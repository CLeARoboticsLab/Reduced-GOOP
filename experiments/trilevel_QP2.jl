using LinearAlgebra
using Random

using BlockArrays: Block, BlockArray
using NonlinearSolve
using QuasiGOOP

const PATH_ATOL = 1e-3
const PATH_CONVERGENCE_TOL = 2e-5

function rand_psd(n, r)
	R = randn(r, n)
	return R' * R
end

function default_path_options(; verbose = false)
	return QuasiGOOP.PATHOptions(;
		convergence_tolerance = PATH_CONVERGENCE_TOL,
		ϵ₀ = 0.0,
		cumulative_iteration_limit = 1000000,
		proximal_perturbation = 1e-2,
		major_iteration_limit = 10000,
		minor_iteration_limit = 15000,
		nms_initial_reference_factor = 50000,
		nms_maximum_watchdogs = 8000,
		nms_memory_size = 16000,
		nms_mstep_frequency = 5000,
		lemke_start_type = "advanced",
		lemke_rank_deficiency_iterations = 50,
		restart_limit = 120,
		gradient_step_limit = 120,
		use_basics = true,
		use_start = true,
		verbose,
	)
end

function build_trilevel_qp_problem(; seed = nothing)
	isnothing(seed) || Random.seed!(seed)
	n = 4

	Q₁ = rand_psd(n, 1)
	c₁ = rand(n)
	Q₂ = rand_psd(n, 1)
	c₂ = rand(n)
	Q₃ = rand_psd(n, 1)
	c₃ = rand(n)

	x = BlockArray(zeros(n), [n])
	θ = BlockArray([0.0], [1])
	A_eq = [1.0 0.0 1.0 1.0; 0.0 1.0 1.0 0.0]
	b_eq = [1.0, 2.0]
	A_ineq = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
	b_ineq = [0.5, 0.5]
	initial_guess = [0.5, 2.0, 0.0, 0.5]

	J₁(x, θ) = 0.5 * x[Block(1)]' * Q₁ * x[Block(1)] + c₁' * x[Block(1)]
	J₂(x, θ) = 0.5 * x[Block(1)]' * Q₂ * x[Block(1)] + c₂' * x[Block(1)]
	J₃(x, θ) = 0.5 * x[Block(1)]' * Q₃ * x[Block(1)] + c₃' * x[Block(1)]
	g_eq(x, θ) = A_eq * x[Block(1)] .- b_eq
	g_ineq(x, θ) = A_ineq * x[Block(1)] .- b_ineq

	problem = QuasiGOOP.ParametricGOOP(
		x,
		θ;
		preferences = [[J₁, J₂, J₃]],
		is_prioritized_constraint = [[false, false, false]],
		equality_constraints = [g_eq],
		inequality_constraints = [g_ineq],
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)

	return (; problem, n, objective = J₁, initial_guess)
end

function solve_with_path(problem, primal_initial_guess; mcp_system, label)
	mcp = mcp_system(problem)
	initial_guess = zeros(mcp.problem_size)
	initial_guess[1:length(primal_initial_guess)] .= primal_initial_guess

	output = QuasiGOOP.solve(
		QuasiGOOP.PATHSolver(),
		mcp,
		zeros(sum(problem.parameter_dims));
		z₀ = initial_guess,
		options = default_path_options(),
	)

	println("$label MCP size: ", mcp.problem_size)
	println("$label status: ", output.status)
	println("$label residual: ", output.info.residual)

	return (; mcp, output, primals = output.z[1:length(primal_initial_guess)])
end

""" Natural MCP residual: project z - F(z) back onto the variable bounds and measure the fixed-point error. 
This is zero exactly when the MCP conditions hold: 
	1. free variables satisfy F[i] = 0, 
	2. lower-bounded variables satisfy z[i] ∈ [0.0, ∞), F[i] >= 0, and complementarity,
# Interpretation: 
	1. If z[i] is free, its bounds are (-∞, ∞) so clamp(z[i] - F[i], -∞, ∞) = z[i] - F[i], and residual[i] = F[i] = 0.
	2. If z[i] has lower bound 0.0, so clamp(z[i] - F[i], 0.0, ∞) = max(z[i] - F[i], 0.0), and residual[i] = z[i] - max(z[i] - F[i], 0.0).
	If z[i] > 0.0, then F[i]  = 0 by complementarity. So residual[i] = z[i] - max(z[i] - 0.0, 0.0) = 0.0
	If z[i] = 0.0, then F[i] >= 0 by complementarity. So residual[i] = 0.0 - max(0.0 - F[i], 0.0) = min(F[i], 0.0) = 0.0.
	If z[i] = 0.0 and F[i] < 0.0, then residual[i] = 0.0 - max(0.0 - F[i], 0.0) = F[i] < 0 => residual is non-zero. 
"""
function compute_mcp_residual!(residual, z, F, lower_bounds, upper_bounds)
	@inbounds for i in eachindex(z)
		projected = clamp(z[i] - F[i], lower_bounds[i], upper_bounds[i])
		residual[i] = z[i] - projected
	end
	Main.@infiltrate
	return residual
end

function complete_dual_check_for_primal(complete, primal, parameter_value)
	mcp = complete.mcp
	num_primals = length(primal)
	num_duals = mcp.problem_size - num_primals
	dual_initial_guess = complete.output.z[(num_primals+1):end]

	function fixed_primal_residual(duals, p)
		z_eltype = promote_type(eltype(duals), eltype(primal), eltype(parameter_value))
		z = Vector{z_eltype}(undef, mcp.problem_size)
		z[1:num_primals] .= primal
		z[(num_primals+1):end] .= duals

		F = similar(z)
		mcp.f!(F, z, parameter_value)

		residual = similar(z)
		compute_mcp_residual!(residual, z, F, mcp.lower_bounds, mcp.upper_bounds)
		return residual
	end

	prob = NonlinearLeastSquaresProblem(fixed_primal_residual, dual_initial_guess)
	dual_check_message = ""
	sol = try
		NonlinearSolve.solve(prob; maxiters = 10_000, abstol = 1e-10, reltol = 1e-10)
	catch err
		dual_check_message = "NonlinearSolve failed: $(sprint(showerror, err)); retried with local damped Gauss-Newton fallback."
		try
			damped_gauss_newton_least_squares(fixed_primal_residual, dual_initial_guess)
		catch fallback_err
			dual_check_message *= " Fallback failed: $(sprint(showerror, fallback_err))."
			(;
				u = dual_initial_guess,
				retcode = :DualCheckFailed,
				residual_norm = Inf,
				iterations = 0,
			)
		end
	end
	residual = try
		fixed_primal_residual(sol.u, nothing)
	catch err
		dual_check_message *= " Residual evaluation failed: $(sprint(showerror, err))."
		fill(Inf, mcp.problem_size)
	end
	residual_norm = norm(residual, Inf)

	z = copy(complete.output.z)
	z[1:num_primals] .= primal
	z[(num_primals+1):end] .= sol.u

	return (;
		sol,
		z,
		duals = sol.u,
		residual,
		residual_norm,
		valid_complete_mcp_solution = residual_norm <= PATH_CONVERGENCE_TOL,
		num_duals,
		message = dual_check_message,
	)
end

function main()
	(; problem, n, objective, initial_guess) = build_trilevel_qp_problem(; seed = 1)
	parameter_value = zeros(sum(problem.parameter_dims))

	complete = solve_with_path(
		problem,
		initial_guess;
		mcp_system = QuasiGOOP.generate_mcp_complete_kkt_system,
		label = "complete",
	)

	reduced = solve_with_path(
		problem,
		complete.primals; # warm-start the reduced MCP with the complete MCP solution's primal variables
		mcp_system = QuasiGOOP.generate_mcp_reduced_kkt_system,
		label = "reduced",
	)


	primal_error = maximum(abs.(reduced.primals .- complete.primals))
	complete_path_solved = Int(complete.output.status) == 1
	reduced_path_solved = Int(reduced.output.status) == 1
	reduced_primal_matches_complete =
		isapprox(reduced.primals, complete.primals; atol = PATH_ATOL)
	reduced_matches_complete =
		reduced_path_solved &&
		reduced.output.info.residual <= PATH_ATOL &&
		reduced_primal_matches_complete

	# If reduced PATH solves to a different primal point, check whether the complete
	# MCP can still be satisfied by solving only for its dual variables.
	complete_dual_check = if complete_path_solved && reduced_path_solved && !reduced_primal_matches_complete
		complete_dual_check_for_primal(complete, reduced.primals, parameter_value)
	else
		nothing
	end

	println("\nComparison")
	println("complete primals: ", complete.primals)
	println("reduced primals:  ", reduced.primals)
	println("primal error: ", primal_error)
	println("complete objective: ", objective(BlockArray(complete.primals, problem.primal_dims), BlockArray([0.0], problem.parameter_dims)))
	println("reduced objective:  ", objective(BlockArray(reduced.primals, problem.primal_dims), BlockArray([0.0], problem.parameter_dims)))
	println("reduced matches complete: ", reduced_matches_complete)
	if !isnothing(complete_dual_check)
		println("\nComplete MCP dual check at reduced primal")
		println("retcode: ", complete_dual_check.sol.retcode)
		println("dual variables solved: ", complete_dual_check.num_duals)
		println("MCP residual Inf-norm: ", complete_dual_check.residual_norm)
		println("valid complete MCP solution: ", complete_dual_check.valid_complete_mcp_solution)
		if !isempty(complete_dual_check.message)
			println("message: ", complete_dual_check.message)
		end
	end
end

main()







# other helpers
function finite_difference_jacobian(f, u, residual)
	J = Matrix{Float64}(undef, length(residual), length(u))
	u_step = copy(u)
	for j in eachindex(u)
		step = sqrt(eps(Float64)) * max(1.0, abs(u[j]))
		u_step[j] = u[j] + step
		trial_residual = f(u_step, nothing)
		if all(isfinite, trial_residual) && all(isfinite, residual)
			J[:, j] .= (trial_residual .- residual) ./ step
		else
			J[:, j] .= 0.0
		end
		u_step[j] = u[j]
	end
	return J
end

function damped_gauss_newton_least_squares(f, initial_guess; maxiters = 200, tol = PATH_CONVERGENCE_TOL)
	u = copy(initial_guess)
	residual = f(u, nothing)
	residual_norm = norm(residual, Inf)
	damping = 1e-8
	iterations = 0

	for iter in 1:maxiters
		iterations = iter
		residual_norm <= tol && break
		all(isfinite, residual) || break

		J = finite_difference_jacobian(f, u, residual)
		normal_matrix = J' * J + damping * I
		normal_rhs = -(J' * residual)
		all(isfinite, normal_matrix) && all(isfinite, normal_rhs) || break
		step = normal_matrix \ normal_rhs
		all(isfinite, step) || break

		accepted = false
		for line_search_step in 0:20
			alpha = 0.5^line_search_step
			trial_u = u .+ alpha .* step
			trial_residual = f(trial_u, nothing)
			trial_norm = norm(trial_residual, Inf)
			if isfinite(trial_norm) && trial_norm < residual_norm
				u = trial_u
				residual = trial_residual
				residual_norm = trial_norm
				damping = max(damping / 10, 1e-12)
				accepted = true
				break
			end
		end

		if !accepted
			damping *= 10
		end
	end

	return (;
		u,
		retcode = residual_norm <= tol ? :Converged : :Stalled,
		residual_norm,
		iterations,
	)
end
