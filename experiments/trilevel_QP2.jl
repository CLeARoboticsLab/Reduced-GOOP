using LinearAlgebra
using Logging
using Random
using Statistics

using BlockArrays: Block, BlockArray
using NonlinearSolve
using QuasiGOOP

const NUM_TRIALS = 100
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
		cumulative_iteration_limit = 100000,
		proximal_perturbation = 1e-2,
		major_iteration_limit = 3000,
		minor_iteration_limit = 3000,
		nms_initial_reference_factor = 100,
		nms_maximum_watchdogs = 100,
		nms_memory_size = 100,
		nms_mstep_frequency = 100,
		lemke_start_type = "advanced",
		lemke_rank_deficiency_iterations = 50,
		restart_limit = 500,
		gradient_step_limit = 500,
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
	initial_guess = [2.0, 0.0, 0.0, 0.0]

	J₁(x, θ) = 0.5 * x[Block(1)]' * Q₁ * x[Block(1)] + c₁' * x[Block(1)]
	J₂(x, θ) = 0.5 * x[Block(1)]' * Q₂ * x[Block(1)] + c₂' * x[Block(1)]
	J₃(x, θ) = 0.5 * x[Block(1)]' * Q₃ * x[Block(1)] + c₃' * x[Block(1)]

	# Equality-only bounded, non-singleton feasible set:
	# {x in R^4 : sum(x) = 2, ||x||^2 = 4}.
	g_eq(x, θ) = [
		sum(x[Block(1)]) - 2.0,
		sum(abs2, x[Block(1)]) - 4.0,
	]

	problem = QuasiGOOP.ParametricGOOP(
		x,
		θ;
		preferences = [[J₁, J₂, J₃]],
		is_prioritized_constraint = [[false, false, false]],
		equality_constraints = [g_eq],
		inequality_constraints = [nothing],
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)

	return (; problem, initial_guess)
end

function solve_with_path(problem, primal_initial_guess; mcp_system)
	mcp = mcp_system(problem)
	initial_guess = zeros(mcp.problem_size)
	initial_guess[1:length(primal_initial_guess)] .= primal_initial_guess

	output = with_logger(NullLogger()) do
		QuasiGOOP.solve(
			QuasiGOOP.PATHSolver(),
			mcp,
			zeros(sum(problem.parameter_dims));
			z₀ = initial_guess,
			options = default_path_options(),
		)
	end

	return (; mcp, output, primals = output.z[1:length(primal_initial_guess)])
end

function path_solved(output)
	return Int(output.status) == 1
end

function residual_summary(residuals)
	return (minimum(residuals), median(residuals), maximum(residuals))
end

""" Compute MCP residual: project z - F(z) back onto the variable bounds and measure the fixed-point error. 
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
	return residual
end

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

		!accepted && (damping *= 10)
	end

	return (;
		u,
		retcode = residual_norm <= tol ? :Converged : :Stalled,
		residual_norm,
		iterations,
	)
end

function complete_dual_check_for_primal(complete, primal, parameter_value)
	mcp = complete.mcp
	num_primals = length(primal)
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

	function residual_norm_for_solution(candidate)
		residual = try
			fixed_primal_residual(candidate.u, nothing)
		catch
			return Inf
		end
		return norm(residual, Inf)
	end

	function fallback_solution()
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

	dual_check_message = ""
	prob = NonlinearLeastSquaresProblem(fixed_primal_residual, dual_initial_guess)
	sol = try
		with_logger(NullLogger()) do
			NonlinearSolve.solve(prob; maxiters = 10_000, abstol = 1e-10, reltol = 1e-10)
		end
	catch err
		dual_check_message = "NonlinearSolve failed: $(sprint(showerror, err)); fallback used."
		fallback_solution()
	end

	sol_residual_norm = residual_norm_for_solution(sol)
	if sol_residual_norm > PATH_CONVERGENCE_TOL
		fallback = fallback_solution()
		fallback_residual_norm = residual_norm_for_solution(fallback)
		if fallback_residual_norm < sol_residual_norm
			dual_check_message *= isempty(dual_check_message) ? "Fallback improved stalled NonlinearSolve result." : " Fallback improved stalled NonlinearSolve result."
			sol = fallback
		end
	end

	residual = try
		fixed_primal_residual(sol.u, nothing)
	catch err
		dual_check_message *= " Residual evaluation failed: $(sprint(showerror, err))."
		fill(Inf, mcp.problem_size)
	end
	residual_norm = norm(residual, Inf)

	return (;
		sol,
		residual_norm,
		valid_at_path_atol = isfinite(residual_norm) && residual_norm <= PATH_ATOL,
		valid_at_convergence_tol = isfinite(residual_norm) && residual_norm <= PATH_CONVERGENCE_TOL,
		num_duals = length(dual_initial_guess),
		message = dual_check_message,
	)
end

function run_trials(num_trials = NUM_TRIALS)
	complete_solved = 0
	reduced_solved = 0
	both_solved = 0
	matches_path_tol = 0
	mismatches_path_tol = 0
	mismatches_valid_at_path_atol = 0
	mismatches_valid_at_convergence_tol = 0

	status_pairs = Dict{Tuple{String, String}, Int}()
	complete_residuals = Float64[]
	reduced_residuals = Float64[]
	primal_errors = Float64[]
	dual_check_residuals = Float64[]
	mismatches = []
	first_both_solved = nothing

	for seed in 1:num_trials
		(; problem, initial_guess) = build_trilevel_qp_problem(; seed)
		parameter_value = zeros(sum(problem.parameter_dims))

		complete = solve_with_path(
			problem,
			initial_guess;
			mcp_system = QuasiGOOP.generate_mcp_complete_kkt_system,
		)
		reduced = solve_with_path(
			problem,
			initial_guess;
			mcp_system = QuasiGOOP.generate_mcp_reduced_kkt_system,
		)

		complete_ok = path_solved(complete.output)
		reduced_ok = path_solved(reduced.output)
		complete_solved += complete_ok
		reduced_solved += reduced_ok

		push!(complete_residuals, complete.output.info.residual)
		push!(reduced_residuals, reduced.output.info.residual)

		status_key = (string(complete.output.status), string(reduced.output.status))
		status_pairs[status_key] = get(status_pairs, status_key, 0) + 1

		if complete_ok && reduced_ok
			both_solved += 1
			primal_error = maximum(abs.(complete.primals .- reduced.primals))
			push!(primal_errors, primal_error)

			primal_error <= PATH_ATOL && (matches_path_tol += 1)

			if first_both_solved === nothing
				first_both_solved = (;
					seed,
					primal_error,
					complete_size = complete.mcp.problem_size,
					reduced_size = reduced.mcp.problem_size,
					complete_residual = complete.output.info.residual,
					reduced_residual = reduced.output.info.residual,
					complete_primals = complete.primals,
					reduced_primals = reduced.primals,
				)
			end

				if primal_error > PATH_ATOL
					mismatches_path_tol += 1
					dual_check = complete_dual_check_for_primal(complete, reduced.primals, parameter_value)
					push!(dual_check_residuals, dual_check.residual_norm)
					dual_check.valid_at_path_atol && (mismatches_valid_at_path_atol += 1)
					dual_check.valid_at_convergence_tol && (mismatches_valid_at_convergence_tol += 1)

					push!(
						mismatches,
						(;
						seed,
						primal_error,
						complete_residual = complete.output.info.residual,
							reduced_residual = reduced.output.info.residual,
							complete_primals = complete.primals,
							reduced_primals = reduced.primals,
							dual_check_residual = dual_check.residual_norm,
							dual_check_valid_at_path_atol = dual_check.valid_at_path_atol,
							dual_check_valid_at_convergence_tol = dual_check.valid_at_convergence_tol,
							dual_check_retcode = dual_check.sol.retcode,
							dual_check_message = dual_check.message,
						),
					)
				end
			end

		seed % 10 == 0 && println("completed ", seed, "/", num_trials)
	end

	println("\nSummary: equality-only trilevel QP, same initial guess for complete and reduced")
	println("constraints: sum(x) = 2, sum(abs2, x) = 4, no inequalities")
	println("initial guess: [2.0, 0.0, 0.0, 0.0]")
	println("total runs: ", num_trials)
	println("complete PATH solved: ", complete_solved)
	println("reduced PATH solved: ", reduced_solved)
	println("both PATH solved: ", both_solved)
	println("both solved and primal matched <= ", PATH_ATOL, ": ", matches_path_tol)
	println("both solved but primal mismatch > ", PATH_ATOL, ": ", mismatches_path_tol)
	println("mismatches with complete duals found <= ", PATH_ATOL, ": ", mismatches_valid_at_path_atol)
	println("mismatches with complete duals found <= ", PATH_CONVERGENCE_TOL, ": ", mismatches_valid_at_convergence_tol)
	println("status pairs: ", status_pairs)
	println("complete residual min/median/max: ", residual_summary(complete_residuals))
	println("reduced residual min/median/max: ", residual_summary(reduced_residuals))

	if !isempty(primal_errors)
		println("both-solved primal error min/median/max: ", residual_summary(primal_errors))
	end
	if !isempty(dual_check_residuals)
		println("mismatch dual-check residual min/median/max: ", residual_summary(dual_check_residuals))
	end

	if first_both_solved !== nothing
		println("\nFirst both-solved trial")
		println("seed: ", first_both_solved.seed)
		println("MCP sizes complete/reduced: ", first_both_solved.complete_size, " / ", first_both_solved.reduced_size)
		println("primal error: ", first_both_solved.primal_error)
		println("residuals complete/reduced: ", first_both_solved.complete_residual, " / ", first_both_solved.reduced_residual)
		println("complete primals: ", first_both_solved.complete_primals)
		println("reduced primals:  ", first_both_solved.reduced_primals)
	end

	if !isempty(mismatches)
		println("\nFirst primal mismatches above PATH_ATOL")
		for item in Iterators.take(mismatches, 10)
			println(
				"seed=", item.seed,
				" primal_error=", item.primal_error,
					" complete_res=", item.complete_residual,
					" reduced_res=", item.reduced_residual,
					" dual_check_res=", item.dual_check_residual,
					" dual_valid_path_atol=", item.dual_check_valid_at_path_atol,
					" dual_valid_convergence_tol=", item.dual_check_valid_at_convergence_tol,
					" dual_retcode=", item.dual_check_retcode,
					" complete_primals=", item.complete_primals,
					" reduced_primals=", item.reduced_primals,
					isempty(item.dual_check_message) ? "" : " dual_message=$(item.dual_check_message)",
				)
			end
	end
end

run_trials()


"""
For 15 of the 16 primal mismatch cases, the reduced primal can be extended with complete MCP duals and 
is therefore also a valid complete-MCP solution. 
The one unvalidated case in the printed first-10 list is seed 33, with dual-check residual about 0.22069.
"""