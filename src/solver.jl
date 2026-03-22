using LinearAlgebra

abstract type SolverType end
struct InteriorPoint <: SolverType end

""" Basic interior point solver, based on Nocedal & Wright, ch. 19.
Computes step directions `δz` by solving the relaxed primal-dual system, i.e.
						 ∇F(z; ϵ) δz = -F(z; ϵ).

Given a step direction `δz`, performs a "fraction to the boundary" linesearch,
i.e., for `(x, s)` it chooses step size `α_s` such that
			  α_s = max(α ∈ [0, 1] : s + α δs ≥ (1 - τ) s)
and for `y` it chooses step size `α_s` such that
			  α_y = max(α ∈ [0, 1] : y + α δy ≥ (1 - τ) y).

A typical value of τ is 0.995. Once we converge to ||F(z; \epsilon)|| ≤ ϵ,
we typically decrease ϵ by a factor of 0.1 or 0.2, with smaller values chosen
when the previous subproblem is solved in fewer iterations.

Positional arguments:
	- `mcp::GOOPKKTSystem`: the mixed complementarity problem to solve. #TODO: FIX LATER
	- `θ::AbstractVector{<:Real}`: the parameter vector.

Keyword arguments:
	- `x₀::AbstractVector{<:Real}`: the initial primal variable.
	- `y₀::AbstractVector{<:Real}`: the initial dual variable.
	- `s₀::AbstractVector{<:Real}`: the initial slack variable.
	- `ϵ₀::Real`: the initial relaxation scale.
	- `tol::Real = 1e-4`: the tolerance for the KKT error.
	- `max_inner_iters::Int = 20`: the maximum number of inner iterations.
	- `max_outer_iters::Int = 50`: the maximum number of outer iterations.
	- `tightening_rate::Real = 0.1`: the rate at which to tighten the tolerance.
	- `loosening_rate::Real = 0.5`: the rate at which to loosen the tolerance.
	- `min_stepsize::Real = 1e-2`: the minimum step size for the linesearch.
	- `linesearch::Symbol = :backtracking`: linesearch mode (`:backtracking` or `:fraction_to_boundary`).
	- `verbose::Bool = false`: whether to print debug information.
	- `linear_solve_algorithm::LinearSolve.SciMLLinearSolveAlgorithm`: the linear solve algorithm to use. Any solver from `LinearSolve.jl` that can handle nonsquare system can be used.
	- `convergence_log::Union{Nothing,AbstractDict} = nothing`: optional output dictionary populated with convergence traces.
	- `measure_solve_time::Bool = false`: if true, returns solve time measured with `@btime` (warmup run excludes compile) and includes `solve_time_sec`/`solve_time_ns` fields.
	- `benchmark_samples::Int = 1`: number of @btime samples when `measure_solve_time = true`.
	- `benchmark_evals::Int = 1`: number of evals per sample when `measure_solve_time = true`.
"""
function solve(
	::InteriorPoint,
	mcp::GOOPKKTSystem,
	θ::AbstractVector{<:Real};
	z₀ = nothing,
	tol = 1e-4,
	η₀ = 1e-4,
	ϵ₀ = :auto,
	max_inner_iters = 20,
	max_outer_iters = 50,
	tightening_rate = 0.1,
	loosening_rate = 0.5,
	min_stepsize = 1e-4,
	linesearch = :backtracking,
	verbose = false,
	linear_solve_algorithm = LinearSolve.KrylovJL_LSMR(), # LinearSolve.KrylovJL_LSMR(), # KrylovJL_CRAIGMR() for non-square KKT systems
	convergence_log = nothing,
	use_linsolve = false,
)
	# z = @something(z₀, begin
	# 	z = zeros(mcp.variable_dimension)
	# 	z[mcp.preference_slack_dims] .= 1.0
	# 	z[mcp.interior_point_slack_dims] .= 1.0
	# 	z[mcp.inequality_constraint_dual_dims] .= 1.0
	# 	z
	# end)
	z = zeros(mcp.variable_dimension)
	z[mcp.preference_slack_dims] .= 1.0
	z[mcp.interior_point_slack_dims] .= 1.0
	z[mcp.inequality_constraint_dual_dims] .= 1.0

	if !isnothing(z₀)
		z[mcp.primal_dims] .= z₀
	end

	x = @view z[Not(vcat(mcp.preference_slack_dims, mcp.interior_point_slack_dims, mcp.inequality_constraint_dual_dims))]
	s = @view z[mcp.preference_slack_dims]
	σ = @view z[mcp.interior_point_slack_dims]
	γ = @view z[mcp.inequality_constraint_dual_dims]

	# Set up common memory.
	∇F = mcp.∇F_z!.result_buffer
	F = zeros(mcp.kkt_dimension)
	δz = zeros(mcp.variable_dimension)
	δx = @view δz[Not(vcat(mcp.preference_slack_dims, mcp.interior_point_slack_dims, mcp.inequality_constraint_dual_dims))]
	δs = @view δz[mcp.preference_slack_dims]
	δσ = @view δz[mcp.interior_point_slack_dims]
	δγ = @view δz[mcp.inequality_constraint_dual_dims]

	use_linsolve && (linsolve = init(LinearProblem(∇F, δz), linear_solve_algorithm, maxiters = 100000))

	# Main solver loop.
	if ϵ₀ === :auto
		is_warmstarted = !isnothing(z₀)
		if is_warmstarted
			ϵ = tol
		else
			ϵ = one(tol)
		end
	else
		ϵ = ϵ₀
	end

	# Initialize regularization parameter.
	η = η₀

	status = :solved
	linesearch ∈ (:backtracking, :fraction_to_boundary) ||
		throw(ArgumentError("Unsupported linesearch $(linesearch). Use :backtracking or :fraction_to_boundary."))
	total_iters = 0
	inner_iters = 1
	outer_iters = 1
	kkt_error = Inf
	is_fraction_to_boundary_linesearch = (linesearch == :fraction_to_boundary)
	has_convergence_log = !isnothing(convergence_log)
	kkt_error_history = Float64[]
	total_iteration_history = Int[]
	outer_iteration_history = Int[]
	inner_iteration_history = Int[]
	outer_end_total_iterations = Int[]
	outer_end_trace_indices = Int[]
	while outer_iters < max_outer_iters || iszero(total_iters)
		inner_iters = 1
		status = :solved

		verbose && @info "Outer iteration $(outer_iters): ϵ = $ϵ, kkt_error = $kkt_error"

		while inner_iters < max_inner_iters &&(kkt_error > tol) # (!is_fraction_to_boundary_linesearch || kkt_error > tol)
			total_iters += 1
			# Compute the Newton step.
			# TODO: Can add some adaptive regularization.
			# TODO: use a linear operator with a lazy gradient computation here.
			mcp.F!(F, z; θ, ϵ, η = 0.0)
			mcp.∇F_z!(∇F, z; θ, ϵ, η)
			# @assert all(.!isnan.(F)) "Found NaN in F - aborting!"
			# @assert all(.!isnan.(∇F)) "Found NaN in ∇F - aborting!"
			# verbose && println("inner iter $inner_iters, condition number of ∇F = $(cond(collect(∇F),2))")
			verbose && println("inner iter $inner_iters")
			# Check the primals
			# verbose && println("current primal x: ", round.(z[mcp.primal_dims]; digits = 4))

			# # Solve δz using the specified linear solver.
			# linsolve.A = ∇F
			# linsolve.b = -F
			# solution = solve!(linsolve)
			# if !SciMLBase.successful_retcode(solution) &&
			#    (solution.retcode !== SciMLBase.ReturnCode.Default)
			# 	verbose &&
			# 		@warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
			# 	status = :failed
			# 	break
			# end
			# δz .= solution.u

			# Solve δz via direct pseudoinverse
			δz .= ∇F \ (-F)

			# verbose && println("current δx: ", round.(δz[mcp.primal_dims]; digits = 4))

			if linesearch == :fraction_to_boundary
				α_σ = fraction_to_the_boundary_linesearch(σ, δσ; tol = min_stepsize)
				α_γ = fraction_to_the_boundary_linesearch(γ, δγ; tol = min_stepsize)
				verbose && println("fraction_to_boundary linesearch α_σ = $α_σ, α_γ = $α_γ")
				if isnan(α_σ) || isnan(α_γ)
					verbose && @warn "Fraction-to-boundary linesearch failed. Exiting prematurely."
					status = :failed
					break
				end

				# Update regularization parameter.
				if min(α_σ, α_γ) == 1.0
					verbose && printstyled("Full step taken... Decreasing η. ($η -> $(η * (1 - exp(-tightening_rate * inner_iters))))\n"; color = :blue)
					η *= 1 - exp(-tightening_rate)
				else
					verbose && printstyled("Partial step (<1.0) taken... Increasing η. ($η -> $(η * (1 + exp(-loosening_rate * inner_iters))))\n"; color = :red)
					η *= 1 + exp(-loosening_rate)
				end
			else
				# backtracking linesearch
				α = 1.0
				F_z = norm(F, 2)
				z_trial = similar(z)
				@. z_trial = z + α * δz
				mcp.F!(F, z_trial; θ, ϵ, η)
				F_z_next = norm(F, 2)
				while (F_z_next >= 1.0 * F_z) || (any(@. σ + α * δσ < 0) || any(@. γ + α * δγ < 0))
					if α < min_stepsize
						verbose && @warn "Backtracking linesearch failed. Exiting prematurely."
						status = :failed
						break
					end

					α *= 0.5 # decay
					@. z_trial = z + α * δz
					mcp.F!(F, z_trial; θ, ϵ, η)
					F_z_next = norm(F, 2)
				end
				if status === :failed
					break
				end
				verbose && println("backtracking linesearch α = $α")
				α_σ = α
				α_γ = α
			end

			# Update variables accordingly.
			@. x += α_σ * δx
			@. s += α_σ * δs
			@. σ += α_σ * δσ
			@. γ += α_γ * δγ

			kkt_error = norm(F, 2)
			if has_convergence_log
				push!(kkt_error_history, kkt_error)
				push!(total_iteration_history, total_iters)
				push!(outer_iteration_history, outer_iters)
				push!(inner_iteration_history, inner_iters)
			end

			verbose && println("KKT error = $kkt_error")

			inner_iters += 1
		end

		if has_convergence_log &&
		   !isempty(total_iteration_history) &&
		   (isempty(outer_end_total_iterations) || last(outer_end_total_iterations) != total_iters)
			push!(outer_end_total_iterations, total_iters)
			push!(outer_end_trace_indices, length(total_iteration_history))
		end

		if linesearch == :fraction_to_boundary
			if kkt_error <= ϵ <= tol
				break
			end
			ϵ *= if status === :solved
				1 - exp(-tightening_rate)
			else
				1 + exp(-loosening_rate)
			end
			ϵ = min(ϵ, one(ϵ))
		end

		outer_iters += 1
	end

	if outer_iters == max_outer_iters
		status = (kkt_error <= tol) ? :solved : :failed
	end
	if has_convergence_log
		convergence_log["kkt_error_history"] = kkt_error_history
		convergence_log["total_iteration_history"] = total_iteration_history
		convergence_log["outer_iteration_history"] = outer_iteration_history
		convergence_log["inner_iteration_history"] = inner_iteration_history
		convergence_log["outer_end_total_iterations"] = outer_end_total_iterations
		convergence_log["outer_end_trace_indices"] = outer_end_trace_indices
	end

	(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters)
end

"""Helper function to compute the step size `α` which solves:
				   α* = max(α ∈ [0, 1] : v + α δ ≥ (1 - τ) v).
"""
function fraction_to_the_boundary_linesearch(v, δ; max_stepsize = 1.0, τ = 0.995, decay = 0.5, tol = 1e-4)
	α = max_stepsize
	while any(@. v + α * δ < (1 - τ) * v)
		if α < tol
			return NaN
		end

		α *= decay
	end

	α
end
