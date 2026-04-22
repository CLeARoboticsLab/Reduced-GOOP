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
	use_proximal = false,
	ρx₀ = 1e-2,
	ρs₀ = 0.0,
	ρx_min = 1e-8,
	ρx_max = 1e4,
	ρs_min = 0.0,
	ρs_max = 1e4,
	ρ_decrease = 0.5,
	ρ_increase = 2.0,
	stable_step_threshold = 0.8,
	unstable_step_threshold = 1e-2,
	stable_inner_iter_threshold = 4,
	stable_min_inner_iters = 2,
	print_proximal_trace = false,
	proximal_slack_reference = :current,
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
	∇F =
		hasproperty(mcp.∇F_z!, :result_buffer) ? mcp.∇F_z!.result_buffer :
		zeros(mcp.kkt_dimension, mcp.variable_dimension)
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
	ρx = clamp(ρx₀, ρx_min, ρx_max)
	ρs = clamp(ρs₀, ρs_min, ρs_max)
	ρx_start = ρx
	ρs_start = ρs

	status = :solved
	linesearch ∈ (:backtracking, :fraction_to_boundary) ||
		throw(ArgumentError("Unsupported linesearch $(linesearch). Use :backtracking or :fraction_to_boundary."))
	proximal_slack_reference ∈ (:current, :zero) ||
		throw(
			ArgumentError(
				"Unsupported proximal_slack_reference $(proximal_slack_reference). Use :current or :zero.",
			),
		)
	total_iters = 0
	inner_iters = 1
	outer_iters = 1
	kkt_error = Inf
	is_fraction_to_boundary_linesearch = (linesearch == :fraction_to_boundary)
	has_convergence_log = !isnothing(convergence_log)
	track_proximal_trace = use_proximal && (has_convergence_log || print_proximal_trace)
	proximal_outer_trace = Any[]
	proximal_inner_trace = Any[]
	proximal_rho_update_trace = Any[]
	kkt_error_history = Float64[]
	total_iteration_history = Int[]
	outer_iteration_history = Int[]
	inner_iteration_history = Int[]
	outer_end_total_iterations = Int[]
	outer_end_trace_indices = Int[]
	call_F! = if use_proximal
		(result, z; θ, ϵ, η, x_ref, s_ref, ρx, ρs) -> mcp.F!(
			result,
			z;
			θ,
			ϵ,
			η,
			x_ref,
			s_ref,
			ρx,
			ρs,
		)
	else
		(result, z; θ, ϵ, η, x_ref, s_ref, ρx, ρs) -> mcp.F!(result, z; θ, ϵ, η)
	end
	call_∇F_z! = if use_proximal
		(result, z; θ, ϵ, η, x_ref, s_ref, ρx, ρs) -> mcp.∇F_z!(
			result,
			z;
			θ,
			ϵ,
			η,
			x_ref,
			s_ref,
			ρx,
			ρs,
		)
	else
		(result, z; θ, ϵ, η, x_ref, s_ref, ρx, ρs) -> mcp.∇F_z!(result, z; θ, ϵ, η)
	end
	while outer_iters < max_outer_iters || iszero(total_iters)
		inner_iters = 1
		status = :solved
		line_search_failed = false
		linear_solve_failed = false
		accepted_step_sizes = Float64[]
		# Keep proximal references fixed throughout the inner Newton loop.
		x_ref_val = copy(z[mcp.primal_dims])
		s_ref_val =
			(use_proximal && proximal_slack_reference === :zero) ?
			zeros(eltype(z), length(mcp.preference_slack_dims)) :
			copy(z[mcp.preference_slack_dims])
		ρx_eval = use_proximal ? ρx : 0.0
		ρs_eval = use_proximal ? ρs : 0.0
		if track_proximal_trace
			outer_entry = (
				outer_iteration = outer_iters,
				ρx = ρx_eval,
				ρs = ρs_eval,
				x_ref = copy(x_ref_val),
				s_ref = copy(s_ref_val),
			)
			push!(proximal_outer_trace, outer_entry)
			if print_proximal_trace
				println("proximal outer iteration $(outer_iters)")
				println("  ρx = $(outer_entry.ρx), ρs = $(outer_entry.ρs)")
				println("  x_ref = $(outer_entry.x_ref)")
				println("  s_ref = $(outer_entry.s_ref)")
			end
		end
		call_F!(
			F,
			z;
			θ,
			ϵ,
			η = 0.0,
			x_ref = x_ref_val,
			s_ref = s_ref_val,
			ρx = ρx_eval,
			ρs = ρs_eval,
		)
		outer_residual_start = norm(F, 2)
		kkt_error = outer_residual_start

		verbose &&
			@info "Outer iteration $(outer_iters): ϵ = $ϵ, kkt_error = $kkt_error, ρx = $ρx_eval, ρs = $ρs_eval"

		while inner_iters < max_inner_iters && (kkt_error > tol) # (!is_fraction_to_boundary_linesearch || kkt_error > tol)
			total_iters += 1
			if track_proximal_trace
				inner_entry = (
					outer_iteration = outer_iters,
					inner_iteration = inner_iters,
					total_iteration = total_iters,
					ρx = ρx_eval,
					ρs = ρs_eval,
					x_ref = copy(x_ref_val),
					s_ref = copy(s_ref_val),
				)
				push!(proximal_inner_trace, inner_entry)
				if print_proximal_trace
					println(
						"proximal inner iteration outer=$(outer_iters), inner=$(inner_iters), total=$(total_iters)",
					)
					println("  ρx = $(inner_entry.ρx), ρs = $(inner_entry.ρs)")
					println("  x_ref = $(inner_entry.x_ref)")
					println("  s_ref = $(inner_entry.s_ref)")
				end
			end
			# Compute the Newton step.
			# TODO: Can add some adaptive regularization.
			# TODO: use a linear operator with a lazy gradient computation here.
			call_F!(
				F,
				z;
				θ,
				ϵ,
				η = 0.0,
				x_ref = x_ref_val,
				s_ref = s_ref_val,
				ρx = ρx_eval,
				ρs = ρs_eval,
			)
			call_∇F_z!(
				∇F,
				z;
				θ,
				ϵ,
				η,
				x_ref = x_ref_val,
				s_ref = s_ref_val,
				ρx = ρx_eval,
				ρs = ρs_eval,
			)
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

			# Solve δz 
			try
				δz .= ∇F \ (-F)
			catch err
				if verbose
					@warn "Linear solve failed. Exiting prematurely." exception = err
				end
				status = :failed
				linear_solve_failed = true
				break
			end
			# δz .= pinv(Matrix(∇F)) * (-F) # minimum-norm sol

			# verbose && println("current δx: ", round.(δz[mcp.primal_dims]; digits = 4))

			if linesearch == :fraction_to_boundary
				α_σ = fraction_to_the_boundary_linesearch(σ, δσ; tol = min_stepsize)
				α_γ = fraction_to_the_boundary_linesearch(γ, δγ; tol = min_stepsize)
				verbose && println("fraction_to_boundary linesearch α_σ = $α_σ, α_γ = $α_γ")
				if isnan(α_σ) || isnan(α_γ)
					verbose && @warn "Fraction-to-boundary linesearch failed. Exiting prematurely."
					status = :failed
					line_search_failed = true
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
				call_F!(
					F,
					z_trial;
					θ,
					ϵ,
					η,
					x_ref = x_ref_val,
					s_ref = s_ref_val,
					ρx = ρx_eval,
					ρs = ρs_eval,
				)
				F_z_next = norm(F, 2)
				while (F_z_next >= 1.0 * F_z) || (any(@. σ + α * δσ < 0) || any(@. γ + α * δγ < 0))
					if α < min_stepsize
						verbose && @warn "Backtracking linesearch failed. Exiting prematurely."
						status = :failed
						line_search_failed = true
						break
					end

					α *= 0.5 # decay
					@. z_trial = z + α * δz
					call_F!(
						F,
						z_trial;
						θ,
						ϵ,
						η,
						x_ref = x_ref_val,
						s_ref = s_ref_val,
						ρx = ρx_eval,
						ρs = ρs_eval,
					)
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
			push!(accepted_step_sizes, min(α_σ, α_γ))

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
		inner_iters_used = inner_iters - 1
		outer_residual_end = kkt_error
		avg_accepted_step =
			isempty(accepted_step_sizes) ? 0.0 : sum(accepted_step_sizes) / length(accepted_step_sizes)
		min_accepted_step = isempty(accepted_step_sizes) ? 0.0 : minimum(accepted_step_sizes)
		residual_ratio = outer_residual_end / max(outer_residual_start, eps(Float64))
		residual_decreased_significantly = (inner_iters_used > 0) && (residual_ratio <= 0.5)
		residual_stagnated_or_increased = (inner_iters_used > 0) && (residual_ratio >= 0.95)
		no_failure = (status !== :failed) && !line_search_failed && !linear_solve_failed
		is_stable_outer_iteration =
			no_failure &&
			(inner_iters_used >= stable_min_inner_iters) &&
			(avg_accepted_step >= stable_step_threshold) &&
			(min_accepted_step >= unstable_step_threshold) &&
			(inner_iters_used <= stable_inner_iter_threshold) &&
			residual_decreased_significantly
		many_tiny_steps =
			(inner_iters_used > 0) &&
			((avg_accepted_step < unstable_step_threshold) || (min_accepted_step < unstable_step_threshold))
		too_many_inner_iterations =
			(inner_iters_used > 0) &&
			(inner_iters_used > stable_inner_iter_threshold)
		is_unstable_outer_iteration =
			line_search_failed ||
			linear_solve_failed ||
			(status === :failed) ||
			many_tiny_steps ||
			too_many_inner_iterations ||
			residual_stagnated_or_increased
		ρx_before_update = ρx
		ρs_before_update = ρs
		stability_state =
			is_stable_outer_iteration ? :stable :
			is_unstable_outer_iteration ? :unstable : :neutral
		if use_proximal
			if is_stable_outer_iteration
				ρx = max(ρx_min, ρ_decrease * ρx)
				ρs = max(ρs_min, ρ_decrease * ρs)
			elseif is_unstable_outer_iteration
				ρx = min(ρx_max, ρ_increase * ρx)
				ρs = min(ρs_max, ρ_increase * ρs)
			end
		end
		if track_proximal_trace
			ρ_update_entry = (
				outer_iteration = outer_iters,
				stability = stability_state,
				ρx_before = ρx_before_update,
				ρx_after = ρx,
				ρs_before = ρs_before_update,
				ρs_after = ρs,
				outer_residual_start,
				outer_residual_end,
				avg_accepted_step,
				min_accepted_step,
				inner_iters_used,
			)
			push!(proximal_rho_update_trace, ρ_update_entry)
			if print_proximal_trace
				println("proximal outer update $(outer_iters) ($(ρ_update_entry.stability))")
				println(
					"  ρx: $(ρ_update_entry.ρx_before) -> $(ρ_update_entry.ρx_after), ρs: $(ρ_update_entry.ρs_before) -> $(ρ_update_entry.ρs_after)",
				)
				println(
					"  residual: $(ρ_update_entry.outer_residual_start) -> $(ρ_update_entry.outer_residual_end), avg_step=$(ρ_update_entry.avg_accepted_step), min_step=$(ρ_update_entry.min_accepted_step)",
				)
			end
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
		# (kkt_error <= tol) && Main.@infiltrate
	end
	if has_convergence_log
		convergence_log["kkt_error_history"] = kkt_error_history
		convergence_log["total_iteration_history"] = total_iteration_history
		convergence_log["outer_iteration_history"] = outer_iteration_history
		convergence_log["inner_iteration_history"] = inner_iteration_history
		convergence_log["outer_end_total_iterations"] = outer_end_total_iterations
		convergence_log["outer_end_trace_indices"] = outer_end_trace_indices
		convergence_log["proximal_outer_trace"] = proximal_outer_trace
		convergence_log["proximal_inner_trace"] = proximal_inner_trace
		convergence_log["proximal_rho_update_trace"] = proximal_rho_update_trace
		convergence_log["proximal_start_ρx"] = ρx_start
		convergence_log["proximal_start_ρs"] = ρs_start
		convergence_log["proximal_final_ρx"] = ρx
		convergence_log["proximal_final_ρs"] = ρs
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
