using LinearAlgebra

abstract type SolverType end
struct InteriorPoint <: SolverType end
struct PATHSolver <: SolverType end

Base.@kwdef struct InteriorPointOptions
	tol::Float64
	η₀::Float64
	ϵ₀::Union{Float64, Symbol}
	max_inner_iters::Int
	max_outer_iters::Int
	tightening_rate::Float64
	loosening_rate::Float64
	min_stepsize::Float64
	linesearch::Symbol
	linear_solve_algorithm::LinearSolve.SciMLLinearSolveAlgorithm
	use_linsolve::Bool
	record_convergence::Bool
	record_condition_number::Bool = false
	record_solver_diagnostics::Bool = false
	solver_diagnostics_limit::Int = 1000
	max_eta_retries::Int = 5
	eta_retry_growth::Float64 = 2.0
	η_min::Float64 = 1e-20
	η_max::Float64 = 1e-1
	perturbation_enabled::Bool = false
	stagnation_iters::Int = 50
	stagnation_rtol::Float64 = 1e-3
	perturbation_scale::Float64 = 1e-4
	max_perturbations::Int = 20
	tsvd_threshold::Float64 = 0.0
	use_marquardt_scaling::Bool = false
	verbose::Bool

end

Base.@kwdef struct PATHOptions
	convergence_tolerance::Float64
	ϵ₀::Union{Float64, Symbol}
	cumulative_iteration_limit::Int
	proximal_perturbation::Float64
	major_iteration_limit::Int
	minor_iteration_limit::Int
	nms_initial_reference_factor::Int
	nms_maximum_watchdogs::Int
	nms_memory_size::Int
	nms_mstep_frequency::Int
	lemke_start_type::String
	lemke_rank_deficiency_iterations::Int
	restart_limit::Int
	gradient_step_limit::Int
	use_basics::Bool
	use_start::Bool
	verbose::Bool
end

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
	- `record_convergence::Bool = false`: if true, record and return `kkt_error_history`.
	- `record_condition_number::Bool = false`: if true, record and return `condition_number_history`.
	- `record_solver_diagnostics::Bool = false`: if true, record singular-value and step diagnostics.
	- `solver_diagnostics_limit::Int = 1000`: maximum number of inner-iteration diagnostic rows to record.
	- `perturbation_enabled::Bool = false`: if true, perturb `x` after repeated KKT residual stagnation.
	- `stagnation_iters::Int = 5`: accepted iterations without relative KKT improvement before perturbing.
	- `stagnation_rtol::Real = 1e-3`: relative KKT improvement threshold used to reset the stagnation counter.
	- `perturbation_scale::Real = 1e-4`: fixed standard deviation for random perturbations.
	- `max_perturbations::Int = 2`: maximum number of perturbation attempts per solve.
	- `measure_solve_time::Bool = false`: if true, returns solve time measured with `@btime` (warmup run excludes compile) and includes `solve_time_sec`/`solve_time_ns` fields.
	- `benchmark_samples::Int = 1`: number of @btime samples when `measure_solve_time = true`.
	- `benchmark_evals::Int = 1`: number of evals per sample when `measure_solve_time = true`.
"""
function solve(
	::InteriorPoint,
	mcp::GOOPKKTSystem,
	θ::AbstractVector{<:Real};
	z₀ = nothing,
	options::InteriorPointOptions,
)
	tol = options.tol
	η₀ = options.η₀
	ϵ₀ = options.ϵ₀
	max_inner_iters = options.max_inner_iters
	max_outer_iters = options.max_outer_iters
	tightening_rate = options.tightening_rate
	loosening_rate = options.loosening_rate
	min_stepsize = options.min_stepsize
	linesearch = options.linesearch
	linear_solve_algorithm = options.linear_solve_algorithm
	use_linsolve = options.use_linsolve
	record_convergence = options.record_convergence
	record_condition_number = options.record_condition_number
	record_solver_diagnostics = options.record_solver_diagnostics
	solver_diagnostics_limit = options.solver_diagnostics_limit
	verbose = options.verbose
	max_eta_retries = options.max_eta_retries
	eta_retry_growth = options.eta_retry_growth
	η_min = options.η_min
	η_max = options.η_max
	perturbation_enabled = options.perturbation_enabled
	stagnation_iters = options.stagnation_iters
	stagnation_rtol = options.stagnation_rtol
	perturbation_scale = options.perturbation_scale
	max_perturbations = options.max_perturbations
	tsvd_threshold = options.tsvd_threshold
	use_marquardt_scaling = options.use_marquardt_scaling
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

	x_dims = Not(vcat(mcp.preference_slack_dims, mcp.interior_point_slack_dims, mcp.inequality_constraint_dual_dims))
	x = @view z[x_dims]
	s = @view z[mcp.preference_slack_dims]
	σ = @view z[mcp.interior_point_slack_dims]
	γ = @view z[mcp.inequality_constraint_dual_dims]

	# Set up common memory.
	∇F = mcp.∇F_z!.result_buffer
	F = zeros(mcp.kkt_dimension)
	F_trial = zeros(mcp.kkt_dimension)
	Jδz = zeros(mcp.kkt_dimension)
	z_trial = similar(z)
	δz = zeros(mcp.variable_dimension)
	δx = @view δz[x_dims]
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
	best_kkt_error = Inf
	iters_since_improvement = 0
	num_perturbations = 0
	is_fraction_to_boundary_linesearch = (linesearch == :fraction_to_boundary)
	kkt_error_history = Float64[]
	condition_number_history = Float64[]
	solver_diagnostics = NamedTuple[]
	record_solver_diagnostics && solver_diagnostics_limit < 0 &&
		throw(ArgumentError("solver_diagnostics_limit must be nonnegative."))
	while outer_iters < max_outer_iters || iszero(total_iters)
		inner_iters = 1
		status = :solved

		verbose && @info "Outer iteration $(outer_iters): ϵ = $ϵ, kkt_error = $kkt_error"

		while inner_iters < max_inner_iters && (kkt_error > tol) # (!is_fraction_to_boundary_linesearch || kkt_error > tol)
			total_iters += 1
			# Compute the residual at the current iterate (η does not enter F, only ∇F).
			mcp.F!(F, z; θ, ϵ, η = 0.0)
			# @assert all(.!isnan.(F)) "Found NaN in F - aborting!"
			verbose && println("inner iter $inner_iters")
			condition_number = NaN

			if linesearch == :fraction_to_boundary
				mcp.∇F_z!(∇F, z; θ, ϵ, η = 0.0)
				Jmat = Matrix(∇F)
				Jsvd = svd(Jmat)
				condition_number = record_condition_number ? Jsvd.S[1] / Jsvd.S[end] : NaN
				verbose && record_condition_number && println("condition number of ∇F: ", condition_number)
				# Unified TSVD+Tikhonov step: modes below tsvd_threshold*σ₁ are zeroed (hard cutoff),
				# remaining modes use the Tikhonov filter σ/(σ²+η). tsvd_threshold=0 → pure Tikhonov.
				threshold_abs = tsvd_threshold * Jsvd.S[1]
				filters = @. ifelse(Jsvd.S >= threshold_abs, Jsvd.S / (Jsvd.S^2 + η), zero(eltype(Jsvd.S)))
				δz .= -Jsvd.V * (filters .* (Jsvd.U' * F))
				record_solver_diagnostics && _record_solver_diagnostic!(
					solver_diagnostics,
					solver_diagnostics_limit,
					total_iters,
					norm(F, 2),
					Jsvd.S,
					δz,
				)

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
				# Backtracking linesearch: if the line search exhausts at the current η,
				# grow η and re-solve the Newton step rather than failing outright.
				F_z = norm(F, 2)
				eta_retries = 0
				# J is fixed at this iterate; only η changes between retries, so compute SVD once.
				mcp.∇F_z!(∇F, z; θ, ϵ, η = 0.0)
				Jmat = Matrix(∇F)
				Jsvd = if use_marquardt_scaling
					d = vec(sum(abs2, Jmat; dims = 1)) # diagonal entries of Jmat' * Jmat
					d .= max.(d, 1e-16) # prevent Jscaled from having NaN or Inf entries
					Jscaled = Jmat ./ (sqrt.(d))' # J̃ = JD^(-1/2), J is (m, n), d is (n, 1)
					svd(Jscaled)
				else
					svd(Jmat)
				end
				condition_number = record_condition_number ? Jsvd.S[1] / Jsvd.S[end] : NaN
				verbose && record_condition_number && println("condition number of ∇F: ", condition_number)
				local α, pred_reduction, actual_reduction
				while true
					# filters = @. ifelse(Jsvd.S >= tsvd_threshold, Jsvd.S / (Jsvd.S^2 + max(η, 6e-5)), zero(eltype(Jsvd.S))) 
					filters = @. ifelse(
						Jsvd.S >= tsvd_threshold,
						# Jsvd.S / (Jsvd.S^2 + max(η, 6e-5)),
						Jsvd.S / (Jsvd.S^2 + η), zero(eltype(Jsvd.S)), # max(η, 1e-8)
					)
					δz .= if use_marquardt_scaling
						scaled_step = -Jsvd.V * (filters .* (Jsvd.U' * F))
						scaled_step ./ sqrt.(d) # undo the scaling
					else
						-Jsvd.V * (filters .* (Jsvd.U' * F))
						# δz .= -pinv(Jmat) * F # minimum-norm solution
					end

					α = 1.0
					@. z_trial = z + α * δz
					mcp.F!(F_trial, z_trial; θ, ϵ, η = 0.0)
					F_z_next = norm(F_trial, 2)
					while (F_z_next >= 1.0 * F_z) || (any(@. σ + α * δσ < 0) || any(@. γ + α * δγ < 0))
						if α < min_stepsize
							break # exhausted at this η — escalate below
						end

						α *= 0.5 # decay
						@. z_trial = z + α * δz
						mcp.F!(F_trial, z_trial; θ, ϵ, η = 0.0)
						F_z_next = norm(F_trial, 2)
					end

					if α >= min_stepsize
						mul!(Jδz, ∇F, δz)
						pred_reduction = F_z^2 - norm(F .+ α .* Jδz, 2)^2
						actual_reduction = F_z^2 - F_z_next^2
						break
					end

					eta_retries += 1
					if eta_retries > max_eta_retries
						verbose && @warn "Backtracking linesearch failed after $max_eta_retries η-retries. Exiting prematurely."
						status = :failed
						break
					end
					verbose && printstyled(
						"Backtracking exhausted at η=$η. Retrying with $(eta_retry_growth >= 1 ? "larger" : "smaller") η ($η -> $(η * eta_retry_growth)), attempt $eta_retries/$max_eta_retries\n";
						color = :yellow,
					)
					η = min(η * eta_retry_growth, η_max)
				end
				record_solver_diagnostics && _record_solver_diagnostic!(
					solver_diagnostics,
					solver_diagnostics_limit,
					total_iters,
					F_z,
					Jsvd.S,
					δz,
				)
				if status === :failed
					break
				end

				F .= F_trial # commit the accepted trial residual to compute kkt error at the end of this iteration

				# Levenberg-Marquardt gain-ratio update for the next Newton iteration's η.
				# https://www.cs.cornell.edu/courses/cs4220/2023sp/lec/2023-04-19.pdf
				ρ_low = 0.40
				ρ_high = 0.75
				ρ = pred_reduction > 0 ? actual_reduction / pred_reduction : -Inf
				if ρ ≤ ρ_low
					verbose && printstyled("Poor gain ratio (ρ = $ρ)... Increasing η. ($η -> $(η * (1 + exp(-loosening_rate))))\n"; color = :red)
					η = min(η * (1 + exp(-loosening_rate)), η_max) # 1 < (1 + e⁻ʳ) ≤ 2
				elseif ρ > ρ_high
					verbose && printstyled("Good gain ratio (ρ = $ρ)... Decreasing η. ($η -> $(η * (1 - exp(-tightening_rate))))\n"; color = :blue)
					η = max(η * (1 - exp(-tightening_rate)), η_min) # 0 ≤ (1 - e⁻ʳ) < 1
				end
				verbose && println("backtracking linesearch α = $α, gain ratio ρ = $ρ")
				α_σ = α
				α_γ = α
			end

			# Update variables accordingly.
			@. x += α_σ * δx
			@. s += α_σ * δs
			@. σ += α_σ * δσ
			@. γ += α_γ * δγ

			kkt_error = norm(F, 2)
			if kkt_error < best_kkt_error * (1 - stagnation_rtol)
				best_kkt_error = kkt_error
				iters_since_improvement = 0
			else
				iters_since_improvement += 1
				verbose && println("No significant improvement in KKT error for $iters_since_improvement iterations (best_kkt_error = $best_kkt_error, current kkt_error = $kkt_error).")
			end

			if perturbation_enabled && kkt_error > tol &&
			   iters_since_improvement >= stagnation_iters &&
			   #    num_perturbations < max_perturbations &&
			   kkt_error < 1.0

				verbose && println("Stagnation detected: perturbing x to escape local minimum (num_perturbations = $num_perturbations).")


				z_trial .= z
				z_trial[x_dims] .+= perturbation_scale .* randn(length(x))
				mcp.F!(F_trial, z_trial; θ, ϵ, η = 0.0)
				trial_kkt_error = norm(F_trial, 2)
				num_perturbations += 1

				if isfinite(trial_kkt_error) && all(isfinite, F_trial) &&
				   trial_kkt_error <= 1.05 * kkt_error

					z .= z_trial
					F .= F_trial
					kkt_error = trial_kkt_error
					best_kkt_error = min(best_kkt_error, kkt_error)
					iters_since_improvement = 0
					verbose && printstyled("...Applied perturbation to x; KKT error = $kkt_error\n", color = :green)
				else
					verbose && println("...Rejected perturbation; trial KKT error = $trial_kkt_error")
				end
			end

			if record_convergence
				push!(kkt_error_history, kkt_error)
			end
			if record_condition_number
				push!(condition_number_history, condition_number)
			end

			verbose && println("KKT error = $kkt_error")

			inner_iters += 1
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

	# Evaluate final convergence unconditionally — the outer_iters == max_outer_iters
	# check was unreachable when max_outer_iters=1 because the loop increments
	# outer_iters to 2 before the check.
	status = (kkt_error <= tol) ? :solved : :failed

	(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters, kkt_error_history, condition_number_history, solver_diagnostics)
end

function _record_solver_diagnostic!(diagnostics, limit, inner_iter, kkt_error, singular_values, δz)
	length(diagnostics) >= limit && return
	length(δz) >= 145 || throw(DimensionMismatch("solver diagnostics require δz to have at least 145 entries."))

	push!(
		diagnostics,
		(;
			inner_iter,
			kkt_error,
			singular_values_lt_1e_6 = count(value -> value < 1e-6, singular_values),
			singular_values_lt_1e_2 = count(value -> value < 1e-2, singular_values),
			max_singular_value = maximum(singular_values),
			min_singular_value = minimum(singular_values),
			max_abs_delta_z_1_144 = maximum(abs, @view(δz[1:144])),
			max_abs_delta_z_145_end = maximum(abs, @view(δz[145:end])),
		),
	)
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


function solve(
	::PATHSolver,
	mcp::ParametricMCP,
	θ::AbstractVector{<:Real};
	z₀ = nothing,
	options::PATHOptions,
)

	initial_guess = zeros(mcp.problem_size)
	if !isnothing(z₀)
		initial_guess[1:length(z₀)] .= z₀
	end

	z, status, info = ParametricMCPs.solve(
		mcp,
		[θ; options.ϵ₀]; # ϵ₀ relaxation parameter is embedded into θ as θ[end] or θ[sum(paramters)+1]
		initial_guess,
		# output_options = "yes",
		# output_warnings = "yes",
		cumulative_iteration_limit = options.cumulative_iteration_limit,
		proximal_perturbation = options.proximal_perturbation,
		major_iteration_limit = options.major_iteration_limit,
		minor_iteration_limit = options.minor_iteration_limit,
		convergence_tolerance = options.convergence_tolerance, #1e-1
		nms_initial_reference_factor = options.nms_initial_reference_factor,
		nms_maximum_watchdogs = options.nms_maximum_watchdogs,
		nms_memory_size = options.nms_memory_size,
		nms_mstep_frequency = options.nms_mstep_frequency,
		lemke_start_type = options.lemke_start_type,
		lemke_rank_deficiency_iterations = options.lemke_rank_deficiency_iterations,
		restart_limit = options.restart_limit,
		gradient_step_limit = options.gradient_step_limit,
		use_basics = options.use_basics,
		use_start = options.use_start,
		verbose = options.verbose,
		crash_iteration_limit = 100,
		crash_nbchange_limit = 100,
	)

	(; status, z, ϵ = options.ϵ₀, info)
end
