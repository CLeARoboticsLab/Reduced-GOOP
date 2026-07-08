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
	record_convergence::Bool = false
	record_condition_number::Bool = false
	max_eta_retries::Int = 5
	eta_retry_growth::Float64 = 2.0
	η_min::Float64 = 0.0
	η_max::Float64 = 1e-1
	ρ_low::Float64 = 0.75
	ρ_high::Float64 = 0.75
	eta_increase_factor::Float64 = 2.0
	eta_decrease_factor::Float64 = 0.5
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
	- `record_convergence::Bool = false`: if true, record and return `kkt_error_history` and `eta_history`.
	- `record_condition_number::Bool = false`: if true, record and return `condition_number_history`.
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
	verbose = options.verbose
	max_eta_retries = options.max_eta_retries
	eta_retry_growth = options.eta_retry_growth
	η_min = options.η_min
	η_max = options.η_max
	ρ_low = options.ρ_low
	ρ_high = options.ρ_high
	eta_increase_factor = options.eta_increase_factor
	eta_decrease_factor = options.eta_decrease_factor
	perturbation_enabled = options.perturbation_enabled
	stagnation_iters = options.stagnation_iters
	stagnation_rtol = options.stagnation_rtol
	perturbation_scale = options.perturbation_scale
	max_perturbations = options.max_perturbations
	tsvd_threshold = options.tsvd_threshold
	use_marquardt_scaling = options.use_marquardt_scaling
	@timeit TO "solver initialization" begin
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

		# Preallocated per-iteration workspaces. The sparse Jacobian structure is
		# fixed at construction, so the dense SVD buffer is refilled in place via
		# precomputed scatter indices instead of allocating Matrix(∇F) each
		# iteration; the SVD step and range-space projection reuse the vectors
		# below instead of allocating temporaries.
		Jdense = zeros(mcp.kkt_dimension, mcp.variable_dimension)
		jacobian_scatter_indices = _dense_scatter_indices(∇F)
		svd_mode_count = min(mcp.kkt_dimension, mcp.variable_dimension)
		svd_filters = zeros(svd_mode_count)
		svd_coefficients = zeros(svd_mode_count)
		range_coefficients = zeros(svd_mode_count)
		F_range = zeros(mcp.kkt_dimension)
		# Marquardt column scaling D^(-1/2): holds sqrt.(max.(diag(JᵀJ), 1e-16));
		# the reshaped row alias lets sum! reduce column norms without allocating.
		marquardt_scale = zeros(mcp.variable_dimension)
		marquardt_scale_row = reshape(marquardt_scale, 1, :)

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
		eta_history = Float64[]
		alpha_history = Float64[]
		rho_history = Float64[]
	end
	while outer_iters < max_outer_iters || iszero(total_iters)
		inner_iters = 1
		status = :solved

		verbose && @info "Outer iteration $(outer_iters): ϵ = $ϵ, kkt_error = $kkt_error"

		while inner_iters < max_inner_iters && (kkt_error > tol) # (!is_fraction_to_boundary_linesearch || kkt_error > tol)
			@timeit TO "inner iteration loop" begin
				total_iters += 1
				# Compute the residual at the current iterate
				@timeit TO "residual evaluation" mcp.F!(F, z; θ, ϵ, η = 0.0)
				# @assert all(.!isnan.(F)) "Found NaN in F - aborting!"
				verbose && println("inner iter $inner_iters")
				condition_number = NaN
				# Gain ratio of the accepted step; stays NaN on paths that do not
				# compute it (fraction-to-boundary linesearch).
				ρ = NaN

				if linesearch == :fraction_to_boundary
					@timeit TO "Jacobian evaluation" mcp.∇F_z!(∇F, z; θ, ϵ, η = 0.0)
					@timeit TO "KKT system assembly" _densify!(Jdense, ∇F, jacobian_scatter_indices)
					Jsvd = @timeit TO "Newton step / linear solve" _robust_svd!(
						Jdense,
						A -> _densify!(A, ∇F, jacobian_scatter_indices),
					)
					condition_number = @timeit TO "condition number evaluation" begin
						record_condition_number ? Jsvd.S[1] / Jsvd.S[end] : NaN
					end

					verbose && record_condition_number && println("condition number of ∇F: ", condition_number)
					# Unified TSVD+Tikhonov step: modes below tsvd_threshold*σ₁ are zeroed (hard cutoff),
					# remaining modes use the Tikhonov filter σ/(σ²+η). tsvd_threshold=0 → pure Tikhonov.
					@timeit TO "Newton step / linear solve" begin
						threshold_abs = tsvd_threshold * Jsvd.S[1]
						@. svd_filters = ifelse(Jsvd.S >= threshold_abs, Jsvd.S / (Jsvd.S^2 + η), 0.0)
						mul!(svd_coefficients, Jsvd.U', F)
						svd_coefficients .*= svd_filters
						mul!(δz, Jsvd.V, svd_coefficients, -1.0, false)
					end

					@timeit TO "line search" begin
						α_σ = fraction_to_the_boundary_linesearch(σ, δσ; tol = min_stepsize)
						α_γ = fraction_to_the_boundary_linesearch(γ, δγ; tol = min_stepsize)
					end
					verbose && println("fraction_to_boundary linesearch α_σ = $α_σ, α_γ = $α_γ")
					if isnan(α_σ) || isnan(α_γ)
						verbose && @warn "Fraction-to-boundary linesearch failed. Exiting prematurely."
						status = :failed
						break
					end

					# Update regularization parameter.
					@timeit TO "regularization" begin
						if min(α_σ, α_γ) == 1.0
							verbose && printstyled("Full step taken... Decreasing η. ($η -> $(η * (1 - exp(-tightening_rate * inner_iters))))\n"; color = :blue)
							η *= 1 - exp(-tightening_rate)
						else
							verbose && printstyled("Partial step (<1.0) taken... Increasing η. ($η -> $(η * (1 + exp(-loosening_rate * inner_iters))))\n"; color = :red)
							η *= 1 + exp(-loosening_rate)
						end
					end
				else
					# Backtracking linesearch: if the line search exhausts at the current η,
					# grow η and re-solve the Newton step rather than failing outright.
					F_z = norm(F, 2)
					eta_retries = 0
					# J is fixed at this iterate; only η changes between retries, so compute SVD once.
					@timeit TO "Jacobian evaluation" mcp.∇F_z!(∇F, z; θ, ϵ, η = 0.0)
					@timeit TO "KKT system assembly" _densify!(Jdense, ∇F, jacobian_scatter_indices)
					Jsvd = @timeit TO "Newton step / linear solve" begin
						if use_marquardt_scaling
							# J̃ = JD^(-1/2), J is (m, n), d = diag(Jᵀ * J) is (n, 1), d : marquardt_scale
							# Marquardt column scaling: normalize each Jacobian column by sqrt(diag(J'J)), with a small floor to avoid division by zero.
							# The scaling is applied to the dense buffer in place. 
							# On LAPACK-driver fallback the refill closure re-densifies and re-scales.
							sum!(abs2, marquardt_scale_row, Jdense)
							@. marquardt_scale = sqrt(max(marquardt_scale, 1e-16)) # prevent NaN or Inf entries
							Jdense ./= marquardt_scale' # Divide each column of Jdense by the corresponding scale. The transpose ' makes marquardt_scale behave like a row vector, so broadcasting works column-wise.
							_robust_svd!(
								Jdense,
								A -> begin
									_densify!(A, ∇F, jacobian_scatter_indices)
									A ./= marquardt_scale'
								end,
							)
						else
							_robust_svd!(Jdense, A -> _densify!(A, ∇F, jacobian_scatter_indices))
						end
					end
					# Check numerical rank of Jacobian. Full row rank <=> # nonzero singular values == # rows
					verbose && println("Numerical rank of ∇F: $(count(σ -> σ > 1e-10 * Jsvd.S[1], Jsvd.S))) / $(mcp.kkt_dimension) rows)")

					condition_number = @timeit TO "condition number evaluation" begin
						record_condition_number ? Jsvd.S[1] / Jsvd.S[end] : NaN
					end

					# Check: current residual has components outside Range(∇F): F + α∇Fδz can only modify Fᵣ, the component of F in Range(∇F).
					# Only relevant near the solution — use_range_step is false whenever
					# kkt_error ≥ 1e-3, so the projection is skipped entirely there.
					use_range_step = false
					if kkt_error < 1e-3
						@timeit TO "range-space residual projection" begin
							τ = 1e-8 * maximum(Jsvd.S)
							r = count(>(τ), Jsvd.S)
							Uᵣ = @view Jsvd.U[:, 1:r]
							range_coefficients_r = @view range_coefficients[1:r]
							mul!(range_coefficients_r, Uᵣ', F)
							mul!(F_range, Uᵣ, range_coefficients_r)
							F_perp_norm = sqrt(_shifted_norm2(F, -1.0, F_range))
							# Compute newton step using Fᵣ
							use_range_step = (F_perp_norm / norm(F, 2)) > 0.3
						end
						verbose && printstyled("|F_perp|| / ||F|| = $(F_perp_norm / F_z)\n", color = :green) # large is > 0.3
						if use_range_step
							verbose && printstyled("Using range-space step (Fᵣ) instead of full-space step (F) because ||F_perp|| / ||F|| = $(F_perp_norm / norm(F, 2))\n", color = :yellow)
						end
					end

					# if inner_iters > 20 && kkt_error < 5e-3
					# 	η = 0.0
					# end

					local α, pred_reduction, actual_reduction
					while true
						@timeit TO "Newton step / linear solve" begin
							@. svd_filters = ifelse(
								Jsvd.S >= tsvd_threshold,
								(Jsvd.S / (Jsvd.S^2 + η)), 0.0, # max(η, 1e-8)
							)
							residual = use_range_step ? F_range : F
							mul!(svd_coefficients, Jsvd.U', residual)
							svd_coefficients .*= svd_filters
							mul!(δz, Jsvd.V, svd_coefficients, -1.0, false) # # Compute δz = -V * svd_coefficients in place, discarding the old δz.
							# -pinv(Jmat; atol=1e-8, rtol = sqrt(eps(real(float(oneunit(eltype(Jmat))))))) * residual # minimum-norm solution
							if use_marquardt_scaling
								δz ./= marquardt_scale # undo the D^(-1/2) scaling
							end
						end

						@timeit TO "line search" begin
							α = 1.0
							@. z_trial = z + α * δz
							@timeit TO "residual evaluation" mcp.F!(F_trial, z_trial; θ, ϵ, η = 0.0)
							F_z_next = norm(F_trial, 2)
							while (F_z_next >= 1.0 * F_z) ||
									  _nonnegativity_violated(σ, δσ, α) ||
									  _nonnegativity_violated(γ, δγ, α)
								if α < min_stepsize
									break # exhausted at this η — escalate below
								end

								α *= 0.5 # decay
								@. z_trial = z + α * δz
								@timeit TO "residual evaluation" mcp.F!(F_trial, z_trial; θ, ϵ, η = 0.0)
								F_z_next = norm(F_trial, 2)
							end
						end

						if α >= min_stepsize
							@timeit TO "line search" begin
								mul!(Jδz, ∇F, δz)
								pred_reduction = F_z^2 - _shifted_norm2(F, α, Jδz)
								actual_reduction = F_z^2 - F_z_next^2
							end
							break
						end

						@timeit TO "regularization" begin
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
					end
					if status === :failed
						break
					end

					F .= F_trial

					# Levenberg-Marquardt gain-ratio update for the next Newton iteration's η.
					# https://www.cs.cornell.edu/courses/cs4220/2023sp/lec/2023-04-19.pdf
					@timeit TO "regularization" begin
						ρ = pred_reduction > 0 ? actual_reduction / pred_reduction : -Inf
						full_step_taken = α ≥ 0.99
						if ρ ≤ ρ_low || !full_step_taken
							verbose && printstyled("Poor gain ratio or backtracked step (ρ = $ρ, α = $α)... Increasing η. ($η -> $(min(η * eta_increase_factor, η_max)))\n"; color = :red)
							η = min(η * (1 + exp(-loosening_rate)), η_max) # 1 < (1 + e⁻ʳ) ≤ 2
						elseif ρ > ρ_high
							verbose && printstyled("Good gain ratio on full step (ρ = $ρ)... Decreasing η. ($η -> $(max(η * eta_decrease_factor, η_min)))\n"; color = :blue)
							η = max(η * (1 - exp(-tightening_rate)), η_min) # 0 ≤ (1 - e⁻ʳ) < 1
						else # ρ_low < ρ ≤ ρ_high on a full step: keep η
							verbose && printstyled("Full step with moderate gain ratio (ρ = $ρ)... Keeping η = $η.\n", color = :green)
						end
					end
					verbose && println("backtracking linesearch α = $α, gain ratio ρ = $ρ")
					α_σ = α
					α_γ = α
				end

				# Update variables accordingly.
				@timeit TO "iterate update and bookkeeping" begin
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
				end

				if perturbation_enabled && kkt_error > tol &&
				   iters_since_improvement >= stagnation_iters &&
				   #    num_perturbations < max_perturbations &&
				   kkt_error < 1.0

					@timeit TO "perturbation" begin
						verbose && println("Stagnation detected: perturbing x to escape local minimum (num_perturbations = $num_perturbations).")


						z_trial .= z
						z_trial[x_dims] .+= perturbation_scale .* randn(length(x))
						@timeit TO "residual evaluation" mcp.F!(F_trial, z_trial; θ, ϵ, η = 0.0)
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
				end

				@timeit TO "iterate update and bookkeeping" begin
					if record_convergence
						push!(kkt_error_history, kkt_error)
						push!(eta_history, η)
						push!(alpha_history, min(α_σ, α_γ))
						push!(rho_history, ρ)
					end
					if record_condition_number
						push!(condition_number_history, condition_number)
					end
				end

				verbose && println("KKT error = $kkt_error")

				inner_iters += 1
			end
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

	result = (; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters)
	if record_convergence || record_condition_number
		return (;
			result...,
			kkt_error_history,
			condition_number_history,
			eta_history,
			alpha_history,
			rho_history,
		)
	end
	result
end

"""
SVD with a driver fallback: the default divide-and-conquer LAPACK driver
(`gesdd`) can throw `LAPACKException` on severely ill-conditioned matrices;
retry with the slower but more robust QR-iteration driver (`gesvd`).
"""
function _robust_svd(A)
	try
		svd(A)
	catch error
		error isa LinearAlgebra.LAPACKException || rethrow()
		svd(A; alg = LinearAlgebra.QRIteration())
	end
end

"""
In-place variant of [`_robust_svd`](@ref) for a preallocated buffer: `svd!`
destroys `A`, so on driver failure `refill!` must restore the buffer contents
before the QR-iteration retry.
"""
function _robust_svd!(A, refill!)
	try
		svd!(A)
	catch error
		error isa LinearAlgebra.LAPACKException || rethrow()
		svd!(refill!(A); alg = LinearAlgebra.QRIteration())
	end
end

"Linear indices of a sparse matrix's stored entries in its dense counterpart, in `nonzeros` order."
function _dense_scatter_indices(sparse_matrix)
	num_rows = size(sparse_matrix, 1)
	rows = SparseArrays.rowvals(sparse_matrix)
	indices = Vector{Int}(undef, SparseArrays.nnz(sparse_matrix))
	for j in axes(sparse_matrix, 2)
		for k in SparseArrays.nzrange(sparse_matrix, j)
			indices[k] = (j - 1) * num_rows + rows[k]
		end
	end
	indices
end

"Refill dense `dest` from sparse `src` using precomputed `scatter_indices`, without allocating."
function _densify!(dest, src, scatter_indices)
	fill!(dest, zero(eltype(dest)))
	stored_values = SparseArrays.nonzeros(src)
	@inbounds for k in eachindex(scatter_indices, stored_values)
		dest[scatter_indices[k]] = stored_values[k]
	end
	dest
end

"Allocation-free check for `any(values .+ α .* direction .< 0)`."
function _nonnegativity_violated(values, direction, α)
	@inbounds for i in eachindex(values, direction)
		if values[i] + α * direction[i] < 0
			return true
		end
	end
	false
end

"Allocation-free `norm(a .+ scale .* b, 2)^2`."
function _shifted_norm2(a, scale, b)
	accumulator = zero(promote_type(eltype(a), eltype(b)))
	@inbounds for i in eachindex(a, b)
		value = a[i] + scale * b[i]
		accumulator += value * value
	end
	accumulator
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
