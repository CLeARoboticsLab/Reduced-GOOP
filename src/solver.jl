using LinearAlgebra

abstract type SolverType end
struct InteriorPoint <: SolverType end
struct PATHSolver <: SolverType end

Base.@kwdef struct InteriorPointOptions
    tol::Float64
    η₀::Float64
    ϵ₀::Union{Float64,Symbol}
    max_inner_iters::Int
    max_outer_iters::Int
    tightening_rate::Float64
    loosening_rate::Float64
    min_stepsize::Float64
    linesearch::Symbol
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
    linear_solver::Symbol = :svd
    armijo_constant::Float64 = 1e-4
    warmstart_slack_floor::Float64 = 1e-4
    warmstart_dual_floor::Float64 = 1e-4
    warmstart_dual_cap::Float64 = 1e8
    klu_singularity_eta_growth::Float64 = 100.0
    klu_singularity_max_retries::Int = 3
    reuse_factorization_iters::Int = 0
    reuse_quality_threshold::Float64 = 0.9
    verbose::Bool
end

Base.@kwdef struct PATHOptions
    convergence_tolerance::Float64
    ϵ₀::Union{Float64,Symbol}
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
	- `linear_solver::Symbol = :svd`: `:svd` (dense SVD with Tikhonov filter) or `:klu`
	  (sparse KLU on the augmented system `[I J; Jᵀ -ηI]`, same step direction, one
	  symbolic analysis then numeric-only refactorizations). `:klu` does not support
	  `record_condition_number`, `tsvd_threshold > 0`, or `use_marquardt_scaling`.
	- `armijo_constant::Float64 = 1e-4`: sufficient-decrease constant `c` of the
	  backtracking Armijo condition on φ(z) = ‖F‖²/2; `0.0` recovers the plain
	  decrease test.
	- `warmstart_slack_floor`, `warmstart_dual_floor`, `warmstart_dual_cap`: positivity
	  safeguards applied to slacks/inequality duals when `z₀` has full length
	  (`variable_dimension`), which warmstarts duals and slacks in addition to primals.
	- `klu_singularity_eta_growth::Float64 = 100.0`: factor applied to η after a
	  singular KLU factorization. The failed numeric object is discarded before
	  retrying. This is independent of `eta_retry_growth`, which handles an
	  exhausted line search.
	- `klu_singularity_max_retries::Int = 3`: maximum number of fresh KLU rebuilds
	  at successively escalated η values before using the dense-SVD fallback.
	- `reuse_factorization_iters::Int = 0`: with `:klu` + backtracking, reuse the
	  Jacobian/factorization for up to this many consecutive iterations while the
	  last step was full and reduced ‖F‖ below `reuse_quality_threshold` (modified
	  Newton); `0` disables reuse.
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
    linear_solver = options.linear_solver
    armijo_constant = options.armijo_constant
    klu_singularity_eta_growth = options.klu_singularity_eta_growth
    klu_singularity_max_retries = options.klu_singularity_max_retries
    reuse_factorization_iters = options.reuse_factorization_iters
    reuse_quality_threshold = options.reuse_quality_threshold

    linear_solver ∈ (:svd, :klu) || throw(
        ArgumentError("Unsupported linear_solver $(linear_solver). Use :svd or :klu."),
    )
    klu_singularity_eta_growth >= 1 || throw(
        ArgumentError("klu_singularity_eta_growth must be at least 1."),
    )
    klu_singularity_max_retries >= 0 || throw(
        ArgumentError("klu_singularity_max_retries must be nonnegative."),
    )
    use_klu = linear_solver === :klu
    if use_klu && (record_condition_number || tsvd_threshold > 0 || use_marquardt_scaling)
        throw(
            ArgumentError(
                "linear_solver = :klu supports neither record_condition_number, tsvd_threshold > 0, nor use_marquardt_scaling; use :svd for these features.",
            ),
        )
    end
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
            if length(z₀) == mcp.variable_dimension
                # Full warmstart: primals, equality duals, slacks, and inequality duals.
                z .= z₀
                _safeguard_warmstart!(z, mcp, options)
            elseif length(z₀) == length(mcp.primal_dims)
                z[mcp.primal_dims] .= z₀
            else
                throw(
                    ArgumentError(
                        "z₀ must have length $(mcp.variable_dimension) (full warmstart) or $(length(mcp.primal_dims)) (primal-only warmstart); got $(length(z₀)).",
                    ),
                )
            end
        end

        x_dims = Not(
            vcat(
                mcp.preference_slack_dims,
                mcp.interior_point_slack_dims,
                mcp.inequality_constraint_dual_dims,
            ),
        )
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
        # below instead of allocating temporaries. On the :klu path the augmented
        # sparse system replaces the dense buffer entirely (the rare singular
        # fallback allocates its own dense copy on the spot).
        Jdense = use_klu ? zeros(0, 0) : zeros(mcp.kkt_dimension, mcp.variable_dimension)
        jacobian_scatter_indices = use_klu ? Int[] : _dense_scatter_indices(∇F)
        aug_cache =
            use_klu ?
            _build_augmented_kkt_cache(∇F, mcp.kkt_dimension, mcp.variable_dimension) :
            nothing
        # Modified-Newton (factorization reuse) state; only active on the :klu
        # path of the backtracking branch when reuse_factorization_iters > 0.
        iters_since_jacobian = typemax(Int)  # force a fresh Jacobian on the first iteration
        last_alpha = 0.0
        last_step_quality = Inf
        η_in_K = NaN
        svd_mode_count = min(mcp.kkt_dimension, mcp.variable_dimension)
        svd_filters = zeros(svd_mode_count)
        svd_coefficients = zeros(svd_mode_count)
        range_coefficients = zeros(svd_mode_count)
        F_range = zeros(mcp.kkt_dimension)
        # Marquardt column scaling D^(-1/2): holds sqrt.(max.(diag(JᵀJ), 1e-16));
        # the reshaped row alias lets sum! reduce column norms without allocating.
        marquardt_scale = zeros(mcp.variable_dimension)
        marquardt_scale_row = reshape(marquardt_scale, 1, :)

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
        linesearch ∈ (:backtracking, :fraction_to_boundary) || throw(
            ArgumentError(
                "Unsupported linesearch $(linesearch). Use :backtracking or :fraction_to_boundary.",
            ),
        )
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
        klu_singular_retries = Ref(0)
        svd_fallback_count = Ref(0)
        klu_singularity_diagnostics = NamedTuple[]
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
                    if use_klu
                        @timeit TO "KKT system assembly" _update_augmented_kkt!(
                            aug_cache,
                            ∇F,
                            η,
                        )
                        η =
                            @timeit TO "Newton step / linear solve" _klu_step_with_fallback!(
                                δz,
                                aug_cache,
                                ∇F,
                                F,
                                η,
                                η_max,
                                verbose,
                                eta_growth = klu_singularity_eta_growth,
                                max_retries = klu_singularity_max_retries,
                                singular_retry_counter = klu_singular_retries,
                                svd_fallback_counter = svd_fallback_count,
                                singularity_diagnostics = klu_singularity_diagnostics,
                            )
                    else
                        @timeit TO "KKT system assembly" _densify!(
                            Jdense,
                            ∇F,
                            jacobian_scatter_indices,
                        )
                        Jsvd = @timeit TO "Compute SVD" _robust_svd!(
                            Jdense,
                            A -> _densify!(A, ∇F, jacobian_scatter_indices),
                        )
                        condition_number = @timeit TO "condition number evaluation" begin
                            record_condition_number ? Jsvd.S[1] / Jsvd.S[end] : NaN
                        end

                        verbose &&
                            record_condition_number &&
                            println("condition number of ∇F: ", condition_number)
                        # Unified TSVD+Tikhonov step: modes below tsvd_threshold*σ₁ are zeroed (hard cutoff),
                        # remaining modes use the Tikhonov filter σ/(σ²+η). tsvd_threshold=0 → pure Tikhonov.
                        @timeit TO "Newton step / linear solve" begin
                            threshold_abs = tsvd_threshold * Jsvd.S[1]
                            @. svd_filters = ifelse(
                                Jsvd.S >= threshold_abs,
                                Jsvd.S / (Jsvd.S^2 + η),
                                0.0,
                            )
                            mul!(svd_coefficients, Jsvd.U', F)
                            svd_coefficients .*= svd_filters
                            mul!(δz, Jsvd.V, svd_coefficients, -1.0, false)
                        end
                    end

                    @timeit TO "line search" begin
                        α_σ =
                            fraction_to_the_boundary_linesearch(σ, δσ; tol = min_stepsize)
                        α_γ =
                            fraction_to_the_boundary_linesearch(γ, δγ; tol = min_stepsize)
                    end
                    verbose &&
                        println("fraction_to_boundary linesearch α_σ = $α_σ, α_γ = $α_γ")
                    if isnan(α_σ) || isnan(α_γ)
                        verbose &&
                            @warn "Fraction-to-boundary linesearch failed. Exiting prematurely."
                        status = :failed
                        break
                    end

                    # Update regularization parameter.
                    @timeit TO "regularization" begin
                        if min(α_σ, α_γ) == 1.0
                            verbose && printstyled(
                                "Full step taken... Decreasing η. ($η -> $(η * (1 - exp(-tightening_rate * inner_iters))))\n";
                                color = :blue,
                            )
                            η *= 1 - exp(-tightening_rate)
                        else
                            verbose && printstyled(
                                "Partial step (<1.0) taken... Increasing η. ($η -> $(η * (1 + exp(-loosening_rate * inner_iters))))\n";
                                color = :red,
                            )
                            η *= 1 + exp(-loosening_rate)
                        end
                    end
                else
                    # Backtracking linesearch: if the line search exhausts at the current η,
                    # grow η and re-solve the Newton step rather than failing outright.
                    F_z = norm(F, 2)
                    eta_retries = 0
                    reused_jacobian = false
                    needs_refactor = true
                    use_range_step = false
                    local Jsvd
                    if use_klu
                        # Modified Newton: reuse the previous Jacobian (and its KLU
                        # factorization) while the last accepted step was full and made
                        # good progress; a stale direction is guarded by the line search.
                        reused_jacobian =
                            reuse_factorization_iters > 0 &&
                            iters_since_jacobian < reuse_factorization_iters &&
                            last_alpha >= 0.99 &&
                            last_step_quality <= reuse_quality_threshold
                        if !reused_jacobian
                            @timeit TO "Jacobian evaluation" mcp.∇F_z!(
                                ∇F,
                                z;
                                θ,
                                ϵ,
                                η = 0.0,
                            )
                            @timeit TO "KKT system assembly" _update_augmented_kkt!(
                                aug_cache,
                                ∇F,
                                η,
                            )
                            η_in_K = η
                            iters_since_jacobian = 0
                        elseif η != η_in_K
                            # Stale Jacobian values but a new η: rewrite the diagonal only.
                            @timeit TO "KKT system assembly" _update_augmented_eta!(
                                aug_cache,
                                η,
                            )
                            η_in_K = η
                        else
                            needs_refactor = false
                        end
                    else
                        # J is fixed at this iterate; only η changes between retries, so compute SVD once.
                        @timeit TO "Jacobian evaluation" mcp.∇F_z!(∇F, z; θ, ϵ, η = 0.0)
                        @timeit TO "KKT system assembly" _densify!(
                            Jdense,
                            ∇F,
                            jacobian_scatter_indices,
                        )
                        Jsvd = @timeit TO "Compute SVD" begin
                            if use_marquardt_scaling
                                # J̃ = JD^(-1/2), J is (m, n), d = diag(Jᵀ * J) is (n, 1), d : marquardt_scale
                                # Marquardt column scaling: normalize each Jacobian column by sqrt(diag(J'J)), with a small floor to avoid division by zero.
                                # The scaling is applied to the dense buffer in place. 
                                # On LAPACK-driver fallback the refill closure re-densifies and re-scales.
                                sum!(abs2, marquardt_scale_row, Jdense)
                                @. marquardt_scale = sqrt(max(marquardt_scale, 1e-16)) # prevent NaN or Inf entries
                                Jdense ./= marquardt_scale' # divides each column of Jdense by the corresponding scale. The transpose ' makes marquardt_scale behave like a row vector, so broadcasting works column-wise.
                                _robust_svd!(
                                    Jdense,
                                    A -> begin
                                        _densify!(A, ∇F, jacobian_scatter_indices)
                                        A ./= marquardt_scale'
                                    end,
                                )
                            else
                                _robust_svd!(
                                    Jdense,
                                    A -> _densify!(A, ∇F, jacobian_scatter_indices),
                                )
                            end
                        end
                        # Check numerical rank of Jacobian. Full row rank <=> # nonzero singular values == # rows
                        verbose && println(
                            "Numerical rank of ∇F: $(count(σ -> σ > 1e-10 * Jsvd.S[1], Jsvd.S))) / $(mcp.kkt_dimension) rows)",
                        )

                        condition_number = @timeit TO "condition number evaluation" begin
                            record_condition_number ? Jsvd.S[1] / Jsvd.S[end] : NaN
                        end

                        # Check: current residual has components outside Range(∇F): F + α∇Fδz can only modify Fᵣ, the component of F in Range(∇F).
                        # Only relevant near the solution — use_range_step is false whenever
                        # kkt_error ≥ 1e-3, so the projection is skipped entirely there.
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
                            verbose && printstyled(
                                "|F_perp|| / ||F|| = $(F_perp_norm / F_z)\n",
                                color = :green,
                            ) # large is > 0.3
                            if use_range_step
                                verbose && printstyled(
                                    "Using range-space step (Fᵣ) instead of full-space step (F) because ||F_perp|| / ||F|| = $(F_perp_norm / norm(F, 2))\n",
                                    color = :yellow,
                                )
                            end
                        end
                    end

                    local α, pred_reduction, actual_reduction, F_z_next
                    while true
                        @timeit TO "Newton step / linear solve" begin
                            if use_klu
                                η_used = _klu_step_with_fallback!(
                                    δz,
                                    aug_cache,
                                    ∇F,
                                    F,
                                    η,
                                    η_max,
                                    verbose;
                                    refactor = needs_refactor,
                                    eta_growth = klu_singularity_eta_growth,
                                    max_retries = klu_singularity_max_retries,
                                    singular_retry_counter = klu_singular_retries,
                                    svd_fallback_counter = svd_fallback_count,
                                    singularity_diagnostics = klu_singularity_diagnostics,
                                )
                                if η_used != η
                                    η = η_used
                                    η_in_K = η
                                end
                                needs_refactor = false
                            else
                                @. svd_filters = ifelse(
                                    Jsvd.S >= tsvd_threshold,
                                    (Jsvd.S / (Jsvd.S^2 + η)),
                                    0.0, # max(η, 1e-8)
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
                        end

                        @timeit TO "line search" begin
                            # Armijo condition on the merit function φ(z) = ‖F‖²/2:
                            # accept α iff φ(z + αδz) ≤ φ(z) + c·α·∇φᵀδz with ∇φᵀδz = Fᵀ(Jδz).
                            # Strict decrease alone (F_z_next < F_z) accepts arbitrarily
                            # small improvements and can grind for thousands of iterations
                            # near an unattainable tolerance. A non-descent slope (possible
                            # with a stale reused Jacobian or a heavily regularized step) is
                            # clamped to 0, which degenerates to the plain decrease test.
                            mul!(Jδz, ∇F, δz)
                            armijo_slope = min(dot(F, Jδz), 0.0)
                            α = 1.0
                            @. z_trial = z + α * δz
                            @timeit TO "residual evaluation" mcp.F!(
                                F_trial,
                                z_trial;
                                θ,
                                ϵ,
                                η = 0.0,
                            )
                            F_z_next = norm(F_trial, 2)
                            while (
                                      F_z_next^2 >=
                                      F_z^2 + 2.0 * armijo_constant * α * armijo_slope
                                  ) ||
                                  _nonnegativity_violated(σ, δσ, α) ||
                                  _nonnegativity_violated(γ, δγ, α)
                                if α < min_stepsize
                                    break # exhausted at this η — escalate below
                                end

                                α *= 0.5 # decay
                                @. z_trial = z + α * δz
                                @timeit TO "residual evaluation" mcp.F!(
                                    F_trial,
                                    z_trial;
                                    θ,
                                    ϵ,
                                    η = 0.0,
                                )
                                F_z_next = norm(F_trial, 2)
                            end
                        end

                        if α >= min_stepsize
                            @timeit TO "line search" begin
                                # Jδz already holds ∇F * δz from the Armijo slope above.
                                pred_reduction = F_z^2 - _shifted_norm2(F, α, Jδz)
                                actual_reduction = F_z^2 - F_z_next^2
                            end
                            break
                        end

                        @timeit TO "regularization" begin
                            if use_klu && reused_jacobian
                                # Line search exhausted on a stale Jacobian — refresh it at
                                # the current η before escalating the regularization.
                                verbose && printstyled(
                                    "Backtracking exhausted on a reused Jacobian. Refreshing Jacobian at η=$η.\n";
                                    color = :yellow,
                                )
                                @timeit TO "Jacobian evaluation" mcp.∇F_z!(
                                    ∇F,
                                    z;
                                    θ,
                                    ϵ,
                                    η = 0.0,
                                )
                                @timeit TO "KKT system assembly" _update_augmented_kkt!(
                                    aug_cache,
                                    ∇F,
                                    η,
                                )
                                η_in_K = η
                                iters_since_jacobian = 0
                                reused_jacobian = false
                                needs_refactor = true
                            else
                                eta_retries += 1
                                if eta_retries > max_eta_retries
                                    verbose &&
                                        @warn "Backtracking linesearch failed after $max_eta_retries η-retries. Exiting prematurely."
                                    status = :failed
                                    break
                                end
                                verbose && printstyled(
                                    "Backtracking exhausted at η=$η. Retrying with $(eta_retry_growth >= 1 ? "larger" : "smaller") η ($η -> $(η * eta_retry_growth)), attempt $eta_retries/$max_eta_retries\n";
                                    color = :yellow,
                                )
                                η = min(η * eta_retry_growth, η_max)
                                if use_klu
                                    @timeit TO "KKT system assembly" _update_augmented_eta!(
                                        aug_cache,
                                        η,
                                    )
                                    η_in_K = η
                                    needs_refactor = true
                                end
                            end
                        end
                    end
                    if status === :failed
                        break
                    end

                    F .= F_trial

                    # Levenberg-Marquardt gain-ratio update for the next Newton iteration's η.
                    # https://www.cs.cornell.edu/courses/cs4220/2023sp/lec/2023-04-19.pdf
                    if use_klu && reused_jacobian
                        # Modified-Newton iteration on a stale Jacobian: keep η frozen —
                        # the predicted reduction from stale ∇F would corrupt the gain ratio.
                        verbose && println(
                            "Reused Jacobian: skipping gain-ratio update, keeping η = $η.",
                        )
                    else
                        @timeit TO "regularization" begin
                            ρ =
                                pred_reduction > 0 ? actual_reduction / pred_reduction :
                                -Inf
                            full_step_taken = α ≥ 0.99
                            if ρ ≤ ρ_low || !full_step_taken
                                verbose && printstyled(
                                    "Poor gain ratio or backtracked step (ρ = $ρ, α = $α)... Increasing η. ($η -> $(min(η * eta_increase_factor, η_max)))\n";
                                    color = :red,
                                )
                                η = min(η * (1 + exp(-loosening_rate)), η_max) # 1 < (1 + e⁻ʳ) ≤ 2
                            elseif ρ > ρ_high
                                verbose && printstyled(
                                    "Good gain ratio on full step (ρ = $ρ)... Decreasing η. ($η -> $(max(η * eta_decrease_factor, η_min)))\n";
                                    color = :blue,
                                )
                                η = max(η * (1 - exp(-tightening_rate)), η_min) # 0 ≤ (1 - e⁻ʳ) < 1
                            else # ρ_low < ρ ≤ ρ_high on a full step: keep η
                                verbose && printstyled(
                                    "Full step with moderate gain ratio (ρ = $ρ)... Keeping η = $η.\n",
                                    color = :green,
                                )
                            end
                        end
                    end
                    if use_klu
                        last_alpha = α
                        last_step_quality = F_z_next / F_z
                        if reused_jacobian
                            iters_since_jacobian += 1
                        end
                    end
                    verbose &&
                        println("backtracking linesearch α = $α, gain ratio ρ = $ρ")
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
                        verbose && println(
                            "No significant improvement in KKT error for $iters_since_improvement iterations (best_kkt_error = $best_kkt_error, current kkt_error = $kkt_error).",
                        )
                    end
                end

                if perturbation_enabled &&
                   kkt_error > tol &&
                   iters_since_improvement >= stagnation_iters &&
                   #    num_perturbations < max_perturbations &&
                   kkt_error < 1.0
                    @timeit TO "perturbation" begin
                        verbose && println(
                            "Stagnation detected: perturbing x to escape local minimum (num_perturbations = $num_perturbations).",
                        )

                        z_trial .= z
                        z_trial[x_dims] .+= perturbation_scale .* randn(length(x))
                        @timeit TO "residual evaluation" mcp.F!(
                            F_trial,
                            z_trial;
                            θ,
                            ϵ,
                            η = 0.0,
                        )
                        trial_kkt_error = norm(F_trial, 2)
                        num_perturbations += 1

                        if isfinite(trial_kkt_error) &&
                           all(isfinite, F_trial) &&
                           trial_kkt_error <= 1.05 * kkt_error
                            z .= z_trial
                            F .= F_trial
                            kkt_error = trial_kkt_error
                            best_kkt_error = min(best_kkt_error, kkt_error)
                            iters_since_improvement = 0
                            verbose && printstyled(
                                "...Applied perturbation to x; KKT error = $kkt_error\n",
                                color = :green,
                            )
                        else
                            verbose && println(
                                "...Rejected perturbation; trial KKT error = $trial_kkt_error",
                            )
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

    result = (;
        status,
        z,
        x,
        s,
        σ,
        γ,
        kkt_error,
        ϵ,
        outer_iters,
        total_iters,
        klu_singular_retries = klu_singular_retries[],
        svd_fallback_count = svd_fallback_count[],
        klu_singularity_diagnostics,
    )
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

"Ipopt-style push of the positive variables away from the boundary before a fully warmstarted solve."
function _safeguard_warmstart!(z, mcp, options)
    κ_slack = options.warmstart_slack_floor
    κ_dual_lo = options.warmstart_dual_floor
    κ_dual_hi = options.warmstart_dual_cap
    @views begin
        z[mcp.preference_slack_dims] .= max.(z[mcp.preference_slack_dims], κ_slack)
        z[mcp.interior_point_slack_dims] .=
            max.(z[mcp.interior_point_slack_dims], κ_slack)
        z[mcp.inequality_constraint_dual_dims] .=
            clamp.(z[mcp.inequality_constraint_dual_dims], κ_dual_lo, κ_dual_hi)
    end
    z
end

"""
Sparse augmented system for the Tikhonov-regularized least-squares Newton step.

The KKT Jacobian `J` is rectangular (m×n), so the regularized step
`δz = -(JᵀJ + ηI)⁻¹JᵀF` is obtained from the square quasi-definite system in
its √η-balanced form (γ = √η, any diagonal pair with product η gives the same
δz in exact arithmetic):

	[ γI_m    J   ] [ w  ]   [ -F ]
	[  Jᵀ   -γI_n ] [ δz ] = [  0 ]

The balanced scaling keeps the matrix condition ~κ(J) even when the
regularization escalates to large η; the naive `[I J; Jᵀ -ηI]` form loses
relative accuracy in the (then ~1/η-sized) step through cancellation exactly
in the hard, heavily regularized regime, producing noisier directions than
the SVD filter there.

The sparsity pattern is fixed across iterations and η updates, so KLU's
symbolic analysis is computed once and only numeric refactorizations happen
per iteration. `J_positions`/`Jt_positions` map `nonzeros(J)` (CSC order) into
`nonzeros(K)` for the two off-diagonal blocks; `identity_positions` and
`eta_positions` address the two η-dependent diagonal blocks.
"""
struct AugmentedKKTCache
    K::SparseArrays.SparseMatrixCSC{Float64,Int}
    klu_fact::Base.RefValue{Any}
    J_positions::Vector{Int}
    Jt_positions::Vector{Int}
    identity_positions::Vector{Int}
    eta_positions::Vector{Int}
    rhs::Vector{Float64}
    sol::Vector{Float64}
    m::Int
    n::Int
end

"Build the augmented-system cache once from the fixed sparsity pattern of `∇F`."
function _build_augmented_kkt_cache(
    ∇F::SparseArrays.SparseMatrixCSC,
    m::Integer,
    n::Integer,
)
    rows, cols, _ = SparseArrays.findnz(∇F)
    nnzJ = length(rows)
    Is = Vector{Int}(undef, m + n + 2 * nnzJ)
    Js = similar(Is)
    k = 0
    for i in 1:m # I_m block
        k += 1
        Is[k] = i
        Js[k] = i
    end
    for t in 1:nnzJ # J block
        k += 1
        Is[k] = rows[t]
        Js[k] = m + cols[t]
    end
    for t in 1:nnzJ # Jᵀ block
        k += 1
        Is[k] = m + cols[t]
        Js[k] = rows[t]
    end
    for i in 1:n # -ηI_n block
        k += 1
        Is[k] = m + i
        Js[k] = m + i
    end
    # The four blocks occupy disjoint (row, col) regions, so `sparse` never merges entries.
    K = SparseArrays.sparse(Is, Js, ones(length(Is)), m + n, m + n)

    # Locate each structural entry's nzval slot (CSC-aware: J and Jᵀ orderings differ).
    K_rowvals = SparseArrays.rowvals(K)
    locate = (i, j) -> begin
        range = SparseArrays.nzrange(K, j)
        range[searchsortedfirst(view(K_rowvals, range), i)]
    end
    J_positions = [locate(rows[t], m + cols[t]) for t in 1:nnzJ]
    Jt_positions = [locate(m + cols[t], rows[t]) for t in 1:nnzJ]
    identity_positions = [locate(i, i) for i in 1:m]
    eta_positions = [locate(m + i, m + i) for i in 1:n]

    AugmentedKKTCache(
        K,
        Ref{Any}(nothing),
        J_positions,
        Jt_positions,
        identity_positions,
        eta_positions,
        zeros(m + n),
        zeros(m + n),
        m,
        n,
    )
end

"Scatter the current Jacobian values and η into the augmented matrix, without allocating."
function _update_augmented_kkt!(cache::AugmentedKKTCache, ∇F, η)
    K_nonzeros = SparseArrays.nonzeros(cache.K)
    J_nonzeros = SparseArrays.nonzeros(∇F)
    @inbounds for k in eachindex(J_nonzeros)
        K_nonzeros[cache.J_positions[k]] = J_nonzeros[k]
        K_nonzeros[cache.Jt_positions[k]] = J_nonzeros[k]
    end
    _update_augmented_eta!(cache, η)
    cache
end

"Rewrite only the η-dependent diagonals (used by the η-retry loop, where J is unchanged)."
function _update_augmented_eta!(cache::AugmentedKKTCache, η)
    K_nonzeros = SparseArrays.nonzeros(cache.K)
    # Balanced scaling γ = √η with a floor keeping both diagonal blocks
    # strictly definite even if η underflows to 0.
    γ = sqrt(max(η, 1e-12))
    @inbounds for position in cache.identity_positions
        K_nonzeros[position] = γ
    end
    @inbounds for position in cache.eta_positions
        K_nonzeros[position] = -γ
    end
    cache
end

"""
Solve the augmented system for `δz` with rhs `[-F; 0]`. The first call performs
KLU's symbolic analysis; later calls refactorize numerically in place (skipped
entirely when `refactor = false`, i.e. modified-Newton reuse).
"""
function _solve_augmented!(δz, cache::AugmentedKKTCache, F; refactor = true)
    @views cache.rhs[1:(cache.m)] .= .-F
    fill!(view(cache.rhs, (cache.m + 1):(cache.m + cache.n)), 0.0)
    if cache.klu_fact[] === nothing
        cache.klu_fact[] = @timeit TO "KLU refactorization" KLU.klu(cache.K)
    elseif refactor
        @timeit TO "KLU refactorization" KLU.klu!(cache.klu_fact[], cache.K)
    end
    cache.sol .= cache.rhs
    ldiv!(cache.klu_fact[], cache.sol)
    δz .= view(cache.sol, (cache.m + 1):(cache.m + cache.n)) # copy last n elements
    δz
end

"""
KLU step with recovery and escalation: on a singular factorization, discard
the failed numeric object, grow η, and rebuild so KLU can choose new numeric
pivots. Falls back to a dense SVD only after trying the final escalated η.
Returns the η actually used, so the caller can keep its regularization
bookkeeping consistent.
"""
function _klu_step_with_fallback!(
    δz,
    cache::AugmentedKKTCache,
    ∇F,
    F,
    η,
    η_max,
    verbose;
    refactor = true,
    eta_growth = 100.0,
    max_retries = 3,
    singular_retry_counter = nothing,
    svd_fallback_counter = nothing,
    singularity_diagnostics = nothing,
)
    η_used = η
    needs_refactor = refactor
    # Attempts include the original η plus `max_retries` successively escalated
    # values. The previous fixed loop escalated after its last failure but never
    # actually tried the final η before invoking dense SVD.
    for attempt in 0:max_retries
        had_numeric_factorization = !isnothing(cache.klu_fact[])
        try
            _solve_augmented!(δz, cache, F; refactor = needs_refactor)
            return η_used
        catch error
            error isa LinearAlgebra.SingularException || rethrow()
            !isnothing(singular_retry_counter) && (singular_retry_counter[] += 1)
            if !isnothing(singularity_diagnostics)
                jacobian_values = SparseArrays.nonzeros(∇F)
                max_abs = maximum(abs, jacobian_values; init = 0.0)
                min_abs_nonzero = minimum(
                    (abs(value) for value in jacobian_values if !iszero(value));
                    init = Inf,
                )
                push!(
                    singularity_diagnostics,
                    (;
                        η = η_used,
                        jacobian_max_abs = max_abs,
                        jacobian_min_abs_nonzero = min_abs_nonzero,
                        attempted_inplace_refactor = had_numeric_factorization,
                    ),
                )
            end
            # KLU's failed in-place numeric refactorization is not guaranteed to
            # remain reusable. Force a fresh factorization on the next attempt.
            cache.klu_fact[] = nothing
            attempt == max_retries && break
            η_used = min(max(η_used, 1e-8) * eta_growth, η_max)
            verbose && printstyled(
                "KLU factorization singular. Retrying with η = $η_used.\n";
                color = :yellow,
            )
            _update_augmented_eta!(cache, η_used)
            needs_refactor = true
        end
    end
    verbose &&
        @warn "KLU factorization singular after η escalation; falling back to a dense SVD step for this iteration."
    !isnothing(svd_fallback_counter) && (svd_fallback_counter[] += 1)
    _svd_fallback_step!(δz, ∇F, F, η_used)
    η_used
end

"Rare-path dense Tikhonov step used when KLU cannot factorize the augmented system."
function _svd_fallback_step!(δz, ∇F, F, η)
    Jsvd = _robust_svd(Matrix(∇F))
    coefficients = Jsvd.U' * F
    @. coefficients *= Jsvd.S / (Jsvd.S^2 + η)
    mul!(δz, Jsvd.V, coefficients, -1.0, false)
    δz
end

"""Helper function to compute the step size `α` which solves:
				   α* = max(α ∈ [0, 1] : v + α δ ≥ (1 - τ) v).
"""
function fraction_to_the_boundary_linesearch(
    v,
    δ;
    max_stepsize = 1.0,
    τ = 0.995,
    decay = 0.5,
    tol = 1e-4,
)
    α = max_stepsize
    while any(@. v + α * δ < (1 - τ) * v)
        if α < tol
            return NaN
        end

        α *= decay
    end

    α
end
