using LinearAlgebra

abstract type SolverType end
struct InteriorPoint <: SolverType end

Base.@kwdef struct InteriorPointOptions
    tol::Float64
    η₀::Float64
    ϵ₀::Union{Float64,Symbol}
    max_inner_iters::Int
    max_outer_iters::Int
    tightening_rate::Float64
    loosening_rate::Float64
    min_stepsize::Float64
    linesearch::Symbol = :backtracking
    record_convergence::Bool = false
    record_condition_number::Bool = false
    max_eta_retries::Int = 5
    eta_retry_growth::Float64 = 2.0
    η_min::Float64 = 0.0
    η_max::Float64 = 1e-1
    ρ_low::Float64 = 0.75
    ρ_high::Float64 = 0.75
    tsvd_threshold::Float64 = 0.0
    use_marquardt_scaling::Bool = false
    linear_solver::Symbol = :klu
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
    - `z₀`: optional primal-only or full-length warm start.
    - `options::InteriorPointOptions`: solver and diagnostic settings.

Selected `InteriorPointOptions` fields:
    - `record_convergence`: record KKT-error, η, raw direction norm, accepted
      step-size, and gain-ratio histories.
    - `record_condition_number`: record dense-SVD condition-number history.
    - `linear_solver::Symbol = :klu`: `:svd` (dense SVD with Tikhonov filter), `:klu`
      (sparse KLU on the balanced augmented system `[√ηI J; Jᵀ -√ηI]`, with one
      symbolic analysis followed by numeric refactorizations; η is the Tikhonov
      shift), its explicit alias `:klu_sqrt_eta`, or `:klu_eta` (the same KLU path
      on `[ηI J; Jᵀ -ηI]`, so the shift is η², the convention of the Scholtes
      `:normal` backend). The KLU variants do not support `record_condition_number`,
      `tsvd_threshold > 0`, or `use_marquardt_scaling`.
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
    tsvd_threshold = options.tsvd_threshold
    use_marquardt_scaling = options.use_marquardt_scaling
    linear_solver = options.linear_solver
    armijo_constant = options.armijo_constant
    klu_singularity_eta_growth = options.klu_singularity_eta_growth
    klu_singularity_max_retries = options.klu_singularity_max_retries
    reuse_factorization_iters = options.reuse_factorization_iters
    reuse_quality_threshold = options.reuse_quality_threshold

    linear_solver ∈ (:svd, :klu, :klu_sqrt_eta, :klu_eta) || throw(
        ArgumentError(
            "Unsupported linear_solver $(linear_solver). Use :svd, :klu, :klu_sqrt_eta or :klu_eta.",
        ),
    )
    klu_singularity_eta_growth >= 1 ||
        throw(ArgumentError("klu_singularity_eta_growth must be at least 1."))
    klu_singularity_max_retries >= 0 ||
        throw(ArgumentError("klu_singularity_max_retries must be nonnegative."))
    use_klu = linear_solver !== :svd
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
            _build_augmented_kkt_cache(
                ∇F,
                mcp.kkt_dimension,
                mcp.variable_dimension;
                sqrt_eta = linear_solver !== :klu_eta,
            ) : nothing
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
        stopping_criterion = false
        is_fraction_to_boundary_linesearch = (linesearch == :fraction_to_boundary)
        kkt_error_history = Float64[]
        condition_number_history = Float64[]
        eta_history = Float64[]
        alpha_history = Float64[]
        delta_z_norm_history = Float64[]
        rho_history = Float64[]
        klu_singular_retries = Ref(0)
        svd_fallback_count = Ref(0)
    end
    while outer_iters < max_outer_iters || iszero(total_iters)
        inner_iters = 1
        status = :solved

        verbose && @info "Outer iteration $(outer_iters): ϵ = $ϵ, kkt_error = $kkt_error"

        while inner_iters < max_inner_iters && kkt_error > tol
            @timeit TO "inner iteration loop" begin
                total_iters += 1
                # Compute the residual at the current iterate
                @timeit TO "residual evaluation" mcp.F!(F, z; θ, ρ = ϵ)
                kkt_error = norm(F, 2)
                stopping_criterion = kkt_error <= tol
                stopping_criterion && break
                # @assert all(.!isnan.(F)) "Found NaN in F - aborting!"
                verbose && println("inner iter $inner_iters")
                condition_number = NaN
                # Gain ratio of the accepted step; stays NaN on paths that do not
                # compute it (fraction-to-boundary linesearch).
                ρ = NaN

                if linesearch == :fraction_to_boundary
                    @timeit TO "Jacobian evaluation" mcp.∇F_z!(∇F, z; θ, ρ = ϵ)
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
                    F_z = kkt_error
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
                                ρ = ϵ,
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
                        @timeit TO "Jacobian evaluation" mcp.∇F_z!(∇F, z; θ, ρ = ϵ)
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

                    current_merit2 = F_z^2
                    local α, pred_reduction, actual_reduction, F_z_next, trial_merit2
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
                            mul!(Jδz, ∇F, δz)
                            armijo_slope = min(dot(F, Jδz), 0.0)
                            α = 1.0
                            @. z_trial = z + α * δz
                            @timeit TO "residual evaluation" mcp.F!(
                                F_trial,
                                z_trial;
                                θ,
                                ρ = ϵ,
                            )
                            F_z_next = norm(F_trial, 2)
                            trial_merit2 = F_z_next^2
                            armijo_failed =
                                F_z_next^2 >= F_z^2 + 2.0 * armijo_constant * α * armijo_slope
                            while armijo_failed ||
                                      _nonnegativity_violated(σ, δσ, α) ||
                                      _nonnegativity_violated(γ, δγ, α)
                                if α < min_stepsize
                                    break # exhausted at this η — escalate below
                                end

                                α *= 0.5
                                @. z_trial = z + α * δz
                                @timeit TO "residual evaluation" mcp.F!(
                                    F_trial,
                                    z_trial;
                                    θ,
                                    ρ = ϵ,
                                )
                                F_z_next = norm(F_trial, 2)
                                trial_merit2 = F_z_next^2
                                armijo_failed =
                                    F_z_next^2 >=
                                    F_z^2 + 2.0 * armijo_constant * α * armijo_slope
                            end
                        end

                        if α >= min_stepsize
                            @timeit TO "line search" begin
                                pred_reduction =
                                    current_merit2 - _shifted_norm2(F, α, Jδz)
                                actual_reduction = current_merit2 - trial_merit2
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
                                    ρ = ϵ,
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
                                    "Poor gain ratio or backtracked step (ρ = $ρ, α = $α)... Increasing η. ($η -> $(min(η * (1 + exp(-loosening_rate)), η_max)))\n";
                                    color = :red,
                                )
                                η = min(η * (1 + exp(-loosening_rate)), η_max) # 1 < (1 + e⁻ʳ) ≤ 2
                            elseif ρ > ρ_high
                                verbose && printstyled(
                                    "Good gain ratio on full step (ρ = $ρ)... Decreasing η. ($η -> $(max(η * (1 - exp(-tightening_rate)), η_min)))\n";
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

                    if is_fraction_to_boundary_linesearch
                        @timeit TO "residual evaluation" mcp.F!(F, z; θ, ρ = ϵ)
                    else
                        F .= F_trial # backtracking linesearch already evaluated the accepted point so reuse
                    end

                    kkt_error = norm(F, 2)
                end

                @timeit TO "iterate update and bookkeeping" begin
                    if record_convergence
                        push!(kkt_error_history, kkt_error)
                        push!(eta_history, η)
                        push!(alpha_history, min(α_σ, α_γ))
                        push!(delta_z_norm_history, norm(δz, 2))
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

        if status === :solved
            break
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

    # Evaluate stopping criterion/convergence at the returned iterate.
    @timeit TO "residual evaluation" mcp.F!(F, z; θ, ρ = ϵ)
    kkt_error = norm(F, 2)
    status = kkt_error <= tol ? :solved : :failed

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
    )
    if record_convergence || record_condition_number
        return (;
            result...,
            kkt_error_history,
            condition_number_history,
            eta_history,
            alpha_history,
            delta_z_norm_history,
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

`sqrt_eta` selects what η means. `true` (`:klu`, `:klu_sqrt_eta`): the diagonals are
`±√η`, so η is the Tikhonov shift, `δz = -(JᵀJ + ηI)⁻¹JᵀF`. `false` (`:klu_eta`):
the diagonals are `±η`, so the shift is η², `δz = -(JᵀJ + η²I)⁻¹JᵀF`, the convention
of the `:normal` Scholtes backend and of ScholtesReducedGOOP.jl's `:klu`.
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
    sqrt_eta::Bool
end

"Build the augmented-system cache once from the fixed sparsity pattern of `∇F`."
function _build_augmented_kkt_cache(
    ∇F::SparseArrays.SparseMatrixCSC,
    m::Integer,
    n::Integer;
    sqrt_eta::Bool = true,
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
        sqrt_eta,
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
    # Balanced scaling γ = √η (or γ = η, see `sqrt_eta`) with a floor keeping both
    # diagonal blocks strictly definite even if η underflows to 0.
    γ = cache.sqrt_eta ? sqrt(max(η, 1e-12)) : max(η, 1e-12)
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
)
    η_used = η
    needs_refactor = refactor
    # Attempts include the original η plus `max_retries` successively escalated
    # values. The previous fixed loop escalated after its last failure but never
    # actually tried the final η before invoking dense SVD.
    for attempt in 0:max_retries
        try
            _solve_augmented!(δz, cache, F; refactor = needs_refactor)
            return η_used
        catch error
            error isa LinearAlgebra.SingularException || rethrow()
            !isnothing(singular_retry_counter) && (singular_retry_counter[] += 1)
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
    # The dense step uses the same effective Tikhonov shift as the factorization it
    # replaces: η for `sqrt_eta = true`, η² for `sqrt_eta = false`.
    shift = cache.sqrt_eta ? max(η_used, 1e-12) : max(η_used, 1e-12)^2
    _svd_fallback_step!(δz, ∇F, F, shift)
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

"""
Solver tag for the primal-dual interior point method: 
    `solve(PDIP(), kkt, θ; z₀, options = PDIPOptions(...))`.
"""
struct PDIP <: SolverType end

"""
Options for `solve(PDIP(), …)`.

  - `ρ_schedule`: the relaxation of each level. With `linesearch = :backtracking` the levels are
    solved in turn. With `:fraction_to_boundary`, ρ starts at `first(ρ_schedule)` and, as in the old
    interior-point solver, is multiplied by `1 - exp(-tightening_rate)` after a solved level and by
    `1 + exp(-loosening_rate)` after an unsolved one (capped at 1), for at most `max_outer_iters` levels.
  - `max_inner_iters`: Newton steps per level (Algorithm 5.1's t_max).
  - `tol`: bar on ‖K₀‖ = ‖F(z; ρ = 0)‖, checked before every Newton step; below it the solve stops.
    `tol_inner`: bar on ‖F(z; ρ)‖; below it the level is solved.
  - `linesearch`: `:backtracking` (Algorithm 5.1: α ← β α while ‖F(z + α δz; ρ)‖ ≥ ‖F(z; ρ)‖ or a
    coordinate in `nonnegative_dims` would not stay positive, so that an accepted step decreases ‖F‖
    strictly; the step fails below `min_step`) or `:fraction_to_boundary` (the old
    solver's τ-rule: γ moves by α_γ, every other variable by α_σ, with no decrease test).
  - `linear_solver`: `:svd` (default), `:sparseQR_J`, `:sparseQR_Jᵀ`, `:klu` or `:ldl`; see
    `_newton_step!`.
  - `η > 0`: the shift of `:klu` and `:ldl`, whose step is −Jᵀ(JJᵀ + ηI)⁻¹F (→ −J⁺F as η → 0).
  - `record_convergence`: also return the per-iteration histories.
"""
Base.@kwdef struct PDIPOptions
    ρ_schedule::Vector{Float64} = [10.0^(-k) for k in 0:10]
    max_inner_iters::Int = 200
    max_outer_iters::Int = 20
    tol::Float64 = 1e-8
    tol_inner::Float64 = 1e-11
    linesearch::Symbol = :backtracking
    linear_solver::Symbol = :svd
    η::Float64 = 1e-12
    β::Float64 = 0.5
    min_step::Float64 = 1e-14
    τ::Float64 = 0.995
    tightening_rate::Float64 = 0.1
    loosening_rate::Float64 = 0.5
    record_convergence::Bool = false
    verbose::Bool = false
end

"""
    solve(PDIP(), kkt::GOOPKKTSystem, θ; z₀ = nothing, options = PDIPOptions())

Algorithm 5.1 of manuscript/main.pdf: Newton's method on F(z; ρ) = 0 for each relaxation ρ, with the
step J δz = −F (J = ∇F) and a line search that keeps `kkt.nonnegative_dims` positive: the slacks σ
and the inequality duals γ, of the reduced or the complete system. The preference slacks s are not
bounded directly: s ≥ 0 is a row of G (s − σ = 0).

`z₀` is a full iterate (length `kkt.variable_dimension`, positive on `nonnegative_dims`), 
a primal one (length of `kkt.primal_dims`; then s, σ and γ start at 1 and every other variable at 0), or
`nothing` (the same start with x = 0).

Returns `(; status, termination_reason, z, x, ρ, residual, relaxed_residual, newton_steps,
residual_evaluations, svd_fallbacks)`, and with `record_convergence` also `ρ_history`,
`relaxed_residual_history`, `residual_history` (one entry per residual evaluation) and
`step_size_history` (one per Newton step). `residual` is ‖K₀‖ = ‖F(z; ρ = 0)‖ at the returned `z`,
`status` is `:solved` iff it is below `tol`, and `relaxed_residual` is ‖F(z; ρ)‖ at the last level.
"""
function solve(
    ::PDIP,
    kkt::GOOPKKTSystem,
    θ::AbstractVector{<:Real};
    z₀ = nothing,
    options::PDIPOptions = PDIPOptions(),
)
    (; ρ_schedule, max_inner_iters, max_outer_iters, tol, tol_inner, linesearch, linear_solver, η) =
        options
    (; β, min_step, τ, tightening_rate, loosening_rate, record_convergence, verbose) = options
    linesearch ∈ (:backtracking, :fraction_to_boundary) || throw(
        ArgumentError(
            "Unsupported linesearch $(linesearch). Use :backtracking or :fraction_to_boundary.",
        ),
    )
    linear_solver ∈ (:svd, :sparseQR_J, :sparseQR_Jᵀ, :klu, :ldl) || throw(
        ArgumentError(
            "Unsupported linear_solver $(linear_solver). Use :svd, :sparseQR_J, :sparseQR_Jᵀ, :klu or :ldl.",
        ),
    )
    η > 0 || throw(ArgumentError("η must be positive, got $η"))
    isempty(ρ_schedule) && throw(ArgumentError("ρ_schedule is empty"))
    kkt.kind === :scholtes && throw(ArgumentError("PDIP cannot solve the Scholtes system: its exact products σ̂ ⊙ ϕ = 0 have no interior. Use ProjectedLM."))
    0 < β < 1 && 0 < min_step < 1 ||
        throw(ArgumentError("β and min_step must lie in (0, 1), got $β and $min_step"))

    # Initial point: s, σ and γ at 1, every other variable at 0, unless z₀ gives them.
    z = zeros(kkt.variable_dimension)
    z[kkt.preference_slack_dims] .= 1.0
    z[kkt.interior_point_slack_dims] .= 1.0
    z[kkt.inequality_constraint_dual_dims] .= 1.0
    if !isnothing(z₀)
        if length(z₀) == kkt.variable_dimension
            z .= z₀
        elseif length(z₀) == length(kkt.primal_dims)
            z[kkt.primal_dims] .= z₀
        else
            throw(
                ArgumentError(
                    "z₀ must have length $(kkt.variable_dimension) (full) or $(length(kkt.primal_dims)) (primal); got $(length(z₀)).",
                ),
            )
        end
    end
    σ = @view z[kkt.interior_point_slack_dims]
    γ = @view z[kkt.inequality_constraint_dual_dims]
    all(>(0), view(z, kkt.nonnegative_dims)) ||
        throw(ArgumentError("z₀ must be positive on nonnegative_dims (σ and γ)."))

    # Set up common memory.
    J = kkt.∇F_z!.result_buffer
    F = zeros(kkt.kkt_dimension) # F(z; ρ)
    F_trial = zeros(kkt.kkt_dimension) # F(z + α δz; ρ)
    F₀ = zeros(kkt.kkt_dimension) # K₀(z): F(z; 0) (for the Scholtes system with u = 0), for ‖K₀‖
    δz = zeros(kkt.variable_dimension)
    z_trial = similar(z)
    nonnegative_trial = @view z_trial[kkt.nonnegative_dims] # @view to avoid copy of selected entries
    δσ = @view δz[kkt.interior_point_slack_dims]
    δγ = @view δz[kkt.inequality_constraint_dual_dims]
    not_γ = setdiff(1:(kkt.variable_dimension), kkt.inequality_constraint_dual_dims)

    ρ_history, relaxed_residual_history, residual_history, step_size_history =
        Float64[], Float64[], Float64[], Float64[]
    newton_steps, residual_evaluations, svd_fallbacks = 0, 0, 0
    residual = Inf
    termination_reason = :max_inner_iters
    converged = false
    ρ = first(ρ_schedule)
    level = 1
    while true
        kkt.F!(F, z; θ, ρ)
        relaxed_residual = norm(F)
        level_solved = false
        termination_reason = :max_inner_iters
        for _ in 1:max_inner_iters
            # Stopping tests: ‖K₀‖ < tol ends the solve, ‖F(z; ρ)‖ < tol_inner ends this level.
            residual_evaluations += 1
            kkt.K₀!(F₀, z; θ)
            residual = norm(F₀)
            if record_convergence
                push!(ρ_history, ρ)
                push!(relaxed_residual_history, relaxed_residual)
                push!(residual_history, residual)
            end
            if !isfinite(relaxed_residual)
                termination_reason = :nonfinite_residual
                break
            end
            if residual < tol
                converged = true
                break
            end
            if relaxed_residual < tol_inner
                level_solved = true
                termination_reason = :inner_tolerance_reached
                break
            end

            # Newton step: J δz = −F.
            kkt.∇F_z!(J, z; θ, ρ)
            svd_fallbacks += _newton_step!(δz, J, F, linear_solver, η).fallback

            if linesearch === :backtracking
                # Algorithm 5.1: α ← β α while ‖F(z + α δz; ρ)‖ ≥ ‖F(z; ρ)‖ or a coordinate in
                # nonnegative_dims would not stay positive. The decrease is strict: a step that
                # leaves ‖F‖ unchanged (at tiny α, z + α δz rounds to z) makes no progress.
                α = 1.0
                accepted = false
                while α >= min_step
                    @. z_trial = z + α * δz
                    if all(>(0), nonnegative_trial) # keep σ and γ positive. 
                        kkt.F!(F_trial, z_trial; θ, ρ)
                        if norm(F_trial) < relaxed_residual
                            accepted = true
                            break
                        end
                    end
                    α *= β
                end
                if !accepted
                    termination_reason = :line_search_failed
                    break
                end
                z .= z_trial
                F .= F_trial
            else
                # Fraction to the boundary: γ moves by α_γ, every other variable by α_σ.
                α_σ = fraction_to_the_boundary_linesearch(σ, δσ; τ, decay = β, tol = min_step)
                α_γ = fraction_to_the_boundary_linesearch(γ, δγ; τ, decay = β, tol = min_step)
                if isnan(α_σ) || isnan(α_γ)
                    termination_reason = :line_search_failed
                    break
                end
                @views z[not_γ] .+= α_σ .* δz[not_γ]
                γ .+= α_γ .* δγ
                kkt.F!(F, z; θ, ρ)
                α = min(α_σ, α_γ)
            end
            relaxed_residual = norm(F)
            newton_steps += 1
            record_convergence && push!(step_size_history, α)
        end
        verbose && println(
            "ρ = $ρ: $termination_reason, ‖F(z; ρ)‖ = $relaxed_residual, ‖K₀‖ = $residual",
        )
        converged && break

        # The next level.
        if linesearch === :backtracking
            level == length(ρ_schedule) && break
            level += 1
            ρ = ρ_schedule[level]
        else
            level == max_outer_iters && break
            level += 1
            ρ *= level_solved ? 1 - exp(-tightening_rate) : 1 + exp(-loosening_rate)
            ρ = min(ρ, 1.0)
        end
    end

    # Evaluate the stopping criterion at the returned iterate.
    kkt.K₀!(F₀, z; θ)
    residual = norm(F₀)
    converged = residual < tol
    converged && (termination_reason = :tolerance_reached)
    result = (;
        status = converged ? :solved : :failed,
        termination_reason,
        z,
        x = z[kkt.primal_dims],
        ρ,
        residual,
        relaxed_residual = norm(F),
        newton_steps,
        residual_evaluations,
        svd_fallbacks,
    )
    record_convergence || return result
    (; result..., ρ_history, relaxed_residual_history, residual_history, step_size_history)
end

"""
Solver tag for the projected Levenberg–Marquardt method: 
    `solve(ProjectedLM(), kkt, θ; z₀, options = ProjectedLMOptions(...))`.
"""
struct ProjectedLM <: SolverType end

"""
Options for `solve(ProjectedLM(), …)`. The fields it shares with `PDIPOptions` mean the same.

  - `ρ_schedule`: the relaxation of each stage; the levels are solved in turn.
  - `max_inner_iters`: Newton steps per stage.
  - `tol`: bar on ‖K₀‖ = ‖F(z; ρ = 0)‖, checked before every Newton step; below it the solve stops.
    `tol_inner`: bar on ‖F(z; ρ)‖; below it the level is solved.
  - `linear_solver`: `:normal` (default), `:klu` or `:ldl`; see `_newton_step!`.
  - `η`: the first Levenberg–Marquardt parameter; the step on the free coordinates is
    −J_Fᵀ(J_F J_Fᵀ + η²I)⁻¹F. After each line search η is updated from the gain ratio (the actual over
    the predicted decrease of ½‖F‖²): multiplied by `1 + exp(-η_loosening_rate)` when the ratio is at
    most `gain_low` or the step was shortened (α < 0.99), by `1 - exp(-η_tightening_rate)` when it
    exceeds `gain_high`, and kept within [`η_min`, `η_max`].
  - `τ`, `proj_eps`: the two-metric projection. Each step clips the coordinates in `nonnegative_dims`
    at their floor (1 − τ) z_k; a coordinate within min(`proj_eps`, ‖projected gradient‖) of its floor
    whose gradient pushes it lower is held: it leaves the Levenberg–Marquardt step and takes its own
    scaled gradient step.
  - `β`, `min_step`, `c`: the line search along the projected arc: α ← β α until ‖F‖ falls by the
    fraction c α (Armijo), and strictly; the step fails below `min_step`.
  - `record_convergence`: also return the per-iteration histories.
"""
Base.@kwdef struct ProjectedLMOptions
    ρ_schedule::Vector{Float64} = [10.0^(-k) for k in 0:10]
    max_inner_iters::Int = 200
    tol::Float64 = 1e-8
    tol_inner::Float64 = 1e-11
    linear_solver::Symbol = :normal
    η::Float64 = 1e-8
    η_min::Float64 = 1e-8
    η_max::Float64 = 1e2
    gain_low::Float64 = 0.25
    gain_high::Float64 = 0.75
    η_tightening_rate::Float64 = 1.2
    η_loosening_rate::Float64 = 3.0
    τ::Float64 = 0.99
    proj_eps::Float64 = 1e-12
    β::Float64 = 0.5
    min_step::Float64 = 1e-14
    c::Float64 = 1e-4
    record_convergence::Bool = false
    verbose::Bool = false
end

"""
    solve(ProjectedLM(), kkt::GOOPKKTSystem, θ; z₀ = nothing, options = ProjectedLMOptions())

The projected Levenberg–Marquardt method: Newton's method on F(z; ρ) = 0 for each relaxation ρ, 
    keeping `kkt.nonnegative_dims` positive. Each Newton step differs from PDIP's in four places:

  1. the direction: the coordinates in `nonnegative_dims` at their floor (1 − τ) z_k that the gradient
     g = JᵀF of ½‖F‖² pushes lower are held; the other coordinates take the Levenberg–Marquardt step
     −J_Fᵀ(J_F J_Fᵀ + η²I)⁻¹F (J_F: J with the held columns zeroed), each held one the scaled gradient
     step −g_k / ‖J e_k‖², stopped at its floor (Bertsekas' two-metric projection);
  2. the trial point: z + α δz with every coordinate in `nonnegative_dims` clipped at its floor;
  3. the acceptance test: ‖F‖ falls by the fraction c α (Armijo), and strictly;
  4. the η update from the gain ratio, after the line search.

`z₀` and the outputs are as for `PDIP`; the outputs add `η`, the η after the last step (to continue a
solve from), and with `record_convergence` `η_history` (the η of each Newton step).
"""
function solve(
    ::ProjectedLM,
    kkt::GOOPKKTSystem,
    θ::AbstractVector{<:Real};
    z₀ = nothing,
    options::ProjectedLMOptions = ProjectedLMOptions(),
)
    (; ρ_schedule, max_inner_iters, tol, tol_inner, linear_solver, η) = options
    (; η_min, η_max, gain_low, gain_high, η_tightening_rate, η_loosening_rate) = options
    (; τ, proj_eps, β, min_step, c, record_convergence, verbose) = options
    linear_solver ∈ (:normal, :klu, :ldl) || throw(
        ArgumentError("Unsupported linear_solver $(linear_solver). Use :normal, :klu or :ldl."),
    )
    isfinite(η) && η > 0 || throw(ArgumentError("η must be finite and positive, got $η"))
    0 < η_min <= η_max || throw(ArgumentError("η_min and η_max must satisfy 0 < η_min <= η_max"))
    0 <= gain_low < gain_high ||
        throw(ArgumentError("gain_low and gain_high must satisfy 0 <= gain_low < gain_high"))
    isempty(ρ_schedule) && throw(ArgumentError("ρ_schedule is empty"))
    0 < β < 1 && 0 < min_step < 1 ||
        throw(ArgumentError("β and min_step must lie in (0, 1), got $β and $min_step"))
    0 < τ < 1 || throw(ArgumentError("τ must lie in (0, 1), got $τ"))

    # Initial point: s, σ and γ at 1, every other variable at 0, unless z₀ gives them.
    z = zeros(kkt.variable_dimension)
    z[kkt.preference_slack_dims] .= 1.0
    z[kkt.interior_point_slack_dims] .= 1.0
    z[kkt.inequality_constraint_dual_dims] .= 1.0
    if !isnothing(z₀)
        if length(z₀) == kkt.variable_dimension
            z .= z₀
        elseif length(z₀) == length(kkt.primal_dims)
            z[kkt.primal_dims] .= z₀
        else
            throw(
                ArgumentError(
                    "z₀ must have length $(kkt.variable_dimension) (full) or $(length(kkt.primal_dims)) (primal); got $(length(z₀)).",
                ),
            )
        end
    end
    all(>(0), view(z, kkt.nonnegative_dims)) ||
        throw(ArgumentError("z₀ must be positive on nonnegative_dims (σ and γ)."))

    # Set up common memory.
    J = kkt.∇F_z!.result_buffer
    J_F = copy(J) # J with the held columns zeroed
    F = zeros(kkt.kkt_dimension) # F(z; ρ)
    F_trial = zeros(kkt.kkt_dimension) # F(z_trial; ρ)
    F₀ = zeros(kkt.kkt_dimension) # K₀(z): F(z; 0) (for the Scholtes system with u = 0), for ‖K₀‖
    δz = zeros(kkt.variable_dimension)
    z_trial = similar(z)
    nonnegative_trial = @view z_trial[kkt.nonnegative_dims] # @view to avoid copy of selected entries
    g = zeros(kkt.variable_dimension) # JᵀF, the gradient of ½‖F‖²
    floors = zeros(length(kkt.nonnegative_dims)) # (1 − τ) z_k for k in nonnegative_dims
    held = falses(length(kkt.nonnegative_dims))
    Δz = similar(z) # z_trial − z, for the gain ratio
    JΔz = zeros(kkt.kkt_dimension) # J Δz, the change of F the linear model predicts

    η = clamp(η, η_min, η_max)
    ρ_history, relaxed_residual_history, residual_history, step_size_history, η_history =
        Float64[], Float64[], Float64[], Float64[], Float64[]
    newton_steps, residual_evaluations, svd_fallbacks = 0, 0, 0
    residual = Inf
    termination_reason = :max_inner_iters
    converged = false
    ρ = first(ρ_schedule)
    level = 1
    while true
        kkt.F!(F, z; θ, ρ)
        relaxed_residual = norm(F)
        termination_reason = :max_inner_iters
        for _ in 1:max_inner_iters
            # Stopping tests: ‖K₀‖ < tol ends the solve, ‖F(z; ρ)‖ < tol_inner ends this level.
            residual_evaluations += 1
            kkt.K₀!(F₀, z; θ)
            residual = norm(F₀)
            if record_convergence
                push!(ρ_history, ρ)
                push!(relaxed_residual_history, relaxed_residual)
                push!(residual_history, residual)
            end
            if !isfinite(relaxed_residual)
                termination_reason = :nonfinite_residual
                break
            end
            if residual < tol
                converged = true
                break
            end
            if relaxed_residual < tol_inner
                termination_reason = :inner_tolerance_reached
                break
            end

            # Two-metric projection: hold the coordinates at their floor that the gradient pushes lower.
            kkt.∇F_z!(J, z; θ, ρ)
            mul!(g, transpose(J), F)
            projected_gradient² = 0.0 # ‖z − max(z − g, floor)‖² over nonnegative_dims
            for (k, i) in enumerate(kkt.nonnegative_dims)
                floors[k] = (1 - τ) * z[i]
                projected_gradient² += (z[i] - max(z[i] - g[i], floors[k]))^2
            end
            ε = min(proj_eps, sqrt(projected_gradient²))
            copyto!(SparseArrays.nonzeros(J_F), SparseArrays.nonzeros(J))
            for (k, i) in enumerate(kkt.nonnegative_dims)
                held[k] = z[i] - floors[k] <= ε && g[i] > 0 # held
                held[k] && (view(SparseArrays.nonzeros(J_F), SparseArrays.nzrange(J_F, i)) .*= 0.0) # free
            end

            # Newton step on the free coordinates: J_F δz = −F, regularized by η² (Levenberg–Marquardt).
            step = _newton_step!(δz, J_F, F, linear_solver, η; shift = abs2)
            svd_fallbacks += step.fallback
            # A held coordinate's own step: −g_k / ‖J e_k‖², stopped at its floor.
            for (k, i) in enumerate(kkt.nonnegative_dims)
                held[k] || continue
                column_norm² = 0.0
                for p in SparseArrays.nzrange(J, i)
                    column_norm² += SparseArrays.nonzeros(J)[p]^2
                end
                δz[i] = max(-g[i] / (column_norm² + 1e-16), floors[k] - z[i])
            end
            if !all(isfinite, δz)
                termination_reason = :invalid_step
                break
            end

            # Line search along the projected arc: α ← β α while ‖F‖ does not fall by c α (Armijo), and
            # strictly: below α ≈ 1e-12, 1 − c α rounds to 1 and the Armijo test alone would accept a step
            # that leaves ‖F‖ unchanged (measured: output/refactor/step24/variants.jl).
            α = 1.0
            accepted = false
            trial_residual = Inf
            while α >= min_step
                @. z_trial = z + α * δz
                for (k, i) in enumerate(kkt.nonnegative_dims)
                    z_trial[i] < floors[k] && (z_trial[i] = floors[k]) # clip at the floor
                end
                if all(>(0), nonnegative_trial) # keep σ and γ positive.
                    kkt.F!(F_trial, z_trial; θ, ρ)
                    trial_residual = norm(F_trial)
                    if trial_residual <= (1 - c * α) * relaxed_residual && trial_residual < relaxed_residual
                        accepted = true
                        break
                    end
                end
                α *= β
            end

            # η from the gain ratio, the actual over the predicted decrease of ½‖F‖².
            gain_ratio = -Inf
            if accepted
                @. Δz = z_trial - z
                mul!(JΔz, J, Δz)
                gain_ratio = _gain_ratio(F, JΔz, relaxed_residual, trial_residual)
            end
            record_convergence && push!(η_history, step.η)
            η = _scheduled_η(
                step.η,
                gain_ratio,
                accepted ? α : 0.0;
                gain_low,
                gain_high,
                η_min,
                η_max,
                η_tightening_rate,
                η_loosening_rate,
            )
            if !accepted
                termination_reason = :line_search_failed
                break
            end
            z .= z_trial
            F .= F_trial
            relaxed_residual = norm(F)
            newton_steps += 1
            record_convergence && push!(step_size_history, α)
        end
        verbose && println(
            "ρ = $ρ: $termination_reason, ‖F(z; ρ)‖ = $relaxed_residual, ‖K₀‖ = $residual, η = $η",
        )
        converged && break

        # The next level.
        level == length(ρ_schedule) && break
        level += 1
        ρ = ρ_schedule[level]
    end

    # Evaluate the stopping criterion at the returned iterate.
    kkt.K₀!(F₀, z; θ)
    residual = norm(F₀)
    converged = residual < tol
    converged && (termination_reason = :tolerance_reached)
    result = (;
        status = converged ? :solved : :failed,
        termination_reason,
        z,
        x = z[kkt.primal_dims],
        ρ,
        residual,
        relaxed_residual = norm(F),
        newton_steps,
        residual_evaluations,
        svd_fallbacks,
        η,
    )
    record_convergence || return result
    (; result..., ρ_history, relaxed_residual_history, residual_history, step_size_history, η_history)
end

"""
The gain ratio of a step: the actual decrease of ½‖F‖², from `residual` = ‖F‖ to `trial_residual`, over
the decrease that the linear model F + J Δz predicts (`JΔz` = J Δz); −Inf when none is predicted.
"""
function _gain_ratio(F, JΔz, residual, trial_residual)
    predicted = -dot(F, JΔz) - 0.5 * dot(JΔz, JΔz)
    actual = (residual - trial_residual) * (0.5residual + 0.5trial_residual)
    predicted > 0 && isfinite(predicted) && isfinite(actual) ? actual / predicted : -Inf
end

"""
The next Levenberg–Marquardt parameter: larger after a poor (`gain_ratio` ≤ `gain_low`) or shortened
(α < 0.99) step, smaller after a good one (`gain_ratio` > `gain_high`), within [`η_min`, `η_max`].
"""
function _scheduled_η(η, gain_ratio, α; gain_low, gain_high, η_min, η_max, η_tightening_rate, η_loosening_rate)
    if gain_ratio <= gain_low || isnan(gain_ratio) || α < 0.99
        return clamp(η * (1 + exp(-η_loosening_rate)), η_min, η_max)
    elseif gain_ratio > gain_high
        return clamp(η * (1 - exp(-η_tightening_rate)), η_min, η_max)
    end
    clamp(η, η_min, η_max)
end

"""
The Newton step δz with J δz = −F, by `linear_solver`, for `PDIP` (Algorithm 5.1) and `ProjectedLM` (on
the masked Jacobian J_F). The regularized solvers shift JJᵀ by s = `shift(η)`, δz = −Jᵀ(JJᵀ + sI)⁻¹F:
PDIP by η (`shift = identity`), ProjectedLM by η² (`shift = abs2`, Levenberg–Marquardt). `:klu` and `:ldl`
solve the balanced augmented system [√sI J; Jᵀ −√sI] [w; δz] = [−F; 0]; `:normal` factorizes JJᵀ + sI
itself. Returns `(; fallback, η)`: `fallback` is 1 when the step fell back to the `:svd` step
(`:sparseQR_Jᵀ` with J without full row rank, or a singular or failed factorization), else 0; `η` is the
η the step used (`:normal` raises it when its factorization fails).
"""
function _newton_step!(δz, J, F, linear_solver, η; shift = identity)
    m, n = size(J)
    if linear_solver === :svd
        # :svd: the minimum-norm step −J⁺F by the dense pseudoinverse (Algorithm 5.1).
        δz .= pinv(Matrix(J)) * (-F)
    elseif linear_solver === :sparseQR_J
        # :sparseQR_J: J \ (−F) by a sparse QR of J, a least-squares step, not the minimum-norm one.
        δz .= J \ (-F)
    elseif linear_solver === :sparseQR_Jᵀ
        # :sparseQR_Jᵀ: −J⁺F by a sparse QR of Jᵀ, Jᵀ[prow, pcol] = QR, when J has full row rank.
        QR = qr(SparseArrays.sparse(J'))
        if rank(QR) < m
            δz .= pinv(Matrix(J)) * (-F) # J without full row rank: the :svd step
            return (; fallback = 1, η)
        end
        c = -(UpperTriangular(QR.R)' \ F[QR.pcol]) # R₁ᵀ c = −Pcᵀ F
        δz[QR.prow] .= QR.Q * [c; zeros(n - m)] # δz = Prᵀ Q [c; 0]
    elseif linear_solver === :klu
        # :klu: −Jᵀ(JJᵀ + sI)⁻¹F by a sparse LU (KLU) of the augmented matrix [√sI J; Jᵀ −√sI].
        s = shift(η)
        K = [√s*SparseArrays.sparse(1.0I, m, m) J; J' -√s*SparseArrays.sparse(1.0I, n, n)]
        solution = try
            KLU.klu(K) \ [-F; zeros(n)]
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
            δz .= pinv(Matrix(J)) * (-F) # a singular factorization: the :svd step
            return (; fallback = 1, η)
        end
        δz .= view(solution, (m + 1):(m + n))
    elseif linear_solver === :ldl
        # :ldl: −Jᵀ(JJᵀ + sI)⁻¹F by a sparse LDLᵀ (CHOLMOD) of the augmented matrix [√sI J; Jᵀ −√sI].
        s = shift(η)
        K = [√s*SparseArrays.sparse(1.0I, m, m) J; J' -√s*SparseArrays.sparse(1.0I, n, n)]
        factor = ldlt(Symmetric(K, :L); check = false) # one triangle: J may store zeros
        if !issuccess(factor)
            δz .= pinv(Matrix(J)) * (-F) # a failed factorization: the :svd step
            return (; fallback = 1, η)
        end
        δz .= view(factor \ [-F; zeros(n)], (m + 1):(m + n))
    elseif linear_solver === :normal
        # :normal: −Jᵀ(JJᵀ + sI)⁻¹F by a sparse Cholesky (CHOLMOD) of JJᵀ + sI. A failed factorization
        # (s below the round-off of JJᵀ when J is rank-deficient) retries with 100 η, up to 5 times, and
        # then takes the :svd step, as the Scholtes engine does.
        JJᵀ = Symmetric(J * J', :U)
        η₀ = η
        for _ in 0:5
            factor = cholesky(JJᵀ; shift = shift(η), check = false)
            if issuccess(factor)
                mul!(δz, transpose(J), factor \ (-F))
                all(isfinite, δz) && return (; fallback = 0, η)
            end
            η *= 100
        end
        δz .= pinv(Matrix(J)) * (-F) # every factorization failed: the :svd step
        return (; fallback = 1, η = η₀)
    else
        throw(ArgumentError("Unsupported linear_solver $(linear_solver)."))
    end
    (; fallback = 0, η)
end
