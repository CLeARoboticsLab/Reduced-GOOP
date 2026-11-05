using LinearAlgebra

abstract type SolverType end

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
  - `η > 0`: the Levenberg–Marquardt parameter of `:klu` and `:ldl`, whose step is
    −Jᵀ(JJᵀ + η²I)⁻¹F (→ −J⁺F as η → 0); the default 1e-6 regularizes JJᵀ by 1e-12 I. `:svd` and
    the sparse QR steps do not use it.
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
    η::Float64 = 1e-6
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
    proj_eps >= 0 || throw(ArgumentError("proj_eps must be nonnegative, got $proj_eps"))
    0 < c < 1 || throw(ArgumentError("c must lie in (0, 1), got $c"))

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
    normal_equations = _normal_equations(J_F, linear_solver) # J_F J_Fᵀ and its factor, kept across steps (:normal)
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
            step = _newton_step!(δz, J_F, F, linear_solver, η, normal_equations)
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
            # that leaves ‖F‖ unchanged.
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
the masked Jacobian J_F). The regularized solvers take the Levenberg–Marquardt step with parameter η,
δz = −Jᵀ(JJᵀ + η²I)⁻¹F, the minimizer of ½‖F + J δz‖² + ½η²‖δz‖² (→ −J⁺F as η → 0): `:klu` and `:ldl`
solve the balanced augmented system [ηI J; Jᵀ −ηI] [w; δz] = [−F; 0]; `:normal` factorizes JJᵀ + η²I
itself, kept in `normal_equations` from one step to the next. Returns `(; fallback, η)`: `fallback` is 1
when the step fell back to the `:svd` step (`:sparseQR_Jᵀ` with J without full row rank, or a singular or
failed factorization), else 0; `η` is the η the step used (`:normal` raises it when its factorization fails).
"""
#TODO: Same treatment (as :normal) for :klu and :ldl: Fill K in place and reuse symbolic factorization (https://klu.juliasparse.org/dev/)
function _newton_step!(δz, J, F, linear_solver, η, normal_equations = _normal_equations(J, linear_solver))
    m, n = size(J)
    if linear_solver === :svd
        # :svd: the minimum-norm step −J⁺F by the dense pseudoinverse (Algorithm 5.1).
        δz .= pinv(Matrix(J)) * (-F)
    elseif linear_solver === :sparseQR_J
        # :sparseQR_J: a least-squares step by a sparse QR of J (SPQR), not the minimum-norm one. `qr(J)`, not
        # `J \ (−F)`: for a square J, `\` takes an LU, which fails when J is singular.
        δz .= qr(J) \ (-F)
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
        # :klu: −Jᵀ(JJᵀ + η²I)⁻¹F by a sparse LU (KLU) of the augmented matrix [ηI J; Jᵀ −ηI].
        K = [η*SparseArrays.sparse(1.0I, m, m) J; J' -η*SparseArrays.sparse(1.0I, n, n)]
        solution = try
            KLU.klu(K) \ [-F; zeros(n)]
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
            δz .= pinv(Matrix(J)) * (-F) # a singular factorization: the :svd step
            return (; fallback = 1, η)
        end
        δz .= view(solution, (m + 1):(m + n))
    elseif linear_solver === :ldl
        # :ldl: −Jᵀ(JJᵀ + η²I)⁻¹F by a sparse LDLᵀ (CHOLMOD) of the augmented matrix [ηI J; Jᵀ −ηI].
        K = [η*SparseArrays.sparse(1.0I, m, m) J; J' -η*SparseArrays.sparse(1.0I, n, n)]
        factor = ldlt(Symmetric(K, :L); check = false) # one triangle: J may store zeros
        if !issuccess(factor)
            δz .= pinv(Matrix(J)) * (-F) # a failed factorization: the :svd step
            return (; fallback = 1, η)
        end
        δz .= view(factor \ [-F; zeros(n)], (m + 1):(m + n))
    elseif linear_solver === :normal
        # :normal: −Jᵀ(JJᵀ + η²I)⁻¹F by a sparse Cholesky (CHOLMOD) of JJᵀ + η²I. A failed factorization
        # (η² below the round-off of JJᵀ when J is rank-deficient) retries with 100 η, up to 5 times, and
        # then takes the :svd step, as the Scholtes engine does.
        JJᵀ = _gram!(normal_equations, J)
        η₀ = η
        for _ in 0:5
            factor = _cholesky!(normal_equations, JJᵀ, η) # CHOLMOD factorizes JJᵀ + η²I
            if issuccess(factor)
                w = ldiv!(normal_equations.w, factor, normal_equations.minus_F .= .-F) # (JJᵀ + η²I) w = −F
                mul!(δz, transpose(J), w)
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

"""
The normal equations of the `:normal` step, kept from one Newton step to the next. J's pattern is the same at
every step of a solve (J_F zeroes the held columns' values, not their entries), so the upper triangle of JJᵀ is
refilled in place (`_gram!`) in the matrix CHOLMOD factorizes, CHOLMOD's symbolic analysis of the first
factorization serves the later ones (`_cholesky!`), and the solve writes into kept vectors: no step rebuilds or
copies JJᵀ, or repeats the analysis.
"""
mutable struct _NormalEquations
    "The upper triangle of JJᵀ in CHOLMOD's own format, made once: every factorization reads it."
    JJᵀ::SparseArrays.CHOLMOD.Sparse{Float64,Int}
    "`JJᵀ`'s values, in CHOLMOD's memory, as a Julia vector: `_gram!` refills them in place."
    values::Vector{Float64}
    "The pairs (p, q) of positions in J's nonzeros whose products the entries of `JJᵀ` sum, entry by entry."
    products::Vector{Tuple{Int,Int}}
    "For each entry of `JJᵀ`, its range in `products`."
    entry_products::Vector{UnitRange{Int}}
    "The last successful Cholesky factor of JJᵀ + η²I, whose symbolic analysis the next one reuses."
    factor::Union{Nothing,SparseArrays.CHOLMOD.Factor{Float64,Int}}
    "The solve's result w, with (JJᵀ + η²I) w = −F."
    w::Vector{Float64}
    "The solve's right-hand side −F."
    minus_F::Vector{Float64}
end

"""
`_NormalEquations` for J's pattern, or `nothing` when `linear_solver` is not `:normal`. The entry (i, k), i ≤ k, of
JJᵀ sums J[i, j] J[k, j] over the columns j where both are stored, in increasing j: the order in which
SparseArrays' product `J * J'` adds them, so that `_gram!` is bitwise `triu(J * J')`.
"""
function _normal_equations(J, linear_solver)
    linear_solver === :normal || return nothing
    rows = SparseArrays.rowvals(J)
    ones_J = SparseArrays.SparseMatrixCSC(size(J)..., SparseArrays.getcolptr(J), rows, ones(SparseArrays.nnz(J)))
    upper = triu(ones_J * ones_J') # every entry JJᵀ can store (ones: no cancellation)
    products_of = [Tuple{Int,Int}[] for _ in 1:SparseArrays.nnz(upper)]
    for j in 1:size(J, 2), q in SparseArrays.nzrange(J, j), p in first(SparseArrays.nzrange(J, j)):q
        i, k = rows[p], rows[q] # i ≤ k: J's rows are sorted within a column
        entries = SparseArrays.nzrange(upper, k)
        entry = entries[searchsortedfirst(view(SparseArrays.rowvals(upper), entries), i)]
        push!(products_of[entry], (p, q))
    end
    ends = cumsum(length.(products_of))
    entry_products = [(stop - length(products) + 1):stop for (stop, products) in zip(ends, products_of)]
    # CHOLMOD's copy of JJᵀ, and a Julia vector onto its values (in `upper`'s order; the memory is CHOLMOD's, kept
    # alive by the struct's `JJᵀ`), so that `_gram!` writes straight into the matrix CHOLMOD factorizes.
    JJᵀ = SparseArrays.CHOLMOD.Sparse(Symmetric(upper, :U))
    raw = unsafe_load(pointer(JJᵀ))
    values = unsafe_wrap(Array, Ptr{Float64}(raw.x), SparseArrays.nnz(upper))
    raw.stype == 1 && raw.packed == 1 && values == SparseArrays.nonzeros(upper) ||
        error("CHOLMOD's copy of JJᵀ does not keep its entries in order")
    m = size(J, 1)
    products = collect(Iterators.flatten(products_of)) # (not `reduce(vcat, …; init)`: quadratic)
    _NormalEquations(JJᵀ, values, products, entry_products, nothing, zeros(m), zeros(m))
end

"The upper triangle of JJᵀ from J's values, refilled in place in CHOLMOD's copy, which it returns."
function _gram!(normal_equations, J)
    (; values, products, entry_products) = normal_equations
    J_values = SparseArrays.nonzeros(J)
    @inbounds for (entry, range) in enumerate(entry_products)
        p, q = products[first(range)]
        total = J_values[p] * J_values[q] # the first product, then the rest, as `J * J'` adds them
        for t in (first(range) + 1):last(range)
            p, q = products[t]
            total += J_values[p] * J_values[q]
        end
        values[entry] = total
    end
    normal_equations.JJᵀ
end

"The Cholesky factor of JJᵀ + η²I: CHOLMOD's symbolic analysis once, then only the numeric factorization."
function _cholesky!(normal_equations, JJᵀ, η)
    factor = if isnothing(normal_equations.factor)
        cholesky(JJᵀ; shift = η^2, check = false)
    else
        cholesky!(normal_equations.factor, JJᵀ; shift = η^2, check = false)
    end
    normal_equations.factor = issuccess(factor) ? factor : nothing # a failed factor is not reused
    factor
end
