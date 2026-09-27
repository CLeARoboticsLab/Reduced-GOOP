# scholtes_solver.jl -- Newton on the Scholtes-form reduced KKT system, down a ρ homotopy.
#
# Ported from ScholtesReducedGOOP.jl `src/solve.jl` (Jingqi Li). With
# w = [y; s; u] and R(w; ρ) as in src/scholtes_residual.jl:
#
#     w ← w₀(z₀, ρ₁)                                   strictly interior
#     for ρ in schedule
#         repeat at most max_inner times
#             r ← R(w; ρ)
#             if ‖K₀(w)‖ < tol       : stop, converged
#             if ‖r‖      < tol_inner : this level is solved, next ρ
#             d ← projected minimum-norm least-squares step at w
#             α ← largest of 1, 1/2, 1/4, … with merit(P(w + αd)) ≤ (1 − 1e-4 α)‖r‖
#             optionally update η from the gain ratio and the step length
#             if no such α           : this level has stalled, next ρ
#             w ← P(w + αd)
#
# `P` is the projected rule's arc and the merit is ∞ outside the sign box, so
# positivity is a property of the direction, never a test on the step length.

struct Scholtes <: SolverType end

"""
Options for `solve(::Scholtes, …)`. Defaults are the source's `solve_goop` defaults,
except `linear_solver`, which defaults to `:normal` (the robotic-arm setting).

  - `rho_init = 1.0`, `rho_min = 1e-10`: the geometric ρ schedule, one level per decade.
    `rho_schedule` (a vector) overrides both.
  - `max_inner = 200`: Newton steps per ρ level.
  - `tol = 1e-8`: bar on the true residual ‖K₀‖ (converged). `tol_inner = 1e-11`: bar on
    ‖R(w; ρ)‖ (this level solved).
  - `projected_step::Bool = true`: the two-metric projected bound rule. It is REQUIRED:
    the exact φ/σ rows are only handled by the projected step, and this package
    supports only φ = true, so `false` throws.
  - `tau = 0.99`: projection floor `(1 − τ)·c_k(w)`. `proj_eps = 1e-12`: the projected
    rule's active-set radius.
  - `nonmonotone = 1`: Armijo against the worst of the last M residual norms.
  - `linear_solver`: `:normal` (Cholesky of `JJᵀ + η²I`; default), `:klu_eta` (LU of
    `[ηI J; Jᵀ −ηI]`, the source's `:klu`), `:klu_sqrt_eta` (LU of `[√η I J; Jᵀ −√η I]`,
    this package's interior-point `:klu` convention, where η is the Tikhonov shift) or
    `:svd` (dense `pinv`). `refine = 0`: iterative-refinement sweeps.
  - `eta_init = 1e-8`: initial Tikhonov parameter. `eta_schedule = false`: update η
    after each line search from the gain ratio (`gain_low = 0.25`, `gain_high = 0.75`,
    factors `1 ± exp(−rate)` with `tightening_rate = 1.2`, `loosening_rate = 3.0`,
    clamped to `[eta_min, eta_max] = [1e-8, 1e2]`).
"""
Base.@kwdef struct ScholtesOptions
    rho_init::Float64 = 1.0
    rho_min::Float64 = 1e-10
    rho_schedule::Union{Nothing,Vector{Float64}} = nothing
    max_inner::Int = 200
    tol::Float64 = 1e-8
    tol_inner::Float64 = 1e-11
    projected_step::Bool = true
    tau::Float64 = 0.99
    proj_eps::Float64 = 1e-12
    nonmonotone::Int = 1
    linear_solver::Symbol = :normal
    refine::Int = 0
    eta_init::Float64 = 1e-8
    eta_schedule::Bool = false
    eta_min::Float64 = 1e-8
    eta_max::Float64 = 1e2
    gain_low::Float64 = 0.25
    gain_high::Float64 = 0.75
    tightening_rate::Float64 = 1.2
    loosening_rate::Float64 = 3.0
    verbose::Bool = false
end

"A copy of `options` with the given fields replaced."
function _with_options(options::ScholtesOptions; kwargs...)
    fields = (f => getfield(options, f) for f in fieldnames(ScholtesOptions))
    ScholtesOptions(; fields..., kwargs...)
end

"""
    geometric_schedule(rho_init; stop = 1e-10, per_decade = 1) -> Vector{Float64}

A ρ schedule descending geometrically from `rho_init` to `stop`.
"""
function geometric_schedule(rho_init::Real; stop = 1e-10, per_decade = 1)
    rho_init > 0 || throw(ArgumentError("rho_init must be > 0, got $rho_init"))
    stop > 0 || throw(ArgumentError("stop must be > 0, got $stop"))
    per_decade > 0 || throw(ArgumentError("per_decade must be > 0, got $per_decade"))
    stop >= rho_init && return [float(rho_init)]
    return [10.0^e for e in log10(rho_init):(-1 / per_decade):log10(stop)]
end

# Gain ratio for the merit ‖R‖²/2; `jd = J(w)(w_trial − w)` includes the projection.
function _gain_reductions(r, jd, nr, trial_norm)
    pred_reduction = -dot(r, jd) - 0.5 * dot(jd, jd)
    actual_reduction = (nr - trial_norm) * (0.5nr + 0.5trial_norm)
    gain_ratio =
        pred_reduction > 0 && isfinite(pred_reduction) && isfinite(actual_reduction) ?
        actual_reduction / pred_reduction : -Inf
    return pred_reduction, actual_reduction, gain_ratio
end

function _scheduled_eta(eta, gain_ratio, alpha; gain_low, gain_high, eta_min, eta_max, tightening_rate, loosening_rate)
    if gain_ratio <= gain_low || isnan(gain_ratio) || alpha < 0.99
        return clamp(eta * (1 + exp(-loosening_rate)), eta_min, eta_max)
    elseif gain_ratio > gain_high
        return clamp(eta * (1 - exp(-tightening_rate)), eta_min, eta_max)
    end
    return clamp(eta, eta_min, eta_max)
end

# The starting iterate: duals 0, every original multiplier γ_j = 0.1ρ₁/s_j so the
# complementarity rows hold exactly at w₀, then the φ slacks are re-evaluated and
# φ_j = 0.1ρ₁/σ_j, and u = max(ρ₁ − s⊙b, 1e-6ρ₁) on the relaxed rows.
function _initial_w(kkt::ScholtesKKTSystem, z0, ρ₁::Float64; gamma_cols)
    _set_rho!(kkt, ρ₁)
    n, nc = kkt.n, kkt.n_comp
    scholtes = nc > 0
    comp_frac = scholtes ? 0.1 : 1.0
    y = zeros(n)
    y[kkt.primal_dims] .= float.(collect(z0))
    nc == 0 && return y

    s = _a(kkt, y)
    ng = kkt.original_nc
    all(>(0), view(s, 1:ng)) || throw(
        ArgumentError(
            "every g^i(z0) must be > 0 STRICTLY; worst = $(minimum(s)). Move z0 into the interior, or relax the constraint.",
        ),
    )
    if gamma_cols !== nothing
        @inbounds for j in 1:ng
            y[gamma_cols[j]] = comp_frac * ρ₁ / s[j]
        end
        s = _a(kkt, y)
        for j in (ng + 1):nc
            y[gamma_cols[j]] = comp_frac * ρ₁ / s[j]
        end
    end
    bv = _b(kkt, y)
    all(>(0), bv) ||
        throw(ArgumentError("initial multipliers must be > 0; worst = $(minimum(bv))."))
    u = max.(ρ₁ .- s .* bv, 1e-6 * ρ₁)
    return [y; s; u[findall(!iszero, kkt.u_index)]]
end

"""
    scholtes_warm_start(kkt, θ, z, ρ; eq_cols = Int[], eq = Float64[], s_floor = 1e-8) -> w

A continuation start at primal `z` and level `ρ`, optionally keeping the
equality/policy multipliers `eq` at columns `eq_cols` (source `_warm_w`).
"""
function scholtes_warm_start(kkt::ScholtesKKTSystem, θ, z, ρ; eq_cols = Int[], eq = Float64[], s_floor = 1e-8)
    _set_theta!(kkt, θ)
    _set_rho!(kkt, ρ)
    cols = _gamma_columns(kkt)
    y = zeros(kkt.n)
    y[kkt.primal_dims] .= z
    ng = kkt.original_nc
    s = max.(_a(kkt, y), s_floor)
    y[cols[1:ng]] .= 0.1ρ ./ max.(s[1:ng], ρ)
    y[eq_cols] .= eq
    a = _a(kkt, y)
    s[(ng + 1):end] .= max.(a[(ng + 1):end], min(s_floor, 0.1ρ))
    y[cols[(ng + 1):end]] .= 0.1ρ ./ max.(s[(ng + 1):end], ρ)
    u = max.(ρ .- s .* _b(kkt, y), 1e-6ρ)
    return [y; s; u[findall(!iszero, kkt.u_index)]]
end

"""
    solve(::Scholtes, kkt::ScholtesKKTSystem, θ; z₀ = nothing, w₀ = nothing,
          options = ScholtesOptions(), trace = nothing, step_trace = nothing)

Newton on the Scholtes-form reduced KKT system down a ρ homotopy (see the top of
src/scholtes_solver.jl). `z₀` is a strictly feasible primal start (default zeros);
`w₀` a full iterate `[y; s; u]` from a previous result. `trace` / `step_trace` are
called before / after each line search with the source's fields.

Returns a NamedTuple: `status` (`:solved` iff `converged`), `z` (primal), `w`, `s`, `u`,
`rho` (the last level reached), `residual` (‖K₀(w)‖, the unperturbed KKT residual),
`converged` (`residual < tol`), `sign_min`, `shortfall` (‖s ⊙ b‖∞), `iters`, `history`
(‖R‖ per iteration), `klu_fallbacks` (dense fallback steps), `eta`.
"""
function solve(
    ::Scholtes,
    kkt::ScholtesKKTSystem,
    θ::AbstractVector{<:Real};
    z₀ = nothing,
    w₀ = nothing,
    options::ScholtesOptions = ScholtesOptions(),
    trace = nothing,
    step_trace = nothing,
)
    (; rho_init, rho_min, rho_schedule, max_inner, tol, tol_inner, tau, proj_eps) = options
    (; nonmonotone, linear_solver, refine, eta_init, eta_schedule, eta_min, eta_max) = options
    (; gain_low, gain_high, tightening_rate, loosening_rate, verbose) = options

    options.projected_step || throw(
        ArgumentError(
            "phi = true requires projected_step = true (exact φ/σ rows); set projected_step = true.",
        ),
    )
    nonmonotone >= 1 || throw(ArgumentError("nonmonotone must be >= 1, got $nonmonotone"))
    refine >= 0 || throw(ArgumentError("refine must be non-negative, got $refine"))
    solver = _scholtes_step_solver(linear_solver)
    isfinite(eta_init) && eta_init > 0 ||
        throw(ArgumentError("eta_init must be finite and positive"))
    all(isfinite, (eta_min, eta_max)) && 0 < eta_min <= eta_max ||
        throw(ArgumentError("eta bounds must be finite and satisfy 0 < eta_min <= eta_max"))
    all(isfinite, (gain_low, gain_high)) && 0 <= gain_low < gain_high || throw(
        ArgumentError("gain thresholds must be finite and satisfy 0 <= gain_low < gain_high"),
    )
    all(isfinite, (tightening_rate, loosening_rate)) &&
        tightening_rate >= 0 &&
        loosening_rate >= 0 ||
        throw(ArgumentError("tightening_rate and loosening_rate must be finite and nonnegative"))
    adaptive_eta = eta_schedule && linear_solver !== :svd
    initial_eta = adaptive_eta ? clamp(float(eta_init), eta_min, eta_max) : float(eta_init)

    schedule =
        isnothing(rho_schedule) ? geometric_schedule(rho_init; stop = rho_min) :
        collect(float.(rho_schedule))
    isempty(schedule) && throw(ArgumentError("rho_schedule is empty"))
    n, nc = kkt.n, kkt.n_comp
    n_total = length(kkt.primal_dims)

    _set_theta!(kkt, θ)
    kkt.Jvalid[] = false
    _set_rho!(kkt, schedule[1])
    gamma_cols = _gamma_columns(kkt)
    res = ScholtesResiduals(kkt)
    ctx, bk = _build_scholtes_step_ctx(
        solver,
        kkt;
        gamma_cols,
        eta_init = initial_eta,
        eta_max,
        eta_sticky = !adaptive_eta,
        refine,
    )
    rule = _ProjectedRule(ctx.nw; tau, eps = proj_eps)

    w = if isnothing(w₀)
        _initial_w(kkt, isnothing(z₀) ? zeros(n_total) : z₀, schedule[1]; gamma_cols)
    else
        collect(float.(w₀))
    end
    length(w) == _n_w(res) ||
        throw(ArgumentError("w₀ has length $(length(w)), expected $(_n_w(res))"))

    m = res.m
    r_cur, r_try = zeros(m), zeros(m)
    b_cur, b_try = zeros(nc), zeros(nc)
    w_try = zeros(length(w))
    displacement = adaptive_eta ? zeros(length(w)) : Float64[]
    model_change = adaptive_eta ? zeros(m) : Float64[]
    recent = fill(-Inf, nonmonotone)
    history = Float64[]
    iters, best_k0, ρ = 0, Inf, schedule[1]
    residual_valid = false

    for level in schedule
        ρ = level
        residual_valid = false
        _set_rho!(kkt, ρ)
        if nc > 0
            _b!(kkt, b_cur, view(w, 1:n))
            all(isfinite, w) && _sign_min(res, w, b_cur) > 0 || throw(
                ArgumentError("warm start violates a required slack, gamma, phi, or u bound"),
            )
        end
        slot = 0
        fill!(recent, -Inf)
        for _ in 1:max_inner
            r = residual_valid ? r_cur : _resid!(res, r_cur, b_cur, w, ρ)
            residual_valid = true
            nr = norm(r)
            push!(history, nr)
            iters += 1
            isfinite(nr) || break

            k0 = _norm_K0(res, r, w, ρ)
            k0 < best_k0 && (best_k0 = k0)
            k0 < tol && @goto done
            nr < tol_inner && break

            # The source keeps this nonmonotone reference but its Armijo test below
            # reads `nr`; kept as ported so iteration counts match the source.
            slot = slot % length(recent) + 1
            @inbounds recent[slot] = nr

            dw = _bounded_step!(rule, ctx, w, b_cur, r)
            (length(dw) == length(w) && all(isfinite, dw)) || break
            trace === nothing || trace((;
                iter = iters,
                rho = ρ,
                w,
                b = b_cur,
                r,
                dw,
                col_scale = ctx.col_scale,
                eta = bk.eta_used,
            ))

            alpha, ok = 1.0, false
            backtracks, trial_norm = 0, Inf
            while alpha > 1e-14
                _project!(rule, w_try, w, alpha, dw)
                trial_norm = _merit!(res, r_try, b_try, w_try, ρ)
                if trial_norm <= (1 - 1e-4 * alpha) * nr
                    ok = true
                    break
                end
                alpha /= 2
                backtracks += 1
            end
            pred_reduction, actual_reduction, gain_ratio = NaN, NaN, NaN
            if adaptive_eta
                if ok
                    @. displacement = w_try - w
                    _model_Jv!(solver, model_change, bk, w, b_cur, displacement)
                    pred_reduction, actual_reduction, gain_ratio =
                        _gain_reductions(r, model_change, nr, trial_norm)
                else
                    pred_reduction, actual_reduction, gain_ratio = 0.0, 0.0, -Inf
                end
                eta_before = isfinite(bk.eta_used) ? bk.eta_used : ctx.eta[]
                ctx.eta[] = _scheduled_eta(
                    eta_before,
                    gain_ratio,
                    ok ? alpha : 0.0;
                    gain_low,
                    gain_high,
                    eta_min,
                    eta_max,
                    tightening_rate,
                    loosening_rate,
                )
                verbose && println(
                    "    gain = $gain_ratio  alpha = $(ok ? alpha : 0.0)  eta: $eta_before -> $(ctx.eta[])",
                )
            end
            step_trace === nothing || step_trace((;
                iter = iters,
                rho = ρ,
                w,
                b = b_cur,
                r,
                dw,
                col_scale = ctx.col_scale,
                eta = bk.eta_used,
                alpha = ok ? alpha : 0.0,
                accepted = ok,
                backtracks,
                trial_norm,
                w_trial = w_try,
                eta_next = ctx.eta[],
                pred_reduction,
                actual_reduction,
                gain_ratio,
            ))
            ok || break
            copyto!(w, w_try)
            r_cur, r_try = r_try, r_cur
            b_cur, b_try = b_try, b_cur
            residual_valid = true
        end
        verbose && println(
            "  rho = $ρ  iters = $iters  ||R|| = $(isempty(history) ? NaN : history[end])  ||K_0|| = $best_k0",
        )
    end
    @label done

    residual_valid || _resid!(res, r_cur, b_cur, w, ρ)
    k0 = _norm_K0(res, r_cur, w, ρ)
    s = nc == 0 ? Float64[] : w[(n + 1):(n + nc)]
    u = nc > 0 ? w[(n + nc + 1):end] : Float64[]
    converged = k0 < tol
    return (;
        status = converged ? :solved : :failed,
        z = w[kkt.primal_dims],
        w,
        s,
        u,
        rho = ρ,
        residual = k0,
        converged,
        sign_min = _sign_min(res, w, b_cur),
        shortfall = nc == 0 ? 0.0 : maximum(abs, s .* b_cur),
        iters,
        history,
        klu_fallbacks = bk.fallbacks,
        eta = !adaptive_eta && bk.normal !== nothing ? max(ctx.eta[], bk.normal.eta[]) :
              ctx.eta[],
    )
end
