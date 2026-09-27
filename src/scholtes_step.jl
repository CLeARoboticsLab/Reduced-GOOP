# scholtes_step.jl -- one bounded Newton step: backends, step context, projected rule.
#
# Ported from ScholtesReducedGOOP.jl `src/step/{stepctx,steps,rules}.jl`. Only the
# two-metric projected bound rule (`:projected`) is ported: the explicit φ rows
# need it, and this package supports only the φ formulation.

abstract type _ScholtesStepSolver end
struct _NormalSolver <: _ScholtesStepSolver end
struct _KLUEtaSolver <: _ScholtesStepSolver end
struct _KLUSqrtEtaSolver <: _ScholtesStepSolver end
struct _DenseSolver <: _ScholtesStepSolver end

const _SaddleSolver = Union{_KLUEtaSolver,_KLUSqrtEtaSolver}
const SCHOLTES_LINEAR_SOLVERS = (:normal, :klu_eta, :klu_sqrt_eta, :svd)

function _scholtes_step_solver(name::Symbol)
    name === :normal && return _NormalSolver()
    name === :klu_eta && return _KLUEtaSolver()
    name === :klu_sqrt_eta && return _KLUSqrtEtaSolver()
    name === :svd && return _DenseSolver()
    throw(
        ArgumentError(
            "linear_solver must be one of $(SCHOLTES_LINEAR_SOLVERS) with complementarity = :scholtes, got $name",
        ),
    )
end

"""
Which column of y each complementarity row reads its multiplier (γ or φ) off, or
`nothing` if `b(y)` is not a coordinate projection.
"""
function _gamma_columns(kkt::ScholtesKKTSystem)
    kkt.n_comp == 0 && return nothing
    rows, cols = kkt.S_b
    length(rows) == kkt.n_comp || return nothing
    kkt.J_b!(kkt.Jb_buf, randn(kkt.n), kkt.θ, kkt.ρ[])
    kkt.Jvalid[] = false
    all(v -> isapprox(v, 1.0; atol = 1e-12), SparseArrays.nonzeros(kkt.Jb_buf)) || return nothing
    out, seen = zeros(Int, kkt.n_comp), falses(kkt.n_comp)
    for k in eachindex(rows)
        seen[rows[k]] && return nothing
        seen[rows[k]] = true
        out[rows[k]] = cols[k]
    end
    return all(seen) ? out : nothing
end

"""
The caches a backend runs against and the count of dense fallbacks: pinv steps for
`:normal`, SVD fallback steps of `_klu_step_with_fallback!` for the KLU variants.
"""
mutable struct _ScholtesBackend{K<:ScholtesKKTSystem}
    kkt::K
    scholtes::Bool
    saddle::Union{Nothing,_ScholtesSaddle}
    normal::Union{Nothing,_AugmentedNormal}
    dense::Union{Nothing,_AugmentedDense}
    fallbacks::Int
    eta_used::Float64
end

_build_backend(::_NormalSolver, kkt, scholtes) =
    _ScholtesBackend(kkt, scholtes, nothing, _build_augmented_normal(kkt; scholtes), nothing, 0, NaN)
_build_backend(::_KLUEtaSolver, kkt, scholtes) = _ScholtesBackend(
    kkt,
    scholtes,
    _build_scholtes_saddle(kkt; scholtes, sqrt_eta = false),
    nothing,
    nothing,
    0,
    NaN,
)
_build_backend(::_KLUSqrtEtaSolver, kkt, scholtes) = _ScholtesBackend(
    kkt,
    scholtes,
    _build_scholtes_saddle(kkt; scholtes, sqrt_eta = true),
    nothing,
    nothing,
    0,
    NaN,
)
_build_backend(::_DenseSolver, kkt, scholtes) =
    _ScholtesBackend(kkt, scholtes, nothing, nothing, _build_augmented_dense(kkt; scholtes), 0, NaN)

@inline _yv(bk::_ScholtesBackend, w) = view(w, 1:(bk.kkt.n))
@inline _sv(bk::_ScholtesBackend, w) = view(w, (bk.kkt.n + 1):(bk.kkt.n + bk.kkt.n_comp))

function _dense!(bk::_ScholtesBackend, w, b, scale)
    bk.dense === nothing && (bk.dense = _build_augmented_dense(bk.kkt; scholtes = bk.scholtes))
    return _augmented_dense!(bk.dense, bk.kkt, _yv(bk, w), _sv(bk, w), b, scale)
end

# The masked minimum-norm step by this backend's own η ladder, or ok = false. The KLU
# variants always return a step (their SVD fallback is inside `_klu_step_with_fallback!`)
# and record the η they used.
_sparse_step(::_DenseSolver, ::_ScholtesBackend, w, b, r, col_scale; kw...) = (Float64[], false)
function _sparse_step(::_SaddleSolver, bk::_ScholtesBackend, w, b, r, col_scale; eta_init, eta_max, out, kw...)
    fallbacks_before = bk.saddle.svd_fallbacks[]
    delta, η_used = _saddle_step!(bk.saddle, bk.kkt, _yv(bk, w), _sv(bk, w), b, r, col_scale; eta_init, eta_max, out)
    bk.fallbacks += bk.saddle.svd_fallbacks[] - fallbacks_before
    bk.eta_used = η_used
    return delta, true
end
_sparse_step(::_NormalSolver, bk::_ScholtesBackend, w, b, r, col_scale; eta_sticky = true, eta_max = 1e2, kw...) =
    _normal_step!(bk.normal, bk.kkt, _yv(bk, w), _sv(bk, w), b, r, col_scale; sticky = eta_sticky, kw...)

# The η the step just used, including retry escalation.
_factor_eta(::_SaddleSolver, bk::_ScholtesBackend) = bk.eta_used
_factor_eta(::_NormalSolver, bk::_ScholtesBackend) = bk.normal.eta_fac[]
_factor_eta(::_DenseSolver, ::_ScholtesBackend) = NaN

_Jtv!(::_SaddleSolver, out, bk, w, b, u) =
    _saddle_Jtv!(out, bk.saddle, bk.kkt, _yv(bk, w), _sv(bk, w), b, u)
_Jtv!(::_NormalSolver, out, bk, w, b, u) =
    _normal_Jtv!(out, bk.normal, bk.kkt, _yv(bk, w), _sv(bk, w), b, u)
_Jtv!(::_DenseSolver, out, bk, w, b, u) =
    mul!(out, transpose(_dense!(bk, w, b, _Unmasked())), u)

_coldiag!(::_SaddleSolver, out, bk, w, b) =
    _augmented_coldiag!(out, bk.saddle.entries, bk.kkt, _yv(bk, w), _sv(bk, w), b)
_coldiag!(::_NormalSolver, out, bk, w, b) =
    _augmented_coldiag!(out, bk.normal.entries, bk.kkt, _yv(bk, w), _sv(bk, w), b)
_coldiag!(::_DenseSolver, out, bk, w, b) =
    (sum!(abs2, reshape(out, 1, length(out)), _dense!(bk, w, b, _Unmasked())); out)

# J(w)·d at the pre-step iterate, for the η schedule's predicted reduction.
_model_Jv!(::_SaddleSolver, out, bk, w, b, d) =
    _saddle_model_Jv!(out, bk.saddle, bk.kkt, _yv(bk, w), _sv(bk, w), b, d)
_model_Jv!(::_NormalSolver, out, bk, w, b, d) =
    _normal_model_Jv!(out, bk.normal, bk.kkt, _yv(bk, w), _sv(bk, w), b, d)
_model_Jv!(::_DenseSolver, out, bk, w, b, d) = mul!(out, _dense!(bk, w, b, _Unmasked()), d)

"""
Everything the bound rule needs from the solver: the column mask it writes
(`col_scale[c] = 0` pins variable c), the masked minimum-norm `solve(w, b, r)`, the
unmasked `Jtmul(w, b, u) = Jᵀu` and `coldiag(w, b) = diag(JᵀJ)` (internal buffers,
valid until the next call), and the requested Tikhonov parameter `eta`.
"""
struct _ScholtesStepCtx{K,G,S,T,C}
    kkt::K
    n::Int
    nc::Int
    nw::Int
    scholtes::Bool
    gam_col::G
    col_scale::Vector{Float64}
    solve::S
    Jtmul::T
    coldiag::C
    eta::Base.RefValue{Float64}
end

function _build_scholtes_step_ctx(
    solver::_ScholtesStepSolver,
    kkt::ScholtesKKTSystem;
    gamma_cols = _gamma_columns(kkt),
    eta_init = 1e-8,
    eta_max = 1e2,
    eta_sticky::Bool = true,
    refine::Int = 0,
)
    n, nc = kkt.n, kkt.n_comp
    scholtes = nc > 0
    nw = n + nc + (scholtes ? _n_u(kkt) : 0)
    bk = _build_backend(solver, kkt, scholtes)
    col_scale = ones(nw)
    eta = Ref(float(eta_init))
    gq, cd = zeros(nw), zeros(nw)
    direction = zeros(nw)

    function solve(w, b, r)
        bk.eta_used = NaN
        d, ok = _sparse_step(solver, bk, w, b, r, col_scale; eta_init = eta[], refine, eta_sticky, eta_max, out = direction)
        if ok
            bk.eta_used = abs(_factor_eta(solver, bk))
            return d
        end
        # `:normal` fell through its ladder (the source's pinv fallback); `:svd` is pinv.
        solver isa _DenseSolver || (bk.fallbacks += 1)
        return LinearAlgebra.pinv(_dense!(bk, w, b, col_scale)) * (-r)
    end
    Jtmul(w, b, u) = _Jtv!(solver, gq, bk, w, b, u)
    coldiag(w, b) = _coldiag!(solver, cd, bk, w, b)

    ctx = _ScholtesStepCtx(
        kkt,
        n,
        nc,
        nw,
        scholtes,
        j -> gamma_cols === nothing ? 0 : gamma_cols[j],
        col_scale,
        solve,
        Jtmul,
        coldiag,
        eta,
    )
    return ctx, bk
end

# The sign-constrained coordinates of w: every slack, the b column (γ or φ) when b
# is a coordinate projection, and on relaxed rows the u coordinate.
@inline function _sign_coords(f, ctx::_ScholtesStepCtx, w)
    n, nc = ctx.n, ctx.nc
    @inbounds for j in 1:nc
        f(n + j, w[n + j])
        c = ctx.gam_col(j)
        c != 0 && f(c, w[c])
        ucol = _u_col(ctx.kkt, j)
        ctx.scholtes && ucol != 0 && f(ucol, w[ucol])
    end
end

# Refill `idx` with the sign-constrained coordinates and `bnd` with `scale · c_k(w)`.
@inline function _sign_box!(idx::Vector{Int}, bnd::Vector{Float64}, ctx::_ScholtesStepCtx, w, scale::Float64)
    empty!(idx)
    empty!(bnd)
    _sign_coords(ctx, w) do i, v
        push!(idx, i)
        push!(bnd, scale * v)
    end
    return idx
end

"""
Bertsekas' two-metric projection, adapted (source `:projected`). At w with r = R(w):

 1. floor `flr_k = (1 − τ) c_k(w)`, a box re-centred on w;
 2. hold `A = { k : c_k(w) − flr_k ≤ ε_w and (Jᵀr)_k > 0 }`;
 3. free block: masked minimum-norm solve; held block: `d_i = max(−g_i / H_ii, flr_k − w_i)`
    with `H_ii = ‖J e_i‖² + 1e-16`;
 4. trial point `clip(w + α d, flr)`.

`ε_w = min(eps, projected-gradient norm)`.
"""
struct _ProjectedRule
    tau::Float64
    eps::Float64
    idx::Vector{Int}
    flr::Vector{Float64}
    held::Vector{Int}
    gbuf::Vector{Float64}
    function _ProjectedRule(nw::Int; tau::Real = 0.99, eps::Real = 1e-12)
        (0 < tau < 1) || throw(ArgumentError("tau must lie in (0,1), got $tau"))
        eps >= 0 || throw(ArgumentError("proj_eps must be >= 0, got $eps"))
        new(float(tau), float(eps), Int[], Float64[], Int[], zeros(nw))
    end
end

const _PROJ_ETA2 = 1e-16

function _bounded_step!(rule::_ProjectedRule, ctx::_ScholtesStepCtx, w, b, r)
    empty!(rule.held)
    _sign_box!(rule.idx, rule.flr, ctx, w, 1 - rule.tau)
    fill!(ctx.col_scale, 1.0)
    isempty(rule.idx) && return ctx.solve(w, b, r)

    g = copyto!(rule.gbuf, ctx.Jtmul(w, b, r))
    pg = 0.0
    @inbounds for (k, i) in enumerate(rule.idx)
        pg += (w[i] - max(w[i] - g[i], rule.flr[k]))^2
    end
    epsk = min(rule.eps, sqrt(pg))
    @inbounds for (k, i) in enumerate(rule.idx)
        if w[i] - rule.flr[k] <= epsk && g[i] > 0
            ctx.col_scale[i] = 0.0
            push!(rule.held, i)
        end
    end

    dw = ctx.solve(w, b, r)
    (isempty(rule.held) || length(dw) != ctx.nw || !all(isfinite, dw)) && return dw

    h = ctx.coldiag(w, b)
    @inbounds for (k, i) in enumerate(rule.idx)
        ctx.col_scale[i] == 0.0 || continue
        dw[i] = max(-g[i] / (h[i] + _PROJ_ETA2), rule.flr[k] - w[i])
    end
    return dw
end

# The projected arc: free coordinates may overshoot and are clipped to the floor.
function _project!(rule::_ProjectedRule, wtry, w, alpha, dw)
    @. wtry = w + alpha * dw
    @inbounds for (k, i) in enumerate(rule.idx)
        wtry[i] < rule.flr[k] && (wtry[i] = rule.flr[k])
    end
    return wtry
end
