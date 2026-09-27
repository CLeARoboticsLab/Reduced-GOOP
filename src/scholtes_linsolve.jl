# scholtes_linsolve.jl -- the Scholtes Newton step's linear algebra.
#
# Ported from ScholtesReducedGOOP.jl `src/step/linsolve.jl`. The augmented Jacobian
#
#     ∇R = [ J_nc(y)          0        0 ]
#          [ J_a(y)          −I        0 ]
#          [ diag(s)·J_b(y)   diag(b)  I ]      (the u column only on relaxed rows)
#
# is rectangular and rank deficient, so the step is the minimum-norm least-squares
#
#     δ = −Jᵀ(η² I + J Jᵀ)⁻¹ r
#
# computed by one of:
#
#     :normal        Cholesky of J Jᵀ + η² I                       AugmentedNormal
#     :klu_eta       LU of [ηI J; Jᵀ −ηI]   (the source's :klu convention)  AugmentedKKTCache
#     :klu_sqrt_eta  LU of [√η I J; Jᵀ −√η I]  (this package's IP convention, where η
#                    is the Tikhonov shift itself)                 AugmentedKKTCache
#     :svd           dense pinv (the source's :pinv)               _AugmentedDense
#
# A column mask `col_scale` pins the coordinates the projected bound rule holds.

"""
Every structural entry of the augmented Jacobian, flattened: `block` (1 = J_nc,
2 = J_a, 3 = −I, 4 = s·J_b, 5 = diag(b), 6 = +I on u), its row within that block
`src_r`, its index into that block's `nonzeros` `src_k`, and its column `col`.
"""
struct _EntryList
    src_r::Vector{Int}
    src_k::Vector{Int}
    block::Vector{Int}
    col::Vector{Int}
end

function _augmented_pattern(kkt::ScholtesKKTSystem; scholtes::Bool = false)
    n_nc, ncomp, n = kkt.n_nc, kkt.n_comp, kkt.n
    m = n_nc + 2 * ncomp
    p = n + ncomp + (scholtes ? _n_u(kkt) : 0)
    rows, cols, src_r, src_k, block = Int[], Int[], Int[], Int[], Int[]
    entry!(r, c, sr, sk, blk) =
        (push!(rows, r); push!(cols, c); push!(src_r, sr); push!(src_k, sk); push!(block, blk))

    let (r, c) = kkt.S_nc
        for k in eachindex(r)
            entry!(r[k], c[k], r[k], k, 1)
        end
    end
    let (r, c) = kkt.S_a
        for k in eachindex(r)
            entry!(n_nc + r[k], c[k], r[k], k, 2)
        end
    end
    for j in 1:ncomp
        entry!(n_nc + j, n + j, j, 0, 3)
    end
    let (r, c) = kkt.S_b
        for k in eachindex(r)
            entry!(n_nc + ncomp + r[k], c[k], r[k], k, 4)
        end
    end
    for j in 1:ncomp
        entry!(n_nc + ncomp + j, n + j, j, 0, 5)
    end
    if scholtes
        for j in 1:ncomp
            uj = kkt.u_index[j]
            uj == 0 || entry!(n_nc + ncomp + j, n + ncomp + uj, j, 0, 6)
        end
    end
    return (; rows, cols, entries = _EntryList(src_r, src_k, block, cols), m, p)
end

@inline function _entry_value(blk, r, sk, vnc, va, vb, s, bv)
    blk == 1 ? vnc[sk] :
    blk == 2 ? va[sk] :
    blk == 3 ? -1.0 :
    blk == 4 ? s[r] * vb[sk] :
    blk == 5 ? bv[r] : 1.0
end

# An all-ones column mask for the unmasked matvecs.
struct _Unmasked end
@inline Base.getindex(::_Unmasked, ::Int) = 1.0

@inline function _scatter(f, e::_EntryList, kkt::ScholtesKKTSystem, y, s, bv, col_scale)
    vnc, va, vb = _fill_jacobians!(kkt, y)
    @inbounds for k in eachindex(e.block)
        f(
            k,
            _entry_value(e.block[k], e.src_r[k], e.src_k[k], vnc, va, vb, s, bv) *
            col_scale[e.col[k]],
        )
    end
    return nothing
end

"Refill the three Jacobian buffers at `y`, memoized on the value of `y`."
function _fill_jacobians!(kkt::ScholtesKKTSystem, y)
    if !(kkt.Jvalid[] && _same_y(kkt.Jy, y))
        kkt.J_nc!(kkt.Jnc_buf, y, kkt.θ, kkt.ρ[])
        kkt.J_a!(kkt.Ja_buf, y, kkt.θ, kkt.ρ[])
        kkt.J_b!(kkt.Jb_buf, y, kkt.θ, kkt.ρ[])
        copyto!(kkt.Jy, y)
        kkt.Jvalid[] = true
    end
    return (
        SparseArrays.nonzeros(kkt.Jnc_buf),
        SparseArrays.nonzeros(kkt.Ja_buf),
        SparseArrays.nonzeros(kkt.Jb_buf),
    )
end

@inline function _same_y(cached::Vector{Float64}, y)
    length(cached) == length(y) || return false
    @inbounds for i in eachindex(cached)
        cached[i] == y[i] || return false
    end
    return true
end

function _slot_finder(A::SparseArrays.SparseMatrixCSC)
    Arows = SparseArrays.rowvals(A)
    return (i, j) ->
        (rng = SparseArrays.nzrange(A, j); rng[searchsortedfirst(view(Arows, rng), i)])
end

# The η-escalation ladder: try `attempt!(δ, η)`, on failure grow η. `ok = false`
# means every attempt failed and the caller falls back to a dense step.
function _escalate(attempt!, cache, eta0::Real, growth::Real, retries::Int; forget_factor::Bool, out = nothing)
    eta = float(eta0)
    delta = out === nothing ? Vector{Float64}(undef, cache.p) : out
    length(delta) == cache.p || throw(DimensionMismatch("step output has the wrong length"))
    for _ in 0:retries
        attempt!(delta, eta) && return (delta, true)
        forget_factor && (cache.fact[] = nothing)
        eta *= growth
    end
    return (Float64[], false)
end

const _REFINE_RTOL = 1e-14

# --- saddle-point LU (:klu_eta, :klu_sqrt_eta) --------------------------------------
#
# Both variants run on this package's `AugmentedKKTCache` (src/solver.jl): the
# augmented Jacobian J is assembled in place into a fixed-pattern sparse matrix with
# the column mask applied, and `_klu_step_with_fallback!` factorizes
# `[dI J; Jᵀ −dI]` with `d = √η` (`:klu_sqrt_eta`) or `d = η` (`:klu_eta`), escalating η
# on a singular factorization and falling back to a dense Tikhonov SVD step.

"""
The masked augmented Jacobian `J` (m × p, fixed pattern, `Jpos[k]` the slot of entry
k) and the `AugmentedKKTCache` built on its pattern, plus the singular-retry and
SVD-fallback counters of `_klu_step_with_fallback!`.
"""
struct _ScholtesSaddle
    J::SparseArrays.SparseMatrixCSC{Float64,Int}
    Jpos::Vector{Int}
    entries::_EntryList
    cache::AugmentedKKTCache
    singular_retries::Base.RefValue{Int}
    svd_fallbacks::Base.RefValue{Int}
    m::Int
    p::Int
end

function _build_scholtes_saddle(kkt::ScholtesKKTSystem; scholtes::Bool = false, sqrt_eta::Bool = true)
    pat = _augmented_pattern(kkt; scholtes)
    J = SparseArrays.sparse(pat.rows, pat.cols, ones(length(pat.rows)), pat.m, pat.p)
    at = _slot_finder(J)
    Jpos = [at(pat.rows[t], pat.cols[t]) for t in eachindex(pat.rows)]
    cache = _build_augmented_kkt_cache(J, pat.m, pat.p; sqrt_eta)
    _ScholtesSaddle(J, Jpos, pat.entries, cache, Ref(0), Ref(0), pat.m, pat.p)
end

"Scatter the current Jacobian blocks and the column mask into `S.J`. Allocation-free."
function _fill_saddle!(S::_ScholtesSaddle, kkt, y, s, bv, col_scale)
    nz, Jpos = SparseArrays.nonzeros(S.J), S.Jpos
    _scatter(S.entries, kkt, y, s, bv, col_scale) do k, v
        @inbounds nz[Jpos[k]] = v
    end
    return S
end

"""
The masked minimum-norm step at this iterate: fill J, refresh the augmented matrix,
and run `_klu_step_with_fallback!`. Returns `(δ, η_used)`; the step always exists (the
SVD fallback covers a singular factorization).
"""
function _saddle_step!(S::_ScholtesSaddle, kkt, y, s, bv, r, col_scale; eta_init, eta_max, out)
    _fill_saddle!(S, kkt, y, s, bv, col_scale)
    _update_augmented_kkt!(S.cache, S.J, eta_init)
    η_used = _klu_step_with_fallback!(
        out,
        S.cache,
        S.J,
        r,
        eta_init,
        eta_max,
        false;
        singular_retry_counter = S.singular_retries,
        svd_fallback_counter = S.svd_fallbacks,
    )
    return out, η_used
end

function _saddle_Jtv!(out, S::_ScholtesSaddle, kkt, y, s, bv, u)
    _fill_saddle!(S, kkt, y, s, bv, _Unmasked())
    return mul!(out, transpose(S.J), u)
end

function _saddle_model_Jv!(out, S::_ScholtesSaddle, kkt, y, s, bv, d)
    _fill_saddle!(S, kkt, y, s, bv, _Unmasked())
    return mul!(out, S.J, d)
end

# --- normal equations (:normal) -----------------------------------------------------

"""
`J` as an `m × p` sparse matrix and the Cholesky factorization of `J Jᵀ + η² I`.
`eta` is the last η that factorized (the sticky floor), `eta_fac` the η the standing
factors carry. `gram_*` replay `triu(J Jᵀ)` on `J`'s pattern so it is refilled in place.
"""
struct _AugmentedNormal
    J::SparseArrays.SparseMatrixCSC{Float64,Int}
    fact::Base.RefValue{Any}
    eta::Base.RefValue{Float64}
    eta_fac::Base.RefValue{Float64}
    nesc::Base.RefValue{Int}
    entries::_EntryList
    Jpos::Vector{Int}
    lam::Vector{Float64}
    rhs::Vector{Float64}
    res::Vector{Float64}
    cor::Vector{Float64}
    tp::Vector{Float64}
    gram::Base.RefValue{Any}
    gram_ptr::Vector{Int}
    gram_a::Vector{Int}
    gram_b::Vector{Int}
    m::Int
    p::Int
end

"""
The upper triangle of `J Jᵀ` replayed on `J`'s pattern, in the order SparseArrays'
Gustavson product accumulates it, so `_gram_values!` is bitwise `triu(J * J')`.
"""
function _gram_plan(J::SparseArrays.SparseMatrixCSC)
    m = size(J, 1)
    colptr = SparseArrays.getcolptr(J)
    slotJ = SparseArrays.SparseMatrixCSC(
        size(J)...,
        copy(colptr),
        copy(SparseArrays.rowvals(J)),
        collect(1:SparseArrays.nnz(J)),
    )
    Jt = copy(transpose(slotJ))
    ones_ = SparseArrays.SparseMatrixCSC(
        size(J)...,
        copy(colptr),
        copy(SparseArrays.rowvals(J)),
        ones(SparseArrays.nnz(J)),
    )
    U = LinearAlgebra.triu(ones_ * transpose(ones_))
    rowJ, rowJt, rowU = SparseArrays.rowvals(J), SparseArrays.rowvals(Jt), SparseArrays.rowvals(U)

    slot_of = zeros(Int, m)
    count = zeros(Int, SparseArrays.nnz(U))
    visit(f) =
        for i in 1:m
            for s in SparseArrays.nzrange(U, i)
                slot_of[rowU[s]] = s
            end
            for jp in SparseArrays.nzrange(Jt, i)
                j, bj = rowJt[jp], SparseArrays.nonzeros(Jt)[jp]
                for kp in SparseArrays.nzrange(J, j)
                    rowJ[kp] <= i && f(slot_of[rowJ[kp]], kp, bj)
                end
            end
        end
    visit((s, _, _) -> (count[s] += 1))
    ptr = cumsum([1; count])
    a, b = Vector{Int}(undef, ptr[end] - 1), Vector{Int}(undef, ptr[end] - 1)
    next = ptr[1:(end - 1)]
    visit((s, ka, kb) -> (a[next[s]] = ka; b[next[s]] = kb; next[s] += 1))
    return ptr, a, b
end

# Plain `v += x*y`, exactly as `spcolmul!` accumulates (no @simd/muladd).
function _gram_values!(x, nzJ::Vector{Float64}, ptr::Vector{Int}, a::Vector{Int}, b::Vector{Int})
    @inbounds for s in 1:(length(ptr) - 1)
        q = ptr[s]
        v = nzJ[a[q]] * nzJ[b[q]]
        for q in (ptr[s] + 1):(ptr[s + 1] - 1)
            v += nzJ[a[q]] * nzJ[b[q]]
        end
        x[s] = v
    end
    return x
end

# `J Jᵀ` as CHOLMOD wants it: built once as `Sparse(Symmetric(triu(J J')))`, then
# refilled in place along the gram plan. Falls back to `Symmetric(J * J')` if the
# handle does not map 1:1 onto the plan.
function _normal_gram!(cache::_AugmentedNormal)
    S = cache.gram[]
    if S === nothing
        C = cache.J * transpose(cache.J)
        U = LinearAlgebra.triu(C)
        S = SparseArrays.CHOLMOD.Sparse(LinearAlgebra.Symmetric(U))
        raw = unsafe_load(pointer(S))
        planned = length(cache.gram_ptr) - 1
        (
            raw.nzmax == SparseArrays.nnz(U) == planned &&
            raw.stype == 1 &&
            SparseArrays.getcolptr(U)[end] - 1 == planned
        ) || return LinearAlgebra.Symmetric(C)
        cache.gram[] = S
        return S
    end
    raw = unsafe_load(pointer(S))
    x = unsafe_wrap(Array, Ptr{Float64}(raw.x), length(cache.gram_ptr) - 1)
    _gram_values!(x, SparseArrays.nonzeros(cache.J), cache.gram_ptr, cache.gram_a, cache.gram_b)
    return S
end

function _build_augmented_normal(kkt::ScholtesKKTSystem; scholtes::Bool = false)
    pat = _augmented_pattern(kkt; scholtes)
    m, p = pat.m, pat.p
    J = SparseArrays.sparse(pat.rows, pat.cols, ones(length(pat.rows)), m, p)
    at = _slot_finder(J)
    gram_ptr, gram_a, gram_b = _gram_plan(J)
    _AugmentedNormal(
        J,
        Ref{Any}(nothing),
        Ref(0.0),
        Ref(0.0),
        Ref(0),
        pat.entries,
        [at(pat.rows[t], pat.cols[t]) for t in eachindex(pat.rows)],
        zeros(m),
        zeros(m),
        zeros(m),
        zeros(m),
        zeros(p),
        Ref{Any}(nothing),
        gram_ptr,
        gram_a,
        gram_b,
        m,
        p,
    )
end

function _update_normal!(cache::_AugmentedNormal, kkt, y, s, bv, col_scale)
    nz, Jpos = SparseArrays.nonzeros(cache.J), cache.Jpos
    _scatter(cache.entries, kkt, y, s, bv, col_scale) do k, v
        @inbounds nz[Jpos[k]] = v
    end
    return cache
end

function _normal_factor!(cache::_AugmentedNormal, kkt, y, s, bv, eta::Float64, col_scale)
    _update_normal!(cache, kkt, y, s, bv, col_scale)
    A = _normal_gram!(cache)
    try
        if cache.fact[] === nothing
            cache.fact[] = LinearAlgebra.cholesky(A; shift = eta^2)
        else
            LinearAlgebra.cholesky!(cache.fact[], A; shift = eta^2)
        end
        cache.eta_fac[] = eta
        return true
    catch err
        cache.nesc[] += 1
        err isa LinearAlgebra.PosDefException && return false
        if err isa ArgumentError || err isa DimensionMismatch
            cache.fact[] = nothing
            return false
        end
        rethrow()
    end
end

function _normal_resolve!(out, cache::_AugmentedNormal, r; refine::Int = 0)
    cache.fact[] === nothing && return false
    try
        @. cache.rhs = -r
        ldiv!(cache.lam, cache.fact[], cache.rhs)
        nb = refine > 0 ? norm(cache.rhs) : 0.0
        if nb > 0
            eta2 = cache.eta_fac[]^2
            for _ in 1:refine
                mul!(cache.tp, transpose(cache.J), cache.lam)
                mul!(cache.res, cache.J, cache.tp)
                @. cache.res = cache.rhs - cache.res - eta2 * cache.lam
                norm(cache.res) <= _REFINE_RTOL * nb && break
                ldiv!(cache.cor, cache.fact[], cache.res)
                all(isfinite, cache.cor) || break
                @. cache.lam += cache.cor
            end
        end
        mul!(out, transpose(cache.J), cache.lam)
        return all(isfinite, out)
    catch err
        err isa LinearAlgebra.PosDefException || rethrow()
        return false
    end
end

# Sticky η: the ladder starts at max(last η that worked, eta_init) unless `sticky = false`
# (the η schedule), and a failed attempt discards the factorization.
_normal_step!(cache::_AugmentedNormal, kkt, y, s, bv, r, col_scale; eta_init = 1e-8, eta_growth = 100.0, max_retries = 5, refine::Int = 0, sticky::Bool = true, out = nothing) =
    _escalate(
        cache,
        sticky ? max(cache.eta[], float(eta_init)) : float(eta_init),
        eta_growth,
        max_retries;
        forget_factor = true,
        out,
    ) do delta, eta
        ok =
            _normal_factor!(cache, kkt, y, s, bv, eta, col_scale) &&
            _normal_resolve!(delta, cache, r; refine)
        ok && (cache.eta[] = eta)
        ok
    end

function _normal_Jtv!(out, cache::_AugmentedNormal, kkt, y, s, bv, u)
    _update_normal!(cache, kkt, y, s, bv, _Unmasked())
    return mul!(out, transpose(cache.J), u)
end

function _normal_model_Jv!(out, cache::_AugmentedNormal, kkt, y, s, bv, d)
    _update_normal!(cache, kkt, y, s, bv, _Unmasked())
    return mul!(out, cache.J, d)
end

"`diag(JᵀJ)` of the unmasked augmented Jacobian: one pass over the entry list."
function _augmented_coldiag!(out, entries::_EntryList, kkt, y, s, bv)
    col = entries.col
    fill!(out, 0.0)
    _scatter(entries, kkt, y, s, bv, _Unmasked()) do k, v
        @inbounds out[col[k]] += v * v
    end
    return out
end

# --- dense (:svd, and every sparse backend's fallback) ------------------------------

struct _AugmentedDense
    J::Matrix{Float64}
    entries::_EntryList
    pos::Vector{Int}
    m::Int
    p::Int
end

function _build_augmented_dense(kkt::ScholtesKKTSystem; scholtes::Bool = false)
    pat = _augmented_pattern(kkt; scholtes)
    pos = [(pat.cols[t] - 1) * pat.m + pat.rows[t] for t in eachindex(pat.rows)]
    return _AugmentedDense(zeros(pat.m, pat.p), pat.entries, pos, pat.m, pat.p)
end

function _augmented_dense!(cache::_AugmentedDense, kkt, y, s, bv, col_scale)
    J, pos = cache.J, cache.pos
    _scatter(cache.entries, kkt, y, s, bv, col_scale) do k, v
        @inbounds J[pos[k]] = v
    end
    return J
end
