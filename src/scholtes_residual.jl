# scholtes_residual.jl -- R(w; ρ), ‖K₀‖, the sign box and the merit.
#
# Ported from ScholtesReducedGOOP.jl `src/residual.jl`. The iterate is
# w = [y; s; u] and
#
#     R(w; ρ) = [ F_nc(y) ; a(y) − s ; s ⊙ b(y) + u − ρ𝟙 ]
#
# where exact φ rows (u_index = 0) carry neither u nor ρ in their product row. With
# no complementarity rows at all the u block is absent and the solve is plain
# Newton on F_nc.

"""
Scratch for evaluating `R(w; ρ)` and its companions on a `ScholtesKKTSystem`. The
caller owns the residual and `b(y)` buffers, because the line search needs two of
each live at once.
"""
struct ScholtesResiduals{K<:ScholtesKKTSystem}
    kkt::K
    n::Int
    nc::Int
    n_nc::Int
    m::Int
    scholtes::Bool
    Fb::Vector{Float64}
    ab::Vector{Float64}
end

function ScholtesResiduals(kkt::ScholtesKKTSystem)
    n, nc, n_nc = kkt.n, kkt.n_comp, kkt.n_nc
    ScholtesResiduals(kkt, n, nc, n_nc, n_nc + 2nc, nc > 0, zeros(n_nc), zeros(nc))
end

"Width of `w`: `[y; s]`, plus `u` when there are complementarity rows."
_n_w(R::ScholtesResiduals) = R.n + R.nc + (R.scholtes ? _n_u(R.kkt) : 0)

"""
    _resid!(R, dst, bbuf, w, ρ; fill_b = true) -> dst

`R(w; ρ)` into `dst`, refreshing `bbuf = b(y)` on the way. `fill_b = false` says `bbuf`
already holds `b(y)` for this `w` (only `_merit!` passes it).
"""
function _resid!(R::ScholtesResiduals, dst, bbuf, w, ρ; fill_b::Bool = true)
    _set_rho!(R.kkt, ρ)
    n, nc, n_nc = R.n, R.nc, R.n_nc
    y = view(w, 1:n)
    fill_b && nc > 0 && _b!(R.kkt, bbuf, y)
    _F_nc!(R.kkt, R.Fb, y)
    nc > 0 && _a!(R.kkt, R.ab, y)
    u_index = R.kkt.u_index
    @inbounds begin
        copyto!(dst, 1, R.Fb, 1, n_nc)
        for j in 1:nc
            dst[n_nc + j] = R.ab[j] - w[n + j]
            uj = u_index[j]
            dst[n_nc + nc + j] =
                uj == 0 ? w[n + j] * bbuf[j] :
                w[n + j] * bbuf[j] - ρ + (R.scholtes ? w[n + nc + uj] : 0.0)
        end
    end
    return dst
end

"""
    _norm_K0(R, Rvec, w, ρ) -> Float64

`‖K₀(w)‖`, the unperturbed KKT residual, recovered from `Rvec = R(w; ρ)`: every ρ
offset is removed, both in the relaxed product rows and in the φ slack-definition
rows (whose `a` contains ρ).
"""
function _norm_K0(R::ScholtesResiduals, Rvec, w, ρ)
    n, nc, n_nc = R.n, R.nc, R.n_nc
    unshifted_end = n_nc + R.kkt.original_nc
    u_index = R.kkt.u_index
    acc = 0.0
    @inbounds for i in 1:(n_nc + nc)
        v = Rvec[i]
        i > unshifted_end && (v -= ρ)
        acc += v^2
    end
    @inbounds for j in 1:nc
        uj = u_index[j]
        v =
            uj == 0 ? Rvec[n_nc + nc + j] :
            Rvec[n_nc + nc + j] + ρ - (R.scholtes ? w[n + nc + uj] : 0.0)
        acc += v^2
    end
    return sqrt(acc)
end

"""
    _sign_min(R, w, bbuf) -> Float64

The smallest sign-box margin: every slack, every b coordinate (γ or φ) and every
original u. `Inf` when there are no complementarity rows.
"""
function _sign_min(R::ScholtesResiduals, w, bbuf)
    R.nc == 0 && return Inf
    n, nc = R.n, R.nc
    u_index = R.kkt.u_index
    mn = Inf
    @inbounds for j in 1:nc
        mn = min(mn, w[n + j])
        mn = min(mn, bbuf[j])
        uj = u_index[j]
        R.scholtes && uj != 0 && (mn = min(mn, w[n + nc + uj]))
    end
    return mn
end

"""
    _merit!(R, dst, bbuf, w, ρ) -> Float64

The line-search merit: `Inf` outside the sign box, `‖R(w; ρ)‖` inside it.
"""
function _merit!(R::ScholtesResiduals, dst, bbuf, w, ρ)
    _set_rho!(R.kkt, ρ)
    R.nc > 0 && _b!(R.kkt, bbuf, view(w, 1:(R.n)))
    _sign_min(R, w, bbuf) > 0 || return Inf
    return norm(_resid!(R, dst, bbuf, w, ρ; fill_b = false))
end
