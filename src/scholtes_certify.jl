# scholtes_certify.jl -- judging a fixed-ρ solve.
#
# Ported from ScholtesReducedGOOP.jl `src/certify.jl`.
# ‖K₀‖ adds a part Newton earns (stationarity and slack definitions) to a part
# pinned at O(ρ) by the relaxation (complementarity), so at a fixed ρ a solve is
# judged on the three hypotheses of Scholtes' theorem instead: `stat_feas`, the
# `shortfall` ‖s ⊙ γ‖∞ ≈ ρ, and the constraint margin.

"""
    stat_feas(kkt, θ, result) -> Float64

The part of ‖K₀‖ that can converge: `‖F_nc(y)‖` together with `‖a(y) − s‖`.
"""
function stat_feas(kkt::ScholtesKKTSystem, θ, r)
    _set_theta!(kkt, θ)
    _set_rho!(kkt, r.rho)
    y = view(r.w, 1:(kkt.n))
    e = norm(_F_nc(kkt, y))
    kkt.n_comp == 0 && return e
    return hypot(e, norm(_a(kkt, y) .- r.s))
end

"""
    certified_ladder(; rho = 1e-2, stop = 1e-10) -> Vector{Vector{Float64}}

The default escalation for `solve_certified`: one fixed level, then two progressively
finer walks down to it, then the full schedule.
"""
certified_ladder(; rho = 1e-2, stop = 1e-10) = [
    [float(rho)],
    geometric_schedule(1e-1; stop = rho, per_decade = 1),
    geometric_schedule(1.0; stop = rho, per_decade = 2),
    geometric_schedule(1.0; stop = stop, per_decade = 4),
]

# One-level rungs get a large budget; multi-level rungs fall through on budget.
_rung_budget(schedule) = length(schedule) == 1 ? 4000 : 500

"""
    solve_certified(kkt, θ, z₀; ladder = certified_ladder(), tol_stat = 1e-6,
                    margin_tol = -1e-8, shortfall_factor = 10.0, max_inner = nothing,
                    options = ScholtesOptions()) -> NamedTuple

Solve, check the certificate, and escalate the ρ walk until it passes. A rung is
accepted when `stat_feas < tol_stat`, every exact (φ) pair's product is below
`tol_stat`, `shortfall ≤ shortfall_factor · min(ρ)` and the worst constraint
`min a(y) ≥ margin_tol`. Returns `(; result, rung, certified, stat, shortfall, margin,
attempts)`; `rung = 0` when no rung passed.
"""
function solve_certified(
    kkt::ScholtesKKTSystem,
    θ,
    z₀;
    ladder = certified_ladder(),
    tol_stat = 1e-6,
    margin_tol = -1e-8,
    shortfall_factor = 10.0,
    max_inner = nothing,
    verbose = false,
    options::ScholtesOptions = ScholtesOptions(),
)
    isempty(ladder) && throw(ArgumentError("ladder is empty"))
    local last_certificate
    for (k, schedule) in enumerate(ladder)
        rung_options = _with_options(
            options;
            rho_schedule = collect(float.(schedule)),
            max_inner = something(max_inner, _rung_budget(schedule)),
        )
        r = solve(Scholtes(), kkt, θ; z₀, options = rung_options)
        st = stat_feas(kkt, θ, r)
        mg = kkt.n_comp == 0 ? Inf : minimum(_a(kkt, view(r.w, 1:(kkt.n))))
        exact_rows = findall(iszero, kkt.u_index)
        bv = _b(kkt, view(r.w, 1:(kkt.n)))
        exact_error = maximum(abs, r.s[exact_rows] .* bv[exact_rows]; init = 0.0)
        ok =
            st < tol_stat &&
            exact_error < tol_stat &&
            r.shortfall <= shortfall_factor * minimum(schedule) &&
            mg >= margin_tol
        verbose && println(
            "  rung $k ($(length(schedule)) levels, rho_min = $(minimum(schedule))): stat $st  shortfall $(r.shortfall)  margin $mg  $(ok ? "PASS" : "fail")",
        )
        last_certificate = (;
            result = r,
            rung = ok ? k : 0,
            certified = ok,
            stat = st,
            shortfall = r.shortfall,
            margin = mg,
            attempts = k,
        )
        ok && return last_certificate
    end
    return last_certificate
end
