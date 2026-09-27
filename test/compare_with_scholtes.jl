# Per-solve comparison of the ported Scholtes tests (test/scholtes.jl) against a recording
# of ScholtesReducedGOOP.jl's own test suite (the Step 1 target reference).
#
#   julia --project=. test/compare_with_scholtes.jl <scholtes_tests.jls> [out.md]
#
# Every `solve(::Scholtes, …)` made while test/scholtes.jl runs is recorded with its
# testset. A port solve is paired with the source solve of the same testset, problem
# size and solver options (backend names mapped: :klu → :klu_eta, :pinv → :svd), in
# order. Source solves of configurations the port does not have (:ldl, :freeze, :ideal,
# scholtes = false, SolverWorkspace, φ = false) are left unpaired, and so are port solves
# of backends the source lacks (:klu_sqrt_eta).
#
# "Match" is the plan's acceptance criterion, per solve: same `converged` and ρ reached;
# ‖Δz‖∞ ≤ 1e-6; ‖K₀‖ within a relative 1e-6 (absolute floor 1e-10); iteration counts
# within ±1. The non-unique-answer testset is compared as a set of distinct points.

using ReducedGOOP, Test, Serialization, Printf, LinearAlgebra

const REF = ARGS[1]
const OUT = length(ARGS) >= 2 ? ARGS[2] : nothing
const RECORDS = Any[]

function ReducedGOOP.solve(s::ReducedGOOP.Scholtes, kkt::ReducedGOOP.ScholtesKKTSystem,
                           θ::AbstractVector{Float64}; kw...)
    r = invoke(ReducedGOOP.solve,
               Tuple{ReducedGOOP.Scholtes,ReducedGOOP.ScholtesKKTSystem,AbstractVector{<:Real}},
               s, kkt, θ; kw...)
    ts = Test.get_testset()
    o = get(kw, :options, ReducedGOOP.ScholtesOptions())
    push!(RECORDS, (; testset = ts isa Test.FallbackTestSet ? "" : ts.description,
                      n = kkt.n, ls = o.linear_solver, sched = o.rho_schedule,
                      max_inner = o.max_inner, eta_schedule = o.eta_schedule,
                      eta_init = o.eta_init, nonmonotone = o.nonmonotone, tol = o.tol,
                      has_w0 = get(kw, :w₀, nothing) !== nothing,
                      z = copy(r.z), converged = r.converged, rho = r.rho,
                      residual = r.residual, iters = r.iters))
    r
end

@testset "ROOT" begin
    include(joinpath(@__DIR__, "scholtes.jl"))
end

ref = deserialize(REF)
maplsym(ls) = ls === :klu ? :klu_eta : ls === :pinv ? :svd : ls
src_sig(s) = (s.testset, s.dims.n, maplsym(get(s.options, :linear_solver, :klu)),
              s.schedule, get(s.options, :max_inner, 200), get(s.options, :eta_schedule, false),
              get(s.options, :eta_init, 1e-8), get(s.options, :nonmonotone, 1), s.has_w0)
function port_sig(p)
    sched = p.sched === nothing ? ReducedGOOP.geometric_schedule(1.0; stop = 1e-10) : p.sched
    (p.testset, p.n, p.ls, sched, p.max_inner, p.eta_schedule, p.eta_init, p.nonmonotone, p.has_w0)
end
# the port's testset names that differ from the source's
const RENAMED = Dict(
    "Scholtes (ported from ScholtesReducedGOOP.jl)" => "ScholtesReducedGOOP",
    "the projected bound rule stays in the sign box" => "the three bound rules agree, and all stay in the sign box",
    "refinement brings :normal onto the saddle-point step" => "refinement makes the backends compute the SAME step",
    "every backend solves with the projected rule -- no crashes" => "every (backend, rule, closure) pair solves -- no crashes",
    "what the projected rule promises the line search at alpha = 1" => "what each rule promises the line search at alpha = 1",
)
ports = [merge(p, (; testset = get(RENAMED, p.testset, p.testset))) for p in RECORDS]
srcs = [s for s in ref.solves if get(s.options, :step, :projected) === :projected &&
                                 get(s.options, :scholtes, true) === true && !s.had_workspace]

pools = Dict{Any,Vector{Any}}()
for s in srcs
    push!(get!(pools, src_sig(s), Any[]), s)
end
rows = []
for p in ports
    pool = get(pools, port_sig(p), nothing)
    (pool === nothing || isempty(pool)) && (push!(rows, (; p, s = nothing)); continue)
    push!(rows, (; p, s = popfirst!(pool)))
end

function verdict(p, s)
    dz = length(p.z) == length(s.z) ? norm(p.z .- s.z, Inf) : Inf
    dk = abs(p.residual - s.residual) / max(abs(s.residual), 1e-10)
    ok = p.converged == s.converged && p.rho == s.rho && dz <= 1e-6 &&
         (dk <= 1e-6 || abs(p.residual - s.residual) <= 1e-10) && abs(p.iters - s.iters) <= 1
    (; ok, dz, dk, di = p.iters - s.iters)
end

io = IOBuffer()
w(a...) = (print(io, a...); print(a...))
w("# Ported Scholtes tests vs ScholtesReducedGOOP.jl, per solve\n\n")
w("| testset | paired solves | match | worst ‖Δz‖∞ | worst Δiters | converged/ρ mismatches | unpaired (port-only backend or config) |\n")
w("|---|---|---|---|---|---|---|\n")
nonunique = "non-unique answers: Scholtes recovers multiple points"
totals = [0, 0, 0]
failures = []
for ts in unique(r.p.testset for r in rows)
    rs = [r for r in rows if r.p.testset == ts]
    paired = [r for r in rs if r.s !== nothing]
    vs = [verdict(r.p, r.s) for r in paired]
    if ts == nonunique && !isempty(paired)
        # a set, not points: compare the distinct answers both suites recovered
        dedupe(zs) = foldl((acc, z) -> any(q -> norm(q .- z, Inf) <= 1e-4, acc) ? acc : push!(acc, z), zs; init = Vector{Float64}[])
        P = dedupe([r.p.z for r in paired if r.p.converged])
        S = dedupe([r.s.z for r in paired if r.s.converged])
        same = length(P) == length(S) && all(p -> any(q -> norm(p .- q, Inf) <= 1e-4, S), P)
        w(@sprintf("| %s | %d | set: %d vs %d distinct points, %s | | | | %d |\n", ts, length(paired),
                   length(P), length(S), same ? "same set" : "DIFFERENT set", length(rs) - length(paired)))
        totals .+= (length(paired), same ? length(paired) : 0, 0)
        continue
    end
    nmatch = count(v -> v.ok, vs)
    mism = count(i -> paired[i].p.converged != paired[i].s.converged || paired[i].p.rho != paired[i].s.rho, eachindex(paired))
    w(@sprintf("| %s | %d | %d | %s | %s | %d | %d |\n", ts, length(paired), nmatch,
               isempty(vs) ? "" : @sprintf("%.1e", maximum(v -> v.dz, vs)),
               isempty(vs) ? "" : string(maximum(v -> abs(v.di), vs)), mism, length(rs) - length(paired)))
    totals .+= (length(paired), nmatch, 0)
    for (r, v) in zip(paired, vs)
        v.ok || push!(failures, (ts, r, v))
    end
end
unmatched_src = sum(length, values(pools); init = 0)
w(@sprintf("\n**Total: %d paired solves, %d match (%.0f%%). %d source solves had no port counterpart.**\n",
           totals[1], totals[2], 100 * totals[2] / max(totals[1], 1), unmatched_src))
if !isempty(failures)
    w("\n## Solves outside tolerance\n\n| testset | n | backend | ρ port/src | converged port/src | iters port/src | ‖Δz‖∞ | ‖K₀‖ port/src |\n|---|---|---|---|---|---|---|---|\n")
    for (ts, r, v) in failures
        w(@sprintf("| %s | %d | %s | %.0e/%.0e | %s/%s | %d/%d | %.2e | %.2e/%.2e |\n", ts, r.p.n, r.p.ls,
                   r.p.rho, r.s.rho, r.p.converged, r.s.converged, r.p.iters, r.s.iters, v.dz,
                   r.p.residual, r.s.residual))
    end
end
OUT === nothing || write(OUT, String(take!(io)))
