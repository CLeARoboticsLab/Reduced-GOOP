# The Scholtes-mode test suite, translated from ScholtesReducedGOOP.jl's test/runtests.jl
# (Jingqi Li). The problems are the source's, written against a flat primal `z` and
# adapted to `ParametricGOOP` by `sproblem` (one block per player, no parameters).
#
# Not translated (φ = true only, _port_logs/DECISIONS.md D19): the effective-Γ
# (`phi = false`) testset, the `:freeze` / `:ideal` bound rules, the `scholtes = false`
# closure, the `:ldl` backend, `SolverWorkspace`, and the fixed-primal check that needs
# the source's analysis environment. Backend names: the source's `:klu` is `:klu_eta`
# here (same saddle matrix), `:pinv` is `:svd`; `:klu_sqrt_eta` is added where the
# source loops over backends.

using LinearAlgebra
using Random
using SparseArrays
using BlockArrays: BlockArray

const RG = ReducedGOOP
const θ0 = Float64[]
# The source's default solver options (its default backend is `:klu`).
const SRC_DEFAULTS = RG.ScholtesOptions(linear_solver = :klu_eta)

# --- the source's problem model, on ParametricGOOP --------------------------------------
function sproblem(; ni, objectives, equality = nothing, inequality = nothing, z0 = nothing)
    N = length(ni)
    z0v = isnothing(z0) ? zeros(sum(ni)) : Float64.(collect(z0))
    lift(f) = (x, θ) -> f(collect(x))
    eqs = isnothing(equality) ? fill(nothing, N) :
          [isempty(equality[i](z0v)) ? nothing : lift(equality[i]) for i in 1:N]
    ineqs = isnothing(inequality) ? fill(nothing, N) :
            [isempty(inequality[i](z0v)) ? nothing : lift(inequality[i]) for i in 1:N]
    goop = RG.ParametricGOOP(
        BlockArray(copy(z0v), ni),
        BlockArray(zeros(0), [0]);
        preferences = [Function[lift(f) for f in objectives[i]] for i in 1:N],
        is_prioritized_constraint = [fill(false, length(objectives[i])) for i in 1:N],
        equality_constraints = eqs,
        inequality_constraints = ineqs,
    )
    (; goop, z0 = z0v, ni, N, np = sum(ni), Ki = length.(objectives),
       mEi = goop.equality_dims, mIi = goop.inequality_dims)
end

skkt(p; quasi = false, quasi_order = 2) = RG.generate_slacked_reduced_kkt_system(
    p.goop; complementarity = :scholtes, drop_higher_order_terms = quasi, quasi_order)

const _LS = Dict(:klu => :klu_eta, :pinv => :svd)
"""
`solve_goop(c, n, z0; kwargs...)` in the source's spelling, on the port. The source's
default backend is `:klu`, so an unspecified `linear_solver` is `:klu_eta` here (the
port's own default, `:normal`, is the source's robotic-arm setting, not its test default).
"""
function ssolve(k, z0; w0 = nothing, trace = nothing, step_trace = nothing, kw...)
    kw = Dict{Symbol,Any}(kw)
    haskey(kw, :linear_solver) || (kw[:linear_solver] = :klu_eta)
    haskey(kw, :linear_solver) && (kw[:linear_solver] = get(_LS, kw[:linear_solver], kw[:linear_solver]))
    haskey(kw, :rho_schedule) && (kw[:rho_schedule] = collect(float.(kw[:rho_schedule])))
    RG.solve(RG.Scholtes(), k, θ0; z₀ = z0, w₀ = w0, trace, step_trace,
             options = RG.ScholtesOptions(; kw...))
end
ssolve(p::NamedTuple, z0 = p.z0; kw...) = ssolve(skkt(p), z0; kw...)
sfeasible(p, z) = RG.is_feasible(p.goop, z, θ0)
F_nc(k, y) = (RG._set_theta!(k, θ0); RG._F_nc(k, y))
a_of(k, y) = (RG._set_theta!(k, θ0); RG._a(k, y))
b_of(k, y) = (RG._set_theta!(k, θ0); RG._b(k, y))
step_ctx(k, ls; kw...) = RG._build_scholtes_step_ctx(RG._scholtes_step_solver(get(_LS, ls, ls)), k; kw...)

# --- the source's test problems ------------------------------------------------------------
prob_A = sproblem(ni = [1], objectives = [[z -> z[1]^2]], inequality = [z -> [z[1] + 2.0]], z0 = [1.0])
prob_B = sproblem(ni = [1], objectives = [[z -> z[1]]], inequality = [z -> [z[1]]], z0 = [1.0])
prob_C = sproblem(ni = [1, 1],
    objectives = [[z -> z[1]^2, z -> (z[1] - (1.0 + 0.25 * z[2]))^2],
                  [z -> z[2]^2, z -> (z[2] - (-1.0 + 0.25 * z[1]))^2]],
    inequality = [z -> [z[1] + 2.0], z -> [z[2] + 2.0]], z0 = [0.0, 0.0])
box2(z) = [z[1] + 2.0, 2.0 - z[1], z[2] + 2.0, 2.0 - z[2]]
prob_D = sproblem(ni = [2], objectives = [[z -> (z[1] + z[2])^2, z -> (z[1] + z[2] - 1.0)^2]],
                  inequality = [box2], z0 = [0.0, 0.0])
seg_t(z) = (z[1] - z[2] + 1.0) / 2
seg_dist(z) = (t = clamp(seg_t(z), -1.0, 2.0); norm(z .- [t, 1 - t]))
prob_E = sproblem(ni = [2], objectives = [[z -> z[1]^2 + z[2]^2, z -> (z[1] - 0.5)^2,
                                           z -> (z[1] + z[2] - 1.0)^2]],
                  inequality = [box2], z0 = [0.0, 0.0])
prob_F = sproblem(ni = [2, 2],
    objectives = [[z -> z[1]^2 + 0.5 * z[4]^2, z -> (z[1] - 0.3 * z[3])^2 + z[2]^2,
                   z -> (z[2] + z[4] - 0.5)^2, z -> (z[1] + z[2] - 1.0)^2 + 0.2 * z[1] * z[3]],
                  [z -> z[3]^2 + z[4]^2, z -> (z[3] + 0.25 * z[1])^2,
                   z -> (z[4] - 0.4)^2 + z[3] * z[2], z -> (z[3] - z[4] + 0.2)^2]],
    equality = [z -> [z[1] + z[2] + 0.5 * z[3] - 0.7], z -> Float64[]],
    inequality = [z -> [z[1] + 3.0, 3.0 - z[1], z[2] + 3.0], z -> [z[3] + 3.0, 3.0 - z[4]]],
    z0 = [0.0, 0.0, 0.0, 0.0])
prob_G = sproblem(ni = [2], objectives = [[z -> z[2]^2, z -> z[1]^2,
                                           z -> (z[1] - 0.5)^2 + (z[2] + 0.25)^2 + 0.1 * z[1]^3]],
                  inequality = [box2], z0 = [0.0, 0.0])
prob_G_z = [(-2 + sqrt(4 + 1.2)) / 0.6, -0.25]
prob_single = sproblem(ni = [1, 1],
    objectives = [[z -> (z[1] - (1.0 + 0.25 * z[2]))^2], [z -> (z[2] - (-1.0 + 0.25 * z[1]))^2]],
    inequality = [z -> [z[1] + 2.0], z -> [z[2] + 2.0]], z0 = [0.0, 0.0])

@testset "Scholtes (ported from ScholtesReducedGOOP.jl)" begin

@testset "problem model" begin
    @test prob_C.N == 2 && prob_C.Ki == [2, 2]
    @test prob_C.np == 2
    @test prob_C.mIi == [1, 1] && prob_C.mEi == [0, 0]
    @test sfeasible(prob_A, [0.0]).feasible
    @test !sfeasible(prob_A, [-3.0]).feasible
    @test sfeasible(prob_A, [-3.0]).worst ≈ -1.0
end

@testset "reduced KKT shape" begin
    k = skkt(prob_C)
    @test k.n_nc + k.n_comp == 2 * (2 * 1 + 0 + 2 * 1 + 1)   # rows, incl. one exact φ pair per player
    @test k.n == 2 + 2 * ((2 - 1) * 1 + 0 + 3)              # z, then duals per player
    @test k.n_comp == 6
    @test k.n == k.n_nc + k.n_comp                          # K = 2, no equalities: square
end

@testset "quasi-stationarity: shape is unchanged" begin
    for prob in (prob_C, prob_E, prob_F, prob_G)
        ke, kq = skkt(prob; quasi = false), skkt(prob; quasi = true)
        @test string.(kq.vars) == string.(ke.vars)
        @test kq.n_nc == ke.n_nc && kq.n_comp == ke.n_comp
        @test length(kq.S_nc[1]) <= length(ke.S_nc[1])
        @test issubset(Set(zip(kq.S_nc...)), Set(zip(ke.S_nc...)))
    end
    @test_throws ArgumentError skkt(prob_E; quasi = true, quasi_order = 0)
end

@testset "quasi == reduced on quadratic objectives over affine constraints" begin
    Random.seed!(5)
    for prob in (prob_A, prob_B, prob_C, prob_D, prob_E, prob_F)
        ke, kq = skkt(prob; quasi = false), skkt(prob; quasi = true)
        RG._set_rho!(ke, 0.3); RG._set_rho!(kq, 0.3)
        worst = 0.0
        for _ in 1:25
            y = randn(ke.n)
            worst = max(worst, maximum(abs, F_nc(kq, y) .- F_nc(ke, y)),
                        maximum(abs, a_of(kq, y) .- a_of(ke, y); init = 0.0),
                        maximum(abs, b_of(kq, y) .- b_of(ke, y); init = 0.0))
        end
        @test worst == 0.0
        @test kq.S_nc == ke.S_nc
    end
    for prob in (prob_E, prob_F)
        re = ssolve(skkt(prob; quasi = false), prob.z0)
        rq = ssolve(skkt(prob; quasi = true), prob.z0)
        @test rq.z == re.z
        @test rq.iters == re.iters
    end
end

@testset "quasi truncates for real on non-quadratic data" begin
    ke, kq = skkt(prob_G; quasi = false), skkt(prob_G; quasi = true)
    RG._set_rho!(ke, 0.3); RG._set_rho!(kq, 0.3)
    Random.seed!(6)
    y = randn(ke.n)
    @test !isapprox(F_nc(kq, y), F_nc(ke, y); atol = 1e-8)
    @test length(kq.S_nc[1]) < length(ke.S_nc[1])
    re, rq = ssolve(ke, prob_G.z0), ssolve(kq, prob_G.z0)
    @test re.converged && rq.converged
    @test rq.sign_min > 0
    @test rq.z ≈ prob_G_z atol = 1e-5
    @test rq.z ≈ re.z atol = 1e-6
end

@testset "quasi_order = 1 keeps first-order information only" begin
    k1 = skkt(prob_E; quasi = true, quasi_order = 1)
    psi_cols = findall(v -> startswith(string(v), "psi"), k1.vars)
    @test !isempty(psi_cols)
    @test isempty(intersect(Set(k1.S_nc[2]), Set(psi_cols)))
    Random.seed!(8)
    ke, kq = skkt(prob_G; quasi = false), skkt(prob_G; quasi = true, quasi_order = 10)
    RG._set_rho!(ke, 0.3); RG._set_rho!(kq, 0.3)
    y = randn(ke.n)
    @test F_nc(kq, y) == F_nc(ke, y)
end

@testset "quasi composes with explicit Scholtes complementarity" begin
    r = ssolve(skkt(prob_E; quasi = true), prob_E.z0)
    @test r.converged && r.sign_min > 0
    @test r.z[1] ≈ 0.5 atol = 1e-5
    @test r.z[1] + r.z[2] ≈ 1.0 atol = 1e-5
end

@testset "compiled functions agree in and out of place" begin
    k = skkt(prob_C)
    RG._set_theta!(k, θ0); RG._set_rho!(k, 0.3)
    y = randn(k.n)
    v = zeros(k.n_nc); RG._F_nc!(k, v, y); @test v ≈ F_nc(k, y)
    va = zeros(k.n_comp); RG._a!(k, va, y); @test va ≈ a_of(k, y)
    vb = zeros(k.n_comp); RG._b!(k, vb, y); @test vb ≈ b_of(k, y)
    k.J_b!(k.Jb_buf, y, k.θ, k.ρ[])
    @test all(==(1.0), nonzeros(k.Jb_buf))
    @test k.Jb_buf * y ≈ b_of(k, y)
end

@testset "the Jacobian memo answers for the y it was filled at, and no other" begin
    k = skkt(prob_D); RG._set_theta!(k, θ0); RG._set_rho!(k, 0.3)
    y1, y2 = randn(k.n), randn(k.n)
    fresh = skkt(prob_D); RG._set_theta!(fresh, θ0); RG._set_rho!(fresh, 0.3)
    ref(y) = (fresh.Jvalid[] = false; RG._fill_jacobians!(fresh, y); copy(nonzeros(fresh.Jnc_buf)))
    RG._fill_jacobians!(k, y1); @test nonzeros(k.Jnc_buf) == ref(y1)
    RG._fill_jacobians!(k, y1); @test nonzeros(k.Jnc_buf) == ref(y1)
    RG._fill_jacobians!(k, y2); @test nonzeros(k.Jnc_buf) == ref(y2)
    RG._fill_jacobians!(k, y1); @test nonzeros(k.Jnc_buf) == ref(y1)
    w = copy(y1); v = view(w, 1:k.n)
    RG._fill_jacobians!(k, v)
    w .= y2
    RG._fill_jacobians!(k, v); @test nonzeros(k.Jnc_buf) == ref(y2)
    RG._fill_jacobians!(k, y1)
    RG._gamma_columns(k)
    @test !k.Jvalid[]
end

@testset "inactive constraint, all backends" begin
    k = skkt(prob_A)
    for ls in (:klu_eta, :klu_sqrt_eta, :normal, :svd)
        r = ssolve(k, [1.0]; linear_solver = ls)
        @test r.converged
        @test r.z[1] ≈ 0.0 atol = 1e-6
        @test r.sign_min > 0
        @test r.klu_fallbacks == 0 || ls in (:klu_eta, :klu_sqrt_eta)
    end
end

@testset "active constraint and coupled game" begin
    r = ssolve(skkt(prob_B), prob_B.z0)
    @test r.converged && abs(r.z[1]) < 1e-5
    r = ssolve(prob_C)
    @test r.converged
    @test r.z ≈ [0.8, -0.8] atol = 1e-5
end

@testset "the certificate separates a solve from a stall" begin
    k = skkt(prob_C)
    r = ssolve(k, prob_C.z0; tol = 1e-8)
    @test RG.stat_feas(k, θ0, r) >= 0
    @test RG.stat_feas(k, θ0, r) <= r.residual + 1e-12
    @test r.converged && RG.stat_feas(k, θ0, r) < 1e-6
    k1 = skkt(prob_A)
    r1 = ssolve(k1, [1.0])
    @test RG.stat_feas(k1, θ0, r1) >= 0
end

@testset "solve_certified returns a certificate, and it is honest" begin
    for prob in (prob_A, prob_C, prob_D)
        k = skkt(prob)
        cr = RG.solve_certified(k, θ0, prob.z0; options = SRC_DEFAULTS)
        @test cr.attempts >= 1
        @test cr.stat ≈ RG.stat_feas(k, θ0, cr.result) atol = 1e-12
        @test cr.shortfall == cr.result.shortfall
        if cr.certified
            @test cr.rung >= 1
            @test cr.stat < 1e-6
            @test cr.margin >= -1e-8
        else
            @test cr.rung == 0
        end
    end
    k = skkt(prob_C)
    hopeless = RG.solve_certified(k, θ0, prob_C.z0; ladder = [[1e-2]], tol_stat = 1e-30,
                                   options = SRC_DEFAULTS)
    @test !hopeless.certified && hopeless.rung == 0
    @test hopeless.attempts == 1
    @test_throws ArgumentError RG.solve_certified(k, θ0, prob_C.z0; ladder = Vector{Float64}[])
    L = RG.certified_ladder()
    @test all(!isempty, L)
    @test length(L[1]) == 1
    @test issorted(length.(L))
    @test minimum(L[end]) <= minimum(L[1])
end

@testset "warm start reproduces the solve" begin
    k = skkt(prob_C)
    r1 = ssolve(k, prob_C.z0)
    r2 = ssolve(k, prob_C.z0; w0 = r1.w, rho_schedule = [r1.rho])
    @test r2.converged
    @test r2.z ≈ r1.z atol = 1e-8
    @test r2.iters <= r1.iters
    @test_throws ArgumentError ssolve(k, prob_C.z0; w0 = zeros(3))
end

@testset "a start on the boundary is rejected, not silently accepted" begin
    @test_throws ArgumentError ssolve(prob_A, [-2.0])
end

@testset "non-monotone line search" begin
    k = skkt(prob_C)
    r1 = ssolve(k, prob_C.z0; nonmonotone = 1)
    lv = ssolve(k, prob_C.z0; nonmonotone = 1, rho_schedule = [1e-2])
    @test all(lv.history[i + 1] <= lv.history[i] for i in 1:(length(lv.history) - 1))
    for M in (2, 5)
        r = ssolve(k, prob_C.z0; nonmonotone = M)
        @test r.converged
        @test r.sign_min > 0
        @test r.z ≈ r1.z atol = 1e-6
    end
    r0 = ssolve(k, prob_C.z0; nonmonotone = 3)
    @test r0.converged && r0.sign_min > 0
    @test_throws ArgumentError ssolve(k, prob_C.z0; nonmonotone = 0)
end

@testset "the projected bound rule stays in the sign box" begin
    for prob in (prob_A, prob_B, prob_single)
        r = ssolve(skkt(prob), prob.z0)
        @test r.converged
        @test r.sign_min > 0
    end
    @test_throws ArgumentError ssolve(prob_A, prob_A.z0; projected_step = false)
    @test_throws ArgumentError ssolve(prob_A, prob_A.z0; tau = 1.5)
    @test_throws ArgumentError ssolve(prob_A, prob_A.z0; proj_eps = -1.0)
end

@testset "schedules" begin
    @test RG.geometric_schedule(1.0; stop = 1e-2, per_decade = 1) ≈ [1.0, 1e-1, 1e-2]
    @test length(RG.geometric_schedule(1.0; stop = 1e-2, per_decade = 4)) == 9
    @test RG.geometric_schedule(1e-3; stop = 1e-2) == [1e-3]
    @test_throws ArgumentError RG.geometric_schedule(1.0; stop = 0.0)
    @test_throws ArgumentError RG.geometric_schedule(0.0)
    @test_throws ArgumentError ssolve(prob_A, [1.0]; rho_min = 0.0)
    @test_throws ArgumentError ssolve(prob_A, [1.0]; linear_solver = :lu)
end

@testset "non-unique answers: Scholtes recovers multiple points" begin
    Random.seed!(11)
    k = skkt(prob_D)
    starts = RG.grid_starts(prob_D.goop, θ0, -1.5:0.75:1.5, -1.5:0.75:1.5)
    @test length(starts) >= 20
    S = RG.solve_multi(k, θ0, starts; options = SRC_DEFAULTS)
    @test S.rejected == 0
    @test length(S.points) > 1
    @test maximum(seg_dist(r.z) for r in S.results) < 1e-5
    ts = [seg_t(p) for p in S.points]
    @test maximum(ts) - minimum(ts) > 0.5
    @test all(r.sign_min > 0 for r in S.results)
    @test length(S.assignment) == length(S.results)
    @test maximum(S.assignment) == length(S.points)
end

@testset "random starts are strictly feasible" begin
    Random.seed!(3)
    for z in RG.random_starts(prob_D.goop, θ0, 25; radius = 1.5, center = prob_D.z0)
        @test sfeasible(prob_D, z).worst > 0
    end
    @test_throws ArgumentError RG.random_starts(prob_A.goop, θ0, 5; center = [-10.0], radius = 0.1)
end

@testset "the rectangularity surplus, exactly" begin
    surplus(K, n, mE) = ((K - 1) * (K - 2) ÷ 2) * n + (K - 1) * mE
    mk(K, n, mE, mI) = sproblem(ni = [n],
        objectives = [[let k = k; z -> sum(z[i]^2 for i in 1:n) + k * z[1]; end for k in 1:K]],
        equality = mE > 0 ? [let m = mE; z -> [z[j] - 0.1j for j in 1:m]; end] : nothing,
        inequality = mI > 0 ? [let m = mI; z -> [z[j] + 2.0 for j in 1:m]; end] : nothing,
        z0 = zeros(n))
    rows(k) = k.n_nc + k.n_comp
    for K in 2:4, n in (2, 4), mE in (0, 1), mI in (1, 2)
        k = skkt(mk(K, n, mE, mI))
        @test k.n - rows(k) == surplus(K, n, mE)
    end
    for K in 3:4, n in (2,), mE in (0, 1), mI in (0, 1)
        k = skkt(mk(K, n, mE, mI))
        @test k.n > rows(k)
    end
    for n in (2, 4), mI in (1, 2)
        sq = skkt(mk(2, n, 0, mI)); @test sq.n == rows(sq)
        re = skkt(mk(2, n, 1, mI)); @test re.n > rows(re)
    end
end

@testset "refinement brings :normal onto the saddle-point step" begin
    k = skkt(prob_C)
    r = ssolve(k, prob_C.z0; rho_schedule = [1.0], max_inner = 3)
    w = copy(r.w)
    res = RG.ScholtesResiduals(k)
    Rv, bv = zeros(res.m), zeros(k.n_comp)
    rv = RG._resid!(res, Rv, bv, w, 1e-3)
    @test norm(rv) > 1e-3
    step_for(ls, nref) = begin
        ctx, _ = step_ctx(k, ls; refine = nref)
        copy(RG._bounded_step!(RG._ProjectedRule(ctx.nw), ctx, w, bv, rv))
    end
    ref = step_for(:klu_eta, 0)
    @test norm(ref) > 0
    raw = norm(step_for(:normal, 0) - ref) / norm(ref)
    fine = norm(step_for(:normal, 4) - ref) / norm(ref)
    @test fine <= raw * (1 + 1e-9) + 1e-15
    raw > 1e-10 && @test fine < raw / 10
end

@testset "every backend solves with the projected rule -- no crashes" begin
    for prob in (prob_A, prob_single), ls in (:klu_eta, :klu_sqrt_eta, :normal, :svd)
        r = ssolve(skkt(prob), prob.z0; linear_solver = ls)
        @test r.converged
        @test r.sign_min > 0
    end
end

@testset "the Jacobian memo can never serve stale values" begin
    @test RG._same_y([1.0, 2.0], [1.0, 2.0])
    @test !RG._same_y([NaN, 2.0], [NaN, 2.0])
    @test !RG._same_y([1.0], [1.0, 2.0])
    pnl = sproblem(ni = [2], objectives = [[z -> sin(z[1]) * z[2]^3, z -> cos(z[1] * z[2]) + z[1]^4]],
                   inequality = [z -> [z[1] + 2.0, z[2] + 2.0]], z0 = [0.1, 0.1])
    k = skkt(pnl); RG._set_theta!(k, θ0); RG._set_rho!(k, 0.3)
    grab(y) = copy(RG._fill_jacobians!(k, y)[1])
    y1, y2 = fill(0.3, k.n), fill(0.7, k.n)
    j1, j2 = grab(y1), grab(y2)
    @test !all(j1 .== j2)
    @test all(grab(y1) .== j1)
    buf = copy(y1); grab(buf); buf .= y2
    @test all(grab(buf) .== j2)
    grab(y1); RG._gamma_columns(k)
    @test all(grab(y1) .== j1)
    for ls in (:klu_eta, :klu_sqrt_eta, :normal)
        a = ssolve(skkt(prob_C), prob_C.z0; linear_solver = ls)
        km = skkt(prob_C)
        b = ssolve(km, prob_C.z0; linear_solver = ls)
        @test a.z == b.z && a.iters == b.iters
    end
end

@testset "the :normal in-place Gram refresh equals a fresh product" begin
    # (the source tests its `:ldl` CHOLMOD refresh here; `:ldl` is not ported, and
    # `:normal`'s in-place triu(J Jᵀ) refill is the equivalent memcpy contract)
    k = skkt(prob_E); RG._set_theta!(k, θ0); RG._set_rho!(k, 0.3)
    cache = RG._build_augmented_normal(k; scholtes = true)
    Random.seed!(11)
    for _ in 1:4
        y = randn(k.n)
        sv = abs.(randn(k.n_comp)) .+ 0.5
        bv = abs.(randn(k.n_comp)) .+ 0.5
        RG._update_normal!(cache, k, y, sv, bv, RG._Unmasked())
        S = RG._normal_gram!(cache)
        S = RG._normal_gram!(cache)       # the second call is the in-place refill
        U = triu(cache.J * transpose(cache.J))
        if S isa SparseArrays.CHOLMOD.Sparse
            raw = unsafe_load(pointer(S))
            @test raw.stype == 1
            @test unsafe_wrap(Array, Ptr{Float64}(raw.x), raw.nzmax) == nonzeros(U)
        else
            @test Matrix(S) == Matrix(Symmetric(cache.J * transpose(cache.J)))
        end
    end
end

@testset "the new knobs validate their arguments" begin
    for kw in ((; nonmonotone = 0), (; linear_solver = :nope), (; rho_min = 0.0),
               (; refine = -1), (; rho_schedule = Float64[]), (; projected_step = false),
               (; reuse_factorization_iters = 1))
        @test_throws ArgumentError ssolve(prob_A; kw...)
    end
end

@testset "an explicit rho_schedule is honoured exactly" begin
    r = ssolve(skkt(prob_C), prob_C.z0; rho_schedule = [1.0, 1e-3], max_inner = 40)
    @test r.rho >= 1e-3 - 1e-15
    @test r.sign_min > 0
end

@testset "every backend computes the same least-squares step" begin
    Random.seed!(4)
    for prob in (prob_C, prob_E)
        k = skkt(prob); RG._set_theta!(k, θ0); RG._set_rho!(k, 0.5)
        res = RG.ScholtesResiduals(k)
        w = RG._initial_w(k, prob.z0, 0.5; gamma_cols = RG._gamma_columns(k))
        w .+= 0.01 .* randn(length(w))
        w[(k.n + 1):end] .= abs.(w[(k.n + 1):end]) .+ 0.2
        Rv, bv = zeros(res.m), zeros(res.nc)
        RG._resid!(res, Rv, bv, w, 0.5)
        J = copy(RG._augmented_dense!(RG._build_augmented_dense(k; scholtes = true), k,
                 view(w, 1:k.n), view(w, (k.n + 1):(k.n + k.n_comp)), bv, ones(length(w))))
        p = size(J, 2)
        F = svd(J)
        pnv = -F.V * ((F.S .> 1e-10 * F.S[1]) .* (F.U' * Rv) ./ max.(F.S, 1e-300))
        tik3 = -F.V * ((F.S ./ (F.S .^ 2 .+ 1e-6)) .* (F.U' * Rv))
        base = nothing
        for ls in (:klu_eta, :normal, :svd)
            sweeps = ls === :normal ? 2 : 0
            ctx, _ = step_ctx(k, ls; eta_init = 1e-8, refine = sweeps)
            d = ctx.solve(w, bv, Rv)
            @test length(d) == p && all(isfinite, d)
            defect = norm(J * d + Rv) / norm(Rv)
            if ls === :normal
                @test isfinite(defect)
            else
                base === nothing && (base = defect)
                @test abs(defect - base) <= 1e-8 + 1e-5 * base
            end
            ctx2, _ = step_ctx(k, ls; eta_init = 1e-3, refine = sweeps)
            d2 = ctx2.solve(w, bv, Rv)
            ref = ls === :svd ? pnv : tik3
            @test norm(d2 - ref) <= 1e-6 * max(norm(ref), 1e-12)
        end
        # `:klu_sqrt_eta` at η = 1e-6 builds the same matrix as `:klu_eta` at 1e-3.
        ctx3, _ = step_ctx(k, :klu_sqrt_eta; eta_init = 1e-6)
        @test norm(ctx3.solve(w, bv, Rv) - tik3) <= 1e-6 * max(norm(tik3), 1e-12)
    end
end

@testset "what the projected rule promises the line search at alpha = 1" begin
    Random.seed!(5)
    for prob in (prob_A, prob_B, prob_single)
        k = skkt(prob); RG._set_theta!(k, θ0); RG._set_rho!(k, 0.5)
        res = RG.ScholtesResiduals(k)
        w = RG._initial_w(k, prob.z0, 0.5; gamma_cols = RG._gamma_columns(k))
        w .+= 0.01 .* randn(length(w))
        w[(k.n + 1):end] .= abs.(w[(k.n + 1):end]) .+ 0.2
        Rv, bv = zeros(res.m), zeros(res.nc)
        RG._resid!(res, Rv, bv, w, 0.5)
        ctx, _ = step_ctx(k, :klu_eta)
        rule = RG._ProjectedRule(ctx.nw)
        d = RG._bounded_step!(rule, ctx, w, bv, Rv)
        wt = similar(w)
        RG._project!(rule, wt, w, 1.0, d)
        @test isfinite(RG._merit!(res, zeros(res.m), zeros(res.nc), wt, 0.5))
        g = copy(ctx.Jtmul(w, bv, Rv))
        @test dot(g, d) <= 1e-9 * max(1.0, norm(g) * norm(d))
    end
end

@testset "the returned rho is the level the walk actually ended on" begin
    sched = [1.0, 1e-2, 1e-4]
    for prob in (prob_A, prob_C, prob_D)
        r = ssolve(skkt(prob), prob.z0; rho_schedule = sched, tol = 0.0)
        @test r.rho == sched[end]
        @test r.sign_min > 0
    end
    rc = ssolve(skkt(prob_A), prob_A.z0; rho_schedule = sched)
    @test rc.converged && rc.rho in sched
end

@testset "line-search-driven eta schedule" begin
    options = (; gain_low = 0.25, gain_high = 0.75, eta_min = 1e-8, eta_max = 1e2,
               tightening_rate = 1.2, loosening_rate = 3.0)
    up, down = 1 + exp(-3.0), 1 - exp(-1.2)
    next_eta(eta, gain, alpha) = RG._scheduled_eta(eta, gain, alpha; options...)
    @test next_eta(1.0, 0.25, 1.0) ≈ up
    @test next_eta(1.0, 0.75, 1.0) == 1.0
    @test next_eta(1.0, 0.8, 1.0) ≈ down
    @test next_eta(1.0, 1.0, 0.5) ≈ up
    @test next_eta(1.0, 1.0, 0.99) ≈ down
    @test next_eta(1.0, -Inf, 0.0) ≈ up
    @test next_eta(1e2, -1.0, 1.0) == 1e2
    @test next_eta(1e-8, 1.0, 1.0) == 1e-8
    @test RG._gain_reductions([1.0], [0.0], 1.0, 1.0) == (0.0, 0.0, -Inf)

    p = sproblem(ni = [1], objectives = [[z -> z[1]^2]], z0 = [1.0])
    k = skkt(p)
    for backend in (:normal, :klu_eta, :klu_sqrt_eta)
        seen = NamedTuple[]
        rr = ssolve(k, p.z0; linear_solver = backend, eta_schedule = true, eta_init = 0.1,
                    rho_schedule = [1.0, 0.1], max_inner = 1, tol = 0.0, tol_inner = 0.0,
                    step_trace = nt -> push!(seen, (; rho = nt.rho, eta = nt.eta,
                        eta_next = nt.eta_next, gain = nt.gain_ratio, alpha = nt.alpha)))
        @test length(seen) == 2
        @test all(t -> t.alpha == 1.0 && t.gain ≈ 1.0, seen)
        @test seen[1].eta ≈ 0.1
        @test seen[1].eta_next ≈ 0.1down
        @test seen[2].eta ≈ seen[1].eta_next
        @test rr.eta ≈ seen[2].eta_next ≈ 0.1down^2
        restarted = Float64[]
        ssolve(k, p.z0; w0 = rr.w, linear_solver = backend, eta_schedule = true,
               eta_init = rr.eta, rho_schedule = [0.01], max_inner = 1, tol = 0.0,
               tol_inner = 0.0, trace = nt -> push!(restarted, nt.eta))
        @test only(restarted) ≈ rr.eta
    end

    k = skkt(prob_C)
    snapshots = NamedTuple[]
    result = ssolve(k, prob_C.z0; linear_solver = :normal, eta_schedule = true, eta_init = 1e-3,
                    rho_schedule = [1e-2], max_inner = 12, tol = 0.0, tol_inner = 0.0,
                    step_trace = nt -> push!(snapshots, (; nt..., w = copy(nt.w),
                        w_trial = copy(nt.w_trial), r = copy(nt.r), b = copy(nt.b))))
    dc = RG._build_augmented_dense(k; scholtes = true)
    @test !isempty(snapshots)
    for nt in snapshots
        if nt.accepted
            J = RG._augmented_dense!(dc, k, view(nt.w, 1:k.n),
                view(nt.w, (k.n + 1):(k.n + k.n_comp)), nt.b, ones(length(nt.w)))
            jd = J * (nt.w_trial - nt.w)
            predicted = -dot(nt.r, jd) - dot(jd, jd) / 2
            actual = (norm(nt.r)^2 - nt.trial_norm^2) / 2
            @test nt.pred_reduction ≈ predicted atol = 1e-12 rtol = 1e-9
            @test nt.actual_reduction ≈ actual atol = 1e-12 rtol = 1e-9
            @test nt.gain_ratio ≈ (predicted > 0 ? actual / predicted : -Inf)
        else
            @test nt.gain_ratio == -Inf
            @test nt.actual_reduction == nt.pred_reduction == 0.0
        end
        @test nt.eta_next ≈ next_eta(nt.eta, nt.gain_ratio, nt.alpha)
    end
    @test result.eta == snapshots[end].eta_next

    # The normal backend's sticky floor must be bypassable.
    RG._set_theta!(k, θ0)
    ctx, bk = step_ctx(k, :normal; eta_init = 1e-3, eta_sticky = false)
    w = RG._initial_w(k, prob_C.z0, 1e-2; gamma_cols = RG._gamma_columns(k))
    res = RG.ScholtesResiduals(k)
    rv, bv = zeros(res.m), zeros(res.nc)
    RG._resid!(res, rv, bv, w, 1e-2)
    bk.normal.eta[] = 1.0
    ctx.solve(w, bv, rv)
    @test bk.eta_used == 1e-3

    for schedule in (false, true)
        rr = ssolve(p; linear_solver = :svd, eta_schedule = schedule, eta_init = 0.1)
        @test rr.eta == 0.1
    end
    for kw in ((; eta_init = 0.0), (; eta_min = -1.0), (; eta_min = 2.0, eta_max = 1.0),
               (; gain_low = 0.8, gain_high = 0.5), (; tightening_rate = -1.0),
               (; loosening_rate = Inf))
        @test_throws ArgumentError ssolve(p; eta_schedule = true, kw...)
    end
end

@testset "effective Tikhonov parameter in solver traces" begin
    k = skkt(prob_C)
    for backend in (:normal, :klu_eta, :klu_sqrt_eta, :svd)
        before, after = Float64[], Float64[]
        ssolve(k, prob_C.z0; linear_solver = backend, eta_init = 1e-3, rho_schedule = [1e-2],
               max_inner = 3, tol = 0.0, tol_inner = 0.0,
               trace = nt -> push!(before, nt.eta), step_trace = nt -> push!(after, nt.eta))
        @test !isempty(before)
        @test isequal(before, after)
        @test backend === :svd ? all(isnan, before) : all(==(1e-3), before)
    end
    RG._set_theta!(k, θ0)
    ctx, bk = step_ctx(k, :normal; eta_init = 1e-8)
    w = RG._initial_w(k, prob_C.z0, 1e-2; gamma_cols = RG._gamma_columns(k))
    res = RG.ScholtesResiduals(k)
    rv, bv = zeros(res.m), zeros(res.nc)
    RG._resid!(res, rv, bv, w, 1e-2)
    bk.normal.eta[] = 1e-3
    ctx.solve(w, bv, rv)
    @test bk.eta_used == bk.normal.eta_fac[] == 1e-3
    @test ctx.eta[] == 1e-8
    ctx.coldiag(w, bv)
    @test bk.eta_used == 1e-3
end

@testset "the reported residual is the residual OF the returned point" begin
    for prob in (prob_A, prob_C, prob_E), rf in (0, 2), ls in (:klu_eta, :normal)
        k = skkt(prob)
        r = ssolve(k, prob.z0; refine = rf, linear_solver = ls)
        res = RG.ScholtesResiduals(k)
        Rv = zeros(res.m); bv = zeros(res.nc)
        RG._resid!(res, Rv, bv, r.w, r.rho)
        @test r.residual ≈ RG._norm_K0(res, Rv, r.w, r.rho) atol = 0 rtol = 1e-12
        @test r.sign_min ≈ RG._sign_min(res, r.w, bv) atol = 0 rtol = 1e-12
        @test r.z == r.w[1:prob.np]
        k.n_comp == 0 || @test r.shortfall ≈ maximum(abs, r.s .* bv) atol = 0 rtol = 1e-12
    end
end

@testset "inequalities hold exactly at a Scholtes solution" begin
    # Active constraints (prob_B, prob_D's and prob_E's box) end ON their bound, not past
    # it. prob_F is left out: it does not converge under the default options, in the
    # source either (Step 1 reference: ‖K₀‖ ≈ 4e-4).
    for prob in (prob_B, prob_C, prob_D, prob_E)
        r = ssolve(skkt(prob), prob.z0; linear_solver = :normal)
        @test r.converged
        @test sfeasible(prob, r.z).worst >= -1e-8
    end
end

@testset "the merit options are gone" begin
    base = (; tol = 1e-6, η₀ = 0.0, ϵ₀ = 1e-9, max_inner_iters = 10, max_outer_iters = 1,
            tightening_rate = 2.0, loosening_rate = 0.5, min_stepsize = 1e-20, verbose = false)
    @test RG.InteriorPointOptions(; base...) isa RG.InteriorPointOptions
    for kw in ((; use_feasibility_merit = true), (; feasibility_tol = 1e-3), (; μ₀ = 1.0),
               (; μ_max = 1e4), (; mu_growth = 10.0))
        @test_throws MethodError RG.InteriorPointOptions(; base..., kw...)
    end
end

@testset "explicit gamma / nonnegative inequality phi" begin
    p = sproblem(ni = [1], objectives = [[z -> -z[1], z -> z[1]]], inequality = [z -> [z[1]]], z0 = [1.0])
    @test_throws ArgumentError RG.generate_slacked_reduced_kkt_system(p.goop;
        complementarity = :scholtes, phi = false)
    k = skkt(p); RG._set_theta!(k, θ0); rho = 0.1; RG._set_rho!(k, rho)
    @test all(!startswith(string(v), "Gam") for v in k.vars)
    @test k.original_nc == 2
    @test k.n_comp == 3 && k.u_index == [1, 2, 0]
    cols = RG._gamma_columns(k)
    res = RG.ScholtesResiduals(k)
    w = RG._initial_w(k, p.z0, rho; gamma_cols = cols)
    rv = zeros(res.m); bv = zeros(res.nc)
    @test isfinite(RG._merit!(res, rv, bv, w, rho))
    @test all(>(0), w[cols])
    @test a_of(k, w[1:k.n])[3] ≈ 0.9rho
    # exact boundary witness: γ₁ = 0, γ₂ = 1, φ = 1, z = ρ
    y = zeros(k.n); y[1] = rho; y[cols] = [0.0, 1.0, 1.0]
    ww = [y; rho; rho; 0.0; rho; 0.0]
    @test norm(RG._resid!(res, rv, bv, ww, rho)) < 1e-14
    ww[cols[3]] = -1
    @test F_nc(k, ww[1:k.n])[1] ≈ -2
    @test isinf(RG._merit!(res, rv, bv, ww, rho))
    w[cols[1]] = -0.02
    @test !isfinite(RG._merit!(res, rv, bv, w, rho))
    w[cols[1]] = 0.02
    w[cols[3]] = -0.1
    @test isinf(RG._merit!(res, rv, bv, w, rho))
    w[cols[3]] = 0.1
    RG._resid!(res, rv, bv, w, rho)
    dense = RG._build_augmented_dense(k; scholtes = true)
    J = copy(RG._augmented_dense!(dense, k, w[1:k.n], w[(k.n + 1):(k.n + k.n_comp)], bv, ones(length(w))))
    v = sin.(1.0:length(w)); h = 1e-6
    rp = copy(RG._resid!(res, rv, bv, w + h * v, rho)); rm = copy(RG._resid!(res, rv, bv, w - h * v, rho))
    @test norm((rp - rm) / (2h) - J * v) < 1e-8
    RG._resid!(res, rv, bv, w, rho)
    k0 = RG._norm_K0(res, rv, w, rho)
    wzero = copy(w); wzero[(k.n + k.n_comp + 1):end] .= 0
    @test k0 ≈ norm(RG._resid!(res, rv, bv, wzero, 0.0))
    RG._set_rho!(k, rho)
    RG._resid!(res, rv, bv, w, rho)
    for backend in (:normal, :klu_eta, :klu_sqrt_eta, :svd)
        ctx, _ = step_ctx(k, backend)
        rule = RG._ProjectedRule(ctx.nw)
        d = RG._bounded_step!(rule, ctx, w, bv, rv)
        @test all(isfinite, d)
        @test cols[3] in rule.idx
        @test cols[1] in rule.idx
        trial = similar(w); RG._project!(rule, trial, w, 1.0, d)
        @test trial[cols[3]] > 0
    end
    p3 = sproblem(ni = [1], objectives = [[z -> z[1], z -> z[1], z -> z[1]]], inequality = [z -> [z[1]]], z0 = [1.0])
    k3 = skkt(p3)
    @test k3.original_nc == 3
    @test k3.n_comp == 6
    @test all(l.kind === :phi_upper for l in k3.labels[4:end])
    @test all(iszero, k3.u_index[4:end])
    k = skkt(p)
    r = ssolve(k, p.z0; rho_schedule = [0.1, 0.01], linear_solver = :normal, max_inner = 1000,
               tol = 0.0, tol_inner = 1e-9)
    res = RG.ScholtesResiduals(k)
    rv = RG._resid!(res, zeros(res.m), zeros(res.nc), r.w, r.rho)
    @test norm(rv) < 1e-8
    @test r.z[1] ≈ 0.01 atol = 1e-8
    @test maximum(abs, r.s[3:end] .* b_of(k, r.w[1:k.n])[3:end]) < 1e-8
    @test r.sign_min > 0
    single = sproblem(ni = [1], objectives = [[z -> z[1]]], inequality = [z -> [z[1]]], z0 = [1.0])
    r = ssolve(single)
    @test r.converged && abs(r.z[1]) < 1e-5
    emptyprob = sproblem(ni = [1], objectives = [[z -> z[1]^2]], z0 = [1.0])
    r = ssolve(emptyprob)
    @test r.converged && abs(r.z[1]) < 1e-7
    # a parameter change invalidates the memo even when y is unchanged
    k = skkt(p); RG._set_theta!(k, θ0); cols = RG._gamma_columns(k)
    w = RG._initial_w(k, p.z0, 0.1; gamma_cols = cols)
    RG._fill_jacobians!(k, w[1:k.n]); @test k.Jvalid[]
    a = copy(a_of(k, w[1:k.n])); RG._set_rho!(k, 0.2)
    @test !k.Jvalid[]
    @test a_of(k, w[1:k.n])[3] - a[3] ≈ 0.1
end

end
