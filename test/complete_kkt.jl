module CompleteKKTTests

using Test, ReducedGOOP, Symbolics, BlockArrays, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "benchmark_problems.jl"))
const RG = ReducedGOOP
exact(goop) = RG.generate_slacked_complete_kkt_system(goop; complementarity = :exact)
perturbed(goop) = RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed)
const BUILDERS = (exact, perturbed)

function evaluate(kkt, y, rho; theta = zeros(kkt.parameter_dimension))
    r = zeros(kkt.n_nc)
    kkt.F_nc!(r, y, theta, rho)
    r
end

"The values of the auxiliary slacks' defining expressions (`slack_expressions`) at `y`."
function slack_values(kkt, y; theta = zeros(kkt.parameter_dimension))
    backend = RG.SymbolicTracingUtils.SymbolicsBackend()
    θ = RG.SymbolicTracingUtils.make_variables(backend, :θ, kkt.parameter_dimension)
    ρ = only(RG.SymbolicTracingUtils.make_variables(backend, :ρ, 1))
    G! = RG._compile_kkt_block(kkt.slack_expressions, kkt.vars, θ, ρ; codegen = :fast_differentiation,
        fd_codegen_chunk_size = 32, backend_options = (;))[1]
    G = zeros(length(kkt.slack_expressions))
    G!(G, y, theta, 0.0)
    G
end

function jacobian(kkt, y, rho)
    J = copy(kkt.Jnc_buf)
    kkt.J_nc!(J, y, zeros(kkt.parameter_dimension), rho)
    J
end

function problem(preferences, flags; inequality = nothing)
    RG.ParametricGOOP(BlockArray([0.0], [1]), BlockArray(Float64[], [0]);
        preferences = [preferences], is_prioritized_constraint = [flags],
        equality_constraints = [nothing], inequality_constraints = [inequality])
end

@testset "Complete KKT requires at least two levels per player" for builder in BUILDERS
    for preferences in (Function[], Function[(x, θ) -> x[1]^2])
        @test_throws ArgumentError builder(
            problem(preferences, fill(false, length(preferences))),
        )
    end
    mixed = RG.ParametricGOOP(BlockArray(zeros(2), [1, 1]), BlockArray(Float64[], [0]);
        preferences = [Function[(x, θ) -> x[1]^2, (x, θ) -> x[1]^2],
                       Function[(x, θ) -> x[2]^2]],
        is_prioritized_constraint = [[false, false], [false]],
        equality_constraints = [nothing, nothing],
        inequality_constraints = [nothing, nothing])
    err = try
        builder(mixed)
    catch error
        error
    end
    @test err isa ArgumentError
    @test occursin("player 2 has K = 1", sprint(showerror, err))
end

@testset "Exact complete system: manuscript equations (2.12)-(2.17)" begin
    p = RG.ParametricGOOP(BlockArray(zeros(2), [2]), BlockArray(Float64[], [0]);
        preferences = [Function[(x,t) -> (x[1]+2x[2])^2/2,
            (x,t) -> cosh(x[1]-1)+x[2]^2/2]],
        is_prioritized_constraint = [[false,false]],
        equality_constraints = [(x,t) -> [x[2]-x[1]]],
        inequality_constraints = [(x,t) -> [x[1]+1, 2-x[1]+x[2]]])
    k = exact(p)
    inner, outer = k.levels[1][2], k.levels[1][1]
    x = k.vars[k.primal_dims]
    f, g = [x[2]-x[1]], [x[1]+1, 2-x[1]+x[2]]
    L2 = cosh(x[1]-1)+x[2]^2/2-inner.lambda'*f-inner.gamma'*g
    F2 = [Symbolics.gradient(L2,x); f; g .* inner.gamma]
    G2 = [g; inner.gamma]
    L1 = (x[1]+2x[2])^2/2-outer.lambda'*F2-outer.gamma'*G2
    F1 = [Symbolics.gradient(L1,[x;inner.lambda;inner.gamma]);
        G2 .* outer.gamma; F2]
    G1 = [g;outer.gamma;inner.gamma]
    reference = [F1;g-k.vars[k.interior_point_slack_dims]]
    @test length(reference) == k.n_nc == k.n
    @test all(iszero, Symbolics.simplify.(reference-k.F_symbolic))
    @test iszero(Symbolics.simplify(L1-outer.L))
    @test all(iszero, Symbolics.simplify.(G1-k.G_symbolic))
    @test length(inner.complementary_slackness) == 2
    @test length(outer.complementary_slackness) == 4
    y = 0.5 .+ 0.1sin.(1:k.n)
    @test evaluate(k,y,1.0) == evaluate(k,y,1e-12)
    d = cos.(1:k.n); d ./= norm(d)
    @test jacobian(k,y,1.0)*d ≈ (evaluate(k,y+1e-6d,1.0)-evaluate(k,y-1e-6d,1.0))/2e-6 atol=1e-7

    p3 = RG.ParametricGOOP(BlockArray(zeros(2),[2]),BlockArray(Float64[],[0]);
        preferences=[Function[(x,t)->x[1]^2/2;p.preferences[1]]],
        is_prioritized_constraint=[[false,false,false]],
        equality_constraints=p.equality_constraints,inequality_constraints=p.inequality_constraints)
    k3 = exact(p3)
    top,middle,bottom = k3.levels[1]
    x = k3.vars[k3.primal_dims]
    f,g = [x[2]-x[1]],[x[1]+1,2-x[1]+x[2]]
    L3 = cosh(x[1]-1)+x[2]^2/2-bottom.lambda'*f-bottom.gamma'*g
    F3 = [Symbolics.gradient(L3,x);f;g .* bottom.gamma]
    G3 = [g;bottom.gamma]
    L2 = (x[1]+2x[2])^2/2-middle.lambda'*F3-middle.gamma'*G3
    z3 = [x;bottom.lambda;bottom.gamma]
    F2 = [Symbolics.gradient(L2,z3);G3 .* middle.gamma;F3]
    G2 = [g;middle.gamma;bottom.gamma]
    L1 = x[1]^2/2-top.lambda'*F2-top.gamma'*G2
    z2 = [z3;middle.lambda;middle.gamma]
    reference = [Symbolics.gradient(L1,z2);G2 .* top.gamma;F2;
        g-k3.vars[k3.interior_point_slack_dims]]
    @test k3.n == k3.n_nc == length(reference)
    @test all(iszero,Symbolics.simplify.(reference-k3.F_symbolic))
    y = 0.5 .+ 0.1sin.(1:k3.n)
    d = cos.(1:k3.n);d ./= norm(d)
    @test jacobian(k3,y,1.0)*d ≈ (evaluate(k3,y+1e-6d,1.0)-evaluate(k3,y-1e-6d,1.0))/2e-6 atol=1e-7
end

@testset "Perturbed complete system: exact rows, level slacks, products equal to rho" begin
    # The perturbed system keeps the exact recursion. At the top level each level k's
    # products with g use its own slacks σ_k, with g - σ_k = 0, and every product equals rho.
    for (levels, players, kind) in ((2, 1, :quadratic), (3, 3, :nonlinear))
        case = build_benchmark_problem(; num_players = players, levels, kind)
        e = exact(case.problem)
        k = perturbed(case.problem)
        K, m = length.(case.problem.preferences), case.problem.inequality_dims
        extra = sum((K .- 1) .* m)
        @test k.n == k.n_nc == e.n + extra
        @test length(k.nonnegative_dims) == length(e.nonnegative_dims) + extra
        # An exact point with its slacks on g; the perturbed point shares every other variable by name
        # (exact's σ_{i,1} is level 1's slacks here) and puts every level's slacks on g.
        index = Dict(string(v) => j for (j, v) in enumerate(k.vars))
        ye = 0.5 .+ 0.1sin.(1:e.n)
        ye[e.interior_point_slack_dims] .= slack_values(e, ye)
        y = zeros(k.n)
        for (j, v) in enumerate(e.vars)
            y[index[string(v)]] = ye[j]
        end
        y[k.interior_point_slack_dims] .= slack_values(k, y)
        re, r0, r = evaluate(e, ye, 0.0), evaluate(k, y, 0.0), evaluate(k, y, 1e-3)
        # Every player's exact rows, then the slack rows of all players (exact: m_i, perturbed: K_i m_i each).
        n = sum(length(e.levels[i][1].F) for i in 1:case.problem.num_players)
        @test r0[1:n] ≈ re[1:n] atol=1e-12
        @test norm(r0[(n+1):end]) <= 1e-12
        @test (n + sum(m), n + sum(K .* m)) == (e.n_nc, k.n_nc)
        # rho shifts exactly the complementarity products, (2^K - 1) m_I per player here.
        @test count(!iszero, r - r0) ==
              sum(length(level.complementary_slackness) for player in e.levels for level in player)
        @test all(≈(-1e-3), filter(!iszero, r - r0))
        d = cos.(1:k.n); d ./= norm(d)
        @test jacobian(k,y,1e-3)*d ≈ (evaluate(k,y+1e-6d,1e-3)-evaluate(k,y-1e-6d,1e-3))/2e-6 atol=1e-7
        # Down a ρ chain from z = 0, one solve per stage, as experiments/MonteCarlo_RandomGOOP.jl does.
        θ, w, scale = zeros(k.parameter_dimension), nothing, sqrt(k.n_nc)
        for ρ in (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
            options = RG.ScholtesOptions(; rho_schedule = [ρ], max_inner = 1000, tol = 0.0,
                tol_inner = min(ρ, 1e-2) * scale, projected_step = true, proj_eps = 1e-8, linear_solver = :normal,
                eta_schedule = true, eta_init = 1e-2, eta_min = 1e-8, eta_max = 1e4)
            output = isnothing(w) ? RG.solve(RG.Scholtes(), k, θ; z₀ = zeros(length(case.expected)), options) :
                     RG.solve(RG.Scholtes(), k, θ; w₀ = w, options)
            w = output.w
        end
        @test norm(w[k.primal_dims] - case.expected) < 1e-5
        @test minimum(w[k.nonnegative_dims]) >= 0
    end
end

@testset "Complete empty inequalities, preference bounds, and K >= 2" begin
    for builder in BUILDERS
        p = RG.ParametricGOOP(BlockArray([0.0],[1]),BlockArray(Float64[],[0]);
            preferences=[Function[(x,t)->x[1]^2/2]], is_prioritized_constraint=[[false]],
            equality_constraints=[nothing],inequality_constraints=[nothing])
        @test_throws ArgumentError builder(p)
        for flags in ([false,false,false], [false,false,true], [true,false,true])
            p = RG.ParametricGOOP(BlockArray([0.0],[1]),BlockArray(Float64[],[0]);
                preferences=[Function[flag ? ((x,t)->[1-x[1]]) : ((x,t)->x[1]^2/2) for flag in flags]],
                is_prioritized_constraint=[flags],equality_constraints=[nothing],inequality_constraints=[nothing])
            k = builder(p)
            y = zeros(k.n)
            for level in k.levels[1]
                for v in level.preference_mu
                    y[findfirst(isequal(v),k.vars)] = 1.0
                end
            end
            y[k.interior_point_slack_dims] .= slack_values(k,y)
            @test norm(evaluate(k,y,0.0)) < 1e-12
            @test all(isfinite, jacobian(k,0.5 .+ y,1e-3))
        end
    end
end

@testset "Scholtes projected engine solves the exact complete system" begin
    # The perturbed system is solved down a ρ chain in its own testset above.
    options = RG.ScholtesOptions(;rho_schedule=[1e-9],max_inner=500,tol=1e-7,tol_inner=1e-10,
        eta_init=1e-4,eta_schedule=true,eta_max=1e4,proj_eps=1e-8)
    for levels in (2,3), players in (1,3), kind in (:quadratic,:nonlinear)
        case = build_benchmark_problem(;num_players=players,levels,kind)
        k = exact(case.problem)
        @test k.n_comp == 0
        @test k.n_nc == k.n
        output = RG.solve(RG.Scholtes(),k,zeros(k.parameter_dimension);z₀=case.expected,options)
        @test output.status === :solved
        @test norm(evaluate(k,output.w,1e-9)) < 1e-7
        @test minimum(output.w[k.nonnegative_dims]) >= 0
        @test output.z ≈ case.expected atol=1e-6 rtol=0
        @test RG.is_feasible(case.problem,output.z,zeros(k.parameter_dimension);tol=1e-7).feasible
        @test output.iters > 1
    end
    case = build_benchmark_problem(;num_players=1,levels=2,kind=:quadratic)
    k = exact(case.problem)
    # A start outside the sign box is rejected.
    output = RG.solve(RG.Scholtes(),k,zeros(k.parameter_dimension);z₀=case.expected,options)
    bad = copy(output.w);bad[first(k.nonnegative_dims)] = -1
    @test_throws ArgumentError RG.solve(RG.Scholtes(),k,zeros(k.parameter_dimension);w₀=bad,options)
end


@testset "Complete/reduced unconstrained solution agreement" for levels in (2, 3)
	case = build_unconstrained_quadratic_problem(; levels)

	kkt = exact(case.problem)
	complete = RG.solve(RG.Scholtes(), kkt, zeros(kkt.parameter_dimension);
		z₀ = case.z₀,
		options = RG.ScholtesOptions(; rho_schedule = [1e-9], tol = 1e-9,
			tol_inner = 1e-11, max_inner = 500, eta_init = 1e-6))
	reduced = solve_with_interior_point(
		case.problem,
		case.expected;
		z₀ = case.z₀,
		kkt_system = reduced_kkt_system,
	)

	@test complete.status === :solved
	@test reduced.output.status === :solved
	@test complete.residual <= 1e-8
	@test reduced.output.kkt_error <= 1e-8
	@test isapprox(complete.z, case.expected; atol = IP_ATOL, rtol = 0.0)
	@test isapprox(reduced.primals, case.expected; atol = IP_ATOL, rtol = 0.0)
	@test isapprox(reduced.primals, complete.z; atol = IP_ATOL, rtol = 0.0)
end

end
