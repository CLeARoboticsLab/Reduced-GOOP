module CompleteKKTTests

using Test, ReducedGOOP, Symbolics, BlockArrays, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "benchmark_problems.jl"))
const RG = ReducedGOOP
const BUILDERS = (RG.generated_exact_complete_kkt_system, RG.generate_scholtes_complete_kkt_system)

function evaluate(kkt, y, rho; theta = zeros(kkt.parameter_dimension))
    r = zeros(kkt.n_nc)
    kkt.F_nc!(r, y, theta, rho)
    r
end

"The values of the auxiliary slacks' defining expressions (`slack_expressions`) at `y`."
function slack_values(kkt, y; theta = zeros(kkt.parameter_dimension))
    θ = RG.SymbolicTracingUtils.make_variables(RG.SymbolicTracingUtils.SymbolicsBackend(), :θ,
        kkt.parameter_dimension)
    G! = RG._compile_scholtes_block(kkt.slack_expressions, kkt.vars, θ, Symbolics.variable(:complete_rho),
        Symbolics.variable(:complete_unused_eta); codegen = :fast_differentiation, fd_codegen_chunk_size = 32,
        backend_options = (;))[1]
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

function residual(kkt, z, ε)
    r = zeros(kkt.kkt_dimension)
    kkt.F!(r, z; θ = Float64[], ϵ = ε, η = 0.0)
    r
end

@testset "Complete KKT requires at least two levels per player" begin
    for preferences in (Function[], Function[(x, θ) -> x[1]^2])
        @test_throws ArgumentError RG.generate_slacked_complete_kkt_system(
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
        RG.generate_slacked_complete_kkt_system(mixed)
    catch error
        error
    end
    @test err isa ArgumentError
    @test occursin("player 2 has K = 1", sprint(showerror, err))
end

@testset "Complete KKT omits empty complementarity budgets" begin
    for levels in 2:4
        preferences = Function[(x, θ) -> x[1]^2 / 2 for _ in 1:levels]
        flags = fill(false, levels)
        absent = RG.generate_slacked_complete_kkt_system(problem(preferences, flags))
        empty = RG.generate_slacked_complete_kkt_system(
            problem(preferences, flags; inequality = (x, θ) -> Float64[]),
        )
        @test isequal(absent.F_symbolic, empty.F_symbolic)
        @test isequal(absent.z_symbolic, empty.z_symbolic)
        @test absent.kkt_dimension == absent.variable_dimension == 2^(levels - 1)
        for kkt in (absent, empty)
            @test isempty(kkt.interior_point_slack_dims)
            @test isempty(kkt.inequality_constraint_dual_dims)
            z = fill(0.7, kkt.variable_dimension)
            @test residual(kkt, z, 0.1) == residual(kkt, z, 1e-7)
            @test iszero(norm(residual(kkt, zeros(length(z)), 0.1), Inf))
            if levels == 2
                @test residual(kkt, [0.7, -0.2], 0.1) ≈ [0.9, 0.7]
            end
            # A large positive ε exposes the old artificial residual floor.
            native = RG.generate_scholtes_complete_kkt_system(problem(preferences, flags))
            options = RG.ScholtesOptions(; tol = 1e-10, tol_inner = 1e-12,
                rho_schedule = [0.1], max_inner = 200, eta_init = 1e-6,
                linear_solver = :svd)
            out = RG.solve(RG.Scholtes(), native, Float64[]; z₀ = [0.7], options)
            @test out.status == :solved
            @test norm(evaluate(native, out.w, 0.1), Inf) < 1e-10
            @test abs(out.z[1]) < 1e-10
        end
    end
    for flags in ([true, false], [false, true])
        preferences = Function[
            flag ? ((x, θ) -> Float64[]) : ((x, θ) -> x[1]^2 / 2) for flag in flags
        ]
        kkt = RG.generate_slacked_complete_kkt_system(problem(preferences, flags))
        @test isempty(kkt.preference_slack_dims)
        @test isempty(kkt.interior_point_slack_dims)
        @test isempty(kkt.inequality_constraint_dual_dims)
        z = fill(0.7, kkt.variable_dimension)
        @test residual(kkt, z, 0.1) == residual(kkt, z, 1e-7)
        @test iszero(norm(residual(kkt, zeros(length(z)), 0.1), Inf))
    end
    constrained = RG.ParametricGOOP(BlockArray(zeros(2), [2]), BlockArray(Float64[], [0]);
        preferences = [Function[(x, θ) -> (x[1] - 1)^2 + x[2]^2 for _ in 1:3]],
        is_prioritized_constraint = [fill(false, 3)],
        equality_constraints = [(x, θ) -> [x[2]]], inequality_constraints = [nothing])
    kkt = RG.generate_slacked_complete_kkt_system(constrained)
    certificate = zeros(kkt.variable_dimension)
    certificate[kkt.primal_dims] .= [1.0, 0.0]
    @test isempty(kkt.interior_point_slack_dims)
    @test isempty(kkt.inequality_constraint_dual_dims)
    @test iszero(norm(residual(kkt, certificate, 0.1), Inf))

    # Keep a current preference's budget when only the inherited vector is empty.
    preferences = Function[(x, θ) -> [2 + x[1]], (x, θ) -> x[1]^2 / 2]
    kkt = RG.generate_slacked_complete_kkt_system(problem(preferences, [true, false]))
    ε, h = 0.1, 2.0
    s = 2ε * h / (sqrt(h^2 + 4ε^2) + h - 2ε)
    γₚ, μₛ = ε / (h + s), ε / s
    certificate = [0.0, s, h + s, s, -γₚ, γₚ, μₛ]
    @test length(certificate) == kkt.variable_dimension == kkt.kkt_dimension
    @test norm(residual(kkt, certificate, ε), Inf) < 1e-12
end

@testset "Complete KKT preference slacks" begin
    for flags in ([false, true], [true, false], [false, true, false],
                  [false, false, true], [true, true])
        preferences = Function[
            flag ? ((x, θ) -> [1 + x[1], 1 - x[1]]) : ((x, θ) -> (x[1] - 1)^2)
            for flag in flags
        ]
        kkt = RG.generate_slacked_complete_kkt_system(
            problem(preferences, flags; inequality = (x, θ) -> [x[1] + 2]),
        )
        @test length(kkt.preference_slack_dims) == 2count(flags)
        @test size(kkt.∇F_z!.result_buffer) == (kkt.kkt_dimension, kkt.variable_dimension)
        for level in findall(flags)
            columns(prefix) = findall(v -> startswith(string(v), "$(prefix)_1_$(level)["), kkt.z_symbolic)
            s, γₚ, μₛ = (columns(prefix) for prefix in ("s", "γₚ", "μₛ"))
            @test length(s) == length(γₚ) == length(μₛ) == 2
            @test all(in(kkt.inequality_constraint_dual_dims), μₛ)
            for j in 1:2
                # Stationarity of min sum(s), with both h+s >= 0 and s >= 0.
                stationarity = 1 - kkt.z_symbolic[γₚ[j]] - kkt.z_symbolic[μₛ[j]]
                @test any(f -> isequal(f, stationarity), kkt.F_symbolic)
                @test any(kkt.interior_point_slack_dims) do σ
                    any(f -> isequal(f, kkt.z_symbolic[s[j]] - kkt.z_symbolic[σ]), kkt.F_symbolic)
                end
                if level > 1
                    @test any(kkt.interior_point_slack_dims) do σ
                        any(f -> isequal(f, kkt.z_symbolic[μₛ[j]] - kkt.z_symbolic[σ]), kkt.F_symbolic)
                    end
                end
            end
        end

        z = 0.6 .+ 0.1sin.(1:kkt.variable_dimension)
        J = deepcopy(kkt.∇F_z!.result_buffer)
        kkt.∇F_z!(J, z; θ = Float64[], ϵ = 1e-3, η = 0.0)
        basis = Matrix{Float64}(I, length(z), length(z))
        h = 1e-5
        finite_difference = hcat([
            (residual(kkt, z + h * basis[:, j], 1e-3) -
             residual(kkt, z - h * basis[:, j], 1e-3)) / (2h)
            for j in eachindex(z)
        ]...)
        @test norm(Matrix(J) - finite_difference, Inf) / max(1, norm(J, Inf)) < 1e-7
    end

    preferences = Function[(x, θ) -> x[1]^2, (x, θ) -> [1 + x[1], 1 - x[1]]]
    kkt = RG.generate_slacked_complete_kkt_system(problem(preferences, [false, true]))
    for ε in (1e-3, 1e-5, 1e-7)
        # A symmetric certificate with both preference inequalities strictly feasible.
        s = ε / 4
        γₚ = ε / (4 * (1 + s))
        μₛ = 1 - γₚ
        lower_G = [1 + s, 1 + s, γₚ, γₚ, ε / 2,
                   s, s, μₛ, μₛ, ε - 2μₛ * s]
        t = ε / (1 + 2s + 2ε)
        upper_γ = [t / 2, t / 2, t * (1 + s), t * (1 + s), t,
                   t / 2, t / 2, t * s, t * s, t]
        certificate = [0.0; fill(s, 2); lower_G; zeros(3);
                       fill(γₚ, 2); fill(μₛ, 2); upper_γ]
        @test length(certificate) == kkt.variable_dimension
        @test norm(residual(kkt, certificate, ε), Inf) < 1e-12
        for block in (kkt.preference_slack_dims, kkt.interior_point_slack_dims,
                      kkt.inequality_constraint_dual_dims)
            @test minimum(certificate[block]) >= 0
        end
        native = RG.generate_scholtes_complete_kkt_system(problem(preferences, [false, true]))
        options = RG.ScholtesOptions(; tol = 1e-11, tol_inner = 1e-11,
            rho_schedule = [ε], max_inner = 200, linear_solver = :svd)
        initial = copy(certificate)
        initial[1] = ε / 100
        out = RG.solve(RG.Scholtes(), native, Float64[]; w₀ = initial, options)
        # A fixed positive rho stage converges to its relaxed equations; the
        # solver's :solved status additionally requires the unrelaxed tolerance.
        @test out.status == :solved || out.termination_reason == :inner_tolerance_reached
        @test norm(evaluate(native, out.w, ε), Inf) < 1e-9
        println("Complete preference validation: rho=$ε, stop=$(out.termination_reason), ",
                "iterations=$(out.iters), residual=$(norm(evaluate(native, out.w, ε), Inf))")
    end

    preferences = Function[(x, θ) -> [2 + x[1]], (x, θ) -> (x[1] - 1)^2]
    kkt = RG.generate_slacked_complete_kkt_system(
        problem(preferences, [true, false]; inequality = (x, θ) -> [x[1]]),
    )
    ε = 1e-3
    x = (1 + sqrt(1 + ε)) / 2
    γ = sqrt(1 + ε) - 1
    h = 2 + x
    s = 2ε * h / (sqrt(h^2 + 4ε^2) + h - 2ε)
    γₚ, μₛ = ε / (h + s), ε / s
    t = (ε + γₚ * x) / (1.5ε)
    certificate = [x, s, γ, x, ε / 2, h + s, s, 0.0,
                   γ, t * x, t * γ - γₚ, t, γₚ, μₛ]
    @test length(certificate) == kkt.variable_dimension
    @test norm(residual(kkt, certificate, ε), Inf) < 1e-12
    @test minimum(certificate[kkt.inequality_constraint_dual_dims]) >= 0
end


@testset "Native complete systems: aggregate parity" begin
    for levels in (2, 3), players in (1, 3), kind in (:quadratic, :nonlinear)
        case = build_benchmark_problem(; num_players = players, levels, kind)
        old = RG.generate_slacked_complete_kkt_system(case.problem)
        new = RG.generate_scholtes_complete_kkt_system(case.problem)
        @test isequal(old.z_symbolic, new.z_symbolic)
        @test old.variable_dimension == new.n
        @test old.kkt_dimension == new.n_nc
        @test [collect(old.preference_slack_dims); old.interior_point_slack_dims;
            old.inequality_constraint_dual_dims] == new.nonnegative_dims
        y = 0.7 .+ 0.1sin.(1:new.n)
        for rho in (1e-3, 1e-9)
            r = zeros(old.kkt_dimension)
            old.F!(r, y; θ = zeros(new.parameter_dimension), ϵ = rho, η = 0.0)
            J = copy(old.∇F_z!.result_buffer)
            old.∇F_z!(J, y; θ = zeros(new.parameter_dimension), ϵ = rho, η = 0.0)
            @test evaluate(new, y, rho) ≈ r atol=1e-11 rtol=1e-12
            @test jacobian(new, y, rho) ≈ J atol=1e-11 rtol=1e-12
        end
    end
end

@testset "Exact complete system: manuscript equations (2.12)-(2.17)" begin
    p = RG.ParametricGOOP(BlockArray(zeros(2), [2]), BlockArray(Float64[], [0]);
        preferences = [Function[(x,t) -> (x[1]+2x[2])^2/2,
            (x,t) -> cosh(x[1]-1)+x[2]^2/2]],
        is_prioritized_constraint = [[false,false]],
        equality_constraints = [(x,t) -> [x[2]-x[1]]],
        inequality_constraints = [(x,t) -> [x[1]+1, 2-x[1]+x[2]]])
    k = RG.generated_exact_complete_kkt_system(p)
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
    @test length(inner.products) == 2
    @test length(outer.products) == 4
    y = 0.5 .+ 0.1sin.(1:k.n)
    @test evaluate(k,y,1.0) == evaluate(k,y,1e-12)
    d = cos.(1:k.n); d ./= norm(d)
    @test jacobian(k,y,1.0)*d ≈ (evaluate(k,y+1e-6d,1.0)-evaluate(k,y-1e-6d,1.0))/2e-6 atol=1e-7

    p3 = RG.ParametricGOOP(BlockArray(zeros(2),[2]),BlockArray(Float64[],[0]);
        preferences=[Function[(x,t)->x[1]^2/2;p.preferences[1]]],
        is_prioritized_constraint=[[false,false,false]],
        equality_constraints=p.equality_constraints,inequality_constraints=p.inequality_constraints)
    k3 = RG.generated_exact_complete_kkt_system(p3)
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
    # generated_perturbed_complete_kkt_system keeps the exact recursion. At the top level each level k's
    # products with g use its own slacks σ_k, with g - σ_k = 0, and every product equals rho.
    for (levels, players, kind) in ((2, 1, :quadratic), (3, 3, :nonlinear))
        case = build_benchmark_problem(; num_players = players, levels, kind)
        e = RG.generated_exact_complete_kkt_system(case.problem)
        k = RG.generated_perturbed_complete_kkt_system(case.problem)
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
        # Per player: the exact rows, then the slack rows (exact: m_i, perturbed: K_i m_i).
        oe, ok = 0, 0
        for i in 1:case.problem.num_players
            n = length(e.levels[i][1].F)
            @test r0[(ok+1):(ok+n)] ≈ re[(oe+1):(oe+n)] atol=1e-12
            @test norm(r0[(ok+n+1):(ok+n+K[i]*m[i])]) <= 1e-12
            oe, ok = oe + n + m[i], ok + n + K[i] * m[i]
        end
        @test (oe, ok) == (e.n_nc, k.n_nc)
        # rho shifts exactly the complementarity products, (2^K - 1) m_I per player here.
        @test count(!iszero, r - r0) == sum(length(level.products) for player in e.levels for level in player)
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

@testset "Aggregate parity with prioritized constraints at every level" begin
    for flags in ([true,false],[false,true],[true,true],
                  [true,false,false],[false,true,false],[false,false,true],
                  [true,true,false],[true,false,true],[false,true,true],[true,true,true])
        p = RG.ParametricGOOP(BlockArray(zeros(2),[2]),BlockArray([0.2],[1]);
            preferences=[Function[flag ? ((x,t)->[2+x[1],1-x[2]]) :
                ((x,t)->cosh(x[1]-t[1])+x[2]^2/2) for flag in flags]],
            is_prioritized_constraint=[flags],
            equality_constraints=[(x,t)->[x[2]-x[1]]],
            inequality_constraints=[(x,t)->[x[1]+3]])
        old,new = RG.generate_slacked_complete_kkt_system(p), RG.generate_scholtes_complete_kkt_system(p)
        @test isequal(old.z_symbolic,new.vars)
        @test old.kkt_dimension == new.n_nc
        @test old.variable_dimension == new.n
        y = 0.5 .+ 0.1sin.(1:new.n)
        for rho in (1e-3,1e-9)
            theta = [0.2]
            r = zeros(old.kkt_dimension)
            old.F!(r,y;θ=theta,ϵ=rho,η=0.0)
            J = copy(old.∇F_z!.result_buffer)
            old.∇F_z!(J,y;θ=theta,ϵ=rho,η=0.0)
            Jnew = copy(new.Jnc_buf);new.J_nc!(Jnew,y,theta,rho)
            @test evaluate(new,y,rho;theta) ≈ r atol=1e-10 rtol=1e-12
            @test Jnew ≈ J atol=1e-10 rtol=1e-12
        end
    end
end

@testset "Complete empty inequalities, preference bounds, and K >= 2" begin
    for builder in (BUILDERS..., RG.generated_perturbed_complete_kkt_system)
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
            G! = RG._compile_scholtes_block(k.slack_expressions,k.vars,Symbolics.Num[],
                Symbolics.variable(:complete_rho),Symbolics.variable(:complete_unused_eta);
                codegen=:fast_differentiation,fd_codegen_chunk_size=32,backend_options=(;))[1]
            G = zeros(length(k.slack_expressions)); G!(G,y,Float64[],0.0)
            if k.formulation in (:exact, :perturbed)
                y[k.interior_point_slack_dims] .= G
                @test norm(evaluate(k,y,0.0)) < 1e-12
            elseif !any(flags)
                @test isempty(k.nonnegative_dims)
                @test evaluate(k,y,1.0) == evaluate(k,y,1e-9) == zeros(k.n_nc)
            end
            @test all(isfinite, jacobian(k,0.5 .+ y,1e-3))
        end
    end
end

@testset "Scholtes projected engine solves both complete formulations" begin
    options = RG.ScholtesOptions(;rho_schedule=[1e-9],max_inner=500,tol=1e-7,tol_inner=1e-10,
        eta_init=1e-4,eta_schedule=true,eta_max=1e4,proj_eps=1e-8)
    for levels in (2,3), players in (1,3), kind in (:quadratic,:nonlinear), builder in BUILDERS
        case = build_benchmark_problem(;num_players=players,levels,kind)
        k = builder(case.problem)
        @test k.n_comp == 0
        k.formulation === :exact && @test k.n_nc == k.n
        output = RG.solve(RG.Scholtes(),k,zeros(k.parameter_dimension);z₀=case.expected,options)
        @test output.status === :solved
        @test norm(evaluate(k,output.w,1e-9)) < 1e-7
        @test minimum(output.w[k.nonnegative_dims]) >= 0
        @test output.z ≈ case.expected atol=1e-6 rtol=0
        @test RG.is_feasible(case.problem,output.z,zeros(k.parameter_dimension);tol=1e-7).feasible
        @test output.iters > 1
    end
    case = build_benchmark_problem(;num_players=1,levels=2,kind=:quadratic)
    k = RG.generate_scholtes_complete_kkt_system(case.problem)
    chain = RG.solve(RG.Scholtes(),k,zeros(k.parameter_dimension);z₀=case.expected,
        options=RG._with_options(options;rho_schedule=[1e-3,1e-9],max_inner=1000))
    @test chain.rho == 1e-9
    @test chain.status === :solved
    bad = copy(chain.w);bad[first(k.nonnegative_dims)] = -1
    @test_throws ArgumentError RG.solve(RG.Scholtes(),k,zeros(k.parameter_dimension);w₀=bad,options)
end


@testset "Complete/reduced unconstrained solution agreement" for levels in (2, 3)
	case = build_unconstrained_quadratic_problem(; levels)

	kkt = RG.generate_scholtes_complete_kkt_system(case.problem)
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
