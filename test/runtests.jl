# runtests.jl -- the test suite in four parts; the problems and helpers are in problems.jl.
#
#   1. Construction  each generator builds the system the manuscript defines (no solver)
#   2. Mechanics     what PDIP and ProjectedLM do at each step, on small problems
#   3. Solutions     each system, solved by its solver, reaches the right answer:
#                      reduced (5.2), quasi           PDIP           known solutions (3.1), the coupled game (3.2)
#                      reduced (5.2)                  ProjectedLM    the coupled game (3.2)
#                      complete :perturbed            PDIP           the coupled game (3.2)
#                      Scholtes, quasi Scholtes       ProjectedLM    the source's small problems (3.3)
#                      complete :exact                ProjectedLM    the benchmarks, agreement with reduced (3.4)
#                      Scholtes, complete :exact      PDIP           refused: no interior (2)
#   4. Multistart    solve_multi and the start samplers

include(joinpath(@__DIR__, "problems.jl"))

@testset "1. Construction: each generator builds the system the manuscript defines" begin
    "Rows, unknowns, index sets and Jacobians of every system, checked at fixed points, without a solver."

    @testset "1.1 Reduced (5.2) and quasi systems" begin
        "The interior-point systems' index sets and the quasi truncation."

        @testset "the dual-coordinate index sets of the reduced system" begin
            x_template = BlockArray(zeros(2), [2])
            θ_template = BlockArray(zeros(1), [1])
            problem = ReducedGOOP.ParametricGOOP(
                x_template,
                θ_template;
                preferences = [
                    Function[
                        (x, θ)->sum(abs2, x[Block(1)]),
                        (x, θ)->sum(abs2, x[Block(1)] .- 1.0),
                    ],
                ],
                is_prioritized_constraint = [[false, false]],
                equality_constraints = [(x, θ) -> [x[Block(1)][1]]],
                inequality_constraints = [(x, θ) -> [x[Block(1)][2]]],
            )
            kkt = reduced_kkt_system(problem)

            @test length(kkt.equality_constraint_dual_dims) == 2
            @test length(kkt.stationarity_dual_dims) == 2
            @test kkt.all_equality_stationarity_dual_dims ==
                  vcat(kkt.equality_constraint_dual_dims, kkt.stationarity_dual_dims)
            @test kkt.innermost_stationarity_dual_dims == kkt.stationarity_dual_dims
            @test isempty(
                intersect(
                    kkt.all_equality_stationarity_dual_dims,
                    kkt.inequality_constraint_dual_dims,
                ),
            )
            other_dims = setdiff(
                1:kkt.variable_dimension,
                vcat(
                    kkt.primal_dims,
                    kkt.preference_slack_dims,
                    kkt.interior_point_slack_dims,
                    kkt.inequality_constraint_dual_dims,
                    kkt.all_equality_stationarity_dual_dims,
                ),
            )
            @test length(other_dims) == 1 # Block 10 (Φ) is not carried.
            @test isempty(intersect(kkt.all_equality_stationarity_dual_dims, other_dims))
        end

        @testset "the quasi system: truncation after quasi_order derivatives, and the reduced system on quadratic data and at one level" begin
            # The quasi system truncates every Lagrangian term after `quasi_order` derivatives (default 2). A large
            # order truncates nothing, so its rows evaluate to the reduced system's (whose identically zero rows the
            # quasi system drops); order 1 keeps first-order information only, so no row depends on ψ.
            case = build_benchmark_problem(; num_players = 1, levels = 3, kind = :nonlinear)
            reduced = reduced_kkt_system(case.problem)
            quasi(order) = ReducedGOOP.generate_slacked_quasi_kkt_system(case.problem; quasi_order = order)
            @test_throws ArgumentError quasi(0)
            @test isequal(quasi(2).K_ρ_symbolic, quasi_kkt_system(case.problem).K_ρ_symbolic)
            full = quasi(10)
            kept = findall(row -> !iszero(row), reduced.K_ρ_symbolic)
            @test full.kkt_dimension == length(kept)
            z = 0.5 .+ 0.1 .* sin.(1:(reduced.variable_dimension))
            θ = zeros(reduced.parameter_dimension)
            K_ρ, K_ρ_full = zeros(reduced.kkt_dimension), zeros(full.kkt_dimension)
            reduced.K_ρ!(K_ρ, z; θ, ρ = 1e-3)
            full.K_ρ!(K_ρ_full, z; θ, ρ = 1e-3)
            @test K_ρ_full ≈ K_ρ[kept] rtol = 1e-12
            first_order = quasi(1)
            ψ = findall(v -> startswith(string(v), "ψ_"), first_order.z_symbolic)
            J = first_order.∇K_ρ!.result_buffer
            @test !isempty(ψ)
            @test all(j -> isempty(ReducedGOOP.SparseArrays.nzrange(J, j)), ψ)
            # Nothing is differentiated more than twice on quadratic data, nor more than once at one level: there
            # the quasi system (order 2) is the reduced system less its identically zero rows.
            for case in (build_benchmark_problem(; num_players = 1, levels = 3, kind = :quadratic),
                         build_benchmark_problem(; num_players = 1, levels = 1, kind = :nonlinear))
                r, q = reduced_kkt_system(case.problem), quasi_kkt_system(case.problem)
                kept = findall(row -> !iszero(row), r.K_ρ_symbolic)
                @test q.variable_dimension == r.variable_dimension && q.kkt_dimension == length(kept)
                z = 0.5 .+ 0.1 .* sin.(1:(r.variable_dimension))
                @test residual(q, z, 1e-3) ≈ residual(r, z, 1e-3)[kept] rtol = 1e-12
            end
        end
    end

    @testset "1.2 Scholtes system" begin
        "The reduced system in Scholtes form, row by row as ScholtesReducedGOOP.jl's tests check it; its quasi form; its starts."

        @testset "the source's test problems on ParametricGOOP" begin
            @test prob_C.N == 2 && prob_C.Ki == [2, 2]
            @test prob_C.np == 2
            @test prob_C.mIi == [1, 1] && prob_C.mEi == [0, 0]
            @test sfeasible(prob_A, [0.0]).feasible
            @test !sfeasible(prob_A, [-3.0]).feasible
            @test sfeasible(prob_A, [-3.0]).worst ≈ -1.0
        end

        @testset "unknowns and rows" begin
            k = scholtes_kkt(prob_C)
            # Per player: level 1 relaxes its g (σ, γ, u) and slacks level 2's product (σ̂, ϕ); level 2 relaxes its g.
            @test (length(relaxed_pairs(k)), length(exact_pairs(k))) == (2 * 2, 2 * 1)
            # Rows: the stationarity of x at both levels, then two per pair (its slack row and its product).
            @test k.kkt_dimension == 2 * 2 + 2 * length(k.complementarity_pairs)
            # Unknowns: x, then per player ψ, σ, γ and u per relaxed pair, σ̂ and ϕ per exact one.
            @test k.variable_dimension == 2 + 2 * (1 + 3 * 2 + 2 * 1)
            # K = 2, no equalities: square but for the u's.
            @test k.variable_dimension - k.kkt_dimension == length(relaxed_pairs(k))
        end

        @testset "explicit prioritized-constraint preference slacks" begin
            function prioritized_problem(preferences, flags; inequality = (x, θ) -> [x[1] + 3.0])
                RG.ParametricGOOP(
                    BlockArray(zeros(1), [1]), BlockArray(zeros(0), [0]);
                    preferences = [Function[preferences...]],
                    is_prioritized_constraint = [collect(flags)],
                    equality_constraints = [nothing], inequality_constraints = [inequality],
                )
            end
            build_prioritized(goop; codegen = :fast_differentiation) =
                RG.generate_slacked_scholtes_reduced_kkt_system(goop; codegen)

            @testset "inherited constraints and current-level differentiation" begin
                # Preference variables and inequalities persist above their source level,
                # but reduced stationarity differentiates only x and the current slack.
                for flags in ((true, false, false), (false, true, false),
                              (false, false, true), (true, false, true), (false, true, true))
                    preferences = [flag ? ((x, θ) -> [x[1] + 1.0, 2.0 - x[1]]) :
                                   ((x, θ) -> (x[1] - 0.5)^2) for flag in flags]
                    k = build_prioritized(prioritized_problem(preferences, flags))
                    K = length(flags)
                    dimensions = [1 + (flags[level] ? 2 : 0) for level in 1:K]
                    inequality_dimensions = [1 + 4count(identity, flags[level:K]) for level in 1:K]
                    labels = [pair.label for pair in k.complementarity_pairs]
                    relaxed, exact = relaxed_pairs(k), exact_pairs(k)
                    @test k.primal_dims == 1:1
                    @test length(k.preference_slack_dims) == 2count(identity, flags)
                    @test k.kkt_dimension == sum(dimensions) + 2length(labels)
                    @test relaxed == 1:sum(inequality_dimensions) # every level's G, then the lower products
                    @test length(exact) == sum((level - 1) * inequality_dimensions[level] for level in 2:K)
                    @test all(l.kind === :phi_upper for l in labels[exact])

                    columns = [pair.multiplier for pair in k.complementarity_pairs]
                    @test allunique(columns[relaxed])
                    y = zeros(k.variable_dimension)
                    y[1] = 0.25
                    y[k.preference_slack_dims] .= 0.2 .+ 0.1 .* collect(1:length(k.preference_slack_dims))
                    y[columns[relaxed]] .= 0.01 .* collect(1:length(relaxed))
                    # Keeping all policy and phi multipliers zero isolates the direct terms.
                    values = residual(k, y, 0.1)
                    a, b = targets(k, y, 0.1), multipliers(k, y)
                    Ja = target_jacobian(k, y, 0.1)
                    # K_ρ lists a level's stationarity rows, then two rows per pair of that level (its slack row and its
                    # product), then the levels below: level ℓ's stationarity follows the rows of the levels above it.
                    level_rows(level) = dimensions[level] + 2count(l -> l.level == level, labels)
                    stationarity_row(level, component) = sum(level_rows, 1:(level - 1); init = 0) + 1 + component
                    expected_slack_rows = Int[]
                    slack_number = 0
                    for source in 1:K
                        flags[source] || continue
                        for component in 1:2
                            slack_number += 1
                            slack_column = k.preference_slack_dims[slack_number]
                            for level in 1:source
                                row(kind) = only(findall(l -> l.kind === kind && l.level == level &&
                                    l.lower_level == source && l.inequality == component, labels))
                                pref_row, bound_row = row(:preference), row(:preference_slack)
                                level == source && push!(expected_slack_rows, pref_row)
                                @test a[bound_row] == y[slack_column]
                                @test a[pref_row] ≈ (component == 1 ? 1.25 : 1.75) + y[slack_column]
                                @test findall(!iszero, Ja[bound_row, :]) == [slack_column]
                                @test Ja[bound_row, slack_column] == Ja[pref_row, slack_column] == 1.0
                                if level == source
                                    @test values[stationarity_row(level, component)] ≈ 1.0 - b[pref_row] - b[bound_row]
                                end
                            end
                        end
                    end
                    @test k.preference_slack_rows == expected_slack_rows
                    # Each exact phi pair bounds one lower-level relaxed product, including
                    # products for h + preference_slack and preference_slack itself.
                    inherited_rows = Int[]
                    for row in exact
                        source_column = only(findall(!iszero, Ja[row, columns[relaxed]]))
                        push!(inherited_rows, source_column)
                        @test labels[source_column].level == labels[row].lower_level
                        @test a[row] ≈ 0.1 - a[source_column] * b[source_column]
                    end
                    for row in relaxed
                        @test count(==(row), inherited_rows) == labels[row].level - 1
                    end
                    # An inherited slack can affect an upper Lagrangian without acquiring
                    # an upper stationarity equation: it is an induced variable there.
                    y[columns[exact]] .= 0.02
                    for source in 1:K, component in 1:2, level in 1:(source - 1)
                        flags[source] || continue
                        source_slack = 2count(identity, flags[1:(source - 1)]) + component
                        slack_column = k.preference_slack_dims[source_slack]
                        original_rows = findall(l -> l.kind !== :phi_upper && l.level == level, labels)
                        phi_rows = findall(l -> l.kind === :phi_upper && l.level == level, labels)
                        function upper_lagrangian(at)
                            first_slack = 2count(identity, flags[1:(level - 1)]) + 1
                            objective = flags[level] ? sum(at[k.preference_slack_dims[
                                first_slack:(first_slack + 1)]]) : (at[1] - 0.5)^2
                            av, bv = targets(k, at, 0.1), multipliers(k, at)
                            objective - sum(av[row] * bv[row] for row in original_rows) -
                                sum(av[row] * bv[row] for row in phi_rows)
                        end
                        own_pref = only(findall(l -> l.kind === :preference && l.level == level &&
                            l.lower_level == source && l.inequality == component, labels))
                        own_bound = only(findall(l -> l.kind === :preference_slack && l.level == level &&
                            l.lower_level == source && l.inequality == component, labels))
                        contribution = 0.0
                        for lower in (level + 1):source, kind in (:preference, :preference_slack)
                            original_row = only(findall(l -> l.kind === kind && l.level == lower &&
                                l.lower_level == source && l.inequality == component, labels))
                            inequality = original_row - sum(inequality_dimensions[1:(lower - 1)]; init = 0)
                            phi_row = only(findall(l -> l.kind === :phi_upper && l.level == level &&
                                l.lower_level == lower && l.inequality == inequality, labels))
                            contribution += y[columns[phi_row]] * b[original_row]
                        end
                        plus, minus = copy(y), copy(y)
                        plus[slack_column] += 1e-6
                        minus[slack_column] -= 1e-6
                        derivative = (upper_lagrangian(plus) - upper_lagrangian(minus)) / 2e-6
                        @test derivative ≈ -b[own_pref] - b[own_bound] + contribution atol = 1e-8
                        @test abs(derivative) > 1e-4
                    end
                end
            end

            @testset "native and FastDifferentiation agree with numerical Jacobians" begin
                goop = prioritized_problem(Function[
                    (x, θ) -> (x[1] - 0.5)^2,
                    (x, θ) -> [x[1] + 1.0, 2.0 - x[1]],
                    (x, θ) -> [exp(x[1]) - 0.1, 2.0 - x[1]^2],
                ], (false, true, true))
                native = build_prioritized(goop; codegen = :native)
                fd = build_prioritized(goop)
                @test string.(native.z_symbolic) == string.(fd.z_symbolic)
                y = 0.2 .* cos.(collect(1:(native.variable_dimension)))
                rho, h = 0.2, 1e-6
                direction = sin.(collect(1:(native.variable_dimension)))
                @test residual(native, y, rho) ≈ residual(fd, y, rho) atol = 1e-12 rtol = 1e-12
                @test targets(native, y, rho) ≈ targets(fd, y, rho) atol = 1e-12 rtol = 1e-12
                @test Matrix(jacobian(native, y, rho)) ≈ Matrix(jacobian(fd, y, rho)) atol = 1e-12 rtol = 1e-12
                numeric = (residual(fd, y + h * direction, rho) - residual(fd, y - h * direction, rho)) / (2h)
                @test numeric ≈ jacobian(fd, y, rho) * direction atol = 2e-7 rtol = 2e-7
            end

            @testset "sum of preference slacks preserves the L1 minimizer continuum" begin
                # min e1 + e2 subject to x + e1 >= 0, -1 - x + e2 >= 0, e >= 0
                # has value 1 for all x in [-1, 0]. The reduced KKT system need not select
                # a unique x; a smooth violation power would have a unique inner x = -0.5.
                goop = prioritized_problem(Function[
                    (x, θ) -> (x[1] + 0.75)^2,
                    (x, θ) -> [x[1], -1.0 - x[1]],
                ], (false, true); inequality = nothing)
                k = build_prioritized(goop)
                pairs = k.complementarity_pairs
                # Two full boundary certificates exhibit the L1 continuum without imposing
                # uniqueness on the reduced outer KKT system.
                for x in (-0.75, -0.5)
                    z = zeros(k.variable_dimension)
                    z[1] = x
                    z[k.preference_slack_dims] .= [-x, 1.0 + x]
                    for pair in pairs
                        if pair.label.kind === :preference
                            z[pair.multiplier] = pair.label.level == 2 ? 1.0 :
                                (pair.label.inequality == 1 ? 2(x + 0.75) : 0.0)
                        end
                    end
                    # The slacks on their targets at ρ = 0 (σ = G, σ̂ = −Gₘ ⊙ γₘ), ϕ = 0 and u = 0: a root of K_ρ at ρ = 0.
                    a = targets(k, z, 0.0)
                    @test all(>=(0), a) && all(>=(0), multipliers(k, z))
                    z[[pair.slack for pair in pairs]] .= a
                    @test norm(residual(k, z, 0.0)) == 0.0
                    @test sum(z[k.preference_slack_dims]) == 1.0
                end
                cold = scholtes_start(k, [-0.25], 0.1)
                @test all(>(0), cold[k.preference_slack_dims])
                @test all(>(0), targets(k, cold, 0.1)[relaxed_pairs(k)])
                result = plm_solve(k, [-0.25]; ρ_schedule = 10.0 .^ collect(-2:-1:-10), max_inner_iters = 1000,
                    tol = 1e-7, tol_inner = 1e-11)
                @test result.status === :solved
                @test length(result.x) == 1
                @test result.x == result.z[k.primal_dims]
                @test all(isfinite, result.x)
                preference_slacks = result.z[k.preference_slack_dims]
                @test preference_slacks ≈ [-result.x[1], 1.0 + result.x[1]] atol = 1e-4
                @test sum(preference_slacks) ≈ 1.0 atol = 1e-5
                eq_cols = findall(v -> startswith(string(v), "ψ_"), k.z_symbolic)
                w0 = scholtes_start(k, result.x, 1e-10; warm = true, eq_cols, eq = result.z[eq_cols], preference_slacks)
                @test w0[k.primal_dims] == result.x
                @test all(>(0), w0[k.preference_slack_dims])
                @test all(>(0), targets(k, w0, 1e-10)[relaxed_pairs(k)])
                @test all(w0[k.preference_slack_dims] .>= preference_slacks)
                @test w0[k.preference_slack_dims] ≈ preference_slacks atol = 1e-7
                for invalid_seed in ([1.0], [NaN, 1.0], [1.0, Inf])
                    @test_throws ArgumentError scholtes_start(k, result.x, 1e-10; warm = true, preference_slacks = invalid_seed)
                end
                for linear_solver in (:normal, :klu, :ldl)
                    options = RG.ProjectedLMOptions(; linear_solver, ρ_schedule = [1e-10], max_inner_iters = 2, tol = 0.0,
                        tol_inner = 0.0)
                    warm = RG.solve(RG.ProjectedLM(), k, θ0; z₀ = w0, options)
                    @test warm.x == warm.z[k.primal_dims]
                    @test length(warm.x) == 1
                    @test all(isfinite, warm.z[k.preference_slack_dims])
                end
            end
        end

        @testset "the quasi system: the same unknowns, only identically zero rows dropped" begin
            # The quasi system keeps the full system's unknowns, complementarity pairs and slack targets, so the same
            # starts serve both. Its truncation can leave stationarity rows identically zero; those are dropped, as in
            # the interior-point quasi system (none on the source's problems, three on the 3-player benchmark).
            benchmark = build_benchmark_problem(; num_players = 3, levels = 2, kind = :quadratic).problem
            for (goop, dropped) in ((prob_C.goop, 0), (prob_E.goop, 0), (prob_F.goop, 0), (prob_G.goop, 0), (benchmark, 3))
                ke = RG.generate_slacked_scholtes_reduced_kkt_system(goop)
                kq = RG.generate_slacked_scholtes_reduced_kkt_system(goop; drop_higher_order_terms = true)
                @test string.(kq.z_symbolic) == string.(ke.z_symbolic)
                @test kq.complementarity_pairs == ke.complementarity_pairs
                @test isequal(kq.slack_expressions, ke.slack_expressions)
                @test kq.kkt_dimension == ke.kkt_dimension - dropped
                @test !any(iszero, kq.K_ρ_symbolic)
                # Every dropped row is a stationarity row: the slack and product rows are the full system's, in order.
                has_slack = slack_row_filter(ke)
                @test isequal(filter(has_slack, kq.K_ρ_symbolic), filter(has_slack, ke.K_ρ_symbolic))
                @test count(has_slack, kq.K_ρ_symbolic) == 2length(kq.complementarity_pairs)
                @test nnz(kq.∇K_ρ!.result_buffer) <= nnz(ke.∇K_ρ!.result_buffer)
            end
            @test_throws ArgumentError scholtes_kkt(prob_E; quasi = true, quasi_order = 0)
        end

        @testset "quasi_order = 1 keeps first-order information only" begin
            k1 = scholtes_kkt(prob_E; quasi = true, quasi_order = 1)
            psi_cols = findall(v -> startswith(string(v), "ψ_"), k1.z_symbolic)
            @test !isempty(psi_cols)
            # ψ multiplies the policy gradient π, whose derivative is second-order: no row depends on ψ.
            @test all(j -> isempty(nzrange(k1.∇K_ρ!.result_buffer, j)), psi_cols)
            Random.seed!(8)
            ke, kq = scholtes_kkt(prob_G; quasi = false), scholtes_kkt(prob_G; quasi = true, quasi_order = 10)
            z = randn(ke.variable_dimension)
            @test residual(kq, z, 0.3) == residual(ke, z, 0.3)
        end

        @testset "generate_slacked_scholtes_quasi_kkt_system is the quasi Scholtes system" begin
            for quasi_order in (1, 2)
                kq = RG.generate_slacked_scholtes_quasi_kkt_system(prob_G.goop; quasi_order)
                ke = scholtes_kkt(prob_G; quasi = true, quasi_order)
                @test string.(kq.z_symbolic) == string.(ke.z_symbolic)
                @test isequal(kq.K_ρ_symbolic, ke.K_ρ_symbolic)
            end
        end

        @testset "the rectangularity surplus, exactly" begin
            surplus(K, n, mE) = ((K - 1) * (K - 2) ÷ 2) * n + (K - 1) * mE
            mk(K, n, mE, mI) = sproblem(ni = [n],
                objectives = [[let k = k; z -> sum(z[i]^2 for i in 1:n) + k * z[1]; end for k in 1:K]],
                equality = mE > 0 ? [let m = mE; z -> [z[j] - 0.1j for j in 1:m]; end] : nothing,
                inequality = mI > 0 ? [let m = mI; z -> [z[j] + 2.0 for j in 1:m]; end] : nothing,
                z0 = zeros(n))
            # Each complementarity pair adds its slack and two rows (its slack row and its product), and each
            # relaxed one also its u: the unknowns beyond the rows are the reduced system's surplus and the u's.
            excess(k) = k.variable_dimension - k.kkt_dimension - length(relaxed_pairs(k))
            for K in 2:4, n in (2, 4), mE in (0, 1), mI in (1, 2)
                @test excess(scholtes_kkt(mk(K, n, mE, mI))) == surplus(K, n, mE)
            end
            for K in 3:4, n in (2,), mE in (0, 1), mI in (0, 1)
                @test excess(scholtes_kkt(mk(K, n, mE, mI))) > 0
            end
            for n in (2, 4), mI in (1, 2)
                @test excess(scholtes_kkt(mk(2, n, 0, mI))) == 0
                @test excess(scholtes_kkt(mk(2, n, 1, mI))) > 0
            end
        end

        @testset "explicit gamma / nonnegative inequality phi" begin
            p = sproblem(ni = [1], objectives = [[z -> -z[1], z -> z[1]]], inequality = [z -> [z[1]]], z0 = [1.0])
            k = scholtes_kkt(p)
            rho = 0.1
            pairs = k.complementarity_pairs
            @test all(!startswith(string(v), "Gam") for v in k.z_symbolic)
            @test length(relaxed_pairs(k)) == 2
            @test length(pairs) == 3 && exact_pairs(k) == [3]
            cols = [pair.multiplier for pair in pairs]
            w = scholtes_start(k, p.z0, rho)
            @test all(>(0), w[k.nonnegative_dims])
            @test all(>(0), w[cols])
            @test targets(k, w, rho)[3] ≈ 0.9rho
            # exact boundary witness: γ₁ = 0, γ₂ = 1, φ = 1, z = ρ; σ on G (= ρ), σ̂ on ρ − G₂γ₂ (= 0), u = ρ − σ ⊙ γ
            y = zeros(k.variable_dimension)
            y[1] = rho
            y[cols] = [0.0, 1.0, 1.0]
            y[[pair.slack for pair in pairs]] .= targets(k, y, rho)
            for pair in pairs[relaxed_pairs(k)]
                y[pair.u_relaxation] = rho - y[pair.slack] * y[pair.multiplier]
            end
            @test norm(residual(k, y, rho)) < 1e-14
            yy = copy(y)
            yy[cols[3]] = -1
            @test residual(k, yy, rho)[1] ≈ -2
            # A start outside the sign box (ϕ < 0, or γ < 0) is refused.
            @test_throws ArgumentError RG.solve(RG.ProjectedLM(), k, θ0; z₀ = yy)
            yy = copy(w)
            yy[cols[1]] = -0.02
            @test_throws ArgumentError RG.solve(RG.ProjectedLM(), k, θ0; z₀ = yy)
            v = sin.(1.0:(k.variable_dimension))
            h = 1e-6
            @test norm((residual(k, w + h * v, rho) - residual(k, w - h * v, rho)) / (2h) - jacobian(k, w, rho) * v) < 1e-8
            # K₀ is K_ρ at ρ = 0 with u = 0.
            K = zeros(k.kkt_dimension)
            k.K₀!(K, w; θ = θ0)
            wzero = copy(w)
            wzero[[pair.u_relaxation for pair in pairs[relaxed_pairs(k)]]] .= 0
            @test K == residual(k, wzero, 0.0)
            # One projected step of every linear solver keeps ϕ (and the rest of the sign box) positive.
            for linear_solver in (:normal, :klu, :ldl)
                options = RG.ProjectedLMOptions(; ρ_schedule = [rho], max_inner_iters = 1, tol = 0.0, tol_inner = 0.0,
                    linear_solver)
                r = RG.solve(RG.ProjectedLM(), k, θ0; z₀ = w, options)
                @test r.newton_steps == 1
                @test all(isfinite, r.z)
                @test r.z[cols[3]] > 0
                @test minimum(r.z[k.nonnegative_dims]) > 0
            end
            p3 = sproblem(ni = [1], objectives = [[z -> z[1], z -> z[1], z -> z[1]]], inequality = [z -> [z[1]]], z0 = [1.0])
            k3 = scholtes_kkt(p3)
            @test length(relaxed_pairs(k3)) == 3
            @test length(k3.complementarity_pairs) == 6
            @test all(pair -> pair.label.kind === :phi_upper && pair.u_relaxation == 0, k3.complementarity_pairs[4:end])
            r = plm_solve(k, p.z0; ρ_schedule = [0.1, 0.01], max_inner_iters = 1000, tol = 0.0, tol_inner = 1e-9)
            @test norm(residual(k, r.z, r.ρ)) < 1e-8
            @test r.x[1] ≈ 0.01 atol = 1e-8
            @test maximum(abs, [r.z[pair.slack] * r.z[pair.multiplier] for pair in pairs[exact_pairs(k)]]) < 1e-8
            @test minimum(r.z[k.nonnegative_dims]) > 0
            single = sproblem(ni = [1], objectives = [[z -> z[1]]], inequality = [z -> [z[1]]], z0 = [1.0])
            r = plm_solve(single)
            @test r.status === :solved && abs(r.x[1]) < 1e-5
            emptyprob = sproblem(ni = [1], objectives = [[z -> z[1]^2]], z0 = [1.0])
            r = plm_solve(emptyprob)
            @test r.status === :solved && abs(r.x[1]) < 1e-7
            # ρ enters the exact pairs' targets ρ𝟙 − Gₘ ⊙ γₘ.
            @test targets(k, w, 0.2)[3] - targets(k, w, 0.1)[3] ≈ 0.1
        end

        @testset "initialize_scholtes!: the warm and cold starts" begin
            # Warm (a continuation start): s = max(seed, margin, margin − h) with margin = max(s_floor, 0.1ρ),
            # σ = max(G, s_floor), γ = 0.1ρ / max(σ, ρ), σ̂ = max(ρ𝟙 − Gₘ ⊙ γₘ, min(s_floor, 0.1ρ)),
            # ϕ = 0.1ρ / max(σ̂, ρ), u = max(ρ − σ ⊙ γ, 1e-6ρ). Cold (the source's start): margin 1 and no seed,
            # σ = G > 0, γ = 0.1ρ / σ, σ̂ = ρ𝟙 − Gₘ ⊙ γₘ, ϕ = 0.1ρ / σ̂, u likewise. G, ρ𝟙 − Gₘ ⊙ γₘ and h are the
            # system's slack targets.
            prioritized = RG.ParametricGOOP(BlockArray([0.0], [1]), BlockArray(Float64[], [0]);
                preferences = [Function[(x, t) -> [1 - x[1]], (x, t) -> x[1]^2 / 2, (x, t) -> [x[1] + 2]]],
                is_prioritized_constraint = [[true, false, true]], equality_constraints = [nothing],
                inequality_constraints = [(x, t) -> [3 - x[1]]])
            kp = RG.generate_slacked_scholtes_reduced_kkt_system(prioritized)
            s_floor = 1e-8
            for (k, x) in ((scholtes_kkt(prob_E), [0.1, 0.2]), (kp, [0.5])), ρ in (1.0, 1e-3), warm in (true, false)
                z = scholtes_start(k, x, ρ; warm)
                pairs, a = k.complementarity_pairs, targets(k, z, ρ)
                @test z[k.primal_dims] == x
                @test all(>(0), z[k.nonnegative_dims])
                unslacked = copy(z)
                unslacked[k.preference_slack_dims] .= 0.0
                h = targets(k, unslacked, ρ)[k.preference_slack_rows]
                margin = warm ? max(s_floor, 0.1ρ) : 1.0
                @test z[k.preference_slack_dims] == max.(margin, margin .- h)
                for j in relaxed_pairs(k)
                    pair = pairs[j]
                    σ, γ, target = z[pair.slack], z[pair.multiplier], a[j]
                    @test σ == (warm ? max(target, s_floor) : target)
                    @test γ == (warm ? 0.1ρ / max(σ, ρ) : 0.1 * ρ / σ)
                    @test z[pair.u_relaxation] == max(ρ - σ * γ, 1e-6ρ)
                end
                for j in exact_pairs(k)
                    pair = pairs[j]
                    σ̂, ϕ, target = z[pair.slack], z[pair.multiplier], a[j]
                    @test σ̂ == (warm ? max(target, min(s_floor, 0.1ρ)) : target)
                    @test ϕ == (warm ? 0.1ρ / max(σ̂, ρ) : 0.1 * ρ / σ̂)
                end
                # The cold start puts every slack on its target: the slack rows of K_ρ hold.
                warm || @test maximum(abs, residual(k, z, ρ)[slack_rows(k)]) <= 1e-14
            end
            # The multipliers kept from a previous stage, and a seed for the preference slacks.
            k = scholtes_kkt(prob_E)
            eq_cols = findall(v -> startswith(string(v), "ψ_"), k.z_symbolic)
            eq = 0.1 .* collect(1:length(eq_cols))
            @test scholtes_start(k, [0.1, 0.2], 1e-3; warm = true, eq_cols, eq)[eq_cols] == eq
            seed = fill(5.0, length(kp.preference_slack_dims))
            @test scholtes_start(kp, [0.5], 1e-3; warm = true, preference_slacks = seed)[kp.preference_slack_dims] == seed
            # The cold start needs every inequality strictly satisfied: prob_A's g(z) = z + 2 is 0 at z = −2.
            @test_throws ArgumentError scholtes_start(scholtes_kkt(prob_A), [-2.0], 1.0)
            # Only the Scholtes system has the pairs and slack targets the start reads.
            @test_throws ArgumentError RG.initialize_scholtes!(zeros(1), RG.generate_slacked_reduced_kkt_system(prob_E.goop),
                θ0, [0.1, 0.2], 1e-3)
        end
    end

    @testset "1.3 Complete systems" begin
        "The manuscript's equations (2.12)-(2.17), the perturbed rows (6.1), K >= 2, empty inequalities and preference bounds."

        @testset "Complete KKT requires at least two levels per player" for builder in BUILDERS
            for preferences in (Function[], Function[(x, θ) -> x[1]^2])
                @test_throws ArgumentError builder(
                    one_player_problem(preferences, fill(false, length(preferences))),
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
            x = k.z_symbolic[k.primal_dims]
            f, g = [x[2]-x[1]], [x[1]+1, 2-x[1]+x[2]]
            L2 = cosh(x[1]-1)+x[2]^2/2-inner.lambda'*f-inner.gamma'*g
            F2 = [Symbolics.gradient(L2,x); f; g .* inner.gamma]
            G2 = [g; inner.gamma]
            L1 = (x[1]+2x[2])^2/2-outer.lambda'*F2-outer.gamma'*G2
            F1 = [Symbolics.gradient(L1,[x;inner.lambda;inner.gamma]);
                G2 .* outer.gamma; F2]
            G1 = [g;outer.gamma;inner.gamma]
            reference = [F1;g-k.z_symbolic[k.interior_point_slack_dims]]
            @test length(reference) == k.kkt_dimension == k.variable_dimension
            @test all(iszero, Symbolics.simplify.(reference-k.K_ρ_symbolic))
            @test iszero(Symbolics.simplify(L1-outer.L))
            @test all(iszero, Symbolics.simplify.(G1-k.G_symbolic))
            @test length(inner.complementary_slackness) == 2
            @test length(outer.complementary_slackness) == 4
            y = 0.5 .+ 0.1sin.(1:k.variable_dimension)
            @test residual(k,y,1.0) == residual(k,y,1e-12)
            d = cos.(1:k.variable_dimension); d ./= norm(d)
            @test jacobian(k,y,1.0)*d ≈ (residual(k,y+1e-6d,1.0)-residual(k,y-1e-6d,1.0))/2e-6 atol=1e-7

            p3 = RG.ParametricGOOP(BlockArray(zeros(2),[2]),BlockArray(Float64[],[0]);
                preferences=[Function[(x,t)->x[1]^2/2;p.preferences[1]]],
                is_prioritized_constraint=[[false,false,false]],
                equality_constraints=p.equality_constraints,inequality_constraints=p.inequality_constraints)
            k3 = exact(p3)
            top,middle,bottom = k3.levels[1]
            x = k3.z_symbolic[k3.primal_dims]
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
                g-k3.z_symbolic[k3.interior_point_slack_dims]]
            @test k3.variable_dimension == k3.kkt_dimension == length(reference)
            @test all(iszero,Symbolics.simplify.(reference-k3.K_ρ_symbolic))
            y = 0.5 .+ 0.1sin.(1:k3.variable_dimension)
            d = cos.(1:k3.variable_dimension);d ./= norm(d)
            @test jacobian(k3,y,1.0)*d ≈ (residual(k3,y+1e-6d,1.0)-residual(k3,y-1e-6d,1.0))/2e-6 atol=1e-7
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
                @test k.variable_dimension == k.kkt_dimension == e.variable_dimension + extra
                @test length(k.nonnegative_dims) == length(e.nonnegative_dims) + extra
                # An exact point with its slacks on g; the perturbed point shares every other variable by name
                # (exact's σ_{i,1} is level 1's slacks here) and puts every level's slacks on g.
                index = Dict(string(v) => j for (j, v) in enumerate(k.z_symbolic))
                ye = 0.5 .+ 0.1sin.(1:e.variable_dimension)
                ye[e.interior_point_slack_dims] .= slack_values(e, ye)
                y = zeros(k.variable_dimension)
                for (j, v) in enumerate(e.z_symbolic)
                    y[index[string(v)]] = ye[j]
                end
                y[k.interior_point_slack_dims] .= slack_values(k, y)
                re, r0, r = residual(e, ye, 0.0), residual(k, y, 0.0), residual(k, y, 1e-3)
                # Every player's exact rows, then the slack rows of all players (exact: m_i, perturbed: K_i m_i each).
                n = sum(length(e.levels[i][1].F) for i in 1:case.problem.num_players)
                @test r0[1:n] ≈ re[1:n] atol=1e-12
                @test norm(r0[(n+1):end]) <= 1e-12
                @test (n + sum(m), n + sum(K .* m)) == (e.kkt_dimension, k.kkt_dimension)
                # rho shifts exactly the complementarity products, (2^K - 1) m_I per player here.
                @test count(!iszero, r - r0) ==
                      sum(length(level.complementary_slackness) for player in e.levels for level in player)
                @test all(≈(-1e-3), filter(!iszero, r - r0))
                d = cos.(1:k.variable_dimension); d ./= norm(d)
                @test jacobian(k,y,1e-3)*d ≈ (residual(k,y+1e-6d,1e-3)-residual(k,y-1e-6d,1e-3))/2e-6 atol=1e-7
                # Down a ρ chain from z = 0, one solve per stage, as experiments/MonteCarlo_RandomGOOP.jl does.
                θ, z, scale = zeros(k.parameter_dimension), zeros(length(case.expected)), sqrt(k.kkt_dimension)
                for ρ in (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
                    options = RG.ProjectedLMOptions(; ρ_schedule = [ρ], max_inner_iters = 1000, tol = 0.0,
                        tol_inner = min(ρ, 1e-2) * scale, proj_eps = 1e-8, η = 1e-2, η_min = 1e-8, η_max = 1e4)
                    z = RG.solve(RG.ProjectedLM(), k, θ; z₀ = z, options).z
                end
                @test norm(z[k.primal_dims] - case.expected) < 1e-5
                @test minimum(z[k.nonnegative_dims]) > 0
            end
        end

        @testset "Complete empty inequalities and preference bounds" begin
            for builder in BUILDERS
                for flags in ([false,false,false], [false,false,true], [true,false,true])
                    p = RG.ParametricGOOP(BlockArray([0.0],[1]),BlockArray(Float64[],[0]);
                        preferences=[Function[flag ? ((x,t)->[1-x[1]]) : ((x,t)->x[1]^2/2) for flag in flags]],
                        is_prioritized_constraint=[flags],equality_constraints=[nothing],inequality_constraints=[nothing])
                    k = builder(p)
                    y = zeros(k.variable_dimension)
                    for level in k.levels[1]
                        for v in level.preference_mu
                            y[findfirst(isequal(v),k.z_symbolic)] = 1.0
                        end
                    end
                    y[k.interior_point_slack_dims] .= slack_values(k,y)
                    @test norm(residual(k,y,0.0)) < 1e-12
                    @test all(isfinite, jacobian(k,0.5 .+ y,1e-3))
                end
            end
        end
    end

    @testset "1.4 Every system" begin
        "What all generators share: the sign box, the slack rows and the compiled code."

        @testset "the sign box and the slack rows of the reduced, complete and Scholtes systems" begin
            # Every inequality, the preference slacks' s ≥ 0 included, is a row expression − σ = 0 with
            # σ ≥ 0: the sign box (nonnegative_dims) is σ and the inequality duals, never s. For the complete
            # systems, slack_targets! evaluates the expressions (slack_expressions) the slacks σ stand for.
            p = RG.ParametricGOOP(BlockArray([0.0], [1]), BlockArray(Float64[], [0]);
                preferences = [Function[(x, t) -> [1 - x[1]], (x, t) -> x[1]^2 / 2, (x, t) -> [x[1] + 2]]],
                is_prioritized_constraint = [[true, false, true]], equality_constraints = [nothing],
                inequality_constraints = [(x, t) -> [3 - x[1]]])
            reduced = RG.generate_slacked_reduced_kkt_system(p)
            complete_systems = (RG.generate_slacked_complete_kkt_system(p; complementarity = :exact),
                RG.generate_slacked_complete_kkt_system(p; complementarity = :perturbed))
            for k in (reduced, complete_systems...)
                σ, s = k.z_symbolic[k.interior_point_slack_dims], k.z_symbolic[k.preference_slack_dims]
                @test k.nonnegative_dims == [k.interior_point_slack_dims; k.inequality_constraint_dual_dims]
                @test !isempty(s) && isempty(intersect(k.nonnegative_dims, k.preference_slack_dims))
                # s ≥ 0 has its slack rows: every preference slack minus a slack σ is a row of K_ρ.
                @test all(v -> any(w -> any(isequal(v - w), k.K_ρ_symbolic), σ), s)
            end
            @test isempty(reduced.slack_expressions)
            @test (reduced.kind, reduced.complementarity) == (:reduced, :interior_point)
            @test [(k.kind, k.complementarity) for k in complete_systems] == [(:complete, :exact), (:complete, :perturbed)]
            for k in complete_systems
                σ = k.z_symbolic[k.interior_point_slack_dims]
                @test length(k.slack_expressions) == length(σ)
                # Every slack row expression − σ is a row of K_ρ.
                @test all(((e, v),) -> any(isequal(e - v), k.K_ρ_symbolic), zip(k.slack_expressions, σ))
                y = 0.5 .+ 0.1 .* sin.(1:k.variable_dimension)
                G = zeros(length(σ))
                k.slack_targets!(G, y; θ = Float64[], ρ = 0.0)
                values = Dict(k.z_symbolic .=> y)
                @test G ≈ [Symbolics.value(Symbolics.substitute(e, values)) for e in k.slack_expressions]
            end
            # The Scholtes system's sign box is every complementarity pair's slack and multiplier (σ, γ; σ̂, ϕ) and
            # the relaxed pairs' u; its s ≥ 0, too, is a slack row.
            scholtes = RG.generate_slacked_scholtes_reduced_kkt_system(p)
            @test (scholtes.kind, scholtes.complementarity) == (:scholtes, :scholtes)
            pairs = scholtes.complementarity_pairs
            @test sort(scholtes.nonnegative_dims) == sort([[pair.slack for pair in pairs]; [pair.multiplier for pair in pairs];
                [pair.u_relaxation for pair in pairs if pair.u_relaxation != 0]])
            @test isempty(intersect(scholtes.nonnegative_dims, scholtes.preference_slack_dims))
            σ, s = scholtes.z_symbolic[scholtes.interior_point_slack_dims], scholtes.z_symbolic[scholtes.preference_slack_dims]
            @test all(v -> any(w -> any(isequal(v - w), scholtes.K_ρ_symbolic), σ), s)
        end

        @testset "Symbolics and FastDifferentiation code generation give the same residuals and Jacobians" begin
            # The :fast_differentiation codegen path must produce the same KKT residual
            # and Jacobian as the native Symbolics code generator: differentiation stays
            # in Symbolics and only the compiled evaluators change. The hierarchy below
            # includes an innermost prioritized inequality so its preference slacks (and
            # their rows) are exercised through both generators.
            num_players = 3
            kind = :quadratic
            primal_dims = fill(PRIMAL_DIM, num_players)
            parameter_dims = fill(1, num_players)
            x_template = BlockArray(zeros(sum(primal_dims)), primal_dims)
            θ_template = BlockArray(zeros(sum(parameter_dims)), parameter_dims)
            expected_blocks = [multiplayer_expected_block(player) for player in 1:num_players]
            raw_targets = [multiplayer_raw_target(player) for player in 1:num_players]
            coordinate_groups = preference_coordinate_groups(2)

            preferences = [
                Function[
                    objective_from_target(
                        kind,
                        player,
                        coordinate_groups[1],
                        raw_targets[player],
                    ),
                    objective_from_target(
                        kind,
                        player,
                        coordinate_groups[2],
                        raw_targets[player],
                    ),
                    multiplayer_constraint(kind, player, expected_blocks),
                ] for player in 1:num_players
            ]
            problem = ReducedGOOP.ParametricGOOP(
                x_template,
                θ_template;
                preferences,
                is_prioritized_constraint = [[false, false, true] for _ in 1:num_players],
                equality_constraints = fill(nothing, num_players),
                inequality_constraints = fill(nothing, num_players),
            )

            native = ReducedGOOP.generate_slacked_reduced_kkt_system(problem; codegen = :native)
            fdgen = ReducedGOOP.generate_slacked_reduced_kkt_system(
                problem;
                codegen = :fast_differentiation,
            )
            fdgen_chunked = ReducedGOOP.generate_slacked_reduced_kkt_system(
                problem;
                codegen = :fast_differentiation,
                fd_codegen_chunk_size = 7,
            )

            @test fdgen.kkt_dimension == native.kkt_dimension
            @test fdgen.variable_dimension == native.variable_dimension
            @test fdgen_chunked.kkt_dimension == native.kkt_dimension
            @test fdgen_chunked.variable_dimension == native.variable_dimension

            reference_symbolic_jacobian = ReducedGOOP.SymbolicTracingUtils.sparse_jacobian(
                native.K_ρ_symbolic,
                native.z_symbolic,
            )
            optimized_symbolic_jacobian =
                ReducedGOOP._build_symbolics_sparse_jacobian(native.K_ρ_symbolic, native.z_symbolic)
            @test isequal(reference_symbolic_jacobian, optimized_symbolic_jacobian)

            n = native.variable_dimension
            m = sum(problem.parameter_dims)
            for scale in (0.3, 1.7), ρ in (0.1, 0.01)
                z = scale .* sin.(1:n)
                θ = scale .* cos.(1:m)

                F_native = zeros(native.kkt_dimension)
                F_fd = zeros(fdgen.kkt_dimension)
                native.K_ρ!(F_native, z; θ, ρ)
                fdgen.K_ρ!(F_fd, z; θ, ρ)
                @test isapprox(F_native, F_fd; atol = 1e-12, rtol = 1e-12)

                J_native = copy(native.∇K_ρ!.result_buffer)
                J_fd = copy(fdgen.∇K_ρ!.result_buffer)
                F_fd_chunked = zeros(fdgen_chunked.kkt_dimension)
                fdgen_chunked.K_ρ!(F_fd_chunked, z; θ, ρ)
                @test isapprox(F_native, F_fd_chunked; atol = 1e-12, rtol = 1e-12)

                J_fd_chunked = copy(fdgen_chunked.∇K_ρ!.result_buffer)
                native.∇K_ρ!(J_native, z; θ, ρ)
                fdgen.∇K_ρ!(J_fd, z; θ, ρ)
                fdgen_chunked.∇K_ρ!(J_fd_chunked, z; θ, ρ)
                @test isapprox(J_native, J_fd; atol = 1e-12, rtol = 1e-12)
                @test isapprox(J_native, J_fd_chunked; atol = 1e-12, rtol = 1e-12)
            end
        end
    end
end

@testset "2. Solver mechanics: what PDIP and ProjectedLM do at each step" begin
    "The Newton step, the line searches, ProjectedLM's η, stages and termination, the reported residuals, the starts and options each solver accepts, and warm starts, on small problems."

    @testset "the Newton step of every linear solver" begin
        # δz with J δz = −K_ρ: `:svd`, and `:sparseQR_Jᵀ` when J has full row rank, take the minimum-norm step −J⁺K_ρ;
        # `:klu`, `:ldl` and `:normal` the Levenberg–Marquardt step −Jᵀ(JJᵀ + η²I)⁻¹F = −V diag(s / (s² + η²)) UᵀF
        # (J = U diag(s) Vᵀ); `:sparseQR_J` a least-squares step. At a point of each kind of system.
        case = build_benchmark_problem(; num_players = 1, levels = 3, kind = :quadratic)
        systems = (RG.generate_slacked_reduced_kkt_system(case.problem),
            RG.generate_slacked_complete_kkt_system(case.problem; complementarity = :perturbed),
            RG.generate_slacked_scholtes_reduced_kkt_system(case.problem))
        for k in systems
            θ, ρ, η = zeros(k.parameter_dimension), 1e-2, 1e-3
            z = k.kind === :scholtes ? RG.initialize_scholtes!(zeros(k.variable_dimension), k, θ, case.expected, ρ) :
                0.5 .+ 0.1 .* sin.(1:(k.variable_dimension))
            K_ρ = zeros(k.kkt_dimension)
            k.K_ρ!(K_ρ, z; θ, ρ)
            J = copy(k.∇K_ρ!.result_buffer)
            k.∇K_ρ!(J, z; θ, ρ)
            U, s, V = svd(Matrix(J))
            minimum_norm = -pinv(Matrix(J)) * K_ρ
            levenberg_marquardt = -V * (s ./ (s .^ 2 .+ η^2) .* (U' * K_ρ))
            δz = zeros(k.variable_dimension)
            @test RG._newton_step!(δz, J, K_ρ, :svd, η) == (; fallback = 0, η)
            @test δz == minimum_norm
            for linear_solver in (:klu, :ldl, :normal)
                @test RG._newton_step!(δz, J, K_ρ, linear_solver, η) == (; fallback = 0, η)
                @test δz ≈ levenberg_marquardt rtol = 1e-6
            end
            step = RG._newton_step!(δz, J, K_ρ, :sparseQR_Jᵀ, η)
            if rank(Matrix(J)) == k.kkt_dimension
                @test step.fallback == 0
                @test δz ≈ minimum_norm rtol = 1e-8
            else
                @test step.fallback == 1 # J without full row rank: the :svd step
                @test δz == minimum_norm
            end
            @test RG._newton_step!(δz, J, K_ρ, :sparseQR_J, η) == (; fallback = 0, η)
            @test norm(J' * (J * δz + K_ρ)) <= 1e-8 * opnorm(Matrix(J))^2 * norm(K_ρ) # the normal equations
            @test_throws ArgumentError RG._newton_step!(δz, J, K_ρ, :lu, η)
        end
        # :normal retries with 100 η when CHOLMOD cannot factorize JJᵀ + η²I. Below η ≈ 2e-162, η² underflows to 0
        # and the singular JJᵀ is factorized as it is: from η = 1e-170, the last of the six tries, at 1e-160,
        # succeeds. A zero J fails every try and takes the :svd step, reporting the η it was given.
        δz = zeros(2)
        step = RG._newton_step!(δz, sparse([1.0 0.0; 0.0 0.0]), [1.0, 1.0], :normal, 1e-170)
        @test step.fallback == 0
        @test step.η ≈ 1e-160
        @test δz == [-1.0, 0.0]
        δz = zeros(1)
        @test RG._newton_step!(δz, spzeros(1, 1), [1.0], :normal, 1e-200) == (; fallback = 1, η = 1e-200)
        @test δz == [0.0]
    end

    @testset "PDIP's fraction-to-the-boundary line search and the sparse QR steps" begin
        # On the reduced system (wide) and the complete one (square, and singular along the solve), whose J is
        # rank-deficient: :sparseQR_Jᵀ then takes the :svd step (a fallback), :sparseQR_J a least-squares step.
        goop = coupled_problem(d -> cosh(d) - 1)
        for k in (RG.generate_slacked_reduced_kkt_system(goop),
                  RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed)),
            (options, fallbacks) in (((; linesearch = :fraction_to_boundary), false),
                                     ((; linear_solver = :sparseQR_J), false),
                                     ((; linear_solver = :sparseQR_Jᵀ), true))
            outputs = pdip_chain(k; options...)
            z = last(outputs).z
            @test all(o -> o.termination_reason === :inner_tolerance_reached, outputs)
            @test any(o -> o.svd_fallbacks > 0, outputs) == fallbacks
            @test norm(z[k.primal_dims] - COUPLED_EXPECTED) < 1e-8
            @test minimum(z[k.nonnegative_dims]) > 0
        end
    end

    @testset "ProjectedLM's gain ratio and η schedule" begin
        rates = (; gain_low = 0.25, gain_high = 0.75, η_min = 1e-8, η_max = 1e2, η_tightening_rate = 1.2,
            η_loosening_rate = 3.0)
        up, down = 1 + exp(-3.0), 1 - exp(-1.2)
        next_η(η, gain_ratio, α) = RG._scheduled_η(η, gain_ratio, α; rates...)
        @test next_η(1.0, 0.25, 1.0) ≈ up # a poor step
        @test next_η(1.0, 0.75, 1.0) == 1.0
        @test next_η(1.0, 0.8, 1.0) ≈ down # a good one
        @test next_η(1.0, 1.0, 0.5) ≈ up # a shortened one
        @test next_η(1.0, 1.0, 0.99) ≈ down
        @test next_η(1.0, -Inf, 0.0) ≈ up # a failed line search
        @test next_η(1e2, -1.0, 1.0) == 1e2 # within [η_min, η_max]
        @test next_η(1e-8, 1.0, 1.0) == 1e-8
        # The actual over the predicted decrease of ½‖K_ρ‖²: 1 for a step that lands where the linear model K_ρ + JΔz
        # says, −Inf when the model predicts no decrease.
        @test RG._gain_ratio([1.0], [-1.0], 1.0, 0.0) == 1.0
        @test RG._gain_ratio([1.0], [-1.0], 1.0, 0.5) == 0.75
        @test RG._gain_ratio([1.0], [0.0], 1.0, 1.0) == -Inf
    end

    @testset "ProjectedLM's η along a solve, and a solve continued from its η" begin
        k = RG.generate_slacked_reduced_kkt_system(free_problem(x -> x^2))
        down = 1 - exp(-1.2)
        for linear_solver in (:normal, :klu, :ldl)
            options = RG.ProjectedLMOptions(; ρ_schedule = [1.0, 0.1], max_inner_iters = 1, tol = 0.0, tol_inner = 0.0,
                linear_solver, η = 0.1, record_convergence = true)
            r = RG.solve(RG.ProjectedLM(), k, Float64[]; z₀ = [1.0], options)
            # K_ρ = 2x is linear: every full step lands where the model says (gain ratio 1), and η shrinks.
            @test r.step_size_history == [1.0, 1.0]
            @test r.η_history ≈ [0.1, 0.1down]
            @test r.η ≈ 0.1down^2
            # A solve continues from the returned η.
            options = RG.ProjectedLMOptions(; ρ_schedule = [0.01], max_inner_iters = 1, tol = 0.0, tol_inner = 0.0,
                linear_solver, η = r.η, record_convergence = true)
            @test only(RG.solve(RG.ProjectedLM(), k, Float64[]; z₀ = r.z, options).η_history) == r.η
        end
    end

    @testset "the stages and termination reasons of both solvers" begin
        quadratic = RG.generate_slacked_reduced_kkt_system(free_problem(x -> x^2))
        # K_ρ = atan(x): from x = 2 the full Newton step overshoots, to |atan(x)| larger than at x = 2.
        overshoot = RG.generate_slacked_reduced_kkt_system(free_problem(x -> x * atan(x) - log(1 + x^2) / 2))
        for (solver, Options) in SOLVERS
            solve_from(k, z₀; kwargs...) =
                RG.solve(solver, k, Float64[]; z₀, options = Options(; ρ_schedule = [1e-3], kwargs...))
            r = solve_from(quadratic, [1.0]; tol = Inf)
            @test (r.status, r.termination_reason, r.newton_steps) == (:solved, :tolerance_reached, 0)
            r = solve_from(quadratic, [1.0]; tol = 0.0, tol_inner = Inf)
            @test (r.status, r.termination_reason, r.newton_steps) == (:failed, :inner_tolerance_reached, 0)
            r = solve_from(quadratic, [1.0]; max_inner_iters = 0, tol = 0.0, record_convergence = true)
            @test (r.termination_reason, r.newton_steps) == (:max_inner_iters, 0)
            @test isempty(r.norm_K_ρ_history)
            r = solve_from(quadratic, [1.0]; max_inner_iters = 1, tol = 0.0, tol_inner = 0.0, record_convergence = true)
            @test (r.termination_reason, r.newton_steps) == (:max_inner_iters, 1)
            @test r.norm_K_ρ < only(r.norm_K_ρ_history)
            # The returned iterate is judged again: a last step can meet tol after the last test before a step.
            r = solve_from(quadratic, [1.0]; max_inner_iters = 1, tol = 1e-6, tol_inner = 0.0)
            @test (r.status, r.termination_reason) == (:solved, :tolerance_reached)
            r = solve_from(quadratic, [NaN])
            @test (r.status, r.termination_reason) == (:failed, :nonfinite_residual)
            @test isnan(r.norm_K_ρ)
            # With min_step = 0.9 the line search tries α = 1 only, and the overshooting step fails it.
            r = solve_from(overshoot, [2.0]; min_step = 0.9, tol = 0.0, tol_inner = 0.0)
            @test (r.status, r.termination_reason, r.newton_steps) == (:failed, :line_search_failed, 0)
            @test r.z == [2.0]
        end
    end

    @testset "both solvers report the residuals of the point they return" begin
        goop = coupled_problem(d -> cosh(d) - 1)
        reduced = RG.generate_slacked_reduced_kkt_system(goop)
        complete = RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed)
        scholtes = RG.generate_slacked_scholtes_reduced_kkt_system(goop)
        θ = Float64[]
        runs = [(RG.PDIP(), RG.PDIPOptions, reduced, zeros(4)), (RG.PDIP(), RG.PDIPOptions, complete, zeros(4)),
                (RG.ProjectedLM(), RG.ProjectedLMOptions, reduced, zeros(4)),
                (RG.ProjectedLM(), RG.ProjectedLMOptions, complete, zeros(4)),
                (RG.ProjectedLM(), RG.ProjectedLMOptions, scholtes, scholtes_start(scholtes, zeros(4), 1.0; θ))]
        for (solver, Options, k, z₀) in runs, tol in (1e-6, 0.0)
            r = RG.solve(solver, k, θ; z₀, options = Options(; ρ_schedule = [1.0, 1e-2, 1e-4], max_inner_iters = 30, tol))
            K = zeros(k.kkt_dimension)
            k.K₀!(K, r.z; θ)
            @test r.norm_K₀ == norm(K)
            k.K_ρ!(K, r.z; θ, r.ρ)
            @test r.norm_K_ρ == norm(K)
            @test r.x == r.z[k.primal_dims]
            @test r.status === (r.norm_K₀ < tol ? :solved : :failed)
            @test minimum(r.z[k.nonnegative_dims]) > 0
        end
    end

    @testset "the starts each solver accepts, and the inputs and options it refuses" begin
        case = build_benchmark_problem(; num_players = 1, levels = 2, kind = :quadratic)
        k = RG.generate_slacked_reduced_kkt_system(case.problem)
        θ = zeros(k.parameter_dimension)
        for (solver, Options) in SOLVERS
            # A primal start: s, σ and γ at 1, every other variable at 0.
            z = RG.solve(solver, k, θ; z₀ = case.expected, options = Options(; max_inner_iters = 0)).z
            @test z[k.primal_dims] == case.expected
            ones_dims = [k.preference_slack_dims; k.interior_point_slack_dims; k.inequality_constraint_dual_dims]
            @test all(==(1.0), z[ones_dims])
            @test all(iszero, z[setdiff(1:(k.variable_dimension), [k.primal_dims; ones_dims])])
            # A start of neither length, or not positive on nonnegative_dims (σ and γ), is rejected.
            @test_throws ArgumentError RG.solve(solver, k, θ; z₀ = zeros(k.variable_dimension + 1))
            bad = copy(z)
            bad[first(k.nonnegative_dims)] = 0.0
            @test_throws ArgumentError RG.solve(solver, k, θ; z₀ = bad)
            @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; ρ_schedule = Float64[]))
            @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; β = 1.0))
            @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; min_step = 0.0))
            @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; η = 0.0))
            @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; linear_solver = :lu))
        end
        for options in (RG.PDIPOptions(; linesearch = :exact), RG.PDIPOptions(; linear_solver = :normal))
            @test_throws ArgumentError RG.solve(RG.PDIP(), k, θ; options)
        end
        for options in (RG.ProjectedLMOptions(; linear_solver = :svd), RG.ProjectedLMOptions(; η = Inf),
                        RG.ProjectedLMOptions(; η_min = 0.0), RG.ProjectedLMOptions(; η_min = 2.0, η_max = 1.0),
                        RG.ProjectedLMOptions(; gain_low = 0.8, gain_high = 0.5),
                        RG.ProjectedLMOptions(; gain_low = -0.1), RG.ProjectedLMOptions(; τ = 0.0),
                        RG.ProjectedLMOptions(; τ = 1.0), RG.ProjectedLMOptions(; proj_eps = -1.0),
                        RG.ProjectedLMOptions(; c = 0.0), RG.ProjectedLMOptions(; c = 1.0))
            @test_throws ArgumentError RG.solve(RG.ProjectedLM(), k, θ; options)
        end
        # Algorithm 5.1 has no rule for the Scholtes system's exact products σ̂ ⊙ ϕ = 0.
        scholtes = RG.generate_slacked_scholtes_reduced_kkt_system(case.problem)
        @test_throws ArgumentError RG.solve(RG.PDIP(), scholtes, θ; z₀ = scholtes_start(scholtes, case.expected, 1.0; θ))
        # Nor for the exact complete system's products σ ⊙ γ = 0.
        @test_throws ArgumentError RG.solve(RG.PDIP(), exact(case.problem), θ; z₀ = case.expected)
    end

    @testset "PDIP continued from its previous full solution converges at once" begin
        problem = build_benchmark_problem(; num_players = 3, levels = 2, kind = :quadratic)
        kkt = reduced_kkt_system(problem.problem)
        θ = zeros(sum(problem.problem.parameter_dims))
        expected = reduce(vcat, [multiplayer_expected_block(player) for player in 1:3])
        options = default_pdip_options()

        cold = ReducedGOOP.solve(ReducedGOOP.PDIP(), kkt, θ; z₀ = expected, options)
        @test cold.status === :solved

        # Resolving from the previous full solution (primals + duals + slacks) must
        # converge almost immediately.
        warm = ReducedGOOP.solve(ReducedGOOP.PDIP(), kkt, θ; z₀ = cold.z, options)
        @test warm.status === :solved
        @test warm.newton_steps <= cold.newton_steps
        @test warm.newton_steps <= 14
        @test isapprox(warm.x, expected; atol = 1e-6)
    end
end

@testset "3. Solutions: each system, solved by its solver, reaches the right answer" begin
    "Each system with the solver that suits it, on problems with known answers (the table at the top of this file)."

    @testset "3.1 Reduced (5.2) and quasi systems by PDIP" begin
        "The benchmark families with known solutions and active sets."

        @testset "known solutions and active sets: one or three players, one to three levels" begin
            #=
            Single-Player Box-Constrained Benchmark Family
            ----------------------------------------------

            Decision variable:

            x in R^5

            Known solution:

            x* = [0.0, 0.6, 1.0, 0.4, 1.4]

            Raw objective target:

            t = [-0.4, 0.6, 1.4, 0.4, 1.4]

            Constraints:

            Quadratic/linear variant:
             x - lower >= 0
             upper - x >= 0

            Nonlinear/nonlinear variant:
             exp.(x - lower) - 1 >= 0
             exp.(upper - x) - 1 >= 0

            lower = [0, 0, 0, 0, 0]
            upper = [2, 2, 1, 2, 2]

            Intentional active set at x*:

            active:   x_1 = lower_1 = 0,  x_3 = upper_3 = 1
            inactive: all other lower and upper bounds

            The expected solution is the projection of t onto the box constraints.
            =#
            @testset "Single-Player Problems" begin
                @testset "Single-Level" begin
                    @testset "Quadratic Objective, Linear Constraints" begin
                        test_known_solution_case(; num_players = 1, levels = 1, kind = :quadratic)
                    end

                    @testset "Nonlinear Objective, Nonlinear Constraints" begin
                        test_known_solution_case(; num_players = 1, levels = 1, kind = :nonlinear)
                    end
                end
            end

            #=
            Three-Player Coupled Generalized Nash Benchmark Family
            -----------------------------------------------------

            Decision variables:

            x_i in R^5,  i = 1,2,3

            Known solution:

            x_1* = [0.40, 0.90, 0.55, 1.20, 0.80]
            x_2* = [0.45, 1.00, 0.60, 1.30, 0.90]
            x_3* = [0.50, 1.10, 0.65, 1.40, 1.00]

            Player i's raw objective target is:

            t_i = x_i* + [0.6, 0, -0.4, 0, 0]

            So each player would prefer coordinate 1 to be larger and coordinate 3 to be
            smaller than the final solution. The coupled constraints create the active
            resources that force the equilibrium to x*.

            Coupled resource expression:

            c_{i,j}(x) = x_{i,j} + 0.25 * sum(x_{k,j} for k != i)

            Player i constraints:

            C_{i,1} - c_{i,1}(x) >= 0    active upper resource
            C_{i,2} - c_{i,2}(x) >= 0    inactive upper resource, residual slack 1
            c_{i,3}(x) - F_{i,3} >= 0    active lower resource
            c_{i,4}(x) - F_{i,4} >= 0    inactive lower resource, residual slack 1

            where:

            C_{i,1} = c_{i,1}(x*)
            C_{i,2} = c_{i,2}(x*) + 1
            F_{i,3} = c_{i,3}(x*)
            F_{i,4} = c_{i,4}(x*) - 1

            For the nonlinear constraint transform, these inactive constraints evaluate
            to exp(1) - 1, although the underlying affine residual slack is 1.

            This is a true generalized Nash setup: every player's feasible set depends
            on all players' decision variables. The active coordinate-1 and coordinate-3
            resource equations are player-specific and jointly nonsingular, so the known
            solution fixes the distribution across players, not just aggregate totals.

            Each player's single-level objective covers coordinates {1,2,3,4,5}.
            =#
            @testset "Three-Player Problems" begin

                # Testing Reduced KKT system 

                @testset "Single-Level" begin
                    @testset "Reduced GOOP, Quadratic Objective, Linear Constraints" begin
                        test_known_solution_case(; num_players = 3, levels = 1, kind = :quadratic)
                    end

                    @testset "Reduced GOOP, Nonlinear Objective, Nonlinear Constraints" begin
                        test_known_solution_case(; num_players = 3, levels = 1, kind = :nonlinear)
                    end
                end

                @testset "Bi-Level" begin
                    @testset "Reduced GOOP, Quadratic Objective, Linear Constraints" begin
                        test_known_solution_case(; num_players = 3, levels = 2, kind = :quadratic)
                    end

                    @testset "Reduced GOOP, Nonlinear Objective, Nonlinear Constraints" begin
                        test_known_solution_case(;
                            num_players = 3,
                            levels = 2,
                            kind = :nonlinear,
                            atol = 5e-7,
                        )
                    end
                end

                @testset "Three-Level" begin
                    @testset "Reduced GOOP, Quadratic Objective, Linear Constraints" begin
                        test_known_solution_case(; num_players = 3, levels = 3, kind = :quadratic)
                    end

                    @testset "Reduced GOOP, Nonlinear Objective, Nonlinear Constraints" begin
                        test_known_solution_case(;
                            num_players = 3,
                            levels = 3,
                            kind = :nonlinear,
                            atol = 5e-7,
                        )
                    end
                end

                # The quasi system where it differs from the reduced one: nonlinear data at two and three levels (on
                # quadratic data and at one level it is the reduced system; see 1.1).

                @testset "Bi-Level" begin
                    @testset "Quasi GOOP, Nonlinear Objective, Nonlinear Constraints" begin
                        test_known_solution_case(;
                            num_players = 3,
                            levels = 2,
                            kind = :nonlinear,
                            atol = 5e-7,
                            kkt_system = quasi_kkt_system
                        )
                    end
                end

                @testset "Three-Level" begin
                    @testset "Quasi GOOP, Nonlinear Objective, Nonlinear Constraints" begin
                        test_known_solution_case(;
                            num_players = 3,
                            levels = 3,
                            kind = :nonlinear,
                            atol = 5e-7,
                            kkt_system = quasi_kkt_system
                        )
                    end
                end
            end
        end
    end

    @testset "3.2 Reduced (5.2) and :perturbed complete systems on the coupled game" begin
        "The two-player, three-level game with a unique equilibrium and a rank-deficient J."

        @testset "PDIP on both systems with every linear solver: the equilibrium, every step lowering ‖K_ρ‖ strictly" begin
            for innermost in (d -> d^2 / 2, d -> cosh(d) - 1)
                goop = coupled_problem(innermost)
                for k in (RG.generate_slacked_reduced_kkt_system(goop),
                          RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed))
                    θ = zeros(k.parameter_dimension)
                    # J is rank-deficient at the start (s, σ and γ at 1, x = 0), and :svd still takes −J⁺K_ρ.
                    start = RG.solve(RG.PDIP(), k, θ; z₀ = zeros(4), options = RG.PDIPOptions(; max_inner_iters = 0)).z
                    J = copy(k.∇K_ρ!.result_buffer)
                    k.∇K_ρ!(J, start; θ, ρ = 1.0)
                    @test rank(Matrix(J)) < k.kkt_dimension
                    for linear_solver in (:svd, :klu, :ldl)
                        outputs = pdip_chain(k; linear_solver)
                        z = last(outputs).z
                        @test all(o -> o.termination_reason === :inner_tolerance_reached, outputs)
                        @test all(o -> o.svd_fallbacks == 0, outputs)
                        # Every accepted step lowers ‖K_ρ(z)‖ strictly.
                        @test all(o -> all(<(0), diff(o.norm_K_ρ_history)), outputs)
                        @test norm(z[k.primal_dims] - COUPLED_EXPECTED) < 1e-8
                        @test minimum(z[k.nonnegative_dims]) > 0
                    end
                    # The ‖K₀‖ < tol stop, checked before every Newton step: with tol far above ‖K₀‖, a solve stops
                    # before its first Newton step.
                    for linear_solver in (:svd, :klu)
                        options = RG.PDIPOptions(; ρ_schedule = [1.0], tol = 1e10, linear_solver)
                        output = RG.solve(RG.PDIP(), k, θ; z₀ = zeros(4), options)
                        @test output.newton_steps == 0
                        @test output.termination_reason === :tolerance_reached
                    end
                end
            end
        end

        @testset "ProjectedLM on the reduced system down a ρ chain" begin
            # One solve per stage from z = 0, as the intersection study's reduced solves. The 3-player nonlinear case
            # levels off at ‖K₀‖/√m ≈ 1e-8 with a primal error of ≈ 4e-6; a wiring error would miss by O(1).
            for (levels, players, kind) in ((2, 1, :quadratic), (3, 3, :nonlinear))
                case = build_benchmark_problem(; num_players = players, levels, kind)
                k = RG.generate_slacked_reduced_kkt_system(case.problem)
                θ, z, scale = zeros(k.parameter_dimension), zeros(length(case.expected)), sqrt(k.kkt_dimension)
                for ρ in (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
                    options = RG.ProjectedLMOptions(; ρ_schedule = [ρ], max_inner_iters = 1000, tol = 0.0,
                        tol_inner = min(ρ, 1e-2) * scale, proj_eps = 1e-8, η = 1e-2, η_min = 1e-8, η_max = 1e4)
                    z = RG.solve(RG.ProjectedLM(), k, θ; z₀ = z, options).z
                end
                @test norm(z[k.primal_dims] - case.expected) < 1e-5
                @test minimum(z[k.nonnegative_dims]) > 0
            end
        end
    end

    @testset "3.3 Scholtes system by ProjectedLM" begin
        "ScholtesReducedGOOP.jl's small problems: inactive and active constraints, a coupled game, the ρ schedule, and the quasi form."

        @testset "every linear solver: an inactive constraint and a coupled game, inside the sign box with ‖K_ρ‖ falling" begin
            # Every iterate is positive on nonnegative_dims (σ, γ, ϕ, σ̂, u), and every accepted step lowers ‖K_ρ(z)‖
            # strictly within a stage.
            for prob in (prob_A, prob_single), linear_solver in (:normal, :klu, :ldl)
                k = scholtes_kkt(prob)
                r = plm_solve(k, prob.z0; linear_solver, record_convergence = true)
                @test r.status === :solved
                @test minimum(r.z[k.nonnegative_dims]) > 0
                history, ρs = r.norm_K_ρ_history, r.ρ_history
                @test all(history[i + 1] < history[i] for i in 1:(length(history) - 1) if ρs[i + 1] == ρs[i])
                if prob === prob_A
                    @test r.x[1] ≈ 0.0 atol = 1e-6
                    @test r.svd_fallbacks == 0
                end
            end
        end

        @testset "active constraints and a coupled game: the solutions, exact inequalities at them, and a warm restart" begin
            # Active constraints (prob_B, prob_D's and prob_E's box) end ON their bound, not past it. prob_F is left
            # out: it does not converge under the default options, in the source either (‖K₀‖ ≈ 4e-4).
            solutions = map((prob_B, prob_C, prob_D, prob_E)) do prob
                k = scholtes_kkt(prob)
                r = plm_solve(k, prob.z0; record_convergence = true)
                @test r.status === :solved
                @test sfeasible(prob, r.x).worst >= -1e-8
                @test minimum(r.z[k.nonnegative_dims]) > 0
                history, ρs = r.norm_K_ρ_history, r.ρ_history
                @test all(history[i + 1] < history[i] for i in 1:(length(history) - 1) if ρs[i + 1] == ρs[i])
                (; k, r)
            end
            @test abs(solutions[1].r.x[1]) < 1e-5
            @test solutions[2].r.x ≈ [0.8, -0.8] atol = 1e-5
            # Restarted from its own solution at its last stage, prob_C's solve stays there.
            (; k, r) = solutions[2]
            warm = RG.solve(RG.ProjectedLM(), k, θ0; z₀ = r.z, options = RG.ProjectedLMOptions(; ρ_schedule = [r.ρ], η = r.η))
            @test warm.status === :solved
            @test warm.x ≈ r.x atol = 1e-8
            @test warm.newton_steps <= r.newton_steps
        end

        @testset "the ρ schedule: the returned ρ is one of its stages, the last when only ‖K_ρ‖ ends a stage" begin
            k = scholtes_kkt(prob_C)
            r = plm_solve(k, prob_C.z0; ρ_schedule = [1.0, 1e-3], max_inner_iters = 40, record_convergence = true)
            @test r.ρ in (1.0, 1e-3)
            @test issubset(r.ρ_history, [1.0, 1e-3])
            @test minimum(r.z[k.nonnegative_dims]) > 0
            sched = [1.0, 1e-2, 1e-4]
            for prob in (prob_A, prob_C, prob_D)
                k = scholtes_kkt(prob)
                r = plm_solve(k, prob.z0; ρ_schedule = sched, tol = 0.0)
                @test r.ρ == sched[end]
                @test minimum(r.z[k.nonnegative_dims]) > 0
            end
            rc = plm_solve(prob_A; ρ_schedule = sched)
            @test rc.status === :solved && rc.ρ in sched
        end

        @testset "quasi == reduced on quadratic objectives over affine constraints" begin
            Random.seed!(5)
            for prob in (prob_A, prob_B, prob_C, prob_D, prob_E, prob_F)
                ke, kq = scholtes_kkt(prob; quasi = false), scholtes_kkt(prob; quasi = true)
                @test kq.kkt_dimension == ke.kkt_dimension
                worst = 0.0
                for _ in 1:25
                    z = randn(ke.variable_dimension)
                    worst = max(worst, maximum(abs, residual(kq, z, 0.3) .- residual(ke, z, 0.3)),
                                maximum(abs, targets(kq, z, 0.3) .- targets(ke, z, 0.3); init = 0.0))
                end
                @test worst == 0.0
                @test findnz(kq.∇K_ρ!.result_buffer)[1:2] == findnz(ke.∇K_ρ!.result_buffer)[1:2]
            end
            for prob in (prob_E, prob_F)
                re = plm_solve(scholtes_kkt(prob; quasi = false), prob.z0)
                rq = plm_solve(scholtes_kkt(prob; quasi = true), prob.z0)
                @test rq.z == re.z
                @test rq.newton_steps == re.newton_steps
            end
        end

        @testset "quasi truncates for real on non-quadratic data" begin
            ke, kq = scholtes_kkt(prob_G; quasi = false), scholtes_kkt(prob_G; quasi = true)
            Random.seed!(6)
            z = randn(ke.variable_dimension)
            @test !isapprox(residual(kq, z, 0.3), residual(ke, z, 0.3); atol = 1e-8)
            @test nnz(kq.∇K_ρ!.result_buffer) < nnz(ke.∇K_ρ!.result_buffer)
            re, rq = plm_solve(ke, prob_G.z0), plm_solve(kq, prob_G.z0)
            @test re.status === :solved && rq.status === :solved
            @test minimum(rq.z[kq.nonnegative_dims]) > 0
            @test rq.x ≈ prob_G_z atol = 1e-5
            @test rq.x ≈ re.x atol = 1e-6
        end

        @testset "quasi composes with explicit Scholtes complementarity" begin
            k = scholtes_kkt(prob_E; quasi = true)
            r = plm_solve(k, prob_E.z0)
            @test r.status === :solved && minimum(r.z[k.nonnegative_dims]) > 0
            @test r.x[1] ≈ 0.5 atol = 1e-5
            @test r.x[1] + r.x[2] ≈ 1.0 atol = 1e-5
        end
    end

    @testset "3.4 Complete systems by ProjectedLM" begin
        "The exact complete system's solutions, and the complete system against the reduced one."

        @testset "ProjectedLM solves the exact complete system" begin
            # The perturbed system is solved down a ρ chain in its own testset above.
            options = RG.ProjectedLMOptions(;ρ_schedule=[1e-9],max_inner_iters=500,tol=1e-7,tol_inner=1e-10,
                η=1e-4,η_max=1e4,proj_eps=1e-8)
            for levels in (2,3), players in (1,3), kind in (:quadratic,:nonlinear)
                case = build_benchmark_problem(;num_players=players,levels,kind)
                k = exact(case.problem)
                @test k.kkt_dimension == k.variable_dimension
                output = RG.solve(RG.ProjectedLM(),k,zeros(k.parameter_dimension);z₀=case.expected,options)
                @test output.status === :solved
                @test norm(residual(k,output.z,1e-9)) < 1e-7
                @test minimum(output.z[k.nonnegative_dims]) > 0
                @test output.x ≈ case.expected atol=1e-6 rtol=0
                @test RG.is_feasible(case.problem,output.x,zeros(k.parameter_dimension);tol=1e-7).feasible
                @test output.newton_steps > 0
            end
            case = build_benchmark_problem(;num_players=1,levels=2,kind=:quadratic)
            k = exact(case.problem)
            # A start outside the sign box is rejected.
            output = RG.solve(RG.ProjectedLM(),k,zeros(k.parameter_dimension);z₀=case.expected,options)
            bad = copy(output.z);bad[first(k.nonnegative_dims)] = -1
            @test_throws ArgumentError RG.solve(RG.ProjectedLM(),k,zeros(k.parameter_dimension);z₀=bad,options)
        end

        @testset "the complete and reduced systems find the same unconstrained solution" for levels in (2, 3)
            case = build_unconstrained_quadratic_problem(; levels)

            kkt = exact(case.problem)
            complete = RG.solve(RG.ProjectedLM(), kkt, zeros(kkt.parameter_dimension);
                z₀ = case.z₀,
                options = RG.ProjectedLMOptions(; ρ_schedule = [1e-9], tol = 1e-9,
                    tol_inner = 1e-11, max_inner_iters = 500, η = 1e-6))
            reduced = solve_with_pdip(
                case.problem,
                case.expected;
                z₀ = case.z₀,
                kkt_system = reduced_kkt_system,
            )

            @test complete.status === :solved
            @test reduced.output.status === :solved
            @test complete.norm_K₀ <= 1e-8
            @test reduced.output.norm_K₀ <= 1e-8
            @test isapprox(complete.x, case.expected; atol = IP_ATOL, rtol = 0.0)
            @test isapprox(reduced.primals, case.expected; atol = IP_ATOL, rtol = 0.0)
            @test isapprox(reduced.primals, complete.x; atol = IP_ATOL, rtol = 0.0)
        end
    end
end

@testset "4. Multistart: solve_multi and the start samplers" begin
    "solve_multi on a non-unique answer and from ball_starts, and strictly feasible random starts."

    @testset "non-unique answers: solve_multi recovers several points of the set" begin
        # prob_D's answer is the segment z₁ + z₂ = 1 in the box: primal starts on a grid (each started by
        # initialize_scholtes!) land at different points of it.
        k = scholtes_kkt(prob_D)
        starts = RG.grid_starts(prob_D.goop, θ0, -1.5:0.75:1.5, -1.5:0.75:1.5)
        @test length(starts) >= 20
        S = RG.solve_multi(RG.ProjectedLM(), k, θ0, starts)
        @test S.rejected == 0
        @test length(S.points) > 1
        @test maximum(seg_dist(r.x) for r in S.results) < 1e-5
        ts = [seg_t(p) for p in S.points]
        @test maximum(ts) - minimum(ts) > 0.5
        @test all(minimum(r.z[k.nonnegative_dims]) > 0 for r in S.results)
        @test length(S.assignment) == length(S.results)
        @test maximum(S.assignment) == length(S.points)
    end

    @testset "solve_multi from ball_starts: a fixed-ρ study around a solution" begin
        goop = coupled_problem(d -> cosh(d) - 1)
        k = RG.generate_slacked_reduced_kkt_system(goop)
        θ = zeros(k.parameter_dimension)
        options = RG.PDIPOptions(; ρ_schedule = [0.1], max_inner_iters = 1000, tol = 0.0, tol_inner = 1e-11,
            linear_solver = :klu, record_convergence = true)
        # Held at one ρ, a solve never meets the ‖K₀‖ test, so it is accepted on its inner tolerance.
        accept(r) = r.termination_reason === :inner_tolerance_reached
        center = RG.solve(RG.PDIP(), k, θ; z₀ = zeros(4), options).z # the stage's solution y*_ρ
        others = setdiff(1:(k.variable_dimension), k.primal_dims)
        for mode in (:primal_only, :whole_iterate)
            starts = RG.ball_starts(k, center, 10; radius = 0.5, mode, rng = Xoshiro(1))
            @test length(starts) == 10
            @test all(z -> norm(z - center) <= 0.5 + 1e-12, starts)
            @test all(z -> minimum(z[k.nonnegative_dims]) > 0, starts)
            mode === :primal_only && @test all(z -> z[others] == center[others], starts)
            S = RG.solve_multi(RG.PDIP(), k, θ, starts; options, accept)
            @test S.rejected == 0 && S.starts == starts
            # The equilibrium is unique: every start lands on it, and the dedupe keeps one point.
            @test length(S.points) == 1 && all(==(1), S.assignment)
            @test norm(only(S.points) - center[k.primal_dims]) < 1e-8
            # One ‖K_ρ‖ per stopping test: the start, then one per Newton step.
            @test all(r -> length(r.norm_K_ρ_history) == r.newton_steps + 1, S.results)
        end
        # The same rng state gives two systems with as many primal coordinates the same primal offsets (recovered
        # as start − center, so up to the round-off of adding them to different centers).
        kc = RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed)
        center_c = RG.solve(RG.PDIP(), kc, θ; z₀ = zeros(4), options).z
        offsets(k, c) = [z[k.primal_dims] - c[k.primal_dims] for z in RG.ball_starts(k, c, 3; rng = Xoshiro(2))]
        @test all(map((a, b) -> isapprox(a, b; atol = 1e-14), offsets(k, center), offsets(kc, center_c)))
        # Primal starts (PDIP fills the rest), the default acceptance (`status`), and a rejection.
        pdip = RG.PDIPOptions(; linear_solver = :klu)
        S = RG.solve_multi(RG.PDIP(), k, θ, [zeros(4), fill(0.2, 4)]; options = pdip)
        @test S.rejected == 0 && length(S.points) == 1
        @test RG.solve_multi(RG.PDIP(), k, θ, [zeros(4)]; options = pdip, accept = r -> false).rejected == 1
        # A full start off the sign box is rejected; a start of neither length is an error.
        @test RG.solve_multi(RG.PDIP(), k, θ, [zeros(k.variable_dimension)]; options = pdip).rejected == 1
        @test_throws ArgumentError RG.solve_multi(RG.PDIP(), k, θ, [zeros(3)]; options = pdip)
        @test_throws ArgumentError RG.ball_starts(k, center, 1; mode = :box)
        @test_throws ArgumentError RG.ball_starts(k, center[1:3], 1)
        @test_throws ArgumentError RG.ball_starts(k, center, 1; radius = 0.0)
        # Far beyond the sign box no draw keeps it positive.
        @test_throws ArgumentError RG.ball_starts(k, center, 1; radius = 1e6, mode = :whole_iterate, max_tries = 10)
    end

    @testset "random starts are strictly feasible" begin
        Random.seed!(3)
        for z in RG.random_starts(prob_D.goop, θ0, 25; radius = 1.5, center = prob_D.z0)
            @test sfeasible(prob_D, z).worst > 0
        end
        @test_throws ArgumentError RG.random_starts(prob_A.goop, θ0, 5; center = [-10.0], radius = 0.1)
    end
end
