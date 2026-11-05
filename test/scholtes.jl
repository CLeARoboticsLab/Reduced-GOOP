# The Scholtes test suite, translated from ScholtesReducedGOOP.jl's test/runtests.jl
# (Jingqi Li). The problems are the source's, written against a flat primal `z` and
# adapted to `ParametricGOOP` by `sproblem` (one block per player, no parameters).
#
# Part 1, "The Scholtes system": the system of `generate_slacked_scholtes_reduced_kkt_system`,
# started by `initialize_scholtes!` and solved by `solve(ProjectedLM(), …)`. These are the
# source's tests of the system and of its solves; the solvers' own tests are in test/solver.jl.
# Part 2, "The Scholtes engine": the source's tests of the engine's own features (its
# Jacobian memo, certificate, multistart, non-monotone line search, refinement, in-place
# Gram refresh and option checks), as ported; they go with the engine (src/scholtes_*.jl).
#
# From the engine's port, not translated (φ = true only, _port_logs/DECISIONS.md D19): the
# effective-Γ (`phi = false`) testset, the `:freeze` / `:ideal` bound rules, the
# `scholtes = false` closure, the `:ldl` backend, `SolverWorkspace`, and the fixed-primal
# check that needs the source's analysis environment. Backend names in part 2: the source's
# `:klu` is `:klu_eta` (same saddle matrix), `:pinv` is `:svd`; `:klu_sqrt_eta` is added
# where the source loops over backends.

using LinearAlgebra
using Random
using SparseArrays
using BlockArrays: BlockArray
using Symbolics: Symbolics

const RG = ReducedGOOP
const θ0 = Float64[]

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
sfeasible(p, z) = RG.is_feasible(p.goop, z, θ0)

# --- part 1: the Scholtes system, started by initialize_scholtes!, solved by ProjectedLM ------
scholtes_kkt(p; quasi = false, quasi_order = 2) = RG.generate_slacked_scholtes_reduced_kkt_system(
    p.goop; drop_higher_order_terms = quasi, quasi_order)

"The cold start (the source's own start) at the primal `x` and the level ρ, or with `warm` the warm one."
scholtes_start(k, x, ρ; warm = false, kwargs...) =
    RG.initialize_scholtes!(zeros(k.variable_dimension), k, θ0, x, ρ; warm, kwargs...)

"""
`ProjectedLM` from the cold start at the primal `x` and the first level of `ρ_schedule` (by default the
source's ladder 1, 0.1, …, 1e-10), with the other options given.
"""
function plm_solve(k, x; ρ_schedule = [10.0^(-j) for j in 0:10], kwargs...)
    RG.solve(RG.ProjectedLM(), k, θ0; z₀ = scholtes_start(k, x, first(ρ_schedule)),
        options = RG.ProjectedLMOptions(; ρ_schedule, kwargs...))
end
plm_solve(p::NamedTuple; kwargs...) = plm_solve(scholtes_kkt(p), p.z0; kwargs...)

"K_ρ(z)."
function residual(k, z, ρ)
    K_ρ = zeros(k.kkt_dimension)
    k.K_ρ!(K_ρ, z; θ = θ0, ρ)
    K_ρ
end

"∇K_ρ(z)."
function jacobian(k, z, ρ)
    J = copy(k.∇K_ρ!.result_buffer)
    k.∇K_ρ!(J, z; θ = θ0, ρ)
    J
end

"The slack targets at z, one per complementarity pair: G for σ, then ρ𝟙 − Gₘ ⊙ γₘ for σ̂ (the engine's block a)."
function targets(k, z, ρ)
    a = zeros(length(k.slack_expressions))
    k.slack_targets!(a, z; θ = θ0, ρ)
    a
end

"Every complementarity pair's multiplier, γ then ϕ (the engine's block b)."
multipliers(k, z) = z[[pair.multiplier for pair in k.complementarity_pairs]]

"The relaxed pairs (σ ⊙ γ + u = ρ𝟙) and the exact ones (σ̂ ⊙ ϕ = 0), by position in `complementarity_pairs`."
relaxed_pairs(k) = findall(pair -> pair.u_relaxation != 0, k.complementarity_pairs)
exact_pairs(k) = findall(pair -> pair.u_relaxation == 0, k.complementarity_pairs)

"The row of K_ρ that puts each pair's slack on its target: G − σ = 0, or ρ𝟙 − Gₘ ⊙ γₘ − σ̂ = 0."
slack_rows(k) = [only(findall(isequal(e - k.z_symbolic[pair.slack]), k.K_ρ_symbolic))
                 for (e, pair) in zip(k.slack_expressions, k.complementarity_pairs)]

"The Jacobian of the slack targets at z: the slack rows of ∇K_ρ, less their −1 in the slack's own column."
function target_jacobian(k, z, ρ)
    Ja = Matrix(jacobian(k, z, ρ)[slack_rows(k), :])
    for (j, pair) in enumerate(k.complementarity_pairs)
        Ja[j, pair.slack] += 1.0
    end
    Ja
end

"Whether a row of K_ρ holds a slack σ or σ̂: the slack and product rows do, the stationarity rows do not."
function slack_row_filter(k)
    slacks = Set(Symbolics.unwrap.(k.z_symbolic[[pair.slack for pair in k.complementarity_pairs]]))
    row -> any(in(slacks), Symbolics.get_variables(row))
end

# --- part 2: the engine's helpers, as ported -------------------------------------------------
# The source's default solver options (its default backend is `:klu`).
const SRC_DEFAULTS = RG.ScholtesOptions(linear_solver = :klu_eta)

skkt(p; quasi = false, quasi_order = 2) = RG._generate_scholtes_reduced_kkt_system(
    p.goop; drop_higher_order_terms = quasi, quasi_order)

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

@testset "The Scholtes system (ported from ScholtesReducedGOOP.jl)" begin

@testset "problem model" begin
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

@testset "quasi composes with explicit Scholtes complementarity" begin
    k = scholtes_kkt(prob_E; quasi = true)
    r = plm_solve(k, prob_E.z0)
    @test r.status === :solved && minimum(r.z[k.nonnegative_dims]) > 0
    @test r.x[1] ≈ 0.5 atol = 1e-5
    @test r.x[1] + r.x[2] ≈ 1.0 atol = 1e-5
end

@testset "inactive constraint, all linear solvers" begin
    k = scholtes_kkt(prob_A)
    for linear_solver in (:normal, :klu, :ldl)
        r = plm_solve(k, [1.0]; linear_solver)
        @test r.status === :solved
        @test r.x[1] ≈ 0.0 atol = 1e-6
        @test minimum(r.z[k.nonnegative_dims]) > 0
        @test r.svd_fallbacks == 0
    end
end

@testset "active constraint and coupled game" begin
    r = plm_solve(prob_B)
    @test r.status === :solved && abs(r.x[1]) < 1e-5
    r = plm_solve(prob_C)
    @test r.status === :solved
    @test r.x ≈ [0.8, -0.8] atol = 1e-5
end

@testset "warm start reproduces the solve" begin
    k = scholtes_kkt(prob_C)
    r1 = plm_solve(k, prob_C.z0)
    r2 = RG.solve(RG.ProjectedLM(), k, θ0; z₀ = r1.z, options = RG.ProjectedLMOptions(; ρ_schedule = [r1.ρ], η = r1.η))
    @test r2.status === :solved
    @test r2.x ≈ r1.x atol = 1e-8
    @test r2.newton_steps <= r1.newton_steps
    @test_throws ArgumentError RG.solve(RG.ProjectedLM(), k, θ0; z₀ = zeros(3))
end

@testset "a start on the boundary is rejected, not silently accepted" begin
    # prob_A's g(z) = z + 2 is 0 at z = −2, and the cold start needs g > 0 strictly; a start that is not
    # positive on nonnegative_dims is refused by the solver.
    k = scholtes_kkt(prob_A)
    @test_throws ArgumentError plm_solve(k, [-2.0])
    z = scholtes_start(k, [1.0], 1.0)
    z[first(k.nonnegative_dims)] = 0.0
    @test_throws ArgumentError RG.solve(RG.ProjectedLM(), k, θ0; z₀ = z)
end

@testset "the projected steps stay in the sign box and lower ‖K_ρ‖" begin
    # Every iterate is positive on nonnegative_dims (σ, γ, ϕ, σ̂, u), and every accepted step lowers
    # ‖K_ρ(z)‖ strictly within a stage.
    for prob in (prob_A, prob_B, prob_single)
        k = scholtes_kkt(prob)
        r = plm_solve(k, prob.z0; record_convergence = true)
        @test r.status === :solved
        @test minimum(r.z[k.nonnegative_dims]) > 0
        history, ρs = r.norm_K_ρ_history, r.ρ_history
        @test all(history[i + 1] < history[i] for i in 1:(length(history) - 1) if ρs[i + 1] == ρs[i])
    end
    @test_throws ArgumentError plm_solve(prob_A; τ = 1.5)
    @test_throws ArgumentError plm_solve(prob_A; proj_eps = -1.0)
end

@testset "every linear solver solves with the projected steps" begin
    for prob in (prob_A, prob_single), linear_solver in (:normal, :klu, :ldl)
        k = scholtes_kkt(prob)
        r = plm_solve(k, prob.z0; linear_solver)
        @test r.status === :solved
        @test minimum(r.z[k.nonnegative_dims]) > 0
    end
end

@testset "an explicit ρ_schedule is honoured exactly" begin
    k = scholtes_kkt(prob_C)
    r = plm_solve(k, prob_C.z0; ρ_schedule = [1.0, 1e-3], max_inner_iters = 40, record_convergence = true)
    @test r.ρ in (1.0, 1e-3)
    @test issubset(r.ρ_history, [1.0, 1e-3])
    @test minimum(r.z[k.nonnegative_dims]) > 0
end

@testset "the returned ρ is the level the walk actually ended on" begin
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

@testset "inequalities hold exactly at a Scholtes solution" begin
    # Active constraints (prob_B, prob_D's and prob_E's box) end ON their bound, not past
    # it. prob_F is left out: it does not converge under the default options, in the
    # source either (Step 1 reference: ‖K₀‖ ≈ 4e-4).
    for prob in (prob_B, prob_C, prob_D, prob_E)
        r = plm_solve(prob)
        @test r.status === :solved
        @test sfeasible(prob, r.x).worst >= -1e-8
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
    # Only the Scholtes system has the pairs and slack targets the start reads.
    @test_throws ArgumentError RG.initialize_scholtes!(zeros(1), RG.generate_slacked_reduced_kkt_system(prob_E.goop),
        θ0, [0.1, 0.2], 1e-3)
end

end

@testset "The Scholtes engine (ported from ScholtesReducedGOOP.jl; goes with src/scholtes_*.jl)" begin

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

@testset "the augmented KLU step of :klu_eta and :klu_sqrt_eta" begin
	problem = build_benchmark_problem(; num_players = 3, levels = 2, kind = :quadratic)
	kkt = RG.generate_slacked_reduced_kkt_system(problem.problem)
	θ = zeros(sum(problem.problem.parameter_dims))

	@testset "Direction equivalence with the SVD Tikhonov step" begin
		# The augmented quasi-definite system [[I J]; [Jᵀ -ηI]] must reproduce the
		# dense-SVD step δz = -V diag(S/(S²+η)) Uᵀ F exactly (tsvd_threshold = 0,
		# no Marquardt scaling).
		n = kkt.variable_dimension
		m = kkt.kkt_dimension
		z = 0.3 .* sin.(1:n)
		ρ = 0.1

		F = zeros(m)
		kkt.K_ρ!(F, z; θ, ρ)
		J = kkt.∇K_ρ!.result_buffer
		kkt.∇K_ρ!(J, z; θ, ρ)

		# High-η cases lock in the √η-balanced augmented scaling: the naive
		# [I J; Jᵀ -ηI] form loses the tiny ~1/η step to cancellation there.
		for η in (1e-6, 1e-2, 1e2, 1e6)
			cache = RG._build_augmented_kkt_cache(J, m, n)
			RG._update_augmented_kkt!(cache, J, η)
			δz_klu = zeros(n)
			RG._solve_augmented!(δz_klu, cache, F)

			Jsvd = RG.LinearAlgebra.svd(Matrix(J))
			δz_svd = -Jsvd.V * ((Jsvd.S ./ (Jsvd.S .^ 2 .+ η)) .* (Jsvd.U' * F))

			@test isapprox(δz_klu, δz_svd; rtol = 1e-8)
		end
	end

	@testset "Singular numeric refactorization rebuilds its cache" begin
		# Reproduce the cache state that matters here: a successful numeric
		# factorization followed by a singular in-place `klu!` update. The
		# retry must discard that failed numeric object, rebuild at escalated η,
		# and avoid the dense-SVD fallback.
		J = RG.SparseArrays.sparse([1], [1], [1.0], 1, 1)
		cache = RG._build_augmented_kkt_cache(J, 1, 1)
		RG._update_augmented_kkt!(cache, J, 1e-6)
		δz = zeros(1)
		F = ones(1)
		RG._solve_augmented!(δz, cache, F)

		# Make the next numeric update singular while retaining the same sparse
		# pattern. The retry's η update restores a nonsingular diagonal system.
		RG.SparseArrays.nonzeros(J) .= 0.0
		RG.SparseArrays.nonzeros(cache.K) .= 0.0
		singular_retries = Ref(0)
		svd_fallbacks = Ref(0)
		η_used = RG._klu_step_with_fallback!(
			δz,
			cache,
			J,
			F,
			1e-6,
			1e2,
			false;
			singular_retry_counter = singular_retries,
			svd_fallback_counter = svd_fallbacks,
		)

		@test singular_retries[] == 1
		@test svd_fallbacks[] == 0
		@test η_used ≈ 1e-4
		@test all(isfinite, δz)
	end
end

@testset "the new knobs validate their arguments" begin
    for kw in ((; nonmonotone = 0), (; linear_solver = :nope), (; rho_min = 0.0),
               (; refine = -1), (; rho_schedule = Float64[]), (; projected_step = false),
               (; reuse_factorization_iters = 1))
        @test_throws ArgumentError ssolve(prob_A; kw...)
    end
end

end
