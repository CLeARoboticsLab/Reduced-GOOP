#=
Quasi vs. Reduced KKT under an innermost prioritized-constraint preference
=========================================================================

`:quasi` is `:reduced` with `drop_higher_order_terms = true`: the recursive
policy-gradient terms are truncated above second order. The two reformulations
therefore encode *different* stationarity systems, and any difference in their
solutions is a modelling difference, not a solver tolerance artifact.

This file asks whether that truncation is visible on the benchmark families in
`benchmark_problems.jl` when the inequality constraints are moved out of the
hard-constraint slot and into the innermost (highest-priority) preference level
— the formulation used by `experiments/robotic_arm_core.jl`:

    robot_preferences            = [control, goal, load_balance, robot_inequality]
    robot_is_prioritized_constraint = [false, false, false, true]

Both formulations are built from the *same* `ParametricGOOP`, solved with the
same initial guess, the same `InteriorPointOptions`, and the same stopping
criteria, so the only difference is the KKT generator.

Findings this file pins down (see the printed summary):

  * `:quasi` and `:reduced` are *not* the same system here. The residual maps
    differ by O(1) in relative terms as soon as any prioritized constraint is
    violated, and — for `cosh` objectives — everywhere.
  * The dropped terms carry policy multipliers, so the gap scales as O(dual²)
    and is *exactly* zero when the duals are zero. Whether a given solve ever
    sees the difference therefore depends on how large its multipliers grow.
  * Where both converge, they converge to genuinely different equilibria:
    ‖Δx‖∞ ≈ 0.67 (48 % relative) on the quadratic family, with second-level
    preference values differing by a factor of four. Each point solves its own
    KKT system to ‖F‖ ≤ 1e-8 while leaving the other's residual at 1e4–1e11, so
    this is a formulation difference, not solver noise.
  * `:reduced` is markedly harder to solve in this encoding: on the `cosh`
    family it diverges or trips a LAPACK NaN under every configuration tried,
    while `:quasi` converges in ~110 iterations.
  * Relaxing the sparse path to `tol = 1e-4` turns its stall into a reported
    `:solved`, and the accepted iterate is the *same* point for both
    formulations (‖Δx‖∞ ≤ 4e-6). It sits ≈0.06 from the reference `:quasi`
    solution but ≈0.73 from the reference `:reduced` one — that is, the sparse
    path converges toward the `:quasi` answer even when solving `:reduced`,
    because its multipliers never grow enough to activate the dropped terms.
    A relaxed tolerance therefore hides the formulation difference rather than
    exposing it.

Run standalone with (the root environment already carries every dependency, and
`test/Manifest.toml` must stay absent):

    julia --project=. test/quasi_vs_reduced_preference_inequality.jl
=#

using Test

using BlockArrays: Block, BlockArray
using LinearAlgebra: norm
using ReducedGOOP

@isdefined(build_benchmark_problem) ||
    include(joinpath(@__DIR__, "benchmark_problems.jl"))

# The ϵ-relaxed residual floors at ≈1.22ϵ, so ϵ₀ is pinned strictly below `tol`
# rather than left at `:auto`; otherwise the outer loop can stop at a residual
# that is dominated by the relaxation instead of by the formulation. This matters
# more than usual here: `:backtracking` never shrinks ϵ on its own.
#
# `linear_solver = :klu` rules out `use_marquardt_scaling`, `tsvd_threshold > 0`,
# and `record_condition_number` — the sparse path has no dense factorization to
# scale or inspect.
function comparison_interior_point_options(;
    tol = 1e-8,
    linesearch = :fraction_to_boundary,
    linear_solver = :svd,
    use_marquardt_scaling = true,
    verbose = false,
)
    @assert 1e-9 < tol "ϵ₀ = 1e-9 must stay strictly below tol"
    return ReducedGOOP.InteriorPointOptions(;
        tol,
        η₀ = 0.0,
        ϵ₀ = 1e-9,
        max_inner_iters = 5000,
        max_outer_iters = 20,
        tightening_rate = 2.0,
        loosening_rate = 0.5,
        min_stepsize = 1e-20,
        linesearch,
        linear_solver,
        record_convergence = false,
        record_condition_number = false,
        eta_retry_growth = 0.3,
        tsvd_threshold = 0.0,
        use_marquardt_scaling,
        verbose,
    )
end

# Three solver configurations, because the verdict depends on which one is used.
#
#   svd_fraction_to_boundary  matches `runtests.jl`; the only path that reaches
#                             ‖F‖ ≤ 1e-8 on these problems at all.
#   klu_backtracking          far cheaper per iteration, but stalls at ‖F‖ ≈ 1e-5
#                             and reports `:failed` against a 1e-8 tolerance.
#   klu_backtracking_relaxed  identical to the above except `tol = 1e-4`, chosen
#                             to sit just above that stall so the same iterate is
#                             *accepted* instead of rejected. Comparing it against
#                             the tight configurations separates "the sparse path
#                             finds a different answer" from "the sparse path
#                             finds the same answer but cannot certify it".
const SOLVER_CONFIGURATIONS = [
    (;
        name = :svd_fraction_to_boundary,
        tol = 1e-8,
        options = comparison_interior_point_options(;
            tol = 1e-8,
            linesearch = :fraction_to_boundary,
            linear_solver = :svd,
            use_marquardt_scaling = true,
        ),
    ),
    (;
        name = :klu_backtracking,
        tol = 1e-8,
        options = comparison_interior_point_options(;
            tol = 1e-8,
            linesearch = :backtracking,
            linear_solver = :klu,
            use_marquardt_scaling = false,
        ),
    ),
    (;
        name = :klu_backtracking_relaxed,
        tol = 1e-4,
        options = comparison_interior_point_options(;
            tol = 1e-4,
            linesearch = :backtracking,
            linear_solver = :klu,
            use_marquardt_scaling = false,
        ),
    ),
]

"The configuration every other one is measured against."
const REFERENCE_CONFIGURATION = first(SOLVER_CONFIGURATIONS).name

"""
    innermost_preference_inequality_goop(problem)

Rebuild `problem` with every player's hard inequality constraint `g(x, θ) ≥ 0`
appended as the innermost preference level (`is_prioritized_constraint = true`)
and no hard inequality constraints left. Preferences are stored
`[outermost, ..., innermost]`, so appending puts the constraint at the highest
priority — the `robotic_arm_core.jl` layout.
"""
function innermost_preference_inequality_goop(problem)
    num_players = problem.num_players
    @assert all(!isnothing, problem.inequality_constraints)

    preferences = [
        Function[problem.preferences[player]..., problem.inequality_constraints[player]] for player in 1:num_players
    ]
    is_prioritized_constraint = [
        Bool[problem.is_prioritized_constraint[player]..., true] for
        player in 1:num_players
    ]

    x_template = BlockArray(zeros(sum(problem.primal_dims)), problem.primal_dims)
    θ_template = BlockArray(zeros(sum(problem.parameter_dims)), problem.parameter_dims)

    return ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences,
        is_prioritized_constraint,
        equality_constraints = fill(nothing, num_players),
        inequality_constraints = fill(nothing, num_players),
        shared_equality_constraint = nothing,
        shared_inequality_constraint = nothing,
    )
end

"""
    preference_values(goop, primals)

Per-player preference values at `primals`, ordered `[outermost, ..., innermost]`.
Objective levels report their scalar value; prioritized-constraint levels report
the smooth penalty `sum(smooth_piecewise_preference_objective.(h, level))`, which
is exactly zero iff the constraint is satisfied.
"""
function preference_values(goop, primals)
    x_blocks = BlockArray(collect(primals), goop.primal_dims)
    θ_blocks = BlockArray(zeros(sum(goop.parameter_dims)), goop.parameter_dims)

    return map(1:goop.num_players) do player
        levels = zip(goop.preferences[player], goop.is_prioritized_constraint[player])
        map(enumerate(levels)) do (level, (preference, is_constraint))
            value = preference(x_blocks, θ_blocks)
            is_constraint ?
            sum(ReducedGOOP.smooth_piecewise_preference_objective.(value, level)) : value
        end
    end
end

"""
    inequality_residuals(problem, primals)

Raw `g(x, θ)` values of the *original* hard inequality constraints, per player.
"""
function inequality_residuals(problem, primals)
    x_blocks = BlockArray(collect(primals), problem.primal_dims)
    θ_blocks = BlockArray(zeros(sum(problem.parameter_dims)), problem.parameter_dims)
    return [
        problem.inequality_constraints[p](x_blocks, θ_blocks) for
        p in 1:problem.num_players
    ]
end

"Largest constraint violation max(-g, 0) over all players."
max_inequality_violation(problem, primals) =
    maximum(g -> maximum(max.(-g, 0.0)), inequality_residuals(problem, primals))

"Residual vector F(z; θ, ϵ) of `kkt` evaluated at an arbitrary point `z`."
function kkt_residual(kkt, z, θ; ϵ)
    F = zeros(kkt.variable_dimension)
    kkt.F!(F, z; θ, ϵ, η = 0.0)
    return F
end

kkt_residual_norm(kkt, z, θ; ϵ) = norm(kkt_residual(kkt, z, θ; ϵ), 2)

"""
    probe_point(kkt, primals; slack, dual)

A deterministic interior point of `kkt`'s variable space: the given primals,
strictly positive slacks, and uniform duals. Used to compare the two residual
maps as *functions*, independent of where either solver happened to stop.
"""
function probe_point(kkt, primals; slack = 1.0, dual = 0.5)
    z = fill(dual, kkt.variable_dimension)
    z[kkt.primal_dims] .= primals
    z[kkt.preference_slack_dims] .= slack
    z[kkt.interior_point_slack_dims] .= slack
    return z
end

"""
    residual_gap(reduced_kkt, quasi_kkt, z, θ; ϵ)

`‖F_reduced(z) - F_quasi(z)‖∞`, absolute and relative. Nonzero anywhere means the
dropped higher-order policy terms are not identically zero for this problem, so
`:quasi` and `:reduced` are genuinely different systems rather than two spellings
of one.
"""
function residual_gap(reduced_kkt, quasi_kkt, z, θ; ϵ)
    F_reduced = kkt_residual(reduced_kkt, z, θ; ϵ)
    F_quasi = kkt_residual(quasi_kkt, z, θ; ϵ)
    absolute = norm(F_reduced .- F_quasi, Inf)
    scale = max(norm(F_reduced, Inf), norm(F_quasi, Inf), eps())
    return (; absolute, relative = absolute / scale)
end

"Solve `goop` with one reformulation and collect everything worth comparing."
function solve_variant(goop, base_problem, variant::Symbol; z₀, options)
    generator = if variant === :reduced
        ReducedGOOP.generate_slacked_reduced_kkt_system
    elseif variant === :quasi
        ReducedGOOP.generate_slacked_quasi_kkt_system
    else
        error("Unknown variant: $variant")
    end

    build_time = @elapsed kkt = generator(goop)

    # A diverging iterate can push NaNs into the dense Tikhonov SVD, which LAPACK
    # reports as an argument error rather than a solver failure. Treat it as one
    # more (interesting) outcome instead of aborting the whole comparison.
    local output
    solve_time = @elapsed output = try
        ReducedGOOP.solve(
            ReducedGOOP.InteriorPoint(),
            kkt,
            zeros(sum(goop.parameter_dims));
            z₀,
            options,
        )
    catch err
        err isa InterruptException && rethrow()
        @warn "$variant solve threw" exception = err
        z = fill(NaN, kkt.variable_dimension)
        (;
            status = :errored,
            z,
            kkt_error = NaN,
            ϵ = NaN,
            outer_iters = -1,
            total_iters = -1,
        )
    end

    primals = output.z[kkt.primal_dims]
    diverged = any(!isfinite, primals)

    return (;
        variant,
        kkt,
        output,
        primals,
        diverged,
        status = output.status,
        kkt_error = output.kkt_error,
        outer_iters = output.outer_iters,
        total_iters = output.total_iters,
        ϵ = output.ϵ,
        variable_dimension = kkt.variable_dimension,
        preference_values = preference_values(goop, primals),
        inequality_residuals = inequality_residuals(base_problem, primals),
        max_violation = max_inequality_violation(base_problem, primals),
        build_time,
        solve_time,
    )
end

relative_difference(a, b) = norm(a .- b, Inf) / max(norm(a, Inf), norm(b, Inf), eps())

"""
    compare_reduced_vs_quasi(; num_players, levels, kind, z₀ = nothing, options = ...)

Build the benchmark problem, move its inequalities to the innermost preference,
solve the result with both `:reduced` and `:quasi`, and return the two solves
plus their pairwise differences.
"""
function compare_reduced_vs_quasi(;
    num_players::Int,
    levels::Int,
    kind::Symbol,
    z₀ = nothing,
    options = comparison_interior_point_options(),
    case = build_benchmark_problem(; num_players, levels, kind),
)
    goop = innermost_preference_inequality_goop(case.problem)
    initial_guess = isnothing(z₀) ? zeros(sum(case.primal_dims)) : z₀

    reduced = solve_variant(goop, case.problem, :reduced; z₀ = initial_guess, options)
    quasi = solve_variant(goop, case.problem, :quasi; z₀ = initial_guess, options)

    primal_abs = norm(reduced.primals .- quasi.primals, Inf)
    primal_rel = relative_difference(reduced.primals, quasi.primals)

    preference_abs = [
        abs.(reduced.preference_values[p] .- quasi.preference_values[p]) for
        p in 1:num_players
    ]
    preference_rel = [
        relative_difference(reduced.preference_values[p], quasi.preference_values[p])
        for p in 1:num_players
    ]

    # Cross-residuals: how far each solution is from satisfying the *other*
    # formulation's stationarity system. This separates "the two solvers landed
    # on the same point" from "the two systems have the same solution set".
    same_layout = reduced.variable_dimension == quasi.variable_dimension
    θ = zeros(sum(goop.parameter_dims))
    cross_residuals = if same_layout
        (;
            reduced_system_at_quasi_solution = kkt_residual_norm(
                reduced.kkt,
                quasi.output.z,
                θ;
                ϵ = reduced.ϵ,
            ),
            quasi_system_at_reduced_solution = kkt_residual_norm(
                quasi.kkt,
                reduced.output.z,
                θ;
                ϵ = quasi.ϵ,
            ),
        )
    else
        nothing
    end

    # Formulation-level probes: compare the two residual maps at points chosen
    # without reference to either solve.
    #
    # `feasible` uses the benchmark's known constrained solution, where every
    # g(x, θ) ≥ 0; `violating` shifts every coordinate down by 1, which drives the
    # lower-resource constraints strictly negative for both `kind`s. The
    # prioritized-constraint penalty is `ifelse(h ≥ 0, 0, (-h)^(level+2))`, so it
    # and *all* of its derivatives vanish identically on the feasible side —
    # that is where the two formulations must coincide and where they must not.
    #
    # `zero_dual` re-probes the violating point with every dual set to zero. The
    # terms `:quasi` drops are the ones carrying policy multipliers, so the gap
    # vanishes exactly there and grows as O(dual²) away from it. That is why a
    # solver path which keeps the multipliers small never separates the two
    # formulations no matter how infeasible its iterates are.
    residual_gaps = if same_layout
        gap_at(primals; ϵ = 1e-9, dual = 0.5) = residual_gap(
            reduced.kkt,
            quasi.kkt,
            probe_point(reduced.kkt, primals; dual),
            θ;
            ϵ,
        )
        (;
            at_initial_guess = gap_at(initial_guess),
            at_feasible_probe = gap_at(case.expected),
            at_violating_probe = gap_at(case.expected .- 1.0),
            at_violating_probe_zero_duals = gap_at(case.expected .- 1.0; dual = 0.0),
            at_reduced_solution = residual_gap(
                reduced.kkt,
                quasi.kkt,
                reduced.output.z,
                θ;
                ϵ = reduced.ϵ,
            ),
        )
    else
        nothing
    end

    return (;
        case,
        goop,
        initial_guess,
        reduced,
        quasi,
        residual_gaps,
        primal_abs,
        primal_rel,
        preference_abs,
        preference_rel,
        violation_abs = abs(reduced.max_violation - quasi.max_violation),
        kkt_error_abs = abs(reduced.kkt_error - quasi.kkt_error),
        same_layout,
        cross_residuals,
    )
end

function report_comparison(comparison; label)
    (; reduced, quasi) = comparison
    println("\n", "="^78)
    println("  $label")
    println("="^78)
    print_row(name, r, q) = println(rpad("  " * name, 34), rpad(string(r), 24), string(q))
    print_row("", "reduced", "quasi")
    print_row("status", reduced.status, quasi.status)
    print_row(
        "outer / inner iterations",
        "$(reduced.outer_iters) / $(reduced.total_iters)",
        "$(quasi.outer_iters) / $(quasi.total_iters)",
    )
    print_row("final KKT residual", reduced.kkt_error, quasi.kkt_error)
    print_row("final ϵ", reduced.ϵ, quasi.ϵ)
    print_row(
        "KKT variable dimension",
        reduced.variable_dimension,
        quasi.variable_dimension,
    )
    print_row("max inequality violation", reduced.max_violation, quasi.max_violation)
    print_row(
        "build / solve time [s]",
        "$(round(reduced.build_time; digits = 2)) / $(round(reduced.solve_time; digits = 3))",
        "$(round(quasi.build_time; digits = 2)) / $(round(quasi.solve_time; digits = 3))",
    )

    println("\n  primal solution:")
    println("    ‖Δx‖∞ (absolute) = ", comparison.primal_abs)
    println("    ‖Δx‖∞ (relative) = ", comparison.primal_rel)

    println("\n  preference values per player [outermost … innermost]:")
    for player in 1:comparison.goop.num_players
        println("    player $player")
        println("      reduced = ", reduced.preference_values[player])
        println("      quasi   = ", quasi.preference_values[player])
        println("      |Δ|     = ", comparison.preference_abs[player])
        println("      rel Δ   = ", comparison.preference_rel[player])
    end

    if !isnothing(comparison.cross_residuals)
        println("\n  cross-residuals (same variable layout):")
        println(
            "    ‖F_reduced(z_quasi)‖ = ",
            comparison.cross_residuals.reduced_system_at_quasi_solution,
        )
        println(
            "    ‖F_quasi(z_reduced)‖ = ",
            comparison.cross_residuals.quasi_system_at_reduced_solution,
        )
    end

    if !isnothing(comparison.residual_gaps)
        println("\n  residual-map gap ‖F_reduced(z) - F_quasi(z)‖∞ (abs / rel):")
        for (name, gap) in pairs(comparison.residual_gaps)
            println("    $(rpad(name, 22)) ", gap.absolute, "  /  ", gap.relative)
        end
    end

    if !isnothing(comparison.cross_residuals) &&
       reduced.status === :solved &&
       quasi.status === :solved
        println("\n  raw inequality residuals g(x, θ) at each solution:")
        println("    reduced = ", reduced.inequality_residuals)
        println("    quasi   = ", quasi.inequality_residuals)
    end
    println()
    return comparison
end

# Threshold separating "same solution up to solver noise" from "the truncation
# changed the equilibrium". Both formulations are driven to ‖F‖ ≤ 1e-8 and the
# interior-point iterates inherit roughly sqrt(tol) primal accuracy, so anything
# above 1e-4 in relative terms is a genuine formulation difference.
const FORMULATION_DIFFERENCE_TOL = 1e-4

# A point that solves one system but leaves the other with a residual this large
# proves the two stationarity systems do not share a solution, independent of any
# argument about which local solution the solver happened to find.
const CROSS_RESIDUAL_TOL = 1e-6

"""
    initial_guesses(case)

Deterministic starting points shared by both formulations. `expected` is the
benchmark's known constrained solution (strictly feasible for the inequalities);
`zeros` and `offset` probe whether any discrepancy is start-dependent, i.e. two
local solutions of one system rather than two different systems.
"""
function initial_guesses(case)
    return [
        :expected => copy(case.expected),
        :zeros => zeros(length(case.expected)),
        :offset => case.expected .+ 0.3,
    ]
end

"Assert that one solve is a bona fide solution of the system it was built from."
function check_solve(solve; label, tol)
    @test solve.status === :solved
    @test !solve.diverged
    @test solve.kkt_error <= tol

    # The innermost level is a prioritized constraint, so its smooth penalty —
    # not the raw residual — is what the formulation drives to zero. The penalty
    # is `(-h)^(level+2)`, so it shrinks far faster than the residual and stays
    # comfortably under `tol` even on the relaxed configuration.
    innermost_penalty = maximum(last, solve.preference_values)
    @test innermost_penalty <= max(1e-8, tol)
    return (; label, innermost_penalty)
end

function test_reduced_vs_quasi(; num_players::Int, levels::Int, kind::Symbol)
    case = build_benchmark_problem(; num_players, levels, kind)

    results =
        map(SOLVER_CONFIGURATIONS) do config
            (; name, tol, options) = config
            map(initial_guesses(case)) do (start_name, z₀)
                comparison = compare_reduced_vs_quasi(;
                    num_players,
                    levels,
                    kind,
                    z₀,
                    options,
                    case,
                )
                report_comparison(
                    comparison;
                    label = "$(num_players)-player, $(levels)-level + innermost inequality " *
                            "preference, kind = :$kind, solver = :$name (tol = $tol), " *
                            "z₀ = :$start_name",
                )

                both_solved =
                    comparison.reduced.status === :solved &&
                    comparison.quasi.status === :solved
                if both_solved
                    check_solve(comparison.reduced; label = "reduced/$start_name", tol)
                    check_solve(comparison.quasi; label = "quasi/$start_name", tol)
                end

                (;
                    config_name = name,
                    config_tol = tol,
                    start_name,
                    comparison,
                    both_solved,
                )
            end
        end |>
        Iterators.flatten |>
        collect

    # Formulation-level invariants, independent of whether any solve converged.
    # These are the crisp answer to "are :quasi and :reduced the same system?".
    gaps = first(results).comparison.residual_gaps
    @test !isnothing(gaps)

    # Once any prioritized constraint is violated, the penalty
    # `(-h)^(level+2)` contributes derivatives above order 2 that `:quasi` drops,
    # so the two residual maps differ — for every problem kind.
    @test gaps.at_violating_probe.relative > FORMULATION_DIFFERENCE_TOL

    # …but only through the policy multipliers. Zero the duals at that same
    # violating point and the two maps agree exactly.
    @test gaps.at_violating_probe_zero_duals.absolute == 0.0

    # On the feasible side the penalty and all its derivatives vanish identically
    # (`ifelse(h ≥ 0, 0, …)`), so the only remaining source of higher-order terms
    # is the objectives themselves. Quadratic objectives have zero third
    # derivatives and the two maps coincide exactly there; `cosh` objectives do
    # not, and the maps differ everywhere.
    if kind === :quadratic
        @test gaps.at_feasible_probe.absolute <= 1e-12
    else
        @test gaps.at_feasible_probe.relative > FORMULATION_DIFFERENCE_TOL
    end

    return (; kind, case, results)
end

"‖a - b‖∞, or NaN if either point is not finite."
function primal_distance(a, b)
    (any(!isfinite, a) || any(!isfinite, b)) && return NaN
    return norm(a .- b, Inf)
end

"""
    cross_configuration_report(entry)

For each initial guess and each formulation, measure how far every solver
configuration's answer sits from the reference configuration's answer — both for
the *same* formulation and for the *other* one.

The second column is the interesting one. If a relaxed sparse solve is simply a
lower-accuracy version of the reference, it sits near the reference solution for
its own formulation. If instead it sits equidistant from both, or nearer the
other formulation's solution, then it stopped somewhere the choice of
formulation had not yet separated the two trajectories.
"""
function cross_configuration_report(entry)
    println("\n", "-"^78)
    println("  CROSS-CONFIGURATION primal distances, kind = :$(entry.kind)")
    println("  reference = :$REFERENCE_CONFIGURATION")
    println("-"^78)
    println(
        rpad("  z₀", 10),
        rpad("variant", 9),
        rpad("solver", 26),
        rpad("status", 9),
        rpad("‖F‖", 12),
        rpad("Δ vs ref same-variant", 24),
        "Δ vs ref other-variant",
    )

    by_key = Dict((r.config_name, r.start_name) => r.comparison for r in entry.results)

    for (start_name, _) in initial_guesses(entry.case)
        reference = get(by_key, (REFERENCE_CONFIGURATION, start_name), nothing)
        isnothing(reference) && continue

        for variant in (:reduced, :quasi)
            other = variant === :reduced ? :quasi : :reduced
            reference_same = getproperty(reference, variant).primals
            reference_other = getproperty(reference, other).primals

            for config in SOLVER_CONFIGURATIONS
                comparison = get(by_key, (config.name, start_name), nothing)
                isnothing(comparison) && continue
                solve = getproperty(comparison, variant)
                println(
                    rpad("  $start_name", 10),
                    rpad(string(variant), 9),
                    rpad(string(config.name), 26),
                    rpad(string(solve.status), 9),
                    rpad(string(round(solve.kkt_error; sigdigits = 3)), 12),
                    rpad(string(primal_distance(solve.primals, reference_same)), 24),
                    string(primal_distance(solve.primals, reference_other)),
                )
            end
        end
    end
    println()
end

function summarize(all_results)
    for entry in all_results
        cross_configuration_report(entry)
    end

    println("\n", "="^78)
    println("  SUMMARY — :reduced vs :quasi, inequality as innermost preference")
    println("="^78)
    println(
        rpad("  problem", 12),
        rpad("solver", 26),
        rpad("tol", 8),
        rpad("z₀", 10),
        rpad("‖Δx‖∞", 24),
        rpad("relative", 24),
        "verdict",
    )
    for entry in all_results, result in entry.results
        (; comparison, both_solved) = result
        verdict = if !both_solved
            "NOT CONVERGED ($(comparison.reduced.status)/$(comparison.quasi.status), " *
            "‖F‖ = $(round(comparison.reduced.kkt_error; sigdigits = 3))/" *
            "$(round(comparison.quasi.kkt_error; sigdigits = 3)))"
        elseif comparison.primal_rel <= FORMULATION_DIFFERENCE_TOL
            "within numerical tolerance"
        else
            "MEANINGFUL formulation difference"
        end
        println(
            rpad("  $(entry.kind)", 12),
            rpad(string(result.config_name), 26),
            rpad(string(result.config_tol), 8),
            rpad(string(result.start_name), 10),
            rpad(string(comparison.primal_abs), 24),
            rpad(string(comparison.primal_rel), 24),
            verdict,
        )
    end

    println("\n  relative residual-map gap ‖F_reduced(z) - F_quasi(z)‖∞ / ‖F‖∞")
    println("  Feasible/violating probes depend only on the formulation, not on")
    println("  the solver, so one row per problem is enough.")
    println(
        rpad("  problem", 12),
        rpad("feasible probe", 24),
        rpad("violating probe", 24),
        "z₀ = zeros probe",
    )
    for entry in all_results
        gaps = first(entry.results).comparison.residual_gaps
        isnothing(gaps) && continue
        zeros_gap = something(
            findfirst(r -> r.start_name === :zeros, entry.results),
            firstindex(entry.results),
        )
        println(
            rpad("  $(entry.kind)", 12),
            rpad(string(gaps.at_feasible_probe.relative), 24),
            rpad(string(gaps.at_violating_probe.relative), 24),
            string(
                entry.results[zeros_gap].comparison.residual_gaps.at_initial_guess.relative,
            ),
        )
    end
    println()
end

@testset "Quasi vs Reduced — inequalities as innermost preference" begin
    all_results = []

    @testset "Three-level quadratic" begin
        push!(
            all_results,
            test_reduced_vs_quasi(; num_players = 3, levels = 3, kind = :quadratic),
        )
    end

    @testset "Three-level nonconvex" begin
        push!(
            all_results,
            test_reduced_vs_quasi(; num_players = 3, levels = 3, kind = :nonlinear),
        )
    end

    summarize(all_results)
end
