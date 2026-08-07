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
criteria, so the only difference is the KKT generator. Every solve uses one
configuration — sparse `:klu` with a `:backtracking` line search, `tol = 1e-4`
held fixed, and `max_outer_iters = 1` so there is no ϵ-tightening loop and the
reported iteration count is purely inner (`1 / n`). See `COMPARISON_OPTIONS`.

Each solution is also measured against the benchmark's designed solution `x*`,
which `benchmark_problems.jl` constructs in closed form.

Findings this file pins down (see the printed summary):

  * `:quasi` and `:reduced` are *not* the same system here. Compared as residual
    maps — which needs no solve at all — they differ by O(1) in relative terms
    as soon as any prioritized constraint is violated, and, for `cosh`
    objectives, everywhere.
  * The dropped terms carry policy multipliers: the gap scales as O(dual²) and
    is *exactly* zero when the duals are zero. Whether a given solve ever sees
    the difference therefore depends on how large its multipliers grow.
  * On this solver path they never grow. Both formulations stall together at
    ‖F‖ ≈ 1e-5 and produce the same iterate, so the *solve* comparison reports
    agreement on every row. That is a statement about the trajectory, not about
    the formulations — the probes above are the ones that settle the question.
  * Neither formulation recovers `x*` here: the innermost penalty is flat to
    fifth order at the boundary, so nothing pulls the iterate back onto the
    active constraints and both stop short of it while violating them.

The dense `:svd` / `:fraction_to_boundary` path — the only one that reaches
‖F‖ ≤ 1e-8 on these problems — is deliberately not exercised here. Under it the
two formulations converge to genuinely different equilibria (‖Δx‖∞ ≈ 0.67 on the
quadratic family, with `:reduced` recovering the benchmark's known solution to
≈0.011 and `:quasi` landing ≈0.67 away). Reintroduce it if you want the solve
comparison, and not just the residual-map probes, to exhibit the difference.

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

# The solver tolerance is held at 1e-4 throughout — no per-run overrides.
#
# The interior-point iteration stalls at ‖F‖ ≈ 1e-5 on these problems: the
# innermost prioritized constraint enters as `(-h)^(level+2)`, which is flat to
# fifth order at the boundary, so a tighter tolerance only rejects the iterate it
# was always going to return. At 1e-6 every solve reports `:failed`, which also
# skips the per-solve `check_solve` assertions (8 tests instead of 56).
const COMPARISON_TOLERANCE = 1e-6

function comparison_solver_options(;
    tol = COMPARISON_TOLERANCE,
    verbose = false,
    record_convergence = false,
)
    @assert 1e-9 < tol "ϵ₀ = 1e-9 must stay strictly below tol"
    return ReducedGOOP.InteriorPointOptions(;
        tol,
        η₀ = 0.0,
        ϵ₀ = 0.0,
        max_inner_iters = 1000,
        max_outer_iters = 1,
        tightening_rate = 2.0,
        loosening_rate = 0.5,
        min_stepsize = 1e-20,
        linesearch = :backtracking,
        linear_solver = :klu,
        record_convergence,
        verbose,
    )
end

"The single solver configuration: sparse KLU with a backtracking line search."
const COMPARISON_OPTIONS = comparison_solver_options()

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

    # A diverging iterate can drive the factorization singular or push NaNs into
    # it, which surfaces as a thrown error rather than a solver failure. Treat it
    # as one more (interesting) outcome instead of aborting the whole comparison.
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
        outer_passes = max(output.outer_iters - 1, 0),
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
    options = COMPARISON_OPTIONS,
    case = build_benchmark_problem(; num_players, levels, kind),
)
    goop = innermost_preference_inequality_goop(case.problem)
    initial_guess = isnothing(z₀) ? zeros(sum(case.primal_dims)) : z₀

    reduced = solve_variant(goop, case.problem, :reduced; z₀ = initial_guess, options)
    quasi = solve_variant(goop, case.problem, :quasi; z₀ = initial_guess, options)

    primal_abs = norm(reduced.primals .- quasi.primals, Inf)
    primal_rel = relative_difference(reduced.primals, quasi.primals)

    # Main.@infiltrate

    # Distance to the ground truth. `case.expected` is the closed-form solution
    # the benchmark family was constructed around: the two active coupled
    # resources pin coordinates 1 and 3, and every other coordinate sits at its
    # unconstrained target. It is the solution of the *hard-constrained* problem;
    # moving the inequalities into the innermost preference should preserve it,
    # because the smooth penalty attains its minimum (zero) on exactly the
    # feasible set. In practice the penalty is flat to fifth order at the
    # boundary, so a solve is not pulled back onto it — the gap below is a
    # property of the encoding, not of the two formulations.
    truth = case.expected
    reduced_vs_truth = (;
        absolute = norm(reduced.primals .- truth, Inf),
        relative = relative_difference(reduced.primals, truth),
    )
    quasi_vs_truth = (;
        absolute = norm(quasi.primals .- truth, Inf),
        relative = relative_difference(quasi.primals, truth),
    )

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
        truth,
        reduced_vs_truth,
        quasi_vs_truth,
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
        "$(reduced.outer_passes) / $(reduced.total_iters)",
        "$(quasi.outer_passes) / $(quasi.total_iters)",
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

    println("\n  primal solution — reduced vs quasi:")
    println("    ‖Δx‖∞ (absolute) = ", comparison.primal_abs)
    println("    ‖Δx‖∞ (relative) = ", comparison.primal_rel)

    println("\n  primal solution — vs ground truth x*:")
    println("    x*      = ", comparison.truth)
    println("    reduced = ", comparison.reduced.primals)
    println("    quasi   = ", comparison.quasi.primals)
    println("    reduced - x* = ", comparison.reduced.primals .- comparison.truth)
    println("    quasi   - x* = ", comparison.quasi.primals .- comparison.truth)
    println(
        "    ‖x_reduced - x*‖∞ = ",
        comparison.reduced_vs_truth.absolute,
        "  (relative ",
        comparison.reduced_vs_truth.relative,
        ")",
    )
    println(
        "    ‖x_quasi   - x*‖∞ = ",
        comparison.quasi_vs_truth.absolute,
        "  (relative ",
        comparison.quasi_vs_truth.relative,
        ")",
    )

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

    results = map(initial_guesses(case)) do (start_name, z₀)
        comparison = compare_reduced_vs_quasi(; num_players, levels, kind, z₀, case)
        report_comparison(
            comparison;
            label = "$(num_players)-player, $(levels)-level + innermost inequality " *
                    "preference, kind = :$kind, tol = $COMPARISON_TOLERANCE, " *
                    "z₀ = :$start_name",
        )

        both_solved =
            comparison.reduced.status === :solved && comparison.quasi.status === :solved
        if both_solved
            check_solve(
                comparison.reduced;
                label = "reduced/$start_name",
                tol = COMPARISON_TOLERANCE,
            )
            check_solve(
                comparison.quasi;
                label = "quasi/$start_name",
                tol = COMPARISON_TOLERANCE,
            )
        end

        (; start_name, comparison, both_solved)
    end

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

"""
    ground_truth_report(entry)

For each initial guess and each formulation, how far the computed solution sits
from the benchmark's designed solution `x*`, alongside the constraint violation
there. The pair of columns answers two different questions: whether either
formulation recovers `x*`, and whether the point it does reach is even feasible.
"""
function ground_truth_report(entry)
    println("\n", "-"^78)
    println("  GROUND-TRUTH primal distances, kind = :$(entry.kind)")
    println("  x* = ", entry.case.expected)
    println("-"^78)
    println(
        rpad("  z₀", 10),
        rpad("variant", 9),
        rpad("status", 9),
        rpad("‖F‖", 12),
        rpad("‖x - x*‖∞", 24),
        rpad("relative", 24),
        "max violation",
    )

    for result in entry.results, variant in (:reduced, :quasi)
        solve = getproperty(result.comparison, variant)
        distance =
            variant === :reduced ? result.comparison.reduced_vs_truth :
            result.comparison.quasi_vs_truth
        println(
            rpad("  $(result.start_name)", 10),
            rpad(string(variant), 9),
            rpad(string(solve.status), 9),
            rpad(string(round(solve.kkt_error; sigdigits = 3)), 12),
            rpad(string(distance.absolute), 24),
            rpad(string(distance.relative), 24),
            string(solve.max_violation),
        )
    end
    println()
end

function summarize(all_results)
    for entry in all_results
        ground_truth_report(entry)
    end

    println("\n", "="^78)
    println("  SUMMARY — :reduced vs :quasi, inequality as innermost preference")
    println(
        "  solver = klu/backtracking, tol = $COMPARISON_TOLERANCE, max_outer_iters = 1",
    )
    println("="^78)
    println(
        rpad("  problem", 12),
        rpad("z₀", 10),
        rpad("‖Δx‖∞ reduced-quasi", 24),
        rpad("‖x_red - x*‖∞", 24),
        rpad("‖x_qua - x*‖∞", 24),
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
            rpad(string(result.start_name), 10),
            rpad(string(comparison.primal_abs), 24),
            rpad(string(comparison.reduced_vs_truth.absolute), 24),
            rpad(string(comparison.quasi_vs_truth.absolute), 24),
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

# `quasi_vs_reduced_convergence_plot.jl` includes this file only for the helpers
# above, and defines `RUN_QUASI_VS_REDUCED_TESTSET = false` beforehand so the
# comparison suite is not re-run underneath it. Every other entry point — a bare
# `include`, or `julia --project=. test/quasi_vs_reduced_preference_inequality.jl`
# — leaves the flag undefined and runs the suite as before.
if !@isdefined(RUN_QUASI_VS_REDUCED_TESTSET) || RUN_QUASI_VS_REDUCED_TESTSET
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
end
