#=
Convergence plots for `quasi_vs_reduced_preference_inequality.jl`
=================================================================

Re-runs the comparison of that file — three-player, three-level benchmarks with
the hard inequality moved into the innermost preference level, solved with both
`:reduced` and `:quasi` — with `record_convergence = true`, and plots the KKT
residual ‖F(z)‖₂ against the Newton iteration count.

Nothing about the problems, the solver configuration, or the initial guesses is
re-specified here: `innermost_preference_inequality_goop`,
`comparison_solver_options`, and `initial_guesses` are taken from the comparison
file itself, which is included with `RUN_QUASI_VS_REDUCED_TESTSET = false` so its
`@testset` does not re-run underneath this script. The only knob added is
`record_convergence`.

The figures come from `experiments/Plotting.jl` (reached through
`experiments/Intersection.jl`, which is where `safe_log10_history` and
`save_figure` live). One figure is written per problem family and initial guess,
overlaying the two formulations' traces; each plots log₁₀‖F‖₂, matching
`save_convergence_diagnostics` in `Intersection.jl`. Trace-averaging across
initial guesses is deliberately not done — the three starts converge to
different points, so a mean trace over them describes no single solve.
`:quasi` is drawn dashed because on these problems the two traces agree to about
thirteen digits: a solid overlay would hide `:reduced` entirely. That agreement is
the comparison file's own finding — the dropped higher-order terms carry policy
multipliers, and the multipliers never grow along this solver path.

CairoMakie lives in the `experiments` environment, not in the root or `test`
ones, so this script runs from there (`test/Manifest.toml` must stay absent):

    julia --project=experiments test/quasi_vs_reduced_convergence_plot.jl

Figures land in `test/plots/quasi_vs_reduced_convergence/<timestamp>/`; runs are
never overwritten.
=#

using Dates: format, now

using ReducedGOOP

# Definitions only — the comparison suite itself is skipped. See the guard at the
# bottom of the included file.
const RUN_QUASI_VS_REDUCED_TESTSET = false
include(joinpath(@__DIR__, "quasi_vs_reduced_preference_inequality.jl"))

# `Intersection` is included for its plotting layer alone: `plot_convergence_plot`,
# `plot_convergence_plot_aggregate_comparison`, `safe_log10_history`, and
# `save_figure`. Including the module runs no experiment.
include(joinpath(@__DIR__, "..", "experiments", "Intersection.jl"))

"The comparison configuration, with per-iteration KKT-error recording turned on."
const CONVERGENCE_OPTIONS = comparison_solver_options(; record_convergence = true)

"Problem families plotted — the two exercised by the comparison suite."
const PLOTTED_PROBLEMS = [
    (; num_players = 3, levels = 3, kind = :quadratic),
    (; num_players = 3, levels = 3, kind = :nonlinear),
]

const VARIANTS = (:reduced, :quasi)

const VARIANT_LABELS = (; reduced = "Reduced", quasi = "Quasi")

"""
    kkt_system(goop, variant)

Build the `:reduced` or `:quasi` KKT system for `goop`. Built once per
(problem, variant) and reused across initial guesses — the generators depend on
the `ParametricGOOP` alone, and each build costs far more than a solve.
"""
function kkt_system(goop, variant::Symbol)
    generator = if variant === :reduced
        ReducedGOOP.generate_slacked_reduced_kkt_system
    elseif variant === :quasi
        ReducedGOOP.generate_slacked_quasi_kkt_system
    else
        error("Unknown variant: $variant")
    end
    return generator(goop)
end

"""
    solve_recording(kkt, goop; z₀, options)

Solve `kkt` from `z₀` and return the per-iteration KKT-error trace alongside the
usual summary fields. A solve that throws (a singular factorization on a
diverging iterate) is reported as `:errored` with whatever trace it produced, so
one bad configuration does not abort the whole sweep.
"""
function solve_recording(kkt, goop; z₀, options = CONVERGENCE_OPTIONS)
    output = try
        ReducedGOOP.solve(
            ReducedGOOP.InteriorPoint(),
            kkt,
            zeros(sum(goop.parameter_dims));
            z₀,
            options,
        )
    catch err
        err isa InterruptException && rethrow()
        @warn "solve threw" exception = err
        (;
            status = :errored,
            kkt_error = NaN,
            total_iters = 0,
            kkt_error_history = Float64[],
        )
    end

    return (;
        status = output.status,
        kkt_error = output.kkt_error,
        total_iters = output.total_iters,
        kkt_error_history = collect(output.kkt_error_history),
    )
end

"""
    convergence_traces(; num_players, levels, kind)

Solve one benchmark family with both formulations from every initial guess of
`initial_guesses`, returning the recorded traces grouped by initial guess.
"""
function convergence_traces(; num_players::Int, levels::Int, kind::Symbol)
    case = build_benchmark_problem(; num_players, levels, kind)
    goop = innermost_preference_inequality_goop(case.problem)

    systems = map(VARIANTS) do variant
        build_time = @elapsed kkt = kkt_system(goop, variant)
        @info "built KKT system" kind variant build_time dimension =
            kkt.variable_dimension
        variant => kkt
    end
    systems = Dict(systems)

    runs = map(initial_guesses(case)) do (start_name, z₀)
        solves = map(VARIANTS) do variant
            solve = solve_recording(systems[variant], goop; z₀)
            @info "solved" kind start_name variant solve.status solve.kkt_error solve.total_iters
            variant => solve
        end
        (; start_name, solves = Dict(solves))
    end

    return (; kind, case, goop, runs)
end

"""
    save_convergence_figures(traces, plots_dir)

Write one overlay per initial guess for one problem family, and return the paths
written.
"""
function save_convergence_figures(traces, plots_dir)
    (; kind, runs) = traces
    mkpath(plots_dir)
    written = String[]

    log_trace(solve) = Intersection.safe_log10_history(solve.kkt_error_history)
    has_trace(solve) = !isempty(solve.kkt_error_history)

    for run in runs
        reduced = run.solves[:reduced]
        quasi = run.solves[:quasi]
        (has_trace(reduced) && has_trace(quasi)) || continue

        figure, _ = Intersection.plot_convergence_plot_aggregate_comparison(;
            reduced_kkt_error_histories = [log_trace(reduced)],
            complete_kkt_error_histories = [log_trace(quasi)],
            reduced_label = VARIANT_LABELS.reduced,
            complete_label = VARIANT_LABELS.quasi,
            complete_linestyle = :dash,
        )
        path = joinpath(plots_dir, "convergence_$(kind)_z0-$(run.start_name).pdf")
        Intersection.save_figure(path, figure)
        push!(written, path)
    end

    return written
end

"Tabulate where each trace ended, so the figures can be read without the log."
function report_traces(all_traces)
    println("\n", "="^78)
    println("  Recorded convergence traces — tol = $COMPARISON_TOLERANCE")
    println("="^78)
    println(
        rpad("  problem", 12),
        rpad("z₀", 10),
        rpad("variant", 10),
        rpad("status", 10),
        rpad("iters", 8),
        "final ‖F‖₂",
    )
    for traces in all_traces, run in traces.runs, variant in VARIANTS
        solve = run.solves[variant]
        println(
            rpad("  $(traces.kind)", 12),
            rpad(string(run.start_name), 10),
            rpad(string(variant), 10),
            rpad(string(solve.status), 10),
            rpad(string(solve.total_iters), 8),
            string(round(solve.kkt_error; sigdigits = 3)),
        )
    end
    println()
end

function main()
    run_id = format(now(), "yyyy-mm-dd_HH-MM-SS")
    plots_dir = joinpath(@__DIR__, "plots", "quasi_vs_reduced_convergence", run_id)

    all_traces = map(PLOTTED_PROBLEMS) do problem
        traces = convergence_traces(; problem...)
        save_convergence_figures(traces, plots_dir)
        traces
    end

    report_traces(all_traces)
    println("  figures written to $(plots_dir)")
    for path in sort(readdir(plots_dir))
        println("    $path")
    end
    println()

    return (; plots_dir, all_traces)
end

main()
