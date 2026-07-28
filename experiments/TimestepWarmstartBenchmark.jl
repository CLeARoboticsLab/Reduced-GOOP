module TimestepWarmstartBenchmark

using Dates: Dates
using JLD2: JLD2
using LinearAlgebra: norm
using Printf: @sprintf
using Statistics: mean, median, std

# Reuse an already-loaded (e.g. Revise-tracked via includet) receding-horizon
# module instead of baking in a private copy: `includet` is non-recursive, so a
# copy created by a plain `include` here would never pick up edits.
if isdefined(Main, :Robotic_arm_receding)
    const RR = Main.Robotic_arm_receding
else
    include(joinpath(@__DIR__, "Robotic_arm_receding.jl"))
    const RR = Robotic_arm_receding
end

"""
Planning grids compared by the benchmark. `T` is held fixed so both grids build
an identically sized KKT system; only the discretization step differs. The
coarse grid therefore looks 5× further into the future, but problem size is
removed as a confound — any measured difference is attributable to `Δt` and to
how far the state moves between consecutive MPC solves.
"""
const GRIDS = (
    (; label = "dt0.070", Δt = 0.07, planning_horizon = 20),
    (; label = "dt0.080", Δt = 0.08, planning_horizon = 20),
    (; label = "dt0.090", Δt = 0.09, planning_horizon = 20),
    (; label = "dt0.100", Δt = 0.1, planning_horizon = 20),
)
"""
Every dual warm-start strategy `Robotic_arm_receding.demo` supports. Note the
mode is spelled `:primal_only` (not `:primals_only`).
"""
const WARMSTART_MODES = RR.DUAL_WARMSTART_MODES

# ── Benchmark entry point ──────────────────────────────────────────────────────

"""
	run_benchmark(; kwargs...)

Sweep every (planning grid, dual warm-start mode) pair through the
receding-horizon robotic-arm MPC loop and report online solve cost.

Tests the hypothesis that a smaller `Δt` makes the shifted warm start more
effective and lowers the online solve time. All solver options, tolerances,
initial conditions and scenario parameters are held identical across cells;
only `Δt` and `dual_warmstart` vary. Plotting and per-step file output are
disabled.

The first MPC step of each cell is reported separately: it solves from the cold
default warm start *and* pays the one-time JIT of that cell's freshly generated
KKT residual code. The headline comparison uses steps `2:num_mpc_steps`, which
are the genuinely warm-started online solves.

Returns `(; results, summaries, verdict, run_dir)`.
"""
function run_benchmark(;
    num_mpc_steps = 20,
    grids = GRIDS,
    warmstart_modes = WARMSTART_MODES,
    # One throwaway solve to move generic solver/BLAS JIT out of the measured
    # cells. Per-cell KKT codegen is still charged to that cell's first step.
    warmup = false,
    save_summary = true,
    # Forwarded verbatim to every cell, so any override stays common to all of
    # them and cannot confound the comparison.
    demo_options = (;),
)
    num_mpc_steps >= 2 || throw(
        ArgumentError("Need at least 2 MPC steps to separate cold from warm solves."),
    )
    for mode in warmstart_modes
        mode in RR.DUAL_WARMSTART_MODES || throw(
            ArgumentError(
                "Unknown warm-start mode $(mode); expected one of " *
                "$(join(RR.DUAL_WARMSTART_MODES, ", ")).",
            ),
        )
    end

    run_id = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    run_dir = if save_summary
        dir = joinpath(dirname(@__DIR__), "runs", "timestep_warmstart_benchmark", run_id)
        mkpath(dir)
        dir
    else
        nothing
    end

    if warmup
        @info "Warm-up solve (discarded) to absorb generic JIT..."
        _run_cell(
            (; label = "warmup", Δt = 0.5, planning_horizon = 4),
            first(warmstart_modes),
            2,
            demo_options,
        )
    end

    trajectory_dir = if save_summary
        dir = joinpath(run_dir, "trajectories")
        mkpath(dir)
        dir
    else
        nothing
    end

    results = NamedTuple[]
    for grid in grids, mode in warmstart_modes
        @info "Benchmarking Δt=$(grid.Δt), T=$(grid.planning_horizon), warmstart=$(mode)"
        result = _run_cell(grid, mode, num_mpc_steps, demo_options)
        push!(results, result)
        if save_summary && isnothing(result.error)
            _save_trajectory(result, trajectory_dir)
        end
    end

    summaries = map(_summarize, results)
    verdict = _verdict(summaries)
    report = _report(summaries, verdict; num_mpc_steps)
    println(report)

    if save_summary
        write(joinpath(run_dir, "summary.md"), report)
        write(joinpath(run_dir, "results.csv"), _csv(summaries))
        JLD2.save_object(
            joinpath(run_dir, "benchmark_results.jld2"),
            Dict(
                "run_id" => run_id,
                "num_mpc_steps" => num_mpc_steps,
                "grids" => collect(grids),
                "warmstart_modes" => collect(warmstart_modes),
                "demo_options" => demo_options,
                "results" => results,
                "summaries" => summaries,
                "verdict" => verdict,
            ),
        )
        println("benchmark outputs saved under: ", run_dir)
    end

    (; results, summaries, verdict, run_dir)
end

# ── Cell execution ─────────────────────────────────────────────────────────────

"""
Run one (grid, warm-start mode) cell. Failures are captured rather than
propagated so a single bad cell does not abort the sweep.
"""
function _run_cell(grid, mode, num_mpc_steps, demo_options)
    cell = (;
        grid_label = grid.label,
        Δt = grid.Δt,
        planning_horizon = grid.planning_horizon,
        mode,
    )
    wall_time = 0.0
    output = try
        wall_time = @elapsed out = RR.demo(;
            num_mpc_steps,
            planning_horizon = grid.planning_horizon,
            Δt = grid.Δt,
            dual_warmstart = mode,
            save_outputs = false,
            show_interactive_trajectory = false,
            verbose = false,
            demo_options...,
        )
        out
    catch err
        @error "cell failed" grid_label = grid.label mode exception =
            (err, catch_backtrace())
        return merge(cell, (; error = sprint(showerror, err), wall_time))
    end

    merge(
        cell,
        (;
            error = nothing,
            wall_time,
            step_solve_times = output.step_solve_times,
            step_total_iters = output.step_total_iters,
            step_attempt_iters = output.step_attempt_iters,
            step_statuses = output.step_statuses,
            step_kkt_errors = output.step_kkt_errors,
            step_rescue_statuses = output.step_rescue_statuses,
            kkt_dimension = output.kkt_dimension,
            variable_dimension = output.variable_dimension,
            kkt_build_time_sec = output.kkt_build_time_sec,
            problem_setup_time_sec = output.problem_setup_time_sec,
            # Mirrors the demo default; needed to tell a capped step apart from a
            # genuine stall when summarizing.
            max_inner_iters = get(demo_options, :max_inner_iters, 500),
            # Kept so trajectory quality can be inspected independently of solver
            # cost: a cheap solve that never approaches the goal is not a win.
            closed_loop_strategies = output.closed_loop_strategies,
        ),
    )
end

# ── Trajectory quality ─────────────────────────────────────────────────────────

"""
Goal positions for the scenario. Independent of `Δt` and `T`, so they are read
once from the shared scenario config rather than duplicated here.
"""
function _goal_positions()
    sc = RR.RA.demo_scenario_config()
    (; sc.goal_position1, sc.goal_position2, sc.goal_position3)
end

"""
Split a closed-loop state history into the three physical bodies. Player 1
stacks both arms as `[arm1; arm2] ∈ R⁶`; player 2 is the child in `R³`.
"""
function _bodies(closed_loop_strategies)
    arm_states = closed_loop_strategies[1].xs
    child_states = closed_loop_strategies[2].xs
    (;
        arm1 = [collect(x[1:3]) for x in arm_states],
        arm2 = [collect(x[4:6]) for x in arm_states],
        child = [collect(x[1:3]) for x in child_states],
    )
end

"""
Distance-to-goal summary for one closed-loop run. `final_*` is where the body
ended after all MPC steps; `min_*` is the closest it ever got, which separates
"never approached" from "approached then drifted".
"""
function _trajectory_metrics(closed_loop_strategies)
    goals = _goal_positions()
    bodies = _bodies(closed_loop_strategies)
    distances(path, goal) = [norm(p .- goal) for p in path]
    d1 = distances(bodies.arm1, goals.goal_position1)
    d2 = distances(bodies.arm2, goals.goal_position2)
    d3 = distances(bodies.child, goals.goal_position3)
    (;
        initial_dist_arm1 = d1[1],
        initial_dist_arm2 = d2[1],
        final_dist_arm1 = d1[end],
        final_dist_arm2 = d2[end],
        final_dist_child = d3[end],
        min_dist_arm1 = minimum(d1),
        min_dist_arm2 = minimum(d2),
        # Fraction of the initial gap the arms actually closed.
        progress_arm1 = 1 - d1[end] / d1[1],
        progress_arm2 = 1 - d2[end] / d2[1],
    )
end

"""
Write one cell's closed-loop trajectory as JLD2 (full states and controls) plus
a flat CSV of body positions for quick inspection.
"""
function _save_trajectory(result, trajectory_dir)
    tag = "$(result.grid_label)_$(result.mode)"
    JLD2.save_object(
        joinpath(trajectory_dir, "trajectory_$(tag).jld2"),
        Dict(
            "grid_label" => result.grid_label,
            "dt" => result.Δt,
            "planning_horizon" => result.planning_horizon,
            "dual_warmstart" => result.mode,
            "closed_loop_strategies" => result.closed_loop_strategies,
            "step_statuses" => result.step_statuses,
            "step_total_iters" => result.step_total_iters,
            "step_solve_times" => result.step_solve_times,
            "goals" => _goal_positions(),
        ),
    )

    bodies = _bodies(result.closed_loop_strategies)
    io = IOBuffer()
    println(
        io,
        "step,t,arm1_x,arm1_y,arm1_z,arm2_x,arm2_y,arm2_z,child_x,child_y,child_z",
    )
    for k in eachindex(bodies.arm1)
        a1, a2, c = bodies.arm1[k], bodies.arm2[k], bodies.child[k]
        println(io, join((k - 1, (k - 1) * result.Δt, a1..., a2..., c...), ","))
    end
    write(joinpath(trajectory_dir, "trajectory_$(tag).csv"), String(take!(io)))
end

# ── Summarization ──────────────────────────────────────────────────────────────

_stats(v) = (;
    mean = mean(v),
    median = median(v),
    min = minimum(v),
    max = maximum(v),
    std = length(v) > 1 ? std(v) : 0.0,
    total = sum(v),
)

"""
Reduce one cell to the quantities the hypothesis is judged on.

`warm_*` covers steps 2:end (shifted warm start). `cold_*` is step 1, which
solves from the default warm start. `iter_speedup` is the clean measure of
warm-start effectiveness — iteration counts are unaffected by the one-time JIT
that inflates step 1's wall time. `realtime_factor` is the online-feasibility
measure: solve time divided by the control period it must fit inside.
"""
function _summarize(result)
    isnothing(result.error) || return merge(
        (; result.grid_label, result.Δt, result.planning_horizon, result.mode),
        (; failed = true, error = result.error),
    )

    times = result.step_solve_times
    iters = result.step_total_iters
    warm_times = times[2:end]
    warm_iters = iters[2:end]
    warm_statuses = result.step_statuses[2:end]
    time_stats = _stats(warm_times)
    iter_stats = _stats(warm_iters)

    (;
        result.grid_label,
        result.Δt,
        result.planning_horizon,
        result.mode,
        failed = false,
        error = nothing,
        result.kkt_dimension,
        result.variable_dimension,
        result.kkt_build_time_sec,
        cold_time = times[1],
        cold_iters = iters[1],
        cold_status = result.step_statuses[1],
        warm_time_mean = time_stats.mean,
        warm_time_median = time_stats.median,
        warm_time_min = time_stats.min,
        warm_time_max = time_stats.max,
        warm_time_std = time_stats.std,
        warm_time_total = time_stats.total,
        warm_iters_mean = iter_stats.mean,
        warm_iters_median = iter_stats.median,
        warm_iters_max = iter_stats.max,
        # > 1 means the warm start saved work relative to the cold solve.
        iter_speedup = iters[1] / iter_stats.median,
        time_speedup = times[1] / time_stats.median,
        # Seconds of solve per second of control period; < 1 is real-time capable.
        realtime_factor = time_stats.mean / result.Δt,
        warm_converged = count(==(:solved), warm_statuses),
        warm_steps = length(warm_statuses),
        rescues = count(!=(:not_run), result.step_rescue_statuses),
        max_kkt_error = maximum(result.step_kkt_errors),
        # Steps that exhausted the Newton budget rather than converging or
        # stalling: these measure the cost of the cap, not a solve time.
        capped_steps = count(>=(result.max_inner_iters - 1), iters),
        max_inner_iters = result.max_inner_iters,
        _trajectory_metrics(result.closed_loop_strategies)...,
    )
end

# ── Verdict ────────────────────────────────────────────────────────────────────

"""
Judge the hypothesis per warm-start mode as a monotonic trend across `Δt`.

The hypothesis "smaller Δt warm-starts better and solves faster" predicts that
online cost rises monotonically with `Δt`: sorting cells by ascending `Δt`, both
the median warm solve time and the median warm iteration count should be
non-decreasing. Anything else (flat, non-monotone, or decreasing) refutes it.
Generalizes to any number of grids, so a two-grid sweep reduces to the obvious
pairwise comparison.
"""
function _verdict(summaries)
    usable = filter(s -> !s.failed, summaries)
    modes = unique(s.mode for s in usable)
    per_mode = map(modes) do mode
        cells = sort(filter(s -> s.mode == mode, usable); by = s -> s.Δt)
        length(cells) < 2 && return (; mode, comparable = false)
        Δts = [c.Δt for c in cells]
        times = [c.warm_time_median for c in cells]
        iters = [c.warm_iters_median for c in cells]
        realtime = [c.realtime_factor for c in cells]
        (;
            mode,
            comparable = true,
            Δts,
            warm_time_medians = times,
            warm_iters_medians = iters,
            realtime_factors = realtime,
            time_supported = issorted(times),
            iters_supported = issorted(iters),
            realtime_supported = issorted(realtime),
            fastest_Δt = Δts[argmin(times)],
            fewest_iters_Δt = Δts[argmin(iters)],
        )
    end
    comparable = filter(v -> v.comparable, per_mode)
    (;
        per_mode,
        time_supported_count = count(v -> v.time_supported, comparable),
        iters_supported_count = count(v -> v.iters_supported, comparable),
        realtime_supported_count = count(v -> v.realtime_supported, comparable),
        comparable_count = length(comparable),
    )
end

# ── Reporting ──────────────────────────────────────────────────────────────────

_grid_name(s) = "Δt=$(s.Δt), T=$(s.planning_horizon)"

function _report(summaries, verdict; num_mpc_steps)
    io = IOBuffer()
    println(io, "# Δt / warm-start benchmark (receding-horizon robotic arm)\n")
    println(
        io,
        "`num_mpc_steps = $(num_mpc_steps)`; step 1 is the cold solve, ",
        "steps 2–$(num_mpc_steps) are the warm-started online solves.\n",
    )

    println(io, "## Warm online solves (steps 2–$(num_mpc_steps))\n")
    println(
        io,
        "| grid | warm start | KKT dim | warm mean (s) | warm median (s) | warm min–max (s) | warm iters (med) | conv | capped | max ‖F‖ |",
    )
    println(io, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries
        if s.failed
            println(
                io,
                "| $(_grid_name(s)) | $(s.mode) | — | FAILED: $(s.error) | | | | | | |",
            )
            continue
        end
        println(
            io,
            "| $(_grid_name(s)) | $(s.mode) | $(s.kkt_dimension) | ",
            @sprintf("%.4f", s.warm_time_mean),
            " | ",
            @sprintf("%.4f", s.warm_time_median),
            " | ",
            @sprintf("%.4f–%.4f", s.warm_time_min, s.warm_time_max),
            " | ",
            @sprintf("%.1f", s.warm_iters_median),
            " | ",
            "$(s.warm_converged)/$(s.warm_steps) | ",
            "$(s.capped_steps)/$(num_mpc_steps) | ",
            @sprintf("%.2e", s.max_kkt_error),
            " |",
        )
    end

    println(io, "\n## Cold first solve vs. warm solves\n")
    println(
        io,
        "| grid | warm start | cold time (s) | cold iters | warm iters (med) | iter speedup | time speedup | solve/Δt |",
    )
    println(io, "|---|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries
        s.failed && continue
        println(
            io,
            "| $(_grid_name(s)) | $(s.mode) | ",
            @sprintf("%.3f", s.cold_time),
            " | ",
            "$(s.cold_iters) | ",
            @sprintf("%.1f", s.warm_iters_median),
            " | ",
            @sprintf("%.2f×", s.iter_speedup),
            " | ",
            @sprintf("%.2f×", s.time_speedup),
            " | ",
            @sprintf("%.2f", s.realtime_factor),
            " |",
        )
    end
    println(
        io,
        "\n`iter speedup` = cold iters / median warm iters (warm-start effectiveness; ",
        "immune to the one-time JIT that inflates the cold wall time). ",
        "`solve/Δt` < 1 means the loop keeps up with its own control period.\n",
    )

    println(io, "\n## Trajectory quality (closed loop)\n")
    println(
        io,
        "| grid | warm start | arm1 final dist | arm2 final dist | arm1 closest | arm2 closest | arm1 progress | arm2 progress | child final dist |",
    )
    println(io, "|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries
        s.failed && continue
        println(
            io,
            "| $(_grid_name(s)) | $(s.mode) | ",
            @sprintf("%.3f", s.final_dist_arm1),
            " | ",
            @sprintf("%.3f", s.final_dist_arm2),
            " | ",
            @sprintf("%.3f", s.min_dist_arm1),
            " | ",
            @sprintf("%.3f", s.min_dist_arm2),
            " | ",
            @sprintf("%.1f%%", 100 * s.progress_arm1),
            " | ",
            @sprintf("%.1f%%", 100 * s.progress_arm2),
            " | ",
            @sprintf("%.3f", s.final_dist_child),
            " |",
        )
    end
    if !isempty(summaries) && !first(summaries).failed
        println(
            io,
            @sprintf(
                "\nInitial gap to goal: arm1 %.3f, arm2 %.3f. `progress` = fraction of that gap closed.\n",
                first(summaries).initial_dist_arm1,
                first(summaries).initial_dist_arm2
            ),
        )
    end

    println(io, "## Hypothesis\n")
    println(
        io,
        "Claim: a smaller Δt warm-starts better and solves faster online. ",
        "Supported only if cost rises monotonically with Δt.\n",
    )
    for v in verdict.per_mode
        if !v.comparable
            println(io, "- `$(v.mode)`: not comparable (a cell failed).")
            continue
        end
        trace(values, fmt) = join(
            (@sprintf("Δt=%g: %s", d, fmt(x)) for (d, x) in zip(v.Δts, values)),
            ", ",
        )
        println(
            io,
            "- `$(v.mode)`: online solve time ",
            v.time_supported ? "**supported**" : "**not supported**",
            " (",
            trace(v.warm_time_medians, x -> @sprintf("%.4f s", x)),
            "); ",
            "warm iterations ",
            v.iters_supported ? "**supported**" : "**not supported**",
            " (",
            trace(v.warm_iters_medians, x -> @sprintf("%.1f", x)),
            "); ",
            "real-time margin ",
            v.realtime_supported ? "**supported**" : "**not supported**",
            " (",
            trace(v.realtime_factors, x -> @sprintf("%.2f", x)),
            "). ",
            @sprintf("Fastest at Δt=%g.", v.fastest_Δt),
        )
    end
    println(
        io,
        "\nOverall: online solve time supported in ",
        "$(verdict.time_supported_count)/$(verdict.comparable_count) modes, ",
        "warm iterations in $(verdict.iters_supported_count)/$(verdict.comparable_count), ",
        "real-time margin in $(verdict.realtime_supported_count)/$(verdict.comparable_count).",
    )
    String(take!(io))
end

function _csv(summaries)
    io = IOBuffer()
    println(
        io,
        "grid_label,dt,planning_horizon,warmstart,failed,kkt_dimension,kkt_build_time_sec,",
        "cold_time,cold_iters,warm_time_mean,warm_time_median,warm_time_min,warm_time_max,",
        "warm_time_std,warm_iters_mean,warm_iters_median,iter_speedup,time_speedup,",
        "realtime_factor,warm_converged,warm_steps,rescues,max_kkt_error,capped_steps,max_inner_iters,",
        "final_dist_arm1,final_dist_arm2,final_dist_child,min_dist_arm1,min_dist_arm2,",
        "progress_arm1,progress_arm2",
    )
    for s in summaries
        if s.failed
            println(
                io,
                "$(s.grid_label),$(s.Δt),$(s.planning_horizon),$(s.mode),true",
                ","^27,
            )
            continue
        end
        println(
            io,
            join(
                (
                    s.grid_label,
                    s.Δt,
                    s.planning_horizon,
                    s.mode,
                    false,
                    s.kkt_dimension,
                    s.kkt_build_time_sec,
                    s.cold_time,
                    s.cold_iters,
                    s.warm_time_mean,
                    s.warm_time_median,
                    s.warm_time_min,
                    s.warm_time_max,
                    s.warm_time_std,
                    s.warm_iters_mean,
                    s.warm_iters_median,
                    s.iter_speedup,
                    s.time_speedup,
                    s.realtime_factor,
                    s.warm_converged,
                    s.warm_steps,
                    s.rescues,
                    s.max_kkt_error,
                    s.capped_steps,
                    s.max_inner_iters,
                    s.final_dist_arm1,
                    s.final_dist_arm2,
                    s.final_dist_child,
                    s.min_dist_arm1,
                    s.min_dist_arm2,
                    s.progress_arm1,
                    s.progress_arm2,
                ),
                ",",
            ),
        )
    end
    String(take!(io))
end

end
