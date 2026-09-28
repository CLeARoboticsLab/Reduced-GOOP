module Robotic_arm_final

# The robotic-arm pot carry, open loop or receding horizon, in one entry point.
#
#   julia --project=experiments -e 'include("experiments/Robotic_arm_final.jl"); Robotic_arm_final.demo()'
#   Robotic_arm_final.demo(receding_horizon = 10)      # 10 closed-loop steps
#   Robotic_arm_final.demo(plot_fig = false)           # no figures
#
# Step 1 is ScholtesReducedGOOP.jl's `demo()`: a 2-step warm-up solve (so the table shows
# solve times), the ρ sweep (a cold row per ρ, then the warm-z+eq chain) with one table
# line per row, and the plan picked as the source's demo picks it. `receding_horizon = 1`
# stops there (open loop). `receding_horizon = N > 1` then runs N closed-loop steps: the
# first knot of the current plan is executed (a perfect plant model), and the same
# compiled system (x₀ is the parameter θ) is re-solved from the executed state,
# warm-started from the previous plan shifted by one knot (warm-z+eq at the previous ρ,
# then coarser ρ), with a cold sweep as the fallback: from the shifted plan nudged into the
# strict interior (`interior_start`), or from the zero-control guess.
#
# Figures (`plot_fig`): the source's set for the step-1 sweep (a PDF and an HTML per
# initial guess and per row, plus robotic_arm_convergence.pdf), and for a receding run
# the executed closed-loop trajectory (robotic_arm_closed_loop.{pdf,html}). No data files
# are written.

using LinearAlgebra: norm
using Printf: @printf, @sprintf
using Dates: Dates
using ReducedGOOP

isdefined(Main, :RoboticArmCore) || Base.include(Main, joinpath(@__DIR__, "robotic_arm_core.jl"))
const Core_ = Main.RoboticArmCore

# The plotting stack is loaded only when figures are drawn, after every solve: a plotting
# package loaded up front invalidates precompiled Symbolics/ReducedGOOP code and slows the
# KKT build several-fold.
const PLOTTING_PATH = joinpath(@__DIR__, "Robotic_arm_plotting.jl")
"A function of `RoboticArmPlotting` (included on first use), looked up in the latest world."
function plotting(name::Symbol)
    isdefined(Main, :RoboticArmPlotting) || Base.include(Main, PLOTTING_PATH)
    P = Base.invokelatest(getglobal, Main, :RoboticArmPlotting)
    return Base.invokelatest(getglobal, P, name)
end

"The plan advanced one knot: x_{t+1} → x_t, u_{t+1} → u_t, the last state held, u_T = 0."
function shift_plan(sc, z)
    out = similar(z)
    T = sc.horizon
    for i in 1:2, t in 1:T
        src = min(t + 1, T)
        ks, kd = Core_.state_start(sc, i, src), Core_.state_start(sc, i, t)
        out[(kd + 1):(kd + Core_.NX[i])] .= z[(ks + 1):(ks + Core_.NX[i])]
        ks, kd = Core_.control_start(sc, i, src), Core_.control_start(sc, i, t)
        out[(kd + 1):(kd + Core_.NU[i])] .= t == T ? 0.0 : z[(ks + 1):(ks + Core_.NU[i])]
    end
    return out
end

"""
    warm_step(ctx, θ, z0, previous, coarser_rhos; eq_cols, linear_solver, max_inner)

One closed-loop warm start: warm-z+eq from `previous` shifted one knot, at `previous.rho`
and then each of `coarser_rhos`, stopping at the first converged solve. Returns
`(; label, result, attempts, shifted_z)`, `result === nothing` if none converged.
`demo`'s warm-up calls it too, so step 1 compiles exactly the code the MPC steps run.
"""
function warm_step(ctx, θ, z0, previous, coarser_rhos; eq_cols, linear_solver,
                   max_inner = Core_.MAX_INNER)
    shifted_z = shift_plan(ctx.scenario, previous.z)
    attempts = Any[]
    for ρ in unique((previous.rho, coarser_rhos...))
        w₀ = ReducedGOOP.scholtes_warm_start(ctx.kkt, θ, shifted_z, ρ;
                                              eq_cols, eq = previous.w[eq_cols])
        r = ReducedGOOP.solve(ReducedGOOP.Scholtes(), ctx.kkt, θ; z₀ = z0, w₀,
            options = Core_.solver_options(ctx, ρ; linear_solver, max_inner))
        push!(attempts, r)
        r.residual / sqrt(ctx.m) < Core_.TOL &&
            return (; label = @sprintf("warm (rho %.0e)", ρ), result = r, attempts, shifted_z)
    end
    return (; label = "", result = nothing, attempts, shifted_z)
end

"""
    interior_start(problem, z, θ, anchor; λs = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)) -> z′ or nothing

A cold start sets its slacks to s = g(z₀) and needs them STRICTLY positive, but a converged
plan sits on its active constraints (g ≈ 0 to solver tolerance), so its shift is rejected as
it stands. To make the shifted plan actually usable, pull `z` a little toward `anchor` 
(the zero-control guess: the executed state held at rest, zero speeds) and returns the first 
blend (1 − λ)·z + λ·anchor with every g > 0, for λ in `λs`, so the start stays mostly `z`. 
Returns `z` itself if it is already strictly feasible, and `nothing` if no blend is.
"""
function interior_start(problem, z, θ, anchor; λs = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5))
    strictly_feasible(v) = ReducedGOOP.is_feasible(problem, v, θ).worst > 0
    strictly_feasible(z) && return z
    for λ in λs
        blend = (1 - λ) .* z .+ λ .* anchor
        strictly_feasible(blend) && return blend
    end
    return nothing
end

"""
    closed_loop_trajectory(sc, executed, applied) -> (sc_exec, z_exec)

The executed run as a plan of its own: a scenario with `horizon = length(applied)` and the
primal vector holding the executed states `executed[2:end]` and the applied controls, so
the plan figures can draw it.
"""
function closed_loop_trajectory(sc, executed, applied)
    N = length(applied)
    fields = fieldnames(Core_.ScenarioConfig)
    sc_exec = Core_.ScenarioConfig(; merge(NamedTuple{fields}(getfield.(Ref(sc), fields)),
                                          (; horizon = N))...)
    z = zeros(sum((Core_.NX[i] + Core_.NU[i]) * N for i in 1:2))
    for t in 1:N, i in 1:2
        ks, kc = Core_.state_start(sc_exec, i, t), Core_.control_start(sc_exec, i, t)
        z[(ks + 1):(ks + Core_.NX[i])] .= executed[t + 1][i]
        z[(kc + 1):(kc + Core_.NU[i])] .= applied[t][i]
    end
    return sc_exec, z
end

"""
    demo(; receding_horizon = 1, plot_fig = true, output_dir = nothing,
         scenario_kwargs = (;), linear_solver = LINEAR_SOLVER, stop_at_tol = false,
         select = :goal_error, coarser_rhos = (1e-7, 1e-5))

- `receding_horizon`: `1` solves open loop; `N > 1` runs `N` closed-loop steps (step 1
  is the open-loop solve, and its plan's first knot is the first executed step).
- `plot_fig`: write figures to `output_dir` (default
  `data/robotic_arm_final/<timestamp>/`); `false` writes nothing.
- `scenario_kwargs`: keyword overrides of `ScenarioConfig` (horizon, x_init, d_min, …).
- `stop_at_tol`: step 1's sweep stops at the first converged row (fine → coarse)
  instead of running every row.
- `select`: how step 1 picks its plan among converged rows: `:goal_error` (closest to the goal) or
    `:residual` (smallest ‖K₀‖/√m, the source's MPC planner).
- `coarser_rhos`: the coarser ρ tried, after the previous (T-1) plan's ρ, when a closed-loop warm
  start does not converge; then a cold sweep following the default RHO_SWEEP.

Returns `nothing`.
"""
function demo(; receding_horizon::Integer = 1, plot_fig::Bool = true, output_dir = nothing,
              scenario_kwargs::NamedTuple = (;), linear_solver::Symbol = Core_.LINEAR_SOLVER,
              stop_at_tol::Bool = false, select::Symbol = :goal_error,
              coarser_rhos = (1e-7, 1e-5))
    receding_horizon >= 1 || throw(ArgumentError("receding_horizon must be >= 1"))
    select in (:goal_error, :residual) ||
        throw(ArgumentError("select must be :goal_error or :residual, got $select"))
    sc = Core_.ScenarioConfig(; scenario_kwargs...)
    output_dir = something(output_dir, joinpath(@__DIR__, "..", "data", "robotic_arm_final",
                                                Dates.format(Dates.now(), "yyyymmdd_HHMMSS")))
    θ = Core_.scenario_parameters(sc.x_init)

    # ── step 1: the source's open-loop demo ─────────────────────────────────────────────
    ctx = Core_.build_context(sc)
    scale = sqrt(ctx.m)
    k0(r) = r.residual / scale
    @printf("\n%d primal variables, residual rows m = %d (n_nc = %d, n_c = %d), sqrt(m) = %.3f\n",
            length(ctx.kkt.primal_dims), ctx.m, ctx.kkt.n_nc, ctx.kkt.n_comp, scale)
    @printf("KKT build + codegen %.1f s; linear_solver = %s, projected step\n\n",
            ctx.build_time, linear_solver)
    @info "Starting demo with initial (T = 1) scenario parameters"
    # The closed-loop warm step too (`warm_step`, the very function steps 2…N call): the
    # sweep only runs a warm start when its best cold row is not the finest ρ, so otherwise
    # the first closed-loop step would pay its compilation (~0.2 s against ~0.03 s solves).
    tj = @elapsed let z = Core_.zero_control_guess(sc, sc.x_init)
        r = ReducedGOOP.solve(ReducedGOOP.Scholtes(), ctx.kkt, θ; z₀ = z,
            options = Core_.solver_options(ctx, Core_.RHO_SWEEP[1]; linear_solver, max_inner = 2))
        warm_step(ctx, θ, z, r, coarser_rhos; eq_cols = Core_.eq_columns(ctx.kkt), linear_solver,
                  max_inner = 2)
    end
    @printf("solver specialization compiled in %.1fs (once per session)\n\n", tj)

    @printf("%-22s %-6s %-7s %-16s %-9s %-8s %-8s\n", "", "iters", "time_s", "||K_0||/sqrt(m)",
            "goal_err", "min_gap", "tilt")
    rows = Any[]
    cold_traces = Dict{Float64,NamedTuple}()
    t_sweep = @elapsed cold, warm = Core_.solve_rho_sweep(ctx, θ; linear_solver, stop_at_tol,
        collect_traces = plot_fig,
        on_result = (tag, ρ, r, t, tr) -> begin
            tag == "cold" && tr !== nothing && (cold_traces[ρ] = tr)
            m = Core_.plan_metrics(sc, r.z, θ)
            @printf("%-22s %-6d %-7.2f %-16.3e %-9.4f %-8.3f %-8.3f\n", @sprintf("rho = %.0e, %s", ρ, tag),
                    r.iters, t, k0(r), m.goal_error, m.min_gap, m.max_tilt)
            push!(rows, (; tag, rho = ρ, result = r, metrics = m))
        end)

    # Converged rows ranked by goal error (the source's demo) or by ‖K₀‖ (its planner);
    # if none converged, the cold RHO_REF row, or the smallest residual.
    converged = filter(row -> k0(row.result) < Core_.TOL, rows)
    ref_rows = filter(row -> row.tag == "cold" && row.rho == Core_.RHO_REF, rows)
    best = !isempty(converged) ?
           argmin(row -> select === :goal_error ? row.metrics.goal_error : row.result.residual, converged) :
           !isempty(ref_rows) ? first(ref_rows) : argmin(row -> row.result.residual, rows)
    isempty(converged) && @warn "no row met ||K_0||/sqrt(m) < $(Core_.TOL); the plan need not be feasible"
    m = best.metrics
    @printf("\nplan: rho = %.0e (%s), goal error %.4f, min gap %.3f (d_min %.2f), max tilt %.3f, handle drift %.1e, worst g %.1e\n",
            best.rho, best.tag, m.goal_error, m.min_gap, sc.d_min, m.max_tilt, m.max_handle_drift,
            min(m.min_safety, m.min_arm_speed, m.min_child_speed, m.min_child_reach))

    # ── steps 2…N: receding horizon ─────────────────────────────────────────────────────
    steps = [(; k = 1, mode = "sweep ($(best.tag))", result = best.result, time = t_sweep)]
    x = deepcopy(sc.x_init)
    executed = [deepcopy(x)]
    applied = Vector{Vector{Float64}}[]
    eq_cols = Core_.eq_columns(ctx.kkt)
    execute!(z) = begin
        push!(applied, [Core_.control(sc, z, 1, 1), Core_.control(sc, z, 2, 1)])
        x = [Core_.state(sc, z, 1, 1), Core_.state(sc, z, 2, 1)]
        push!(executed, deepcopy(x))
    end
    if receding_horizon > 1
        @printf("\nreceding horizon: %d closed-loop steps\n", receding_horizon)
        execute!(best.result.z)
        for k in 2:receding_horizon
            previous = steps[end].result
            θk = Core_.scenario_parameters(x)
            z0 = Core_.zero_control_guess(sc, x)
            mode, result, attempts = "", nothing, Any[]
            st = @timed begin
                # warm-z+eq from the shifted plan, at the previous ρ and then coarser ρ
                ws = warm_step(ctx, θk, z0, previous, coarser_rhos; eq_cols, linear_solver,
                               max_inner = Core_.MAX_INNER)  # the warm-up's keyword set
                shifted_z = ws.shifted_z
                append!(attempts, ws.attempts)
                ws.result === nothing || ((mode, result) = (ws.label, ws.result))
                # Fallback mechanism: Cold sweep from the shifted plan, nudged into the strict interior
                # (`interior_start`), or from the zero-control guess if no nudge works. A
                # cold start sets s = g(z₀) and γ = 0.1ρ/s, so it needs g(z₀) > 0 STRICTLY:
                # a start on an active state constraint (g = 0, e.g. the child on its reach
                # bound, which zero control holds at every knot) is rejected. If neither
                # start is strictly feasible, the sweep is skipped.
                cold_start = nothing
                if result === nothing
                    cold_start = something(interior_start(ctx.problem, shifted_z, θk, z0),
                                           ReducedGOOP.is_feasible(ctx.problem, z0, θk).worst > 0 ? z0 : Some(nothing))
                end
                # Re-solve the problem at the executed state θk from cold_start.
                if cold_start !== nothing
                    c, w = Core_.solve_rho_sweep(ctx, θk; z0 = cold_start, linear_solver,
                                                 stop_at_tol = true)
                    append!(attempts, vcat([r for (_, r) in c], [r for (_, _, r, _) in w]))
                    r = argmin(k0, attempts)
                    if k0(r) < Core_.TOL
                        mode = cold_start === z0 ? "sweep (zero control)" : "sweep (shifted plan)"
                        result = r
                    end
                end
                if result === nothing
                    mode, result = "best attempt (not converged)", argmin(k0, attempts)
                end
            end
            mk = Core_.plan_metrics(sc, result.z, θk)
            t = st.time
            @printf("step %2d  %-22s rho %.0e  %4d iters  %.3f s (JIT %.3f, GC %.3f)  ||K_0||/sqrt(m) %.2e  gap %.3f  goal %.4f\n",
                    k, mode, result.rho, result.iters, t, st.compile_time, st.gctime, k0(result),
                    mk.min_gap, mk.goal_error)
            push!(steps, (; k, mode, result, time = t))
            execute!(result.z)
        end
        times = [s.time for s in steps[2:end]]
        @printf("\nclosed loop: %d steps executed; per-step solve time after step 1: median %.3f s, max %.3f s; %d/%d warm-started\n",
                length(applied), sort(times)[cld(length(times), 2)], maximum(times),
                count(s -> startswith(s.mode, "warm"), steps), receding_horizon - 1)
        gap = minimum(norm(0.5 .* (e[1][1:3] .+ e[1][4:6])[1:2] .- e[2][1:2]) for e in executed)
        @printf("executed: final pot goal error %.4f, min horizontal pot-child gap %.3f (d_min %.2f)\n",
                norm(0.5 .* (x[1][1:3] .+ x[1][4:6]) .- sc.pot_goal), gap, sc.d_min)
    end

    # ── figures ─────────────────────────────────────────────────────────────────────────
    # These plots come only from the first MPC step's ρ sweep. 
    if plot_fig
        guesses = ("zero_control" => Core_.zero_control_guess(sc),
                   "direct_path" => Core_.direct_path_guess(sc))
        tp = @elapsed n = Base.invokelatest(plotting(:write_robotic_arm_plots), sc, cold, warm,
            cold_traces; max_inner = Core_.MAX_INNER, tol = Core_.TOL, tol_scale = scale,
            guesses, output_dir)
        if receding_horizon > 1
            sc_exec, z_exec = closed_loop_trajectory(sc, executed, applied)
            Base.invokelatest(plotting(:use_scenario!), sc_exec)
            Base.invokelatest(plotting(:plot_robotic_arm), z_exec; rho = NaN,
                              path = joinpath(output_dir, "robotic_arm_closed_loop.pdf"))
            Base.invokelatest(plotting(:plot_robotic_arm_interactive), z_exec; rho = NaN,
                              path = joinpath(output_dir, "robotic_arm_closed_loop.html"),
                              title = "pot carry,  closed loop ($(length(applied)) steps)")
            n += 2
        end
        @printf("\nwrote %d figures to %s in %.1f s\n", n, normpath(output_dir), tp)
    end
    return nothing
end

end # module Robotic_arm_final
