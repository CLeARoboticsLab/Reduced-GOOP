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
# then coarser ρ), with a cold sweep as the fallback.
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
         select = :goal_error, warm_rhos = (1e-7, 1e-5))

- `receding_horizon`: `1` solves open loop; `N > 1` runs `N` closed-loop steps (step 1
  is the open-loop solve, and its plan's first knot is the first executed step).
- `plot_fig`: write figures to `output_dir` (default
  `data/robotic_arm_final/<timestamp>/`); `false` writes nothing.
- `scenario_kwargs`: keyword overrides of `ScenarioConfig` (horizon, x_init, d_min, …).
- `stop_at_tol`: step 1's sweep stops at the first converged row (fine → coarse)
  instead of running every row.
- `select`: how step 1 picks its plan among converged rows: `:goal_error` (the source's
  `demo()`) or `:residual` (smallest ‖K₀‖/√m, the source's MPC planner).
- `warm_rhos`: the coarser ρ tried, after the previous plan's ρ, when a closed-loop warm
  start does not converge; then a cold sweep.

Returns `nothing`.
"""
function demo(; receding_horizon::Integer = 1, plot_fig::Bool = true, output_dir = nothing,
              scenario_kwargs::NamedTuple = (;), linear_solver::Symbol = Core_.LINEAR_SOLVER,
              stop_at_tol::Bool = false, select::Symbol = :goal_error,
              warm_rhos = (1e-7, 1e-5))
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
    tj = @elapsed ReducedGOOP.solve(ReducedGOOP.Scholtes(), ctx.kkt, θ;
        z₀ = Core_.zero_control_guess(sc, sc.x_init),
        options = Core_.solver_options(ctx, Core_.RHO_SWEEP[1]; linear_solver, max_inner = 2))
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
            t = @elapsed begin
                # warm-z+eq from the shifted plan, at the previous ρ and then coarser ρ (a
                # coarser ρ buys basin); the warm start floors the slacks, so it needs no
                # strictly feasible point
                shifted = shift_plan(sc, previous.z)
                for ρ in unique((previous.rho, warm_rhos...))
                    w₀ = ReducedGOOP.scholtes_warm_start(ctx.kkt, θk, shifted, ρ;
                                                          eq_cols, eq = previous.w[eq_cols])
                    r = ReducedGOOP.solve(ReducedGOOP.Scholtes(), ctx.kkt, θk; z₀ = z0, w₀,
                        options = Core_.solver_options(ctx, ρ; linear_solver))
                    push!(attempts, r)
                    if k0(r) < Core_.TOL
                        mode, result = @sprintf("warm (rho %.0e)", ρ), r
                        break
                    end
                end
                # cold sweep, only from a strictly feasible zero-control guess (a state
                # executed ON an active constraint makes it infeasible)
                if result === nothing && ReducedGOOP.is_feasible(ctx.problem, z0, θk).worst > 0
                    c, w = Core_.solve_rho_sweep(ctx, θk; z0, linear_solver, stop_at_tol = true)
                    append!(attempts, vcat([r for (_, r) in c], [r for (_, _, r, _) in w]))
                    r = argmin(k0, attempts)
                    if k0(r) < Core_.TOL
                        mode, result = "sweep", r
                    end
                end
                if result === nothing
                    mode, result = "best attempt (not converged)", argmin(k0, attempts)
                end
            end
            mk = Core_.plan_metrics(sc, result.z, θk)
            @printf("step %2d  %-22s rho %.0e  %4d iters  %.3f s  ||K_0||/sqrt(m) %.2e  gap %.3f  goal %.4f\n",
                    k, mode, result.rho, result.iters, t, k0(result), mk.min_gap, mk.goal_error)
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
