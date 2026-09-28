module Robotic_arm

# Open-loop robotic-arm pot carry: the ρ sweep of robotic_arm_core.jl, reported the way
# ScholtesReducedGOOP.jl's `demo()` reports it, with data and figures saved per run.
#
#   julia --project=experiments -e 'include("experiments/Robotic_arm.jl"); Robotic_arm.demo()'
#
# GOOP_PLOT=0 skips the figures. The figures are the source's: one PDF and one HTML per
# initial guess and per solve (Robotic_arm_plotting.jl).

using JLD2: jldsave
using LinearAlgebra: norm
using Printf: @printf, @sprintf
using Dates: Dates
using ReducedGOOP

const ROBOTIC_ARM_CORE_PATH = joinpath(@__DIR__, "robotic_arm_core.jl")
isdefined(Main, :RoboticArmCore) || Base.include(Main, ROBOTIC_ARM_CORE_PATH)
using Main.RoboticArmCore
const Core_ = Main.RoboticArmCore
# The plotting stack is loaded only when figures are drawn, after every solve: a plotting
# package loaded up front invalidates precompiled Symbolics/ReducedGOOP code and slows
# the KKT build several-fold.
const PLOTTING_PATH = joinpath(@__DIR__, "Robotic_arm_plotting.jl")
"A function of `RoboticArmPlotting` (included on first use), looked up in the latest world."
function plotting(name::Symbol)
    isdefined(Main, :RoboticArmPlotting) || Base.include(Main, PLOTTING_PATH)
    P = Base.invokelatest(getglobal, Main, :RoboticArmPlotting)
    return Base.invokelatest(getglobal, P, name)
end

"""
    demo(; scenario_kwargs = (;), linear_solver = LINEAR_SOLVER, stop_at_tol = false,
         run_id = nothing, save = true, plot = save && ENV["GOOP_PLOT"] != "0")

Build the scenario (keyword overrides of `ScenarioConfig` in `scenario_kwargs`), run the ρ
sweep (a cold row per ρ, then the warm-z+eq chain), print one line per row, pick the
converged row with the smallest goal error (as the source's `demo()` does; the MPC
planner in Robotic_arm_receding.jl ranks by ‖K₀‖/√m, as the source's planner does), and
save the rows, the chosen plan and its metrics to `data/robotic_arm_scholtes/<run_id>/`
(`sweep.jld2`), with the source's figures beside it when `plot`: a PDF and an HTML for
each initial guess and each row. Returns `nothing`, as the source's `demo()` does.
`save = false` writes nothing (no .jld2, no figures), as the source's `demo()` writes
nothing without its plotting environment.
"""
function demo(; scenario_kwargs::NamedTuple = (;), linear_solver::Symbol = LINEAR_SOLVER,
              stop_at_tol::Bool = false, run_id = nothing, save::Bool = true,
              plot::Bool = save && get(ENV, "GOOP_PLOT", "1") != "0")
    sc = Core_.ScenarioConfig(; scenario_kwargs...)
    run_id = something(run_id, Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    run_dir = joinpath(@__DIR__, "..", "data", "robotic_arm_scholtes", run_id)
    θ = Core_.scenario_parameters(sc.x_init)

    ctx = Core_.build_context(sc)
    scale = sqrt(ctx.m)
    @printf("\n%d primal variables, residual rows m = %d (n_nc = %d, n_c = %d), sqrt(m) = %.3f\n",
            length(ctx.kkt.primal_dims), ctx.m, ctx.kkt.n_nc, ctx.kkt.n_comp, scale)
    @printf("KKT build + codegen %.1f s; linear_solver = %s, projected step\n\n", ctx.build_time, linear_solver)
    # As the source's demo(): compile the solver for this system with a 2-step solve, so
    # the table's times are solve times.
    tj = @elapsed ReducedGOOP.solve(ReducedGOOP.Scholtes(), ctx.kkt, θ;
        z₀ = Core_.zero_control_guess(sc, sc.x_init),
        options = Core_.solver_options(ctx, Core_.RHO_SWEEP[1]; linear_solver, max_inner = 2))
    @printf("solver specialization compiled in %.1fs (once per session)\n\n", tj)
    @printf("%-22s %-6s %-7s %-16s %-9s %-8s %-8s\n", "", "iters", "time_s", "||K_0||/sqrt(m)",
            "goal_err", "min_gap", "tilt")
    rows = Any[]
    cold_traces = Dict{Float64,NamedTuple}()
    cold, warm = Core_.solve_rho_sweep(ctx, θ; linear_solver, stop_at_tol,
        collect_traces = plot,
        on_result = (tag, ρ, r, t, tr) -> begin
            tag == "cold" && tr !== nothing && (cold_traces[ρ] = tr)
            m = Core_.plan_metrics(sc, r.z, θ)
            @printf("%-22s %-6d %-7.2f %-16.3e %-9.4f %-8.3f %-8.3f\n", @sprintf("rho = %.0e, %s", ρ, tag),
                    r.iters, t, r.residual / scale, m.goal_error, m.min_gap, m.max_tilt)
            push!(rows, (; tag, rho = ρ, result = r, time = t, metrics = m))
        end)

    # As the source's demo() ranks: converged rows by goal error; if none converged, the
    # cold RHO_REF row (or, when the sweep has no such row, the smallest residual).
    converged = filter(row -> row.result.residual / scale < Core_.TOL, rows)
    ref_rows = filter(row -> row.tag == "cold" && row.rho == Core_.RHO_REF, rows)
    best = !isempty(converged) ? argmin(row -> row.metrics.goal_error, converged) :
           !isempty(ref_rows) ? first(ref_rows) : argmin(row -> row.result.residual, rows)
    isempty(converged) && @warn "no row met ||K_0||/sqrt(m) < $(Core_.TOL); the reported plan need not be feasible"
    m = best.metrics
    @printf("\nplan: rho = %.0e (%s), goal error %.4f, min gap %.3f (d_min %.2f), max tilt %.3f, handle drift %.1e, worst g %.1e\n",
            best.rho, best.tag, m.goal_error, m.min_gap, sc.d_min, m.max_tilt, m.max_handle_drift,
            min(m.min_safety, m.min_arm_speed, m.min_child_speed, m.min_child_reach))

    save || return nothing
    mkpath(run_dir)
    jldsave(joinpath(run_dir, "sweep.jld2");
            scenario = sc, rows = [(; row.tag, row.rho, row.time, row.metrics, z = row.result.z,
                                      w = row.result.w, iters = row.result.iters,
                                      residual = row.result.residual, history = row.result.history)
                                   for row in rows],
            chosen = (; best.tag, best.rho, z = best.result.z), build_time = ctx.build_time)
    if plot
        # Both guesses, as the source draws them, and which one the sweep started from.
        guesses = ("zero_control" => Core_.zero_control_guess(sc), "direct_path" => Core_.direct_path_guess(sc))
        tp = @elapsed plot_count = Base.invokelatest(plotting(:write_robotic_arm_plots), sc, cold, warm, cold_traces;
            max_inner = Core_.MAX_INNER, tol = Core_.TOL, tol_scale = scale, guesses, output_dir = run_dir)
        @printf("\nwrote %d figures (a PDF and an HTML per solve, plus %d guesses) in %.1f s\n",
                plot_count, length(guesses), tp)
        println("the sweep started from zero_control")
    end
    println("wrote ", normpath(run_dir))
    return nothing
end

end # module Robotic_arm
