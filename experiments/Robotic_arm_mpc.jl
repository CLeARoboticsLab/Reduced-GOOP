module Robotic_arm_mpc

# Closed-loop receding-horizon MPC for the robotic-arm pot carry.
#
#   julia --project=experiments -e 'include("experiments/Robotic_arm_mpc.jl"); Robotic_arm_mpc.demo()'
#
# The KKT system is compiled ONCE: x₀ is the parameter θ (robotic_arm_core.jl), so every
# MPC step re-solves the same system from the executed state. Step 1 runs the ρ sweep
# (`stop_at_tol`: fine → coarse, first converged row). Every later step warm-starts from
# the previous plan shifted by one knot, with its equality/policy duals (warm-z+eq) at the
# previous row's ρ, and falls back to a cold `stop_at_tol` sweep if that does not
# converge. The first knot of each plan is executed (a perfect model of the plant).

using JLD2: jldsave
using LinearAlgebra: norm
using Printf: @printf, @sprintf
using Dates: Dates
using ReducedGOOP

const ROBOTIC_ARM_CORE_PATH = joinpath(@__DIR__, "robotic_arm_core.jl")
isdefined(Main, :RoboticArmCore) || Base.include(Main, ROBOTIC_ARM_CORE_PATH)
using Main.RoboticArmCore
const Core_ = Main.RoboticArmCore
include(joinpath(@__DIR__, "robotic_arm_visualization.jl"))

"The previous plan advanced one knot: x_{t+1} → x_t, u_{t+1} → u_t, the last knot held."
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
    out
end

"""
    demo(; num_steps = 20, scenario_kwargs = (;), linear_solver = LINEAR_SOLVER,
         run_id = nothing, plot = ENV["GOOP_PLOT"] != "0")

Run `num_steps` closed-loop MPC steps and save the executed trajectory, per-step solve
data and figures to `data/robotic_arm_scholtes_mpc/<run_id>/`.
"""
function demo(; num_steps::Integer = 20, scenario_kwargs::NamedTuple = (;),
              linear_solver::Symbol = LINEAR_SOLVER, run_id = nothing,
              plot::Bool = get(ENV, "GOOP_PLOT", "1") != "0")
    sc = Core_.ScenarioConfig(; scenario_kwargs...)
    run_id = something(run_id, Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    run_dir = joinpath(@__DIR__, "..", "data", "robotic_arm_scholtes_mpc", run_id)
    ctx = Core_.build_context(sc)
    scale = sqrt(ctx.m)
    k0(r) = r.residual / scale
    @printf("KKT build + codegen %.1f s, m = %d; %d closed-loop steps, linear_solver = %s\n",
            ctx.build_time, ctx.m, num_steps, linear_solver)

    x = deepcopy(sc.x_init)
    executed = [deepcopy(x)]
    steps = Any[]
    previous = nothing
    eq_cols = Core_.eq_columns(ctx.kkt)
    for k in 1:num_steps
        θ = Core_.scenario_parameters(x)
        z0 = Core_.zero_control_guess(sc, x)
        mode, result = "", nothing
        t = @elapsed begin
            attempts = Any[]
            if previous !== nothing
                # warm-z+eq from the shifted plan, at the previous ρ and then coarser
                # levels (a coarser ρ buys basin). The warm start floors the slacks, so
                # it does not need a strictly feasible point.
                shifted = shift_plan(sc, previous.z)
                for ρ in unique((previous.rho, 1e-7, 1e-5))
                    w₀ = ReducedGOOP.scholtes_warm_start(ctx.kkt, θ, shifted, ρ;
                                                          eq_cols, eq = previous.w[eq_cols])
                    r = ReducedGOOP.solve(ReducedGOOP.Scholtes(), ctx.kkt, θ; z₀ = z0, w₀,
                        options = Core_.solver_options(ctx, ρ; linear_solver))
                    push!(attempts, r)
                    if k0(r) < Core_.TOL
                        mode, result = @sprintf("warm (rho %.0e)", ρ), r
                        break
                    end
                end
            end
            # Cold sweep: only from a strictly feasible zero-control guess (a state
            # executed ON an active constraint makes it infeasible).
            if result === nothing &&
               ReducedGOOP.is_feasible(ctx.problem, z0, θ).worst > 0
                cold, warm = Core_.solve_rho_sweep(ctx, θ; z0, linear_solver, stop_at_tol = true)
                append!(attempts, vcat([r for (_, r) in cold], [r for (_, _, r, _) in warm]))
                r = argmin(k0, attempts)
                if k0(r) < Core_.TOL
                    mode, result = "sweep", r
                end
            end
            if result === nothing
                result = argmin(k0, attempts)
                mode = "best attempt (not converged)"
            end
        end
        m = Core_.plan_metrics(sc, result.z, θ)
        @printf("step %2d  %-22s rho %.0e  %4d iters  %.3f s  ||K_0||/sqrt(m) %.2e  gap %.3f  goal %.4f\n",
                k, mode, result.rho, result.iters, t, k0(result), m.min_gap, m.goal_error)
        push!(steps, (; k, mode, rho = result.rho, iters = result.iters, time = t,
                        residual = k0(result), metrics = m, z = result.z))
        # Execute the first knot of the plan.
        x = [Core_.state(sc, result.z, 1, 1), Core_.state(sc, result.z, 2, 1)]
        push!(executed, deepcopy(x))
        previous = result
    end

    times = [s.time for s in steps]
    @printf("\nper-step solve time: first %.2f s, then median %.3f s, max %.3f s; %d/%d warm-started\n",
            times[1], length(times) > 1 ? sort(times[2:end])[cld(length(times) - 1, 2)] : NaN,
            maximum(times), count(s -> startswith(s.mode, "warm"), steps), num_steps)
    mkpath(run_dir)
    jldsave(joinpath(run_dir, "mpc.jld2"); scenario = sc, executed, steps, build_time = ctx.build_time)
    plot && save_plan_figure(sc, steps[1].z, joinpath(run_dir, "first_plan.pdf"); title = "MPC step 1 plan")
    println("wrote ", normpath(run_dir))
    return (; ctx, executed, steps)
end

end # module Robotic_arm_mpc
