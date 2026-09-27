module Robotic_arm_receding

# Python-facing (juliacall) entry points for the robotic-arm pot carry, with the same
# contract as ScholtesReducedGOOP.jl `examples/robotic_arm.jl` section 9:
#
#   Python: jl.include(".../experiments/Robotic_arm_receding.jl")
#           ctx     = jl.Robotic_arm_receding.build_mpc_context(obs, use_nominal_initial_state=True, bases=bases)
#           goals   = jl.Robotic_arm_receding.get_current_goal_positions(ctx, obs)
#           plan_fn = jl.Robotic_arm_receding.create_planner_from_context(ctx, horizon,
#                         planner_freq=10, low_level_freq=10)
#           action  = np.array(plan_fn(obs))   # one interpolated waypoint, or [] when done
#
# The planner solves ONCE, lazily on its first call (the ρ sweep of
# robotic_arm_core.jl), then streams that open-loop plan one interpolated waypoint at a
# time. x₀ is the parameter θ, so the same compiled system can be re-solved from a
# measured state; the closed-loop MPC that does so is experiments/Robotic_arm_mpc.jl.

using LinearAlgebra: norm
using Printf: @printf
using ReducedGOOP

const ROBOTIC_ARM_CORE_PATH = joinpath(@__DIR__, "robotic_arm_core.jl")
isdefined(Main, :RoboticArmCore) || Base.include(Main, ROBOTIC_ARM_CORE_PATH)
using Main.RoboticArmCore
const Core_ = Main.RoboticArmCore

export build_mpc_context, create_planner_from_context, get_current_goal_positions

"An UNSOLVED planning problem: the scenario and its compiled Scholtes KKT system."
struct MpcContext{C}
    arm::C
    planning_horizon::Int
    x_init::Vector{Vector{Float64}}
    pot_goal::Vector{Float64}
    rho_schedule::Vector{Float64}
    tol::Float64
    max_inner::Int
    linear_solver::Symbol
    proj_eps::Float64
    output_dir::String
end

"""
    build_mpc_context(obs; planning_horizon = 30, bases = nothing,
                      use_nominal_initial_state = false, rho_schedule = RHO_SWEEP, tol = TOL,
                      max_inner = MAX_INNER, linear_solver = LINEAR_SOLVER, proj_eps = PROJ_EPS,
                      child_reach_max = 0.4, output_dir)

Configure the scenario from `obs`'s end-effector positions (`robot0_eef_pos`,
`robot1_eef_pos`, `robot2_eef_pos`), or from the nominal robosuite state when
`use_nominal_initial_state`, and compile its KKT system. Does not solve. `bases`, when
given, are the two Pandas' and the child's reach centres; nominal contexts use the
nominal centres, as the source does.
"""
function build_mpc_context(
    obs;
    planning_horizon::Integer = 30,
    bases = nothing,
    use_nominal_initial_state::Bool = false,
    rho_schedule = RHO_SWEEP,
    tol::Float64 = TOL,
    max_inner::Integer = MAX_INNER,
    linear_solver::Symbol = LINEAR_SOLVER,
    proj_eps::Float64 = PROJ_EPS,
    child_reach_max::Real = 0.4,
    output_dir::AbstractString = joinpath(@__DIR__, "..", "data", "robotic_arm_scholtes"),
)
    nominal = Core_.default_scenario_config()
    g1, g2, c0 = if use_nominal_initial_state
        (nominal.x_init[1][1:3], nominal.x_init[1][4:6], nominal.x_init[2])
    else
        (Vector{Float64}(obs["robot0_eef_pos"]), Vector{Float64}(obs["robot1_eef_pos"]),
         Vector{Float64}(obs["robot2_eef_pos"]))
    end
    reach_bases = use_nominal_initial_state || bases === nothing ? nominal.bases :
                  [Float64.(collect(b)) for b in bases]
    x_init = [vcat(g1, g2), copy(c0)]
    sc = Core_.ScenarioConfig(; horizon = Int(planning_horizon), x_init, bases = reach_bases,
                              child_reach_max = Float64(child_reach_max))
    isempty(rho_schedule) && error("rho_schedule must contain at least one level")
    t0 = time()
    arm = Core_.build_context(sc)
    @printf("[Robotic_arm_receding] build_mpc_context: %.1f s (KKT build + codegen %.1f s)\n",
            time() - t0, arm.build_time)
    MpcContext(arm, Int(planning_horizon), x_init, copy(sc.pot_goal),
               collect(float.(rho_schedule)), tol, Int(max_inner), linear_solver, proj_eps,
               String(output_dir))
end

"""
    create_planner_from_context(ctx, num_mpc_steps = ctx.planning_horizon; planner_freq = 10,
                                low_level_freq = 10, select_by_goal_distance = false,
                                stop_at_tol = false)

A stateful `plan_fn(obs) -> Vector{Float64}`. On its first call it runs the ρ sweep once
and keeps one row: the converged row (‖K₀‖/√m < tol) with the smallest ‖K₀‖/√m, or, if
none converged, the smallest ‖K₀‖/√m overall (with a warning).
`select_by_goal_distance = true` keeps the row closest to the goal instead.
`stop_at_tol` returns at the first converged row. Every call streams
`[gripper1; 1; gripper2; 1; child]` waypoints, `low_level_freq ÷ planner_freq` per knot,
and returns `Float64[]` after `num_mpc_steps` knots. `obs` is accepted but unused.
"""
function create_planner_from_context(
    ctx::MpcContext,
    num_mpc_steps::Integer = ctx.planning_horizon;
    planner_freq::Integer = 10,
    low_level_freq::Integer = 10,
    select_by_goal_distance::Bool = false,
    stop_at_tol::Bool = false,
)
    low_level_freq >= planner_freq ||
        error("low_level_freq ($(low_level_freq)) must be >= planner_freq ($(planner_freq))")
    ratio = div(low_level_freq, planner_freq)
    low_level_freq == planner_freq * ratio ||
        @warn "low_level_freq $(low_level_freq) is not an exact multiple of planner_freq $(planner_freq); using ratio=$(ratio)"
    num_mpc_steps <= ctx.planning_horizon ||
        @warn "clamping the number of mpc steps requested ($(num_mpc_steps)) to the planning horizon ($(ctx.planning_horizon))"
    num_mpc_steps = min(num_mpc_steps, ctx.planning_horizon)

    sc = ctx.arm.scenario
    buffer = Vector{Float64}[]
    t = Ref(0)
    z = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    scale = sqrt(ctx.arm.m)
    k0(r) = r.residual / scale
    goal_dist(zz) = norm(Core_.pot_centre(sc, zz, sc.horizon) .- sc.pot_goal)

    function solved_z()
        z[] === nothing || return z[]
        @printf("[Robotic_arm_receding] T = %d, residual rows m = %d, sqrt(m) = %.3f\n",
                ctx.planning_horizon, ctx.arm.m, scale)
        θ = Core_.scenario_parameters(ctx.x_init)
        st = @timed Core_.solve_rho_sweep(ctx.arm, θ; rho_schedule = ctx.rho_schedule,
            linear_solver = ctx.linear_solver, proj_eps = ctx.proj_eps,
            max_inner = ctx.max_inner, tol = ctx.tol, stop_at_tol,
            on_result = (tag, ρ, r, elapsed, tr) -> @printf(
                "[Robotic_arm_receding]   rho=%.0e %-10s ||K_0||/sqrt(m)=%.3e %-10s goal_dist=%.5f  %4d iters %.2f s\n",
                ρ, tag, k0(r), k0(r) < ctx.tol ? "converged" : "", goal_dist(r.z), r.iters, elapsed))
        cold, warm = st.value
        @printf("[Robotic_arm_receding] sweep: %.2f s -- JIT %.2f s, solves %.2f s\n",
                st.time, st.compile_time, st.time - st.compile_time)
        candidates = vcat([("cold", r) for (_, r) in cold], [(tag, r) for (tag, _, r, _) in warm])
        ok = filter(c -> k0(c[2]) < ctx.tol, candidates)
        scheme, result = select_by_goal_distance ?
                         argmin(c -> goal_dist(c[2].z), candidates) :
                         argmin(c -> k0(c[2]), isempty(ok) ? candidates : ok)
        isempty(ok) && @warn "create_planner_from_context: no candidate reached tol (best ||K_0||/sqrt(m) = $(k0(result)), tol = $(ctx.tol))"
        @printf("[Robotic_arm_receding] solved: %s, rho=%.0e, ||K_0||/sqrt(m)=%.3e, goal_dist=%.5f\n",
                scheme, result.rho, k0(result), goal_dist(result.z))
        z[] = result.z
    end

    grip_at(zz, k, a) = k == 0 ? ctx.x_init[1][(3a - 2):(3a)] : Core_.gripper(sc, zz, k, a)
    child_at(zz, k) = k == 0 ? ctx.x_init[2] : Core_.child(sc, zz, k)

    function plan_next_fn(obs)
        if isempty(buffer)
            t[] >= num_mpc_steps && return Float64[]
            zz = solved_z()
            k_prev = t[]
            t[] += 1
            k_next = t[]
            g1p, g2p, cp = grip_at(zz, k_prev, 1), grip_at(zz, k_prev, 2), child_at(zz, k_prev)
            g1n, g2n, cn = grip_at(zz, k_next, 1), grip_at(zz, k_next, 2), child_at(zz, k_next)
            for i in 1:ratio
                α = i / ratio
                push!(buffer, vcat(g1p .+ α .* (g1n .- g1p), [1.0],
                                   g2p .+ α .* (g2n .- g2p), [1.0],
                                   cp .+ α .* (cn .- cp)))
            end
        end
        popfirst!(buffer)
    end
    plan_next_fn
end

"Goal markers for overlay visualization: the pot goal and the current pot centre."
function get_current_goal_positions(ctx::MpcContext, obs)::Vector{Vector{Float64}}
    eef0 = collect(Float64, obs["robot0_eef_pos"])
    eef1 = collect(Float64, obs["robot1_eef_pos"])
    [copy(ctx.pot_goal), 0.5 .* (eef0 .+ eef1)]
end

end # module Robotic_arm_receding
