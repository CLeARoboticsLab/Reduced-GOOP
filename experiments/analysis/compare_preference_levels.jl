# Compare converged 3-level and 4-level equilibria: preference level values
# (in the solver's own penalty form), raw feasibility margins, control cost,
# pot-center path shape, and step-vs-goal alignment.
using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "experiments"); io = devnull)
using JLD2
using LinearAlgebra

runs_root = joinpath(REPO_ROOT, "data", "robotic_arm_open_loop", "runs")
run_ids =
    ["Robotic_arm_single_robot_agent_3_levels", "Robotic_arm_single_robot_agent_4_levels"]

penalty(h, level) = sum(max(0.0 - hi, 0.0)^(level + 2) for hi in h)
fmt(v) = round(v; sigdigits = 5)
goal_center = [0.0, -5.0, 5.0]

for run_id in run_ids
    dir = joinpath(runs_root, run_id, "data", "problem")
    meta = JLD2.load_object(joinpath(dir, "run_metadata.jld2"))
    prob = JLD2.load_object(joinpath(dir, "problem_data_instance_1.jld2"))
    prob isa AbstractVector && (prob = prob[1])
    sol = JLD2.load_object(
        joinpath(dir, "solution", "solution_dict_instance_1_eps0.1.jld2"),
    )

    arm_speed_limit = meta["arm_speed_limit"]
    child_speed_limit = meta["child_speed_limit"]
    collision_avoidance = meta["collision_avoidance"]
    goal1, goal2 = prob["goal_position1"], prob["goal_position2"]
    # robot_inequality sits at the innermost level: 3 in the 3-level run, 4 in
    # the 4-level run; the generator's penalty exponent is level + 2.
    robot_level = occursin("4_levels", run_id) ? 4 : 3

    arm, child = sol["strategies"]
    T = length(arm.xs)
    pot(t) = 0.5 .* (arm.xs[t][1:3] .+ arm.xs[t][4:6])

    println("═══ $run_id ═══")
    println("status: $(sol["status"])   kkt_error: $(fmt(sol["kkt_error"]))   T = $T")

    safety = [sum(abs2, pot(t) .- child.xs[t]) - collision_avoidance^2 for t in 1:T]
    speed1 = [arm_speed_limit^2 - sum(abs2, u[1:3]) for u in arm.us]
    speed2 = [arm_speed_limit^2 - sum(abs2, u[4:6]) for u in arm.us]
    child_speed = [child_speed_limit^2 - sum(abs2, u[1:2]) for u in child.us]
    robot_h = vcat(safety, vec(permutedims(hcat(speed1, speed2))))

    nviol(r) = count(<(0), r)
    println(
        "violations: safety $(nviol(safety))/$T (min $(fmt(minimum(safety)))), ",
        "arm1 speed $(nviol(speed1))/$T (min $(fmt(minimum(speed1)))), ",
        "arm2 speed $(nviol(speed2))/$T (min $(fmt(minimum(speed2)))), ",
        "child speed $(nviol(child_speed))/$T (min $(fmt(minimum(child_speed))))",
    )

    control_cost = sum(sum(u .^ 2) for u in arm.us)
    goal_val = sum(abs2, pot(T) .- 0.5 .* (goal1 .+ goal2))
    allowance = 0.1
    load_balance = sum(1:T) do t
        balance = (arm.xs[t][3] - arm.xs[t][6])^2
        balance < allowance^2 ? 0.0 : balance - allowance^2
    end
    pot_approach = sum(sum(abs2, child.xs[t] .- pot(t)) for t in 1:T)

    println(
        "player 1: control = $(fmt(control_cost)), goal = $(fmt(goal_val)), load_balance = $(fmt(load_balance)), robot_ineq penalty (level $robot_level) = $(fmt(penalty(robot_h, robot_level)))",
    )
    println(
        "player 2: pot_approach = $(fmt(pot_approach)), child_speed penalty (level 2) = $(fmt(penalty(child_speed, 2)))",
    )

    # Path shape: alignment of each pot-center step with the goal direction,
    # lateral drift, and height profile ("swerve" = large angles, x-drift,
    # and a z-dip toward the ground-bound child).
    angles = [
        begin
            v = pot(t + 1) .- pot(t);
            d = goal_center .- pot(t)
            round(acosd(clamp(dot(v, d) / (norm(v) * norm(d)), -1, 1)); digits = 1)
        end for t in 1:(T - 1)
    ]
    println("step-vs-goal angles [deg]: $angles")
    println("pot x path: $([round(pot(t)[1]; digits = 3) for t in 1:T])")
    println("pot z path: $([round(pot(t)[3]; digits = 3) for t in 1:T])")
    println()
end
