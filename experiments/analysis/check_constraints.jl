# Evaluate the raw inequality residuals of a converged robotic-arm solution:
# per-timestep child-safety, arm-speed, and child-speed margins, plus the
# handle-grasp equality. Negative residual = violated.
#
# Usage: julia experiments/analysis/check_constraints.jl [run_id]
# Default run_id: Robotic_arm_single_robot_agent_3_levels
using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "experiments"); io = devnull)
using JLD2

run_id = isempty(ARGS) ? "Robotic_arm_single_robot_agent_3_levels" : ARGS[1]
run_dir = joinpath(
    REPO_ROOT,
    "data",
    "robotic_arm_open_loop",
    "runs",
    run_id,
    "data",
    "problem",
)
meta = JLD2.load_object(joinpath(run_dir, "run_metadata.jld2"))
sol = JLD2.load_object(
    joinpath(run_dir, "solution", "solution_dict_instance_1_eps0.1.jld2"),
)

arm_speed_limit = meta["arm_speed_limit"]
child_speed_limit = meta["child_speed_limit"]
collision_avoidance = meta["collision_avoidance"]
d_p = meta["dₚ"]

println("run: $run_id")
println(
    "limits: arm_speed=$(arm_speed_limit), child_speed=$(child_speed_limit), collision_avoidance=$(collision_avoidance), dₚ=$(d_p)",
)
println("status: ", sol["status"], "   kkt_error: ", sol["kkt_error"], "   ϵ: ", sol["ϵ"])
haskey(sol, "preference_values") &&
    println("stored preference_values: ", sol["preference_values"])

strategies = sol["strategies"]
arm = strategies[1]   # combined two-arm: 6D states/controls
child = strategies[2] # 3D
T = length(arm.xs)
println("planning horizon T = $T")

fmt(v) = round(v; sigdigits = 4)

println(
    "\n── Player 1: robot_child_safety_inequality (pot center outside child sphere) ──",
)
safety = map(1:T) do t
    pot_center = 0.5 .* (arm.xs[t][1:3] .+ arm.xs[t][4:6])
    sum(abs2, pot_center .- child.xs[t]) - collision_avoidance^2
end
for t in 1:T
    flag = safety[t] < 0 ? "  <-- VIOLATED" : ""
    println("  t=$t: residual = $(fmt(safety[t]))$flag")
end

println(
    "\n── Player 1: robot_arm_speed_inequality (per-arm speed ≤ $(arm_speed_limit)) ──",
)
speed1 = [arm_speed_limit^2 - sum(abs2, u[1:3]) for u in arm.us]
speed2 = [arm_speed_limit^2 - sum(abs2, u[4:6]) for u in arm.us]
for t in 1:T
    f1 = speed1[t] < 0 ? "  <-- VIOLATED (arm1)" : ""
    f2 = speed2[t] < 0 ? "  <-- VIOLATED (arm2)" : ""
    println("  t=$t: arm1 = $(fmt(speed1[t]))$f1,  arm2 = $(fmt(speed2[t]))$f2")
end

println(
    "\n── Player 2: child_ground_speed_inequality (ground speed ≤ $(child_speed_limit)) ──",
)
child_speed = [child_speed_limit^2 - sum(abs2, u[1:2]) for u in child.us]
for t in 1:T
    flag = child_speed[t] < 0 ? "  <-- VIOLATED" : ""
    println("  t=$t: residual = $(fmt(child_speed[t]))$flag")
end

println("\n── (hard equality) handle grasp: |p1 - p2|² - dₚ² should be ≈ 0 ──")
grasp = [sum(abs2, arm.xs[t][1:3] .- arm.xs[t][4:6]) - d_p^2 for t in 1:T]
println("  max |residual| = ", fmt(maximum(abs, grasp)))

println("\n── Summary ──")
println(
    "player 1 worst residuals: safety = $(fmt(minimum(safety))), arm1 speed = $(fmt(minimum(speed1))), arm2 speed = $(fmt(minimum(speed2)))",
)
println(
    "  combined robot_inequality min = $(fmt(min(minimum(safety), minimum(speed1), minimum(speed2))))",
)
println("player 2 worst residual: child speed = $(fmt(minimum(child_speed)))")
