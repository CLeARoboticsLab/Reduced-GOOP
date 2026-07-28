# Is the default warmstart Euclidean-close to the swerving equilibrium, which
# would explain why the 3-level solve lands there? Measure primal distances
# from the default warmstart to (a) the swerving 3-level equilibrium, (b) the
# direct-basin 3-level point (results/cross_warmstart_solution.jld2), and
# (c) the direct 4-level equilibrium.
#
# Finding: NO — the warmstart is FARTHER from the swerve (10.79) than from the
# direct equilibrium (6.6), and its robot block nearly coincides with the
# 4-level solution's robot block (0.15). The 3-level solve moves the robot 8.9
# to reach the swerve when a 3-level-admissible direct point sits 3.2 away.
# The basin is set by the Newton flow through the flat inner-level valley, not
# by Euclidean proximity.
#
# Requires experiments/Robotic_arm.jl in its 3-LEVEL configuration.
using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "experiments"); io = devnull)

using JLD2
using LinearAlgebra

include(joinpath(REPO_ROOT, "experiments", "Robotic_arm.jl"))

scenario_config = Robotic_arm.demo_scenario_config()
(; problem, flatten_parameters) = Robotic_arm.get_setup(scenario_config)
n = sum(problem.primal_dims)
d1 = problem.primal_dims[1]

instance_states = (;
    initial_state1 = scenario_config.base_initial_state1,
    initial_state2 = scenario_config.base_initial_state2,
    initial_state3 = scenario_config.initial_state3,
)
(; warmstart_solution) =
    Robotic_arm.build_default_warmstart(instance_states, scenario_config)
w = collect(warmstart_solution)[1:n]

runs_root = joinpath(REPO_ROOT, "data", "robotic_arm_open_loop", "runs")
swerve = collect(
    JLD2.load_object(
        joinpath(
            runs_root,
            "Robotic_arm_single_robot_agent_3_levels",
            "data",
            "problem",
            "solution",
            "solution_dict_instance_1_eps0.1.jld2",
        ),
    )["x"],
)[1:n]
direct3 = collect(
    JLD2.load_object(joinpath(@__DIR__, "results", "cross_warmstart_solution.jld2"))["x"],
)[1:n]
direct4 = collect(
    JLD2.load_object(
        joinpath(
            runs_root,
            "Robotic_arm_single_robot_agent_4_levels",
            "data",
            "problem",
            "solution",
            "solution_dict_instance_1_eps0.1.jld2",
        ),
    )["x"],
)[1:n]

fmt(v) = round(v; sigdigits = 4)
dist(a, b) = fmt(norm(a .- b))
p1(v) = v[1:d1]
p2(v) = v[(d1 + 1):n]

for (label, target) in [
    ("swerving 3-level eq", swerve),
    ("direct 3-level eq  ", direct3),
    ("direct 4-level eq  ", direct4),
]
    println(
        "‖warmstart − $(label)‖ : total $(dist(w, target)),  robot-block $(dist(p1(w), p1(target))),  child-block $(dist(p2(w), p2(target)))",
    )
end
println()
println(
    "‖swerve − direct3‖ = $(dist(swerve, direct3)) (inter-equilibrium gap, robot $(dist(p1(swerve), p1(direct3))), child $(dist(p2(swerve), p2(direct3))))",
)
