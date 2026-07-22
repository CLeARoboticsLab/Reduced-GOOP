# Reverse cross-warmstart: warmstart the 4-LEVEL problem with the swerving
# 3-level solution.
#
# Finding: the solver ESCAPES (102 iterations, primal moves 7.0 of the 8.8
# gap between the two equilibria): the z-dip is eliminated, safety clearance
# is restored (−0.046 → +4.44), goal improves 55.28 → 51.11. Together with
# cross_warmstart.jl this establishes the asymmetry: the direct trajectory is
# a solution of BOTH systems; the swerving trajectory is a solution of the
# 3-level system ONLY — the 4-level system's extra control-stationarity rows
# exclude it. Note the escape is not driven by control cost (451.16 → 452.10,
# it slightly rises; speed saturates either way) but by the added optimality
# conditions.
#
# Requires experiments/Robotic_arm.jl in its 3-LEVEL configuration; the
# 4-level problem is built here by prepending min-control at the outermost slot.
using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "experiments"); io = devnull)

using JLD2
using BlockArrays
using ReducedGOOP

include(joinpath(REPO_ROOT, "experiments", "Robotic_arm.jl"))

runs_root = joinpath(REPO_ROOT, "data", "robotic_arm_open_loop", "runs")
sol3 = JLD2.load_object(joinpath(runs_root, "Robotic_arm_single_robot_agent_3_levels", "data", "problem", "solution", "solution_dict_instance_1_eps0.1.jld2"))
sol4 = JLD2.load_object(joinpath(runs_root, "Robotic_arm_single_robot_agent_4_levels", "data", "problem", "solution", "solution_dict_instance_1_eps0.1.jld2"))

scenario_config = Robotic_arm.demo_scenario_config()
(; problem, flatten_parameters) = Robotic_arm.get_setup(scenario_config)
@assert length(problem.preferences[1]) == 3

# ── Build the 4-level problem: prepend min-control at the outermost slot ──
arm_state_dim = scenario_config.dynamics[1].state_dimension
arm_control_dim = scenario_config.dynamics[1].control_dimension
control_objective_p1 = function (z, _)
	(; us) = Robotic_arm.unflatten_trajectory(z[Block(1)], arm_state_dim, arm_control_dim)
	sum(sum(u .^ 2) for u in us)
end
preferences4 = [
	vcat([control_objective_p1], problem.preferences[1]),
	problem.preferences[2],
]
is_constraint4 = [
	vcat([false], problem.is_prioritized_constraint[1]),
	problem.is_prioritized_constraint[2],
]
dummy_primals = BlockArray(zeros(sum(problem.primal_dims)), problem.primal_dims)
dummy_parameters = BlockArray(zeros(sum(problem.parameter_dims)), problem.parameter_dims)
problem4 = ReducedGOOP.ParametricGOOP(
	dummy_primals,
	dummy_parameters;
	preferences = preferences4,
	is_prioritized_constraint = is_constraint4,
	equality_constraints = problem.equality_constraints,
	inequality_constraints = [nothing, nothing],
	shared_equality_constraint = nothing,
	shared_inequality_constraint = nothing,
)

instance_states = (;
	initial_state1 = scenario_config.base_initial_state1,
	initial_state2 = scenario_config.base_initial_state2,
	initial_state3 = scenario_config.initial_state3,
)
θ = Robotic_arm.build_instance_parameters(flatten_parameters, instance_states, scenario_config).θ

println("building 4-level KKT system...")
kkt4 = @time ReducedGOOP.generate_slacked_reduced_kkt_system(
	problem4;
	backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
	backend_options = (;),
	codegen = :fast_differentiation,
)

n_primal = sum(problem.primal_dims)
z₀ = collect(sol3["x"][1:n_primal])   # the swerving 3-level solution

options = ReducedGOOP.InteriorPointOptions(;
	tol = 0.01,
	η₀ = 1e-6,
	η_max = 1e6,
	ϵ₀ = 0.1,
	max_inner_iters = 500,
	max_outer_iters = 1,
	tightening_rate = 1.2,
	loosening_rate = 3.0,
	min_stepsize = 1e-20,
	linesearch = :backtracking,
	record_convergence = true,
	record_condition_number = true,
	eta_retry_growth = 2.0,
	ρ_low = 0.75,
	ρ_high = 0.75,
	tsvd_threshold = 0.0,
	use_marquardt_scaling = false,
	verbose = false,
)

println("solving 4-level problem from the swerving 3-level warmstart...")
output = @time ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt4, θ; z₀, options)
println("status: $(output.status), total_iters: $(output.total_iters), kkt_error: $(round(output.kkt_error; sigdigits = 5))")

x = collect(output.x[1:n_primal])
mkpath(joinpath(@__DIR__, "results"))
JLD2.save_object(joinpath(@__DIR__, "results", "cross_warmstart_reverse_solution.jld2"),
	Dict("x" => x, "z" => collect(output.z), "kkt_error" => output.kkt_error, "status" => string(output.status), "total_iters" => output.total_iters))

fmt(v) = round(v; sigdigits = 5)
function analyze(label, primal)
	strategies = Robotic_arm.extract_player_strategies(primal, problem.primal_dims, scenario_config.dynamics)
	arm, child = strategies
	T = length(arm.xs)
	goal1, goal2 = scenario_config.goal_position1, scenario_config.goal_position2
	goal_val = sum(abs2, 0.5 .* (arm.xs[end][1:3] .+ arm.xs[end][4:6]) .- 0.5 .* (goal1 .+ goal2))
	control = sum(sum(u .^ 2) for u in arm.us)
	safety = minimum(sum(abs2, 0.5 .* (arm.xs[t][1:3] .+ arm.xs[t][4:6]) .- child.xs[t]) - 4.0 for t in 1:T)
	pot_x = [round(0.5 * (arm.xs[t][1] + arm.xs[t][4]); digits = 3) for t in 1:T]
	pot_z = [round(0.5 * (arm.xs[t][3] + arm.xs[t][6]); digits = 3) for t in 1:T]
	println("$label: goal = $(fmt(goal_val)), control = $(fmt(control)), safety margin = $(fmt(safety))")
	println("    pot x path: $pot_x")
	println("    pot z path: $pot_z")
	collect(primal)
end

println()
p_ws  = analyze("swerving 3-level warmstart ", z₀)
p_new = analyze("4-level solve from it      ", x)
p_dir = analyze("4-level direct equilibrium ", collect(sol4["x"][1:n_primal]))

println()
println("‖converged − swerving warmstart‖   = ", fmt(sqrt(sum(abs2, p_new .- p_ws))))
println("‖converged − direct 4-level‖       = ", fmt(sqrt(sum(abs2, p_new .- p_dir))))
