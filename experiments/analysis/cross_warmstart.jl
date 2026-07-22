# Forward cross-warmstart: warmstart the 3-LEVEL problem with the 4-level
# solution's primal trajectory.
#
# Finding: the solve stays at the direct (goal ≈ 50) equilibrium instead of
# returning to the swerving goal ≈ 55.28 equilibrium the default warmstart
# produces — i.e. the direct trajectory IS admissible for the 3-level system;
# the default warmstart's basin just never finds it. The small drift (speed
# overshoot relaxing 0.104 → 0.053) is the penalty-exponent mismatch: the
# innermost constraint level uses violation^5 in the 3-level problem but
# violation^6 in the 4-level problem (exponent = level + 2).
#
# Requires experiments/Robotic_arm.jl in its 3-LEVEL configuration
# (control_objective commented out of goop_preferences).
using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "experiments"); io = devnull)

using JLD2
using ReducedGOOP

include(joinpath(REPO_ROOT, "experiments", "Robotic_arm.jl"))

runs_root = joinpath(REPO_ROOT, "data", "robotic_arm_open_loop", "runs")
sol4 = JLD2.load_object(joinpath(runs_root, "Robotic_arm_single_robot_agent_4_levels", "data", "problem", "solution", "solution_dict_instance_1_eps0.1.jld2"))
prob4 = JLD2.load_object(joinpath(runs_root, "Robotic_arm_single_robot_agent_4_levels", "data", "problem", "problem_data_instance_1.jld2"))
prob4 isa AbstractVector && (prob4 = prob4[1])
sol3 = JLD2.load_object(joinpath(runs_root, "Robotic_arm_single_robot_agent_3_levels", "data", "problem", "solution", "solution_dict_instance_1_eps0.1.jld2"))

# ── Rebuild the 3-level problem ──
scenario_config = Robotic_arm.demo_scenario_config()
(; problem, flatten_parameters) = Robotic_arm.get_setup(scenario_config)
@assert length(problem.preferences[1]) == 3 "expected 3-level hierarchy for player 1, got $(length(problem.preferences[1]))"

# θ must match the saved runs: assert the scenario states equal the recorded ones.
@assert scenario_config.base_initial_state1 == prob4["initial_state1"]
@assert scenario_config.base_initial_state2 == prob4["initial_state2"]
@assert scenario_config.goal_position1 == prob4["goal_position1"]
@assert scenario_config.goal_position2 == prob4["goal_position2"]
@assert scenario_config.initial_state3 == prob4["initial_state3"]
@assert scenario_config.goal_position3 == prob4["goal_position3"]

instance_states = (;
	initial_state1 = prob4["initial_state1"],
	initial_state2 = prob4["initial_state2"],
	initial_state3 = prob4["initial_state3"],
)
θ = Robotic_arm.build_instance_parameters(flatten_parameters, instance_states, scenario_config).θ

println("building 3-level KKT system...")
kkt = @time ReducedGOOP.generate_slacked_reduced_kkt_system(
	problem;
	backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
	backend_options = (;),
	codegen = :fast_differentiation,
)

# ── Warmstart with the 4-level solution's primal trajectory ──
n_primal = sum(problem.primal_dims)
z₀ = collect(sol4["x"][1:n_primal])

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

println("solving 3-level problem from 4-level warmstart...")
output = @time ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt, θ; z₀, options)
println("status: $(output.status), total_iters: $(output.total_iters), kkt_error: $(round(output.kkt_error; sigdigits = 5))")

x = collect(output.x[1:n_primal])
mkpath(joinpath(@__DIR__, "results"))
JLD2.save_object(joinpath(@__DIR__, "results", "cross_warmstart_solution.jld2"),
	Dict("x" => x, "z" => collect(output.z), "kkt_error" => output.kkt_error, "status" => string(output.status), "total_iters" => output.total_iters))

fmt(v) = round(v; sigdigits = 5)

function analyze(label, primal)
	strategies = Robotic_arm.extract_player_strategies(primal, problem.primal_dims, scenario_config.dynamics)
	arm, child = strategies
	T = length(arm.xs)
	goal1, goal2 = scenario_config.goal_position1, scenario_config.goal_position2
	goal_val = sum(abs2, 0.5 .* (arm.xs[end][1:3] .+ arm.xs[end][4:6]) .- 0.5 .* (goal1 .+ goal2))
	control = sum(sum(u .^ 2) for u in arm.us)
	speed_res = vcat([25.0 - sum(abs2, u[1:3]) for u in arm.us[1:end-1]], [25.0 - sum(abs2, u[4:6]) for u in arm.us[1:end-1]])
	safety = minimum(sum(abs2, 0.5 .* (arm.xs[t][1:3] .+ arm.xs[t][4:6]) .- child.xs[t]) - 4.0 for t in 1:T)
	pot = sum(sum(abs2, child.xs[t] .- 0.5 .* (arm.xs[t][1:3] .+ arm.xs[t][4:6])) for t in 1:T)
	println("$label: goal = $(fmt(goal_val)), control = $(fmt(control)), worst speed residual = $(fmt(minimum(speed_res))), Σ overshoot = $(fmt(-sum(min.(speed_res, 0.0)))), safety margin = $(fmt(safety)), pot_approach = $(fmt(pot))")
	(; goal_val, control, primal = collect(primal))
end

println()
r_ws  = analyze("4-level warmstart point   ", z₀)
r_new = analyze("3-level solve from it     ", x)
r_def = analyze("3-level default equilibrium", collect(sol3["x"][1:n_primal]))

println()
println("‖converged − warmstart‖        = ", fmt(sqrt(sum(abs2, r_new.primal .- r_ws.primal))))
println("‖converged − default 3-level‖  = ", fmt(sqrt(sum(abs2, r_new.primal .- r_def.primal))))
println("‖warmstart − default 3-level‖  = ", fmt(sqrt(sum(abs2, r_ws.primal .- r_def.primal))))
