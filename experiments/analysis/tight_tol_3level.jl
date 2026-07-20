# Does tightening tol remove the swerving equilibrium in the 3-level setting?
# Re-solve the 3-level problem from the SAME default warmstart with tol = 1e-3
# and 1e-4 (the original run used tol = 1e-2 and accepted the swerve at 0.0098).
#
# Finding: NO. At tol = 1e-3 the solver exhausts 2000 iterations
# (status :failed), stalls at kkt_error 1.57e-3, and the stall point swerves
# MORE than the tol = 1e-2 solution (x-drift −0.48 → −0.75, worst step angle
# 30° → 35°, goal 55.28 → 56.17, safety still violated). The swerve is a
# genuine near-stationary branch of the 3-level system, not a tolerance
# artifact — solver strictness entrenches it instead of removing it.
#
# Requires experiments/Robotic_arm.jl in its 3-LEVEL configuration.
using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "experiments"); io = devnull)

using JLD2
using LinearAlgebra
using ReducedGOOP

include(joinpath(REPO_ROOT, "experiments", "Robotic_arm.jl"))

scenario_config = Robotic_arm.demo_scenario_config()
(; problem, flatten_parameters) = Robotic_arm.get_setup(scenario_config)
@assert length(problem.preferences[1]) == 3

instance_states = (;
	initial_state1 = scenario_config.base_initial_state1,
	initial_state2 = scenario_config.base_initial_state2,
	initial_state3 = scenario_config.initial_state3,
)
θ = Robotic_arm.build_instance_parameters(flatten_parameters, instance_states, scenario_config).θ
(; warmstart_solution) = Robotic_arm.build_default_warmstart(instance_states, scenario_config)

println("building 3-level KKT system...")
kkt = @time ReducedGOOP.generate_slacked_reduced_kkt_system(
	problem;
	backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
	backend_options = (;),
	codegen = :fast_differentiation,
)

n_primal = sum(problem.primal_dims)
fmt(v) = round(v; sigdigits = 5)
goal_center = [0.0, -5.0, 5.0]

function analyze(label, primal)
	strategies = Robotic_arm.extract_player_strategies(primal, problem.primal_dims, scenario_config.dynamics)
	arm, child = strategies
	T = length(arm.xs)
	pot(t) = 0.5 .* (arm.xs[t][1:3] .+ arm.xs[t][4:6])
	goal_val = sum(abs2, pot(T) .- goal_center)
	control = sum(sum(u .^ 2) for u in arm.us)
	safety = minimum(sum(abs2, pot(t) .- child.xs[t]) - 4.0 for t in 1:T)
	angles = [begin
		v = pot(t + 1) .- pot(t); d = goal_center .- pot(t)
		round(acosd(clamp(dot(v, d) / (norm(v) * norm(d)), -1, 1)); digits = 1)
	end for t in 1:(T-1)]
	println("$label: goal = $(fmt(goal_val)), control = $(fmt(control)), safety margin = $(fmt(safety))")
	println("    step-vs-goal angles [deg]: $angles")
	println("    pot x: $([round(pot(t)[1]; digits = 3) for t in 1:T])")
	println("    pot z: $([round(pot(t)[3]; digits = 3) for t in 1:T])")
end

mkpath(joinpath(@__DIR__, "results"))
for tol in [1e-3, 1e-4]
	options = ReducedGOOP.InteriorPointOptions(;
		tol,
		η₀ = 1e-6,
		η_max = 1e6,
		ϵ₀ = 0.1,
		max_inner_iters = 2000,
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
		perturbation_enabled = false,
		stagnation_rtol = 1e-1,
		perturbation_scale = 1e-6,
		tsvd_threshold = 0.0,
		use_marquardt_scaling = false,
		verbose = false,
	)
	println("\n════ tol = $tol ════")
	output = @time ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt, θ; z₀ = warmstart_solution, options)
	println("status: $(output.status), total_iters: $(output.total_iters), kkt_error: $(fmt(output.kkt_error))")
	analyze("  converged point", collect(output.x[1:n_primal]))
	JLD2.save_object(joinpath(@__DIR__, "results", "tight_tol_3level_tol$(tol).jld2"),
		Dict("x" => collect(output.x), "kkt_error" => output.kkt_error, "status" => string(output.status), "total_iters" => output.total_iters, "tol" => tol))
end
