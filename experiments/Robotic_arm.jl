module Robotic_arm

using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, Distributions, Random
using ReducedGOOP
using TimerOutputs: @timeit, reset_timer!

const TO = ReducedGOOP.TO

abstract type DynamicsModel end
struct SingleIntegrator3D <: DynamicsModel end

include(joinpath(@__DIR__, "plotting.jl"))
include(joinpath(@__DIR__, "3d_plotting.jl"))
include(joinpath(@__DIR__, "dynamics.jl"))

# ── Problem definition ─────────────────────────────────────────────────────────

function get_setup(
	num_players;
	dynamics = build_dynamics(SingleIntegrator()),
	control_bounds = (; lb = [-10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0]),
	planning_horizon = 10,
	dₚ = 2.0,
	collision_avoidance = 1.0,
	speed_limit = 1.5,
	map_end = 10,
	lane_width = 2,
	use_scalarized_baseline = false,
	use_social_equilibrium_baseline = false,
)
	num_players == 3 || error("robotic_arm Zero-Sum GOOP setup expects exactly three players.")

	state_dimension = dynamics.state_dimension
	control_dimension = dynamics.control_dimension
	dynamics_dimension = state_dimension + control_dimension
	position_dimension = 3

	primals_per_player = dynamics_dimension * planning_horizon
	primal_dimensions = fill(primals_per_player, num_players)
	parameter_dimensions = fill(state_dimension + 2 * position_dimension, num_players) # (state, 3D goal, 3D obstacle)

	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
	dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

	unflatten_parameters = function (θ)
		θ_iter = Iterators.Stateful(θ)
		initial_state = first(θ_iter, state_dimension)
		goal_position = first(θ_iter, position_dimension)
		obstacle_position = first(θ_iter, position_dimension)
		(; initial_state, goal_position, obstacle_position)
	end

	function flatten_parameters(; initial_state, goal_position, obstacle_position)
		length(initial_state) == state_dimension || error("Expected initial_state length $(state_dimension), got $(length(initial_state)).")
		length(goal_position) == position_dimension || error("Expected 3D goal_position, got length $(length(goal_position)).")
		length(obstacle_position) == position_dimension || error("Expected 3D obstacle_position, got length $(length(obstacle_position)).")
		vcat(initial_state, goal_position, obstacle_position)
	end

	function squared_violation(h)
		"h(x) ≥ 0	<=> (min(h(x), 0))^2 = 0"
		return (min(h, 0))^2
	end

	function smooth_piecewise_preference_objective(
		preference,
		level;
		ϵ = 0.0,
	)
		ifelse(preference ≥ ϵ, 0.0, (ϵ - preference)^(level + 2))
	end

	trajectory(z; player) =
		unflatten_trajectory(z[Block(player)], state_dimension, control_dimension)

	function goal_objective(; player)
		player in (1, 2) || error("Center goal objective belongs only to robot players 1 and 2.")
		function (z, θ)
			(; xs) = trajectory(z; player = 1)
			xs1 = copy(xs)
			(; xs) = trajectory(z; player = 2)
			xs2 = copy(xs)
			params1 = unflatten_parameters(θ[Block(1)])
			params2 = unflatten_parameters(θ[Block(2)])

			center_position = 0.5 .* (xs1[end] .+ xs2[end])
			center_goal_position = 0.5 .* (params1.goal_position .+ params2.goal_position)
			goal_deviation = center_position .- center_goal_position
			sum(goal_deviation .^ 2)
		end
	end

	function control_objective(; player)
		function (z, _)
			(; xs, us) = trajectory(z; player)
			sum(sum(u .^ 2) for u in us)
		end
	end

	function control_bound_inequality(; player)
		function (z, _)
			(; lb, ub) = control_bounds
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) = trajectory(z; player)
			mapreduce(vcat, us) do u
				vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
			end
		end
	end

	player_equality_constraints = [
		function (z, θ)
			(; lb, ub) = control_bounds
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) = trajectory(z; player = i)
			(; initial_state, goal_position) = unflatten_parameters(θ[Block(i)])

			initial_state_constraint = xs[1] - initial_state
			dynamics_constraints = mapreduce(vcat, 1:(length(xs)-1)) do t
				# xs[t] - dynamics(xs[t-1], us[t-1], t)
				dynamics.residual(z[Block(i)], t)
			end
			empty_constraint = similar(initial_state_constraint, 0)

			# The robot grippers carry the pot to a raised delivery height. The child/pet
			# stays on the ground and does not receive this terminal height constraint.
			terminal_state_constraint = i <= 2 ?
				[xs[end][position_dimension] - goal_position[position_dimension]] :
				empty_constraint
			terminal_state_constraint = empty_constraint

			# Player 3 is modeled with the same 3D single-integrator dynamics, but its
			# vertical velocity is fixed to zero. Since its initial z is zero, the child
			# remains on the ground for the full trajectory.
			child_ground_constraint = i == 3 ?
				mapreduce(u -> [u[position_dimension]], vcat, us) :
				empty_constraint

			vcat(
				initial_state_constraint,
				dynamics_constraints,
				terminal_state_constraint,
				child_ground_constraint,
			)
		end for i in 1:num_players
	]

	shared_equality_constraint = function (z, θ)
		# Players 1 and 2 jointly carry the hot pot. This hard handle-grasp
		# constraint keeps the two robot grippers separated by the fixed 3D handle
		# distance dₚ at all times. Player 3 never directly manipulates the pot and
		# is deliberately excluded from this equality constraint.
		(; xs, us) = trajectory(z; player = 1)
		xs1 = copy(xs)
		(; xs, us) = trajectory(z; player = 2)
		xs2 = copy(xs)
		mapreduce(vcat, eachindex(xs1)) do t
			gripper_separation = xs1[t].- xs2[t]
			sum(abs2, gripper_separation) - dₚ^2
		end
	end

	equality_constraints = [
		i <= 2 ?
			((z, θ) -> vcat(
				player_equality_constraints[i](z, θ),
				shared_equality_constraint(z, θ),
			)) :
			player_equality_constraints[i]
		for i in 1:num_players
	]

	function robot_child_safety_inequality(; player)
		player in (1, 2) || error("Robot-child safety constraints belong only to robot players 1 and 2.")
		function (z, θ)
			(; xs) = trajectory(z; player)
			robot_xs = copy(xs)
			(; xs) = trajectory(z; player = 3)
			child_xs = copy(xs)

			# The child/pet is not constrained by this residual. Instead, each robot
			# player must keep its gripper outside the child-centered safety sphere.
			# This is the proof-of-concept Zero-Sum GOOP coupling: Player 3's curious
			# motion restricts the feasible robot motions indirectly through safety.
			mapreduce(vcat, eachindex(robot_xs)) do t
				separation = robot_xs[t] .- child_xs[t]
				[sum(abs2, separation) - collision_avoidance^2]
			end
		end
	end

	function robot_arm_speed_inequality(; player)
		player in (1, 2) || error("Robot arm speed constraints belong only to robot players 1 and 2.")
		function (z, θ)
			(; us) = trajectory(z; player)

			mapreduce(vcat, us) do u
				[speed_limit^2 - sum(abs2, u[1:position_dimension])]
			end
		end
	end

	function robot_inequality(; player)
		safety_inequality = robot_child_safety_inequality(; player)
		speed_inequality = robot_arm_speed_inequality(; player)

		function (z, θ)
			vcat(
				safety_inequality(z, θ),
				speed_inequality(z, θ),
			)
		end
	end

	function child_ground_speed_inequality(z, θ)
		(; us) = trajectory(z; player = 3)

		# This is a physical child-motion bound, not a robot safety constraint. The
		# safety constraints remain robot-only; Player 3 is merely limited to a
		# plausible ground speed so it cannot teleport into the pot center.
		mapreduce(vcat, us) do u
			[(speed_limit/2)^2 - sum(abs2, u[1:2])]
		end
	end

	inequality_constraints = [
		robot_inequality(; player = 1),
		robot_inequality(; player = 2),
		child_ground_speed_inequality,
	]

	# Load balance objective
	function load_balance_objective(z, θ)
		(; xs, us) = trajectory(z; player = 1)
		xs1 = copy(xs)
		(; xs, us) = trajectory(z; player = 2)
		xs2 = copy(xs)
		sum(eachindex(xs1)) do t
    		(xs1[t][position_dimension] - xs2[t][position_dimension])^2
		end
	end

	function negative_load_balance_objective(z, θ)
		-load_balance_objective(z, θ)
	end

	function pot_approach_objective(z, θ)
		(; xs) = trajectory(z; player = 1)
		xs1 = copy(xs)
		(; xs) = trajectory(z; player = 2)
		xs2 = copy(xs)
		(; xs) = trajectory(z; player = 3)
		xs3 = copy(xs)

		# The child/pet is curious, not physically coupled to the pot.
		# Approach the pot center c = (p₁ + p₂) / 2.
		sum(eachindex(xs3)) do t
			pot_center = 0.5 .* (xs1[t] .+ xs2[t])
			sum(abs2, xs3[t] .- pot_center)
		end
	end

	preferences = [
		[
			# control_objective(; player = 1),
			goal_objective(; player = 1),
			load_balance_objective,
			# goal_objective(; player = 1),
		],
		[
			# control_objective(; player = 2),
			# goal_objective(; player = 2),
			load_balance_objective,
			goal_objective(; player = 2),
		],
		[
			negative_load_balance_objective,
			pot_approach_objective,
		],
	]

	# Preference hierarchy: [lowest priority, ..., highest priority]
	is_prioritized_constraint = [[false, false], [false, false], [false, false]]

	function build_goop_problem()
		@timeit TO "ParametricGOOP construction" begin
			ReducedGOOP.ParametricGOOP(
				dummy_primals,
				dummy_parameters;
				preferences = use_scalarized_baseline ? scalarized_preferences : preferences,
				is_prioritized_constraint = use_scalarized_baseline ? scalarized_is_prioritized_constraint : is_prioritized_constraint,
				equality_constraints,
				inequality_constraints,
				shared_equality_constraint = nothing,
				shared_inequality_constraint = nothing,
			)
		end
	end

	function evaluate_preference_level(preference, is_constraint, level, z, θ)
		preference_value = preference(z, θ)
		if is_constraint
			return sum(smooth_piecewise_preference_objective.(preference_value, level))
		end
		preference_value
	end

	function scalarized_player_preference(player_preferences, player_is_prioritized_constraint)
		@assert length(player_preferences) == length(player_is_prioritized_constraint)

		function (z, θ)
			accumulated_preference = 0
			for (level, (preference, is_constraint)) in
				enumerate(zip(player_preferences, player_is_prioritized_constraint))
				accumulated_preference += evaluate_preference_level(
					preference,
					is_constraint,
					level,
					z,
					θ,
				)
			end
			accumulated_preference
		end
	end

	" Scalarized baseline (Nash / no hierarchy): flattens hierarchical preferences into a single objective per player"
	scalarized_player_preferences =
		map(scalarized_player_preference, preferences, is_prioritized_constraint)
	scalarized_preferences = [[preference] for preference in scalarized_player_preferences]
	scalarized_is_prioritized_constraint = [[false] for _ in scalarized_preferences]

	" Social equilibrium (no Nash / no hierarchy) baseline: sum of all players' preferences, single optimization problem"
	objective = function (z, θ)
		accumulated_objective = 0
		for preference in scalarized_player_preferences
			accumulated_objective += preference(z, θ)
		end
		accumulated_objective
	end
	equality_constraint = function (z, θ)
		vcat(
			mapreduce(f -> f(z, θ), vcat, player_equality_constraints),
			shared_equality_constraint(z, θ),
		)
	end
	inequality_constraint = nothing
	inequality_constraint = function (z, θ)
		vcat(
			inequality_constraints[1](z, θ),
			inequality_constraints[2](z, θ),
			inequality_constraints[3](z, θ),
		)
	end
	primal_dimension = sum(primal_dimensions)
	parameter_dimension = sum(parameter_dimensions)
	equality_dimension = length(equality_constraint(dummy_primals, dummy_parameters))
	inequality_dimension = length(inequality_constraint(dummy_primals, dummy_parameters))

	function build_social_problem()
		@timeit TO "ParametricOptimizationProblem construction" begin
			ReducedGOOP.ParametricOptimizationProblem(;
				objective,
				equality_constraint,
				inequality_constraint,
				parameter_dimension,
				primal_dimension,
				equality_dimension,
				inequality_dimension,
				num_players,
			)
		end
	end

	# Build problem
	problem = @timeit TO "preference and constraint construction" begin
		use_social_equilibrium_baseline ? build_social_problem() : build_goop_problem()
	end

	(; problem, flatten_parameters)
end

# ── Experiment entry point ─────────────────────────────────────────────────────

function demo(;
	planning_horizon = 20,
	map_end = 10,
	lane_width = 2,
	verbose = false,
	plot = false,
	save = false,
	rng_seed = 123,
	random_initial_state = false,
	debug = false,
	use_scalarized_baseline = false,
	use_social_equilibrium_baseline = false,
	show_interactive_trajectory = false,
	# ── Real-world simulation kwargs (pass from Python after grip) ──────────
	# When provided the solver operates in MuJoCo world-frame (metres).
	sim_eef0 = nothing,          # robot0 EEF position [x, y, z] (world frame)
	sim_eef1 = nothing,          # robot1 EEF position [x, y, z] (world frame)
	sim_eef2 = nothing,          # robot2/GR1 EEF position [x, y, z] (world frame)
	sim_lift_height = 0.35,      # target lift height above grip position (metres)
)
	reset_timer!(TO)
	@timeit TO "experiment setup" Random.seed!(rng_seed)

	# ── Settings ───────────────────────────────────────────────────────────────
	run_id = "Robotic_arm"
	dynamics_model = Robotic_arm.SingleIntegrator3D()
	goop_version = :reduced               # :complete | :reduced | :quasi
	solver = ReducedGOOP.InteriorPoint() # ReducedGOOP.InteriorPoint() | ReducedGOOP.PATHSolver()
	linesearch = :backtracking          # :backtracking | :fraction_to_boundary
	compute_warmstart = true # Whether to compute a warmstart trajectory via rollout (true) or load from file (false)

	# ── World-frame mode detection ────────────────────────────────────────────
	# When sim_eef0/1/2 are provided (from Python after grip), the solver uses
	# MuJoCo world-frame metres. Otherwise abstract coordinates are used.
	use_world_frame = !isnothing(sim_eef0) && !isnothing(sim_eef1)

	# ── Problem parameters ─────────────────────────────────────────────────────
	num_players          = 3
	# World-frame values are realistic physical quantities (metres, m/s).
	# Abstract values are the legacy standalone-test scale.
	collision_avoidance  = use_world_frame ? 0.25 : 2.0   # safety sphere radius
	child_initial_buffer = use_world_frame ? 0.5  : 4.0
	speed_limit          = use_world_frame ? 0.5  : 3.0   # m/s (world) or abstract
	num_instances        = 1
	perturbation_scale   = use_world_frame ? 0.01 : 0.3
	dₚ                   = use_world_frame ? 0.34 : 2.0   # handle separation (m)

	# ── Solver schedule ────────────────────────────────────────────────────────
	epsilon_schedule         = [0.1]
	max_inner_iters_schedule = fill(1000, length(epsilon_schedule))

	# ── Cache paths ───────────────────────────────────────────────────────────
	# Use separate cache files for world-frame vs abstract-coordinate modes so the
	# two parameter sets (different dₚ, speed_limit, etc.) don't share a KKT cache.
	root_path           = joinpath(@__DIR__, "data", run_id)
	cache_suffix        = use_world_frame ? "_world" : ""
	kkt_cache_file      = joinpath(root_path, "cache_kkt_system$(cache_suffix).jld2")
	solution_cache_file = joinpath(root_path, "cache_solution$(cache_suffix).jld2")

	# Disable plotting, saving, and interactive trajectory if cache exists
	if isfile(kkt_cache_file)
		plot = false
		save = false
		show_interactive_trajectory = false
	end

	# ── Scenario ───────────────────────────────────────────────────────────────
	# Single Integrator 3D: state = [px, py, pz]
	if use_world_frame
		# World-frame (metres): robot 0 on +y side, robot 1 on −y side, GR1 on −x side.
		# Table surface z ≈ 0.8 m; handles at z ≈ 0.84 m, ±0.17 m from pot centre in y.
		# Goals: lift the pot straight up by sim_lift_height while holding the handles.
		base_initial_state1 = collect(Float64, sim_eef0)
		base_initial_state2 = collect(Float64, sim_eef1)
		goal_position1      = base_initial_state1 .+ [0.0, 0.0, sim_lift_height]
		goal_position2      = base_initial_state2 .+ [0.0, 0.0, sim_lift_height]
		initial_state3      = isnothing(sim_eef2) ? [-0.55, 0.0, 1.0] : collect(Float64, sim_eef2)
		goal_position3      = 0.5 .* (base_initial_state1 .+ base_initial_state2)  # pot centre
	else
		# Abstract coordinates for standalone testing.
		base_initial_state1 = [-1.0,  6.0, 0.0]
		base_initial_state2 = [ 1.0,  6.0, 0.0]
		goal_position1      = [-1.0, -5.0, 5.0]
		goal_position2      = [ 1.0, -5.0, 5.0]
		initial_state3      = [-6.0, -1.0, 0.0]
		goal_position3      = [ 0.0,  0.0, 0.0]
	end
	obstacle_position   = [0.25, 0.15, 0.0]   # placeholder

	# ── Build dynamics and problem ─────────────────────────────────────────────
	state_dimension   = 3
	control_dimension = 3
	Δt                = 0.1
	# World-frame: control = velocity (m/s); ±1 m/s is realistic for a robot arm.
	# Abstract: keep legacy large bounds.
	control_bounds    = use_world_frame ?
		(; lb = [-1.0, -1.0, -1.0], ub = [1.0, 1.0, 1.0]) :
		(; lb = [-10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0])

	dynamics = @timeit TO "dynamics construction" build_dynamics(dynamics_model; Δt, state_dimension, control_dimension)

	@timeit TO "problem setup" begin
		(; problem, flatten_parameters) = get_setup(
			num_players;
			dynamics,
			control_bounds,
			planning_horizon,
			dₚ,
			collision_avoidance,
			speed_limit,
			map_end,
			lane_width,
			use_scalarized_baseline,
			use_social_equilibrium_baseline,
		)
	end
	kkt_generators = if solver isa ReducedGOOP.InteriorPoint
		Dict(
			:complete => ReducedGOOP.generate_slacked_complete_kkt_system,
			:reduced  => ReducedGOOP.generate_slacked_reduced_kkt_system,
			:quasi    => ReducedGOOP.generate_slacked_quasi_kkt_system,
		)
	else
		Dict(
			:complete => ReducedGOOP.generate_mcp_complete_kkt_system,
			:reduced  => ReducedGOOP.generate_mcp_reduced_kkt_system,
		)
	end

	GOOP_kkt_generator = get(kkt_generators, goop_version, nothing)
	isnothing(GOOP_kkt_generator) && error("Unknown GOOP version: $(goop_version)")

	@info "Building KKT system for $(goop_version) GOOP formulation and $(solver) solver..."
	# Check if problem is not an instance of GOOPKKTSystem. Otherwise, build GOOPKKTSystem.
	GOOP_kkt_system = @timeit TO "KKT construction" begin
		if isfile(kkt_cache_file)
			@info "Loading cached KKT system from $(kkt_cache_file)"
			JLD2.load_object(kkt_cache_file)
		elseif problem isa ReducedGOOP.GOOPKKTSystem
			problem
		else
			kkt = GOOP_kkt_generator(problem)
			JLD2.save_object(kkt_cache_file, kkt)
			kkt
		end
	end

	if solver isa ReducedGOOP.InteriorPoint
		println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
		println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)
	else
		println("[PATH] MCP Dimension: ", GOOP_kkt_system.problem_size)
		println("[PATH] Variable Dimension: ", length(GOOP_kkt_system.lower_bounds))
	end

	dynamics_dimension = dynamics.state_dimension + dynamics.control_dimension
	primal_dimension   = dynamics_dimension * planning_horizon

	# ── Per-instance solver ────────────────────────────────────────────────────
	function solve_game_instance(θ; z₀, ϵ₀, max_inner_iters)
		options = @timeit TO "solver options construction" if solver isa ReducedGOOP.InteriorPoint
			ReducedGOOP.InteriorPointOptions(;
				tol = 1e-2, #1e-4
				η₀ = 5e-5, # 5e-5, 0.0 to turn off Tikhonov
				ϵ₀,
				max_inner_iters,
				max_outer_iters = 1,
				tightening_rate = 1.2, # high => weak decrease in η
				loosening_rate = 3.0, # low => strong increase in η
				min_stepsize = 1e-20,
				linesearch,
				linear_solve_algorithm = ReducedGOOP.LinearSolve.KrylovJL_LSMR(),
				use_linsolve = false,
				record_convergence = true,
				record_condition_number = true,
				eta_retry_growth = 0.3,
				perturbation_enabled = false,
				stagnation_rtol = 1e-1,
				perturbation_scale = 1e-6,
				tsvd_threshold = 0.0, # 0.0: pure Tikhonov, > 0 and η = 0: pure TSVD
				use_marquardt_scaling = false,
				verbose,
			)
		else
			ReducedGOOP.PATHOptions(;
				convergence_tolerance = 1e-4,
				ϵ₀,
				cumulative_iteration_limit = 1_000_000,
				proximal_perturbation = 1e-2,
				major_iteration_limit = 10_000,
				minor_iteration_limit = 15_000,
				nms_initial_reference_factor = 50_000,
				nms_maximum_watchdogs = 8_000,
				nms_memory_size = 16_000,
				nms_mstep_frequency = 5_000,
				lemke_start_type = "advanced",
				lemke_rank_deficiency_iterations = 50,
				restart_limit = 120,
				gradient_step_limit = 120,
				use_basics = true,
				use_start = true,
				verbose,
			)
		end

		@info "Solving game instance with $(solver)..."
		kkt_error_history = Float64[]
		condition_number_history = Float64[]
		eta_history = Float64[]
		total_iters = 0
		solver_status = :solved
		elapsed_time = @elapsed begin
			output = @timeit TO "solver invocation" ReducedGOOP.solve(
					solver,
					GOOP_kkt_system,
					θ;
					z₀,
					options,
				)
			if solver isa ReducedGOOP.InteriorPoint
				(;
					status,
					z,
					x,
					kkt_error,
					ϵ,
					total_iters,
					kkt_error_history,
					condition_number_history,
				) = output
				eta_history = hasproperty(output, :eta_history) ? output.eta_history : Float64[]
				if status == :failed
					println("  [solver exit] total_iters=$(total_iters), kkt_error=$(round(kkt_error; sigdigits=4)), tol=$(options.tol)")
				end
				solver_status = status
			else
				(; status, z, ϵ, info) = output
				@show status
				Int(status) != 1 && return nothing
				kkt_error = info.residual
				x = z[1:(num_players*primal_dimension)]
				solver_status = :solved
			end
		end

		strategies = @timeit TO "solution postprocessing" extract_player_strategies(
				x,
				num_players,
				primal_dimension,
				dynamics,
			)

		solution_dict = Dict(
			"strategies" => strategies,
			"z" => z,
			"x" => x,
			"solve_time_sec" => elapsed_time,
			"kkt_error" => kkt_error,
			"ϵ" => ϵ,
			"status" => solver_status,
			"total_iters" => total_iters,
			"kkt_error_history" => kkt_error_history,
			"condition_number_history" => condition_number_history,
			"eta_history" => eta_history,
		)

		(; strategies, solution_dict)
	end

	# ── Output directories ─────────────────────────────────────────────────────
	(;
		run_dir,
		problem_data_dir,
		solution_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		speed_plots_dir,
		control_plots_dir,
		distance_plots_dir,
		warmstart_plots_dir,
	) = @timeit TO "output directory setup" prepare_robotic_arm_output_dirs(root_path; debug)

	# ── Main solve loop ────────────────────────────────────────────────────────
	instance_problem_data = Dict{String, Any}[]
	solved_attempts       = 0
	total_attempts        = 0

	solution_dict = nothing

	while solved_attempts < num_instances
		total_attempts += 1
		initial_state1, initial_state2 = if random_initial_state
			(
				sample_initial_state(
					dynamics_model,
					base_initial_state1,
					dynamics.state_dimension,
					perturbation_scale,
				),
				sample_initial_state(
					dynamics_model,
					base_initial_state2,
					dynamics.state_dimension,
					perturbation_scale,
				),
			)
		else
			(copy(base_initial_state1), copy(base_initial_state2))
		end
		# initial_state3, goal_position3 = choose_child_initial_and_goal(
		# 	initial_state1,
		# 	initial_state2,
		# 	goal_position1,
		# 	goal_position2,
		# 	collision_avoidance,
		# 	child_initial_buffer = child_initial_buffer,
		# )

		println(
			"solved $(solved_attempts)/$(num_instances), attempt $(total_attempts), goop version $(goop_version): ",
		)
		println("initial_state1:", initial_state1)
		println("goal_position1:", goal_position1)
		println("initial_state2:", initial_state2)
		println("goal_position2:", goal_position2)
		println("initial_state3:", initial_state3)
		println("goal_position3:", goal_position3)

		(; θ1, θ2, θ3, θ) = @timeit TO "instance parameter construction" build_instance_parameters(
				flatten_parameters,
				initial_state1,
				initial_state2,
				initial_state3,
				goal_position1,
				goal_position2,
				goal_position3,
				obstacle_position,
			)

		(; warmstart_solution) = @timeit TO "warmstart construction" if compute_warmstart
			# World-frame hint: lift straight up. Abstract: move in −y and +z.
			robot_control_hint = use_world_frame ?
				[0.0, 0.0, sim_lift_height / (Δt * (planning_horizon - 1))] :
				[0.0, -1.0, 1.0]
			build_default_warmstart(
				planning_horizon,
				dynamics,
				initial_state1,
				initial_state2,
				initial_state3,
				goal_position3;
				speed_limit,
				robot_control_hint,
			)
		else
			(;
				warmstart_solution = load("experiments/solution_dict_instance_1_eps0.1.jld2")["single_stored_object"]["x"][1:(num_players*primal_dimension)],
			)
		end
		if plot
			@timeit TO "warmstart visualization" begin
				save_warmstart_visualizations(
					warmstart_solution,
					warmstart_plots_dir,
					total_attempts,
					solved_attempts + 1,
					num_players,
					primal_dimension,
					dynamics,
					map_end,
					lane_width,
					θ1,
					θ2,
					θ3,
					goal_position1,
					goal_position2,
					goal_position3,
					collision_avoidance,
					dₚ,
					speed_limit,
					control_bounds,
				)
			end
		end

		epsilon_results = Pair{Float64, Any}[]
		stage_warmstart = warmstart_solution
		solve_sequence_succeeded = true
		instance_total_solve_time_sec = 0.0

		for (ϵ₀, max_inner_iters) in zip(epsilon_schedule, max_inner_iters_schedule)
			loaded_from_cache = isfile(solution_cache_file)
			result = try
				if loaded_from_cache
					@info "Loading cached solution from $(solution_cache_file)"
					cached_dict = JLD2.load_object(solution_cache_file)
					cached_strategies = extract_player_strategies(cached_dict["x"], num_players, primal_dimension, dynamics)
					(; strategies = cached_strategies, solution_dict = cached_dict)
				else
					r = solve_game_instance(θ; z₀ = stage_warmstart, ϵ₀, max_inner_iters)
					isnothing(r) || JLD2.save_object(solution_cache_file, r.solution_dict)
					r
				end
			catch err
				rethrow(err)
			end
			if isnothing(result)
				println(
					"attempt $(total_attempts): failed to converge for ϵ₀ = $(ϵ₀), resampling.",
				)
				solve_sequence_succeeded = false
				break
			end
			push!(epsilon_results, ϵ₀ => result)
			instance_total_solve_time_sec += result.solution_dict["solve_time_sec"]
			# When loading from cache we always accept the result (it was accepted once already).
			# Only treat a fresh solve as a failure if the solver didn't converge.
			if !loaded_from_cache && result.solution_dict["status"] == :failed
				println(
					"attempt $(total_attempts): failed to converge for ϵ₀ = $(ϵ₀), saving diagnostics.",
				)
				solve_sequence_succeeded = false
				break
			elseif loaded_from_cache && result.solution_dict["status"] == :failed
				@warn "Loaded cached solution has status :failed (solver did not converge on original run). Using it anyway."
			end
			stage_warmstart = warmstart_solution
		end

		if !solve_sequence_succeeded
			failed_instance_idx = solved_attempts + 1
			if save
				for (ϵ₀, result) in epsilon_results
					failed_suffix = "_attempt_$(total_attempts)_failed"
					JLD2.save_object(
						joinpath(
							solution_data_dir,
							"solution_dict_instance_$(failed_instance_idx)_eps$(ϵ₀)$(failed_suffix).jld2",
						),
						result.solution_dict,
					)
					save_convergence_diagnostics(
						result.solution_dict,
						convergence_plots_dir,
						failed_instance_idx,
						ϵ₀;
						filename_suffix = failed_suffix,
					)
				end
			end
			if !random_initial_state
				println("solver failed for default initial states.")
				break
			end
			continue
		end

		push!(
			instance_problem_data,
			Dict(
				"attempt_idx" => total_attempts,
				"initial_state1" => initial_state1,
				"goal_position1" => goal_position1,
				"initial_state2" => initial_state2,
				"goal_position2" => goal_position2,
				"initial_state3" => initial_state3,
				"goal_position3" => goal_position3,
				"total_solve_time_sec" => instance_total_solve_time_sec,
			),
		)

		solved_attempts += 1
		println(
			"instance $(solved_attempts) total solve time: $(round(instance_total_solve_time_sec; digits = 5)) sec",
		)
		println("instance $(solved_attempts) converged preference values by ϵ:")

		if save
			@timeit TO "problem data save" begin
				JLD2.save_object(
					joinpath(problem_data_dir, "problem_data_instance_$(solved_attempts).jld2"),
					instance_problem_data,
				)
			end
		end

		for (ϵ₀, result) in epsilon_results
			solution_dict = result.solution_dict
			
			if save || plot
				@timeit TO "solution output and plotting" begin
					if save
						JLD2.save_object(
						joinpath(
							solution_data_dir,
							"solution_dict_instance_$(solved_attempts)_eps$(ϵ₀).jld2",
							),
							solution_dict,
							)
					end
					if plot
						trajectory_fig, _ = plot_single_integrator_3d_trajectories(
							;
							map_end,
							lane_width,
							strategy = result.strategies,
							θ1,
							θ2,
							θ3,
							goal_position1,
							goal_position2,
							goal_position3,
							collision_avoidance,
						)
						speed_fig, _ = speed_plot(;
							strategy = result.strategies,
							speed_limit = speed_limit,
							dynamics_model,
							speed_limit_players = 1:num_players,
						)
						control_fig, _ = control_plot(;
							strategy = result.strategies,
							control_lb = control_bounds.lb,
							control_ub = control_bounds.ub,
						)
						distance_fig, _ = inter_player_distance_plot(;
							strategy = result.strategies,
							reference_distance = dₚ,
						)

						CairoMakie.save(
							joinpath(
								trajectory_plots_dir,
								"trajectory_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
							),
							trajectory_fig,
						)
						CairoMakie.save(
							joinpath(
								speed_plots_dir,
								"speed_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
							),
							speed_fig,
						)
						CairoMakie.save(
							joinpath(
								control_plots_dir,
								"control_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
							),
							control_fig,
						)
						CairoMakie.save(
							joinpath(
								distance_plots_dir,
								"distance_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
							),
							distance_fig,
						)
						if show_interactive_trajectory
							interactive_trajectory_path = joinpath(
								trajectory_plots_dir,
								"trajectory_interactive_instance_$(solved_attempts)_eps$(ϵ₀).html",
							)
							plot_trajectory_3d_interactive(
								;
								map_end,
								lane_width,
								strategy = result.strategies,
								θ1,
								θ2,
								θ3,
								goal_position1,
								goal_position2,
								goal_position3,
								collision_avoidance,
								display_figure = false,
								save_path = interactive_trajectory_path,
							)
							println("saved interactive trajectory browser file: ", interactive_trajectory_path)
						end
					end
				end
			end
		end
	end

	if save
		@timeit TO "run metadata save" begin
			JLD2.save_object(
				joinpath(problem_data_dir, "run_metadata.jld2"),
				Dict(
					"run_id" => run_id,
					"debug" => debug,
					"run_dir" => run_dir,
					"rng_seed" => rng_seed,
					"dynamics_model" => dynamics_model_name(dynamics_model),
					"num_instances" => num_instances,
					"random_initial_state" => random_initial_state,
					"use_scalarized_baseline" => use_scalarized_baseline,
					"use_social_equilibrium_baseline" => use_social_equilibrium_baseline,
					"show_interactive_trajectory" => show_interactive_trajectory,
					"epsilon_schedule" => epsilon_schedule,
					"max_inner_iters_schedule" => max_inner_iters_schedule,
					"speed_limit" => speed_limit,
					"collision_avoidance" => collision_avoidance,
					"child_initial_buffer" => child_initial_buffer,
					"dₚ" => dₚ,
					"perturbation_scale" => perturbation_scale,
				),
			)
		end
	end

	println("\nTiming summary:")
	show(TO)
	println()

	# Attach scenario parameters to solution dict so Python can read them back.
	if !isnothing(solution_dict)
		merge!(solution_dict, Dict(
			"initial_state1" => base_initial_state1,
			"goal_position1" => goal_position1,
			"initial_state2" => base_initial_state2,
			"goal_position2" => goal_position2,
			"initial_state3" => initial_state3,
			"goal_position3" => goal_position3,
			"use_world_frame" => use_world_frame,
			"dp" => dₚ,
		))
		if use_world_frame
			println("\n[world-frame] Planned trajectory summary:")
			for (player_idx, strategy) in enumerate(solution_dict["strategies"])
				println("  Player $(player_idx):")
				for (t, x) in enumerate(strategy.xs)
					println("    t=$(lpad(t-1, 2))  x=[$(round(x[1]; digits=4)), $(round(x[2]; digits=4)), $(round(x[3]; digits=4))]")
				end
			end
		end
	end

	return solution_dict;
end

# ── Output / plotting helpers ──────────────────────────────────────────────────

function save_warmstart_visualizations(
	warmstart_solution,
	warmstart_plots_dir,
	total_attempts,
	instance_idx,
	num_players,
	primal_dimension,
	dynamics,
	map_end,
	lane_width,
	θ1,
	θ2,
	θ3,
	goal_position1,
	goal_position2,
	goal_position3,
	collision_avoidance,
	dₚ,
	speed_limit,
	control_bounds,
)
	warmstart_strategies = extract_player_strategies(
		warmstart_solution,
		num_players,
		primal_dimension,
		dynamics,
	)

	warmstart_fig, _ = plot_single_integrator_3d_trajectories(
		;
		map_end,
		lane_width,
		strategy = warmstart_strategies,
		θ1,
		θ2,
		θ3,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
	)
	warmstart_speed_fig, _ = speed_plot(
		;
		strategy = warmstart_strategies,
		speed_limit = speed_limit,
		dynamics_model = dynamics.model,
		speed_limit_players = 1:num_players,
	)
	warmstart_control_fig, _ = control_plot(
		;
		strategy = warmstart_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	warmstart_distance_fig, _ = inter_player_distance_plot(
		;
		strategy = warmstart_strategies,
		reference_distance = dₚ,
	)

	CairoMakie.save(
		joinpath(
			warmstart_plots_dir,
			"warmstart_attempt_$(total_attempts)_instance_$(instance_idx).pdf",
		),
		warmstart_fig,
	)
	CairoMakie.save(
		joinpath(
			warmstart_plots_dir,
			"warmstart_speed_attempt_$(total_attempts)_instance_$(instance_idx).pdf",
		),
		warmstart_speed_fig,
	)
	CairoMakie.save(
		joinpath(
			warmstart_plots_dir,
			"warmstart_control_attempt_$(total_attempts)_instance_$(instance_idx).pdf",
		),
		warmstart_control_fig,
	)
	CairoMakie.save(
		joinpath(
			warmstart_plots_dir,
			"warmstart_distance_attempt_$(total_attempts)_instance_$(instance_idx).pdf",
		),
		warmstart_distance_fig,
	)
end

function prepare_robotic_arm_output_dirs(root_path; debug)
	run_dir = if debug
		joinpath(root_path, "debug")
	else
		joinpath(root_path, "runs")
	end

	data_dir              = joinpath(run_dir, "data")
	problem_data_dir      = joinpath(data_dir, "problem")
	solution_data_dir     = joinpath(problem_data_dir, "solution")
	plots_dir             = joinpath(run_dir, "plots")
	trajectory_plots_dir  = joinpath(plots_dir, "trajectories")
	convergence_plots_dir = joinpath(plots_dir, "convergence")
	speed_plots_dir       = joinpath(plots_dir, "speed")
	control_plots_dir     = joinpath(plots_dir, "controls")
	distance_plots_dir    = joinpath(plots_dir, "distance")
	warmstart_plots_dir   = joinpath(plots_dir, "warmstart")

	for dir in (
		problem_data_dir,
		solution_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		speed_plots_dir,
		control_plots_dir,
		distance_plots_dir,
		warmstart_plots_dir,
	)
		mkpath(dir)
	end

	(;
		run_dir,
		data_dir,
		problem_data_dir,
		solution_data_dir,
		plots_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		speed_plots_dir,
		control_plots_dir,
		distance_plots_dir,
		warmstart_plots_dir,
	)
end

# ── Internal utilities ─────────────────────────────────────────────────────────

function build_dynamics(
	model::SingleIntegrator3D;
	Δt = 0.5,
	state_dimension = 3,
	control_dimension = 3,
)
	state_dimension == 3 || error("SingleIntegrator3D expects a 3D state.")
	control_dimension == 3 || error("SingleIntegrator3D expects a 3D control input.")

	residual(z, t) = single_integrator_3d(z, t; Δt, state_dimension, control_dimension)
	step(x, u, t) = single_integrator_3d_step(x, u; Δt)
	(; model, residual, step, Δt, state_dimension, control_dimension)
end

function dynamics_model_name(::SingleIntegrator3D)
	"single_integrator_3d"
end


function extract_player_strategies(
	primal_solution,
	num_players,
	primal_dimension,
	dynamics,
)
	map(1:num_players) do player_idx
		start_idx = primal_dimension * (player_idx - 1) + 1
		end_idx   = start_idx + primal_dimension - 1
		unflatten_trajectory(
			primal_solution[start_idx:end_idx],
			dynamics.state_dimension,
			dynamics.control_dimension,
		)
	end
end

function evaluate_preferences_at_solution(problem, x, θ)
	x_block = BlockArray(collect(x), problem.primal_dims)
	θ_block = BlockArray(collect(θ), problem.parameter_dims)
	map(1:problem.num_players) do player
		map(problem.preferences[player]) do preference
			preference(x_block, θ_block)
		end
	end
end

function planar_distance(a, b)
	sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2)
end

function choose_child_initial_and_goal(
	initial_state1,
	initial_state2,
	goal_position1,
	goal_position2,
	collision_avoidance,
	;
	child_initial_buffer = 1.2,
)
	start_center = 0.5 .* (initial_state1 .+ initial_state2)
	goal_center = 0.5 .* (goal_position1 .+ goal_position2)
	path_xy = goal_center[1:2] .- start_center[1:2]
	path_norm = sqrt(sum(abs2, path_xy))
	path_norm > 0 || error("Cannot place child for a degenerate robot delivery path.")

	# Put the child on the side of the nominal pot path, then aim its nominal
	# ground-level warm start toward the delivery corridor. This starts outside
	# the initial safety radius while naturally intercepting the robots' route.
	side_xy = [path_xy[2], -path_xy[1]] ./ path_norm
	initial_offset = collision_avoidance + child_initial_buffer
	goal_offset = 0.25 * collision_avoidance

	child_initial_xy = start_center[1:2] .+ 0.35 .* path_xy .+ initial_offset .* side_xy
	child_goal_xy = start_center[1:2] .+ 0.80 .* path_xy .+ goal_offset .* side_xy
	child_initial = [child_initial_xy[1], child_initial_xy[2], 0.0]
	child_goal = [child_goal_xy[1], child_goal_xy[2], 0.0]

	initial_clearance = min(
		planar_distance(child_initial, initial_state1),
		planar_distance(child_initial, initial_state2),
	)
	if initial_clearance <= collision_avoidance
		child_initial[1:2] .+= (collision_avoidance - initial_clearance + 0.5) .* side_xy
	end

	(child_initial, child_goal)
end

function build_instance_parameters(
	flatten_parameters,
	initial_state1,
	initial_state2,
	initial_state3,
	goal_position1,
	goal_position2,
	goal_position3,
	obstacle_position,
)
	θ1 = flatten_parameters(;
		initial_state = initial_state1,
		goal_position = goal_position1,
		obstacle_position = obstacle_position,
	)
	θ2 = flatten_parameters(;
		initial_state = initial_state2,
		goal_position = goal_position2,
		obstacle_position = obstacle_position,
	)
	θ3 = flatten_parameters(;
		initial_state = initial_state3,
		goal_position = goal_position3,
		obstacle_position = obstacle_position,
	)
	(; θ1, θ2, θ3, θ = [θ1..., θ2..., θ3...])
end

function build_default_warmstart(
	planning_horizon,
	dynamics,
	initial_state1,
	initial_state2,
	initial_state3,
	goal_position3;
	speed_limit = 1.5,
	robot_control_hint = [0.0, -1.0, 1.0],
)
	player1_warmstart = build_constant_control_warmstart(
		planning_horizon,
		dynamics,
		initial_state1,
		robot_control_hint,
	)

	player2_warmstart = build_constant_control_warmstart(
		planning_horizon,
		dynamics,
		initial_state2,
		robot_control_hint,
	)

	player3_warmstart = build_ground_line_warmstart(
		planning_horizon,
		dynamics,
		initial_state3,
		goal_position3,
	)

	# player2_vx_profile = fill(0.0, planning_horizon)
	# player2_vy_profile = vcat(1.0, fill(speed_limit, planning_horizon - 1))
	# player2_warmstart = build_planar_di_speed_profile_warmstart(
	# 	planning_horizon,
	# 	dynamics,
	# 	initial_state2;
	# 	vx_profile = player2_vx_profile,
	# 	vy_profile = player2_vy_profile,
	# )

	warmstart_solution = flatten_warmstart_solution(
		planning_horizon,
		[player1_warmstart.xs, player2_warmstart.xs, player3_warmstart.xs],
		[player1_warmstart.us, player2_warmstart.us, player3_warmstart.us],
	)
	(; warmstart_solution)
end

function infer_planar_di_timestep(dynamics)
	x0       = [0.0, 0.0, 0.0, 0.0]
	u_unit_y = [0.0, 1.0]
	x1       = dynamics.step(x0, u_unit_y, 1)
	dt       = x1[4]
	dt <= 0.0 && error("Failed to infer planar double-integrator timestep from dynamics.")
	dt
end

function build_constant_control_warmstart(planning_horizon, dynamics, initial_state, constant_control)
	length(initial_state) == dynamics.state_dimension || error("Initial state dimension mismatch.")
	length(constant_control) == dynamics.control_dimension || error("Control dimension mismatch.")

	xs = [collect(initial_state)]
	us = [collect(constant_control)]
	for t in 1:(planning_horizon-1)
		push!(xs, dynamics.step(xs[t], us[1], t))
		push!(us, copy(us[1]))
	end
	us[end] = zeros(dynamics.control_dimension)
	(; xs, us)
end

function build_ground_line_warmstart(planning_horizon, dynamics, initial_state, goal_position)
	length(initial_state) == dynamics.state_dimension || error("Initial state dimension mismatch.")
	length(goal_position) == dynamics.state_dimension || error("Goal position dimension mismatch.")
	planning_horizon >= 2 || error("Ground-line warm start requires at least two time steps.")

	total_time = dynamics.Δt * (planning_horizon - 1)
	constant_control = (goal_position .- initial_state) ./ total_time
	constant_control[3] = 0.0

	xs = [collect(initial_state)]
	us = [collect(constant_control)]
	for t in 1:(planning_horizon-1)
		push!(xs, dynamics.step(xs[t], us[1], t))
		xs[end][3] = 0.0
		push!(us, copy(us[1]))
	end
	us[end] = zeros(dynamics.control_dimension)
	(; xs, us)
end

function build_planar_di_speed_profile_warmstart(
	planning_horizon,
	dynamics,
	initial_state;
	vx_profile,
	vy_profile,
)
	length(initial_state) == dynamics.state_dimension || error("Expected $(dynamics.state_dimension)D planar-double-integrator state.")
	dynamics.control_dimension == 2 || error("Expected 2D control input for planar-double-integrator.")
	length(vx_profile) == planning_horizon || error("vx_profile length must equal planning_horizon.")
	length(vy_profile) == planning_horizon || error("vy_profile length must equal planning_horizon.")

	dt = infer_planar_di_timestep(dynamics)
	x0 = [initial_state[1], initial_state[2], vx_profile[1], vy_profile[1]]
	xs = [x0]
	us = Vector{Vector{Float64}}()
	for t in 1:(planning_horizon-1)
		u = [
			(vx_profile[t+1] - vx_profile[t]) / dt,
			(vy_profile[t+1] - vy_profile[t]) / dt,
		]
		push!(us, u)
		push!(xs, dynamics.step(xs[t], u, t))
	end
	push!(us, [0.0, 0.0])
	(; xs, us)
end

function flatten_warmstart_solution(planning_horizon, warmstart_x, warmstart_u)
	length(warmstart_x) == length(warmstart_u) ||
		error("warmstart_x and warmstart_u must have the same number of players.")
	warmstart_solution = Float64[]
	for player in eachindex(warmstart_x)
		length(warmstart_x[player]) == planning_horizon ||
			error("State warm-start horizon mismatch for player $(player).")
		length(warmstart_u[player]) == planning_horizon ||
			error("Control warm-start horizon mismatch for player $(player).")
		warmstart_primals = mapreduce(vcat, 1:planning_horizon) do t
			vcat(warmstart_x[player][t], warmstart_u[player][t])
		end
		append!(warmstart_solution, warmstart_primals)
	end
	warmstart_solution
end

function safe_log10_history(history)
	map(history) do value
		isfinite(value) && value > 0 ? log10(value) : NaN
	end
end

function save_convergence_diagnostics(solution_dict, convergence_plots_dir, instance_idx, ϵ₀; filename_suffix = "")
	kkt_error_history = get(solution_dict, "kkt_error_history", Float64[])
	if !isempty(kkt_error_history)
		convergence_fig, _ = plot_convergence_plot(;
			kkt_error_history = safe_log10_history(kkt_error_history),
			total_iters = solution_dict["total_iters"],
		)
		CairoMakie.save(
			joinpath(
				convergence_plots_dir,
				"convergence_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			convergence_fig,
		)
	end

	condition_number_history = get(solution_dict, "condition_number_history", Float64[])
	if !isempty(condition_number_history)
		condition_number_fig, _ = plot_condition_number_plot(;
			condition_number_history = safe_log10_history(condition_number_history),
			total_iters = solution_dict["total_iters"],
		)
		CairoMakie.save(
			joinpath(
				convergence_plots_dir,
				"condition_number_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			condition_number_fig,
		)
	end

	eta_history = get(solution_dict, "eta_history", Float64[])
	if !isempty(eta_history)
		eta_fig, _ = plot_eta_plot(;
			eta_history = safe_log10_history(eta_history),
			total_iters = solution_dict["total_iters"],
		)
		CairoMakie.save(
			joinpath(
				convergence_plots_dir,
				"eta_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			eta_fig,
		)
	end
end


end
