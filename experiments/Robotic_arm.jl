module Robotic_arm

# Forward BLAS/LAPACK to Apple Accelerate (AMX): the solver's per-iteration
# cost is dominated by a dense SVD, which runs ~2.7x faster under Accelerate
# than OpenBLAS on Apple silicon. Process-global via libblastrampoline.
@static if Sys.isapple()
	using AppleAccelerate: AppleAccelerate
end

using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, Random
using ReducedGOOP
using TimerOutputs: @timeit, reset_timer!

const TO = ReducedGOOP.TO

abstract type DynamicsModel end
struct SingleIntegrator3D <: DynamicsModel end

include(joinpath(@__DIR__, "plotting.jl"))
include(joinpath(@__DIR__, "3d_plotting.jl"))
include(joinpath(@__DIR__, "dynamics.jl"))

# ── Configuration ────────────────────────────────────────────────────────────────────────

"""Stable numerical and geometric configuration for the robotic-arm game."""
Base.@kwdef struct ScenarioConfig{DM,D,CB}
	dynamics_model::DM
	dynamics::D
	control_bounds::CB
	num_players::Int = 2
	planning_horizon::Int = 12
	position_dimension::Int = 3
	map_end::Float64 = 10.0
	lane_width::Float64 = 2.0
	Δt::Float64 = 0.1
	dₚ::Float64 = 2.0
	collision_avoidance::Float64 = 2.0
	safety_buffer_margin::Float64 = 1.0
	child_initial_buffer::Float64 = 4.0
	arm_speed_limit::Float64 = 5.0
	child_speed_limit::Float64 = 3.0
	base_initial_state1::Vector{Float64} = [-1.0, 6.0, 0.0]
	base_initial_state2::Vector{Float64} = [1.0, 6.0, 0.0]
	goal_position1::Vector{Float64} = [-1.0, -5.0, 5.0]
	goal_position2::Vector{Float64} = [1.0, -5.0, 5.0]
	initial_state3::Vector{Float64} = [-6.0, -1.0, 0.0]
	goal_position3::Vector{Float64} = [0.0, 0.0, 0.0]
	obstacle_position::Vector{Float64} = [0.25, 0.15, 0.0]
	use_scalarized_baseline::Bool = false
	use_social_equilibrium_baseline::Bool = false
end

"""Output locations and display/file-format choices for experiment plots."""
Base.@kwdef struct VisualizationConfig{D}
	dirs::D
	show_interactive_trajectory::Bool = false
	static_extension::String = "pdf"
	interactive_extension::String = "html"
end

"""Per-attempt parameter blocks passed to solver and plotting routines."""
struct InstanceParameters{T1,T2,T3,T}
	θ1::T1
	θ2::T2
	θ3::T3
	θ::T
end

# ── Problem definition ─────────────────────────────────────────────────────────

function get_setup(scenario_config::ScenarioConfig)
	(;
		num_players,
		dynamics,
		control_bounds,
		planning_horizon,
		dₚ,
		collision_avoidance,
		arm_speed_limit,
		child_speed_limit,
		use_scalarized_baseline,
		use_social_equilibrium_baseline,
	) = scenario_config
	num_players == 2 || error("robotic_arm Zero-Sum GOOP setup expects exactly two players (two-arm robot and child).")
	length(dynamics) == num_players || error("Expected one dynamics entry per player.")

	# ── Shared dimensions and parameter layout ──────────────────────────────────────────────────────────

	position_dimension = 3
	# Player 1 is the combined two-arm agent: state/control stack both arms as
	# [arm1; arm2]. Player 2 is the child/pet on the ground.
	arm1_range = 1:position_dimension
	arm2_range = (position_dimension+1):(2*position_dimension)

	primal_dimensions = [
		(dyn.state_dimension + dyn.control_dimension) * planning_horizon for dyn in dynamics
	]
	# Per player: initial state, goal of the same length, and a 3D obstacle.
	parameter_dimensions = [
		2 * dyn.state_dimension + position_dimension for dyn in dynamics
	]

	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
	dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

	# ── Shared parameter and trajectory helpers ─────────────────────────────────────────────────────────────

	unflatten_parameters = function (θ; player)
		state_dimension = dynamics[player].state_dimension
		θ_iter = Iterators.Stateful(θ)
		initial_state = first(θ_iter, state_dimension)
		goal_position = first(θ_iter, state_dimension)
		obstacle_position = first(θ_iter, position_dimension)
		(; initial_state, goal_position, obstacle_position)
	end

	function flatten_parameters(;
		player,
		initial_state,
		goal_position,
		obstacle_position,
	)
		state_dimension = dynamics[player].state_dimension
		length(initial_state) == state_dimension || error("Expected initial_state length $(state_dimension), got $(length(initial_state)).")
		length(goal_position) == state_dimension || error("Expected goal_position length $(state_dimension), got $(length(goal_position)).")
		length(obstacle_position) == position_dimension || error("Expected 3D obstacle_position, got length $(length(obstacle_position)).")
		vcat(initial_state, goal_position, obstacle_position)
	end

	trajectory(z; player) = unflatten_trajectory(
		z[Block(player)],
		dynamics[player].state_dimension,
		dynamics[player].control_dimension,
	)

	# ── Preference objectives ──────────────────────────────────────────────────────────────────

	function goal_objective(z, θ)
		# The two-arm agent delivers the pot center to the center of the two
		# per-arm goals; both quantities are recovered from the stacked state.
		(; xs) = trajectory(z; player = 1)
		(; goal_position) = unflatten_parameters(θ[Block(1)]; player = 1)

		center_position = 0.5 .* (xs[end][arm1_range] .+ xs[end][arm2_range])
		center_goal_position = 0.5 .* (goal_position[arm1_range] .+ goal_position[arm2_range])
		goal_deviation = center_position .- center_goal_position
		terminal_objective = sum(goal_deviation .^ 2)

		terminal_objective
	end

	function control_objective(; player)
		function (z, _)
			(; us) = trajectory(z; player)
			sum(sum(u .^ 2) for u in us)
		end
	end

	function load_balance_objective(z, θ; allowance = 0.1)
		(; xs, us) = trajectory(z; player = 1)
		sum(eachindex(xs)) do t
			balance = (xs[t][arm1_range[end]] - xs[t][arm2_range[end]])^2
			swerve = us[t][arm1_range[1]]^2 + us[t][arm2_range[1]]^2
			# Squared excess keeps the deadzone but makes the penalty C¹ at the
			# boundary; the raw excess has a gradient jump there, which pins the
			# equilibrium on a non-stationary kink and blocks KKT convergence.
			ifelse(balance < allowance^2, 0.0, balance - allowance^2)
		end
	end

	function negative_load_balance_objective(z, θ)
		-load_balance_objective(z, θ)
	end

	function pot_approach_objective(z, θ)
		arm_trajectory = trajectory(z; player = 1)
		child_trajectory = trajectory(z; player = 2)

		# The child/pet is curious, not physically coupled to the pot.
		# Approach the pot center c = (p₁ + p₂) / 2.
		sum(eachindex(child_trajectory.xs)) do t
			arm_state = arm_trajectory.xs[t]
			pot_center = 0.5 .* (arm_state[arm1_range] .+ arm_state[arm2_range])
			sum(abs2, child_trajectory.xs[t] .- pot_center)
		end
	end

	function negative_pot_approach_objective(z, θ)
		-pot_approach_objective(z, θ)
	end

	# ── Control-bound constraints ──────────────────────────────────────────────────────────
	function control_bound_inequality(; player)
		function (z, _)
			# The combined two-arm agent tiles the single-arm bounds over both arms.
			lb = player == 1 ? vcat(control_bounds.lb, control_bounds.lb) : control_bounds.lb
			ub = player == 1 ? vcat(control_bounds.ub, control_bounds.ub) : control_bounds.ub
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) = trajectory(z; player)
			mapreduce(vcat, us) do u
				vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
			end
		end
	end

	# ── Equality and inequality constraints ──────────────────────────────────────────────────────────
	player_equality_constraints = [
		function (z, θ)
			(; xs, us) = trajectory(z; player = i)
			(; initial_state) = unflatten_parameters(θ[Block(i)]; player = i)

			initial_state_constraint = xs[1] - initial_state
			dynamics_constraints = mapreduce(vcat, 1:(length(xs)-1)) do t
				# xs[t] - dynamics(xs[t-1], us[t-1], t)
				dynamics[i].residual(z[Block(i)], t)
			end
			empty_constraint = similar(initial_state_constraint, 0)

			# The two arms jointly carry the hot pot. This hard handle-grasp
			# constraint keeps the two grippers separated by the fixed 3D handle
			# distance dₚ at all times.
			handle_grasp_constraint = i == 1 ?
				mapreduce(vcat, eachindex(xs)) do t
					gripper_separation = xs[t][arm1_range] .- xs[t][arm2_range]
					[sum(abs2, gripper_separation) - dₚ^2]
				end :
				empty_constraint

			# The child is modeled with the same 3D single-integrator dynamics, but its
			# vertical velocity is fixed to zero.
			child_ground_constraint = i == 2 ?
				mapreduce(u -> [u[position_dimension]], vcat, us) :
				empty_constraint

			vcat(
				initial_state_constraint,
				dynamics_constraints,
				handle_grasp_constraint,
				child_ground_constraint,
			)
		end for i in 1:num_players
	]

	equality_constraints = player_equality_constraints

	function robot_child_safety_inequality(z, θ)
		arm_trajectory = trajectory(z; player = 1)
		child_trajectory = trajectory(z; player = 2)

		# The child/pet is not constrained by this residual. Instead, the two-arm
		# player must keep the pot center c = (p₁ + p₂) / 2 outside the
		# child-centered safety sphere. This is the proof-of-concept Zero-Sum
		# GOOP coupling: the child's curious motion restricts the feasible robot
		# motions indirectly through safety.
		mapreduce(vcat, eachindex(child_trajectory.xs)) do t
			arm_state = arm_trajectory.xs[t]
			pot_center = 0.5 .* (arm_state[arm1_range] .+ arm_state[arm2_range])
			separation = pot_center .- child_trajectory.xs[t]
			[sum(abs2, separation) - collision_avoidance^2]
		end
	end

	function robot_arm_speed_inequality(z, θ)
		(; us) = trajectory(z; player = 1)

		# Each arm individually respects the same speed limit as before.
		mapreduce(vcat, us) do u
			[
				arm_speed_limit^2 - sum(abs2, u[arm1_range]),
				arm_speed_limit^2 - sum(abs2, u[arm2_range]),
			]
		end
	end

	function robot_inequality(z, θ)
		vcat(
			robot_child_safety_inequality(z, θ),
			robot_arm_speed_inequality(z, θ),
		)
	end

	function child_ground_speed_inequality(z, θ)
		(; us) = trajectory(z; player = 2)

		# This is a physical child-motion bound, not a robot safety constraint. The
		# safety constraints remain robot-only; Player 3 is merely limited to a
		# plausible ground speed so it cannot teleport into the pot center.
		mapreduce(vcat, us) do u
			[(child_speed_limit)^2 - sum(abs2, u[1:2])]
		end
	end

	inequality_constraints = [
		robot_inequality,
		child_ground_speed_inequality,
	]

	# ── 1. GOOP formulation ──────────────────────────────────────────────────────────────────────────────
	# Preference hierarchy: [lowest priority, ..., highest priority].
	goop_preferences = [
		[
			# control_objective(; player = 1),
			goal_objective,
			load_balance_objective,
			robot_inequality,
		],
		[
			pot_approach_objective,
			child_ground_speed_inequality
		],
	]
	goop_is_prioritized_constraint = [[false, false, true], [false, true]]

	function build_goop_problem()
		@timeit TO "ParametricGOOP construction" ReducedGOOP.ParametricGOOP(
				dummy_primals,
				dummy_parameters;
				preferences = goop_preferences,
				is_prioritized_constraint = goop_is_prioritized_constraint,
				equality_constraints,
				inequality_constraints = [nothing, nothing],
				shared_equality_constraint = nothing,
				shared_inequality_constraint = nothing,
			)
	end

	# ── 2. Scalarized baseline formulation ─────────────────────────────────────────────────────────────
	# Flatten each player's hierarchy into one smooth objective while retaining
	# the same per-player equalities and ParametricGOOP solver representation.
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

	scalarized_player_preferences =
		map(
			scalarized_player_preference,
			goop_preferences,
			goop_is_prioritized_constraint,
		)
	scalarized_preferences = [[preference] for preference in scalarized_player_preferences]
	scalarized_is_prioritized_constraint = [[false] for _ in scalarized_preferences]

	function build_scalarized_problem()
		@timeit TO "ParametricGOOP construction" ReducedGOOP.ParametricGOOP(
				dummy_primals,
				dummy_parameters;
				preferences = scalarized_preferences,
				is_prioritized_constraint = scalarized_is_prioritized_constraint,
				equality_constraints,
				inequality_constraints = [nothing, nothing],
				shared_equality_constraint = nothing,
				shared_inequality_constraint = nothing,
			)
	end

	# ── 3. Social equilibrium formulation ────────────────────────────────────────────────────────────────
	# Sum the scalarized player objectives and concatenate all player constraints
	# into a single ParametricOptimizationProblem.
	social_objective = function (z, θ)
		accumulated_objective = 0
		for preference in scalarized_player_preferences
			accumulated_objective += preference(z, θ)
		end
		accumulated_objective
	end
	social_equality_constraint = function (z, θ)
		mapreduce(f -> f(z, θ), vcat, player_equality_constraints)
	end
	social_inequality_constraint = function (z, θ)
		vcat(
			inequality_constraints[1](z, θ),
			inequality_constraints[2](z, θ),
		)
	end
	social_primal_dimension = sum(primal_dimensions)
	social_parameter_dimension = sum(parameter_dimensions)
	social_equality_dimension =
		length(social_equality_constraint(dummy_primals, dummy_parameters))
	social_inequality_dimension =
		length(social_inequality_constraint(dummy_primals, dummy_parameters))

	# TODO: Pass the unequal per-player primal and parameter block dimensions;
	# ParametricOptimizationProblem currently splits both totals evenly.
	function build_social_problem()
		@timeit TO "ParametricOptimizationProblem construction" ReducedGOOP.ParametricOptimizationProblem(;
				objective = social_objective,
				equality_constraint = social_equality_constraint,
				inequality_constraint = social_inequality_constraint,
				parameter_dimension = social_parameter_dimension,
				primal_dimension = social_primal_dimension,
				equality_dimension = social_equality_dimension,
				inequality_dimension = social_inequality_dimension,
				num_players,
			)
	end

	# Social equilibrium retains precedence when both baseline flags are enabled.
	problem = @timeit TO "preference and constraint construction" if use_social_equilibrium_baseline
		build_social_problem()
	elseif use_scalarized_baseline
		build_scalarized_problem()
	else
		build_goop_problem()
	end

	(; problem, flatten_parameters)
end

# ── Experiment entry point ─────────────────────────────────────────────────────

function demo(;
	map_end = 10,
	lane_width = 2,
	verbose = false,
	rng_seed = 123,
	random_initial_state = false,
	debug = false,
	use_scalarized_baseline = false,
	use_social_equilibrium_baseline = false,
	show_interactive_trajectory = false,
)
	reset_timer!(TO)
	@timeit TO "experiment setup" Random.seed!(rng_seed)

	# ── Settings ───────────────────────────────────────────────────────────────
	run_id = "Robotic_arm_single_robot_agent_trial1"
	dynamics_model = Robotic_arm.SingleIntegrator3D()
	goop_version = :reduced      # :complete | :reduced | :quasi
	linesearch = :backtracking   # :backtracking | :fraction_to_boundary
	compute_warmstart = true # Whether to compute a warmstart trajectory via rollout (true) or load from file (false)

	# ── Problem parameters ─────────────────────────────────────────────────────
	# Player 1: combined two-arm agent, Player 2: child/pet.
	num_players          = 2
	planning_horizon     = 12
	collision_avoidance  = 2.0
	child_initial_buffer = 4.0
	arm_speed_limit		 = 5.0
	child_speed_limit    = 3.0
	num_instances        = 1
	perturbation_scale   = 0.3
	dₚ                   = 2.0


	# ── Solver schedule ────────────────────────────────────────────────────────
	epsilon_schedule         = [0.1]
	max_inner_iters_schedule = fill(500, length(epsilon_schedule))

	# ── Scenario ───────────────────────────────────────────────────────────────
	# Single Integrator 3D: state = [px, py, pz]
	base_initial_state1 = [-1.0,  6.0, 1.5]
	base_initial_state2 = [ 1.0,  6.0, 1.5]
	goal_position1      = [-1.0, -5.0, 5.0]
	goal_position2      = [ 1.0, -5.0, 5.0]
	initial_state3      = [-3.0, -1.0, 0.0]
	goal_position3	    = [ 0.0,  0.0, 0.0]
	obstacle_position   = [0.25, 0.15, 0.0]   # placeholder

	# ── Build dynamics and problem ─────────────────────────────────────────────
	state_dimension   = 3
	control_dimension = 3
	Δt                = 0.1
	control_bounds    = (; lb = [-10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0])

	# Per-player dynamics: the combined two-arm agent stacks both arms into one
	# 6D single integrator; the child keeps the plain 3D single integrator.
	dynamics = @timeit TO "dynamics construction" [
		build_two_arm_dynamics(dynamics_model; Δt),
		build_dynamics(dynamics_model; Δt, state_dimension, control_dimension),
	]
	scenario_config = ScenarioConfig(;
		dynamics_model,
		dynamics,
		control_bounds,
		num_players,
		planning_horizon,
		position_dimension = state_dimension,
		map_end = Float64(map_end),
		lane_width = Float64(lane_width),
		Δt,
		dₚ,
		collision_avoidance,
		child_initial_buffer,
		arm_speed_limit,
		child_speed_limit,
		base_initial_state1,
		base_initial_state2,
		goal_position1,
		goal_position2,
		initial_state3,
		goal_position3,
		obstacle_position,
		use_scalarized_baseline,
		use_social_equilibrium_baseline,
	)

	(; problem, flatten_parameters) = @timeit TO "problem setup" get_setup(scenario_config)
	# This example currently supports only the interior-point solver.
	solver = ReducedGOOP.InteriorPoint()
	kkt_generators = Dict(
		:complete => ReducedGOOP.generate_slacked_complete_kkt_system,
		:reduced  => ReducedGOOP.generate_slacked_reduced_kkt_system,
		:quasi    => ReducedGOOP.generate_slacked_quasi_kkt_system,
	)

	GOOP_kkt_generator = get(kkt_generators, goop_version, nothing)
	isnothing(GOOP_kkt_generator) && error("Unknown GOOP version: $(goop_version)")

	@info "Building KKT system for $(goop_version) GOOP formulation and $(solver) solver..."
	# Check if problem is not an instance of GOOPKKTSystem. Otherwise, build GOOPKKTSystem.
	GOOP_kkt_system = @timeit TO "KKT construction" begin
		if problem isa ReducedGOOP.GOOPKKTSystem
			problem
		else
			GOOP_kkt_generator(problem)
		end
	end

	println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
	println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)

	primal_dimensions = [
		(dyn.state_dimension + dyn.control_dimension) * planning_horizon for dyn in dynamics
	]

	# ── Per-instance solver ────────────────────────────────────────────────────
	function solve_game_instance(θ; z₀, ϵ₀, max_inner_iters)
		options = @timeit TO "solver options construction" ReducedGOOP.InteriorPointOptions(;
				tol = 0.01, #1e-4
				η₀ = 1e-6, # 0.0 to turn off Tikhonov regularization
				η_max = 1e6,
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
				eta_retry_growth = 2.0,
				ρ_low = 0.75,
				ρ_high = 0.75,
				perturbation_enabled = false,
				stagnation_rtol = 1e-1,
				perturbation_scale = 1e-6,
				tsvd_threshold = 0.0, # 0.0: pure Tikhonov, > 0 and η = 0: pure TSVD
				use_marquardt_scaling = false,
				verbose,
			)

		@info "Solving game instance with $(solver)..."
		kkt_error_history = Float64[]
		condition_number_history = Float64[]
		eta_history = Float64[]
		alpha_history = Float64[]
		rho_history = Float64[]
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
			alpha_history = hasproperty(output, :alpha_history) ? output.alpha_history : Float64[]
			rho_history = hasproperty(output, :rho_history) ? output.rho_history : Float64[]
			if status == :failed
				println("  [solver exit] total_iters=$(total_iters), kkt_error=$(round(kkt_error; sigdigits=4)), tol=$(options.tol)")
			end
			solver_status = status
		end

		strategies = @timeit TO "solution postprocessing" extract_player_strategies(
				x,
				primal_dimensions,
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
			"alpha_history" => alpha_history,
			"rho_history" => rho_history,
			"arm_speed_limit" => arm_speed_limit,
			"child_speed_limit" => child_speed_limit,
		)

		(; strategies, solution_dict)
	end

	# ── Output directories ─────────────────────────────────────────────────────
	output_dirs = @timeit TO "output directory setup" prepare_robotic_arm_output_dirs(run_id; debug)
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
	) = output_dirs
	visualization_config = VisualizationConfig(;
		dirs = output_dirs,
		show_interactive_trajectory,
	)

	# ── Main solve loop ────────────────────────────────────────────────────────
	instance_problem_data = Dict{String, Any}[]
	solved_attempts       = 0
	total_attempts        = 0

	while solved_attempts < num_instances
		total_attempts += 1
		initial_state1, initial_state2 = if random_initial_state
			(
				sample_initial_state(
					dynamics_model,
					base_initial_state1,
					state_dimension,
					perturbation_scale,
				),
				sample_initial_state(
					dynamics_model,
					base_initial_state2,
					state_dimension,
					perturbation_scale,
				),
			)
		else
			(copy(base_initial_state1), copy(base_initial_state2))
		end
		println(
			"solved $(solved_attempts)/$(num_instances), attempt $(total_attempts), goop version $(goop_version): ",
		)
		println("initial_state1:", initial_state1)
		println("goal_position1:", goal_position1)
		println("initial_state2:", initial_state2)
		println("goal_position2:", goal_position2)
		println("initial_state3:", initial_state3)
		println("goal_position3:", goal_position3)

		instance_states = (; initial_state1, initial_state2, initial_state3)
		instance_parameters = @timeit TO "instance parameter construction" build_instance_parameters(
			flatten_parameters,
			instance_states,
			scenario_config,
		)
		(; θ1, θ2, θ3, θ) = instance_parameters

		(; warmstart_solution) = @timeit TO "warmstart construction" if compute_warmstart
			build_default_warmstart(
				instance_states,
				scenario_config,
			)
		else
			(;
				warmstart_solution = load("experiments/solution_dict_instance_1_eps0.1.jld2")["single_stored_object"]["x"][1:sum(primal_dimensions)],
			)
		end
		@timeit TO "warmstart visualization" save_warmstart_visualizations(
			warmstart_solution;
			total_attempts,
			instance_idx = solved_attempts + 1,
			primal_dimensions,
			instance_parameters,
			scenario_config,
			visualization_config,
		)

		epsilon_results = Pair{Float64, Any}[]
		stage_warmstart = warmstart_solution
		solve_sequence_succeeded = true
		instance_total_solve_time_sec = 0.0

		for (ϵ₀, max_inner_iters) in zip(epsilon_schedule, max_inner_iters_schedule)
			result = try
				solve_game_instance(θ; z₀ = stage_warmstart, ϵ₀, max_inner_iters)
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
			if result.solution_dict["status"] == :failed
				println(
					"attempt $(total_attempts): failed to converge for ϵ₀ = $(ϵ₀), saving diagnostics.",
				)
				solve_sequence_succeeded = false
				break
			end
			stage_warmstart = warmstart_solution
		end

		if !solve_sequence_succeeded
			failed_instance_idx = solved_attempts + 1
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
				"arm_speed_limit" => arm_speed_limit,
				"child_speed_limit" => child_speed_limit,
				"total_solve_time_sec" => instance_total_solve_time_sec,
			),
		)

		solved_attempts += 1
		println(
			"instance $(solved_attempts) total solve time: $(round(instance_total_solve_time_sec; digits = 5)) sec",
		)
		println("instance $(solved_attempts) converged preference values by ϵ:")

		@timeit TO "problem data save" begin
			JLD2.save_object(
				joinpath(problem_data_dir, "problem_data_instance_$(solved_attempts).jld2"),
				instance_problem_data,
			)
		end

		for (ϵ₀, result) in epsilon_results
			solution_dict = result.solution_dict

			@timeit TO "solution output and plotting" begin
				JLD2.save_object(
					joinpath(
						solution_data_dir,
						"solution_dict_instance_$(solved_attempts)_eps$(ϵ₀).jld2",
					),
					solution_dict,
				)
				save_convergence_diagnostics(solution_dict, convergence_plots_dir, solved_attempts, ϵ₀)
				save_solution_visualizations(
					result.strategies,
					solved_attempts,
					ϵ₀;
					instance_parameters,
					scenario_config,
					visualization_config,
				)
			end
		end
	end

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
				"arm_speed_limit" => arm_speed_limit,
				"child_speed_limit" => child_speed_limit,
				"collision_avoidance" => collision_avoidance,
				"child_initial_buffer" => child_initial_buffer,
				"dₚ" => dₚ,
				"perturbation_scale" => perturbation_scale,
			),
		)
	end

	println("\nTiming summary:")
	show(TO)
	println()
end

# ── Output / plotting helpers ──────────────────────────────────────────────────

function save_warmstart_visualizations(
	warmstart_solution;
	total_attempts,
	instance_idx,
	primal_dimensions,
	instance_parameters::InstanceParameters,
	scenario_config::ScenarioConfig,
	visualization_config::VisualizationConfig,
)
	(; θ1, θ2, θ3) = instance_parameters
	(;
		dynamics,
		map_end,
		lane_width,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
		dₚ,
		arm_speed_limit,
		child_speed_limit,
		control_bounds,
	) = scenario_config
	(; warmstart_plots_dir) = visualization_config.dirs
	static_extension = visualization_config.static_extension
	# Plots keep the per-arm view: split the combined two-arm strategy back
	# into [arm1, arm2, child].
	warmstart_strategies = split_arm_strategies(
		extract_player_strategies(
			warmstart_solution,
			primal_dimensions,
			dynamics,
		),
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
		speed_limit = arm_speed_limit,
		dynamics_model = dynamics[1].model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
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
		safety_distance = collision_avoidance,
	)

	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_attempt_$(total_attempts)_instance_$(instance_idx).$(static_extension)",
		),
		warmstart_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_speed_attempt_$(total_attempts)_instance_$(instance_idx).$(static_extension)",
		),
		warmstart_speed_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_control_attempt_$(total_attempts)_instance_$(instance_idx).$(static_extension)",
		),
		warmstart_control_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_distance_attempt_$(total_attempts)_instance_$(instance_idx).$(static_extension)",
		),
		warmstart_distance_fig,
	)
end

"""Save the standard static plots and optional interactive trajectory for one solution."""
function save_solution_visualizations(
	strategies,
	instance_idx,
	ϵ₀;
	instance_parameters::InstanceParameters,
	scenario_config::ScenarioConfig,
	visualization_config::VisualizationConfig,
)
	(; θ1, θ2, θ3) = instance_parameters
	(;
		dynamics_model,
		control_bounds,
		map_end,
		lane_width,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
		dₚ,
		arm_speed_limit,
		child_speed_limit,
	) = scenario_config
	(;
		trajectory_plots_dir,
		speed_plots_dir,
		control_plots_dir,
		distance_plots_dir,
	) = visualization_config.dirs
	static_extension = visualization_config.static_extension
	interactive_extension = visualization_config.interactive_extension

	# Plots keep the per-arm view of the combined two-arm agent.
	plot_strategies = split_arm_strategies(strategies)
	trajectory_fig, _ = plot_single_integrator_3d_trajectories(;
		map_end,
		lane_width,
		strategy = plot_strategies,
		θ1,
		θ2,
		θ3,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
	)
	speed_fig, _ = speed_plot(;
		strategy = plot_strategies,
		speed_limit = arm_speed_limit,
		dynamics_model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
	)
	control_fig, _ = control_plot(;
		strategy = plot_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	distance_fig, _ = inter_player_distance_plot(;
		strategy = plot_strategies,
		reference_distance = dₚ,
		safety_distance = collision_avoidance,
	)

	plot_specs = (
		(trajectory_plots_dir, "trajectory", trajectory_fig),
		(speed_plots_dir, "speed", speed_fig),
		(control_plots_dir, "control", control_fig),
		(distance_plots_dir, "distance", distance_fig),
	)
	for (output_dir, plot_name, figure) in plot_specs
		save_figure(
			joinpath(output_dir, "$(plot_name)_instance_$(instance_idx)_eps$(ϵ₀).$(static_extension)"),
			figure,
		)
	end

	if visualization_config.show_interactive_trajectory
		interactive_trajectory_path = joinpath(
			trajectory_plots_dir,
			"trajectory_interactive_instance_$(instance_idx)_eps$(ϵ₀).$(interactive_extension)",
		)
		plot_trajectory_3d_interactive(;
			map_end,
			lane_width,
			strategy = plot_strategies,
			θ1,
			θ2,
			θ3,
			goal_position1,
			goal_position2,
			goal_position3,
			collision_avoidance,
			reference_distance = dₚ,
			display_figure = false,
			save_path = interactive_trajectory_path,
		)
		println("saved interactive trajectory browser file: ", interactive_trajectory_path)
	end
end

function prepare_robotic_arm_output_dirs(run_id; debug)
	run_dir = if debug
		joinpath("data", "robotic_arm_open_loop", "debug", run_id)
	else
		joinpath("data", "robotic_arm_open_loop", "runs", run_id)
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

"""
Two 3D single-integrator arms stacked into one joint agent with state
[p₁; p₂] ∈ R⁶ and control [u₁; u₂] ∈ R⁶. The single-integrator residual and
step are dimension-generic, so the stacked agent reuses them directly.
"""
function build_two_arm_dynamics(model::SingleIntegrator3D; Δt = 0.5)
	state_dimension = 6
	control_dimension = 6

	residual(z, t) = single_integrator_3d(z, t; Δt, state_dimension, control_dimension)
	step(x, u, t) = single_integrator_3d_step(x, u; Δt)
	(; model, residual, step, Δt, state_dimension, control_dimension)
end

function dynamics_model_name(::SingleIntegrator3D)
	"single_integrator_3d"
end


function extract_player_strategies(
	primal_solution,
	primal_dimensions,
	dynamics,
)
	offsets = cumsum(vcat(0, primal_dimensions[1:(end-1)]))
	map(eachindex(primal_dimensions)) do player_idx
		start_idx = offsets[player_idx] + 1
		end_idx   = offsets[player_idx] + primal_dimensions[player_idx]
		unflatten_trajectory(
			primal_solution[start_idx:end_idx],
			dynamics[player_idx].state_dimension,
			dynamics[player_idx].control_dimension,
		)
	end
end

"""
Split the combined two-arm strategy (player 1) into per-arm 3D strategies so
the existing per-arm plotting code can be reused unchanged. Returns
[arm1, arm2, child, ...] in the layout the plots expect.
"""
function split_arm_strategies(strategies; position_dimension = 3)
	combined = strategies[1]
	arm1 = (;
		xs = [x[1:position_dimension] for x in combined.xs],
		us = [u[1:position_dimension] for u in combined.us],
	)
	arm2 = (;
		xs = [x[(position_dimension+1):(2*position_dimension)] for x in combined.xs],
		us = [u[(position_dimension+1):(2*position_dimension)] for u in combined.us],
	)
	vcat([arm1, arm2], strategies[2:end])
end

function build_instance_parameters(
	flatten_parameters,
	instance_states,
	scenario_config::ScenarioConfig,
)
	(; initial_state1, initial_state2, initial_state3) = instance_states
	(;
		goal_position1,
		goal_position2,
		goal_position3,
		obstacle_position,
	) = scenario_config
	# Solver-facing parameter blocks: player 1 stacks both arms, player 2 is the child.
	θ_arms = flatten_parameters(;
		player = 1,
		initial_state = vcat(initial_state1, initial_state2),
		goal_position = vcat(goal_position1, goal_position2),
		obstacle_position = obstacle_position,
	)
	θ_child = flatten_parameters(;
		player = 2,
		initial_state = initial_state3,
		goal_position = goal_position3,
		obstacle_position = obstacle_position,
	)

	# Per-arm 3D blocks kept in the legacy layout purely for plotting, which
	# reads the initial position from the first three entries.
	θ1 = vcat(initial_state1, goal_position1, obstacle_position)
	θ2 = vcat(initial_state2, goal_position2, obstacle_position)
	θ3 = copy(θ_child)

	InstanceParameters(θ1, θ2, θ3, [θ_arms..., θ_child...])
end

function limit_control_speed(control, speed_limit, speed_indices)
	speed_limit >= 0 || error("Speed limit must be nonnegative.")
	bounded_control = collect(control)
	control_speed = sqrt(sum(abs2, bounded_control[speed_indices]))
	if control_speed > speed_limit && control_speed > 0
		bounded_control[speed_indices] .= bounded_control[speed_indices] .* (speed_limit / control_speed)
	end
	bounded_control
end

function build_default_warmstart(
	instance_states,
	scenario_config::ScenarioConfig,
)
	(; initial_state1, initial_state2, initial_state3) = instance_states
	(;
		planning_horizon,
		dynamics,
		goal_position1,
		goal_position2,
		goal_position3,
		arm_speed_limit,
		child_speed_limit,
	) = scenario_config
	arm_dynamics, child_dynamics = dynamics
	planning_horizon >= 2 || error("Robot warm start requires at least two time steps.")
	length(goal_position1) == child_dynamics.state_dimension || error("Goal position 1 dimension mismatch.")
	length(goal_position2) == child_dynamics.state_dimension || error("Goal position 2 dimension mismatch.")

	total_time = arm_dynamics.Δt * (planning_horizon - 1)
	start_center = 0.5 .* (initial_state1 .+ initial_state2)
	goal_center = 0.5 .* (goal_position1 .+ goal_position2)
	# Same 3D constant control for both arms (as before), stacked into the
	# combined agent's 6D control.
	robot_warmstart_control = limit_control_speed(
		(goal_center .- start_center) ./ total_time,
		arm_speed_limit,
		1:child_dynamics.control_dimension,
	)
	arms_warmstart = build_constant_control_warmstart(
		planning_horizon,
		arm_dynamics,
		vcat(initial_state1, initial_state2),
		vcat(robot_warmstart_control, robot_warmstart_control),
	)

	child_warmstart = build_ground_line_warmstart(
		planning_horizon,
		child_dynamics,
		initial_state3,
		goal_position3;
		ground_speed_limit = child_speed_limit,
	)

	warmstart_solution = flatten_warmstart_solution(
		planning_horizon,
		[arms_warmstart.xs, child_warmstart.xs],
		[arms_warmstart.us, child_warmstart.us],
	)
	(; warmstart_solution)
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

function build_ground_line_warmstart(
	planning_horizon,
	dynamics,
	initial_state,
	goal_position;
	ground_speed_limit = Inf,
)
	length(initial_state) == dynamics.state_dimension || error("Initial state dimension mismatch.")
	length(goal_position) == dynamics.state_dimension || error("Goal position dimension mismatch.")
	planning_horizon >= 2 || error("Ground-line warm start requires at least two time steps.")

	total_time = dynamics.Δt * (planning_horizon - 1)
	constant_control = (goal_position .- initial_state) ./ total_time
	constant_control[3] = 0.0
	constant_control = limit_control_speed(constant_control, ground_speed_limit, 1:2)

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
		save_figure(
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
		save_figure(
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
		save_figure(
			joinpath(
				convergence_plots_dir,
				"eta_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			eta_fig,
		)
	end

	alpha_history = get(solution_dict, "alpha_history", Float64[])
	if !isempty(alpha_history)
		alpha_fig, _ = plot_alpha_plot(;
			alpha_history,
			total_iters = solution_dict["total_iters"],
		)
		save_figure(
			joinpath(
				convergence_plots_dir,
				"alpha_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			alpha_fig,
		)
	end

	rho_history = get(solution_dict, "rho_history", Float64[])
	if !isempty(rho_history)
		rho_fig, _ = plot_rho_plot(;
			rho_history,
			total_iters = solution_dict["total_iters"],
		)
		save_figure(
			joinpath(
				convergence_plots_dir,
				"rho_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			rho_fig,
		)
	end
end


end
