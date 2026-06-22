module Intersection

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
	unflatten_trajectory, state_dim, control_dim, control_bounds
using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, Distributions, Random
using Symbolics, NonlinearSolve, LinearAlgebra
using Printf, Statistics
using ParametricMCPs
using ReducedGOOP

include(joinpath(@__DIR__, "Plotting.jl"))

# ── Problem definition ─────────────────────────────────────────────────────────

function get_setup(
	num_players;
	dynamics = UnicycleDynamics,
	dynamics_model = :unicycle,
	planning_horizon = 5,
	collision_avoidance = 1.0,
	velocity_limit = 1.5,
	map_end = 7,
	lane_width = 2,
)
	state_dimension = state_dim(dynamics)
	control_dimension = control_dim(dynamics)
	primals_per_player = (state_dimension + control_dimension) * planning_horizon
	primal_dimensions = fill(primals_per_player, num_players)
	parameter_dimensions = fill(state_dimension + 4, num_players) # (state, goal, obstacle)

	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
	dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

	unflatten_parameters = function (θ)
		θ_iter = Iterators.Stateful(θ)
		initial_state = first(θ_iter, state_dimension)
		goal_position = first(θ_iter, 2)
		obstacle_position = first(θ_iter, 2)
		(; initial_state, goal_position, obstacle_position)
	end

	function flatten_parameters(; initial_state, goal_position, obstacle_position)
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

	control_objectives = [
		function (z, _)
			(; xs, us) =
				unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
			sum(sum(u .^ 2) for u in us)
		end,
		function (z, _)
			(; xs, us) =
				unflatten_trajectory(z[Block(2)], state_dimension, control_dimension)
			sum(sum(u .^ 2) for u in us)
		end,
	]

	control_bound_inequality = function (z, i)
		(; lb, ub) = control_bounds(dynamics)
		lb_mask = findall(!isinf, lb)
		ub_mask = findall(!isinf, ub)
		(; xs, us) =
			unflatten_trajectory(z[Block(i)], state_dimension, control_dimension)
		mapreduce(vcat, us) do u
			vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
		end
	end

	function shared_collision_avoidance(z, _)
		trajectories = map(
			i ->
				unflatten_trajectory(z[Block(i)], state_dimension, control_dimension),
			1:num_players,
		)
		xs = map(trajectory -> trajectory.xs, trajectories)
		@assert length(xs) == num_players
		mapreduce(vcat, 2:length(xs[1])) do k
			[sum((xs[1][k][1:2] - xs[2][k][1:2]) .^ 2) - collision_avoidance^2]
		end
	end

	inequality_constraints = [
		function (z, θ)
			(; lb, ub) = control_bounds(dynamics)
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) =
				unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
			vcat(
				# mapreduce(vcat, us) do u
				# 	vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
				# end,
				mapreduce(vcat, 1:length(xs)) do k
					px = xs[k][1]
					py = xs[k][2]
					vcat(
						px + map_end,
						-px + map_end,
						py + lane_width,
						-py + lane_width,
					) # -7 ≤ pₓ ≤ 7, -2 ≤ py ≤ 2
				end,
				shared_collision_avoidance(z, θ),
			)
		end, function (z, θ)
			(; lb, ub) = control_bounds(dynamics)
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) =
				unflatten_trajectory(z[Block(2)], state_dimension, control_dimension)
			vcat(
				# mapreduce(vcat, us) do u
				# 	vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
				# end,
				mapreduce(vcat, 1:length(xs)) do k
					px = xs[k][1]
					py = xs[k][2]
					vcat(
						px + lane_width,
						-px + lane_width,
						py + map_end,
						-py + map_end,
					) # -2 ≤ pₓ ≤ 2, -7 ≤ py ≤ 7
				end,
				shared_collision_avoidance(z, θ),
			)
		end,
	]

	player_equality_constraints = [
		function (z, θ)
			(; lb, ub) = control_bounds(dynamics)
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) = unflatten_trajectory(z[Block(i)], state_dimension, control_dimension)
			(; initial_state) = unflatten_parameters(θ[Block(i)])

			initial_state_constraint = xs[1] - initial_state
			dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
				xs[k] - dynamics(xs[k-1], us[k-1], k)
			end

			vcat(
				initial_state_constraint,
				dynamics_constraints,
				# squared_violation.(
				# mapreduce(vcat, us) do u
				# 	vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
				# end,
				# ),
				# squared_violation.(
				# 	mapreduce(vcat, 1:length(xs)) do k
				# 		px = xs[k][1]
				# 		py = xs[k][2]
				# 		i == 1 ? vcat(
				# 			px + map_end,
				# 			-px + map_end,
				# 			py + lane_width,
				# 			-py + lane_width,
				# 		) : vcat(
				# 			px + lane_width,
				# 			-px + lane_width,
				# 			py + map_end,
				# 			-py + map_end,
				# 		)
				# 	end),
			)
		end for i in 1:num_players
	]

	shared_equality_constraint = function (z, θ)
		squared_violation.(shared_collision_avoidance(z, θ))
	end

	equality_constraints = [
		(z, θ) -> vcat(player_equality_constraints[i](z, θ), shared_equality_constraint(z, θ))
		for i in 1:num_players
	]

	preferences = [
		[
			# Minimize control effort 
			control_objectives[1],

			# Drive under speed limit
			# function (z, _)
			# 	(; xs) = unflatten_trajectory(
			# 		z[Block(1)],
			# 		state_dimension,
			# 		control_dimension,
			# 	)
			# 	mapreduce(vcat, 1:length(xs)) do k
			# 		velocity_limit_constraints(xs[k], dynamics_model; velocity_limit)
			# 	end
			# end,

			# Reach the goal (highest priority for P1)
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(1)],
					state_dimension,
					control_dimension,
				)
				(; goal_position) = unflatten_parameters(θ[Block(1)])
				goal_deviation = xs[end][1:2] .- goal_position
				sum(goal_deviation .^ 2) #+ sum(smooth_piecewise_preference_objective.(control_bound_inequality(z, 1), 1))
			end,

			# Lane bounds + collision avoidance (constraint, both players)
			# inequality_constraints[1],
		],
		[
			# Minimize control effort
			control_objectives[2],

			# Reach the goal
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(2)],
					state_dimension,
					control_dimension,
				)
				(; goal_position) = unflatten_parameters(θ[Block(2)])
				goal_deviation = xs[end][1:2] .- goal_position
				sum(goal_deviation .^ 2) #+ sum(smooth_piecewise_preference_objective.(control_bound_inequality(z, 2), 2))
			end,

			# Drive under speed limit (highest priority for P2)
			function (z, _)
				(; xs) = unflatten_trajectory(
					z[Block(2)],
					state_dimension,
					control_dimension,
				)
				mapreduce(vcat, 1:length(xs)) do k
					velocity_limit_constraints(xs[k], dynamics_model; velocity_limit)
				end
			end,

			# Lane bounds + collision avoidance (constraint, both players)
			# inequality_constraints[2],
		],
	]

	# Preference hierarchy: [lowest priority, ..., highest priority]
	is_prioritized_constraint = [[false, false], [false, false, true]]

	" Scalarized baseline (Nash / no hierarchy): flattens hierarchical preferences into a single objective per player"
	use_scalarized_baseline = false
	scalarized_preferences = map(
		preferences,
		is_prioritized_constraint,
	) do player_preferences, player_is_prioritized_constraint
		@assert length(player_preferences) == length(player_is_prioritized_constraint)

		scalarized_preference = function (z, θ)
			accumulated_preference = 0
			for (level, (preference, is_constraint)) in
				enumerate(zip(player_preferences, player_is_prioritized_constraint))
				preference_value = preference(z, θ)
				accumulated_preference += if is_constraint
					sum(smooth_piecewise_preference_objective.(preference_value, level))
				else
					preference_value
				end
			end
			accumulated_preference
		end
		[scalarized_preference]
	end
	scalarized_is_prioritized_constraint = [[false] for _ in scalarized_preferences]

	" Social equilibrium (no Nash / no hierarchy) baseline: sum of all players' preferences, single optimization problem"
	use_social_equilibrium_baseline = false
	objective = function (z, θ)
		accumulated_objective = 0
		for (player_preferences, player_is_prioritized_constraint) in
			zip(preferences, is_prioritized_constraint)
			@assert length(player_preferences) == length(player_is_prioritized_constraint)

			for (level, (preference, is_constraint)) in
				enumerate(zip(player_preferences, player_is_prioritized_constraint))
				preference_value = preference(z, θ)
				accumulated_objective += if is_constraint
					sum(smooth_piecewise_preference_objective.(preference_value, level))
				else
					preference_value
				end
			end
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
	primal_dimension = sum(primal_dimensions)
	parameter_dimension = sum(parameter_dimensions)
	equality_dimension = length(equality_constraint(dummy_primals, dummy_parameters))
	inequality_dimension = 0


	# Build problem
	problem = if !use_social_equilibrium_baseline
		ReducedGOOP.ParametricGOOP(
			dummy_primals,
			dummy_parameters;
			preferences = use_scalarized_baseline ? scalarized_preferences : preferences,
			is_prioritized_constraint = use_scalarized_baseline ? scalarized_is_prioritized_constraint : is_prioritized_constraint,
			equality_constraints,
			inequality_constraints = [nothing, nothing],
			shared_equality_constraint = nothing,
			shared_inequality_constraint = nothing,
		)
	else
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

	(; problem, flatten_parameters)
end

# ── Experiment entry point ─────────────────────────────────────────────────────

function demo(;
	map_end = 7,
	lane_width = 2,
	verbose = false,
	rng_seed = 123,
	random_initial_state = true,
	debug = false,
)
	Random.seed!(rng_seed)

	# ── Settings ───────────────────────────────────────────────────────────────
	run_id = "0_IP_test_Marquardt_scaling_w_adaptive_eta_and_tsvd"
	dynamics_model = :planar_double_integrator   # :planar_double_integrator | :unicycle
	goop_version = :reduced                    # :complete | :reduced | :quasi
	solver = ReducedGOOP.InteriorPoint() # ReducedGOOP.InteriorPoint() | ReducedGOOP.PATHSolver()
	linesearch = :backtracking          # :backtracking | :fraction_to_boundary
	compute_warmstart = true # Whether to compute a warmstart trajectory via rollout (true) or load from file (false)

	# ── Problem parameters ─────────────────────────────────────────────────────
	num_players           = 2
	planning_horizon      = 12
	collision_avoidance   = 1.5
	speed_component_limit = 2.0
	control_bounds        = (; lb = [-10.0, -10.0], ub = [10.0, 10.0])
	num_instances         = 1
	perturbation_scale    = 0.3

	# ── Solver schedule ────────────────────────────────────────────────────────
	epsilon_schedule         = [0.1]
	max_inner_iters_schedule = fill(3000, length(epsilon_schedule))

	# ── Scenario ───────────────────────────────────────────────────────────────
	# Planar double integrator: state = [px, py, vx, vy]
	base_initial_state1 = [-4.0, -1.0, 3.0, 0.0]
	base_initial_state2 = [1.0, -5.0, 0.0, 1.5]
	# Unicycle: state = [px, py, speed, heading] — uncomment to switch
	# base_initial_state1 = [-6.0, -1.0, 0.0, 0.0]
	# base_initial_state2 = [1.0, -6.0, 1.3, π/2]

	goal_position1    = [6.0, -1.0]
	goal_position2    = [1.0, 6.0]
	obstacle_position = [0.25, 0.15]   # placeholder

	# ── Build dynamics and problem ─────────────────────────────────────────────
	dynamics = build_intersection_dynamics(dynamics_model; dt = 0.2, control_bounds)

	(; problem, flatten_parameters) = get_setup(
		num_players;
		dynamics,
		dynamics_model,
		planning_horizon,
		collision_avoidance,
		velocity_limit = speed_component_limit,
		map_end,
		lane_width,
	)
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

	GOOP_kkt_system = get(kkt_generators, goop_version, nothing)
	isnothing(GOOP_kkt_system) && error("Unknown GOOP version: $(goop_version)")

	@info "Building KKT system for $(goop_version) GOOP formulation and $(solver) solver..."
	# Check if problem is not an instance of GOOPKKTSystem. Otherwise, build GOOPKKTSystem.
	if problem isa ReducedGOOP.GOOPKKTSystem
		GOOP_kkt_system = problem
	else
		GOOP_kkt_system = GOOP_kkt_system(problem)
	end

	if solver isa ReducedGOOP.InteriorPoint
		println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
		println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)
	else
		println("[PATH] MCP Dimension: ", GOOP_kkt_system.problem_size)
		println("[PATH] Variable Dimension: ", length(GOOP_kkt_system.lower_bounds))
	end

	dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
	primal_dimension   = dynamics_dimension * planning_horizon

	# ── Per-instance solver ────────────────────────────────────────────────────
	function solve_game_instance(θ; z₀, ϵ₀, max_inner_iters)
		options = if solver isa ReducedGOOP.InteriorPoint
			ReducedGOOP.InteriorPointOptions(;
				tol = 1e-3, #1e-4
				η₀ = 2e-5, # 5e-5, less than 1e-4
				ϵ₀,
				max_inner_iters,
				max_outer_iters = 1,
				tightening_rate = 2.0, # high => weak decrease in η
				loosening_rate = 0.5, # low => strong increase in η
				min_stepsize = 1e-20,
				linesearch,
				linear_solve_algorithm = ReducedGOOP.LinearSolve.KrylovJL_LSMR(),
				use_linsolve = false,
				record_convergence = true,
				record_condition_number = true,
				record_solver_diagnostics = true,
				solver_diagnostics_limit = 100,
				eta_retry_growth = 0.3,
				perturbation_enabled = false,
				stagnation_rtol = 1e-1,
				perturbation_scale = 1e-6,
				tsvd_threshold = 0.0, # 0.0: pure Tikhonov, > 0 and η = 0: pure TSVD
				use_marquardt_scaling = true,
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
		solver_diagnostics = NamedTuple[]
		total_iters = 0
		solver_status = :solved
		elapsed_time = @elapsed begin
			output = ReducedGOOP.solve(
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
					solver_diagnostics,
				) = output
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
		if solver isa ReducedGOOP.InteriorPoint
			save_solver_diagnostics_report(
				solver_diagnostics,
				normpath(joinpath(@__DIR__, "..", "solver_diagnostics.pdf"));
				max_rows = 1000,
			)
		end

		strategies = extract_player_strategies(
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
			"solver_diagnostics" => solver_diagnostics,
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
		velocity_plots_dir,
		control_plots_dir,
		warmstart_plots_dir,
	) = prepare_intersection_output_dirs(run_id; debug)

	# ── Main solve loop ────────────────────────────────────────────────────────
	instance_problem_data = Dict{String, Any}[]
	solved_attempts       = 0
	total_attempts        = 0

	while solved_attempts < num_instances
		total_attempts += 1
		initial_state1, initial_state2 = if random_initial_state
			(
				sample_initial_state(
					base_initial_state1,
					dynamics,
					dynamics_model,
					perturbation_scale,
				),
				sample_initial_state(
					base_initial_state2,
					dynamics,
					dynamics_model,
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

		(; θ1, θ2, θ) = build_instance_parameters(
			flatten_parameters,
			initial_state1,
			initial_state2,
			goal_position1,
			goal_position2,
			obstacle_position,
		)

		(; warmstart_solution) = if compute_warmstart
			build_default_warmstart(
				planning_horizon,
				dynamics,
				initial_state1,
				initial_state2;
				speed_component_limit,
			)
		else
			(;
				warmstart_solution = load("experiments/solution_dict_instance_1_eps0.1.jld2")["single_stored_object"]["x"][1:(num_players*primal_dimension)],
			)
		end
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
			goal_position1,
			goal_position2,
			speed_component_limit,
			dynamics_model,
			control_bounds,
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
				"total_solve_time_sec" => instance_total_solve_time_sec,
			),
		)

		solved_attempts += 1
		println(
			"instance $(solved_attempts) total solve time: $(round(instance_total_solve_time_sec; digits = 5)) sec",
		)
		println("instance $(solved_attempts) converged preference values by ϵ:")

		JLD2.save_object(
			joinpath(problem_data_dir, "problem_data_instance_$(solved_attempts).jld2"),
			instance_problem_data,
		)

		for (ϵ₀, result) in epsilon_results
			solution_dict = result.solution_dict
			# preference_values = evaluate_preferences_at_solution(
			# 	problem,
			# 	solution_dict["x"][1:(num_players*primal_dimension)],
			# 	θ,
			# )
			# solution_dict["preference_values"] = preference_values
			# println("  ϵ₀ = $(round(ϵ₀; digits = 5)):")
			# println("  kkt_error = $(solution_dict["kkt_error"])")
			# for (player_idx, player_preferences) in enumerate(preference_values)
			# 	println("    player $(player_idx): $(_fmt5(player_preferences))")
			# end

			JLD2.save_object(
				joinpath(
					solution_data_dir,
					"solution_dict_instance_$(solved_attempts)_eps$(ϵ₀).jld2",
				),
				solution_dict,
			)

			save_convergence_diagnostics(solution_dict, convergence_plots_dir, solved_attempts, ϵ₀)

			trajectory_fig, _ = plot_intersection_trajectories(
				;
				map_end,
				lane_width,
				strategy = result.strategies,
				θ1,
				θ2,
				goal_position1,
				goal_position2,
			)
			velocity_fig, _ = velocity_plot(;
				strategy = result.strategies,
				velocity_limit = speed_component_limit,
				dynamics_model,
			)
			control_fig, _ = control_plot(;
				strategy = result.strategies,
				control_lb = control_bounds.lb,
				control_ub = control_bounds.ub,
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
					velocity_plots_dir,
					"velocity_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
				),
				velocity_fig,
			)
			CairoMakie.save(
				joinpath(
					control_plots_dir,
					"control_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
				),
				control_fig,
			)
		end
	end

	JLD2.save_object(
		joinpath(problem_data_dir, "run_metadata.jld2"),
		Dict(
			"run_id"                   => run_id,
			"debug"                    => debug,
			"run_dir"                  => run_dir,
			"rng_seed"                 => rng_seed,
			"dynamics_model"           => dynamics_model,
			"num_instances"            => num_instances,
			"random_initial_state"     => random_initial_state,
			"epsilon_schedule"         => epsilon_schedule,
			"max_inner_iters_schedule" => max_inner_iters_schedule,
			"velocity_limit"           => speed_component_limit,
			"perturbation_scale"       => perturbation_scale,
		),
	)
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
	goal_position1,
	goal_position2,
	speed_component_limit,
	dynamics_model,
	control_bounds,
)
	warmstart_strategies = extract_player_strategies(
		warmstart_solution,
		num_players,
		primal_dimension,
		dynamics,
	)

	warmstart_fig, _ = plot_intersection_trajectories(
		;
		map_end,
		lane_width,
		strategy = warmstart_strategies,
		θ1,
		θ2,
		goal_position1,
		goal_position2,
	)
	warmstart_velocity_fig, _ = velocity_plot(
		;
		strategy = warmstart_strategies,
		velocity_limit = speed_component_limit,
		dynamics_model,
	)
	warmstart_control_fig, _ = control_plot(
		;
		strategy = warmstart_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
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
			"warmstart_velocity_attempt_$(total_attempts)_instance_$(instance_idx).pdf",
		),
		warmstart_velocity_fig,
	)
	CairoMakie.save(
		joinpath(
			warmstart_plots_dir,
			"warmstart_control_attempt_$(total_attempts)_instance_$(instance_idx).pdf",
		),
		warmstart_control_fig,
	)
end

function prepare_intersection_output_dirs(run_id; debug)
	run_dir = if debug
		joinpath("data", "Intersection_open_loop", "debug", run_id)
	else
		joinpath("data", "Intersection_open_loop", "runs", run_id)
	end

	data_dir              = joinpath(run_dir, "data")
	problem_data_dir      = joinpath(data_dir, "problem")
	solution_data_dir     = joinpath(problem_data_dir, "solution")
	plots_dir             = joinpath(run_dir, "plots")
	trajectory_plots_dir  = joinpath(plots_dir, "trajectories")
	convergence_plots_dir = joinpath(plots_dir, "convergence")
	velocity_plots_dir    = joinpath(plots_dir, "velocities")
	control_plots_dir     = joinpath(plots_dir, "controls")
	warmstart_plots_dir   = joinpath(plots_dir, "warmstart")

	for dir in (
		problem_data_dir,
		solution_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		velocity_plots_dir,
		control_plots_dir,
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
		velocity_plots_dir,
		control_plots_dir,
		warmstart_plots_dir,
	)
end

# ── Internal utilities ─────────────────────────────────────────────────────────

function build_intersection_dynamics(
	dynamics_model::Symbol;
	dt = 0.3,
	control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0]),
)
	if dynamics_model === :planar_double_integrator
		return planar_double_integrator(; dt, control_bounds)
	elseif dynamics_model === :unicycle
		return UnicycleDynamics(; dt, control_bounds)
	end
	throw(ArgumentError("Unsupported dynamics_model $(dynamics_model)"))
end

function velocity_limit_constraints(x, dynamics_model::Symbol; velocity_limit = 1.5)
	if dynamics_model === :planar_double_integrator
		vx, vy = x[3], x[4]
		return vcat(
			vx + velocity_limit,
			-vx + velocity_limit,
			vy + velocity_limit,
			-vy + velocity_limit,
		)
	elseif dynamics_model === :unicycle
		speed = x[3]
		return vcat(speed, -speed + velocity_limit) # 0 ≤ speed ≤ velocity_limit
	end
	throw(ArgumentError("Unsupported dynamics_model $(dynamics_model)"))
end

function sample_initial_state(
	base_initial_state,
	dynamics,
	dynamics_model::Symbol,
	perturbation_scale,
)
	noise = rand(Uniform(-perturbation_scale, perturbation_scale), state_dim(dynamics))
	if dynamics_model === :planar_double_integrator
		return base_initial_state .+ (base_initial_state .!= 0.0) .* noise
	elseif dynamics_model === :unicycle
		initial_state = copy(base_initial_state)
		initial_state[1:3] .+= noise[1:3] # perturb (px, py, speed) only
		return initial_state
	end
	throw(ArgumentError("Unsupported dynamics_model $(dynamics_model)"))
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
			state_dim(dynamics),
			control_dim(dynamics),
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

function build_instance_parameters(
	flatten_parameters,
	initial_state1,
	initial_state2,
	goal_position1,
	goal_position2,
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
	(; θ1, θ2, θ = [θ1..., θ2...])
end

function build_default_warmstart(
	planning_horizon,
	dynamics,
	initial_state1,
	initial_state2;
	speed_component_limit = 1.5,
)
	player1_warmstart = build_constant_control_warmstart(
		planning_horizon,
		dynamics,
		initial_state1,
		[0.0, 0.0],
	)

	player2_warmstart = build_constant_control_warmstart(
		planning_horizon,
		dynamics,
		initial_state2,
		[0.0, 0.0],
	)

	# player2_vx_profile = fill(0.0, planning_horizon)
	# player2_vy_profile = vcat(1.0, fill(speed_component_limit, planning_horizon - 1))
	# player2_warmstart = build_planar_di_velocity_profile_warmstart(
	# 	planning_horizon,
	# 	dynamics,
	# 	initial_state2;
	# 	vx_profile = player2_vx_profile,
	# 	vy_profile = player2_vy_profile,
	# )

	warmstart_solution = flatten_warmstart_solution(
		planning_horizon,
		[player1_warmstart.xs, player2_warmstart.xs],
		[player1_warmstart.us, player2_warmstart.us],
	)
	(; warmstart_solution)
end

function dynamics_step(dynamics, x, u, k)
	if applicable(dynamics, x, u, k)
		return dynamics(x, u, k)
	end
	return dynamics(x, u)
end

function infer_planar_di_timestep(dynamics)
	x0       = [0.0, 0.0, 0.0, 0.0]
	u_unit_y = [0.0, 1.0]
	x1       = dynamics_step(dynamics, x0, u_unit_y, 1)
	dt       = x1[4]
	dt <= 0.0 && error("Failed to infer planar double-integrator timestep from dynamics.")
	dt
end

function build_constant_control_warmstart(planning_horizon, dynamics, initial_state, constant_control)
	length(initial_state) == state_dim(dynamics) || error("Initial state dimension mismatch.")
	length(constant_control) == control_dim(dynamics) || error("Control dimension mismatch.")

	xs = [collect(initial_state)]
	us = [collect(constant_control)]
	for k in 1:(planning_horizon-1)
		push!(xs, dynamics_step(dynamics, xs[k], us[1], k))
		push!(us, copy(us[1]))
	end
	us[end] = zeros(control_dim(dynamics))
	(; xs, us)
end

function build_planar_di_velocity_profile_warmstart(
	planning_horizon,
	dynamics,
	initial_state;
	vx_profile,
	vy_profile,
)
	length(initial_state) == 4 || error("Expected 4D planar-double-integrator state.")
	control_dim(dynamics) == 2 || error("Expected 2D control input for planar-double-integrator.")
	length(vx_profile) == planning_horizon || error("vx_profile length must equal planning_horizon.")
	length(vy_profile) == planning_horizon || error("vy_profile length must equal planning_horizon.")

	dt = infer_planar_di_timestep(dynamics)
	x0 = [initial_state[1], initial_state[2], vx_profile[1], vy_profile[1]]
	xs = [x0]
	us = Vector{Vector{Float64}}()
	for k in 1:(planning_horizon-1)
		u = [
			(vx_profile[k+1] - vx_profile[k]) / dt,
			(vy_profile[k+1] - vy_profile[k]) / dt,
		]
		push!(us, u)
		push!(xs, dynamics_step(dynamics, xs[k], u, k))
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
		warmstart_primals = mapreduce(vcat, 1:planning_horizon) do k
			vcat(warmstart_x[player][k], warmstart_u[player][k])
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
end

function save_solver_diagnostics_report(diagnostics, output_path; max_rows = 1000)
	max_rows >= 0 || throw(ArgumentError("max_rows must be nonnegative."))
	rows = diagnostics[1:min(length(diagnostics), max_rows)]
	isempty(rows) && error("No solver diagnostics were recorded; cannot generate $(output_path).")

	iterations = getproperty.(rows, :inner_iter)
	metric_specs = [
		(:kkt_error, "kkt_error"),
		(:singular_values_lt_1e_6, "count(Jsvd.S < 1e-6)"),
		(:singular_values_lt_1e_2, "count(Jsvd.S < 1e-2)"),
		(:max_singular_value, "maximum(Jsvd.S)"),
		(:min_singular_value, "minimum(Jsvd.S)"),
		(:max_abs_delta_z_1_144, "maximum(abs.(delta_z[1:144]))"),
		(:max_abs_delta_z_145_end, "maximum(abs.(delta_z[145:end]))"),
	]

	mktempdir() do report_dir
		plots_path = joinpath(report_dir, "solver_diagnostic_plots.pdf")
		fig = CairoMakie.Figure(size = (1400, 1800), fontsize = 18)
		axes = [
			CairoMakie.Axis(fig[1, 1], title = "KKT error", xlabel = "inner iteration", ylabel = "kkt_error"),
			CairoMakie.Axis(fig[1, 2], title = "Small singular values", xlabel = "inner iteration", ylabel = "count"),
			CairoMakie.Axis(fig[2, 1], title = "Maximum singular value", xlabel = "inner iteration", ylabel = "maximum(Jsvd.S)"),
			CairoMakie.Axis(fig[2, 2], title = "Minimum singular value", xlabel = "inner iteration", ylabel = "minimum(Jsvd.S)"),
			CairoMakie.Axis(fig[3, 1], title = "Maximum primal-step magnitude", xlabel = "inner iteration", ylabel = "max abs delta_z[1:144]"),
			CairoMakie.Axis(fig[3, 2], title = "Maximum remaining-step magnitude", xlabel = "inner iteration", ylabel = "max abs delta_z[145:end]"),
		]

		CairoMakie.lines!(axes[1], iterations, getproperty.(rows, :kkt_error); linewidth = 2)
		CairoMakie.lines!(axes[2], iterations, getproperty.(rows, :singular_values_lt_1e_6); label = "< 1e-6", linewidth = 2)
		CairoMakie.lines!(axes[2], iterations, getproperty.(rows, :singular_values_lt_1e_2); label = "< 1e-2", linewidth = 2)
		CairoMakie.axislegend(axes[2]; position = :lt)
		CairoMakie.lines!(axes[3], iterations, getproperty.(rows, :max_singular_value); linewidth = 2)
		CairoMakie.lines!(axes[4], iterations, getproperty.(rows, :min_singular_value); linewidth = 2)
		CairoMakie.lines!(axes[5], iterations, getproperty.(rows, :max_abs_delta_z_1_144); linewidth = 2)
		CairoMakie.lines!(axes[6], iterations, getproperty.(rows, :max_abs_delta_z_145_end); linewidth = 2)
		CairoMakie.save(plots_path, fig)

		tex_path = joinpath(report_dir, "solver_diagnostics.tex")
		open(tex_path, "w") do io
			println(io, raw"\documentclass[10pt]{article}")
			println(io, raw"\usepackage[letterpaper,landscape,margin=0.5in]{geometry}")
			println(io, raw"\usepackage{booktabs,longtable,graphicx}")
			println(io, raw"\setlength{\tabcolsep}{4pt}")
			println(io, raw"\begin{document}")
			println(io, raw"\section*{Solver Diagnostics}")
			println(io, "Recorded $(length(diagnostics)) inner iterations; reporting $(length(rows)) rows (cap: $(max_rows)).")
			println(io, raw"\subsection*{Summary statistics}")
			println(io, raw"\begin{tabular}{lrrrr}")
			println(io, "\\toprule Metric & Minimum & Maximum & Mean & Median \\\\")
			println(io, raw"\midrule")
			for (field, label) in metric_specs
				values = Float64.(getproperty.(rows, field))
				println(
					io,
					"\\texttt{$(_latex_escape(label))} & $(_diagnostic_number(minimum(values))) & $(_diagnostic_number(maximum(values))) & $(_diagnostic_number(mean(values))) & $(_diagnostic_number(median(values))) \\\\",
				)
			end
			println(io, raw"\bottomrule")
			println(io, raw"\end{tabular}")
			println(io, raw"\clearpage")
			println(io, raw"\subsection*{Metrics versus inner iteration}")
			println(io, raw"\begin{center}")
			println(io, raw"\includegraphics[width=0.91\textwidth,height=0.78\textheight,keepaspectratio]{solver_diagnostic_plots.pdf}")
			println(io, raw"\end{center}")
			println(io, raw"\clearpage")
			println(io, raw"\subsection*{Per-iteration diagnostics}")
			println(io, raw"\scriptsize")
			println(io, raw"\begin{longtable}{rrrrrrrr}")
			println(io, "\\toprule Iter & KKT error & \$N_{<10^{-6}}\$ & \$N_{<10^{-2}}\$ & \$\\sigma_{\\max}\$ & \$\\sigma_{\\min}\$ & \$\\max|\\delta z_{1:144}|\$ & \$\\max|\\delta z_{145:}|\$ \\\\")
			println(io, raw"\midrule")
			println(io, raw"\endfirsthead")
			println(io, "\\toprule Iter & KKT error & \$N_{<10^{-6}}\$ & \$N_{<10^{-2}}\$ & \$\\sigma_{\\max}\$ & \$\\sigma_{\\min}\$ & \$\\max|\\delta z_{1:144}|\$ & \$\\max|\\delta z_{145:}|\$ \\\\")
			println(io, raw"\midrule")
			println(io, raw"\endhead")
			for row in rows
				println(
					io,
					"$(row.inner_iter) & $(_diagnostic_number(row.kkt_error)) & $(row.singular_values_lt_1e_6) & $(row.singular_values_lt_1e_2) & $(_diagnostic_number(row.max_singular_value)) & $(_diagnostic_number(row.min_singular_value)) & $(_diagnostic_number(row.max_abs_delta_z_1_144)) & $(_diagnostic_number(row.max_abs_delta_z_145_end)) \\\\",
				)
			end
			println(io, raw"\bottomrule")
			println(io, raw"\end{longtable}")
			println(io, raw"\end{document}")
		end

		isnothing(Sys.which("pdflatex")) && error("pdflatex is required to generate $(output_path).")
		latex_command = Cmd(`pdflatex -interaction=nonstopmode -halt-on-error solver_diagnostics.tex`; dir = report_dir)
		for _ in 1:2
			latex_output = IOBuffer()
			process = run(
				pipeline(ignorestatus(latex_command); stdout = latex_output, stderr = latex_output),
			)
			success(process) || error(
				"Failed to generate solver diagnostics PDF with pdflatex:\n$(String(take!(latex_output)))",
			)
		end
		mkpath(dirname(output_path))
		cp(joinpath(report_dir, "solver_diagnostics.pdf"), output_path; force = true)
	end

	@info "Saved solver diagnostics report" output_path rows = length(rows)
	output_path
end

_diagnostic_number(value) = @sprintf("%.6e", value)

function _latex_escape(text)
	replace(text, "_" => raw"\_", "<" => raw"\textless{}")
end


_fmt5(x) = x isa Real ? round(x; digits = 5) :
		   x isa AbstractArray ? map(_fmt5, x) : x

# ── Advanced diagnostic ────────────────────────────────────────────────────────
# Recovers dual variables for the complete KKT system given a fixed primal.
# The call site in demo() is commented out; kept here for reference.

function recover_complete_kkt_duals_for_fixed_primal(
	complete_kkt_system,
	fixed_primal,
	θ;
	ϵ₀,
	η₀,
	dual_init = nothing,
)
	F_symbolic        = complete_kkt_system.F_symbolic
	z_symbolic        = complete_kkt_system.z_symbolic
	primal_indices    = collect(complete_kkt_system.primal_dims)
	nonprimal_indices = setdiff(collect(eachindex(z_symbolic)), primal_indices)
	nonprimal_symbols = z_symbolic[nonprimal_indices]

	substitution_dict = Dict{Any, Any}()
	for (sym, val) in zip(z_symbolic[primal_indices], fixed_primal)
		substitution_dict[sym] = val
	end
	backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend()
	θ_symbols = ReducedGOOP.SymbolicTracingUtils.make_variables(backend, :θ, length(θ))
	for (sym, val) in zip(θ_symbols, θ)
		substitution_dict[sym] = val
	end
	let
		ϵ_sym = only(ReducedGOOP.SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
		η_sym = only(ReducedGOOP.SymbolicTracingUtils.make_variables(backend, :η, 1))
		substitution_dict[ϵ_sym] = ϵ₀
		substitution_dict[η_sym] = η₀
	end

	F_symbolic_after_sub = Symbolics.substitute.(F_symbolic, Ref(substitution_dict))
	F_eval = first(
		Symbolics.build_function(
			F_symbolic_after_sub,
			nonprimal_symbols;
			expression = Val(false),
		),
	)
	dual_residual(u, _) = F_eval(u)

	u₀     = isnothing(dual_init) ? zeros(length(nonprimal_indices)) : copy(dual_init)
	prob     = NonlinearLeastSquaresProblem(dual_residual, u₀)
	dual_sol = NonlinearSolve.solve(prob)

	z_recovered = zeros(length(z_symbolic))
	z_recovered[primal_indices] = fixed_primal
	z_recovered[nonprimal_indices] = dual_sol.u

	F_recovered = zeros(complete_kkt_system.kkt_dimension)
	complete_kkt_system.F!(F_recovered, z_recovered; θ, ϵ = ϵ₀, η = η₀)
	kkt_error_recovered = norm(F_recovered, 2)

	Dict(
		"status"        => string(dual_sol.retcode),
		"kkt_error"     => kkt_error_recovered,
		"fixed_primal"  => fixed_primal,
		"dual_solution" => dual_sol.u,
		"z_recovered"   => z_recovered,
	)
end

end
