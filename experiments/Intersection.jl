module Intersection

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
	unflatten_trajectory, state_dim, control_dim, control_bounds
using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, Distributions, Random
using Symbolics, NonlinearSolve, LinearAlgebra
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
	primals_per_agent = (state_dimension + control_dimension) * planning_horizon
	primal_dimensions = fill(primals_per_agent, num_players)
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
		return (min(h, 0))^4
	end

	control_objectives = [
		function (z, _)
			(; us) =
				unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
			sum(sum(u .^ 2) for u in us)
		end,
		function (z, _)
			(; us) =
				unflatten_trajectory(z[Block(2)], state_dimension, control_dimension)
			sum(sum(u .^ 2) for u in us)
		end,
	]

	function shared_inequality_constraint(z, _)
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
		# xs[1][end][1]^2 + xs[2][end][1]^2 - 1000 # xs[1][T] + xs[2][T] ≥ 0 for testing
	end

	inequality_constraints = [
		function (z, θ)
			(; lb, ub) = control_bounds(dynamics)
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) =
				unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
			vcat(
				mapreduce(vcat, us) do u
					vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
				end,
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
				shared_inequality_constraint(z, θ),
			)
		end, function (z, θ)
			(; lb, ub) = control_bounds(dynamics)
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) =
				unflatten_trajectory(z[Block(2)], state_dimension, control_dimension)
			vcat(
				mapreduce(vcat, us) do u
					vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
				end,
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
				shared_inequality_constraint(z, θ),
			)
		end,
	]

	equality_constraints = [
		function (z, θ)
			(; lb, ub) = control_bounds(dynamics)
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) =
				unflatten_trajectory(z[Block(i)], state_dimension, control_dimension)
			(; initial_state) = unflatten_parameters(θ[Block(i)])
			initial_state_constraint = xs[1] - initial_state
			dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
				xs[k] - dynamics(xs[k-1], us[k-1], k)
			end
			vcat(
				initial_state_constraint,
				dynamics_constraints,
				# squared_violation.(
				# 	mapreduce(vcat, us) do u
				# 		vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
				# 	end,
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
				# squared_violation.(shared_inequality_constraint(z, θ)),
			)
		end for i in 1:num_players
	]

	preferences = [
		[
			# Minimize control effort 
			control_objectives[1],

			# Drive under speed limit
			function (z, _)
				(; xs) = unflatten_trajectory(
					z[Block(1)],
					state_dimension,
					control_dimension,
				)
				mapreduce(vcat, 1:length(xs)) do k
					velocity_limit_constraints(xs[k], dynamics_model; velocity_limit)
				end
			end,

			# Reach the goal (highest priority for P1)
			function (z, θ)
				(; xs) = unflatten_trajectory(
					z[Block(1)],
					state_dimension,
					control_dimension,
				)
				(; goal_position) = unflatten_parameters(θ[Block(1)])
				goal_deviation = xs[end][1:2] .- goal_position
				sum(goal_deviation .^ 2)
			end,

			# Lane bounds + collision avoidance (constraint, both players)
			inequality_constraints[1],
		],
		[
			# Minimize control effort
			control_objectives[2],

			# Reach the goal
			function (z, θ)
				(; xs) = unflatten_trajectory(
					z[Block(2)],
					state_dimension,
					control_dimension,
				)
				(; goal_position) = unflatten_parameters(θ[Block(2)])
				goal_deviation = xs[end][1:2] .- goal_position
				sum(goal_deviation .^ 2)
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
			inequality_constraints[2],
		],
	]


	# Preference hierarchy: [lowest priority, ..., highest priority]
	is_prioritized_constraint = [[false, true, false, true], [false, false, true, true]]

	problem = ReducedGOOP.ParametricGOOP(
		dummy_primals,
		dummy_parameters;
		preferences,
		is_prioritized_constraint,
		equality_constraints,
		inequality_constraints = [nothing, nothing],
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)

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
	run_id = "0_IP_reduced_four_levels_extra_level_ca_eta"
	dynamics_model = :planar_double_integrator   # :planar_double_integrator | :unicycle
	goop_version = :reduced                    # :complete | :reduced | :quasi
	solver = ReducedGOOP.InteriorPoint() # ReducedGOOP.InteriorPoint() | ReducedGOOP.PATHSolver()
	linesearch = :backtracking          # :backtracking | :fraction_to_boundary
	compute_warmstart = true # Whether to compute a warmstart trajectory via rollout (true) or load from file (false)

	# ── Problem parameters ─────────────────────────────────────────────────────
	num_players           = 2
	planning_horizon      = 8
	collision_avoidance   = 1.3
	speed_component_limit = 1.5
	control_bounds        = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
	num_instances         = 1
	perturbation_scale    = 0.3

	# ── Solver schedule ────────────────────────────────────────────────────────
	epsilon_schedule         = [0.1]
	max_inner_iters_schedule = fill(100_000, length(epsilon_schedule))

	# ── Scenario ───────────────────────────────────────────────────────────────
	# Planar double integrator: state = [px, py, vx, vy]
	base_initial_state1 = [-6.0, -1.0, 2.5, 0.0]
	base_initial_state2 = [1.0, -6.0, 0.0, 1.0]
	# Unicycle: state = [px, py, speed, heading] — uncomment to switch
	# base_initial_state1 = [-6.0, -1.0, 0.0, 0.0]
	# base_initial_state2 = [1.0, -6.0, 1.3, π/2]

	goal_position1    = [6.0, -1.0]
	goal_position2    = [1.0, 6.0]
	obstacle_position = [0.25, 0.15]   # placeholder

	# ── Build dynamics and problem ─────────────────────────────────────────────
	dynamics = build_intersection_dynamics(dynamics_model; dt = 0.5, control_bounds)

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
	GOOP_kkt_system = GOOP_kkt_system(problem)

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
				tol = 1e-4,
				η₀ = 1e-7,
				ϵ₀,
				max_inner_iters,
				max_outer_iters = 1,
				tightening_rate = 1.0,
				loosening_rate = 0.3,
				min_stepsize = 1e-20,
				linesearch,
				linear_solve_algorithm = ReducedGOOP.LinearSolve.KrylovJL_LSMR(),
				use_linsolve = false,
				record_convergence = true,
				record_condition_number = true,
				eta_retry_growth = 0.3,
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
				initial_state2,
			)
		else
			(;
				warmstart_solution = load("experiments/solution_dict_instance_1_eps0.1.jld2")["single_stored_object"]["x"][1:96],
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
			preference_values = evaluate_preferences_at_solution(
				problem,
				solution_dict["x"][1:(num_players*primal_dimension)],
				θ,
			)
			solution_dict["preference_values"] = preference_values
			println("  ϵ₀ = $(round(ϵ₀; digits = 5)):")
			println("  kkt_error = $(solution_dict["kkt_error"])")
			for (player_idx, player_preferences) in enumerate(preference_values)
				println("    player $(player_idx): $(_fmt5(player_preferences))")
			end

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
	initial_state2,
)
	player1_warmstart = build_constant_control_warmstart(
		planning_horizon,
		dynamics,
		initial_state1,
		[0.0, 0.0],
	)

	player2_vx_profile = fill(0.0, planning_horizon)
	player2_vy_profile = vcat(1.0, fill(1.5, planning_horizon - 1))
	player2_warmstart = build_planar_di_velocity_profile_warmstart(
		planning_horizon,
		dynamics,
		initial_state2;
		vx_profile = player2_vx_profile,
		vy_profile = player2_vy_profile,
	)

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
