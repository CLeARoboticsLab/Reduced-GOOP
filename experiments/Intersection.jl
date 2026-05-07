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
using QuasiGOOP

include(joinpath(@__DIR__, "Plotting.jl"))

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

	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions) # THIS will be x
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

	objectives = [
		function (z, θ)
			(; xs, us) =
				unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
			(; goal_position) = unflatten_parameters(θ[Block(2)]) # Player 2
			goal_deviation = xs[end][1:2] .- goal_position
			# [
			# 	goal_deviation .+ 0.01
			# 	-goal_deviation .+ 0.01
			# ]
			sum(sum(u .^ 2) for u in us) #+ sum(goal_deviation .^ 2)
		end,
		function (z, θ)
			(; xs, us) =
				unflatten_trajectory(z[Block(2)], state_dimension, control_dimension)
			(; goal_position) = unflatten_parameters(θ[Block(2)]) # Player 2
			goal_deviation = xs[end][1:2] .- goal_position
			# [
			# 	goal_deviation .+ 0.01
			# 	-goal_deviation .+ 0.01
			# ]
			sum(sum(u .^ 2) for u in us) #+ sum(goal_deviation .^ 2)
		end,
	]

	equality_constraints = [
		function (z, θ)
			(; xs, us) =
				unflatten_trajectory(z[Block(i)], state_dimension, control_dimension)
			(; initial_state) = unflatten_parameters(θ[Block(i)]) # Player i θ[Block(i)]
			initial_state_constraint = xs[1] - initial_state
			# zero_initial_control = us[1]
			dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
				xs[k] - dynamics(xs[k-1], us[k-1], k)
			end
			# vcat(initial_state_constraint, zero_initial_control, dynamics_constraints)
			vcat(initial_state_constraint, dynamics_constraints)
		end for i in 1:num_players
	]

	function shared_inequality_constraint(z, θ)
		trajectories = map(
			i ->
				unflatten_trajectory(z[Block(i)], state_dimension, control_dimension),
			1:num_players,
		)
		xs = map(trajectory -> trajectory.xs, trajectories)
		@assert length(xs) == num_players
		# Avoid collision between 2 players.
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
				# control bounds (box)
				mapreduce(vcat, us) do u
					vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
				end,

				# stay within the intersection. R1 (ambulance)
				mapreduce(vcat, 1:length(xs)) do k
					px = xs[k][1]
					py = xs[k][2]
					position_constraints = vcat(
						px + map_end,
						-px + map_end,
						py + lane_width,
						-py + lane_width,
					) # -7 ≤ pₓ ≤ 7, -2 ≤ py ≤ 2
					vcat(position_constraints)
				end,

				# add shared collision-avoidance constraints
				# shared_inequality_constraint(z, θ),
			)
		end,
		function (z, θ)
			(; lb, ub) = control_bounds(dynamics)
			lb_mask = findall(!isinf, lb)
			ub_mask = findall(!isinf, ub)
			(; xs, us) =
				unflatten_trajectory(z[Block(2)], state_dimension, control_dimension)
			vcat(
				# control bounds (box)
				mapreduce(vcat, us) do u
					vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
				end,

				# stay within the intersection. R2 (car)
				mapreduce(vcat, 1:length(xs)) do k
					px = xs[k][1]
					py = xs[k][2]
					position_constraints = vcat(
						px + lane_width,
						-px + lane_width,
						py + map_end,
						-py + map_end,
					) # -2 ≤ pₓ ≤ 2, -7 ≤ py ≤ 7
					vcat(position_constraints)
				end,

				# add shared collision-avoidance constraints
				# shared_inequality_constraint(z, θ),
			)
		end,
		# for now, two robots
	]

	prioritized_preferences = [
		[

			# Drive under speed limit 
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(1)],
					state_dimension,
					control_dimension,
				)
				mapreduce(vcat, 1:length(xs)) do k
					velocity_limit_constraints(xs[k], dynamics_model; velocity_limit)
				end
			end,

			# reach the goal. (highest priority)
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(1)],
					state_dimension,
					control_dimension,
				)
				(; goal_position) = unflatten_parameters(θ[Block(1)]) # Player 1 θ[Block(i)] Ambuluance
				goal_deviation = xs[end][1:2] .- goal_position
				# [
				# 	goal_deviation .+ 0.01
				# 	-goal_deviation .+ 0.01
				# ]
				sum(goal_deviation .^ 2) #+ 0.1sum(sum(u .^ 2) for u in us)
				# sum(sum(u .^ 2) for u in us)
			end,
		],
		[
			# reach the goal.
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(2)],
					state_dimension,
					control_dimension,
				)
				(; goal_position) = unflatten_parameters(θ[Block(2)]) # Player 2
				goal_deviation = xs[end][1:2] .- goal_position
				# [
				# 	goal_deviation .+ 0.01
				# 	-goal_deviation .+ 0.01
				# ]
				sum(goal_deviation .^ 2) #+ 0.1sum(sum(u .^ 2) for u in us)
				# sum(sum(u .^ 2) for u in us)
			end,


			# Drive under speed limit (highest priority)
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(2)],
					state_dimension,
					control_dimension,
				)
				mapreduce(vcat, 1:length(xs)) do k
					velocity_limit_constraints(xs[k], dynamics_model; velocity_limit)
				end
			end,
		],
	]

	# Specify prioritized constraint [lowest priority, ..., highest priority]
	# is_prioritized_constraint = [[false, false], [false, false]]
	# preferences = [vcat(objectives[player], prioritized_preferences[player]) for player in 1:num_players]

	# is_prioritized_constraint = [[false, false], [false, false]]
	# preferences = [vcat(objectives[player], prioritized_preferences[player]) for player in 1:num_players]
	# preferences = [vcat(prioritized_preferences[player], objectives[player]) for player in 1:num_players]

	# is_prioritized_constraint = [[false, true, false], [false, false, true]]
	# preferences = [vcat(objectives[player], prioritized_preferences[player]) for player in 1:num_players]

	is_prioritized_constraint = [[true, false], [false, true]]
	preferences = prioritized_preferences

	# is_prioritized_constraint = [[false], [false]]
	# preferences = [[objectives[player]] for player in 1:num_players]
	# preferences = [prioritized_preferences[player] for player in 1:num_players]


	problem = QuasiGOOP.ParametricGOOP(
		dummy_primals, # x
		dummy_parameters; # θ
		preferences,
		is_prioritized_constraint,
		equality_constraints,
		inequality_constraints, # = [nothing, nothing],
		shared_equality_constraint = nothing, # keep these under individual constraints
		shared_inequality_constraint = nothing,
	)

	(; problem, flatten_parameters)
end

function demo(;
	map_end = 7,
	lane_width = 2,
	verbose = false,
	rng_seed = 123,
	random_initial_state = true,
	debug = false,
)
	Random.seed!(rng_seed)

	# Problem setup
	num_players = 2
	control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
	planning_horizon = 4
	collision_avoidance = 1.3
	speed_component_limit = 1.5
	num_instances = 1
	epsilon_schedule = [1.0*0.5^(j-1) for j in 1:13] # 1:11 
	max_inner_iters_schedule = fill(5000, length(epsilon_schedule))
	perturbation_scale = 0.3
	dynamics_model = :planar_double_integrator # :unicycle, :planar_double_integrator
	goop_version = :reduced # :complete, :reduced, :quasi 
	solver = QuasiGOOP.PATHSolver() # QuasiGOOP.InteriorPoint(), QuasiGOOP.PATHSolver()
	dynamics = build_intersection_dynamics(dynamics_model; dt = 0.5, control_bounds)

	# run_id = "run_$(dynamics_model)_$(goop_version)_1_pref_$(num_instances)_instances_horizon_$(planning_horizon)_linesearch_$(linesearch)_goal_reaching_3"

	run_id = "0_PATH_init_vel2_1.0"
	# run_id = "0_PDIP_init_vel2_1.0_w_total_complementarity"

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

	kkt_generators = if solver isa QuasiGOOP.InteriorPoint
		Dict(
			:complete => QuasiGOOP.generate_slacked_complete_kkt_system,
			:reduced => QuasiGOOP.generate_slacked_reduced_kkt_system,
			:quasi => QuasiGOOP.generate_slacked_quasi_kkt_system,
		)
	else
		Dict(
			:complete => QuasiGOOP.generate_mcp_complete_kkt_system,
			:reduced => QuasiGOOP.generate_mcp_reduced_kkt_system,
			# :quasi => QuasiGOOP.generate_mcp_quasi_kkt_system,
		)
	end

	GOOP_kkt_system = get(kkt_generators, goop_version, nothing) # return nothing if goop_version key is not found
	isnothing(GOOP_kkt_system) && error("Unknown GOOP version: $(goop_version)")
	GOOP_kkt_system = GOOP_kkt_system(problem)

	if solver isa QuasiGOOP.InteriorPoint
		println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
		println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)
	else
		println("[PATH] MCP Dimension: ", GOOP_kkt_system.problem_size)
		println("[PATH] Variable Dimension: ", length(GOOP_kkt_system.lower_bounds))
	end

	function solve_game_instance(θ; z₀, ϵ₀, max_inner_iters)
		options = if solver isa QuasiGOOP.InteriorPoint
			QuasiGOOP.InteriorPointOptions(;
				tol = 1e-5,
				η₀ = 0.0,
				ϵ₀,
				max_inner_iters,
				max_outer_iters = 2,
				tightening_rate = 0.001,
				loosening_rate = 0.05,
				min_stepsize = 1e-20,
				linesearch = :backtracking, # :backtracking, :fraction_to_boundary
				linear_solve_algorithm = QuasiGOOP.LinearSolve.KrylovJL_LSMR(),
				use_linsolve = false,
				record_convergence = true,
				verbose,
			)
		else
			QuasiGOOP.PATHOptions(;
				convergence_tolerance = 1e-4, #1e-1
				ϵ₀,
				cumulative_iteration_limit = 1000000,
				proximal_perturbation = 1e-2,
				major_iteration_limit = 10000,
				minor_iteration_limit = 15000,
				nms_initial_reference_factor = 50000,
				nms_maximum_watchdogs = 8000,
				nms_memory_size = 16000,
				nms_mstep_frequency = 5000,
				lemke_start_type = "advanced",
				lemke_rank_deficiency_iterations = 50,
				restart_limit = 120,
				gradient_step_limit = 120,
				use_basics = true,
				use_start = true,
				verbose,
			)
		end
		elapsed_time = @elapsed begin
			output = QuasiGOOP.solve(
				solver, # QuasiGOOP.InteriorPoint(), QuasiGOOP.PATHSolver()
				GOOP_kkt_system,
				θ;
				z₀,
				options,
			)
			if solver isa QuasiGOOP.InteriorPoint
				(; status, z, x, s, σ, γ, kkt_error, ϵ, total_iters, kkt_error_history) = output
				status == :failed && return nothing
			else
				(; status, z, ϵ, info) = output
				@show status
				Int(status) != 1 && return nothing
				kkt_error = info.residual
				x = z[1:(num_players*primal_dimension)]
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
			# "s" => s,
			"solve_time_sec" => elapsed_time,
			"kkt_error" => kkt_error,
			"ϵ" => ϵ,
			# "total_iters" => total_iters, # TODO: PATH solver does not give total_iters and kkt error history
			# "kkt_error_history" => kkt_error_history,
		)

		(; strategies, solution_dict)
	end

	dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
	primal_dimension = dynamics_dimension * planning_horizon
	obstacle_position = [0.25, 0.15] # placeholder
	base_initial_state1, base_initial_state2 = default_intersection_initial_states(dynamics_model)
	goal_position1 = [6.0, -1.0]
	goal_position2 = [1.0, 6.0]

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

	instance_problem_data = Dict{String, Any}[]
	solved_attempts = 0
	total_attempts = 0

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

		(; warmstart_solution, player2_warmstart_primal) = build_default_warmstart(
			planning_horizon,
			dynamics,
			initial_state1,
			initial_state2,
			num_players,
			primal_dimension,
		)
		# # Use control-optimal trajectory as warmstart
		# data = load_object("./data/Intersection_open_loop/debug/0_10goal_control/data/problem/solution/solution_dict_instance_1_eps0.0001953125.jld2")
		# warmstart_solution = data["z"][1:(num_players*primal_dimension)]
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
		stage_warmstart = warmstart_solution #warmstart_solution #load_object("stage_warmstart.jld2") #warmstart_solution
		solve_sequence_succeeded = true
		instance_total_solve_time_sec = 0.0
		for (ϵ₀, max_inner_iters) in zip(epsilon_schedule, max_inner_iters_schedule)
			result = try
				solve_game_instance(
					θ;
					z₀ = stage_warmstart,
					ϵ₀,
					max_inner_iters,
				)
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

			# # Recover dual variables for the complete KKT system given the fixed primal from the current solution
			# solved_primal = result.solution_dict["x"][1:(num_players*primal_dimension)]
			# fixed_primal = copy(solved_primal)
			# fixed_primal[(primal_dimension + 1):(num_players*primal_dimension)] .=
			# 	player2_warmstart_primal

			# dual_init = if length(result.solution_dict["z"]) == complete_GOOP_kkt_system.variable_dimension
			# 	result.solution_dict["z"][setdiff(
			# 		collect(1:complete_GOOP_kkt_system.variable_dimension),
			# 		collect(complete_GOOP_kkt_system.primal_dims),
			# 	)]
			# else
			# 	nothing
			# end
			# recovery_dict = recover_complete_kkt_duals_for_fixed_primal(
			# 	complete_GOOP_kkt_system,
			# 	fixed_primal,
			# 	θ;
			# 	ϵ₀,
			# 	η₀ = 0.0,
			# 	dual_init,
			# )
			# result.solution_dict["fixed_primal_dual_recovery"] = recovery_dict
			# println(
			# 	"  [dual recovery @ ϵ=$(round(ϵ₀; digits = 5))] status=$(recovery_dict["status"]), ||F||₂=$(round(recovery_dict["kkt_error"]; digits = 6))",
			# )

			# warmstart next solve with the previous solution's primal variables
			# TODO: this needs to be fixed for PATH, use very first warmstart for now
			stage_warmstart = warmstart_solution # result.solution_dict["z"][1:(num_players*primal_dimension)]
		end
		if !solve_sequence_succeeded
			if !random_initial_state
				println("deterministic mode enabled: solver failed for default initial states.")
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
		format_5dp(x) = x isa Real ? round(x; digits = 5) :
						x isa AbstractArray ? map(format_5dp, x) : x
		println(
			"instance $(solved_attempts) total solve time: $(round(instance_total_solve_time_sec; digits = 5)) sec",
		)
		println("instance $(solved_attempts) converged preference values by ϵ:")
		# Save per-instance setup/summary once (all epsilons share this metadata).
		JLD2.save_object(
			joinpath(problem_data_dir, "problem_data_instance_$(solved_attempts).jld2"),
			instance_problem_data,
		)
		# Single pass over epsilon results: post-process, serialize, and export plots.
		for (ϵ₀, result) in epsilon_results
			solution_dict = result.solution_dict
			# Evaluate preference values at the solved primal trajectory for reporting.
			preference_values = evaluate_preferences_at_solution(
				problem,
				solution_dict["x"][1:(num_players*primal_dimension)],
				θ,
			)
			solution_dict["preference_values"] = preference_values
			println("  ϵ₀ = $(round(ϵ₀; digits = 5)):")
			slack_start = 1
			for (player_idx, player_preferences) in enumerate(preference_values)
				println("    player $(player_idx): $(format_5dp(player_preferences))")
				# player_slack_dim = sum(
				# 	length(vec(player_preferences[pref_idx])) for pref_idx in eachindex(player_preferences) if
				# 												  problem.is_prioritized_constraint[player_idx][pref_idx]
				# )
				# player_slacks = if player_slack_dim > 0
				# 	solution_dict["s"][
				# 		slack_start:(slack_start+player_slack_dim-1)
				# 	]
				# else
				# 	Float64[]
				# end
				# slack_start += player_slack_dim
				# println("      preference_slacks: $(format_5dp(player_slacks))")
				# # player_idx == 2 && println("	  γ_pref: $(format_5dp(solution_dict["z"][353:368]))") # γₚ_2_2 for current setup
				# # player_idx == 2 && println("	  σ_pref: $(format_5dp(solution_dict["z"][113:128]))") # σₚ_2_2 for current setup
			end

			# Save the full solution dictionary for this epsilon stage.
			JLD2.save_object(
				joinpath(
					solution_data_dir,
					"solution_dict_instance_$(solved_attempts)_eps$(ϵ₀).jld2",
				),
				solution_dict,
			)

			# # Build diagnostic figures from the saved solution payload.
			# convergence_fig, _ = plot_convergence_plot(
			# 	;
			# 	kkt_error_history = log10.(solution_dict["kkt_error_history"]),
			# 	total_iters = solution_dict["total_iters"],
			# )
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
			# Export one figure per metric to keep downstream analysis simple.
			# CairoMakie.save(
			# 	joinpath(
			# 		convergence_plots_dir,
			# 		"convergence_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
			# 	),
			# 	convergence_fig,
			# )
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
		end # end of single pass over epsilon results
	end

	JLD2.save_object(
		joinpath(problem_data_dir, "run_metadata.jld2"),
		Dict(
			"run_id" => run_id,
			"debug" => debug,
			"run_dir" => run_dir,
			"rng_seed" => rng_seed,
			"dynamics_model" => dynamics_model,
			"num_instances" => num_instances,
			"random_initial_state" => random_initial_state,
			"epsilon_schedule" => epsilon_schedule,
			"max_inner_iters_schedule" => max_inner_iters_schedule,
			"velocity_limit" => speed_component_limit,
			"perturbation_scale" => perturbation_scale,
		),
	)

end # end of demo()



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

function default_intersection_initial_states(dynamics_model::Symbol)
	if dynamics_model === :planar_double_integrator
		return [-6.0, -1.0, 2.0, 0.0], [1.0, -5.0, 0.0, 1.0]
	elseif dynamics_model === :unicycle
		# Unicycle state is (px, py, speed, heading), so pi/2 points upward.
		return [-6.0, -1.0, 0.0, 0.0], [1.0, -6.0, 1.3, pi / 2]
	end
	throw(ArgumentError("Unsupported dynamics_model $(dynamics_model)"))
end

function velocity_limit_constraints(x, dynamics_model::Symbol; velocity_limit = 1.5)
	if dynamics_model === :planar_double_integrator
		vx, vy = x[3], x[4]
		return vcat(vx + velocity_limit, -vx + velocity_limit, vy + velocity_limit, -vy + velocity_limit)
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
		end_idx = start_idx + primal_dimension - 1
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

function recover_complete_kkt_duals_for_fixed_primal(
	complete_kkt_system,
	fixed_primal,
	θ;
	ϵ₀,
	η₀,
	dual_init = nothing,
)
	F_symbolic = complete_kkt_system.F_symbolic
	z_symbolic = complete_kkt_system.z_symbolic
	primal_indices = collect(complete_kkt_system.primal_dims)
	nonprimal_indices = setdiff(collect(eachindex(z_symbolic)), primal_indices)
	nonprimal_symbols = z_symbolic[nonprimal_indices]

	substitution_dict = Dict{Any, Any}()
	for (sym, val) in zip(z_symbolic[primal_indices], fixed_primal)
		substitution_dict[sym] = val
	end
	backend = QuasiGOOP.SymbolicTracingUtils.SymbolicsBackend()
	θ_symbols =
		QuasiGOOP.SymbolicTracingUtils.make_variables(backend, :θ, length(θ))
	for (sym, val) in zip(θ_symbols, θ)
		substitution_dict[sym] = val
	end
	let
		ϵ_sym = only(QuasiGOOP.SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
		η_sym = only(QuasiGOOP.SymbolicTracingUtils.make_variables(backend, :η, 1))
		substitution_dict[ϵ_sym] = ϵ₀
		substitution_dict[η_sym] = η₀
	end

	F_symbolic_after_sub =
		Symbolics.substitute.(F_symbolic, Ref(substitution_dict))
	F_eval = first(
		Symbolics.build_function(
			F_symbolic_after_sub,
			nonprimal_symbols;
			expression = Val(false),
		),
	)
	dual_residual(u, p) = F_eval(u)

	u₀ = isnothing(dual_init) ? zeros(length(nonprimal_indices)) : copy(dual_init)
	prob = NonlinearLeastSquaresProblem(dual_residual, u₀)
	dual_sol = NonlinearSolve.solve(prob)

	z_recovered = zeros(length(z_symbolic))
	z_recovered[primal_indices] = fixed_primal
	z_recovered[nonprimal_indices] = dual_sol.u

	F_recovered = zeros(complete_kkt_system.kkt_dimension)
	complete_kkt_system.F!(F_recovered, z_recovered; θ, ϵ = ϵ₀, η = η₀)
	kkt_error_recovered = norm(F_recovered, 2)

	Dict(
		"status" => string(dual_sol.retcode),
		"kkt_error" => kkt_error_recovered,
		"fixed_primal" => fixed_primal,
		"dual_solution" => dual_sol.u,
		"z_recovered" => z_recovered,
	)
end

function prepare_intersection_output_dirs(run_id; debug)
	run_dir = if debug
		joinpath("data", "Intersection_open_loop", "debug", run_id)
	else
		joinpath("data", "Intersection_open_loop", "runs", run_id)
	end

	data_dir = joinpath(run_dir, "data")
	problem_data_dir = joinpath(data_dir, "problem")
	solution_data_dir = joinpath(problem_data_dir, "solution")
	plots_dir = joinpath(run_dir, "plots")
	trajectory_plots_dir = joinpath(plots_dir, "trajectories")
	convergence_plots_dir = joinpath(plots_dir, "convergence")
	velocity_plots_dir = joinpath(plots_dir, "velocities")
	control_plots_dir = joinpath(plots_dir, "controls")
	warmstart_plots_dir = joinpath(plots_dir, "warmstart")

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
	num_players,
	primal_dimension,
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
	player2_warmstart_primal =
		warmstart_solution[(primal_dimension+1):(num_players*primal_dimension)]
	(; warmstart_solution, player2_warmstart_primal)
end

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

function build_warmstart_solution(num_players, planning_horizon, dynamics, warmstart_x, warmstart_u)
	warmstart_solution = []
	for k in 1:num_players
		for i in 1:(planning_horizon-1)
			next_state = if applicable(dynamics, warmstart_x[k][i], warmstart_u[k][1], i)
				dynamics(warmstart_x[k][i], warmstart_u[k][1], i)
			else
				dynamics(warmstart_x[k][i], warmstart_u[k][1])
			end
			push!(warmstart_x[k], next_state)
			push!(warmstart_u[k], warmstart_u[k][1])
		end
		pop!(warmstart_u[k])
		push!(warmstart_u[k], [0.0, 0.0])

		warmstart_primals = mapreduce(vcat, 1:planning_horizon) do i
			vcat(warmstart_x[k][i], warmstart_u[k][i])
		end
		push!(warmstart_solution, warmstart_primals)
	end
	vcat(warmstart_solution...)
end

function dynamics_step(dynamics, x, u, k)
	if applicable(dynamics, x, u, k)
		return dynamics(x, u, k)
	end
	return dynamics(x, u)
end

function infer_planar_di_timestep(dynamics)
	x0 = [0.0, 0.0, 0.0, 0.0]
	u_unit_y = [0.0, 1.0]
	x1 = dynamics_step(dynamics, x0, u_unit_y, 1)
	dt = x1[4]
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
	for player in 1:length(warmstart_x)
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


end
