module Intersection

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
	OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, ProgressMeter, Dates, Distributions
using Random
using QuasiGOOP

include(joinpath(@__DIR__, "Plotting.jl"))

function get_setup(
	num_players;
	dynamics = UnicycleDynamics,
	planning_horizon = 5,
	collision_avoidance = 1.0,
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
			# (; goal_position) = unflatten_parameters(θ[Block(1)]) # Player 1 θ[Block(i)] Ambuluance
			# goal_deviation = xs[end][1:2] .- goal_position
			0.5*sum(sum(u .^ 2) for u in us) #+ sum(goal_deviation .^ 2) # ||x - x\_goal||^ 2
		end, #for i in 1:num_players
		function (z, θ)
			(; xs, us) =
				unflatten_trajectory(z[Block(2)], state_dimension, control_dimension)
			# (; goal_position) = unflatten_parameters(θ[Block(2)]) # Player 2 θ[Block(i)] Car
			# goal_deviation = xs[end][1:2] .- goal_position
			0.5*sum(sum(u .^ 2) for u in us) #+ sum(goal_deviation .^ 2) # ||x - x\_goal||^ 2
		end,
	]

	equality_constraints = [
		function (z, θ)
			(; xs, us) =
				unflatten_trajectory(z[Block(i)], state_dimension, control_dimension)
			(; initial_state) = unflatten_parameters(θ[Block(i)]) # Player i θ[Block(i)]
			initial_state_constraint = xs[1] - initial_state
			dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
				xs[k] - dynamics(xs[k-1], us[k-1], k)
			end
			vcat(initial_state_constraint, dynamics_constraints)
		end for i in 1:num_players
	]

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
					px, py, vx, vy = xs[k]
					position_constraints = vcat(
						px + map_end,
						-px + map_end,
						py + lane_width,
						-py + lane_width,
					) # -7 ≤ pₓ ≤ 7, -2 ≤ py ≤ 2
					vcat(position_constraints)
				end,
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
					px, py, vx, vy = xs[k]
					position_constraints = vcat(
						px + lane_width,
						-px + lane_width,
						py + map_end,
						-py + map_end,
					) # -2 ≤ pₓ ≤ 2, -7 ≤ py ≤ 7
					# velocity_constraints = vcat(vx + 1.5, -vx + 1.5, vy + 1.5, -vy + 1.5)
					# vcat(velocity_constraints,position_constraints)
					vcat(position_constraints)
				end,
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
					px, py, vx, vy = xs[k]
					vcat(vx + 2.0, -vx + 2.0, vy + 2.0, -vy + 2.0)
				end
			end,

			# # Keep center (yellow) line
			# function (z, θ)
			# 	(; xs, us) = unflatten_trajectory(
			# 		z[Block(1)],
			# 		state_dimension,
			# 		control_dimension,
			# 	)
			# 	mapreduce(vcat, 1:length(xs)) do k
			# 		px, py, vx, vy = xs[k]
			# 		-py # py ≤ 0.0
			# 	end
			# end,

			# reach the goal. (highest priority)
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(1)],
					state_dimension,
					control_dimension,
				)
				(; goal_position) = unflatten_parameters(θ[Block(1)]) # Player 1 θ[Block(i)] Ambuluance
				goal_deviation = xs[end][1:2] .- goal_position
				[
					goal_deviation .+ 0.01
					-goal_deviation .+ 0.01
				]
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
				[
					goal_deviation .+ 0.01
					-goal_deviation .+ 0.01
				]
			end,

			# Drive under speed limit 
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(2)],
					state_dimension,
					control_dimension,
				)
				mapreduce(vcat, 1:length(xs)) do k
					px, py, vx, vy = xs[k]
					vcat(vx + 2.0, -vx + 2.0, vy + 2.0, -vy + 2.0)
				end
			end,

			# # Keep center (yellow) line (highest priority)
			# function (z, θ)
			# 	(; xs, us) = unflatten_trajectory(
			# 		z[Block(2)],
			# 		state_dimension,
			# 		control_dimension,
			# 	)
			# 	mapreduce(vcat, 1:length(xs)) do k
			# 		px, py, vx, vy = xs[k]
			# 		px + 0.0 # px ≥ 0.0
			# 	end
			# end,
		],
	]

	# Specify prioritized constraint [lowest priority, ..., highest priority]
	is_prioritized_constraint = [[false, true, true], [false, true, true]]
	preferences = [vcat(objectives[player], prioritized_preferences[player]) for player in 1:num_players]

	# Shared constraints
	function shared_inequality_constraint(z, θ)
		trajectories = map(
			i ->
				unflatten_trajectory(z[Block(i)], state_dimension, control_dimension),
			1:num_players,
		)
		xs = map(trajectory -> trajectory.xs, trajectories)
		@assert length(xs) == num_players
		# Avoid collision between 2 players
		mapreduce(vcat, 2:length(xs[1])) do k
			[sum((xs[1][k][1:2] - xs[2][k][1:2]) .^ 2) - collision_avoidance^2]
		end
	end

	problem = QuasiGOOP.ParametricGOOP(
		dummy_primals, # x
		dummy_parameters; # θ
		preferences,
		is_prioritized_constraint,
		equality_constraints,
		inequality_constraints,
		shared_equality_constraint = nothing,
		shared_inequality_constraint,
	)

	(; problem, flatten_parameters)
end

function demo(; map_end = 7, lane_width = 2, verbose = false, rng_seed = 123)
	Random.seed!(rng_seed)

	# Problem setup
	num_players = 2
	control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
	dynamics = planar_double_integrator(; dt = 0.3, control_bounds) # x := (px, py, vx, vy) and u := (ax, ay).
	planning_horizon = 15
	collision_avoidance = 1.5
	num_instances = 1
	epsilon_schedule = [1.0*0.5^(j-1) for j in 1:15]
	max_inner_iters_schedule = fill(200, length(epsilon_schedule))
	perturbation_scale = 0.2
	linesearch = :backtracking # :backtracking, :fraction_to_boundary
	goop_version = :reduced # :complete, :reduced 
	receding_horizon_steps = 0 # 0 for single-step only

	run_id = "run_6_$(goop_version)_system_3_pref_$(num_instances)_instances"

	(; problem, flatten_parameters) = get_setup(
		num_players;
		dynamics,
		planning_horizon,
		collision_avoidance,
		map_end,
		lane_width,
	)

	if goop_version === :complete
		GOOP_kkt_system = QuasiGOOP.generate_slacked_complete_kkt_system(problem)
	else
		GOOP_kkt_system = QuasiGOOP.generate_slacked_reduced_kkt_system(problem)
	end
	println("MCP Dimension: ", GOOP_kkt_system.kkt_dimension)
	println("variable Dimension: ", GOOP_kkt_system.variable_dimension)

	dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
	primal_dimension = dynamics_dimension * planning_horizon

	# Run-time record
	runtime = Float64[]

	function get_receding_horizon_solution(θ; z₀, ϵ₀, max_inner_iters)
		convergence_log = Dict{String, Any}()
		elapsed_time = @elapsed begin
			(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				GOOP_kkt_system,
				θ;
				tol = 1e-7, # 5e-3
				η₀ = 0.0, # 0.5
				ϵ₀, # 5.0
				max_inner_iters, # 20
				max_outer_iters = 2, # 50
				tightening_rate = 0.001, # 0.1
				loosening_rate = 0.05, # 0.5
				min_stepsize = 1e-20,
				z₀,
				verbose = true,
				convergence_log = convergence_log,
				linesearch, # :backtracking, :fraction_to_boundary
			)
		end
		push!(runtime, elapsed_time)
		if status == :failed
			return nothing
		end

		strategies = mapreduce(vcat, 1:num_players) do i
			start_idx = primal_dimension * (i-1) + 1
			end_idx = start_idx + primal_dimension - 1
			x_segment = x[start_idx:end_idx]   # same as sol.primals[i][1:primal_dimension]

			unflatten_trajectory(
				x_segment,
				state_dim(dynamics),
				control_dim(dynamics),
			)
		end
		# Save solution
		solution_dict = Dict(
			"strategies" => strategies,
			"z" => z,
			"x" => x,
			"s" => s,
			"kkt_error" => kkt_error,
			"ϵ" => ϵ,
			"outer_iters" => outer_iters,
			"total_iters" => total_iters,
			"kkt_error_history" => get(convergence_log, "kkt_error_history", Float64[]),
			"total_iteration_history" => get(convergence_log, "total_iteration_history", Int[]),
			"outer_iteration_history" => get(convergence_log, "outer_iteration_history", Int[]),
			"inner_iteration_history" => get(convergence_log, "inner_iteration_history", Int[]),
			"outer_end_total_iterations" => get(convergence_log, "outer_end_total_iterations", Int[]),
			"outer_end_trace_indices" => get(convergence_log, "outer_end_trace_indices", Int[]),
			"per iteration runtime" => mean(runtime),
		)
		(; strategies, solution_dict)
	end

	obstacle_position = [0.25, 0.15] # placeholder
	base_initial_state1 = [-6.0, -1.0, 1.5, 0.0]
	base_initial_state2 = [1.0, -5.0, 0.0, 1.0]
	goal_position1 = [6.0, -1.0]
	goal_position2 = [1.0, 6.0]

	run_dir = joinpath("data", "Intersection_open_loop", "runs", run_id)
	data_dir = joinpath(run_dir, "data")
	problem_data_dir = joinpath(data_dir, "problem")
	solution_data_dir = joinpath(problem_data_dir, "solution")
	histories_data_dir = joinpath(data_dir, "histories")
	plots_dir = joinpath(run_dir, "plots")
	trajectory_plots_dir = joinpath(plots_dir, "trajectories")
	convergence_plots_dir = joinpath(plots_dir, "convergence")
	for dir in (
		problem_data_dir,
		solution_data_dir,
		histories_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
	)
		mkpath(dir)
	end

	instance_problem_data = Dict{String, Any}[]
	kkt_error_histories_per_eps = Dict(ϵ => Vector{Vector{Float64}}() for ϵ in epsilon_schedule)
	solved_attempts = 0
	total_attempts = 0

	while solved_attempts < num_instances
		total_attempts += 1
		initial_state1 = base_initial_state1 .+ rand(Uniform(-perturbation_scale, perturbation_scale), state_dim(dynamics))
		initial_state2 = base_initial_state2 .+ rand(Uniform(-perturbation_scale, perturbation_scale), state_dim(dynamics))
		println(
			"solved $(solved_attempts)/$(num_instances), attempt $(total_attempts), goop version $(goop_version): ",
		)
		println("initial_state1:", initial_state1)
		println("goal_position1:", goal_position1)
		println("initial_state2:", initial_state2)
		println("goal_position2:", goal_position2)

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
		θ = [θ1..., θ2...]

		warmstart_x = [[initial_state1], [initial_state2]]
		warmstart_u = [[[2.0, 0.0]], [[0.0, 2.0]]] # some constant control
		warmstart_solution = build_warmstart_solution(
			num_players,
			planning_horizon,
			dynamics,
			warmstart_x,
			warmstart_u,
		)

		epsilon_results = Pair{Float64, Any}[]
		stage_warmstart = warmstart_solution
		solve_sequence_succeeded = true
		for (ϵ₀, max_inner_iters) in zip(epsilon_schedule, max_inner_iters_schedule)
			result = try
				get_receding_horizon_solution(θ; z₀ = stage_warmstart, ϵ₀, max_inner_iters)
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

			# warmstart next solve with the previous solution's primal variables
			stage_warmstart = result.solution_dict["z"][1:(num_players*primal_dimension)]
		end
		if !solve_sequence_succeeded
			# TODO: Speed this up using GLMakie observable, toggling only initial state
			continue
		end

		push!(instance_problem_data, Dict(
			"attempt_idx" => total_attempts,
			"initial_state1" => initial_state1,
			"goal_position1" => goal_position1,
			"initial_state2" => initial_state2,
			"goal_position2" => goal_position2,
		))

		solved_attempts += 1
		JLD2.save_object(
			joinpath(problem_data_dir, "problem_data_instance_$(solved_attempts).jld2"),
			instance_problem_data,
		)
		for (ϵ₀, result) in epsilon_results
			JLD2.save_object(
				joinpath(
					solution_data_dir,
					"solution_dict_instance_$(solved_attempts)_eps$(ϵ₀).jld2",
				),
				result.solution_dict,
			)

			# Compute and save distance between trajectories for the two players
			xs1 = result.solution_dict["strategies"][1].xs
			xs2 = result.solution_dict["strategies"][2].xs
			trajectory_len = min(length(xs1), length(xs2))
			trajectory_distance = [
				sqrt(sum((xs1[k][1:2] - xs2[k][1:2]) .^ 2)) for k in 1:trajectory_len
			]
			JLD2.save_object(
				joinpath(
					histories_data_dir,
					"trajectory_distance_instance_$(solved_attempts)_eps$(ϵ₀).jld2",
				),
				trajectory_distance,
			)

			push!(kkt_error_histories_per_eps[ϵ₀], log10.(result.solution_dict["kkt_error_history"]))
			convergence_fig, _ = plot_convergence_plot(
				;
				kkt_error_history = log10.(result.solution_dict["kkt_error_history"]),
				total_iteration_history = result.solution_dict["total_iteration_history"],
				outer_end_total_iterations = result.solution_dict["outer_end_total_iterations"],
				outer_end_trace_indices = result.solution_dict["outer_end_trace_indices"],
			)
			trajectory_fig, _ = plot_intersection_trajectories(
				;
				map_end,
				lane_width,
				strategy = Observable(result.strategies),
				θ1 = Observable(θ1),
				θ2 = Observable(θ2),
				goal_position1 = Observable(goal_position1),
				goal_position2 = Observable(goal_position2),
			)
			CairoMakie.save(
				joinpath(
					convergence_plots_dir,
					"convergence_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
				),
				convergence_fig,
			)
			CairoMakie.save(
				joinpath(
					trajectory_plots_dir,
					"trajectory_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
				),
				trajectory_fig,
			)
		end
	end

	for ϵ₀ in epsilon_schedule
		aggregate_convergence_fig, _ = plot_convergence_plot_aggregate(
			;
			kkt_error_histories = kkt_error_histories_per_eps[ϵ₀],
		)
		CairoMakie.save(
			joinpath(convergence_plots_dir, "convergence_aggregate_eps$(ϵ₀).pdf"),
			aggregate_convergence_fig,
		)
	end

	JLD2.save_object(
		joinpath(histories_data_dir, "kkt_error_histories_per_eps.jld2"),
		kkt_error_histories_per_eps,
	)
	JLD2.save_object(
		joinpath(problem_data_dir, "run_metadata.jld2"),
		Dict(
			"run_id" => run_id,
			"rng_seed" => rng_seed,
			"num_instances" => num_instances,
			"epsilon_schedule" => epsilon_schedule,
			"max_inner_iters_schedule" => max_inner_iters_schedule,
			"perturbation_scale" => perturbation_scale,
		),
	)

end

function build_warmstart_solution(num_players, planning_horizon, dynamics, warmstart_x, warmstart_u)
	warmstart_solution = []
	for k in 1:num_players
		for i in 1:(planning_horizon-1)
			push!(warmstart_x[k], dynamics(warmstart_x[k][i], warmstart_u[k][1]))
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


end
