module Intersection

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
	OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, ProgressMeter, Dates, Distributions
using Random
using QuasiGOOP

include(joinpath(@__DIR__, "IntersectionPlotting.jl"))

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
					vcat(vx + 1.5, -vx + 1.5, vy + 1.5, -vy + 1.5)
				end
			end,

			# Keep center (yellow) line
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(1)],
					state_dimension,
					control_dimension,
				)
				mapreduce(vcat, 1:length(xs)) do k
					px, py, vx, vy = xs[k]
					-py # py ≤ 0.0
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
					vcat(vx + 1.5, -vx + 1.5, vy + 1.5, -vy + 1.5)
				end
			end,

			# Keep center (yellow) line (highest priority)
			function (z, θ)
				(; xs, us) = unflatten_trajectory(
					z[Block(2)],
					state_dimension,
					control_dimension,
				)
				mapreduce(vcat, 1:length(xs)) do k
					px, py, vx, vy = xs[k]
					px + 0.0 # px ≥ 0.0
				end
			end,
		],
	]

	# Specify prioritized constraint [lowest priority, ..., highest priority]
	is_prioritized_constraint = [[false, true, true, true], [false, true, true, true]]
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

function demo(; map_end = 7, lane_width = 2, verbose = false, rng_seed = 1234)
	Random.seed!(rng_seed)

	# Problem setup
	num_players = 2
	control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
	dynamics = planar_double_integrator(; dt = 0.3, control_bounds) # x := (px, py, vx, vy) and u := (ax, ay).
	planning_horizon = 15
	collision_avoidance = 1.5
	num_instances = 10
	receding_horizon_steps = 0 # 0 for single-step only

	(; problem, flatten_parameters) = get_setup(
		num_players;
		dynamics,
		planning_horizon,
		collision_avoidance,
		map_end,
		lane_width,
	)

	dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
	primal_dimension = dynamics_dimension * planning_horizon

	# Run-time record
	runtime = Float64[]

	function get_receding_horizon_solution(θ; warmstart_solution)
		GOOP_kkt_system = QuasiGOOP.generate_slacked_kkt_system(problem)
		convergence_log = Dict{String,Any}()
		elapsed_time = @elapsed begin
			(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				GOOP_kkt_system,
				θ;
				tol = 1e-4, # 5e-3
				η₀ = 0.0, # 0.5
				ϵ₀ = 1.0, # 5.0
				max_inner_iters = 50, # 20
				max_outer_iters = 2, # 50
				tightening_rate = 0.001, # 0.1
				loosening_rate = 0.05, # 0.5
				min_stepsize = 1e-5,
				z₀ = warmstart_solution,
				verbose = true,
				convergence_log = convergence_log,
				linesearch = :backtracking, # :backtracking, :fraction_to_boundary
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
		)
		(; strategies, solution_dict)
	end

	obstacle_position = [0.25, 0.15] # placeholder
	base_initial_state1 = [-6.0, -1.0, 1.5, 0.0]
	base_initial_state2 = [1.0, -5.0, 0.0, 1.0]
	goal_position1 = [6.0, -1.0]
	goal_position2 = [1.0, 6.5]
	perturbation_scale = 0.1

	instance_problem_data = Dict{String,Any}[]
	log_kkt_error_histories = Vector{Float64}[]
	solved_attempts = 0
	total_attempts = 0

	while solved_attempts < num_instances
		total_attempts += 1 
		initial_state1 = base_initial_state1 .+ rand(Uniform(-perturbation_scale, perturbation_scale), state_dim(dynamics))
		initial_state2 = base_initial_state2 .+ rand(Uniform(-perturbation_scale, perturbation_scale), state_dim(dynamics))
		println(
			"solved $(solved_attempts)/$(num_instances), attempt $(total_attempts): "
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
		warmstart_u = [[[1.5, 0.0]], [[0.0, 3.0]]] # some constant control
		warmstart_solution = build_warmstart_solution(
			num_players,
			planning_horizon,
			dynamics,
			warmstart_x,
			warmstart_u,
		)

		result = try
			get_receding_horizon_solution(θ; warmstart_solution)
		catch err
			rethrow(err)
		end
		if isnothing(result)
			println(
				"attempt $(total_attempts): failed to converge, resampling.",
			)
			# TODO: Speed this up using GLMakie observable, toggling  only initial state
			continue
		end

		push!(instance_problem_data, Dict(
			"attempt_idx" => total_attempts,
			"initial_state1" => initial_state1,
			"goal_position1" => goal_position1,
			"initial_state2" => initial_state2,
			"goal_position2" => goal_position2,
		))

		strategies = result.strategies
		kkt_error_history = result.solution_dict["kkt_error_history"]
		push!(log_kkt_error_histories, log10.(max.(kkt_error_history, eps(Float64))))

		figure, _ = plot_intersection_trajectories(
			;
			map_end,
			lane_width,
			strategy = Observable(strategies),
			θ1 = Observable(θ1),
			θ2 = Observable(θ2),
			goal_position1 = Observable(goal_position1),
			goal_position2 = Observable(goal_position2),
		)


		JLD2.save_object(
		"./data/Intersection_closed_loop/problem/problem_data_$(solved_attempts).jld2",
		instance_problem_data,
		)

		solved_attempts += 1
		GLMakie.save(
			"data/Intersection_closed_loop/trajectory_instance_$(solved_attempts).png",
			figure,
		)
	end

	aggregate_convergence_fig, _ = plot_convergence_plot_aggregate(
		;
		log_kkt_error_histories,
	)
	GLMakie.save(
		"./data/Intersection_closed_loop/GOOP_plots/convergence_aggregate.png",
		aggregate_convergence_fig,
	)

	JLD2.save_object(
		"./data/Intersection_closed_loop/problem/log_kkt_error_histories.jld2",
		log_kkt_error_histories,
	)

	return (; log_kkt_error_histories)
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
