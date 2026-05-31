module Intersection

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
	unflatten_trajectory, state_dim, control_dim, control_bounds
using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, Distributions
using Random
using QuasiGOOP

include(joinpath(@__DIR__, "Plotting.jl"))

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
		return [-6.0, -1.0, 2.0, 0.0], [1.0, -5.0, 0.0, 2.0]
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
			# (; goal_position) = unflatten_parameters(θ[Block(1)]) # Player 1 θ[Block(i)] Ambuluance
			# goal_deviation = xs[end][1:2] .- goal_position
			sum(sum(u .^ 2) for u in us) #+ sum(goal_deviation .^ 2) # ||x - x\_goal||^ 2
		end, #for i in 1:num_players
		function (z, θ)
			(; xs, us) =
				unflatten_trajectory(z[Block(2)], state_dimension, control_dimension)
			# (; goal_position) = unflatten_parameters(θ[Block(2)]) # Player 2 θ[Block(i)] Car
			# goal_deviation = xs[end][1:2] .- goal_position
			sum(sum(u .^ 2) for u in us) #+ sum(goal_deviation .^ 2) # ||x - x\_goal||^ 2
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
					# velocity_constraints = vcat(vx + 1.5, -vx + 1.5, vy + 1.5, -vy + 1.5)
					# vcat(velocity_constraints,position_constraints)
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
			# # Keep center (yellow) line
			# function (z, θ)
			# 	(; xs, us) = unflatten_trajectory(
			# 		z[Block(1)],
			# 		state_dimension,
			# 		control_dimension,
			# 	)
			# 	mapreduce(vcat, 1:length(xs)) do k
			# 		py = xs[k][2]
			# 		-py # py ≤ 0.0
			# 	end
			# end,

			# # Drive under speed limit 
			# function (z, θ)
			# 	(; xs, us) = unflatten_trajectory(
			# 		z[Block(1)],
			# 		state_dimension,
			# 		control_dimension,
			# 	)
			# 	mapreduce(vcat, 1:length(xs)) do k
			# 		velocity_limit_constraints(xs[k], dynamics_model; velocity_limit)
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
				# [
				# 	goal_deviation .+ 0.01
				# 	-goal_deviation .+ 0.01
				# ]
				0*sum(goal_deviation .^ 2)
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
				sum(goal_deviation .^ 2)
			end,

			# # Keep center (yellow) line 
			# function (z, θ)
			# 	(; xs, us) = unflatten_trajectory(
			# 		z[Block(2)],
			# 		state_dimension,
			# 		control_dimension,
			# 	)
			# 	mapreduce(vcat, 1:length(xs)) do k
			# 		px = xs[k][1]
			# 		px + 0.0 # px ≥ 0.0
			# 	end
			# end,

			# # Drive under speed limit (highest priority)
			# function (z, θ)
			# 	(; xs, us) = unflatten_trajectory(
			# 		z[Block(2)],
			# 		state_dimension,
			# 		control_dimension,
			# 	)
			# 	mapreduce(vcat, 1:length(xs)) do k
			# 		velocity_limit_constraints(xs[k], dynamics_model; velocity_limit)
			# 	end
			# end,
		],

	]

	# Specify prioritized constraint [lowest priority, ..., highest priority]
	# is_prioritized_constraint = [[false, false, true, true], [false, true, true, false]]
	is_prioritized_constraint = [[false, false], [false, false]]
	preferences = [vcat(objectives[player], prioritized_preferences[player]) for player in 1:num_players]

	problem = QuasiGOOP.ParametricGOOP(
		dummy_primals, # x
		dummy_parameters; # θ
		preferences,
		is_prioritized_constraint,
		equality_constraints,
		inequality_constraints,
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
)
	Random.seed!(rng_seed)

	# Problem setup
	num_players = 2
	control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
	planning_horizon = 15
	collision_avoidance = 1.0
	speed_component_limit = 1.5
	num_instances = 1
	epsilon_schedule = [1.0*0.5^(j-1) for j in 1:3] # 1:11 
	max_inner_iters_schedule = fill(5000, length(epsilon_schedule))
	perturbation_scale = 0.3
	dynamics_model = :planar_double_integrator # :unicycle, :planar_double_integrator
	linesearch = :backtracking # :backtracking, :fraction_to_boundary
	goop_version = :reduced # :complete, :reduced, :quasi 
	dynamics = build_intersection_dynamics(dynamics_model; dt = 0.25, control_bounds)

	run_id = "run_debug3_swapped_$(dynamics_model)_$(goop_version)_system_4_pref_$(num_instances)_instances_horizon_$(planning_horizon)_linesearch_$(linesearch)"

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

	if goop_version === :complete
		GOOP_kkt_system = QuasiGOOP.generate_slacked_complete_kkt_system(problem)
	elseif goop_version === :reduced
		GOOP_kkt_system = QuasiGOOP.generate_slacked_reduced_kkt_system(problem)
	else
		GOOP_kkt_system = QuasiGOOP.generate_slacked_quasi_kkt_system(problem)
	end

	println("MCP Dimension: ", GOOP_kkt_system.kkt_dimension)
	println("variable Dimension: ", GOOP_kkt_system.variable_dimension)

	dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
	primal_dimension = dynamics_dimension * planning_horizon

	function get_receding_horizon_solution(θ; z₀, ϵ₀, max_inner_iters)
		convergence_log = Dict{String, Any}()
		elapsed_time = @elapsed begin
			(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				GOOP_kkt_system,
				θ;
				tol = 1e-5, # 1e-7 for backtracking, 1e-5 for fraction-to-boundary
				η₀ = 0.0, # 0.0 for backtracking, 0.001 for fraction-to-boundary
				ϵ₀, # 5.0
				max_inner_iters, # 20
				max_outer_iters = 2, # 2 for backtracking, 50 for fraction-to-boundary
				tightening_rate = 0.001, # 0.1
				loosening_rate = 0.05, # 0.5
				min_stepsize = 1e-20,
				z₀,
				verbose = verbose,
				convergence_log = convergence_log,
				linesearch,
			)
		end
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
			"solve_time_sec" => elapsed_time,
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
	base_initial_state1, base_initial_state2 =
		default_intersection_initial_states(dynamics_model)
	goal_position1 = [-1.0, -1.0]
	goal_position2 = [1.0, 6.0]

	run_dir = joinpath("data", "Intersection_open_loop", "runs", run_id)
	data_dir = joinpath(run_dir, "data")
	problem_data_dir = joinpath(data_dir, "problem")
	solution_data_dir = joinpath(problem_data_dir, "solution")
	histories_data_dir = joinpath(data_dir, "histories")
	plots_dir = joinpath(run_dir, "plots")
	trajectory_plots_dir = joinpath(plots_dir, "trajectories")
	convergence_plots_dir = joinpath(plots_dir, "convergence")
	velocity_plots_dir = joinpath(plots_dir, "velocities")
	for dir in (
		problem_data_dir,
		solution_data_dir,
		histories_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		velocity_plots_dir,
	)
		mkpath(dir)
	end

	instance_problem_data = Dict{String, Any}[]
	kkt_error_histories_per_eps = Dict(ϵ => Vector{Vector{Float64}}() for ϵ in epsilon_schedule)
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
		warmstart_u = [[[0.0, 0.0]], [[0.0, 0.0]]] # some constant control
		warmstart_solution = build_warmstart_solution(
			num_players,
			planning_horizon,
			dynamics,
			warmstart_x,
			warmstart_u,
		)

		epsilon_results = Pair{Float64, Any}[]
		stage_warmstart = warmstart_solution #warmstart_solution #load_object("stage_warmstart.jld2") #warmstart_solution
		solve_sequence_succeeded = true
		instance_total_solve_time_sec = 0.0
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
			instance_total_solve_time_sec += result.solution_dict["solve_time_sec"]

			# warmstart next solve with the previous solution's primal variables
			stage_warmstart = result.solution_dict["z"][1:(num_players*primal_dimension)]
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
		println(
			"instance $(solved_attempts) total solve time: $(round(instance_total_solve_time_sec; digits = 4)) sec",
		)
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
			CairoMakie.save(
				joinpath(
					velocity_plots_dir,
					"velocity_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
				),
				velocity_fig,
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
			"dynamics_model" => dynamics_model,
			"num_instances" => num_instances,
			"random_initial_state" => random_initial_state,
			"epsilon_schedule" => epsilon_schedule,
			"max_inner_iters_schedule" => max_inner_iters_schedule,
			"velocity_limit" => speed_component_limit,
			"perturbation_scale" => perturbation_scale,
		),
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


end
