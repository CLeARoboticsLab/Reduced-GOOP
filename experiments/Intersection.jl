module Intersection

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
	OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
using BlockArrays
using JLD2, ProgressMeter, Dates

using QuasiGOOP

include("intersection_plotting.jl")

function get_setup(
	num_players;
	dynamics = UnicycleDynamics,
	planning_horizon = 5,
	collision_avoidance = 1.0,
	map_end = 7,
	lane_width = 2,
	relaxation_mode = :standard,
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

function demo(; map_end = 7, lane_width = 2, verbose = false)
	# Algorithm setting
	# σ = 20
	# κ = 0.6
	# max_iterations = 10
	# tolerance = 5e-2
	relaxation_mode = :standard

	num_players = 2
	control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
	dynamics = planar_double_integrator(; dt = 0.3, control_bounds) # x := (px, py, vx, vy) and u := (ax, ay).
	planning_horizon = 10
	collision_avoidance = 1.5

	(; problem, flatten_parameters) = get_setup(
		num_players;
		dynamics,
		planning_horizon,
		collision_avoidance,
		map_end,
		lane_width,
		relaxation_mode,
	)

	dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
	primal_dimension = dynamics_dimension * planning_horizon

	# Run-time record
	runtime = Float64[]

	function get_receding_horizon_solution(θ; warmstart_solution)
		GOOP_kkt_system = QuasiGOOP.generate_slacked_kkt_system(problem)
		elapsed_time = @elapsed begin
			(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				GOOP_kkt_system,
				θ;
				tol = 5e-3, # 5e-3
				η₀ = 0.5, # 0.5
				ϵ₀ = 5.0, # 5.0
				max_inner_iters = 30, # 20
				max_outer_iters = 50, # 50
				tightening_rate = 0.01, # 0.1
				loosening_rate = 0.05, # 0.5
				min_stepsize = 1e-5,
				z₀ = warmstart_solution,
				verbose = true,
			)
		end
		push!(runtime, elapsed_time)
		if status == :failed
			error("GOOP solver failed to converge.")
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
		)
		file_name = "intersection_"*string(now())*".jld2"
		JLD2.save_object(
			"./data/Intersection_closed_loop/GOOP_solution/$(file_name)",
			solution_dict,
		)
		strategies
	end

	obstacle_position = Observable([0.25, 0.15]) # placeholder
	# Player 1
	initial_state1 = Observable([-6.0, -1.0, 1.5, 0.0])
	# initial_state1 = Observable([-4.16585, -1.21562, 2.16838, -0.431245])
	goal_position1 = Observable([6.0, -1.0])
	θ1 = GLMakie.@lift flatten_parameters(; # θ is a flat (column) vector of parameters
		initial_state = $initial_state1,
		goal_position = $goal_position1,
		obstacle_position = $obstacle_position,
	)

	# Player 2
	initial_state2 = Observable([1.0, -5.0, 0.0, 1.0])
	# initial_state2 = Observable([0.808618, -3.75084, -0.382764, 1.49955])
	goal_position2 = Observable([1.0, 6.0])
	θ2 = GLMakie.@lift flatten_parameters(;
		initial_state = $initial_state2,
		goal_position = $goal_position2,
		obstacle_position = $obstacle_position,
	)
	θ = GLMakie.@lift [$θ1..., $θ2...]

	println("initial_state1:", initial_state1)
	println("goal_position1:", goal_position1)
	println("initial_state2:", initial_state2)
	println("goal_position2:", goal_position2)

	problem_data = Dict(
		"initial_state1" => initial_state1[],
		"goal_position1" => goal_position1[],
		"initial_state2" => initial_state2[],
		"goal_position2" => goal_position2[],
	)
	JLD2.save_object(
		"./data/Intersection_closed_loop/problem/problem_data.jld2",
		problem_data,
	)

	# Warmstart solution
	warmstart_x = [[initial_state1[]], [initial_state2[]]]
	warmstart_u = [[[2.0, 0.0]], [[0.0, 4.0]]] # some constant control
	warmstart_solution = build_warmstart_solution(num_players, planning_horizon, dynamics, warmstart_x, warmstart_u)
	# warmstart_solution = nothing 

	strategy = GLMakie.@lift let
		result = get_receding_horizon_solution($θ; warmstart_solution)
		warmstart_solution = nothing # TODO
		result
	end

	figure, ax = plot_intersection_trajectories(
		;
		map_end,
		lane_width,
		strategy,
		θ1,
		θ2,
		goal_position1,
		goal_position2,
	)

	# Save img 
	# Main.@infiltrate
	GLMakie.save("data/Intersection_closed_loop/trajectory.png", figure)
	Main.@infiltrate

	# closed_loop + receding horizon demo
	time_step = 1
	while time_step < 1 #15
		println("time_step: ", time_step)
		GLMakie.save("data/Intersection_closed_loop/trajectory$(time_step-1).png", figure)
		# Update the positions of the vehicles
		println("Update initial state1")
		θ1.val[1:state_dim(dynamics)] = first(strategy[]).xs[begin+1] # Asynchronous update: mutate p1's initial state without triggering others
		println("Update initial state2")
		initial_state2[] = strategy[][2].xs[begin+1]
		# Main.@infiltrate
		time_step += 1
	end

	# Store speed data for Intersection
	horizontal_speed_data = Vector{Vector{Float64}}[]
	vertical_speed_data = Vector{Vector{Float64}}[]
	openloop_distance1 = Vector{Float64}[]

	# Store openloop speed data
	push!(horizontal_speed_data, [vcat(strategy[][1].xs...)[3:4:end], vcat(strategy[][2].xs...)[3:4:end]])#, vcat(strategy[3].xs...)[3:4:end]])
	push!(vertical_speed_data, [vcat(strategy[][1].xs...)[4:4:end], vcat(strategy[][2].xs...)[4:4:end]])#, vcat(strategy[3].xs...)[4:4:end]])

	# Store openloop distance data
	push!(openloop_distance1, [sqrt(sum((strategy[][1].xs[k][1:2] - strategy[][2].xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])

	# Visualize horizontal speed
	T = 1
	fig = GLMakie.Figure() # limits = (nothing, (nothing, 0.7))
	ax2 = GLMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "speed", title = "Horizontal Speed")
	GLMakie.scatterlines!(ax2, 0:(planning_horizon-1), horizontal_speed_data[T][1], label = "Vehicle 1", color = :blue)
	GLMakie.scatterlines!(ax2, 0:(planning_horizon-1), horizontal_speed_data[T][2], label = "Vehicle 2", color = :red)
	GLMakie.lines!(ax2, 0:(planning_horizon-1), [1.5 for _ in 0:(planning_horizon-1)], color = :black, linestyle = :dash)
	fig[2, 1:2] = GLMakie.Legend(fig, ax2, framevisible = false, orientation = :horizontal)

	# Visualize vertical speed
	ax3 = GLMakie.Axis(fig[1, 2]; xlabel = "time step", ylabel = "speed", title = "Vertical Speed")
	GLMakie.scatterlines!(ax3, 0:(planning_horizon-1), vertical_speed_data[T][1], label = "Vehicle 1", color = :blue)
	GLMakie.scatterlines!(ax3, 0:(planning_horizon-1), vertical_speed_data[T][2], label = "Vehicle 2", color = :red)
	GLMakie.lines!(ax3, 0:(planning_horizon-1), [1.5 for _ in 0:(planning_horizon-1)], color = :black, linestyle = :dash)

	GLMakie.save("./data/Intersection_closed_loop/GOOP_plots/speed.png", fig)

	# Visualize distance bw vehicles , limits = (nothing, (collision_avoidance-0.05, 0.4)) 
	fig = GLMakie.Figure() # limits = (nothing, (nothing, 0.7))
	ax4 = GLMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "distance", title = "Distance bw vehicles")
	GLMakie.scatterlines!(ax4, 0:(planning_horizon-1), openloop_distance1[T], label = "B/w Agent 1 & Agent 2", color = :black, marker = :star5, markersize = 20)
	GLMakie.lines!(ax4, 0:(planning_horizon-1), [1.0 for _ in 0:(planning_horizon-1)], color = :black, linestyle = :dash)
	fig[2, 1] = GLMakie.Legend(fig, ax4, framevisible = false, orientation = :horizontal)

	GLMakie.save("./data/Intersection_closed_loop/GOOP_plots/" * "distance_bw_vehicles.png", fig)

	# Store distance from center yellow line 
	distance_from_center = Vector{Vector{Float64}}[]
	push!(distance_from_center, [vcat(strategy[][1].xs...)[2:4:end], vcat(-strategy[][2].xs...)[1:4:end]])#, vcat(strategy[3].xs...)[2:4:end]])

	# Visualize distance from center yellow line
	fig = GLMakie.Figure() # limits = (nothing, (nothing, 0.7))
	ax5 = GLMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "distance", title = "Position from center yellow line")
	GLMakie.scatterlines!(ax5, 0:(planning_horizon-1), distance_from_center[T][1], label = "Vehicle 1", color = :blue)
	GLMakie.scatterlines!(ax5, 0:(planning_horizon-1), distance_from_center[T][2], label = "Vehicle 2", color = :red)
	GLMakie.lines!(ax5, 0:(planning_horizon-1), [0.0 for _ in 0:(planning_horizon-1)], color = :black, linestyle = :dash)
	fig[2, 1] = GLMakie.Legend(fig, ax5, framevisible = false, orientation = :horizontal)

	GLMakie.save("./data/Intersection_closed_loop/GOOP_plots/" * "position_from_center.png", fig)

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
