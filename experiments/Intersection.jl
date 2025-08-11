module Intersection

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
using BlockArrays
using JLD2, ProgressMeter, Dates

using QuasiGOOP

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
                unflatten_trajectory(z[Block(i)], state_dimension, control_dimension)
            0.5*sum(sum(u .^ 2) for u in us)
        end for i in 1:num_players
    ]

    equality_constraints = [ # private: z[Block(1)] are original private primals
        function (z, θ)
            (; xs, us) =
                unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
            (; initial_state) = unflatten_parameters(θ[Block(i)])
            initial_state_constraint = xs[1] - initial_state
            dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
                xs[k] - dynamics(xs[k - 1], us[k - 1], k)
            end
            vcat(initial_state_constraint, dynamics_constraints)
        end for i in 1:num_players
    ]
    equality_dimensions = [
        length(equality_constraints[i](dummy_primals, dummy_parameters)) for
        i in 1:num_players
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
                unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
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
    inequality_dimensions = [
        length(inequality_constraints[i](dummy_primals, dummy_parameters)) for
        i in 1:num_players
    ]

    prioritized_preferences = [
        [
            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(
                    z[Block(1)],
                    state_dimension,
                    control_dimension,
                )
                (; goal_position) = unflatten_parameters(θ[Block(1)]) # Player 1 θ[Block(i)]
                goal_deviation = xs[end][1:2] .- goal_position
                [
                    goal_deviation .+ 0.01
                    -goal_deviation .+ 0.01
                ]
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
                    -py + 0.0 # py ≤ 0.0
                end
            end,

            # Drive under speed limit 
            function (z, θ)
                (; xs, us) = unflatten_trajectory(
                    z[Block(1)],
                    state_dimension,
                    control_dimension,
                )
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    velocity_constraints =
                        vcat(vx + 1.5, -vx + 1.5, vy + 1.5, -vy + 1.5)
                    vcat(velocity_constraints)
                end
            end,
        ],
        [
            # Keep center (yellow) line
            function (z, θ)
                (; xs, us) = unflatten_trajectory(
                    z[Block(1)],
                    state_dimension,
                    control_dimension,
                )
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    px + 0.0 # px ≥ 0.0
                end
            end,

            # Drive under speed limit 
            function (z, θ)
                (; xs, us) = unflatten_trajectory(
                    z[Block(1)],
                    state_dimension,
                    control_dimension,
                )
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    velocity_constraints =
                        vcat(vx + 1.5, -vx + 1.5, vy + 1.5, -vy + 1.5)
                    vcat(velocity_constraints)
                end
            end,

            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(
                    z[Block(1)],
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
        ],
    ]

    # Specify prioritized constraint
    is_prioritized_constraint = [[true, true, true], [true, true, true]] #, [true, true]]

    # Construct a preferences object
    preferences = ordered_preferences(prioritized_preferences, is_prioritized_constraint)

    # Shared constraints
    function shared_equality_constraints(z, θ)
        [0]
    end

    function shared_inequality_constraints(z, θ)
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

    problem = ParametricQuasiGOOP(;
        objectives,
        equality_constraints,
        inequality_constraints,
        preferences,
        shared_equality_constraints,
        shared_inequality_constraints,
        primal_dimensions,
        parameter_dimensions,
        equality_dimensions,
        inequality_dimensions,
    )

    (; problem, flatten_parameters)
end

function demo(; map_end = 7, lane_width = 2, verbose = false)
    # Algorithm setting
    σ = 20
    κ = 0.6
    max_iterations = 10
    tolerance = 5e-2
    relaxation_mode = :standard

    num_players = 2
    control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
    dynamics = planar_double_integrator(; dt = 1.0, control_bounds) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 7
    collision_avoidance = 1.3

    (; problem, flatten_parameters) = get_setup(
        num_players;
        dynamics,
        planning_horizon,
        collision_avoidance,
        map_end,
        lane_width,
        relaxation_mode,
    )

    warmstart_solution = nothing

    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    # Run-time record
    runtime = Float64[]

    function get_receding_horizon_solution(θ; warmstart_solution)
        # Measure run time
        # elapsed_time = @elapsed begin
        #     (; relaxation, solution, residual) = solve_relaxed_pop_game(
        #         problem,
        #         warmstart_solution,
        #         θ;
        #         ϵ,
        #         κ,
        #         max_iterations,
        #         tolerance,
        #         verbose,
        #     )
        # end
        elapsed_time = @elapsed begin
            # Main.@infiltrate
            sol = solve(problem, θ, σ, κ, max_iterations, tolerance; verbose)
        end
        push!(runtime, elapsed_time)
        Main.@infiltrate

        # Choose the solution with best complementarity residual
        # min_residual_idx = argmin(residual)
        # println("residual: ", residual[min_residual_idx])
        # println("relaxation: ", relaxation[min_residual_idx])
        # println("slacks: ", solution[min_residual_idx].slacks)
        strategies = mapreduce(vcat, 1:num_players) do i
            unflatten_trajectory(
                sol.primals[i][1:primal_dimension],
                state_dim(dynamics),
                control_dim(dynamics),
            )
        end
        # Save solution
        solution_dict = Dict(
            # "residual" => residual[min_residual_idx],
            # "relaxation" => relaxation[min_residual_idx],
            # "slacks" => solution[min_residual_idx].slacks,
            "strategy1" => strategies[1],
            "strategy2" => strategies[2],
            "primals" => sol.primals,
        )
        file_name = "intersection_"*string(now())*".jld2"
        JLD2.save_object(
            "./data/Intersection_closed_loop/GOOP_solution/$(file_name)",
            solution_dict,
        )
        # JLD2.save_object("./data/Intersection_closed_loop/GOOP_solution/intersection.jld2", solution_dict)

        (; strategies, sol)
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

    strategy = GLMakie.@lift let
        result = get_receding_horizon_solution($θ; warmstart_solution)
        warmstart_solution = nothing
        result.strategies
    end

    # Plot using GLMakie
    figure = GLMakie.Figure()
    ax = GLMakie.Axis(
        figure[1, 1];
        aspect = 1,
        xgridvisible = false,
        ygridvisible = false,
        backgroundcolor = :lightgreen,
    )
    GLMakie.hidedecorations!(ax)
    GLMakie.hidespines!(ax)

    # Draw intersection
    offset = 0.2

    vertical_road_background = GLMakie.Polygon(
        GLMakie.Point2f[
            (-lane_width-offset, -map_end),
            (lane_width+offset, -map_end),
            (lane_width+offset, map_end),
            (-lane_width-offset, map_end),
        ],
    )
    GLMakie.poly!(vertical_road_background, color = :white)
    GLMakie.lines!(
        ax,
        [-lane_width-offset, -lane_width-offset],
        [-map_end, -lane_width],
        color = :black,
        linewidth = 1,
    )
    GLMakie.lines!(
        ax,
        [-lane_width-offset, -lane_width-offset],
        [map_end, lane_width],
        color = :black,
        linewidth = 1,
    )
    GLMakie.lines!(
        ax,
        [lane_width+offset, lane_width+offset],
        [-map_end, -lane_width],
        color = :black,
        linewidth = 1,
    )
    GLMakie.lines!(
        ax,
        [lane_width+offset, lane_width+offset],
        [map_end, lane_width],
        color = :black,
        linewidth = 1,
    )

    horizontal_road_background = GLMakie.Polygon(
        GLMakie.Point2f[
            (-map_end, -lane_width-offset),
            (map_end, -lane_width-offset),
            (map_end, lane_width+offset),
            (-map_end, lane_width+offset),
        ],
    )
    GLMakie.poly!(horizontal_road_background, color = :white)
    GLMakie.lines!(
        ax,
        [-lane_width-offset, -map_end],
        [-lane_width-offset, -lane_width-offset],
        color = :black,
        linewidth = 1,
    )
    GLMakie.lines!(
        ax,
        [-lane_width-offset, -map_end],
        [lane_width+offset, lane_width+offset],
        color = :black,
        linewidth = 1,
    )
    GLMakie.lines!(
        ax,
        [lane_width+offset, map_end],
        [lane_width+offset, lane_width+offset],
        color = :black,
        linewidth = 1,
    )
    GLMakie.lines!(
        ax,
        [lane_width+offset, map_end],
        [-lane_width-offset, -lane_width-offset],
        color = :black,
        linewidth = 1,
    )

    vertical_road = GLMakie.Polygon(
        GLMakie.Point2f[
            (-lane_width, -map_end),
            (lane_width, -map_end),
            (lane_width, map_end),
            (-lane_width, map_end),
        ],
    )
    GLMakie.poly!(vertical_road, color = :gray)
    horizontal_road = GLMakie.Polygon(
        GLMakie.Point2f[
            (-map_end, -lane_width),
            (map_end, -lane_width),
            (map_end, lane_width),
            (-map_end, lane_width),
        ],
    )
    GLMakie.poly!(horizontal_road, color = :gray)

    # Lane markings (dashed center lines)
    GLMakie.lines!(ax, [-lane_width, -map_end], [0, 0], color = :yellow, linewidth = 2)
    GLMakie.lines!(ax, [-lane_width, -map_end], [0, 0], color = :yellow, linewidth = 2)
    GLMakie.lines!(ax, [lane_width, map_end], [0, 0], color = :yellow, linewidth = 2)
    GLMakie.lines!(ax, [lane_width, map_end], [0, 0], color = :yellow, linewidth = 2)
    GLMakie.lines!(ax, [0, 0], [-lane_width, -map_end], color = :yellow, linewidth = 2)
    GLMakie.lines!(ax, [0, 0], [-lane_width, -map_end], color = :yellow, linewidth = 2)
    GLMakie.lines!(ax, [0, 0], [lane_width, map_end], color = :yellow, linewidth = 2)
    GLMakie.lines!(ax, [0, 0], [lane_width, map_end], color = :yellow, linewidth = 2)

    # Add directional arrows using arrows!
    xs = [-3, 3, 1, -1]  # Starting x-coordinates for arrows
    ys = [-1, 1, -3, 3]  # Starting y-coordinates for arrows
    us = [1, -1, 0, 0]      # Arrow x-directions
    vs = [0, 0, 1, -1]      # Arrow y-directions

    GLMakie.arrows!(
        xs,
        ys,
        us,
        vs;
        arrowsize = 15,
        lengthscale = 0.5,
        arrowcolor = :white,
        linecolor = :white,
        linewidth = 3,
    )

    # Visuliaze trajectories
    # goop_data = load_object("data/Intersection_closed_loop/GOOP_solution/intersection.jld2")
    # # NOTE: For some reason, GLMakie plots trajectories after reflecting the points along y=x. 
    # # To fix this, we need to reflect the points along y=x before plotting them.
    # goop_strategy1_xs = Observable([vcat(v[2], v[1], v[3:end]) for v in goop_data["strategy1"].xs])
    # goop_strategy2_xs = Observable([vcat(v[2], v[1], v[3:end]) for v in goop_data["strategy2"].xs])
    # goop_strategy1 = Observable(goop_data["strategy1"])
    # goop_strategy2 = Observable(goop_data["strategy2"])

    # goop_strategy1 = GLMakie.@lift OpenLoopStrategy($goop_strategy1_xs, $goop_strategy1.us)
    # goop_strategy2 = GLMakie.@lift OpenLoopStrategy($goop_strategy2_xs, $goop_strategy2.us)

    # GLMakie.plot!(ax, goop_strategy1, color = :blue)
    # GLMakie.plot!(ax, goop_strategy2, color = :red)
    Main.@infiltrate
    strategy1 = GLMakie.@lift OpenLoopStrategy($strategy[1].xs, $strategy[1].us)
    strategy2 = GLMakie.@lift OpenLoopStrategy($strategy[2].xs, $strategy[2].us)
    GLMakie.plot!(ax, strategy1, color = :blue)
    GLMakie.plot!(ax, strategy2, color = :red)

    # Visualize initial states 
    GLMakie.scatter!(
        ax,
        GLMakie.@lift([GLMakie.Point2f($θ1[1:2]), GLMakie.Point2f($θ2[1:2])]),
        markersize = 20,
        color = [:blue, :red],
    )

    # Visualize goal positions
    GLMakie.scatter!(
        ax,
        GLMakie.@lift(GLMakie.Point2f($goal_position1)),
        markersize = 20,
        marker = :star5,
        color = :blue,
    )
    GLMakie.scatter!(
        ax,
        GLMakie.@lift(GLMakie.Point2f($goal_position2)),
        markersize = 20,
        marker = :star5,
        color = :red,
    )

    # Save img 
    GLMakie.save("data/Intersection_closed_loop/trajectory.png", figure)

    # # closed_loop + receding horizon demo
    # time_step = 2
    # while time_step < 3 #15
    #     println("time_step: ", time_step)
    #     GLMakie.save("data/Intersection_closed_loop/trajectory$(time_step).png", figure)
    #     # Update the positions of the vehicles
    #     println("Update initial state1")
    #     θ1.val[1:state_dim(dynamics)] = first(strategy[]).xs[begin + 1] # Asynchronous update: mutate p1's initial state without triggering others
    #     println("Update initial state2")
    #     initial_state2[] = strategy[][2].xs[begin + 1]
    #     time_step += 1
    # end

    # # Store speed data for Intersection
    # horizontal_speed_data = Vector{Vector{Float64}}[]
    # vertical_speed_data = Vector{Vector{Float64}}[]
    # openloop_distance1 = Vector{Float64}[]

    # # Store openloop speed data
    # push!(horizontal_speed_data, [vcat(strategy[][1].xs...)[3:4:end], vcat(strategy[][2].xs...)[3:4:end]])#, vcat(strategy[3].xs...)[3:4:end]])
    # push!(vertical_speed_data, [vcat(strategy[][1].xs...)[4:4:end], vcat(strategy[][2].xs...)[4:4:end]])#, vcat(strategy[3].xs...)[4:4:end]])

    # # Store openloop distance data
    # push!(openloop_distance1, [sqrt(sum((strategy[][1].xs[k][1:2] - strategy[][2].xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])

    # # Visualize horizontal speed
    # T = 1
    # fig = GLMakie.Figure() # limits = (nothing, (nothing, 0.7))
    # ax2 = GLMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "speed", title = "Horizontal Speed")
    # GLMakie.scatterlines!(ax2, 0:planning_horizon-1, horizontal_speed_data[T][1], label = "Vehicle 1", color = :blue)
    # GLMakie.scatterlines!(ax2, 0:planning_horizon-1, horizontal_speed_data[T][2], label = "Vehicle 2", color = :red)
    # GLMakie.lines!(ax2, 0:planning_horizon-1, [1.5 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
    # fig[2,1:2] = GLMakie.Legend(fig, ax2, framevisible = false, orientation = :horizontal)

    # # Visualize vertical speed
    # ax3 = GLMakie.Axis(fig[1, 2]; xlabel = "time step", ylabel = "speed", title = "Vertical Speed")
    # GLMakie.scatterlines!(ax3, 0:planning_horizon-1, vertical_speed_data[T][1], label = "Vehicle 1", color = :blue)
    # GLMakie.scatterlines!(ax3, 0:planning_horizon-1, vertical_speed_data[T][2], label = "Vehicle 2", color = :red)
    # GLMakie.lines!(ax3, 0:planning_horizon-1, [1.5 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

    # GLMakie.save("./data/Intersection_closed_loop/GOOP_plots/speed.png", fig)

    # # Visualize distance bw vehicles , limits = (nothing, (collision_avoidance-0.05, 0.4)) 
    # fig = GLMakie.Figure() # limits = (nothing, (nothing, 0.7))
    # ax4 = GLMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "distance", title = "Distance bw vehicles")
    # GLMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance1[T], label = "B/w Agent 1 & Agent 2", color = :black, marker = :star5, markersize = 20)
    # GLMakie.lines!(ax4, 0:planning_horizon-1, [1.0 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
    # fig[2,1] = GLMakie.Legend(fig, ax4, framevisible = false, orientation = :horizontal)

    # GLMakie.save("./data/Intersection_closed_loop/GOOP_plots/" * "distance_bw_vehicles.png", fig)

    # # Store distance from center yellow line 
    # distance_from_center = Vector{Vector{Float64}}[]
    # push!(distance_from_center, [vcat(strategy[][1].xs...)[2:4:end], vcat(-strategy[][2].xs...)[1:4:end]])#, vcat(strategy[3].xs...)[2:4:end]])

    # # Visualize distance from center yellow line
    # fig = GLMakie.Figure() # limits = (nothing, (nothing, 0.7))
    # ax5 = GLMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "distance", title = "Position from center yellow line")
    # GLMakie.scatterlines!(ax5, 0:planning_horizon-1, distance_from_center[T][1], label = "Vehicle 1", color = :blue)
    # GLMakie.scatterlines!(ax5, 0:planning_horizon-1, distance_from_center[T][2], label = "Vehicle 2", color = :red)
    # GLMakie.lines!(ax5, 0:planning_horizon-1, [0.0 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
    # fig[2,1] = GLMakie.Legend(fig, ax5, framevisible = false, orientation = :horizontal)

    # GLMakie.save("./data/Intersection_closed_loop/GOOP_plots/" * "position_from_center.png", fig)

end

end
