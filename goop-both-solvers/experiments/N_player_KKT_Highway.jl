struct HighwayExperiment
    num_players::Int
    dynamics::AbstractDynamics
    planning_horizon::Int
    collision_avoidance::Real
    hybrid_goop::Bool
    objectives::Vector{Any}
    equality_constraints::Vector{Any}
    inequality_constraints::Vector{Any}
    prioritized_preferences::Vector{Vector{Any}}
    is_prioritized_constraint::Vector{Vector{Bool}}
    shared_equality_constraints::Any
    shared_inequality_constraints::Any
    primal_dimensions::Vector{Int}
    parameter_dimensions::Vector{Int}
    equality_dimensions::Vector{Int}
    inequality_dimensions::Vector{Int}
    shared_equality_dimension::Int
    shared_inequality_dimension::Int
end

function get_default_dynamics()
    control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
    planar_double_integrator(; control_bounds)
end

function HighwayExperiment(;
    num_players = 3,
    dynamics = get_default_dynamics(),
    planning_horizon = 5,
    collision_avoidance = 0.2,
    hybrid_goop = false,
)
    function unflatten_parameters(θ)
        θ_iter = Iterators.Stateful(θ)
        initial_state = first(θ_iter, state_dimension)
        goal_position = first(θ_iter, 2)
        obstacle_position = first(θ_iter, 2)
        (; initial_state, goal_position, obstacle_position)
    end

    state_dimension = state_dim(dynamics)
    control_dimension = control_dim(dynamics)
    primals_per_agent = (state_dimension + control_dimension) * planning_horizon
    primal_dimensions = fill(primals_per_agent, num_players)
    # Add four extra dimensions per plaer, two for the goal and two for the obstacle
    # TODO PHM: We are not incorporating the obstacle in any constraints!
    parameter_dimensions = fill(state_dimension + 4, num_players)

    dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
    dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

    objectives = [
        function (z, θ)
            # Consider variable number of objective levels
            # TODO PHM: Check that this doesn't significantly slow things down
            index = min(i, length(blocksizes(z)))
            (; xs, us) =
                unflatten_trajectory(z[Block(index)], state_dimension, control_dimension)
            0.5 * sum(sum(u .^ 2) for u in us)
        end for i in 1:num_players
    ]

    equality_constraints = [
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
            (; initial_state) = unflatten_parameters(θ[Block(i)])
            initial_state_constraint = xs[1] - initial_state
            dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
                xs[k] - dynamics(xs[k - 1], us[k - 1], k)
            end
            vcat(initial_state_constraint, dynamics_constraints)
        end for i in 1:num_players
    ]

    inequality_constraints = [
        function (z, θ)
            (; lb, ub) = control_bounds(dynamics)
            lb_mask = findall(!isinf, lb)
            ub_mask = findall(!isinf, ub)
            (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
            vcat(
                # control bounds (box)
                mapreduce(vcat, us) do u
                    vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
                end,

                # stay within the playing field (for planar_double_integrator)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    position_constraints = vcat(px + 1.0, -px + 1.0, py + 0.2, -py + 0.2)
                    vcat(position_constraints)
                end,
            )
        end for _ in 1:num_players
    ]

    prioritized_preferences = [
        [
            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                (; goal_position) = unflatten_parameters(θ[Block(1)]) # Player 1 θ[Block(i)]
                xs[end][1] - goal_position[1] # px[end] ≥ 0.9
            end,

            # Speed limit
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    velocity_constraints = vcat(vx + 0.2, -vx + 0.2, vy + 0.2, -vy + 0.2)
                    vcat(velocity_constraints)
                end
            end,
        ],
        [
            # Speed limit
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    velocity_constraints = vcat(vx + 0.2, -vx + 0.2, vy + 0.2, -vy + 0.2)
                    vcat(velocity_constraints)
                end
            end,

            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                (; goal_position) = unflatten_parameters(θ[Block(2)]) # Player 2
                xs[end][1] - goal_position[1] # px[end] ≥ 0.9
            end,
        ],
        [
            # Speed limit
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    velocity_constraints = vcat(vx + 0.2, -vx + 0.2, vy + 0.2, -vy + 0.2)
                    vcat(velocity_constraints)
                end
            end,

            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                (; goal_position) = unflatten_parameters(θ[Block(3)]) # Player 3
                xs[end][1] - goal_position[1] # px[end] ≥ 0.9
            end,
        ],
    ]

    # Specify prioritized constraint
    is_prioritized_constraint = [[true, true], [true, true], [true, true]]
    # In hybrid GOOP, the intermediate level is transcribed into penalty-based method
    if hybrid_goop == true
        is_prioritized_constraint = [[true, false], [true, false], [true, false]]
    end

    # Shared constraints
    function shared_equality_constraints(z, θ)
        [0]
    end

    function shared_inequality_constraints(z, θ)
        trajectories = map(
            i -> unflatten_trajectory(z[Block(i)], state_dimension, control_dimension),
            1:num_players,
        )
        xs = map(trajectory -> trajectory.xs, trajectories)
        @assert length(xs) == num_players
        # PHM: Note that this is fixed, it doesn't really depend on num_players
        # Avoid collision between 3 players
        mapreduce(vcat, 2:length(xs[1])) do k
            [
                sum((xs[1][k][1:2] - xs[2][k][1:2]) .^ 2) - collision_avoidance^2
                sum((xs[1][k][1:2] - xs[3][k][1:2]) .^ 2) - collision_avoidance^2
                sum((xs[2][k][1:2] - xs[3][k][1:2]) .^ 2) - collision_avoidance^2
            ]
        end
    end

    equality_dimensions =
        [length(equality_constraints[i](dummy_primals, dummy_parameters)) for i in 1:num_players]
    inequality_dimensions =
        [length(inequality_constraints[i](dummy_primals, dummy_parameters)) for i in 1:num_players]

    shared_equality_dimension = length(shared_equality_constraints(dummy_primals, dummy_parameters))
    shared_inequality_dimension =
        length(shared_inequality_constraints(dummy_primals, dummy_parameters))

    HighwayExperiment(
        num_players,
        dynamics,
        planning_horizon,
        collision_avoidance,
        hybrid_goop,
        objectives,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        is_prioritized_constraint,
        shared_equality_constraints,
        shared_inequality_constraints,
        primal_dimensions,
        parameter_dimensions,
        equality_dimensions,
        inequality_dimensions,
        shared_equality_dimension,
        shared_inequality_dimension,
    )
end

function flatten_parameters(; initial_state, goal_position, obstacle_position)
    vcat(initial_state, goal_position, obstacle_position)
end

function load_problem_parameters(problem_folder::String, filename::String)
    # Load problem data
    problem_data = JLD2.load_object(joinpath(problem_folder, filename))

    obstacle_position = problem_data["obstacle_position"]

    # Player 1
    initial_state1 = problem_data["initial_state1"]
    goal_position1 = problem_data["goal_position1"]
    θ1 = flatten_parameters(; # θ is a flat (column) vector of parameters
        initial_state = initial_state1,
        goal_position = goal_position1,
        obstacle_position = obstacle_position,
    )

    # Player 2
    initial_state2 = problem_data["initial_state2"]
    goal_position2 = problem_data["goal_position2"]
    θ2 = flatten_parameters(;
        initial_state = initial_state2,
        goal_position = goal_position2,
        obstacle_position = obstacle_position,
    )

    # Player 3
    initial_state3 = problem_data["initial_state3"]
    goal_position3 = problem_data["goal_position3"]
    θ3 = flatten_parameters(;
        initial_state = initial_state3,
        goal_position = goal_position3,
        obstacle_position = obstacle_position,
    )

    θ = [θ1..., θ2..., θ3...]
    initial_states = [initial_state1, initial_state2, initial_state3]

    return (; θ, initial_states)
end

function plot_strategies(
    experiment::HighwayExperiment,
    strategy,
    folder::String,
    filename_base::String,
)
    (; planning_horizon) = experiment

    # Store speed data for Highway
    horizontal_speed_data = Vector{Vector{Float64}}[]
    vertical_speed_data = Vector{Vector{Float64}}[]
    openloop_distance1 = Vector{Float64}[]
    openloop_distance2 = Vector{Float64}[]
    openloop_distance3 = Vector{Float64}[]

    # Store openloop speed data
    push!(
        horizontal_speed_data,
        [
            vcat(strategy[1].xs...)[3:4:end],
            vcat(strategy[2].xs...)[3:4:end],
            vcat(strategy[3].xs...)[3:4:end],
        ],
    )
    push!(
        vertical_speed_data,
        [
            vcat(strategy[1].xs...)[4:4:end],
            vcat(strategy[2].xs...)[4:4:end],
            vcat(strategy[3].xs...)[4:4:end],
        ],
    )

    # Store openloop distance data
    push!(
        openloop_distance1,
        [
            sqrt(sum((strategy[1].xs[k][1:2] - strategy[2].xs[k][1:2]) .^ 2)) for
            k in 1:planning_horizon
        ],
    )
    push!(
        openloop_distance2,
        [
            sqrt(sum((strategy[1].xs[k][1:2] - strategy[3].xs[k][1:2]) .^ 2)) for
            k in 1:planning_horizon
        ],
    )
    push!(
        openloop_distance3,
        [
            sqrt(sum((strategy[2].xs[k][1:2] - strategy[3].xs[k][1:2]) .^ 2)) for
            k in 1:planning_horizon
        ],
    )

    # Visualize horizontal speed
    T = 1
    fig = CairoMakie.Figure()
    ax2 = CairoMakie.Axis(
        fig[1, 1];
        xlabel = "time step",
        ylabel = "speed",
        title = "Horizontal Speed",
    )
    CairoMakie.scatterlines!(
        ax2,
        0:(planning_horizon - 1),
        horizontal_speed_data[T][1],
        label = "Vehicle 1",
        color = :blue,
    )
    CairoMakie.scatterlines!(
        ax2,
        0:(planning_horizon - 1),
        horizontal_speed_data[T][2],
        label = "Vehicle 2",
        color = :red,
    )
    CairoMakie.scatterlines!(
        ax2,
        0:(planning_horizon - 1),
        horizontal_speed_data[T][3],
        label = "Vehicle 3",
        color = :green,
    )
    CairoMakie.lines!(
        ax2,
        0:(planning_horizon - 1),
        [0.2 for _ in 0:(planning_horizon - 1)],
        color = :black,
        linestyle = :dash,
    )
    fig[2, 1:2] = CairoMakie.Legend(fig, ax2, framevisible = false, orientation = :horizontal)

    # Visualize vertical speed
    ax3 =
        CairoMakie.Axis(fig[1, 2]; xlabel = "time step", ylabel = "speed", title = "Vertical Speed")
    CairoMakie.scatterlines!(
        ax3,
        0:(planning_horizon - 1),
        vertical_speed_data[T][1],
        label = "Vehicle 1",
        color = :blue,
    )
    CairoMakie.scatterlines!(
        ax3,
        0:(planning_horizon - 1),
        vertical_speed_data[T][2],
        label = "Vehicle 2",
        color = :red,
    )
    CairoMakie.scatterlines!(
        ax3,
        0:(planning_horizon - 1),
        vertical_speed_data[T][3],
        label = "Vehicle 3",
        color = :green,
    )
    CairoMakie.lines!(
        ax3,
        0:(planning_horizon - 1),
        [0.2 for _ in 0:(planning_horizon - 1)],
        color = :black,
        linestyle = :dash,
    )

    save_plot(joinpath(folder, "speed_$filename_base"), fig)

    # Visualize distance bw vehicles
    fig = CairoMakie.Figure()
    ax4 = CairoMakie.Axis(
        fig[1, 1];
        xlabel = "time step",
        ylabel = "distance",
        title = "Distance bw vehicles",
    )
    CairoMakie.scatterlines!(
        ax4,
        0:(planning_horizon - 1),
        openloop_distance1[T],
        label = "B/w Agent 1 & Agent 2",
        color = :black,
        marker = :star5,
        markersize = 20,
    )
    CairoMakie.scatterlines!(
        ax4,
        0:(planning_horizon - 1),
        openloop_distance2[T],
        label = "B/w Agent 1 & Agent 3",
        color = :orange,
        marker = :diamond,
        markersize = 20,
    )
    CairoMakie.scatterlines!(
        ax4,
        0:(planning_horizon - 1),
        openloop_distance3[T],
        label = "B/w Agent 2 & Agent 3",
        color = :purple,
        marker = :circle,
        markersize = 20,
    )
    CairoMakie.lines!(
        ax4,
        0:(planning_horizon - 1),
        [0.2 for _ in 0:(planning_horizon - 1)],
        color = :black,
        linestyle = :dash,
    )
    fig[2, 1] = CairoMakie.Legend(fig, ax4, framevisible = false, orientation = :horizontal)

    save_plot(joinpath(folder, "distance_$filename_base"), fig)
end