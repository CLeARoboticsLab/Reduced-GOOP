module RoboticArmCore

#=
	Plotting-free core of the robotic-arm experiment.

	Everything here builds and post-processes the game: scenario configuration,
	problem construction, warm starts, and trajectory packing/unpacking. Nothing
	here draws, saves, or depends on a plotting stack, so a solver-only entry
	point (`Robotic_arm_mpc.jl`) can load this module without pulling in
	CairoMakie/WGLMakie/GLMakie.

	The module is loaded once as `Main.RoboticArmCore` and reused by the
	open-loop, receding-horizon, and solver-only entry points. Loading this file
	with Revise's `includet` therefore gives every entry point the same revised
	methods and type identities.
=#

using BlockArrays
using ReducedGOOP
using TimerOutputs: @timeit

export TO,
    DynamicsModel,
    SingleIntegrator3D,
    ScenarioConfig,
    InstanceParameters,
    get_setup,
    demo_scenario_config,
    build_dynamics,
    build_two_arm_dynamics,
    dynamics_model_name,
    extract_player_strategies,
    extract_initial_controls,
    split_arm_strategies,
    evaluate_preferences_at_solution,
    summarize_dual_blocks,
    build_instance_parameters,
    limit_control_speed,
    build_default_warmstart,
    build_up_and_over_warmstart,
    build_constant_control_warmstart,
    build_ground_line_warmstart,
    flatten_warmstart_solution,
    unflatten_trajectory,
    single_integrator_3d,
    single_integrator_3d_step

const TO = ReducedGOOP.TO

abstract type DynamicsModel end
struct SingleIntegrator3D <: DynamicsModel end

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
    use_running_goal_cost::Bool = false
    use_up_and_over_warmstart::Bool = false
    use_world_frame::Bool = false
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
        planning_horizon,
        dₚ,
        collision_avoidance,
        arm_speed_limit,
        child_speed_limit,
        use_scalarized_baseline,
        use_social_equilibrium_baseline,
        use_running_goal_cost,
        use_world_frame,
    ) = scenario_config
    num_players == 2 || error(
        "robotic_arm Zero-Sum GOOP setup expects exactly two players (two-arm robot and child).",
    )
    length(dynamics) == num_players || error("Expected one dynamics entry per player.")

    # ── Common dimensions and parameter layout ──────────────────────────────────────────────────────────

    position_dimension = 3
    # Player 1 is the combined two-arm agent: state/control stack both arms as
    # [arm1; arm2]. Player 2 is the child/pet on the ground.
    arm_state_dimension, state_remainder = divrem(dynamics[1].state_dimension, 2)
    arm_control_dimension, control_remainder = divrem(dynamics[1].control_dimension, 2)
    iszero(state_remainder) || error("Expected an even stacked two-arm state dimension.")
    iszero(control_remainder) ||
        error("Expected an even stacked two-arm control dimension.")
    arm_state_dimension == position_dimension || error(
        "Expected each arm state to have $(position_dimension) position dimensions, " *
        "got $(arm_state_dimension).",
    )
    child_control_dimension = dynamics[2].control_dimension
    child_control_dimension == position_dimension || error(
        "Expected a $(position_dimension)D child control, got " *
        "$(child_control_dimension)D.",
    )
    arm1_state_range = 1:arm_state_dimension
    arm2_state_range = (arm_state_dimension + 1):(2 * arm_state_dimension)
    arm1_control_range = 1:arm_control_dimension
    arm2_control_range = (arm_control_dimension + 1):(2 * arm_control_dimension)
    child_horizontal_control_range = 1:(child_control_dimension - 1)
    child_vertical_control_index = child_control_dimension

    primal_dimensions = [
        (dyn.state_dimension + dyn.control_dimension) * planning_horizon for
        dyn in dynamics
    ]
    # Per player: initial state, initial control, goal state, and a 3D obstacle.
    parameter_dimensions = [
        dyn.state_dimension +
        dyn.control_dimension +
        dyn.state_dimension +
        position_dimension for dyn in dynamics
    ]

    dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
    dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

    # ── Common parameter and trajectory helpers ─────────────────────────────────────────────────────────────

    unflatten_parameters = function (θ; player)
        state_dimension = dynamics[player].state_dimension
        control_dimension = dynamics[player].control_dimension
        θ_iter = Iterators.Stateful(θ)
        initial_state = first(θ_iter, state_dimension)
        initial_control = first(θ_iter, control_dimension)
        goal_position = first(θ_iter, state_dimension)
        obstacle_position = first(θ_iter, position_dimension)
        (; initial_state, initial_control, goal_position, obstacle_position)
    end

    function flatten_parameters(;
        player,
        initial_state,
        initial_control,
        goal_position,
        obstacle_position,
    )
        state_dimension = dynamics[player].state_dimension
        control_dimension = dynamics[player].control_dimension
        length(initial_state) == state_dimension || error(
            "Expected initial_state length $(state_dimension), got $(length(initial_state)).",
        )
        length(initial_control) == control_dimension || error(
            "Expected initial_control length $(control_dimension), got $(length(initial_control)).",
        )
        length(goal_position) == state_dimension || error(
            "Expected goal_position length $(state_dimension), got $(length(goal_position)).",
        )
        length(obstacle_position) == position_dimension || error(
            "Expected 3D obstacle_position, got length $(length(obstacle_position)).",
        )
        vcat(initial_state, initial_control, goal_position, obstacle_position)
    end

    trajectory(z; player) = unflatten_trajectory(
        z[Block(player)],
        dynamics[player].state_dimension,
        dynamics[player].control_dimension,
    )

    # ── Preference objectives ──────────────────────────────────────────────────────────────────

    function goal_objective(z, θ)
        (; xs, us) = trajectory(z; player = 1)
        (; goal_position) = unflatten_parameters(θ[Block(1)]; player = 1)

        center_goal_position =
            0.5 .* (goal_position[arm1_state_range] .+ goal_position[arm2_state_range])
        if use_running_goal_cost
            sum(eachindex(xs)) do t
                center_position =
                    0.5 .* (xs[t][arm1_state_range] .+ xs[t][arm2_state_range])
                sum(abs2, center_position .- center_goal_position)
            end
        else # terminal cost only
            center_position =
                0.5 .* (xs[end][arm1_state_range] .+ xs[end][arm2_state_range])
            sum(abs2, center_position .- center_goal_position)
        end
    end

    function control_objective(; player)
        function (z, _)
            (; us) = trajectory(z; player)
            sum(sum(u .^ 2) for u in us)
        end
    end

    function load_balance_objective(z, θ; allowance = 0.5)
        (; xs, us) = trajectory(z; player = 1)
        sum(eachindex(xs)) do t
            balance = (xs[t][arm1_state_range[end]] - xs[t][arm2_state_range[end]])^2
            ifelse(balance < allowance^2, 0.0, balance - allowance^2)
        end
    end

    function pot_approach_objective(z, θ)
        arm_trajectory = trajectory(z; player = 1)
        child_trajectory = trajectory(z; player = 2)

        sum(eachindex(child_trajectory.xs)) do t
            arm_state = arm_trajectory.xs[t]
            pot_center =
                0.5 .* (arm_state[arm1_state_range] .+ arm_state[arm2_state_range])
            sum(abs2, child_trajectory.xs[t] .- pot_center)
        end
    end

    # ── Equality and inequality constraints ──────────────────────────────────────────────────────────
    player_equality_constraints = [
        function (z, θ)
            (; xs, us) = trajectory(z; player = i)
            (; initial_state, initial_control) =
                unflatten_parameters(θ[Block(i)]; player = i)

            initial_state_constraint = xs[1] - initial_state
            initial_control_constraint = us[1] - initial_control
            dynamics_constraints = mapreduce(vcat, 1:(length(xs) - 1)) do t
                # xs[t] - dynamics(xs[t-1], us[t-1], t)
                dynamics[i].residual(z[Block(i)], t)
            end
            empty_constraint = similar(initial_state_constraint, 0)

            # The two arms jointly carry the hot pot. This hard handle-grasp
            # constraint keeps the two grippers separated by the fixed 3D handle
            # distance dₚ at all times.
            handle_grasp_constraint =
                i == 1 ?
                mapreduce(vcat, eachindex(xs)) do t
                    gripper_separation =
                        xs[t][arm1_state_range] .- xs[t][arm2_state_range]
                    [sum(abs2, gripper_separation) - dₚ^2]
                end : empty_constraint

            # The child is modeled with the same 3D single-integrator dynamics. In the
            # abstract scenario the child is on the ground so vertical velocity is zero;
            # in world-frame mode the GR1 EEF is not on a ground plane, so skip this.
            child_ground_constraint =
                (i == 2 && !use_world_frame) ?
                mapreduce(u -> [u[child_vertical_control_index]], vcat, us) :
                empty_constraint

            vcat(
                initial_state_constraint,
                initial_control_constraint,
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

        mapreduce(vcat, eachindex(child_trajectory.xs)) do t
            arm_state = arm_trajectory.xs[t]
            pot_center =
                0.5 .* (arm_state[arm1_state_range] .+ arm_state[arm2_state_range])
            separation = pot_center .- child_trajectory.xs[t]
            [sum(abs2, separation) - collision_avoidance^2]
        end
    end

    function robot_arm_speed_inequality(z, θ)
        (; us) = trajectory(z; player = 1)

        # Each arm individually respects the same speed limit.
        mapreduce(vcat, us) do u
            [
                arm_speed_limit^2 - sum(abs2, u[arm1_control_range]),
                arm_speed_limit^2 - sum(abs2, u[arm2_control_range]),
            ]
        end
    end

    function robot_inequality(z, θ)
        vcat(robot_child_safety_inequality(z, θ), robot_arm_speed_inequality(z, θ))
    end

    function child_ground_speed_inequality(z, θ)
        (; us) = trajectory(z; player = 2)

        # Child cannot teleport into the pot center.
        mapreduce(vcat, us) do u
            [(child_speed_limit)^2 - sum(abs2, u[child_horizontal_control_range])]
        end
    end

    inequality_constraints = [robot_inequality, child_ground_speed_inequality]

    # ── 1. GOOP formulation ──────────────────────────────────────────────────────────────────────────────
    # Preference hierarchy: [lowest priority, ..., highest priority].
    robot_preferences = Function[
        control_objective(; player = 1),
        goal_objective,
        load_balance_objective,
        robot_inequality,
    ]
    robot_is_prioritized_constraint = [false, false, false, true]

    child_preferences = Function[pot_approach_objective, child_ground_speed_inequality]
    child_is_prioritized_constraint = [false, true]

    goop_preferences = [robot_preferences, child_preferences]
    goop_is_prioritized_constraint =
        [robot_is_prioritized_constraint, child_is_prioritized_constraint]

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

    function smooth_piecewise_preference_objective(preference, level; ϵ = 0.0)
        ifelse(preference ≥ ϵ, 0.0, (ϵ - preference)^(level + 2))
    end

    function evaluate_preference_level(preference, is_constraint, level, z, θ)
        preference_value = preference(z, θ)
        if is_constraint
            return sum(smooth_piecewise_preference_objective.(preference_value, level))
        end
        preference_value
    end

    function scalarized_player_preference(
        player_preferences,
        player_is_prioritized_constraint,
    )
        @assert length(player_preferences) == length(player_is_prioritized_constraint)

        function (z, θ)
            accumulated_preference = 0
            for (level, (preference, is_constraint)) in
                enumerate(zip(player_preferences, player_is_prioritized_constraint))
                accumulated_preference +=
                    evaluate_preference_level(preference, is_constraint, level, z, θ)
            end
            accumulated_preference
        end
    end

    scalarized_player_preferences = map(
        scalarized_player_preference,
        goop_preferences,
        goop_is_prioritized_constraint,
    )
    scalarized_preferences =
        [[preference] for preference in scalarized_player_preferences]
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
        vcat(inequality_constraints[1](z, θ), inequality_constraints[2](z, θ))
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

"""
Build the exact `ScenarioConfig` used by `demo`. Kept as a separate function so
external tools (e.g. `SecondOrderCheck`) can rebuild the identical problem
without duplicating the configuration.
"""
function demo_scenario_config(;
    use_scalarized_baseline = false,
    use_social_equilibrium_baseline = false,
    use_running_goal_cost = false,
    use_up_and_over_warmstart = false,
    # Discretization is a kwarg so sweeps (e.g. the Δt/T benchmark) can vary the
    # planning grid without duplicating the rest of the scenario.
    planning_horizon = 30,
    Δt = 0.1,
    # ── Real-world simulation kwargs (pass from Python after grip) ──────────
    # When provided the solver operates in MuJoCo world-frame (metres).
    sim_eef0 = nothing,          # robot0 EEF position [x, y, z] (world frame)
    sim_eef1 = nothing,          # robot1 EEF position [x, y, z] (world frame)
    sim_eef2 = nothing,          # robot2/GR1 EEF position [x, y, z] (world frame)
    sim_lift_height = 0.35,      # target lift height above grip position (metres)
)
    dynamics_model = SingleIntegrator3D()

    # ── World-frame mode detection ────────────────────────────────────────────
    # When sim_eef0/1 are provided (from Python after grip), the solver uses
    # MuJoCo world-frame metres. Otherwise abstract coordinates are used.
    use_world_frame = !isnothing(sim_eef0) && !isnothing(sim_eef1)

    # ── Problem parameters ─────────────────────────────────────────────────────
    # Player 1: combined two-arm agent, Player 2: child/pet.
    num_players = 2
    collision_avoidance = use_world_frame ? 0.25 : 2.5   # safety sphere radius
    child_initial_buffer = use_world_frame ? 0.4  : 4.0
    arm_speed_limit      = use_world_frame ? 0.5  : 5.0   # m/s (world) or abstract
    child_speed_limit    = use_world_frame ? 0.3  : 3.0   # m/s (world) or abstract
    # In world-frame mode, dₚ is the actual Euclidean distance between the two
    # grippers at the moment of grip (measured from the sim). Using 2.0 (abstract
    # units) in world-frame makes the initial-state + handle-grasp equality
    # constraints contradictory, so the problem is infeasible before the solver
    # takes a single step.
    dₚ = use_world_frame ? sqrt(sum(abs2, sim_eef0 .- sim_eef1)) : 2.0

    # ── Scenario ───────────────────────────────────────────────────────────────
    # Single Integrator 3D: state = [px, py, pz]
    if use_world_frame
        # World-frame (metres): robot 0 on +y side, robot 1 on −y side, GR1 on −x side.
        # Table surface z ≈ 0.8 m; handles at z ≈ 0.84 m, ±0.17 m from pot centre in y.
        # Goals: lift the pot straight up by sim_lift_height while holding the handles.
        base_initial_state1 = collect(Float64, sim_eef0)
        base_initial_state2 = collect(Float64, sim_eef1)
        goal_position1      = base_initial_state1 .+ [0.0, 0.0, sim_lift_height]
        goal_position2      = base_initial_state2 .+ [0.0, 0.0, sim_lift_height]
        initial_state3      = isnothing(sim_eef2) ? [-0.55, 0.0, 1.0] : collect(Float64, sim_eef2)
        goal_position3      = 0.5 .* (base_initial_state1 .+ base_initial_state2)  # pot centre
    else
        # Abstract coordinates for standalone testing.
        base_initial_state1 = [-1.0, 6.0, 1.5]
        base_initial_state2 = [1.0, 6.0, 1.5]
        goal_position1 = [-1.0, -5.0, 3.5]
        goal_position2 = [1.0, -5.0, 3.5]
        initial_state3 = [-3.0, -1.0, 0.0]
        goal_position3 = [0.0, 0.0, 0.0]
    end
    obstacle_position = [0.25, 0.15, 0.0]   # placeholder

    # Miscellaneous
    map_end = 10
    lane_width = 2

    # ── Build dynamics ─────────────────────────────────────────────────────────
    state_dimension = 3
    control_dimension = 3
    # World-frame: control = velocity (m/s); ±1 m/s is realistic for a robot arm.
    # Abstract: keep legacy large bounds.
    control_bounds = use_world_frame ?
        (; lb = [-1.0, -1.0, -1.0], ub = [1.0, 1.0, 1.0]) :
        (; lb = [-10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0])

    # Per-player dynamics: the combined two-arm agent stacks both arms into one
    # 6D single integrator; the child keeps the plain 3D single integrator.
    dynamics = @timeit TO "dynamics construction" [
        build_two_arm_dynamics(dynamics_model; Δt),
        build_dynamics(dynamics_model; Δt, state_dimension, control_dimension),
    ]

    ScenarioConfig(;
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
        use_running_goal_cost,
        use_up_and_over_warmstart,
        use_world_frame,
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

function extract_player_strategies(primal_solution, primal_dimensions, dynamics)
    offsets = cumsum(vcat(0, primal_dimensions[1:(end - 1)]))
    map(eachindex(primal_dimensions)) do player_idx
        start_idx = offsets[player_idx] + 1
        end_idx = offsets[player_idx] + primal_dimensions[player_idx]
        unflatten_trajectory(
            primal_solution[start_idx:end_idx],
            dynamics[player_idx].state_dimension,
            dynamics[player_idx].control_dimension,
        )
    end
end

"""Extract the per-arm and child controls at the first trajectory knot."""
function extract_initial_controls(primal_solution, primal_dimensions, dynamics)
    strategies = extract_player_strategies(primal_solution, primal_dimensions, dynamics)
    combined_arm_control = collect(strategies[1].us[1])
    arm_control_dimension, remainder = divrem(dynamics[1].control_dimension, 2)
    iszero(remainder) || error(
        "Expected an even stacked two-arm control dimension, got " *
        "$(dynamics[1].control_dimension).",
    )
    length(combined_arm_control) == dynamics[1].control_dimension || error(
        "Expected a stacked two-arm control of length " *
        "$(dynamics[1].control_dimension), got $(length(combined_arm_control)).",
    )
    length(strategies) >= 2 || error("Expected a child strategy.")
    child_control = collect(strategies[2].us[1])
    length(child_control) == dynamics[2].control_dimension || error(
        "Expected a child control of length $(dynamics[2].control_dimension), got " *
        "$(length(child_control)).",
    )
    (;
        initial_control1 = combined_arm_control[1:arm_control_dimension],
        initial_control2 = combined_arm_control[(arm_control_dimension + 1):(2 * arm_control_dimension)],
        initial_control3 = child_control,
    )
end

"""
Split the combined two-arm strategy (player 1) into per-arm 3D strategies so
the existing per-arm plotting code can be reused unchanged. Returns
[arm1, arm2, child, ...] in the layout the plots expect.
"""
function split_arm_strategies(strategies)
    combined = strategies[1]
    state_dimension, state_remainder = divrem(length(combined.xs[1]), 2)
    control_dimension, control_remainder = divrem(length(combined.us[1]), 2)
    iszero(state_remainder) || error("Expected an even stacked two-arm state dimension.")
    iszero(control_remainder) ||
        error("Expected an even stacked two-arm control dimension.")
    arm1 = (;
        xs = [x[1:state_dimension] for x in combined.xs],
        us = [u[1:control_dimension] for u in combined.us],
    )
    arm2 = (;
        xs = [x[(state_dimension + 1):(2 * state_dimension)] for x in combined.xs],
        us = [u[(control_dimension + 1):(2 * control_dimension)] for u in combined.us],
    )
    vcat([arm1, arm2], strategies[2:end])
end

"""
Evaluate each player's preference levels at a converged primal solution,
ordered [lowest priority, ..., highest priority]. Objective preferences report
their scalar value; prioritized-constraint preferences report the same smooth
violation penalty the KKT generator minimizes at that level,
`sum(smooth_piecewise_preference_objective.(h, level))`, which is 0 iff the
constraint is satisfied.
"""
function evaluate_preferences_at_solution(problem, x, θ)
    x_block = BlockArray(collect(x), problem.primal_dims)
    θ_block = BlockArray(collect(θ), problem.parameter_dims)
    map(1:problem.num_players) do player
        levels =
            zip(problem.preferences[player], problem.is_prioritized_constraint[player])
        map(enumerate(levels)) do (level, (preference, is_constraint))
            value = preference(x_block, θ_block)
            if is_constraint
                sum(ReducedGOOP.smooth_piecewise_preference_objective.(value, level))
            else
                value
            end
        end
    end
end

"""
Summarize each named block of the solver vector `z` (grouped by symbolic
variable name, e.g. `ψ_1_2`, `λ_2_1`) with max, min, ‖·‖₂, and the number of
near-zero coordinates (|·| ≤ `zero_tol`). A large near-zero count on an inner
level's duals signals that the level is numerically inactive there.
"""
function summarize_dual_blocks(kkt_system, z; zero_tol = 1e-6)
    name_of(sym) = split(string(sym), "[")[1]
    groups = Dict{String,Vector{Int}}()
    for (i, sym) in enumerate(kkt_system.z_symbolic)
        push!(get!(groups, name_of(sym), Int[]), i)
    end
    map(sort!(collect(keys(groups)))) do name
        v = view(z, groups[name])
        (;
            name,
            len = length(v),
            max = maximum(v),
            min = minimum(v),
            norm2 = sqrt(sum(abs2, v)),
            n_near_zero = count(x -> abs(x) <= zero_tol, v),
            zero_tol,
        )
    end
end

function build_instance_parameters(
    flatten_parameters,
    instance_states,
    scenario_config::ScenarioConfig,
)
    (; initial_state1, initial_state2, initial_state3) = instance_states
    dynamics = scenario_config.dynamics
    arm_control_dimension, remainder = divrem(dynamics[1].control_dimension, 2)
    iszero(remainder) || error(
        "Expected an even stacked two-arm control dimension, got " *
        "$(dynamics[1].control_dimension).",
    )
    child_control_dimension = dynamics[2].control_dimension
    # Keep callers that only provide states working by using the physically
    # stationary initial condition as the default control boundary.
    initial_control1 =
        hasproperty(instance_states, :initial_control1) ?
        instance_states.initial_control1 :
        zeros(eltype(initial_state1), arm_control_dimension)
    initial_control2 =
        hasproperty(instance_states, :initial_control2) ?
        instance_states.initial_control2 :
        zeros(eltype(initial_state2), arm_control_dimension)
    initial_control3 =
        hasproperty(instance_states, :initial_control3) ?
        instance_states.initial_control3 :
        zeros(eltype(initial_state3), child_control_dimension)
    (; goal_position1, goal_position2, goal_position3, obstacle_position) =
        scenario_config
    # Solver-facing parameter blocks: player 1 stacks both arms, player 2 is the child.
    θ_arms = flatten_parameters(;
        player = 1,
        initial_state = vcat(initial_state1, initial_state2),
        initial_control = vcat(initial_control1, initial_control2),
        goal_position = vcat(goal_position1, goal_position2),
        obstacle_position = obstacle_position,
    )
    θ_child = flatten_parameters(;
        player = 2,
        initial_state = initial_state3,
        initial_control = initial_control3,
        goal_position = goal_position3,
        obstacle_position = obstacle_position,
    )

    # Per-arm 3D blocks are retained for plotting, which reads the initial
    # position from the first three entries.
    θ1 = vcat(initial_state1, initial_control1, goal_position1, obstacle_position)
    θ2 = vcat(initial_state2, initial_control2, goal_position2, obstacle_position)
    θ3 = copy(θ_child)

    InstanceParameters(θ1, θ2, θ3, [θ_arms..., θ_child...])
end

function limit_control_speed(control, speed_limit, speed_indices)
    speed_limit >= 0 || error("Speed limit must be nonnegative.")
    bounded_control = collect(control)
    control_speed = sqrt(sum(abs2, bounded_control[speed_indices]))
    if control_speed > speed_limit && control_speed > 0
        bounded_control[speed_indices] .=
            bounded_control[speed_indices] .* (speed_limit / control_speed)
    end
    bounded_control
end

function build_default_warmstart(instance_states, scenario_config::ScenarioConfig)
    (; initial_state1, initial_state2, initial_state3) = instance_states
    (;
        planning_horizon,
        dynamics,
        goal_position1,
        goal_position2,
        goal_position3,
        collision_avoidance,
        safety_buffer_margin,
        arm_speed_limit,
        child_speed_limit,
        use_up_and_over_warmstart,
    ) = scenario_config
    arm_dynamics, child_dynamics = dynamics
    arm_state_dimension, state_remainder = divrem(arm_dynamics.state_dimension, 2)
    arm_control_dimension, remainder = divrem(arm_dynamics.control_dimension, 2)
    iszero(state_remainder) || error(
        "Expected an even stacked two-arm state dimension, got " *
        "$(arm_dynamics.state_dimension).",
    )
    iszero(remainder) || error(
        "Expected an even stacked two-arm control dimension, got " *
        "$(arm_dynamics.control_dimension).",
    )
    arm_state_dimension == arm_control_dimension || error(
        "SingleIntegrator3D requires matching per-arm state and control dimensions.",
    )
    planning_horizon >= 2 || error("Robot warm start requires at least two time steps.")
    length(goal_position1) == arm_state_dimension ||
        error("Goal position 1 dimension mismatch.")
    length(goal_position2) == arm_state_dimension ||
        error("Goal position 2 dimension mismatch.")

    arms_warmstart = if use_up_and_over_warmstart
        # Up-and-over warm start: the interior-point solver is local, so start
        # in the evasion basin (rise above the child's safety sphere, then
        # advance) instead of the straight line that heads at the child.
        build_up_and_over_warmstart(
            planning_horizon,
            arm_dynamics,
            vcat(initial_state1, initial_state2),
            vcat(goal_position1, goal_position2);
            clearance_altitude = collision_avoidance + safety_buffer_margin,
            speed_limit = arm_speed_limit,
            child_initial_position = initial_state3,
        )
    else
        total_time = arm_dynamics.Δt * (planning_horizon - 1)
        start_center = 0.5 .* (initial_state1 .+ initial_state2)
        goal_center = 0.5 .* (goal_position1 .+ goal_position2)
        # Same 3D constant control for both arms, stacked into the combined
        # agent's 6D control.
        robot_warmstart_control = limit_control_speed(
            (goal_center .- start_center) ./ total_time,
            arm_speed_limit,
            1:arm_control_dimension,
        )
        build_constant_control_warmstart(
            planning_horizon,
            arm_dynamics,
            vcat(initial_state1, initial_state2),
            vcat(robot_warmstart_control, robot_warmstart_control),
        )
    end

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

"""
Three-phase evasion warm start for the stacked two-arm agent: rise to
`clearance_altitude`, cruise toward the goal center at altitude, then descend
to the goal height. Both arms receive the identical 3D control, so the initial
gripper separation (and hence the handle-grasp equality) is preserved exactly.

`lateral_evasion_offset > 0` mixes into the rise phase a horizontal component
directed away from `child_initial_position`, for scenarios where vertical
clearance alone does not escape the pursuer's corridor.
"""
function build_up_and_over_warmstart(
    planning_horizon,
    dynamics,
    initial_state,
    goal_position;
    clearance_altitude,
    speed_limit,
    child_initial_position = nothing,
    lateral_evasion_offset = 0.0,
    position_dimension = 3,
    # Keep warm-start controls strictly inside the speed bound: at short Δt
    # the rise phase saturates the limit, and a boundary-active warm start
    # (zero slack) degrades the interior-point iteration.
    speed_margin = 0.95,
)
    length(initial_state) == dynamics.state_dimension ||
        error("Initial state dimension mismatch.")
    length(goal_position) == dynamics.state_dimension ||
        error("Goal position dimension mismatch.")
    planning_horizon >= 2 ||
        error("Up-and-over warm start requires at least two time steps.")

    arm_state_dimension, state_remainder = divrem(dynamics.state_dimension, 2)
    arm_control_dimension, control_remainder = divrem(dynamics.control_dimension, 2)
    iszero(state_remainder) || error("Expected an even stacked two-arm state dimension.")
    iszero(control_remainder) ||
        error("Expected an even stacked two-arm control dimension.")
    arm_state_dimension == position_dimension || error(
        "Expected each arm state to have $(position_dimension) position dimensions, " *
        "got $(arm_state_dimension).",
    )
    arm_control_dimension == position_dimension || error(
        "SingleIntegrator3D requires matching per-arm state and control dimensions.",
    )
    arm1_state_range = 1:arm_state_dimension
    arm2_state_range = (arm_state_dimension + 1):(2 * arm_state_dimension)
    vertical_state_index = position_dimension
    vertical_control_index = arm_control_dimension
    horizontal_control_range = 1:(arm_control_dimension - 1)
    Δt = dynamics.Δt
    num_steps = planning_horizon - 1
    num_rise_steps = max(1, round(Int, num_steps / 3))
    num_descend_steps = max(1, round(Int, num_steps / 3))
    num_cruise_steps = max(0, num_steps - num_rise_steps - num_descend_steps)

    goal_center =
        0.5 .* (goal_position[arm1_state_range] .+ goal_position[arm2_state_range])
    start_center =
        0.5 .* (initial_state[arm1_state_range] .+ initial_state[arm2_state_range])
    evasion_direction =
        if lateral_evasion_offset > 0 && !isnothing(child_initial_position)
            away = start_center[1:2] .- child_initial_position[1:2]
            away_norm = sqrt(sum(abs2, away))
            away_norm > 0 ? away ./ away_norm : zeros(2)
        else
            zeros(2)
        end

    xs = [collect(initial_state)]
    us = Vector{Float64}[]
    for t in 1:num_steps
        center = 0.5 .* (xs[t][arm1_state_range] .+ xs[t][arm2_state_range])
        remaining_time = (num_steps - t + 1) * Δt
        center_control = (goal_center .- center) ./ remaining_time
        if t <= num_rise_steps
            remaining_rise_time = (num_rise_steps - t + 1) * Δt
            center_control[vertical_control_index] =
                (clearance_altitude - center[vertical_state_index]) / remaining_rise_time
            center_control[horizontal_control_range] .+=
                evasion_direction .* (lateral_evasion_offset / (num_rise_steps * Δt))
        elseif t <= num_rise_steps + num_cruise_steps
            center_control[vertical_control_index] = 0.0
        end
        center_control = limit_control_speed(
            center_control,
            speed_margin * speed_limit,
            1:arm_control_dimension,
        )
        push!(us, vcat(center_control, center_control))
        push!(xs, dynamics.step(xs[t], us[t], t))
    end
    push!(us, zeros(dynamics.control_dimension))
    (; xs, us)
end

function build_constant_control_warmstart(
    planning_horizon,
    dynamics,
    initial_state,
    constant_control,
)
    length(initial_state) == dynamics.state_dimension ||
        error("Initial state dimension mismatch.")
    length(constant_control) == dynamics.control_dimension ||
        error("Control dimension mismatch.")

    xs = [collect(initial_state)]
    us = [collect(constant_control)]
    for t in 1:(planning_horizon - 1)
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
    length(initial_state) == dynamics.state_dimension ||
        error("Initial state dimension mismatch.")
    length(goal_position) == dynamics.state_dimension ||
        error("Goal position dimension mismatch.")
    planning_horizon >= 2 ||
        error("Ground-line warm start requires at least two time steps.")
    dynamics.state_dimension == dynamics.control_dimension ||
        error("SingleIntegrator3D requires matching state and control dimensions.")
    vertical_state_index = dynamics.state_dimension
    vertical_control_index = dynamics.control_dimension
    horizontal_control_range = 1:(vertical_control_index - 1)

    total_time = dynamics.Δt * (planning_horizon - 1)
    constant_control = (goal_position .- initial_state) ./ total_time
    constant_control[vertical_control_index] = 0.0
    constant_control = limit_control_speed(
        constant_control,
        ground_speed_limit,
        horizontal_control_range,
    )

    xs = [collect(initial_state)]
    us = [collect(constant_control)]
    for t in 1:(planning_horizon - 1)
        push!(xs, dynamics.step(xs[t], us[1], t))
        xs[end][vertical_state_index] = 0.0
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

end
