module Intersection

using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, Distributions, Random
using ReducedGOOP
using TimerOutputs: @timeit, reset_timer!

const TO = ReducedGOOP.TO

abstract type DynamicsModel end
struct PlanarDoubleIntegrator <: DynamicsModel end
struct Unicycle <: DynamicsModel end
struct Bicycle <: DynamicsModel end

include(joinpath(@__DIR__, "plotting.jl"))
include(joinpath(@__DIR__, "dynamics.jl"))

# ── Problem definition ─────────────────────────────────────────────────────────

function get_setup(
    num_players;
    dynamics = build_intersection_dynamics(PlanarDoubleIntegrator()),
    control_bounds = (; lb = [-10.0, -10.0], ub = [10.0, 10.0]),
    planning_horizon = 5,
    collision_avoidance = 1.0,
    speed_limit = 1.5,
    map_end = 7,
    lane_width = 2,
    use_scalarized_baseline = false,
    use_social_equilibrium_baseline = false,
)
    num_players == 2 || error("Intersection setup currently expects exactly two players.")

    state_dimension = dynamics.state_dimension
    control_dimension = dynamics.control_dimension
    dynamics_dimension = state_dimension + control_dimension

    primals_per_player = dynamics_dimension * planning_horizon
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

    function smooth_piecewise_preference_objective(preference, level; ϵ = 0.0)
        ifelse(preference ≥ ϵ, 0.0, (ϵ - preference)^(level + 2))
    end

    trajectory(z; player) =
        unflatten_trajectory(z[Block(player)], state_dimension, control_dimension)

    function lane_bounds(x; player)
        px = x[1]
        py = x[2]
        if player == 1
            return vcat(px + map_end, -px + map_end, py + lane_width, -py + lane_width) # -7 ≤ pₓ ≤ 7, -2 ≤ py ≤ 2
        elseif player == 2
            return vcat(px + lane_width, -px + lane_width, py + map_end, -py + map_end) # -2 ≤ pₓ ≤ 2, -7 ≤ py ≤ 7
        end
        error("Lane bounds are only defined for players 1 and 2.")
    end

    function control_objective(; player)
        function (z, _)
            (; xs, us) = trajectory(z; player)
            sum(sum(u .^ 2) for u in us)
        end
    end

    function goal_objective(; player)
        function (z, θ)
            (; xs, us) = trajectory(z; player)
            (; goal_position) = unflatten_parameters(θ[Block(player)])
            goal_deviation = xs[end][1:2] .- goal_position
            sum(goal_deviation .^ 2) #+ sum(smooth_piecewise_preference_objective.(control_bound_inequality(z, player), player))
        end
    end

    function speed_limit_preference(; player)
        function (z, _)
            (; xs) = trajectory(z; player)
            mapreduce(vcat, 1:length(xs)) do t
                speed_limit_constraints(dynamics.model, xs[t]; speed_limit)
            end
        end
    end

    control_bound_inequality = function (z, i)
        (; lb, ub) = control_bounds
        lb_mask = findall(!isinf, lb)
        ub_mask = findall(!isinf, ub)
        (; xs, us) = trajectory(z; player = i)
        mapreduce(vcat, us) do u
            vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
        end
    end

    function shared_collision_avoidance(z, _)
        trajectories = map(i -> trajectory(z; player = i), 1:num_players)
        xs = map(trajectory -> trajectory.xs, trajectories)
        @assert length(xs) == num_players
        mapreduce(vcat, 2:length(xs[1])) do t
            [sum((xs[1][t][1:2] - xs[2][t][1:2]) .^ 2) - collision_avoidance^2]
        end
    end

    inequality_constraints = [
        function (z, θ)
            (; lb, ub) = control_bounds
            lb_mask = findall(!isinf, lb)
            ub_mask = findall(!isinf, ub)
            (; xs, us) = trajectory(z; player = 1)
            vcat(
                # mapreduce(vcat, us) do u
                # 	vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
                # end,
                # mapreduce(vcat, 1:length(xs)) do t
                # 	lane_bounds(xs[t]; player = 1)
                # end,
                shared_collision_avoidance(z, θ),
            )
        end,
        function (z, θ)
            (; lb, ub) = control_bounds
            lb_mask = findall(!isinf, lb)
            ub_mask = findall(!isinf, ub)
            (; xs, us) = trajectory(z; player = 2)
            vcat(
                # mapreduce(vcat, us) do u
                # 	vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
                # end,
                # mapreduce(vcat, 1:length(xs)) do t
                # 	lane_bounds(xs[t]; player = 2)
                # end,
                shared_collision_avoidance(z, θ),
            )
        end,
    ]

    player_equality_constraints = [
        function (z, θ)
            (; lb, ub) = control_bounds
            lb_mask = findall(!isinf, lb)
            ub_mask = findall(!isinf, ub)
            (; xs, us) = trajectory(z; player = i)
            (; initial_state) = unflatten_parameters(θ[Block(i)])

            initial_state_constraint = xs[1] - initial_state
            dynamics_constraints = mapreduce(vcat, 1:(length(xs) - 1)) do t
                # xs[t] - dynamics(xs[t-1], us[t-1], t)
                dynamics.residual(z[Block(i)], t)
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
                # 	mapreduce(vcat, 1:length(xs)) do t
                # 		px = xs[t][1]
                # 		py = xs[t][2]
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
        (z, θ) -> vcat(
            player_equality_constraints[i](z, θ),
            shared_equality_constraint(z, θ),
        ) for i in 1:num_players
    ]

    preferences = [
        [
            # Minimize control effort 
            control_objective(; player = 1),

            # # Drive under speed limit
            speed_limit_preference(; player = 1),

            # Reach the goal (highest priority for P1)
            goal_objective(; player = 1),

            # Lane bounds + collision avoidance (constraint, both players)
            # inequality_constraints[1],
        ],
        [
            # Minimize control effort
            control_objective(; player = 2),

            # Reach the goal
            goal_objective(; player = 2),

            # Drive under speed limit (highest priority for P2)
            speed_limit_preference(; player = 2),

            # Lane bounds + collision avoidance (constraint, both players)
            # inequality_constraints[2],
        ],
    ]

    # Preference hierarchy: [lowest priority, ..., highest priority]
    is_prioritized_constraint = [[false, true, false], [false, false, true]]

    function build_goop_problem()
        @timeit TO "ParametricGOOP construction" begin
            ReducedGOOP.ParametricGOOP(
                dummy_primals,
                dummy_parameters;
                preferences = use_scalarized_baseline ? scalarized_preferences :
                              preferences,
                is_prioritized_constraint = use_scalarized_baseline ?
                                            scalarized_is_prioritized_constraint :
                                            is_prioritized_constraint,
                equality_constraints,
                inequality_constraints = [nothing, nothing],
                shared_equality_constraint = nothing,
                shared_inequality_constraint = nothing,
            )
        end
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

    " Scalarized baseline (Nash / no hierarchy): flattens hierarchical preferences into a single objective per player"
    scalarized_player_preferences =
        map(scalarized_player_preference, preferences, is_prioritized_constraint)
    scalarized_preferences =
        [[preference] for preference in scalarized_player_preferences]
    scalarized_is_prioritized_constraint = [[false] for _ in scalarized_preferences]

    " Social equilibrium (no Nash / no hierarchy) baseline: sum of all players' preferences, single optimization problem"
    objective = function (z, θ)
        accumulated_objective = 0
        for preference in scalarized_player_preferences
            accumulated_objective += preference(z, θ)
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

    function build_social_problem()
        @timeit TO "ParametricOptimizationProblem construction" begin
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
    end

    # Build problem
    problem = @timeit TO "preference and constraint construction" begin
        use_social_equilibrium_baseline ? build_social_problem() : build_goop_problem()
    end

    (; problem, flatten_parameters)
end

# ── Experiment entry point ─────────────────────────────────────────────────────

function demo(;
    map_end = 7,
    lane_width = 2,
    eta_0 = 5e-5,
    verbose = false,
    rng_seed = 123,
    random_initial_state = false,
    debug = false,
    run_id = "0_IP_test_cleanup_trial_again",
    use_scalarized_baseline = false,
    use_social_equilibrium_baseline = false,
    show_interactive_trajectory = false,
)
    reset_timer!(TO)
    @timeit TO "experiment setup" Random.seed!(rng_seed)

    # ── Settings ───────────────────────────────────────────────────────────────
    dynamics_model = Intersection.PlanarDoubleIntegrator()
    goop_version = :reduced                    # :complete | :reduced | :quasi
    solver = ReducedGOOP.InteriorPoint()
    linesearch = :backtracking          # :backtracking | :fraction_to_boundary
    compute_warmstart = true # Whether to compute a warmstart trajectory via rollout (true) or load from file (false)
    use_marquardt_scaling = true

    # ── Problem parameters ─────────────────────────────────────────────────────
    num_players = 2
    planning_horizon = 12
    collision_avoidance = 1.5
    speed_limit = 2.0
    num_instances = 1
    perturbation_scale = 0.3

    # ── Solver schedule ────────────────────────────────────────────────────────
    epsilon_schedule = [0.3, 0.1]
    max_inner_iters_schedule = fill(1000, length(epsilon_schedule))

    # ── Scenario ───────────────────────────────────────────────────────────────
    # Planar double integrator: state = [px, py, vx, vy]
    base_initial_state1 = [-4.0, -1.0, 3.0, 0.0]
    base_initial_state2 = [1.0, -5.0, 0.0, 1.5]
    # Unicycle: state = [px, py, speed, heading] — uncomment to switch
    # base_initial_state1 = [-6.0, -1.0, 0.0, 0.0]
    # base_initial_state2 = [1.0, -6.0, 1.3, π/2]

    goal_position1 = [6.0, -1.0]
    goal_position2 = [1.0, 6.0]
    obstacle_position = [0.25, 0.15]   # placeholder

    # ── Build dynamics and problem ─────────────────────────────────────────────
    state_dimension = 4
    control_dimension = 2
    Δt = 0.2
    control_bounds = (; lb = [-10.0, -10.0], ub = [10.0, 10.0])

    dynamics = @timeit TO "dynamics construction" build_intersection_dynamics(
        dynamics_model;
        Δt,
        state_dimension,
        control_dimension,
    )

    @timeit TO "problem setup" begin
        (; problem, flatten_parameters) = get_setup(
            num_players;
            dynamics,
            control_bounds,
            planning_horizon,
            collision_avoidance,
            speed_limit,
            map_end,
            lane_width,
            use_scalarized_baseline,
            use_social_equilibrium_baseline,
        )
    end
    kkt_generators = Dict(
        :complete => ReducedGOOP.generate_slacked_complete_kkt_system,
        :reduced => ReducedGOOP.generate_slacked_reduced_kkt_system,
        :quasi => ReducedGOOP.generate_slacked_quasi_kkt_system,
    )

    GOOP_kkt_generator = get(kkt_generators, goop_version, nothing)
    isnothing(GOOP_kkt_generator) && error("Unknown GOOP version: $(goop_version)")

    @info "Building KKT system for $(goop_version) GOOP formulation and $(solver) solver..."
    # Check if problem is not an instance of GOOPKKTSystem. Otherwise, build GOOPKKTSystem.
    GOOP_kkt_system = @timeit TO "KKT construction" begin
        if problem isa ReducedGOOP.GOOPKKTSystem
            problem
        else
            GOOP_kkt_generator(problem)
        end
    end

    println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
    println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)

    dynamics_dimension = dynamics.state_dimension + dynamics.control_dimension
    primal_dimension = dynamics_dimension * planning_horizon

    # ── Per-instance solver ────────────────────────────────────────────────────
    function solve_game_instance(θ; z₀, ϵ₀, max_inner_iters)
        options =
            @timeit TO "solver options construction" ReducedGOOP.InteriorPointOptions(;
                tol = 1e-3, #1e-4
                η₀ = eta_0, # 5e-5, 0.0 to turn off Tikhonov
                ϵ₀,
                max_inner_iters,
                max_outer_iters = 1,
                tightening_rate = 1.2, # high => weak decrease in η
                loosening_rate = 4.0, # low => strong increase in η
                min_stepsize = 1e-20,
                linesearch,
                record_convergence = true,
                record_condition_number = true,
                eta_retry_growth = 0.3,
                tsvd_threshold = 0.0, # 0.0: pure Tikhonov, > 0 and η = 0: pure TSVD
                use_marquardt_scaling = true,
                verbose,
            )

        @info "Solving game instance with $(solver)..."
        kkt_error_history = Float64[]
        condition_number_history = Float64[]
        eta_history = Float64[]
        total_iters = 0
        solver_status = :solved
        elapsed_time = @elapsed begin
            output = @timeit TO "solver invocation" ReducedGOOP.solve(
                solver,
                GOOP_kkt_system,
                θ;
                z₀,
                options,
            )
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
            eta_history =
                hasproperty(output, :eta_history) ? output.eta_history : Float64[]
            if status == :failed
                println(
                    "  [solver exit] total_iters=$(total_iters), kkt_error=$(round(kkt_error; sigdigits=4)), tol=$(options.tol)",
                )
            end
            solver_status = status
        end

        strategies = @timeit TO "solution postprocessing" extract_player_strategies(
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
            "eta_history" => eta_history,
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
        speed_plots_dir,
        control_plots_dir,
        warmstart_plots_dir,
    ) = @timeit TO "output directory setup" prepare_intersection_output_dirs(
        run_id;
        debug,
    )

    # ── Main solve loop ────────────────────────────────────────────────────────
    instance_problem_data = Dict{String,Any}[]
    solved_attempts = 0
    total_attempts = 0

    while solved_attempts < num_instances
        total_attempts += 1
        initial_state1, initial_state2 = if random_initial_state
            (
                sample_initial_state(
                    dynamics_model,
                    base_initial_state1,
                    dynamics.state_dimension,
                    perturbation_scale,
                ),
                sample_initial_state(
                    dynamics_model,
                    base_initial_state2,
                    dynamics.state_dimension,
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

        (; θ1, θ2, θ) =
            @timeit TO "instance parameter construction" build_instance_parameters(
                flatten_parameters,
                initial_state1,
                initial_state2,
                goal_position1,
                goal_position2,
                obstacle_position,
            )

        (; warmstart_solution) = @timeit TO "warmstart construction" if compute_warmstart
            build_default_warmstart(
                planning_horizon,
                dynamics,
                initial_state1,
                initial_state2;
                speed_limit,
            )
        else
            (;
                warmstart_solution = load(
                    "experiments/solution_dict_instance_1_eps0.1.jld2",
                )["single_stored_object"]["x"][1:(num_players * primal_dimension)],
            )
        end
        @timeit TO "warmstart visualization" begin
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
                speed_limit,
                control_bounds,
            )
        end

        epsilon_results = Pair{Float64,Any}[]
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
            # Chain stages: warmstart the next (tighter) ϵ stage from this stage's
            # primal solution instead of the initial rollout.
            stage_warmstart =
                result.solution_dict["x"][1:(num_players * primal_dimension)]
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

        @timeit TO "problem data save" begin
            JLD2.save_object(
                joinpath(
                    problem_data_dir,
                    "problem_data_instance_$(solved_attempts).jld2",
                ),
                instance_problem_data,
            )
        end

        for (ϵ₀, result) in epsilon_results
            solution_dict = result.solution_dict

            @timeit TO "solution output and plotting" begin
                JLD2.save_object(
                    joinpath(
                        solution_data_dir,
                        "solution_dict_instance_$(solved_attempts)_eps$(ϵ₀).jld2",
                    ),
                    solution_dict,
                )
                save_convergence_diagnostics(
                    solution_dict,
                    convergence_plots_dir,
                    solved_attempts,
                    ϵ₀,
                )
                trajectory_fig, _ = plot_intersection_trajectories(;
                    map_end,
                    lane_width,
                    strategy = result.strategies,
                    θ1,
                    θ2,
                    goal_position1,
                    goal_position2,
                )
                speed_fig, _ = speed_plot(;
                    strategy = result.strategies,
                    speed_limit = speed_limit,
                    dynamics_model,
                )
                control_fig, _ = control_plot(;
                    strategy = result.strategies,
                    control_lb = control_bounds.lb,
                    control_ub = control_bounds.ub,
                )

                save_figure(
                    joinpath(
                        trajectory_plots_dir,
                        "trajectory_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
                    ),
                    trajectory_fig,
                )
                save_figure(
                    joinpath(
                        speed_plots_dir,
                        "speed_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
                    ),
                    speed_fig,
                )
                save_figure(
                    joinpath(
                        control_plots_dir,
                        "control_instance_$(solved_attempts)_eps$(ϵ₀).pdf",
                    ),
                    control_fig,
                )
                if show_interactive_trajectory
                    interactive_trajectory_path = joinpath(
                        trajectory_plots_dir,
                        "trajectory_interactive_instance_$(solved_attempts)_eps$(ϵ₀).html",
                    )
                    plot_intersection_trajectories_interactive(;
                        map_end,
                        lane_width,
                        strategy = result.strategies,
                        θ1,
                        θ2,
                        goal_position1,
                        goal_position2,
                        speed_limit,
                        collision_avoidance,
                        display_figure = false,
                        save_path = interactive_trajectory_path,
                    )
                    println(
                        "saved interactive trajectory browser file: ",
                        interactive_trajectory_path,
                    )
                end
            end
        end
    end

    @timeit TO "run metadata save" begin
        JLD2.save_object(
            joinpath(problem_data_dir, "run_metadata.jld2"),
            Dict(
                "run_id" => run_id,
                "debug" => debug,
                "run_dir" => run_dir,
                "rng_seed" => rng_seed,
                "dynamics_model" => dynamics_model_name(dynamics_model),
                "num_instances" => num_instances,
                "random_initial_state" => random_initial_state,
                "use_scalarized_baseline" => use_scalarized_baseline,
                "use_social_equilibrium_baseline" => use_social_equilibrium_baseline,
                "show_interactive_trajectory" => show_interactive_trajectory,
                "epsilon_schedule" => epsilon_schedule,
                "max_inner_iters_schedule" => max_inner_iters_schedule,
                "eta_0" => eta_0,
                "tightening_rate" => tightening_rate,
                "loosening_rate" => loosening_rate,
                "use_marquardt_scaling" => use_marquardt_scaling,
                "speed_limit" => speed_limit,
                "perturbation_scale" => perturbation_scale,
            ),
        )
    end

    println("\nTiming summary:")
    show(TO)
    println()
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
    speed_limit,
    control_bounds,
)
    warmstart_strategies = extract_player_strategies(
        warmstart_solution,
        num_players,
        primal_dimension,
        dynamics,
    )

    warmstart_fig, _ = plot_intersection_trajectories(;
        map_end,
        lane_width,
        strategy = warmstart_strategies,
        θ1,
        θ2,
        goal_position1,
        goal_position2,
    )
    warmstart_speed_fig, _ = speed_plot(;
        strategy = warmstart_strategies,
        speed_limit = speed_limit,
        dynamics_model = dynamics.model,
    )
    warmstart_control_fig, _ = control_plot(;
        strategy = warmstart_strategies,
        control_lb = control_bounds.lb,
        control_ub = control_bounds.ub,
    )

    save_figure(
        joinpath(
            warmstart_plots_dir,
            "warmstart_attempt_$(total_attempts)_instance_$(instance_idx).pdf",
        ),
        warmstart_fig,
    )
    save_figure(
        joinpath(
            warmstart_plots_dir,
            "warmstart_speed_attempt_$(total_attempts)_instance_$(instance_idx).pdf",
        ),
        warmstart_speed_fig,
    )
    save_figure(
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

    data_dir = joinpath(run_dir, "data")
    problem_data_dir = joinpath(data_dir, "problem")
    solution_data_dir = joinpath(problem_data_dir, "solution")
    plots_dir = joinpath(run_dir, "plots")
    trajectory_plots_dir = joinpath(plots_dir, "trajectories")
    convergence_plots_dir = joinpath(plots_dir, "convergence")
    speed_plots_dir = joinpath(plots_dir, "speed")
    control_plots_dir = joinpath(plots_dir, "controls")
    warmstart_plots_dir = joinpath(plots_dir, "warmstart")

    for dir in (
        problem_data_dir,
        solution_data_dir,
        trajectory_plots_dir,
        convergence_plots_dir,
        speed_plots_dir,
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
        speed_plots_dir,
        control_plots_dir,
        warmstart_plots_dir,
    )
end

# ── Internal utilities ─────────────────────────────────────────────────────────

function build_intersection_dynamics(
    model::PlanarDoubleIntegrator;
    Δt = 0.5,
    state_dimension = 4,
    control_dimension = 2,
)
    state_dimension == 4 || error("PlanarDoubleIntegrator expects a 4D state.")
    control_dimension == 2 || error("PlanarDoubleIntegrator expects a 2D control input.")

    residual(z, t) =
        planar_double_integrator(z, t; Δt, state_dimension, control_dimension)
    step(x, u, t) = planar_double_integrator_step(x, u; Δt)
    (; model, residual, step, Δt, state_dimension, control_dimension)
end

function build_intersection_dynamics(
    model::Unicycle;
    Δt = 0.5,
    state_dimension = 4,
    control_dimension = 2,
)
    residual(z, t) = unicycle_dynamics(z, t; Δt, state_dimension, control_dimension)
    step(x, u, t) = unicycle_step(x, u; Δt)
    (; model, residual, step, Δt, state_dimension, control_dimension)
end

function build_intersection_dynamics(
    model::Bicycle;
    Δt = 0.5,
    state_dimension = 4,
    control_dimension = 2,
)
    residual(z, t) = bicycle_dynamics(z, t; Δt, state_dimension, control_dimension)
    step(x, u, t) = bicycle_step(x, u; Δt)
    (; model, residual, step, Δt, state_dimension, control_dimension)
end

function dynamics_model_name(::PlanarDoubleIntegrator)
    "planar_double_integrator"
end

function dynamics_model_name(::Unicycle)
    "unicycle"
end

function dynamics_model_name(::Bicycle)
    "bicycle"
end

function speed_limit_constraints(::PlanarDoubleIntegrator, x; speed_limit = 1.5)
    vx, vy = x[3], x[4]
    # Vehicle speed may not exceed the limit: vx² + vy² ≤ speed_limit².
    # Kept squared so the residual stays smooth at zero velocity.
    return [speed_limit^2 - (vx^2 + vy^2)]
end

function speed_limit_constraints(::Unicycle, x; speed_limit = 1.5)
    speed = x[3]
    return vcat(speed, -speed + speed_limit) # 0 ≤ speed ≤ speed_limit
end

function speed_limit_constraints(::Bicycle, x; speed_limit = 1.5)
    speed = x[3]
    return vcat(speed, -speed + speed_limit) # 0 ≤ speed ≤ speed_limit
end

function sample_initial_state(
    ::PlanarDoubleIntegrator,
    base_initial_state,
    state_dimension,
    perturbation_scale,
)
    noise = rand(Uniform(-perturbation_scale, perturbation_scale), state_dimension)
    return base_initial_state .+ (base_initial_state .!= 0.0) .* noise
end

function sample_initial_state(
    ::Unicycle,
    base_initial_state,
    state_dimension,
    perturbation_scale,
)
    noise = rand(Uniform(-perturbation_scale, perturbation_scale), state_dimension)
    initial_state = copy(base_initial_state)
    initial_state[1:3] .+= noise[1:3] # perturb (px, py, speed) only
    return initial_state
end

function sample_initial_state(
    ::Bicycle,
    base_initial_state,
    state_dimension,
    perturbation_scale,
)
    noise = rand(Uniform(-perturbation_scale, perturbation_scale), state_dimension)
    initial_state = copy(base_initial_state)
    initial_state[1:3] .+= noise[1:3] # perturb (px, py, speed) only
    return initial_state
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
            dynamics.state_dimension,
            dynamics.control_dimension,
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
    speed_limit = 1.5,
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
    # player2_vy_profile = vcat(1.0, fill(speed_limit, planning_horizon - 1))
    # player2_warmstart = build_planar_di_speed_profile_warmstart(
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

function infer_planar_di_timestep(dynamics)
    x0 = [0.0, 0.0, 0.0, 0.0]
    u_unit_y = [0.0, 1.0]
    x1 = dynamics.step(x0, u_unit_y, 1)
    dt = x1[4]
    dt <= 0.0 && error("Failed to infer planar double-integrator timestep from dynamics.")
    dt
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

function build_planar_di_speed_profile_warmstart(
    planning_horizon,
    dynamics,
    initial_state;
    vx_profile,
    vy_profile,
)
    length(initial_state) == dynamics.state_dimension ||
        error("Expected $(dynamics.state_dimension)D planar-double-integrator state.")
    dynamics.control_dimension == 2 ||
        error("Expected 2D control input for planar-double-integrator.")
    length(vx_profile) == planning_horizon ||
        error("vx_profile length must equal planning_horizon.")
    length(vy_profile) == planning_horizon ||
        error("vy_profile length must equal planning_horizon.")

    dt = infer_planar_di_timestep(dynamics)
    x0 = [initial_state[1], initial_state[2], vx_profile[1], vy_profile[1]]
    xs = [x0]
    us = Vector{Vector{Float64}}()
    for t in 1:(planning_horizon - 1)
        u = [
            (vx_profile[t + 1] - vx_profile[t]) / dt,
            (vy_profile[t + 1] - vy_profile[t]) / dt,
        ]
        push!(us, u)
        push!(xs, dynamics.step(xs[t], u, t))
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
        warmstart_primals = mapreduce(vcat, 1:planning_horizon) do t
            vcat(warmstart_x[player][t], warmstart_u[player][t])
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

function save_convergence_diagnostics(
    solution_dict,
    convergence_plots_dir,
    instance_idx,
    ϵ₀;
    filename_suffix = "",
)
    kkt_error_history = get(solution_dict, "kkt_error_history", Float64[])
    if !isempty(kkt_error_history)
        convergence_fig, _ = plot_convergence_plot(;
            kkt_error_history = safe_log10_history(kkt_error_history),
            total_iters = solution_dict["total_iters"],
        )
        save_figure(
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
        save_figure(
            joinpath(
                convergence_plots_dir,
                "condition_number_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
            ),
            condition_number_fig,
        )
    end

    eta_history = get(solution_dict, "eta_history", Float64[])
    if !isempty(eta_history)
        eta_fig, _ = plot_eta_plot(;
            eta_history = safe_log10_history(eta_history),
            total_iters = solution_dict["total_iters"],
        )
        save_figure(
            joinpath(
                convergence_plots_dir,
                "eta_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
            ),
            eta_fig,
        )
    end
end

end
