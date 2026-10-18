# Intersection.jl -- two crossing vehicles, solved with reduced Scholtes GOOP.
# Run from the repository root (including the file alone does not solve):
#
#  julia --project=experiments -e 'include("experiments/Intersection.jl"); Intersection.demo(; debug=true, run_id="intersection_lanes", verbose=true)'
#
# Add plot_fig=false to skip plots; use a new run_id for each experiment.
# State [px,py,vx,vy], control [ax,ay]; default dt=0.2 s, 12 knots (2.2 s).
# P1 goes (-4,-1)->(6,-1); P2 goes (1,-5)->(1,6); the first state is fixed.
# Preferences, LOWEST -> HIGHEST: P1 control -> speed -> goal;
#                               P2 control -> goal -> speed.
# Speed levels minimize sum(preference_slack), with h+slack>=0 and slack>=0.
# Linear vx/vy bounds are +/-speed_limit, scaled by max-min (default 4).
# Hard lanes: P1 |px|<=map_end, |py|<=lane_width; P2 swaps bounds. Rows /full width.
# Hard collision separation is 1.5 m at knots 2:T; controls are not bounded.
# Warm rho chain: rollout first, then previous primals, preference slacks, lambda/psi.
# Gamma/phi, solver slacks, relaxation variables, and eta restart at each rho.
# Defaults: rho=1e-1,...,1e-10, max_inner_iters=500, tol=1e-5, linear_solver=:normal.
# KKT convergence: ||K0||/sqrt(m)<tol; inner convergence: ||R||/sqrt(m)<rho.
# [solver exit] table reports residuals, goal errors, separation, and stop reasons.
# Output: data/Intersection_open_loop/{debug|runs}/<run_id>/plots/.
# One combined PDF per rho includes the trajectory, diagnostics, and warm start.
# Edit get_setup for preferences/constraints, demo for scenario/options; JLD2 data is in data/problem/.

module Intersection

using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, Distributions, Random
using ReducedGOOP
using Printf: @printf
using TimerOutputs: @timeit, reset_timer!

const TO = ReducedGOOP.TO

abstract type DynamicsModel end
struct PlanarDoubleIntegrator <: DynamicsModel end
struct Unicycle <: DynamicsModel end
struct Bicycle <: DynamicsModel end

include(joinpath(@__DIR__, "Intersection_plotting.jl"))
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
    all(v -> isfinite(v) && v > 0, (map_end, lane_width)) ||
        throw(ArgumentError("map_end and lane_width must be finite and positive"))

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
        xmax, ymax = if player == 1
            (map_end, lane_width)
        elseif player == 2
            (lane_width, map_end)
        else
            error("Lane bounds are only defined for players 1 and 2.")
        end
        xmin, ymin = -xmax, -ymax
        xscale, yscale = xmax - xmin, ymax - ymin
        return [
            (px - xmin) / xscale,
            (xmax - px) / xscale,
            (py - ymin) / yscale,
            (ymax - py) / yscale,
        ]
    end

    function lane_bound_inequality(z; player)
        (; xs) = trajectory(z; player)
        mapreduce(x -> lane_bounds(x; player), vcat, xs)
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

    # Each player carries its own lane bounds and the shared collision avoidance
    # as hard inequalities, with its own duals. Speed limits are preferences below.
    inequality_constraints = [
        (z, θ) -> vcat(
            lane_bound_inequality(z; player = i),
            # speed_limit_preference(; player = i)(z, θ),
            shared_collision_avoidance(z, θ),
        ) for i in 1:num_players
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

    equality_constraints = player_equality_constraints

    preferences = [
        [
            # Minimize control effort
            control_objective(; player = 1),

            # Speed limit preference
            speed_limit_preference(; player = 1),

            # Reach the goal (highest priority for P1)
            goal_objective(; player = 1),

        ],
        [
            # Minimize control effort
            control_objective(; player = 2),

            # Reach the goal
            goal_objective(; player = 2),

            # Speed limit preference (highest priority for P2)
            speed_limit_preference(; player = 2),
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
                inequality_constraints,
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
        mapreduce(f -> f(z, θ), vcat, player_equality_constraints)
    end
    inequality_constraint = function (z, θ)
        vcat(
            mapreduce(i -> lane_bound_inequality(z; player = i), vcat, 1:num_players),
            mapreduce(i -> speed_limit_preference(; player = i)(z, θ), vcat, 1:num_players),
            shared_collision_avoidance(z, θ),
        )
    end
    primal_dimension = sum(primal_dimensions)
    parameter_dimension = sum(parameter_dimensions)
    function build_social_problem()
        # One decision maker controls both trajectories. Restore the two vehicle
        # blocks inside the callbacks used by the existing objective/constraints.
        vehicle_blocks(f) = (z, θ) -> f(
            BlockArray(collect(z), primal_dimensions),
            BlockArray(collect(θ), parameter_dimensions),
        )
        @timeit TO "social ParametricGOOP construction" begin
            ReducedGOOP.ParametricGOOP(
                BlockArray(zeros(primal_dimension), [primal_dimension]),
                BlockArray(zeros(parameter_dimension), [parameter_dimension]);
                preferences = [[vehicle_blocks(objective)]],
                is_prioritized_constraint = [[false]],
                equality_constraints = [vehicle_blocks(equality_constraint)],
                inequality_constraints = [vehicle_blocks(inequality_constraint)],
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
    eta_0 = 1e-4,
    verbose = false,
    rng_seed = 123,
    random_initial_state = false,
    debug = false,
    run_id = "ScholtesGOOP version",
    use_scalarized_baseline = false,
    use_social_equilibrium_baseline = false,
    show_interactive_trajectory = false,
    planning_horizon = 12,
    rho_schedule = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
    max_inner_iters = 500,
    linear_solver = :normal,
    tol = 1e-5,
    plot_fig = true,
)
    reset_timer!(TO)
    @timeit TO "experiment setup" Random.seed!(rng_seed)

    # ── Settings ───────────────────────────────────────────────────────────────
    dynamics_model = Intersection.PlanarDoubleIntegrator()
    goop_version = :reduced
    solver = ReducedGOOP.Scholtes()
    compute_warmstart = true # Whether to compute a warmstart trajectory via rollout (true) or load from file (false)

    # ── Problem parameters ─────────────────────────────────────────────────────
    num_players = 2
    collision_avoidance = 1.5
    speed_limit = 2.0
    num_instances = 1
    perturbation_scale = 0.3

    # ── Solver schedule ────────────────────────────────────────────────────────
    isempty(rho_schedule) && throw(ArgumentError("rho_schedule must not be empty"))
    all(ρ -> isfinite(ρ) && ρ > 0, rho_schedule) ||
        throw(ArgumentError("rho_schedule entries must be finite and positive"))

    # ── Scenario ───────────────────────────────────────────────────────────────
    # Planar double integrator: state = [px, py, vx, vy]
    base_initial_state1 = [-4.0, -1.0, 1.99, 0.0]
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
    @info "Building KKT system for $(goop_version) GOOP formulation and $(solver) solver..."
    GOOP_kkt_system = @timeit TO "KKT construction" begin
        ReducedGOOP.generate_slacked_reduced_kkt_system(
            problem; complementarity = :scholtes, codegen = :fast_differentiation,
        )
    end
    m = GOOP_kkt_system.n_nc + 2GOOP_kkt_system.n_comp
    residual_scale = sqrt(m)
    eq_cols = findall(
        v -> startswith(string(v), "lam_") || startswith(string(v), "psi_"),
        GOOP_kkt_system.vars,
    )
    residuals = ReducedGOOP.ScholtesResiduals(GOOP_kkt_system)
    println("[Primal-Dual] KKT Dimension: ", m)
    println("[Primal-Dual] variable Dimension: ",
        GOOP_kkt_system.n + GOOP_kkt_system.n_comp +
        count(!iszero, GOOP_kkt_system.u_index))

    dynamics_dimension = dynamics.state_dimension + dynamics.control_dimension
    primal_dimension = dynamics_dimension * planning_horizon

    # ── Per-instance solver ────────────────────────────────────────────────────
    function solve_game_instance(θ; z₀, w₀, ρ)
        options =
            @timeit TO "solver options construction" ReducedGOOP.ScholtesOptions(;
                rho_schedule = [Float64(ρ)],
                max_inner = max_inner_iters,
                tol = tol * residual_scale,
                tol_inner = ρ * residual_scale,
                projected_step = true,
                proj_eps = 1e-8,
                linear_solver,
                eta_schedule = true,
                eta_init = eta_0,
                eta_min = 1e-8,
                eta_max = 1e4,
                gain_low = 0.25,
                gain_high = 0.75,
                tightening_rate = 1.2,
                loosening_rate = 3.0,
                verbose,
            )

        @info "Solving game instance with $(solver)..."
        kkt_error_history = Float64[]
        eta_history = Float64[]
        trace = nt -> begin
            push!(kkt_error_history, ReducedGOOP._norm_K0(residuals, nt.r, nt.w, nt.rho))
            push!(eta_history, nt.eta)
        end
        elapsed_time = @elapsed begin
            output = @timeit TO "solver invocation" ReducedGOOP.solve(
                solver,
                GOOP_kkt_system,
                θ;
                z₀,
                w₀,
                options,
                trace,
            )
        end
        stop_reason = solver_stop_reason(output, options; scale = residual_scale)
        push!(kkt_error_history, output.residual)
        push!(eta_history, output.eta)

        strategies = @timeit TO "solution postprocessing" extract_player_strategies(
            output.z,
            num_players,
            primal_dimension,
            dynamics,
        )

        solution_dict = Dict(
            "strategies" => strategies,
            "warmstart_strategies" => extract_player_strategies(
                z₀, num_players, primal_dimension, dynamics,
            ),
            "z" => output.z,
            "w" => output.w,
            "preference_slacks" => output.preference_slacks,
            "x" => output.z, # Primal compatibility for existing trajectory readers.
            "solve_time_sec" => elapsed_time,
            "kkt_error" => output.residual,
            "kkt_error_per_equation" => output.residual / residual_scale,
            "rho" => output.rho,
            "status" => output.status,
            "termination_reason" => output.termination_reason,
            "stop_reason" => stop_reason,
            "relaxed_residual" => output.relaxed_residual,
            "total_iters" => output.iters,
            "kkt_error_history" => kkt_error_history,
            "residual_history" => output.history,
            "eta_history" => eta_history,
            "solver_options" => options,
            "speed_constraint" => :component,
        )

        (; strategies, solution_dict)
    end

    # ── Output directories ─────────────────────────────────────────────────────
    (;
        run_dir,
        problem_data_dir,
        solution_data_dir,
        plots_dir,
    ) = @timeit TO "output directory setup" prepare_intersection_output_dirs(
        run_id;
        debug,
    )

    # ── Main solve loop ────────────────────────────────────────────────────────
    instance_problem_data = Dict{String,Any}[]
    stage_plots = NamedTuple[]
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
                    "experiments/solution_dict_instance_1_rho1.0e-5.jld2",
                )["single_stored_object"]["x"][1:(num_players * primal_dimension)],
            )
        end
        rho_results = Pair{Float64,Any}[]
        stage_warmstart = warmstart_solution
        previous_w = nothing
        stage_preference_slacks = nothing
        solve_sequence_succeeded = false
        instance_total_solve_time_sec = 0.0

        for ρ in rho_schedule
            # The rollout can violate collision avoidance. Initialize positive
            # solver slacks even then, and retain λ/ψ for subsequent ρ stages,
            # as in the robotic-arm example's warm-z+eq continuation.
            eq = isnothing(previous_w) ? zeros(length(eq_cols)) : previous_w[eq_cols]
            w₀ = ReducedGOOP.scholtes_warm_start(
                GOOP_kkt_system, θ, stage_warmstart, ρ; eq_cols, eq,
                preference_slacks = stage_preference_slacks,
            )
            result = solve_game_instance(θ; z₀ = stage_warmstart, w₀, ρ)
            push!(rho_results, Float64(ρ) => result)
            instance_total_solve_time_sec += result.solution_dict["solve_time_sec"]
            solve_sequence_succeeded |= result.solution_dict["status"] == :solved
            # A coarse stage can miss the unperturbed KKT tolerance while still
            # providing a useful start for a finer stage; continue the schedule.
            stage_warmstart = result.solution_dict["z"]
            previous_w = result.solution_dict["w"]
            stage_preference_slacks = result.solution_dict["preference_slacks"]
        end

        print_intersection_results(rho_results;
            instance = solved_attempts + 1, attempt = total_attempts,
            residual_scale,
            goal_position1, goal_position2,
        )

        if plot_fig
            for (ρ, result) in rho_results
                page = (;
                    solution = result.solution_dict, rho = ρ,
                    instance = solved_attempts + 1, attempt = total_attempts,
                    warmstart_strategies = result.solution_dict["warmstart_strategies"],
                    map_end, lane_width, θ1, θ2,
                    goal_position1, goal_position2, speed_limit, collision_avoidance,
                    dynamics_model, timestep = Δt, tol, residual_scale,
                )
                report_path = joinpath(plots_dir,
                    "Intersection_instance_$(solved_attempts + 1)_rho$(ρ)_attempt_$(total_attempts).pdf")
                push!(stage_plots, (; path = report_path, page))
            end
        end

        if !solve_sequence_succeeded
            failed_instance_idx = solved_attempts + 1
            for (ρ, result) in rho_results
                failed_suffix = "_attempt_$(total_attempts)_failed"
                JLD2.save_object(
                    joinpath(
                        solution_data_dir,
                        "solution_dict_instance_$(failed_instance_idx)_rho$(ρ)$(failed_suffix).jld2",
                    ),
                    result.solution_dict,
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

        @timeit TO "problem data save" begin
            JLD2.save_object(
                joinpath(
                    problem_data_dir,
                    "problem_data_instance_$(solved_attempts).jld2",
                ),
                instance_problem_data,
            )
        end

        for (ρ, result) in rho_results
            solution_dict = result.solution_dict
            @timeit TO "solution data save" begin
                JLD2.save_object(
                    joinpath(
                        solution_data_dir,
                        "solution_dict_instance_$(solved_attempts)_rho$(ρ).jld2",
                    ),
                    solution_dict,
                )
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
                "solver" => "Scholtes",
                "complementarity" => "scholtes",
                "rho_schedule" => rho_schedule,
                "max_inner_iters" => max_inner_iters,
                "linear_solver" => linear_solver,
                "tol_per_equation" => tol,
                "plot_fig" => plot_fig,
                "eta_0" => eta_0,
                "eta_schedule" => true,
                "tightening_rate" => 1.2,
                "loosening_rate" => 3.0,
                "speed_limit" => speed_limit,
                "speed_constraint" => :component,
                "velocity_min" => -speed_limit,
                "velocity_max" => speed_limit,
                "velocity_scale" => 2 * speed_limit,
                "perturbation_scale" => perturbation_scale,
            ),
        )
    end

    # Save solve data and metadata before rendering, so a plotting error cannot lose them.
    for (; path, page) in stage_plots
        @timeit TO "combined stage plotting" save_figure(path, intersection_report_figure(page))
        println("saved combined stage plot: ", path)
        if show_interactive_trajectory
            plot_intersection_trajectories_interactive(;
                map_end = page.map_end, lane_width = page.lane_width,
                strategy = page.solution["strategies"], θ1 = page.θ1, θ2 = page.θ2,
                goal_position1 = page.goal_position1, goal_position2 = page.goal_position2,
                collision_avoidance = page.collision_avoidance,
                speed_limit = nothing, display_figure = false,
                save_path = replace(path, ".pdf" => ".html"),
            )
        end
    end

    println("\nTiming summary:")
    show(TO)
    println()
end

# ── Output / plotting helpers ──────────────────────────────────────────────────

"Physical goal errors and minimum separation for the result table."
function intersection_result_metrics(strategy;
    goal_position1, goal_position2,
)
    goals = (goal_position1, goal_position2)
    goal = [sqrt(sum(abs2, strategy[i].xs[end][1:2] - goals[i])) for i in 1:2]
    n = min(length(strategy[1].xs), length(strategy[2].xs))
    min_sep = minimum(sqrt(sum(abs2, strategy[1].xs[t][1:2] - strategy[2].xs[t][1:2])) for t in 1:n)
    (; goal, min_sep)
end

"Print every rho, including unsuccessful stages, in the overtake example's table style."
function print_intersection_results(rho_results; instance, attempt, residual_scale,
    goal_position1, goal_position2,
    io = stdout,
)
    println(io, "\n[solver exit] instance $(instance), attempt $(attempt): warm rho chain")
    println(io, "Residuals /sqrt(m); goal errors and minimum separation in m.")
    @printf(io, "%-9s %-7s %6s %8s %12s %12s %8s %8s %8s  %s\n",
        "rho", "status", "iters", "time_s", "R/sqrt(m)", "K0/sqrt(m)",
        "goal1", "goal2", "min_sep", "stop")
    for (ρ, result) in rho_results
        row = result.solution_dict
        metrics = intersection_result_metrics(row["strategies"];
            goal_position1, goal_position2,
        )
        @printf(io, "%-9.0e %-7s %6d %8.3f %12.2e %12.2e %8.3f %8.3f %8.3f  %s\n",
            ρ, string(row["status"]), row["total_iters"], row["solve_time_sec"],
            row["relaxed_residual"] / residual_scale, row["kkt_error"] / residual_scale,
            metrics.goal...,
            metrics.min_sep, row["stop_reason"])
    end
    println(io)
    return nothing
end

function solver_stop_reason(output, options; scale = 1.0)
    k0_name = scale == 1 ? "||K₀||" : "||K₀||/√m"
    r_name = scale == 1 ? "||R||₂" : "||R||₂/√m"
    reason = output.termination_reason
    reason == :tolerance_reached &&
        return "KKT tolerance reached ($(k0_name) < $(round(options.tol / scale; sigdigits = 3)))"
    reason == :inner_tolerance_reached &&
        return "relaxed-system tolerance reached ($(r_name) < $(round(options.tol_inner / scale; sigdigits = 3)))"
    reason == :max_inner && return "iteration limit reached (max_inner=$(options.max_inner))"
    reason == :nonfinite_residual && return "solver failure: non-finite residual"
    reason == :invalid_step && return "solver failure: invalid Newton step"
    reason == :line_search_failed && return "line-search stall (no accepted step)"
    reason == :no_iterations && return "no iterations"
    return string(reason)
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
    for dir in (problem_data_dir, solution_data_dir, plots_dir)
        mkpath(dir)
    end
    (; run_dir, problem_data_dir, solution_data_dir, plots_dir)
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
    isfinite(speed_limit) && speed_limit > 0 ||
        throw(ArgumentError("speed_limit must be finite and positive"))
    vx, vy = x[3], x[4]
    velocity_min, velocity_max = -speed_limit, speed_limit
    velocity_scale = velocity_max - velocity_min
    # Scaled component bounds; each row is interpreted as h ≥ 0.
    return [
        (vx - velocity_min) / velocity_scale,
        (velocity_max - vx) / velocity_scale,
        (vy - velocity_min) / velocity_scale,
        (velocity_max - vy) / velocity_scale,
    ]
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

end
