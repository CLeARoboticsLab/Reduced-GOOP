# julia --project=experiments -e 'include("experiments/MonteCarlo_Intersection.jl"); MonteCarlo_Intersection.run()'
module MonteCarlo_Intersection

using BlockArrays,
    LinearAlgebra, SparseArrays, Random, Statistics, Serialization, Dates, Printf, SHA
import ReducedGOOP
import NonlinearSolve as NS
import Symbolics
import CairoMakie as CM
include(joinpath(@__DIR__, "Intersection.jl"))

# 1. Problem definition

function objective_variants(settings)
    # 2a²(cosh(d/a)-1) matches d² in value, gradient, and curvature at zero.
    # a controls growth: control scale 1.5 m/s², goal scale 6 m; these are not error scales.
    [
        (;
            name = :quadratic,
            label = "Quadratic",
            objective = (d, a) -> d^2,
            control_scale = settings.control_shape_scale,
            goal_scale = settings.goal_shape_scale,
        ),
        (;
            name = :cosh,
            label = "Scaled cosh",
            objective = (d, a) -> 2a^2 * (cosh(d/a)-1),
            control_scale = settings.control_shape_scale,
            goal_scale = settings.goal_shape_scale,
        ),
    ]
end

function get_setup(
    num_players;
    dynamics,
    control_bounds,
    planning_horizon,
    collision_avoidance,
    min_speed,
    max_speed,
    map_end,
    lane_width,
    variant,
)
    (; problem, flatten_parameters) = Intersection.get_setup(
        num_players;
        dynamics,
        control_bounds,
        planning_horizon,
        collision_avoidance,
        min_speed,
        max_speed,
        map_end,
        lane_width,
    )
    state_dimension = dynamics.state_dimension
    control_dimension = dynamics.control_dimension
    trajectory(z; player) = Intersection.unflatten_trajectory(
        z[Block(player)],
        state_dimension,
        control_dimension,
    )
    unflatten_parameters = function (θ)
        θ_iter = Iterators.Stateful(θ)
        initial_state = first(θ_iter, state_dimension)
        goal_position = first(θ_iter, 2)
        obstacle_position = first(θ_iter, 2)
        (; initial_state, goal_position, obstacle_position)
    end

    function control_objective(; player)
        function (z, _)
            (; xs, us) = trajectory(z; player)
            sum(variant.objective(u[j], variant.control_scale) for u in us for j in 1:2)
        end
    end
    function goal_objective(; player)
        function (z, θ)
            (; xs, us) = trajectory(z; player)
            (; goal_position) = unflatten_parameters(θ[Block(player)])
            goal_deviation = xs[end][1:2] .- goal_position
            sum(variant.objective(goal_deviation[j], variant.goal_scale) for j in 1:2)
        end
    end

    preferences = [Function[p...] for p in problem.preferences]
    if variant.name !== :quadratic
        preferences[1][1] = control_objective(; player = 1)
        preferences[1][3] = goal_objective(; player = 1)
        preferences[2][1] = control_objective(; player = 2)
        preferences[2][2] = goal_objective(; player = 2)
    end
    problem = ReducedGOOP.ParametricGOOP(;
        preferences,
        problem.is_prioritized_constraint,
        problem.equality_constraints,
        problem.inequality_constraints,
        problem.primal_dims,
        problem.parameter_dims,
        problem.equality_dims,
        problem.inequality_dims,
        problem.num_players,
    )
    (; problem, flatten_parameters)
end

# 2. Experiment settings

function run(; overrides = (;))
    settings = merge(
        (;
            trials = 2,
            seed = 123,
            T = 7,
            dt = 0.5,
            objective_variants = (:quadratic, :cosh),
            complete_formulations = (:aggregate, :exact),
            comparison_modes = (:fixed_primal, :warm_start),
            base_states = ([-6.0, -1.0, 1.5, 0.0], [1.0, -5.0, 0.0, 2.0]),
            goals = ([6.0, -1.0], [1.0, 6.0]),
            obstacle = [0.25, 0.15],
            sampling_halfwidths = ([0.3, 0.3, 0.15, 0.0], [0.3, 0.3, 0.0, 0.15]),
            sampling_distribution = :uniform,
            initial_lane_margin = 0.05,
            initial_separation_margin = 0.05,
            max_sampling_attempts = 1000,
            map_end = 7.0,
            lane_width = 2.0,
            separation = 1.3,
            min_speed = 0.7,
            max_speed = 3.0,
            control_bounds = (; lb = [-3.0, -3.0], ub = [3.0, 3.0]),
            control_shape_scale = 1.5,
            goal_shape_scale = 6.0,
            ρ_r = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            ρ_c = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            max_inner = 500,
            NS_iters = 1000,
            reduced_tol = 1e-5,
            complete_tol = 1e-4,
            fixed_tol = 1e-4,
            component_tol = 1e-3,
            physical_tol = 1e-4,
            sign_tol = 1e-12,
            linear_solver = :normal,
            eta_init = 1e-4,
            retry_eta = [1e-2],
            positive_seed = 1.0,
            positive_floor = 1e-10,
            error_scales = (; position = 1.0, velocity = 1.0, control = 1.0),
            small_rms = 0.01,
            small_max = 0.05,
            output_dir = joinpath(
                dirname(@__DIR__),
                "output",
                "intersection_kkt_comparison",
            ),
        ),
        overrides,
    )
    0 < settings.fixed_tol <= 1e-4 || error("fixed_tol must lie in (0, 1e-4]")
    settings.trials > 0 && settings.T >= 2 || error("positive trials and T >= 2 required")
    all(ρ -> isfinite(ρ) && ρ > 0, [settings.ρ_r; settings.ρ_c]) ||
        error("invalid relaxation schedule")
    isempty(settings.ρ_r) || isempty(settings.ρ_c) ? error("empty schedule") : nothing
    mkpath(settings.output_dir)
    directory = mktempdir(
        settings.output_dir;
        prefix = Dates.format(now(), "yyyymmdd_HHMMSS") * "_",
        cleanup = false,
    )
    rng = Xoshiro(settings.seed)
    instances, sampling = sample_instances(rng, settings)
    results, metadata = Any[], Any[]
    save_results(directory, settings, instances, sampling, metadata, results)
    length(instances) == settings.trials || error(
        "Sampling exhausted $(settings.max_sampling_attempts) attempts; rejection records saved in $directory",
    )
    println(
        "Diagnostic run: $(settings.trials) instances, planning horizon = $(settings.T), ",
        "$((settings.T-1)*settings.dt) seconds. Output: $directory",
    )
    variants =
        filter(v -> v.name in settings.objective_variants, objective_variants(settings))
    modes = [
        (;
            name = :fixed_primal,
            label = "Fixed-primal recovery",
            solver = :nonlinear_solve,
        ),
        (; name = :warm_start, label = "Warm-started complete", solver = :scholtes),
    ]
    modes = filter(m -> m.name in settings.comparison_modes, modes)
    builders = [
        (; name = :aggregate, build = ReducedGOOP.generate_scholtes_complete_kkt_system),
        (; name = :exact, build = ReducedGOOP.generated_exact_complete_kkt_system),
    ]
    builders = filter(b -> b.name in settings.complete_formulations, builders)
    total = length(variants)*length(instances)*length(builders)*length(modes)
    done = 0
    num_players = 2
    num_instances = settings.trials
    dynamics_model = Intersection.PlanarDoubleIntegrator()
    planning_horizon = settings.T
    time_step = settings.dt
    state_dimension = 4
    control_dimension = 2
    Δt = time_step
    dynamics_dimension = state_dimension + control_dimension
    primal_dimension = dynamics_dimension * planning_horizon
    control_bounds = settings.control_bounds
    collision_avoidance = settings.separation
    min_speed, max_speed = settings.min_speed, settings.max_speed
    map_end, lane_width = settings.map_end, settings.lane_width
    goal_position1, goal_position2 = settings.goals
    obstacle_position = settings.obstacle
    dynamics = Intersection.build_intersection_dynamics(
        dynamics_model;
        Δt,
        state_dimension,
        control_dimension,
    )
    for variant in variants
        (; problem, flatten_parameters) = get_setup(
            num_players;
            dynamics,
            control_bounds,
            planning_horizon,
            collision_avoidance,
            min_speed,
            max_speed,
            map_end,
            lane_width,
            variant,
        )
        println("Building reduced KKT system for $(variant.name) variant...");
        flush(stdout)
        kkt_construction = @timed ReducedGOOP.generate_slacked_reduced_kkt_system(
            problem;
            complementarity = :scholtes,
            codegen = :fast_differentiation,
        )
        GOOP_kkt_system = kkt_construction.value
        formulations = map(builders) do builder
            println(
                "Building $(builder.name) complete KKT system for $(variant.name) variant...",
            );
            flush(stdout)
            built = @timed builder.build(problem)
            kkt = built.value
            slack_targets = compile_slack_targets(kkt)
            fixed = fixed_primal_formulation(kkt)
            mapping = variable_mapping(GOOP_kkt_system, kkt, problem)
            builder.name === :exact && @assert kkt.n == kkt.n_nc
            @assert size(fixed.J) == (kkt.n_nc, kkt.n-length(kkt.primal_dims))
            (;
                builder.name,
                kkt,
                slack_targets,
                fixed,
                mapping,
                construction_seconds = built.time,
            )
        end
        push!(
            metadata,
            (;
                variant = variant.name,
                label = variant.label,
                objective = (; variant.control_scale, variant.goal_scale),
                reduced = (;
                    rows = GOOP_kkt_system.n_nc+2GOOP_kkt_system.n_comp,
                    columns = GOOP_kkt_system.n+GOOP_kkt_system.n_comp+ReducedGOOP._n_u(
                        GOOP_kkt_system,
                    ),
                    construction_seconds = kkt_construction.time,
                ),
                complete = [
                    (; p.name, p.fixed.dimensions, p.mapping, p.construction_seconds) for
                    p in formulations
                ],
            ),
        )
        warmup = instances[1]
        initial_state1, initial_state2 = warmup.states
        (; θ1, θ2, θ) = Intersection.build_instance_parameters(
            flatten_parameters,
            initial_state1,
            initial_state2,
            goal_position1,
            goal_position2,
            obstacle_position,
        )
        (; warmstart_solution) = Intersection.build_yielding_warmstart(
            planning_horizon,
            dynamics,
            initial_state1,
            initial_state2,
            goal_position1;
            min_speed,
            max_speed,
        )
        println(
            "Untimed warm-up: $(variant.name), reduced and every selected complete formulation...",
        );
        flush(stdout)
        warm_reduced =
            solve_reduced(GOOP_kkt_system, problem, θ, warmstart_solution, settings)
        warm_physical = warm_reduced.output.z
        for formulation in formulations, mode in modes
            println("Warm-up: $(variant.name), $(formulation.name), $(mode.name)");
            flush(stdout)
            initial_complete =
                complete_initialization(formulation, warm_reduced.output, θ, settings)
            solve_complete(
                formulation,
                mode,
                problem,
                θ,
                warm_physical,
                initial_complete,
                settings;
                retry_values = Float64[],
            )
        end
        # Warm-up results are discarded; every measured trial rebuilds its starts.
        @info "Starting Monte Carlo simulation..." num_instances
        for (trial, instance) in enumerate(instances)
            println("Trial $trial/$num_instances: solving reduced $(variant.name) system");
            initial_state1, initial_state2 = instance.states
            println("  Player 1 initial state [px, py, vx, vy]: $initial_state1")
            println("  Player 2 initial state [px, py, vx, vy]: $initial_state2")
            flush(stdout)
            (; θ1, θ2, θ) = Intersection.build_instance_parameters(
                flatten_parameters,
                initial_state1,
                initial_state2,
                goal_position1,
                goal_position2,
                obstacle_position,
            )
            (; warmstart_solution) = Intersection.build_yielding_warmstart(
                planning_horizon,
                dynamics,
                initial_state1,
                initial_state2,
                goal_position1;
                min_speed,
                max_speed,
            )
            @info "Solving reduced system for trial $trial" variant.name
            reduced_sol =
                solve_reduced(GOOP_kkt_system, problem, θ, warmstart_solution, settings)
            println(
                "Reduced solve finished: status=$(reduced_sol.output.status), reason=$(reduced_sol.output.termination_reason), accepted=$(reduced_sol.accepted)",
            )
            flush(stdout)

            strategies = Intersection.extract_player_strategies(
                reduced_sol.output.z,
                num_players,
                primal_dimension,
                dynamics,
            )
            checks = Any[]
            record = (;
                trial,
                variant = variant.name,
                θ,
                initial = copy(warmstart_solution),
                strategies,
                reduced = reduced_sol,
                checks,
            )
            push!(results, record)
            save_results(directory, settings, instances, sampling, metadata, results)
            println("Starting complete KKT system for trial $trial...")
            for formulation in formulations, mode in modes # formulations = (:aggregate, :exact), modes = (:fixed_primal, :warmstart)
                if reduced_sol.accepted
                    println(
                        "Trial $trial: $(variant.name), $(formulation.name), $(mode.name)",
                    );
                    flush(stdout)
                    start = complete_initialization(
                        formulation,
                        reduced_sol.output,
                        θ,
                        settings,
                    )
                    result = try
                        solve_complete(
                            formulation,
                            mode,
                            problem,
                            θ,
                            reduced_sol.output.z,
                            start,
                            settings,
                        )
                    catch error
                        validation = merge(
                            reduced_sol.validation,
                            (; accepted = false, relaxed_rms = NaN, unrelaxed_rms = NaN),
                        )
                        (;
                            accepted = false,
                            attempts = Any[],
                            validation,
                            physical = copy(reduced_sol.output.z),
                            solution = start,
                            status = :exception,
                            reason = :exception,
                            error = sprint(showerror, error, catch_backtrace()),
                            iterations = 0,
                            seconds = 0.0,
                            compile_seconds = 0.0,
                        )
                    end
                    metrics = trajectory_metrics(
                        reduced_sol.output.z,
                        result.physical,
                        settings,
                    )
                    category =
                        !result.accepted ? :numerical_failure :
                        mode.name === :fixed_primal ? :fixed_primal_compatible :
                        metrics.small ? :small_adjustment : :substantial_adjustment
                    check = (;
                        formulation = formulation.name,
                        mode = mode.name,
                        solver = mode.solver,
                        result,
                        metrics,
                        category,
                    )
                    push!(checks, check)
                    if result.accepted
                        @printf(
                            "\ntrial=%d %s %s %s: iters=%d/%d R=%.2e/%.2e K0=%.2e/%.2e time=%.3f/%.3fs RMS=%.2e max=%.2e [%s]\n",
                            trial,
                            variant.name,
                            formulation.name,
                            mode.name,
                            reduced_sol.iterations,
                            result.iterations,
                            reduced_sol.validation.relaxed_rms,
                            result.validation.relaxed_rms,
                            reduced_sol.validation.unrelaxed_rms,
                            result.validation.unrelaxed_rms,
                            reduced_sol.seconds,
                            result.seconds,
                            metrics.rms,
                            metrics.maximum,
                            category
                        )
                    end
                else
                    push!(
                        checks,
                        (;
                            formulation = formulation.name,
                            mode = mode.name,
                            solver = mode.solver,
                            category = :reduced_failed,
                            reason = "Final reduced solution did not pass acceptance",
                        ),
                    )
                end
                done += 1
                filled = floor(Int, 24done/total)
                print(
                    "\r[",
                    repeat("=", filled),
                    repeat(" ", 24-filled),
                    "] $done/$total checks",
                );
                flush(stdout)
                save_results(directory, settings, instances, sampling, metadata, results)
            end
        end
    end
    println()
    open(joinpath(directory, "final_summary.txt"), "w") do io
        print_final_summary(results, settings; io)
    end
    plot_results(directory)
    println("Saved diagnostic results and plots: $directory")
    print_final_summary(results, settings)
    (; directory, settings, instances, results, metadata)
end

# 3. Initialization and Monte Carlo execution

function sample_instances(rng, settings)
    settings.sampling_distribution === :uniform || error("Supported sampler: :uniform")
    instances, attempts = Any[], Any[]
    for attempt in 1:settings.max_sampling_attempts
        states = [
            base .+ width .* (2rand(rng, 4) .- 1) for
            (base, width) in zip(settings.base_states, settings.sampling_halfwidths)
        ]
        lane_margins = [
            minimum((
                settings.map_end-abs(states[i][i]),
                settings.lane_width-abs(states[i][3 - i]),
            )) for i in 1:2
        ]
        separation_margin = norm(states[1][1:2]-states[2][1:2])-settings.separation
        accepted =
            minimum(lane_margins) > settings.initial_lane_margin &&
            separation_margin > settings.initial_separation_margin
        push!(
            attempts,
            (;
                attempt,
                states,
                lane_margins,
                separation_margin,
                accepted,
                reason = accepted ? "accepted" : "initial lane/separation strict margin",
            ),
        )
        accepted && push!(instances, (; states, sampling_attempt = attempt))
        length(instances) == settings.trials && return instances, attempts
    end
    instances, attempts
end

function print_kkt_header(label; io = stdout)
    println(io, "\n[solver exit] $label: warm rho chain")
    println(io, "Residuals /sqrt(m); goal errors and minimum separation in m.")
    @printf(
        io,
        "%-9s %-7s %6s %8s %12s %12s %8s %8s %8s  %s\n",
        "rho",
        "status",
        "iters",
        "time_s",
        "R/sqrt(m)",
        "K0/sqrt(m)",
        "goal1",
        "goal2",
        "min_sep",
        "stop"
    )
    flush(io)
end

function print_kkt_stage(;
    ρ,
    status,
    iterations,
    seconds,
    relaxed_rms,
    unrelaxed_rms,
    physical,
    settings,
    stop,
    io = stdout,
)
    strategies = Intersection.extract_player_strategies(
        physical,
        2,
        6settings.T,
        (; state_dimension = 4, control_dimension = 2),
    )
    metrics = Intersection.intersection_result_metrics(
        strategies;
        goal_position1 = settings.goals[1],
        goal_position2 = settings.goals[2],
    )
    @printf(
        io,
        "%-9.0e %-7s %6d %8.3f %12.2e %12.2e %8.3f %8.3f %8.3f  %s\n",
        ρ,
        string(status),
        iterations,
        seconds,
        relaxed_rms,
        unrelaxed_rms,
        metrics.goal...,
        metrics.min_sep,
        stop
    )
    flush(io)
end

function solve_reduced(GOOP_kkt_system, problem, θ, warmstart_solution, settings)
    solver = ReducedGOOP.Scholtes()
    max_inner_iters = settings.max_inner
    tol = settings.reduced_tol
    eta_0 = settings.eta_init
    linear_solver = settings.linear_solver
    verbose = false
    rho_schedule = settings.ρ_r
    m = GOOP_kkt_system.n_nc + 2GOOP_kkt_system.n_comp
    residual_scale = sqrt(m)
    eq_cols = findall(
        v -> startswith(string(v), "lam_") || startswith(string(v), "psi_"),
        GOOP_kkt_system.vars,
    )
    residuals = ReducedGOOP.ScholtesResiduals(GOOP_kkt_system)
    stage_warmstart = warmstart_solution
    previous_w = nothing
    stage_preference_slacks = nothing
    stages = Any[]
    output = nothing
    print_kkt_header("reduced KKT, attempt 1")
    for ρ in rho_schedule
        eq = isnothing(previous_w) ? zeros(length(eq_cols)) : previous_w[eq_cols]
        w₀ = ReducedGOOP.scholtes_warm_start(
            GOOP_kkt_system,
            θ,
            stage_warmstart,
            ρ;
            eq_cols,
            eq,
            preference_slacks = stage_preference_slacks,
        )
        z₀ = stage_warmstart
        options = ReducedGOOP.ScholtesOptions(;
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
        kkt_error_history = Float64[]
        eta_history = Float64[]
        relaxed_residual_history = Float64[]
        trace =
            nt -> begin
                push!(kkt_error_history, ReducedGOOP._norm_K0(residuals, nt.r, nt.w, nt.rho))
                push!(eta_history, nt.eta)
                push!(relaxed_residual_history, norm(nt.r))
            end
        timed =
            @timed ReducedGOOP.solve(solver, GOOP_kkt_system, θ; z₀, w₀, options, trace)
        output = timed.value
        push!(kkt_error_history, output.residual)
        push!(eta_history, output.eta)
        push!(relaxed_residual_history, output.relaxed_residual)
        history = [
            (; raw, unrelaxed, eta) for (raw, unrelaxed, eta) in
            zip(relaxed_residual_history, kkt_error_history, eta_history)
        ]
        push!(
            stages,
            (;
                ρ,
                parameter = :rho_reduced,
                status = output.status,
                reason = output.termination_reason,
                iterations = output.iters,
                seconds = max(0.0, timed.time-timed.compile_time-timed.recompile_time),
                wall_seconds = timed.time,
                compile_seconds = timed.compile_time+timed.recompile_time,
                history,
                initial = copy(w₀),
                final = copy(output.w),
            ),
        )
        print_kkt_stage(;
            ρ,
            status = output.status,
            iterations = output.iters,
            seconds = last(stages).seconds,
            relaxed_rms = output.relaxed_residual/residual_scale,
            unrelaxed_rms = output.residual/residual_scale,
            physical = output.z,
            settings,
            stop = Intersection.solver_stop_reason(
                output,
                options;
                scale = residual_scale,
            ),
        )
        stage_warmstart = output.z
        previous_w = output.w
        stage_preference_slacks = output.preference_slacks
    end
    r = zeros(m)
    b = zeros(GOOP_kkt_system.n_comp)
    ρ = last(rho_schedule)
    ReducedGOOP._resid!(residuals, r, b, output.w, ρ)
    relaxed_rms = norm(r)/residual_scale
    unrelaxed_rms = ReducedGOOP._norm_K0(residuals, r, output.w, ρ)/residual_scale
    feasibility =
        ReducedGOOP.is_feasible(problem, output.z, θ; tol = settings.physical_tol)
    x = BlockArray(output.z, problem.primal_dims)
    θ_blocks = BlockArray(θ, problem.parameter_dims)
    equality_max =
        maximum(abs, reduce(vcat, (f(x, θ_blocks) for f in problem.equality_constraints)))
    sign_violation = max(0.0, -output.sign_min)
    validation = (;
        relaxed_rms,
        unrelaxed_rms,
        sign_violation,
        primal = (; equality_max, inequality_violation = max(0.0, -feasibility.worst)),
        accepted = all(isfinite, r) &&
                   max(relaxed_rms, unrelaxed_rms)<=tol &&
                   maximum(abs, r)<=settings.component_tol &&
                   feasibility.feasible &&
                   equality_max<=settings.physical_tol &&
                   sign_violation<=settings.sign_tol,
    )

    (;
        accepted = output.status === :solved && validation.accepted,
        output,
        validation,
        stages,
        iterations = sum(s.iterations for s in stages),
        seconds = sum(s.seconds for s in stages),
        compile_seconds = sum(s.compile_seconds for s in stages),
    )
end

function compile_slack_targets(kkt)
    backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend()
    θ = ReducedGOOP.SymbolicTracingUtils.make_variables(
        backend,
        :θ,
        kkt.parameter_dimension,
    )
    raw! = ReducedGOOP._build_fd_codegen_function(
        kkt.slack_expressions,
        kkt.vars,
        θ,
        Symbolics.variable(:complete_rho),
        Symbolics.variable(:complete_unused_eta);
        chunk_size = 32,
    )
    function (y, θ, ρ)
        G = zeros(length(kkt.slack_expressions))
        raw!(G, y, θ, ρ, 0.0)
        G
    end
end

"Retain every complete equation; only the nonphysical Jacobian columns are unknowns."
function fixed_primal_formulation(kkt)
    physical = collect(kkt.primal_dims)
    free = setdiff(collect(1:kkt.n), physical)
    full = zeros(kkt.n)
    full_J = copy(kkt.Jnc_buf)
    rows, cols, _ = findnz(full_J)
    slots = sparse(rows, cols, collect(1:nnz(full_J)), size(full_J)...)
    selected_slots = nonzeros(slots[:, free])
    J = copy(full_J[:, free])
    residual! = function (r, v, parameters)
        full[physical] .= parameters.physical
        full[free] .= v
        kkt.F_nc!(r, full, parameters.θ, parameters.ρ)
        r
    end
    jacobian! = function (J, v, parameters)
        full[physical] .= parameters.physical
        full[free] .= v
        kkt.J_nc!(full_J, full, parameters.θ, parameters.ρ)
        nonzeros(J) .= nonzeros(full_J)[selected_slots]
        J
    end
    lower = fill(-Inf, length(free))
    lower[findall(in(kkt.nonnegative_dims), free)] .= 0.0
    (;
        free,
        physical,
        J,
        residual!,
        jacobian!,
        lower,
        dimensions = (; rows = kkt.n_nc, columns = length(free), full_columns = kkt.n),
    )
end

function variable_mapping(reduced, complete, problem)
    reduced_names = Dict(string(v) => j for (j, v) in enumerate(reduced.vars))
    complete_names = Dict(string(v) => j for (j, v) in enumerate(complete.vars))
    source, destination, labels = Int[], Int[], String[]
    offset = first(reduced.preference_slack_dims)-1
    for (i, levels) in enumerate(complete.levels), k in eachindex(levels)
        level = levels[k]
        for (j, v) in enumerate(level.preference_slack)
            push!(source, offset+j);
            push!(destination, complete_names[string(v)])
            push!(labels, "player $i level $k preference slack $j")
        end
        offset += length(level.preference_slack)
        k == length(levels) || continue
        for (j, v) in enumerate(level.lambda)
            push!(source, reduced_names["lam_p$(i)_K$j"])
            push!(destination, complete_names[string(v)])
            push!(labels, "player $i innermost original equality multiplier $j")
        end
        for (j, v) in enumerate(level.gamma)
            row = findfirst(
                label ->
                    label.player == i &&
                    label.level == k &&
                    label.kind == :gamma &&
                    label.inequality == j,
                reduced.labels,
            )
            row === nothing && continue
            push!(source, only(reduced.S_b[2][findall(==(row), reduced.S_b[1])]))
            push!(destination, complete_names[string(v)])
            push!(labels, "player $i innermost original inequality multiplier $j")
        end
    end
    (;
        source,
        destination,
        labels,
        physical = collect(zip(reduced.primal_dims, complete.primal_dims)),
        unmapped = "Other equality/induced multipliers start at zero; other nonnegative coordinates use the configured positive seed.",
    )
end

function complete_initialization(formulation, output, θ, settings)
    kkt = formulation.kkt
    y = zeros(kkt.n)
    y[kkt.primal_dims] .= output.z
    y[kkt.nonnegative_dims] .= settings.positive_seed
    y[formulation.mapping.destination] .= output.w[formulation.mapping.source]
    y[kkt.nonnegative_dims] .= max.(y[kkt.nonnegative_dims], settings.positive_floor)
    ρ = formulation.name === :exact ? 0.0 : first(settings.ρ_c)
    y[kkt.interior_point_slack_dims] .=
        max.(formulation.slack_targets(y, θ, ρ), settings.positive_floor)
    y
end

function solve_complete(
    formulation,
    mode,
    problem,
    θ,
    physical,
    initial,
    settings;
    retry_values = settings.retry_eta,
)
    kkt = formulation.kkt
    schedule = formulation.name === :exact ? [0.0] : settings.ρ_c
    attempts = Any[]
    for (attempt, eta) in enumerate([settings.eta_init; retry_values])
        y = copy(initial)
        stages = Any[]
        print_kkt_header(
            "complete KKT, $(formulation.name), $(mode.name), attempt $attempt",
        )
        for ρ in schedule
            stage_initial = copy(y)
            history = Any[]
            if mode.solver === :scholtes
                residuals = ReducedGOOP.ScholtesResiduals(kkt)
                trace =
                    nt -> push!(
                        history,
                        (;
                            raw = norm(nt.r),
                            unrelaxed = ReducedGOOP._norm_K0(residuals, nt.r, nt.w, ρ),
                            eta = nt.eta,
                        ),
                    )
                residual_scale = sqrt(kkt.n_nc)
                options = ReducedGOOP.ScholtesOptions(;
                    rho_schedule = [max(Float64(ρ), 1e-12)],
                    max_inner = settings.max_inner,
                    tol = settings.complete_tol * residual_scale,
                    tol_inner = max(ρ, 1e-12) * residual_scale,
                    projected_step = true,
                    proj_eps = 1e-8,
                    linear_solver = settings.linear_solver,
                    eta_schedule = true,
                    eta_init = eta,
                    eta_min = 1e-8,
                    eta_max = 1e4,
                    gain_low = 0.25,
                    gain_high = 0.75,
                    tightening_rate = 1.2,
                    loosening_rate = 3.0,
                )
                timed = @timed ReducedGOOP.solve(
                    ReducedGOOP.Scholtes(),
                    kkt,
                    θ;
                    w₀ = y,
                    options,
                    trace,
                )
                output = timed.value;
                y = copy(output.w)
                status, reason, iterations =
                    output.status, output.termination_reason, output.iters
                status_ok = output.status === :solved
                stop = Intersection.solver_stop_reason(
                    output,
                    options;
                    scale = residual_scale,
                )
                push!(
                    history,
                    (;
                        raw = output.relaxed_residual,
                        unrelaxed = output.residual,
                        eta = output.eta,
                    ),
                )
            else
                fixed = formulation.fixed
                parameters = (; θ, physical, ρ)
                unrelaxed_buffer, full = zeros(kkt.n_nc), copy(y)
                f! = function (r, v, p)
                    fixed.residual!(r, v, p)
                    full[fixed.free] .= v;
                    full[fixed.physical] .= p.physical
                    kkt.F_nc!(unrelaxed_buffer, full, p.θ, 0.0)
                    push!(
                        history,
                        (; raw = norm(r), unrelaxed = norm(unrelaxed_buffer), eta = NaN),
                    )
                    r ./= sqrt(kkt.n_nc)
                    nothing
                end
                jac! =
                    (J, v, p) -> (
                        fixed.jacobian!(J, v, p);
                        nonzeros(J) ./= sqrt(kkt.n_nc);
                        nothing
                    )
                nf = NS.NonlinearFunction{true}(
                    f!;
                    jac = jac!,
                    jac_prototype = copy(fixed.J),
                    resid_prototype = zeros(kkt.n_nc),
                )
                nonlinear_problem = NS.NonlinearLeastSquaresProblem(
                    nf,
                    y[fixed.free],
                    parameters;
                    lb = fixed.lower,
                )
                algorithm = NS.BoundedTrustRegion(;
                    linsolve = NS.QRFactorization(),
                    concrete_jac = true,
                    initial_trust_radius = attempt == 1 ? 0.0 : 0.1,
                    gtol = 0.0,
                )
                timed = @timed NS.solve(
                    nonlinear_problem,
                    algorithm;
                    abstol = settings.fixed_tol,
                    reltol = 0.0,
                    maxiters = settings.NS_iters,
                )
                output = timed.value;
                y[fixed.free] .= output.u;
                y[fixed.physical] .= physical
                status, reason, iterations = Symbol(string(output.retcode)),
                Symbol(string(output.retcode)),
                output.stats.nsteps
                status_ok = output.retcode == NS.ReturnCode.Success
                stop = string(output.retcode)
            end
            push!(
                stages,
                (;
                    ρ,
                    parameter = formulation.name===:exact ? :unrelaxed :
                                :epsilon_complete,
                    status,
                    reason,
                    iterations,
                    status_ok,
                    seconds = max(
                        0.0,
                        timed.time-timed.compile_time-timed.recompile_time,
                    ),
                    wall_seconds = timed.time,
                    compile_seconds = timed.compile_time+timed.recompile_time,
                    history,
                    initial = stage_initial,
                    final = copy(y),
                ),
            )
            # Evaluate the returned iterate; the last NonlinearSolve callback may be a rejected trial.
            r, r0 = zeros(kkt.n_nc), zeros(kkt.n_nc)
            kkt.F_nc!(r, y, θ, ρ)
            kkt.F_nc!(r0, y, θ, 0.0)
            print_kkt_stage(;
                ρ,
                status = status_ok ? :solved : :failed,
                iterations,
                seconds = last(stages).seconds,
                relaxed_rms = norm(r)/sqrt(kkt.n_nc),
                unrelaxed_rms = norm(r0)/sqrt(kkt.n_nc),
                physical = y[kkt.primal_dims],
                settings,
                stop,
            )
        end
        r, r0 = zeros(kkt.n_nc), zeros(kkt.n_nc)
        kkt.F_nc!(r, y, θ, last(schedule))
        kkt.F_nc!(r0, y, θ, 0.0)
        relaxed_rms, unrelaxed_rms = norm(r)/sqrt(kkt.n_nc), norm(r0)/sqrt(kkt.n_nc)
        feasibility = ReducedGOOP.is_feasible(
            problem,
            y[kkt.primal_dims],
            θ;
            tol = settings.physical_tol,
        )
        x = BlockArray(y[kkt.primal_dims], problem.primal_dims)
        θ_blocks = BlockArray(θ, problem.parameter_dims)
        equality_max = maximum(
            abs,
            reduce(vcat, (f(x, θ_blocks) for f in problem.equality_constraints)),
        )
        sign_violation = max(0.0, -minimum(y[kkt.nonnegative_dims]; init = Inf))
        # The full residual retains stationarity, equality, complementarity, and slack equations.
        validation = (;
            relaxed_rms,
            unrelaxed_rms,
            sign_violation,
            primal = (;
                equality_max,
                inequality_violation = max(0.0, -feasibility.worst),
            ),
            accepted = all(isfinite, r) &&
                       max(relaxed_rms, unrelaxed_rms)<=settings.complete_tol &&
                       maximum(abs, r)<=settings.component_tol &&
                       feasibility.feasible &&
                       equality_max<=settings.physical_tol &&
                       sign_violation<=settings.sign_tol,
        )

        accepted = last(stages).status_ok && validation.accepted
        push!(
            attempts,
            (;
                attempt,
                eta,
                initial = copy(initial),
                stages,
                validation,
                accepted,
                solution = copy(y),
            ),
        )
        accepted && break
    end
    final = last(attempts)
    mode.name === :fixed_primal && @assert final.solution[kkt.primal_dims] == physical
    (;
        accepted = final.accepted,
        attempts,
        validation = final.validation,
        physical = final.solution[kkt.primal_dims],
        solution = final.solution,
        status = last(final.stages).status,
        reason = last(final.stages).reason,
        iterations = sum(s.iterations for a in attempts for s in a.stages),
        seconds = sum(s.seconds for a in attempts for s in a.stages),
        compile_seconds = sum(s.compile_seconds for a in attempts for s in a.stages),
    )
end

# 4. Saved results

function trajectory_metrics(reduced, complete, settings)
    reduced_columns, complete_columns = reshape(reduced, 6, :), reshape(complete, 6, :)
    groups = ((:position, 1:2), (:velocity, 3:4), (:control, 5:6))
    errors = map(groups) do (name, rows)
        reference = vec(reduced_columns[rows, :]);
        δ = vec(complete_columns[rows, :])-reference
        scale = getproperty(settings.error_scales, name)
        (;
            name,
            absolute = norm(δ),
            relative = norm(δ)/max(norm(reference), 1e-12scale*sqrt(length(δ))),
            rms = norm(δ)/sqrt(length(δ)),
            maximum = maximum(abs, δ),
            scaled = δ/scale,
        )
    end
    δ = reduce(vcat, (g.scaled for g in errors))
    reference = reduce(
        vcat,
        (
            vec(reduced_columns[rows, :])/getproperty(settings.error_scales, name) for
            (name, rows) in groups
        ),
    )
    rms, maximum_error = norm(δ)/sqrt(length(δ)), maximum(abs, δ)
    (;
        rms,
        maximum = maximum_error,
        relative = norm(δ)/max(norm(reference), 1e-12sqrt(length(δ))),
        groups = errors,
        small = all(
            norm(g.scaled)/sqrt(length(g.scaled))<=settings.small_rms for g in errors
        ) && maximum_error<=settings.small_max,
    )
end

function save_results(directory, settings, instances, sampling, metadata, results)
    root = dirname(@__DIR__)
    source_files = [
        @__FILE__,
        joinpath(@__DIR__, "Intersection.jl"),
        joinpath(@__DIR__, "dynamics.jl"),
        joinpath(root, "Project.toml"),
        joinpath(root, "Manifest.toml"),
        joinpath(@__DIR__, "Project.toml"),
        joinpath(@__DIR__, "Manifest.toml"),
    ]
    append!(
        source_files,
        [
            joinpath(root, "src", name) for name in (
                "scholtes_complete_kkt.jl",
                "scholtes_kkt.jl",
                "scholtes_solver.jl",
                "scholtes_residual.jl",
                "scholtes_step.jl",
                "scholtes_linsolve.jl",
            )
        ],
    )
    snapshot_dir=joinpath(directory, "sources")
    if !isdir(snapshot_dir)
        for file in source_files
            destination=joinpath(snapshot_dir, relpath(file, root))
            mkpath(dirname(destination));
            cp(file, destination)
        end
    end
    sources = Dict(
        relpath(file, root) =>
            bytes2hex(sha256(read(joinpath(snapshot_dir, relpath(file, root))))) for
        file in source_files
    )
    payload = (;
        settings,
        instances,
        sampling,
        metadata,
        results,
        sources,
        julia = string(VERSION),
        nonlinear_solve = string(Base.pkgversion(NS)),
        blas_threads = BLAS.get_num_threads(),
    )
    temporary = joinpath(directory, "results.jls.tmp")
    serialize(temporary, payload)
    mv(temporary, joinpath(directory, "results.jls"); force = true)
    open(joinpath(directory, "summary.csv"), "w") do io
        println(
            io,
            "trial,objective,formulation,mode,category,reduced_accepted,complete_accepted,status,reason,reduced_iters,complete_iters,reduced_seconds,complete_seconds,compile_seconds,relaxed_rms,unrelaxed_rms,trajectory_rms,trajectory_max",
        )
        for row in results, check in row.checks
            if haskey(check, :result)
                r=check.result
                println(
                    io,
                    join(
                        (
                            row.trial,
                            row.variant,
                            check.formulation,
                            check.mode,
                            check.category,
                            row.reduced.accepted,
                            r.accepted,
                            r.status,
                            r.reason,
                            row.reduced.iterations,
                            r.iterations,
                            row.reduced.seconds,
                            r.seconds,
                            r.compile_seconds,
                            r.validation.relaxed_rms,
                            r.validation.unrelaxed_rms,
                            check.metrics.rms,
                            check.metrics.maximum,
                        ),
                        ',',
                    ),
                )
            else
                println(
                    io,
                    join(
                        (
                            row.trial,
                            row.variant,
                            check.formulation,
                            check.mode,
                            check.category,
                            false,
                            false,
                            :skipped,
                            :reduced_failed,
                            row.reduced.iterations,
                            0,
                            row.reduced.seconds,
                            0,
                            0,
                            NaN,
                            NaN,
                            NaN,
                            NaN,
                        ),
                        ',',
                    ),
                )
            end
        end
    end
end

function print_final_summary(results, settings; io = stdout)
    passed(row, formulation, mode) = any(
        check ->
            check.formulation === formulation &&
            check.mode === mode &&
            haskey(check, :result) &&
            check.result.accepted,
        row.checks,
    )
    fixed_enabled = :fixed_primal in settings.comparison_modes
    aggregate_enabled = :aggregate in settings.complete_formulations
    exact_enabled = :exact in settings.complete_formulations
    println(io, "\n[final summary] Reduced convergence and complete KKT compatibility")
    println(
        io,
        "Reduced: accepted / trials. Fixed-primal: passed / accepted reduced solves.",
    )
    @printf(
        io,
        "%-12s %12s %16s %16s %16s\n",
        "objective",
        "reduced",
        "aggregate fixed",
        "exact fixed",
        "both fixed"
    )
    for variant in settings.objective_variants
        rows = filter(row -> row.variant === variant, results)
        converged = filter(row -> row.reduced.accepted, rows)
        nred = length(converged)
        aggregate = count(row -> passed(row, :aggregate, :fixed_primal), converged)
        exact = count(row -> passed(row, :exact, :fixed_primal), converged)
        both = count(
            row ->
                passed(row, :aggregate, :fixed_primal) &&
                passed(row, :exact, :fixed_primal),
            converged,
        )
        @printf(
            io,
            "%-12s %12s %16s %16s %16s\n",
            string(variant),
            "$nred/$(length(rows))",
            fixed_enabled && aggregate_enabled ? "$aggregate/$nred" : "n/a",
            fixed_enabled && exact_enabled ? "$exact/$nred" : "n/a",
            fixed_enabled && aggregate_enabled && exact_enabled ? "$both/$nred" : "n/a"
        )
    end
    println(io, "Fixed-primal passes verify compatibility to the configured tolerances.")
    println(
        io,
        "0/0 means no accepted reduced solves; a failed check does not prove incompatibility.",
    )
    if :warm_start in settings.comparison_modes
        println(
            io,
            "\nWarm-started complete solves (trajectory may change; not a fixed-primal certificate):",
        )
        @printf(io, "%-12s %16s %16s\n", "objective", "aggregate warm", "exact warm")
        for variant in settings.objective_variants
            converged =
                filter(row -> row.variant === variant && row.reduced.accepted, results)
            nred = length(converged)
            aggregate = count(row -> passed(row, :aggregate, :warm_start), converged)
            exact = count(row -> passed(row, :exact, :warm_start), converged)
            @printf(
                io,
                "%-12s %16s %16s\n",
                string(variant),
                aggregate_enabled ? "$aggregate/$nred" : "n/a",
                exact_enabled ? "$exact/$nred" : "n/a"
            )
        end
    end
    flush(io)
    nothing
end

# 5. Plots and summary

function plot_history!(axis, histories, label, color, counts_file)
    isempty(histories) && return
    average, deviation, counts = Float64[], Float64[], Int[]
    for j in 1:maximum(length, histories)
        values = [
            log10(max(h[j], 1e-16)) for
            h in histories if length(h)>=j && isfinite(h[j]) && h[j]>=0
        ]
        isempty(values) && break
        push!(average, mean(values));
        push!(deviation, length(values)>=2 ? std(values) : NaN)
        push!(counts, length(values))
    end
    x = collect(eachindex(average))
    CM.scatterlines!(
        axis,
        x,
        10.0 .^ average;
        color,
        label,
        markersize = 4,
        linewidth = 2,
    )
    eligible = findall(>=(2), counts)
    if !isempty(eligible)
        CM.band!(
            axis,
            x[eligible],
            10.0 .^ (average[eligible]-deviation[eligible]),
            10.0 .^ (average[eligible]+deviation[eligible]);
            color = (color, 0.2),
        )
    end
    open(counts_file, "w") do io
        println(io, "sample,count,mean_log10_residual_rms,std_log10_residual_rms")
        foreach(
            j->println(io, "$j,$(counts[j]),$(average[j]),$(deviation[j])"),
            eachindex(counts),
        )
    end
end

"Intersection's map and component profiles, with reduced and complete iterates side by side."
function plot_trial_trajectories!(layout, row, check, settings)
    CM.Label(
        layout[1, 1:2],
        "Trial $(row.trial) | circle: start; star: goal; dotted: bound / preference";
        fontsize = 20,
    )
    complete = haskey(check, :result) ? check.result : nothing
    for (column, name, physical, status, accepted) in (
        (
            1,
            "Reduced",
            row.reduced.output.z,
            row.reduced.output.status,
            row.reduced.accepted,
        ),
        (
            2,
            "Complete",
            isnothing(complete) ? nothing : complete.physical,
            isnothing(complete) ? :skipped : complete.status,
            !isnothing(complete) && complete.accepted,
        ),
    )
        ax = CM.Axis(
            layout[2, column];
            title = "$name | $status | accepted=$accepted",
            aspect = CM.DataAspect(),
            xlabel = "x [m]",
            ylabel = "y [m]",
            xgridvisible = false,
            ygridvisible = false,
        )
        Intersection.draw_intersection_map!(
            ax;
            map_end = settings.map_end,
            lane_width = settings.lane_width,
        )
        CM.limits!(
            ax,
            -settings.map_end,
            settings.map_end,
            -settings.map_end,
            settings.map_end,
        )
        if isnothing(physical)
            CM.text!(
                ax,
                0.5,
                0.5;
                text = "Skipped: reduced solve failed",
                space = :relative,
                align = (:center, :center),
            )
            continue
        end
        strategies = Intersection.extract_player_strategies(
            physical,
            2,
            6settings.T,
            (; state_dimension = 4, control_dimension = 2),
        )
        for player in 1:2
            xs = strategies[player].xs
            color = Intersection.player_plot_color(player)
            CM.scatterlines!(
                ax,
                [x[1] for x in xs],
                [x[2] for x in xs];
                color,
                linewidth = 3,
                markersize = 6,
            )
            CM.scatter!(
                ax,
                [xs[1][1]],
                [xs[1][2]];
                color,
                marker = :circle,
                markersize = 16,
            )
            goal = settings.goals[player]
            CM.scatter!(ax, [goal[1]], [goal[2]]; color, marker = :star5, markersize = 22)
        end
        velocities = CM.GridLayout(layout[3, column])
        accelerations = CM.GridLayout(layout[4, column])
        for component in 1:2
            direction = component == 1 ? "x" : "y"
            velocity = CM.Axis(
                velocities[1, component];
                title = "Velocity v$direction",
                xlabel = "time [s]",
                ylabel = "v$direction [m/s]",
            )
            acceleration = CM.Axis(
                accelerations[1, component];
                title = "Acceleration a$direction",
                xlabel = "time [s]",
                ylabel = "a$direction [m/s^2]",
            )
            for player in 1:2
                color = Intersection.player_plot_color(player)
                xs, us = strategies[player].xs, strategies[player].us
                CM.lines!(
                    velocity,
                    (0:(length(xs) - 1)) .* settings.dt,
                    [x[component + 2] for x in xs];
                    color,
                    linewidth = 2,
                )
                CM.lines!(
                    acceleration,
                    (0:(length(us) - 1)) .* settings.dt,
                    [u[component] for u in us];
                    color,
                    linewidth = 2,
                )
                bounds = Intersection.player_speed_bounds(
                    player;
                    min_speed = settings.min_speed,
                    max_speed = settings.max_speed,
                )
                CM.hlines!(
                    velocity,
                    [bounds.lower[component], bounds.upper[component]];
                    color,
                    linestyle = :dot,
                )
            end
            limits = filter(
                isfinite,
                [
                    settings.control_bounds.lb[component],
                    settings.control_bounds.ub[component],
                ],
            )
            isempty(limits) ||
                CM.hlines!(acceleration, limits; color = :black, linestyle = :dot)
        end
    end
    CM.Legend(
        layout[5, 1:2],
        [
            CM.LineElement(color = Intersection.player_plot_color(player), linewidth = 3)
            for player in 1:2
        ],
        ["Player 1", "Player 2"];
        orientation = :horizontal,
        framevisible = false,
    )
    CM.rowsize!(layout, 2, 340)
    CM.rowsize!(layout, 3, 180)
    CM.rowsize!(layout, 4, 180)
    CM.rowgap!(layout, 25)
    CM.colgap!(layout, 35)
end

"Regenerate every plot from saved numeric data, without constructing or solving a KKT system."
function plot_results(directory)
    cp(@__FILE__, joinpath(directory, "plotting_source.jl"); force = true)
    data = deserialize(joinpath(directory, "results.jls"))
    settings, results = data.settings, data.results
    open(joinpath(directory, "report.md"), "w") do io
        println(io, "# Intersection KKT comparison: diagnostic two-instance study\n")
        println(
            io,
            "$(settings.T) state knots, $(settings.T-1) intervals, dt=$(settings.dt) s; horizon=$((settings.T-1)*settings.dt) s. Physical primals are all state/control entries, including terminal controls.\n",
        )
        println(
            io,
            "Initial-state sampling uses independent uniform deviations within base +/- sampling_halfwidths; zero-width coordinates remain fixed. Initial lane and separation margins are checked in metres and saved for every draw, including rejections. The original hard collision inequalities apply at knots 2:T; initial separation is additionally checked at knot 1. No continuous-time safety claim is made.\n",
        )
        println(
            io,
            "## Dimensions and variable mapping\n\n|Objective|System|Rows retained|Full unknowns|Unknowns with physical trajectory fixed|\n|---|---|---:|---:|---:|",
        )
        for metadata in data.metadata, formulation in metadata.complete
            d=formulation.dimensions
            println(
                io,
                "|$(metadata.variant)|$(formulation.name)|$(d.rows)|$(d.full_columns)|$(d.columns)|",
            )
        end
        println(
            io,
            "\nBoth systems use identical physical ordering: player, time knot, [px,py,vx,vy,ax,ay]. Preference slacks map by player/source preference/component, not flat index order. The innermost original equality and inequality multipliers also map by player and original row. All other equality/induced multipliers start at zero; remaining sign-constrained variables use positive_seed, followed by auxiliary-slack initialization from their defining expressions. Full source/destination index maps and labels are saved in metadata.\n",
        )
        println(
            io,
            "Reduced and full complete solves use Scholtes. Fixed-primal recovery uses NonlinearSolve BoundedTrustRegion with analytic free-column Jacobians and nonnegative-coordinate bounds. Exact complete equations have no epsilon schedule; aggregate uses rho_c as legacy epsilon. All complete rows remain after fixing physical columns.\n",
        )
        println(
            io,
            "Residual curves show the full relaxed and unrelaxed L2 residuals divided by sqrt(number of equations) on logarithmic axes, concatenated across stages, against solver-trace samples (Scholtes iterates; NonlinearSolve residual evaluations, including rejected trials). Reduced curves include all sampled trials; complete curves include all attempted checks, using the final retry. Histograms use accepted pairs only. Curves use available samples only: no padding after a run ends. Lines are geometric means; bands are mean +/- one sample standard deviation in log10 residual, transformed back to the logarithmic axis at indices with at least two contributors; contributor counts are saved separately. Stage endpoints and raw histories are in results.jls. Each comparison PDF also shows every trial's reduced and complete position, velocity, and acceleration profiles side by side, including failed returned iterates with explicit status labels.\n",
        )
        println(
            io,
            "NonlinearSolve tolerance=$(settings.fixed_tol); independent complete RMS tolerance=$(settings.complete_tol), component tolerance=$(settings.component_tol), physical feasibility tolerance=$(settings.physical_tol). Fixed trajectories retain the reduced solution's finite physical residual; these rows are not removed. Success requires a successful solver exit plus all independent checks. Relaxed and unrelaxed residuals are checked separately. The full residual includes stationarity, equality, complementarity, and auxiliary-slack equations; the script additionally records multiplier signs and original physical feasibility. Failures do not prove incompatibility.\n",
        )
        println(
            io,
            "Warm-started complete times are refinement costs. Construction and independent warm-ups are excluded. Reported solve seconds subtract Julia's measured compilation and recompilation time from wall time (clamped at zero); wall and compilation seconds are saved separately, and nonzero measured compilation is flagged below. Both starts and every attempt/stage are saved. No warm-up or previous-trial solution seeds measured trials.\n",
        )
        println(
            io,
            "Trajectory normalization: position=$(settings.error_scales.position) m, velocity=$(settings.error_scales.velocity) m/s, control=$(settings.error_scales.control) m/s^2. Small adjustment requires each group scaled RMS <= $(settings.small_rms), max scaled component <= $(settings.small_max). Relative norms use a 1e-12 RMS denominator floor. Statistics below are descriptive, not evidence of theoretical equivalence.\n",
        )
        println(
            io,
            "|Objective|Complete|Mode|Reduced|Conditional complete|Overall pairs|\n|---|---|---|---:|---:|---:|",
        )
        for variant in settings.objective_variants,
            formulation in settings.complete_formulations,
            mode in settings.comparison_modes

            rows = filter(r->r.variant===variant, results)
            pairs = [
                (; row, check) for row in rows for check in row.checks if
                check.formulation===formulation && check.mode===mode
            ]
            nred = count(r->r.reduced.accepted, rows)
            successful =
                filter(p->haskey(p.check, :result) && p.check.result.accepted, pairs)
            println(
                io,
                "|$variant|$formulation|$mode|$nred/$(length(rows))|$(length(successful))/$nred|$(length(successful))/$(length(rows))|",
            )
            println("Plotting $variant / $formulation / $mode...");
            flush(stdout)
            attempted = filter(
                p -> haskey(p.check, :result) && !isempty(p.check.result.attempts),
                pairs,
            )
            figure = Intersection.serif_figure(
                size = (1600, 480+900length(rows)),
                fontsize = 18,
                figure_padding = 30,
            )
            CM.Label(figure[1, 1:2], "$variant / $formulation / $mode"; fontsize = 26)
            convergence = CM.GridLayout(figure[2, 1:2])
            metadata = only(filter(m->m.variant===variant, data.metadata))
            complete_rows =
                only(filter(p->p.name===formulation, metadata.complete)).dimensions.rows
            prefix = joinpath(directory, "$(variant)_$(formulation)_$(mode)")
            for (column, field, title, suffix) in (
                (1, :unrelaxed, "True KKT residual", "_k0"),
                (2, :raw, "Relaxed residual", ""),
            )
                axis = CM.Axis(
                    convergence[1, column];
                    xlabel = "Trace sample (stages concatenated)",
                    ylabel = "Residual / sqrt(m)",
                    yscale = log10,
                    title = "$title\nReduced n=$(length(rows)); complete n=$(length(attempted))",
                )
                histories = [
                    reduce(
                        vcat,
                        (
                            [
                                getproperty(h, field)/sqrt(complete_rows) for
                                h in s.history
                            ] for s in last(p.check.result.attempts).stages
                        ),
                    ) for p in attempted
                ]
                reduced_histories = [
                    reduce(
                        vcat,
                        (
                            [
                                getproperty(h, field)/sqrt(metadata.reduced.rows) for
                                h in s.history
                            ] for s in row.reduced.stages
                        ),
                    ) for row in rows
                ]
                if isempty(histories) && isempty(reduced_histories)
                    CM.text!(
                        axis,
                        0.5,
                        0.5;
                        text = "No solver histories",
                        space = :relative,
                        align = (:center, :center),
                    )
                else
                    plot_history!(
                        axis,
                        histories,
                        "Complete",
                        :navy,
                        prefix*"_complete$(suffix)_contributors.csv",
                    )
                    plot_history!(
                        axis,
                        reduced_histories,
                        "Reduced",
                        :darkorange,
                        prefix*"_reduced$(suffix)_contributors.csv",
                    )
                    CM.hlines!(
                        axis,
                        [settings.reduced_tol];
                        color = :darkorange,
                        linestyle = :dot,
                    )
                    CM.hlines!(
                        axis,
                        [settings.complete_tol];
                        color = :navy,
                        linestyle = :dot,
                    )
                    CM.axislegend(axis; position = :rt, labelsize = 14)
                end
            end
            second = CM.Axis(
                convergence[1, 3],
                xlabel = mode===:fixed_primal ? "Final unrelaxed residual RMS" :
                         "Scaled trajectory RMS",
                ylabel = "Accepted pairs",
                title = mode===:fixed_primal ?
                        "Fixed-primal recovery $(length(successful))/$nred" :
                        "Trajectory adjustment (successful pairs)",
            )
            values = [
                mode===:fixed_primal ? p.check.result.validation.unrelaxed_rms :
                p.check.metrics.rms for p in successful
            ]
            if isempty(values)
                CM.text!(
                    second,
                    0.5,
                    0.5;
                    text = "No accepted pairs",
                    space = :relative,
                    align = (:center, :center),
                )
            else
                CM.hist!(
                    second,
                    values;
                    bins = max(1, min(10, length(values))),
                    color = :steelblue,
                )
            end
            for (index, row) in enumerate(rows)
                check = only(
                    filter(
                        c -> c.formulation === formulation && c.mode === mode,
                        row.checks,
                    ),
                )
                layout = CM.GridLayout(figure[index + 2, 1:2])
                plot_trial_trajectories!(layout, row, check, settings)
            end
            CM.rowsize!(figure.layout, 2, 300)
            CM.rowgap!(figure.layout, 30)
            CM.resize_to_layout!(figure)
            Intersection.save_figure(prefix*".pdf", figure)
            isfile(prefix*".png") && rm(prefix*".png")
        end
        println(io, "\n## Failures and measured compilation\n")
        for row in results
            row.reduced.compile_seconds>0 && println(
                io,
                "- $(row.variant) trial $(row.trial): reduced measured compilation $(row.reduced.compile_seconds) s.",
            )
            !row.reduced.accepted && println(
                io,
                "- $(row.variant) trial $(row.trial): reduced rejected; status=$(row.reduced.output.status), reason=$(row.reduced.output.termination_reason), R=$(row.reduced.validation.relaxed_rms), K0=$(row.reduced.validation.unrelaxed_rms), physical equality=$(row.reduced.validation.primal.equality_max), inequality=$(row.reduced.validation.primal.inequality_violation).",
            )
            for check in row.checks
                haskey(check, :result) || continue
                r=check.result
                !r.accepted && println(
                    io,
                    "- $(row.variant) trial $(row.trial), $(check.formulation), $(check.mode): $(r.status), $(r.reason); R=$(r.validation.relaxed_rms), K0=$(r.validation.unrelaxed_rms), physical equality=$(r.validation.primal.equality_max), inequality=$(r.validation.primal.inequality_violation), attempts=$(length(r.attempts)).",
                )
                haskey(r, :error) && println(io, "  Exception: $(r.error)")
                r.compile_seconds>0 && println(
                    io,
                    "- $(row.variant) trial $(row.trial), $(check.formulation), $(check.mode): measured compilation $(r.compile_seconds) s.",
                )
            end
        end
    end
    directory
end

end
