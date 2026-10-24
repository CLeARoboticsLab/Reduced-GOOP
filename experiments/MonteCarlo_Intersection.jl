"""
Monte Carlo check of whether reduced-GOOP solutions also solve the complete KKT system,
on the two-player intersection game from Intersection.jl.

For every objective variant and every sampled initial state:
 1. Solve the REDUCED KKT system with Scholtes down `rho_schedule`, starting from
    Intersection's yielding initial guess. Every solve, reduced or complete, is accepted iff
    its unrelaxed KKT residual ‖K0‖/√m is at most `tol` (reduced) or `complete_tol`. A ρ chain
    runs all its stages and is judged, and kept, at the stage with the lowest ‖K0‖; a complete
    check likewise keeps its best try.
 2. For an accepted reduced solution, test each COMPLETE formulation in each mode:
      :fixed_primal  keep the reduced trajectory fixed and solve only for the other complete
                     variables (multipliers, slacks) with Scholtes, x entering as a parameter.
                     A pass means the complete system holds, to tolerance, with the reduced trajectory.
      :warmstart     solve the whole complete system with Scholtes from the mapped reduced
                     solution, and report how far the trajectory moves.

Terms
  aggregate  generate_scholtes_complete_kkt_system: the legacy slacked complete system;
             complementarity enters as aggregate rows ρ − γᵀG ≥ 0, with ρ (the legacy ε)
             run down `complete_rho_schedule`.
  exact      generated_exact_complete_kkt_system: componentwise complementarity, no
             relaxation, so it is solved once at ρ = 0.
  R, K0      residual norm at the stage's ρ, and at ρ = 0; always reported as RMS,
             norm / √(number of equations).
  physical   the stacked trajectory z, ordered player, knot, [px, py, vx, vy, ax, ay].

Entry points: `run(; overrides)`, and `plot_results(directory)`, which redraws report.pdf
from results.jls without building or solving anything. Each run writes a timestamped
directory under output/intersection_kkt_comparison/ with two files: results.jls (all the
numbers; rewritten after every check, so a crash keeps partial results) and report.pdf (a
summary page, then one page per variant/formulation/mode).

    julia --project=experiments -e 'include("experiments/MonteCarlo_Intersection.jl");
        MonteCarlo_Intersection.run(overrides = (; num_instances = 50))'
"""
module MonteCarlo_Intersection

using BlockArrays, LinearAlgebra, SparseArrays, Random, Statistics, Serialization, Dates, Printf
import ReducedGOOP
import Symbolics
import CairoMakie as CM
include(joinpath(@__DIR__, "Intersection.jl"))

# ── Problem definition ─────────────────────────────────────────────────────────

function objective_variants(settings)
    # 2a²(cosh(d/a)-1) matches d² in value, gradient, and curvature at zero.
    # a controls growth: control scale 1.5 m/s², goal scale 6 m; these are not error scales.
    [
        (;
            name=:quadratic,
            label="Quadratic",
            objective=(d, a) -> d^2, # reference only: get_setup keeps Intersection's objectives
            control_scale=settings.control_shape_scale,
            goal_scale=settings.goal_shape_scale,
        ),
        (;
            name=:cosh,
            label="Scaled cosh",
            objective=(d, a) -> 2a^2 * (cosh(d/a)-1),
            control_scale=settings.control_shape_scale,
            goal_scale=settings.goal_shape_scale,
        ),
    ]
end

"""
Intersection.get_setup, with the control and goal objectives replaced by `variant.objective`
for every variant except :quadratic (see below). Each player's preferences run lowest ->
highest priority (see Intersection.get_setup):
    P1: [control, speed (prioritized constraint), goal]
    P2: [control, goal, speed (prioritized constraint)]
so the indices below pick the control and goal entries. :quadratic keeps Intersection's
built-in objectives, which are the same sums of squares.
"""
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

    # Vector{Function}, so single entries can be replaced; ParametricGOOP is immutable, so it
    # is rebuilt below with the new preferences.
    preferences = [Function[p...] for p in problem.preferences]
    # TODO: install variant.objective for :quadratic too, so objective_variants is the one
    # source of truth. It changes the generated expressions; compare a run before and after.
    if variant.name !== :quadratic
        preferences[1][1] = control_objective(; player=1)
        preferences[1][3] = goal_objective(; player=1)
        preferences[2][1] = control_objective(; player=2)
        preferences[2][2] = goal_objective(; player=2)
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

# ── Settings and experiment driver ─────────────────────────────────────────────

"""
Default experiment settings, grouped by purpose; a setting that Intersection.demo also has uses
demo's name. Replace entries with `run(overrides = (; num_instances = 50))`. Saved verbatim in
results.jls.
"""
default_settings() = (;
    # Monte Carlo
    num_instances=2,                             # accepted initial states per objective variant
    rng_seed=123,                                # Xoshiro seed for sampling
    objective_variants=(:cosh,),                 # names in objective_variants(settings); (:quadratic, :cosh)
    complete_formulations=(:exact,),             # names in `builders` (run)            ; (:exact, :aggregate)
    comparison_modes=(:fixed_primal,),           # names in `modes` (run)               ; (:fixed_primal, :warmstart)
    # Scenario (Intersection.jl). Units m, s; state [px, py, vx, vy]; P1 drives along x, P2 along y.
    planning_horizon=7,                          # state knots; horizon (planning_horizon - 1) * time_step = 3 s
    time_step=0.5,
    base_initial_state1=[-6.0, -1.0, 1.5, 0.0],  # unperturbed initial states; sampling perturbs them
    base_initial_state2=[1.0, -5.0, 0.0, 2.0],
    goal_position1=[6.0, -1.0],
    goal_position2=[1.0, 6.0],
    obstacle_position=[0.25, 0.15],              # part of θ, but no cost or constraint uses it
    map_end=7.0,                                 # hard: |along-lane position| <= map_end
    lane_width=2.0,                              # hard: |cross-lane position| <= lane_width
    collision_avoidance=1.3,                     # hard: distance between players >= this, from knot 2
    min_speed=0.7,                               # preference: |transverse velocity| <= min_speed (a CAP)
    max_speed=3.0,                               # preference: |forward velocity| <= max_speed
    control_bounds=(; lb=[-3.0, -3.0], ub=[3.0, 3.0]), # hard, m/s²
    # Initial-state sampling: uniform in base ± perturbation_scale (zero scale = fixed), then rejection.
    sampling_distribution=:uniform,              # the only sampler implemented
    perturbation_scale1=[0.3, 0.3, 0.15, 0.0],   # per coordinate of base_initial_state1
    perturbation_scale2=[0.3, 0.3, 0.0, 0.15],   # per coordinate of base_initial_state2
    initial_lane_margin=0.05,                    # m, strict clearance from the lane bounds at knot 1
    initial_separation_margin=0.05,              # m, strict clearance beyond collision_avoidance at knot 1
    max_sampling_attempts=1000,                  # draws before giving up; rejections are saved
    # Objective shape, cosh variant: a in 2a²(cosh(d/a) - 1). Not the error scales below.
    control_shape_scale=1.5,                     # m/s²
    goal_shape_scale=6.0,                        # m
    # Solvers
    #! format: off
    rho_schedule          = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], # reduced system
    complete_rho_schedule = [1.0,  1e-1, 1e-2, 1e-3, 1e-4, 1e-5], # aggregate complete system; exact runs at ρ = 0
    #! format: on
    max_inner_iters=500,                         # Scholtes Newton steps per ρ stage
    linear_solver=:normal,                       # ScholtesOptions.linear_solver
    eta_0=1e-5,                                  # initial regularization η
    # One complete-solve retry per entry, with that η.
    retry_eta=[1e-4, 1e-3, 1e-2],
    # Acceptance: a solve is accepted iff its unrelaxed KKT residual ‖K0‖ / √(number of equations) <= tol,
    # at the best stage of its ρ chain (and, for a complete check, its best try).
    tol=2e-4,                                    # reduced system
    complete_tol=1e-3,                           # complete systems
    inner_ρ_tol=1e-2,                            # Scholtes ρ stage's inner stop: ‖R‖/√m < min(ρ, inner_ρ_tol). 1.0 for old behavior
    # Complete-system start (complete_initialization)
    positive_seed=1.0,                           # unmapped nonnegative variables
    positive_floor=1e-10,                        # floor for every nonnegative variable
    # Trajectory comparison, :warmstart mode
    error_scales=(; position=1.0, velocity=1.0, control=1.0), # m, m/s, m/s²
    small_rms=0.01,                              # "small adjustment": every group's scaled RMS <= small_rms
    small_max=0.05,                              #   and every |scaled entry| <= small_max
    output_dir=joinpath(dirname(@__DIR__), "output", "intersection_kkt_comparison"),
)

function run(; overrides=(;))
    defaults = default_settings()
    unknown = setdiff(keys(overrides), keys(defaults))
    isempty(unknown) || error("Unknown setting(s) in overrides: $(join(unknown, ", "))")
    settings = merge(defaults, overrides)
    settings.num_instances > 0 && settings.planning_horizon >= 2 || error("invalid num_instances or planning_horizon")
    schedules = (settings.rho_schedule, settings.complete_rho_schedule)
    all(s -> !isempty(s) && all(ρ -> isfinite(ρ) && ρ > 0, s), schedules) || error("invalid relaxation schedule")
    mkpath(settings.output_dir)
    directory = mktempdir(
        settings.output_dir;
        prefix=Dates.format(now(), "yyyymmdd_HHMMSS") * "_",
        cleanup=false,
    )
    rng = Xoshiro(settings.rng_seed)
    instances, sampling = sample_instances(rng, settings)
    results, metadata = Any[], Any[]
    # Save before checking the sample count, so the rejection records survive a sampling failure.
    save_results(directory, settings, instances, sampling, metadata, results)
    length(instances) == settings.num_instances || error(
        "Sampling exhausted $(settings.max_sampling_attempts) attempts; rejection records saved in $directory",
    )
    println(
        "Diagnostic run: $(settings.num_instances) instances, planning horizon = $(settings.planning_horizon), ",
        "$((settings.planning_horizon-1)*settings.time_step) seconds. Output: $directory",
    )
    variants = filter(v -> v.name in settings.objective_variants, objective_variants(settings))
    modes = [
        (; name=:fixed_primal, label="Fixed-primal recovery"),
        (; name=:warmstart, label="Warm-started complete"),
    ]
    modes = filter(m -> m.name in settings.comparison_modes, modes)
    builders = [
        (; name=:aggregate, build=ReducedGOOP.generate_scholtes_complete_kkt_system),
        (; name=:exact, build=ReducedGOOP.generated_exact_complete_kkt_system),
    ]
    builders = filter(b -> b.name in settings.complete_formulations, builders)
    total = length(variants)*length(instances)*length(builders)*length(modes)
    done = 0
    (; num_instances, planning_horizon, time_step, control_bounds, collision_avoidance) = settings
    (; min_speed, max_speed, map_end, lane_width, goal_position1, goal_position2, obstacle_position) = settings
    num_players = 2
    dynamics_model = Intersection.PlanarDoubleIntegrator()
    state_dimension = 4
    control_dimension = 2
    Δt = time_step
    dynamics_dimension = state_dimension + control_dimension
    primal_dimension = dynamics_dimension * planning_horizon # per player
    dynamics = Intersection.build_intersection_dynamics(
        dynamics_model;
        Δt,
        state_dimension,
        control_dimension,
    )
    println(tolerance_line(settings))
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
        println("Building reduced KKT system for $(variant.name) variant...")
        flush(stdout)
        kkt_construction = @timed ReducedGOOP.generate_slacked_reduced_kkt_system(
            problem;
            complementarity=:scholtes,
            codegen=:fast_differentiation,
        )
        GOOP_kkt_system = kkt_construction.value
        formulations = map(builders) do builder
            println(
                "Building :$(builder.name) complete KKT system for $(variant.name) variant...",
            )
            flush(stdout)
            built = @timed builder.build(problem)
            kkt = built.value
            slack_targets = compile_slack_targets(kkt)
            fixed = fixed_primal_formulation(kkt)
            mapping = variable_mapping(GOOP_kkt_system, kkt)
            # The exact system is square; the fixed-primal Jacobian keeps every row but drops
            # the physical columns.
            builder.name === :exact && @assert kkt.n == kkt.n_nc
            @assert size(fixed.system.Jnc_buf) == (kkt.n_nc, kkt.n-length(kkt.primal_dims))
            (;
                builder.name,
                kkt,
                slack_targets,
                fixed,
                mapping,
                construction_seconds=built.time,
            )
        end
        push!(
            metadata,
            (;
                variant=variant.name,
                label=variant.label,
                objective=(; variant.control_scale, variant.goal_scale),
                # Reduced Scholtes system: rows of R(w; ρ), columns of w = [y; s; u].
                reduced=(;
                    rows=GOOP_kkt_system.n_nc+2GOOP_kkt_system.n_comp,
                    columns=GOOP_kkt_system.n+GOOP_kkt_system.n_comp+ReducedGOOP._n_u(
                        GOOP_kkt_system,
                    ),
                    construction_seconds=kkt_construction.time,
                ),
                complete=[
                    (; p.name, p.fixed.dimensions, p.mapping, p.construction_seconds) for
                    p in formulations
                ],
            ),
        )
        # Untimed warm-up on the first instance, so measured trials don't pay first-call (JIT)
        # costs. Not related to the :warmstart comparison mode.
        warmup = instances[1]
        initial_state1, initial_state2 = warmup.states
        (; θ) = Intersection.build_instance_parameters(
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
        print("Warm-up for $(variant.name) (untimed; compiles every solver path; results discarded)... ")
        flush(stdout)
        warmup_seconds = @elapsed begin
            context = "warm-up, $(variant.name)"
            warm_reduced =
                solve_reduced(GOOP_kkt_system, problem, θ, warmstart_solution, settings; context, io=devnull)
            for formulation in formulations, mode in modes
                initial_complete =
                    complete_initialization(formulation, warm_reduced.output, θ, settings)
                solve_complete(
                    formulation,
                    mode,
                    problem,
                    θ,
                    warm_reduced.output.z,
                    initial_complete,
                    settings;
                    retry_values=Float64[],
                    context,
                    io=devnull,
                )
            end
        end
        println("done in $(round(warmup_seconds; digits=1)) s")
        # Warm-up results are discarded; every measured trial rebuilds its starts.
        for (trial, instance) in enumerate(instances)
            println("\n══ Trial $trial/$num_instances · $(variant.name) ══")
            initial_state1, initial_state2 = instance.states
            println("  P1 initial state [px, py, vx, vy]: $(round.(initial_state1; digits=3))")
            println("  P2 initial state [px, py, vx, vy]: $(round.(initial_state2; digits=3))")
            flush(stdout)
            (; θ) = Intersection.build_instance_parameters(
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
            context = "trial $trial, $(variant.name)"
            reduced_sol =
                solve_reduced(GOOP_kkt_system, problem, θ, warmstart_solution, settings; context)
            n_checks = length(formulations) * length(modes)
            println(
                reduced_verdict(reduced_sol, settings),
                reduced_sol.accepted ? " Running its $n_checks complete checks." :
                " Skipping its $n_checks complete checks.",
            )
            flush(stdout)

            strategies = Intersection.extract_player_strategies(
                reduced_sol.output.z,
                num_players,
                primal_dimension,
                dynamics,
            )
            # `checks` is filled in place by the loop below. The record already in `results` sees
            # each push, so every save_results call stores the progress so far.
            checks = Any[]
            record = (;
                trial,
                variant=variant.name,
                θ,
                initial=copy(warmstart_solution),
                strategies,
                reduced=reduced_sol,
                checks,
            )
            push!(results, record)
            save_results(directory, settings, instances, sampling, metadata, results)
            # formulations: :aggregate, :exact; modes: :fixed_primal, :warmstart
            for formulation in formulations, mode in modes
                if reduced_sol.accepted
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
                            settings;
                            context,
                        )
                    catch err
                        validation = merge(
                            reduced_sol.validation,
                            (; accepted=false, relaxed_rms=NaN, unrelaxed_rms=NaN, max_abs_residual=NaN),
                        )
                        (;
                            accepted=false,
                            attempts=Any[],
                            validation,
                            physical=copy(reduced_sol.output.z),
                            solution=start,
                            status=:exception,
                            reason=:exception,
                            error=sprint(showerror, err, catch_backtrace()),
                            iterations=0,
                            seconds=0.0,
                            compile_seconds=0.0,
                        )
                    end
                    metrics = trajectory_metrics(
                        reduced_sol.output.z,
                        result.physical,
                        settings,
                    )
                    # :numerical_failure        complete solve not accepted (no conclusion either way)
                    # :fixed_primal_compatible  fixed-primal solve accepted
                    # :small_adjustment         :warmstart accepted; trajectory moved within small_rms/small_max
                    # :substantial_adjustment   :warmstart accepted; trajectory moved more
                    # (:reduced_failed, below, marks checks skipped because the reduced solve failed.)
                    category =
                        !result.accepted ? :numerical_failure :
                        mode.name === :fixed_primal ? :fixed_primal_compatible :
                        metrics.small ? :small_adjustment : :substantial_adjustment
                    check = (;
                        formulation=formulation.name,
                        mode=mode.name,
                        result,
                        metrics,
                        category,
                    )
                    push!(checks, check)
                    println(complete_verdict(formulation, mode, result, metrics, category, settings))
                else
                    push!(
                        checks,
                        (;
                            formulation=formulation.name,
                            mode=mode.name,
                            category=:reduced_failed,
                            reason="Best reduced stage did not pass acceptance",
                        ),
                    )
                end
                done += 1
                # One bar line per complete check that ran; skipped checks advance it once, below.
                reduced_sol.accepted && print_progress(done, total)
                save_results(directory, settings, instances, sampling, metadata, results)
            end
            reduced_sol.accepted || print_progress(done, total)
        end
    end
    println()
    plot_results(directory)
    println("Saved results.jls and report.pdf: $directory")
    print_final_summary(results, settings)
    (; directory, settings, instances, results, metadata)
end
# End of Monte Carlo Intersection experiments


# ── Sampling and solves ────────────────────────────────────────────────────────

"Rejection-sample `num_instances` initial-state pairs; every draw, accepted or not, is recorded."
function sample_instances(rng, settings)
    settings.sampling_distribution === :uniform || error("Supported sampler: :uniform")
    instances, attempts = Any[], Any[]
    base_initial_states = (settings.base_initial_state1, settings.base_initial_state2)
    perturbation_scales = (settings.perturbation_scale1, settings.perturbation_scale2)
    for attempt in 1:settings.max_sampling_attempts
        # 2rand - 1 is uniform on [-1, 1]; a zero scale keeps that coordinate fixed.
        states = [
            base .+ scale .* (2rand(rng, 4) .- 1) for
            (base, scale) in zip(base_initial_states, perturbation_scales)
        ]
        # P1 drives along x and P2 along y, so for player i, coordinate i is along-lane
        # (bound map_end) and coordinate 3 - i is cross-lane (bound lane_width).
        lane_margins = [
            minimum((
                settings.map_end-abs(states[i][i]),
                settings.lane_width-abs(states[i][3-i]),
            )) for i in 1:2
        ]
        separation_margin = norm(states[1][1:2]-states[2][1:2])-settings.collision_avoidance
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
                reason=accepted ? "accepted" : "initial lane/separation strict margin",
            ),
        )
        accepted && push!(instances, (; states, sampling_attempt=attempt))
        length(instances) == settings.num_instances && return instances, attempts
    end
    instances, attempts
end

"Compact number for printouts: 1e-4, 2.5e-3, 0.1."
function compact(x)
    isfinite(x) || return string(x)
    x == 0 && return "0"
    exponent = floor(Int, log10(abs(x)))
    -1 <= exponent <= 1 && return string(round(x; sigdigits=3))
    mantissa = round(x / 10.0^exponent; sigdigits=3)
    "$(isinteger(mantissa) ? Int(mantissa) : mantissa)e$exponent"
end

"The acceptance tolerances, printed before the first solve and on the report's summary page."
tolerance_line(settings) = "Tolerances: reduced ‖K0‖/√m ≤ $(compact(settings.tol)), complete ‖K0‖/√m ≤ $(compact(settings.complete_tol))"

"Progress bar on its own line: other output interleaves with it, so it is not redrawn in place."
function print_progress(done, total; width=24)
    filled = floor(Int, width * done / total)
    println("[", repeat("=", filled), repeat(" ", width - filled), "] $done/$total checks")
    flush(stdout)
end

"One-line outcome of a reduced solve, printed under its stage table."
function reduced_verdict(reduced, settings)
    k0, tol = compact(reduced.validation.unrelaxed_rms), compact(settings.tol)
    stage = "ρ = $(compact(reduced.stages[reduced.best_stage].ρ)), stage $(reduced.best_stage) of $(length(reduced.stages))"
    reduced.accepted ? "→ reduced accepted at $stage: ‖K0‖/√m = $k0 ≤ $tol." :
    "→ reduced REJECTED: best ‖K0‖/√m = $k0 ($stage) > $tol."
end

"One-line outcome of a complete check, printed under its stage tables."
function complete_verdict(formulation, mode, result, metrics, category, settings)
    name = ":$(formulation.name), :$(mode.name) mode"
    haskey(result, :error) && return "→ $name FAILED with an exception: $(first(split(result.error, '\n')))"
    final = result.attempts[result.best_attempt]
    tries = length(result.attempts) == 1 ? "1 try" : "$(length(result.attempts)) tries"
    stage = "ρ = $(compact(final.stages[final.best_stage].ρ))"
    k0, tol = compact(result.validation.unrelaxed_rms), compact(settings.complete_tol)
    result.accepted ||
        return "→ $name REJECTED after $tries: best ‖K0‖/√m = $k0 (try $(result.best_attempt), $stage) > $tol"
    moved =
        mode.name === :fixed_primal ? "trajectory held fixed" :
        "trajectory moved RMS $(compact(metrics.rms)) ($(replace(string(category), '_' => ' ')))"
    "→ $name accepted after $tries at $stage: ‖K0‖/√m = $k0 ≤ $tol; $moved"
end

"Header above each solve's ρ-stage table; `label` names the solve and the η₀ this try uses."
function print_kkt_header(label; io=stdout)
    println(io, "\n[solver exit] $label")
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
    io=stdout,
)
    # 2 players with 6 primals per knot; extract_player_strategies reads only these two
    # dimensions from its `dynamics` argument.
    strategies = Intersection.extract_player_strategies(
        physical,
        2,
        6settings.planning_horizon,
        (; state_dimension=4, control_dimension=2),
    )
    metrics = Intersection.intersection_result_metrics(
        strategies;
        settings.goal_position1,
        settings.goal_position2,
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

function solve_reduced(GOOP_kkt_system, problem, θ, warmstart_solution, settings; context, io=stdout)
    solver = ReducedGOOP.Scholtes()
    (; max_inner_iters, tol, eta_0, linear_solver, rho_schedule, inner_ρ_tol) = settings
    verbose = false
    # R(w; ρ) = [F_nc(y); a(y) - s; s ⊙ b(y) + u - ρ] has m = n_nc + 2 n_comp rows
    # (src/scholtes_residual.jl). Settings hold RMS tolerances, so they are scaled by √m.
    m = GOOP_kkt_system.n_nc + 2GOOP_kkt_system.n_comp
    residual_scale = sqrt(m)
    # λ (lam_*) and ψ (psi_*) are unsigned, so they carry over to the next ρ stage.
    # scholtes_warm_start re-centres γ/φ, s and u for the new ρ, and each stage restarts η at
    # eta_0: the same warm-ρ chain as Intersection.demo.
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
    best = nothing
    print_kkt_header("$context | reduced KKT | η₀ = $(compact(eta_0))"; io)
    for ρ in rho_schedule
        eq = isnothing(previous_w) ? zeros(length(eq_cols)) : previous_w[eq_cols]
        w₀ = ReducedGOOP.scholtes_warm_start(
            GOOP_kkt_system,
            θ,
            stage_warmstart,
            ρ;
            eq_cols,
            eq,
            preference_slacks=stage_preference_slacks,
        )
        z₀ = stage_warmstart
        # Same options as Intersection.demo. Of the fixed constants, only proj_eps, eta_max and
        # eta_schedule differ from the ScholtesOptions defaults; the rest come from settings.
        options = ReducedGOOP.ScholtesOptions(;
            rho_schedule=[Float64(ρ)],
            max_inner=max_inner_iters,
            tol=tol * residual_scale,
            # A stage's inner loop stops at ‖R‖/√m < min(ρ, inner_ρ_tol): coarse stages are solved
            # more tightly than their ρ.
            tol_inner=min(ρ, inner_ρ_tol) * residual_scale,
            projected_step=true,
            proj_eps=1e-8,
            linear_solver,
            eta_schedule=true,
            eta_init=eta_0,
            eta_min=1e-8,
            eta_max=1e4,
            gain_low=0.25,
            gain_high=0.75,
            tightening_rate=1.2,
            loosening_rate=3.0,
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
                parameter=:rho_reduced,
                status=output.status,
                reason=output.termination_reason,
                iterations=output.iters,
                seconds=max(0.0, timed.time-timed.compile_time-timed.recompile_time),
                wall_seconds=timed.time,
                compile_seconds=timed.compile_time+timed.recompile_time,
                history,
                initial=copy(w₀),
                final=copy(output.w),
            ),
        )
        print_kkt_stage(;
            ρ,
            status=output.status,
            iterations=output.iters,
            seconds=last(stages).seconds,
            relaxed_rms=output.relaxed_residual/residual_scale,
            unrelaxed_rms=output.residual/residual_scale,
            physical=output.z,
            settings,
            io,
            stop=Intersection.solver_stop_reason(
                output,
                options;
                scale=residual_scale,
            ),
        )
        stage_warmstart = output.z
        previous_w = output.w
        stage_preference_slacks = output.preference_slacks
        # Track the stage with the lowest ‖K0‖ (a non-finite residual counts as Inf).
        k0 = isfinite(output.residual) ? output.residual : Inf
        if isnothing(best) || k0 < best.k0
            best = (; output, ρ, k0, index=length(stages))
        end
    end
    # Judge and keep the best stage, not the last: each stage restarts γ/φ, s and u, so a later
    # stage can end worse than an earlier, converged one.
    (; output, ρ) = best
    r = zeros(m)
    b = zeros(GOOP_kkt_system.n_comp)
    ReducedGOOP._resid!(residuals, r, b, output.w, ρ)
    relaxed_rms = norm(r)/residual_scale
    unrelaxed_rms = ReducedGOOP._norm_K0(residuals, r, output.w, ρ)/residual_scale
    feasibility = ReducedGOOP.is_feasible(problem, output.z, θ) # only `worst` is used
    x = BlockArray(output.z, problem.primal_dims)
    θ_blocks = BlockArray(θ, problem.parameter_dims)
    equality_max =
        maximum(abs, reduce(vcat, (f(x, θ_blocks) for f in problem.equality_constraints)))
    sign_violation = max(0.0, -output.sign_min)
    # Acceptance depends only on the unrelaxed KKT residual ‖K0‖/√m; every other field is a
    # saved diagnostic.
    validation = (;
        relaxed_rms,
        unrelaxed_rms,
        max_abs_residual=maximum(abs, r),
        sign_violation,
        primal=(; equality_max, inequality_violation=max(0.0, -feasibility.worst)),
        accepted=unrelaxed_rms <= tol,
    )

    (;
        accepted=validation.accepted,
        output,
        validation,
        stages,
        best_stage=best.index,
        iterations=sum(s.iterations for s in stages),
        seconds=sum(s.seconds for s in stages),
        compile_seconds=sum(s.compile_seconds for s in stages),
    )
end

"""
Compile `(y, θ, ρ) -> G`, the values that the complete system's auxiliary slacks σ must equal
at a solution, so complete_initialization can start each σ on its defining equation. The
symbols :complete_rho and :complete_unused_eta must match the names used when the complete
system was built (ReducedGOOP's _generate_complete_kkt_system), so that the ρ inside
`slack_expressions` binds to the ρ argument here.
"""
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
        chunk_size=32,
    )
    function (y, θ, ρ)
        G = zeros(length(kkt.slack_expressions))
        raw!(G, y, θ, ρ, 0.0)
        G
    end
end

"""
Fixed-primal view of a complete KKT system, for `solve(Scholtes(), system, [θ; x]; w₀)`. Every
complete equation is kept; the physical columns x become parameters and the other (free) columns
are the unknowns, so the system is rectangular (n_nc rows, n minus the physical columns). Since x
enters as a parameter, one compiled system serves every trial.
"""
function fixed_primal_formulation(kkt)
    physical = collect(kkt.primal_dims)
    free = setdiff(collect(1:kkt.n), physical)
    nθ, nx = kkt.parameter_dimension, length(physical)
    # y = [x; everything else], so the free columns are exactly nx+1:n and the complete system's
    # sign-constrained coordinates shift down by nx.
    free == collect((nx+1):kkt.n) || error("expected the physical columns to come first in y")
    full = zeros(kkt.n)
    full_J = copy(kkt.Jnc_buf)
    # `slots` has full_J's sparsity pattern with values 1:nnz. Its free columns list, in CSC
    # order, which entries of nonzeros(full_J) form J = full_J[:, free], so J! can refresh J's
    # values in place without building a new sparse matrix.
    rows, cols, _ = findnz(full_J)
    slots = sparse(rows, cols, collect(1:nnz(full_J)), size(full_J)...)
    selected_slots = nonzeros(slots[:, free])
    J = copy(full_J[:, free])
    θ_buffer = zeros(nθ)
    function fill_full!(v, θx)
        full[physical] .= view(θx, (nθ+1):(nθ+nx))
        full[free] .= v
        θ_buffer .= view(θx, 1:nθ)
        full
    end
    F! = (out, v, θx, ρ) -> kkt.F_nc!(out, fill_full!(v, θx), θ_buffer, ρ)
    J! = function (J_out, v, θx, ρ)
        kkt.J_nc!(full_J, fill_full!(v, θx), θ_buffer, ρ)
        nonzeros(J_out) .= view(nonzeros(full_J), selected_slots)
        J_out
    end
    # The unrelaxed residual that _norm_K0 evaluates for :aggregate (exact ignores it).
    function complete_residual!(out, v, θx, ρ)
        kkt.complete_residual!(out, fill_full!(v, θx), θ_buffer, ρ)
    end
    nonnegative_dims = kkt.nonnegative_dims .- nx
    # ScholtesKKTSystem and CompleteKKTSystem are ReducedGOOP internals; this builds them the way
    # _generate_complete_kkt_system does (src/scholtes_complete_kkt.jl), with no new codegen.
    no_rows = spzeros(0, length(free))
    inner = ReducedGOOP.ScholtesKKTSystem(
        length(free),                     # n: unknowns are the free columns
        kkt.n_nc,                         # every complete equation
        0,                                # no complementarity block, as in the complete system
        1:0,                              # no primal coordinates among the unknowns
        kkt.preference_slack_dims .- nx,
        Int[],
        nθ + nx,                          # parameters [θ; x]
        F!,
        kkt.a!,
        kkt.b!,
        J!,
        kkt.J_a!,
        kkt.J_b!,
        J,
        no_rows,
        copy(no_rows),
        findnz(J)[1:2],
        (Int[], Int[]),
        (Int[], Int[]),
        Int[],
        0,
        NamedTuple[],
        kkt.vars[free],
        zeros(nθ + nx),
        Ref(0.0),
        zeros(length(free)),
        Ref(false),
    )
    system = ReducedGOOP.CompleteKKTSystem(inner, (; kkt.formulation, complete_residual!, nonnegative_dims))
    (; free, physical, system, dimensions=(; rows=kkt.n_nc, columns=length(free), full_columns=kkt.n))
end

"""
Index map used to start complete solves: `complete_y[destination] .= reduced_w[source]`.
Maps every preference slack and, for each player, the innermost level's multipliers of the
ORIGINAL equalities (λ) and inequalities (γ). Outer-level multipliers belong to different
rows in the two systems (ψ/φ in reduced, inherited lower-level KKT rows in complete), so
they start at 0 or positive_seed (see complete_initialization).
"""
function variable_mapping(reduced, complete)
    reduced_names = Dict(string(v) => j for (j, v) in enumerate(reduced.vars))
    complete_names = Dict(string(v) => j for (j, v) in enumerate(complete.vars))
    source, destination, labels = Int[], Int[], String[]
    offset = first(reduced.preference_slack_dims)-1
    for (i, levels) in enumerate(complete.levels), k in eachindex(levels)
        level = levels[k]
        for (j, v) in enumerate(level.preference_slack)
            push!(source, offset+j)
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
        physical=collect(zip(reduced.primal_dims, complete.primal_dims)),
        unmapped="Other equality/induced multipliers start at zero; other nonnegative coordinates use the configured positive seed.",
    )
end

"""
Starting point for a complete solve from an accepted reduced solution. The order matters:
 1. physical coordinates <- reduced z;
 2. every nonnegative coordinate <- positive_seed;
 3. mapped preference slacks and multipliers <- reduced w (variable_mapping);
 4. nonnegative coordinates floored at positive_floor;
 5. auxiliary slacks σ <- their defining expressions (compile_slack_targets) at the first ρ
    of the solve (0 for exact, first(complete_rho_schedule) for aggregate), floored as in 4.
"""
function complete_initialization(formulation, output, θ, settings)
    kkt = formulation.kkt
    y = zeros(kkt.n)
    y[kkt.primal_dims] .= output.z
    y[kkt.nonnegative_dims] .= settings.positive_seed
    y[formulation.mapping.destination] .= output.w[formulation.mapping.source]
    y[kkt.nonnegative_dims] .= max.(y[kkt.nonnegative_dims], settings.positive_floor)
    ρ = formulation.name === :exact ? 0.0 : first(settings.complete_rho_schedule)
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
    retry_values=settings.retry_eta,
    context,
    io=stdout,
)
    kkt = formulation.kkt
    # exact is unrelaxed: one stage at ρ = 0. aggregate runs down complete_rho_schedule (ρ plays
    # the legacy ε). Each retry restarts from `initial` with the next η.
    schedule = formulation.name === :exact ? [0.0] : settings.complete_rho_schedule
    # :warmstart solves the whole complete system. :fixed_primal solves the fixed-primal system,
    # whose unknowns are the free columns and whose parameters are [θ; x], so x cannot move.
    # Note: until 2026-10-22 :fixed_primal used NonlinearSolve's BoundedTrustRegion. On these
    # rank-deficient systems it stopped far from a stationary point (projected gradient ≫ 0) after
    # 1000+ steps, while Scholtes on the fixed-primal system met the tolerance in under 100.
    fixed_mode = mode.name === :fixed_primal
    system = fixed_mode ? formulation.fixed.system : kkt
    parameters = fixed_mode ? [θ; physical] : θ
    unknowns = fixed_mode ? formulation.fixed.free : Colon()
    residuals = ReducedGOOP.ScholtesResiduals(system)
    residual_scale = sqrt(kkt.n_nc)
    attempts = Any[]
    for (attempt, eta) in enumerate([settings.eta_0; retry_values])
        y = copy(initial)
        stages = Any[]
        best = nothing
        try_label =
            attempt == 1 ? "first try" :
            attempt == 2 ? "retry (first try rejected)" : "retry $(attempt - 1) (earlier tries rejected)"
        print_kkt_header(
            "$context | complete :$(formulation.name), :$(mode.name) mode | $try_label: η₀ = $(compact(eta))";
            io,
        )
        for ρ in schedule
            stage_initial = copy(y)
            history = Any[]
            trace =
                nt -> push!(
                    history,
                    (;
                        raw=norm(nt.r),
                        unrelaxed=ReducedGOOP._norm_K0(residuals, nt.r, nt.w, ρ),
                        eta=nt.eta,
                    ),
                )
            # Same options as solve_reduced, with the complete tolerance and this attempt's η.
            # The 1e-12 floor is defensive: exact ignores ρ (its residual is evaluated at ρ = 0)
            # and complete_rho_schedule stays above it, so it changes nothing today; for exact
            # it only keeps ρ and tol_inner positive. As in solve_reduced, the inner loop stops
            # at ‖R‖/√m < min(ρ, inner_ρ_tol).
            options = ReducedGOOP.ScholtesOptions(;
                rho_schedule=[max(Float64(ρ), 1e-12)],
                max_inner=settings.max_inner_iters,
                tol=settings.complete_tol * residual_scale,
                tol_inner=min(max(ρ, 1e-12), settings.inner_ρ_tol) * residual_scale,
                projected_step=true,
                proj_eps=1e-8,
                linear_solver=settings.linear_solver,
                eta_schedule=true,
                eta_init=eta,
                eta_min=1e-8,
                eta_max=1e4,
                gain_low=0.25,
                gain_high=0.75,
                tightening_rate=1.2,
                loosening_rate=3.0,
            )
            timed = @timed ReducedGOOP.solve(
                ReducedGOOP.Scholtes(),
                system,
                parameters;
                w₀=y[unknowns],
                options,
                trace,
            )
            output = timed.value
            y[unknowns] .= output.w
            push!(
                history,
                (;
                    raw=output.relaxed_residual,
                    unrelaxed=output.residual,
                    eta=output.eta,
                ),
            )
            push!(
                stages,
                (;
                    ρ,
                    parameter=formulation.name===:exact ? :unrelaxed :
                              :epsilon_complete,
                    status=output.status,
                    reason=output.termination_reason,
                    iterations=output.iters,
                    status_ok=output.status === :solved,
                    seconds=max(
                        0.0,
                        timed.time-timed.compile_time-timed.recompile_time,
                    ),
                    wall_seconds=timed.time,
                    compile_seconds=timed.compile_time+timed.recompile_time,
                    history,
                    initial=stage_initial,
                    final=copy(y),
                ),
            )
            # Residuals of the full complete iterate, for the table and the best-stage choice.
            r, r0 = zeros(kkt.n_nc), zeros(kkt.n_nc)
            kkt.F_nc!(r, y, θ, ρ)
            kkt.F_nc!(r0, y, θ, 0.0)
            print_kkt_stage(;
                ρ,
                status=output.status,
                iterations=output.iters,
                seconds=last(stages).seconds,
                relaxed_rms=norm(r)/sqrt(kkt.n_nc),
                unrelaxed_rms=norm(r0)/sqrt(kkt.n_nc),
                physical=y[kkt.primal_dims],
                settings,
                stop=Intersection.solver_stop_reason(output, options; scale=residual_scale),
                io,
            )
            # Track the stage with the lowest ‖K0‖ (a non-finite residual counts as Inf).
            k0 = isfinite(norm(r0)) ? norm(r0) : Inf
            if isnothing(best) || k0 < best.k0
                best = (; y=copy(y), ρ, k0, index=length(stages))
            end
        end
        # Judge and keep the best stage, not the last, as in solve_reduced.
        y = best.y
        r, r0 = zeros(kkt.n_nc), zeros(kkt.n_nc)
        kkt.F_nc!(r, y, θ, best.ρ)
        kkt.F_nc!(r0, y, θ, 0.0)
        relaxed_rms, unrelaxed_rms = norm(r)/sqrt(kkt.n_nc), norm(r0)/sqrt(kkt.n_nc)
        feasibility = ReducedGOOP.is_feasible(problem, y[kkt.primal_dims], θ) # only `worst` is used
        x = BlockArray(y[kkt.primal_dims], problem.primal_dims)
        θ_blocks = BlockArray(θ, problem.parameter_dims)
        equality_max = maximum(
            abs,
            reduce(vcat, (f(x, θ_blocks) for f in problem.equality_constraints)),
        )
        sign_violation = max(0.0, -minimum(y[kkt.nonnegative_dims]; init=Inf))
        # The full residual retains stationarity, equality, complementarity, and slack equations.
        # Acceptance depends only on its unrelaxed (ρ = 0) RMS, not on the solver's exit status;
        # every other field is a saved diagnostic.
        validation = (;
            relaxed_rms,
            unrelaxed_rms,
            max_abs_residual=maximum(abs, r),
            sign_violation,
            primal=(;
                equality_max,
                inequality_violation=max(0.0, -feasibility.worst),
            ),
            accepted=unrelaxed_rms <= settings.complete_tol,
        )

        accepted = validation.accepted
        push!(
            attempts,
            (;
                attempt,
                eta,
                initial=copy(initial),
                stages,
                validation,
                accepted,
                best_stage=best.index,
                solution=copy(y),
            ),
        )
        accepted && break
    end
    # Keep the best try, as a ρ chain keeps its best stage: a retry restarts from `initial` and can
    # end worse than an earlier try. An accepted try is the last and has the lowest ‖K0‖.
    best_attempt = argmin([isfinite(a.validation.unrelaxed_rms) ? a.validation.unrelaxed_rms : Inf for a in attempts])
    final = attempts[best_attempt]
    mode.name === :fixed_primal && @assert final.solution[kkt.primal_dims] == physical
    (;
        accepted=final.accepted,
        attempts,
        best_attempt,
        validation=final.validation,
        physical=final.solution[kkt.primal_dims],
        solution=final.solution,
        status=final.stages[final.best_stage].status,
        reason=final.stages[final.best_stage].reason,
        iterations=sum(s.iterations for a in attempts for s in a.stages),
        seconds=sum(s.seconds for a in attempts for s in a.stages),
        compile_seconds=sum(s.compile_seconds for a in attempts for s in a.stages),
    )
end

# ── Metrics, saved results and final summary ───────────────────────────────────

"""
Differences between the reduced and complete trajectories, per group (position, velocity,
control) and overall after dividing by settings.error_scales. `small` decides between a small
and a substantial adjustment. In fixed-primal mode the trajectories are identical, so every
metric is 0.
"""
function trajectory_metrics(reduced, complete, settings)
    reduced_columns, complete_columns = reshape(reduced, 6, :), reshape(complete, 6, :)
    groups = ((:position, 1:2), (:velocity, 3:4), (:control, 5:6))
    errors = map(groups) do (name, rows)
        reference = vec(reduced_columns[rows, :])
        δ = vec(complete_columns[rows, :])-reference
        scale = getproperty(settings.error_scales, name)
        (;
            name,
            absolute=norm(δ),
            relative=norm(δ)/max(norm(reference), 1e-12scale*sqrt(length(δ))),
            rms=norm(δ)/sqrt(length(δ)),
            maximum=maximum(abs, δ),
            scaled=δ/scale,
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
        maximum=maximum_error,
        relative=norm(δ)/max(norm(reference), 1e-12sqrt(length(δ))),
        groups=errors,
        small=all(
            norm(g.scaled)/sqrt(length(g.scaled))<=settings.small_rms for g in errors
        ) && maximum_error<=settings.small_max,
    )
end

"Write results.jls via a temporary file and a rename, so an interrupted save never leaves a truncated file."
function save_results(directory, settings, instances, sampling, metadata, results)
    payload = (;
        settings,
        instances,
        sampling,
        metadata,
        results,
        julia=string(VERSION),
        blas_threads=BLAS.get_num_threads(),
    )
    temporary = joinpath(directory, "results.jls.tmp")
    serialize(temporary, payload)
    mv(temporary, joinpath(directory, "results.jls"); force=true)
end

function print_final_summary(results, settings; io=stdout)
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
    if :warmstart in settings.comparison_modes
        println(
            io,
            "\nWarm-started complete solves (trajectory may change; not a fixed-primal certificate):",
        )
        @printf(io, "%-12s %16s %16s\n", "objective", "aggregate warm", "exact warm")
        for variant in settings.objective_variants
            converged =
                filter(row -> row.variant === variant && row.reduced.accepted, results)
            nred = length(converged)
            aggregate = count(row -> passed(row, :aggregate, :warmstart), converged)
            exact = count(row -> passed(row, :exact, :warmstart), converged)
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

# ── Plots and report ───────────────────────────────────────────────────────────

const MONO_FONT = "DejaVu Sans Mono" # ships with Makie, so report.pdf renders the same everywhere

"""
Plot the geometric mean of `histories` at each trace index (line) with ± one standard
deviation of the log10 values (band, only where at least two histories contribute).
"""
function plot_history!(axis, histories, label, color)
    isempty(histories) && return
    average, deviation, counts = Float64[], Float64[], Int[]
    for j in 1:maximum(length, histories)
        values = [
            log10(max(h[j], 1e-16)) for
            h in histories if length(h)>=j && isfinite(h[j]) && h[j]>=0
        ]
        isempty(values) && break
        push!(average, mean(values))
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
        markersize=4,
        linewidth=2,
    )
    eligible = findall(>=(2), counts)
    if !isempty(eligible)
        CM.band!(
            axis,
            x[eligible],
            10.0 .^ (average[eligible]-deviation[eligible]),
            10.0 .^ (average[eligible]+deviation[eligible]);
            color=(color, 0.2),
        )
    end
end

"Intersection's map and component profiles, with reduced and complete iterates side by side."
function plot_trial_trajectories!(layout, row, check, settings)
    CM.Label(
        layout[1, 1:2],
        "Trial $(row.trial) | circle: start; star: goal; dotted: bound / preference";
        fontsize=20,
    )
    complete = haskey(check, :result) ? check.result : nothing
    # Subtitles give K0/sqrt(m) of the plotted iterate: the reduced solve's best stage, and the best
    # stage of the complete check's best try. An exception plots the reduced trajectory and has no K0.
    reduced_k0 = let stage = row.reduced.stages[row.reduced.best_stage]
        "K0/sqrt(m) = $(compact(row.reduced.validation.unrelaxed_rms)) at rho = $(compact(stage.ρ))"
    end
    complete_k0 =
        isnothing(complete) ? "not run" :
        isempty(complete.attempts) ? "no K0: the complete solve threw an exception" :
        let final = complete.attempts[complete.best_attempt], stage = final.stages[final.best_stage]
            "K0/sqrt(m) = $(compact(final.validation.unrelaxed_rms)) at rho = $(compact(stage.ρ)), try $(final.attempt)"
        end
    for (column, name, physical, status, accepted, k0) in (
        (
            1,
            "Reduced",
            row.reduced.output.z,
            row.reduced.output.status,
            row.reduced.accepted,
            reduced_k0,
        ),
        (
            2,
            "Complete",
            isnothing(complete) ? nothing : complete.physical,
            isnothing(complete) ? :skipped : complete.status,
            !isnothing(complete) && complete.accepted,
            complete_k0,
        ),
    )
        ax = CM.Axis(
            layout[2, column];
            title="$name | $status | accepted=$accepted",
            subtitle=k0,
            aspect=CM.DataAspect(),
            xlabel="x [m]",
            ylabel="y [m]",
            xgridvisible=false,
            ygridvisible=false,
        )
        Intersection.draw_intersection_map!(
            ax;
            map_end=settings.map_end,
            lane_width=settings.lane_width,
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
                text="Skipped: reduced solve failed",
                space=:relative,
                align=(:center, :center),
            )
            continue
        end
        # Same stand-in `dynamics` as in print_kkt_stage.
        strategies = Intersection.extract_player_strategies(
            physical,
            2,
            6settings.planning_horizon,
            (; state_dimension=4, control_dimension=2),
        )
        for player in 1:2
            xs = strategies[player].xs
            color = Intersection.player_plot_color(player)
            CM.scatterlines!(
                ax,
                [x[1] for x in xs],
                [x[2] for x in xs];
                color,
                linewidth=3,
                markersize=10,
            )
            CM.scatter!(
                ax,
                [xs[1][1]],
                [xs[1][2]];
                color,
                marker=:circle,
                markersize=16,
            )
            goal = (settings.goal_position1, settings.goal_position2)[player]
            CM.scatter!(ax, [goal[1]], [goal[2]]; color, marker=:star5, markersize=22)
        end
        velocities = CM.GridLayout(layout[3, column])
        accelerations = CM.GridLayout(layout[4, column])
        for component in 1:2
            direction = component == 1 ? "x" : "y"
            velocity = CM.Axis(
                velocities[1, component];
                title="Velocity v$direction",
                xlabel="time [s]",
                ylabel="v$direction [m/s]",
            )
            acceleration = CM.Axis(
                accelerations[1, component];
                title="Acceleration a$direction",
                xlabel="time [s]",
                ylabel="a$direction [m/s^2]",
            )
            for player in 1:2
                color = Intersection.player_plot_color(player)
                xs, us = strategies[player].xs, strategies[player].us
                CM.lines!(
                    velocity,
                    (0:(length(xs)-1)) .* settings.time_step,
                    [x[component+2] for x in xs];
                    color,
                    linewidth=2,
                )
                CM.lines!(
                    acceleration,
                    (0:(length(us)-1)) .* settings.time_step,
                    [u[component] for u in us];
                    color,
                    linewidth=2,
                )
                bounds = Intersection.player_speed_bounds(
                    player;
                    min_speed=settings.min_speed,
                    max_speed=settings.max_speed,
                )
                CM.hlines!(
                    velocity,
                    [bounds.lower[component], bounds.upper[component]];
                    color,
                    linestyle=:dot,
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
                CM.hlines!(acceleration, limits; color=:black, linestyle=:dot)
        end
    end
    CM.Legend(
        layout[5, 1:2],
        [
            CM.LineElement(color=Intersection.player_plot_color(player), linewidth=3)
            for player in 1:2
        ],
        ["Player 1", "Player 2"];
        orientation=:horizontal,
        framevisible=false,
    )
    CM.rowsize!(layout, 2, 340)
    CM.rowsize!(layout, 3, 180)
    CM.rowsize!(layout, 4, 180)
    CM.rowgap!(layout, 25)
    CM.colgap!(layout, 35)
end

"""
Draw report.pdf from results.jls alone, without building or solving any KKT system: a summary
page, then one page per variant/formulation/mode.
"""
function plot_results(directory)
    data = deserialize(joinpath(directory, "results.jls"))
    (; settings) = data
    pages = [summary_page(data, directory)]
    for variant in settings.objective_variants,
        formulation in settings.complete_formulations,
        mode in settings.comparison_modes

        println("Plotting $variant, :$formulation, :$mode mode...")
        flush(stdout)
        push!(pages, comparison_page(data, variant, formulation, mode))
    end
    path = joinpath(directory, "report.pdf")
    Intersection._with_filesystem_retry(description="report save to $path") do
        save_multipage_pdf(path, pages)
    end
    directory
end

"""
Write `figures` as the pages of one PDF, each page sized to its figure. CairoMakie's `save`
writes one figure per file, so this drives its Cairo backend directly: one PDF surface,
resized, drawn and emitted (`show_page`) once per page, then finished. It relies on CairoMakie internals (`Screen` built
on a surface, `cairo_draw`, `scaled_scene_resolution`); checked with CairoMakie 0.15.
"""
function save_multipage_pdf(path, figures)
    config = CM.Makie.merge_screen_config(CM.ScreenConfig, Dict{Symbol,Any}())
    surface = nothing
    for figure in figures
        CM.Makie.update_state_before_display!(figure)
        scene = CM.Makie.get_scene(figure)
        width, height = CM.scaled_scene_resolution(CM.PDF, config, scene)
        if isnothing(surface)
            surface = CM.Cairo.CairoPDFSurface(path, width, height)
        else
            ccall(
                (:cairo_pdf_surface_set_size, CM.Cairo.Cairo_jll.libcairo),
                Cvoid,
                (Ptr{Cvoid}, Cdouble, Cdouble),
                surface.ptr,
                width,
                height,
            )
        end
        screen = CM.Screen(scene, config, surface)
        CM.Makie.push_screen!(scene, screen)
        CM.cairo_draw(screen, scene)
        CM.Cairo.show_page(screen.context)
    end
    CM.Cairo.finish(surface)
    path
end

"Text page that opens report.pdf: tolerances, pass counts, dimensions, settings, method notes and failures."
function summary_page(data, directory)
    (; settings, results) = data
    figure = Intersection.serif_figure(size=(1600, 1200), fontsize=18, figure_padding=40)
    row = Ref(0)
    next_row() = (row[] += 1)
    label!(text; kwargs...) = CM.Label(
        figure[next_row(), 1],
        text;
        halign=:left,
        justification=:left,
        kwargs...,
    )
    heading!(text) = label!(text; fontsize=24)
    # Labels sized to their text sit at the left. Only long prose wraps: a wrapped Label fills the
    # column and centres its lines, which reads as left-aligned because the lines are full.
    paragraph!(text) = label!(text; word_wrap=true, tellwidth=false)
    monospace!(text) = label!(text; font=MONO_FONT, fontsize=14)
    number(x) = @sprintf("%.3e", x)

    label!("Intersection KKT comparison: $(settings.num_instances) instances per objective variant"; fontsize=32)
    label!("Run $(basename(directory)); Julia $(data.julia), $(data.blas_threads) BLAS threads.")
    monospace!(tolerance_line(settings))

    heading!("Pass counts")
    buffer = IOBuffer()
    print_final_summary(results, settings; io=buffer)
    monospace!(strip(String(take!(buffer))))
    lines = [@sprintf("%-10s %-11s %-19s %8s %21s %14s", "objective", "complete", "mode", "reduced", "conditional complete", "overall pairs")]
    for variant in settings.objective_variants,
        formulation in settings.complete_formulations,
        mode in settings.comparison_modes

        variant_rows = filter(r -> r.variant === variant, results)
        nred = count(r -> r.reduced.accepted, variant_rows)
        passed = count(
            r -> any(
                c -> c.formulation === formulation && c.mode === mode && haskey(c, :result) && c.result.accepted,
                r.checks,
            ),
            variant_rows,
        )
        n = length(variant_rows)
        push!(lines, @sprintf("%-10s %-11s %-19s %8s %21s %14s", variant, ":$formulation", ":$mode mode", "$nred/$n", "$passed/$nred", "$passed/$n"))
    end
    monospace!(join(lines, "\n"))

    heading!("Dimensions")
    lines = [@sprintf("%-10s %-11s %14s %14s %30s", "objective", "system", "rows retained", "full unknowns", "unknowns with physical fixed")]
    for metadata in data.metadata, formulation in metadata.complete
        d = formulation.dimensions
        push!(lines, @sprintf("%-10s %-11s %14d %14d %30d", metadata.variant, ":$(formulation.name)", d.rows, d.full_columns, d.columns))
    end
    monospace!(join(lines, "\n"))

    heading!("Settings")
    entries = ["$k = $v" for (k, v) in pairs(settings) if k !== :output_dir]
    half = cld(length(entries), 2)
    columns = CM.GridLayout(figure[next_row(), 1]; halign=:left)
    for (column, part) in enumerate((entries[1:half], entries[(half+1):end]))
        CM.Label(
            columns[1, column],
            join(part, "\n");
            font=MONO_FONT,
            fontsize=13,
            halign=:left,
            justification=:left,
        )
    end
    CM.colgap!(columns, 80)

    heading!("Method notes")
    horizon = settings.planning_horizon
    label!(
        "$horizon state knots, $(horizon - 1) intervals, time_step = $(settings.time_step) s; horizon = $((horizon - 1) * settings.time_step) s. Physical primals are all state/control entries, including terminal controls.",
    )
    paragraph!(
        "Initial-state sampling uses independent uniform deviations within base_initial_state +/- perturbation_scale, per player; zero-scale coordinates remain fixed. Initial lane and separation margins are checked in metres and saved for every draw, including rejections. The original hard collision inequalities apply from knot 2; initial separation is additionally checked at knot 1. No continuous-time safety claim is made.",
    )
    paragraph!(
        "Both systems use identical physical ordering: player, time knot, [px, py, vx, vy, ax, ay]. Preference slacks map by player/source preference/component, not flat index order. The innermost original equality and inequality multipliers also map by player and original row. All other equality/induced multipliers start at zero; remaining sign-constrained variables use positive_seed, followed by auxiliary-slack initialization from their defining expressions. Full source/destination index maps and labels are saved in results.jls.",
    )
    paragraph!(
        "Every solve uses Scholtes; each rho stage's inner loop stops at R/sqrt(m) < min(rho, $(settings.inner_ρ_tol)), at max_inner_iters, or on a line-search stall, unless K0 reaches its tolerance first. :fixed_primal recovery solves the complete system with the physical trajectory as a parameter, so only the remaining columns move; nonnegative coordinates stay inside the sign box. Exact complete equations have no epsilon schedule; aggregate uses complete_rho_schedule as legacy epsilon. All complete rows remain after fixing physical columns.",
    )
    paragraph!(
        "Residual curves show the full relaxed and unrelaxed L2 residuals divided by sqrt(number of equations) on logarithmic axes, concatenated across stages, against solver-trace samples (Scholtes iterates). Reduced curves include all sampled trials; complete curves include all attempted checks, using each check's kept (best) try. Histograms use accepted pairs only. Curves use available samples only: no padding after a run ends. Lines are geometric means; bands are mean +/- one sample standard deviation in log10 residual, drawn at indices with at least two contributors. Stage endpoints and raw histories are in results.jls. Each comparison page also shows every trial's reduced and complete position, velocity, and acceleration profiles side by side, including failed returned iterates with explicit status labels.",
    )
    paragraph!(
        "A solve is accepted iff its unrelaxed KKT residual K0/sqrt(m) is at most $(settings.tol) (reduced) or $(settings.complete_tol) (complete); the solver's exit status is not required. A rho chain runs all its stages and is judged at the stage with the lowest K0, and a complete check keeps its best try (retries restart with the next eta); that iterate is the one kept, compared and plotted. Fixed trajectories retain the reduced solution's finite physical residual; these rows are not removed. The full residual includes stationarity, equality, complementarity, and auxiliary-slack equations. The relaxed residual, largest residual entry, multiplier signs and original physical feasibility are recorded as diagnostics only. Failures do not prove incompatibility.",
    )
    paragraph!(
        ":warmstart complete times are refinement costs. Construction and independent warm-ups are excluded. Reported solve seconds subtract Julia's measured compilation and recompilation time from wall time (clamped at zero); wall and compilation seconds are saved separately, and nonzero measured compilation is flagged below. Both starts and every attempt/stage are saved. No warm-up or previous-trial solution seeds measured trials.",
    )
    paragraph!(
        "Trajectory normalization: position = $(settings.error_scales.position) m, velocity = $(settings.error_scales.velocity) m/s, control = $(settings.error_scales.control) m/s^2. Small adjustment requires each group's scaled RMS <= $(settings.small_rms) and max scaled component <= $(settings.small_max). Relative norms use a 1e-12 RMS denominator floor. Statistics here are descriptive, not evidence of theoretical equivalence.",
    )

    heading!("Failures and measured compilation")
    failures = String[]
    for row in results
        reduced = row.reduced
        reduced.compile_seconds > 0 && push!(
            failures,
            "$(row.variant) trial $(row.trial): reduced measured compilation $(reduced.compile_seconds) s",
        )
        reduced.accepted || push!(
            failures,
            "$(row.variant) trial $(row.trial): reduced rejected; status=$(reduced.output.status), reason=$(reduced.output.termination_reason), R=$(number(reduced.validation.relaxed_rms)), K0=$(number(reduced.validation.unrelaxed_rms)), physical equality=$(number(reduced.validation.primal.equality_max)), inequality=$(number(reduced.validation.primal.inequality_violation))",
        )
        for check in row.checks
            haskey(check, :result) || continue
            complete = check.result
            name = "$(row.variant) trial $(row.trial), :$(check.formulation), :$(check.mode) mode"
            complete.accepted || push!(
                failures,
                "$name: $(complete.status), $(complete.reason); R=$(number(complete.validation.relaxed_rms)), K0=$(number(complete.validation.unrelaxed_rms)), physical equality=$(number(complete.validation.primal.equality_max)), inequality=$(number(complete.validation.primal.inequality_violation)), attempts=$(length(complete.attempts))",
            )
            # The full backtrace stays in results.jls; the page shows its first lines.
            haskey(complete, :error) &&
                push!(failures, "  Exception: " * join(first(split(complete.error, '\n'), 5), "\n    "))
            complete.compile_seconds > 0 &&
                push!(failures, "$name: measured compilation $(complete.compile_seconds) s")
        end
    end
    # Wrapped by hand (monospace widths are exact) so the block stays left-aligned.
    function wrap(line; width=170)
        wrapped, current = String[], ""
        for word in split(line, ' ')
            if !isempty(current) && length(current) + 1 + length(word) > width
                push!(wrapped, current)
                current = "    " * word
            else
                current = isempty(current) ? String(word) : current * " " * word
            end
        end
        join(push!(wrapped, current), "\n")
    end
    monospace!(isempty(failures) ? "none" : join(wrap.(failures), "\n"))

    CM.colsize!(figure.layout, 1, CM.Fixed(1520))
    CM.rowgap!(figure.layout, 16)
    CM.resize_to_layout!(figure)
    figure
end

"One report page: residual histories, the acceptance histogram, then every trial's trajectories."
function comparison_page(data, variant, formulation, mode)
    (; settings, results) = data
    rows = filter(r->r.variant===variant, results)
    pairs = [
        (; row, check) for row in rows for check in row.checks if
                                           check.formulation===formulation && check.mode===mode
    ]
    nred = count(r->r.reduced.accepted, rows)
    successful = filter(p->haskey(p.check, :result) && p.check.result.accepted, pairs)
    attempted = filter(p -> haskey(p.check, :result) && !isempty(p.check.result.attempts), pairs)
    figure = Intersection.serif_figure(size=(1600, 480+900length(rows)), fontsize=18, figure_padding=30)
    CM.Label(figure[1, 1:2], "$variant | :$formulation, :$mode mode"; fontsize=26)
    convergence = CM.GridLayout(figure[2, 1:2])
    metadata = only(filter(m->m.variant===variant, data.metadata))
    complete_rows = only(filter(p->p.name===formulation, metadata.complete)).dimensions.rows
    for (column, field, title) in ((1, :unrelaxed, "True KKT residual"), (2, :raw, "Relaxed residual"))
        axis = CM.Axis(
            convergence[1, column];
            xlabel="Trace sample (stages concatenated)",
            ylabel="Residual / sqrt(m)",
            yscale=log10,
            title="$title\nReduced n=$(length(rows)); complete n=$(length(attempted))",
        )
        histories = [
            reduce(
                vcat,
                (
                    [getproperty(h, field)/sqrt(complete_rows) for h in s.history] for
                    s in p.check.result.attempts[p.check.result.best_attempt].stages
                ),
            ) for p in attempted
        ]
        reduced_histories = [
            reduce(
                vcat,
                ([getproperty(h, field)/sqrt(metadata.reduced.rows) for h in s.history] for s in row.reduced.stages),
            ) for row in rows
        ]
        if isempty(histories) && isempty(reduced_histories)
            CM.text!(axis, 0.5, 0.5; text="No solver histories", space=:relative, align=(:center, :center))
        else
            plot_history!(axis, histories, "Complete", :navy)
            plot_history!(axis, reduced_histories, "Reduced", :darkorange)
            CM.hlines!(axis, [settings.tol]; color=:darkorange, linestyle=:dot)
            CM.hlines!(axis, [settings.complete_tol]; color=:navy, linestyle=:dot)
            CM.axislegend(axis; position=:rt, labelsize=14)
        end
    end
    # At most 4 x ticks, labelled like the console (9.4e-4): the default labels (9.4×10⁻⁴, ~9 ticks)
    # overlap whenever the values span less than a decade.
    second = CM.Axis(
        convergence[1, 3],
        xticks=CM.WilkinsonTicks(4),
        xtickformat=values -> compact.(values),
        xlabel=mode===:fixed_primal ? "Unrelaxed residual RMS (best stage)" : "Scaled trajectory RMS",
        ylabel="Accepted pairs",
        title=mode===:fixed_primal ? "Fixed-primal recovery $(length(successful))/$nred" :
              "Trajectory adjustment (successful pairs)",
    )
    values = [
        mode===:fixed_primal ? p.check.result.validation.unrelaxed_rms : p.check.metrics.rms for
        p in successful
    ]
    if isempty(values)
        CM.text!(second, 0.5, 0.5; text="No accepted pairs", space=:relative, align=(:center, :center))
    else
        CM.hist!(second, values; bins=max(1, min(10, length(values))), color=:steelblue)
    end
    for (index, row) in enumerate(rows)
        check = only(filter(c -> c.formulation === formulation && c.mode === mode, row.checks))
        layout = CM.GridLayout(figure[index+2, 1:2])
        plot_trial_trajectories!(layout, row, check, settings)
    end
    CM.rowsize!(figure.layout, 2, 300)
    CM.rowgap!(figure.layout, 30)
    CM.resize_to_layout!(figure)
    figure
end

end
