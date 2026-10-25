"""
Monte Carlo check of whether reduced-GOOP solutions also solve the complete KKT system,
on the two-player intersection game from Intersection.jl.

For every objective variant and every sampled initial state:
 1. Solve the REDUCED KKT system with Scholtes down `rho_schedule`, starting from
    Intersection's yielding initial guess. Every solve, reduced or complete, is accepted iff
    its unrelaxed KKT residual ‖K₀‖/√m is at most `tol` (reduced) or `complete_tol`. A ρ chain
    runs all its stages and is judged, and kept, at the stage with the lowest ‖K₀‖; a complete
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
  R, K₀      residual norm at the stage's ρ, and at ρ = 0; always reported as RMS,
             norm / √(number of equations).
  physical   the stacked trajectory z, ordered player, knot, [px, py, vx, vy, ax, ay].

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
            objective=(d, a) -> d^2,
            control_scale=settings.control_shape_scale,
            goal_scale=settings.goal_shape_scale,
        ),
        (;
            name=:cosh,
            objective=(d, a) -> 2a^2 * (cosh(d/a)-1),
            control_scale=settings.control_shape_scale,
            goal_scale=settings.goal_shape_scale,
        ),
    ]
end

"""
Intersection.get_setup.
objective_variants is the one definition of each objective. Each player's
preferences run lowest -> highest priority (see Intersection.get_setup):
    P1: [control, speed (prioritized constraint), goal]
    P2: [control, goal, speed (prioritized constraint)]
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
    preferences[1][1] = control_objective(; player=1)
    preferences[1][3] = goal_objective(; player=1)
    preferences[2][1] = control_objective(; player=2)
    preferences[2][2] = goal_objective(; player=2)
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

# Two players with planar double-integrator dynamics: state [px, py, vx, vy] and control [ax, ay] at
# every knot, so a player's slice of the stacked trajectory z is (4 + 2) × planning_horizon long.
const NUM_PLAYERS = 2
const STATE_DIMENSION = 4
const CONTROL_DIMENSION = 2

"Intersection's planar double-integrator dynamics at the settings' time step."
function intersection_dynamics(settings)
    Intersection.build_intersection_dynamics(
        Intersection.PlanarDoubleIntegrator();
        Δt=settings.time_step,
        state_dimension=STATE_DIMENSION,
        control_dimension=CONTROL_DIMENSION,
    )
end

"Each player's states `xs` and controls `us` from a stacked trajectory z."
function player_strategies(z, settings)
    # extract_player_strategies reads only these two dimensions from its `dynamics` argument.
    Intersection.extract_player_strategies(
        z,
        NUM_PLAYERS,
        (STATE_DIMENSION+CONTROL_DIMENSION) * settings.planning_horizon,
        (; state_dimension=STATE_DIMENSION, control_dimension=CONTROL_DIMENSION),
    )
end

"θ and Intersection's yielding initial guess for one sampled instance; the warm-up and every trial start here."
function instance_inputs(instance, flatten_parameters, dynamics, settings)
    initial_state1, initial_state2 = instance.states
    (; θ) = Intersection.build_instance_parameters(
        flatten_parameters,
        initial_state1,
        initial_state2,
        settings.goal_position1,
        settings.goal_position2,
        settings.obstacle_position,
    )
    (; warmstart_solution) = Intersection.build_yielding_warmstart(
        settings.planning_horizon,
        dynamics,
        initial_state1,
        initial_state2,
        settings.goal_position1;
        settings.min_speed,
        settings.max_speed,
    )
    (; θ, initial_guess=warmstart_solution)
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
    # Acceptance: a solve is accepted iff its unrelaxed KKT residual ‖K₀‖ / √(number of equations) <= tol,
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

# Driver steps, called by `run` in this order; each prints its own console lines.

"Default settings merged with `overrides`, then checked: an unknown name or an empty schedule is an error."
function validated_settings(overrides)
    defaults = default_settings()
    unknown = setdiff(keys(overrides), keys(defaults))
    isempty(unknown) || error("Unknown setting(s) in overrides: $(join(unknown, ", "))")
    settings = merge(defaults, overrides)
    settings.num_instances > 0 && settings.planning_horizon >= 2 || error("invalid num_instances or planning_horizon")
    schedules = (settings.rho_schedule, settings.complete_rho_schedule)
    all(s -> !isempty(s) && all(ρ -> isfinite(ρ) && ρ > 0, s), schedules) || error("invalid relaxation schedule")
    settings
end

"""
A directory for this run, named by its start time; a second run started in the same second gets
_2, _3, ... so no run overwrites another.
"""
function run_directory(output_dir)
    mkpath(output_dir)
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    directory = joinpath(output_dir, stamp)
    for suffix in Iterators.countfrom(2)
        ispath(directory) || break
        directory = joinpath(output_dir, "$(stamp)_$suffix")
    end
    mkdir(directory)
end

"""
One objective variant's problem and systems: the reduced KKT system, then for each complete
formulation its KKT system, σ targets, fixed-primal system and variable mapping. Prints the system
sizes, and returns everything with the variant's results.jls metadata (sizes, construction times).
"""
function build_systems(variant, builders, dynamics, settings)
    (; problem, flatten_parameters) = get_setup(
        NUM_PLAYERS;
        dynamics,
        settings.control_bounds,
        settings.planning_horizon,
        settings.collision_avoidance,
        settings.min_speed,
        settings.max_speed,
        settings.map_end,
        settings.lane_width,
        variant,
    )
    println("Building reduced KKT system for $(variant.name) variant...")
    flush(stdout)
    reduced_construction = @timed ReducedGOOP.generate_slacked_reduced_kkt_system(
        problem;
        complementarity=:scholtes,
        codegen=:fast_differentiation,
    )
    reduced_kkt = reduced_construction.value
    formulations = map(builders) do builder
        println("Building :$(builder.name) complete KKT system for $(variant.name) variant...")
        flush(stdout)
        complete_construction = @timed builder.build(problem)
        complete_kkt = complete_construction.value
        slack_targets = compile_slack_targets(complete_kkt)
        fixed = fixed_primal_formulation(complete_kkt)
        mapping = variable_mapping(reduced_kkt, complete_kkt)
        # The exact system is square; the fixed-primal Jacobian keeps every row but drops the
        # physical columns.
        builder.name === :exact && @assert complete_kkt.n == complete_kkt.n_nc
        @assert size(fixed.system.Jnc_buf) == (complete_kkt.n_nc, complete_kkt.n-length(complete_kkt.primal_dims))
        (;
            builder.name,
            complete_kkt,
            slack_targets,
            fixed,
            mapping,
            construction_seconds=complete_construction.time,
        )
    end
    metadata = (;
        variant=variant.name,
        objective=(; variant.control_scale, variant.goal_scale),
        # Reduced KKT system: variables y, equations F_nc and complementarity pairs; then the
        # Scholtes system the solver works on: rows of R(w; ρ), columns of w = [y; s; u].
        reduced=(;
            variables=reduced_kkt.n,
            equations=reduced_kkt.n_nc,
            complementarity=reduced_kkt.n_comp,
            rows=reduced_kkt.n_nc+2reduced_kkt.n_comp,
            columns=reduced_kkt.n+reduced_kkt.n_comp+ReducedGOOP._n_u(reduced_kkt),
            construction_seconds=reduced_construction.time,
        ),
        complete=[(; p.name, p.fixed.dimensions, p.mapping, p.construction_seconds) for p in formulations],
    )
    println("KKT system sizes, $(variant.name) variant:")
    foreach(line -> println("  ", line), size_lines(metadata))
    (; variant, problem, flatten_parameters, reduced_kkt, formulations, metadata)
end

"""
Untimed JIT warm-up on `instance`, so measured trials don't pay first-call compilation: one reduced
solve and one try of every complete check, results discarded. Not related to the :warmstart mode.
"""
function jit_warmup(systems, modes, instance, dynamics, settings)
    (; variant, problem, flatten_parameters, reduced_kkt, formulations) = systems
    (; θ, initial_guess) = instance_inputs(instance, flatten_parameters, dynamics, settings)
    print("Warm-up for $(variant.name) (untimed; compiles every solver path; results discarded)... ")
    flush(stdout)
    seconds = @elapsed begin
        context = "warm-up, $(variant.name)"
        reduced = solve_scholtes_reduced(reduced_kkt, problem, θ, initial_guess, settings; context, io=devnull)
        for formulation in formulations, mode in modes
            start = complete_initialization(formulation, reduced.output, θ, settings)
            solve_scholtes_complete(
                formulation,
                mode,
                problem,
                θ,
                reduced.output.z,
                start,
                settings;
                retry_values=Float64[],
                context,
                io=devnull,
            )
        end
    end
    println("done in $(round(seconds; digits=1)) s")
end

"""
One trial's reduced solve: print the trial banner, solve from Intersection's yielding initial guess,
print the verdict, and return the trial's record. Its `checks` start empty; `run` fills them.
"""
function solve_reduced_kkt(systems, trial, instance, dynamics, num_checks, settings)
    (; variant, problem, flatten_parameters, reduced_kkt) = systems
    println("\n══ Trial $trial/$(settings.num_instances) · $(variant.name) ══")
    initial_state1, initial_state2 = instance.states
    println("  P1 initial state [px, py, vx, vy]: $(round.(initial_state1; digits=3))")
    println("  P2 initial state [px, py, vx, vy]: $(round.(initial_state2; digits=3))")
    flush(stdout)
    (; θ, initial_guess) = instance_inputs(instance, flatten_parameters, dynamics, settings)
    context = trial_context(trial, variant.name)
    reduced = solve_scholtes_reduced(reduced_kkt, problem, θ, initial_guess, settings; context)
    println(
        reduced_verdict(reduced, settings),
        reduced.accepted ? " Running its $num_checks complete checks." : " Skipping its $num_checks complete checks.",
    )
    flush(stdout)
    (;
        trial,
        variant=variant.name,
        θ,
        initial=copy(initial_guess),
        strategies=player_strategies(reduced.output.z, settings),
        reduced,
        checks=Any[],
    )
end

"Label naming a trial in the solver tables."
trial_context(trial, variant_name) = "trial $trial, $variant_name"

"""
One complete check of a trial: start the complete system from the accepted reduced solution, solve
it in `mode`, print the verdict and return the check. A rejected reduced solve skips the check.
"""
function check_complete_kkt(record, systems, formulation, mode, settings)
    (; θ, reduced) = record
    reduced.accepted || return (;
        formulation=formulation.name,
        mode=mode.name,
        category=:reduced_failed,
        reason="Best reduced stage did not pass acceptance",
    )
    start = complete_initialization(formulation, reduced.output, θ, settings)
    complete = try
        solve_scholtes_complete(
            formulation,
            mode,
            systems.problem,
            θ,
            reduced.output.z,
            start,
            settings;
            context=trial_context(record.trial, record.variant),
        )
    catch err
        validation = merge(
            reduced.validation,
            (; accepted=false, relaxed_rms=NaN, unrelaxed_rms=NaN, max_abs_residual=NaN),
        )
        (;
            accepted=false,
            attempts=Any[],
            validation,
            physical=copy(reduced.output.z),
            solution=start,
            status=:exception,
            reason=:exception,
            error=sprint(showerror, err, catch_backtrace()),
            iterations=0,
            seconds=0.0,
            compile_seconds=0.0,
        )
    end
    metrics = trajectory_metrics(reduced.output.z, complete.physical, settings)
    # :numerical_failure        complete solve not accepted (no conclusion either way)
    # :fixed_primal_compatible  fixed-primal solve accepted
    # :small_adjustment         :warmstart accepted; trajectory moved within small_rms/small_max
    # :substantial_adjustment   :warmstart accepted; trajectory moved more
    # (:reduced_failed, above, marks checks skipped because the reduced solve failed.)
    category =
        !complete.accepted ? :numerical_failure :
        mode.name === :fixed_primal ? :fixed_primal_compatible :
        metrics.small ? :small_adjustment : :substantial_adjustment
    println(complete_verdict(formulation, mode, complete, metrics, category, settings))
    (;
        formulation=formulation.name,
        mode=mode.name,
        result=complete,
        metrics,
        category,
    )
end

"""
Run the experiment described in the module docstring: sample the instances, then for each objective
variant build its systems, warm up, and solve every trial's reduced system and complete checks.
Writes results.jls (after every check) and report.pdf to a new directory under `output_dir`, and
returns the run's directory, settings, instances, results and metadata.
"""
function run(; overrides=(;))
    settings = validated_settings(overrides)
    directory = run_directory(settings.output_dir)
    instances, sampling = sample_instances(Xoshiro(settings.rng_seed), settings)
    results, metadata = Any[], Any[]
    save() = save_results(directory, settings, instances, sampling, metadata, results)
    # Save before checking the sample count, so the rejection records survive a sampling failure.
    save()
    length(instances) == settings.num_instances || error(
        "Sampling exhausted $(settings.max_sampling_attempts) attempts; rejection records saved in $directory",
    )
    println(
        "Diagnostic run: $(settings.num_instances) instances, planning horizon = $(settings.planning_horizon), ",
        "$((settings.planning_horizon-1)*settings.time_step) seconds. Output: $directory",
    )
    variants = filter(v -> v.name in settings.objective_variants, objective_variants(settings))
    modes = filter(m -> m.name in settings.comparison_modes, [(; name=:fixed_primal), (; name=:warmstart)])
    builders = [
        (; name=:aggregate, build=ReducedGOOP.generate_scholtes_complete_kkt_system),
        (; name=:exact, build=ReducedGOOP.generated_exact_complete_kkt_system),
    ]
    builders = filter(b -> b.name in settings.complete_formulations, builders)
    num_checks = length(builders) * length(modes)
    total, done = length(variants) * length(instances) * num_checks, 0
    dynamics = intersection_dynamics(settings)
    println(tolerance_line(settings))
    for variant in variants
        systems = build_systems(variant, builders, dynamics, settings)
        push!(metadata, systems.metadata)
        jit_warmup(systems, modes, first(instances), dynamics, settings)
        # Warm-up results are discarded; every measured trial rebuilds its starts.
        for (trial, instance) in enumerate(instances)
            record = solve_reduced_kkt(systems, trial, instance, dynamics, num_checks, settings)
            # `record.checks` is filled in place below. The record already in `results` sees each
            # push, so every save stores the progress so far.
            push!(results, record)
            save()
            # formulations: :aggregate, :exact; modes: :fixed_primal, :warmstart
            for formulation in systems.formulations, mode in modes
                push!(record.checks, check_complete_kkt(record, systems, formulation, mode, settings))
                done += 1
                # One bar line per complete check that ran; skipped checks advance it once, below.
                record.reduced.accepted && print_progress(done, total)
                save()
            end
            record.reduced.accepted || print_progress(done, total)
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
            base .+ scale .* (2rand(rng, STATE_DIMENSION) .- 1) for
            (base, scale) in zip(base_initial_states, perturbation_scales)
        ]
        # P1 drives along x and P2 along y, so for player i, coordinate i is along-lane
        # (bound map_end) and coordinate 3 - i is cross-lane (bound lane_width).
        lane_margins = [
            minimum((
                settings.map_end-abs(states[i][i]),
                settings.lane_width-abs(states[i][3-i]),
            )) for i in 1:NUM_PLAYERS
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

"""
A variant's system sizes, printed after the systems are built and on the report's summary page:
each KKT system's variables and equations, then the size its solver works on (the reduced
system's Scholtes form; the complete system's fixed-primal unknowns).
"""
function size_lines(metadata)
    (; reduced) = metadata
    lines = [
        @sprintf(
            "%-18s %5d variables · %d equations + %d complementarity pairs",
            "reduced",
            reduced.variables,
            reduced.equations,
            reduced.complementarity,
        ),
        @sprintf("%-18s (solved in Scholtes form: %d equations × %d unknowns)", "", reduced.rows, reduced.columns),
    ]
    for complete in metadata.complete
        (; rows, columns, full_columns) = complete.dimensions
        push!(lines, @sprintf("%-18s %5d variables · %d equations", "complete :$(complete.name)", full_columns, rows))
        push!(lines, @sprintf("%-18s (fixed-primal: %d equations × %d unknowns)", "", rows, columns))
    end
    lines
end

"The acceptance tolerances, printed before the first solve and on the report's summary page."
tolerance_line(settings) = "Tolerances: reduced ‖K₀‖/√m ≤ $(compact(settings.tol)), complete ‖K₀‖/√m ≤ $(compact(settings.complete_tol))"

"Progress bar on its own line: other output interleaves with it, so it is not redrawn in place."
function print_progress(done, total; width=24)
    filled = floor(Int, width * done / total)
    println("[", repeat("=", filled), repeat(" ", width - filled), "] $done/$total checks")
    flush(stdout)
end

"One-line outcome of a reduced solve, printed under its stage table."
function reduced_verdict(reduced, settings)
    k₀, tol = compact(reduced.validation.unrelaxed_rms), compact(settings.tol)
    stage = "ρ = $(compact(reduced.stages[reduced.best_stage].ρ)), stage $(reduced.best_stage) of $(length(reduced.stages))"
    reduced.accepted ? "→ reduced accepted at $stage: ‖K₀‖/√m = $k₀ ≤ $tol." :
    "→ reduced REJECTED: best ‖K₀‖/√m = $k₀ ($stage) > $tol."
end

"One-line outcome of a complete check, printed under its stage tables."
function complete_verdict(formulation, mode, result, metrics, category, settings)
    name = ":$(formulation.name), :$(mode.name) mode"
    haskey(result, :error) && return "→ $name FAILED with an exception: $(first(split(result.error, '\n')))"
    final = result.attempts[result.best_attempt]
    tries = length(result.attempts) == 1 ? "1 try" : "$(length(result.attempts)) tries"
    stage = "ρ = $(compact(final.stages[final.best_stage].ρ))"
    k₀, tol = compact(result.validation.unrelaxed_rms), compact(settings.complete_tol)
    result.accepted ||
        return "→ $name REJECTED after $tries: best ‖K₀‖/√m = $k₀ (try $(result.best_attempt), $stage) > $tol"
    moved =
        mode.name === :fixed_primal ? "trajectory held fixed" :
        "trajectory moved RMS $(compact(metrics.rms)) ($(replace(string(category), '_' => ' ')))"
    "→ $name accepted after $tries at $stage: ‖K₀‖/√m = $k₀ ≤ $tol; $moved"
end

"Header above each solve's ρ-stage table; `label` names the solve and the η₀ this try uses."
function print_kkt_header(label; io=stdout)
    println(io, "\n[solver exit] $label")
    println(io, "Residuals are scaled, ‖·‖/√m; goal errors and minimum separation in m.")
    @printf(
        io,
        "%-9s %-7s %6s %8s %12s %12s %8s %8s %8s  %s\n",
        "ρ",
        "status",
        "iters",
        "time_s",
        "‖R‖/√m",
        "‖K₀‖/√m",
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
    metrics = Intersection.intersection_result_metrics(
        player_strategies(physical, settings);
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

"""
Scholtes options for one ρ stage of any solve, reduced or complete; only ρ, the acceptance
tolerance `tol` (an RMS) and the initial η differ between solves. Same options as
Intersection.demo: of the fixed constants, only proj_eps, eta_max and eta_schedule differ from the
ScholtesOptions defaults; the rest come from settings.
"""
function scholtes_options(settings; ρ, tol, eta, residual_scale)
    ReducedGOOP.ScholtesOptions(;
        rho_schedule=[Float64(ρ)],
        max_inner=settings.max_inner_iters,
        tol=tol * residual_scale,
        # A stage's inner loop stops at ‖R‖/√m < min(ρ, inner_ρ_tol): coarse stages are solved
        # more tightly than their ρ.
        tol_inner=min(ρ, settings.inner_ρ_tol) * residual_scale,
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
end

"""
Recorder for one ρ stage's solver trace: `trace` appends (; raw = ‖R‖, unrelaxed = ‖K₀‖, eta) for
each iterate to `history`, and the caller appends the solver's final iterate the same way.
"""
function history_recorder(residuals, ρ)
    history = NamedTuple{(:raw, :unrelaxed, :eta),NTuple{3,Float64}}[]
    trace =
        nt -> push!(history, (; raw=norm(nt.r), unrelaxed=ReducedGOOP._norm_K₀(residuals, nt.r, nt.w, ρ), eta=nt.eta))
    (; history, trace)
end

"""
One ρ stage's record: the solver's outcome, its timing (`seconds` excludes measured compilation),
its trace, and the iterate it started from and ended at.
"""
function stage_record(; ρ, parameter, timed, history, initial, final)
    output = timed.value
    (;
        ρ,
        parameter,
        status=output.status,
        reason=output.termination_reason,
        iterations=output.iters,
        seconds=max(0.0, timed.time-timed.compile_time-timed.recompile_time),
        wall_seconds=timed.time,
        compile_seconds=timed.compile_time+timed.recompile_time,
        history,
        initial,
        final,
    )
end

"""
Acceptance and diagnostics of a kept iterate, whose trajectory is `z`. It is accepted iff its scaled
residual `unrelaxed_rms` is at most `tol`; every other field is a saved diagnostic. `r` is the
relaxed residual vector at the stage's ρ.
"""
function validate_solution(problem, θ, z; r, relaxed_rms, unrelaxed_rms, sign_violation, tol)
    feasibility = ReducedGOOP.is_feasible(problem, z, θ) # only `worst` is used
    x = BlockArray(z, problem.primal_dims)
    θ_blocks = BlockArray(θ, problem.parameter_dims)
    equality_max = maximum(abs, reduce(vcat, (f(x, θ_blocks) for f in problem.equality_constraints)))
    (;
        relaxed_rms,
        unrelaxed_rms,
        max_abs_residual=maximum(abs, r),
        sign_violation,
        primal=(; equality_max, inequality_violation=max(0.0, -feasibility.worst)),
        accepted=unrelaxed_rms <= tol,
    )
end

function solve_scholtes_reduced(reduced_kkt, problem, θ, initial_guess, settings; context, io=stdout)
    # R(w; ρ) = [F_nc(y); a(y) - s; s ⊙ b(y) + u - ρ] has m = n_nc + 2 n_comp rows
    # (src/scholtes_residual.jl). Settings hold RMS tolerances, so they are scaled by √m.
    m = reduced_kkt.n_nc + 2reduced_kkt.n_comp
    residual_scale = sqrt(m)
    # λ (lam_*) and ψ (psi_*) are unsigned, so they carry over to the next ρ stage.
    # scholtes_warm_start re-centres γ/φ, s and u for the new ρ, and each stage restarts η at
    # eta_0: the same warm-ρ chain as Intersection.demo.
    eq_cols = findall(
        v -> startswith(string(v), "lam_") || startswith(string(v), "psi_"),
        reduced_kkt.vars,
    )
    residuals = ReducedGOOP.ScholtesResiduals(reduced_kkt)
    stage_initial_guess = initial_guess
    previous_w = nothing
    stage_preference_slacks = nothing
    stages = Any[]
    best = nothing
    print_kkt_header("$context | reduced KKT | η₀ = $(compact(settings.eta_0))"; io)
    for ρ in settings.rho_schedule
        eq = isnothing(previous_w) ? zeros(length(eq_cols)) : previous_w[eq_cols]
        w₀ = ReducedGOOP.scholtes_warm_start(
            reduced_kkt,
            θ,
            stage_initial_guess,
            ρ;
            eq_cols,
            eq,
            preference_slacks=stage_preference_slacks,
        )
        options = scholtes_options(settings; ρ, tol=settings.tol, eta=settings.eta_0, residual_scale)
        (; history, trace) = history_recorder(residuals, ρ)
        timed = @timed ReducedGOOP.solve(
            ReducedGOOP.Scholtes(),
            reduced_kkt,
            θ;
            z₀=stage_initial_guess,
            w₀,
            options,
            trace,
        )
        output = timed.value
        push!(history, (; raw=output.relaxed_residual, unrelaxed=output.residual, eta=output.eta))
        push!(stages, stage_record(; ρ, parameter=:rho_reduced, timed, history, initial=copy(w₀), final=copy(output.w)))
        print_kkt_stage(;
            ρ,
            status=output.status,
            iterations=output.iters,
            seconds=last(stages).seconds,
            relaxed_rms=output.relaxed_residual/residual_scale,
            unrelaxed_rms=output.residual/residual_scale,
            physical=output.z,
            settings,
            stop=Intersection.solver_stop_reason(output, options; scale=residual_scale),
            io,
        )
        stage_initial_guess = output.z
        previous_w = output.w
        stage_preference_slacks = output.preference_slacks
        # Track the stage with the lowest ‖K₀‖ (a non-finite residual counts as Inf).
        k₀ = isfinite(output.residual) ? output.residual : Inf
        if isnothing(best) || k₀ < best.k₀
            best = (; output, ρ, k₀, index=length(stages))
        end
    end
    # Judge and keep the best stage, not the last: each stage restarts γ/φ, s and u, so a later
    # stage can end worse than an earlier, converged one.
    (; output, ρ) = best
    r = zeros(m)
    b = zeros(reduced_kkt.n_comp)
    ReducedGOOP._resid!(residuals, r, b, output.w, ρ)
    validation = validate_solution(
        problem,
        θ,
        output.z;
        r,
        relaxed_rms=norm(r)/residual_scale,
        unrelaxed_rms=ReducedGOOP._norm_K₀(residuals, r, output.w, ρ)/residual_scale,
        sign_violation=max(0.0, -output.sign_min),
        tol=settings.tol,
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
    # The unrelaxed residual that _norm_K₀ evaluates for :aggregate (exact ignores it).
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
ORIGINAL equalities (λ) and inequalities (γ). The reduced system names these by player, level
and index (preference_slack_p1_k2_3, lam_p1_K3, gam_p1_K3; src/scholtes_kkt.jl), so each
complete variable's counterpart is looked up by that name. Outer-level multipliers belong to
different rows in the two systems (ψ/φ in reduced, inherited lower-level KKT rows in complete), so
they start at 0 or positive_seed (see complete_initialization).
"""
function variable_mapping(reduced, complete)
    reduced_names = Dict(string(v) => j for (j, v) in enumerate(reduced.vars))
    complete_names = Dict(string(v) => j for (j, v) in enumerate(complete.vars))
    source, destination, labels = Int[], Int[], String[]
    function add_pair!(reduced_name, complete_variable, label)
        push!(source, reduced_names[reduced_name])
        push!(destination, complete_names[string(complete_variable)])
        push!(labels, label)
    end
    for (i, levels) in enumerate(complete.levels), (k, level) in enumerate(levels)
        for (j, v) in enumerate(level.preference_slack)
            add_pair!("preference_slack_p$(i)_k$(k)_$j", v, "player $i level $k preference slack $j")
        end
        k == length(levels) || continue
        for (j, v) in enumerate(level.lambda)
            add_pair!("lam_p$(i)_K$j", v, "player $i innermost original equality multiplier $j")
        end
        for (j, v) in enumerate(level.gamma)
            add_pair!("gam_p$(i)_K$j", v, "player $i innermost original inequality multiplier $j")
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
    (; complete_kkt) = formulation
    y = zeros(complete_kkt.n)
    y[complete_kkt.primal_dims] .= output.z
    y[complete_kkt.nonnegative_dims] .= settings.positive_seed
    y[formulation.mapping.destination] .= output.w[formulation.mapping.source]
    y[complete_kkt.nonnegative_dims] .= max.(y[complete_kkt.nonnegative_dims], settings.positive_floor)
    ρ = formulation.name === :exact ? 0.0 : first(settings.complete_rho_schedule)
    y[complete_kkt.interior_point_slack_dims] .=
        max.(formulation.slack_targets(y, θ, ρ), settings.positive_floor)
    y
end

function solve_scholtes_complete(
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
    (; complete_kkt) = formulation
    # exact is unrelaxed: one stage at ρ = 0. aggregate runs down complete_rho_schedule (ρ plays
    # the legacy ε). Each retry restarts from `initial` with the next η.
    schedule = formulation.name === :exact ? [0.0] : settings.complete_rho_schedule
    # :warmstart solves the whole complete system. :fixed_primal solves the fixed-primal system,
    # whose unknowns are the free columns and whose parameters are [θ; x], so x cannot move.
    # Note: until 2026-10-22 :fixed_primal used NonlinearSolve's BoundedTrustRegion. On these
    # rank-deficient systems it stopped far from a stationary point (projected gradient ≫ 0) after
    # 1000+ steps, while Scholtes on the fixed-primal system met the tolerance in under 100.
    fixed_mode = mode.name === :fixed_primal
    system = fixed_mode ? formulation.fixed.system : complete_kkt
    parameters = fixed_mode ? [θ; physical] : θ
    unknowns = fixed_mode ? formulation.fixed.free : Colon()
    residuals = ReducedGOOP.ScholtesResiduals(system)
    residual_scale = sqrt(complete_kkt.n_nc)
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
            # The 1e-12 floor is defensive: exact ignores ρ (its residual is evaluated at ρ = 0)
            # and complete_rho_schedule stays above it, so it changes nothing today; for exact
            # it only keeps ρ and tol_inner positive.
            options = scholtes_options(settings; ρ=max(ρ, 1e-12), tol=settings.complete_tol, eta, residual_scale)
            (; history, trace) = history_recorder(residuals, ρ)
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
            push!(history, (; raw=output.relaxed_residual, unrelaxed=output.residual, eta=output.eta))
            parameter = formulation.name === :exact ? :unrelaxed : :epsilon_complete
            push!(stages, stage_record(; ρ, parameter, timed, history, initial=stage_initial, final=copy(y)))
            # Residuals of the full complete iterate, for the table and the best-stage choice.
            r, r0 = zeros(complete_kkt.n_nc), zeros(complete_kkt.n_nc)
            complete_kkt.F_nc!(r, y, θ, ρ)
            complete_kkt.F_nc!(r0, y, θ, 0.0)
            print_kkt_stage(;
                ρ,
                status=output.status,
                iterations=output.iters,
                seconds=last(stages).seconds,
                relaxed_rms=norm(r)/residual_scale,
                unrelaxed_rms=norm(r0)/residual_scale,
                physical=y[complete_kkt.primal_dims],
                settings,
                stop=Intersection.solver_stop_reason(output, options; scale=residual_scale),
                io,
            )
            # Track the stage with the lowest ‖K₀‖ (a non-finite residual counts as Inf).
            k₀ = isfinite(norm(r0)) ? norm(r0) : Inf
            if isnothing(best) || k₀ < best.k₀
                best = (; y=copy(y), ρ, k₀, index=length(stages))
            end
        end
        # Judge and keep the best stage, not the last, as in solve_scholtes_reduced. The full residual
        # retains stationarity, equality, complementarity, and slack equations.
        y = best.y
        r, r0 = zeros(complete_kkt.n_nc), zeros(complete_kkt.n_nc)
        complete_kkt.F_nc!(r, y, θ, best.ρ)
        complete_kkt.F_nc!(r0, y, θ, 0.0)
        validation = validate_solution(
            problem,
            θ,
            y[complete_kkt.primal_dims];
            r,
            relaxed_rms=norm(r)/residual_scale,
            unrelaxed_rms=norm(r0)/residual_scale,
            sign_violation=max(0.0, -minimum(y[complete_kkt.nonnegative_dims]; init=Inf)),
            tol=settings.complete_tol,
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
    # end worse than an earlier try. An accepted try is the last and has the lowest ‖K₀‖.
    best_attempt = argmin([isfinite(a.validation.unrelaxed_rms) ? a.validation.unrelaxed_rms : Inf for a in attempts])
    final = attempts[best_attempt]
    mode.name === :fixed_primal && @assert final.solution[complete_kkt.primal_dims] == physical
    (;
        accepted=final.accepted,
        attempts,
        best_attempt,
        validation=final.validation,
        physical=final.solution[complete_kkt.primal_dims],
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
    # One column per player and knot: [px, py, vx, vy, ax, ay].
    knot_size = STATE_DIMENSION + CONTROL_DIMENSION
    reduced_columns, complete_columns = reshape(reduced, knot_size, :), reshape(complete, knot_size, :)
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
            max_abs=maximum(abs, δ),
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
    rms, max_abs = norm(δ)/sqrt(length(δ)), maximum(abs, δ)
    (;
        rms,
        max_abs,
        relative=norm(δ)/max(norm(reference), 1e-12sqrt(length(δ))),
        groups=errors,
        small=all(
            norm(g.scaled)/sqrt(length(g.scaled))<=settings.small_rms for g in errors
        ) && max_abs<=settings.small_max,
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

"""
Pass counts, one line per objective, complete formulation and mode, then a key: printed at the end
of a run and on the report's summary page.
"""
function pass_count_lines(results, settings)
    lines = [
        @sprintf(
            "%-10s %-11s %-19s %8s %21s %14s",
            "objective",
            "complete",
            "mode",
            "reduced",
            "complete accepted",
            "overall pairs",
        ),
    ]
    for variant_name in settings.objective_variants,
        formulation_name in settings.complete_formulations,
        mode_name in settings.comparison_modes

        rows = filter(r -> r.variant === variant_name, results)
        num_reduced_accepted = count(r -> r.reduced.accepted, rows)
        passed_check(c) =
            c.formulation === formulation_name && c.mode === mode_name && haskey(c, :result) && c.result.accepted
        passed = count(r -> any(passed_check, r.checks), rows)
        n = length(rows)
        push!(
            lines,
            @sprintf(
                "%-10s %-11s %-19s %8s %21s %14s",
                variant_name,
                ":$formulation_name",
                ":$mode_name mode",
                "$num_reduced_accepted/$n",
                "$passed/$num_reduced_accepted",
                "$passed/$n",
            ),
        )
    end
    push!(
        lines,
        "",
        "reduced: accepted / trials; complete accepted: accepted checks / accepted reduced solves;",
        "overall pairs: accepted checks / trials. A failed check does not prove incompatibility.",
    )
    lines
end

"Final console summary: the pass counts, as on the report's summary page."
function print_final_summary(results, settings; io=stdout)
    println(io, "\n[final summary] Pass counts")
    foreach(line -> println(io, line), pass_count_lines(results, settings))
    flush(io)
    nothing
end

# ── Plots and report ───────────────────────────────────────────────────────────

const MONO_FONT = "DejaVu Sans Mono" # ships with Makie, so report.pdf renders the same everywhere

"""
Plot the geometric mean of `histories` at each trace index (line) with ± one standard
deviation of the log10 values (band, only where at least two histories contribute). With a
single history it draws that history as is, with no band.
"""
function plot_history!(axis, histories, label, color)
    isempty(histories) && return
    average, deviation, counts = Float64[], Float64[], Int[]
    for j in 1:maximum(length, histories)
        values = [
            log10(max(h[j], 1e-16)) for
            h in histories if length(h)>=j && isfinite(h[j]) && h[j]>=0
        ]
        # No finite value at this index: leave a gap rather than stop, so x stays the sample index
        # (the trial pages' ρ stage markers count samples).
        if isempty(values)
            push!(average, NaN)
            push!(deviation, NaN)
            push!(counts, 0)
            continue
        end
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

"Line colour of each complete check in the residual plots and histograms; the reduced solve is dark orange."
const CHECK_COLORS = Dict(
    (:exact, :fixed_primal) => :navy,
    (:exact, :warmstart) => :seagreen,
    (:aggregate, :fixed_primal) => :purple,
    (:aggregate, :warmstart) => :goldenrod,
)

"A solve's `field` trace (:unrelaxed for K₀, :raw for R) over all its stages, as RMS over `equations` rows."
function stage_history(stages, field, equations)
    reduce(vcat, ([getproperty(h, field)/sqrt(equations) for h in s.history] for s in stages))
end

"""
Draw `field` (:unrelaxed for K₀, :raw for R) for the reduced solves of `rows` and for each complete
check's kept try, with both tolerances. One row gives that trial's own curves; several give their
geometric mean ± band (plot_history!), and `counted` adds the number of curves to each label.
"""
function plot_residuals!(axis, rows, metadata, field, settings; counted=false)
    label(name, n) = counted ? "$name (n=$n)" : name
    reduced = [stage_history(r.reduced.stages, field, metadata.reduced.rows) for r in rows]
    plot_history!(axis, reduced, label("Reduced", length(reduced)), :darkorange)
    for formulation_name in settings.complete_formulations, mode_name in settings.comparison_modes
        equations = only(filter(c -> c.name === formulation_name, metadata.complete)).dimensions.rows
        # Skipped checks have no result, and a check that threw has no tries.
        kept = [
            c.result for r in rows for c in r.checks if
            c.formulation === formulation_name && c.mode === mode_name && haskey(c, :result) &&
                !isempty(c.result.attempts)
        ]
        histories = [stage_history(k.attempts[k.best_attempt].stages, field, equations) for k in kept]
        plot_history!(
            axis,
            histories,
            label(":$formulation_name, :$mode_name", length(histories)),
            CHECK_COLORS[(formulation_name, mode_name)],
        )
    end
    CM.hlines!(
        axis,
        [settings.tol];
        color=:darkorange,
        linestyle=:dot,
        label="Reduced tolerance, ‖K₀‖/√m = $(compact(settings.tol))",
    )
    # Black: every complete check shares this tolerance, so it takes none of their colours.
    CM.hlines!(
        axis,
        [settings.complete_tol];
        color=:black,
        linestyle=:dot,
        label="Complete tolerance, ‖K₀‖/√m = $(compact(settings.complete_tol))",
    )
end

"Legend of `axis`'s labelled plots at `position`, a cell under the plot, so it covers no part of it."
function legend_under!(position, axis; nbanks=1)
    CM.Legend(position, axis; orientation=:horizontal, nbanks, framevisible=false, labelsize=13, tellwidth=false)
end

"""
Aggregate-page histogram of one comparison mode over the accepted checks, one colour per complete
formulation: the fixed-primal scaled residual, or how far :warmstart moved the trajectory. Its
legend goes at `legend_position`, under the plot.
"""
function plot_acceptance_histogram!(position, legend_position, rows, settings, mode_name)
    fixed = mode_name === :fixed_primal
    # At most 4 x ticks, labelled like the console (9.4e-4): the default labels (9.4×10⁻⁴, ~9 ticks)
    # overlap whenever the values span less than a decade.
    axis = CM.Axis(
        position;
        xticks=CM.WilkinsonTicks(4),
        xtickformat=values -> compact.(values),
        xlabel=fixed ? "‖K₀‖/√m (best stage)" : "Scaled trajectory RMS",
        ylabel="Accepted pairs",
        title=fixed ? "Fixed-primal recovery" : "Trajectory adjustment (:warmstart)",
    )
    num_reduced_accepted = count(r -> r.reduced.accepted, rows)
    plotted = false
    for formulation_name in settings.complete_formulations
        accepted = [
            c for r in rows for c in r.checks if
            c.formulation === formulation_name && c.mode === mode_name && haskey(c, :result) && c.result.accepted
        ]
        isempty(accepted) && continue
        values = [fixed ? c.result.validation.unrelaxed_rms : c.metrics.rms for c in accepted]
        CM.hist!(
            axis,
            values;
            bins=max(1, min(10, length(values))),
            color=(CHECK_COLORS[(formulation_name, mode_name)], 0.6),
            label=":$formulation_name, $(length(values))/$num_reduced_accepted accepted",
        )
        plotted = true
    end
    if plotted
        legend_under!(legend_position, axis)
    else
        CM.text!(axis, 0.5, 0.5; text="No accepted pairs", space=:relative, align=(:center, :center))
    end
end

"""
Report page for one trial: initial states and verdicts, this trial's residual histories, the
reduced and :warmstart complete trajectories next to their vehicle separation, then velocity and
acceleration profiles overlaid (solid: reduced; dashed: complete). A :fixed_primal check keeps the
reduced trajectory, so it gets a verdict line and residual curves but no trajectory of its own.
"""
function trial_page(data, row)
    (; settings) = data
    metadata = only(filter(m -> m.variant === row.variant, data.metadata))
    figure = Intersection.serif_figure(size=(1600, 1600), fontsize=18, figure_padding=30)
    CM.Label(figure[1, 1], "Trial $(row.trial) | $(row.variant)"; fontsize=26)
    s1, s2 = data.instances[row.trial].states
    # The one wording, used once per check, for a check that never ran.
    skipped = "skipped (reduced solve rejected)"
    verdicts = [
        "P1 initial state [px, py, vx, vy]: $(round.(s1; digits=3))",
        "P2 initial state [px, py, vx, vy]: $(round.(s2; digits=3))",
        reduced_verdict(row.reduced, settings),
    ]
    for check in row.checks
        formulation, mode = (; name=check.formulation), (; name=check.mode)
        push!(
            verdicts,
            haskey(check, :result) ?
            complete_verdict(formulation, mode, check.result, check.metrics, check.category, settings) :
            "→ :$(check.formulation), :$(check.mode) mode $skipped",
        )
    end
    CM.Label(figure[2, 1], join(verdicts, "\n"); font=MONO_FONT, fontsize=14, halign=:left, justification=:left)

    residuals = CM.GridLayout(figure[3, 1])
    for (column, field, title, ylabel) in (
        (1, :unrelaxed, "True KKT residual", "‖K₀‖/√m"),
        (2, :raw, "Relaxed residual", "‖R‖/√m"),
    )
        axis = CM.Axis(
            residuals[1, column];
            xlabel="Trace sample (stages concatenated)",
            ylabel,
            yscale=log10,
            title,
        )
        plot_residuals!(axis, [row], metadata, field, settings)
        # Both plots have the same entries, so one legend under the pair serves both.
        column == 1 && legend_under!(residuals[2, 1:2], axis)
        field === :raw || continue
        # Relaxed residual only: mark where each reduced ρ stage starts, and its ρ. R jumps up there
        # because each stage re-initializes γ/φ, s and u. The x limits are frozen so the labels can
        # sit at a fixed height, in relative coordinates, on the log-scaled y axis.
        starts = cumsum([1; [length(s.history) for s in row.reduced.stages[1:(end-1)]]]) .- 0.5
        CM.reset_limits!(axis)
        (; origin, widths) = axis.finallimits[]
        CM.xlims!(axis, origin[1], origin[1]+widths[1])
        CM.vlines!(axis, starts; color=(:darkorange, 0.5), linestyle=:dash, linewidth=1)
        CM.text!(
            axis,
            [CM.Point2f((x-origin[1])/widths[1], 0.03) for x in starts];
            text=["ρ = $(compact(s.ρ))" for s in row.reduced.stages],
            space=:relative,
            rotation=π/2,
            align=(:left, :top),
            fontsize=13,
            color=:chocolate,
        )
    end

    # Trajectories: the reduced solution, then each formulation's :warmstart result. Subtitles give
    # the scaled residual K₀/sqrt(m) of the plotted iterate: the reduced solve's best stage, and the
    # best stage of the complete check's best try. A check that threw keeps the reduced trajectory
    # and has no residual.
    subtitle(k₀, ρ) = "Scaled residual = $(compact(k₀)) at ρ = $(compact(ρ))"
    # A skipped check says so once, in its status, and has no subtitle. Its reason goes on its own
    # line, so the title stays as narrow as its map.
    check_status(check) = haskey(check, :result) ? string(check.result.status) : replace(skipped, " (" => "\n(")
    function check_subtitle(check)
        haskey(check, :result) || return ""
        isempty(check.result.attempts) && return "no residual: the complete solve threw an exception"
        final = check.result.attempts[check.result.best_attempt]
        subtitle(final.validation.unrelaxed_rms, final.stages[final.best_stage].ρ)
    end
    # A :fixed_primal check keeps the reduced trajectory, so its name, status and residual go under
    # the reduced map.
    reduced_stage = row.reduced.stages[row.reduced.best_stage]
    reduced_lines = [subtitle(row.reduced.validation.unrelaxed_rms, reduced_stage.ρ)]
    for check in filter(c -> c.mode === :fixed_primal, row.checks)
        push!(reduced_lines, "Complete :$(check.formulation), :fixed_primal | $(check_status(check))")
        isempty(check_subtitle(check)) || push!(reduced_lines, check_subtitle(check))
    end
    trajectories = NamedTuple[]
    push!(
        trajectories,
        (;
            name="Reduced",
            physical=row.reduced.output.z,
            status=row.reduced.output.status,
            k₀=join(reduced_lines, "\n"),
            linestyle=:solid,
        ),
    )
    for check in filter(c -> c.mode === :warmstart, row.checks)
        complete = haskey(check, :result) ? check.result : nothing
        k₀ = check_subtitle(check)
        push!(
            trajectories,
            (;
                name="Complete :$(check.formulation), :warmstart",
                physical=isnothing(complete) ? nothing : complete.physical,
                status=check_status(check),
                k₀,
                linestyle=check.formulation === :exact ? :dash : :dashdot,
            ),
        )
    end

    maps = CM.GridLayout(figure[4, 1])
    separation = CM.Axis(
        maps[1, length(trajectories)+1];
        title="Vehicle separation",
        xlabel="time [s]",
        ylabel="distance [m]",
    )
    profiles = CM.GridLayout(figure[5, 1])
    panels = [
        CM.Axis(profiles[1, column]; title, xlabel="time [s]", ylabel) for
        (column, (title, ylabel)) in enumerate((
            ("Velocity vx", "vx [m/s]"),
            ("Velocity vy", "vy [m/s]"),
            ("Acceleration ax", "ax [m/s^2]"),
            ("Acceleration ay", "ay [m/s^2]"),
        ))
    ]
    for component in 1:2
        for player in 1:NUM_PLAYERS
            bounds = Intersection.player_speed_bounds(player; settings.min_speed, settings.max_speed)
            CM.hlines!(
                panels[component],
                [bounds.lower[component], bounds.upper[component]];
                color=Intersection.player_plot_color(player),
                linestyle=:dot,
            )
        end
        limits = filter(isfinite, [settings.control_bounds.lb[component], settings.control_bounds.ub[component]])
        isempty(limits) || CM.hlines!(panels[2+component], limits; color=:black, linestyle=:dot)
    end
    for (column, trajectory) in enumerate(trajectories)
        (; name, physical, status, k₀, linestyle) = trajectory
        # Only the solver status: Scholtes reports :solved exactly when K₀ meets the tolerance that
        # acceptance uses, so an accepted flag would repeat it (the verdict lines still state it).
        ax = CM.Axis(
            maps[1, column];
            title="$name | $status",
            subtitle=k₀,
            aspect=CM.DataAspect(),
            xlabel="x [m]",
            ylabel="y [m]",
            xgridvisible=false,
            ygridvisible=false,
        )
        Intersection.draw_intersection_map!(ax; map_end=settings.map_end, lane_width=settings.lane_width)
        CM.limits!(ax, -settings.map_end, settings.map_end, -settings.map_end, settings.map_end)
        # A skipped check has no trajectory; its title already says why.
        isnothing(physical) && continue
        strategies = player_strategies(physical, settings)
        Intersection.plot_vehicle_separation!(
            separation,
            strategies;
            timestep=settings.time_step,
            linestyle,
            label=name,
        )
        for player in 1:NUM_PLAYERS
            (; xs, us) = strategies[player]
            color = Intersection.player_plot_color(player)
            CM.scatterlines!(ax, [x[1] for x in xs], [x[2] for x in xs]; color, linewidth=3, markersize=10)
            CM.scatter!(ax, [xs[1][1]], [xs[1][2]]; color, marker=:circle, markersize=16)
            goal = (settings.goal_position1, settings.goal_position2)[player]
            CM.scatter!(ax, [goal[1]], [goal[2]]; color, marker=:star5, markersize=22)
            for component in 1:2
                CM.lines!(
                    panels[component],
                    (0:(length(xs)-1)) .* settings.time_step,
                    [x[component+2] for x in xs];
                    color,
                    linestyle,
                    linewidth=2,
                )
                CM.lines!(
                    panels[2+component],
                    (0:(length(us)-1)) .* settings.time_step,
                    [u[component] for u in us];
                    color,
                    linestyle,
                    linewidth=2,
                )
            end
        end
    end
    CM.hlines!(
        separation,
        [settings.collision_avoidance];
        color=:black,
        linestyle=:dot,
        label="Collision avoidance, $(settings.collision_avoidance) m",
    )
    CM.axislegend(separation; position=:ct, labelsize=13)
    CM.Legend(
        figure[6, 1],
        vcat(
            [CM.LineElement(color=Intersection.player_plot_color(player), linewidth=3) for player in 1:NUM_PLAYERS],
            [CM.LineElement(color=:gray30, linestyle=t.linestyle, linewidth=2) for t in trajectories],
        ),
        vcat("Player 1", "Player 2", [t.name for t in trajectories]);
        orientation=:horizontal,
        framevisible=false,
    )
    CM.Label(
        figure[7, 1],
        "Circle: start; star: goal. Dotted: speed bounds in each player's colour, control bounds in black.";
        fontsize=14,
    )
    # Fixed page, map (square, as tall as its row) and separation widths, so the maps and their
    # separation plot stay together however many trajectories a page has.
    CM.colsize!(figure.layout, 1, CM.Fixed(1540))
    foreach(column -> CM.colsize!(maps, column, CM.Fixed(380)), eachindex(trajectories))
    CM.colsize!(maps, length(trajectories)+1, CM.Fixed(480))
    CM.rowsize!(figure.layout, 3, 320)  # residual plots and their legend
    CM.rowsize!(figure.layout, 4, 380)
    CM.rowsize!(figure.layout, 5, 220)
    CM.rowgap!(figure.layout, 24)
    CM.resize_to_layout!(figure)
    figure
end

"""
Draw report.pdf from results.jls alone, without building or solving any KKT system: a summary
page, an aggregate page (all trials' residual histories and the acceptance histograms), then one
page per objective variant and trial.
"""
function plot_results(directory)
    data = deserialize(joinpath(directory, "results.jls"))
    pages = [summary_page(data, directory), aggregate_page(data)]
    for row in data.results
        println("Plotting $(row.variant) trial $(row.trial)...")
        flush(stdout)
        push!(pages, trial_page(data, row))
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

"Text page that opens report.pdf: tolerances, pass counts, system sizes, settings, method notes and failures."
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
    paragraph!(
        "Scaled residual: ‖K₀‖/√m, the norm of the unrelaxed KKT residual (ρ = 0) divided by √m, m = number of equations. The scaled relaxed residual ‖R‖/√m is the same at the stage's ρ.",
    )

    heading!("Pass counts")
    monospace!(join(pass_count_lines(results, settings), "\n"))

    heading!("System sizes")
    monospace!(join(["$(m.variant):\n  " * join(size_lines(m), "\n  ") for m in data.metadata], "\n"))

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
        "Every solve uses Scholtes; each ρ stage's inner loop stops once its scaled relaxed residual drops below min(ρ, $(settings.inner_ρ_tol)), at max_inner_iters, or on a line-search stall, unless the scaled residual reaches its tolerance first. :fixed_primal recovery solves the complete system with the physical trajectory as a parameter, so only the remaining columns move; nonnegative coordinates stay inside the sign box. Exact complete equations have no ε schedule; aggregate uses complete_rho_schedule as the legacy ε. All complete rows remain after fixing physical columns.",
    )
    paragraph!(
        "Residual curves show the scaled relaxed and unrelaxed residuals on logarithmic axes, concatenated across stages, against solver-trace samples (Scholtes iterates). Reduced curves include all sampled trials; complete curves include all attempted checks, using each check's kept (best) try. Histograms use accepted pairs only. Curves use available samples only: no padding after a run ends, and a gap where a sample is not finite. On the aggregate page, lines are geometric means and bands are mean +/- one sample standard deviation in log10 residual, drawn at indices with at least two contributors. Stage endpoints and raw histories are in results.jls. Each trial page shows that trial's own curves (the relaxed-residual plot marks where each reduced ρ stage starts), its reduced and :warmstart complete trajectories next to their vehicle separation, and their velocity and acceleration profiles overlaid (solid: reduced; dashed: complete), including failed returned iterates with explicit status labels.",
    )
    paragraph!(
        "A solve is accepted iff its scaled residual is at most $(settings.tol) (reduced) or $(settings.complete_tol) (complete); the solver's exit status is not required. A ρ chain runs all its stages and is judged at the stage with the lowest scaled residual, and a complete check keeps its best try (retries restart with the next η); that iterate is the one kept, compared and plotted. Fixed trajectories retain the reduced solution's finite physical residual; these rows are not removed. The full residual includes stationarity, equality, complementarity, and auxiliary-slack equations. The scaled relaxed residual, largest residual entry, multiplier signs and original physical feasibility are recorded as diagnostics only. Failures do not prove incompatibility.",
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
            "$(row.variant) trial $(row.trial): reduced rejected; status=$(reduced.output.status), reason=$(reduced.output.termination_reason), ‖R‖/√m=$(number(reduced.validation.relaxed_rms)), ‖K₀‖/√m=$(number(reduced.validation.unrelaxed_rms)), physical equality=$(number(reduced.validation.primal.equality_max)), inequality=$(number(reduced.validation.primal.inequality_violation))",
        )
        for check in row.checks
            haskey(check, :result) || continue
            complete = check.result
            name = "$(row.variant) trial $(row.trial), :$(check.formulation), :$(check.mode) mode"
            complete.accepted || push!(
                failures,
                "$name: $(complete.status), $(complete.reason); ‖R‖/√m=$(number(complete.validation.relaxed_rms)), ‖K₀‖/√m=$(number(complete.validation.unrelaxed_rms)), physical equality=$(number(complete.validation.primal.equality_max)), inequality=$(number(complete.validation.primal.inequality_violation)), attempts=$(length(complete.attempts))",
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

"""
Report page 2: for each objective variant, every trial's residual histories (geometric mean ±
band; reduced and each complete check) and the acceptance histograms of each comparison mode.
"""
function aggregate_page(data)
    (; settings, results) = data
    variant_names = settings.objective_variants
    figure = Intersection.serif_figure(size=(1600, 100+420length(variant_names)), fontsize=18, figure_padding=30)
    CM.Label(figure[1, 1], "All trials: residual histories and acceptance"; fontsize=26)
    for (index, variant_name) in enumerate(variant_names)
        rows = filter(r -> r.variant === variant_name, results)
        metadata = only(filter(m -> m.variant === variant_name, data.metadata))
        layout = CM.GridLayout(figure[index+1, 1])
        for (column, field, title, ylabel) in (
            (1, :unrelaxed, "True KKT residual", "‖K₀‖/√m"),
            (2, :raw, "Relaxed residual", "‖R‖/√m"),
        )
            axis = CM.Axis(
                layout[1, column];
                xlabel="Trace sample (stages concatenated)",
                ylabel,
                yscale=log10,
                title="$variant_name | $title",
            )
            plot_residuals!(axis, rows, metadata, field, settings; counted=true)
            # Both plots have the same entries, so one legend under the pair serves both.
            column == 1 && legend_under!(layout[2, 1:2], axis; nbanks=2)
        end
        for (column, mode_name) in enumerate(settings.comparison_modes)
            plot_acceptance_histogram!(layout[1, 2+column], layout[2, 2+column], rows, settings, mode_name)
        end
        CM.rowsize!(layout, 1, 320)
    end
    # Nothing else fixes the page width, so without this resize_to_layout! shrinks it to the minimum.
    CM.colsize!(figure.layout, 1, CM.Fixed(1540))
    CM.rowgap!(figure.layout, 30)
    CM.resize_to_layout!(figure)
    figure
end

end
