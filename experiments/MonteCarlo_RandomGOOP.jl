"""
Monte Carlo comparison of the reduced and complete KKT systems on random GOOPs, for the manuscript's §6:
Table 2 (system sizes and solve times), Table 3 (setup times), Fig 2 (Newton steps and solve time per ρ
stage) and Fig 4 (‖z_complete − z_reduced‖₂). All of them come from one run, so they describe the same
problems. Fig 3 (local convergence) comes from RandomGOOP_local_convergence.jl, on one instance per class of
this study.

For every objective class, number of preference levels K and sampled instance, solve from z = 0, by
Algorithm 5.1 (`solve(PDIP(), …)` with `linear_solver`, :klu by default) down
`rho_schedule`:
 1. the ρ-perturbed REDUCED KKT system (5.2), and
 2. for K in `complete_levels`, the ρ-perturbed COMPLETE KKT system (6.1), using nothing from the
    reduced solve, and report ‖z_complete − z_reduced‖₂.
A stage at ρ stops once ‖K_ρ‖₂ < ε_ρ = ρ, at a line-search failure (α < 1e-14) or after `max_inner_iters`
Newton steps. The whole solve stops, accepted, as soon as ‖K₀‖₂ < `epsilon_0`, checked before every Newton
step; a solve that never gets there is rejected, judged at its best stage.

Problem. N players with n primal variables each; z ∈ R^{Nn} aggregates all players' variables, and every
function below depends on all of z, so the players are coupled. Player i's levels, outermost first:
    J_1 = ‖z‖², or ½‖R_1 z + p_1‖² with quadratic_top_level = :rank2 (quadratic class only); for k = 2..K
      :quadratic     J_k = ½‖R_k z + p_k‖²  (= ½zᵀQ_k z + q_kᵀz + ½‖p_k‖², Q_k = R_kᵀR_k rank 2, q_k = R_kᵀp_k)
      :nonquadratic  J_k = exp(v_kᵀz − s_k) + exp(−(v_kᵀz − s_k)),  ‖v_k‖ = 1
    equalities     H z − b = 0                                  (m_E rows)
    inequalities   on D(z) = R_K z ∈ R² (:quadratic) or v_Kᵀz ∈ R (:nonquadratic), what the innermost
                   objective depends on (m_I = 2 rows). The quadratic class is a quadratic GOOP, with
                   convex quadratic objectives and linear constraints, so its inequalities are always
                   :linear; the nonquadratic class's are `nonquadratic_constraint_type`:
      :linear        W D − c ≥ 0
      :nonlinear     r_j² − ‖D − c_j‖² ≥ 0
Every instance draws new data for every player (sample_instances). The player's own columns of H and of
each level's R_k or v_k are orthonormal rows of a random orthogonal matrix (own_rows, `own_blocks`); the
other players' columns are Gaussian, scaled by κ = `coupling` (coupled_rows). The data enter through θ,
so each (class, K) system is built once. z = 0 is strictly feasible for every inequality by
construction (sample_inequality), and every solve, reduced or complete, starts there.

Terms
  perturbed  the ρ-perturbed complete system (6.1), the complete counterpart of the reduced (5.2): the
             exact complete system's equations with level-indexed slacks in the products and every
             product = ρ.
  R, K₀      the residual ‖·‖₂ at the stage's ρ, and at ρ = 0.

    julia --project=experiments -e 'include("experiments/MonteCarlo_RandomGOOP.jl");
        MonteCarlo_RandomGOOP.run(overrides = (; num_instances = 10, num_levels = 2:3))'
"""
module MonteCarlo_RandomGOOP

using BlockArrays, LinearAlgebra, Random, Serialization, Printf
import ReducedGOOP
include(joinpath(@__DIR__, "MonteCarlo_Intersection.jl"))
# MonteCarlo_Intersection's helpers (run directory, saving, solver traces, validation, number format) are
# reused unchanged.
const MCI = MonteCarlo_Intersection

# ── Problem definition ─────────────────────────────────────────────────────────

"""
The objective classes (module docstring). `levels(k)` names level k's objective, :norm, :quadratic or
:cosh, with level 1 the outermost; `constraint_type` is the form of the inequalities: always :linear
for the quadratic class (a quadratic GOOP has linear constraints), and `nonquadratic_constraint_type`
(:linear or :nonlinear) for the nonquadratic class.
"""
function objective_classes(settings)
    [
        (;
            name=:quadratic,
            levels=k -> k == 1 && settings.quadratic_top_level === :norm ? :norm : :quadratic,
            constraint_type=:linear,
        ),
        (;
            name=:nonquadratic,
            levels=k -> k == 1 ? :norm : :cosh,
            constraint_type=settings.nonquadratic_constraint_type,
        ),
    ]
end

"""
One level's objective from its data. :norm is ‖z‖². :quadratic is ½‖R z + p‖², the manuscript's
(4.11) ½zᵀQz + qᵀz with Q = RᵀR (PSD, rank 2) and q = Rᵀp ∈ col(Q), plus the constant ½‖p‖², which
changes no derivative. :cosh is exp(t) + exp(−t) with t = vᵀz − s.
"""
function level_objective(kind, data, z)
    kind === :norm && return sum(abs2, z)
    kind === :quadratic && return sum(abs2, data.R * z + data.p) / 2
    t = dot(data.v, z) - data.s
    exp(t) + exp(-t)
end

"""
D(z), what the innermost objective depends on: R_K z ∈ R² (:quadratic) or [v_Kᵀz] ∈ R¹ (:cosh).
J_k(z) = ½‖R_k z + p_k‖² = ½ zᵀQ_k z + q_kᵀz + ½‖p_k‖², with Q_k = R_kᵀR_k and q_k = R_kᵀp_k.
R_k has 2 rows (layout R = (2, N·n) in parameter_layout), so Q_k has rank 2. 
"""
innermost_image(kind, data, z) = kind === :quadratic ? data.R * z : [dot(data.v, z)]

"The innermost objective's unconstrained optimum of D: −p_K (:quadratic) or [s_K] (:cosh)."
innermost_target(kind, data) = kind === :quadratic ? -data.p : [data.s]

"Inequality values from a player's inequality data: :linear W D − c, :nonlinear r_j² − ‖D − c_j‖²."
function inequality_values(constraint_type, inequality, D)
    constraint_type === :linear && return inequality.W * D - inequality.c
    [inequality.r[j]^2 - sum(abs2, D - inequality.C[j, :]) for j in eachindex(inequality.r)]
end

"""
Shapes of one player's data, in θ order: each level's (R, p), (v, s) or nothing (:norm), then H and b,
then the inequality data, (W, c) for :linear or (C, r) for :nonlinear, where row j of C is c_j. Shape ()
is a scalar.
"""
function parameter_layout(class, num_levels, settings)
    (; num_players, primal_dimension, equality_dimension, inequality_dimension) = settings
    n = num_players * primal_dimension
    level_shapes = map(1:num_levels) do k
        kind = class.levels(k)
        kind === :norm ? (;) : kind === :quadratic ? (; R=(2, n), p=(2,)) : (; v=(n,), s=())
    end
    d = class.levels(num_levels) === :quadratic ? 2 : 1
    inequality_shapes =
        class.constraint_type === :linear ? (; W=(inequality_dimension, d), c=(inequality_dimension,)) :
        (; C=(inequality_dimension, d), r=(inequality_dimension,))
    (; levels=level_shapes, H=(equality_dimension, n), b=(equality_dimension,), inequality=inequality_shapes)
end

"Number of θ entries per player."
function layout_length(layout)
    groups = [layout.levels; (; layout.H, layout.b); layout.inequality]
    sum(prod(shape) for group in groups for shape in values(group); init=0)
end

"A player's data from its θ block, as (; levels, H, b, inequality); the inverse of flatten_parameters."
function unflatten_parameters(θ, layout)
    θ_iter = Iterators.Stateful(θ)
    read(shape) = shape == () ? first(θ_iter) : reshape(first(θ_iter, prod(shape)), shape)
    read_group(shapes) = map(read, shapes)
    levels = map(read_group, layout.levels)
    (; H, b) = read_group((; layout.H, layout.b))
    (; levels, H, b, inequality=read_group(layout.inequality))
end

"A player's θ block from its data (sample_instances); the inverse of unflatten_parameters."
function flatten_parameters(data, layout)
    θ = Float64[]
    append_group!(entries, shapes) = foreach(name -> append!(θ, entries[name]), keys(shapes))
    foreach(append_group!, data.levels, layout.levels)
    append_group!(data, (; layout.H, layout.b))
    append_group!(data.inequality, layout.inequality)
    θ
end

"""
The GOOP of `class` with `num_levels` levels per player (module docstring), and the layout of each
player's θ block. Each player's preferences run outermost → innermost: preferences[i][1] is J_1.
"""
function get_setup(class, num_levels, settings)
    (; num_players, primal_dimension) = settings
    layout = parameter_layout(class, num_levels, settings)
    primal_dims = fill(primal_dimension, num_players)
    parameter_dims = fill(layout_length(layout), num_players)
    dummy_primals = BlockArray(zeros(sum(primal_dims)), primal_dims)
    dummy_parameters = BlockArray(zeros(sum(parameter_dims)), parameter_dims)
    data(θ; player) = unflatten_parameters(θ[Block(player)], layout)

    preferences = [
        Function[(z, θ) -> level_objective(class.levels(k), data(θ; player).levels[k], z) for k in 1:num_levels]
        for player in 1:num_players
    ]
    equality_constraints = [
        function (z, θ)
            (; H, b) = data(θ; player)
            H * z - b
        end for player in 1:num_players
    ]
    inequality_constraints = [
        function (z, θ)
            (; levels, inequality) = data(θ; player)
            D = innermost_image(class.levels(num_levels), levels[num_levels], z)
            inequality_values(class.constraint_type, inequality, D)
        end for player in 1:num_players
    ]
    problem = ReducedGOOP.ParametricGOOP(
        dummy_primals,
        dummy_parameters;
        preferences,
        is_prioritized_constraint=[fill(false, num_levels) for _ in 1:num_players],
        equality_constraints,
        inequality_constraints,
    )
    (; problem, layout)
end

"θ and the initial guess of one instance: every solve, reduced or complete, starts at z = 0."
instance_inputs(instance, settings) =
    (; instance.θ, initial_guess=zeros(settings.num_players * settings.primal_dimension))

# ── Settings and experiment driver ─────────────────────────────────────────────

"""
Default experiment settings, grouped by purpose; a setting that MonteCarlo_Intersection also has uses
its name. Replace entries with `run(overrides = (; num_instances = 10))`. Saved verbatim in results.jls.
The solver is the paper's, Algorithm 5.1 (stage_options); beyond the stage tolerances its one setting is
`linear_solver`.
"""
default_settings() = (;
    # Monte Carlo
    num_instances=100,                           # instances per objective class and K
    rng_seed=123,                                # (class, K) samples from Xoshiro(rng_seed + 100·class index + K)
    objective_classes=(:quadratic, :nonquadratic), # names in objective_classes(settings)
    # K, the preference levels of every player, per class (levels_for; a range applies to every class).
    # Cosh code generation time grows fast with K: its systems at K = 5, 6 do not build in reasonable time.
    num_levels=(; quadratic=2:6, nonquadratic=2:4),
    # K whose complete system is built, per class (levels_for); other K solve only the reduced system.
    # Quadratic K = 6 builds none: with :svd each Newton step there is a dense pinv of a 3248 × 3248 Jacobian
    # (3–6 s), ≈ 5–9 h for the 100 instances.
    complete_levels=(; quadratic=2:5, nonquadratic=2:4),
    # Problem (module docstring)
    num_players=4,                               # N
    primal_dimension=10,                         # n, primal variables per player
    coupling=0.03,                               # κ, scale of the other players' columns in R_k, v_k and H (coupled_rows)
    own_blocks=:orthogonal,                      # own columns of H, R_k, v_k: :orthogonal (orthonormal rows) or :gaussian (own_rows)
    equality_dimension=3,                        # m_E, coupled linear equalities per player
    inequality_dimension=2,                      # m_I, inequalities per player, on D(z)
    quadratic_top_level=:norm,                   # J_1 of the quadratic class: :norm (‖z‖²) or :rank2 (½‖R_1 z + p_1‖²)
    nonquadratic_constraint_type=:nonlinear,     # inequalities of the nonquadratic class: :linear or :nonlinear (quadratic: :linear)
    target_exclusion=0.5,                        # probability that a player's inequalities exclude its innermost target
    target_floor=0.25,                           # minimum ‖target‖ of the innermost level (redraw_target)
    # Algorithm 5.1, in plain ‖·‖₂ (your calls, 2026-10-30; module docstring)
    #! format: off
    # 13 stages, to 1e-12: every system meets ‖K₀‖₂ < epsilon_0 by then (your call, 2026-10-30).
    rho_schedule = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12], # both systems
    #! format: on
    max_inner_iters=1000,                        # t_max, Newton steps per ρ stage
    gc_before_solve=true,                        # a full garbage collection before each solve, outside its timing (solve_algorithm51)
    linear_solver=:klu,                         # PDIPOptions.linear_solver: :klu (Levenberg–Marquardt, η = 1e-6) or :svd (the exact pinv)
    epsilon_0=1e-10,                             # ε₀, the success tolerance on the unperturbed ‖K₀‖₂
    # Fig 4 compares the reduced and complete iterates at the end of this stage, so both are at the same ρ
    # (the name keeps the figure's earlier number, as does the CSV column distance_fig3).
    fig3_rho=1e-11,
    # Start (cold_initialization)
    positive_seed=1.0,                           # nonnegative variables
    positive_floor=1e-10,                        # floor for every nonnegative variable
    # ‖z_complete − z_reduced‖₂ up to this counts as the same solution (pass counts).
    same_solution_tol=1e-5,
    output_dir=joinpath(dirname(@__DIR__), "output", "random_goop_study"),
)

"""
The K values under `name` in `setting`: a range applies to every name, and a named tuple gives each its
own. `num_levels` and `complete_levels` are keyed by class name.
"""
levels_for(setting, name) = setting isa NamedTuple ? setting[name] : setting

"""
Default settings merged with `overrides`, then checked: an unknown name, class, constraint type, top
level or own-block kind, a K below 2 or an empty schedule is an error.
"""
function validated_settings(overrides)
    defaults = default_settings()
    unknown = setdiff(keys(overrides), keys(defaults))
    isempty(unknown) || error("Unknown setting(s) in overrides: $(join(unknown, ", "))")
    settings = merge(defaults, overrides)
    all(in((:quadratic, :nonquadratic)), settings.objective_classes) ||
        error("objective_classes must be :quadratic and/or :nonquadratic")
    settings.nonquadratic_constraint_type in (:linear, :nonlinear) ||
        error("nonquadratic_constraint_type must be :linear or :nonlinear")
    settings.quadratic_top_level in (:norm, :rank2) || error("quadratic_top_level must be :norm or :rank2")
    settings.own_blocks in (:orthogonal, :gaussian) || error("own_blocks must be :orthogonal or :gaussian")
    for name in (:num_levels, :complete_levels)
        setting = settings[name]
        setting isa NamedTuple && !all(c -> haskey(setting, c), settings.objective_classes) &&
            error("$name must give a range for every class in objective_classes")
    end
    # The complete system needs two levels.
    settings.num_instances > 0 && all(c -> minimum(levels_for(settings.num_levels, c)) >= 2, settings.objective_classes) ||
        error("invalid num_instances or num_levels")
    0 <= settings.coupling <= 1 || error("coupling must lie in [0, 1]")
    settings.target_floor >= 0 || error("target_floor must be nonnegative")
    settings.epsilon_0 > 0 || error("epsilon_0 must be positive")
    any(r -> isapprox(r, settings.fig3_rho; rtol=1e-9), settings.rho_schedule) ||
        error("fig3_rho must be a stage of rho_schedule")
    # Two rows fill the innermost image: two half-planes or disks in R², a slab or two intervals in R.
    settings.inequality_dimension == 2 || error("inequality_dimension must be 2")
    !isempty(settings.rho_schedule) && all(ρ -> isfinite(ρ) && ρ > 0, settings.rho_schedule) ||
        error("invalid relaxation schedule")
    settings
end

"""
One class's problem and systems at K = num_levels: the reduced system (5.2) and, for K in complete_levels,
the ρ-perturbed complete system (6.1), each with its σ targets. Prints the system sizes, and returns
everything with its results.jls metadata (sizes, construction times).
"""
function build_systems(class, num_levels, settings)
    (; problem, layout) = get_setup(class, num_levels, settings)
    label = "$(class.name), K = $num_levels"
    # Both systems are GOOPKKTSystems, solved alike by PDIP (solve_algorithm51). The reduced system's σ
    # start at g(0), once per level (interior_point_slack_targets).
    print("Building the reduced KKT system (5.2) for $label (symbolic construction, Jacobian and code generation)... ")
    flush(stdout)
    construction = @timed ReducedGOOP.generate_slacked_reduced_kkt_system(problem)
    println("done in $(round(construction.time; digits=1)) s")
    kkt = construction.value
    reduced = (;
        name=:interior_point,
        kkt,
        slack_targets=interior_point_slack_targets(problem, kkt),
        construction_seconds=construction.time,
    )
    complete = nothing
    if num_levels in levels_for(settings.complete_levels, class.name)
        print("Building the ρ-perturbed complete KKT system (6.1) for $label (symbolic construction, Jacobian and code generation)... ")
        flush(stdout)
        construction = @timed ReducedGOOP.generate_slacked_complete_kkt_system(problem; complementarity=:perturbed)
        println("done in $(round(construction.time; digits=1)) s")
        kkt = construction.value
        @assert kkt.variable_dimension == kkt.kkt_dimension # square
        complete = (;
            name=:perturbed,
            kkt,
            slack_targets=MCI.slack_target_values(kkt),
            construction_seconds=construction.time,
        )
    end
    # Sizes of the systems the solver works on: variables y and equations, no complementarity pairs.
    sizes(p) = (; p.name, variables=p.kkt.variable_dimension, equations=p.kkt.kkt_dimension, p.construction_seconds)
    metadata = (;
        class=class.name,
        num_levels,
        levels=[class.levels(k) for k in 1:num_levels],
        class.constraint_type,
        reduced=sizes(reduced),
        complete=isnothing(complete) ? nothing : sizes(complete),
    )
    println("KKT system sizes, $label:")
    for (system, p) in (("reduced", metadata.reduced), ("complete", metadata.complete))
        isnothing(p) || println("  ", rpad("$system :$(p.name)", 26), p.variables, " variables · ", p.equations, " equations")
    end
    (; class, num_levels, problem, layout, reduced, complete, metadata)
end

"""
Start of the σ slacks of the :interior_point reduced system: each player's inequality values g_i(z), once
per level, in the order that system holds them (players in turn, levels outermost first).
"""
function interior_point_slack_targets(problem, kkt)
    names = string.(kkt.z_symbolic[kkt.interior_point_slack_dims])
    order = [(i, k) for i in 1:problem.num_players for k in eachindex(problem.preferences[i]) for _ in 1:problem.inequality_dims[i]]
    @assert all(startswith(name, "σ_$(i)_$(k)") for (name, (i, k)) in zip(names, order)) "unexpected σ order"
    function (y, θ, ρ)
        z = BlockArray(y[kkt.primal_dims], problem.primal_dims)
        θ_blocks = BlockArray(θ, problem.parameter_dims)
        reduce(vcat, (repeat(problem.inequality_constraints[i](z, θ_blocks), length(problem.preferences[i])) for i in 1:problem.num_players); init=Float64[])
    end
end

"""
Untimed JIT warm-up on `instance`, so measured trials don't pay first-call compilation: one solve of each
system, results discarded.
"""
function jit_warmup(systems, instance, settings)
    (; class, num_levels, problem) = systems
    (; θ, initial_guess) = instance_inputs(instance, settings)
    print("Warm-up for $(class.name), K = $num_levels (one untimed solve of each system on instance 1; results discarded)... ")
    flush(stdout)
    # One-time cost of each system's first solve: its wall time and the part spent compiling (setup.csv).
    timings = NamedTuple[]
    for (i, (system, formulation)) in enumerate(solved_systems(systems))
        context = "warm-up, $(class.name), K = $num_levels"
        timed = @timed solve_algorithm51(formulation, problem, θ, initial_guess, settings; context, io=devnull)
        compile_seconds = timed.compile_time + timed.recompile_time
        push!(timings, (; system, formulation=formulation.name, first_solve_seconds=timed.time, compile_seconds))
        print(
            i == 1 ? "" : ", ",
            "$system done in $(round(timed.time; digits=1)) s (compile_seconds $(round(compile_seconds; digits=1)) s)",
        )
        flush(stdout)
    end
    println()
    timings
end

"The systems each trial of one (class, K) solves: the reduced one, then the complete one if built."
solved_systems(systems) =
    isnothing(systems.complete) ? [(:reduced, systems.reduced)] : [(:reduced, systems.reduced), (:complete, systems.complete)]

"""
One trial's reduced solve: print the trial banner, solve from the instance's initial guess
(instance_inputs), print the verdict, and return the trial's record. Its `checks` start empty; `run`
fills them.
"""
function solve_reduced_kkt(systems, trial, instance, settings)
    (; class, num_levels) = systems
    println("\n══ Trial $trial/$(settings.num_instances) · $(class.name), K = $num_levels ══")
    excluded = [p.excluded for p in instance.players]
    println("  Players whose inequalities exclude their innermost target: ", any(excluded) ? join(findall(excluded), ", ") : "none")
    flush(stdout)
    (; θ, initial_guess) = instance_inputs(instance, settings)
    context = trial_context(trial, class.name, num_levels)
    reduced = solve_algorithm51(systems.reduced, systems.problem, θ, initial_guess, settings; context)
    println(solve_verdict("reduced (5.2)", reduced, settings), isnothing(systems.complete) ? "" : " Running its complete solve.")
    flush(stdout)
    (; trial, class=class.name, num_levels, θ, excluded, reduced, checks=Any[])
end

"Label naming a trial in the solver tables."
trial_context(trial, class_name, num_levels) = "trial $trial, $class_name, K = $num_levels"

"""
One trial's complete solve, with its verdict printed. It uses only the trial's initial guess, so it runs
whether or not the reduced solve was accepted. An exception (in practice, LAPACK's SVD not converging)
rejects the solve.
"""
function check_complete_kkt(record, systems, instance, settings)
    (; θ, reduced) = record
    (; initial_guess) = instance_inputs(instance, settings)
    complete = try
        solve_algorithm51(
            systems.complete,
            systems.problem,
            θ,
            initial_guess,
            settings;
            context=trial_context(record.trial, record.class, record.num_levels),
        )
    catch err
        start = cold_initialization(systems.complete, initial_guess, θ, settings)
        validation = (;
            rms_K_ρ=NaN,
            rms_K₀=NaN,
            min_K₀=NaN,
            max_K₀=NaN,
            max_abs_K_ρ=NaN,
            sign_violation=NaN,
            primal=(; equality_max=NaN, inequality_violation=NaN),
            accepted=false,
        )
        (;
            accepted=false,
            validation,
            physical=start[systems.complete.kkt.primal_dims],
            solution=start,
            stages=Any[],
            best_stage=0,
            status=:exception,
            reason=:exception,
            error=sprint(showerror, err, catch_backtrace()),
            newton_steps=0,
            seconds=0.0,
            compile_seconds=0.0,
        )
    end
    metrics = solution_distance(reduced.physical, complete.physical)
    # :numerical_failure   complete solve not accepted (no conclusion either way)
    # :reduced_rejected    complete accepted, but there is no accepted reduced solution to compare
    # :same_solution       both accepted; ‖z_complete − z_reduced‖₂ <= same_solution_tol
    # :different_solution  both accepted; the solutions differ by more
    category =
        !complete.accepted ? :numerical_failure :
        !reduced.accepted ? :reduced_rejected :
        metrics.absolute <= settings.same_solution_tol ? :same_solution : :different_solution
    compared =
        category === :reduced_rejected ? " No accepted reduced solution to compare." :
        complete.accepted ? " ‖z_complete − z_reduced‖₂ = $(MCI.compact(metrics.absolute)) ($(replace(string(category), '_' => ' ')))." : ""
    println(solve_verdict("complete :perturbed", complete, settings), compared)
    (; formulation=:perturbed, result=complete, metrics, distance_fig3=fig3_distance(reduced, complete, systems, settings), category)
end

"""
Fig 4's ‖z_complete − z_reduced‖₂: the two iterates at the end of the stage ρ = `fig3_rho`, so both are at
the same ρ (your call, 2026-10-30). `nothing` unless both solves were accepted.
"""
function fig3_distance(reduced, complete, systems, settings)
    reduced.accepted && complete.accepted || return nothing
    at(stages) = findfirst(s -> isapprox(s.ρ, settings.fig3_rho; rtol=1e-9), stages)
    i, j = at(reduced.stages), at(complete.stages)
    (isnothing(i) || isnothing(j)) && return nothing
    primal(p) = p.kkt.primal_dims
    norm(complete.stages[j].final[primal(systems.complete)] - reduced.stages[i].final[primal(systems.reduced)])
end

"""
Run the experiment described in the module docstring: for each objective class and K, sample the
instances, build the systems, warm up, and solve every trial's reduced system and complete solves.
Writes results.jls (after every solve) and the CSV files to a new directory under `output_dir`, and
returns the run's directory, settings, instances, results and metadata.
"""
function run(; overrides=(;))
    settings = validated_settings(overrides)
    directory = MCI.run_directory(settings.output_dir)
    instances, results, metadata = Any[], Any[], Any[]
    save() = MCI.save_results(directory, settings, instances, nothing, metadata, results)
    classes = filter(c -> c.name in settings.objective_classes, objective_classes(settings))
    total, done = sum(c -> length(levels_for(settings.num_levels, c.name)), classes) * settings.num_instances, 0
    println(
        "Random GOOP study: $(settings.num_instances) instances per class and K, K ∈ $(settings.num_levels), ",
        "N = $(settings.num_players), n = $(settings.primal_dimension). Output: $directory",
    )
    println(
        "Algorithm 5.1 (PDIP, $(settings.linear_solver)): stage ρ to ‖K_ρ‖₂ < ρ; ",
        "the solve stops, accepted, once ‖K₀‖₂ < ε₀ = $(MCI.compact(settings.epsilon_0))",
    )
    # Untimed warm-up build, discarded: a session's first build also compiles the builders, which made the
    # first timed build ≈ 10× slower (run 20261028_231456: quadratic K = 2 reduced 8.9 s, K = 3 0.9 s).
    let class = first(classes), num_levels = first(levels_for(settings.num_levels, class.name))
        print("Warm-up build for $(class.name), K = $num_levels (untimed; discarded)... ")
        seconds = @elapsed redirect_stdout(() -> build_systems(class, num_levels, settings), devnull)
        println("done in $(round(seconds; digits=1)) s")
    end
    for class in classes, num_levels in levels_for(settings.num_levels, class.name)
        class_instances = sample_instances(instance_rng(class, num_levels, settings), class, num_levels, settings)
        push!(instances, (; class=class.name, num_levels, instances=class_instances))
        systems = build_systems(class, num_levels, settings)
        push!(metadata, systems.metadata)
        save()
        warmup = jit_warmup(systems, first(class_instances), settings)
        # The warm-up's one-time costs join this system's metadata (setup.csv).
        metadata[end] = merge(metadata[end], (; warmup))
        # Warm-up results are discarded; every measured trial starts afresh.
        for (trial, instance) in enumerate(class_instances)
            record = solve_reduced_kkt(systems, trial, instance, settings)
            # `record.checks` is filled in place below. The record already in `results` sees each
            # push, so every save stores the progress so far.
            push!(results, record)
            isnothing(systems.complete) || push!(record.checks, check_complete_kkt(record, systems, instance, settings))
            # results.jls grows to hundreds of MB over a full run, so it is rewritten every 10 trials and
            # after the last trial of each class and K, not after every solve.
            (trial % 10 == 0 || trial == length(class_instances)) && save()
            done += 1
            print_progress(done, total)
        end
    end
    println()
    export_csv(directory)
    println("Saved results.jls and the CSV files: $directory")
    plot_results(directory)
    print_final_summary(results, settings)
    (; directory, settings, instances, results, metadata)
end
# End of Monte Carlo random GOOP experiments


# ── Sampling and solves ────────────────────────────────────────────────────────

"The random stream of one (class, K): adding a class or a K leaves every other stream unchanged."
instance_rng(class, num_levels, settings) =
    Xoshiro(settings.rng_seed + 100findfirst(==(class.name), (:quadratic, :nonquadratic)) + num_levels)

"Player `player`'s own columns of z."
own_columns(player, settings) = ((player-1)*settings.primal_dimension+1):(player*settings.primal_dimension)

"Rows a level contributes to its player's own rows: R (:quadratic) 2, v (:cosh) 1, :norm none."
level_rows(kind) = kind === :quadratic ? 2 : kind === :cosh ? 1 : 0

"""
`num_rows` own-column rows of one player, N(0, 1/n) for `own_blocks = :gaussian`, so their norms are
≈ 1. For :orthogonal, the first min(num_rows, n) are instead the rows of a random orthogonal matrix,
so they are orthonormal. With these as the player's rows in priority order, the best response stays
well conditioned: Gaussian blocks are often nearly singular once the rows m_E + 2(K−1) (quadratic)
approach n, whatever κ is. Rows beyond n stay Gaussian.
"""
function own_rows(rng, num_rows, settings)
    n = settings.primal_dimension
    A = randn(rng, num_rows, n) / sqrt(n)
    settings.own_blocks === :orthogonal || return A
    # Q from the QR factorization of a Gaussian matrix, with the signs of R's diagonal moved into Q, is
    # uniformly distributed over the orthogonal matrices.
    F = qr(randn(rng, n, n))
    Q = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    k = min(num_rows, n)
    A[1:k, :] = Q[1:k, :]
    A
end

"""
Rows acting on z: player `player`'s own columns are `own` (own_rows), and the other players' columns
κ·N(0, 1/n) with κ = settings.coupling. κ < 1 keeps the players coupled but makes each best response
depend less on the others than on its own variables, which keeps the equilibrium unique and well
conditioned.
"""
function coupled_rows(rng, own, player, settings)
    (; num_players, primal_dimension, coupling) = settings
    A = coupling * randn(rng, size(own, 1), num_players * primal_dimension) / sqrt(primal_dimension)
    A[:, own_columns(player, settings)] = own
    A
end

"""
`num_instances` instances of `class` at K = num_levels. Each draws, for every player: its own rows in
priority order, innermost first (own_rows: H, then levels K, K−1, …, 1); H from coupled_rows (its own
block has full row rank); each level's data (sample_level); a Gaussian b; and the inequality data
(sample_inequality). Returns each instance's θ and its players' data, including whether the player's
inequalities exclude its innermost target.
"""
function sample_instances(rng, class, num_levels, settings)
    (; num_players, primal_dimension, equality_dimension) = settings
    layout = parameter_layout(class, num_levels, settings)
    kinds = [class.levels(k) for k in 1:num_levels]
    innermost = kinds[num_levels]
    map(1:settings.num_instances) do _
        players = map(1:num_players) do player
            own = Iterators.Stateful(eachrow(own_rows(rng, equality_dimension + sum(level_rows, kinds), settings)))
            next_own(count) = permutedims(reduce(hcat, Iterators.take(own, count); init=zeros(primal_dimension, 0)))
            H = coupled_rows(rng, next_own(equality_dimension), player, settings)
            rank(H[:, own_columns(player, settings)]) == equality_dimension ||
                error("player $player: own block of H is rank deficient")
            levels = reverse([sample_level(rng, kinds[k], next_own(level_rows(kinds[k])), player, settings) for k in num_levels:-1:1])
            levels[num_levels] = redraw_target(rng, innermost, levels[num_levels], settings)
            b = randn(rng, equality_dimension)
            target = innermost_target(innermost, levels[num_levels])
            (; inequality, excluded) = sample_inequality(rng, class.constraint_type, target, settings)
            (; levels, H, b, inequality, excluded, target)
        end
        blocks = [flatten_parameters(p, layout) for p in players]
        @assert all(unflatten_parameters(block, layout) == (; p.levels, p.H, p.b, p.inequality) for (block, p) in zip(blocks, players))
        (; θ=reduce(vcat, blocks), players)
    end
end

"""
One level's data for `player`, whose own columns are `own` (own_rows). :quadratic: R from coupled_rows
and a Gaussian p. :cosh: v from coupled_rows, normalized to ‖v‖ = 1, and a Gaussian s. :norm has none.
"""
function sample_level(rng, kind, own, player, settings)
    kind === :norm && return (;)
    kind === :quadratic && return (; R=coupled_rows(rng, own, player, settings), p=randn(rng, 2))
    (; v=normalize(vec(coupled_rows(rng, own, player, settings))), s=randn(rng))
end

"""
The innermost level's data with its offset (p, or s) redrawn until its target (innermost_target) has norm
at least `target_floor`. The inequality geometry scales with ‖target‖ (sample_inequality); near zero it
shrinks so far that the perturbed reduced system (5.2) cannot resolve it (tuning: failures at ‖target‖ ≤
0.07, stages at the iteration cap up to 0.17).
"""
function redraw_target(rng, kind, data, settings)
    while norm(innermost_target(kind, data)) < settings.target_floor
        data = kind === :quadratic ? merge(data, (; p=randn(rng, 2))) : merge(data, (; s=randn(rng)))
    end
    data
end

"""
One player's inequality data: two constraints on D(z) ∈ R^d, each strictly satisfied at D = 0, the
image of the zero initial guess. `target` is the innermost objective's unconstrained optimum of D. With
probability settings.target_exclusion one constraint excludes it, so that constraint is active at the
equilibrium with a positive multiplier; otherwise both contain it. All scales are relative to ‖target‖.
  :linear     W D − c ≥ 0: unit rows w_j (random in R², the pair ±1 in R, a slab).
  :nonlinear  r_j² − ‖D − c_j‖² ≥ 0: disks (intervals in R) with centers c_j = ‖target‖·U(0.2, 1)·u_j
              for unit u_j as above.
"""
function sample_inequality(rng, constraint_type, target, settings)
    d, m = length(target), settings.inequality_dimension
    scale = norm(target)
    excluded = rand(rng) < settings.target_exclusion
    directions = d == 1 ? reshape([1.0, -1.0], m, 1) : reduce(vcat, (normalize(randn(rng, d))' for _ in 1:m))
    # The excluding constraint faces the target: its direction has a negative inner product with it.
    j = argmin(directions * target)
    excluded && dot(directions[j, :], target) >= 0 && (directions[j, :] .*= -1)
    margin() = scale * (0.1 + 0.4rand(rng))     # strict clearance of 0 and, unless excluded, of the target
    fraction() = 0.2 + 0.6rand(rng)              # where an excluding boundary sits between 0 and the target
    if constraint_type === :linear
        W = directions
        # w_jᵀD − c_j ≥ 0 holds at D = 0 iff c_j < 0, and at the target iff c_j ≤ w_jᵀtarget.
        c = [min(0.0, dot(W[i, :], target)) - margin() for i in 1:m]
        excluded && (c[j] = fraction() * dot(W[j, :], target))
        return (; inequality=(; W, c), excluded)
    end
    C = directions .* [scale * (0.2 + 0.8rand(rng)) for _ in 1:m]
    # r_j² − ‖D − c_j‖² ≥ 0 holds at D = 0 iff r_j > ‖c_j‖, and at the target iff r_j ≥ ‖target − c_j‖.
    r = [max(norm(C[i, :]), norm(target - C[i, :])) + margin() for i in 1:m]
    if excluded
        # ‖target − c_j‖ > ‖c_j‖ because c_jᵀtarget ≤ 0, so this radius keeps 0 and drops the target.
        inner, outer = norm(C[j, :]), norm(target - C[j, :])
        r[j] = inner + fraction() * (outer - inner)
    end
    (; inequality=(; C, r), excluded)
end

"""
Starting point of every solve, reduced or complete: the steps of MonteCarlo_Intersection's
complete_initialization without its step 3, so nothing comes from another solve.
 1. physical coordinates <- the instance's initial guess (z = 0);
 2. every nonnegative coordinate <- positive_seed, floored at positive_floor;
 3. every unsigned multiplier stays 0;
 4. auxiliary slacks σ <- their defining expressions (the formulation's slack_targets) at the first ρ,
    floored as in 2.
"""
function cold_initialization(formulation, initial_guess, θ, settings)
    (; kkt) = formulation
    y = zeros(kkt.variable_dimension)
    y[kkt.primal_dims] .= initial_guess
    y[kkt.nonnegative_dims] .= max(settings.positive_seed, settings.positive_floor)
    y[kkt.interior_point_slack_dims] .=
        max.(formulation.slack_targets(y, θ, first(settings.rho_schedule)), settings.positive_floor)
    @assert y[kkt.primal_dims] == initial_guess
    y
end

"Header above each solve's ρ-stage table; `label` names the solve."
function print_kkt_header(label; io=stdout)
    println(io, "\n[solver exit] $label")
    println(io, "Residuals ‖·‖₂; K₀ min/max are its smallest and largest rows; min g is the smallest inequality value.")
    @printf(
        io,
        "%-9s %-7s %6s %8s %12s %12s %10s %10s %9s %10s  %s\n",
        "ρ",
        "status",
        "steps",
        "time_s",
        "‖K_ρ‖",
        "‖K₀‖",
        "K₀ min",
        "K₀ max",
        "‖z‖",
        "min g",
        "stop"
    )
    flush(io)
end

function print_kkt_stage(;
    ρ,
    status,
    newton_steps,
    seconds,
    norm_K_ρ,
    norm_K₀,
    min_K₀,
    max_K₀,
    physical,
    problem,
    θ,
    stop,
    io=stdout,
)
    worst = ReducedGOOP.is_feasible(problem, physical, θ).worst
    @printf(
        io,
        "%-9.0e %-7s %6d %8.3f %12.2e %12.2e %10.2e %10.2e %9.3f %10.2e  %s\n",
        ρ,
        string(status),
        newton_steps,
        seconds,
        norm_K_ρ,
        norm_K₀,
        min_K₀,
        max_K₀,
        norm(physical),
        worst,
        stop
    )
    flush(io)
end

"""
PDIPOptions of one ρ stage of Algorithm 5.1 (with settings.linear_solver), in plain ‖·‖₂: the stage
stops at ‖K_ρ‖₂ < ε_ρ = ρ, and the solver's ‖K₀‖₂ < epsilon_0 test, before every Newton step, stops the
whole solve. The histories make the stage's trace (stage_history), which results.jls keeps.
"""
function stage_options(settings, ρ)
    ReducedGOOP.PDIPOptions(;
        ρ_schedule=[Float64(ρ)],
        max_inner_iters=settings.max_inner_iters,
        tol=settings.epsilon_0,
        tol_inner=ρ,
        linear_solver=settings.linear_solver,
        record_convergence=true,
    )
end

"""
A stage's trace from PDIP's histories (one entry per stopping test): one row, ‖K_ρ‖₂ (`norm_K_ρ`) and ‖K₀‖₂
(`norm_K₀`), per iterate a Newton step was taken from, then the stage's last iterate: `newton_steps` + 1 rows.
"""
function stage_history(output)
    history = [
        (; norm_K_ρ=output.norm_K_ρ_history[k], norm_K₀=output.norm_K₀_history[k], eta=NaN) for k in 1:output.newton_steps
    ]
    push!(history, (; norm_K_ρ=output.norm_K_ρ, norm_K₀=output.norm_K₀, eta=NaN))
end

"""
Algorithm 5.1 on one system, `formulation` (the reduced (5.2) or the complete :perturbed), from the cold
start down `rho_schedule` (stage_options). Each stage starts at the previous one's last iterate. The solve
is judged at its best stage, the one with the lowest ‖K₀‖₂: accepted iff ‖K₀‖₂ <= epsilon_0 there. After
the solver's ‖K₀‖₂ < epsilon_0 stop, every later stage returns at its first stopping test, with no Newton
step. With `gc_before_solve`, a full garbage collection runs first, outside the stages' timing: a solve takes
10 ms to a few seconds, and a collection that the previous solves' garbage triggers inside it would be
timed as part of it.
"""
function solve_algorithm51(formulation, problem, θ, initial_guess, settings; context, io=stdout)
    (; kkt) = formulation
    settings.gc_before_solve && GC.gc()
    y = cold_initialization(formulation, initial_guess, θ, settings)
    # The full residual at the stage's ρ and at ρ = 0 (K₀), for the table and the best-stage choice.
    r, r0 = zeros(kkt.kkt_dimension), zeros(kkt.kkt_dimension)
    stages = Any[]
    best = nothing
    system = formulation.name === :interior_point ? "reduced (5.2)" : "complete :$(formulation.name)"
    print_kkt_header("$context | $system | Algorithm 5.1 (PDIP, $(settings.linear_solver))"; io)
    for ρ in settings.rho_schedule
        stage_initial = copy(y)
        options = stage_options(settings, ρ)
        timed = @timed ReducedGOOP.solve(ReducedGOOP.PDIP(), kkt, θ; z₀=y, options)
        output = timed.value
        y .= output.z
        push!(stages, MCI.stage_record(; ρ, parameter=:rho, timed, history=stage_history(output), initial=stage_initial, final=copy(y)))
        kkt.K_ρ!(r, y; θ, ρ)
        kkt.K₀!(r0, y; θ)
        print_kkt_stage(;
            ρ,
            status=output.status,
            newton_steps=output.newton_steps,
            seconds=last(stages).seconds,
            norm_K_ρ=norm(r),
            norm_K₀=norm(r0),
            min_K₀=minimum(r0),
            max_K₀=maximum(r0),
            physical=y[kkt.primal_dims],
            problem,
            θ,
            stop=MCI.Intersection.solver_stop_reason(output, options; scale=1.0),
            io,
        )
        # Track the stage with the lowest ‖K₀‖₂ (a non-finite residual counts as Inf).
        k₀ = isfinite(norm(r0)) ? norm(r0) : Inf
        if isnothing(best) || k₀ < best.k₀
            best = (; y=copy(y), ρ, k₀, index=length(stages))
        end
    end
    # Judge the best stage, not the last. The full residual retains stationarity, equality,
    # complementarity and slack equations.
    y = best.y
    kkt.K_ρ!(r, y; θ, ρ=best.ρ)
    kkt.K₀!(r0, y; θ)
    validation = MCI.validate_solution(
        problem,
        θ,
        y[kkt.primal_dims];
        r,
        k₀=r0,
        rms_K_ρ=norm(r),
        rms_K₀=norm(r0),
        sign_violation=max(0.0, -minimum(y[kkt.nonnegative_dims]; init=Inf)),
        tol=settings.epsilon_0,
    )
    (;
        accepted=validation.accepted,
        validation,
        physical=y[kkt.primal_dims],
        solution=y,
        stages,
        best_stage=best.index,
        status=stages[best.index].status,
        reason=stages[best.index].reason,
        newton_steps=sum(s.newton_steps for s in stages),
        seconds=sum(s.seconds for s in stages),
        compile_seconds=sum(s.compile_seconds for s in stages),
    )
end

# ── Metrics, saved results and final summary ───────────────────────────────────

"‖z_complete − z_reduced‖₂, absolute (Fig 3) and relative to ‖z_reduced‖₂, and the largest entry of the difference."
function solution_distance(reduced, complete)
    δ = complete - reduced
    (; absolute=norm(δ), relative=norm(δ)/max(norm(reduced), 1e-12), max_abs=maximum(abs, δ))
end

"""
One-line outcome of a solve, printed under its stage table: accepted iff ‖K₀‖₂ <= epsilon_0 at its best
stage.
"""
function solve_verdict(name, result, settings)
    haskey(result, :error) && return "→ $name FAILED with an exception: $(first(split(result.error, '\n')))"
    stage = "ρ = $(MCI.compact(result.stages[result.best_stage].ρ)), stage $(result.best_stage) of $(length(result.stages))"
    k₀, ε₀ = MCI.compact(result.validation.rms_K₀), MCI.compact(settings.epsilon_0)
    range = MCI.k₀_range(result.validation)
    result.accepted ? "→ $name accepted at $stage: ‖K₀‖₂ = $k₀ ≤ ε₀ = $ε₀; $range." :
    "→ $name REJECTED: best ‖K₀‖₂ = $k₀ ($stage) > ε₀ = $ε₀; $range."
end

"Progress bar on its own line: other output interleaves with it, so it is not redrawn in place."
function print_progress(done, total; width=24)
    filled = floor(Int, width * done / total)
    println("[", repeat("=", filled), repeat(" ", width - filled), "] $done/$total trials")
    flush(stdout)
end

"Pass counts, one line per class and K, then a key: printed at the end of a run."
function pass_count_lines(results, settings)
    lines = [@sprintf("%-13s %2s %8s %9s %9s %9s %11s", "class", "K", "reduced", "complete", "both", "same z", "max ‖Δz‖₂")]
    for class_name in settings.objective_classes, num_levels in levels_for(settings.num_levels, class_name)
        rows = filter(r -> r.class === class_name && r.num_levels == num_levels, results)
        isempty(rows) && continue
        n = length(rows)
        num_reduced = count(r -> r.reduced.accepted, rows)
        checks = [c for r in rows for c in r.checks]
        if isempty(checks) # no complete system at this K (complete_levels)
            push!(lines, @sprintf("%-13s %2d %8s %9s", class_name, num_levels, "$num_reduced/$n", "—"))
            continue
        end
        num_complete = count(c -> c.result.accepted, checks)
        compared = filter(c -> c.category in (:same_solution, :different_solution), checks)
        num_same = count(c -> c.category === :same_solution, compared)
        distances = [c.metrics.absolute for c in compared]
        push!(
            lines,
            @sprintf(
                "%-13s %2d %8s %9s %9s %9s %11s",
                class_name,
                num_levels,
                "$num_reduced/$n",
                "$num_complete/$n",
                "$(length(compared))/$n",
                "$num_same/$(length(compared))",
                isempty(distances) ? "—" : MCI.compact(maximum(distances)),
            ),
        )
    end
    push!(
        lines,
        "",
        "reduced, complete: accepted / trials; both: trials where both were accepted; same z: of those,",
        "‖z_complete − z_reduced‖₂ ≤ same_solution_tol = $(MCI.compact(settings.same_solution_tol)).",
    )
    lines
end

"Final console summary: the pass counts."
function print_final_summary(results, settings; io=stdout)
    println(io, "\n[final summary] Pass counts")
    foreach(line -> println(io, line), pass_count_lines(results, settings))
    flush(io)
    nothing
end

# ── CSV exports for the figures and Table 2 ────────────────────────────────────

"Write `rows` (tuples) under `header` as comma-separated values."
function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        foreach(row -> println(io, join(row, ",")), rows)
    end
    path
end

"""
Figures and tables of a run (experiments/RandomGOOP_plotting.py: Tables 2 and 3, Figs 2–4, under their
run names and, for the manuscript, under its file names), written to <directory>/figures/ from the CSV
files. Runs the Python script with uv. A failure only warns, since results.jls and the CSV files are
already saved; rerun with plot_results(directory).
"""
function plot_results(directory)
    script = joinpath(@__DIR__, "RandomGOOP_plotting.py")
    # Base.run: this module's own `run` is the experiment driver. --no-project --isolated: a fresh
    # environment with matplotlib and numpy, whatever virtual environment the shell has active.
    command = `uv run --quiet --no-project --isolated --with matplotlib --with numpy python -I $script $directory`
    try
        Base.run(command)
        println("Saved the figures and tables: $(joinpath(directory, "figures"))")
    catch err
        @warn "Plotting failed; the results and CSV files are saved. Rerun plot_results(\"$directory\")." exception = err
    end
    directory
end

"""
The run's CSV files, read by experiments/RandomGOOP_plotting.py:
  sizes.csv     one row per (class, K, system): variables, equations, build time
  setup.csv     one row per (class, K, system): build time, and the warm-up's first-solve time and the
                part of it spent compiling (jit_warmup)
  solves.csv    one row per solve (system reduced, formulation interior_point: the (5.2); complete,
                perturbed): accepted, solve time (no compilation), Newton steps,
                ‖K₀‖₂ of the judged iterate, the distance to the trial's reduced z, and Fig 4's distance
                between the two iterates at the end of the stage ρ = fig3_rho (distance_fig3)
  stages.csv    one row per ρ stage of every solve: Newton steps, solve time (no
                compilation), compile time, status, and ‖K_ρ‖₂, ‖K₀‖₂ at the stage's last iterate
  settings.csv  the scalar settings (dimensions for the size formulas, tolerances, coupling)
`newton_steps` counts the Newton steps taken (a step the line search rejects is not one).
"""
function export_csv(directory)
    (; settings, metadata, results) = deserialize(joinpath(directory, "results.jls"))
    scalars = [(name, value) for (name, value) in pairs(settings) if value isa Union{Real,Symbol}]
    write_csv(joinpath(directory, "settings.csv"), ("name", "value"), scalars)
    sizes, setup = Any[], Any[]
    for m in metadata
        for (system, p) in (("reduced", m.reduced), ("complete", m.complete))
            isnothing(p) || push!(sizes, (m.class, m.num_levels, system, p.name, p.variables, p.equations, p.construction_seconds))
        end
        for w in get(m, :warmup, NamedTuple[])
            build = (w.system === :reduced ? m.reduced : m.complete).construction_seconds
            push!(setup, (m.class, m.num_levels, w.system, w.formulation, build, w.first_solve_seconds, w.compile_seconds))
        end
    end
    write_csv(
        joinpath(directory, "sizes.csv"),
        ("class", "K", "system", "formulation", "variables", "equations", "construction_seconds"),
        sizes,
    )
    write_csv(
        joinpath(directory, "setup.csv"),
        ("class", "K", "system", "formulation", "build_seconds", "first_solve_seconds", "compile_seconds"),
        setup,
    )
    solves, stage_rows = Any[], Any[]
    function add_solve!(key, result; distance="", distance_fig3="")
        push!(solves, (key..., result.accepted, result.seconds, result.newton_steps, result.validation.rms_K₀, distance, distance_fig3))
        for (stage, s) in enumerate(result.stages)
            last_norm_K_ρ, last_norm_K₀ = isempty(s.history) ? (NaN, NaN) : (last(s.history).norm_K_ρ, last(s.history).norm_K₀)
            push!(stage_rows, (key..., stage, s.ρ, s.newton_steps, s.seconds, s.compile_seconds, s.status, last_norm_K_ρ, last_norm_K₀))
        end
    end
    for r in results
        add_solve!((r.class, r.num_levels, r.trial, "reduced", "interior_point"), r.reduced)
        for c in r.checks
            distance = c.category in (:same_solution, :different_solution) ? c.metrics.absolute : ""
            add_solve!((r.class, r.num_levels, r.trial, "complete", c.formulation), c.result; distance, distance_fig3=something(c.distance_fig3, ""))
        end
    end
    write_csv(
        joinpath(directory, "stages.csv"),
        ("class", "K", "trial", "system", "formulation", "stage", "rho", "newton_steps", "seconds", "compile_seconds", "status", "norm_K_ρ", "norm_K₀"),
        stage_rows,
    )
    write_csv(
        joinpath(directory, "solves.csv"),
        ("class", "K", "trial", "system", "formulation", "accepted", "seconds", "newton_steps", "norm_K₀", "distance", "distance_fig3"),
        solves,
    )
    directory
end

end # module MonteCarlo_RandomGOOP
