"""
Monte Carlo comparison of the reduced and complete KKT systems on random GOOPs, for the manuscript's
§6.1 (Table 2: system sizes and solve times) and §6.2 (Fig 2: convergence; Fig 3: ‖z_complete −
z_reduced‖₂). All three come from one run, so they describe the same problems.

For every objective class, number of preference levels K and sampled instance:
 1. Solve the REDUCED KKT system, the first of `reduced_formulations` ((5.2) as :interior_point, or
    :scholtes), with Scholtes down `rho_schedule`, from z = 0. It is accepted iff its unrelaxed KKT
    residual ‖K₀‖/√m is at most `tol`, at the best stage of its ρ chain. A further reduced formulation
    is solved likewise and compared with it.
 2. For each COMPLETE formulation, in each mode:
      :cold          solve the whole complete system with Scholtes from the same z = 0, using nothing
                     from the reduced solve (cold_initialization), and report ‖z_complete − z_reduced‖₂.
      :fixed_primal  keep the reduced solution fixed and solve for the other complete variables, as
                     in MonteCarlo_Intersection.
    A complete solve is accepted iff ‖K₀‖/√m ≤ `complete_tol`, at the best stage of its best try.

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
  exact, aggregate  the complete formulations, as in MonteCarlo_Intersection: exact keeps componentwise
                    complementarity and is solved once at ρ = 0; aggregate is the legacy slacked system,
                    solved down `complete_rho_schedule`.
  perturbed         the ρ-perturbed complete system, the complete counterpart of the reduced (5.2):
                    exact's equations with level-indexed slacks in the products and every product = ρ,
                    solved down `complete_rho_schedule`.
  R, K₀             residual norm at the stage's ρ, and at ρ = 0; reported as RMS, norm / √m.

    julia --project=experiments -e 'include("experiments/MonteCarlo_RandomGOOP.jl");
        MonteCarlo_RandomGOOP.run(overrides = (; num_instances = 10, num_levels = 2:3))'
"""
module MonteCarlo_RandomGOOP

using BlockArrays, LinearAlgebra, Random, Serialization, Printf
import ReducedGOOP
include(joinpath(@__DIR__, "MonteCarlo_Intersection.jl"))
# MCI.f is MonteCarlo_Intersection's f, reused unchanged.
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

Solver, formulation and comparison options, and whether each works with either Newton method (✗: an
error in validated_settings or ScholtesOptions):

    setting                value            what it is                                         :pdip  :scholtes
    newton_method          :pdip            Algorithm 5.1: step −J⁺K_ρ, backtracking that
                                            keeps s, γ > 0; one try per solve
                           :scholtes        projected, η-regularized Newton: bound rule,
                                            η schedule (eta_0, eta_tightening_rate) and
                                            complete-solve retries (retry_eta)
    linear_solver          :svd             dense pinv, the exact J⁺; cost grows as n³         ✓      ✓ (η unused)
                           :normal          sparse Cholesky of J Jᵀ + η²I; under :pdip,        ✓      ✓
                                            η = eta_pdip = 1e-6, the Tikhonov limit of J⁺
                           :klu_eta         sparse LU of [ηI J; Jᵀ −ηI]                        ✗      ✓
                           :klu_sqrt_eta    the same with √η                                   ✗      ✓
    reduced_formulations   :interior_point  the paper's (5.2)                                  ✓      ✓
                           :scholtes        the Scholtes-relaxed reduced system                ✗      ✓
    complete_formulations  :perturbed       ρ-perturbed complete system, down the ρ chain      ✓      ✓
                           :exact           complete system (2.12)–(2.17), one stage at ρ = 0  ✓¹     ✓
                           :aggregate       legacy aggregate system, down the ρ chain          ✓¹     ✓
    comparison_modes       :cold            complete solve from z = 0; z compared              ✓      ✓
                           :fixed_primal    complete solve with z held at the reduced z;       ✗      ✓
                                            needs :scholtes as the first reduced formulation

¹ Allowed but never run; at ρ = 0, :exact lies outside Theorem 5.1, which needs ρ > 0. Under :pdip,
eta_0, eta_tightening_rate and retry_eta are unused. The defaults are the paper's setting: :pdip, :svd,
(:interior_point,), (:perturbed,), (:cold,).
"""
default_settings() = (;
    # Monte Carlo
    num_instances=100,                           # instances per objective class and K
    rng_seed=123,                                # (class, K) samples from Xoshiro(rng_seed + 100·class index + K)
    objective_classes=(:quadratic, :nonquadratic), # names in objective_classes(settings)
    # K, the preference levels of every player, per class (levels_for; a range applies to every class).
    # Cosh code generation time grows fast with K: its systems at K = 5, 6 do not build in reasonable time.
    num_levels=(; quadratic=2:6, nonquadratic=2:4),
    # K whose complete systems are built, per class and, optionally, per formulation (levels_for); other K
    # solve only the reduced system. Cosh K = 4 builds :exact and :perturbed only: :exact's build and
    # compile took ≈ 40 min, and the cosh :aggregate build is several times slower. Quadratic K = 6 builds
    # no complete system: under linear_solver = :svd each :perturbed step there is a dense pinv of a
    # 3248 × 3248 Jacobian (8–15 s), ≈ 13–24 h for the 100 instances.
    complete_levels=(; quadratic=2:5, nonquadratic=(; exact=2:4, perturbed=2:4, aggregate=2:3)),
    complete_formulations=(:perturbed,),         # names in `builders` (run): :perturbed, :exact, and optionally the legacy :aggregate
    # :interior_point = the paper's (5.2), and/or :scholtes (build_systems). The first is each trial's
    # reduced solve, which every other solve is compared with (solve_reference_reduced).
    reduced_formulations=(:interior_point,),
    comparison_modes=(:cold,),                   # names in `modes` (run)               ; (:cold, :fixed_primal)
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
    # Solvers
    #! format: off
    rho_schedule          = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14], # reduced systems
    complete_rho_schedule = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14], # perturbed and aggregate complete systems; exact runs at ρ = 0
    #! format: on
    max_inner_iters=1000,                        # Scholtes Newton steps per ρ stage
    linear_solver=:svd,                          # ScholtesOptions.linear_solver
    # ScholtesOptions.method: :scholtes (projected bound rule, η schedule) or :pdip, the manuscript's
    # Algorithm 5.1 (step −J⁺K_ρ, backtracking that keeps s, γ > 0), the paper's setting. Under :pdip,
    # linear_solver = :svd takes J⁺ as the exact dense pinv, and :normal its Tikhonov limit at
    # η = ScholtesOptions.eta_pdip = 1e-6. Under :pdip the η settings below are unused, every solve gets
    # one try, and :scholtes cannot be a reduced formulation.
    newton_method=:pdip,
    eta_0=1e-2,                                  # initial regularization η: every ρ stage, and a complete solve's first try
    # η shrinks by 1 − e^(−rate) after each step whose gain ratio exceeds 0.75 (ScholtesOptions.tightening_rate).
    # At 1.2 (×0.70) η is still ≈ 1e-3 when the residual reaches 1e-14, so the local rate is linear; at
    # 0.01 (×0.01) η reaches eta_min within a few local steps, which are then Newton steps, and Fig 2's
    # stages converge quadratically (tuning: median order ≈ 2 for (5.2) and :perturbed).
    eta_tightening_rate=0.01,
    # One complete-solve retry per entry, with that η; all below eta_0, so no retry repeats the first try.
    retry_eta=[1e-3, 1e-4, 1e-5],
    # Acceptance: a solve is accepted iff its unrelaxed KKT residual ‖K₀‖ / √(number of equations) <= tol,
    # at the best stage of its ρ chain (and, for a complete solve, its best try).
    # One tolerance for every system: the :interior_point reduced system (5.2) tops out near 1e-12,
    # while :scholtes and :exact reach 1e-13. The chains end at ρ = 1e-14.
    tol=1e-11,                                   # reduced systems
    complete_tol=1e-11,                          # complete systems
    inner_ρ_tol=1e-2,                            # Scholtes ρ stage's inner stop: ‖R‖/√m < min(ρ, inner_ρ_tol)
    # Fig 2's stages are solved to full convergence instead, ‖R‖/√m < tight_inner_tol (scholtes_options). Not
    # every stage: below ρ ≈ 1e-6 the reduced relaxed residual levels off near 0.01·ρ, so a stage there would
    # only burn max_inner_iters.
    tight_rhos=[1.0, 0.1],
    tight_inner_tol=1e-13,
    # Complete-system start (cold_initialization; complete_initialization for :fixed_primal)
    positive_seed=1.0,                           # nonnegative variables
    positive_floor=1e-10,                        # floor for every nonnegative variable
    # Comparison, :cold mode
    # ‖z_complete − z_reduced‖₂ up to this counts as the same solution. Above the precision floor that
    # tol = complete_tol = 1e-8 leaves (distances up to ≈ 2e-6 in the step-1 runs).
    same_solution_tol=1e-5,
    output_dir=joinpath(dirname(@__DIR__), "output", "random_goop_study"),
)

"""
The K values under `name` in `setting`: a range applies to every name, and a named tuple gives each its
own. `num_levels` is keyed by class name; `complete_levels` by class name and then, optionally, by
formulation name, as levels_for(levels_for(complete_levels, class), formulation).
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
    # The first reduced formulation is each trial's reduced solve. The :fixed_primal mode starts from the
    # :scholtes variables (MCI.variable_mapping), so it needs :scholtes first.
    !isempty(settings.reduced_formulations) && all(in((:scholtes, :interior_point)), settings.reduced_formulations) ||
        error("reduced_formulations must be :scholtes and/or :interior_point")
    :fixed_primal in settings.comparison_modes && first(settings.reduced_formulations) !== :scholtes &&
        error("the :fixed_primal mode needs :scholtes as the first reduced formulation")
    # Algorithm 5.1 has no rule for the :scholtes reduced system's complementarity pairs.
    settings.newton_method in (:scholtes, :pdip) || error("newton_method must be :scholtes or :pdip")
    settings.newton_method === :pdip && :scholtes in settings.reduced_formulations &&
        error("newton_method = :pdip needs reduced_formulations without :scholtes")
    # The complete system needs two levels.
    for name in (:num_levels, :complete_levels)
        setting = settings[name]
        setting isa NamedTuple && !all(c -> haskey(setting, c), settings.objective_classes) &&
            error("$name must give a range for every class in objective_classes")
    end
    for class_name in settings.objective_classes
        by_formulation = levels_for(settings.complete_levels, class_name)
        by_formulation isa NamedTuple && !all(f -> haskey(by_formulation, f), settings.complete_formulations) &&
            error("complete_levels.$class_name must give a range for every formulation in complete_formulations")
    end
    settings.num_instances > 0 && all(c -> minimum(levels_for(settings.num_levels, c)) >= 2, settings.objective_classes) ||
        error("invalid num_instances or num_levels")
    0 <= settings.coupling <= 1 || error("coupling must lie in [0, 1]")
    settings.target_floor >= 0 || error("target_floor must be nonnegative")
    settings.tol > 0 && settings.complete_tol > 0 || error("tolerances must be positive")
    # Two rows fill the innermost image: two half-planes or disks in R², a slab or two intervals in R.
    settings.inequality_dimension == 2 || error("inequality_dimension must be 2")
    schedules = (settings.rho_schedule, settings.complete_rho_schedule)
    all(s -> !isempty(s) && all(ρ -> isfinite(ρ) && ρ > 0, s), schedules) || error("invalid relaxation schedule")
    settings
end

"""
One class's problem and systems at K = num_levels: the reduced KKT system (:scholtes, if in
reduced_formulations), the :interior_point formulation (the paper's (5.2), with its σ targets), then (for K in
complete_levels) for each complete formulation its KKT system, σ targets, fixed-primal system and
variable mapping. Prints the system sizes, and returns everything with its results.jls metadata
(sizes, construction times).
"""
function build_systems(class, num_levels, builders, settings)
    (; problem, layout) = get_setup(class, num_levels, settings)
    label = "$(class.name), K = $num_levels"
    reduced_kkt, reduced_seconds = nothing, NaN
    if :scholtes in settings.reduced_formulations
        println("Building reduced KKT system for $label...")
        flush(stdout)
        reduced_construction = @timed ReducedGOOP.generate_slacked_reduced_kkt_system(
            problem;
            complementarity=:scholtes,
            codegen=:fast_differentiation,
        )
        reduced_kkt, reduced_seconds = reduced_construction.value, reduced_construction.time
    end
    # The paper's (5.2): the :interior_point reduced system, handed to Scholtes() in the complete systems'
    # container, so it is solved like a complete formulation (solve_scholtes_projected) down the reduced
    # rho_schedule. Its σ start is g(0) per level (interior_point_slack_targets).
    reduced_formulations = map(filter(!=(:scholtes), collect(settings.reduced_formulations))) do name
        println("Building :$name reduced KKT system for $label...")
        flush(stdout)
        construction = @timed ReducedGOOP.generate_interior_point_scholtes_kkt_system(problem)
        complete_kkt = construction.value
        (; name, complete_kkt, slack_targets=interior_point_slack_targets(problem, complete_kkt), construction_seconds=construction.time)
    end
    class_levels = levels_for(settings.complete_levels, class.name)
    builders = filter(b -> num_levels in levels_for(class_levels, b.name), builders)
    formulations = map(builders) do builder
        println("Building :$(builder.name) complete KKT system for $label...")
        flush(stdout)
        complete_construction = @timed builder.build(problem)
        complete_kkt = complete_construction.value
        slack_targets = MCI.compile_slack_targets(complete_kkt)
        fixed = MCI.fixed_primal_formulation(complete_kkt)
        mapping = isnothing(reduced_kkt) ? nothing : MCI.variable_mapping(reduced_kkt, complete_kkt)
        # The exact and perturbed systems are square; the fixed-primal Jacobian keeps every row but
        # drops the physical columns.
        builder.name in (:exact, :perturbed) && @assert complete_kkt.n == complete_kkt.n_nc
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
        class=class.name,
        num_levels,
        levels=[class.levels(k) for k in 1:num_levels],
        class.constraint_type,
        # Reduced KKT system (:scholtes; nothing if not built): variables y, equations F_nc and
        # complementarity pairs; then the Scholtes system the solver works on: rows of R(w; ρ), columns
        # of w = [y; s; u].
        reduced=isnothing(reduced_kkt) ? nothing : (;
            variables=reduced_kkt.n,
            equations=reduced_kkt.n_nc,
            complementarity=reduced_kkt.n_comp,
            rows=reduced_kkt.n_nc+2reduced_kkt.n_comp,
            columns=reduced_kkt.n+reduced_kkt.n_comp+ReducedGOOP._n_u(reduced_kkt),
            construction_seconds=reduced_seconds,
        ),
        complete=[(; p.name, p.fixed.dimensions, p.mapping, p.construction_seconds) for p in formulations],
        # Further reduced formulations (:interior_point): variables y and equations, no complementarity pairs.
        reduced_formulations=[
            (; p.name, variables=p.complete_kkt.n, equations=p.complete_kkt.n_nc, p.construction_seconds) for
            p in reduced_formulations
        ],
    )
    println("KKT system sizes, $label:")
    # MCI.size_lines lists the :scholtes reduced system, then the complete ones.
    isnothing(metadata.reduced) || foreach(line -> println("  ", line), MCI.size_lines(metadata))
    for p in metadata.reduced_formulations
        println("  reduced :", rpad(p.name, 16), p.variables, " variables · ", p.equations, " equations")
    end
    for c in metadata.complete
        isnothing(metadata.reduced) &&
            println("  complete :", rpad(c.name, 15), c.dimensions.full_columns, " variables · ", c.dimensions.rows, " equations")
    end
    (; class, num_levels, problem, layout, reduced_kkt, reduced_formulations, formulations, metadata)
end

"""
Start of the σ slacks of the :interior_point reduced system: each player's inequality values g_i(z), once
per level, in the order that system holds them (players in turn, levels outermost first).
"""
function interior_point_slack_targets(problem, kkt)
    names = string.(kkt.vars[kkt.interior_point_slack_dims])
    order = [(i, k) for i in 1:problem.num_players for k in eachindex(problem.preferences[i]) for _ in 1:problem.inequality_dims[i]]
    @assert all(startswith(name, "σ_$(i)_$(k)") for (name, (i, k)) in zip(names, order)) "unexpected σ order"
    function (y, θ, ρ)
        z = BlockArray(y[kkt.primal_dims], problem.primal_dims)
        θ_blocks = BlockArray(θ, problem.parameter_dims)
        reduce(vcat, (repeat(problem.inequality_constraints[i](z, θ_blocks), length(problem.preferences[i])) for i in 1:problem.num_players); init=Float64[])
    end
end

"""
Untimed JIT warm-up on `instance`, so measured trials don't pay first-call compilation: one solve of
every reduced system and one try of every complete solve, results discarded.
"""
function jit_warmup(systems, modes, instance, settings)
    (; class, num_levels, problem, formulations) = systems
    (; θ, initial_guess) = instance_inputs(instance, settings)
    print("Warm-up for $(class.name), K = $num_levels (untimed; compiles every solver path; results discarded)... ")
    flush(stdout)
    # One-time cost of each system's first solve: its wall time and the part spent compiling (setup.csv).
    timings = NamedTuple[]
    seconds = @elapsed begin
        context = "warm-up, $(class.name), K = $num_levels"
        timed = @timed solve_reference_reduced(systems, θ, initial_guess, settings; context, io=devnull)
        reduced = timed.value
        push!(timings, (; system=:reduced, formulation=first(settings.reduced_formulations), first_solve_seconds=timed.time, compile_seconds=timed.compile_time+timed.recompile_time))
        # A further reduced formulation is solved like a complete one, :cold only.
        jobs = [[(f, (; name=:cold), :reduced) for f in further_reduced(systems, settings)]; [(f, m, :complete) for f in formulations for m in modes]]
        for (formulation, mode, system) in jobs
            (; start, physical) = projected_start(formulation, mode, reduced, initial_guess, θ, settings)
            timed = @timed solve_scholtes_projected(
                formulation,
                mode,
                problem,
                θ,
                physical,
                start,
                settings;
                retry_values=Float64[],
                context,
                io=devnull,
            )
            push!(timings, (; system, formulation=formulation.name, first_solve_seconds=timed.time, compile_seconds=timed.compile_time+timed.recompile_time))
        end
    end
    println("done in $(round(seconds; digits=1)) s")
    timings
end

"""
One trial's reduced solve: print the trial banner, solve from the instance's initial guess
(instance_inputs), print the verdict, and return the trial's record. Its `checks` start empty; `run`
fills them.
"""
function solve_reduced_kkt(systems, trial, instance, num_checks, settings)
    (; class, num_levels) = systems
    println("\n══ Trial $trial/$(settings.num_instances) · $(class.name), K = $num_levels ══")
    excluded = [p.excluded for p in instance.players]
    println("  Players whose inequalities exclude their innermost target: ", any(excluded) ? join(findall(excluded), ", ") : "none")
    flush(stdout)
    (; θ, initial_guess) = instance_inputs(instance, settings)
    context = trial_context(trial, class.name, num_levels)
    reduced = solve_reference_reduced(systems, θ, initial_guess, settings; context)
    println(MCI.reduced_verdict(reduced, settings), num_checks == 0 ? "" : " Running its $num_checks other solves.")
    flush(stdout)
    reduced_formulation = first(settings.reduced_formulations)
    (; trial, class=class.name, num_levels, θ, excluded, reduced_formulation, reduced, checks=Any[])
end

"""
A trial's reduced solve, which every other solve is compared with: the first of reduced_formulations.
:scholtes is solved by solve_scholtes_reduced; :interior_point ((5.2)) by solve_scholtes_projected from
the :cold start, and returned with the fields of a :scholtes result that the checks and exports read
(output.z, stages, best_stage).
"""
function solve_reference_reduced(systems, θ, initial_guess, settings; context, io=stdout)
    (; problem, reduced_kkt, reduced_formulations) = systems
    first(settings.reduced_formulations) === :scholtes &&
        return solve_scholtes_reduced(reduced_kkt, problem, θ, initial_guess, settings; context, io)
    formulation, mode = only(reduced_formulations), (; name=:cold)
    (; start, physical) = projected_start(formulation, mode, nothing, initial_guess, θ, settings)
    result = solve_scholtes_projected(formulation, mode, problem, θ, physical, start, settings; context, io)
    final = result.attempts[result.best_attempt]
    (; result..., output=(; z=result.physical, w=result.solution), final.stages, final.best_stage)
end

"The reduced formulations other than each trial's reduced solve: solved like complete ones and compared with it."
further_reduced(systems, settings) = filter(f -> f.name !== first(settings.reduced_formulations), systems.reduced_formulations)

"Label naming a trial in the solver tables."
trial_context(trial, class_name, num_levels) = "trial $trial, $class_name, K = $num_levels"

"""
A complete solve's start and the z it holds fixed. :cold starts at the instance's initial guess with
nothing from the reduced solve (cold_initialization) and holds nothing fixed; :fixed_primal starts from
the reduced solution (complete_initialization) and holds its z fixed.
"""
function projected_start(formulation, mode, reduced, initial_guess, θ, settings)
    mode.name === :cold && return (; start=cold_initialization(formulation, initial_guess, θ, settings), physical=nothing)
    (; start=MCI.complete_initialization(formulation, reduced.output, θ, settings), physical=reduced.output.z)
end

"""
One complete solve of a trial in `mode`, with its verdict printed. :cold uses only the trial's initial
guess, so it runs whether or not the reduced solve was accepted; :fixed_primal needs an accepted reduced
solution, so a rejected one skips it.
"""
function check_projected_kkt(record, systems, instance, formulation, mode, settings)
    (; θ, reduced) = record
    mode.name === :fixed_primal && !reduced.accepted && return (;
        formulation=formulation.name,
        mode=mode.name,
        category=:reduced_failed,
        reason="Best reduced stage did not pass acceptance",
    )
    (; initial_guess) = instance_inputs(instance, settings)
    (; start, physical) = projected_start(formulation, mode, reduced, initial_guess, θ, settings)
    complete = try
        solve_scholtes_projected(
            formulation,
            mode,
            systems.problem,
            θ,
            physical,
            start,
            settings;
            context=trial_context(record.trial, record.class, record.num_levels),
        )
    catch err
        validation = (;
            relaxed_rms=NaN,
            unrelaxed_rms=NaN,
            unrelaxed_min=NaN,
            unrelaxed_max=NaN,
            max_abs_residual=NaN,
            sign_violation=NaN,
            primal=(; equality_max=NaN, inequality_violation=NaN),
            accepted=false,
        )
        (;
            accepted=false,
            attempts=Any[],
            validation,
            physical=start[formulation.complete_kkt.primal_dims],
            solution=start,
            status=:exception,
            reason=:exception,
            error=sprint(showerror, err, catch_backtrace()),
            iterations=0,
            seconds=0.0,
            compile_seconds=0.0,
        )
    end
    metrics = solution_distance(reduced.output.z, complete.physical)
    # :numerical_failure        complete solve not accepted (no conclusion either way)
    # :fixed_primal_compatible  fixed-primal solve accepted
    # :reduced_rejected         :cold accepted, but there is no accepted reduced solution to compare
    # :same_solution            :cold accepted; ‖z_complete − z_reduced‖₂ <= same_solution_tol
    # :different_solution       :cold accepted; the solutions differ by more
    # (:reduced_failed, above, marks :fixed_primal checks skipped because the reduced solve failed.)
    category =
        !complete.accepted ? :numerical_failure :
        mode.name === :fixed_primal ? :fixed_primal_compatible :
        !reduced.accepted ? :reduced_rejected :
        metrics.absolute <= settings.same_solution_tol ? :same_solution : :different_solution
    println(projected_verdict(formulation, mode, complete, metrics, category, settings))
    (;
        formulation=formulation.name,
        mode=mode.name,
        result=complete,
        metrics,
        category,
    )
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
    modes = filter(m -> m.name in settings.comparison_modes, [(; name=:cold), (; name=:fixed_primal)])
    builders = [
        (; name=:aggregate, build=ReducedGOOP.generate_scholtes_complete_kkt_system),
        (; name=:exact, build=ReducedGOOP.generated_exact_complete_kkt_system),
        (; name=:perturbed, build=ReducedGOOP.generated_perturbed_complete_kkt_system),
    ]
    builders = filter(b -> b.name in settings.complete_formulations, builders)
    total, done = sum(c -> length(levels_for(settings.num_levels, c.name)), classes) * settings.num_instances, 0
    println(
        "Random GOOP study: $(settings.num_instances) instances per class and K, K ∈ $(settings.num_levels), ",
        "N = $(settings.num_players), n = $(settings.primal_dimension). Output: $directory",
    )
    println(MCI.tolerance_line(settings))
    for class in classes, num_levels in levels_for(settings.num_levels, class.name)
        class_instances =sample_instances(instance_rng(class, num_levels, settings), class, num_levels, settings)
        push!(instances, (; class=class.name, num_levels, instances=class_instances))
        systems = build_systems(class, num_levels, builders, settings)
        push!(metadata, systems.metadata)
        save()
        warmup = jit_warmup(systems, modes, first(class_instances), settings)
        # The warm-up's one-time costs join this system's metadata (setup.csv).
        metadata[end] = merge(metadata[end], (; warmup))
        num_checks = length(further_reduced(systems, settings)) + length(systems.formulations) * length(modes)
        # Warm-up results are discarded; every measured trial starts afresh.
        for (trial, instance) in enumerate(class_instances)
            record = solve_reduced_kkt(systems, trial, instance, num_checks, settings)
            # `record.checks` is filled in place below. The record already in `results` sees each
            # push, so every save stores the progress so far.
            push!(results, record)
            # A further reduced formulation ((5.2) when :scholtes is the reduced solve), from the same cold
            # start as the complete solves and compared with the reduced solution.
            for formulation in further_reduced(systems, settings)
                push!(record.checks, check_projected_kkt(record, systems, instance, formulation, (; name=:cold), settings))
            end
            # formulations: :aggregate, :exact, :perturbed; modes: :cold, :fixed_primal
            for formulation in systems.formulations, mode in modes
                push!(record.checks, check_projected_kkt(record, systems, instance, formulation, mode, settings))
            end
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
Starting point for a :cold complete solve: the steps of MonteCarlo_Intersection's complete_initialization
without its step 3, so nothing comes from the reduced solve.
 1. physical coordinates <- the instance's initial guess (the reduced solve's start);
 2. every nonnegative coordinate <- positive_seed, floored at positive_floor;
 3. every unsigned multiplier stays 0;
 4. auxiliary slacks σ <- their defining expressions (compile_slack_targets) at the first ρ of the
    solve (0 for exact, first(complete_rho_schedule) for perturbed and aggregate), floored as in 2.
"""
function cold_initialization(formulation, initial_guess, θ, settings)
    (; complete_kkt) = formulation
    y = zeros(complete_kkt.n)
    y[complete_kkt.primal_dims] .= initial_guess
    y[complete_kkt.nonnegative_dims] .= max(settings.positive_seed, settings.positive_floor)
    ρ = formulation.name === :exact ? 0.0 : first(settings.complete_rho_schedule)
    y[complete_kkt.interior_point_slack_dims] .=
        max.(formulation.slack_targets(y, θ, ρ), settings.positive_floor)
    @assert y[complete_kkt.primal_dims] == initial_guess
    y
end

"Header above each solve's ρ-stage table; `label` names the solve and the η₀ this try uses."
function print_kkt_header(label; io=stdout)
    println(io, "\n[solver exit] $label")
    println(io, "Residuals are scaled, ‖·‖/√m; K₀ min/max are its smallest and largest unscaled rows; min g is the smallest inequality value.")
    @printf(
        io,
        "%-9s %-7s %6s %8s %12s %12s %10s %10s %9s %10s  %s\n",
        "ρ",
        "status",
        "iters",
        "time_s",
        "‖R‖/√m",
        "‖K₀‖/√m",
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
    iterations,
    seconds,
    relaxed_rms,
    unrelaxed_rms,
    unrelaxed_min,
    unrelaxed_max,
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
        iterations,
        seconds,
        relaxed_rms,
        unrelaxed_rms,
        unrelaxed_min,
        unrelaxed_max,
        norm(physical),
        worst,
        stop
    )
    flush(io)
end

"""
Scholtes options of one ρ stage: MonteCarlo_Intersection's scholtes_options, with η's tightening rate
from `eta_tightening_rate`, the Newton method from `newton_method`, and the stages in `tight_rhos`
(Fig 2's) stopping at ‖R‖/√m < tight_inner_tol instead of min(ρ, inner_ρ_tol), so their Newton
iterations show full convergence, as in the manuscript's Fig 2.
"""
function scholtes_options(settings; ρ, tol, eta, residual_scale)
    options = MCI.scholtes_options(settings; ρ, tol, eta, residual_scale)
    options = ReducedGOOP._with_options(options; tightening_rate=settings.eta_tightening_rate, method=settings.newton_method)
    any(r -> isapprox(r, ρ; rtol=1e-9), settings.tight_rhos) || return options
    ReducedGOOP._with_options(options; tol_inner=settings.tight_inner_tol * residual_scale)
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
    # R(w; ρ), b(y) and K₀ at each stage's final iterate, for the table and the validation.
    r, b, k₀_rows = zeros(m), zeros(reduced_kkt.n_comp), zeros(m)
    stage_initial_guess = initial_guess
    previous_w = nothing
    stage_preference_slacks = nothing
    stages = Any[]
    best = nothing
    print_kkt_header("$context | reduced KKT | η₀ = $(MCI.compact(settings.eta_0))"; io)
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
        (; history, trace) = MCI.history_recorder(residuals, ρ)
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
        push!(
            stages,
            MCI.stage_record(; ρ, parameter=:rho_reduced, timed, history, initial=copy(w₀), final=copy(output.w)),
        )
        ReducedGOOP._resid!(residuals, r, b, output.w, ρ)
        ReducedGOOP._K₀!(k₀_rows, residuals, r, output.w, ρ)
        print_kkt_stage(;
            ρ,
            status=output.status,
            iterations=output.iters,
            seconds=last(stages).seconds,
            relaxed_rms=output.relaxed_residual/residual_scale,
            unrelaxed_rms=output.residual/residual_scale,
            unrelaxed_min=minimum(k₀_rows),
            unrelaxed_max=maximum(k₀_rows),
            physical=output.z,
            problem,
            θ,
            stop=MCI.Intersection.solver_stop_reason(output, options; scale=residual_scale),
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
    ReducedGOOP._resid!(residuals, r, b, output.w, ρ)
    ReducedGOOP._K₀!(k₀_rows, residuals, r, output.w, ρ)
    validation = MCI.validate_solution(
        problem,
        θ,
        output.z;
        r,
        k₀=k₀_rows,
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

function solve_scholtes_projected(
    formulation,
    mode,
    problem,
    θ,
    physical,
    initial,
    settings;
    # :pdip ignores η₀, so a retry would repeat the first try.
    retry_values=settings.newton_method === :pdip ? Float64[] : settings.retry_eta,
    context,
    io=stdout,
)
    (; complete_kkt) = formulation
    # exact is unrelaxed: one stage at ρ = 0. perturbed and aggregate run down complete_rho_schedule
    # (ρ plays the legacy ε). Each retry restarts from `initial` with the next η. The :interior_point reduced
    # system (5.2) is solved here too, with the reduced system's schedule and tolerance (ρ plays its ϵ).
    reduced_system = formulation.name === :interior_point
    schedule = formulation.name === :exact ? [0.0] : reduced_system ? settings.rho_schedule : settings.complete_rho_schedule
    tol = reduced_system ? settings.tol : settings.complete_tol
    # :cold solves the whole complete system. :fixed_primal solves the fixed-primal system, whose
    # unknowns are the free columns and whose parameters are [θ; x], so x cannot move.
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
        method = settings.newton_method === :pdip ? "Algorithm 5.1 (:pdip)" : "η₀ = $(MCI.compact(eta))"
        print_kkt_header(
            "$context | $(reduced_system ? "reduced" : "complete") :$(formulation.name), :$(mode.name) mode | $try_label: $method";
            io,
        )
        for ρ in schedule
            stage_initial = copy(y)
            # The 1e-12 floor is defensive: exact ignores ρ (its residual is evaluated at ρ = 0)
            # and complete_rho_schedule stays above it, so it changes nothing today; for exact
            # it only keeps ρ and tol_inner positive.
            options = scholtes_options(settings; ρ=max(ρ, 1e-12), tol, eta, residual_scale)
            # The unrelaxed :exact stage has no ρ to follow: it runs to the acceptance tolerance itself.
            formulation.name === :exact && (options = ReducedGOOP._with_options(options; tol_inner=tol * residual_scale))
            (; history, trace) = MCI.history_recorder(residuals, ρ)
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
            parameter = formulation.name === :exact ? :unrelaxed : reduced_system ? :epsilon_reduced : :epsilon_complete
            push!(stages, MCI.stage_record(; ρ, parameter, timed, history, initial=stage_initial, final=copy(y)))
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
                unrelaxed_min=minimum(r0),
                unrelaxed_max=maximum(r0),
                physical=y[complete_kkt.primal_dims],
                problem,
                θ,
                stop=MCI.Intersection.solver_stop_reason(output, options; scale=residual_scale),
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
        validation = MCI.validate_solution(
            problem,
            θ,
            y[complete_kkt.primal_dims];
            r,
            k₀=r0,
            relaxed_rms=norm(r)/residual_scale,
            unrelaxed_rms=norm(r0)/residual_scale,
            sign_violation=max(0.0, -minimum(y[complete_kkt.nonnegative_dims]; init=Inf)),
            tol,
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

"‖z_complete − z_reduced‖₂, absolute (Fig 3) and relative to ‖z_reduced‖₂, and the largest entry of the difference."
function solution_distance(reduced, complete)
    δ = complete - reduced
    (; absolute=norm(δ), relative=norm(δ)/max(norm(reduced), 1e-12), max_abs=maximum(abs, δ))
end

"One-line outcome of a complete solve, printed under its stage tables."
function projected_verdict(formulation, mode, result, metrics, category, settings)
    name = ":$(formulation.name), :$(mode.name) mode"
    haskey(result, :error) && return "→ $name FAILED with an exception: $(first(split(result.error, '\n')))"
    final = result.attempts[result.best_attempt]
    tries = length(result.attempts) == 1 ? "1 try" : "$(length(result.attempts)) tries"
    stage = "ρ = $(MCI.compact(final.stages[final.best_stage].ρ))"
    reduced_system = formulation.name === :interior_point
    k₀, tol = MCI.compact(result.validation.unrelaxed_rms), MCI.compact(reduced_system ? settings.tol : settings.complete_tol)
    range = MCI.k₀_range(result.validation)
    result.accepted ||
        return "→ $name REJECTED after $tries: best ‖K₀‖/√m = $k₀ (try $(result.best_attempt), $stage) > $tol; $range"
    compared =
        mode.name === :fixed_primal ? "reduced z held fixed" :
        category === :reduced_rejected ? "no accepted reduced solution to compare" :
        "‖z$(reduced_system ? "" : "_complete") − z_reduced$(reduced_system ? "(:scholtes)" : "")‖₂ = $(MCI.compact(metrics.absolute)) ($(replace(string(category), '_' => ' ')))"
    "→ $name accepted after $tries at $stage: ‖K₀‖/√m = $k₀ ≤ $tol; $range; $compared"
end

"Progress bar on its own line: other output interleaves with it, so it is not redrawn in place."
function print_progress(done, total; width=24)
    filled = floor(Int, width * done / total)
    println("[", repeat("=", filled), repeat(" ", width - filled), "] $done/$total trials")
    flush(stdout)
end

"""
Pass counts, one line per class, K, complete formulation and mode, then a key: printed at the end of a
run.
"""
function pass_count_lines(results, settings)
    lines = [
        @sprintf(
            "%-13s %2s %-10s %-14s %8s %9s %9s %9s %11s",
            "class",
            "K",
            "complete",
            "mode",
            "reduced",
            "complete",
            "both",
            "same z",
            "max ‖Δz‖₂",
        ),
    ]
    for class_name in settings.objective_classes, num_levels in levels_for(settings.num_levels, class_name)
        rows = filter(r -> r.class === class_name && r.num_levels == num_levels, results)
        isempty(rows) && continue
        n = length(rows)
        num_reduced = count(r -> r.reduced.accepted, rows)
        checks = [c for r in rows for c in r.checks]
        if isempty(checks)
            push!(lines, @sprintf("%-13s %2d %-10s %-14s %8s", class_name, num_levels, "—", "—", "$num_reduced/$n"))
            continue
        end
        # The further reduced formulations (:cold only) first, then the complete ones.
        others = [collect(settings.reduced_formulations)[2:end]; collect(settings.complete_formulations)]
        for formulation_name in others, mode_name in settings.comparison_modes
            matching = filter(c -> c.formulation === formulation_name && c.mode === mode_name, checks)
            isempty(matching) && continue  # this formulation is not built at this K (complete_levels)
            num_complete = count(c -> haskey(c, :result) && c.result.accepted, matching)
            compared = filter(c -> c.category in (:same_solution, :different_solution, :fixed_primal_compatible), matching)
            num_same = count(c -> c.category in (:same_solution, :fixed_primal_compatible), compared)
            distances = [c.metrics.absolute for c in compared]
            push!(
                lines,
                @sprintf(
                    "%-13s %2d %-10s %-14s %8s %9s %9s %9s %11s",
                    class_name,
                    num_levels,
                    ":$formulation_name",
                    ":$mode_name",
                    "$num_reduced/$n",
                    "$num_complete/$n",
                    "$(length(compared))/$n",
                    "$num_same/$(length(compared))",
                    isempty(distances) ? "—" : MCI.compact(maximum(distances)),
                ),
            )
        end
    end
    push!(
        lines,
        "",
        "reduced, complete: accepted / trials; both: trials where both were accepted; same z: of those,",
        "‖z_complete − z_reduced‖₂ ≤ same_solution_tol = $(MCI.compact(settings.same_solution_tol)) (a fixed-primal pass holds z fixed).",
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
Figures and tables of a run (experiments/RandomGOOP_plotting.py: Fig 2, Fig 3, Table 2 and the timing
comparison), written to <directory>/figures/ from the CSV files. Runs the Python script with uv. A failure
only warns, since results.jls and the CSV files are already saved; rerun with plot_results(directory).
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
  sizes.csv     one row per (class, K, system): variables, equations, complementarity pairs, build time
  setup.csv     one row per (class, K, system): build time, and the warm-up's first-solve time and the
                part of it spent compiling (jit_warmup)
  solves.csv    one row per solve (system reduced: formulation scholtes or interior_point; complete: exact,
                perturbed or aggregate): accepted, solve time (no compilation), Newton steps, ‖K₀‖/√m, distance to
                the trial's reduced z, and for complete solves the distance to the :interior_point one
  stages.csv    one row per ρ stage of every solve (each try of a complete solve): Newton steps, solve
                time (no compilation), compile time, status, and ‖R‖, ‖K₀‖ at the stage's last iterate
  traces.csv    one row per Newton iterate of every ρ stage (a complete solve's kept try): ‖R‖, ‖K₀‖
  settings.csv  the scalar settings (dimensions for the size formulas, tolerances, coupling)
"""
function export_csv(directory)
    (; settings, metadata, results) = deserialize(joinpath(directory, "results.jls"))
    scalars = [(name, value) for (name, value) in pairs(settings) if value isa Union{Real,Symbol}]
    write_csv(joinpath(directory, "settings.csv"), ("name", "value"), scalars)
    sizes = Any[]
    for m in metadata
        isnothing(m.reduced) ||
            push!(sizes, (m.class, m.num_levels, "reduced", "scholtes", m.reduced.variables, m.reduced.equations, m.reduced.complementarity, m.reduced.construction_seconds))
        for p in get(m, :reduced_formulations, NamedTuple[])
            push!(sizes, (m.class, m.num_levels, "reduced", p.name, p.variables, p.equations, 0, p.construction_seconds))
        end
        for c in m.complete
            push!(sizes, (m.class, m.num_levels, "complete", c.name, c.dimensions.full_columns, c.dimensions.rows, 0, c.construction_seconds))
        end
    end
    write_csv(
        joinpath(directory, "sizes.csv"),
        ("class", "K", "system", "formulation", "variables", "equations", "complementarity", "construction_seconds"),
        sizes,
    )
    setup = Any[]
    for m in metadata, w in get(m, :warmup, NamedTuple[])
        # :none is the :scholtes reduced system in runs saved before reduced_formulations existed.
        formulation = w.formulation === :none ? :scholtes : w.formulation
        build =
            w.system === :complete ? only(c.construction_seconds for c in m.complete if c.name === formulation) :
            formulation === :scholtes ? m.reduced.construction_seconds :
            only(p.construction_seconds for p in m.reduced_formulations if p.name === formulation)
        push!(setup, (m.class, m.num_levels, w.system, formulation, build, w.first_solve_seconds, w.compile_seconds))
    end
    write_csv(
        joinpath(directory, "setup.csv"),
        ("class", "K", "system", "formulation", "build_seconds", "first_solve_seconds", "compile_seconds"),
        setup,
    )
    solves, stage_rows, traces = Any[], Any[], Any[]
    function add_traces!(key, stages)
        for (stage, s) in enumerate(stages), (iteration, h) in enumerate(s.history)
            push!(traces, (key..., stage, s.ρ, iteration, h.raw, h.unrelaxed))
        end
    end
    function add_stages!(key, attempt, stages)
        for (stage, s) in enumerate(stages)
            last_raw, last_unrelaxed = isempty(s.history) ? (NaN, NaN) : (last(s.history).raw, last(s.history).unrelaxed)
            push!(stage_rows, (key..., attempt, stage, s.ρ, s.iterations, s.seconds, s.compile_seconds, s.status, last_raw, last_unrelaxed))
        end
    end
    for r in results
        key = (r.class, r.num_levels, r.trial)
        # The trial's reduced solve (:scholtes in runs saved before reduced_formulation existed). A (5.2)
        # reduced solve keeps every try, like a complete one.
        reference = get(r, :reduced_formulation, :scholtes)
        push!(solves, (key..., "reduced", reference, "", r.reduced.accepted, r.reduced.seconds, r.reduced.iterations, r.reduced.validation.unrelaxed_rms, "", ""))
        add_traces!((key..., "reduced", reference), r.reduced.stages)
        if haskey(r.reduced, :attempts)
            foreach(a -> add_stages!((key..., "reduced", reference), a.attempt, a.stages), r.reduced.attempts)
        else
            add_stages!((key..., "reduced", reference), 1, r.reduced.stages)
        end
        # The accepted :interior_point reduced solution, if any: complete solves are also measured against it.
        ip = findfirst(c -> c.formulation === :interior_point && haskey(c, :result) && c.result.accepted, r.checks)
        for c in r.checks
            haskey(c, :result) || continue
            system = c.formulation === :interior_point ? "reduced" : "complete"
            distance = c.category in (:same_solution, :different_solution) ? c.metrics.absolute : ""
            # When (5.2) is the trial's reduced solve, `distance` already measures against it.
            distance_ip =
                system != "complete" || !c.result.accepted ? "" :
                reference === :interior_point ? distance :
                isnothing(ip) ? "" : norm(c.result.physical - r.checks[ip].result.physical)
            push!(solves, (key..., system, c.formulation, c.mode, c.result.accepted, c.result.seconds, c.result.iterations, c.result.validation.unrelaxed_rms, distance, distance_ip))
            isempty(c.result.attempts) || add_traces!((key..., system, c.formulation), c.result.attempts[c.result.best_attempt].stages)
            foreach(a -> add_stages!((key..., system, c.formulation), a.attempt, a.stages), c.result.attempts)
        end
    end
    write_csv(
        joinpath(directory, "stages.csv"),
        ("class", "K", "trial", "system", "formulation", "try", "stage", "rho", "iterations", "seconds", "compile_seconds", "status", "raw", "unrelaxed"),
        stage_rows,
    )
    write_csv(
        joinpath(directory, "solves.csv"),
        ("class", "K", "trial", "system", "formulation", "mode", "accepted", "seconds", "iterations", "unrelaxed_rms", "distance", "distance_interior_point"),
        solves,
    )
    write_csv(
        joinpath(directory, "traces.csv"),
        ("class", "K", "trial", "system", "formulation", "stage", "rho", "iteration", "raw", "unrelaxed"),
        traces,
    )
    directory
end

end # module MonteCarlo_RandomGOOP
