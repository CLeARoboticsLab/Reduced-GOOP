# The explicit three-player, three-level game, plus a driver that solves it and
# prints the result.
#
# This is a report script, not part of `runtests.jl` — running it *is* the test:
#
#     julia --project=experiments test/explicit_three_player_solve.jl
#
# The experiments environment supplies CairoMakie for the history PDFs and
# NonlinearSolve for the full hard-KKT warm-start comparison.
#
# Edit `build_explicit_three_player_problem` below and rerun; every number in
# the report is recomputed from the problem definition, so nothing has to be
# kept in sync by hand.
#
# `benchmark_problems.jl` is included only for its shared helpers
# (`default_interior_point_options`, `reduced_kkt_system`, and friends). The
# problem itself is defined here and depends on nothing in that file.

using Printf: @printf, @sprintf, Format, format

using BlockArrays: Block, BlockArray
using LinearAlgebra: ColumnNorm, norm
using NonlinearSolve
using ReducedGOOP
using SparseArrays: nonzeros, nzrange, rowvals

const CAIRO_MAKIE_AVAILABLE = !isnothing(Base.find_package("CairoMakie"))
if CAIRO_MAKIE_AVAILABLE
    @eval import CairoMakie
end

@isdefined(default_interior_point_options) ||
    include(joinpath(@__DIR__, "benchmark_problems.jl"))

# `innermost_preference_inequality_goop` lives in the comparison script. Setting
# the guard flag first suppresses that file's `@testset`, so including it here
# only imports the helper instead of running a second suite.
if !@isdefined(innermost_preference_inequality_goop)
    const RUN_QUASI_VS_REDUCED_TESTSET = false
    # `includet` keeps Revise tracking in a REPL session; plain `include` covers
    # the documented standalone invocation, where Revise is not loaded.
    (@isdefined(includet) ? includet : include)(
        joinpath(@__DIR__, "quasi_vs_reduced_preference_inequality.jl"),
    )
end

const LEVEL_NAMES = ("outermost", "middle", "innermost")

#=
Explicit three-player, three-level benchmark
--------------------------------------------

`build_benchmark_problem(; num_players = 3, levels = 3, kind = :quadratic)`
assembles this same game out of closure factories, which makes it awkward to
change one objective or one constraint by hand. The builder below writes every
objective, every constraint row, and every constant out in full so the game can
be edited directly.

Layout, with x_{p,j} = `x[Block(p)][j]` and n = 5 coordinates per player.

Preferences are stored `[outermost, ..., innermost]`, so for each player p:

  level 1 (outermost, lowest priority):  (x_{p,5} - t_{p,5})^2
  level 2 (middle):                      (x_{p,3} - t_{p,3})^2 + (x_{p,4} - t_{p,4})^2
  level 3 (innermost, highest priority): (x_{p,1} - t_{p,1})^2 + (x_{p,2} - t_{p,2})^2

`t_p` is player p's *raw* target — where the objective would sit with no
constraints:

  t_1 = [1.00, 0.90, 0.15, 1.20, 0.80]
  t_2 = [1.05, 1.00, 0.20, 1.30, 0.90]
  t_3 = [1.10, 1.10, 0.25, 1.40, 1.00]

No expected primal solution is stored in this problem definition. The report
solves the hard-inequality formulation first and uses that returned primal
vector, as computed, as the reference for the two preference formulations.

The coupled resource `c_{p,j}(x) = x_{p,j} + 0.25 * sum(x_{k,j} for k != p)`
gives every player a feasible set that depends on all players' variables — a
true generalized Nash game. The constants below are `c_{p,j}(x*)`, offset by
±1 on the rows that are meant to stay inactive:

  row 1: C_{p,1} - c_{p,1}(x) >= 0    active,   pins coordinate 1 down
  row 2: C_{p,2} - c_{p,2}(x) >= 0    inactive, slack 1 at x*
  row 3: c_{p,3}(x) - F_{p,3} >= 0    active,   pins coordinate 3 up
  row 4: c_{p,4}(x) - F_{p,4} >= 0    inactive, slack 1 at x*

=#
function build_explicit_three_player_problem()
    num_players = 3
    primal_dims = fill(5, num_players)       # 5 coordinates per player
    parameter_dims = fill(1, num_players)    # θ is unused; one dummy per player
    x_template = BlockArray(zeros(sum(primal_dims)), primal_dims)
    θ_template = BlockArray(zeros(sum(parameter_dims)), parameter_dims)

    # ---- Player 1 preferences: [outermost, middle, innermost] --------------
    J₁_outer(x, θ) = (x[Block(1)][5] - 0.80)^2
    J₁_middle(x, θ) = (x[Block(1)][3] - 0.15)^2 + (x[Block(1)][4] - 1.20)^2
    J₁_inner(x, θ) = (x[Block(1)][1] - 1.00)^2 + (x[Block(1)][2] - 0.90)^2

    # ---- Player 2 preferences ----------------------------------------------
    J₂_outer(x, θ) = (x[Block(2)][5] - 0.90)^2
    J₂_middle(x, θ) = (x[Block(2)][3] - 0.20)^2 + (x[Block(2)][4] - 1.30)^2
    J₂_inner(x, θ) = (x[Block(2)][1] - 1.05)^2 + (x[Block(2)][2] - 1.00)^2

    # ---- Player 3 preferences ----------------------------------------------
    J₃_outer(x, θ) = (x[Block(3)][5] - 1.00)^2
    J₃_middle(x, θ) = (x[Block(3)][3] - 0.25)^2 + (x[Block(3)][4] - 1.40)^2
    J₃_inner(x, θ) = (x[Block(3)][1] - 1.10)^2 + (x[Block(3)][2] - 1.10)^2

    # ---- Player 1 coupled inequalities, g₁(x, θ) >= 0 -----------------------
    g₁(x, θ) = [
        0.6375 - (x[Block(1)][1] + 0.25 * (x[Block(2)][1] + x[Block(3)][1]))
        2.4250 - (x[Block(1)][2] + 0.25 * (x[Block(2)][2] + x[Block(3)][2]))
        (x[Block(1)][3] + 0.25 * (x[Block(2)][3] + x[Block(3)][3])) - 0.8625
        (x[Block(1)][4] + 0.25 * (x[Block(2)][4] + x[Block(3)][4])) - 0.8750
    ]

    # ---- Player 2 coupled inequalities --------------------------------------
    g₂(x, θ) = [
        0.6750 - (x[Block(2)][1] + 0.25 * (x[Block(1)][1] + x[Block(3)][1]))
        2.5000 - (x[Block(2)][2] + 0.25 * (x[Block(1)][2] + x[Block(3)][2]))
        (x[Block(2)][3] + 0.25 * (x[Block(1)][3] + x[Block(3)][3])) - 0.9000
        (x[Block(2)][4] + 0.25 * (x[Block(1)][4] + x[Block(3)][4])) - 0.9500
    ]

    # ---- Player 3 coupled inequalities --------------------------------------
    g₃(x, θ) = [
        0.7125 - (x[Block(3)][1] + 0.25 * (x[Block(1)][1] + x[Block(2)][1]))
        2.5750 - (x[Block(3)][2] + 0.25 * (x[Block(1)][2] + x[Block(2)][2]))
        (x[Block(3)][3] + 0.25 * (x[Block(1)][3] + x[Block(2)][3])) - 0.9375
        (x[Block(3)][4] + 0.25 * (x[Block(1)][4] + x[Block(2)][4])) - 1.0250
    ]

    problem = ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences = [
            [J₁_outer, J₁_middle, J₁_inner],
            [J₂_outer, J₂_middle, J₂_inner],
            [J₃_outer, J₃_middle, J₃_inner],
        ],
        is_prioritized_constraint = [
            [false, false, false],
            [false, false, false],
            [false, false, false],
        ],
        equality_constraints = [nothing, nothing, nothing],
        inequality_constraints = [g₁, g₂, g₃],
        shared_equality_constraint = nothing,
        shared_inequality_constraint = nothing,
    )

    return (; problem, primal_dims)
end

"""
    solve_explicit_three_player(; formulation, kkt_system, z₀, options)

Build the explicit three-player problem, solve it, and return the solver output
alongside everything needed to describe the solution.

`formulation` picks how each player's `g_p(x, θ) ≥ 0` enters the game:

  - `:hard_inequality` — as a genuine inequality constraint, handled by the
    interior-point slack/dual machinery. This is the benchmark's native form.
  - `:innermost_preference` — appended as the highest-priority preference level
    via [`innermost_preference_inequality_goop`](@ref), leaving the problem with
    no hard inequalities at all. The constraint then enters the residual through
    the smooth penalty `(-g)^(level+2)`, so it is satisfied only to the extent
    that penalty is minimized, and the interior-point relaxation `ϵ` becomes
    inert (there are no inequality duals and no interior-point slacks left).

`kkt_system` selects the reformulation (`reduced_kkt_system`,
`quasi_kkt_system`, or `complete_kkt_system`). `z₀` defaults to the zero vector
over primals only, which is what the solver pads out internally.

Objectives and constraint values in the returned tuple are always evaluated
against the *original* problem, never the reformulated one, so the two
formulations report the same quantities and can be compared directly.
"""
function solve_explicit_three_player(;
    formulation::Symbol = :hard_inequality,
    kkt_system = reduced_kkt_system,
    z₀ = nothing,
    options = default_interior_point_options(),
)
    @assert formulation in (:hard_inequality, :innermost_preference)

    case = build_explicit_three_player_problem()
    problem = case.problem
    solved_problem = if formulation === :hard_inequality
        problem
    else
        innermost_preference_inequality_goop(problem)
    end
    local kkt
    kkt_construction_time = @elapsed kkt = kkt_system(solved_problem)

    θ = zeros(sum(problem.parameter_dims))
    initial_guess = isnothing(z₀) ? zeros(sum(case.primal_dims)) : z₀

    run_solver() = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀ = initial_guess,
        options,
    )
    first_solve_time = @elapsed run_solver()
    GC.gc()
    local output
    elapsed_solve_time = @elapsed output = run_solver()

    primals = output.z[kkt.primal_dims]
    x_blocks = BlockArray(primals, case.primal_dims)
    θ_blocks = BlockArray(θ, problem.parameter_dims)

    objectives = [
        [preference(x_blocks, θ_blocks) for preference in problem.preferences[player]] for player in 1:problem.num_players
    ]
    constraints = [
        problem.inequality_constraints[player](x_blocks, θ_blocks) for
        player in 1:problem.num_players
    ]

    # `F` is recomputed here rather than trusted from the solve so the reported
    # residual is unambiguously the one at the returned `z`.
    residual = zeros(kkt.kkt_dimension)
    kkt.F!(residual, output.z; θ, ϵ = output.ϵ, η = 0.0)

    return (;
        formulation,
        case,
        solved_problem,
        kkt,
        output,
        primals,
        x_blocks,
        objectives,
        constraints,
        kkt_construction_time,
        first_solve_time,
        elapsed_solve_time,
        residual_norm = norm(residual, 2),
        max_violation = maximum(
            maximum(max.(0.0, .-constraints[player])) for player in 1:problem.num_players
        ),
    )
end

"How each formulation describes itself in the report headers."
formulation_title(formulation::Symbol) =
    formulation === :hard_inequality ?
    "hard inequality (interior-point slacks and duals)" :
    formulation === :innermost_preference ?
    "innermost preference (smooth penalty, no hard inequalities)" :
    "innermost preference with feasibility-augmented merit"

"Print `result` relative to the hard-inequality primal reference."
function report_explicit_three_player(
    result;
    label = result.formulation,
    reference_primals,
)
    (; case, kkt, output, primals, objectives, constraints) = result
    problem = case.problem
    num_players = problem.num_players

    println("=" ^ 72)
    println("Explicit three-player, three-level GOOP")
    println("  formulation: ", formulation_title(label))
    println("=" ^ 72)

    println("\n-- Problem ------------------------------------------------------")
    @printf("  players                %d\n", num_players)
    @printf(
        "  preference levels      %d per player (%d solved)\n",
        length(problem.preferences[1]),
        length(result.solved_problem.preferences[1])
    )
    @printf(
        "  primal variables       %d (%d per player)\n",
        sum(case.primal_dims),
        case.primal_dims[1]
    )
    @printf("  inequality rows        %d per player\n", length(constraints[1]))
    @printf(
        "  KKT system             %d equations, %d unknowns\n",
        kkt.kkt_dimension,
        kkt.variable_dimension
    )

    println("\n-- Solve --------------------------------------------------------")
    @printf("  status                 %s\n", output.status)
    @printf("  outer / total iters    %d / %d\n", output.outer_iters, output.total_iters)
    @printf("  final KKT error        %.6e\n", output.kkt_error)
    @printf("  ‖F(z_returned)‖₂       %.6e\n", result.residual_norm)
    @printf("  KKT construction time  %.6f s\n", result.kkt_construction_time)
    @printf("  first solve call       %.6f s  (may include JIT)\n", result.first_solve_time)
    @printf("  warmed solve time      %.6f s\n", result.elapsed_solve_time)
    @printf("  final ϵ                %.6e\n", output.ϵ)
    @printf("  final μ                %.6e\n", output.μ)
    @printf("  feasibility error      %.6e\n", output.feasibility_error)
    @printf("  KLU singular retries   %d\n", output.klu_singular_retries)
    @printf("  SVD fallbacks          %d\n", output.svd_fallback_count)

    println("\n-- Primal solution ----------------------------------------------")
    println("            x_p,1     x_p,2     x_p,3     x_p,4     x_p,5")
    x_blocks = result.x_blocks
    for player in 1:num_players
        block = x_blocks[Block(player)]
        @printf("  player %d %s\n", player, join((@sprintf("%9.5f", v) for v in block)))
    end

    println("\n-- Objective values (preferences are [outermost … innermost]) ----")
    for player in 1:num_players
        println("  player $player:")
        for (level, value) in enumerate(objectives[player])
            name = level <= length(LEVEL_NAMES) ? LEVEL_NAMES[level] : "level $level"
            @printf("    J_%d^(%d)  %-10s %.6e\n", player, level, name, value)
        end
        @printf("    %-22s %.6e\n", "sum over levels", sum(objectives[player]))
    end

    println("\n-- Inequality constraints, g_p(x) ≥ 0 ---------------------------")
    for player in 1:num_players
        println("  player $player:")
        for (row, value) in enumerate(constraints[player])
            state = value < -1e-8 ? "VIOLATED" : value <= 1e-6 ? "active" : "inactive"
            @printf("    row %d    %12.6e   %s\n", row, value, state)
        end
    end
    @printf("  max violation          %.6e\n", result.max_violation)

    println("\n-- Distance to hard-inequality primal reference -----------------")
    @printf("  ‖x - x_hard‖∞          %.6e\n", norm(primals .- reference_primals, Inf))
    @printf("  ‖x - x_hard‖₂          %.6e\n", norm(primals .- reference_primals, 2))

    println("\n-- Dual and slack blocks ----------------------------------------")
    _min = v -> isempty(v) ? "     —" : @sprintf("%.6e", minimum(v))
    @printf(
        "  preference slacks s    %2d entries, min %s\n",
        length(output.s),
        _min(output.s)
    )
    @printf(
        "  interior-point σ       %2d entries, min %s\n",
        length(output.σ),
        _min(output.σ)
    )
    @printf(
        "  inequality duals γ     %2d entries, min %s\n",
        length(output.γ),
        _min(output.γ)
    )
    println("=" ^ 72)

    return nothing
end

"""
    compare_formulations(results)

Print the three formulations side by side. `results` maps a formulation symbol to
the tuple returned by [`solve_explicit_three_player`](@ref).

The rows worth watching when inequality rows are commented out of
`build_explicit_three_player_problem`:

  - `max violation` — the hard-inequality solve drives this to ~0 because the
    constraint is enforced exactly. The preference solve cannot: `(-g)^(level+2)`
    is flat to high order at the boundary, so a small residual still permits a
    sizeable violation.
  - `‖x - x_hard‖∞` — how far each preference formulation lands from the primal
    vector returned by the hard-inequality solve in this same run.
  - `KKT unknowns / equations` — the preference formulation is underdetermined,
    which is why its residual stalls rather than converging.
"""
function compare_formulations(results)
    order = (
        :hard_inequality,
        :innermost_preference,
        :innermost_preference_augmented,
    )
    labels = ("hard inequality", "preference", "preference + feasibility")
    reference_primals = results[:hard_inequality].primals

    println("\n", "=" ^ 72)
    println("Formulation comparison")
    println("=" ^ 72)
    @printf("  %-24s %22s %22s %22s\n", "", labels...)

    # `Format` is built at runtime because each row carries its own spec;
    # `@sprintf` would need the format to be a literal.
    row(name, spec, f) = @printf(
        "  %-24s %22s %22s %22s\n",
        name,
        format(Format(spec), f(results[order[1]])),
        format(Format(spec), f(results[order[2]])),
        format(Format(spec), f(results[order[3]])),
    )

    row("status", "%s", r -> r.output.status)
    row("KKT equations", "%d", r -> r.kkt.kkt_dimension)
    row("KKT unknowns", "%d", r -> r.kkt.variable_dimension)
    row("total iterations", "%d", r -> r.output.total_iters)
    row("KKT construction time (s)", "%.6f", r -> r.kkt_construction_time)
    row("first solve call (s)", "%.6f", r -> r.first_solve_time)
    row("warmed solve time (s)", "%.6f", r -> r.elapsed_solve_time)
    row("final KKT error", "%.6e", r -> r.output.kkt_error)
    row("‖F(z_returned)‖₂", "%.6e", r -> r.residual_norm)
    row("max violation", "%.6e", r -> r.max_violation)
    row("‖x - x_hard‖∞", "%.6e", r -> norm(r.primals .- reference_primals, Inf))
    row("final μ", "%.6e", r -> r.output.μ)
    row("feasibility error", "%.6e", r -> r.output.feasibility_error)
    row("interior-point slacks", "%d", r -> length(r.output.σ))
    row("inequality duals", "%d", r -> length(r.output.γ))
    row("preference slacks", "%d", r -> length(r.output.s))
    for (level, name) in enumerate(LEVEL_NAMES)
        row(
            "Σ_p J_p^($level) ($name)",
            "%.6e",
            r -> sum(r.objectives[p][level] for p in 1:r.case.problem.num_players),
        )
    end

    println("=" ^ 72)

    return nothing
end

"""
    solve_hard_kkt_with_nonlinear_solve(hard_result, preference_result)

Fix the feasibility-augmented preference primals and solve only for the remaining
hard-system coordinates. Hard-only slack and inequality-dual blocks start at one,
while unrestricted duals start at zero. The default `epsilon = 0` evaluates the
unperturbed hard KKT equations, which is necessary when a fixed primal lies
exactly on an active inequality boundary.

Nonnegative free coordinates are parameterized as `z_i = u_i^2`. The resulting
least-squares problem has all hard KKT equations but excludes the fixed primal
columns from its unknown vector and Jacobian.
"""
function solve_hard_kkt_with_nonlinear_solve(
    hard_result,
    preference_result;
    epsilon = 0.0,
    abstol = 1e-6,
    reltol = 1e-6,
    maxiters = 2000,
)
    hard_kkt = hard_result.kkt
    problem = hard_result.case.problem
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative."))

    z = zeros(hard_kkt.variable_dimension)
    z[hard_kkt.primal_dims] .= preference_result.primals
    z[hard_kkt.preference_slack_dims] .= 1.0
    z[hard_kkt.interior_point_slack_dims] .= 1.0
    z[hard_kkt.inequality_constraint_dual_dims] .= 1.0

    positive_coordinate = falses(hard_kkt.variable_dimension)
    positive_coordinate[hard_kkt.preference_slack_dims] .= true
    positive_coordinate[hard_kkt.interior_point_slack_dims] .= true
    positive_coordinate[hard_kkt.inequality_constraint_dual_dims] .= true

    is_fixed_primal = falses(hard_kkt.variable_dimension)
    is_fixed_primal[hard_kkt.primal_dims] .= true
    free_dims = findall(!, is_fixed_primal)
    u0 = zeros(length(free_dims))
    for (free_index, z_index) in enumerate(free_dims)
        value = z[z_index]
        u0[free_index] = positive_coordinate[z_index] ? sqrt(value) : value
    end

    theta = zeros(sum(problem.parameter_dims))
    residual = zeros(hard_kkt.kkt_dimension)
    initial_residual = similar(residual)
    full_jacobian = copy(hard_kkt.∇F_z!.result_buffer)
    transformed_jacobian = zeros(hard_kkt.kkt_dimension, length(free_dims))
    full_rows = rowvals(full_jacobian)
    full_values = nonzeros(full_jacobian)

    function unpack!(u)
        for (free_index, z_index) in enumerate(free_dims)
            z[z_index] =
                positive_coordinate[z_index] ? u[free_index]^2 : u[free_index]
        end
        return nothing
    end

    function hard_residual!(result, u, _)
        unpack!(u)
        hard_kkt.F!(result, z; θ = theta, ϵ = epsilon, η = 0.0)
        return nothing
    end

    function hard_jacobian!(result, u, _)
        unpack!(u)
        hard_kkt.∇F_z!(full_jacobian, z; θ = theta, ϵ = epsilon, η = 0.0)
        fill!(result, 0.0)
        for (free_column, full_column) in enumerate(free_dims)
            chain_scale =
                positive_coordinate[full_column] ? 2.0 * u[free_column] : 1.0
            for stored_index in nzrange(full_jacobian, full_column)
                result[full_rows[stored_index], free_column] =
                    chain_scale * full_values[stored_index]
            end
        end
        return nothing
    end

    hard_residual!(initial_residual, u0, nothing)
    nonlinear_function = NonlinearFunction(
        hard_residual!;
        jac = hard_jacobian!,
        jac_prototype = transformed_jacobian,
        resid_prototype = residual,
    )
    least_squares_problem = NonlinearLeastSquaresProblem(nonlinear_function, u0)
    algorithm = LevenbergMarquardt(
        disable_geodesic = true,
        linsolve = NonlinearSolve.LinearSolve.QRFactorization(ColumnNorm()),
    )
    run_nonlinear_solve() = solve(
        least_squares_problem,
        algorithm;
        abstol,
        reltol,
        maxiters,
    )
    first_solve_time = @elapsed run_nonlinear_solve()
    GC.gc()
    local solution
    elapsed_time = @elapsed solution = run_nonlinear_solve()

    unpack!(solution.u)
    hard_kkt.F!(residual, z; θ = theta, ϵ = epsilon, η = 0.0)
    hard_jacobian!(transformed_jacobian, solution.u, nothing)
    least_squares_gradient = transpose(transformed_jacobian) * residual

    primals = copy(z[hard_kkt.primal_dims])
    x_blocks = BlockArray(primals, hard_result.case.primal_dims)
    theta_blocks = BlockArray(theta, problem.parameter_dims)
    constraints = [
        problem.inequality_constraints[player](x_blocks, theta_blocks) for
        player in 1:problem.num_players
    ]
    minimum_constraint = minimum(
        value for player_constraints in constraints for value in player_constraints
    )
    maximum_violation = maximum(
        max(-value, 0.0) for player_constraints in constraints for value in player_constraints
    )
    sigma = @view z[hard_kkt.interior_point_slack_dims]
    gamma = @view z[hard_kkt.inequality_constraint_dual_dims]
    complementarity_error = maximum(
        abs(sigma[index] * gamma[index] - epsilon) for index in eachindex(sigma, gamma)
    )

    return (;
        solution,
        z,
        primals,
        constraints,
        residual,
        epsilon,
        first_solve_time,
        elapsed_time,
        fixed_primal_count = length(hard_kkt.primal_dims),
        free_variable_count = length(free_dims),
        primal_movement = norm(primals .- preference_result.primals, Inf),
        initial_kkt_error = norm(initial_residual, 2),
        kkt_error = norm(residual, 2),
        residual_inf = norm(residual, Inf),
        least_squares_gradient_inf = norm(least_squares_gradient, Inf),
        minimum_constraint,
        maximum_violation,
        minimum_sigma = minimum(sigma),
        minimum_gamma = minimum(gamma),
        complementarity_error,
    )
end

"Compare fixed-primal hard-KKT dual recovery with the interior-point solve."
function report_hard_kkt_nonlinear_comparison(hard_result, preference_result, nonlinear_result)
    hard_kkt = hard_result.kkt
    warmstart_movement = norm(nonlinear_result.primals .- preference_result.primals, Inf)
    interior_point_distance = norm(nonlinear_result.primals .- hard_result.primals, Inf)
    full_kkt_variable_distance = norm(nonlinear_result.z .- hard_result.output.z, Inf)
    sigma_distance = norm(
        nonlinear_result.z[hard_kkt.interior_point_slack_dims] .-
        hard_result.output.z[hard_kkt.interior_point_slack_dims],
        Inf,
    )
    gamma_distance = norm(
        nonlinear_result.z[hard_kkt.inequality_constraint_dual_dims] .-
        hard_result.output.z[hard_kkt.inequality_constraint_dual_dims],
        Inf,
    )

    println("\n", "=" ^ 72)
    println("Hard-KKT dual recovery with preference + feasibility primals fixed")
    println("=" ^ 72)
    @printf(
        "  hard system                    %d equations, %d variables\n",
        hard_kkt.kkt_dimension,
        hard_kkt.variable_dimension,
    )
    @printf("  fixed primals / free variables %d / %d\n", nonlinear_result.fixed_primal_count, nonlinear_result.free_variable_count)
    @printf("  fixed-primal max violation     %.6e\n", preference_result.max_violation)
    @printf("  hard-KKT epsilon               %.6e\n", nonlinear_result.epsilon)
    @printf("  NonlinearSolve retcode         %s\n", nonlinear_result.solution.retcode)
    @printf("  NonlinearSolve iterations      %d\n", nonlinear_result.solution.stats.nsteps)
    @printf("  first solve call (s)           %.6f  (may include JIT)\n", nonlinear_result.first_solve_time)
    @printf("  warmed dual solve time (s)     %.6f\n", nonlinear_result.elapsed_time)

    println("\n  Fixed-primal hard-KKT least-squares result")
    println("    fixed primal solution        ", nonlinear_result.primals)
    @printf("    initial KKT residual         %.6e\n", nonlinear_result.initial_kkt_error)
    @printf(
        "    final KKT residual           %.6e  (∞-norm %.6e)\n",
        nonlinear_result.kkt_error,
        nonlinear_result.residual_inf,
    )
    @printf("    ‖J_free' F‖∞                %.6e\n", nonlinear_result.least_squares_gradient_inf)
    @printf("    minimum original constraint  %.6e\n", nonlinear_result.minimum_constraint)
    @printf("    maximum original violation   %.6e\n", nonlinear_result.maximum_violation)
    @printf("    minimum sigma                %.6e\n", nonlinear_result.minimum_sigma)
    @printf("    minimum gamma                %.6e\n", nonlinear_result.minimum_gamma)
    @printf("    max |sigma_i gamma_i-epsilon| %.6e\n", nonlinear_result.complementarity_error)

    println("\n  Comparison with the interior-point hard solve")
    @printf("    interior-point status        %s\n", hard_result.output.status)
    @printf("    interior-point KKT residual  %.6e\n", hard_result.residual_norm)
    @printf("    interior-point iterations    %d\n", hard_result.output.total_iters)
    @printf("    interior-point time (s)      %.6f\n", hard_result.elapsed_solve_time)
    @printf("    ‖x_fixed-x_preference‖∞      %.6e\n", warmstart_movement)
    @printf("    ‖x_fixed-x_IP‖∞              %.6e\n", interior_point_distance)
    @printf("    ‖sigma_NLS-sigma_IP‖∞       %.6e\n", sigma_distance)
    @printf("    ‖gamma_NLS-gamma_IP‖∞       %.6e\n", gamma_distance)
    @printf("    ‖z_NLS-z_IP‖∞               %.6e\n", full_kkt_variable_distance)
    @printf(
        "    fixed primal feasible         %s\n",
        nonlinear_result.maximum_violation == 0.0 ? "yes" : "no",
    )
    println("=" ^ 72)
    return nothing
end
"Map positive history values to base-10 logarithms, leaving invalid values as gaps."
_positive_log10(values) = [
    isfinite(value) && value > 0 ? log10(value) : NaN for value in values
]

"""
    save_step_history_comparison(results; path)

Write one PDF comparing the raw Newton-direction norm and accepted backtracking
factor for the hard, preference, and feasibility-augmented preference solves.
Every recorded iteration is passed to CairoMakie; rasterizing the line artists
keeps the PDF compact even for very long solves.
"""
function save_step_history_comparison(
    results;
    path = joinpath(
        @__DIR__,
        "..",
        "output",
        "pdf",
        "explicit_three_player_step_history.pdf",
    ),
)
    if !CAIRO_MAKIE_AVAILABLE
        @warn "CairoMakie is unavailable; run with `julia --project=experiments test/explicit_three_player_solve.jl` to generate the step-history PDF."
        return nothing
    end

    order = (
        :hard_inequality,
        :innermost_preference,
        :innermost_preference_augmented,
    )
    labels = Dict(
        :hard_inequality => "Hard inequality",
        :innermost_preference => "Innermost preference",
        :innermost_preference_augmented => "Preference + feasibility",
    )
    colors = Dict(
        :hard_inequality => :black,
        :innermost_preference => :dodgerblue,
        :innermost_preference_augmented => :orangered,
    )
    figure = CairoMakie.Figure(
        size = (1050, 780),
        fonts = (;
            regular = "TeX Gyre Termes Makie",
            bold = "TeX Gyre Termes Makie",
        ),
    )
    direction_axis = CairoMakie.Axis(
        figure[1, 1];
        title = "Explicit three-player solve: Newton step histories",
        ylabel = "log₁₀ ‖δzₖ‖₂",
        xscale = log10,
    )
    alpha_axis = CairoMakie.Axis(
        figure[2, 1];
        xlabel = "accepted Newton iteration (log scale)",
        ylabel = "log₁₀ αₖ",
        xscale = log10,
    )

    for key in order
        output = results[key].output
        direction_norms = output.delta_z_norm_history
        alphas = output.alpha_history
        @assert length(direction_norms) == length(alphas)
        iterations = collect(1:length(alphas))
        CairoMakie.lines!(
            direction_axis,
            iterations,
            _positive_log10(direction_norms);
            color = colors[key],
            linewidth = 2,
            label = labels[key],
            rasterize = 2,
        )
        CairoMakie.lines!(
            alpha_axis,
            iterations,
            _positive_log10(alphas);
            color = colors[key],
            linewidth = 2,
            rasterize = 2,
        )
    end

    CairoMakie.axislegend(direction_axis; position = :lb, framevisible = false)
    CairoMakie.linkxaxes!(direction_axis, alpha_axis)
    CairoMakie.hidexdecorations!(direction_axis; grid = false)
    CairoMakie.rowgap!(figure.layout, 14)

    mkpath(dirname(path))
    CairoMakie.save(path, figure)
    return abspath(path)
end

"""
    save_kkt_error_history_comparison(results; path)

Write one PDF comparing the KKT residual norm at every accepted iteration for
the same three solves and with the same color assignment as the step-history
figure. Three further vertically stacked panels track the augmented-Lagrangian
solve alone: the constrained least-squares stationarity residual
‖JᵀF-Cᵀλ_f‖∞ used by the outer termination test, the multiplier norm ‖λ_f‖∞,
and the penalty weight μₖ. Those histories are NaN for the other two solves.
"""
function save_kkt_error_history_comparison(
    results;
    path = joinpath(
        @__DIR__,
        "..",
        "output",
        "pdf",
        "explicit_three_player_kkt_error_history.pdf",
    ),
)
    if !CAIRO_MAKIE_AVAILABLE
        @warn "CairoMakie is unavailable; run with `julia --project=experiments test/explicit_three_player_solve.jl` to generate the KKT-error PDF."
        return nothing
    end

    order = (
        :hard_inequality,
        :innermost_preference,
        :innermost_preference_augmented,
    )
    labels = Dict(
        :hard_inequality => "Hard inequality",
        :innermost_preference => "Innermost preference",
        :innermost_preference_augmented => "Preference + feasibility",
    )
    colors = Dict(
        :hard_inequality => :black,
        :innermost_preference => :dodgerblue,
        :innermost_preference_augmented => :orangered,
    )

    figure = CairoMakie.Figure(
        size = (1050, 1200),
        fonts = (;
            regular = "TeX Gyre Termes Makie",
            bold = "TeX Gyre Termes Makie",
        ),
    )
    kkt_axis = CairoMakie.Axis(
        figure[1, 1];
        title = "Explicit three-player solve: KKT and augmented-Lagrangian histories",
        ylabel = "log₁₀ ‖F(zₖ)‖₂",
        xscale = log10,
    )
    constrained_stationarity_axis = CairoMakie.Axis(
        figure[2, 1];
        ylabel = "log₁₀ ‖JᵀF-Cᵀλ_f‖∞",
        xscale = log10,
    )
    lambda_axis = CairoMakie.Axis(
        figure[3, 1];
        ylabel = "log₁₀ ‖λ_f,ₖ‖∞",
        xscale = log10,
    )
    mu_axis = CairoMakie.Axis(
        figure[4, 1];
        xlabel = "accepted Newton iteration (log scale)",
        ylabel = "log₁₀ μₖ",
        xscale = log10,
    )

    for key in order
        output = results[key].output
        errors = output.kkt_error_history
        iterations = collect(1:length(errors))
        CairoMakie.lines!(
            kkt_axis,
            iterations,
            _positive_log10(errors);
            color = colors[key],
            linewidth = 2,
            label = labels[key],
            rasterize = 2,
        )
        # The AL histories are NaN except for the feasibility-merit solve, so
        # these calls draw nothing for the other two formulations.
        al_panels = (
            (
                constrained_stationarity_axis,
                output.constrained_stationarity_history,
            ),
            (lambda_axis, output.lambda_f_norm_history),
            (mu_axis, output.mu_history),
        )
        for (axis, history) in al_panels
            any(isfinite, history) || continue
            CairoMakie.lines!(
                axis,
                collect(1:length(history)),
                _positive_log10(history);
                color = colors[key],
                linewidth = 2,
                rasterize = 2,
            )
        end
    end

    CairoMakie.axislegend(kkt_axis; position = :lb, framevisible = false)
    CairoMakie.linkxaxes!(
        kkt_axis,
        constrained_stationarity_axis,
        lambda_axis,
        mu_axis,
    )
    for axis in (kkt_axis, constrained_stationarity_axis, lambda_axis)
        CairoMakie.hidexdecorations!(axis; grid = false)
    end
    CairoMakie.rowgap!(figure.layout, 14)
    mkpath(dirname(path))
    CairoMakie.save(path, figure)
    return abspath(path)
end

hard_options = default_interior_point_options(; record_convergence = true)
preference_options = default_interior_point_options(;
    η₀ = 1e-6,
    record_convergence = true,
)
augmented_preference_options = default_interior_point_options(;
    η₀ = 1e-6,
    use_feasibility_merit = true,
    μ₀ = 1.0,
    μ_max = 1e4,
    mu_growth = 10.0,
    record_convergence = true,
)

results = Dict(
    :hard_inequality => solve_explicit_three_player(;
        formulation = :hard_inequality,
        options = hard_options,
    ),
    :innermost_preference => solve_explicit_three_player(;
        formulation = :innermost_preference,
        options = preference_options,
    ),
    :innermost_preference_augmented => solve_explicit_three_player(;
        formulation = :innermost_preference,
        options = augmented_preference_options,
    ),
)

reference_primals = results[:hard_inequality].primals
for label in (:hard_inequality, :innermost_preference, :innermost_preference_augmented)
    report_explicit_three_player(results[label]; label, reference_primals)
    println()
end

compare_formulations(results)

hard_kkt_nonlinear_result = solve_hard_kkt_with_nonlinear_solve(
    results[:hard_inequality],
    results[:innermost_preference_augmented],
)
report_hard_kkt_nonlinear_comparison(
    results[:hard_inequality],
    results[:innermost_preference_augmented],
    hard_kkt_nonlinear_result,
)

step_history_pdf = save_step_history_comparison(results)
!isnothing(step_history_pdf) && println("\nStep-history PDF written to $step_history_pdf")
kkt_error_history_pdf = save_kkt_error_history_comparison(results)
!isnothing(kkt_error_history_pdf) &&
    println("KKT-error-history PDF written to $kkt_error_history_pdf")

@test results[:innermost_preference_augmented].max_violation <
      results[:innermost_preference].max_violation
for result in values(results)
    @test !isempty(result.output.alpha_history)
    @test length(result.output.delta_z_norm_history) ==
          length(result.output.alpha_history)
    @test length(result.output.kkt_error_history) == length(result.output.alpha_history)
    @test length(result.output.merit_history) == length(result.output.alpha_history)
    @test length(result.output.merit_stationarity_history) ==
          length(result.output.alpha_history)
    @test length(result.output.constrained_stationarity_history) ==
          length(result.output.alpha_history)
    @test length(result.output.lambda_f_norm_history) ==
          length(result.output.alpha_history)
    @test length(result.output.mu_history) == length(result.output.alpha_history)
end
@test any(isfinite, results[:innermost_preference_augmented].output.merit_history)
@test any(
    isfinite,
    results[:innermost_preference_augmented].output.constrained_stationarity_history,
)
@test last(
    results[:innermost_preference_augmented].output.constrained_stationarity_history,
) ≈ results[:innermost_preference_augmented].output.merit_stationarity
@test any(
    isfinite,
    results[:innermost_preference_augmented].output.lambda_f_norm_history,
)
@test any(isfinite, results[:innermost_preference_augmented].output.mu_history)
@test all(isnan, results[:hard_inequality].output.merit_history)
@test all(isnan, results[:hard_inequality].output.constrained_stationarity_history)
@test all(isnan, results[:hard_inequality].output.lambda_f_norm_history)
@test all(isnan, results[:hard_inequality].output.mu_history)
@test all(
    >=(0.0),
    hard_kkt_nonlinear_result.z[results[:hard_inequality].kkt.interior_point_slack_dims],
)
@test all(
    >=(0.0),
    hard_kkt_nonlinear_result.z[results[:hard_inequality].kkt.inequality_constraint_dual_dims],
)
@test hard_kkt_nonlinear_result.epsilon == 0.0
@test hard_kkt_nonlinear_result.primal_movement == 0.0
@test hard_kkt_nonlinear_result.kkt_error <= hard_kkt_nonlinear_result.initial_kkt_error
@test hard_kkt_nonlinear_result.maximum_violation == 0.0
@test hard_kkt_nonlinear_result.minimum_constraint >= 0.0
@test hard_kkt_nonlinear_result.complementarity_error <= 1e-6
@test hard_kkt_nonlinear_result.least_squares_gradient_inf <= 1e-6
@test isfinite(hard_kkt_nonlinear_result.kkt_error)
