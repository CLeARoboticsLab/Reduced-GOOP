# The explicit three-player, three-level game, plus a driver that solves it and
# prints the result.
#
# This is a report script, not part of `runtests.jl` — running it *is* the test:
#
#     julia --project=. test/explicit_three_player_solve.jl
#
# (`--project=test` also works once that environment has been instantiated.)
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
using LinearAlgebra: norm
using ReducedGOOP

@isdefined(default_interior_point_options) ||
    include(joinpath(@__DIR__, "benchmark_problems.jl"))

# `innermost_preference_inequality_goop` lives in the comparison script. Setting
# the guard flag first suppresses that file's `@testset`, so including it here
# only imports the helper instead of running a second suite.
if !@isdefined(innermost_preference_inequality_goop)
    const RUN_QUASI_VS_REDUCED_TESTSET = false
    include(joinpath(@__DIR__, "quasi_vs_reduced_preference_inequality.jl"))
end

"Human-readable name for each preference level, outermost first."
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
constraints. It is x_p* shifted by [+0.6, 0, -0.4, 0, 0], so each player wants
coordinate 1 larger and coordinate 3 smaller than the equilibrium allows:

  t_1 = [1.00, 0.90, 0.15, 1.20, 0.80]     x_1* = [0.40, 0.90, 0.55, 1.20, 0.80]
  t_2 = [1.05, 1.00, 0.20, 1.30, 0.90]     x_2* = [0.45, 1.00, 0.60, 1.30, 0.90]
  t_3 = [1.10, 1.10, 0.25, 1.40, 1.00]     x_3* = [0.50, 1.10, 0.65, 1.40, 1.00]

Coordinates 2, 4, 5 are unconstrained, so they sit at their targets and
x_{p,j}* = t_{p,j} there.

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

    #! format: off
    expected = [
        0.40, 0.90, 0.55, 1.20, 0.80,   # x_1*
        0.45, 1.00, 0.60, 1.30, 0.90,   # x_2*
        0.50, 1.10, 0.65, 1.40, 1.00,   # x_3*
    ]
    #! format: on

    return (; problem, expected, primal_dims)
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
    kkt = kkt_system(solved_problem)

    θ = zeros(sum(problem.parameter_dims))
    initial_guess = isnothing(z₀) ? zeros(sum(case.primal_dims)) : z₀

    output = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀ = initial_guess,
        options,
    )

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
    residual = zeros(kkt.variable_dimension)
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
    "innermost preference (smooth penalty, no hard inequalities)"

"Print `result` from `solve_explicit_three_player` as a readable report."
function report_explicit_three_player(result)
    (; case, kkt, output, primals, objectives, constraints) = result
    problem = case.problem
    num_players = problem.num_players

    println("=" ^ 72)
    println("Explicit three-player, three-level GOOP")
    println("  formulation: ", formulation_title(result.formulation))
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
    @printf("  final ϵ                %.6e\n", output.ϵ)
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

    println("\n-- Distance to the known solution -------------------------------")
    @printf("  ‖x - x*‖∞              %.6e\n", norm(primals .- case.expected, Inf))
    @printf("  ‖x - x*‖₂              %.6e\n", norm(primals .- case.expected, 2))

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

Print the two formulations side by side. `results` maps a formulation symbol to
the tuple returned by [`solve_explicit_three_player`](@ref).

The rows worth watching when inequality rows are commented out of
`build_explicit_three_player_problem`:

  - `max violation` — the hard-inequality solve drives this to ~0 because the
    constraint is enforced exactly. The preference solve cannot: `(-g)^(level+2)`
    is flat to high order at the boundary, so a small residual still permits a
    sizeable violation.
  - `‖x - x*‖∞` — how far each formulation lands from the closed-form solution
    of the *hard-constrained* problem.
  - `KKT unknowns / equations` — the preference formulation is underdetermined,
    which is why its residual stalls rather than converging.
"""
function compare_formulations(results)
    order = (:hard_inequality, :innermost_preference)
    labels = ("hard inequality", "innermost preference")

    println("\n", "=" ^ 72)
    println("Formulation comparison")
    println("=" ^ 72)
    @printf("  %-24s %22s %22s\n", "", labels[1], labels[2])

    # `Format` is built at runtime because each row carries its own spec;
    # `@sprintf` would need the format to be a literal.
    row(name, spec, f) = @printf(
        "  %-24s %22s %22s\n",
        name,
        format(Format(spec), f(results[order[1]])),
        format(Format(spec), f(results[order[2]])),
    )

    row("status", "%s", r -> r.output.status)
    row("KKT equations", "%d", r -> r.kkt.kkt_dimension)
    row("KKT unknowns", "%d", r -> r.kkt.variable_dimension)
    row("total iterations", "%d", r -> r.output.total_iters)
    row("final KKT error", "%.6e", r -> r.output.kkt_error)
    row("‖F(z_returned)‖₂", "%.6e", r -> r.residual_norm)
    row("max violation", "%.6e", r -> r.max_violation)
    row("‖x - x*‖∞", "%.6e", r -> norm(r.primals .- r.case.expected, Inf))
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

    primal_gap = norm(results[order[1]].primals .- results[order[2]].primals, Inf)
    @printf("\n  ‖x_hard - x_preference‖∞  %.6e\n", primal_gap)
    println("=" ^ 72)

    return nothing
end

results = Dict(
    formulation => solve_explicit_three_player(; formulation) for
    formulation in (:hard_inequality, :innermost_preference)
)

for formulation in (:hard_inequality, :innermost_preference)
    report_explicit_three_player(results[formulation])
    println()
end

compare_formulations(results)
