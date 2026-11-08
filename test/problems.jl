# problems.jl -- the test problems of runtests.jl and the helpers that build, evaluate and solve them:
#   - the known-solution benchmark families (one or three players, one to three levels, quadratic or nonlinear);
#   - the small problems of ScholtesReducedGOOP.jl's tests (Jingqi Li), written against a flat primal `z` and
#     adapted to `ParametricGOOP` by `sproblem`;
#   - a coupled two-player, three-level game with a unique equilibrium and a rank-deficient J (the solvers);
#   - one-player, one-variable problems for the complete systems.

using Test, LinearAlgebra, Random, SparseArrays
using BlockArrays: Block, BlockArray
using Symbolics: Symbolics
using ReducedGOOP
const RG = ReducedGOOP

# ── Known-solution benchmark families ───────────────────────────────────────────────────────────────

const IP_ATOL = 1e-7
const PRIMAL_DIM = 5
const COUPLING_WEIGHT = 0.25

#=
Benchmark Problem Families
--------------------------

Small problems whose solutions and active sets are known in closed form, solved by PDIP on the reduced and
quasi systems (runtests.jl, "Reduced and quasi systems").

Notation: x_i is player i's decision vector, and x_{i,j} is the j-th decision
variable of player i. In code this is x[Block(i)][j].

The constrained benchmark problems use n = 5 decision variables per player,
with one, two or three levels.

For both single-player and three-player cases we test two variants:

  Quadratic: J_S(x) = sum((x_j - raw_target_j)^2 for j in S)
             with linear inequalities.

  Nonlinear: J_S(x) = sum(cosh(x_j - raw_target_j) - 1 for j in S)
             with exp-transformed inequalities exp(g(x)) - 1 >= 0.

The raw objective target is the point the objective would choose without any
constraints. The expected solution is the feasible point obtained after the
constraints are enforced. For example, if a raw target component is below a
lower bound or above an upper bound, the constrained solution clips that
component to the corresponding active bound.

The nonlinear constraint transform preserves the same feasible set and active
set as g(x) >= 0 because exp(r) - 1 = 0 iff r = 0, and exp(r) - 1 > 0 iff r > 0.
=#

"""
PDIP options of the known-solution tests: the regularized `:klu` step (Levenberg–Marquardt with the
default η = 1e-6) at a single relaxation ρ = 1e-9, from the known primals. On the multi-level benchmarks J is
rank-deficient near the solution, where the exact step −J⁺K_ρ (`:svd`) stalls, and a ladder from ρ = 1 ends
farther from x*.
"""
default_pdip_options(; record_convergence = false) =
    ReducedGOOP.PDIPOptions(; ρ_schedule = [1e-9], linear_solver = :klu, record_convergence)

# Generate KKT systems 
reduced_kkt_system(problem) = ReducedGOOP.generate_slacked_reduced_kkt_system(problem)
quasi_kkt_system(problem) = ReducedGOOP.generate_slacked_quasi_kkt_system(problem)

function solve_with_pdip(
    problem,
    expected_primals;
    z₀ = nothing,
    kkt_system = reduced_kkt_system,
)
    kkt = kkt_system(problem)
    initial_guess = isnothing(z₀) ? expected_primals : z₀

    output = ReducedGOOP.solve(
        ReducedGOOP.PDIP(),
        kkt,
        zeros(sum(problem.parameter_dims));
        z₀ = initial_guess,
        options = default_pdip_options(),
    )

    return (; output, primals = output.x, kkt)
end

function preference_coordinate_groups(levels::Int)
    if levels == 1
        return [[1, 2, 3, 4, 5]]
    elseif levels == 2
        return [[4, 5], [1, 2, 3]]
    elseif levels == 3
        return [[5], [3, 4], [1, 2]]
    else
        error("Only single-level, bilevel, and trilevel problems are tested.")
    end
end

# Single-player expected solution:
#
#   x* = [0.0, 0.6, 1.0, 0.4, 1.4]
#
# The unconstrained/raw target is:
#
#   t = [-0.4, 0.6, 1.4, 0.4, 1.4]
#
# This is the point the objective would choose without constraints. The box
# constraints clip t_1 up to 0 and t_3 down to 1, giving x* below.
#
# with box constraints 0 <= x <= [2, 2, 1, 2, 2]. Thus coordinate 1 is pushed
# to its lower bound and coordinate 3 is pushed to its upper bound:
#
#   active constraints:   x_1 = 0, x_3 = 1
#   inactive constraints: all other lower/upper bounds.
single_player_expected_block() = [0.0, 0.6, 1.0, 0.4, 1.4]
single_player_raw_target() = [-0.4, 0.6, 1.4, 0.4, 1.4]

# Three-player expected solution:
#
#   x_1* = [0.40, 0.90, 0.55, 1.20, 0.80]
#   x_2* = [0.45, 1.00, 0.60, 1.30, 0.90]
#   x_3* = [0.50, 1.10, 0.65, 1.40, 1.00]
#
# Player i's raw target is x_i* + [0.6, 0, -0.4, 0, 0]. The coupled constraints
# below force coordinate 1 down and coordinate 3 up to the expected point.
function multiplayer_expected_block(player::Int)
    return [
        0.35 + 0.05 * player,
        0.80 + 0.10 * player,
        0.50 + 0.05 * player,
        1.10 + 0.10 * player,
        0.70 + 0.10 * player,
    ]
end

function multiplayer_raw_target(player::Int)
    raw = multiplayer_expected_block(player)
    raw[1] += 0.6
    raw[3] -= 0.4
    return raw
end

function objective_from_target(kind::Symbol, player::Int, coordinates, raw_target)
    if kind === :quadratic
        return let player = player, coordinates = coordinates, raw_target = raw_target
            (x, θ) -> sum((x[Block(player)][j] - raw_target[j])^2 for j in coordinates)
        end
    elseif kind === :nonlinear
        return let player = player, coordinates = coordinates, raw_target = raw_target
            (x, θ) ->
                sum(cosh(x[Block(player)][j] - raw_target[j]) - 1.0 for j in coordinates)
        end
    else
        error("Unknown problem kind: $kind")
    end
end

function transform_constraints(kind::Symbol, residuals)
    if kind === :quadratic
        return residuals
    elseif kind === :nonlinear
        return exp.(residuals) .- 1.0
    else
        error("Unknown problem kind: $kind")
    end
end

function single_player_constraint(kind::Symbol)
    lower = zeros(PRIMAL_DIM)
    upper = [2.0, 2.0, 1.0, 2.0, 2.0]

    return let kind = kind, lower = lower, upper = upper
        (x, θ) -> begin
            xi = x[Block(1)]
            transform_constraints(kind, [xi .- lower; upper .- xi])
        end
    end
end

# Coupled generalized Nash resource expression:
#
#   c_{i,j}(x) = x_{i,j} + 0.25 * sum(x_{k,j} for k != i)
#
# Here x_{i,j} denotes player i's coordinate j, i.e. x[Block(i)][j] in code.
#
# Each player i receives four coupled inequalities:
#
#   C_{i,1} - c_{i,1}(x) >= 0    active upper resource
#   C_{i,2} - c_{i,2}(x) >= 0    inactive upper resource, residual slack 1
#   c_{i,3}(x) - F_{i,3} >= 0    active lower resource
#   c_{i,4}(x) - F_{i,4} >= 0    inactive lower resource, residual slack 1
#
# In the nonlinear variant, these inactive transformed constraints evaluate to
# exp(1) - 1 rather than 1.
#
# Constants C and F are chosen from x*. The active equations are nonsingular
# across the three players because the coupling matrix has 1 on the diagonal
# and 0.25 off diagonal, so the active resources pin the intended distribution,
# not only the aggregate total.
function coupled_coordinate(x, player::Int, coordinate::Int, num_players::Int)
    return x[Block(player)][coordinate] +
           COUPLING_WEIGHT *
           sum(x[Block(other)][coordinate] for other in 1:num_players if other != player)
end

function multiplayer_constraint(kind::Symbol, player::Int, expected_blocks)
    num_players = length(expected_blocks)
    coupled_expected(coordinate) =
        expected_blocks[player][coordinate] +
        COUPLING_WEIGHT * sum(
            expected_blocks[other][coordinate] for
            other in 1:num_players if other != player
        )
    active_upper_capacity = coupled_expected(1)
    inactive_upper_capacity = coupled_expected(2) + 1.0
    active_lower_floor = coupled_expected(3)
    inactive_lower_floor = coupled_expected(4) - 1.0

    return let kind = kind,
        player = player,
        num_players = num_players,
        active_upper_capacity = active_upper_capacity,
        inactive_upper_capacity = inactive_upper_capacity,
        active_lower_floor = active_lower_floor,
        inactive_lower_floor = inactive_lower_floor

        (x, θ) -> begin
            residuals = [
                active_upper_capacity - coupled_coordinate(x, player, 1, num_players)
                inactive_upper_capacity - coupled_coordinate(x, player, 2, num_players)
                coupled_coordinate(x, player, 3, num_players) - active_lower_floor
                coupled_coordinate(x, player, 4, num_players) - inactive_lower_floor
            ]
            transform_constraints(kind, residuals)
        end
    end
end

function build_benchmark_problem(; num_players::Int, levels::Int, kind::Symbol)
    @assert num_players in (1, 3)
    @assert levels in (1, 2, 3)
    @assert kind in (:quadratic, :nonlinear)

    primal_dims = fill(PRIMAL_DIM, num_players)
    parameter_dims = fill(1, num_players)
    x_template = BlockArray(zeros(sum(primal_dims)), primal_dims)
    θ_template = BlockArray(zeros(sum(parameter_dims)), parameter_dims)
    coordinate_groups = preference_coordinate_groups(levels)

    expected_blocks = if num_players == 1
        [single_player_expected_block()]
    else
        [multiplayer_expected_block(player) for player in 1:num_players]
    end
    raw_targets = if num_players == 1
        [single_player_raw_target()]
    else
        [multiplayer_raw_target(player) for player in 1:num_players]
    end

    preferences = [Function[] for _ in 1:num_players]
    for player in 1:num_players
        for coordinates in coordinate_groups
            push!(
                preferences[player],
                objective_from_target(kind, player, coordinates, raw_targets[player]),
            )
        end
    end

    inequality_constraints = if num_players == 1
        [single_player_constraint(kind)]
    else
        [multiplayer_constraint(kind, player, expected_blocks) for player in 1:num_players]
    end

    problem = ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences,
        is_prioritized_constraint = [fill(false, levels) for _ in 1:num_players],
        equality_constraints = fill(nothing, num_players),
        inequality_constraints,
    )

    active_indices = num_players == 1 ? [1, PRIMAL_DIM + 3] : [1, 3]
    constraint_dim = num_players == 1 ? 2 * PRIMAL_DIM : 4
    inactive_indices = setdiff(collect(1:constraint_dim), active_indices)

    return (;
        problem,
        expected = reduce(vcat, expected_blocks),
        primal_dims,
        active_indices,
        inactive_indices,
    )
end

function assert_active_set(
    problem,
    primals,
    primal_dims,
    active_indices,
    inactive_indices,
)
    x_blocks = BlockArray(primals, primal_dims)
    θ_blocks = BlockArray(zeros(sum(problem.parameter_dims)), problem.parameter_dims)

    for player in 1:problem.num_players
        g = problem.inequality_constraints[player](x_blocks, θ_blocks)
        @test all(isapprox.(g[active_indices], 0.0; atol = 5e-7, rtol = 0.0))
        @test all(g[inactive_indices] .> 1e-5)
    end
end

function build_unconstrained_quadratic_problem(; levels::Int = 1)
    @assert levels >= 1
    n = PRIMAL_DIM
    x_template = BlockArray(zeros(n), [n])
    θ_template = BlockArray([0.0], [1])
    expected = [0.1, 0.6, 1.0, 0.4, 1.4]

    objective = let expected = expected
        (x, θ) -> sum((x[Block(1)][j] - expected[j])^2 for j in 1:n)
    end

    problem = ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences = [fill(objective, levels)],
        is_prioritized_constraint = [fill(false, levels)],
        equality_constraints = [nothing],
        inequality_constraints = [nothing],
    )

    return (; problem, expected, z₀ = zeros(n))
end

# `atol` is the primal accuracy demanded of the recovered solution. It is a
# keyword because that accuracy tracks the relaxation ρ linearly — the interior-point
# iterates stop on the relaxed complementarity `σ .* γ = ρ`, so the active constraints
# are only satisfied to O(ρ) and `x` is displaced correspondingly. Deeper
# hierarchies inherit a larger constant, so they are given a looser `atol`
# rather than being excluded from the known-solution assertions.
function test_known_solution_case(;
    num_players::Int,
    levels::Int,
    kind::Symbol,
    atol = IP_ATOL,
    kkt_system = reduced_kkt_system,
)
    case = build_benchmark_problem(; num_players, levels, kind)
    (; output, primals) = solve_with_pdip(case.problem, case.expected; kkt_system)

    @test output.status === :solved
    @test isapprox(primals, case.expected; atol, rtol = 0.0)
    @test output.norm_K₀ <= 1e-7
    assert_active_set(
        case.problem,
        primals,
        case.primal_dims,
        case.active_indices,
        case.inactive_indices,
    )
end

# ── The problems of ScholtesReducedGOOP.jl's tests ──────────────────────────────────────────────────

const θ0 = Float64[]

# The source's problem model, on ParametricGOOP --------------------------------------
function sproblem(; ni, objectives, equality = nothing, inequality = nothing, z0 = nothing)
    N = length(ni)
    z0v = isnothing(z0) ? zeros(sum(ni)) : Float64.(collect(z0))
    lift(f) = (x, θ) -> f(collect(x))
    eqs = isnothing(equality) ? fill(nothing, N) :
          [isempty(equality[i](z0v)) ? nothing : lift(equality[i]) for i in 1:N]
    ineqs = isnothing(inequality) ? fill(nothing, N) :
            [isempty(inequality[i](z0v)) ? nothing : lift(inequality[i]) for i in 1:N]
    goop = RG.ParametricGOOP(
        BlockArray(copy(z0v), ni),
        BlockArray(zeros(0), [0]);
        preferences = [Function[lift(f) for f in objectives[i]] for i in 1:N],
        is_prioritized_constraint = [fill(false, length(objectives[i])) for i in 1:N],
        equality_constraints = eqs,
        inequality_constraints = ineqs,
    )
    (; goop, z0 = z0v, ni, N, np = sum(ni), Ki = length.(objectives),
       mEi = goop.equality_dims, mIi = goop.inequality_dims)
end
sfeasible(p, z) = RG.is_feasible(p.goop, z, θ0)

# The Scholtes system, started by initialize_scholtes!, solved by ProjectedLM ---------------
scholtes_kkt(p; quasi = false, quasi_order = 2) = RG.generate_slacked_scholtes_reduced_kkt_system(
    p.goop; drop_higher_order_terms = quasi, quasi_order)

"The cold start (the source's own start) at the primal `x` and the stage ρ, or with `warm` the warm one."
scholtes_start(k, x, ρ; θ = zeros(k.parameter_dimension), warm = false, kwargs...) =
    RG.initialize_scholtes!(zeros(k.variable_dimension), k, θ, x, ρ; warm, kwargs...)

"""
`ProjectedLM` from the cold start at the primal `x` and the first stage of `ρ_schedule` (by default the
source's ladder 1, 0.1, …, 1e-10), with the other options given.
"""
function plm_solve(k, x; ρ_schedule = [10.0^(-j) for j in 0:10], kwargs...)
    RG.solve(RG.ProjectedLM(), k, θ0; z₀ = scholtes_start(k, x, first(ρ_schedule)),
        options = RG.ProjectedLMOptions(; ρ_schedule, kwargs...))
end
plm_solve(p::NamedTuple; kwargs...) = plm_solve(scholtes_kkt(p), p.z0; kwargs...)

"K_ρ(z) of any system."
function residual(k, z, ρ; θ = zeros(k.parameter_dimension))
    K_ρ = zeros(k.kkt_dimension)
    k.K_ρ!(K_ρ, z; θ, ρ)
    K_ρ
end

"∇K_ρ(z) of any system."
function jacobian(k, z, ρ; θ = zeros(k.parameter_dimension))
    J = copy(k.∇K_ρ!.result_buffer)
    k.∇K_ρ!(J, z; θ, ρ)
    J
end

"The slack targets at z, one per complementarity pair: G for σ, then ρ𝟙 − Gₘ ⊙ γₘ for σ̂ (the source's block a)."
function targets(k, z, ρ)
    a = zeros(length(k.slack_expressions))
    k.slack_targets!(a, z; θ = θ0, ρ)
    a
end

"Every complementarity pair's multiplier, γ then ϕ (the source's block b)."
multipliers(k, z) = z[[pair.multiplier for pair in k.complementarity_pairs]]

"The relaxed pairs (σ ⊙ γ + u = ρ𝟙) and the exact ones (σ̂ ⊙ ϕ = 0), by position in `complementarity_pairs`."
relaxed_pairs(k) = findall(pair -> pair.u_relaxation != 0, k.complementarity_pairs)
exact_pairs(k) = findall(pair -> pair.u_relaxation == 0, k.complementarity_pairs)

"The row of K_ρ that puts each pair's slack on its target: G − σ = 0, or ρ𝟙 − Gₘ ⊙ γₘ − σ̂ = 0."
slack_rows(k) = [only(findall(isequal(e - k.z_symbolic[pair.slack]), k.K_ρ_symbolic))
                 for (e, pair) in zip(k.slack_expressions, k.complementarity_pairs)]

"The Jacobian of the slack targets at z: the slack rows of ∇K_ρ, less their −1 in the slack's own column."
function target_jacobian(k, z, ρ)
    Ja = Matrix(jacobian(k, z, ρ)[slack_rows(k), :])
    for (j, pair) in enumerate(k.complementarity_pairs)
        Ja[j, pair.slack] += 1.0
    end
    Ja
end

"Whether a row of K_ρ holds a slack σ or σ̂: the slack and product rows do, the stationarity rows do not."
function slack_row_filter(k)
    slacks = Set(Symbolics.unwrap.(k.z_symbolic[[pair.slack for pair in k.complementarity_pairs]]))
    row -> any(in(slacks), Symbolics.get_variables(row))
end

# The source's test problems ------------------------------------------------------------
prob_A = sproblem(ni = [1], objectives = [[z -> z[1]^2]], inequality = [z -> [z[1] + 2.0]], z0 = [1.0])
prob_B = sproblem(ni = [1], objectives = [[z -> z[1]]], inequality = [z -> [z[1]]], z0 = [1.0])
prob_C = sproblem(ni = [1, 1],
    objectives = [[z -> z[1]^2, z -> (z[1] - (1.0 + 0.25 * z[2]))^2],
                  [z -> z[2]^2, z -> (z[2] - (-1.0 + 0.25 * z[1]))^2]],
    inequality = [z -> [z[1] + 2.0], z -> [z[2] + 2.0]], z0 = [0.0, 0.0])
box2(z) = [z[1] + 2.0, 2.0 - z[1], z[2] + 2.0, 2.0 - z[2]]
prob_D = sproblem(ni = [2], objectives = [[z -> (z[1] + z[2])^2, z -> (z[1] + z[2] - 1.0)^2]],
                  inequality = [box2], z0 = [0.0, 0.0])
seg_t(z) = (z[1] - z[2] + 1.0) / 2
seg_dist(z) = (t = clamp(seg_t(z), -1.0, 2.0); norm(z .- [t, 1 - t]))
prob_E = sproblem(ni = [2], objectives = [[z -> z[1]^2 + z[2]^2, z -> (z[1] - 0.5)^2,
                                           z -> (z[1] + z[2] - 1.0)^2]],
                  inequality = [box2], z0 = [0.0, 0.0])
prob_F = sproblem(ni = [2, 2],
    objectives = [[z -> z[1]^2 + 0.5 * z[4]^2, z -> (z[1] - 0.3 * z[3])^2 + z[2]^2,
                   z -> (z[2] + z[4] - 0.5)^2, z -> (z[1] + z[2] - 1.0)^2 + 0.2 * z[1] * z[3]],
                  [z -> z[3]^2 + z[4]^2, z -> (z[3] + 0.25 * z[1])^2,
                   z -> (z[4] - 0.4)^2 + z[3] * z[2], z -> (z[3] - z[4] + 0.2)^2]],
    equality = [z -> [z[1] + z[2] + 0.5 * z[3] - 0.7], z -> Float64[]],
    inequality = [z -> [z[1] + 3.0, 3.0 - z[1], z[2] + 3.0], z -> [z[3] + 3.0, 3.0 - z[4]]],
    z0 = [0.0, 0.0, 0.0, 0.0])
prob_G = sproblem(ni = [2], objectives = [[z -> z[2]^2, z -> z[1]^2,
                                           z -> (z[1] - 0.5)^2 + (z[2] + 0.25)^2 + 0.1 * z[1]^3]],
                  inequality = [box2], z0 = [0.0, 0.0])
prob_G_z = [(-2 + sqrt(4 + 1.2)) / 0.6, -0.25]
prob_single = sproblem(ni = [1, 1],
    objectives = [[z -> (z[1] - (1.0 + 0.25 * z[2]))^2], [z -> (z[2] - (-1.0 + 0.25 * z[1]))^2]],
    inequality = [z -> [z[1] + 2.0], z -> [z[2] + 2.0]], z0 = [0.0, 0.0])

# ── The solvers' problems ───────────────────────────────────────────────────────────────────────────

const SOLVERS = ((RG.PDIP(), RG.PDIPOptions), (RG.ProjectedLM(), RG.ProjectedLMOptions))

"A one-player, one-level problem without constraints: min `objective(x)` over x ∈ ℝ."
function free_problem(objective)
    RG.ParametricGOOP(BlockArray([0.0], [1]), BlockArray(Float64[], [0]);
        preferences = [Function[(x, θ) -> objective(x[1])]], is_prioritized_constraint = [[false]],
        equality_constraints = [nothing], inequality_constraints = [nothing])
end

"""
Two players, three levels: ‖x‖² on top, (b_i − 0.3)²/2, and the innermost `innermost`(a_i − 1), with the
coupled constraint a_i + 0.1 a_j ≤ 0.5, whose normal lies in the innermost objective's column space. The
equilibrium is unique, x* = (5/11, 0.3, 5/11, 0.3), and the constraint is active with γ_i = 6/11 > 0. The
lifted multipliers are not, so J is rank-deficient. (The benchmark problems violate the column-space rule,
and Algorithm 5.1 jams on them.)
"""
function coupled_problem(innermost)
    RG.ParametricGOOP(BlockArray(zeros(4), [2, 2]), BlockArray(Float64[], [0]);
        preferences = [Function[(x, t) -> sum(abs2, x), (x, t) -> (x[2i] - 0.3)^2 / 2, (x, t) -> innermost(x[2i - 1] - 1)]
                       for i in 1:2],
        is_prioritized_constraint = [[false, false, false] for _ in 1:2],
        equality_constraints = [nothing, nothing],
        inequality_constraints = [(x, t) -> [0.5 - x[1] - 0.1x[3]], (x, t) -> [0.5 - x[3] - 0.1x[1]]])
end
const COUPLED_EXPECTED = [5 / 11, 0.3, 5 / 11, 0.3]

"""
`PDIP` down a ρ chain from the primal start x = 0, one solve per stage, as experiments/MonteCarlo_RandomGOOP.jl
does: every stage runs to its inner tolerance (tol = 0). Returns the stages' outputs.
"""
function pdip_chain(k; kwargs...)
    θ, z, scale = zeros(k.parameter_dimension), zeros(length(k.primal_dims)), sqrt(k.kkt_dimension)
    outputs = []
    for ρ in 10.0 .^ (0:-1:-10)
        options = RG.PDIPOptions(; ρ_schedule = [ρ], max_inner_iters = 1000, tol = 0.0,
            tol_inner = min(ρ, 1e-2) * scale, record_convergence = true, kwargs...)
        push!(outputs, RG.solve(RG.PDIP(), k, θ; z₀ = z, options))
        z = last(outputs).z
    end
    outputs
end

# ── The complete systems' problems ──────────────────────────────────────────────────────────────────

exact(goop) = RG.generate_slacked_complete_kkt_system(goop; complementarity = :exact)
perturbed(goop) = RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed)
const BUILDERS = (exact, perturbed)

"The values of the auxiliary slacks' defining expressions (`slack_expressions`) at `y`."
function slack_values(kkt, y; theta = zeros(kkt.parameter_dimension))
    G = zeros(length(kkt.slack_expressions))
    kkt.slack_targets!(G, y; θ = theta, ρ = 0.0)
    G
end

"A one-player problem with one variable and the given levels."
function one_player_problem(preferences, flags; inequality = nothing)
    RG.ParametricGOOP(BlockArray([0.0], [1]), BlockArray(Float64[], [0]);
        preferences = [preferences], is_prioritized_constraint = [flags],
        equality_constraints = [nothing], inequality_constraints = [inequality])
end
