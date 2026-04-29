using Test

using BlockArrays: Block, BlockArray
using QuasiGOOP

const PATH_ATOL = 1e-7
const PRIMAL_DIM = 5

#=
Benchmark Problem Families
--------------------------

This file tests the complete-KKT MCP generator with PATH on small problems whose
solutions and active sets are known in closed form.

Notation: x_i is player i's decision vector, and x_{i,j} is the j-th decision
variable of player i. In code this is x[Block(i)][j].

All generated benchmark problems use n = 5 decision variables per player. The
hierarchy depth changes only how these five coordinates are assigned to
preference levels:

  Single-level: all coordinates are optimized at one level.
  Bilevel:      outer level controls coordinates 4:5,
                inner level controls coordinates 1:3.
  Trilevel:     outer level controls coordinate 5,
                middle level controls coordinates 3:4,
                inner level controls coordinates 1:2.

Preferences are stored in the GOOP convention [outermost, ..., innermost].
Because coordinate groups are disjoint, all hierarchy depths have the same known
primal solution for a given player/problem family, while still exercising the
recursive KKT construction.

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

function default_path_options(; verbose = false)
	return QuasiGOOP.PATHOptions(;
		convergence_tolerance = 1e-8,
		ϵ₀ = 0.0,
		cumulative_iteration_limit = 10000,
		proximal_perturbation = 1e-2,
		major_iteration_limit = 1000,
		minor_iteration_limit = 1000,
		nms_initial_reference_factor = 100,
		nms_maximum_watchdogs = 100,
		nms_memory_size = 100,
		nms_mstep_frequency = 100,
		lemke_start_type = "advanced",
		lemke_rank_deficiency_iterations = 50,
		restart_limit = 120,
		gradient_step_limit = 120,
		use_basics = true,
		use_start = true,
		verbose,
	)
end

complete_mcp_system(problem) = QuasiGOOP.generate_mcp_complete_kkt_system(problem)

function solve_with_path(problem, expected_primals; z₀ = nothing)
	mcp = complete_mcp_system(problem)
	initial_guess = if isnothing(z₀)
		guess = zeros(length(mcp.lower_bounds))
		guess[1:length(expected_primals)] .= expected_primals
		guess
	else
		z₀
	end

	output = QuasiGOOP.solve(
		QuasiGOOP.PATHSolver(),
		mcp,
		zeros(sum(problem.parameter_dims));
		z₀ = initial_guess,
		options = default_path_options(),
	)

	return (; output, primals = output.z[1:length(expected_primals)], mcp)
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
			(x, θ) -> sum(cosh(x[Block(player)][j] - raw_target[j]) - 1.0 for j in coordinates)
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
#   C_{i,2} - c_{i,2}(x) >= 0    inactive upper resource, slack 1
#   c_{i,3}(x) - F_{i,3} >= 0    active lower resource
#   c_{i,4}(x) - F_{i,4} >= 0    inactive lower resource, slack 1
#
# Constants C and F are chosen from x*. The active equations are nonsingular
# across the three players because the coupling matrix has 1 on the diagonal
# and 0.25 off diagonal, so the active resources pin the intended distribution,
# not only the aggregate total.
function coupled_coordinate(x, player::Int, coordinate::Int, num_players::Int)
	coupling_weight = 0.25
	return x[Block(player)][coordinate] +
		coupling_weight * sum(x[Block(other)][coordinate] for other in 1:num_players if other != player)
end

function multiplayer_constraint(kind::Symbol, player::Int, expected_blocks)
	num_players = length(expected_blocks)
	coupled_expected(coordinate) =
		expected_blocks[player][coordinate] +
		0.25 * sum(
			expected_blocks[other][coordinate] for other in 1:num_players if other != player
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
		[
			multiplayer_constraint(kind, player, expected_blocks)
			for player in 1:num_players
		]
	end

	problem = QuasiGOOP.ParametricGOOP(
		x_template,
		θ_template;
		preferences,
		is_prioritized_constraint = [fill(false, levels) for _ in 1:num_players],
		equality_constraints = fill(nothing, num_players),
		inequality_constraints,
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
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

function assert_active_set(problem, primals, primal_dims, active_indices, inactive_indices)
	x_blocks = BlockArray(primals, primal_dims)
	θ_blocks = BlockArray(zeros(sum(problem.parameter_dims)), problem.parameter_dims)

	for player in 1:problem.num_players
		g = problem.inequality_constraints[player](x_blocks, θ_blocks)
		@test all(isapprox.(g[active_indices], 0.0; atol = 5e-7, rtol = 0.0))
		@test all(g[inactive_indices] .> 1e-5)
	end
end

function test_known_solution_case(; num_players::Int, levels::Int, kind::Symbol)
	case = build_benchmark_problem(; num_players, levels, kind)
	(; output, primals) = solve_with_path(case.problem, case.expected)

	@test Int(output.status) == 1
	@test isapprox(primals, case.expected; atol = PATH_ATOL, rtol = 0.0)
	@test output.info.residual <= 1e-7
	assert_active_set(
		case.problem,
		primals,
		case.primal_dims,
		case.active_indices,
		case.inactive_indices,
	)
end

@testset "PATH Complete MCP" begin
	#=
	Baseline MCP Sanity Check
	-------------------------

	Problem:

	  min_x 0
	  subject to x >= 0,  x in R^4

	Every feasible x is optimal. PATH returns the canonical complementarity
	solution tested here:

	  x* = [0, 0, 0, 0]

	This case is intentionally degenerate and only verifies that the singleton
	preference/base-case complete MCP path is usable.
	=#
	@testset "Baseline: min 0 subject to x >= 0" begin
		n = 4
		x_template = BlockArray(zeros(n), [n])
		θ_template = BlockArray([0.0], [1])

		zero_objective(x, θ) = 0.0 * sum(x[Block(1)])
		nonnegative_constraint(x, θ) = x[Block(1)]

		problem = QuasiGOOP.ParametricGOOP(
			x_template,
			θ_template;
			preferences = [[zero_objective]],
			is_prioritized_constraint = [[false]],
			equality_constraints = [nothing],
			inequality_constraints = [nonnegative_constraint],
			shared_equality_constraint = nothing,
			shared_inequality_constraint = nothing,
		)

		expected = zeros(n)
		z₀ = [expected; zeros(n)]
		(; output, primals) = solve_with_path(problem, expected; z₀)

		@test Int(output.status) == 1
		@test isapprox(primals, expected; atol = PATH_ATOL, rtol = 0.0)
		@test output.info.residual <= 1e-8
		@test all(isapprox.(primals, 0.0; atol = PATH_ATOL, rtol = 0.0))
	end

	#=
	Single-Player Box-Constrained Benchmark Family
	----------------------------------------------

	Decision variable:

	  x in R^5

	Known solution, shared by single-level, bilevel, and trilevel variants:

	  x* = [0.0, 0.6, 1.0, 0.4, 1.4]

	Raw objective target:

	  t = [-0.4, 0.6, 1.4, 0.4, 1.4]

	Constraints:

	  Quadratic/linear variant:
	    x - lower >= 0
	    upper - x >= 0

	  Nonlinear/nonlinear variant:
	    exp.(x - lower) - 1 >= 0
	    exp.(upper - x) - 1 >= 0

	  lower = [0, 0, 0, 0, 0]
	  upper = [2, 2, 1, 2, 2]

	Intentional active set at x*:

	  active:   x_1 = lower_1 = 0,  x_3 = upper_3 = 1
	  inactive: all other lower and upper bounds

	Hierarchy-specific objectives:

	  Single-level:
	    min_x objective over coordinates {1,2,3,4,5}

	  Bilevel:
	    outer: min objective over coordinates {4,5}
	    subject to x in argmin objective over coordinates {1,2,3}

	  Trilevel:
	    outer: min objective over coordinate {5}
	    subject to x in argmin objective over coordinates {3,4}
	      subject to x in argmin objective over coordinates {1,2}

	The expected solution is the projection of t onto the box constraints.
	=#
	@testset "Single-Player Problems" begin
		@testset "Single-Level" begin
			@testset "Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 1, levels = 1, kind = :quadratic)
			end

			@testset "Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(; num_players = 1, levels = 1, kind = :nonlinear)
			end
		end

		@testset "Bilevel" begin
			@testset "Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 1, levels = 2, kind = :quadratic)
			end

			@testset "Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(; num_players = 1, levels = 2, kind = :nonlinear)
			end
		end

		@testset "Trilevel" begin
			@testset "Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 1, levels = 3, kind = :quadratic)
			end

			@testset "Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(; num_players = 1, levels = 3, kind = :nonlinear)
			end
		end
	end

	#=
	Three-Player Coupled Generalized Nash Benchmark Family
	-----------------------------------------------------

	Decision variables:

	  x_i in R^5,  i = 1,2,3

	Known solution:

	  x_1* = [0.40, 0.90, 0.55, 1.20, 0.80]
	  x_2* = [0.45, 1.00, 0.60, 1.30, 0.90]
	  x_3* = [0.50, 1.10, 0.65, 1.40, 1.00]

	Player i's raw objective target is:

	  t_i = x_i* + [0.6, 0, -0.4, 0, 0]

	So each player would prefer coordinate 1 to be larger and coordinate 3 to be
	smaller than the final solution. The coupled constraints create the active
	resources that force the equilibrium to x*.

	Coupled resource expression:

	  c_{i,j}(x) = x_{i,j} + 0.25 * sum(x_{k,j} for k != i)

	Player i constraints:

	  C_{i,1} - c_{i,1}(x) >= 0    active upper resource
	  C_{i,2} - c_{i,2}(x) >= 0    inactive upper resource, slack 1
	  c_{i,3}(x) - F_{i,3} >= 0    active lower resource
	  c_{i,4}(x) - F_{i,4} >= 0    inactive lower resource, slack 1

	where:

	  C_{i,1} = c_{i,1}(x*)
	  C_{i,2} = c_{i,2}(x*) + 1
	  F_{i,3} = c_{i,3}(x*)
	  F_{i,4} = c_{i,4}(x*) - 1

	This is a true generalized Nash setup: every player's feasible set depends
	on all players' decision variables. The active coordinate-1 and coordinate-3
	resource equations are player-specific and jointly nonsingular, so the known
	solution fixes the distribution across players, not just aggregate totals.

	Hierarchy-specific objectives use the same coordinate groups as the
	single-player family:

	  Single-level: {1,2,3,4,5}
	  Bilevel:      outer {4,5}, inner {1,2,3}
	  Trilevel:     outer {5}, middle {3,4}, inner {1,2}
	=#
	@testset "Three-Player Problems" begin
		@testset "Single-Level" begin
			@testset "Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 1, kind = :quadratic)
			end

			@testset "Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 1, kind = :nonlinear)
			end
		end

		@testset "Bilevel" begin
			@testset "Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 2, kind = :quadratic)
			end

			@testset "Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 2, kind = :nonlinear)
			end
		end

		@testset "Trilevel" begin
			@testset "Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 3, kind = :quadratic)
			end

			@testset "Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 3, kind = :nonlinear)
			end
		end
	end
end
