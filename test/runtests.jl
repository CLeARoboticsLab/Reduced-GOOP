include(joinpath(@__DIR__, "benchmark_problems.jl"))

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

@testset "Interior-Point Reduced & Quasi KKT Systems" begin
	#=
	Single-Player Box-Constrained Benchmark Family
	----------------------------------------------

	Decision variable:

	x in R^5

	Known solution:

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
	C_{i,2} - c_{i,2}(x) >= 0    inactive upper resource, residual slack 1
	c_{i,3}(x) - F_{i,3} >= 0    active lower resource
	c_{i,4}(x) - F_{i,4} >= 0    inactive lower resource, residual slack 1

	where:

	C_{i,1} = c_{i,1}(x*)
	C_{i,2} = c_{i,2}(x*) + 1
	F_{i,3} = c_{i,3}(x*)
	F_{i,4} = c_{i,4}(x*) - 1

	For the nonlinear constraint transform, these inactive constraints evaluate
	to exp(1) - 1, although the underlying affine residual slack is 1.

	This is a true generalized Nash setup: every player's feasible set depends
	on all players' decision variables. The active coordinate-1 and coordinate-3
	resource equations are player-specific and jointly nonsingular, so the known
	solution fixes the distribution across players, not just aggregate totals.

	Each player's single-level objective covers coordinates {1,2,3,4,5}.
	=#
	@testset "Three-Player Problems" begin

        # Testing Reduced KKT system 

		@testset "Single-Level" begin
			@testset "Reduced GOOP, Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 1, kind = :quadratic)
			end

			@testset "Reduced GOOP, Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 1, kind = :nonlinear)
			end
		end

		@testset "Bi-Level" begin
			@testset "Reduced GOOP, Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 2, kind = :quadratic)
			end

			@testset "Reduced GOOP, Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(;
					num_players = 3,
					levels = 2,
					kind = :nonlinear,
					atol = 5e-7,
				)
			end
		end

		@testset "Three-Level" begin
			@testset "Reduced GOOP, Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 3, kind = :quadratic)
			end

			@testset "Reduced GOOP, Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(;
					num_players = 3,
					levels = 3,
					kind = :nonlinear,
					atol = 5e-7,
				)
			end
		end

        # Testing Quasi-KKT system 

        @testset "Single-Level" begin
			@testset "Quasi GOOP, Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 1, kind = :quadratic, kkt_system = quasi_kkt_system)
			end

			@testset "Quasi GOOP, Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 1, kind = :nonlinear, kkt_system = quasi_kkt_system)
			end
		end

		@testset "Bi-Level" begin
			@testset "Quasi GOOP, Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 2, kind = :quadratic, kkt_system = quasi_kkt_system)
			end

			@testset "Quasi GOOP, Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(;
					num_players = 3,
					levels = 2,
					kind = :nonlinear,
					atol = 5e-7,
                    kkt_system = quasi_kkt_system
				)
			end
		end

		@testset "Three-Level" begin
			@testset "Quasi GOOP, Quadratic Objective, Linear Constraints" begin
				test_known_solution_case(; num_players = 3, levels = 3, kind = :quadratic, kkt_system = quasi_kkt_system)
			end

			@testset "Quasi GOOP, Nonlinear Objective, Nonlinear Constraints" begin
				test_known_solution_case(;
					num_players = 3,
					levels = 3,
					kind = :nonlinear,
					atol = 5e-7,
                    kkt_system = quasi_kkt_system
				)
			end
		end
	end
end

@testset "Reduced KKT dual-coordinate metadata" begin
	x_template = BlockArray(zeros(2), [2])
	θ_template = BlockArray(zeros(1), [1])
	problem = ReducedGOOP.ParametricGOOP(
		x_template,
		θ_template;
		preferences = [
			Function[
				(x, θ)->sum(abs2, x[Block(1)]),
				(x, θ)->sum(abs2, x[Block(1)] .- 1.0),
			],
		],
		is_prioritized_constraint = [[false, false]],
		equality_constraints = [(x, θ) -> [x[Block(1)][1]]],
		inequality_constraints = [(x, θ) -> [x[Block(1)][2]]],
	)
	kkt = reduced_kkt_system(problem)

	@test length(kkt.equality_constraint_dual_dims) == 2
	@test length(kkt.stationarity_dual_dims) == 2
	@test kkt.all_equality_stationarity_dual_dims ==
		  vcat(kkt.equality_constraint_dual_dims, kkt.stationarity_dual_dims)
	@test kkt.innermost_stationarity_dual_dims == kkt.stationarity_dual_dims
	@test isempty(
		intersect(
			kkt.all_equality_stationarity_dual_dims,
			kkt.inequality_constraint_dual_dims,
		),
	)
	other_dims = setdiff(
		1:kkt.variable_dimension,
		vcat(
			kkt.primal_dims,
			kkt.preference_slack_dims,
			kkt.interior_point_slack_dims,
			kkt.inequality_constraint_dual_dims,
			kkt.all_equality_stationarity_dual_dims,
		),
	)
	@test length(other_dims) == 1 # Block 10 (Φ) is not carried.
	@test isempty(intersect(kkt.all_equality_stationarity_dual_dims, other_dims))
end

@testset "Quasi order of the interior-point quasi system" begin
	# The quasi system truncates every Lagrangian term after `quasi_order` derivatives (default 2). A large
	# order truncates nothing, so its rows evaluate to the reduced system's (whose identically zero rows the
	# quasi system drops); order 1 keeps first-order information only, so no row depends on ψ.
	case = build_benchmark_problem(; num_players = 1, levels = 3, kind = :nonlinear)
	reduced = reduced_kkt_system(case.problem)
	quasi(order) = ReducedGOOP.generate_slacked_quasi_kkt_system(case.problem; quasi_order = order)
	@test_throws ArgumentError quasi(0)
	@test isequal(quasi(2).K_ρ_symbolic, quasi_kkt_system(case.problem).K_ρ_symbolic)
	full = quasi(10)
	kept = findall(row -> !iszero(row), reduced.K_ρ_symbolic)
	@test full.kkt_dimension == length(kept)
	z = 0.5 .+ 0.1 .* sin.(1:(reduced.variable_dimension))
	θ = zeros(reduced.parameter_dimension)
	K_ρ, K_ρ_full = zeros(reduced.kkt_dimension), zeros(full.kkt_dimension)
	reduced.K_ρ!(K_ρ, z; θ, ρ = 1e-3)
	full.K_ρ!(K_ρ_full, z; θ, ρ = 1e-3)
	@test K_ρ_full ≈ K_ρ[kept] rtol = 1e-12
	first_order = quasi(1)
	ψ = findall(v -> startswith(string(v), "ψ_"), first_order.z_symbolic)
	J = first_order.∇K_ρ!.result_buffer
	@test !isempty(ψ)
	@test all(j -> isempty(ReducedGOOP.SparseArrays.nzrange(J, j)), ψ)
end

@testset "FastDifferentiation Codegen Parity" begin
	# The :fast_differentiation codegen path must produce the same KKT residual
	# and Jacobian as the native Symbolics code generator: differentiation stays
	# in Symbolics and only the compiled evaluators change. The hierarchy below
	# includes an innermost prioritized inequality so its preference slacks (and
	# their rows) are exercised through both generators.
	num_players = 3
	kind = :quadratic
	primal_dims = fill(PRIMAL_DIM, num_players)
	parameter_dims = fill(1, num_players)
	x_template = BlockArray(zeros(sum(primal_dims)), primal_dims)
	θ_template = BlockArray(zeros(sum(parameter_dims)), parameter_dims)
	expected_blocks = [multiplayer_expected_block(player) for player in 1:num_players]
	raw_targets = [multiplayer_raw_target(player) for player in 1:num_players]
	coordinate_groups = preference_coordinate_groups(2)

	preferences = [
		Function[
			objective_from_target(
				kind,
				player,
				coordinate_groups[1],
				raw_targets[player],
			),
			objective_from_target(
				kind,
				player,
				coordinate_groups[2],
				raw_targets[player],
			),
			multiplayer_constraint(kind, player, expected_blocks),
		] for player in 1:num_players
	]
	problem = ReducedGOOP.ParametricGOOP(
		x_template,
		θ_template;
		preferences,
		is_prioritized_constraint = [[false, false, true] for _ in 1:num_players],
		equality_constraints = fill(nothing, num_players),
		inequality_constraints = fill(nothing, num_players),
	)

	native = ReducedGOOP.generate_slacked_reduced_kkt_system(problem; codegen = :native)
	fdgen = ReducedGOOP.generate_slacked_reduced_kkt_system(
		problem;
		codegen = :fast_differentiation,
	)
	fdgen_chunked = ReducedGOOP.generate_slacked_reduced_kkt_system(
		problem;
		codegen = :fast_differentiation,
		fd_codegen_chunk_size = 7,
	)

	@test fdgen.kkt_dimension == native.kkt_dimension
	@test fdgen.variable_dimension == native.variable_dimension
	@test fdgen_chunked.kkt_dimension == native.kkt_dimension
	@test fdgen_chunked.variable_dimension == native.variable_dimension

	reference_symbolic_jacobian = ReducedGOOP.SymbolicTracingUtils.sparse_jacobian(
		native.K_ρ_symbolic,
		native.z_symbolic,
	)
	optimized_symbolic_jacobian =
		ReducedGOOP._build_symbolics_sparse_jacobian(native.K_ρ_symbolic, native.z_symbolic)
	@test isequal(reference_symbolic_jacobian, optimized_symbolic_jacobian)

	n = native.variable_dimension
	m = sum(problem.parameter_dims)
	for scale in (0.3, 1.7), ρ in (0.1, 0.01)
		z = scale .* sin.(1:n)
		θ = scale .* cos.(1:m)

		F_native = zeros(native.kkt_dimension)
		F_fd = zeros(fdgen.kkt_dimension)
		native.K_ρ!(F_native, z; θ, ρ)
		fdgen.K_ρ!(F_fd, z; θ, ρ)
		@test isapprox(F_native, F_fd; atol = 1e-12, rtol = 1e-12)

		J_native = copy(native.∇K_ρ!.result_buffer)
		J_fd = copy(fdgen.∇K_ρ!.result_buffer)
		F_fd_chunked = zeros(fdgen_chunked.kkt_dimension)
		fdgen_chunked.K_ρ!(F_fd_chunked, z; θ, ρ)
		@test isapprox(F_native, F_fd_chunked; atol = 1e-12, rtol = 1e-12)

		J_fd_chunked = copy(fdgen_chunked.∇K_ρ!.result_buffer)
		native.∇K_ρ!(J_native, z; θ, ρ)
		fdgen.∇K_ρ!(J_fd, z; θ, ρ)
		fdgen_chunked.∇K_ρ!(J_fd_chunked, z; θ, ρ)
		@test isapprox(J_native, J_fd; atol = 1e-12, rtol = 1e-12)
		@test isapprox(J_native, J_fd_chunked; atol = 1e-12, rtol = 1e-12)
	end
end

@testset "Full Warmstart" begin
	problem = build_benchmark_problem(; num_players = 3, levels = 2, kind = :quadratic)
	kkt = reduced_kkt_system(problem.problem)
	θ = zeros(sum(problem.problem.parameter_dims))
	expected = reduce(vcat, [multiplayer_expected_block(player) for player in 1:3])
	options = default_pdip_options()

	cold = ReducedGOOP.solve(ReducedGOOP.PDIP(), kkt, θ; z₀ = expected, options)
	@test cold.status === :solved

	# Resolving from the previous full solution (primals + duals + slacks) must
	# converge almost immediately.
	warm = ReducedGOOP.solve(ReducedGOOP.PDIP(), kkt, θ; z₀ = cold.z, options)
	@test warm.status === :solved
	@test warm.newton_steps <= cold.newton_steps
	@test warm.newton_steps <= 14
	@test isapprox(warm.x, expected; atol = 1e-6)

	# A z₀ of neither primal nor full length is rejected.
	@test_throws ArgumentError ReducedGOOP.solve(
		ReducedGOOP.PDIP(),
		kkt,
		θ;
		z₀ = zeros(kkt.variable_dimension + 1),
		options,
	)
end

@testset "The Scholtes system accepts prioritized-constraint levels (smoke)" begin
	# Prioritized vectors introduce explicit preference slacks and their two
	# inequality blocks, including the corresponding Scholtes complementarity.
	num_players = 3
	primal_dims = fill(PRIMAL_DIM, num_players)
	x_template = BlockArray(zeros(sum(primal_dims)), primal_dims)
	θ_template = BlockArray(zeros(num_players), fill(1, num_players))
	expected_blocks = [multiplayer_expected_block(player) for player in 1:num_players]
	raw_targets = [multiplayer_raw_target(player) for player in 1:num_players]
	groups = preference_coordinate_groups(2)
	preferences = [
		Function[
			objective_from_target(:quadratic, player, groups[1], raw_targets[player]),
			objective_from_target(:quadratic, player, groups[2], raw_targets[player]),
			multiplayer_constraint(:quadratic, player, expected_blocks),
		] for player in 1:num_players
	]
	problem = ReducedGOOP.ParametricGOOP(
		x_template,
		θ_template;
		preferences,
		is_prioritized_constraint = [[false, false, true] for _ in 1:num_players],
		equality_constraints = fill(nothing, num_players),
		inequality_constraints = fill(nothing, num_players),
	)
	kkt = ReducedGOOP.generate_slacked_scholtes_reduced_kkt_system(problem)
	@test kkt.kind === :scholtes
	θ = zeros(num_players)
	options = ReducedGOOP.ProjectedLMOptions(max_inner_iters = 50)
	z₀ = ReducedGOOP.initialize_scholtes!(
		zeros(kkt.variable_dimension),
		kkt,
		θ,
		reduce(vcat, expected_blocks),
		first(options.ρ_schedule);
		warm = false,
	)
	output = ReducedGOOP.solve(ReducedGOOP.ProjectedLM(), kkt, θ; z₀, options)
	@test all(isfinite, output.z)
	@test isfinite(output.norm_K₀)
end

include(joinpath(@__DIR__, "solver.jl"))
include(joinpath(@__DIR__, "complete_kkt.jl"))
include(joinpath(@__DIR__, "scholtes.jl"))
