include(joinpath(@__DIR__, "benchmark_problems.jl"))

# `atol` is the primal accuracy demanded of the recovered solution. It is a
# keyword because that accuracy tracks ϵ₀ linearly — the interior-point iterates
# stop on the ϵ-relaxed complementarity `σ .* γ = ϵ`, so the active constraints
# are only satisfied to O(ϵ) and `x` is displaced correspondingly. Deeper
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
	(; output, primals) = solve_with_interior_point(case.problem, case.expected; kkt_system)

	@test output.status === :solved
	@test isapprox(primals, case.expected; atol, rtol = 0.0)
	@test output.kkt_error <= 1e-7
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

@testset "Interior-Point Complete KKT Smoke" begin
	case = build_unconstrained_quadratic_problem()

	complete = solve_with_interior_point(
		case.problem,
		case.expected;
		z₀ = case.z₀,
		kkt_system = complete_kkt_system,
	)
	reduced = solve_with_interior_point(
		case.problem,
		case.expected;
		z₀ = case.z₀,
		kkt_system = reduced_kkt_system,
	)

	@test complete.output.status === :solved
	@test reduced.output.status === :solved
	@test complete.output.kkt_error <= 1e-8
	@test reduced.output.kkt_error <= 1e-8
	@test isapprox(complete.primals, case.expected; atol = IP_ATOL, rtol = 0.0)
	@test isapprox(reduced.primals, case.expected; atol = IP_ATOL, rtol = 0.0)
	@test isapprox(reduced.primals, complete.primals; atol = IP_ATOL, rtol = 0.0)
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
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
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

@testset "FastDifferentiation Codegen Parity" begin
	# The :fast_differentiation codegen path must produce the same KKT residual
	# and Jacobian as the native Symbolics code generator: differentiation stays
	# in Symbolics and only the compiled evaluators change. The hierarchy below
	# includes an innermost prioritized inequality so the ifelse-based penalty
	# terms (and their derivatives) are exercised through both generators.
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
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)

	native = ReducedGOOP.generate_slacked_reduced_kkt_system(problem)
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
		native.F_symbolic,
		native.z_symbolic,
	)
	optimized_symbolic_jacobian =
		ReducedGOOP._build_symbolics_sparse_jacobian(native.F_symbolic, native.z_symbolic)
	@test isequal(reference_symbolic_jacobian, optimized_symbolic_jacobian)

	n = native.variable_dimension
	m = sum(problem.parameter_dims)
	for scale in (0.3, 1.7), (ϵ, η) in ((0.1, 1e-6), (0.01, 0.0))
		z = scale .* sin.(1:n)
		θ = scale .* cos.(1:m)

		F_native = zeros(native.kkt_dimension)
		F_fd = zeros(fdgen.kkt_dimension)
		native.F!(F_native, z; θ, ϵ, η)
		fdgen.F!(F_fd, z; θ, ϵ, η)
		@test isapprox(F_native, F_fd; atol = 1e-12, rtol = 1e-12)

		J_native = copy(native.∇F_z!.result_buffer)
		J_fd = copy(fdgen.∇F_z!.result_buffer)
		F_fd_chunked = zeros(fdgen_chunked.kkt_dimension)
		fdgen_chunked.F!(F_fd_chunked, z; θ, ϵ, η)
		@test isapprox(F_native, F_fd_chunked; atol = 1e-12, rtol = 1e-12)

		J_fd_chunked = copy(fdgen_chunked.∇F_z!.result_buffer)
		native.∇F_z!(J_native, z; θ, ϵ, η)
		fdgen.∇F_z!(J_fd, z; θ, ϵ, η)
		fdgen_chunked.∇F_z!(J_fd_chunked, z; θ, ϵ, η)
		@test isapprox(J_native, J_fd; atol = 1e-12, rtol = 1e-12)
		@test isapprox(J_native, J_fd_chunked; atol = 1e-12, rtol = 1e-12)
	end
end

function klu_test_options(;
	linesearch = :fraction_to_boundary,
	linear_solver = :svd,
	reuse_factorization_iters = 0,
	use_marquardt_scaling = false,
	η₀ = 0.0,
)
	return ReducedGOOP.InteriorPointOptions(;
		# The residual of the ϵ-relaxed system floors at ≈ 1.22·ϵ on this
		# benchmark, and the backtracking mode never shrinks ϵ across outer
		# iterations, so ϵ must sit strictly below tol for `kkt_error ≤ tol`
		# to be attainable (ϵ₀ = :auto would set ϵ = tol on a warmstart).
		tol = 1e-7,
		η₀,
		ϵ₀ = 1e-8,
		max_inner_iters = 5000,
		max_outer_iters = 20,
		tightening_rate = 2.0,
		loosening_rate = 0.5,
		min_stepsize = 1e-20,
		linesearch,
		record_convergence = false,
		record_condition_number = false,
		eta_retry_growth = 0.3,
		tsvd_threshold = 0.0,
		use_marquardt_scaling,
		linear_solver,
		reuse_factorization_iters,
		verbose = false,
	)
end

@testset "KLU Augmented Linear Solver" begin
	problem = build_benchmark_problem(; num_players = 3, levels = 2, kind = :quadratic)
	kkt = reduced_kkt_system(problem.problem)
	θ = zeros(sum(problem.problem.parameter_dims))

	@testset "Direction equivalence with the SVD Tikhonov step" begin
		# The augmented quasi-definite system [[I J]; [Jᵀ -ηI]] must reproduce the
		# dense-SVD step δz = -V diag(S/(S²+η)) Uᵀ F exactly (tsvd_threshold = 0,
		# no Marquardt scaling).
		n = kkt.variable_dimension
		m = kkt.kkt_dimension
		z = 0.3 .* sin.(1:n)
		ϵ = 0.1

		F = zeros(m)
		kkt.F!(F, z; θ, ϵ, η = 0.0)
		J = kkt.∇F_z!.result_buffer
		kkt.∇F_z!(J, z; θ, ϵ, η = 0.0)

		# High-η cases lock in the √η-balanced augmented scaling: the naive
		# [I J; Jᵀ -ηI] form loses the tiny ~1/η step to cancellation there.
		for η in (1e-6, 1e-2, 1e2, 1e6)
			cache = ReducedGOOP._build_augmented_kkt_cache(J, m, n)
			ReducedGOOP._update_augmented_kkt!(cache, J, η)
			δz_klu = zeros(n)
			ReducedGOOP._solve_augmented!(δz_klu, cache, F)

			Jsvd = ReducedGOOP.LinearAlgebra.svd(Matrix(J))
			δz_svd = -Jsvd.V * ((Jsvd.S ./ (Jsvd.S .^ 2 .+ η)) .* (Jsvd.U' * F))

			@test isapprox(δz_klu, δz_svd; rtol = 1e-8)
		end
	end

	@testset "Solve-level agreement between :svd and :klu" begin
		# Backtracking is the MPC configuration; without Marquardt scaling the
		# fraction-to-boundary :svd baseline is not a meaningful reference.
		expected = reduce(vcat, [multiplayer_expected_block(player) for player in 1:3])
		outputs = map((:svd, :klu)) do linear_solver
			ReducedGOOP.solve(
				ReducedGOOP.InteriorPoint(),
				kkt,
				θ;
				z₀ = expected,
				options = klu_test_options(;
					linesearch = :backtracking,
					linear_solver,
					η₀ = 1e-6,
				),
			)
		end
		svd_output, klu_output = outputs
		@test svd_output.status === :solved
		@test klu_output.status === :solved
		@test isapprox(
			klu_output.z[kkt.primal_dims],
			svd_output.z[kkt.primal_dims];
			atol = 1e-6,
		)
		@test isapprox(klu_output.z[kkt.primal_dims], expected; atol = 1e-6)
	end

	@testset "Backtracking :klu with factorization reuse" begin
		expected = reduce(vcat, [multiplayer_expected_block(player) for player in 1:3])
		for reuse_factorization_iters in (0, 2)
			output = ReducedGOOP.solve(
				ReducedGOOP.InteriorPoint(),
				kkt,
				θ;
				z₀ = expected,
				options = klu_test_options(;
					linesearch = :backtracking,
					linear_solver = :klu,
					reuse_factorization_iters,
					η₀ = 1e-6,
				),
			)
			@test output.status === :solved
			@test isapprox(output.z[kkt.primal_dims], expected; atol = 1e-6)
		end
	end

	@testset "Singular numeric refactorization rebuilds its cache" begin
		# Reproduce the cache state that matters here: a successful numeric
		# factorization followed by a singular in-place `klu!` update. The
		# retry must discard that failed numeric object, rebuild at escalated η,
		# and avoid the dense-SVD fallback.
		J = ReducedGOOP.SparseArrays.sparse([1], [1], [1.0], 1, 1)
		cache = ReducedGOOP._build_augmented_kkt_cache(J, 1, 1)
		ReducedGOOP._update_augmented_kkt!(cache, J, 1e-6)
		δz = zeros(1)
		F = ones(1)
		ReducedGOOP._solve_augmented!(δz, cache, F)

		# Make the next numeric update singular while retaining the same sparse
		# pattern. The retry's η update restores a nonsingular diagonal system.
		ReducedGOOP.SparseArrays.nonzeros(J) .= 0.0
		ReducedGOOP.SparseArrays.nonzeros(cache.K) .= 0.0
		singular_retries = Ref(0)
		svd_fallbacks = Ref(0)
		η_used = ReducedGOOP._klu_step_with_fallback!(
			δz,
			cache,
			J,
			F,
			1e-6,
			1e2,
			false;
			singular_retry_counter = singular_retries,
			svd_fallback_counter = svd_fallbacks,
		)

		@test singular_retries[] == 1
		@test svd_fallbacks[] == 0
		@test η_used ≈ 1e-4
		@test all(isfinite, δz)
	end

	@testset "SVD-only features rejected on :klu" begin
		@test_throws ArgumentError ReducedGOOP.solve(
			ReducedGOOP.InteriorPoint(),
			kkt,
			θ;
			options = klu_test_options(;
				linear_solver = :klu,
				use_marquardt_scaling = true,
			),
		)
	end
end

@testset "Full Warmstart" begin
	problem = build_benchmark_problem(; num_players = 3, levels = 2, kind = :quadratic)
	kkt = reduced_kkt_system(problem.problem)
	θ = zeros(sum(problem.problem.parameter_dims))
	expected = reduce(vcat, [multiplayer_expected_block(player) for player in 1:3])
	options =
		klu_test_options(; linesearch = :backtracking, linear_solver = :klu, η₀ = 1e-6)

	cold = ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt, θ; z₀ = expected, options)
	@test cold.status === :solved

	# Resolving from the previous full solution (primals + duals + slacks) must
	# converge almost immediately.
	warm = ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt, θ; z₀ = cold.z, options)
	@test warm.status === :solved
	@test warm.total_iters <= cold.total_iters
	@test warm.total_iters <= 15
	@test isapprox(warm.z[kkt.primal_dims], expected; atol = 1e-6)

	# A z₀ of neither primal nor full length is rejected.
	@test_throws ArgumentError ReducedGOOP.solve(
		ReducedGOOP.InteriorPoint(),
		kkt,
		θ;
		z₀ = zeros(kkt.variable_dimension + 1),
		options,
	)
end
