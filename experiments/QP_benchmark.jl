module QP_benchmark

using QuasiGOOP

using ParametricMCPs
using LinearAlgebra: I, Diagonal, norm, pinv, qr, rank
using Symbolics
using SymbolicTracingUtils
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes, blockedrange
using NonlinearSolve
using InvertedIndices: Not
using SparseArrays: sparse, SparseMatrixCSC
using Random

function print_preference_values(label, num_players, n, problem, z, θ)
	println("[$(label)] Preference values at solution:")
	for (player_idx, player_preferences) in enumerate(problem.preferences)
		values = [pref(z[1:(num_players*n)], θ) for pref in player_preferences]
		println("  player $(player_idx): $(round.(values, digits = 6))")
	end
end

function get_setup(n, num_players, mₑ, mᵢ; num_preferences = 2, r = 1)
	primal_dimensions = fill(n, num_players)
	parameter_dimensions = fill(1, num_players)
	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
	dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

	preferences = [Vector{Function}(undef, num_preferences) for _ in 1:num_players]
	for i in 1:num_players, j in 1:num_preferences
        Qk_local = rand_psd(n * num_players, r)
        q_local = Qk_local * randn(n * num_players) # q ∈ Col(Qk) for boundedness
		preferences[i][j] = let Qk = Qk_local, q = q_local
			(z, θ) -> 0.5 * z' * Qk * z + q' * z 
		end
	end

	is_prioritized_constraint = [
		[false for j in 1:num_preferences] for i in 1:num_players
	]

	equality_constraints = Vector{Function}(undef, num_players)
	for i in 1:num_players
		Hⁱ1 = rand(mₑ, n) # n > mₑ for full row rank
		Hⁱ2 = rand(mₑ, n)
		hⁱ = rand(mₑ)
		equality_constraints[i] = let H = hcat(Hⁱ1, Hⁱ2), h = hⁱ
			(z, θ) -> H * z - h
		end
	end

	inequality_constraints = [
		function (z, θ)
			vcat(
				z[Block(i)] .+ 1.0,  # z ≥ - 1.0
				-z[Block(i)] .+ 1.0, # z ≤ 1.0
			)
		end for i in 1:num_players
	]


	QuasiGOOP.ParametricGOOP(
		dummy_primals,
		dummy_parameters;
		preferences,
		is_prioritized_constraint,
		equality_constraints,
		inequality_constraints = [nothing for _ in 1:num_players],
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)
end


function demo(; rng_seed = 123)
	Random.seed!(rng_seed)

	# Quadratic GOOP Problem setup
	num_players = 2
	num_preferences = 3
	n = 20 # x primal dimension (per player)
	mₑ = 5 # equality constraint dimension
	mᵢ = 0 # inequality constraint dimension
	parameters = BlockArray(zeros(sum(fill(1, num_players))), fill(1, num_players))
	num_instances = 1

	run_id = "run_1"
	linesearch = :backtracking # :backtracking, :fraction_to_boundary
	verbose = false
	tol = 1e-3
	ϵ₀ = 1e-2
	η₀ = 0.0
	max_inner_iters = 50
	max_outer_iters = 2
	min_stepsize = 1e-5

	# Create file dir
	run_dir = joinpath("data", "QP_benchmark", run_id)
	data_dir = joinpath(run_dir, "data")
	problem_data_dir = joinpath(data_dir, "problem")
	solution_data_dir = joinpath(problem_data_dir, "solution")
	histories_data_dir = joinpath(data_dir, "histories")
	plots_dir = joinpath(run_dir, "plots")
	trajectory_plots_dir = joinpath(plots_dir, "trajectories")
	convergence_plots_dir = joinpath(plots_dir, "convergence")
	for dir in (
		problem_data_dir,
		solution_data_dir,
		histories_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
	)
		mkpath(dir)
	end

	instance_problem_data = Dict{String, Any}[]
	# kkt_error_histories_per_eps = Dict(ϵ => Vector{Vector{Float64}}() for ϵ in epsilon_schedule)
	solved_attempts = 0
	total_attempts = 0

	while solved_attempts < num_instances
		total_attempts += 1
		println(
			"solved $(solved_attempts)/$(num_instances), attempt $(total_attempts): ",
		)

		# Generate problem instance 
		problem = get_setup(n, num_players, mₑ, mᵢ; num_preferences)

		# Solve problem with reduced GOOP KKT system 
		reduced_kkt_system = QuasiGOOP.generate_slacked_reduced_kkt_system(problem)
		println("[Reduced] KKT Dimension: ", reduced_kkt_system.kkt_dimension)
		println("[Reduced] Variable Dimension: ", reduced_kkt_system.variable_dimension)
		convergence_log_reduced_system = Dict{String, Any}()
		elapsed_time = @elapsed begin
			(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				reduced_kkt_system,
				parameters;
				tol,
				η₀,
				ϵ₀,
				max_inner_iters,
				max_outer_iters,
				min_stepsize,
				z₀ = zeros(reduced_kkt_system.primal_dims),
				verbose,
				convergence_log = convergence_log_reduced_system,
				linesearch, # :backtracking, :fraction_to_boundary
			)
		end
		println("[Reduced] status = $(status)")
		println("[Reduced] Primal solution: $(round.(z[1:num_players * n], digits = 3))")
		print_preference_values("Reduced", num_players, n, problem, z, parameters)
		println("[Reduced] kkt_error = $(kkt_error)")

		# Solve problem with complete GOOP KKT system
		complete_kkt_system = QuasiGOOP.generate_slacked_complete_kkt_system(problem)
		println("[Complete] KKT Dimension: ", complete_kkt_system.kkt_dimension)
		println("[Complete] Variable Dimension: ", complete_kkt_system.variable_dimension)
		convergence_log_complete_system = Dict{String, Any}()
		elapsed_time = @elapsed begin
			(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				complete_kkt_system,
				parameters;
				tol,
				η₀,
				ϵ₀,
				max_inner_iters,
				max_outer_iters,
				min_stepsize,
				z₀ = zeros(complete_kkt_system.primal_dims),
				verbose,
				convergence_log = convergence_log_complete_system,
				linesearch, # :backtracking, :fraction_to_boundary
			)
		end
		println("[Complete] status = $(status)")
		println("[Complete] Primal solution: $(round.(z[1:num_players * n], digits = 3))")
		print_preference_values("Complete", num_players, n, problem, z, parameters)
		println("[Complete] kkt_error = $(kkt_error)")
		# Solve problem with ParametricMCPs (TODO)


		# Save problem and solution data 


		# Save KKT error histories and plot convergence
        Main.@infiltrate
		solved_attempts += 1
	end


end # end demo





# Helper functions 
function rand_psd(n, r)
	# n: primal dimension, r: matrix rank (<=n)
	R = randn(r, n);
	R' * R;
end

function generate_block_quadratic_problem(n, num_players; r = 1)
	# Build num_players PSD blocks and assemble a block-diagonal matrix.
	@assert num_players >= 1 "num_players must be at least 1"
	Qs = [rand_psd(n, r) for _ in 1:num_players]
	Qblk = sparse(cat(Qs...; dims = (1, 2))) # block matrix
	total_n = n * num_players

	# Random off-diagonal blocks; keep diagonal player-blocks unchanged.
	Qoff = randn(total_n, total_n)
	for i in 1:num_players
		idx = ((i-1)*n+1):(i*n)
		@views Qoff[idx, idx] .= 0.0
	end
	Qₖ = Qblk + sparse(Qoff)
	x = randn(total_n)
	q = Qₖ * x # ∈ Col(Qₖ)
	return Qs, Qblk, Qₖ, q
end

function rand_full_row_rank_matrix(m, n; coupling_scale = 0.5, min_sv = 0.2, max_sv = 2.0)
	@assert m <= n "Need m <= n for full row rank"
	@assert 0.0 < min_sv <= max_sv "Singular-value bounds must satisfy 0 < min_sv <= max_sv"

	# Build a guaranteed full-row-rank core, then mix rows/columns with random orthogonal bases.
	U = Matrix(qr(randn(m, m)).Q)
	V = Matrix(qr(randn(n, n)).Q)
	s = min_sv .+ (max_sv - min_sv) .* rand(m)
	core = hcat(Diagonal(s), coupling_scale * randn(m, n - m))
	return U * core * V'
end

function full_row_rank_constraints(n, num_players, mₑ, mᵢ; feasible_margin = 1e-2)
	@assert num_players >= 1 "num_players must be at least 1"
	m = mₑ + mᵢ
	@assert m <= n "Need mₑ + mᵢ <= n  for full row rank"
	@assert feasible_margin > 0 "feasible_margin must be positive"

	# Build per-player diagonal blocks first (each m x n).
	Ablocks = Vector{Matrix{Float64}}(undef, num_players)
	for p in 1:num_players
		Araw = rand_full_row_rank_matrix(
			m,
			n;
			coupling_scale = 0.35 + 0.5 * rand(),
			min_sv = 0.15,
			max_sv = 2.5,
		)

		# Randomize which constraint rows are treated as equalities vs inequalities.
		row_perm = randperm(m)
		eq_idx = row_perm[1:mₑ]
		ineq_idx = row_perm[(mₑ+1):m]
		Aₑ = Araw[eq_idx, :]
		Aᵢ = Araw[ineq_idx, :]
		Ablocks[p] = vcat(Aₑ, Aᵢ)
	end

	# Block-diagonal matrix with A blocks on the diagonal.
	Hblk = sparse(cat(Ablocks...; dims = (1, 2)))

	# Same diagonal blocks as Hblk, random off-diagonal blocks.
	total_m = m * num_players
	total_n = n * num_players
	Hoff = randn(total_m, total_n)
	for p in 1:num_players
		row_idx = ((p-1)*m+1):(p*m)
		col_idx = ((p-1)*n+1):(p*n)
		@views Hoff[row_idx, col_idx] .= 0.0
	end
	H = Hblk + sparse(Hoff)

	@assert rank(H) == total_m "H must have full row rank"

	# Per-player A/Ae/Ai as row-slices of H (same column dimension as H).
	As = Vector{SparseMatrixCSC{Float64, Int}}(undef, num_players)
	Aₑs = Vector{SparseMatrixCSC{Float64, Int}}(undef, num_players)
	Aᵢs = Vector{SparseMatrixCSC{Float64, Int}}(undef, num_players)
	for p in 1:num_players
		row_base = (p - 1) * m
		As[p] = H[(row_base+1):(row_base+m), :]
		Aₑs[p] = H[(row_base+1):(row_base+mₑ), :]
		Aᵢs[p] = H[(row_base+mₑ+1):(row_base+m), :]
	end

	# Build h so the constraint set is feasible by construction.
	# For witness z_feas:
	#   A_e z_feas - h_e = 0
	#   A_i z_feas - h_i >= feasible_margin
	z_feas = randn(total_n)
	h = Vector(H * z_feas)
	for p in 1:num_players
		if mᵢ == 0
			continue
		end
		row_base = (p - 1) * m
		ineq_rows = (row_base+mₑ+1):(row_base+m)
		h[ineq_rows] .-= feasible_margin .+ abs.(randn(mᵢ))
	end

	hₑs = Vector{Vector{Float64}}(undef, num_players)
	hᵢs = Vector{Vector{Float64}}(undef, num_players)
	for p in 1:num_players
		row_base = (p - 1) * m
		hₑs[p] = h[(row_base+1):(row_base+mₑ)]
		hᵢs[p] = h[(row_base+mₑ+1):(row_base+m)]
	end

	# Sanity check at the witness point.
	for p in 1:num_players
		@assert norm(Aₑs[p] * z_feas - hₑs[p], Inf) <= 1e-8 "Equality constraints not feasible at witness point"
		if mᵢ > 0
			@assert minimum(Aᵢs[p] * z_feas - hᵢs[p]) >= feasible_margin "Inequality constraints not feasible at witness point"
		end
	end

	return As, Aₑs, Aᵢs, Hblk, H, h, hₑs, hᵢs
end

end # end module
