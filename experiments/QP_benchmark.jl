module QP_benchmark

using QuasiGOOP

using ParametricMCPs
using LinearAlgebra: I, Diagonal, norm, pinv, qr, rank
using Symbolics
using SymbolicTracingUtils
using LaTeXStrings: @L_str
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes, blockedrange
using NonlinearSolve
using InvertedIndices: Not
using SparseArrays: sparse, SparseMatrixCSC
using Statistics: mean, std
using JLD2
using Random

include(joinpath(@__DIR__, "Plotting.jl"))

function print_preference_values(label, num_players, n, problem, z, θ)
	println("[$(label)] Preference values at solution:")
	for (player_idx, values) in enumerate(get_preference_values(num_players, n, problem, z, θ))
		println("  player $(player_idx): $(round.(values, digits = 6))")
	end
end

function get_preference_values(num_players, n, problem, z, θ)
	z_primal = z[1:(num_players*n)]
	[[pref(z_primal, θ) for pref in player_preferences] for player_preferences in problem.preferences]
end

function get_setup(n, num_players, mₑ, mᵢ; num_preferences = 2, r = 1)
	primal_dimensions = fill(n, num_players)
	parameter_dimensions = fill(1, num_players)
	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
	dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

	preferences = [Vector{Function}(undef, num_preferences) for _ in 1:num_players]
	for i in 1:num_players, j in 1:num_preferences
		Qk_local = rand_psd(n * num_players, r) # overwrite this
		q_local = Qk_local * randn(n * num_players) # q ∈ Col(Qk) for boundedness
		preferences[i][j] = let Qk = Qk_local, q = q_local
			(z, θ) -> 0.5 * z' * Qk * z + q' * z + 0.1 * (j == num_preferences ? (sum(z))^4 : 0.0)
		end
	end

	is_prioritized_constraint = [
		[false for j in 1:num_preferences] for i in 1:num_players
	]

	equality_constraints = Vector{Function}(undef, num_players)
	for i in 1:num_players
		H_blocks = [rand(mₑ, n) for _ in 1:num_players] # n > mₑ for full row rank
		H = hcat(H_blocks...)
		hⁱ = rand(mₑ)
		equality_constraints[i] = let H = H, hⁱ = hⁱ
			(z, θ) -> H * z - hⁱ
		end
	end

	inequality_constraints = Vector{Function}(undef, num_players)
	for i in 1:num_players
		G_blocks = [rand(mᵢ, n) for _ in 1:num_players]
		G = hcat(G_blocks...)
		gⁱ = rand(mᵢ)
		inequality_constraints[i] = let G = G, gⁱ = gⁱ
			(z, θ) -> G * z - gⁱ
		end
	end
	if num_players > 1
		for i in 1:num_players
			Gⁱ1 = rand(mᵢ, n) # n > mᵢ for full row rank
			Gⁱ2 = rand(mᵢ, n)
			gⁱ = rand(mᵢ)
			inequality_constraints[i] = let G = hcat(Gⁱ1, Gⁱ2), g = gⁱ
				(z, θ) -> G * z - g
			end
		end
	else
		G = rand(mᵢ, n)
		g = rand(mᵢ)
		inequality_constraints[1] = (z, θ) -> G * z - g
	end

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


function demo(;
	num_players = 2,
	num_preferences = 5,
	rng_seed = 123,
	show_convergence_legend = true,
	show_ylabel = true,
)
	Random.seed!(rng_seed)

	# Quadratic GOOP Problem setup
	n = 10 # x primal dimension (per player)
	mₑ = 3 # equality constraint dimension
	mᵢ = 2 # inequality constraint dimension
	parameters = BlockArray(zeros(sum(fill(1, num_players))), fill(1, num_players))
	num_instances = 1

	linesearch = :backtracking # :backtracking, :fraction_to_boundary
	verbose = true
	tol = 200000000 # 2e-2, 2e-1, 2.0
	ϵ₀ = 0.001 #ρ 1e-2, 1e-1, 1.0
	η₀ = 0.0
	max_inner_iters = 30
	max_outer_iters = 2
	min_stepsize = 1e-20
	run_id = "Dongho_run_QP_$(num_players)players_$(num_preferences)prefs_$(ϵ₀)ρ_$(n)pdim_$(mₑ)mₑ_$(mᵢ)mᵢ"

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

	backend = SymbolicTracingUtils.SymbolicsBackend()
	instance_problem_data = Dict{String, Any}[]
	# kkt_error_histories_per_eps = Dict(ϵ => Vector{Vector{Float64}}() for ϵ in epsilon_schedule)
	solved_attempts = 0
	total_attempts = 0
	pref_l2_diffs_per_player = [Float64[] for _ in 1:num_players]
	reduced_elapsed_times = Float64[]
	complete_elapsed_times = Float64[]
	kkt_error_histories_reduced = Vector{Vector{Float64}}()
	kkt_error_histories_complete = Vector{Vector{Float64}}()

	# Create problem
	problem = get_setup(n, num_players, mₑ, mᵢ; num_preferences)

	# Solve problem with reduced GOOP KKT system using params
	@info "Generating reduced KKT system..."
	reduced_kkt_system = QuasiGOOP.generate_slacked_reduced_kkt_system(problem)
	println("[Reduced] KKT Dimension: ", reduced_kkt_system.kkt_dimension)
	println("[Reduced] Variable Dimension: ", reduced_kkt_system.variable_dimension)

	# Solve problem with complete GOOP KKT system
	@info "Generating complete KKT system..."
	complete_kkt_system = QuasiGOOP.generate_slacked_complete_kkt_system(problem)
	println("[Complete] KKT Dimension: ", complete_kkt_system.kkt_dimension)
	println("[Complete] Variable Dimension: ", complete_kkt_system.variable_dimension)

	while solved_attempts < num_instances
		total_attempts += 1
		@info "solved $(solved_attempts)/$(num_instances), attempt $(total_attempts): "

		convergence_log_reduced_system = Dict{String, Any}()
		reduced_elapsed_time = @elapsed begin
			(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				reduced_kkt_system,
				parameters; # this will control the θ values used in the problem definition
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
		println("[Reduced] Elapsed time: $(round(reduced_elapsed_time, digits = 3)) seconds")
		println("[Reduced] status = $(status)")
		reduced_z = copy(z)
		reduced_primal = copy(z[1:(num_players*n)])
		reduced_pref_values = get_preference_values(num_players, n, problem, z, parameters)
		println("[Reduced] Primal solution: $(round.(reduced_primal, digits = 3))")
		print_preference_values("Reduced", num_players, n, problem, z, parameters)
		println("[Reduced] kkt_error = $(kkt_error)")
		reduced_kkt_history = log10.(get(convergence_log_reduced_system, "kkt_error_history", Float64[]))

		if status != :solved
			@info "[Reduced] Attempt $(total_attempts) failed. Retrying with a new instance..."
			continue
		end

		# Solve problem with complete GOOP KKT system
		convergence_log_complete_system = Dict{String, Any}()
		complete_elapsed_time = @elapsed begin
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
		println("[Complete] Elapsed time: $(round(complete_elapsed_time, digits = 3)) seconds")
		println("[Complete] status = $(status)")
		complete_z = copy(z)
		complete_primal = copy(z[1:(num_players*n)])
		complete_pref_values = get_preference_values(num_players, n, problem, z, parameters)
		println("[Complete] Primal solution: $(round.(complete_primal, digits = 3))")
		print_preference_values("Complete", num_players, n, problem, z, parameters)
		println("[Complete] kkt_error = $(kkt_error)")
		complete_kkt_history = log10.(get(convergence_log_complete_system, "kkt_error_history", Float64[]))

		if status != :solved
			@info "[Complete] Attempt $(total_attempts) failed. Retrying with a new instance..."
			continue
		end

		# # If primal solutions diverge, attempt dual recovery for the complete KKT system with reduced primals fixed.
		# if norm(reduced_primal - complete_primal, Inf) > tol
		# 	@warn "[Check] Reduced and complete primals differ (Inf-norm > tol). Trying to recover complete-system duals with reduced primal fixed."
		# 	F_symbolic = complete_kkt_system.F_symbolic
		# 	z_symbolic = complete_kkt_system.z_symbolic
		# 	primal_indices = complete_kkt_system.primal_dims
		# 	nonprimal_indices = Not(primal_indices)

		# 	substitution_dict = Dict{Any, Any}()
		# 	for (sym, val) in zip(z_symbolic[primal_indices], reduced_primal)
		# 		substitution_dict[sym] = val
		# 	end
		# 	let ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
		# 		η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))
		# 		substitution_dict[ϵ] = ϵ₀
		# 		substitution_dict[η] = η₀
		# 	end

		# 	F_symbolic_after_sub = Symbolics.substitute(F_symbolic, substitution_dict)
		# 	F_eval = first(
		# 		Symbolics.build_function(
		# 			F_symbolic_after_sub,
		# 			z_symbolic[nonprimal_indices];
		# 			expression = Val(false),
		# 		),
		# 	)
		# 	test_f(u, p) = F_eval(u)

		# 	@info "[Check] Solving for complete-system duals with reduced primal fixed..."
		# 	z_val = zeros(length(z_symbolic) - length(primal_indices))
		# 	prob =
		# 		(mᵢ == 0) ?
		# 		NonlinearProblem(test_f, z_val) :
		# 		NonlinearLeastSquaresProblem(test_f, z_val)
		# 	dual_sol = NonlinearSolve.solve(prob)

		# 	z_recovered = similar(complete_z)
		# 	z_recovered[primal_indices] = reduced_primal
		# 	z_recovered[nonprimal_indices] = dual_sol.u
		# 	F_recovered = zeros(complete_kkt_system.kkt_dimension)
		# 	complete_kkt_system.F!(F_recovered, z_recovered; θ = parameters, ϵ = ϵ₀, η = η₀)
		# 	kkt_error_recovered = norm(F_recovered, Inf)
		# 	@info "[Check] KKT error (reduced primal + recovered dual) = $(kkt_error_recovered)"
		# 	kkt_error_recovered > tol && @error "kkt_error_recovered is above tol. Recovery may have failed."
		# end

		# Save solutions for this instance
		instance_idx = solved_attempts + 1
		solved_instance_idx = instance_idx
		primal = reduced_primal
		z = reduced_z
		@save joinpath(solution_data_dir, "reduced_solution_instance_$(instance_idx).jld2") solved_instance_idx primal z reduced_pref_values
		primal = complete_primal
		z = complete_z
		@save joinpath(solution_data_dir, "complete_solution_instance_$(instance_idx).jld2") solved_instance_idx primal z complete_pref_values

		# Solve problem with ParametricMCPs (TODO)

		# Compare solutions (primal solution and preference values) and save result
		primal_l2_diff = norm(reduced_primal - complete_primal, 2)
		println("[Compare] L2 norm between reduced and complete primal solutions: $(round(primal_l2_diff, digits = 6))")

		for player_idx in 1:num_players
			reduced_vals = reduced_pref_values[player_idx]
			complete_vals = complete_pref_values[player_idx]
			pref_diff = reduced_vals .- complete_vals
			pref_l2_diff = norm(pref_diff, 2)
			push!(pref_l2_diffs_per_player[player_idx], pref_l2_diff)
			println("[Compare] player $(player_idx) preference L2 difference: $(round(pref_l2_diff, digits = 6))")
		end

		# Compare elapsed times and save results
		push!(reduced_elapsed_times, reduced_elapsed_time)
		push!(complete_elapsed_times, complete_elapsed_time)
		println("[Compare] elapsed time reduced (s): $(round(reduced_elapsed_time, digits = 6))")
		println("[Compare] elapsed time complete (s): $(round(complete_elapsed_time, digits = 6))")

		# Save KKT error histories
		push!(kkt_error_histories_reduced, reduced_kkt_history)
		push!(kkt_error_histories_complete, complete_kkt_history)
		kkt_error_history = reduced_kkt_history
		@save joinpath(histories_data_dir, "kkt_error_history_reduced_instance_$(solved_attempts + 1).jld2") kkt_error_history
		kkt_error_history = complete_kkt_history
		@save joinpath(histories_data_dir, "kkt_error_history_complete_instance_$(solved_attempts + 1).jld2") kkt_error_history

		solved_attempts += 1
	end

	# Report aggregate results across instances
	@save joinpath(histories_data_dir, "pref_l2_diffs_per_player.jld2") pref_l2_diffs_per_player
	@save joinpath(histories_data_dir, "kkt_error_histories_reduced.jld2") kkt_error_histories_reduced
	@save joinpath(histories_data_dir, "kkt_error_histories_complete.jld2") kkt_error_histories_complete
	absolute_differences = abs.(reduced_elapsed_times .- complete_elapsed_times)
	@save joinpath(histories_data_dir, "elapsed_times.jld2") reduced_elapsed_times complete_elapsed_times absolute_differences

	if !isempty(kkt_error_histories_reduced) && any(!isempty, kkt_error_histories_reduced)
		reduced_aggregate_convergence_fig, _ = plot_convergence_plot_aggregate(
			;
			kkt_error_histories = kkt_error_histories_reduced,
		)
		CairoMakie.save(
			joinpath(convergence_plots_dir, "convergence_aggregate_reduced.pdf"),
			reduced_aggregate_convergence_fig,
		)
	end
	if !isempty(kkt_error_histories_complete) && any(!isempty, kkt_error_histories_complete)
		complete_aggregate_convergence_fig, _ = plot_convergence_plot_aggregate(
			;
			kkt_error_histories = kkt_error_histories_complete,
		)
		CairoMakie.save(
			joinpath(convergence_plots_dir, "convergence_aggregate_complete.pdf"),
			complete_aggregate_convergence_fig,
		)
	end
	if (!isempty(kkt_error_histories_reduced) && any(!isempty, kkt_error_histories_reduced)) &&
	   (!isempty(kkt_error_histories_complete) && any(!isempty, kkt_error_histories_complete))
		combined_aggregate_convergence_fig, _ = plot_convergence_plot_aggregate_comparison(
			;
			reduced_kkt_error_histories = kkt_error_histories_reduced,
			complete_kkt_error_histories = kkt_error_histories_complete,
			show_legend = show_convergence_legend,
			show_ylabel = show_ylabel,
		)
		CairoMakie.save(
			joinpath(convergence_plots_dir, "convergence_aggregate_reduced_vs_complete_$(num_preferences)prefs_$(ϵ₀)rho.pdf"),
			combined_aggregate_convergence_fig,
		)
	end

	println("==== Aggregate preference L2-difference stats across $(solved_attempts) instance(s) ====")
	for player_idx in 1:num_players
		vals = pref_l2_diffs_per_player[player_idx]
		if isempty(vals)
			println("[Aggregate] player $(player_idx): no solved instances")
		else
			mean_val = mean(vals)
			std_val = length(vals) > 1 ? std(vals) : 0.0
			println("[Aggregate] player $(player_idx): mean=$(round(mean_val, digits = 6)), std=$(round(std_val, digits = 6)), n=$(length(vals))")
		end
	end
	println("==== Aggregate elapsed-time stats across $(solved_attempts) instance(s) ====")
	if !isempty(reduced_elapsed_times) && !isempty(complete_elapsed_times)
		time_abs_diff = abs.(reduced_elapsed_times .- complete_elapsed_times)
		reduced_std = length(reduced_elapsed_times) > 1 ? std(reduced_elapsed_times) : 0.0
		complete_std = length(complete_elapsed_times) > 1 ? std(complete_elapsed_times) : 0.0
		diff_std = length(time_abs_diff) > 1 ? std(time_abs_diff) : 0.0
		println("[Aggregate] reduced elapsed time (s): mean=$(round(mean(reduced_elapsed_times), digits = 6)), std=$(round(reduced_std, digits = 6)), n=$(length(reduced_elapsed_times))")
		println("[Aggregate] complete elapsed time (s): mean=$(round(mean(complete_elapsed_times), digits = 6)), std=$(round(complete_std, digits = 6)), n=$(length(complete_elapsed_times))")
		println("[Aggregate] |reduced-complete| elapsed time (s): mean=$(round(mean(time_abs_diff), digits = 6)), std=$(round(diff_std, digits = 6)), n=$(length(time_abs_diff))")
	else
		println("[Aggregate] elapsed time: no solved instances")
	end


end # end demo





# Helper functions 
function rand_psd(n, r)
	# n: primal dimension, r: matrix rank (<=n)
	R = randn(r, n);
	R' * R;
end

# function generate_block_quadratic_problem(n, num_players; r = 1)
# 	# Build num_players PSD blocks and assemble a block-diagonal matrix.
# 	@assert num_players >= 1 "num_players must be at least 1"
# 	Qs = [rand_psd(n, r) for _ in 1:num_players]
# 	Qblk = sparse(cat(Qs...; dims = (1, 2))) # block matrix
# 	total_n = n * num_players

# 	# Random off-diagonal blocks; keep diagonal player-blocks unchanged.
# 	Qoff = randn(total_n, total_n)
# 	for i in 1:num_players
# 		idx = ((i-1)*n+1):(i*n)
# 		@views Qoff[idx, idx] .= 0.0
# 	end
# 	Qₖ = Qblk + sparse(Qoff)
# 	x = randn(total_n)
# 	q = Qₖ * x # ∈ Col(Qₖ)
# 	return Qs, Qblk, Qₖ, q
# end

# function rand_full_row_rank_matrix(m, n; coupling_scale = 0.5, min_sv = 0.2, max_sv = 2.0)
# 	@assert m <= n "Need m <= n for full row rank"
# 	@assert 0.0 < min_sv <= max_sv "Singular-value bounds must satisfy 0 < min_sv <= max_sv"

# 	# Build a guaranteed full-row-rank core, then mix rows/columns with random orthogonal bases.
# 	U = Matrix(qr(randn(m, m)).Q)
# 	V = Matrix(qr(randn(n, n)).Q)
# 	s = min_sv .+ (max_sv - min_sv) .* rand(m)
# 	core = hcat(Diagonal(s), coupling_scale * randn(m, n - m))
# 	return U * core * V'
# end

# function full_row_rank_constraints(n, num_players, mₑ, mᵢ; feasible_margin = 1e-2)
# 	@assert num_players >= 1 "num_players must be at least 1"
# 	m = mₑ + mᵢ
# 	@assert m <= n "Need mₑ + mᵢ <= n  for full row rank"
# 	@assert feasible_margin > 0 "feasible_margin must be positive"

# 	# Build per-player diagonal blocks first (each m x n).
# 	Ablocks = Vector{Matrix{Float64}}(undef, num_players)
# 	for p in 1:num_players
# 		Araw = rand_full_row_rank_matrix(
# 			m,
# 			n;
# 			coupling_scale = 0.35 + 0.5 * rand(),
# 			min_sv = 0.15,
# 			max_sv = 2.5,
# 		)

# 		# Randomize which constraint rows are treated as equalities vs inequalities.
# 		row_perm = randperm(m)
# 		eq_idx = row_perm[1:mₑ]
# 		ineq_idx = row_perm[(mₑ+1):m]
# 		Aₑ = Araw[eq_idx, :]
# 		Aᵢ = Araw[ineq_idx, :]
# 		Ablocks[p] = vcat(Aₑ, Aᵢ)
# 	end

# 	# Block-diagonal matrix with A blocks on the diagonal.
# 	Hblk = sparse(cat(Ablocks...; dims = (1, 2)))

# 	# Same diagonal blocks as Hblk, random off-diagonal blocks.
# 	total_m = m * num_players
# 	total_n = n * num_players
# 	Hoff = randn(total_m, total_n)
# 	for p in 1:num_players
# 		row_idx = ((p-1)*m+1):(p*m)
# 		col_idx = ((p-1)*n+1):(p*n)
# 		@views Hoff[row_idx, col_idx] .= 0.0
# 	end
# 	H = Hblk + sparse(Hoff)

# 	@assert rank(H) == total_m "H must have full row rank"

# 	# Per-player A/Ae/Ai as row-slices of H (same column dimension as H).
# 	As = Vector{SparseMatrixCSC{Float64, Int}}(undef, num_players)
# 	Aₑs = Vector{SparseMatrixCSC{Float64, Int}}(undef, num_players)
# 	Aᵢs = Vector{SparseMatrixCSC{Float64, Int}}(undef, num_players)
# 	for p in 1:num_players
# 		row_base = (p - 1) * m
# 		As[p] = H[(row_base+1):(row_base+m), :]
# 		Aₑs[p] = H[(row_base+1):(row_base+mₑ), :]
# 		Aᵢs[p] = H[(row_base+mₑ+1):(row_base+m), :]
# 	end

# 	# Build h so the constraint set is feasible by construction.
# 	# For witness z_feas:
# 	#   A_e z_feas - h_e = 0
# 	#   A_i z_feas - h_i >= feasible_margin
# 	z_feas = randn(total_n)
# 	h = Vector(H * z_feas)
# 	for p in 1:num_players
# 		if mᵢ == 0
# 			continue
# 		end
# 		row_base = (p - 1) * m
# 		ineq_rows = (row_base+mₑ+1):(row_base+m)
# 		h[ineq_rows] .-= feasible_margin .+ abs.(randn(mᵢ))
# 	end

# 	hₑs = Vector{Vector{Float64}}(undef, num_players)
# 	hᵢs = Vector{Vector{Float64}}(undef, num_players)
# 	for p in 1:num_players
# 		row_base = (p - 1) * m
# 		hₑs[p] = h[(row_base+1):(row_base+mₑ)]
# 		hᵢs[p] = h[(row_base+mₑ+1):(row_base+m)]
# 	end

# 	# Sanity check at the witness point.
# 	for p in 1:num_players
# 		@assert norm(Aₑs[p] * z_feas - hₑs[p], Inf) <= 1e-8 "Equality constraints not feasible at witness point"
# 		if mᵢ > 0
# 			@assert minimum(Aᵢs[p] * z_feas - hᵢs[p]) >= feasible_margin "Inequality constraints not feasible at witness point"
# 		end
# 	end

# 	return As, Aₑs, Aᵢs, Hblk, H, h, hₑs, hᵢs
# end

end # end module
