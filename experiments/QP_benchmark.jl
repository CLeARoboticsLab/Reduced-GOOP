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

function get_setup(n, num_players, mₑ, mᵢ; num_preferences = 2, r = 1)
	primal_dimensions = fill(n, num_players)
	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
	flattened_parameters = flatten_params(
		generate_random_parameter(; n, num_players, num_preferences, mₑ, mᵢ)
	)
	@assert length(flattened_parameters) % num_players == 0
	dummy_parameters = BlockArray(flattened_parameters, fill(length(flattened_parameters) ÷ num_players, num_players))

	preferences = [Vector{Function}(undef, num_preferences) for _ in 1:num_players]
	for i in 1:num_players, k in 1:num_preferences
		preferences[i][k] = function (z, θ)
			(; Qk, qk) = unpack_parameters(θ, n, i, k, mₑ, mᵢ; num_players, num_preferences)
			0.5 * z' * Qk * z + qk' * z #+ 0.1 * (k == num_preferences ? (sum(z))^4 : 0.0)
		end
	end

	# 2-level debugging
	# preferences[1][1] = (z,θ) -> (z[1]-1)^4       # J¹₁ primary  (quartic)
	# preferences[1][2] = (z,θ) -> (z[1]-0.5)^2     # J¹₂ secondary
	# preferences[2][1] = (z,θ) -> (z[3]+1)^4       # J²₁ primary  (quartic)
	# preferences[2][2] = (z,θ) -> (z[3]+0.5)^2.    # J²₂ secondary
	# preferences[1][1] = let
	# 	Qk = [0.416968 0.944866 1.048411 0.140553; 0.944866 2.141105 2.37574 0.318499; 1.048411 2.37574 2.636089 0.353402; 0.140553 0.318499 0.353402 0.047378]
	# 	qk = [1.433707; 3.248838; 3.604866; 0.483279]
	# 	(z, θ) -> 0.5 * z' * Qk * z + qk' * z
	# end
	# preferences[1][2] = let
	# 	Qk = [1.799915 -0.55296 -0.795839 1.030905; -0.55296 0.169877 0.244493 -0.316709; -0.795839 0.244493 0.351883 -0.455818; 1.030905 -0.316709 -0.455818 0.590452]
	# 	qk = [-0.444909; 0.136682; 0.196718; -0.254822]
	# 	(z, θ) -> 0.5 * z' * Qk * z + qk' * z
	# end
	# preferences[2][1] = let
	# 	Qk = [0.042467 -0.209993 0.104132 -0.11093; -0.209993 1.03839 -0.514919 0.548533; 0.104132 -0.514919 0.255339 -0.272008; -0.11093 0.548533 -0.272008 0.289765] # level 1
	# 	qk = [0.110712; -0.547456; 0.271474; -0.289196]
	# 	(z, θ) -> 0.5 * z' * Qk * z + qk' * z
	# end
	# preferences[2][2] = let
	# 	Qk = [0.011885 0.029826 0.006225 0.043515; 0.029826 0.074851 0.015621 0.109203; 0.006225 0.015621 0.00326 0.02279; 0.043515 0.109203 0.02279 0.159321] # level 2
	# 	qk = [0.045815; 0.114977; 0.023995; 0.167745]
	# 	(z, θ) -> 0.5 * z' * Qk * z + qk' * z
	# end


	is_prioritized_constraint = [
		[false for j in 1:num_preferences] for i in 1:num_players
	]

	equality_constraints = Vector{Function}(undef, num_players)
	# for i in 1:num_players
	# 	equality_constraints[i] = function (z, θ)
	# 		(; H, h) = unpack_parameters(θ, n, i, 1, mₑ, mᵢ; num_players, num_preferences)
	# 		H * z - h
	# 	end
	# end
	equality_constraints[1] = (z, θ) -> [z[1]^2 + z[2]^2 + 0.2*z[3]*z[4] - 1.0]
	equality_constraints[2] = (z, θ) -> [z[3]^2 + z[4]^2 + 0.2*z[1]*z[2] - 1.0]
	# equality_constraints[1] = (z, θ) -> [sum(z) - 1.0]
	# equality_constraints[2] = (z, θ) -> [sum(z) - 1.0]

	inequality_constraints = Vector{Function}(undef, num_players)
	# for i in 1:num_players
	# 	inequality_constraints[i] = function (z, θ)
	# 		(; G, g) = unpack_parameters(θ, n, i, 1, mₑ, mᵢ; num_players, num_preferences)
	# 		G * z - g
	# 	end
	# end
	inequality_constraints[1] = (z, θ) -> [4.0 - z[1]^2 - 0.5*z[3]^2]
	inequality_constraints[2] = (z, θ) -> [4.0 - z[3]^2 - 0.5*z[1]^2]

	QuasiGOOP.ParametricGOOP(
		dummy_primals,
		dummy_parameters;
		preferences,
		is_prioritized_constraint,
		equality_constraints,
		inequality_constraints,
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
	n = 2 # x primal dimension (per player)
	mₑ = 1 # equality constraint dimension
	mᵢ = 1 # inequality constraint dimension
	num_instances = 2
	linesearch = :backtracking # :backtracking, :fraction_to_boundary
	verbose = true
	tol = 1e-5 # 2e-2, 2e-1, 2.0
	ϵ₀ = 1.0 #ρ 1e-2, 1e-1, 1.0
	η₀ = 0.0
	max_inner_iters = 30
	max_outer_iters = 2
	min_stepsize = 1e-20
	run_id = "Dongho_run_QP_$(num_players)players_$(num_preferences)prefs_$(ϵ₀)ρ_$(n)pdim_$(mₑ)mₑ_$(mᵢ)mᵢ"

	# Create file dir
	run_dir = joinpath("data", "debugging_benchmark", run_id)
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

		# Create problem instance with random parameters
		@info "Generating random parameters..."
		parameters = generate_random_parameter(; n, num_players, num_preferences, mₑ, mᵢ)
		flattened_parameters = flatten_params(parameters)
		param_ranges = player_flat_ranges(parameters)
		# i, k = 1, 1
		# (; Qk, qk, H, h, G, g) = unpack_parameters(flattened_parameters, n, i, k, mₑ, mᵢ; num_players, num_preferences)


		convergence_log_reduced_system = Dict{String, Any}()
		reduced_elapsed_time = @elapsed begin
			(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				reduced_kkt_system,
				flattened_parameters; # this will control the θ values used in the problem definition
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
		reduced_pref_values = get_preference_values(num_players, n, problem, z, flattened_parameters)
		println("[Reduced] Primal solution: $(round.(reduced_primal, digits = 3))")
		print_preference_values("Reduced", num_players, n, problem, z, flattened_parameters)
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
				flattened_parameters;
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
		complete_pref_values = get_preference_values(num_players, n, problem, z, flattened_parameters)
		println("[Complete] Primal solution: $(round.(complete_primal, digits = 3))")
		print_preference_values("Complete", num_players, n, problem, z, flattened_parameters)
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
		# 	complete_kkt_system.F!(F_recovered, z_recovered; θ = flattened_parameters, ϵ = ϵ₀, η = η₀)
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

"Generate a random parameter vector Θ corresponding to a convex QP."
function generate_random_parameter(;
	n = 2,
	r = 1,
	num_players = 2,
	num_preferences = 2,
	mₑ = 1,
	mᵢ = 2,
)

	# params[i] stores all preference cost data for player i, and that player's
	# equality/inequality constraint data.
	params = Vector{Any}(undef, num_players)

	for i in 1:num_players
		# Constraint data is generated once per player.
		H_blocks = [rand(mₑ, n) for _ in 1:num_players] # n > mₑ for full row rank
		H = hcat(H_blocks...)
		h = rand(mₑ)
		G_blocks = [rand(mᵢ, n) for _ in 1:num_players]
		G = hcat(G_blocks...)
		g = rand(mᵢ)

		# Cost data is generated per preference for this player.
		preference_data = [
			let
				Qk_local = rand_psd(n * num_players, r)
				qk_local = Qk_local * randn(n * num_players) # q ∈ Col(Qk) for boundedness
				(Qk = Qk_local, qk = qk_local)
			end for _ in 1:num_preferences
		]

		params[i] = (
			player = i,
			preferences = preference_data,
			equality = (H = H, h = h),
			inequality = (G = G, g = g),
		)
	end
	params
end

"Flatten nested QP benchmark params into a single vector."
function flatten_params(params)
	flat = Float64[]

	for player_data in params
		for pref_data in player_data.preferences
			append!(flat, vec(pref_data.Qk))
			append!(flat, vec(pref_data.qk))
		end
		append!(flat, vec(player_data.equality.H))
		append!(flat, vec(player_data.equality.h))
		append!(flat, vec(player_data.inequality.G))
		append!(flat, vec(player_data.inequality.g))
	end

	flat
end

"Return 1-based per-player index ranges in `flatten_params(params)`."
function player_flat_ranges(params)
	ranges = Vector{UnitRange{Int}}(undef, length(params))
	start_idx = 1

	for (player_idx, player_data) in enumerate(params)
		block_len = 0
		for pref_data in player_data.preferences
			block_len += length(pref_data.Qk) + length(pref_data.qk)
		end
		block_len += length(player_data.equality.H) + length(player_data.equality.h)
		block_len += length(player_data.inequality.G) + length(player_data.inequality.g)

		end_idx = start_idx + block_len - 1
		ranges[player_idx] = start_idx:end_idx
		start_idx = end_idx + 1
	end

	ranges
end

"Unpack flattened θ for a specific player/preference into (Qk, qk, H, h, G, g)."
function unpack_parameters(θ, n, player, preference, mₑ, mᵢ; num_players, num_preferences)
	if !(1 <= player <= num_players)
		throw(ArgumentError("player index $(player) out of bounds 1:$(num_players)"))
	end
	if !(1 <= preference <= num_preferences)
		throw(ArgumentError("preference index $(preference) out of bounds 1:$(num_preferences)"))
	end

	num_primals_total = n * num_players
	qk_len = num_primals_total
	Qk_len = num_primals_total^2
	pref_block_len = Qk_len + qk_len

	H_len = mₑ * num_primals_total
	h_len = mₑ
	G_len = mᵢ * num_primals_total
	g_len = mᵢ
	player_block_len = num_preferences * pref_block_len + H_len + h_len + G_len + g_len

	total_expected_len = num_players * player_block_len
	if length(θ) < total_expected_len
		throw(ArgumentError("flattened θ is too short: got $(length(θ)), need at least $(total_expected_len)"))
	end

	player_base = (player - 1) * player_block_len

	# Preference-specific cost block
	pref_start = player_base + (preference - 1) * pref_block_len + 1
	Qk_start = pref_start
	Qk_end = Qk_start + Qk_len - 1
	qk_start = Qk_end + 1
	qk_end = qk_start + qk_len - 1

	Qk = reshape(θ[Qk_start:Qk_end], num_primals_total, num_primals_total)
	qk = θ[qk_start:qk_end]

	# Player-shared constraint block
	constraint_start = player_base + num_preferences * pref_block_len + 1
	H_start = constraint_start
	H_end = H_start + H_len - 1
	h_start = H_end + 1
	h_end = h_start + h_len - 1
	G_start = h_end + 1
	G_end = G_start + G_len - 1
	g_start = G_end + 1
	g_end = g_start + g_len - 1

	H = reshape(θ[H_start:H_end], mₑ, num_primals_total)
	h = θ[h_start:h_end]
	G = reshape(θ[G_start:G_end], mᵢ, num_primals_total)
	g = θ[g_start:g_end]

	(; Qk, qk, H, h, G, g)
end

end # end module
