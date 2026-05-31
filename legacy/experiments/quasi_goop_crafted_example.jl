using QuasiGOOP
using BlockArrays: BlockArray, Block
using LinearAlgebra: norm
using CairoMakie
using Printf

include(joinpath(@__DIR__, "Plotting.jl"))

function parse_int_list_env(varname::String, default_values::Vector{Int})
	value = get(ENV, varname, "")
	isempty(strip(value)) && return default_values
	return parse.(Int, split(value, ","))
end

function crafted_direction_vector(player::Int, level::Int, n::Int)
	v = zeros(n)
	for j in 1:n
		v[j] = sin((player + level) * j) + 0.1 * (player + level + j)
	end
	return v
end

function player_active_indices(primal_dims::AbstractVector{Int}, player::Int)
	start_idx = sum(primal_dims[1:(player - 1)]) + 1
	return start_idx, start_idx + 1
end

function build_crafted_goop(
	K::Int;
	num_players::Int,
	primal_dims::Vector{Int},
	param_dims::Vector{Int},
	z_star::BlockArray,
)
	@assert length(z_star) == sum(primal_dims)

	dummy_primals = BlockArray(zeros(sum(primal_dims)), primal_dims)
	dummy_parameters = BlockArray(zeros(sum(param_dims)), param_dims)

	preferences = [Vector{Function}(undef, K) for _ in 1:num_players]
	for i in 1:num_players
		for k in 1:K
			v = crafted_direction_vector(i, k, sum(primal_dims))
			preferences[i][k] = (z, θ) -> cosh(v' * (z - z_star))
			# ∇J = sinh(v' * (z - z_star)) * v, ∇²J = cosh(v' * (z - z_star)) * v'v'
			# at z=z*, ∇J = 0 and ∇²J = vv' (PSD)
		end
	end

	is_prioritized_constraint = [fill(false, K) for _ in 1:num_players]

	equality_constraints = [
		(z, θ) -> [exp(z[Block(i)][1] - z_star[Block(i)][1]) + exp(-(z[Block(i)][2] - z_star[Block(i)][2])) - 2.0]
		for i in 1:num_players
	]

	inequality_constraints = [
		(z, θ) -> [exp(-((z[Block(i)][1] - z_star[Block(i)][1])^2 + (z[Block(i)][2] - z_star[Block(i)][2])^2)) - 0.5]
		for i in 1:num_players
	]

	goop = QuasiGOOP.ParametricGOOP(
		dummy_primals,
		dummy_parameters;
		preferences,
		is_prioritized_constraint,
		equality_constraints,
		inequality_constraints,
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)

	θ = zeros(sum(param_dims))
	return goop, θ
end

function run_system(kkt_system, θ, z0; z_star, solve_kwargs)
	# First call: include compilation/JIT costs.
	compile_log = Dict{String, Any}()
	compile_time = @elapsed begin
		QuasiGOOP.solve(
			QuasiGOOP.InteriorPoint(),
			kkt_system,
			θ;
			z₀ = z0,
			convergence_log = compile_log,
			solve_kwargs...,
		)
	end

	# Second call: runtime solve time after compilation.
	solve_log = Dict{String, Any}()
	result = nothing
	solve_time = @elapsed begin
		result = QuasiGOOP.solve(
			QuasiGOOP.InteriorPoint(),
			kkt_system,
			θ;
			z₀ = z0,
			convergence_log = solve_log,
			solve_kwargs...,
		)
	end

	F_val = zeros(kkt_system.kkt_dimension)
	kkt_system.F!(F_val, result.z; θ, ϵ = result.ϵ, η = 0.0)
	final_kkt_residual = norm(F_val, 2)

	primal = result.z[kkt_system.primal_dims]
	primal_error = norm(primal - collect(z_star), 2)

	return (
		compile_time = compile_time,
		solve_time = solve_time,
		result = result,
		primal = copy(primal),
		final_kkt_residual = final_kkt_residual,
		primal_error = primal_error,
		convergence_log = solve_log,
	)
end

function plot_comparison(results_by_k; outdir)
	mkpath(outdir)
	figure = serif_figure(size = (1400, 900))
	axes = CairoMakie.Axis[]
	k_values_sorted = sort(collect(keys(results_by_k)))

	for (idx, K) in enumerate(k_values_sorted)
		row = ((idx - 1) ÷ 2) + 1
		col = ((idx - 1) % 2) + 1
		ax = CairoMakie.Axis(
			figure[row, col];
			title = "K = $(K)",
			xlabel = "Newton iteration",
			ylabel = "KKT residual",
			yscale = log10,
			xgridvisible = true,
			ygridvisible = true,
		)
		push!(axes, ax)

		reduced_log = results_by_k[K][:Reduced].convergence_log
		quasi_log = results_by_k[K][:Quasi].convergence_log

		reduced_hist = get(reduced_log, "kkt_error_history", Float64[])
		quasi_hist = get(quasi_log, "kkt_error_history", Float64[])
		reduced_iter = get(reduced_log, "total_iteration_history", collect(1:length(reduced_hist)))
		quasi_iter = get(quasi_log, "total_iteration_history", collect(1:length(quasi_hist)))

		# Clamp to positive floor for log-scale plotting robustness.
		reduced_hist_plot = max.(reduced_hist, eps(Float64))
		quasi_hist_plot = max.(quasi_hist, eps(Float64))

		reduced_line = CairoMakie.lines!(
			ax,
			reduced_iter,
			reduced_hist_plot;
			color = :dodgerblue,
			linewidth = 3,
			label = "Reduced",
		)
		quasi_line = CairoMakie.lines!(
			ax,
			quasi_iter,
			quasi_hist_plot;
			color = :crimson,
			linewidth = 3,
			label = "Quasi",
		)

		if idx == 1
			CairoMakie.axislegend(
				ax,
				[reduced_line, quasi_line],
				["Reduced", "Quasi"];
				position = :rt,
				labelsize = 14,
			)
		end
	end

	CairoMakie.save(joinpath(outdir, "quasi_goop_crafted_convergence_comparison.pdf"), figure)
	CairoMakie.save(joinpath(outdir, "quasi_goop_crafted_convergence_comparison.png"), figure)
	return figure, axes
end

function print_summary_table(rows)
	println("\nSummary across K")
	println(
		"--------------------------------------------------------------------------------------------------------------",
	)
	@printf(
		"%-4s %-8s %-8s %-12s %-12s %-14s %-14s %-10s %s\n",
		"K",
		"System",
		"Status",
		"Build(s)",
		"Compile(s)",
		"Solve(s)",
		"||F(y*)||",
		"||z-z*||",
		"OuterIters",
	)
	println(
		"--------------------------------------------------------------------------------------------------------------",
	)
	for row in rows
		@printf(
			"%-4d %-8s %-8s %-12.4e %-12.4e %-14.4e %-14.4e %-10.4e %d\n",
			row.K,
			row.system,
			string(row.status),
			row.build_time,
			row.compile_time,
			row.solve_time,
			row.kkt_residual,
			row.primal_error,
			row.outer_iters,
		)
	end
	println(
		"--------------------------------------------------------------------------------------------------------------\n",
	)
end

function main()
	num_players = 2
	primal_dims = [4, 4]
	param_dims = [1, 1]
	default_k_values = [3, 4]
	z_star = BlockArray([fill(0.2, 4); fill(-0.1, 4)], primal_dims)
	z0 = zeros(sum(primal_dims))
	solve_kwargs = (
		tol = 1e-8,
		η₀ = 0.0,
		ϵ₀ = 1e-8,
		max_inner_iters = 100, # for this example, no decay for ϵ
		max_outer_iters = 100,
		min_stepsize = 1e-10,
		linesearch = :backtracking,
		verbose = false,
	)

	k_values = parse_int_list_env("QUASI_GOOP_K_VALUES", default_k_values)

	println("Running crafted quasi-GOOP benchmark...")
	println("z* = $(z_star)")
	println("K sweep = $(k_values)")

	results_by_k = Dict{Int, Dict{Symbol, Any}}()
	summary_rows = NamedTuple[]

	for K in k_values
		println("\n========================= K = $(K) =========================")
		goop, θ = build_crafted_goop(
			K;
			num_players,
			primal_dims,
			param_dims,
			z_star,
		)

		reduced_build_time = @elapsed reduced_kkt = QuasiGOOP.generate_slacked_reduced_kkt_system(goop)
		quasi_build_time = @elapsed quasi_kkt = QuasiGOOP.generate_slacked_quasi_kkt_system(goop)

		reduced = run_system(reduced_kkt, θ, z0; z_star, solve_kwargs)
		quasi = run_system(quasi_kkt, θ, z0; z_star, solve_kwargs)

		results_by_k[K] = Dict(
			:Reduced => merge(reduced, (; build_time = reduced_build_time)),
			:Quasi => merge(quasi, (; build_time = quasi_build_time)),
		)

		println("[Reduced][K=$(K)] primal z = $(round.(reduced.primal; digits = 8))")
		println("[Reduced][K=$(K)] status = $(reduced.result.status)")
		println("[Reduced][K=$(K)] ||z-z*|| = $(reduced.primal_error)")
		println("[Reduced][K=$(K)] ||F(y*)|| = $(reduced.final_kkt_residual)")
		println("[Reduced][K=$(K)] build=$(reduced_build_time)s compile=$(reduced.compile_time)s solve=$(reduced.solve_time)s outer_iters=$(reduced.result.outer_iters)")

		println("[Quasi][K=$(K)] primal z = $(round.(quasi.primal; digits = 8))")
		println("[Quasi][K=$(K)] status = $(quasi.result.status)")
		println("[Quasi][K=$(K)] ||z-z*|| = $(quasi.primal_error)")
		println("[Quasi][K=$(K)] ||F(y*)|| = $(quasi.final_kkt_residual)")
		println("[Quasi][K=$(K)] build=$(quasi_build_time)s compile=$(quasi.compile_time)s solve=$(quasi.solve_time)s outer_iters=$(quasi.result.outer_iters)")
		println(
			"[Compare][K=$(K)] solve_time_delta(reduced-quasi) = $(reduced.solve_time - quasi.solve_time), " *
			"solution_error_delta(reduced-quasi) = $(reduced.primal_error - quasi.primal_error), " *
			"kkt_residual_delta(reduced-quasi) = $(reduced.final_kkt_residual - quasi.final_kkt_residual)",
		)

		push!(
			summary_rows,
			(
				K = K,
				system = "Reduced",
				status = reduced.result.status,
				build_time = reduced_build_time,
				compile_time = reduced.compile_time,
				solve_time = reduced.solve_time,
				kkt_residual = reduced.final_kkt_residual,
				primal_error = reduced.primal_error,
				outer_iters = reduced.result.outer_iters,
			),
		)
		push!(
			summary_rows,
			(
				K = K,
				system = "Quasi",
				status = quasi.result.status,
				build_time = quasi_build_time,
				compile_time = quasi.compile_time,
				solve_time = quasi.solve_time,
				kkt_residual = quasi.final_kkt_residual,
				primal_error = quasi.primal_error,
				outer_iters = quasi.result.outer_iters,
			),
		)
	end

	plots_dir = joinpath(@__DIR__, "results")
	plot_comparison(results_by_k; outdir = plots_dir)
	println("Saved convergence plots to: $(plots_dir)")

	print_summary_table(summary_rows)
end

main()
