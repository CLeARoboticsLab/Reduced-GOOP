module Robotic_arm

# Forward BLAS/LAPACK to Apple Accelerate (AMX): the solver's per-iteration
# cost is dominated by a dense SVD, which runs ~2.7x faster under Accelerate
# than OpenBLAS on Apple silicon. Process-global via libblastrampoline.
@static if Sys.isapple()
	using AppleAccelerate: AppleAccelerate
end

using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using JLD2, Random
using TimerOutputs: @timeit, reset_timer!

# The plotting-free half of this experiment (scenario configuration, problem
# construction, warm starts, trajectory packing) lives in `robotic_arm_core.jl`
# so that solver-only entry points can reuse it without loading a plotting
# stack. It also supplies `ReducedGOOP`, `BlockArrays`, `TO`, the dynamics-model
# types, and `dynamics.jl`.
include(joinpath(@__DIR__, "robotic_arm_core.jl"))
include(joinpath(@__DIR__, "plotting.jl"))
include(joinpath(@__DIR__, "3d_plotting.jl"))

# ── Configuration ────────────────────────────────────────────────────────────────────────

"""Output locations and display/file-format choices for experiment plots."""
Base.@kwdef struct VisualizationConfig{D}
	dirs::D
	show_interactive_trajectory::Bool = false
	static_extension::String = "pdf"
	interactive_extension::String = "html"
end

# ── Experiment entry point ─────────────────────────────────────────────────────

function demo(;
	verbose = false,
	rng_seed = 123,
	random_initial_state = false,
	debug = false,
	use_scalarized_baseline = false,
	use_social_equilibrium_baseline = false,
	show_interactive_trajectory = false,
	use_running_goal_cost = false,
	use_up_and_over_warmstart = false,
	run_id = nothing,
)
	reset_timer!(TO)
	@timeit TO "experiment setup" Random.seed!(rng_seed)

	# ── Settings ───────────────────────────────────────────────────────────────
	run_id = something(
		run_id,
		 "challenging-scenario1"
	)
	goop_version = :quasi     # :complete | :reduced | :quasi
	# Tracing/differentiation backend. The :fast_differentiation tracing backend
	# is unusable here: FD 0.4.5 has an upstream factoring bug on ≥3rd-order
	# derivatives of kinked (abs/ifelse) preference penalties, which this
	# 4-level hierarchy needs. Keep :symbolics.
	kkt_backend = :symbolics # :symbolics | :fast_differentiation
	kkt_backend_options = (;) # forwarded to Symbolics.build_function
	# Code generation for the compiled residual/Jacobian (now a demo kwarg).
	# Differentiation stays in Symbolics; :fast_differentiation only re-emits
	# the same expressions as a hash-consed DAG, which avoids the pathological
	# first-call LLVM compile times of Symbolics codegen on this problem size
	# (solver first-call compile ~47 min → ~1 min at horizon 6, verified
	# numerically identical to 1e-15). Caveat: FD codegen fails with
	# UndefVarError on very large systems (5-level hierarchy at T ≥ 8) — use
	# :native there.
	kkt_codegen = :fast_differentiation # :native | :fast_differentiation
	linesearch = :backtracking   # :backtracking | :fraction_to_boundary
	compute_warmstart = true # Whether to compute a warmstart trajectory via rollout (true) or load from file (false)
	num_instances = 1
	perturbation_scale = 0.3

	# ── Solver schedule ────────────────────────────────────────────────────────
	epsilon_schedule = [0.1]
	max_inner_iters_schedule = fill(500, length(epsilon_schedule))

	# ── Scenario and problem ───────────────────────────────────────────────────
	scenario_config = demo_scenario_config(;
		use_scalarized_baseline,
		use_social_equilibrium_baseline,
		use_running_goal_cost,
		use_up_and_over_warmstart,
	)
	(;
		dynamics_model,
		dynamics,
        planning_horizon,
        Δt,
		base_initial_state1,
		base_initial_state2,
		initial_state3,
		goal_position1,
		goal_position2,
		goal_position3,
		arm_speed_limit,
		child_speed_limit,
		collision_avoidance,
		child_initial_buffer,
		dₚ,
	) = scenario_config
	state_dimension = scenario_config.position_dimension

	(; problem, flatten_parameters) =
		@timeit TO "problem setup" get_setup(scenario_config)
	# This example currently supports only the interior-point solver.
	solver = ReducedGOOP.InteriorPoint()
	kkt_generators = Dict(
		:complete => ReducedGOOP.generate_slacked_complete_kkt_system,
		:reduced => ReducedGOOP.generate_slacked_reduced_kkt_system,
		:quasi => ReducedGOOP.generate_slacked_quasi_kkt_system,
	)

	GOOP_kkt_generator = get(kkt_generators, goop_version, nothing)
	isnothing(GOOP_kkt_generator) && error("Unknown GOOP version: $(goop_version)")

	symbolic_backends = Dict(
		:symbolics => ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
		:fast_differentiation =>
			ReducedGOOP.SymbolicTracingUtils.FastDifferentiationBackend(),
	)
	backend = get(symbolic_backends, kkt_backend, nothing)
	isnothing(backend) && error("Unknown KKT backend: $(kkt_backend)")

	@info "Building $(goop_version) KKT system with ($(kkt_backend) backend, $(kkt_codegen) codegen) and $(solver) solver..."
	# Check if problem is not an instance of GOOPKKTSystem. Otherwise, build GOOPKKTSystem.
	GOOP_kkt_system = @timeit TO "KKT construction" begin
		if problem isa ReducedGOOP.GOOPKKTSystem
			problem
		else
			GOOP_kkt_generator(
				problem;
				backend,
				backend_options = kkt_backend_options,
				codegen = kkt_codegen,
			)
		end
	end

	println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
	println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)

	primal_dimensions = [
		(dyn.state_dimension + dyn.control_dimension) * planning_horizon for
		dyn in dynamics
	]

	# ── Per-instance solver ────────────────────────────────────────────────────
	function solve_game_instance(θ; z₀, ϵ₀, max_inner_iters)
		options =
			@timeit TO "solver options construction" ReducedGOOP.InteriorPointOptions(;
				tol = 0.01, #1e-4
				η₀ = 1e-6, # 0.0 to turn off Tikhonov regularization
				η_max = 1e6,
				ϵ₀,
				max_inner_iters,
				max_outer_iters = 1,
				tightening_rate = 1.2, # high => weak decrease in η
				loosening_rate = 3.0, # low => strong increase in η
				min_stepsize = 1e-20,
				linesearch,
				record_convergence = true,
				record_condition_number = true,
				eta_retry_growth = 2.0,
				ρ_low = 0.75,
				ρ_high = 0.75,
				tsvd_threshold = 0.0, # 0.0: pure Tikhonov, > 0 and η = 0: pure TSVD
				use_marquardt_scaling = false,
				verbose,
			)

		@info "Solving game instance with $(solver)..."
		kkt_error_history = Float64[]
		condition_number_history = Float64[]
		eta_history = Float64[]
		alpha_history = Float64[]
		rho_history = Float64[]
		total_iters = 0
		solver_status = :solved
		elapsed_time = @elapsed begin
			output = @timeit TO "solver invocation" ReducedGOOP.solve(
				solver,
				GOOP_kkt_system,
				θ;
				z₀,
				options,
			)
			(; status, z, x, kkt_error, ϵ, total_iters, kkt_error_history, condition_number_history) = output
			eta_history = hasproperty(output, :eta_history) ? output.eta_history : Float64[]
			alpha_history = hasproperty(output, :alpha_history) ? output.alpha_history : Float64[]
			rho_history = hasproperty(output, :rho_history) ? output.rho_history : Float64[]
			if status == :failed
				println(
					"  [solver exit] total_iters=$(total_iters), kkt_error=$(round(kkt_error; sigdigits=4)), tol=$(options.tol)",
				)
			end
			solver_status = status
		end

		strategies = @timeit TO "solution postprocessing" extract_player_strategies(
			x,
			primal_dimensions,
			dynamics,
		)

		solution_dict = Dict(
			"strategies" => strategies,
			"z" => z,
			"x" => x,
			"solve_time_sec" => elapsed_time,
			"kkt_error" => kkt_error,
			"ϵ" => ϵ,
			"status" => solver_status,
			"total_iters" => total_iters,
			"kkt_error_history" => kkt_error_history,
			"condition_number_history" => condition_number_history,
			"eta_history" => eta_history,
			"alpha_history" => alpha_history,
			"rho_history" => rho_history,
			"arm_speed_limit" => arm_speed_limit,
			"child_speed_limit" => child_speed_limit,
		)

		(; strategies, solution_dict)
	end

	# ── Output directories ─────────────────────────────────────────────────────
	output_dirs =
		@timeit TO "output directory setup" prepare_robotic_arm_output_dirs(run_id; debug)
	(;
		run_dir,
		problem_data_dir,
		solution_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		speed_plots_dir,
		control_plots_dir,
		distance_plots_dir,
		warmstart_plots_dir,
	) = output_dirs
	visualization_config =
		VisualizationConfig(; dirs = output_dirs, show_interactive_trajectory)

	# ── Main solve loop ────────────────────────────────────────────────────────
	instance_problem_data = Dict{String, Any}[]
	solved_attempts = 0
	total_attempts = 0

	while solved_attempts < num_instances
		total_attempts += 1
		initial_state1, initial_state2 = if random_initial_state
			(
				sample_initial_state(
					dynamics_model,
					base_initial_state1,
					state_dimension,
					perturbation_scale,
				),
				sample_initial_state(
					dynamics_model,
					base_initial_state2,
					state_dimension,
					perturbation_scale,
				),
			)
		else
			(copy(base_initial_state1), copy(base_initial_state2))
		end
		println(
			"solved $(solved_attempts)/$(num_instances), attempt $(total_attempts), goop version $(goop_version): ",
		)
		println("initial_state1:", initial_state1)
		println("goal_position1:", goal_position1)
		println("initial_state2:", initial_state2)
		println("goal_position2:", goal_position2)
		println("initial_state3:", initial_state3)
		println("goal_position3:", goal_position3)

		instance_states = (; initial_state1, initial_state2, initial_state3)
		(; warmstart_solution) = @timeit TO "warmstart construction" if compute_warmstart
			build_default_warmstart(instance_states, scenario_config)
		else
			(;
				warmstart_solution = load(
					"experiments/solution_dict_instance_1_eps0.1.jld2",
				)["single_stored_object"]["x"][1:sum(primal_dimensions)],
				)
		end
		initial_controls = extract_initial_controls(
			warmstart_solution,
			primal_dimensions,
			dynamics,
		)
		instance_states = merge(instance_states, initial_controls)
		instance_parameters =
			@timeit TO "instance parameter construction" build_instance_parameters(
				flatten_parameters,
				instance_states,
				scenario_config,
			)
		(; θ1, θ2, θ3, θ) = instance_parameters
		@timeit TO "warmstart visualization" save_warmstart_visualizations(
			warmstart_solution;
			total_attempts,
			instance_idx = solved_attempts + 1,
			primal_dimensions,
			instance_parameters,
			scenario_config,
			visualization_config,
		)

		epsilon_results = Pair{Float64, Any}[]
		stage_warmstart = warmstart_solution
		solve_sequence_succeeded = true
		instance_total_solve_time_sec = 0.0

		for (ϵ₀, max_inner_iters) in zip(epsilon_schedule, max_inner_iters_schedule)
			result = try
				solve_game_instance(θ; z₀ = stage_warmstart, ϵ₀, max_inner_iters)
			catch err
				rethrow(err)
			end
			if isnothing(result)
				println(
					"attempt $(total_attempts): failed to converge for ϵ₀ = $(ϵ₀), resampling.",
				)
				solve_sequence_succeeded = false
				break
			end
			push!(epsilon_results, ϵ₀ => result)
			instance_total_solve_time_sec += result.solution_dict["solve_time_sec"]
			if result.solution_dict["status"] == :failed
				println(
					"attempt $(total_attempts): failed to converge for ϵ₀ = $(ϵ₀), saving diagnostics.",
				)
				solve_sequence_succeeded = false
				break
			end
			stage_warmstart = warmstart_solution
		end

		if !solve_sequence_succeeded
			failed_instance_idx = solved_attempts + 1
			for (ϵ₀, result) in epsilon_results
				failed_suffix = "_attempt_$(total_attempts)_failed"
				JLD2.save_object(
					joinpath(
						solution_data_dir,
						"solution_dict_instance_$(failed_instance_idx)_eps$(ϵ₀)$(failed_suffix).jld2",
					),
					result.solution_dict,
				)
				save_convergence_diagnostics(
					result.solution_dict,
					convergence_plots_dir,
					failed_instance_idx,
					ϵ₀;
					filename_suffix = failed_suffix,
				)
			end
			if !random_initial_state
				println("solver failed for default initial states.")
				break
			end
			continue
		end

		push!(
			instance_problem_data,
			Dict(
				"attempt_idx" => total_attempts,
				"initial_state1" => initial_state1,
				"goal_position1" => goal_position1,
				"initial_state2" => initial_state2,
				"goal_position2" => goal_position2,
				"initial_state3" => initial_state3,
				"goal_position3" => goal_position3,
				"arm_speed_limit" => arm_speed_limit,
				"child_speed_limit" => child_speed_limit,
				"total_solve_time_sec" => instance_total_solve_time_sec,
			),
		)

		solved_attempts += 1
		println(
			"instance $(solved_attempts) total solve time: $(round(instance_total_solve_time_sec; digits = 5)) sec",
		)
		println("instance $(solved_attempts) converged preference values by ϵ:")

		@timeit TO "problem data save" begin
			JLD2.save_object(
				joinpath(
					problem_data_dir,
					"problem_data_instance_$(solved_attempts).jld2",
				),
				instance_problem_data,
			)
		end

		for (ϵ₀, result) in epsilon_results
			solution_dict = result.solution_dict

			preference_values = evaluate_preferences_at_solution(
				problem,
				solution_dict["x"][1:sum(problem.primal_dims)],
				θ,
			)
			solution_dict["preference_values"] = preference_values
			println("  ϵ₀ = $(round(ϵ₀; digits = 5)):")
			println("  kkt_error = $(solution_dict["kkt_error"])")
			for (player_idx, player_preferences) in enumerate(preference_values)
				println("    player $(player_idx): $(round.(player_preferences; sigdigits = 5))")
			end
			dual_summaries = summarize_dual_blocks(GOOP_kkt_system, solution_dict["z"])
			solution_dict["dual_summaries"] = dual_summaries
			println("  dual blocks (max· / min· / ‖·‖₂ / #near-zero):")
			for s in dual_summaries
				s.name == "x" && continue # primal block, reported elsewhere
				println(
					"    $(s.name) (n=$(s.len)): $(round(s.max; sigdigits = 4)) / $(round(s.min; sigdigits = 4)) / $(round(s.norm2; sigdigits = 4)) / $(s.n_near_zero)",
				)
			end

			@timeit TO "solution output and plotting" begin
				JLD2.save_object(
					joinpath(
						solution_data_dir,
						"solution_dict_instance_$(solved_attempts)_eps$(ϵ₀).jld2",
					),
					solution_dict,
				)
				save_convergence_diagnostics(
					solution_dict,
					convergence_plots_dir,
					solved_attempts,
					ϵ₀,
				)
				save_solution_visualizations(
					result.strategies,
					solved_attempts,
					ϵ₀;
					instance_parameters,
					scenario_config,
					visualization_config,
				)
			end
		end
	end

	@timeit TO "run metadata save" begin
		JLD2.save_object(
			joinpath(problem_data_dir, "run_metadata.jld2"),
			Dict(
				"run_id" => run_id,
				"debug" => debug,
				"run_dir" => run_dir,
				"rng_seed" => rng_seed,
				"dynamics_model" => dynamics_model_name(dynamics_model),
				"num_instances" => num_instances,
				"random_initial_state" => random_initial_state,
				"use_scalarized_baseline" => use_scalarized_baseline,
				"use_social_equilibrium_baseline" => use_social_equilibrium_baseline,
				"Δt" => Δt,
				"use_running_goal_cost" => use_running_goal_cost,
				"use_up_and_over_warmstart" => use_up_and_over_warmstart,
				"show_interactive_trajectory" => show_interactive_trajectory,
				"epsilon_schedule" => epsilon_schedule,
				"max_inner_iters_schedule" => max_inner_iters_schedule,
				"arm_speed_limit" => arm_speed_limit,
				"child_speed_limit" => child_speed_limit,
				"collision_avoidance" => collision_avoidance,
				"child_initial_buffer" => child_initial_buffer,
				"dₚ" => dₚ,
				"perturbation_scale" => perturbation_scale,
			),
		)
	end

	println("\nTiming summary:")
	show(TO)
	println()
end

# ── Output / plotting helpers ──────────────────────────────────────────────────

function save_warmstart_visualizations(
	warmstart_solution;
	total_attempts = nothing,
	instance_idx = nothing,
	filename_tag = "attempt_$(total_attempts)_instance_$(instance_idx)",
	primal_dimensions,
	instance_parameters::InstanceParameters,
	scenario_config::ScenarioConfig,
	visualization_config::VisualizationConfig,
)
	(; θ1, θ2, θ3) = instance_parameters
	(;
		dynamics,
		map_end,
		lane_width,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
		dₚ,
		arm_speed_limit,
		child_speed_limit,
		control_bounds,
	) = scenario_config
	(; warmstart_plots_dir) = visualization_config.dirs
	static_extension = visualization_config.static_extension
	# Plots keep the per-arm view: split the combined two-arm strategy back
	# into [arm1, arm2, child].
	warmstart_strategies = split_arm_strategies(
		extract_player_strategies(warmstart_solution, primal_dimensions, dynamics),
	)

	warmstart_fig, _ = plot_single_integrator_3d_trajectories(;
		map_end,
		lane_width,
		strategy = warmstart_strategies,
		θ1,
		θ2,
		θ3,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
	)
	warmstart_speed_fig, _ = speed_plot(;
		strategy = warmstart_strategies,
		speed_limit = arm_speed_limit,
		dynamics_model = dynamics[1].model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
	)
	warmstart_control_fig, _ = control_plot(;
		strategy = warmstart_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	warmstart_distance_fig, _ = inter_player_distance_plot(;
		strategy = warmstart_strategies,
		reference_distance = dₚ,
		safety_distance = collision_avoidance,
	)

	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_$(filename_tag).$(static_extension)",
		),
		warmstart_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_speed_$(filename_tag).$(static_extension)",
		),
		warmstart_speed_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_control_$(filename_tag).$(static_extension)",
		),
		warmstart_control_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_distance_$(filename_tag).$(static_extension)",
		),
		warmstart_distance_fig,
	)
end

"""Save the standard static plots and optional interactive trajectory for one solution."""
function save_solution_visualizations(
	strategies,
	instance_idx,
	ϵ₀;
	instance_parameters::InstanceParameters,
	scenario_config::ScenarioConfig,
	visualization_config::VisualizationConfig,
)
	(; θ1, θ2, θ3) = instance_parameters
	(;
		dynamics_model,
		control_bounds,
		map_end,
		lane_width,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
		dₚ,
		arm_speed_limit,
		child_speed_limit,
	) = scenario_config
	(; trajectory_plots_dir, speed_plots_dir, control_plots_dir, distance_plots_dir) =
		visualization_config.dirs
	static_extension = visualization_config.static_extension
	interactive_extension = visualization_config.interactive_extension

	# Plots keep the per-arm view of the combined two-arm agent.
	plot_strategies = split_arm_strategies(strategies)
	trajectory_fig, _ = plot_single_integrator_3d_trajectories(;
		map_end,
		lane_width,
		strategy = plot_strategies,
		θ1,
		θ2,
		θ3,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
	)
	speed_fig, _ = speed_plot(;
		strategy = plot_strategies,
		speed_limit = arm_speed_limit,
		dynamics_model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
	)
	control_fig, _ = control_plot(;
		strategy = plot_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	distance_fig, _ = inter_player_distance_plot(;
		strategy = plot_strategies,
		reference_distance = dₚ,
		safety_distance = collision_avoidance,
	)

	plot_specs = (
		(trajectory_plots_dir, "trajectory", trajectory_fig),
		(speed_plots_dir, "speed", speed_fig),
		(control_plots_dir, "control", control_fig),
		(distance_plots_dir, "distance", distance_fig),
	)
	for (output_dir, plot_name, figure) in plot_specs
		save_figure(
			joinpath(
				output_dir,
				"$(plot_name)_instance_$(instance_idx)_eps$(ϵ₀).$(static_extension)",
			),
			figure,
		)
	end

	if visualization_config.show_interactive_trajectory
		interactive_trajectory_path = joinpath(
			trajectory_plots_dir,
			"trajectory_interactive_instance_$(instance_idx)_eps$(ϵ₀).$(interactive_extension)",
		)
		plot_trajectory_3d_interactive(;
			map_end,
			lane_width,
			strategy = plot_strategies,
			θ1,
			θ2,
			θ3,
			goal_position1,
			goal_position2,
			goal_position3,
			collision_avoidance,
			reference_distance = dₚ,
			display_figure = false,
			save_path = interactive_trajectory_path,
		)
		println(
			"saved interactive trajectory browser file: ",
			interactive_trajectory_path,
		)
	end
end

function prepare_robotic_arm_output_dirs(run_id; debug)
	run_dir = if debug
		joinpath("data", "robotic_arm_open_loop", "debug", run_id)
	else
		joinpath("data", "robotic_arm_open_loop", "runs", run_id)
	end

	data_dir = joinpath(run_dir, "data")
	problem_data_dir = joinpath(data_dir, "problem")
	solution_data_dir = joinpath(problem_data_dir, "solution")
	plots_dir = joinpath(run_dir, "plots")
	trajectory_plots_dir = joinpath(plots_dir, "trajectories")
	convergence_plots_dir = joinpath(plots_dir, "convergence")
	speed_plots_dir = joinpath(plots_dir, "speed")
	control_plots_dir = joinpath(plots_dir, "controls")
	distance_plots_dir = joinpath(plots_dir, "distance")
	warmstart_plots_dir = joinpath(plots_dir, "warmstart")

	for dir in (
		problem_data_dir,
		solution_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		speed_plots_dir,
		control_plots_dir,
		distance_plots_dir,
		warmstart_plots_dir,
	)
		mkpath(dir)
	end

	(;
		run_dir,
		data_dir,
		problem_data_dir,
		solution_data_dir,
		plots_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		speed_plots_dir,
		control_plots_dir,
		distance_plots_dir,
		warmstart_plots_dir,
	)
end

function safe_log10_history(history)
	map(history) do value
		isfinite(value) && value > 0 ? log10(value) : NaN
	end
end

function save_convergence_diagnostics(
	solution_dict,
	convergence_plots_dir,
	instance_idx,
	ϵ₀;
	filename_suffix = "",
)
	kkt_error_history = get(solution_dict, "kkt_error_history", Float64[])
	if !isempty(kkt_error_history)
		convergence_fig, _ = plot_convergence_plot(;
			kkt_error_history = safe_log10_history(kkt_error_history),
			total_iters = solution_dict["total_iters"],
		)
		save_figure(
			joinpath(
				convergence_plots_dir,
				"convergence_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			convergence_fig,
		)
	end

	condition_number_history = get(solution_dict, "condition_number_history", Float64[])
	if !isempty(condition_number_history)
		condition_number_fig, _ = plot_condition_number_plot(;
			condition_number_history = safe_log10_history(condition_number_history),
			total_iters = solution_dict["total_iters"],
		)
		save_figure(
			joinpath(
				convergence_plots_dir,
				"condition_number_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			condition_number_fig,
		)
	end

	eta_history = get(solution_dict, "eta_history", Float64[])
	if !isempty(eta_history)
		eta_fig, _ = plot_eta_plot(;
			eta_history = safe_log10_history(eta_history),
			total_iters = solution_dict["total_iters"],
		)
		save_figure(
			joinpath(
				convergence_plots_dir,
				"eta_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			eta_fig,
		)
	end

	alpha_history = get(solution_dict, "alpha_history", Float64[])
	if !isempty(alpha_history)
		alpha_fig, _ =
			plot_alpha_plot(; alpha_history, total_iters = solution_dict["total_iters"])
		save_figure(
			joinpath(
				convergence_plots_dir,
				"alpha_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			alpha_fig,
		)
	end

	rho_history = get(solution_dict, "rho_history", Float64[])
	if !isempty(rho_history)
		rho_fig, _ =
			plot_rho_plot(; rho_history, total_iters = solution_dict["total_iters"])
		save_figure(
			joinpath(
				convergence_plots_dir,
				"rho_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			rho_fig,
		)
	end
end

end
