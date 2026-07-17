module Robotic_arm_receding

# Forward BLAS/LAPACK to Apple Accelerate (AMX)
@static if Sys.isapple()
	using AppleAccelerate: AppleAccelerate
end

using CairoMakie: CairoMakie
using JLD2, Random
using Dates: Dates
using Statistics: mean, median, std
using ProgressMeter: ProgressMeter, @showprogress
using ReducedGOOP
using TimerOutputs: @timeit, reset_timer!

# Reuse an already-loaded (e.g. Revise-tracked via includet) Main.Robotic_arm
# instead of baking in a private copy: `includet` is non-recursive, so a copy
# created by a plain `include` here would never pick up edits to Robotic_arm.jl.
if isdefined(Main, :Robotic_arm)
	const RA = Main.Robotic_arm
else
	include(joinpath(@__DIR__, "Robotic_arm.jl"))
	const RA = Robotic_arm
end
const TO = ReducedGOOP.TO

# ── Experiment entry point ─────────────────────────────────────────────────────

"""
Receding-horizon (MPC) version of the robotic-arm experiment.

The symbolic KKT system is built exactly once for a fixed `planning_horizon`.
Every MPC iteration then only updates the initial-state entries of the
parameter vector `θ` and the primal warm start `z₀` before re-solving, so all
precomputed structures (compiled residuals, sparse Jacobian, solver options)
are reused across iterations.

At each MPC step the open-loop game is solved from the current state, the
first control is applied (the system advances to the first predicted state),
and the previous solution is shifted one stage to warm-start the next solve.
Non-converged solves are not fatal: the final iterate is used and the step is
marked as non-converged in the saved plots.
"""
function demo(;
	num_mpc_steps = 20,
	map_end = 10,
	lane_width = 2,
	verbose = false,
	rng_seed = 123,
	debug = false,
	show_interactive_trajectory = true,
	record_condition_number = false,
	record_convergence = true,
)
	reset_timer!(TO)
	@timeit TO "experiment setup" Random.seed!(rng_seed)

	# ── Settings ───────────────────────────────────────────────────────────────
	run_id = "Robotic_arm_receding_" * Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
	solver = ReducedGOOP.InteriorPoint()
	linesearch = :backtracking
	ϵ₀ = 0.1
	max_inner_iters = 500
	use_running_goal_cost = true

	# ── Scenario and problem (identical to the open-loop demo) ─────────────────
	# Player 1: combined two-arm agent, Player 2: child/pet.
	scenario_config = RA.demo_scenario_config(; use_running_goal_cost)
	(;
		dynamics_model,
		dynamics,
		control_bounds,
		num_players,
		planning_horizon,
		base_initial_state1,
		base_initial_state2,
		goal_position1,
		goal_position2,
		initial_state3,
		goal_position3,
		collision_avoidance,
		arm_speed_limit,
		child_speed_limit,
		dₚ,
	) = scenario_config
	state_dimension = scenario_config.position_dimension

	problem_setup_time_sec = @elapsed @timeit TO "problem setup" begin
		(; problem, flatten_parameters) = RA.get_setup(scenario_config)
	end
	# Built exactly once; every MPC iteration reuses the compiled KKT system and
	# only changes θ (initial states) and the warm start. 
	@info "Building KKT system once for the receding-horizon loop (T = $(planning_horizon))..."
	kkt_build_time_sec = @elapsed GOOP_kkt_system = @timeit TO "KKT construction" begin
		ReducedGOOP.generate_slacked_reduced_kkt_system(
			problem;
			backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
			backend_options = (;),
			codegen = :fast_differentiation,
		)
	end
	println(
		"one-time setup: problem construction $(round(problem_setup_time_sec; digits = 2)) s, ",
		"symbolic KKT build $(round(kkt_build_time_sec; digits = 2)) s",
	)
	println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
	println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)

	# Same options as the open-loop demo, except record_condition_number stays a
	# kwarg (default false): it costs a dense SVD per Newton iteration, which is
	# prohibitive across an MPC loop.
	options = ReducedGOOP.InteriorPointOptions(;
		tol = 0.01,
		η₀ = 1e-6,
		η_max = 1e6,
		ϵ₀,
		max_inner_iters,
		max_outer_iters = 1,
		tightening_rate = 1.2,
		loosening_rate = 3.0,
		min_stepsize = 1e-20,
		linesearch,
		linear_solve_algorithm = ReducedGOOP.LinearSolve.KrylovJL_LSMR(),
		use_linsolve = false,
		record_convergence,
		record_condition_number,
		eta_retry_growth = 2.0,
		ρ_low = 0.75,
		ρ_high = 0.75,
		perturbation_enabled = false,
		stagnation_rtol = 1e-1,
		perturbation_scale = 1e-6,
		tsvd_threshold = 0.0,
		use_marquardt_scaling = false,
		verbose,
	)

	primal_dimensions = [
		(dyn.state_dimension + dyn.control_dimension) * planning_horizon for dyn in dynamics
	]

	# ── Output directories ─────────────────────────────────────────────────────
	dirs = @timeit TO "output directory setup" prepare_receding_output_dirs(run_id; debug)
	visualization_config = RA.VisualizationConfig(;
		dirs,
		show_interactive_trajectory,
	)

	# ── MPC state ──────────────────────────────────────────────────────────────
	current_state1 = copy(base_initial_state1)
	current_state2 = copy(base_initial_state2)
	current_state_child = copy(initial_state3)

	closed_loop_xs = [
		[vcat(current_state1, current_state2)],
		[copy(current_state_child)],
	]
	closed_loop_us = [Vector{Float64}[] for _ in 1:num_players]

	instance_states = (;
		initial_state1 = current_state1,
		initial_state2 = current_state2,
		initial_state3 = current_state_child,
	)
	(; warmstart_solution) = @timeit TO "warmstart construction" RA.build_default_warmstart(
		instance_states,
		scenario_config,
	)
	stage_warmstart = warmstart_solution

	step_statuses    = Symbol[]
	step_kkt_errors  = Float64[]
	step_solve_times = Float64[]
	step_total_iters = Int[]
	θ1_initial      = θ2_initial = θ3_initial = nothing

	# ── MPC loop ───────────────────────────────────────────────────────────────
	for k in 1:num_mpc_steps
		# Refresh the initial-state parameters without rebuilding the KKT system.
		instance_states = (;
			initial_state1 = current_state1,
			initial_state2 = current_state2,
			initial_state3 = current_state_child,
		)
		instance_parameters = @timeit TO "instance parameter construction" RA.build_instance_parameters(
			flatten_parameters,
			instance_states,
			scenario_config,
		)
		(; θ1, θ2, θ3, θ) = instance_parameters
		if k == 1
			θ1_initial, θ2_initial, θ3_initial = θ1, θ2, θ3
			@timeit TO "warmstart visualization" RA.save_warmstart_visualizations(
				stage_warmstart;
				total_attempts = 1,
				instance_idx = 1,
				primal_dimensions,
				instance_parameters,
				scenario_config,
				visualization_config,
			)
		end

		elapsed_time = @elapsed output = @timeit TO "solver invocation" ReducedGOOP.solve(
			solver,
			GOOP_kkt_system,
			θ;
			z₀ = stage_warmstart,
			options,
		)
		converged = output.status == :solved
		push!(step_statuses, output.status)
		push!(step_kkt_errors, output.kkt_error)
		push!(step_solve_times, elapsed_time)
		push!(step_total_iters, output.total_iters)
		println(
			"MPC step $(k)/$(num_mpc_steps): status=$(output.status), ",
			"iters=$(output.total_iters), ",
			"kkt_error=$(round(output.kkt_error; sigdigits = 4)), ",
			"time=$(round(elapsed_time; digits = 3)) sec",
		)
		if !converged
			println(
				"  step $(k) did not converge within max_inner_iters=$(max_inner_iters); ",
				"continuing with the final iterate.",
			)
		end

		strategies = @timeit TO "solution postprocessing" RA.extract_player_strategies(
			output.x,
			primal_dimensions,
			dynamics,
		)

		# Apply the first control: advance every player to its first predicted
		# state (equivalent to applying us[1] through the dynamics constraints).
		for player in 1:num_players
			push!(closed_loop_us[player], collect(strategies[player].us[1]))
			push!(closed_loop_xs[player], collect(strategies[player].xs[2]))
		end
		combined_arm_state  = closed_loop_xs[1][end]
		current_state1      = combined_arm_state[1:state_dimension]
		current_state2      = combined_arm_state[(state_dimension+1):(2*state_dimension)]
		current_state_child = closed_loop_xs[2][end]

		solution_dict = Dict(
			"mpc_step" => k,
			"strategies" => strategies,
			"z" => output.z,
			"x" => output.x,
			"warmstart" => stage_warmstart,
			"solve_time_sec" => elapsed_time,
			"kkt_error" => output.kkt_error,
			"ϵ" => output.ϵ,
			"status" => output.status,
			"total_iters" => output.total_iters,
			"kkt_error_history" => output.kkt_error_history,
			"condition_number_history" => output.condition_number_history,
			"eta_history" => output.eta_history,
			"alpha_history" => output.alpha_history,
			"rho_history" => output.rho_history,
			"arm_speed_limit" => arm_speed_limit,
			"child_speed_limit" => child_speed_limit,
		)

		@timeit TO "solution output and plotting" begin
			# The interactive export of the previous step leaves WGLMakie active;
			# static PDF saving needs the CairoMakie backend.
			CairoMakie.activate!()
			status_suffix = converged ? "" : "_notconverged"
			JLD2.save_object(
				joinpath(
					dirs.solution_data_dir,
					"solution_dict_step_$(k)_eps$(ϵ₀)$(status_suffix).jld2",
				),
				solution_dict,
			)
			RA.save_convergence_diagnostics(
				solution_dict,
				dirs.convergence_plots_dir,
				k,
				ϵ₀;
				filename_suffix = status_suffix,
			)
			save_step_plots(
				strategies,
				k,
				num_mpc_steps,
				converged,
				output.kkt_error,
				dirs;
				map_end,
				lane_width,
				θ1,
				θ2,
				θ3,
				goal_position1,
				goal_position2,
				goal_position3,
				collision_avoidance,
				dₚ,
				arm_speed_limit,
				child_speed_limit,
				control_bounds,
				dynamics_model,
				save_interactive = show_interactive_trajectory,
			)
		end

		# Shifted previous solution warm-starts the next solve.
		shifted_strategies = @timeit TO "warmstart shift" shift_strategies(
			strategies,
			dynamics,
			planning_horizon,
		)
		stage_warmstart = RA.flatten_warmstart_solution(
			planning_horizon,
			[strategy.xs for strategy in shifted_strategies],
			[strategy.us for strategy in shifted_strategies],
		)
	end

	# ── Closed-loop outputs ────────────────────────────────────────────────────
	failed_steps = findall(==(:failed), step_statuses)
	println(
		"\nMPC run finished: $(num_mpc_steps - length(failed_steps))/$(num_mpc_steps) steps converged",
		isempty(failed_steps) ? "." : "; non-converged steps: $(join(failed_steps, ", ")).",
	)

	closed_loop_strategies = map(1:num_players) do player
		# Pad with a terminal zero control so xs/us have equal length, matching
		# the trajectory layout the plotting helpers expect.
		(;
			xs = closed_loop_xs[player],
			us = vcat(closed_loop_us[player], [zeros(dynamics[player].control_dimension)]),
		)
	end

	@timeit TO "closed-loop output and plotting" begin
		JLD2.save_object(
			joinpath(dirs.problem_data_dir, "closed_loop_trajectory.jld2"),
			Dict(
				"closed_loop_strategies" => closed_loop_strategies,
				"closed_loop_xs" => closed_loop_xs,
				"closed_loop_us" => closed_loop_us,
				"step_statuses" => step_statuses,
				"step_kkt_errors" => step_kkt_errors,
				"step_solve_times" => step_solve_times,
				"step_total_iters" => step_total_iters,
			),
		)
		save_closed_loop_plots(
			closed_loop_strategies,
			failed_steps,
			num_mpc_steps,
			dirs;
			map_end,
			lane_width,
			θ1 = θ1_initial,
			θ2 = θ2_initial,
			θ3 = θ3_initial,
			goal_position1,
			goal_position2,
			goal_position3,
			collision_avoidance,
			dₚ,
			arm_speed_limit,
			child_speed_limit,
			control_bounds,
			dynamics_model,
			save_interactive = true,
		)
	end

	# ── Runtime report ─────────────────────────────────────────────────────────
	timing_stats = @timeit TO "runtime report" begin
		stats = print_runtime_report(
			problem_setup_time_sec,
			kkt_build_time_sec,
			step_solve_times,
			step_statuses,
		)
		save_solve_time_plots(step_solve_times, step_statuses, dirs.timing_plots_dir)
		JLD2.save_object(
			joinpath(dirs.problem_data_dir, "timing_summary.jld2"),
			Dict(
				"problem_setup_time_sec" => problem_setup_time_sec,
				"kkt_build_time_sec" => kkt_build_time_sec,
				"step_solve_times" => step_solve_times,
				"step_statuses" => step_statuses,
				"all_solve_stats" => stats.all_stats,
				"warm_solve_stats" => stats.warm_stats,
			),
		)
		stats
	end

	@timeit TO "run metadata save" begin
		JLD2.save_object(
			joinpath(dirs.problem_data_dir, "run_metadata.jld2"),
			Dict(
				"run_id" => run_id,
				"debug" => debug,
				"run_dir" => dirs.run_dir,
				"rng_seed" => rng_seed,
				"dynamics_model" => RA.dynamics_model_name(dynamics_model),
				"planning_horizon" => planning_horizon,
				"num_mpc_steps" => num_mpc_steps,
				"ϵ₀" => ϵ₀,
				"max_inner_iters" => max_inner_iters,
				"arm_speed_limit" => arm_speed_limit,
				"child_speed_limit" => child_speed_limit,
				"collision_avoidance" => collision_avoidance,
				"dₚ" => dₚ,
				"initial_state1" => base_initial_state1,
				"initial_state2" => base_initial_state2,
				"initial_state3" => initial_state3,
				"goal_position1" => goal_position1,
				"goal_position2" => goal_position2,
				"goal_position3" => goal_position3,
				"step_statuses" => step_statuses,
				"step_kkt_errors" => step_kkt_errors,
				"step_solve_times" => step_solve_times,
				"problem_setup_time_sec" => problem_setup_time_sec,
				"kkt_build_time_sec" => kkt_build_time_sec,
				"total_solve_time_sec" => sum(step_solve_times),
			),
		)
	end

	println("outputs saved under: ", dirs.run_dir)
	println("\nTiming summary:")
	show(TO)
	println()

	(;
		closed_loop_strategies,
		step_statuses,
		step_kkt_errors,
		step_solve_times,
		timing_stats,
		run_dir = dirs.run_dir,
	)
end

# ── MPC utilities ──────────────────────────────────────────────────────────────

"""
Shift a per-player trajectory one stage forward to warm-start the next MPC
solve. The terminal stage is padded with a zero control passed through the
dynamics (trivially feasible); for the single-integrator
dynamics the appended terminal state repeats the previous final state.
"""
function shift_strategies(strategies, dynamics, planning_horizon)
	map(collect(enumerate(strategies))) do (player, strategy)
		xs = [collect(Float64, x) for x in strategy.xs[2:end]]
		us = [collect(Float64, u) for u in strategy.us[2:end]]
		# terminal_control = copy(us[end])
		terminal_control = zeros(dynamics[player].control_dimension)
		push!(xs, dynamics[player].step(xs[end], terminal_control, planning_horizon))
		push!(us, terminal_control)
		(; xs, us)
	end
end

# ── Runtime reporting ──────────────────────────────────────────────────────────

function _solve_time_stats(times)
	(;
		total = sum(times),
		mean = mean(times),
		median = median(times),
		min = minimum(times),
		max = maximum(times),
		std = length(times) > 1 ? std(times) : 0.0,
	)
end

function _format_stats(stats)
	"total $(round(stats.total; digits = 2)) s, " *
	"mean $(round(stats.mean; digits = 2)) s, " *
	"median $(round(stats.median; digits = 2)) s, " *
	"min $(round(stats.min; digits = 2)) s, " *
	"max $(round(stats.max; digits = 2)) s, " *
	"std $(round(stats.std; digits = 2)) s"
end

"""
Print a runtime report that separates one-time compilation/setup cost from the
repeated per-step solve cost, and return the solve-time statistics.
"""
function print_runtime_report(
	problem_setup_time_sec,
	kkt_build_time_sec,
	step_solve_times,
	step_statuses,
)
	num_steps = length(step_solve_times)
	println("\n── Runtime report ──────────────────────────────────────────────")
	println("One-time setup (paid once, reused by all $(num_steps) solves):")
	println("  problem construction:      $(round(problem_setup_time_sec; digits = 2)) s")
	println("  symbolic KKT system build: $(round(kkt_build_time_sec; digits = 2)) s")
	println("  setup total:               $(round(problem_setup_time_sec + kkt_build_time_sec; digits = 2)) s")
	println("Per-step solve times:")
	for (k, (t, status)) in enumerate(zip(step_solve_times, step_statuses))
		println("  step $(lpad(k, 2)): $(lpad(round(t; digits = 3), 9)) s  ($(status))")
	end
	all_stats = _solve_time_stats(step_solve_times)
	warm_stats = num_steps > 1 ? _solve_time_stats(step_solve_times[2:end]) : nothing
	println("All $(num_steps) solves: $(_format_stats(all_stats))")
	if !isnothing(warm_stats)
		jit_estimate = step_solve_times[1] - warm_stats.median
		println(
			"First solve (step 1) additionally pays the one-time JIT compilation ",
			"of the solver code: $(round(step_solve_times[1]; digits = 2)) s",
		)
		println(
			"  estimated JIT overhead ≈ $(round(jit_estimate; digits = 2)) s ",
			"(step 1 minus median warm solve)",
		)
		println("Warm solves (steps 2–$(num_steps)): $(_format_stats(warm_stats))")
	end
	println("────────────────────────────────────────────────────────────────")
	(; all_stats, warm_stats)
end

"""
Save a histogram of per-step solve times and a per-step bar chart colored by
convergence status. The first solve is excluded from the histogram when it is
dominated by one-time JIT compilation (> 3× the median warm solve); its value
is reported in the title instead so the warm-solve distribution stays readable.
"""
function save_solve_time_plots(step_solve_times, step_statuses, timing_plots_dir)
	CairoMakie.activate!()
	num_steps = length(step_solve_times)

	warm_times = step_solve_times[min(2, num_steps):end]
	exclude_first = num_steps > 2 && step_solve_times[1] > 3 * median(warm_times)
	hist_times = exclude_first ? step_solve_times[2:end] : step_solve_times
	hist_title = exclude_first ?
				 "Warm-solve times (step 1 excluded: $(round(step_solve_times[1]; digits = 1)) s incl. one-time JIT)" :
				 "Per-step solve times"
	hist_fig = RA.serif_figure(size = (900, 600))
	hist_ax = CairoMakie.Axis(
		hist_fig[1, 1];
		xlabel = "solve time [s]",
		ylabel = "number of MPC steps",
		title = hist_title,
	)
	CairoMakie.hist!(
		hist_ax,
		hist_times;
		bins = max(6, ceil(Int, 2 * sqrt(length(hist_times)))),
		color = (:dodgerblue, 0.85),
		strokecolor = :black,
		strokewidth = 1,
	)
	RA.save_figure(joinpath(timing_plots_dir, "solve_time_histogram.pdf"), hist_fig)

	bar_fig = RA.serif_figure(size = (950, 600))
	bar_ax = CairoMakie.Axis(
		bar_fig[1, 1];
		xlabel = "MPC step",
		ylabel = "solve time [s]",
		title = "Solve time per MPC step (green = converged, red = not converged)",
	)
	CairoMakie.barplot!(
		bar_ax,
		1:num_steps,
		step_solve_times;
		color = [status == :solved ? :seagreen : :firebrick for status in step_statuses],
		strokecolor = :black,
		strokewidth = 0.5,
	)
	RA.save_figure(joinpath(timing_plots_dir, "solve_time_per_step.pdf"), bar_fig)
end

# ── Output / plotting helpers ──────────────────────────────────────────────────

function _add_title!(figure, title_text)
	CairoMakie.Label(figure[0, :], title_text; fontsize = 20, tellwidth = false)
end

function _step_title(k, num_mpc_steps, converged, kkt_error)
	title = "MPC step $(k)/$(num_mpc_steps)"
	converged ? title :
	title * " — NOT CONVERGED (KKT error $(round(kkt_error; sigdigits = 3)))"
end

function save_step_plots(
	strategies,
	k,
	num_mpc_steps,
	converged,
	kkt_error,
	dirs;
	map_end,
	lane_width,
	θ1,
	θ2,
	θ3,
	goal_position1,
	goal_position2,
	goal_position3,
	collision_avoidance,
	dₚ,
	arm_speed_limit,
	child_speed_limit,
	control_bounds,
	dynamics_model,
	save_interactive,
)
	# Plots keep the per-arm view of the combined two-arm agent.
	plot_strategies = RA.split_arm_strategies(strategies)
	status_suffix = converged ? "" : "_notconverged"
	title = _step_title(k, num_mpc_steps, converged, kkt_error)

	trajectory_fig, _ = RA.plot_single_integrator_3d_trajectories(
		;
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
	_add_title!(trajectory_fig, title)
	speed_fig, _ = RA.speed_plot(;
		strategy = plot_strategies,
		speed_limit = arm_speed_limit,
		dynamics_model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
	)
	_add_title!(speed_fig, title)
	control_fig, _ = RA.control_plot(;
		strategy = plot_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	_add_title!(control_fig, title)
	distance_fig, _ = RA.inter_player_distance_plot(;
		strategy = plot_strategies,
		reference_distance = dₚ,
		safety_distance = collision_avoidance,
	)
	_add_title!(distance_fig, title)

	RA.save_figure(
		joinpath(dirs.trajectory_plots_dir, "trajectory_step_$(k)$(status_suffix).pdf"),
		trajectory_fig,
	)
	RA.save_figure(
		joinpath(dirs.speed_plots_dir, "speed_step_$(k)$(status_suffix).pdf"),
		speed_fig,
	)
	RA.save_figure(
		joinpath(dirs.control_plots_dir, "control_step_$(k)$(status_suffix).pdf"),
		control_fig,
	)
	RA.save_figure(
		joinpath(dirs.distance_plots_dir, "distance_step_$(k)$(status_suffix).pdf"),
		distance_fig,
	)

	if save_interactive
		interactive_path = joinpath(
			dirs.trajectory_plots_dir,
			"trajectory_interactive_step_$(k)$(status_suffix).html",
		)
		RA.plot_trajectory_3d_interactive(
			;
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
			save_path = interactive_path,
		)
		println("saved interactive trajectory browser file: ", interactive_path)
	end
end

function save_closed_loop_plots(
	closed_loop_strategies,
	failed_steps,
	num_mpc_steps,
	dirs;
	map_end,
	lane_width,
	θ1,
	θ2,
	θ3,
	goal_position1,
	goal_position2,
	goal_position3,
	collision_avoidance,
	dₚ,
	arm_speed_limit,
	child_speed_limit,
	control_bounds,
	dynamics_model,
	save_interactive,
)
	plot_strategies = RA.split_arm_strategies(closed_loop_strategies)
	title = "Closed-loop receding-horizon execution ($(num_mpc_steps) steps)"
	if !isempty(failed_steps)
		title *= " — non-converged steps: $(join(failed_steps, ", "))"
	end
	trajectory_fig, _ = RA.plot_single_integrator_3d_trajectories(
		;
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
	_add_title!(trajectory_fig, title)
	speed_fig, _ = RA.speed_plot(;
		strategy = plot_strategies,
		speed_limit = arm_speed_limit,
		dynamics_model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
	)
	_add_title!(speed_fig, title)
	control_fig, _ = RA.control_plot(;
		strategy = plot_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	_add_title!(control_fig, title)
	distance_fig, _ = RA.inter_player_distance_plot(;
		strategy = plot_strategies,
		reference_distance = dₚ,
		safety_distance = collision_avoidance,
	)
	_add_title!(distance_fig, title)

	RA.save_figure(
		joinpath(dirs.closed_loop_plots_dir, "trajectory_closed_loop.pdf"),
		trajectory_fig,
	)
	RA.save_figure(joinpath(dirs.closed_loop_plots_dir, "speed_closed_loop.pdf"), speed_fig)
	RA.save_figure(
		joinpath(dirs.closed_loop_plots_dir, "control_closed_loop.pdf"),
		control_fig,
	)
	RA.save_figure(
		joinpath(dirs.closed_loop_plots_dir, "distance_closed_loop.pdf"),
		distance_fig,
	)

	if save_interactive
		interactive_path = joinpath(
			dirs.closed_loop_plots_dir,
			"trajectory_interactive_closed_loop.html",
		)
		RA.plot_trajectory_3d_interactive(
			;
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
			save_path = interactive_path,
		)
		println("saved closed-loop interactive trajectory browser file: ", interactive_path)
	end
end

function prepare_receding_output_dirs(run_id; debug)
	run_dir = if debug
		joinpath("data", "robotic_arm_receding_horizon", "debug", run_id)
	else
		joinpath("data", "robotic_arm_receding_horizon", "runs", run_id)
	end

	data_dir              = joinpath(run_dir, "data")
	problem_data_dir      = joinpath(data_dir, "problem")
	solution_data_dir     = joinpath(problem_data_dir, "solution")
	plots_dir             = joinpath(run_dir, "plots")
	trajectory_plots_dir  = joinpath(plots_dir, "trajectories")
	convergence_plots_dir = joinpath(plots_dir, "convergence")
	speed_plots_dir       = joinpath(plots_dir, "speed")
	control_plots_dir     = joinpath(plots_dir, "controls")
	distance_plots_dir    = joinpath(plots_dir, "distance")
	warmstart_plots_dir   = joinpath(plots_dir, "warmstart")
	closed_loop_plots_dir = joinpath(plots_dir, "closed_loop")
	timing_plots_dir      = joinpath(plots_dir, "timing")

	for dir in (
		problem_data_dir,
		solution_data_dir,
		trajectory_plots_dir,
		convergence_plots_dir,
		speed_plots_dir,
		control_plots_dir,
		distance_plots_dir,
		warmstart_plots_dir,
		closed_loop_plots_dir,
		timing_plots_dir,
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
		closed_loop_plots_dir,
		timing_plots_dir,
	)
end

end
