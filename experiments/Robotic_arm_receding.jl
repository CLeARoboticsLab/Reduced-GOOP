module Robotic_arm_receding

# Forward BLAS/LAPACK to Apple Accelerate (AMX)
@static if Sys.isapple()
	using AppleAccelerate: AppleAccelerate
end

using CairoMakie: CairoMakie
using JLD2, Random
using Dates: Dates
using Statistics: mean, median, std
using ReducedGOOP
using TimerOutputs: @timeit, reset_timer!

# Reuse a Revise-tracked core when one is already loaded in Main. The fallback
# keeps this entry point loadable on its own in a fresh Julia process.
const ROBOTIC_ARM_CORE_PATH = joinpath(@__DIR__, "robotic_arm_core.jl")
if !isdefined(Main, :RoboticArmCore)
	Base.include(Main, ROBOTIC_ARM_CORE_PATH)
end
Main.RoboticArmCore isa Module ||
	error("Main.RoboticArmCore exists but is not a module.")
using Main.RoboticArmCore

# Preserve qualified access to the core API from this entry-point module.
for core_symbol in names(Main.RoboticArmCore)
	core_symbol === :RoboticArmCore && continue
	@eval export $core_symbol
end

# Share the visualization layer with the open-loop entry point; neither entry
# point depends on the other.
include(joinpath(@__DIR__, "robotic_arm_visualization.jl"))

const DUAL_WARMSTART_MODES = (
	:primal_only,
	:equality_duals,
	:all_except_innermost_stationarity,
)

# ── Experiment entry point ─────────────────────────────────────────────────────

"""
Receding-horizon (MPC) version of the robotic-arm experiment.

The symbolic KKT system is built exactly once for a fixed `planning_horizon`.
Every MPC iteration then only updates the initial-state and initial-control
entries of the parameter vector `θ` and the primal warm start `z₀` before
re-solving, so all precomputed structures (compiled residuals, sparse Jacobian,
solver options) are reused across iterations.

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
    goop_version = :quasi,     # :complete | :reduced | :quasi
	show_interactive_trajectory = false,
	record_condition_number = false,
	record_convergence = false,
	linear_solver = :klu,
	# FastDifferentiation otherwise emits all sparse-Jacobian entries as one
	# enormous RuntimeGeneratedFunction. Bounded chunks avoid pathological
	# first-call inference/lowering while preserving the same expressions.
	fd_codegen_chunk_size = 128,
	# `:primal_only` resets every non-primal variable; `:equality_duals` carries
	# λᵢₖ; `:all_except_innermost_stationarity` carries every dual except the ψ
	# segments targeting preference level Kⁱ. Only primals are horizon-shifted.
	dual_warmstart = :all_except_innermost_stationarity,
	reuse_factorization_iters = 0,
	# η_max = 1e6 lets a struggling step escalate the regularization into a
	# do-nothing limit cycle: LM steps scale like 1/η, so at η ~ 1e6 the solver
	# takes microscopic "full" steps that pass the line search but reduce the
	# residual by ~1e-5 per iteration until the budget is gone. Successful
	# steps never use η above ~1e-4, so 1e2 is a generous ceiling that makes
	# genuinely stuck solves fail fast instead.
	η_max = 1e2,
	# Growth used only after a singular KLU numeric factorization. Each retry
	# rebuilds the failed factorization cache before applying this increase.
	klu_singularity_eta_growth = 100.0,
	# Re-solve a failed step once from the default (cold) warmstart.
	rescue_failed_steps = true,
	# Armijo sufficient-decrease constant for the backtracking line search;
	# 0.0 recovers the pre-Armijo accept-any-decrease behavior.
	armijo_constant = 1e-4,
	tol = 0.008,
	# Newton iteration budget per MPC step. Raise it when a stiff grid (short
	# horizon relative to the distance the arm must cover) needs more iterations
	# than the default before the step is declared failed.
	max_inner_iters = 500,
	# Initial Levenberg-Marquardt regularization. Worth raising when solves stall
	# or diverge on ill-conditioned grids.
	η₀ = 1e-6,
	save_outputs = true,
	# Planning grid. `nothing` keeps the scenario defaults (T = 30, Δt = 0.1);
	# the Δt/T benchmark overrides both to hold Δt·T fixed.
	planning_horizon = nothing,
	Δt = nothing,
)
	reset_timer!(TO)
	@timeit TO "experiment setup" Random.seed!(rng_seed)

	dual_warmstart in DUAL_WARMSTART_MODES || throw(
		ArgumentError(
			"Unknown dual_warmstart mode $(dual_warmstart); expected one of " *
			"$(join(DUAL_WARMSTART_MODES, ", ")).",
		),
	)

	# ── Settings ───────────────────────────────────────────────────────────────
	run_id =
		"Robotic_arm_receding_" * Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS") *
		"_$(goop_version)_$(dual_warmstart)" *
		(isnothing(planning_horizon) ? "" : "_T$(planning_horizon)") *
		(isnothing(Δt) ? "" : "_dt$(Δt)")
	kkt_backend = :symbolics
	kkt_backend_options = (;)
	kkt_codegen = :fast_differentiation
	solver = ReducedGOOP.InteriorPoint()
	linesearch = :backtracking
	ϵ₀ = 0.1 # placeholder in the robotic arm scenario (no inequality constraints here)
	use_running_goal_cost = false

	# ── Scenario and problem (identical to the open-loop demo) ─────────────────
	# Player 1: combined two-arm agent, Player 2: child/pet.
	scenario_overrides = (;
		(isnothing(planning_horizon) ? (;) : (; planning_horizon))...,
		(isnothing(Δt) ? (;) : (; Δt))...,
	)
	scenario_config =
		demo_scenario_config(; use_running_goal_cost, scenario_overrides...)
	# The scenario is the single source of truth from here on; re-bind the local
	# names so a `nothing` kwarg picks up the scenario default.
	(;
		dynamics_model,
		dynamics,
		control_bounds,
		num_players,
		planning_horizon,
		Δt,
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
	arm_state_dimension, state_remainder = divrem(dynamics[1].state_dimension, 2)
	arm_control_dimension, control_remainder = divrem(dynamics[1].control_dimension, 2)
	iszero(state_remainder) || error("Expected an even stacked two-arm state dimension.")
	iszero(control_remainder) ||
		error("Expected an even stacked two-arm control dimension.")

	problem_setup_time_sec = @elapsed @timeit TO "problem setup" begin
		(; problem, flatten_parameters) = get_setup(scenario_config)
	end

	# Built exactly once; every MPC iteration reuses the compiled KKT system and
	# only changes θ (initial states and controls) and the warm start.
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

	@info "Building $(goop_version) KKT system once for the receding-horizon loop (T = $(planning_horizon))..."
	kkt_build_time_sec = @elapsed GOOP_kkt_system = @timeit TO "KKT construction" begin
		GOOP_kkt_generator(
			problem;
			backend,
			backend_options = kkt_backend_options,
			codegen = kkt_codegen,
			fd_codegen_chunk_size,
		)
	end
	println(
		"one-time setup: problem construction $(round(problem_setup_time_sec; digits = 2)) s, ",
		"symbolic KKT build $(round(kkt_build_time_sec; digits = 2)) s",
	)
	println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
	println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)
	total_equality_stationarity_dual_count =
		length(GOOP_kkt_system.all_equality_stationarity_dual_dims)
	selected_dual_count = if dual_warmstart === :primal_only
		0
	elseif dual_warmstart === :equality_duals
		length(GOOP_kkt_system.equality_constraint_dual_dims)
	else
		total_equality_stationarity_dual_count -
		length(GOOP_kkt_system.innermost_stationarity_dual_dims)
	end
	println(
		"dual warm-start mode: $(dual_warmstart) " *
		"($(selected_dual_count)/$(total_equality_stationarity_dual_count) " *
		"equality/stationarity dual coordinates carried)",
	)

	# Same options as the open-loop demo, except record_condition_number stays a
	# kwarg (default false): it costs a dense SVD per Newton iteration, which is
	# prohibitive across an MPC loop.
	options = ReducedGOOP.InteriorPointOptions(;
		tol,
		η₀ = 1e-6,
		η_max,
		ϵ₀,
		max_inner_iters,
		max_outer_iters = 1,
		tightening_rate = 1.2,
		loosening_rate = 3.0,
		min_stepsize = 1e-20,
		linesearch,
		record_convergence,
		record_condition_number,
		eta_retry_growth = 2.0,
		ρ_low = 0.75,
		ρ_high = 0.75,
		tsvd_threshold = 0.0,
		use_marquardt_scaling = false,
		linear_solver,
		klu_singularity_eta_growth,
		armijo_constant,
		reuse_factorization_iters,
		verbose,
	)

	primal_dimensions = [
		(dyn.state_dimension + dyn.control_dimension) * planning_horizon for
		dyn in dynamics
	]

	# ── Output directories ─────────────────────────────────────────────────────
	dirs = if save_outputs
		@timeit TO "output directory setup" prepare_receding_output_dirs(run_id; debug)
	else
		nothing
	end
	visualization_config =
		save_outputs ? VisualizationConfig(; dirs, show_interactive_trajectory) : nothing

	# ── MPC state ──────────────────────────────────────────────────────────────
	current_state1 = copy(base_initial_state1)
	current_state2 = copy(base_initial_state2)
	current_state_child = copy(initial_state3)

	closed_loop_xs = [[vcat(current_state1, current_state2)], [copy(current_state_child)]]
	closed_loop_us = [Vector{Float64}[] for _ in 1:num_players]

	instance_states = (;
		initial_state1 = current_state1,
		initial_state2 = current_state2,
		initial_state3 = current_state_child,
	)
	(; warmstart_solution) =
		@timeit TO "warmstart construction" build_default_warmstart(
			instance_states,
			scenario_config,
		)
	stage_warmstart = warmstart_solution
	initial_controls = extract_initial_controls(
		stage_warmstart,
		primal_dimensions,
		dynamics,
	)
	current_control1 = initial_controls.initial_control1
	current_control2 = initial_controls.initial_control2
	current_control_child = initial_controls.initial_control3

	step_statuses = Symbol[]
	step_kkt_errors = Float64[]
	step_solve_times = Float64[]
	step_total_iters = Int[]
	step_primary_statuses = Symbol[]
	step_primary_kkt_errors = Float64[]
	step_primary_iters = Int[]
	step_rescue_statuses = Symbol[]
	step_rescue_kkt_errors = Float64[]
	step_rescue_iters = Int[]
	step_attempt_iters = Int[]
	step_klu_singular_retries = Int[]
	step_svd_fallback_counts = Int[]
	θ1_initial = θ2_initial = θ3_initial = nothing

	# ── MPC loop ───────────────────────────────────────────────────────────────
	for k in 1:num_mpc_steps
		# Refresh the state/control boundary parameters without rebuilding the
		# KKT system.
		instance_states = (;
			initial_state1 = current_state1,
			initial_state2 = current_state2,
			initial_state3 = current_state_child,
			initial_control1 = current_control1,
			initial_control2 = current_control2,
			initial_control3 = current_control_child,
		)
		instance_parameters =
			@timeit TO "instance parameter construction" build_instance_parameters(
				flatten_parameters,
				instance_states,
				scenario_config,
			)
		(; θ1, θ2, θ3, θ) = instance_parameters
		if k == 1
			θ1_initial, θ2_initial, θ3_initial = θ1, θ2, θ3
		end

		# Plot the warm start used for this step's solve.
		# The interactive export of the previous step leaves WGLMakie active; static PDF saving needs the CairoMakie backend.
		if save_outputs
			CairoMakie.activate!()
			# A full-z warmstart carries duals and slacks; the visualization only
			# understands the primal block.
			stage_warmstart_primal =
				length(stage_warmstart) == GOOP_kkt_system.variable_dimension ?
				stage_warmstart[GOOP_kkt_system.primal_dims] : stage_warmstart
			@timeit TO "warmstart visualization" save_warmstart_visualizations(
				stage_warmstart_primal;
				filename_tag = "step_$(k)",
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
		primary_output = output
		rescue_status = :not_run
		rescue_kkt_error = NaN
		rescue_iters = 0
		rescue_klu_singular_retries = 0
		rescue_svd_fallback_count = 0
		if output.status == :failed && rescue_failed_steps
			# The shifted warmstart occasionally lands in a bad basin (the solver
			# stalls with η pinned at η_max, or at a merit-function local minimum
			# where ‖F‖ stays large). A fresh solve from the default warmstart is
			# cheap with the KLU path and usually recovers the step.
			println(
				"  step $(k): solve from shifted warmstart failed ",
				"(kkt_error=$(round(output.kkt_error; sigdigits = 3))); ",
				"retrying from the default warmstart.",
			)
			rescue_warmstart =
				build_default_warmstart(instance_states, scenario_config).warmstart_solution
			elapsed_time += @elapsed rescue_output =
				@timeit TO "solver invocation (rescue)" ReducedGOOP.solve(
					solver,
					GOOP_kkt_system,
					θ;
					z₀ = rescue_warmstart,
					options,
				)
			rescue_status = rescue_output.status
			rescue_kkt_error = rescue_output.kkt_error
			rescue_iters = rescue_output.total_iters
			rescue_klu_singular_retries = rescue_output.klu_singular_retries
			rescue_svd_fallback_count = rescue_output.svd_fallback_count
			if rescue_output.status == :solved ||
			   rescue_output.kkt_error < output.kkt_error
				output = rescue_output
			end
		end
		converged = output.status == :solved
		push!(step_statuses, output.status)
		push!(step_kkt_errors, output.kkt_error)
		push!(step_solve_times, elapsed_time)
		push!(step_total_iters, output.total_iters)
		push!(step_primary_statuses, primary_output.status)
		push!(step_primary_kkt_errors, primary_output.kkt_error)
		push!(step_primary_iters, primary_output.total_iters)
		push!(step_rescue_statuses, rescue_status)
		push!(step_rescue_kkt_errors, rescue_kkt_error)
		push!(step_rescue_iters, rescue_iters)
		push!(step_attempt_iters, primary_output.total_iters + rescue_iters)
		push!(
			step_klu_singular_retries,
			primary_output.klu_singular_retries + rescue_klu_singular_retries,
		)
		push!(
			step_svd_fallback_counts,
			primary_output.svd_fallback_count + rescue_svd_fallback_count,
		)
		attempt_summary =
			rescue_iters > 0 ?
			", primary_iters=$(primary_output.total_iters), rescue_iters=$(rescue_iters)" : ""
		println(
			"MPC step $(k)/$(num_mpc_steps): status=$(output.status), ",
			"iters=$(output.total_iters), ",
			"kkt_error=$(round(output.kkt_error; sigdigits = 4)), ",
			"time=$(round(elapsed_time; digits = 3)) sec",
			attempt_summary,
		)
		if !converged
			println(
				"  step $(k) did not converge within max_inner_iters=$(max_inner_iters); ",
				"continuing with the final iterate.",
			)
		end

		strategies = @timeit TO "solution postprocessing" extract_player_strategies(
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
		combined_arm_state = closed_loop_xs[1][end]
		current_state1 = combined_arm_state[1:arm_state_dimension]
		current_state2 =
			combined_arm_state[(arm_state_dimension+1):(2*arm_state_dimension)]
		current_state_child = closed_loop_xs[2][end]
		# The shifted warm start begins at the old second knot, so carry those
		# controls into θ as the next solve's fixed initial controls.
		combined_arm_control = strategies[1].us[2]
		current_control1 = combined_arm_control[1:arm_control_dimension]
		current_control2 =
			combined_arm_control[
				(arm_control_dimension+1):(2*arm_control_dimension)
			]
		current_control_child = strategies[2].us[2]

		if save_outputs
			@timeit TO "solution output and plotting" begin
				solution_dict = Dict(
					"mpc_step" => k,
					"dual_warmstart" => dual_warmstart,
					"strategies" => strategies,
					"z" => output.z,
					"x" => output.x,
					"warmstart" => stage_warmstart,
					"solve_time_sec" => elapsed_time,
					"kkt_error" => output.kkt_error,
					"ϵ" => output.ϵ,
					"status" => output.status,
					"total_iters" => output.total_iters,
					"primary_status" => primary_output.status,
					"primary_kkt_error" => primary_output.kkt_error,
					"primary_iters" => primary_output.total_iters,
					"rescue_status" => rescue_status,
					"rescue_kkt_error" => rescue_kkt_error,
					"rescue_iters" => rescue_iters,
					"attempt_iters" => primary_output.total_iters + rescue_iters,
					"klu_singular_retries" =>
						primary_output.klu_singular_retries + rescue_klu_singular_retries,
					"svd_fallback_count" =>
						primary_output.svd_fallback_count + rescue_svd_fallback_count,
					"kkt_error_history" => get(output, :kkt_error_history, Float64[]),
					"condition_number_history" =>
						get(output, :condition_number_history, Float64[]),
					"eta_history" => get(output, :eta_history, Float64[]),
					"alpha_history" => get(output, :alpha_history, Float64[]),
					"rho_history" => get(output, :rho_history, Float64[]),
					"arm_speed_limit" => arm_speed_limit,
					"child_speed_limit" => child_speed_limit,
				)
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
				save_convergence_diagnostics(
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
		end

		# Shifted previous solution warm-starts the next solve.
		shifted_strategies = @timeit TO "warmstart shift" shift_strategies(
			strategies,
			dynamics,
			planning_horizon,
		)
		shifted_primal = flatten_warmstart_solution(
			planning_horizon,
			[strategy.xs for strategy in shifted_strategies],
			[strategy.us for strategy in shifted_strategies],
		)
		stage_warmstart = build_receding_warmstart(
			shifted_primal,
			output.z,
			GOOP_kkt_system,
			dual_warmstart,
		)
	end

	# ── Closed-loop outputs ────────────────────────────────────────────────────
	failed_steps = findall(==(:failed), step_statuses)
	println(
		"\nMPC run finished: $(num_mpc_steps - length(failed_steps))/$(num_mpc_steps) steps converged",
		isempty(failed_steps) ? "." :
		"; non-converged steps: $(join(failed_steps, ", ")).",
	)

	closed_loop_strategies = map(1:num_players) do player
		# Pad with a terminal zero control so xs/us have equal length, matching
		# the trajectory layout the plotting helpers expect.
		(;
			xs = closed_loop_xs[player],
			us = vcat(
				closed_loop_us[player],
				[zeros(dynamics[player].control_dimension)],
			),
		)
	end

	if save_outputs
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
					"step_primary_statuses" => step_primary_statuses,
					"step_primary_kkt_errors" => step_primary_kkt_errors,
					"step_primary_iters" => step_primary_iters,
					"step_rescue_statuses" => step_rescue_statuses,
					"step_rescue_kkt_errors" => step_rescue_kkt_errors,
					"step_rescue_iters" => step_rescue_iters,
					"step_attempt_iters" => step_attempt_iters,
					"step_klu_singular_retries" => step_klu_singular_retries,
					"step_svd_fallback_counts" => step_svd_fallback_counts,
					"dual_warmstart" => dual_warmstart,
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
	end

	# ── Runtime report ─────────────────────────────────────────────────────────
	timing_stats = @timeit TO "runtime report" begin
		stats = print_runtime_report(
			problem_setup_time_sec,
			kkt_build_time_sec,
			step_solve_times,
			step_statuses,
		)
		if save_outputs
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
		end
		stats
	end

	if save_outputs
		@timeit TO "run metadata save" begin
			JLD2.save_object(
				joinpath(dirs.problem_data_dir, "run_metadata.jld2"),
				Dict(
					"run_id" => run_id,
					"debug" => debug,
					"run_dir" => dirs.run_dir,
					"rng_seed" => rng_seed,
					"dual_warmstart" => dual_warmstart,
					"selected_dual_count" => selected_dual_count,
					"total_equality_stationarity_dual_count" =>
						total_equality_stationarity_dual_count,
					"dynamics_model" => dynamics_model_name(dynamics_model),
					"planning_horizon" => planning_horizon,
					"num_mpc_steps" => num_mpc_steps,
					"tol" => tol,
					"klu_singularity_eta_growth" => klu_singularity_eta_growth,
					"ϵ₀" => ϵ₀,
					"max_inner_iters" => max_inner_iters,
					"fd_codegen_chunk_size" => fd_codegen_chunk_size,
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
					"step_total_iters" => step_total_iters,
					"step_primary_statuses" => step_primary_statuses,
					"step_primary_kkt_errors" => step_primary_kkt_errors,
					"step_primary_iters" => step_primary_iters,
					"step_rescue_statuses" => step_rescue_statuses,
					"step_rescue_kkt_errors" => step_rescue_kkt_errors,
					"step_rescue_iters" => step_rescue_iters,
					"step_attempt_iters" => step_attempt_iters,
					"step_klu_singular_retries" => step_klu_singular_retries,
					"step_svd_fallback_counts" => step_svd_fallback_counts,
					"problem_setup_time_sec" => problem_setup_time_sec,
					"kkt_build_time_sec" => kkt_build_time_sec,
					"total_solve_time_sec" => sum(step_solve_times),
				),
			)
		end
	end

	save_outputs && println("outputs saved under: ", dirs.run_dir)
	println("\nTiming summary:")
	show(TO)
	println()

	(;
		closed_loop_strategies,
		step_statuses,
		step_kkt_errors,
		step_solve_times,
		step_total_iters,
		step_primary_statuses,
		step_primary_kkt_errors,
		step_primary_iters,
		step_rescue_statuses,
		step_rescue_kkt_errors,
		step_rescue_iters,
		step_attempt_iters,
		step_klu_singular_retries,
		step_svd_fallback_counts,
		dual_warmstart,
		selected_dual_count,
		timing_stats,
		planning_horizon,
		Δt,
		kkt_dimension = GOOP_kkt_system.kkt_dimension,
		variable_dimension = GOOP_kkt_system.variable_dimension,
		problem_setup_time_sec,
		kkt_build_time_sec,
		run_dir = isnothing(dirs) ? nothing : dirs.run_dir,
	)
end

# ── MPC utilities ──────────────────────────────────────────────────────────────

"""
Assemble the next MPC warm start. Primals are shifted separately; selected
equality and stationarity duals retain their existing flat coordinates. All
other variables use the same defaults as `ReducedGOOP.solve` (zero, with
positive slacks/inequality duals).
"""
function build_receding_warmstart(
	shifted_primal,
	previous_z,
	kkt_system,
	mode::Symbol,
)
	mode in DUAL_WARMSTART_MODES ||
		throw(ArgumentError("Unknown dual_warmstart mode $(mode)."))
	length(shifted_primal) == length(kkt_system.primal_dims) || throw(
		DimensionMismatch(
			"Shifted primal has length $(length(shifted_primal)); " *
			"expected $(length(kkt_system.primal_dims)).",
		),
	)
	length(previous_z) == kkt_system.variable_dimension || throw(
		DimensionMismatch(
			"Previous KKT point has length $(length(previous_z)); " *
			"expected $(kkt_system.variable_dimension).",
		),
	)

	mode === :primal_only && return copy(shifted_primal)

	T = promote_type(eltype(shifted_primal), eltype(previous_z))
	warmstart = zeros(T, kkt_system.variable_dimension)
	warmstart[kkt_system.preference_slack_dims] .= one(T)
	warmstart[kkt_system.interior_point_slack_dims] .= one(T)
	warmstart[kkt_system.inequality_constraint_dual_dims] .= one(T)
	warmstart[kkt_system.primal_dims] .= shifted_primal
	if mode === :equality_duals
		dims = kkt_system.equality_constraint_dual_dims
		warmstart[dims] .= previous_z[dims]
	else
		# Carry all equality/stationarity duals except those tied to the innermost level.
		dims = kkt_system.all_equality_stationarity_dual_dims
		warmstart[dims] .= previous_z[dims]
		warmstart[kkt_system.innermost_stationarity_dual_dims] .= zero(T)
	end
	warmstart
end

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
	println(
		"  setup total:               $(round(problem_setup_time_sec + kkt_build_time_sec; digits = 2)) s",
	)
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
	hist_title =
		exclude_first ?
		"Warm-solve times (step 1 excluded: $(round(step_solve_times[1]; digits = 1)) s incl. one-time JIT)" :
		"Per-step solve times"
	hist_fig = serif_figure(size = (900, 600))
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
	save_figure(joinpath(timing_plots_dir, "solve_time_histogram.pdf"), hist_fig)

	bar_fig = serif_figure(size = (950, 600))
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
	save_figure(joinpath(timing_plots_dir, "solve_time_per_step.pdf"), bar_fig)
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
	plot_strategies = split_arm_strategies(strategies)
	status_suffix = converged ? "" : "_notconverged"
	title = _step_title(k, num_mpc_steps, converged, kkt_error)

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
	_add_title!(trajectory_fig, title)
	speed_fig, _ = speed_plot(;
		strategy = plot_strategies,
		speed_limit = arm_speed_limit,
		dynamics_model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
	)
	_add_title!(speed_fig, title)
	control_fig, _ = control_plot(;
		strategy = plot_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	_add_title!(control_fig, title)
	distance_fig, _ = inter_player_distance_plot(;
		strategy = plot_strategies,
		reference_distance = dₚ,
		safety_distance = collision_avoidance,
	)
	_add_title!(distance_fig, title)

	save_figure(
		joinpath(dirs.trajectory_plots_dir, "trajectory_step_$(k)$(status_suffix).pdf"),
		trajectory_fig,
	)
	save_figure(
		joinpath(dirs.speed_plots_dir, "speed_step_$(k)$(status_suffix).pdf"),
		speed_fig,
	)
	save_figure(
		joinpath(dirs.control_plots_dir, "control_step_$(k)$(status_suffix).pdf"),
		control_fig,
	)
	save_figure(
		joinpath(dirs.distance_plots_dir, "distance_step_$(k)$(status_suffix).pdf"),
		distance_fig,
	)

	if save_interactive
		interactive_path = joinpath(
			dirs.trajectory_plots_dir,
			"trajectory_interactive_step_$(k)$(status_suffix).html",
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
	plot_strategies = split_arm_strategies(closed_loop_strategies)
	title = "Closed-loop receding-horizon execution ($(num_mpc_steps) steps)"
	if !isempty(failed_steps)
		title *= " — non-converged steps: $(join(failed_steps, ", "))"
	end
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
	_add_title!(trajectory_fig, title)
	speed_fig, _ = speed_plot(;
		strategy = plot_strategies,
		speed_limit = arm_speed_limit,
		dynamics_model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
	)
	_add_title!(speed_fig, title)
	control_fig, _ = control_plot(;
		strategy = plot_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	_add_title!(control_fig, title)
	distance_fig, _ = inter_player_distance_plot(;
		strategy = plot_strategies,
		reference_distance = dₚ,
		safety_distance = collision_avoidance,
	)
	_add_title!(distance_fig, title)

	save_figure(
		joinpath(dirs.closed_loop_plots_dir, "trajectory_closed_loop.pdf"),
		trajectory_fig,
	)
	save_figure(
		joinpath(dirs.closed_loop_plots_dir, "speed_closed_loop.pdf"),
		speed_fig,
	)
	save_figure(
		joinpath(dirs.closed_loop_plots_dir, "control_closed_loop.pdf"),
		control_fig,
	)
	save_figure(
		joinpath(dirs.closed_loop_plots_dir, "distance_closed_loop.pdf"),
		distance_fig,
	)

	if save_interactive
		interactive_path = joinpath(
			dirs.closed_loop_plots_dir,
			"trajectory_interactive_closed_loop.html",
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
			save_path = interactive_path,
		)
		println(
			"saved closed-loop interactive trajectory browser file: ",
			interactive_path,
		)
	end
end

function prepare_receding_output_dirs(run_id; debug)
	run_dir = if debug
		joinpath("data", "robotic_arm_receding_horizon", "debug", run_id)
	else
		joinpath("data", "robotic_arm_receding_horizon", "runs", run_id)
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
	closed_loop_plots_dir = joinpath(plots_dir, "closed_loop")
	timing_plots_dir = joinpath(plots_dir, "timing")

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

# ── Python-facing MPC planner API ─────────────────────────────────────────────
#
# Usage (from Python):
#   ctx     = jl.Robotic_arm_receding.build_mpc_context(obs)
#   plan_fn = jl.Robotic_arm_receding.create_planner_from_context(ctx, obs, 20;
#                 planner_freq=10, low_level_freq=100)
#   action  = np.array(plan_fn(obs))   # call once per simulator step
#   # returns Float64[] after num_mpc_steps solves to signal done
#
# Open-loop gripper approach uses the same interface:
#   plan_fn = jl.create_open_loop_planner(jl.grip_trajectory(obs))
#   action  = np.array(plan_fn(obs))

struct MpcContext
	GOOP_kkt_system::Any
	scenario_config::Any
	flatten_parameters::Any
	primal_dimensions::Vector{Int}
	options::Any
	planning_horizon::Int
	arm_state_dimension::Int
	dual_warmstart::Symbol
end

"""Return goal positions for overlay visualization."""
function get_current_goal_positions(ctx::MpcContext, obs)::Vector{Vector{Float64}}
	(; goal_position1, goal_position2) = ctx.scenario_config
	eef0 = collect(Float64, obs["robot0_eef_pos"])
	eef1 = collect(Float64, obs["robot1_eef_pos"])
	[collect(Float64, goal_position1), collect(Float64, goal_position2), 0.5 .* (eef0 .+ eef1)]
end
export get_current_goal_positions

"""
Build the KKT system once from the post-grip EEF positions (world frame).

`obs` is the Python observation dict immediately after gripping.
"""
function build_mpc_context(
	obs;
	planning_horizon::Integer = 10,
	goop_version::Symbol = :quasi,
	dual_warmstart::Symbol = :all_except_innermost_stationarity,
	fd_codegen_chunk_size::Integer = 128,
	tol::Float64 = 0.008,
	max_inner_iters::Integer = 500,
	# Target lift height (metres above grip position). Must be reachable within
	# the planning horizon: sim_lift_height < planning_horizon * Δt * arm_speed_limit
	# (default: 10 × 0.1 × 0.5 = 0.50 m, so 0.25 m leaves a 2× margin).
	sim_lift_height::Float64 = 0.25,
	verbose::Bool = false,
)
	dual_warmstart in DUAL_WARMSTART_MODES || throw(
		ArgumentError(
			"Unknown dual_warmstart mode $(dual_warmstart); expected one of " *
			"$(join(DUAL_WARMSTART_MODES, ", ")).",
		),
	)

	sim_eef0 = collect(Float64, obs["robot0_eef_pos"])
	sim_eef1 = collect(Float64, obs["robot1_eef_pos"])
	sim_eef2 = collect(Float64, obs["robot2_eef_pos"])

	@info "Building MPC context (goop=$(goop_version), T=$(planning_horizon))..."
	scenario_config = demo_scenario_config(; planning_horizon, sim_eef0, sim_eef1, sim_eef2, sim_lift_height)
	(; dynamics) = scenario_config
	arm_state_dimension, _ = divrem(dynamics[1].state_dimension, 2)

	problem_setup_time = @elapsed (; problem, flatten_parameters) = get_setup(scenario_config)

	kkt_generators = Dict(
		:complete => ReducedGOOP.generate_slacked_complete_kkt_system,
		:reduced => ReducedGOOP.generate_slacked_reduced_kkt_system,
		:quasi => ReducedGOOP.generate_slacked_quasi_kkt_system,
	)
	GOOP_kkt_generator = get(kkt_generators, goop_version, nothing)
	isnothing(GOOP_kkt_generator) && error("Unknown GOOP version: $(goop_version)")

	kkt_build_time = @elapsed GOOP_kkt_system = GOOP_kkt_generator(
		problem;
		backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
		backend_options = (;),
		codegen = :fast_differentiation,
		fd_codegen_chunk_size,
	)
	println(
		"[build_mpc_context] problem $(round(problem_setup_time; digits = 2)) s, ",
		"KKT build $(round(kkt_build_time; digits = 2)) s",
	)

	primal_dimensions = [
		(dyn.state_dimension + dyn.control_dimension) * planning_horizon
		for dyn in dynamics
	]

	options = ReducedGOOP.InteriorPointOptions(;
		tol,
		η₀ = 1e-6,
		η_max = 1e2,
		ϵ₀ = 0.1,
		max_inner_iters,
		max_outer_iters = 1,
		tightening_rate = 1.2,
		loosening_rate = 3.0,
		min_stepsize = 1e-20,
		linesearch = :backtracking,
		linear_solver = :svd,
		armijo_constant = 1e-4,
		eta_retry_growth = 2.0,
		ρ_low = 0.75,
		ρ_high = 0.75,
		tsvd_threshold = 0.0,
		use_marquardt_scaling = false,
		verbose,
	)

	# ── Scenario diagnostics ───────────────────────────────────────────────────
	(; base_initial_state1, base_initial_state2, goal_position1, goal_position2,
	   initial_state3, goal_position3, arm_speed_limit, child_speed_limit,
	   Δt, collision_avoidance, dₚ, use_world_frame) = scenario_config
	total_plan_time = planning_horizon * Δt
	max_reach = arm_speed_limit * total_plan_time
	dist1 = sqrt(sum(abs2, goal_position1 .- base_initial_state1))
	dist2 = sqrt(sum(abs2, goal_position2 .- base_initial_state2))
	println("\n── MPC scenario configuration ──────────────────────────────────────")
	println("  frame:        ", use_world_frame ? "world (metres)" : "abstract")
	println("  horizon:      T=$(planning_horizon) × Δt=$(Δt) s = $(total_plan_time) s total")
	println("  arm speed:    ≤ $(arm_speed_limit) m/s  →  max reach $(round(max_reach; digits = 3)) m per horizon")
	println("  child speed:  ≤ $(child_speed_limit) m/s")
	println("  safety dist:  $(collision_avoidance) m  (grip separation dₚ=$(round(dₚ; digits = 4)) m)")
	println("  arm 1  pos:   $(round.(base_initial_state1; digits = 4))")
	println("         goal:  $(round.(goal_position1; digits = 4))  dist=$(round(dist1; digits = 4)) m")
	println("  arm 2  pos:   $(round.(base_initial_state2; digits = 4))")
	println("         goal:  $(round.(goal_position2; digits = 4))  dist=$(round(dist2; digits = 4)) m")
	println("  child  pos:   $(round.(initial_state3; digits = 4))")
	println("         goal:  $(round.(goal_position3; digits = 4))")
	dist1 > max_reach && @warn "arm 1 goal $(round(dist1; digits=3)) m exceeds horizon reach $(round(max_reach; digits=3)) m"
	dist2 > max_reach && @warn "arm 2 goal $(round(dist2; digits=3)) m exceeds horizon reach $(round(max_reach; digits=3)) m"
	println("─────────────────────────────────────────────────────────────────────")

	MpcContext(
		GOOP_kkt_system,
		scenario_config,
		flatten_parameters,
		primal_dimensions,
		options,
		Int(planning_horizon),
		Int(arm_state_dimension),
		dual_warmstart,
	)
end

"""
Return a stateful `plan_fn(obs) -> Vector{Float64}` closure.

Each call drains one pre-computed low-level waypoint from an internal buffer.
The buffer is refilled by one MPC solve every `ratio = low_level_freq ÷
planner_freq` calls, producing `ratio` linearly interpolated waypoints between
consecutive planned positions for smooth OSC tracking.
Returns `Float64[]` after `num_mpc_steps` solves to signal completion.
"""
function create_planner_from_context(
	ctx::MpcContext,
	obs,
	num_mpc_steps::Integer = 20;
	planner_freq::Integer = 10,
	low_level_freq::Integer = 10,
)
	low_level_freq >= planner_freq ||
		error("low_level_freq ($(low_level_freq)) must be ≥ planner_freq ($(planner_freq))")
	ratio = div(low_level_freq, planner_freq)
	low_level_freq == planner_freq * ratio ||
		@warn "low_level_freq $(low_level_freq) is not an exact multiple of planner_freq $(planner_freq); using ratio=$(ratio)"

	(;
		GOOP_kkt_system,
		scenario_config,
		flatten_parameters,
		primal_dimensions,
		options,
		planning_horizon,
		arm_state_dimension,
		dual_warmstart,
	) = ctx

	sim_eef0 = collect(Float64, obs["robot0_eef_pos"])
	sim_eef1 = collect(Float64, obs["robot1_eef_pos"])
	sim_eef2 = collect(Float64, obs["robot2_eef_pos"])

	initial_instance_states = (;
		initial_state1 = sim_eef0,
		initial_state2 = sim_eef1,
		initial_state3 = sim_eef2,
	)
	(; warmstart_solution) = build_default_warmstart(initial_instance_states, scenario_config)
	stage_warmstart = Ref(warmstart_solution)

	solver = ReducedGOOP.InteriorPoint()
	dynamics = scenario_config.dynamics

	arm_state1 = Ref(copy(sim_eef0))
	arm_state2 = Ref(copy(sim_eef1))
	# Init boundary controls from the warm start so θ and z₀ agree at the first solve.
	initial_controls = extract_initial_controls(warmstart_solution, primal_dimensions, dynamics)
	ctrl1 = Ref(copy(initial_controls.initial_control1))
	ctrl2 = Ref(copy(initial_controls.initial_control2))
	ctrl3 = Ref(copy(initial_controls.initial_control3))
	# TODO: switch child to obs["robot2_eef_pos"] once GR1 tracking is reliable.
	child_state = Ref(copy(sim_eef2))

	buffer = Vector{Float64}[]
	mpc_step = Ref(0)

	function plan_next_fn(obs)
		if isempty(buffer)
			mpc_step[] >= num_mpc_steps && return Float64[]

			current_states = (;
				initial_state1 = arm_state1[],
				initial_state2 = arm_state2[],
				initial_state3 = child_state[],
				initial_control1 = ctrl1[],
				initial_control2 = ctrl2[],
				initial_control3 = ctrl3[],
			)
			(; θ) = build_instance_parameters(flatten_parameters, current_states, scenario_config)

			elapsed_time = @elapsed output = ReducedGOOP.solve(solver, GOOP_kkt_system, θ; z₀ = stage_warmstart[], options)
			if output.status == :failed
				rescue_ws = build_default_warmstart(initial_instance_states, scenario_config).warmstart_solution
				elapsed_time += @elapsed rescue = ReducedGOOP.solve(solver, GOOP_kkt_system, θ; z₀ = rescue_ws, options)
				if rescue.status == :solved || rescue.kkt_error < output.kkt_error
					output = rescue
				end
			end

			strategies = extract_player_strategies(output.x, primal_dimensions, dynamics)

			prev_arm1 = copy(arm_state1[])
			prev_arm2 = copy(arm_state2[])
			prev_child = copy(child_state[])

			combined_next = collect(Float64, strategies[1].xs[2])
			arm_state1[] = combined_next[1:arm_state_dimension]
			arm_state2[] = combined_next[(arm_state_dimension + 1):(2 * arm_state_dimension)]

			combined_ctrl = collect(Float64, strategies[1].us[2])
			ctrl1[] = combined_ctrl[1:arm_state_dimension]
			ctrl2[] = combined_ctrl[(arm_state_dimension + 1):(2 * arm_state_dimension)]
			ctrl3[] = collect(Float64, strategies[2].us[2])
			child_state[] = collect(Float64, strategies[2].xs[2])

			mpc_step[] += 1

			(; goal_position1, goal_position2) = scenario_config
			dist1 = sqrt(sum(abs2, arm_state1[] .- goal_position1))
			dist2 = sqrt(sum(abs2, arm_state2[] .- goal_position2))
			println(
				"[mpc] step=$(mpc_step[])/$(num_mpc_steps), status=$(output.status), ",
				"iters=$(output.total_iters), ",
				"kkt_err=$(round(output.kkt_error; sigdigits = 4)), ",
				"time=$(round(elapsed_time; digits = 3)) s | ",
				"arm1→goal=$(round(dist1; digits = 4)) m  ",
				"arm2→goal=$(round(dist2; digits = 4)) m",
			)

			shifted = shift_strategies(strategies, dynamics, planning_horizon)
			shifted_primal = flatten_warmstart_solution(
				planning_horizon,
				[s.xs for s in shifted],
				[s.us for s in shifted],
			)
			stage_warmstart[] = build_receding_warmstart(
				shifted_primal,
				output.z,
				GOOP_kkt_system,
				dual_warmstart,
			)

			for i in 1:ratio
				α = i / ratio
				push!(buffer, vcat(
					prev_arm1 .+ α .* (arm_state1[] .- prev_arm1), [1.0],
					prev_arm2 .+ α .* (arm_state2[] .- prev_arm2), [1.0],
					prev_child .+ α .* (child_state[] .- prev_child),
				))
			end
		end
		popfirst!(buffer)
	end

	plan_next_fn
end

end
