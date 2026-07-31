#=
	Robotic_arm_mpc.jl — lightweight receding-horizon (MPC) robotic-arm planner.

	=== What this file does ===============================================

	Builds a two-player Game of Ordered Preference and runs it in a receding
	horizon. Player 1 is the *combined two-arm agent* (both arms stacked into
	one 6D single integrator); player 2 is the child/pet. Each player optimizes
	a hierarchy of preferences rather than a single scalar cost.

	The expensive part — differentiating the game and compiling its KKT system —
	happens exactly once, for a fixed `planning_horizon`. Every MPC step then
	only updates the parameter vector θ (initial states and initial controls)
	and the warm start before re-solving, so all compiled residuals, the sparse
	Jacobian, and the solver options are reused across steps.

	Per step: solve the open-loop game from the current state, apply the first
	control, shift the solution one stage forward to warm-start the next solve.
	A non-converged step is not fatal — the final iterate is used and the step
	is reported as `:failed`.

	This is the stripped-down sibling of `Robotic_arm_receding.jl`: same
	algorithm and same numerics, but it writes no files, draws no plots, and
	loads no plotting stack.

	=== >>> INTEGRATION POINT FOR RoboSuite <<< ===========================

	The planner is closed on its own prediction. In `demo`'s MPC loop:

	    push!(closed_loop_xs[player], collect(strategies[player].xs[2]))
	    ...
	    current_state1      = combined_arm_state[1:arm_state_dimension]
	    current_state2      = combined_arm_state[(arm_state_dimension+1):...]
	    current_state_child = closed_loop_xs[2][end]

	`strategies[player].xs[2]` is the *predicted* next state under the game's
	own dynamics. To drive this from RoboSuite, replace those assignments with
	the *measured* state read back from the simulator after applying
	`strategies[player].us[1]`:

	    apply_to_simulator(strategies[1].us[1], strategies[2].us[1])
	    current_state1, current_state2, current_state_child = read_simulator()

	`instance_states` at the top of the loop is the ONLY input surface — nothing
	else in the loop reads the world state. Set `current_state1/2/child` (and,
	if the simulator reports them, `current_control1/2/child`) from measurement
	and everything downstream follows.

	Note that `current_control*` is taken from `us[2]`, not `us[1]`: the shifted
	warm start begins at the old second knot, so the initial-control boundary
	parameter must match it. If you substitute measured states, keep this
	consistent or the warm start and θ will disagree.

	=== Data layout =======================================================

	  closed_loop_xs[1][k] :: Vector{Float64}(6)  # [arm1_xyz; arm2_xyz] at step k
	  closed_loop_us[1][k] :: Vector{Float64}(6)  # [arm1_u;   arm2_u  ]
	  closed_loop_xs[2][k] :: Vector{Float64}(3)  # child xyz
	  closed_loop_us[2][k] :: Vector{Float64}(3)  # child u

	Player 1 is deliberately a single stacked agent, not two players — the two
	arms share one preference hierarchy. Use `split_arm_strategies(strategies)`
	from the core to get a per-arm 3D view ([arm1, arm2, child]).

	=== Functions here ====================================================

	  demo                      entry point; builds once, then runs the MPC loop
	  build_receding_warmstart  assembles the next warm start (primals + duals)
	  shift_strategies          shifts a trajectory one stage forward
	  print_runtime_report      separates one-time setup cost from per-step cost
	  _solve_time_stats,
	  _format_stats             small helpers for the runtime report

	=== Dependencies ======================================================

	  robotic_arm_core.jl   scenario config, problem construction (`get_setup`),
	                        warm starts, trajectory packing/unpacking
	    └── dynamics.jl     single-integrator residual and step
	  ReducedGOOP           generate_slacked_{complete,reduced,quasi}_kkt_system,
	                        InteriorPoint, InteriorPointOptions, solve

	Deliberately NOT loaded: CairoMakie, WGLMakie, GLMakie, JLD2. This file does
	not include `Robotic_arm.jl` (that one carries the plotting layer).

	=== Knobs that matter =================================================

	  goop_version     :quasi (default) | :reduced | :complete — the KKT
	                   reformulation. :quasi is the smallest/fastest.
	  dual_warmstart   :all_except_innermost_stationarity (default) | :equality_duals | primal_only
	  tol              solver convergence tolerance on ‖F‖₂
	  max_inner_iters  Newton iteration budget per solve

	Cost: the one-time KKT build is minutes (dominated by symbolic
	differentiation and code generation); each MPC solve is seconds. Budget a
	simulator tick against the *per-step* number reported at the end, not the
	first step, which additionally pays JIT compilation of the solver.

	=== Usage =============================================================

	    julia --project=experiments
	    include("experiments/Robotic_arm_mpc.jl")
	    result = Robotic_arm_mpc.demo(num_mpc_steps = 20)
	    result.closed_loop_xs      # trajectories
	    result.step_statuses       # per-step :solved / :failed
=#

module Robotic_arm_mpc

# Forward BLAS/LAPACK to Apple Accelerate (AMX): the solver's per-iteration
# cost is dominated by dense linear algebra, which runs faster under Accelerate
# than OpenBLAS on Apple silicon. Process-global via libblastrampoline.
@static if Sys.isapple()
	using AppleAccelerate: AppleAccelerate
end

using Random
using ReducedGOOP
using Statistics: mean, median, std
using TimerOutputs: @timeit, reset_timer!

# Reuse a Revise-tracked core when one is already loaded in Main. The fallback
# keeps this solver-only entry point loadable on its own.
const ROBOTIC_ARM_CORE_PATH = joinpath(@__DIR__, "robotic_arm_core.jl")
if !isdefined(Main, :RoboticArmCore)
	Base.include(Main, ROBOTIC_ARM_CORE_PATH)
end
Main.RoboticArmCore isa Module ||
	error("Main.RoboticArmCore exists but is not a module.")
using Main.RoboticArmCore

# Preserve the solver-only entry point's existing qualified core API.
for core_symbol in names(Main.RoboticArmCore)
	core_symbol === :RoboticArmCore && continue
	@eval export $core_symbol
end

const DUAL_WARMSTART_MODES = (
	:primal_only,
	:equality_duals,
	:all_except_innermost_stationarity,
)

# ── Experiment entry point ─────────────────────────────────────────────────────

"""
Run the receding-horizon robotic-arm game and return the closed-loop trajectories.

The symbolic KKT system is built exactly once for a fixed `planning_horizon`;
every MPC iteration only updates θ (initial states and controls) and the warm
start before re-solving. At each step the open-loop game is solved from the
current state, the first control is applied, and the previous solution is
shifted one stage to warm-start the next solve. Non-converged solves are not
fatal: the final iterate is used and the step is reported as `:failed`.

To drive this from a simulator, replace the state advance inside the loop (see
the "INTEGRATION POINT" section at the top of this file) with the measured
state; `instance_states` is the only place the loop reads the world.

Keyword arguments:
- `num_mpc_steps`: number of closed-loop steps to execute.
- `goop_version`: `:complete`, `:reduced`, or `:quasi` KKT reformulation.
- `dual_warmstart`: which duals to carry between steps; see `DUAL_WARMSTART_MODES`.
- `tol`, `max_inner_iters`: solver convergence tolerance and iteration budget.
- `rescue_failed_steps`: re-solve a failed step once from the default warm start.

Returns a named tuple of `closed_loop_strategies`, `closed_loop_xs`,
`closed_loop_us`, the per-step diagnostics, and `timing_stats`. The section
timings are printed at the end of the run; they live in ReducedGOOP's global
timer `TO`, which the next `demo` call resets.
"""
function demo(;
	num_mpc_steps = 20,
	verbose = false,
	rng_seed = 123,
	goop_version = :quasi,     # :complete | :reduced | :quasi
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
	max_inner_iters = 500,
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
	kkt_backend = :symbolics
	kkt_backend_options = (;)
	# Symbolics differentiates, FastDifferentiation emits the code: the FD
	# tracing backend is unusable for these hierarchies, and pure Symbolics
	# codegen is pathologically slow. This pairing is deliberate.
	kkt_codegen = :fast_differentiation
	solver = ReducedGOOP.InteriorPoint()
	linesearch = :backtracking
	ϵ₀ = 0.1 # placeholder in the robotic arm scenario (no inequality constraints here)
	use_running_goal_cost = false

	# ── Scenario and problem ───────────────────────────────────────────────────
	# Player 1: combined two-arm agent, Player 2: child/pet.
	scenario_config = demo_scenario_config(; use_running_goal_cost)
	(;
		dynamics,
		num_players,
		planning_horizon,
		base_initial_state1,
		base_initial_state2,
		initial_state3,
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

	# `record_convergence` and `record_condition_number` stay off: the former
	# only accumulates diagnostic histories, the latter costs a dense SVD per
	# Newton iteration. Neither affects the iterate path.
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
		record_convergence = false,
		record_condition_number = false,
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

	# ── MPC loop ───────────────────────────────────────────────────────────────
	for k in 1:num_mpc_steps
		# Refresh the state/control boundary parameters without rebuilding the
		# KKT system. >>> RoboSuite: this is the only input surface. <<<
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
		(; θ) = instance_parameters

		elapsed_time = @elapsed output = @timeit TO "solver invocation" ReducedGOOP.solve(
			solver,
			GOOP_kkt_system,
			θ;
			z₀ = stage_warmstart,
			options,
		)
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

		strategies = @timeit TO "solution postprocessing" extract_player_strategies(
			output.x,
			primal_dimensions,
			dynamics,
		)

		# Apply the first control: advance every player to its first predicted
		# state (equivalent to applying us[1] through the dynamics constraints).
		#
		# >>> RoboSuite substitution point <<<
		# `strategies[player].xs[2]` is the game's own prediction. Replace the
		# assignments below with the state measured after applying
		# `strategies[player].us[1]` in the simulator.
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

	# ── Runtime report ─────────────────────────────────────────────────────────
	timing_stats = @timeit TO "runtime report" print_runtime_report(
		problem_setup_time_sec,
		kkt_build_time_sec,
		step_solve_times,
		step_statuses,
	)

	println("\nTiming summary:")
	show(TO)
	println()

	(;
		closed_loop_strategies,
		closed_loop_xs,
		closed_loop_us,
		step_statuses,
		step_kkt_errors,
		step_solve_times,
		step_total_iters,
		dual_warmstart,
		selected_dual_count,
		timing_stats,
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

# ── Python-facing MPC planner API ─────────────────────────────────────────────
#
# Usage (from Python / test.py):
#   ctx     = jl.Robotic_arm_mpc.build_mpc_context(obs, r2_dim)
#   planner = jl.Robotic_arm_mpc.create_planner_from_context(ctx, obs, 20)
#   # use jl.Robotic_arm_mpc.plan_next(planner, obs) to get the next action vector
#
# At every plan_next call the planner reads the actual robot EEF positions from
# the live obs dict, re-solves from that state, and returns the first planned
# control as a RoboSuite action vector — closing the loop on real measurements.

# ── Closed-loop planner ────────────────────────────────────────────────────────
# solve_fn(obs) → Vector{Float64}; return Float64[] to signal completion.

mutable struct ClosedLoopPlanner
	solve_fn::Any
	step::Int
	max_steps::Int
end
export ClosedLoopPlanner

function plan_next(planner::ClosedLoopPlanner, obs)::Vector{Float64}
	planner.step >= planner.max_steps && return Float64[]
	action = planner.solve_fn(obs)
	planner.step += 1
	return action
end
export plan_next

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

"""Return [[r0_goal], [r1_goal], [r2_goal]] (world-frame) from the scenario config."""
function get_goal_positions(ctx::MpcContext)
	(; goal_position1, goal_position2, goal_position3) = ctx.scenario_config
	[collect(Float64, goal_position1), collect(Float64, goal_position2), collect(Float64, goal_position3)]
end
export get_goal_positions

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
Create a `ClosedLoopPlanner` from a pre-built `MpcContext`.

Each `plan_next` call reads the actual robot EEF positions from the live obs,
re-solves from that state (closing the loop on real measurements), shifts the
warm start, and returns the first planned position as a RoboSuite action.
GR1 (robot 2) is frozen at its grip position to avoid controller instability.
`num_mpc_steps` caps the number of `plan_next` calls before signalling done.
"""
function create_planner_from_context(
	ctx::MpcContext,
	obs,
	num_mpc_steps::Integer = 20,
)
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
	# Track boundary controls so θ stays consistent with the shifted warm start.
	# us[2] from each solve becomes the fixed initial control for the next θ.
	ctrl1 = Ref(zeros(arm_state_dimension))
	ctrl2 = Ref(zeros(arm_state_dimension))
	ctrl3 = Ref(zeros(dynamics[2].control_dimension))
	# GR1's controller is unreliable: feed the solver its own predicted child
	# state rather than the real EEF measurement to keep θ self-consistent.
	child_state = Ref(copy(sim_eef2))

	function solve_fn(obs)
		current_eef0 = collect(Float64, obs["robot0_eef_pos"])
		current_eef1 = collect(Float64, obs["robot1_eef_pos"])

		current_states = (;
			initial_state1 = current_eef0,
			initial_state2 = current_eef1,
			initial_state3 = child_state[],
			initial_control1 = ctrl1[],
			initial_control2 = ctrl2[],
			initial_control3 = ctrl3[],
		)
		(; θ) = build_instance_parameters(flatten_parameters, current_states, scenario_config)

		output = ReducedGOOP.solve(solver, GOOP_kkt_system, θ; z₀ = stage_warmstart[], options)
		if output.status == :failed
			rescue_ws = build_default_warmstart(initial_instance_states, scenario_config).warmstart_solution
			rescue = ReducedGOOP.solve(solver, GOOP_kkt_system, θ; z₀ = rescue_ws, options)
			if rescue.status == :solved || rescue.kkt_error < output.kkt_error
				output = rescue
			end
		end
		println(
			"[mpc] status=$(output.status), ",
			"iters=$(output.total_iters), ",
			"kkt_err=$(round(output.kkt_error; sigdigits = 4))",
		)

		strategies = extract_player_strategies(output.x, primal_dimensions, dynamics)

		# Update boundary controls and ideal child state for the next θ.
		combined_ctrl = collect(Float64, strategies[1].us[2])
		ctrl1[] = combined_ctrl[1:arm_state_dimension]
		ctrl2[] = combined_ctrl[(arm_state_dimension + 1):(2 * arm_state_dimension)]
		ctrl3[] = collect(Float64, strategies[2].us[2])
		child_state[] = collect(Float64, strategies[2].xs[2])

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

		combined_next = collect(Float64, strategies[1].xs[2])
		r0_pos = combined_next[1:arm_state_dimension]
		r1_pos = combined_next[(arm_state_dimension + 1):(2 * arm_state_dimension)]
		r2_pos = collect(Float64, strategies[2].xs[2])
		vcat(r0_pos, [1.0], r1_pos, [1.0], r2_pos)  # 1.0 = GRIP_CLOSED
	end

	ClosedLoopPlanner(solve_fn, 0, Int(num_mpc_steps))
end

end
