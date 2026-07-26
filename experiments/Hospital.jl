module Hospital

#=
	Urgency-aware hospital corridor navigation, following:
	"Urgency-Aware Navigation for Emergency-Service Robotics via Ordered-Preference Games"
	(Sections 3-10). This is a first feasibility pass at the paper's 3-agent toy example
	(Sec. 10); the 4-agent extension (Sec. 11) reuses the same machinery.

	Deliberate simplifications vs. the paper (flagged so they can be revisited):
	- Urgency-interference avoidance is currently a *hard equality constraint*
	  (`urgency_interference_equality_violation`), not a lexicographic preference
	  level at all — a deliberate experiment after the preference-level vector
	  form proved hard to fully converge. This is a bigger deviation from the
	  paper than the other items below: interference is no longer prioritized
	  above/below anything, it's simply always enforced like safety/`v_max`.
	- Priority order (for the remaining two preference levels) is corridor ≻ own
	  target (swapped vs. the paper's eq. 17 order of target ≻ corridor) —
	  corridor-following is intentionally prioritized above goal-reaching here.
	- Control set ‖u‖ ≤ v_max (paper eq. 4) is enforced as a hard equality-embedded
	  squared-violation term (same mechanism used for collision avoidance, not the
	  disabled `inequality_constraints` GOOP path) — exact at convergence, but only
	  approximately respected at unconverged intermediate iterates.
	- Nominal corridor region C_i is not specified numerically in the paper for this
	  scenario; here it defaults to the straight (infinite) line through the agent's
	  start and goal, and J_cor is the summed squared distance to that line (not
	  clamped to the start↔goal segment — see `corridor_objective`).
	- The urgency clearance D and safety radii r_i are not given numeric values in the
	  paper's toy example; reasonable defaults are chosen in `demo` and will likely need
	  tuning (this codebase's other experiments are known to be parameter-sensitive).
	- Eq. 12's aggregation rule across same-urgency-level peers is ambiguous: the text
	  says "summation... equal weight" but the equation writes `min`, and the worked
	  4-agent example (eq. 28) writes an explicit `min{H4→2, H4→3}`. This only matters
	  once a player has >1 same-level neighbor, which does not happen in the 3-agent
	  case (each of agents 2, 3 has exactly one higher-urgency neighbor: agent 1). The
	  code below defaults to summing each neighbor's margin independently (via the
	  existing per-element smooth-threshold framework) and emits a warning if a player
	  ever has more than one same-level neighbor, so this must be revisited before
	  trusting the 4-agent case.
=#

using CairoMakie: CairoMakie
using LaTeXStrings: @L_str
using BlockArrays
using JLD2, Dates, Random
using LinearAlgebra: norm
using ReducedGOOP
using TimerOutputs: @timeit, reset_timer!

const TO = ReducedGOOP.TO

include(joinpath(@__DIR__, "dynamics.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

# ── Problem definition ─────────────────────────────────────────────────────────

function get_setup(
	num_players;
	urgency_levels,
	Δt = 0.3,
	planning_horizon = 10,
	v_max = 3.0,
	safety_radius = fill(0.3, num_players),
	urgency_clearance = 1.0,
)
	length(urgency_levels) == num_players ||
		error("urgency_levels must have one entry per player.")

	state_dimension = 2   # p = [px, py]
	control_dimension = 2 # u = [vx, vy], directly controlled velocity (single integrator)
	dynamics_dimension = state_dimension + control_dimension

	primal_dimensions = fill(dynamics_dimension * planning_horizon, num_players)
	parameter_dimensions = fill(state_dimension + state_dimension, num_players) # (initial_state, goal_position)

	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
	dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

	unflatten_parameters = function (θ)
		θ_iter = Iterators.Stateful(θ)
		initial_state = first(θ_iter, state_dimension)
		goal_position = first(θ_iter, state_dimension)
		(; initial_state, goal_position)
	end

	function flatten_parameters(; initial_state, goal_position)
		vcat(initial_state, goal_position)
	end

	function squared_violation(h)
		"h(x) ≥ 0	<=> (min(h(x), 0))^2 = 0"
		return (min(h, 0))^2
	end

	function smooth_piecewise_preference_objective(preference, level; ϵ = 0.0)
		ifelse(preference ≥ ϵ, 0.0, (ϵ - preference)^(level + 2))
	end

	trajectory(z; player) =
		unflatten_trajectory(z[Block(player)], state_dimension, control_dimension)

	function goal_objective(; player)
		"J_tar (paper eq. 14): squared distance between the terminal position and the goal."
		function (z, θ)
			(; xs) = trajectory(z; player)
			(; goal_position) = unflatten_parameters(θ[Block(player)])
			sum((xs[end] .- goal_position) .^ 2)
		end
	end

	function corridor_objective(; player)
		"""
		J_cor (paper eq. 15): summed squared distance to the start→goal corridor
		(default nominal corridor). Simplified to the distance to the *infinite* line
		through start/goal (a closed-form algebraic expression) rather than the
		clamped *segment* projection, since the clamp/projection was a contributor
		to an earlier symbolic-compile blowup in the urgency-margin term. Revisit
		once the baseline solves.
		"""
		function (z, θ)
			(; xs) = trajectory(z; player)
			(; initial_state, goal_position) = unflatten_parameters(θ[Block(player)])
			segment = goal_position .- initial_state
			segment_length_sq = sum(segment .^ 2) + 1e-9
			sum(xs) do x
				offset = x .- initial_state
				cross = segment[1] * offset[2] - segment[2] * offset[1]
				cross^2 / segment_length_sq
			end
		end
	end

	function urgency_interference_equality_violation(z, agent, neighbors)
		"""
		Hard (equality-embedded) version of the urgency-interference margin: per
		neighbor, sum `squared_violation(margin_t)` (0 when clear, positive when
		violated) across every lookahead point into one "total violation" scalar,
		then constrain that scalar to equal 0 — i.e. no violation at *any* point,
		enforced the same way as `control_bound_violation`/`shared_pairwise_
		collision_avoidance` rather than as a lexicographic preference level. This
		removes urgency-avoidance from the preference hierarchy entirely (it
		becomes as hard as safety/`v_max`), which is a deliberate experiment, not
		the paper's design (there, interference is the *highest-priority
		preference*, not a hard constraint).

		This is the confirmed-working version: kkt_error ~0.19 (vs. stuck at
		23-30 for the preference-level vector form), reproduced exactly on
		re-test. A "worst-point" variant (`minimum` over points, then
		`squared_violation` just that one value) was tried and abandoned — like
		the earlier CPA clamp, the `minimum` reintroduced catastrophic symbolic
		differentiation cost (killed after 84 minutes with no result). This sum
		form — `squared_violation` applied per-point *before* summing, no min/max
		anywhere — is the one to keep.
		"""
		clearance_sq = urgency_clearance^2
		trajectory_i = trajectory(z; player = agent)
		horizon_len = length(trajectory_i.xs)
		map(neighbors) do j
			trajectory_j = trajectory(z; player = j)
			sum(1:horizon_len) do t
				q = trajectory_i.xs[t] .- trajectory_j.xs[t]
				w = trajectory_i.us[t] .- trajectory_j.us[t]
				q_sq = sum(q .^ 2)
				qw = sum(q .* w)
				w_sq = sum(w .^ 2)
				min_separation_sq = q_sq - qw^2 / (w_sq + 1e-6)
				squared_violation(min_separation_sq - clearance_sq)
			end
		end
	end

	function control_bound_violation(z, i)
		"‖u‖ ≤ v_max (paper eq. 4), as a single magnitude constraint per step rather than an axis-aligned box."
		v_max_sq = v_max^2
		(; us) = trajectory(z; player = i)
		map(us) do u
			squared_violation(v_max_sq - sum(u .^ 2))
		end
	end

	function shared_pairwise_collision_avoidance(z, _)
		"Hard safety constraint (paper eq. 6), all N(N-1)/2 pairs."
		trajectories = map(i -> trajectory(z; player = i), 1:num_players)
		xs = map(t -> t.xs, trajectories)
		horizon_len = length(xs[1])
		mapreduce(vcat, ((i, j) for i in 1:num_players for j in (i+1):num_players)) do (i, j)
			min_clearance_sq = (safety_radius[i] + safety_radius[j])^2
			map(1:horizon_len) do t
				sum((xs[i][t] - xs[j][t]) .^ 2) - min_clearance_sq
			end
		end
	end

	# Same/higher-urgency neighbors per player (needed both here, to embed urgency
	# as a hard equality constraint, and below, to omit it from the preferences).
	neighbors_by_player = map(1:num_players) do i
		higher_levels = sort(unique(filter(l -> l > urgency_levels[i], urgency_levels)))
		mapreduce(vcat, higher_levels; init = Int[]) do level
			neighbors = [j for j in 1:num_players if j != i && urgency_levels[j] == level]
			if length(neighbors) > 1
				@warn "Player $i has $(length(neighbors)) same-urgency-level ($level) neighbors; see the ambiguity noted at the top of Hospital.jl regarding eq. 12's aggregation rule."
			end
			neighbors
		end
	end

	player_equality_constraints = [
		function (z, θ)
			(; xs) = trajectory(z; player = i)
			(; initial_state) = unflatten_parameters(θ[Block(i)])

			initial_state_constraint = xs[1] - initial_state
			dynamics_constraints = mapreduce(vcat, 1:(length(xs)-1)) do t
				single_integrator_2d(z[Block(i)], t; Δt, state_dimension, control_dimension)
			end

			vcat(
				initial_state_constraint,
				dynamics_constraints,
				control_bound_violation(z, i),
				urgency_interference_equality_violation(z, i, neighbors_by_player[i]),
			)
		end for i in 1:num_players
	]

	shared_equality_constraint = function (z, θ)
		squared_violation.(shared_pairwise_collision_avoidance(z, θ))
	end

	equality_constraints = [
		(z, θ) -> vcat(player_equality_constraints[i](z, θ), shared_equality_constraint(z, θ))
		for i in 1:num_players
	]

	# Preference ordering convention (see CLAUDE.md): [lowest priority, ..., highest priority].
	# Paper priority order (eq. 17): higher-urgency interference ≻ own target ≻ corridor.
	# NOTE: urgency-interference is currently embedded as a *hard equality
	# constraint* (see `urgency_interference_equality_violation` above), not a
	# preference level — a deliberate experiment, so it does not appear here.
	# Deliberately swapped vs. the paper here too: corridor ≻ own target.
	preferences = [
		[goal_objective(; player = i), corridor_objective(; player = i)]
		for i in 1:num_players
	]
	is_prioritized_constraint = [[false, false] for _ in 1:num_players]

	function build_goop_problem()
		@timeit TO "ParametricGOOP construction" begin
			ReducedGOOP.ParametricGOOP(
				dummy_primals,
				dummy_parameters;
				preferences,
				is_prioritized_constraint,
				equality_constraints,
				inequality_constraints = fill(nothing, num_players),
				shared_equality_constraint = nothing,
				shared_inequality_constraint = nothing,
			)
		end
	end

	problem = @timeit TO "preference and constraint construction" build_goop_problem()

	(; problem, flatten_parameters, unflatten_parameters, state_dimension, control_dimension)
end

# ── Experiment entry point ─────────────────────────────────────────────────────

function demo(;
	verbose = true,
	rng_seed = 1,
	debug = false,
	goop_version = :reduced, # :complete | :reduced | :quasi
	planning_horizon = 4,
	Δt = 0.1,
	safety_radius = fill(0.3, 3),
	urgency_clearance = 0.5,
	v_max = 3.0,
	ϵ₀ = 0.1,
	max_inner_iters = 1000,
	η₀ = 5e-5,
	tsvd_threshold = 0.0,
	warmstart_speed_scale = nothing, # per-player scale on warmstart speed; defaults to 0.5 (of v_max) for every player
	linear_solver = :klu, # :svd (dense) or :klu (sparse); per Robotic_arm_receding.jl's defaults
	kkt_backend = :symbolics, # :symbolics or :fast_differentiation
	kkt_codegen = :fast_differentiation,
	fd_codegen_chunk_size = 128, # bounds RuntimeGeneratedFunction size for :fast_differentiation codegen
)
	reset_timer!(TO)
	@timeit TO "experiment setup" Random.seed!(rng_seed)

	# ── Scenario: paper Sec. 10, three-agent crossing-corridor toy example ─────
	num_players = 3
	urgency_levels = [1, 0, 0] # agent 1 = emergency robot; agents 2, 3 = routine, equal priority
	initial_states = [[-2.0, 0.0], [0.0, -2.2], [1.0, 2.0]]
	goal_positions = [[2.0, 0.0], [0.0, 2.0], [1.0, -2.0]]

	(; problem, flatten_parameters, state_dimension, control_dimension) = @timeit TO "problem setup" begin
		get_setup(
			num_players;
			urgency_levels,
			Δt,
			planning_horizon,
			v_max,
			safety_radius,
			urgency_clearance,
		)
	end

	kkt_generators = Dict(
		:complete => ReducedGOOP.generate_slacked_complete_kkt_system,
		:reduced  => ReducedGOOP.generate_slacked_reduced_kkt_system,
		:quasi    => ReducedGOOP.generate_slacked_quasi_kkt_system,
	)
	GOOP_kkt_generator = get(kkt_generators, goop_version, nothing)
	isnothing(GOOP_kkt_generator) && error("Unknown GOOP version: $(goop_version)")

	symbolic_backends = Dict(
		:symbolics => ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
		:fast_differentiation => ReducedGOOP.SymbolicTracingUtils.FastDifferentiationBackend(),
	)
	backend = get(symbolic_backends, kkt_backend, nothing)
	isnothing(backend) && error("Unknown KKT tracing backend: $(kkt_backend)")

	@info "Building KKT system for $(goop_version) hospital GOOP formulation ($(kkt_backend) backend, $(kkt_codegen) codegen)..."
	GOOP_kkt_system = @timeit TO "KKT construction" GOOP_kkt_generator(
		problem;
		backend,
		backend_options = (;),
		codegen = kkt_codegen,
		fd_codegen_chunk_size,
	)
	println("[Primal-Dual] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)
	println("[Primal-Dual] variable Dimension: ", GOOP_kkt_system.variable_dimension)

	θ = @timeit TO "instance parameter construction" vcat(
		(
			flatten_parameters(; initial_state = initial_states[i], goal_position = goal_positions[i])
			for i in 1:num_players
		)...,
	)

	primal_dimension = (state_dimension + control_dimension) * planning_horizon
	speed_scale = something(warmstart_speed_scale, fill(0.5, num_players))
	z₀ = @timeit TO "warmstart construction" build_straight_line_warmstart(
		num_players,
		planning_horizon,
		initial_states,
		goal_positions,
		Δt;
		v_max,
		speed_scale,
	)

	options = ReducedGOOP.InteriorPointOptions(;
		tol = 3e-3,
		η₀,
		ϵ₀,
		max_inner_iters,
		max_outer_iters = 1,
		tightening_rate = 1.2,
		loosening_rate = 3.0,
		min_stepsize = 1e-20,
		linesearch = :backtracking,
		record_convergence = true,
		record_condition_number = (linear_solver == :svd),
		eta_retry_growth = 0.3,
		tsvd_threshold,
		use_marquardt_scaling = (linear_solver == :svd),
		linear_solver,
		verbose,
	)

	@info "Solving 3-agent hospital corridor game..."
	output = @timeit TO "solver invocation" ReducedGOOP.solve(
		ReducedGOOP.InteriorPoint(),
		GOOP_kkt_system,
		θ;
		z₀,
		options,
	)
	(; status, z, x, kkt_error, total_iters, kkt_error_history) = output
	println(
		"status = $(status), kkt_error = $(round(kkt_error; sigdigits = 4)), total_iters = $(total_iters)",
	)

	primal_x = x[1:sum(problem.primal_dims)] # `x` from InteriorPoint is primal++equality-duals (λ); see solver.jl's `x_dims`
	print_constraint_breakdown(problem, primal_x, θ, num_players)

	strategies = extract_player_strategies(x, num_players, primal_dimension, state_dimension, control_dimension)

	run_dir = @timeit TO "output directory setup" prepare_hospital_output_dir(debug)
	fig, _ = plot_hospital_trajectories(; strategy = strategies, initial_states, goal_positions, urgency_levels, urgency_clearance)
	CairoMakie.save(joinpath(run_dir, "trajectories.pdf"), fig)
	if !isempty(kkt_error_history)
		convergence_fig, _ = plot_convergence_plot(;
			kkt_error_history = safe_log10_history(kkt_error_history),
			total_iters,
		)
		CairoMakie.save(joinpath(run_dir, "convergence.pdf"), convergence_fig)
	end
	JLD2.save_object(
		joinpath(run_dir, "solution.jld2"),
		Dict(
			"z" => z,
			"x" => x,
			"status" => status,
			"kkt_error" => kkt_error,
			"kkt_error_history" => kkt_error_history,
			"strategies" => strategies,
			"urgency_levels" => urgency_levels,
			"initial_states" => initial_states,
			"goal_positions" => goal_positions,
		),
	)
	println("Saved trajectory plot and solution to $(run_dir)")

	println("\nTiming summary:")
	show(TO)
	println()

	(; status, strategies, kkt_error, z, run_dir)
end

# ── Output / plotting helpers ──────────────────────────────────────────────────

function prepare_hospital_output_dir(debug)
	timestamp = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
	run_dir = joinpath("data", "Hospital_open_loop", debug ? "debug" : "runs", timestamp)
	mkpath(run_dir)
	run_dir
end

function safe_log10_history(history)
	map(history) do value
		isfinite(value) && value > 0 ? log10(value) : NaN
	end
end

function plot_hospital_trajectories(; strategy, initial_states, goal_positions, urgency_levels, urgency_clearance = nothing)
	num_players = length(strategy)
	palette = [:crimson, :dodgerblue, :seagreen, :darkorange, :purple]
	colors = palette[mod1.(1:num_players, length(palette))]

	figure = serif_figure()
	ax = CairoMakie.Axis(
		figure[1, 1];
		aspect = 1,
		xlabel = "x",
		ylabel = "y",
		xgridvisible = false,
		ygridvisible = false,
	)

	legend_handles = Any[]
	legend_labels = String[]
	for i in 1:num_players
		xs = strategy[i].xs
		trajectory_line = CairoMakie.scatterlines!(
			ax,
			[x[1] for x in xs],
			[x[2] for x in xs];
			color = colors[i],
			linewidth = 2,
		)
		CairoMakie.lines!(
			ax,
			[initial_states[i][1], goal_positions[i][1]],
			[initial_states[i][2], goal_positions[i][2]];
			color = colors[i],
			linestyle = :dash,
			linewidth = 1,
		)
		CairoMakie.scatter!(ax, CairoMakie.Point2f(initial_states[i]); markersize = 16, color = colors[i])
		CairoMakie.scatter!(
			ax,
			CairoMakie.Point2f(goal_positions[i]);
			markersize = 16,
			marker = :star5,
			color = colors[i],
		)
		push!(legend_handles, trajectory_line)
		push!(legend_labels, "Agent $(i) (urgency $(urgency_levels[i]))")
	end

	if !isnothing(urgency_clearance)
		"""
		One circle per player, centered at their *current* position — i.e. `xs[1]`,
		the first (earliest/"now") point of this plan, not `xs[end]` (the most
		future point) — with radius = clearance/2. Two agents' circles overlap
		exactly when their current separation < clearance (r_i + r_j =
		clearance/2 + clearance/2), so overlapping circles visually flag a
		clearance violation right now.
		"""
		θ_circle = range(0, 2π; length = 100)
		radius = urgency_clearance / 2
		for i in 1:num_players
			current_position = strategy[i].xs[1]
			CairoMakie.lines!(
				ax,
				current_position[1] .+ radius .* cos.(θ_circle),
				current_position[2] .+ radius .* sin.(θ_circle);
				color = colors[i],
				linestyle = :dot,
				linewidth = 2,
			)
		end
	end

	CairoMakie.Legend(
		figure[1, 1],
		legend_handles,
		legend_labels;
		framevisible = false,
		tellheight = false,
		tellwidth = false,
		halign = :left,
		valign = :top,
	)

	(figure, ax)
end

# ── Internal utilities ─────────────────────────────────────────────────────────

function print_constraint_breakdown(problem, x, θ, num_players)
	"""
	Evaluate each player's equality-constraint residual (dynamics + control-bound +
	shared collision, all embedded via squared-violation) and preference-level
	values directly at the given (unconverged or converged) primal point. Unlike
	the aggregate KKT error, this identifies *which* constraint/objective is
	responsible for a large residual — e.g. distinguishing a `v_max` violation
	from an unsatisfied interference margin.
	"""
	x_block = BlockArray(collect(x), problem.primal_dims)
	θ_block = BlockArray(collect(θ), problem.parameter_dims)

	println("\n--- Constraint/preference breakdown at reported solution ---")
	for i in 1:num_players
		residual = problem.equality_constraints[i](x_block, θ_block)
		println("  player $(i) equality residual: ‖·‖ = $(round(norm(residual, 2); sigdigits = 4)), max|·| = $(round(maximum(abs.(residual)); sigdigits = 4))")
	end
	for i in 1:num_players
		for (level, (pref, is_constraint)) in enumerate(zip(problem.preferences[i], problem.is_prioritized_constraint[i]))
			value = pref(x_block, θ_block)
			tag = is_constraint ? "constraint, want ≥ 0" : "objective"
			println("  player $(i) level $(level) [$(tag)]: $(round.(value; sigdigits = 4))")
		end
	end
	println("---")
end

function extract_player_strategies(primal_solution, num_players, primal_dimension, state_dimension, control_dimension)
	map(1:num_players) do player_idx
		start_idx = primal_dimension * (player_idx - 1) + 1
		end_idx   = start_idx + primal_dimension - 1
		unflatten_trajectory(primal_solution[start_idx:end_idx], state_dimension, control_dimension)
	end
end

function build_straight_line_warmstart(
	num_players,
	planning_horizon,
	initial_states,
	goal_positions,
	Δt;
	v_max = Inf,
	speed_scale = ones(num_players),
)
	"""
	Constant-velocity warmstart toward each agent's goal; final control is zeroed.
	Speed is `v_max * speed_scale[i]` (a fixed nominal cruise speed, not derived
	from distance/time-to-goal then capped): for a short planning horizon the
	direct distance/time speed can be arbitrarily larger than `v_max`, and
	starting every agent's warmstart pinned right at the `v_max` boundary
	simultaneously is itself a likely source of the solver getting stuck
	fighting the (hard) speed-limit constraint at several timesteps at once.
	Using a nominal fraction of `v_max` keeps the warmstart comfortably feasible
	by construction. `speed_scale[i] = 1.0` for the top-urgency agent and lower
	(e.g. 0.5) for the rest is a reasonable default (see `demo`).
	"""
	warmstart = Float64[]
	for i in 1:num_players
		p0 = initial_states[i]
		g = goal_positions[i]
		direction = g .- p0
		distance = sqrt(sum(direction .^ 2))
		unit_direction = distance > 0 ? direction ./ distance : zeros(2)
		velocity = unit_direction .* v_max .* speed_scale[i]

		xs = [collect(p0)]
		for t in 1:(planning_horizon-1)
			push!(xs, single_integrator_2d_step(xs[t], velocity; Δt))
		end
		us = [collect(velocity) for _ in 1:planning_horizon]
		us[end] = zeros(2)

		for t in 1:planning_horizon
			append!(warmstart, xs[t])
			append!(warmstart, us[t])
		end
	end
	warmstart
end

end
