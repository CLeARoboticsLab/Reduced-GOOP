module HospitalMPC

#=
	Receding-horizon (MPC) wrapper around `Hospital.jl`'s single-shot 3-agent
	hospital-corridor solve. Each MPC step re-solves the same (horizon=4, Δt=0.1)
	problem from the agents' *current* true state, applies only the first
	planned control to advance the true state, and re-plans from there —
	standard closed-loop receding-horizon simulation.

	The GOOP problem and its compiled KKT system are built *once* and reused
	across every MPC step (only the parameter vector θ's initial_state entries
	change step to step), since (re)compiling the symbolic KKT system is by far
	the most expensive part of a single solve (order 15-20s here) and the
	problem structure itself never changes between steps.
=#

include(joinpath(@__DIR__, "Hospital.jl"))
using .Hospital

function run_mpc(;
	total_time = 1.0,
	Δt = 0.1,
	planning_horizon = 4,
	v_max = 3.0,
	urgency_clearance = 0.5,
	interference_formula = :pointwise, # :pointwise (recommended) or :cpa (legacy, kept for comparison)
	safety_radius = fill(0.3, 3),
	goop_version = :complete,
	max_inner_iters = 100,
	verbose = false,
	debug = false,
	linear_solver = :klu, # :svd (dense) or :klu (sparse); per Robotic_arm_receding.jl's defaults
	kkt_backend = :symbolics, # :symbolics or :fast_differentiation
	kkt_codegen = :fast_differentiation,
	fd_codegen_chunk_size = 128, # bounds RuntimeGeneratedFunction size for :fast_differentiation codegen
	make_gif = true, # combine the per-step plan plots into an animated GIF after the run
	gif_fps = 5,
	warmstart_speed_scale = nothing, # per-player scale on warmstart speed; defaults to 0.5 (of v_max) for every player
	stagnation_patience = 3, # stop early if kkt_error fails to improve on its best-seen value for this many consecutive steps
	stagnation_tolerance = 0.05, # relative wobble around the best-seen kkt_error that still counts as "no regression" (avoids false triggers from noise-level fluctuations)
)
	num_players = 3
	urgency_levels = [1, 0, 0] # agent 1 = emergency robot; agents 2, 3 = routine, equal priority
	initial_states = [[-2.0, 0.0], [0.0, -2.2], [1.0, 2.0]]
	goal_positions = [[2.0, 0.0], [0.0, 2.0], [1.0, -2.0]]

	num_mpc_steps = round(Int, total_time / Δt)
	num_mpc_steps > 0 || error("total_time / Δt must be at least 1.")

	(; problem, flatten_parameters, state_dimension, control_dimension) = Hospital.get_setup(
		num_players;
		urgency_levels,
		Δt,
		planning_horizon,
		v_max,
		safety_radius,
		urgency_clearance,
		interference_formula,
	)

	kkt_generators = Dict(
		:complete => Hospital.ReducedGOOP.generate_slacked_complete_kkt_system,
		:reduced  => Hospital.ReducedGOOP.generate_slacked_reduced_kkt_system,
		:quasi    => Hospital.ReducedGOOP.generate_slacked_quasi_kkt_system,
	)
	GOOP_kkt_generator = get(kkt_generators, goop_version, nothing)
	isnothing(GOOP_kkt_generator) && error("Unknown GOOP version: $(goop_version)")

	symbolic_backends = Dict(
		:symbolics => Hospital.ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
		:fast_differentiation => Hospital.ReducedGOOP.SymbolicTracingUtils.FastDifferentiationBackend(),
	)
	backend = get(symbolic_backends, kkt_backend, nothing)
	isnothing(backend) && error("Unknown KKT tracing backend: $(kkt_backend)")

	@info "Building KKT system once for $(num_mpc_steps) MPC steps ($(total_time)s at Δt=$(Δt)), $(kkt_backend) backend, $(kkt_codegen) codegen..."
	GOOP_kkt_system = GOOP_kkt_generator(
		problem;
		backend,
		backend_options = (;),
		codegen = kkt_codegen,
		fd_codegen_chunk_size,
	)
	println("[MPC] KKT Dimension: ", GOOP_kkt_system.kkt_dimension)

	primal_dimension = (state_dimension + control_dimension) * planning_horizon

	options = Hospital.ReducedGOOP.InteriorPointOptions(;
		tol = 3e-3,
		η₀ = 5e-5,
		ϵ₀ = 0.1,
		max_inner_iters,
		max_outer_iters = 1,
		tightening_rate = 1.2,
		loosening_rate = 3.0,
		min_stepsize = 1e-20,
		linesearch = :backtracking,
		record_convergence = false,
		record_condition_number = false,
		eta_retry_growth = 0.3,
		tsvd_threshold = 0.0,
		use_marquardt_scaling = (linear_solver == :svd),
		linear_solver,
		verbose,
	)

	speed_scale = something(warmstart_speed_scale, fill(0.5, num_players))
	current_states = [collect(p) for p in initial_states]
	z₀ = Hospital.build_straight_line_warmstart(
		num_players,
		planning_horizon,
		initial_states,
		goal_positions,
		Δt;
		v_max,
		speed_scale,
	)

	closed_loop_positions = [[collect(initial_states[i])] for i in 1:num_players]
	closed_loop_controls = [Vector{Float64}[] for _ in 1:num_players]
	step_statuses = Symbol[]
	step_kkt_errors = Float64[]

	run_dir = Hospital.prepare_hospital_output_dir(debug)
	steps_dir = joinpath(run_dir, "steps")
	mkpath(steps_dir)

	# Fixed axis limits (same across every saved figure) computed from the scenario's
	# known extent (start + goal positions, which never change across MPC steps), with
	# a margin — rather than reusing the first frame's auto-computed limits, which could
	# be too tight if a later frame's trajectory strays further out.
	all_points = vcat(initial_states, goal_positions)
	margin = 0.5
	fixed_xlim = (minimum(p[1] for p in all_points) - margin, maximum(p[1] for p in all_points) + margin)
	fixed_ylim = (minimum(p[2] for p in all_points) - margin, maximum(p[2] for p in all_points) + margin)

	best_kkt_error = Inf
	best_step = 0
	no_improve_streak = 0
	stagnation_triggered = false

	for step in 1:num_mpc_steps
		θ = vcat(
			(
				flatten_parameters(; initial_state = current_states[i], goal_position = goal_positions[i])
				for i in 1:num_players
			)...,
		)

		Hospital.reset_timer!(Hospital.TO)
		output = Hospital.ReducedGOOP.solve(
			Hospital.ReducedGOOP.InteriorPoint(),
			GOOP_kkt_system,
			θ;
			z₀,
			options,
		)
		(; status, x, kkt_error) = output
		push!(step_statuses, status)
		push!(step_kkt_errors, kkt_error)
		println(
			"[MPC step $(step)/$(num_mpc_steps)] status = $(status), kkt_error = $(round(kkt_error; sigdigits = 4))",
		)

		step_tag_for_log = lpad(step, 3, '0')
		open(joinpath(steps_dir, "step_$(step_tag_for_log)_console.log"), "w") do io
			println(io, "MPC step $(step)/$(num_mpc_steps): status = $(status), kkt_error = $(kkt_error)")
			println(io, "\nTiming summary (this step's solve only -- reset before each step, so this is NOT cumulative):")
			show(io, Hospital.TO)
			println(io)
		end

		if kkt_error < best_kkt_error
			best_kkt_error = kkt_error
			best_step = step
			no_improve_streak = 0
		elseif kkt_error <= best_kkt_error * (1 + stagnation_tolerance)
			no_improve_streak = 0 # within noise tolerance of the best-seen value; not a real regression
		else
			no_improve_streak += 1
		end

		strategies = Hospital.extract_player_strategies(
			x,
			num_players,
			primal_dimension,
			state_dimension,
			control_dimension,
		)

		next_states = map(1:num_players) do i
			Hospital.single_integrator_2d_step(current_states[i], strategies[i].us[1]; Δt)
		end

		applied_controls = [collect(strategies[i].us[1]) for i in 1:num_players]
		for i in 1:num_players
			push!(closed_loop_positions[i], collect(next_states[i]))
			push!(closed_loop_controls[i], applied_controls[i])
		end

		step_tag = lpad(step, 3, '0')
		Hospital.JLD2.save_object(
			joinpath(steps_dir, "step_$(step_tag).jld2"),
			Dict(
				"step" => step,
				"status" => status,
				"kkt_error" => kkt_error,
				"current_state" => current_states,
				"next_state" => next_states,
				"applied_control" => applied_controls,
				"planned_strategies" => strategies,
				"theta" => θ,
			),
		)

		step_fig, step_ax = Hospital.plot_hospital_trajectories(;
			strategy = strategies,
			initial_states = current_states,
			goal_positions,
			urgency_levels,
			urgency_clearance,
			safety_radius,
		)
		Hospital.CairoMakie.xlims!(step_ax, fixed_xlim...)
		Hospital.CairoMakie.ylims!(step_ax, fixed_ylim...)
		Hospital.CairoMakie.save(joinpath(steps_dir, "step_$(step_tag)_plan.pdf"), step_fig)

		z₀ = shift_and_extend_warmstart(
			strategies,
			next_states,
			goal_positions,
			Δt,
			v_max,
			planning_horizon,
			num_players,
			speed_scale,
		)
		current_states = next_states

		if no_improve_streak >= stagnation_patience
			@warn "MPC stalled: kkt_error has not improved for $(stagnation_patience) consecutive steps (best was step $(best_step), kkt_error = $(round(best_kkt_error; sigdigits = 4))). Stopping early at step $(step)."
			stagnation_triggered = true
			break
		end
	end

	if stagnation_triggered && best_step < length(step_kkt_errors)
		println(
			"Truncating MPC output to best-performing step $(best_step)/$(length(step_kkt_errors)) (kkt_error = $(round(best_kkt_error; sigdigits = 4))).",
		)
		closed_loop_positions = [pos[1:(best_step + 1)] for pos in closed_loop_positions]
		closed_loop_controls = [ctrl[1:best_step] for ctrl in closed_loop_controls]
		step_statuses = step_statuses[1:best_step]
		step_kkt_errors = step_kkt_errors[1:best_step]
	end

	fig, ax = plot_mpc_trajectories(; closed_loop_positions, initial_states, goal_positions, urgency_levels, urgency_clearance, safety_radius)
	Hospital.CairoMakie.xlims!(ax, fixed_xlim...)
	Hospital.CairoMakie.ylims!(ax, fixed_ylim...)
	Hospital.CairoMakie.save(joinpath(run_dir, "mpc_trajectories.pdf"), fig)

	Hospital.JLD2.save_object(
		joinpath(run_dir, "mpc_solution.jld2"),
		Dict(
			"closed_loop_positions" => closed_loop_positions,
			"closed_loop_controls" => closed_loop_controls,
			"step_statuses" => step_statuses,
			"step_kkt_errors" => step_kkt_errors,
			"initial_states" => initial_states,
			"goal_positions" => goal_positions,
			"urgency_levels" => urgency_levels,
			"Δt" => Δt,
			"total_time" => total_time,
		),
	)
	println("Saved MPC closed-loop trajectory to $(run_dir) (per-step solves in $(steps_dir))")

	make_gif && _build_mpc_animation(steps_dir, run_dir; fps = gif_fps)

	(; closed_loop_positions, closed_loop_controls, step_statuses, step_kkt_errors, run_dir)
end

"""
	_build_mpc_animation(steps_dir, run_dir; fps = 3)

Combine the per-step `step_*_plan.pdf` plots already saved under `steps_dir`
into `run_dir/mpc_animation.gif`, without re-running the solver. Uses macOS's
`qlmanage` to rasterize each PDF to PNG and `ffmpeg` (palette-based, for
GIF-quality color) to assemble the sequence at `fps` frames per second.
"""
function _build_mpc_animation(steps_dir, run_dir; fps = 3)
	pdf_files = sort(filter(f -> endswith(f, "_plan.pdf"), readdir(steps_dir; join = true)))
	isempty(pdf_files) && return nothing

	frames_dir = mktempdir()
	for (i, pdf) in enumerate(pdf_files)
		run(pipeline(`qlmanage -t -s 800 -o $frames_dir $pdf`; stdout = devnull, stderr = devnull))
		mv(joinpath(frames_dir, basename(pdf) * ".png"), joinpath(frames_dir, "frame_$(lpad(i, 3, '0')).png"))
	end

	gif_path = joinpath(run_dir, "mpc_animation.gif")
	frame_pattern = joinpath(frames_dir, "frame_%03d.png")
	run(
		pipeline(
			`ffmpeg -y -framerate $fps -i $frame_pattern -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $gif_path`;
			stdout = devnull,
			stderr = devnull,
		),
	)
	println("Saved MPC animation ($(fps)Hz) to $(gif_path)")
	gif_path
end

function shift_and_extend_warmstart(strategies, next_states, goal_positions, Δt, v_max, planning_horizon, num_players, speed_scale)
	"""
	Standard MPC warmstart: drop the just-applied first step, shift the rest of
	the previous plan up by one, and append one fresh tail point. The new tail
	point is not a "hold last control" — it heads toward the goal at
	`speed_scale[i] * v_max` (matching `Hospital.build_straight_line_warmstart`'s
	own nominal-speed convention), since holding the last (possibly stale,
	corridor/interference-shaped) control would drift away from the goal
	instead of continuing to approach it.
	"""
	warmstart = Float64[]
	for i in 1:num_players
		old_xs = strategies[i].xs
		old_us = strategies[i].us

		new_xs = Vector{Vector{Float64}}(undef, planning_horizon)
		new_us = Vector{Vector{Float64}}(undef, planning_horizon)

		new_xs[1] = collect(next_states[i])
		for t in 2:(planning_horizon-1)
			new_xs[t] = collect(old_xs[t+1])
		end
		for t in 1:(planning_horizon-2)
			new_us[t] = collect(old_us[t+1])
		end

		tail_direction = goal_positions[i] .- new_xs[planning_horizon-1]
		tail_distance = sqrt(sum(tail_direction .^ 2))
		tail_velocity = tail_distance > 0 ? tail_direction .* (speed_scale[i] * v_max / tail_distance) : zeros(2)
		new_us[planning_horizon-1] = tail_velocity
		new_xs[planning_horizon] = new_xs[planning_horizon-1] .+ Δt .* tail_velocity
		new_us[planning_horizon] = zeros(2)

		for t in 1:planning_horizon
			append!(warmstart, new_xs[t])
			append!(warmstart, new_us[t])
		end
	end
	warmstart
end

function plot_mpc_trajectories(; closed_loop_positions, initial_states, goal_positions, urgency_levels, urgency_clearance = nothing, safety_radius = nothing)
	num_players = length(closed_loop_positions)
	palette = [:crimson, :dodgerblue, :seagreen, :darkorange, :purple]
	colors = palette[mod1.(1:num_players, length(palette))]

	figure = Hospital.serif_figure()
	ax = Hospital.CairoMakie.Axis(
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
		xs = closed_loop_positions[i]
		trajectory_line = Hospital.CairoMakie.scatterlines!(
			ax,
			[x[1] for x in xs],
			[x[2] for x in xs];
			color = colors[i],
			linewidth = 2,
		)
		Hospital.CairoMakie.lines!(
			ax,
			[initial_states[i][1], goal_positions[i][1]],
			[initial_states[i][2], goal_positions[i][2]];
			color = colors[i],
			linestyle = :dash,
			linewidth = 1,
		)
		Hospital.CairoMakie.scatter!(ax, Hospital.CairoMakie.Point2f(initial_states[i]); markersize = 16, color = colors[i])
		Hospital.CairoMakie.scatter!(
			ax,
			Hospital.CairoMakie.Point2f(goal_positions[i]);
			markersize = 16,
			marker = :star5,
			color = colors[i],
		)
		push!(legend_handles, trajectory_line)
		push!(legend_labels, "Agent $(i) (urgency $(urgency_levels[i]))")
	end

	if !isnothing(urgency_clearance)
		"""
		One circle per player, centered at their *current* (most recent) closed-
		loop position, radius = clearance/2: two agents' circles overlap exactly
		when their separation < clearance (r_i + r_j = clearance/2 + clearance/2),
		so overlapping circles visually flag a clearance violation *right now*.
		"""
		θ_circle = range(0, 2π; length = 100)
		radius = urgency_clearance / 2
		for i in 1:num_players
			current_position = closed_loop_positions[i][end]
			Hospital.CairoMakie.lines!(
				ax,
				current_position[1] .+ radius .* cos.(θ_circle),
				current_position[2] .+ radius .* sin.(θ_circle);
				color = colors[i],
				linestyle = :dot,
				linewidth = 2,
			)
		end
	end

	if !isnothing(safety_radius)
		"""
		Physical no-overlap safety circle per player, centered at their *current*
		(most recent) closed-loop position, radius = safety_radius[i] -- the plain
		`shared_pairwise_collision_avoidance` clearance, distinct from (and
		smaller than) the urgency-interference buffer above.
		"""
		θ_circle = range(0, 2π; length = 100)
		for i in 1:num_players
			current_position = closed_loop_positions[i][end]
			Hospital.CairoMakie.lines!(
				ax,
				current_position[1] .+ safety_radius[i] .* cos.(θ_circle),
				current_position[2] .+ safety_radius[i] .* sin.(θ_circle);
				color = colors[i],
				linestyle = :dashdot,
				linewidth = 1,
			)
		end
	end

	Hospital.CairoMakie.Legend(
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

end
