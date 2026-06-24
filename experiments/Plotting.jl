# using TrajectoryGamesBase: OpenLoopStrategy
using CairoMakie: CairoMakie
# using LaTeXStrings: @L_str

const SERIF_FONT = "TeX Gyre Termes Makie"

function serif_figure(; kwargs...)
	return CairoMakie.Figure(;
		fonts = (
			;
			regular = SERIF_FONT,
			bold = SERIF_FONT,
			italic = SERIF_FONT,
			bold_italic = SERIF_FONT,
		),
		kwargs...,
	)
end

function draw_intersection_map!(ax; map_end, lane_width, offset = 0.2)
	vertical_road_background = CairoMakie.Polygon(
		CairoMakie.Point2f[
			(-lane_width - offset, -map_end),
			(lane_width + offset, -map_end),
			(lane_width + offset, map_end),
			(-lane_width - offset, map_end),
		],
	)
	CairoMakie.poly!(ax, vertical_road_background, color = :white)
	CairoMakie.lines!(
		ax,
		[-lane_width - offset, -lane_width - offset],
		[-map_end, -lane_width],
		color = :black,
		linewidth = 1,
	)
	CairoMakie.lines!(
		ax,
		[-lane_width - offset, -lane_width - offset],
		[map_end, lane_width],
		color = :black,
		linewidth = 1,
	)
	CairoMakie.lines!(
		ax,
		[lane_width + offset, lane_width + offset],
		[-map_end, -lane_width],
		color = :black,
		linewidth = 1,
	)
	CairoMakie.lines!(
		ax,
		[lane_width + offset, lane_width + offset],
		[map_end, lane_width],
		color = :black,
		linewidth = 1,
	)

	horizontal_road_background = CairoMakie.Polygon(
		CairoMakie.Point2f[
			(-map_end, -lane_width - offset),
			(map_end, -lane_width - offset),
			(map_end, lane_width + offset),
			(-map_end, lane_width + offset),
		],
	)
	CairoMakie.poly!(ax, horizontal_road_background, color = :white)
	CairoMakie.lines!(
		ax,
		[-lane_width - offset, -map_end],
		[-lane_width - offset, -lane_width - offset],
		color = :black,
		linewidth = 1,
	)
	CairoMakie.lines!(
		ax,
		[-lane_width - offset, -map_end],
		[lane_width + offset, lane_width + offset],
		color = :black,
		linewidth = 1,
	)
	CairoMakie.lines!(
		ax,
		[lane_width + offset, map_end],
		[lane_width + offset, lane_width + offset],
		color = :black,
		linewidth = 1,
	)
	CairoMakie.lines!(
		ax,
		[lane_width + offset, map_end],
		[-lane_width - offset, -lane_width - offset],
		color = :black,
		linewidth = 1,
	)

	vertical_road = CairoMakie.Polygon(
		CairoMakie.Point2f[
			(-lane_width, -map_end),
			(lane_width, -map_end),
			(lane_width, map_end),
			(-lane_width, map_end),
		],
	)
	CairoMakie.poly!(ax, vertical_road, color = :gray)

	horizontal_road = CairoMakie.Polygon(
		CairoMakie.Point2f[
			(-map_end, -lane_width),
			(map_end, -lane_width),
			(map_end, lane_width),
			(-map_end, lane_width),
		],
	)
	CairoMakie.poly!(ax, horizontal_road, color = :gray)

	# Lane center lines
	CairoMakie.lines!(ax, [-lane_width, -map_end], [0, 0], color = :yellow, linewidth = 2)
	CairoMakie.lines!(ax, [lane_width, map_end], [0, 0], color = :yellow, linewidth = 2)
	CairoMakie.lines!(ax, [0, 0], [-lane_width, -map_end], color = :yellow, linewidth = 2)
	CairoMakie.lines!(ax, [0, 0], [lane_width, map_end], color = :yellow, linewidth = 2)

	# Direction arrows
	xs = [-3.0, 3.0, 1.0, -1.0]
	ys = [-1.0, 1.0, -3.0, 3.0]
	us = [1.0, -1.0, 0.0, 0.0]
	vs = [0.0, 0.0, 1.0, -1.0]
	CairoMakie.arrows!(
		ax,
		xs,
		ys,
		us,
		vs;
		arrowsize = 15,
		lengthscale = 0.5,
		arrowcolor = :white,
		linecolor = :white,
		linewidth = 3,
	)

	return ax
end

function plot_intersection_trajectories(;
	map_end,
	lane_width,
	strategy,
	θ1,
	θ2,
	goal_position1,
	goal_position2,
)
	if length(strategy) < 2
		error("plot_intersection_trajectories expects strategies for at least two players.")
	end

	figure = serif_figure()
	legend_label_fontsize = 16
	ax = CairoMakie.Axis(
		figure[1, 1];
		aspect = 1,
		xgridvisible = false,
		ygridvisible = false,
		backgroundcolor = :white,
	)
	CairoMakie.hidedecorations!(ax)
	CairoMakie.hidespines!(ax)

	draw_intersection_map!(ax; map_end, lane_width)

	xs1 = strategy[1].xs
	xs2 = strategy[2].xs
	if isempty(xs1) || isempty(xs2)
		error("plot_intersection_trajectories expects non-empty player trajectories.")
	end

	player1_trajectory = CairoMakie.scatterlines!(
		ax,
		[x[1] for x in xs1],
		[x[2] for x in xs1];
		color = :blue,
		linewidth = 2,
	)
	player2_trajectory = CairoMakie.scatterlines!(
		ax,
		[x[1] for x in xs2],
		[x[2] for x in xs2];
		color = :red,
		linewidth = 2,
	)

	start1_marker = CairoMakie.scatter!(
		ax,
		CairoMakie.Point2f(θ1[1:2]),
		markersize = 20,
		color = :blue,
	)
	start2_marker = CairoMakie.scatter!(
		ax,
		CairoMakie.Point2f(θ2[1:2]),
		markersize = 20,
		color = :red,
	)
	goal1_marker = CairoMakie.scatter!(
		ax,
		CairoMakie.Point2f(goal_position1),
		markersize = 20,
		marker = :star5,
		color = :blue,
	)
	goal2_marker = CairoMakie.scatter!(
		ax,
		CairoMakie.Point2f(goal_position2),
		markersize = 20,
		marker = :star5,
		color = :red,
	)
	CairoMakie.Legend(
		figure[1, 1],
		[player1_trajectory, player2_trajectory, start1_marker, start2_marker, goal1_marker, goal2_marker],
		["Player 1", "Player 2", "Start 1", "Start 2", "Goal 1", "Goal 2"];
		framevisible = false,
		labelsize = legend_label_fontsize,
		orientation = :vertical,
		tellheight = false,
		tellwidth = false,
		halign = :left,
		valign = :top,
		margin = (100, 12, 12, 6),
	)

	return figure, ax
end

function velocity_plot(;
	strategy,
	velocity_limit = 1.5,
	dynamics_model = :planar_double_integrator,
)
	if length(strategy) < 2
		error("velocity_plot expects strategies for at least two players.")
	end

	axis_label_fontsize = 24
	tick_label_fontsize = 22
	legend_label_fontsize = 18
	xs1 = strategy[1].xs
	xs2 = strategy[2].xs
	trajectory_len = min(length(xs1), length(xs2))
	if trajectory_len == 0
		error("velocity_plot expects non-empty player trajectories.")
	end

	horizon_steps = 0:(trajectory_len-1)
	velocity_limit_profile = fill(velocity_limit, trajectory_len)

	figure = serif_figure()
	if dynamics_model === :unicycle
		speed1 = [xs1[k][3] for k in 1:trajectory_len]
		speed2 = [xs2[k][3] for k in 1:trajectory_len]
		ax = CairoMakie.Axis(
			figure[1, 1];
			xlabel = "time step [s]",
			ylabel = "speed [m/s]",
			xlabelsize = axis_label_fontsize,
			ylabelsize = axis_label_fontsize,
			xticklabelsize = tick_label_fontsize,
			yticklabelsize = tick_label_fontsize,
		)
		player1_speed = CairoMakie.scatterlines!(
			ax,
			horizon_steps,
			speed1;
			color = :blue,
			label = "Player 1",
			linewidth = 3,
		)
		player2_speed = CairoMakie.scatterlines!(
			ax,
			horizon_steps,
			speed2;
			color = :red,
			label = "Player 2",
			linewidth = 3,
		)
		speed_limit_line = CairoMakie.lines!(
			ax,
			horizon_steps,
			velocity_limit_profile;
			color = :black,
			linestyle = :dash,
			label = "Speed Limit [$(velocity_limit) m/s]",
			linewidth = 2,
		)
		CairoMakie.Legend(
			figure[1, 1],
			[player1_speed, player2_speed, speed_limit_line],
			["Player 1", "Player 2", "Speed Limit [$(velocity_limit) m/s]"];
			framevisible = false,
			labelsize = legend_label_fontsize,
			orientation = :vertical,
			tellheight = false,
			tellwidth = false,
			halign = :right,
			valign = :top,
		)
		return figure, (ax,)
	end

	vx1 = [xs1[k][3] for k in 1:trajectory_len]
	vy1 = [xs1[k][4] for k in 1:trajectory_len]
	vx2 = [xs2[k][3] for k in 1:trajectory_len]
	vy2 = [xs2[k][4] for k in 1:trajectory_len]

	ax_vx = CairoMakie.Axis(
		figure[1, 1];
		xlabel = "time step [s]",
		ylabel = " horizontal speed [m/s]",
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
	)
	player1_vx = CairoMakie.scatterlines!(
		ax_vx,
		horizon_steps,
		vx1;
		color = :blue,
		label = "Player 1",
		linewidth = 3,
	)
	player2_vx = CairoMakie.scatterlines!(
		ax_vx,
		horizon_steps,
		vx2;
		color = :red,
		label = "Player 2",
		linewidth = 3,
	)
	velocity_limit_line = CairoMakie.lines!(
		ax_vx,
		horizon_steps,
		velocity_limit_profile;
		color = :black,
		linestyle = :dash,
		label = "Speed Limit [$(velocity_limit) m/s]",
		linewidth = 2,
	)

	ax_vy = CairoMakie.Axis(
		figure[1, 2];
		xlabel = "time step [s]",
		ylabel = "vertical speed [m/s]",
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
	)
	CairoMakie.scatterlines!(ax_vy, horizon_steps, vy1; color = :blue, linewidth = 3)
	CairoMakie.scatterlines!(ax_vy, horizon_steps, vy2; color = :red, linewidth = 3)
	CairoMakie.lines!(
		ax_vy,
		horizon_steps,
		velocity_limit_profile;
		color = :black,
		linestyle = :dash,
		linewidth = 2,
	)

	CairoMakie.linkyaxes!(ax_vx, ax_vy)
	CairoMakie.Legend(
		figure[1, 2],
		[player1_vx, player2_vx, velocity_limit_line],
		["Player 1", "Player 2", "Speed Limit [$(velocity_limit) m/s]"];
		framevisible = false,
		labelsize = legend_label_fontsize,
		orientation = :vertical,
		tellheight = false,
		tellwidth = false,
		halign = :right,
		valign = :top,
	)

	return figure, (ax_vx, ax_vy)
end

function control_plot(;
	strategy,
	control_lb = nothing,
	control_ub = nothing,
)
	if length(strategy) < 2
		error("control_plot expects strategies for at least two players.")
	end

	axis_label_fontsize = 24
	tick_label_fontsize = 22
	legend_label_fontsize = 18

	us1 = strategy[1].us
	us2 = strategy[2].us
	control_horizon = min(length(us1), length(us2))
	if control_horizon == 0
		error("control_plot expects non-empty player controls.")
	end

	control_dimension = length(us1[1])
	if control_dimension == 0
		error("control_plot expects control vectors with non-zero dimension.")
	end

	lb = isnothing(control_lb) ? fill(-Inf, control_dimension) : collect(control_lb)
	ub = isnothing(control_ub) ? fill(Inf, control_dimension) : collect(control_ub)
	if length(lb) != control_dimension || length(ub) != control_dimension
		error("control bound lengths must match control dimension $(control_dimension).")
	end

	horizon_steps = 0:(control_horizon-1)
	figure = serif_figure()
	axes = CairoMakie.Axis[]

	player1_handle = nothing
	player2_handle = nothing
	upper_bound_handle = nothing
	lower_bound_handle = nothing

	for control_idx in 1:control_dimension
		u1 = [us1[k][control_idx] for k in 1:control_horizon]
		u2 = [us2[k][control_idx] for k in 1:control_horizon]

		ax = CairoMakie.Axis(
			figure[1, control_idx];
			xlabel = "time step [s]",
			ylabel = "u$(control_idx)",
			xlabelsize = axis_label_fontsize,
			ylabelsize = axis_label_fontsize,
			xticklabelsize = tick_label_fontsize,
			yticklabelsize = tick_label_fontsize,
		)
		push!(axes, ax)

		current_player1 = CairoMakie.scatterlines!(
			ax,
			horizon_steps,
			u1;
			color = :blue,
			linewidth = 3,
		)
		current_player2 = CairoMakie.scatterlines!(
			ax,
			horizon_steps,
			u2;
			color = :red,
			linewidth = 3,
		)
		if isnothing(player1_handle)
			player1_handle = current_player1
			player2_handle = current_player2
		end

		if isfinite(ub[control_idx])
			current_upper = CairoMakie.lines!(
				ax,
				horizon_steps,
				fill(ub[control_idx], control_horizon);
				color = :black,
				linestyle = :dash,
				linewidth = 2,
			)
			if isnothing(upper_bound_handle)
				upper_bound_handle = current_upper
			end
		end
		if isfinite(lb[control_idx])
			current_lower = CairoMakie.lines!(
				ax,
				horizon_steps,
				fill(lb[control_idx], control_horizon);
				color = :black,
				linestyle = :dot,
				linewidth = 2,
			)
			if isnothing(lower_bound_handle)
				lower_bound_handle = current_lower
			end
		end
	end

	if length(axes) > 1
		CairoMakie.linkyaxes!(axes...)
	end

	legend_elements = Any[player1_handle, player2_handle]
	legend_labels = ["Player 1", "Player 2"]
	if !isnothing(upper_bound_handle)
		push!(legend_elements, upper_bound_handle)
		push!(legend_labels, "Upper Control Bound")
	end
	if !isnothing(lower_bound_handle)
		push!(legend_elements, lower_bound_handle)
		push!(legend_labels, "Lower Control Bound")
	end

	CairoMakie.Legend(
		figure[1, control_dimension],
		legend_elements,
		legend_labels;
		framevisible = false,
		labelsize = legend_label_fontsize,
		orientation = :vertical,
		tellheight = false,
		tellwidth = false,
		halign = :right,
		valign = :top,
	)

	return figure, Tuple(axes)
end


function plot_convergence_plot(;
	kkt_error_history,
	total_iters = nothing,
	show_ylabel = true,
)
	axis_label_fontsize = 30
	tick_label_fontsize = 22
	ylabel_text = show_ylabel ? L"$\log(|| \mathcal{K}_{\rho}^{(\ell)} ||_\infty)$" : ""
	figure = serif_figure()
	ax = CairoMakie.Axis(
		figure[1, 1];
		xlabel = "Newton iteration",
		ylabel = ylabel_text,
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
	)

	iteration_axis = collect(1:length(kkt_error_history))
	if !isnothing(total_iters) && total_iters > 0 && total_iters < length(iteration_axis)
		iteration_axis = iteration_axis[1:total_iters]
		kkt_error_history = kkt_error_history[1:total_iters]
	end
	CairoMakie.lines!(ax, iteration_axis, kkt_error_history, color = :dodgerblue, linewidth = 4)

	return figure, ax
end

function plot_condition_number_plot(;
	condition_number_history,
	total_iters = nothing,
	show_ylabel = true,
)
	axis_label_fontsize = 30
	tick_label_fontsize = 22
	ylabel_text = show_ylabel ? L"$\log(\kappa(\nabla F))$" : ""
	figure = serif_figure()
	ax = CairoMakie.Axis(
		figure[1, 1];
		xlabel = "Newton iteration",
		ylabel = ylabel_text,
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
	)

	iteration_axis = collect(1:length(condition_number_history))
	if !isnothing(total_iters) && total_iters > 0 && total_iters < length(iteration_axis)
		iteration_axis = iteration_axis[1:total_iters]
		condition_number_history = condition_number_history[1:total_iters]
	end
	CairoMakie.lines!(ax, iteration_axis, condition_number_history, color = :darkorange, linewidth = 4)

	return figure, ax
end

function plot_eta_plot(;
	eta_history,
	total_iters = nothing,
	show_ylabel = true,
)
	axis_label_fontsize = 30
	tick_label_fontsize = 22
	ylabel_text = show_ylabel ? L"$\log(\eta)$" : ""
	figure = serif_figure()
	ax = CairoMakie.Axis(
		figure[1, 1];
		xlabel = "Newton iteration",
		ylabel = ylabel_text,
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
	)

	iteration_axis = collect(1:length(eta_history))
	if !isnothing(total_iters) && total_iters > 0 && total_iters < length(iteration_axis)
		iteration_axis = iteration_axis[1:total_iters]
		eta_history = eta_history[1:total_iters]
	end
	CairoMakie.lines!(ax, iteration_axis, eta_history, color = :seagreen, linewidth = 4)

	return figure, ax
end

function plot_convergence_plot_aggregate(; kkt_error_histories, show_ylabel = true)
	if isempty(kkt_error_histories)
		error("kkt_error_histories must be non-empty.")
	end
	axis_label_fontsize = 30
	tick_label_fontsize = 22
	ylabel_text = show_ylabel ? L"$\log(|| \mathcal{K}_{\rho}^{(\ell)} ||_2)$" : ""

	max_trace_length = maximum(length, kkt_error_histories)
	num_instances = length(kkt_error_histories)
	mean_kkt_error = fill(NaN, max_trace_length)
	std_kkt_error = fill(NaN, max_trace_length)

	for iteration_idx in 1:max_trace_length
		values_at_iteration = Float64[]
		for instance_idx in 1:num_instances
			trace = kkt_error_histories[instance_idx]
			if iteration_idx <= length(trace)
				push!(values_at_iteration, trace[iteration_idx])
			end
		end
		if !isempty(values_at_iteration)
			mean_value = sum(values_at_iteration) / length(values_at_iteration)
			variance = sum((v - mean_value)^2 for v in values_at_iteration) / length(values_at_iteration)
			mean_kkt_error[iteration_idx] = mean_value
			std_kkt_error[iteration_idx] = sqrt(variance)
		end
	end

	valid_indices = findall(!isnan, mean_kkt_error)
	x = collect(valid_indices)
	y_mean = mean_kkt_error[valid_indices]
	y_lower = y_mean .- std_kkt_error[valid_indices]
	y_upper = y_mean .+ std_kkt_error[valid_indices]

	figure = serif_figure()
	ax = CairoMakie.Axis(
		figure[1, 1];
		xlabel = L"\text{iteration} ~\ell",
		ylabel = ylabel_text,
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
		yticks = -10:4,
		xticks = 0:10:maximum(x),
	)

	CairoMakie.band!(ax, x, y_lower, y_upper; color = (:dodgerblue, 0.25))
	CairoMakie.lines!(ax, x, y_mean; color = :dodgerblue, linewidth = 3)

	return figure, ax
end

function plot_convergence_plot_aggregate_comparison(;
	reduced_kkt_error_histories,
	complete_kkt_error_histories,
	show_legend = true,
	show_ylabel = true,
)
	axis_label_fontsize = 30
	legend_label_fontsize = 26
	tick_label_fontsize = 22
	ylabel_text = show_ylabel ? L"$\log(|| \mathcal{K}_{\rho}^{(\ell)} ||_2)$" : ""
	if isempty(reduced_kkt_error_histories) || isempty(complete_kkt_error_histories)
		error("Both reduced_kkt_error_histories and complete_kkt_error_histories must be non-empty.")
	end

	function aggregate_trace_stats(kkt_error_histories)
		max_trace_length = maximum(length, kkt_error_histories)
		num_instances = length(kkt_error_histories)
		mean_kkt_error = fill(NaN, max_trace_length)
		std_kkt_error = fill(NaN, max_trace_length)

		for iteration_idx in 1:max_trace_length
			values_at_iteration = Float64[]
			for instance_idx in 1:num_instances
				trace = kkt_error_histories[instance_idx]
				if iteration_idx <= length(trace)
					push!(values_at_iteration, trace[iteration_idx])
				end
			end
			if !isempty(values_at_iteration)
				mean_value = sum(values_at_iteration) / length(values_at_iteration)
				variance = sum((v - mean_value)^2 for v in values_at_iteration) / length(values_at_iteration)
				mean_kkt_error[iteration_idx] = mean_value
				std_kkt_error[iteration_idx] = sqrt(variance)
			end
		end

		valid_indices = findall(!isnan, mean_kkt_error)
		x = collect(valid_indices)
		y_mean = mean_kkt_error[valid_indices]
		y_lower = y_mean .- std_kkt_error[valid_indices]
		y_upper = y_mean .+ std_kkt_error[valid_indices]
		(; x, y_mean, y_lower, y_upper)
	end

	reduced_stats = aggregate_trace_stats(reduced_kkt_error_histories)
	complete_stats = aggregate_trace_stats(complete_kkt_error_histories)
	max_x = max(
		isempty(reduced_stats.x) ? 0 : maximum(reduced_stats.x),
		isempty(complete_stats.x) ? 0 : maximum(complete_stats.x),
	)

	x_lim = 20 # max iter for plot
	figure = serif_figure()
	ax = CairoMakie.Axis(
		figure[1, 1];
		xlabel = L"\text{iteration} ~\ell",
		ylabel = ylabel_text,
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
		# yticks = -10:4,
		# xticks = 0:5:x_lim, # 0:5:max_x,
		# yscale = log10,
		# limits = ((0, x_lim), nothing),
	)

	CairoMakie.band!(ax, reduced_stats.x, reduced_stats.y_lower, reduced_stats.y_upper; color = (:dodgerblue, 0.2))
	CairoMakie.lines!(ax, reduced_stats.x, reduced_stats.y_mean; color = :dodgerblue, linewidth = 3, label = "Reduced")

	CairoMakie.band!(ax, complete_stats.x, complete_stats.y_lower, complete_stats.y_upper; color = (:crimson, 0.2))
	CairoMakie.lines!(ax, complete_stats.x, complete_stats.y_mean; color = :crimson, linewidth = 3, label = "Complete")

	if show_legend
		CairoMakie.axislegend(ax; position = :rt, labelsize = legend_label_fontsize)
	end
	return figure, ax
end
