# using TrajectoryGamesBase: OpenLoopStrategy
# using GLMakie: GLMakie
# using LaTeXStrings: @L_str

function draw_intersection_map!(ax; map_end, lane_width, offset = 0.2)
	vertical_road_background = GLMakie.Polygon(
		GLMakie.Point2f[
			(-lane_width - offset, -map_end),
			(lane_width + offset, -map_end),
			(lane_width + offset, map_end),
			(-lane_width - offset, map_end),
		],
	)
	GLMakie.poly!(ax, vertical_road_background, color = :white)
	GLMakie.lines!(
		ax,
		[-lane_width - offset, -lane_width - offset],
		[-map_end, -lane_width],
		color = :black,
		linewidth = 1,
	)
	GLMakie.lines!(
		ax,
		[-lane_width - offset, -lane_width - offset],
		[map_end, lane_width],
		color = :black,
		linewidth = 1,
	)
	GLMakie.lines!(
		ax,
		[lane_width + offset, lane_width + offset],
		[-map_end, -lane_width],
		color = :black,
		linewidth = 1,
	)
	GLMakie.lines!(
		ax,
		[lane_width + offset, lane_width + offset],
		[map_end, lane_width],
		color = :black,
		linewidth = 1,
	)

	horizontal_road_background = GLMakie.Polygon(
		GLMakie.Point2f[
			(-map_end, -lane_width - offset),
			(map_end, -lane_width - offset),
			(map_end, lane_width + offset),
			(-map_end, lane_width + offset),
		],
	)
	GLMakie.poly!(ax, horizontal_road_background, color = :white)
	GLMakie.lines!(
		ax,
		[-lane_width - offset, -map_end],
		[-lane_width - offset, -lane_width - offset],
		color = :black,
		linewidth = 1,
	)
	GLMakie.lines!(
		ax,
		[-lane_width - offset, -map_end],
		[lane_width + offset, lane_width + offset],
		color = :black,
		linewidth = 1,
	)
	GLMakie.lines!(
		ax,
		[lane_width + offset, map_end],
		[lane_width + offset, lane_width + offset],
		color = :black,
		linewidth = 1,
	)
	GLMakie.lines!(
		ax,
		[lane_width + offset, map_end],
		[-lane_width - offset, -lane_width - offset],
		color = :black,
		linewidth = 1,
	)

	vertical_road = GLMakie.Polygon(
		GLMakie.Point2f[
			(-lane_width, -map_end),
			(lane_width, -map_end),
			(lane_width, map_end),
			(-lane_width, map_end),
		],
	)
	GLMakie.poly!(ax, vertical_road, color = :gray)

	horizontal_road = GLMakie.Polygon(
		GLMakie.Point2f[
			(-map_end, -lane_width),
			(map_end, -lane_width),
			(map_end, lane_width),
			(-map_end, lane_width),
		],
	)
	GLMakie.poly!(ax, horizontal_road, color = :gray)

	# Lane center lines
	GLMakie.lines!(ax, [-lane_width, -map_end], [0, 0], color = :yellow, linewidth = 2)
	GLMakie.lines!(ax, [lane_width, map_end], [0, 0], color = :yellow, linewidth = 2)
	GLMakie.lines!(ax, [0, 0], [-lane_width, -map_end], color = :yellow, linewidth = 2)
	GLMakie.lines!(ax, [0, 0], [lane_width, map_end], color = :yellow, linewidth = 2)

	# Direction arrows
	xs = [-3.0, 3.0, 1.0, -1.0]
	ys = [-1.0, 1.0, -3.0, 3.0]
	us = [1.0, -1.0, 0.0, 0.0]
	vs = [0.0, 0.0, 1.0, -1.0]
	GLMakie.arrows!(
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
	figure = GLMakie.Figure()
	ax = GLMakie.Axis(
		figure[1, 1];
		aspect = 1,
		xgridvisible = false,
		ygridvisible = false,
		backgroundcolor = :lightgreen,
	)
	GLMakie.hidedecorations!(ax)
	GLMakie.hidespines!(ax)

	draw_intersection_map!(ax; map_end, lane_width)

	strategy1 = GLMakie.@lift OpenLoopStrategy($strategy[1].xs, $strategy[1].us)
	strategy2 = GLMakie.@lift OpenLoopStrategy($strategy[2].xs, $strategy[2].us)
	GLMakie.plot!(ax, strategy1, color = :blue)
	GLMakie.plot!(ax, strategy2, color = :red)

	GLMakie.scatter!(
		ax,
		GLMakie.@lift([GLMakie.Point2f($θ1[1:2]), GLMakie.Point2f($θ2[1:2])]),
		markersize = 20,
		color = [:blue, :red],
	)
	GLMakie.scatter!(
		ax,
		GLMakie.@lift(GLMakie.Point2f($goal_position1)),
		markersize = 20,
		marker = :star5,
		color = :blue,
	)
	GLMakie.scatter!(
		ax,
		GLMakie.@lift(GLMakie.Point2f($goal_position2)),
		markersize = 20,
		marker = :star5,
		color = :red,
	)

	return figure, ax
end


function plot_convergence_plot(;
	kkt_error_history,
	total_iteration_history,
	outer_end_total_iterations = Int[],
	outer_end_trace_indices = Int[],
)
	figure = GLMakie.Figure()
	ax = GLMakie.Axis(
		figure[1, 1];
		xlabel = L"\text{iteration} ~\ell",
		ylabel = L"$\log(|| \mathcal{K}_{\rho}^{(\ell)} ||_\infty)$",
		yscale = log10,
	)

	safe_kkt_error = max.(kkt_error_history, eps(Float64))
	GLMakie.lines!(ax, total_iteration_history, safe_kkt_error, color = :dodgerblue, linewidth = 2)
	GLMakie.scatter!(ax, total_iteration_history, safe_kkt_error, color = :dodgerblue, markersize = 6)

	if !isempty(outer_end_total_iterations)
		GLMakie.vlines!(
			ax,
			outer_end_total_iterations;
			color = (:gray, 0.4),
			linestyle = :dash,
			linewidth = 1,
		)
	end

	if !isempty(outer_end_trace_indices)
		end_x = total_iteration_history[outer_end_trace_indices]
		end_y = safe_kkt_error[outer_end_trace_indices]
		GLMakie.scatter!(
			ax,
			end_x,
			end_y;
			color = :crimson,
			markersize = 12,
			marker = :utriangle,
		)
	end

	return figure, ax
end

function plot_convergence_plot_aggregate(; log_kkt_error_histories)
	if isempty(log_kkt_error_histories)
		error("log_kkt_error_histories must be non-empty.")
	end

	max_trace_length = maximum(length, log_kkt_error_histories)
	num_instances = length(log_kkt_error_histories)
	mean_log_kkt_error = fill(NaN, max_trace_length)
	std_log_kkt_error = fill(NaN, max_trace_length)

	for iteration_idx in 1:max_trace_length
		values_at_iteration = Float64[]
		for instance_idx in 1:num_instances
			trace = log_kkt_error_histories[instance_idx]
			if iteration_idx <= length(trace)
				push!(values_at_iteration, trace[iteration_idx])
			end
		end
		if !isempty(values_at_iteration)
			mean_value = sum(values_at_iteration) / length(values_at_iteration)
			variance = sum((v - mean_value)^2 for v in values_at_iteration) / length(values_at_iteration)
			mean_log_kkt_error[iteration_idx] = mean_value
			std_log_kkt_error[iteration_idx] = sqrt(variance)
		end
	end

	valid_indices = findall(!isnan, mean_log_kkt_error)
	x = collect(valid_indices)
	y_mean = mean_log_kkt_error[valid_indices]
	y_lower = y_mean .- std_log_kkt_error[valid_indices]
	y_upper = y_mean .+ std_log_kkt_error[valid_indices]

	figure = GLMakie.Figure()
	ax = GLMakie.Axis(
		figure[1, 1];
		xlabel = L"\text{iteration} ~\ell",
		ylabel = L"$\log(|| \mathcal{K}_{\rho}^{(\ell)} ||_2$",
	)

	GLMakie.band!(ax, x, y_lower, y_upper; color = (:dodgerblue, 0.25))
	GLMakie.lines!(ax, x, y_mean; color = :dodgerblue, linewidth = 3)
	set_unit_interval_yticks!(ax, y_lower, y_upper)

	return figure, ax
end

function set_unit_interval_yticks!(ax, ys...)
	all_values = reduce(vcat, ys)
	if isempty(all_values)
		return
	end
	ymin = floor(minimum(all_values))
	ymax = ceil(maximum(all_values))
	ytick_values = collect(ymin:1.0:ymax)
	ytick_labels = [string(round(y; digits = 1)) for y in ytick_values]
	ax.yticks = (ytick_values, ytick_labels)
end
