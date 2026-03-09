# using TrajectoryGamesBase: OpenLoopStrategy
using CairoMakie: CairoMakie
# using LaTeXStrings: @L_str

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
	figure = CairoMakie.Figure()
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

	strategy1 = CairoMakie.@lift OpenLoopStrategy($strategy[1].xs, $strategy[1].us)
	strategy2 = CairoMakie.@lift OpenLoopStrategy($strategy[2].xs, $strategy[2].us)
	CairoMakie.plot!(ax, strategy1, color = :blue)
	CairoMakie.plot!(ax, strategy2, color = :red)

	CairoMakie.scatter!(
		ax,
		CairoMakie.@lift([CairoMakie.Point2f($θ1[1:2]), CairoMakie.Point2f($θ2[1:2])]),
		markersize = 20,
		color = [:blue, :red],
	)
	CairoMakie.scatter!(
		ax,
		CairoMakie.@lift(CairoMakie.Point2f($goal_position1)),
		markersize = 20,
		marker = :star5,
		color = :blue,
	)
	CairoMakie.scatter!(
		ax,
		CairoMakie.@lift(CairoMakie.Point2f($goal_position2)),
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
	show_ylabel = true,
)
	axis_label_fontsize = 30
	tick_label_fontsize = 22
	ylabel_text = show_ylabel ? L"$\log(|| \mathcal{K}_{\rho}^{(\ell)} ||_\infty)$" : ""
	figure = CairoMakie.Figure()
	ax = CairoMakie.Axis(
		figure[1, 1];
		xlabel = L"\text{iteration} ~\ell",
		ylabel = ylabel_text,
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
		yscale = log10,
	)

	safe_kkt_error = max.(kkt_error_history, eps(Float64))
	CairoMakie.lines!(ax, total_iteration_history, safe_kkt_error, color = :dodgerblue, linewidth = 2)
	CairoMakie.scatter!(ax, total_iteration_history, safe_kkt_error, color = :dodgerblue, markersize = 6)

	if !isempty(outer_end_total_iterations)
		CairoMakie.vlines!(
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
		CairoMakie.scatter!(
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

	figure = CairoMakie.Figure()
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
	figure = CairoMakie.Figure()
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
