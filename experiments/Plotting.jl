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
		# yticks = -15:3,
		xticks = 0:5:10, # 0:5:x_lim, 0:5:max_x,
		# yscale = log10,
		limits = ((0, 15), nothing),
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

"""
Compute local convergence order values from a log-error trace:
`p_k = (log(e_{k+1}/e_k)) / (log(e_k/e_{k-1}))`.

When `log_kkt_error_history = log(e_k)` (any log base), this becomes
`p_k = (l_{k+1} - l_k) / (l_k - l_{k-1})`.
"""
function local_order_sequence_from_log(log_kkt_error_history; min_abs_denominator = 1e-12)
	iterations = Int[]
	p_values = Float64[]
	if length(log_kkt_error_history) < 3
		return iterations, p_values
	end

	for k in 2:(length(log_kkt_error_history) - 1)
		l_prev = log_kkt_error_history[k - 1]
		l_curr = log_kkt_error_history[k]
		l_next = log_kkt_error_history[k + 1]
		if !isfinite(l_prev) || !isfinite(l_curr) || !isfinite(l_next)
			continue
		end

		denominator = l_curr - l_prev
		if abs(denominator) <= min_abs_denominator
			continue
		end

		p_k = (l_next - l_curr) / denominator
		isfinite(p_k) || continue
		push!(iterations, k)
		push!(p_values, p_k)
	end
	return iterations, p_values
end

"""
Aggregate local-order traces (`p_k`) across multiple log-error histories.

Returns per-iteration mean/std/count and a tail mean `p_hat` (tail_points = 3).
"""
function aggregate_local_order_stats(log_kkt_error_histories; min_abs_denominator = 1e-12, tail_points = 3)
	p_by_iteration = Dict{Int, Vector{Float64}}()
	per_instance_p = Vector{Vector{Float64}}()

	for log_history in log_kkt_error_histories
		iterations, p_values = local_order_sequence_from_log(log_history; min_abs_denominator)
		if !isempty(p_values)
			push!(per_instance_p, p_values)
			for (k, p_k) in zip(iterations, p_values)
				if haskey(p_by_iteration, k)
					push!(p_by_iteration[k], p_k)
				else
					p_by_iteration[k] = Float64[p_k]
				end
			end
		end
	end

	if isempty(p_by_iteration)
		return (
			x = Int[],
			y_mean = Float64[],
			y_std = Float64[],
			y_lower = Float64[],
			y_upper = Float64[],
			counts = Int[],
			p_hat = NaN,
			per_instance_p = per_instance_p,
		)
	end

	x = sort(collect(keys(p_by_iteration)))
	y_mean = Float64[]
	y_std = Float64[]
	counts = Int[]
	for k in x
		values = p_by_iteration[k]
		μ = sum(values) / length(values)
		σ = sqrt(sum((v - μ)^2 for v in values) / length(values))
		push!(y_mean, μ)
		push!(y_std, σ)
		push!(counts, length(values))
	end
	y_lower = y_mean .- y_std
	y_upper = y_mean .+ y_std

	tail_len = min(tail_points, length(y_mean))
	p_hat = tail_len > 0 ? sum(y_mean[(end - tail_len + 1):end]) / tail_len : NaN
	return (; x, y_mean, y_std, y_lower, y_upper, counts, p_hat, per_instance_p)
end

"""
Plot aggregate local-order traces `p_k` for reduced and complete KKT histories.
"""
function plot_local_order_aggregate_comparison(;
	reduced_log_kkt_error_histories,
	complete_log_kkt_error_histories,
	show_legend = true,
	show_ylabel = true,
	min_abs_denominator = 1e-12,
	tail_points = 5,
)
	reduced_stats = aggregate_local_order_stats(
		reduced_log_kkt_error_histories;
		min_abs_denominator,
		tail_points,
	)
	complete_stats = aggregate_local_order_stats(
		complete_log_kkt_error_histories;
		min_abs_denominator,
		tail_points,
	)
	if isempty(reduced_stats.x) && isempty(complete_stats.x)
		error("No valid local-order points were computed from the provided log-error histories.")
	end

	axis_label_fontsize = 28
	tick_label_fontsize = 20
	legend_label_fontsize = 18
	ylabel_text = show_ylabel ? "local order p_k" : ""

	max_x = max(
		isempty(reduced_stats.x) ? 0 : maximum(reduced_stats.x),
		isempty(complete_stats.x) ? 0 : maximum(complete_stats.x),
	)

	figure = CairoMakie.Figure()
	ax = CairoMakie.Axis(
		figure[1, 1];
		xlabel = "iteration k",
		ylabel = ylabel_text,
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
		xticks = 0:5:max_x,
		# yticks = 0:1:3,
		limits = ((2,15), (0, 10))
	)

	# Reference for quadratic convergence.
	CairoMakie.hlines!(ax, [2.0]; color = :black, linestyle = :dot, linewidth = 2, label = "Quadratic target (p = 2)")

	if !isempty(reduced_stats.x)
		CairoMakie.lines!(ax, reduced_stats.x, reduced_stats.y_mean; color = :crimson, linewidth = 3, label = "Reduced mean p_k")
		if isfinite(reduced_stats.p_hat)
			CairoMakie.hlines!(ax, [reduced_stats.p_hat]; color = :crimson, linestyle = :dash, linewidth = 2, label = "Reduced tail mean p_hat")
		end
	end

	if !isempty(complete_stats.x)
		CairoMakie.lines!(ax, complete_stats.x, complete_stats.y_mean; color = :dodgerblue, linewidth = 3, label = "Complete mean p_k")
		if isfinite(complete_stats.p_hat)
			CairoMakie.hlines!(ax, [complete_stats.p_hat]; color = :dodgerblue, linestyle = :dash, linewidth = 2, label = "Complete tail mean p_hat")
		end
	end

	if show_legend
		CairoMakie.axislegend(ax; position = :rt, labelsize = legend_label_fontsize)
	end

	return figure, ax, reduced_stats, complete_stats
end
