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


