using CairoMakie: CairoMakie

const SINGLE_INTEGRATOR_3D_PLAYER_COLORS = (:blue, :red)
const GLMAKIE_PKGID = Base.PkgId(Base.UUID("e9467ef8-e4e7-5192-8a1a-b1aee30e663a"), "GLMakie")

function _load_glmakie()
	Base.require(GLMAKIE_PKGID)
end

function _trajectory_xyz(xs; player)
	isempty(xs) && error("Player $(player) trajectory is empty.")
	any(length(x) < 3 for x in xs) &&
		error("Player $(player) trajectory must contain at least 3D states.")

	(
		[x[1] for x in xs],
		[x[2] for x in xs],
		[x[3] for x in xs],
	)
end

function _position3(position, label)
	isnothing(position) && error("Missing $(label).")
	length(position) >= 3 || error("$(label) must have at least three entries.")
	collect(position[1:3])
end

function _initial_position_from_parameter_block(parameter_block, label)
	_position3(parameter_block, label)
end

function _padded_limits(values; lower = nothing, upper = nothing, margin = 0.75)
	min_value = minimum(values)
	max_value = maximum(values)
	if !isnothing(lower)
		min_value = min(min_value, lower)
	end
	if !isnothing(upper)
		max_value = max(max_value, upper)
	end

	width = max(max_value - min_value, 1.0)
	padding = margin + 0.05 * width
	(min_value - padding, max_value + padding)
end

function _grid_values(lims, spacing)
	spacing > 0 || error("Grid spacing must be positive.")
	first_value = ceil(lims[1] / spacing) * spacing
	last_value = floor(lims[2] / spacing) * spacing
	values = collect(first_value:spacing:last_value)
	isempty(values) ? collect(lims) : values
end

function _serif_figure(makie; kwargs...)
	return makie.Figure(;
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

function _single_integrator_3d_plot_data(;
	strategy,
	θ1 = nothing,
	θ2 = nothing,
	initial_position1 = nothing,
	initial_position2 = nothing,
	goal_position1,
	goal_position2,
	map_end = nothing,
	workspace_margin = 0.75,
)
	length(strategy) >= 2 ||
		error("3D trajectory plotting expects strategies for at least two players.")

	xs1 = strategy[1].xs
	xs2 = strategy[2].xs
	x1, y1, z1 = _trajectory_xyz(xs1; player = 1)
	x2, y2, z2 = _trajectory_xyz(xs2; player = 2)

	start1 = isnothing(initial_position1) ?
		_initial_position_from_parameter_block(θ1, "θ1") :
		_position3(initial_position1, "initial_position1")
	start2 = isnothing(initial_position2) ?
		_initial_position_from_parameter_block(θ2, "θ2") :
		_position3(initial_position2, "initial_position2")
	goal1 = _position3(goal_position1, "goal_position1")
	goal2 = _position3(goal_position2, "goal_position2")

	all_x = vcat(x1, x2, start1[1], start2[1], goal1[1], goal2[1])
	all_y = vcat(y1, y2, start1[2], start2[2], goal1[2], goal2[2])
	all_z = vcat(z1, z2, start1[3], start2[3], goal1[3], goal2[3])

	xlims = isnothing(map_end) ?
		_padded_limits(all_x; margin = workspace_margin) :
		_padded_limits(all_x; lower = -map_end, upper = map_end, margin = workspace_margin)
	ylims = isnothing(map_end) ?
		_padded_limits(all_y; margin = workspace_margin) :
		_padded_limits(all_y; lower = -map_end, upper = map_end, margin = workspace_margin)
	zlims = _padded_limits(all_z; lower = 0.0, margin = workspace_margin)

	(;
		x1,
		y1,
		z1,
		x2,
		y2,
		z2,
		start1,
		start2,
		goal1,
		goal2,
		xlims,
		ylims,
		zlims,
	)
end

function draw_single_integrator_3d_environment!(
	makie,
	ax;
	xlims,
	ylims,
	zlims,
	grid_spacing = 2.0,
)
	floor_z = clamp(0.0, zlims[1], zlims[2])
	grid_color = (:gray70, 0.45)
	box_color = (:gray45, 0.55)

	for x in _grid_values(xlims, grid_spacing)
		makie.lines!(
			ax,
			[x, x],
			[ylims[1], ylims[2]],
			[floor_z, floor_z];
			color = grid_color,
			linewidth = 0.7,
		)
	end
	for y in _grid_values(ylims, grid_spacing)
		makie.lines!(
			ax,
			[xlims[1], xlims[2]],
			[y, y],
			[floor_z, floor_z];
			color = grid_color,
			linewidth = 0.7,
		)
	end

	corners = (
		(xlims[1], ylims[1]),
		(xlims[2], ylims[1]),
		(xlims[2], ylims[2]),
		(xlims[1], ylims[2]),
	)
	for z in (floor_z, zlims[2])
		for idx in eachindex(corners)
			x1, y1 = corners[idx]
			x2, y2 = corners[mod1(idx + 1, length(corners))]
			makie.lines!(
				ax,
				[x1, x2],
				[y1, y2],
				[z, z];
				color = box_color,
				linewidth = 1,
			)
		end
	end
	for (x, y) in corners
		makie.lines!(
			ax,
			[x, x],
			[y, y],
			[floor_z, zlims[2]];
			color = box_color,
			linewidth = 1,
		)
	end

	return ax
end

function draw_single_integrator_3d_environment!(
	ax;
	xlims,
	ylims,
	zlims,
	grid_spacing = 2.0,
)
	draw_single_integrator_3d_environment!(
		CairoMakie,
		ax;
		xlims,
		ylims,
		zlims,
		grid_spacing,
	)
end

function _plot_single_integrator_3d_trajectories(
	makie;
	strategy,
	θ1 = nothing,
	θ2 = nothing,
	initial_position1 = nothing,
	initial_position2 = nothing,
	goal_position1,
	goal_position2,
	map_end = nothing,
	lane_width = nothing,
	workspace_margin = 0.75,
	figure_size = (900, 720),
	axis_label_fontsize = 24,
	tick_label_fontsize = 18,
	legend_label_fontsize = 16,
	interactive = false,
)
	plot_data = _single_integrator_3d_plot_data(
		;
		strategy,
		θ1,
		θ2,
		initial_position1,
		initial_position2,
		goal_position1,
		goal_position2,
		map_end,
		workspace_margin,
	)
	(;
		x1,
		y1,
		z1,
		x2,
		y2,
		z2,
		start1,
		start2,
		goal1,
		goal2,
		xlims,
		ylims,
		zlims,
	) = plot_data

	figure = _serif_figure(makie; size = figure_size)
	ax = makie.Axis3(
		figure[1, 1];
		xlabel = "x [m]",
		ylabel = "y [m]",
		zlabel = "z [m]",
		xlabelsize = axis_label_fontsize,
		ylabelsize = axis_label_fontsize,
		zlabelsize = axis_label_fontsize,
		xticklabelsize = tick_label_fontsize,
		yticklabelsize = tick_label_fontsize,
		zticklabelsize = tick_label_fontsize,
		aspect = :data,
		azimuth = 0.72pi,
		elevation = 0.18pi,
		perspectiveness = interactive ? 0.55 : 0.35,
		xgridvisible = false,
		ygridvisible = false,
		zgridvisible = false,
		backgroundcolor = :white,
	)
	makie.xlims!(ax, xlims...)
	makie.ylims!(ax, ylims...)
	makie.zlims!(ax, zlims...)

	grid_spacing = isnothing(lane_width) ? 2.0 : max(float(lane_width), 1.0)
	draw_single_integrator_3d_environment!(
		makie,
		ax;
		xlims,
		ylims,
		zlims,
		grid_spacing,
	)

	player1_color, player2_color = SINGLE_INTEGRATOR_3D_PLAYER_COLORS
	player1_path = makie.lines!(
		ax,
		x1,
		y1,
		z1;
		color = player1_color,
		linewidth = interactive ? 4 : 3,
	)
	player2_path = makie.lines!(
		ax,
		x2,
		y2,
		z2;
		color = player2_color,
		linewidth = interactive ? 4 : 3,
	)

	makie.scatter!(ax, x1, y1, z1; color = player1_color, markersize = interactive ? 10 : 7)
	makie.scatter!(ax, x2, y2, z2; color = player2_color, markersize = interactive ? 10 : 7)

	start1_marker = makie.scatter!(
		ax,
		[start1[1]],
		[start1[2]],
		[start1[3]];
		markersize = interactive ? 24 : 20,
		color = player1_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	start2_marker = makie.scatter!(
		ax,
		[start2[1]],
		[start2[2]],
		[start2[3]];
		markersize = interactive ? 24 : 20,
		color = player2_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	current1_marker = makie.scatter!(
		ax,
		[x1[end]],
		[y1[end]],
		[z1[end]];
		marker = :diamond,
		markersize = interactive ? 30 : 24,
		color = player1_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	current2_marker = makie.scatter!(
		ax,
		[x2[end]],
		[y2[end]],
		[z2[end]];
		marker = :diamond,
		markersize = interactive ? 30 : 24,
		color = player2_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	goal1_marker = makie.scatter!(
		ax,
		[goal1[1]],
		[goal1[2]],
		[goal1[3]];
		marker = :star5,
		markersize = interactive ? 32 : 26,
		color = player1_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	goal2_marker = makie.scatter!(
		ax,
		[goal2[1]],
		[goal2[2]],
		[goal2[3]];
		marker = :star5,
		markersize = interactive ? 32 : 26,
		color = player2_color,
		strokecolor = :black,
		strokewidth = 1,
	)

	makie.Legend(
		figure[1, 1],
		[
			player1_path,
			player2_path,
			start1_marker,
			start2_marker,
			current1_marker,
			current2_marker,
			goal1_marker,
			goal2_marker,
		],
		[
			"Player 1",
			"Player 2",
			"Start 1",
			"Start 2",
			"Current 1",
			"Current 2",
			"Goal 1",
			"Goal 2",
		];
		framevisible = false,
		labelsize = legend_label_fontsize,
		orientation = :vertical,
		tellheight = false,
		tellwidth = false,
		halign = :left,
		valign = :top,
		margin = (16, 12, 12, 6),
	)

	return figure, ax
end

function plot_single_integrator_3d_trajectories(; kwargs...)
	CairoMakie.activate!()
	_plot_single_integrator_3d_trajectories(
		CairoMakie;
		kwargs...,
		interactive = false,
	)
end

function plot_trajectory_3d_interactive(;
	display_figure = true,
	figure_size = (1100, 850),
	kwargs...,
)
	GLMakie = _load_glmakie()
	Base.invokelatest(GLMakie.activate!)
	figure, ax = _plot_single_integrator_3d_trajectories(
		GLMakie;
		kwargs...,
		figure_size,
		axis_label_fontsize = 24,
		tick_label_fontsize = 18,
		legend_label_fontsize = 16,
		interactive = true,
	)
	display_figure && Base.invokelatest(display, figure)
	return figure, ax
end

function plot_single_integrator_3d_trajectories_interactive(; kwargs...)
	plot_trajectory_3d_interactive(; kwargs...)
end
