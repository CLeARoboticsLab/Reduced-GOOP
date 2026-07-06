using CairoMakie: CairoMakie

const SINGLE_INTEGRATOR_3D_PLAYER_COLORS = (:blue, :red, :darkorange)
const WGLMAKIE_PKGID = Base.PkgId(Base.UUID("276b4fcb-3e11-5398-bf8b-a0c2d153d008"), "WGLMakie")

function _load_wglmakie()
	Base.require(WGLMAKIE_PKGID)
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

function _horizontal_circle(center, radius; point_count = 96)
	angles = range(0, 2pi; length = point_count)
	xs = [center[1] + radius * cos(angle) for angle in angles]
	ys = [center[2] + radius * sin(angle) for angle in angles]
	zs = fill(center[3], point_count)
	(xs, ys, zs)
end

function _hemisphere_wireframe(center, radius; point_count = 56)
	segments = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}[]

	for elevation in (0.0, pi / 6, pi / 3)
		ring_radius = radius * cos(elevation)
		ring_center = [center[1], center[2], center[3] + radius * sin(elevation)]
		push!(segments, _horizontal_circle(ring_center, ring_radius; point_count))
	end

	arc_angles = range(0, pi; length = point_count)
	for azimuth in (0.0, pi / 4, pi / 2, 3pi / 4)
		xs = [center[1] + radius * cos(angle) * cos(azimuth) for angle in arc_angles]
		ys = [center[2] + radius * cos(angle) * sin(azimuth) for angle in arc_angles]
		zs = [center[3] + radius * sin(angle) for angle in arc_angles]
		push!(segments, (xs, ys, zs))
	end

	segments
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
	θ3 = nothing,
	initial_position1 = nothing,
	initial_position2 = nothing,
	initial_position3 = nothing,
	goal_position1,
	goal_position2,
	goal_position3 = nothing,
	collision_avoidance = nothing,
	map_end = nothing,
	workspace_margin = 0.75,
)
	length(strategy) >= 2 ||
		error("3D trajectory plotting expects strategies for at least two players.")

	xs1 = strategy[1].xs
	xs2 = strategy[2].xs
	x1, y1, z1 = _trajectory_xyz(xs1; player = 1)
	x2, y2, z2 = _trajectory_xyz(xs2; player = 2)
	has_child = length(strategy) >= 3
	x3, y3, z3 = has_child ? _trajectory_xyz(strategy[3].xs; player = 3) : (Float64[], Float64[], Float64[])

	start1 = isnothing(initial_position1) ?
		_initial_position_from_parameter_block(θ1, "θ1") :
		_position3(initial_position1, "initial_position1")
	start2 = isnothing(initial_position2) ?
		_initial_position_from_parameter_block(θ2, "θ2") :
		_position3(initial_position2, "initial_position2")
	start3 = if has_child
		if !isnothing(initial_position3)
			_position3(initial_position3, "initial_position3")
		elseif !isnothing(θ3)
			_initial_position_from_parameter_block(θ3, "θ3")
		else
			[x3[1], y3[1], z3[1]]
		end
	else
		nothing
	end
	goal1 = _position3(goal_position1, "goal_position1")
	goal2 = _position3(goal_position2, "goal_position2")

	all_x = vcat(x1, x2, start1[1], start2[1], goal1[1], goal2[1])
	all_y = vcat(y1, y2, start1[2], start2[2], goal1[2], goal2[2])
	all_z = vcat(z1, z2, start1[3], start2[3], goal1[3], goal2[3])
	if has_child
		append!(all_x, x3)
		append!(all_y, y3)
		append!(all_z, z3)
		push!(all_x, start3[1])
		push!(all_y, start3[2])
		push!(all_z, start3[3])
		if !isnothing(collision_avoidance)
			for k in eachindex(x3)
				append!(all_x, (x3[k] - collision_avoidance, x3[k] + collision_avoidance))
				append!(all_y, (y3[k] - collision_avoidance, y3[k] + collision_avoidance))
				append!(all_z, z3[k] + collision_avoidance)
			end
		end
	end

	xlims = isnothing(map_end) ?
		_padded_limits(all_x; margin = workspace_margin) :
		_padded_limits(all_x; lower = -map_end, upper = map_end, margin = workspace_margin)
	ylims = isnothing(map_end) ?
		_padded_limits(all_y; margin = workspace_margin) :
		_padded_limits(all_y; lower = -map_end, upper = map_end, margin = workspace_margin)
	zlims = _padded_limits(all_z; lower = 0.0, margin = workspace_margin)
	z_data_min = minimum(all_z)
	zlims = (z_data_min < 0.0 ? zlims[1] : 0.0, zlims[2])

	(;
		x1,
		y1,
		z1,
		x2,
		y2,
		z2,
		has_child,
		x3,
		y3,
		z3,
		start1,
		start2,
		start3,
		goal1,
		goal2,
		collision_avoidance,
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
	θ3 = nothing,
	initial_position1 = nothing,
	initial_position2 = nothing,
	initial_position3 = nothing,
	goal_position1,
	goal_position2,
	goal_position3 = nothing,
	collision_avoidance = nothing,
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
		θ3,
		initial_position1,
		initial_position2,
		initial_position3,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
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
		has_child,
		x3,
		y3,
		z3,
		start1,
		start2,
		start3,
		goal1,
		goal2,
		collision_avoidance,
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

	player1_color, player2_color, player3_color = SINGLE_INTEGRATOR_3D_PLAYER_COLORS
	safety_handle = nothing
	if has_child && !isnothing(collision_avoidance)
		safety_stride = max(1, cld(length(x3), 6))
		safety_indices = unique(vcat(collect(1:safety_stride:length(x3)), length(x3)))
		for t in safety_indices
			hemisphere_segments = _hemisphere_wireframe(
				[x3[t], y3[t], z3[t]],
				collision_avoidance,
			)
			for (segment_x, segment_y, segment_z) in hemisphere_segments
				current_safety = makie.lines!(
					ax,
					segment_x,
					segment_y,
					segment_z;
					color = (player3_color, 0.42),
					linestyle = :dash,
					linewidth = interactive ? 2.0 : 1.35,
				)
				if isnothing(safety_handle)
					safety_handle = current_safety
				end
			end
		end
	end

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
	player3_path = nothing
	if has_child
		player3_path = makie.lines!(
			ax,
			x3,
			y3,
			z3;
			color = player3_color,
			linewidth = interactive ? 4 : 3,
			linestyle = :dash,
		)
	end

	makie.scatter!(ax, x1, y1, z1; color = player1_color, markersize = interactive ? 10 : 7)
	makie.scatter!(ax, x2, y2, z2; color = player2_color, markersize = interactive ? 10 : 7)
	if has_child
		makie.scatter!(
			ax,
			x3,
			y3,
			z3;
			color = player3_color,
			marker = :utriangle,
			markersize = interactive ? 12 : 8,
		)
	end

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
	start3_marker = has_child ? makie.scatter!(
		ax,
		[start3[1]],
		[start3[2]],
		[start3[3]];
		marker = :utriangle,
		markersize = interactive ? 26 : 21,
		color = player3_color,
		strokecolor = :black,
		strokewidth = 1,
	) : nothing
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
	current3_marker = has_child ? makie.scatter!(
		ax,
		[x3[end]],
		[y3[end]],
		[z3[end]];
		marker = :diamond,
		markersize = interactive ? 30 : 24,
		color = player3_color,
		strokecolor = :black,
		strokewidth = 1,
	) : nothing
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

	legend_elements = Any[player1_path, player2_path]
	legend_labels = ["Left gripper (P1)", "Right gripper (P2)"]
	if has_child
		push!(legend_elements, player3_path)
		push!(legend_labels, "Child/Pet (P3)")
	end
	append!(
		legend_elements,
		Any[start1_marker, start2_marker, current1_marker, current2_marker, goal1_marker, goal2_marker],
	)
	append!(
		legend_labels,
		["Start 1", "Start 2", "Current 1", "Current 2", "Goal 1", "Goal 2"],
	)
	if has_child
		append!(legend_elements, Any[start3_marker, current3_marker])
		append!(legend_labels, ["Child/Pet Start", "Child/Pet Current"])
	end
	if !isnothing(safety_handle)
		push!(legend_elements, safety_handle)
		push!(legend_labels, "Safety hemisphere")
	end

	makie.Legend(
		figure[1, 1],
		legend_elements,
		legend_labels;
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

function _plot_single_integrator_3d_browser_trajectories(
	makie;
	strategy,
	θ1 = nothing,
	θ2 = nothing,
	θ3 = nothing,
	initial_position1 = nothing,
	initial_position2 = nothing,
	initial_position3 = nothing,
	goal_position1,
	goal_position2,
	goal_position3 = nothing,
	collision_avoidance = nothing,
	map_end = nothing,
	lane_width = nothing,
	workspace_margin = 0.75,
	figure_size = (1100, 850),
	legend_label_fontsize = 16,
)
	plot_data = _single_integrator_3d_plot_data(
		;
		strategy,
		θ1,
		θ2,
		θ3,
		initial_position1,
		initial_position2,
		initial_position3,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
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
		has_child,
		x3,
		y3,
		z3,
		start1,
		start2,
		start3,
		goal1,
		goal2,
		collision_avoidance,
		xlims,
		ylims,
		zlims,
	) = plot_data

	figure = _serif_figure(makie; size = figure_size)
	scene_axis = makie.LScene(figure[1, 1]; show_axis = true)
	makie.cam3d!(scene_axis.scene)

	grid_spacing = isnothing(lane_width) ? 2.0 : max(float(lane_width), 1.0)
	draw_single_integrator_3d_environment!(
		makie,
		scene_axis;
		xlims,
		ylims,
		zlims,
		grid_spacing,
	)

	player1_color, player2_color, player3_color = SINGLE_INTEGRATOR_3D_PLAYER_COLORS
	time_horizon = length(x1) - 1
	time_index = makie.Observable(0)
	safety_handle = nothing
	if has_child && !isnothing(collision_avoidance)
		# One hemisphere that follows the child position at the slider time;
		# the origin-centered wireframe is computed once and only translated.
		base_segments = _hemisphere_wireframe([0.0, 0.0, 0.0], collision_avoidance)
		for (segment_x, segment_y, segment_z) in base_segments
			segment_points = makie.lift(time_index) do t
				i = clamp(t + 1, 1, length(x3))
				[
					makie.Point3f(
						segment_x[k] + x3[i],
						segment_y[k] + y3[i],
						segment_z[k] + z3[i],
					) for k in eachindex(segment_x)
				]
			end
			current_safety = makie.lines!(
				scene_axis,
				segment_points;
				color = (player3_color, 0.42),
				linestyle = :dash,
				linewidth = 2.0,
			)
			if isnothing(safety_handle)
				safety_handle = current_safety
			end
		end
	end

	player1_path = makie.lines!(scene_axis, x1, y1, z1; color = player1_color, linewidth = 4)
	player2_path = makie.lines!(scene_axis, x2, y2, z2; color = player2_color, linewidth = 4)
	player3_path = nothing
	if has_child
		player3_path = makie.lines!(
			scene_axis,
			x3,
			y3,
			z3;
			color = player3_color,
			linewidth = 4,
			linestyle = :dash,
		)
	end

	makie.scatter!(scene_axis, x1, y1, z1; color = player1_color, markersize = 10)
	makie.scatter!(scene_axis, x2, y2, z2; color = player2_color, markersize = 10)
	if has_child
		makie.scatter!(
			scene_axis,
			x3,
			y3,
			z3;
			color = player3_color,
			marker = :utriangle,
			markersize = 12,
		)
	end

	start1_marker = makie.scatter!(
		scene_axis,
		[start1[1]],
		[start1[2]],
		[start1[3]];
		markersize = 24,
		color = player1_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	start2_marker = makie.scatter!(
		scene_axis,
		[start2[1]],
		[start2[2]],
		[start2[3]];
		markersize = 24,
		color = player2_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	start3_marker = has_child ? makie.scatter!(
		scene_axis,
		[start3[1]],
		[start3[2]],
		[start3[3]];
		marker = :utriangle,
		markersize = 26,
		color = player3_color,
		strokecolor = :black,
		strokewidth = 1,
	) : nothing
	current1_position = makie.lift(
		t -> _agent_position_at_time(makie, t, x1, y1, z1),
		time_index,
	)
	current2_position = makie.lift(
		t -> _agent_position_at_time(makie, t, x2, y2, z2),
		time_index,
	)
	current1_marker = makie.scatter!(
		scene_axis,
		current1_position;
		marker = :diamond,
		markersize = 30,
		color = player1_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	current2_marker = makie.scatter!(
		scene_axis,
		current2_position;
		marker = :diamond,
		markersize = 30,
		color = player2_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	current3_marker = if has_child
		current3_position = makie.lift(
			t -> _agent_position_at_time(makie, t, x3, y3, z3),
			time_index,
		)
		makie.scatter!(
			scene_axis,
			current3_position;
			marker = :diamond,
			markersize = 30,
			color = player3_color,
			strokecolor = :black,
			strokewidth = 1,
		)
	else
		nothing
	end
	goal1_marker = makie.scatter!(
		scene_axis,
		[goal1[1]],
		[goal1[2]],
		[goal1[3]];
		marker = :star5,
		markersize = 32,
		color = player1_color,
		strokecolor = :black,
		strokewidth = 1,
	)
	goal2_marker = makie.scatter!(
		scene_axis,
		[goal2[1]],
		[goal2[2]],
		[goal2[3]];
		marker = :star5,
		markersize = 32,
		color = player2_color,
		strokecolor = :black,
		strokewidth = 1,
	)

	legend_elements = Any[player1_path, player2_path]
	legend_labels = ["Left gripper (P1)", "Right gripper (P2)"]
	if has_child
		push!(legend_elements, player3_path)
		push!(legend_labels, "Child/Pet (P3)")
	end
	append!(
		legend_elements,
		Any[start1_marker, start2_marker, current1_marker, current2_marker, goal1_marker, goal2_marker],
	)
	append!(
		legend_labels,
		["Start 1", "Start 2", "Current 1", "Current 2", "Goal 1", "Goal 2"],
	)
	if has_child
		append!(legend_elements, Any[start3_marker, current3_marker])
		append!(legend_labels, ["Child/Pet Start", "Child/Pet Current"])
	end
	if !isnothing(safety_handle)
		push!(legend_elements, safety_handle)
		push!(legend_labels, "Safety hemisphere")
	end

	makie.Legend(
		figure[1, 2],
		legend_elements,
		legend_labels;
		framevisible = false,
		labelsize = legend_label_fontsize,
		orientation = :vertical,
		halign = :left,
		valign = :top,
	)
	makie.center!(scene_axis.scene)

	return figure, scene_axis, time_index, time_horizon
end

function _agent_position_at_time(makie, t, xs, ys, zs)
	i = clamp(t + 1, 1, length(xs))
	[makie.Point3f(xs[i], ys[i], zs[i])]
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
	save_path = nothing,
	figure_size = (1100, 850),
	exportable = true,
	offline = true,
	kwargs...,
)
	WGLMakie = _load_wglmakie()
	Bonito = WGLMakie.Bonito
	Base.invokelatest(WGLMakie.Page; exportable, offline)
	Base.invokelatest(WGLMakie.activate!)
	figure, ax, time_index, time_horizon = _plot_single_integrator_3d_browser_trajectories(
		WGLMakie;
		kwargs...,
		figure_size,
		legend_label_fontsize = 16,
	)
	app = Base.invokelatest(
		_time_slider_app,
		WGLMakie,
		Bonito,
		figure,
		time_index,
		time_horizon,
	)
	if !isnothing(save_path)
		Base.invokelatest(Bonito.export_static, save_path, app)
		# Recording the slider states leaves the time observable at the final
		# step; reset so any subsequent display starts at t = 0 again.
		time_index[] = 0
	end
	display_figure &&
		Base.invokelatest(display, Base.invokelatest(Bonito.BrowserDisplay), app)
	return figure, ax
end

# Wraps the WGLMakie figure in a Bonito app with an HTML "Time" slider.
# `Bonito.record_states` pre-records the scene updates for every slider value,
# so the slider stays functional in the static HTML export (where no Julia
# process is available); camera interaction is client-side and unaffected.
function _time_slider_app(makie, Bonito, figure, time_index, time_horizon)
	Bonito.App(; title = "GOOP interactive 3D trajectory") do session
		time_slider = Bonito.Slider(0:time_horizon; value = time_index[])
		makie.on(time_slider.value) do t
			time_index[] = t
		end
		time_readout = makie.map(t -> "t = $(t) / $(time_horizon)", time_slider.value)
		slider_row = Bonito.DOM.div(
			Bonito.DOM.span("Time"; style = "font-weight: 600; margin-right: 0.75em;"),
			Bonito.DOM.div(time_slider; style = "flex: 1 1 auto; max-width: 640px;"),
			Bonito.DOM.span(time_readout; style = "margin-left: 0.75em; min-width: 5em;");
			style = "display: flex; align-items: center; justify-content: center; " *
				"margin: 0.6em auto; width: 90%; font-family: Georgia, serif; " *
				"font-size: 1.05rem;",
		)
		dom = Bonito.DOM.div(figure, slider_row)
		return Bonito.record_states(session, dom)
	end
end

function plot_single_integrator_3d_trajectories_interactive(; kwargs...)
	plot_trajectory_3d_interactive(; kwargs...)
end
