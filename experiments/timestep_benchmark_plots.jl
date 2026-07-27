"""
Plotting helpers for `TimestepWarmstartBenchmark` results.

Kept separate from the benchmark module so the sweep itself stays free of a
CairoMakie dependency, per the repo convention that plotting lives in
`experiments/` and away from the measurement path.
"""

using CairoMakie: CairoMakie
using JLD2: JLD2

# Categorical slots 1–3 of the reference data-viz palette (light mode). Assigned
# in fixed order, never cycled; this triple is the documented all-pairs-passing
# subset, so it is used unchanged.
const SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
const TEXT_PRIMARY = "#0b0b0b"
const TEXT_SECONDARY = "#52514e"
const GRID_COLOR = ("#0b0b0b", 0.10)
const SURFACE = "#fcfcfb"

"""Short axis-side labels; the legend carries the full mode names."""
const SHORT_LABELS = Dict(
	:primal_only => "primal_only",
	:equality_duals => "equality_duals",
	:all_except_innermost_stationarity => "all_except_innermost",
)

"""
Vertical pixel offsets that keep end-of-line direct labels from overlapping when
two series finish at nearly the same value. Series far apart get no offset.
"""
function _stagger_offsets(values; threshold = 0.06, spread = 13.0)
	order = sortperm(values)
	offsets = zeros(length(values))
	for rank in 2:length(order)
		previous, current = order[rank-1], order[rank]
		if abs(log10(values[current]) - log10(values[previous])) < threshold
			offsets[current] = offsets[previous] + spread
		end
	end
	offsets
end

"""
Greedy word wrap. `Makie.Label` has no `word_wrap_width` in this version, so a
long caption would silently clip at the figure edge instead of wrapping.
"""
function _wrap_text(text, max_chars = 165)
	lines = String[]
	current = ""
	for word in split(text)
		candidate = isempty(current) ? word : current * " " * word
		if length(candidate) > max_chars && !isempty(current)
			push!(lines, current)
			current = word
		else
			current = candidate
		end
	end
	isempty(current) || push!(lines, current)
	join(lines, "\n")
end

"""Readable log-axis ticks: plain values, not `10^x` exponent labels."""
function _log_ticks(values)
	low, high = extrema(values)
	candidates = [
		0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8,
		1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0,
	]
	ticks = filter(t -> low / 1.35 <= t <= high * 1.35, candidates)
	isempty(ticks) && return CairoMakie.Makie.automatic
	labels = map(ticks) do t
		t >= 10 ? string(round(Int, t)) : string(t)
	end
	(ticks, labels)
end

"""
	plot_benchmark(run_dir; filename_stem, title)

Render median warm solve time and median warm Newton iterations against `Δt`,
one line per warm-start strategy, from a saved `benchmark_results.jld2`.

Two panels rather than one dual-axis plot: seconds and iteration counts are
different measures and must not share a y-scale. Both use a log y-axis — the
series span ~18× in time and ~19× in iterations, which a linear axis would
flatten into an unreadable band at the bottom.

Returns the paths written.
"""
function plot_benchmark(
	run_dir;
	filename_stem = "benchmark_warmstart_vs_dt",
	title = "Warm-started online solves vs. Δt (T = 20)",
	caption = nothing,
)
	data = JLD2.load_object(joinpath(run_dir, "benchmark_results.jld2"))
	summaries = filter(s -> !s.failed, data["summaries"])
	isempty(summaries) && error("No successful cells in $(run_dir).")

	modes = unique(s.mode for s in summaries)
	series = map(modes) do mode
		cells = sort(filter(s -> s.mode == mode, summaries); by = s -> s.Δt)
		(;
			mode,
			Δts = [c.Δt for c in cells],
			times = [c.warm_time_median for c in cells],
			iters = [Float64(c.warm_iters_median) for c in cells],
		)
	end

	CairoMakie.activate!()
	figure = CairoMakie.Figure(; size = (1020, 470), backgroundcolor = SURFACE)

	CairoMakie.Label(
		figure[0, 1:2],
		title;
		fontsize = 17,
		font = :bold,
		color = TEXT_PRIMARY,
		halign = :left,
		padding = (0, 0, 4, 0),
	)

	axis_kwargs = (;
		xlabel = "Δt (s)",
		xlabelcolor = TEXT_SECONDARY,
		ylabelcolor = TEXT_SECONDARY,
		xticklabelcolor = TEXT_SECONDARY,
		yticklabelcolor = TEXT_SECONDARY,
		titlecolor = TEXT_PRIMARY,
		titlealign = :left,
		titlesize = 14,
		backgroundcolor = SURFACE,
		xgridcolor = GRID_COLOR,
		ygridcolor = GRID_COLOR,
		xgridwidth = 1,
		ygridwidth = 1,
		topspinevisible = false,
		rightspinevisible = false,
		leftspinecolor = GRID_COLOR,
		bottomspinecolor = GRID_COLOR,
		yscale = log10,
	)

	time_axis = CairoMakie.Axis(
		figure[1, 1];
		title = "Median warm solve time",
		ylabel = "seconds",
		axis_kwargs...,
	)
	iter_axis = CairoMakie.Axis(
		figure[1, 2];
		title = "Median warm Newton iterations",
		ylabel = "iterations",
		axis_kwargs...,
	)

	label_offsets = _stagger_offsets([s.times[end] for s in series])
	plot_handles = CairoMakie.Lines[]
	for (index, s) in enumerate(series)
		color = SERIES_COLORS[mod1(index, length(SERIES_COLORS))]
		for (axis, values) in ((time_axis, s.times), (iter_axis, s.iters))
			CairoMakie.lines!(axis, s.Δts, values; color, linewidth = 2)
			CairoMakie.scatter!(
				axis,
				s.Δts,
				values;
				color,
				markersize = 11,
				# 2px surface ring keeps overlapping markers separable.
				strokecolor = SURFACE,
				strokewidth = 2,
			)
		end
		# Direct labels in text ink (never the series color); the adjacent colored
		# line carries identity.
		CairoMakie.text!(
			time_axis,
			s.Δts[end],
			s.times[end];
			text = SHORT_LABELS[s.mode],
			align = (:left, :center),
			offset = (8, label_offsets[index]),
			fontsize = 11,
			color = TEXT_SECONDARY,
		)
		push!(
			plot_handles,
			CairoMakie.lines!(time_axis, s.Δts[1:1], s.times[1:1]; color, linewidth = 2),
		)
	end

	# Extra right-hand headroom on the time panel only: that is where the
	# end-of-line direct labels sit, and they must not be clipped.
	Δt_values = first(series).Δts
	low, high = extrema(Δt_values)
	span = high - low
	CairoMakie.xlims!(time_axis, low - 0.08span, high + 0.45span)
	CairoMakie.xlims!(iter_axis, low - 0.08span, high + 0.08span)
	tick_labels = [rstrip(rstrip(string(round(d; digits = 3)), '0'), '.') for d in Δt_values]
	for axis in (time_axis, iter_axis)
		axis.xticks = (Δt_values, tick_labels)
	end
	time_axis.yticks = _log_ticks(reduce(vcat, [s.times for s in series]))
	iter_axis.yticks = _log_ticks(reduce(vcat, [s.iters for s in series]))

	CairoMakie.Legend(
		figure[2, 1:2],
		plot_handles,
		[String(s.mode) for s in series];
		orientation = :horizontal,
		framevisible = false,
		labelcolor = TEXT_SECONDARY,
		labelsize = 12,
		padding = (0, 0, 0, 0),
	)

	if !isnothing(caption)
		CairoMakie.Label(
			figure[3, 1:2],
			_wrap_text(caption);
			fontsize = 11,
			color = TEXT_SECONDARY,
			halign = :left,
			justification = :left,
			padding = (0, 0, 0, 2),
		)
	end

	CairoMakie.rowgap!(figure.layout, 6)
	CairoMakie.colgap!(figure.layout, 28)

	paths = String[]
	for extension in ("png", "pdf")
		path = joinpath(run_dir, "$(filename_stem).$(extension)")
		CairoMakie.save(path, figure; px_per_unit = 2)
		push!(paths, path)
	end
	paths
end
