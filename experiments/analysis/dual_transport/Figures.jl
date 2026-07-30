module DualTransportFigures

using CairoMakie
using Statistics: median, quantile

export generate_figures

const FORMULATIONS = ("reduced", "quasi")
const RETAINED_MODES = (
    "equality_duals",
    "all_except_innermost_stationarity",
    "all_duals",
)
const TRANSPORTS = (
    "identity_copy",
    "stage_shift_zero_tail",
    "stage_shift_hold_tail",
)
const PRIMARY_PROJECTION_RTOL = 1e-8

const MODE_LABEL = Dict(
    "primal_only" => "Primal",
    "equality_duals" => "Primal + λ",
    "all_except_innermost_stationarity" => "Primal + λ + ψout",
    "all_duals" => "All duals",
)
const SHORT_MODE_LABEL = Dict(
    "primal_only" => "P",
    "equality_duals" => "P+λ",
    "all_except_innermost_stationarity" => "P+λ+ψout",
    "all_duals" => "All",
)
const TRANSPORT_LABEL = Dict(
    "not_applicable" => "Primal only",
    "identity_copy" => "Identity",
    "stage_shift_zero_tail" => "Shift, zero tail",
    "stage_shift_hold_tail" => "Shift, hold tail",
)
const TRANSPORT_SHORT = Dict(
    "not_applicable" => "",
    "identity_copy" => "I",
    "stage_shift_zero_tail" => "Z",
    "stage_shift_hold_tail" => "H",
)
const CANDIDATE_LABEL = Dict(
    "identity_copy" => "Identity",
    "stage_shift_zero_tail" => "Shift–zero",
    "stage_shift_zero_tail_gamma0p5" => "Shift–zero, γ=0.5",
)

const WONG = Makie.wong_colors()
const TRANSPORT_COLOR = Dict(
    "not_applicable" => RGBf(0.38, 0.38, 0.38),
    "identity_copy" => WONG[1],
    "stage_shift_zero_tail" => WONG[2],
    "stage_shift_hold_tail" => WONG[4],
)
const PERIOD_COLOR = Dict(
    "early" => WONG[3],
    "late" => WONG[6],
)
const ENERGY_COLOR = Dict(
    "row" => WONG[2],
    "null" => WONG[4],
)

const CsvRow = Dict{String, String}

function _split_csv_line(line, path, line_number)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    index = firstindex(line)
    while index <= lastindex(line)
        character = line[index]
        if quoted
            if character == '"'
                next_index = nextind(line, index)
                if next_index <= lastindex(line) &&
                   line[next_index] == '"'
                    write(buffer, '"')
                    index = next_index
                else
                    quoted = false
                end
            else
                write(buffer, character)
            end
        elseif character == '"'
            quoted = true
        elseif character == ','
            push!(fields, String(take!(buffer)))
        else
            write(buffer, character)
        end
        index = nextind(line, index)
    end
    quoted && error("Unclosed CSV quote at $(path):$(line_number).")
    push!(fields, String(take!(buffer)))
    fields
end

function _read_csv(path)
    isfile(path) || return CsvRow[]
    lines = readlines(path)
    isempty(lines) && return CsvRow[]
    header = _split_csv_line(first(lines), path, 1)
    rows = CsvRow[]
    for (row_index, line) in enumerate(Iterators.drop(lines, 1))
        isempty(line) && continue
        line_number = row_index + 1
        fields = _split_csv_line(line, path, line_number)
        length(fields) == length(header) || error(
            "CSV field-count mismatch at $(path):$(line_number).",
        )
        push!(rows, CsvRow(header .=> fields))
    end
    rows
end

function _finite(row, column)
    text = strip(get(row, column, ""))
    isempty(text) && return nothing
    value = tryparse(Float64, text)
    isnothing(value) || !isfinite(value) ? nothing : value
end

_true(row, column) =
    lowercase(strip(get(row, column, ""))) == "true"

function _integer(row, column)
    text = strip(get(row, column, ""))
    isempty(text) && return nothing
    tryparse(Int, text)
end

_pair_key(row) = (
    get(row, "scenario_seed", ""),
    get(row, "transition", ""),
)

function _period(row)
    stored = lowercase(strip(get(row, "period", "")))
    stored in ("early", "late") && return stored
    transition = _integer(row, "transition")
    isnothing(transition) ? "" : transition <= 3 ? "early" : "late"
end

function _symmetric_offsets(count::Integer, half_width::Real = 0.035)
    count <= 0 && return Float64[]
    count == 1 && return [0.0]
    collect(range(-Float64(half_width), Float64(half_width); length = count))
end

function _quartiles(values)
    isempty(values) && return nothing
    (
        low = quantile(values, 0.25),
        center = median(values),
        high = quantile(values, 0.75),
    )
end

function _log_if_positive(values; minimum_span = 10.0)
    finite_values = Float64[
        value for value in values if isfinite(value)
    ]
    isempty(finite_values) && return identity
    all(value -> value > 0.0, finite_values) || return identity
    maximum(finite_values) / minimum(finite_values) >= minimum_span ?
    log10 : identity
end

function _empty_axis!(axis, message)
    text!(
        axis,
        0.5,
        0.5;
        text = message,
        space = :relative,
        align = (:center, :center),
        fontsize = 14,
    )
    hidespines!(axis)
    hidedecorations!(axis)
    nothing
end

function _figure_title!(figure, title, subtitle = "")
    header = GridLayout()
    figure[0, :] = header
    Label(
        header[1, 1],
        title;
        fontsize = 22,
        font = :bold,
        padding = (0, 0, 0, 2),
    )
    isempty(subtitle) || Label(
        header[2, 1],
        subtitle;
        fontsize = 13,
        padding = (0, 0, 0, 4),
    )
    rowgap!(header, 0)
    nothing
end

function _save_versions(output_dir, stem, figure)
    mkpath(output_dir)
    png = joinpath(output_dir, "$(stem).png")
    pdf = joinpath(output_dir, "$(stem).pdf")
    save(png, figure; px_per_unit = 2)
    save(pdf, figure)
    (png, pdf)
end

function _identity_structured_figure(rows)
    figure = Figure(size = (1420, 630))
    transport_offsets = Dict(
        transport => offset for (transport, offset) in
        zip(TRANSPORTS, (-0.23, 0.0, 0.23))
    )
    axes = Axis[]
    axis_scales = Any[]
    legend_handles = Any[]
    legend_labels = String[]

    for (panel, formulation) in enumerate(FORMULATIONS)
        selected = filter(
            row ->
                get(row, "formulation", "") == formulation &&
                get(row, "mode", "") in RETAINED_MODES &&
                get(row, "dual_transport", "") in TRANSPORTS,
            rows,
        )
        all_values = Float64[
            value for row in selected
            for value in (_finite(row, "initial_residual_norm2"),)
            if !isnothing(value)
        ]
        axis_scale = _log_if_positive(all_values)
        axis = Axis(
            figure[1, panel];
            title = "$(uppercasefirst(formulation)) formulation",
            xlabel = "Dual information retained",
            ylabel = panel == 1 ? "Initial KKT residual R₀" : "",
            xticks = (
                1:length(RETAINED_MODES),
                [MODE_LABEL[mode] for mode in RETAINED_MODES],
            ),
            xticklabelrotation = π / 12,
            yscale = axis_scale,
        )
        push!(axes, axis)
        push!(axis_scales, axis_scale)
        isempty(selected) && begin
            _empty_axis!(
                axis,
                "No residual diagnostics\n(raw/residual_diagnostics.csv)",
            )
            continue
        end

        plotted = false
        for (mode_position, mode) in enumerate(RETAINED_MODES)
            mode_rows =
                filter(row -> get(row, "mode", "") == mode, selected)
            paired =
                Dict{Tuple{String, String}, Dict{String, Float64}}()
            for row in mode_rows
                value = _finite(row, "initial_residual_norm2")
                isnothing(value) && continue
                transport = get(row, "dual_transport", "")
                get!(
                    paired,
                    _pair_key(row),
                    Dict{String, Float64}(),
                )[transport] = value
            end
            for values in Base.values(paired)
                transports = [
                    transport for transport in TRANSPORTS if
                    haskey(values, transport)
                ]
                length(transports) >= 2 || continue
                x = [
                    mode_position + transport_offsets[transport]
                    for transport in transports
                ]
                y = [values[transport] for transport in transports]
                lines!(
                    axis,
                    x,
                    y;
                    color = (:gray35, 0.18),
                    linewidth = 0.8,
                )
            end
            for transport in TRANSPORTS
                values = Float64[
                    item[transport] for item in Base.values(paired) if
                    haskey(item, transport)
                ]
                isempty(values) && continue
                x = mode_position + transport_offsets[transport]
                scatter!(
                    axis,
                    x .+ _symmetric_offsets(length(values)),
                    values;
                    color = (TRANSPORT_COLOR[transport], 0.28),
                    markersize = 5,
                )
                summary = _quartiles(values)
                rangebars!(
                    axis,
                    [x],
                    [summary.low],
                    [summary.high];
                    color = TRANSPORT_COLOR[transport],
                    linewidth = 3,
                )
                median_plot = scatter!(
                    axis,
                    [x],
                    [summary.center];
                    color = TRANSPORT_COLOR[transport],
                    strokecolor = :black,
                    strokewidth = 0.7,
                    markersize = 12,
                )
                if panel == 1 && mode_position == 1
                    push!(legend_handles, median_plot)
                    push!(
                        legend_labels,
                        TRANSPORT_LABEL[transport],
                    )
                end
                plotted = true
            end
        end
        plotted ||
            _empty_axis!(axis, "No finite initialization residuals")
    end
    if length(axes) == 2 &&
       first(axis_scales) === last(axis_scales)
        linkyaxes!(axes...)
    end
    if !isempty(legend_handles)
        Legend(
            figure[2, :],
            legend_handles,
            legend_labels;
            orientation = :horizontal,
            nbanks = 1,
            halign = :center,
            tellwidth = false,
        )
    end
    _figure_title!(
        figure,
        "Identity copying versus structured dual transport",
        "Thin lines pair the same source–destination transition; points show medians and interquartile ranges. Formulations share one y-scale.",
    )
    figure
end

function _block_alignment_figure(rows)
    figure = Figure(size = (1320, 540))
    transport_offsets = Dict(
        transport => offset for (transport, offset) in
        zip(TRANSPORTS, (-0.18, 0.0, 0.18))
    )
    for (panel, formulation) in enumerate(FORMULATIONS)
        selected = filter(
            row ->
                get(row, "formulation", "") == formulation &&
                get(row, "dual_transport", "") in TRANSPORTS,
            rows,
        )
        axis = Axis(
            figure[1, panel];
            title = "$(uppercasefirst(formulation)) formulation",
            xlabel = "Stationarity-multiplier action",
            ylabel =
                panel == 1 ?
                "Residual-reducing alignment c" : "",
            xticks = (1:2, ["ψout action (c_out)", "ψin action (c_in)"]),
        )
        hlines!(
            axis,
            [0.0];
            color = (:black, 0.5),
            linestyle = :dash,
            linewidth = 1,
        )
        isempty(selected) && begin
            _empty_axis!(
                axis,
                "No block-action diagnostics\n(raw/residual_action.csv)",
            )
            continue
        end
        plotted = false
        for transport in TRANSPORTS
            transport_rows = filter(
                row -> get(row, "dual_transport", "") == transport,
                selected,
            )
            offset = transport_offsets[transport]
            for row in transport_rows
                c_out = _finite(row, "c_out")
                c_in = _finite(row, "c_in")
                (isnothing(c_out) || isnothing(c_in)) && continue
                lines!(
                    axis,
                    [1 + offset, 2 + offset],
                    [c_out, c_in];
                    color = (TRANSPORT_COLOR[transport], 0.18),
                    linewidth = 0.8,
                )
            end
            for (position, column) in ((1, "c_out"), (2, "c_in"))
                values = Float64[
                    value for row in transport_rows
                    for value in (_finite(row, column),)
                    if !isnothing(value)
                ]
                isempty(values) && continue
                x = position + offset
                scatter!(
                    axis,
                    x .+ _symmetric_offsets(length(values), 0.025),
                    values;
                    color = (TRANSPORT_COLOR[transport], 0.3),
                    markersize = 5,
                )
                summary = _quartiles(values)
                rangebars!(
                    axis,
                    [x],
                    [summary.low],
                    [summary.high];
                    color = TRANSPORT_COLOR[transport],
                    linewidth = 3,
                )
                scatter!(
                    axis,
                    [x],
                    [summary.center];
                    color = TRANSPORT_COLOR[transport],
                    strokecolor = :black,
                    strokewidth = 0.7,
                    markersize = 12,
                    label =
                        position == 1 ?
                        TRANSPORT_LABEL[transport] : nothing,
                )
                plotted = true
            end
        end
        if plotted
            ylims!(axis, -1.05, 1.05)
            axislegend(axis; position = :lb, framevisible = false)
        else
            _empty_axis!(axis, "No finite c_out/c_in values")
        end
    end
    _figure_title!(
        figure,
        "Where transported dual blocks act on the KKT residual",
        "Positive c means the block action points against the residual; thin lines pair c_out and c_in within a transition.",
    )
    figure
end

function _policy_order()
    policies = Tuple{String, String}[
        ("primal_only", "not_applicable"),
    ]
    for mode in RETAINED_MODES, transport in TRANSPORTS
        push!(policies, (mode, transport))
    end
    policies
end

function _policy_label(mode, transport)
    transport == "not_applicable" && return MODE_LABEL[mode]
    "$(SHORT_MODE_LABEL[mode])–$(TRANSPORT_SHORT[transport])"
end

function _policy_rows(rows, formulation, mode, transport)
    filter(
        row ->
            get(row, "formulation", "") == formulation &&
            get(row, "mode", "") == mode &&
            get(row, "dual_transport", "") == transport,
        rows,
    )
end

function _convergence_iterations_figure(rows)
    policies = _policy_order()
    labels = [_policy_label(policy...) for policy in policies]
    figure = Figure(size = (1580, 910))
    legend_handles = Any[]
    legend_labels = String[]
    convergence_axes = Axis[]
    iteration_axes = Axis[]
    for (panel, formulation) in enumerate(FORMULATIONS)
        convergence_axis = Axis(
            figure[1, panel];
            title = "$(uppercasefirst(formulation)) formulation",
            ylabel = panel == 1 ? "Converged fraction" : "",
            xticks = (1:length(policies), fill("", length(policies))),
        )
        iteration_axis = Axis(
            figure[2, panel];
            xlabel = "Warm-start policy",
            ylabel =
                panel == 1 ?
                "Inner iterations (all attempts)" : "",
            xticks = (1:length(policies), labels),
            xticklabelrotation = π / 5,
        )
        push!(convergence_axes, convergence_axis)
        push!(iteration_axes, iteration_axis)
        if isempty(rows)
            _empty_axis!(
                convergence_axis,
                "No replay results\n(raw/replay.csv)",
            )
            _empty_axis!(iteration_axis, "No replay results")
            continue
        end
        any_policy = false
        for (position, (mode, transport)) in enumerate(policies)
            selected = _policy_rows(
                rows,
                formulation,
                mode,
                transport,
            )
            isempty(selected) && continue
            numerator = count(
                row -> _true(row, "direct_converged"),
                selected,
            )
            denominator = length(selected)
            rate = numerator / denominator
            color = TRANSPORT_COLOR[transport]
            bar_plot = barplot!(
                convergence_axis,
                [position],
                [rate];
                color = color,
                strokecolor = :black,
                strokewidth = 0.4,
            )
            if panel == 1 && position in (1, 2, 3, 4)
                push!(legend_handles, bar_plot)
                push!(
                    legend_labels,
                    TRANSPORT_LABEL[transport],
                )
            end
            text!(
                convergence_axis,
                [position],
                [min(rate + 0.035, 1.06)];
                text = ["$(numerator)/$(denominator)"],
                align = (:center, :bottom),
                fontsize = 10,
            )

            iteration_rows = [
                (;
                    value,
                    converged = _true(row, "direct_converged"),
                ) for row in selected
                for value in (_finite(row, "total_inner_iters"),)
                if !isnothing(value)
            ]
            values = getproperty.(iteration_rows, :value)
            if !isempty(values)
                jitter = _symmetric_offsets(length(values), 0.12)
                converged_mask =
                    getproperty.(iteration_rows, :converged)
                any(converged_mask) && scatter!(
                    iteration_axis,
                    position .+ jitter[converged_mask],
                    values[converged_mask];
                    color = (color, 0.38),
                    markersize = 6,
                )
                any(.!converged_mask) && scatter!(
                    iteration_axis,
                    position .+ jitter[.!converged_mask],
                    values[.!converged_mask];
                    color = :firebrick,
                    marker = :xcross,
                    markersize = 10,
                )
                summary = _quartiles(values)
                rangebars!(
                    iteration_axis,
                    [position],
                    [summary.low],
                    [summary.high];
                    color = color,
                    linewidth = 3,
                )
                scatter!(
                    iteration_axis,
                    [position],
                    [summary.center];
                    color,
                    strokecolor = :black,
                    strokewidth = 0.7,
                    markersize = 11,
                )
            end
            any_policy = true
        end
        if any_policy
            ylims!(convergence_axis, 0.0, 1.12)
        else
            _empty_axis!(
                convergence_axis,
                "No $(formulation) replay rows",
            )
            _empty_axis!(iteration_axis, "No iteration data")
        end
    end
    length(convergence_axes) == 2 &&
        linkyaxes!(convergence_axes...)
    length(iteration_axes) == 2 &&
        linkyaxes!(iteration_axes...)
    if !isempty(legend_handles)
        Legend(
            figure[3, :],
            legend_handles,
            legend_labels;
            orientation = :horizontal,
            nbanks = 1,
            halign = :center,
            tellwidth = false,
        )
    end
    _figure_title!(
        figure,
        "Solver reliability and effort by warm-start policy",
        "Bar labels are exact converged/attempted counts; red crosses are nonconverged attempts.",
    )
    figure
end

function _early_late_ratios(rows, formulation)
    grouped = Dict{
        Tuple{String, String, String},
        Dict{String, Float64},
    }()
    for row in rows
        get(row, "formulation", "") == formulation || continue
        mode = get(row, "mode", "")
        mode in RETAINED_MODES || continue
        transport = get(row, "dual_transport", "")
        transport in (
            "identity_copy",
            "stage_shift_zero_tail",
        ) || continue
        value = _finite(row, "initial_residual_norm2")
        isnothing(value) && continue
        key = (
            get(row, "scenario_seed", ""),
            get(row, "transition", ""),
            mode,
        )
        get!(grouped, key, Dict{String, Float64}())[transport] = value
    end
    results = NamedTuple[]
    for (key, values) in grouped
        haskey(values, "identity_copy") || continue
        haskey(values, "stage_shift_zero_tail") || continue
        denominator = values["identity_copy"]
        denominator > 0.0 || continue
        ratio = values["stage_shift_zero_tail"] / denominator
        transition = tryparse(Int, key[2])
        isnothing(transition) && continue
        push!(results, (;
            mode = key[3],
            period = transition <= 3 ? "early" : "late",
            ratio,
        ))
    end
    results
end

function _early_late_figure(rows)
    figure = Figure(size = (1320, 630))
    period_offsets = Dict("early" => -0.14, "late" => 0.14)
    axes = Axis[]
    axis_scales = Any[]
    legend_handles = Dict{String, Any}()
    for (panel, formulation) in enumerate(FORMULATIONS)
        ratios = _early_late_ratios(rows, formulation)
        ratio_values = getproperty.(ratios, :ratio)
        axis_scale = _log_if_positive(
            ratio_values;
            minimum_span = 3.0,
        )
        axis = Axis(
            figure[1, panel];
            title = "$(uppercasefirst(formulation)) formulation",
            xlabel = "Dual information retained",
            ylabel =
                panel == 1 ?
                "R₀(shift–zero) / R₀(identity)" : "",
            xticks = (
                1:length(RETAINED_MODES),
                [MODE_LABEL[mode] for mode in RETAINED_MODES],
            ),
            xticklabelrotation = π / 12,
            yscale = axis_scale,
        )
        push!(axes, axis)
        push!(axis_scales, axis_scale)
        hlines!(
            axis,
            [1.0];
            color = (:black, 0.55),
            linestyle = :dash,
            linewidth = 1.1,
        )
        isempty(ratios) && begin
            _empty_axis!(
                axis,
                "No paired identity/shift–zero diagnostics",
            )
            continue
        end
        for (mode_position, mode) in enumerate(RETAINED_MODES)
            for period in ("early", "late")
                values = Float64[
                    row.ratio for row in ratios if
                    row.mode == mode && row.period == period
                ]
                isempty(values) && continue
                x = mode_position + period_offsets[period]
                scatter!(
                    axis,
                    x .+ _symmetric_offsets(length(values), 0.04),
                    values;
                    color = (PERIOD_COLOR[period], 0.35),
                    markersize = 6,
                )
                summary = _quartiles(values)
                rangebars!(
                    axis,
                    [x],
                    [summary.low],
                    [summary.high];
                    color = PERIOD_COLOR[period],
                    linewidth = 3,
                )
                median_plot = scatter!(
                    axis,
                    [x],
                    [summary.center];
                    color = PERIOD_COLOR[period],
                    strokecolor = :black,
                    strokewidth = 0.7,
                    markersize = 12,
                )
                haskey(legend_handles, period) ||
                    (legend_handles[period] = median_plot)
            end
        end
    end
    if length(axes) == 2 &&
       first(axis_scales) === last(axis_scales)
        linkyaxes!(axes...)
    end
    period_order = ("early", "late")
    shown_periods = [
        period for period in period_order if
        haskey(legend_handles, period)
    ]
    if !isempty(shown_periods)
        Legend(
            figure[2, :],
            [legend_handles[period] for period in shown_periods],
            [uppercasefirst(period) for period in shown_periods];
            orientation = :horizontal,
            nbanks = 1,
            halign = :center,
            tellwidth = false,
        )
    end
    _figure_title!(
        figure,
        "Does structured transport behave differently early and late?",
        "Ratios are paired within each transition; values below one favor stage shifting with a zero tail. Formulations share one y-scale.",
    )
    figure
end

function _case_label(row)
    seed = get(row, "scenario_seed", "?")
    transition = get(row, "transition", "?")
    "seed $(seed), t$(transition)"
end

function _gamma_figure(rows)
    figure = Figure(size = (1420, 920))
    residual_axes = Axis[]
    residual_scales = Any[]
    iteration_axes = Axis[]
    for (panel, formulation) in enumerate(FORMULATIONS)
        selected =
            filter(row -> get(row, "formulation", "") == formulation, rows)
        residual_values = Float64[
            value for row in selected
            for value in (_finite(row, "initial_residual_norm2"),)
            if !isnothing(value)
        ]
        residual_scale = _log_if_positive(residual_values)
        residual_axis = Axis(
            figure[1, panel];
            title = "$(uppercasefirst(formulation)) formulation",
            xlabel = "Innermost-dual damping γ",
            ylabel = panel == 1 ? "Initial KKT residual R₀" : "",
            yscale = residual_scale,
        )
        iteration_axis = Axis(
            figure[2, panel];
            xlabel = "Innermost-dual damping γ",
            ylabel =
                panel == 1 ?
                "Inner iterations" : "",
        )
        push!(residual_axes, residual_axis)
        push!(residual_scales, residual_scale)
        push!(iteration_axes, iteration_axis)
        isempty(selected) && begin
            _empty_axis!(
                residual_axis,
                "No γ sweep\n(raw/damping.csv)",
            )
            _empty_axis!(iteration_axis, "No γ sweep")
            continue
        end
        grouped = Dict{Tuple{String, String}, Vector{CsvRow}}()
        for row in selected
            push!(
                get!(grouped, _pair_key(row), CsvRow[]),
                row,
            )
        end
        case_colors = WONG[1:max(1, min(length(grouped), length(WONG)))]
        legend_handles = Any[]
        legend_labels = String[]
        for ((_, case_rows), color) in zip(
            sort!(collect(grouped); by = first),
            Iterators.cycle(case_colors),
        )
            points = NamedTuple[]
            for row in case_rows
                gamma = _finite(row, "gamma")
                residual = _finite(row, "initial_residual_norm2")
                iterations = _finite(row, "total_inner_iters")
                isnothing(gamma) && continue
                push!(points, (;
                    gamma,
                    residual,
                    iterations,
                    converged = _true(row, "direct_converged"),
                    label = _case_label(row),
                ))
            end
            sort!(points; by = point -> point.gamma)
            isempty(points) && continue
            residual_points =
                filter(point -> !isnothing(point.residual), points)
            if !isempty(residual_points)
                line_plot = lines!(
                    residual_axis,
                    getproperty.(residual_points, :gamma),
                    getproperty.(residual_points, :residual);
                    color = (color, 0.75),
                    linewidth = 1.6,
                )
                push!(legend_handles, line_plot)
                push!(legend_labels, first(points).label)
                scatter!(
                    residual_axis,
                    getproperty.(residual_points, :gamma),
                    getproperty.(residual_points, :residual);
                    color,
                    markersize = 7,
                )
            end
            iteration_points =
                filter(point -> !isnothing(point.iterations), points)
            if !isempty(iteration_points)
                lines!(
                    iteration_axis,
                    getproperty.(iteration_points, :gamma),
                    getproperty.(iteration_points, :iterations);
                    color = (color, 0.45),
                    linewidth = 1.2,
                )
                converged =
                    getproperty.(iteration_points, :converged)
                any(converged) && scatter!(
                    iteration_axis,
                    getproperty.(iteration_points, :gamma)[converged],
                    getproperty.(iteration_points, :iterations)[converged];
                    color,
                    markersize = 7,
                )
                any(.!converged) && scatter!(
                    iteration_axis,
                    getproperty.(iteration_points, :gamma)[.!converged],
                    getproperty.(iteration_points, :iterations)[.!converged];
                    color = :firebrick,
                    marker = :xcross,
                    markersize = 11,
                )
            end
        end
        if !isempty(legend_handles)
            Legend(
                figure[3, panel],
                legend_handles,
                legend_labels;
                orientation = :horizontal,
                nbanks = 1,
                halign = :center,
                tellwidth = false,
            )
        end
    end
    if length(residual_axes) == 2 &&
       first(residual_scales) === last(residual_scales)
        linkyaxes!(residual_axes...)
    end
    length(iteration_axes) == 2 &&
        linkyaxes!(iteration_axes...)
    _figure_title!(
        figure,
        "Conditional damping of transported innermost duals",
        "Lines pair γ values within each selected transition; red crosses mark nonconverged solves. Formulations share y-scales.",
    )
    figure
end

function _first_error_line(row)
    error_text = strip(get(row, "error", ""))
    isempty(error_text) && return ""
    expanded = replace(error_text, "\\n" => "\n", "\\r" => "")
    first(split(expanded, '\n'))
end

function _wrap_words(text, width = 68)
    isempty(text) && return text
    lines = String[]
    current = ""
    for word in split(text)
        if isempty(current)
            current = word
        elseif length(current) + length(word) + 1 <= width
            current *= " " * word
        else
            push!(lines, current)
            current = word
        end
    end
    isempty(current) || push!(lines, current)
    join(lines, "\n")
end

function _projection_blocker(path, rows, formulation)
    isfile(path) ||
        return "Missing raw/projection.csv\nProjection stage has not produced output."
    isempty(rows) && return "raw/projection.csv contains no rows."
    selected =
        filter(row -> get(row, "formulation", "") == formulation, rows)
    isempty(selected) &&
        return "No projection rows for $(formulation)."
    statuses = sort!(unique(
        get(row, "status", "") for row in selected
    ))
    errors = [
        _first_error_line(row) for row in selected if
        !isempty(_first_error_line(row))
    ]
    if !isempty(errors)
        return "Projection unavailable ($(length(selected)) rows).\n" *
               "First blocker (verbatim):\n" *
               _wrap_words(first(errors))
    end
    "Projection rows have no usable row/null energies.\n" *
    "Recorded status: $(join(statuses, ", "))."
end

function _projection_selection(rows, formulation)
    computed = filter(
        row ->
            get(row, "formulation", "") == formulation &&
            lowercase(get(row, "status", "")) == "computed" &&
            get(row, "scale_mode", "") == "scale_aware",
        rows,
    )
    isempty(computed) && return (rows = CsvRow[], rtol = nothing)
    rtols = sort!(unique(Float64[
        value for row in computed
        for value in (_finite(row, "rank_rtol"),)
        if !isnothing(value) && value > 0.0
    ]))
    isempty(rtols) && return (rows = CsvRow[], rtol = nothing)
    index = argmin(
        abs(log10(value) - log10(PRIMARY_PROJECTION_RTOL))
        for value in rtols
    )
    chosen = rtols[index]
    selected = filter(
        row -> begin
            value = _finite(row, "rank_rtol")
            !isnothing(value) &&
                isapprox(value, chosen; rtol = 1e-12, atol = 0.0)
        end,
        computed,
    )
    (rows = selected, rtol = chosen)
end

function _row_null_figure(rows, projection_path)
    figure = Figure(size = (1320, 760))
    candidates = (
        "identity_copy",
        "stage_shift_zero_tail",
        "stage_shift_zero_tail_gamma0p5",
    )
    energy_offsets = Dict("row" => -0.13, "null" => 0.13)
    plotted_axes = Axis[]
    all_plotted_fractions = Float64[]
    legend_handles = Dict{String, Any}()
    panel_notes = Dict{Int, String}()
    for (panel, formulation) in enumerate(FORMULATIONS)
        selection = _projection_selection(rows, formulation)
        selected = selection.rows
        axis = Axis(
            figure[1, panel];
            title = "$(uppercasefirst(formulation)) formulation",
            xlabel = "Dual candidate",
            ylabel =
                panel == 1 ?
                "Scale-metric energy fraction" : "",
            xticks = (
                1:length(candidates),
                [CANDIDATE_LABEL[candidate] for candidate in candidates],
            ),
            xticklabelrotation = π / 10,
        )
        if isempty(selected)
            _empty_axis!(
                axis,
                _projection_blocker(
                    projection_path,
                    rows,
                    formulation,
                ),
            )
            continue
        end
        plotted = false
        for (position, candidate) in enumerate(candidates)
            candidate_rows = filter(
                row -> get(row, "candidate", "") == candidate,
                selected,
            )
            paired = NamedTuple[]
            for row in candidate_rows
                row_fraction =
                    _finite(row, "metric_row_energy_fraction")
                null_fraction =
                    _finite(row, "metric_null_energy_fraction")
                (
                    isnothing(row_fraction) ||
                    isnothing(null_fraction)
                ) && continue
                push!(paired, (; row_fraction, null_fraction))
                push!(
                    all_plotted_fractions,
                    row_fraction,
                    null_fraction,
                )
                lines!(
                    axis,
                    [
                        position + energy_offsets["row"],
                        position + energy_offsets["null"],
                    ],
                    [row_fraction, null_fraction];
                    color = (:gray35, 0.2),
                    linewidth = 0.8,
                )
            end
            for (energy, property) in (
                ("row", :row_fraction),
                ("null", :null_fraction),
            )
                values = Float64[
                    getproperty(item, property) for item in paired
                ]
                isempty(values) && continue
                x = position + energy_offsets[energy]
                scatter!(
                    axis,
                    x .+ _symmetric_offsets(length(values), 0.035),
                    values;
                    color = (ENERGY_COLOR[energy], 0.32),
                    marker = energy == "row" ? :circle : :diamond,
                    markersize = 7,
                )
                summary = _quartiles(values)
                rangebars!(
                    axis,
                    [x],
                    [summary.low],
                    [summary.high];
                    color = ENERGY_COLOR[energy],
                    linewidth = 3,
                )
                median_plot = scatter!(
                    axis,
                    [x],
                    [summary.center];
                    color = ENERGY_COLOR[energy],
                    marker = energy == "row" ? :circle : :diamond,
                    strokecolor = :black,
                    strokewidth = 0.7,
                    markersize = 12,
                )
                haskey(legend_handles, energy) ||
                    (legend_handles[energy] = median_plot)
                plotted = true
            end
        end
        if plotted
            push!(plotted_axes, axis)
            ranks = Int[
                value for row in selected
                for value in (_integer(row, "rank"),)
                if !isnothing(value)
            ]
            nullities = Int[
                value for row in selected
                for value in (_integer(row, "nullity"),)
                if !isnothing(value)
            ]
            note =
                "scale-aware metric, rank rtol=$(selection.rtol)"
            if !isempty(ranks) && !isempty(nullities)
                note *=
                    "\nrank $(minimum(ranks))–$(maximum(ranks)); " *
                    "nullity $(minimum(nullities))–$(maximum(nullities))"
            end
            panel_notes[panel] = note
        else
            _empty_axis!(
                axis,
                "Computed projection rows lack finite energy fractions.",
            )
        end
    end
    for (panel, note) in panel_notes
        Label(
            figure[2, panel],
            note;
            fontsize = 11,
            halign = :center,
            tellwidth = false,
        )
    end
    if !isempty(plotted_axes)
        lower = min(0.0, minimum(all_plotted_fractions))
        upper = max(1.0, maximum(all_plotted_fractions))
        padding = 0.035 * max(upper - lower, 0.1)
        for axis in plotted_axes
            ylims!(axis, lower - padding, upper + padding)
        end
        length(plotted_axes) == 2 && linkyaxes!(plotted_axes...)
    end
    energy_order = ("row", "null")
    shown_energies = [
        energy for energy in energy_order if
        haskey(legend_handles, energy)
    ]
    if !isempty(shown_energies)
        Legend(
            figure[3, :],
            [legend_handles[energy] for energy in shown_energies],
            [
                "$(uppercasefirst(energy)) component"
                for energy in shown_energies
            ];
            orientation = :horizontal,
            nbanks = 1,
            halign = :center,
            tellwidth = false,
        )
    end
    _figure_title!(
        figure,
        "Scale-metric row-space versus null-space energy",
        "Metric row and null fractions sum to one; thin lines pair the same candidate and transition. Projection blockers are reported verbatim.",
    )
    figure
end

function _theme()
    Theme(
        font = "TeX Gyre Termes Makie",
        fontsize = 15,
        Axis = (
            xgridvisible = false,
            ygridcolor = (:gray70, 0.32),
            topspinevisible = false,
            rightspinevisible = false,
            titlesize = 17,
        ),
        Legend = (
            framevisible = false,
            labelsize = 12,
        ),
    )
end

"""
    generate_figures(run_dir)

Generate the six dual-transport study figures as both high-resolution PNG and
vector PDF files under `run_dir/figures`. Missing stage data produce annotated
placeholder panels rather than suppressing a required figure.
"""
function generate_figures(run_dir::AbstractString)
    run_dir = abspath(run_dir)
    raw_dir = joinpath(run_dir, "raw")
    output_dir = joinpath(run_dir, "figures")
    diagnostic_rows =
        _read_csv(joinpath(raw_dir, "residual_diagnostics.csv"))
    action_rows = _read_csv(joinpath(raw_dir, "residual_action.csv"))
    replay_rows = _read_csv(joinpath(raw_dir, "replay.csv"))
    damping_rows = _read_csv(joinpath(raw_dir, "damping.csv"))
    projection_path = joinpath(raw_dir, "projection.csv")
    projection_rows = _read_csv(projection_path)

    figures = with_theme(_theme()) do
        (
            (
                "figure_1_identity_vs_structured_r0",
                _identity_structured_figure(diagnostic_rows),
            ),
            (
                "figure_2_block_action_alignment",
                _block_alignment_figure(action_rows),
            ),
            (
                "figure_3_convergence_iterations_by_policy",
                _convergence_iterations_figure(replay_rows),
            ),
            (
                "figure_4_early_vs_late",
                _early_late_figure(diagnostic_rows),
            ),
            (
                "figure_5_gamma_sweep",
                _gamma_figure(damping_rows),
            ),
            (
                "figure_6_row_null_energy",
                _row_null_figure(
                    projection_rows,
                    projection_path,
                ),
            ),
        )
    end

    paths = String[]
    for (stem, figure) in figures
        append!(paths, _save_versions(output_dir, stem, figure))
    end
    paths
end

end
