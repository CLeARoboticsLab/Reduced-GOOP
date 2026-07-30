module SelectiveWarmstartFigures

using CairoMakie
using Statistics: median, quantile

if !isdefined(Main, :SelectiveWarmstartStudy)
    Base.include(
        Main,
        joinpath(@__DIR__, "SelectiveWarmstartStudy.jl"),
    )
end
const SWS = Main.SelectiveWarmstartStudy

const MODE_ORDER = [
    "primal_only",
    "equality_duals",
    "all_except_innermost_stationarity",
    "all_duals",
]
const MODE_LABELS = [
    "Primal",
    "Primal + λ",
    "Primal + λ + ψout",
    "All duals",
]
const MODE_COLORS = [
    Makie.wong_colors()[6],
    Makie.wong_colors()[2],
    Makie.wong_colors()[1],
    Makie.wong_colors()[4],
]
const FORMULATIONS = ["reduced", "quasi"]

function _symmetric_offsets(count::Integer, half_width::Real = 0.12)
    count >= 0 ||
        throw(ArgumentError("Offset count must be nonnegative; got $(count)."))
    count == 0 && return Float64[]
    count == 1 && return [0.0]
    collect(
        range(
            -Float64(half_width),
            Float64(half_width);
            length = count,
        ),
    )
end

function _finite(row, name)
    value = get(row, name, "")
    isempty(value) && return nothing
    parsed = tryparse(Float64, value)
    isnothing(parsed) || isfinite(parsed) || return nothing
    parsed
end

_true(row, name) = lowercase(get(row, name, "")) == "true"
_solved(row) = _true(row, "direct_converged")

function _empty_axis!(axis, message = "No valid observations")
    text!(axis, 0.5, 0.5; text = message, space = :relative, align = (:center, :center))
    hidespines!(axis)
    hidedecorations!(axis)
end

function _paired_index(rows, formulation, metric; require_solved = false)
    indexed = Dict{String, Dict{String, Float64}}()
    for row in rows
        get(row, "formulation", "") == formulation || continue
        _true(row, "valid_reference_pair") || continue
        require_solved && !_solved(row) && continue
        value = _finite(row, metric)
        isnothing(value) && continue
        key = join(
            (
                get(row, "scenario_seed", ""),
                get(row, "transition", ""),
            ),
            "|",
        )
        indexed_mode = get!(indexed, key, Dict{String, Float64}())
        indexed_mode[get(row, "mode", "")] = value
    end
    if require_solved
        filter!(
            pair -> all(mode -> haskey(last(pair), mode), MODE_ORDER),
            indexed,
        )
    end
    indexed
end

function _paired_mode_figure(rows, metric, ylabel, stem; require_solved = false)
    figure = Figure(size = (1100, 430))
    for (panel, formulation) in enumerate(FORMULATIONS)
        axis = Axis(
            figure[1, panel];
            title = uppercasefirst(formulation) * " GOOP",
            xlabel = "Warm-start information retained",
            ylabel = panel == 1 ? ylabel : "",
            xticks = (1:4, MODE_LABELS),
            xticklabelrotation = π / 9,
            yscale =
                metric == "initial_residual_normalized" ? log10 : identity,
        )
        indexed = _paired_index(
            rows,
            formulation,
            metric;
            require_solved,
        )
        any_observations = false
        for mode_values in Base.values(indexed)
            xs = Int[]
            ys = Float64[]
            for (position, mode) in enumerate(MODE_ORDER)
                haskey(mode_values, mode) || continue
                value = mode_values[mode]
                metric == "initial_residual_normalized" && value <= 0.0 && continue
                push!(xs, position)
                push!(ys, value)
            end
            length(xs) >= 2 || continue
            lines!(axis, xs, ys; color = (:gray45, 0.18), linewidth = 0.8)
            any_observations = true
        end
        for (position, mode) in enumerate(MODE_ORDER)
            values = Float64[
                modes[mode] for modes in Base.values(indexed) if
                haskey(modes, mode) &&
                (
                    metric != "initial_residual_normalized" ||
                    modes[mode] > 0.0
                )
            ]
            isempty(values) && continue
            low, center, high =
                quantile(values, 0.25), median(values), quantile(values, 0.75)
            rangebars!(
                axis,
                [position],
                [low],
                [high];
                color = MODE_COLORS[position],
                linewidth = 4,
            )
            scatter!(
                axis,
                [position],
                [center];
                color = MODE_COLORS[position],
                strokecolor = :black,
                strokewidth = 0.8,
                markersize = 13,
            )
            any_observations = true
        end
        !any_observations && _empty_axis!(axis)
    end
    Label(
        figure[0, :],
        stem == "paired_initial_residual" ?
        "Paired destination initialization residuals" :
        "Paired Newton iteration counts";
        fontsize = 20,
        font = :bold,
    )
    figure
end

function _shift_quality_figure(rows)
    figure = Figure(size = (1000, 430))
    blocks = [
        ("shift_quality_lambda", "λ"),
        ("shift_quality_psi_out", "ψout"),
        ("shift_quality_psi_in", "ψin"),
    ]
    for (panel, formulation) in enumerate(FORMULATIONS)
        axis = Axis(
            figure[1, panel];
            title = uppercasefirst(formulation) * " GOOP",
            xlabel = "Transported block",
            ylabel = panel == 1 ? "Shift-quality ratio qᵦ" : "",
            xticks = (1:3, last.(blocks)),
            yscale = log10,
        )
        hlines!(
            axis,
            [1.0];
            color = (:black, 0.55),
            linestyle = :dash,
            linewidth = 1.2,
        )
        any_observations = false
        # q is a property of the source/destination pair, so deduplicate the
        # identical value repeated across the four replay modes.
        seen = Set{String}()
        for (position, (column, _)) in enumerate(blocks)
            values = Float64[]
            for row in rows
                get(row, "formulation", "") == formulation || continue
                _true(row, "valid_reference_pair") || continue
                key = join(
                    (
                        formulation,
                        get(row, "scenario_seed", ""),
                        get(row, "transition", ""),
                        column,
                    ),
                    "|",
                )
                key in seen && continue
                value = _finite(row, column)
                isnothing(value) && continue
                value > 0.0 || continue
                push!(seen, key)
                push!(values, value)
            end
            isempty(values) && continue
            jitter = _symmetric_offsets(length(values))
            scatter!(
                axis,
                position .+ jitter,
                values;
                color = (:steelblue, 0.35),
                markersize = 5,
            )
            scatter!(
                axis,
                [position],
                [median(values)];
                color = :darkorange,
                strokecolor = :black,
                strokewidth = 0.8,
                markersize = 13,
            )
            any_observations = true
        end
        !any_observations && _empty_axis!(axis)
    end
    Label(
        figure[0, :],
        "How well each dual block tracks the next MPC solution";
        fontsize = 20,
        font = :bold,
    )
    figure
end

function _scaling_figure(rows)
    figure = Figure(size = (1050, 450))
    base_values = Dict{String, Vector{Float64}}()
    for row in rows
        get(row, "mode", "") == "all_duals" || continue
        baseline = _finite(row, "baseline_initial_residual_norm2")
        isnothing(baseline) && continue
        key = join(
            (
                get(row, "formulation", ""),
                get(row, "scenario_seed", ""),
                get(row, "direction", ""),
            ),
            "|",
        )
        push!(get!(base_values, key, Float64[]), baseline)
    end
    base_resolution =
        Dict(key => median(values) for (key, values) in base_values)
    for (panel, formulation) in enumerate(FORMULATIONS)
        axis = Axis(
            figure[1, panel];
            title = uppercasefirst(formulation) * " GOOP",
            xlabel = "Initial-state perturbation ε",
            ylabel = panel == 1 ? "Normalized initial residual r₀" : "",
            xscale = log10,
            yscale = log10,
        )
        any_observations = false
        for (mode_index, mode) in enumerate(MODE_ORDER)
            grouped = Dict{
                String,
                Vector{Tuple{Float64, Float64, Bool, Bool}},
            }()
            for row in rows
                get(row, "formulation", "") == formulation || continue
                get(row, "mode", "") == mode || continue
                epsilon = _finite(row, "epsilon")
                r0 = _finite(row, "initial_residual_normalized")
                R0 = _finite(row, "initial_residual_norm2")
                baseline =
                    _finite(row, "baseline_initial_residual_norm2")
                perturbed_reference =
                    something(
                        _finite(
                            row,
                            "perturbed_reference_residual",
                        ),
                        0.0,
                    )
                isnothing(epsilon) && continue
                isnothing(r0) && continue
                isnothing(R0) && continue
                isnothing(baseline) && continue
                epsilon > 0.0 && r0 > 0.0 || continue
                key = join(
                    (
                        get(row, "scenario_seed", ""),
                        get(row, "direction", ""),
                    ),
                    "|",
                )
                resolution_key = join(
                    (
                        formulation,
                        get(row, "scenario_seed", ""),
                        get(row, "direction", ""),
                    ),
                    "|",
                )
                resolution = max(
                    get(base_resolution, resolution_key, 0.0),
                    perturbed_reference,
                    100 * eps(Float64),
                )
                structural = baseline > 1.25 * resolution
                reliable = R0 > 1.25 * resolution || structural
                push!(
                    get!(
                        grouped,
                        key,
                        Tuple{Float64, Float64, Bool, Bool}[],
                    ),
                    (
                        epsilon,
                        r0,
                        reliable,
                        structural,
                    ),
                )
            end
            labelled = false
            for group_values in Base.values(grouped)
                sort!(group_values; by = first)
                x = first.(group_values)
                y = getindex.(group_values, 2)
                reliable = getindex.(group_values, 3)
                structural = getindex.(group_values, 4)
                lines!(
                    axis,
                    x,
                    y;
                    color = (MODE_COLORS[mode_index], 0.55),
                    linewidth = 1.4,
                    label = labelled ? nothing : MODE_LABELS[mode_index],
                )
                labelled = true
                resolved_perturbation = reliable .& .!structural
                if any(resolved_perturbation)
                    scatter!(
                        axis,
                        x[resolved_perturbation],
                        y[resolved_perturbation];
                        color = MODE_COLORS[mode_index],
                        markersize = 7,
                    )
                end
                if any(structural)
                    scatter!(
                        axis,
                        x[structural],
                        y[structural];
                        color = MODE_COLORS[mode_index],
                        marker = :rect,
                        markersize = 7,
                    )
                end
                if any(.!reliable)
                    scatter!(
                        axis,
                        x[.!reliable],
                        y[.!reliable];
                        color = :transparent,
                        strokecolor = MODE_COLORS[mode_index],
                        strokewidth = 1.2,
                        markersize = 7,
                    )
                end
                any_observations = true
            end
        end
        if any_observations
            axislegend(axis; position = :lt, framevisible = false)
        else
            _empty_axis!(axis)
        end
    end
    Label(
        figure[0, :],
        "Small-perturbation scaling (squares: structural floor; open: unresolved)";
        fontsize = 20,
        font = :bold,
    )
    figure
end

function _residual_iteration_figure(rows)
    figure = Figure(size = (1000, 430))
    for (panel, formulation) in enumerate(FORMULATIONS)
        axis = Axis(
            figure[1, panel];
            title = uppercasefirst(formulation) * " GOOP",
            xlabel = "Normalized initial residual r₀",
            ylabel = panel == 1 ? "Newton iterations" : "",
            xscale = log10,
        )
        any_observations = false
        for (mode_index, mode) in enumerate(MODE_ORDER)
            x = Float64[]
            y = Float64[]
            for row in rows
                get(row, "formulation", "") == formulation || continue
                get(row, "mode", "") == mode || continue
                _true(row, "valid_reference_pair") || continue
                _solved(row) || continue
                r0 = _finite(row, "initial_residual_normalized")
                iterations = _finite(row, "total_inner_iters")
                isnothing(r0) && continue
                isnothing(iterations) && continue
                r0 > 0.0 || continue
                push!(x, r0)
                push!(y, iterations)
            end
            isempty(x) && continue
            scatter!(
                axis,
                x,
                y;
                color = (MODE_COLORS[mode_index], 0.65),
                markersize = 8,
                label = MODE_LABELS[mode_index],
            )
            any_observations = true
        end
        if any_observations
            axislegend(axis; position = :lt, framevisible = false)
        else
            _empty_axis!(axis)
        end
    end
    Label(
        figure[0, :],
        "Initial KKT distance and subsequent solver effort";
        fontsize = 20,
        font = :bold,
    )
    figure
end

function _sensitivity_figure(rows)
    figure = Figure(size = (1000, 700))
    block_order = ["psi_out", "psi_in"]
    block_labels = ["ψout", "ψin"]
    for (panel, formulation) in enumerate(FORMULATIONS)
        raw_axis = Axis(
            figure[1, panel];
            title = uppercasefirst(formulation) * " GOOP",
            ylabel =
                panel == 1 ?
                "‖J[:, b]‖F / √dim(b)" : "",
            xticks = (1:2, block_labels),
        )
        response_axis = Axis(
            figure[2, panel];
            xlabel = "Stationarity-multiplier block",
            ylabel =
                panel == 1 ?
                "‖K(y+a d)-K(y)‖₂ / a" : "",
            xticks = (1:2, block_labels),
            yscale = log10,
        )
        any_raw = false
        any_response = false
        for (position, block) in enumerate(block_order)
            selected = filter(
                row ->
                    get(row, "formulation", "") == formulation &&
                    get(row, "block", "") == block &&
                    _true(row, "reference_accepted"),
                rows,
            )
            # One Jacobian statistic per reference (it is repeated by amplitude).
            raw_by_reference = Dict{String, Float64}()
            amplitudes = Float64[]
            for row in selected
                amplitude = _finite(row, "amplitude")
                !isnothing(amplitude) && push!(amplitudes, amplitude)
            end
            minimum_amplitude =
                isempty(amplitudes) ? nothing : minimum(amplitudes)
            response = Float64[]
            for row in selected
                key = join(
                    (
                        get(row, "scenario_seed", ""),
                        get(row, "step", ""),
                    ),
                    "|",
                )
                raw = _finite(
                    row,
                    "raw_frobenius_per_sqrt_column",
                )
                !isnothing(raw) && (raw_by_reference[key] = raw)
                amplitude = _finite(row, "amplitude")
                directional = _finite(
                    row,
                    "directional_residual_change_per_amplitude",
                )
                if !isnothing(minimum_amplitude) &&
                   amplitude == minimum_amplitude &&
                   !isnothing(directional) &&
                   directional > 0.0
                    push!(response, directional)
                end
            end
            raw_values = collect(values(raw_by_reference))
            if !isempty(raw_values)
                scatter!(
                    raw_axis,
                    fill(position, length(raw_values)),
                    raw_values;
                    color = (:steelblue, 0.4),
                    markersize = 7,
                )
                scatter!(
                    raw_axis,
                    [position],
                    [median(raw_values)];
                    color = :darkorange,
                    strokecolor = :black,
                    strokewidth = 0.8,
                    markersize = 13,
                )
                any_raw = true
            end
            if !isempty(response)
                scatter!(
                    response_axis,
                    fill(position, length(response)),
                    response;
                    color = (:steelblue, 0.4),
                    markersize = 7,
                )
                scatter!(
                    response_axis,
                    [position],
                    [median(response)];
                    color = :darkorange,
                    strokecolor = :black,
                    strokewidth = 0.8,
                    markersize = 13,
                )
                any_response = true
            end
        end
        !any_raw && _empty_axis!(raw_axis)
        !any_response && _empty_axis!(response_axis)
    end
    Label(
        figure[0, :],
        "Outer versus innermost stationarity-multiplier sensitivity";
        fontsize = 20,
        font = :bold,
    )
    figure
end

function _save_versions(output_dir, stem, figure)
    mkpath(output_dir)
    save(joinpath(output_dir, stem * ".png"), figure; px_per_unit = 2)
    save(joinpath(output_dir, stem * ".pdf"), figure)
end

function generate_figures(run_dir::AbstractString)
    set_theme!(
        Theme(
            font = "TeX Gyre Termes Makie",
            fontsize = 15,
            Axis = (
                xgridvisible = false,
                ygridcolor = (:gray70, 0.35),
                topspinevisible = false,
                rightspinevisible = false,
            ),
        ),
    )
    raw_dir = joinpath(run_dir, "raw")
    output_dir = joinpath(run_dir, "figures")
    replay = SWS.read_csv_rows(joinpath(raw_dir, "replay.csv"))
    scaling = SWS.read_csv_rows(joinpath(raw_dir, "scaling.csv"))
    sensitivity = SWS.read_csv_rows(joinpath(raw_dir, "sensitivity.csv"))
    figures = (
        (
            "paired_initial_residual",
            _paired_mode_figure(
                replay,
                "initial_residual_normalized",
                "Normalized initial residual r₀",
                "paired_initial_residual",
            ),
        ),
        (
            "paired_iterations",
            _paired_mode_figure(
                replay,
                "total_inner_iters",
                "Newton iterations",
                "paired_iterations";
                require_solved = true,
            ),
        ),
        ("shift_quality", _shift_quality_figure(replay)),
        ("small_perturbation_scaling", _scaling_figure(scaling)),
        (
            "residual_vs_iterations",
            _residual_iteration_figure(replay),
        ),
        (
            "stationarity_sensitivity",
            _sensitivity_figure(sensitivity),
        ),
    )
    for (stem, figure) in figures
        _save_versions(output_dir, stem, figure)
    end
    [joinpath(output_dir, stem * extension) for
     (stem, _) in figures for extension in (".png", ".pdf")]
end

end
