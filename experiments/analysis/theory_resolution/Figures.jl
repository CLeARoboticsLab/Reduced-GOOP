module TheoryResolutionFigures

using CairoMakie
using Statistics: median, quantile

export generate_figures, validate_figure_schemas

const CsvRow = Dict{String, String}
const FORMULATIONS = ("reduced", "quasi")
const TRANSPORTS = (
    "identity_copy",
    "stage_shift_zero_tail",
    "stage_shift_hold_tail",
)
const SCALING_TRANSPORTS = (
    "identity_copy",
    "stage_shift_zero_tail",
    "stage_shift_hold_tail",
    "diagnostic_oracle_terminal_completion",
)
const GAMMAS = (0.0, 0.25, 1.0)
const PLOT_FLOOR = 1e-14

const WONG = Makie.wong_colors()
const FORMULATION_COLOR = Dict(
    "reduced" => WONG[1],
    "quasi" => WONG[2],
)
const TRANSPORT_COLOR = Dict(
    "identity_copy" => WONG[1],
    "stage_shift_zero_tail" => WONG[2],
    "stage_shift_hold_tail" => WONG[4],
    "diagnostic_oracle_terminal_completion" => WONG[6],
)
const TRANSPORT_LABEL = Dict(
    "identity_copy" => "Fixed index",
    "stage_shift_zero_tail" => "Semantic, zero tail",
    "stage_shift_hold_tail" => "Semantic, hold tail",
    "diagnostic_oracle_terminal_completion" => "Diagnostic oracle terminal",
)
const TRANSPORT_SHORT = Dict(
    "identity_copy" => "fixed",
    "stage_shift_zero_tail" => "shift–zero",
    "stage_shift_hold_tail" => "shift–hold",
)
const TRANSPORT_MARKER = Dict(
    "identity_copy" => :circle,
    "stage_shift_zero_tail" => :rect,
    "stage_shift_hold_tail" => :utriangle,
    "diagnostic_oracle_terminal_completion" => :diamond,
)
const FAMILY_PREFERENCE = (
    "outer_stationarity",
    "innermost_stationarity",
    "stationarity",
    "equality",
    "inequality",
    "complementarity",
    "other",
)
const REQUIRED_SCHEMAS = Dict(
    "production_stagewise_residuals.csv" => (
        "formulation",
        "scenario_seed",
        "transition",
        "dual_transport",
        "equation_family",
        "physical_stage",
        "coordinate_count",
        "residual_norm2",
    ),
    "boundary_ablations.csv" => (
        "formulation",
        "scenario_seed",
        "transition",
        "interior_map",
        "initial_completion",
        "terminal_completion",
        "explicit_terminal_completion",
        "interior_norm2",
        "initial_norm2",
        "terminal_norm2",
    ),
    "gamma_group_summary.csv" => (
        "sample_group",
        "formulation",
        "gamma",
        "total_cases",
        "convergence_rate",
        "jointly_converged_cases",
        "aligned_joint_cases",
        "median_iterations_joint",
    ),
    "globalization_directions.csv" => (
        "case_id",
        "formulation",
        "scenario_seed",
        "transition",
        "gamma",
        "total_iter",
        "raw_residual_norm2",
        "eta",
        "numerical_rank_estimate",
        "row_dimension",
        "rrqr_condition_proxy",
    ),
    "globalization_iterations.csv" => (
        "case_id",
        "formulation",
        "scenario_seed",
        "transition",
        "gamma",
        "total_iter",
        "accepted_alpha",
        "reduction_ratio",
    ),
    "globalization_case_summary.csv" => (
        "case_id",
        "direct_converged",
    ),
    "scaling_benchmark.csv" => (
        "formulation",
        "epsilon",
        "transport",
        "interior_residual_norm2",
        "initial_residual_norm2",
        "terminal_residual_norm2",
        "global_residual_norm2",
    ),
    "scaling_benchmark_slopes.csv" => (
        "formulation",
        "transport",
        "metric",
        "primary_slope",
    ),
)

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
                if next_index <= lastindex(line) && line[next_index] == '"'
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
    !isempty(header) &&
        (header[1] = replace(header[1], '\ufeff' => ""))
    rows = CsvRow[]
    for (offset, line) in enumerate(Iterators.drop(lines, 1))
        isempty(strip(line)) && continue
        fields = _split_csv_line(line, path, offset + 1)
        length(fields) == length(header) || error(
            "CSV field-count mismatch at $(path):$(offset + 1): " *
            "expected $(length(header)), found $(length(fields)).",
        )
        push!(rows, CsvRow(header .=> fields))
    end
    rows
end

function _token(row, column)
    value = lowercase(strip(get(row, column, "")))
    startswith(value, ":") ? value[2:end] : value
end

function _number(row, column)
    value = strip(get(row, column, ""))
    isempty(value) && return nothing
    parsed = tryparse(Float64, value)
    isnothing(parsed) || !isfinite(parsed) ? nothing : parsed
end

function _integer(row, column)
    value = _number(row, column)
    isnothing(value) && return nothing
    rounded = round(Int, value)
    abs(value - rounded) <= 100eps(max(abs(value), 1.0)) ? rounded : nothing
end

_true(row, column) = _token(row, column) in ("true", "1", "yes")
_case_key(row) = (
    _token(row, "formulation"),
    get(row, "scenario_seed", ""),
    get(row, "transition", ""),
)

"""
    validate_figure_schemas(run_dir; throw_on_error = false)

Check only the columns consumed by the publication figures. Missing files and
columns are reported rather than assumed; `generate_figures` remains tolerant
and emits an explicitly labelled empty panel for unavailable phases.
"""
function validate_figure_schemas(
    run_dir::AbstractString;
    throw_on_error::Bool = false,
)
    raw_dir = joinpath(abspath(run_dir), "raw")
    tables = Dict{String, NamedTuple}()
    for (filename, required) in sort!(collect(REQUIRED_SCHEMAS); by = first)
        path = joinpath(raw_dir, filename)
        exists = isfile(path)
        rows = exists ? _read_csv(path) : CsvRow[]
        columns = if !isempty(rows)
            Set(keys(first(rows)))
        elseif exists
            lines = readlines(path)
            isempty(lines) ? Set{String}() :
            Set(_split_csv_line(first(lines), path, 1))
        else
            Set{String}()
        end
        missing_columns =
            sort!(String[column for column in required if !(column in columns)])
        ok = exists && isempty(missing_columns)
        tables[filename] = (;
            path,
            exists,
            row_count = length(rows),
            missing_columns,
            ok,
        )
    end
    valid = all(table.ok for table in values(tables))
    if throw_on_error && !valid
        failures = String[]
        for (filename, table) in sort!(collect(tables); by = first)
            table.ok && continue
            reason = !table.exists ? "missing file" :
                     "missing columns: $(join(table.missing_columns, ", "))"
            push!(failures, "$(filename) ($(reason))")
        end
        error("Theory-resolution figure schema validation failed: " *
              join(failures, "; "))
    end
    (; valid, raw_dir, tables)
end

function _empty_axis!(axis, message)
    text!(
        axis,
        0.5,
        0.5;
        text = message,
        space = :relative,
        align = (:center, :center),
        fontsize = 13,
        color = :gray35,
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
        fontsize = 12,
        color = :gray30,
        padding = (0, 0, 0, 6),
    )
    rowgap!(header, 0)
    nothing
end

function _save_versions(output_dir, stem, figure)
    mkpath(output_dir)
    png = joinpath(output_dir, stem * ".png")
    pdf = joinpath(output_dir, stem * ".pdf")
    save(png, figure; px_per_unit = 2)
    save(pdf, figure)
    (png, pdf)
end

function _ordered_families(rows)
    observed = Set(
        _token(row, "equation_family") for row in rows if
        !isempty(_token(row, "equation_family"))
    )
    ordered = String[
        family for family in FAMILY_PREFERENCE if family in observed
    ]
    append!(ordered, sort!(collect(setdiff(observed, Set(ordered)))))
    ordered
end

function _family_label(family)
    family == "outer_stationarity" && return "outer stationarity"
    family == "innermost_stationarity" && return "inner stationarity"
    replace(family, "_" => " ")
end

function _aggregate_stage_family(rows, stages, families)
    sums = Dict{Tuple{Int, String}, Float64}()
    counts = Dict{Tuple{Int, String}, Int}()
    for row in rows
        stage = _integer(row, "physical_stage")
        family = _token(row, "equation_family")
        norm2 = _number(row, "residual_norm2")
        count = _integer(row, "coordinate_count")
        (
            isnothing(stage) ||
            isnothing(norm2) ||
            isnothing(count) ||
            count <= 0 ||
            !(stage in stages) ||
            !(family in families)
        ) && continue
        key = (stage, family)
        sums[key] = get(sums, key, 0.0) + norm2^2
        counts[key] = get(counts, key, 0) + count
    end
    matrix = fill(NaN, length(stages), length(families))
    for (stage_index, stage) in enumerate(stages)
        for (family_index, family) in enumerate(families)
            key = (stage, family)
            get(counts, key, 0) > 0 || continue
            matrix[stage_index, family_index] =
                sqrt(sums[key] / counts[key])
        end
    end
    matrix
end

function _aggregate_stage(rows, stages)
    sums = zeros(Float64, length(stages))
    counts = zeros(Int, length(stages))
    stage_index = Dict(stage => index for (index, stage) in enumerate(stages))
    for row in rows
        stage = _integer(row, "physical_stage")
        norm2 = _number(row, "residual_norm2")
        count = _integer(row, "coordinate_count")
        (
            isnothing(stage) ||
            isnothing(norm2) ||
            isnothing(count) ||
            count <= 0 ||
            !haskey(stage_index, stage)
        ) && continue
        index = stage_index[stage]
        sums[index] += norm2^2
        counts[index] += count
    end
    [
        counts[index] == 0 ? NaN : sqrt(sums[index] / counts[index]) for
        index in eachindex(stages)
    ]
end

function _log_heatmap_data(matrices)
    positive = Float64[
        value for matrix in matrices for value in matrix if
        isfinite(value) && value > 0.0
    ]
    floor_value =
        isempty(positive) ? PLOT_FLOOR :
        max(PLOT_FLOOR, minimum(positive) * 1e-2)
    transformed = [
        map(matrix) do value
            isfinite(value) ? log10(max(value, floor_value)) : NaN
        end for matrix in matrices
    ]
    finite_logs = Float64[
        value for matrix in transformed for value in matrix if isfinite(value)
    ]
    colorrange = if isempty(finite_logs)
        (-14.0, -13.0)
    else
        low = quantile(finite_logs, 0.02)
        high = quantile(finite_logs, 0.98)
        high <= low + 0.5 ? (low - 0.25, low + 0.75) : (low, high)
    end
    (; transformed, colorrange, floor_value)
end

function _stage_family_heatmap(rows)
    figure = Figure(size = (1500, 820))
    stages = collect(1:20)
    families = _ordered_families(rows)
    isempty(families) && (families = ["unavailable"])
    selections = [
        filter(
            row ->
                _token(row, "formulation") == formulation &&
                _token(row, "dual_transport") == transport,
            rows,
        ) for formulation in FORMULATIONS for transport in TRANSPORTS
    ]
    matrices = [
        _aggregate_stage_family(selection, stages, families) for
        selection in selections
    ]
    transformed = _log_heatmap_data(matrices)
    axes = Axis[]
    index = 0
    for (form_index, formulation) in enumerate(FORMULATIONS)
        for (transport_index, transport) in enumerate(TRANSPORTS)
            index += 1
            axis = Axis(
                figure[form_index, transport_index];
                title = "$(uppercasefirst(formulation)): " *
                        TRANSPORT_SHORT[transport],
                xlabel = form_index == length(FORMULATIONS) ?
                         "physical stage" : "",
                ylabel = transport_index == 1 ? "equation family" : "",
                xticks = (1:2:20, string.(1:2:20)),
                yticks = (
                    1:length(families),
                    _family_label.(families),
                ),
                yticklabelsvisible = transport_index == 1,
            )
            push!(axes, axis)
            if isempty(selections[index])
                _empty_axis!(
                    axis,
                    "No production stagewise rows\nfor this panel",
                )
                continue
            end
            heatmap!(
                axis,
                stages,
                1:length(families),
                transformed.transformed[index];
                colormap = :viridis,
                colorrange = transformed.colorrange,
                nan_color = :gray92,
            )
            vlines!(
                axis,
                [1.5, 19.5];
                color = (:white, 0.6),
                linestyle = :dash,
                linewidth = 1,
            )
        end
    end
    Colorbar(
        figure[1:2, 4];
        colormap = :viridis,
        limits = transformed.colorrange,
        label = "log₁₀ coordinate RMS residual",
    )
    _figure_title!(
        figure,
        "Stage- and family-resolved transported KKT residuals",
        "All 17 valid replay transitions; dashed boundaries separate the new " *
        "initial and terminal stages.",
    )
    figure
end

function _spike_heatmap(rows)
    selected = filter(rows) do row
        _token(row, "formulation") == "reduced" &&
        _integer(row, "scenario_seed") == 202 &&
        something(_integer(row, "transition"), -1) in 4:7 &&
        _token(row, "dual_transport") in TRANSPORTS
    end
    stages = collect(1:20)
    row_specs = [
        (transition, transport) for transition in 4:7 for
        transport in TRANSPORTS
    ]
    matrices = Matrix{Float64}[]
    values = fill(NaN, length(stages), length(row_specs))
    for (row_index, (transition, transport)) in enumerate(row_specs)
        group = filter(
            row ->
                _integer(row, "transition") == transition &&
                _token(row, "dual_transport") == transport,
            selected,
        )
        values[:, row_index] .= _aggregate_stage(group, stages)
    end
    push!(matrices, values)
    transformed = _log_heatmap_data(matrices)
    figure = Figure(size = (1420, 670))
    axis = Axis(
        figure[1, 1];
        xlabel = "physical stage",
        ylabel = "transition and transported dual completion",
        xticks = (1:20, string.(1:20)),
        yticks = (
            1:length(row_specs),
            [
                "t$(transition) · $(TRANSPORT_SHORT[transport])" for
                (transition, transport) in row_specs
            ],
        ),
    )
    if isempty(selected)
        _empty_axis!(
            axis,
            "No reduced seed-202 transitions 4–7\n" *
            "(raw/production_stagewise_residuals.csv)",
        )
    else
        heatmap!(
            axis,
            stages,
            1:length(row_specs),
            transformed.transformed[1];
            colormap = :magma,
            colorrange = transformed.colorrange,
            nan_color = :gray92,
        )
        for boundary in (3.5, 6.5, 9.5)
            hlines!(axis, [boundary]; color = (:white, 0.72), linewidth = 1)
        end
        vlines!(axis, [1.5, 19.5]; color = (:white, 0.72), linewidth = 1.2)
    end
    Colorbar(
        figure[1, 2];
        colormap = :magma,
        limits = transformed.colorrange,
        label = "log₁₀ coordinate RMS residual",
    )
    _figure_title!(
        figure,
        "Localization of the reduced seed-202 transition 4–7 spikes",
        "Residuals are aggregated over equation families at each physical " *
        "stage; fixed-index copying is shown beside semantic completions.",
    )
    figure
end

function _matches(row, conditions)
    all(_token(row, name) == value for (name, value) in conditions)
end

function _factor_pairs(rows, factor, categories, metric, conditions)
    indexed = Dict{Tuple{String, String, String}, Dict{String, Float64}}()
    for row in rows
        _matches(row, conditions) || continue
        category = _token(row, factor)
        category in categories || continue
        value = _number(row, metric)
        isnothing(value) && continue
        indexed_case = get!(
            indexed,
            _case_key(row),
            Dict{String, Float64}(),
        )
        indexed_case[category] = value
    end
    indexed
end

function _paired_factor_axis!(
    axis,
    rows;
    factor,
    categories,
    labels,
    metric,
    conditions,
)
    indexed = _factor_pairs(rows, factor, categories, metric, conditions)
    isempty(indexed) && begin
        _empty_axis!(axis, "Required diagnostic ablation rows unavailable")
        return false
    end
    for (key, values) in indexed
        present = [
            index for (index, category) in enumerate(categories) if
            haskey(values, category)
        ]
        isempty(present) && continue
        x = Float64.(present)
        y = [max(PLOT_FLOOR, values[categories[index]]) for index in present]
        color = get(FORMULATION_COLOR, first(key), :gray45)
        length(x) >= 2 &&
            lines!(axis, x, y; color = (color, 0.20), linewidth = 0.8)
        scatter!(
            axis,
            x,
            y;
            color = (color, 0.32),
            markersize = 5,
        )
    end
    for (position, category) in enumerate(categories)
        values = Float64[
            max(PLOT_FLOOR, item[category]) for item in Base.values(indexed) if
            haskey(item, category)
        ]
        isempty(values) && continue
        low, center, high =
            quantile(values, 0.25), median(values), quantile(values, 0.75)
        rangebars!(
            axis,
            [position],
            [low],
            [high];
            color = :black,
            linewidth = 3,
        )
        scatter!(
            axis,
            [position],
            [center];
            color = :white,
            strokecolor = :black,
            strokewidth = 1.2,
            markersize = 11,
        )
    end
    axis.xticks = (collect(1:length(categories)), collect(labels))
    true
end

function _boundary_ablation_figure(rows)
    figure = Figure(size = (1540, 770))
    specifications = (
        (
            title = "A  inherited coordinate map",
            factor = "interior_map",
            categories = ("fixed_index", "semantic_shift"),
            labels = ("fixed index", "semantic shift"),
            metric = "interior_norm2",
            ylabel = "‖Finterior‖₂",
            conditions = (
                "initial_completion" => "reset",
                "terminal_completion" => "zero",
                "explicit_terminal_completion" => "reset",
            ),
        ),
        (
            title = "B  initial-condition multiplier",
            factor = "initial_completion",
            categories = ("copy", "reset"),
            labels = ("copy", "reset"),
            metric = "initial_norm2",
            ylabel = "‖Finitial‖₂",
            conditions = (
                "interior_map" => "semantic_shift",
                "terminal_completion" => "zero",
                "explicit_terminal_completion" => "reset",
            ),
        ),
        (
            title = "C  missing-successor tail",
            factor = "terminal_completion",
            categories = ("zero", "hold"),
            labels = ("zero tail", "hold tail"),
            metric = "terminal_norm2",
            ylabel = "‖Fterminal‖₂",
            conditions = (
                "interior_map" => "semantic_shift",
                "initial_completion" => "reset",
                "explicit_terminal_completion" => "reset",
            ),
        ),
        (
            title = "D  explicit terminal coordinates",
            factor = "explicit_terminal_completion",
            categories = ("reset", "copy"),
            labels = ("reset", "copy"),
            metric = "terminal_norm2",
            ylabel = "‖Fterminal‖₂",
            conditions = (
                "interior_map" => "semantic_shift",
                "initial_completion" => "reset",
                "terminal_completion" => "zero",
            ),
        ),
    )
    for (panel, specification) in enumerate(specifications)
        row_index = panel <= 2 ? 1 : 2
        column_index = isodd(panel) ? 1 : 2
        axis = Axis(
            figure[row_index, column_index];
            title = specification.title,
            ylabel = specification.ylabel,
            yscale = log10,
            xticklabelrotation = π / 18,
        )
        _paired_factor_axis!(
            axis,
            rows;
            factor = specification.factor,
            categories = specification.categories,
            labels = specification.labels,
            metric = specification.metric,
            conditions = specification.conditions,
        )
    end
    elements = [
        MarkerElement(
            color = (FORMULATION_COLOR[formulation], 0.55),
            marker = :circle,
            markersize = 9,
        ) for formulation in FORMULATIONS
    ]
    Legend(
        figure[3, :],
        elements,
        ["Reduced cases", "Quasi cases"];
        orientation = :horizontal,
        framevisible = false,
        tellheight = true,
    )
    _figure_title!(
        figure,
        "Factorial localization of transport and boundary effects",
        "Diagnostic copy/reset combinations are ablations—not production " *
        "policies. Thin lines pair the same transition; black marks show " *
        "median and interquartile range.",
    )
    figure
end

function _summary_rows(rows, sample_group, formulation)
    selected = filter(
        row ->
            _token(row, "sample_group") == sample_group &&
            _token(row, "formulation") == formulation,
        rows,
    )
    sort!(
        selected;
        by = row -> something(_number(row, "gamma"), Inf),
    )
end

function _gamma_figure(rows)
    figure = Figure(size = (1510, 790))
    sample_groups = ("development", "holdout", "all")
    sample_labels = (
        "Development (6 cases)",
        "Held out (11 cases)",
        "All valid (17 cases)",
    )
    for (column, (sample_group, sample_label)) in
        enumerate(zip(sample_groups, sample_labels))
        convergence_axis = Axis(
            figure[1, column];
            title = sample_label,
            ylabel = column == 1 ? "convergence rate" : "",
            xlabel = "",
            xticks = (collect(GAMMAS), collect(string.(GAMMAS))),
        )
        iteration_axis = Axis(
            figure[2, column];
            ylabel = column == 1 ?
                     "median Newton iterations\n(jointly converged cases)" : "",
            xlabel = "dual damping γ",
            xticks = (collect(GAMMAS), collect(string.(GAMMAS))),
        )
        any_convergence = false
        any_iterations = false
        annotation = String[]
        for formulation in FORMULATIONS
            selected = _summary_rows(rows, sample_group, formulation)
            gamma = Float64[]
            rates = Float64[]
            iterations = Float64[]
            iteration_gamma = Float64[]
            for row in selected
                current_gamma = _number(row, "gamma")
                rate = _number(row, "convergence_rate")
                if !isnothing(current_gamma) && !isnothing(rate)
                    push!(gamma, current_gamma)
                    push!(rates, rate)
                end
                value = _number(row, "median_iterations_joint")
                if !isnothing(current_gamma) && !isnothing(value)
                    push!(iteration_gamma, current_gamma)
                    push!(iterations, value)
                end
            end
            color = FORMULATION_COLOR[formulation]
            if !isempty(gamma)
                lines!(
                    convergence_axis,
                    gamma,
                    rates;
                    color,
                    linewidth = 2.2,
                )
                scatter!(
                    convergence_axis,
                    gamma,
                    rates;
                    color,
                    marker = formulation == "reduced" ? :circle : :rect,
                    markersize = 10,
                )
                any_convergence = true
            end
            if !isempty(iteration_gamma)
                lines!(
                    iteration_axis,
                    iteration_gamma,
                    iterations;
                    color,
                    linewidth = 2.2,
                )
                scatter!(
                    iteration_axis,
                    iteration_gamma,
                    iterations;
                    color,
                    marker = formulation == "reduced" ? :circle : :rect,
                    markersize = 10,
                )
                any_iterations = true
            end
            gamma_quarter = findfirst(
                row -> _number(row, "gamma") == 0.25,
                selected,
            )
            if !isnothing(gamma_quarter)
                row = selected[gamma_quarter]
                joint = _integer(row, "jointly_converged_cases")
                aligned = _integer(row, "aligned_joint_cases")
                total = _integer(row, "total_cases")
                if !isnothing(joint) && !isnothing(aligned) &&
                   !isnothing(total)
                    push!(
                        annotation,
                        "$(uppercasefirst(formulation)): " *
                        "joint $(joint)/$(total), aligned $(aligned)/$(total)",
                    )
                end
            end
        end
        ylims!(convergence_axis, -0.03, 1.05)
        any_convergence ||
            _empty_axis!(convergence_axis, "Group summary unavailable")
        any_iterations ||
            _empty_axis!(iteration_axis, "No jointly converged comparison")
        isempty(annotation) || text!(
            iteration_axis,
            0.03,
            0.97;
            text = join(annotation, "\n"),
            space = :relative,
            align = (:left, :top),
            fontsize = 10,
            color = :gray30,
        )
    end
    elements = [
        LineElement(
            color = FORMULATION_COLOR[formulation],
            linewidth = 2,
        ) for formulation in FORMULATIONS
    ]
    Legend(
        figure[3, :],
        elements,
        ["Reduced", "Quasi"];
        orientation = :horizontal,
        framevisible = false,
    )
    _figure_title!(
        figure,
        "Preregistered γ = 0.25 safeguard: development versus holdout",
        "Iteration summaries are conditioned on joint convergence across " *
        "γ ∈ {0, 0.25, 1}; aligned counts additionally require the fixed " *
        "10⁻³ normalized final-primal separation threshold.",
    )
    figure
end

function _hard_case_key(row)
    (
        _token(row, "formulation"),
        something(_integer(row, "scenario_seed"), -1),
        something(_integer(row, "transition"), -1),
    )
end

function _hard_case_label(key)
    formulation, seed, transition = key
    prefix = formulation == "reduced" ? "R" :
             formulation == "quasi" ? "Q" :
             uppercasefirst(formulation)
    "$(prefix) · seed $(seed) · t$(transition)"
end

function _gamma_color(gamma)
    colors = Makie.to_colormap(:viridis)
    index = clamp(round(Int, 1 + gamma * (length(colors) - 1)), 1, length(colors))
    colors[index]
end

function _group_by_case(rows)
    grouped = Dict{String, Vector{CsvRow}}()
    for row in rows
        case_id = get(row, "case_id", "")
        isempty(case_id) && continue
        push!(get!(grouped, case_id, CsvRow[]), row)
    end
    grouped
end

function _plot_trace_metric!(
    axis,
    grouped,
    metric;
    transform = identity,
    positive_only = false,
    case_styles = Dict{Tuple{String, Int, Int}, Any}(),
    summary = Dict{String, Bool}(),
)
    plotted = false
    for (case_id, rows) in sort!(collect(grouped); by = first)
        sort!(
            rows;
            by = row -> (
                something(_integer(row, "total_iter"), typemax(Int)),
                something(_integer(row, "direction_attempt"), 0),
            ),
        )
        gamma = _number(first(rows), "gamma")
        isnothing(gamma) && continue
        key = _hard_case_key(first(rows))
        style = get(case_styles, key, :solid)
        x = Float64[]
        y = Float64[]
        for row in rows
            iteration = _number(row, "total_iter")
            value = metric(row)
            (
                isnothing(iteration) ||
                isnothing(value) ||
                !isfinite(value) ||
                (positive_only && value <= 0.0)
            ) && continue
            push!(x, iteration)
            push!(y, transform(value))
        end
        isempty(x) && continue
        color = _gamma_color(gamma)
        if length(x) >= 2
            lines!(axis, x, y; color, linestyle = style, linewidth = 1.45)
        end
        scatter!(
            axis,
            x,
            y;
            color,
            marker = :circle,
            markersize = 2.8,
        )
        status = get(summary, case_id, nothing)
        outcome_marker = isnothing(status) ? :diamond :
                         status ? :circle : :xcross
        scatter!(
            axis,
            [last(x)],
            [last(y)];
            color,
            marker = outcome_marker,
            strokecolor = :black,
            strokewidth = 0.5,
            markersize = 8,
        )
        plotted = true
    end
    plotted
end

function _globalization_figure(directions, iterations, summaries)
    figure = Figure(size = (1650, 930))
    direction_groups = _group_by_case(directions)
    iteration_groups = _group_by_case(iterations)
    all_rows = vcat(directions, iterations)
    hard_cases = sort!(
        unique([_hard_case_key(row) for row in all_rows]);
        by = string,
    )
    styles = (:solid, :dash, :dot, :dashdot)
    case_styles = Dict(
        key => styles[mod1(index, length(styles))] for
        (index, key) in enumerate(hard_cases)
    )
    outcome = Dict(
        get(row, "case_id", "") => _true(row, "direct_converged") for
        row in summaries if !isempty(get(row, "case_id", ""))
    )

    specifications = (
        (
            title = "A  raw residual",
            ylabel = "‖F‖₂",
            groups = direction_groups,
            metric = row -> _number(row, "raw_residual_norm2"),
            yscale = log10,
            positive = true,
        ),
        (
            title = "B  accepted step length",
            ylabel = "accepted α",
            groups = iteration_groups,
            metric = row -> _number(row, "accepted_alpha"),
            yscale = log10,
            positive = true,
        ),
        (
            title = "C  regularization",
            ylabel = "η",
            groups = direction_groups,
            metric = row -> _number(row, "eta"),
            yscale = log10,
            positive = true,
        ),
        (
            title = "D  Jacobian rank deficit",
            ylabel = "rows − numerical rank",
            groups = direction_groups,
            metric = row -> begin
                rows = _number(row, "row_dimension")
                rank = _number(row, "numerical_rank_estimate")
                isnothing(rows) || isnothing(rank) ? nothing : rows - rank
            end,
            yscale = identity,
            positive = false,
        ),
        (
            title = "E  RRQR conditioning proxy",
            ylabel = "σmax / σmin,retained",
            groups = direction_groups,
            metric = row -> _number(row, "rrqr_condition_proxy"),
            yscale = log10,
            positive = true,
        ),
        (
            title = "F  accepted model agreement",
            ylabel = "actual / predicted reduction",
            groups = iteration_groups,
            metric = row -> _number(row, "reduction_ratio"),
            yscale = identity,
            positive = false,
        ),
    )
    for (panel, specification) in enumerate(specifications)
        row_index = panel <= 3 ? 1 : 2
        column_index = mod1(panel, 3)
        axis = Axis(
            figure[row_index, column_index];
            title = specification.title,
            xlabel = row_index == 2 ? "Newton iteration" : "",
            ylabel = specification.ylabel,
            yscale = specification.yscale,
        )
        plotted = _plot_trace_metric!(
            axis,
            specification.groups,
            specification.metric;
            positive_only = specification.positive,
            case_styles,
            summary = outcome,
        )
        if panel == 6 && plotted
            hlines!(
                axis,
                [0.0];
                color = (:gray35, 0.45),
                linewidth = 1,
            )
            hlines!(
                axis,
                [0.75];
                color = (:gray35, 0.35),
                linestyle = :dash,
                linewidth = 1,
            )
            hlines!(
                axis,
                [1.0];
                color = (:gray35, 0.35),
                linestyle = :dot,
                linewidth = 1,
            )
        end
        plotted || _empty_axis!(axis, "Per-iteration telemetry unavailable")
    end

    legend_layout = GridLayout()
    figure[3, :] = legend_layout
    case_elements = [
        LineElement(
            color = :gray20,
            linestyle = case_styles[key],
            linewidth = 2,
        ) for key in hard_cases
    ]
    !isempty(case_elements) && Legend(
        legend_layout[1, 1],
        case_elements,
        _hard_case_label.(hard_cases);
        orientation = :horizontal,
        framevisible = false,
    )
    outcome_elements = [
        MarkerElement(
            color = :gray30,
            marker = marker,
            markersize = 10,
        ) for marker in (:circle, :xcross, :diamond)
    ]
    Legend(
        legend_layout[1, 2],
        outcome_elements,
        ["converged", "failed", "classification unavailable"];
        orientation = :horizontal,
        framevisible = false,
    )
    Colorbar(
        legend_layout[1, 3];
        colormap = :viridis,
        limits = (0.0, 1.0),
        vertical = false,
        label = "γ",
        width = 180,
    )
    _figure_title!(
        figure,
        "Hard-case globalization telemetry",
        "Line style identifies the transition, color identifies γ, and the " *
        "terminal marker records the frozen convergence classification.",
    )
    figure
end

function _boundary_norm(row)
    values = (
        _number(row, "initial_residual_norm2"),
        _number(row, "terminal_residual_norm2"),
        _number(row, "global_residual_norm2"),
    )
    any(isnothing, values) && return nothing
    sqrt(sum(value^2 for value in values))
end

function _scaling_series(rows, formulation, transport, metric)
    selected = filter(
        row ->
            _token(row, "formulation") == formulation &&
            _token(row, "transport") == transport,
        rows,
    )
    values = Tuple{Float64, Float64}[]
    for row in selected
        epsilon = _number(row, "epsilon")
        value = metric(row)
        (
            isnothing(epsilon) ||
            epsilon <= 0.0 ||
            isnothing(value) ||
            value <= 0.0
        ) && continue
        push!(values, (epsilon, value))
    end
    sort!(values; by = first)
end

function _scaling_slope(slopes, formulation, transport, metric)
    selected = filter(
        row ->
            _token(row, "formulation") == formulation &&
            _token(row, "transport") == transport &&
            get(row, "metric", "") == metric,
        slopes,
    )
    isempty(selected) ? nothing :
    _number(first(selected), "primary_slope")
end

function _scaling_figure(rows, slopes)
    figure = Figure(size = (1460, 810))
    specifications = (
        (
            title = "Inherited interior",
            ylabel = "‖Finterior‖₂",
            metric = row -> _number(row, "interior_residual_norm2"),
            slope_metric = "interior_residual_norm2",
        ),
        (
            title = "Initial + terminal + global boundary",
            ylabel = "‖Fboundary‖₂",
            metric = _boundary_norm,
            slope_metric = "",
        ),
    )
    legend_handles = Any[]
    legend_labels = String[]
    for (row_index, specification) in enumerate(specifications)
        for (column_index, formulation) in enumerate(FORMULATIONS)
            axis = Axis(
                figure[row_index, column_index];
                title = row_index == 1 ?
                        "$(uppercasefirst(formulation)): " *
                        lowercase(specification.title) :
                        specification.title,
                xlabel = row_index == 2 ? "data perturbation ε" : "",
                ylabel = column_index == 1 ? specification.ylabel : "",
                xscale = log10,
                yscale = log10,
            )
            plotted = false
            for transport in SCALING_TRANSPORTS
                series = _scaling_series(
                    rows,
                    formulation,
                    transport,
                    specification.metric,
                )
                isempty(series) && continue
                x = first.(series)
                y = last.(series)
                color = TRANSPORT_COLOR[transport]
                marker = TRANSPORT_MARKER[transport]
                lines!(axis, x, y; color, linewidth = 2)
                scatter!(
                    axis,
                    x,
                    y;
                    color,
                    marker,
                    markersize = 8,
                    strokecolor = :white,
                    strokewidth = 0.5,
                )
                plotted = true
            end
            if row_index == 1
                anchor = filter(
                    row ->
                        _token(row, "formulation") == formulation &&
                        _token(row, "transport") == "identity_copy" &&
                        something(_number(row, "epsilon"), NaN) == 0.0,
                    rows,
                )
                if !isempty(anchor)
                    intercept =
                        _number(first(anchor), "interior_residual_norm2")
                    if !isnothing(intercept) && intercept > 0.0
                        hlines!(
                            axis,
                            [intercept];
                            color = (TRANSPORT_COLOR["identity_copy"], 0.55),
                            linestyle = :dash,
                            linewidth = 1.4,
                        )
                        text!(
                            axis,
                            0.03,
                            0.96;
                            text = "fixed-index ε→0 intercept = " *
                                   string(round(intercept; sigdigits = 3)),
                            space = :relative,
                            align = (:left, :top),
                            fontsize = 10,
                            color = TRANSPORT_COLOR["identity_copy"],
                        )
                    end
                end
                annotations = String[]
                for transport in SCALING_TRANSPORTS
                    slope = _scaling_slope(
                        slopes,
                        formulation,
                        transport,
                        specification.slope_metric,
                    )
                    isnothing(slope) && continue
                    push!(
                        annotations,
                        "$(_transport_short_name(transport)): " *
                        "m=$(round(slope; digits = 2))",
                    )
                end
                isempty(annotations) || text!(
                    axis,
                    0.98,
                    0.04;
                    text = join(annotations, "\n"),
                    space = :relative,
                    align = (:right, :bottom),
                    fontsize = 9,
                    color = :gray30,
                )
            end
            plotted || _empty_axis!(axis, "Scaling benchmark unavailable")
        end
    end
    for transport in SCALING_TRANSPORTS
        push!(
            legend_handles,
            LineElement(
                color = TRANSPORT_COLOR[transport],
                linewidth = 2,
            ),
        )
        push!(legend_labels, TRANSPORT_LABEL[transport])
    end
    Legend(
        figure[3, :],
        legend_handles,
        legend_labels;
        orientation = :horizontal,
        nbanks = 1,
        framevisible = false,
    )
    _figure_title!(
        figure,
        "Controlled ε-scaling benchmark",
        "T = 20 and Δt = 0.1 throughout. Boundary norm combines initial, " *
        "terminal, and global rows; the oracle terminal completion is " *
        "diagnostic and unavailable online.",
    )
    figure
end

function _transport_short_name(transport)
    transport == "identity_copy" && return "fixed"
    transport == "stage_shift_zero_tail" && return "zero"
    transport == "stage_shift_hold_tail" && return "hold"
    transport == "diagnostic_oracle_terminal_completion" && return "oracle"
    transport
end

"""
    generate_figures(run_dir)

Generate six paired PNG/PDF publication figures from the Phase 1–4 raw CSVs.
The routine never invokes an optimizer. A missing phase produces a labelled
empty panel so partially completed runs can be inspected without fabricating
results.
"""
function generate_figures(run_dir::AbstractString)
    run_dir = abspath(run_dir)
    CairoMakie.activate!()
    set_theme!(
        Theme(
            fontsize = 14,
            Axis = (
                xgridvisible = false,
                ygridcolor = (:gray75, 0.32),
                topspinevisible = false,
                rightspinevisible = false,
                titlesize = 15,
            ),
            Legend = (framevisible = false,),
        ),
    )
    validation = validate_figure_schemas(run_dir; throw_on_error = false)
    raw_dir = validation.raw_dir
    output_dir = joinpath(run_dir, "figures")

    stagewise =
        _read_csv(joinpath(raw_dir, "production_stagewise_residuals.csv"))
    ablations = _read_csv(joinpath(raw_dir, "boundary_ablations.csv"))
    gamma_summary = _read_csv(joinpath(raw_dir, "gamma_group_summary.csv"))
    directions = _read_csv(joinpath(raw_dir, "globalization_directions.csv"))
    iterations = _read_csv(joinpath(raw_dir, "globalization_iterations.csv"))
    globalization_summary =
        _read_csv(joinpath(raw_dir, "globalization_case_summary.csv"))
    scaling = _read_csv(joinpath(raw_dir, "scaling_benchmark.csv"))
    scaling_slopes =
        _read_csv(joinpath(raw_dir, "scaling_benchmark_slopes.csv"))

    figures = (
        (
            "stage_family_heatmap",
            _stage_family_heatmap(stagewise),
        ),
        (
            "reduced_seed202_t4_7_stagewise_heatmap",
            _spike_heatmap(stagewise),
        ),
        (
            "boundary_ablation_localization",
            _boundary_ablation_figure(ablations),
        ),
        (
            "gamma_holdout_validation",
            _gamma_figure(gamma_summary),
        ),
        (
            "globalization_hard_cases",
            _globalization_figure(
                directions,
                iterations,
                globalization_summary,
            ),
        ),
        (
            "epsilon_scaling",
            _scaling_figure(scaling, scaling_slopes),
        ),
    )
    output_paths = String[]
    for (stem, figure) in figures
        append!(output_paths, _save_versions(output_dir, stem, figure))
    end
    output_paths
end

end # module
