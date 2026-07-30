module BaselineAudit

using Statistics
using TOML: TOML

export DEFAULT_PLANNING_HORIZON,
    DEFAULT_DELTA_T,
    DEFAULT_SOLVER_TOLERANCE,
    audit_baseline,
    write_audit

const DEFAULT_PLANNING_HORIZON = 20
const DEFAULT_DELTA_T = 0.1
const DEFAULT_SOLVER_TOLERANCE = 0.008

const REQUIRED_TABLES = (
    "kkt_systems",
    "references",
    "replay",
    "scaling",
    "scaling_slopes",
    "sensitivity",
)

const CsvRow = Dict{String, String}

_fraction(numerator, denominator) = (; numerator, denominator)

function _summary(values)
    finite_values = Float64[
        Float64(value) for value in values if isfinite(value)
    ]
    isempty(finite_values) && return (;
        count = 0,
        minimum = nothing,
        median = nothing,
        maximum = nothing,
    )
    (;
        count = length(finite_values),
        minimum = minimum(finite_values),
        median = Statistics.median(finite_values),
        maximum = maximum(finite_values),
    )
end

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

"""
Read a study CSV without modifying it.

The original study's incremental reader repairs an incomplete final row. An audit
must never alter its input, so malformed rows are errors here.
"""
function _read_csv(path)
    isfile(path) || error("Required baseline table is missing: $(path)")
    lines = readlines(path)
    isempty(lines) && error("Required baseline table is empty: $(path)")
    header = _split_csv_line(first(lines), path, 1)
    rows = CsvRow[]
    for (row_index, line) in enumerate(Iterators.drop(lines, 1))
        line_number = row_index + 1
        isempty(line) && continue
        fields = _split_csv_line(line, path, line_number)
        length(fields) == length(header) || error(
            "CSV field-count mismatch at $(path):$(line_number): " *
            "expected $(length(header)), found $(length(fields)).",
        )
        push!(rows, CsvRow(header .=> fields))
    end
    rows
end

function _raw_directory(path)
    normalized = abspath(normpath(path))
    basename(normalized) == "raw" ? normalized : joinpath(normalized, "raw")
end

function _field_float(row, column)
    text = strip(get(row, column, ""))
    isempty(text) && return nothing
    tryparse(Float64, text)
end

function _finite_float(row, column)
    value = _field_float(row, column)
    isnothing(value) || !isfinite(value) ? nothing : value
end

function _field_int(row, column)
    text = strip(get(row, column, ""))
    isempty(text) && return nothing
    tryparse(Int, text)
end

function _required_int(row, column)
    value = _field_int(row, column)
    isnothing(value) && error(
        "Expected integer field `$(column)` in row $(get(row, "case_id", "<unknown>")).",
    )
    value
end

function _field_bool(row, column)
    text = lowercase(strip(get(row, column, "")))
    text == "true" && return true
    text == "false" && return false
    nothing
end

_is_true(row, column) = something(_field_bool(row, column), false)

function _status_counts(rows, column)
    counts = Dict{String, Int}()
    for row in rows
        value = get(row, column, "")
        counts[value] = get(counts, value, 0) + 1
    end
    [
        (; value, count = counts[value]) for
        value in sort!(collect(keys(counts)))
    ]
end

function _reference_accepted(row, tolerance)
    residual = _finite_float(row, "direct_residual_norm2")
    !isnothing(residual) && residual <= tolerance
end

function _direct_converged(row, tolerance)
    residual = _finite_float(row, "direct_final_residual_norm2")
    !isnothing(residual) && residual <= tolerance
end

function _dimension_audit(rows, expected_planning_horizon)
    dimensions = NamedTuple[]
    for row in sort(rows; by = row -> get(row, "formulation", ""))
        equation_rows = _required_int(row, "kkt_dimension")
        variable_columns = _required_int(row, "variable_dimension")
        block_dimensions = (;
            z = _required_int(row, "z_dimension"),
            lambda = _required_int(row, "lambda_dimension"),
            psi_out = _required_int(row, "psi_out_dimension"),
            psi_in = _required_int(row, "psi_in_dimension"),
            preference_slack =
                _required_int(row, "preference_slack_dimension"),
            interior_point_slack =
                _required_int(row, "interior_point_slack_dimension"),
            inequality_dual =
                _required_int(row, "inequality_dual_dimension"),
            unclassified = _required_int(row, "unclassified_dimension"),
        )
        block_sum = sum(values(block_dimensions))
        nullity_lower_bound = max(variable_columns - equation_rows, 0)
        push!(dimensions, (;
            formulation = get(row, "formulation", ""),
            planning_horizon = _required_int(row, "planning_horizon"),
            equation_rows,
            variable_columns,
            block_dimensions,
            block_sum,
            blocks_cover_variables = block_sum == variable_columns,
            right_nullity = (;
                relation = :at_least,
                lower_bound = nullity_lower_bound,
                exact_value = nothing,
                exact_rank_computed = false,
                reason =
                    "n - m is only a lower bound unless full row rank is established",
            ),
        ))
    end
    horizons =
        sort!(unique([row.planning_horizon for row in dimensions]))
    (;
        rows = dimensions,
        observed_planning_horizons = horizons,
        expected_planning_horizon,
        all_horizons_match =
            !isempty(dimensions) &&
            all(
                row -> row.planning_horizon == expected_planning_horizon,
                dimensions,
            ),
        all_block_sums_match =
            !isempty(dimensions) &&
            all(row -> row.blocks_cover_variables, dimensions),
    )
end

function _reference_audit(rows, tolerance)
    grouped = Dict{Tuple{String, Int}, Vector{CsvRow}}()
    for row in rows
        key = (
            get(row, "formulation", ""),
            _required_int(row, "scenario_seed"),
        )
        push!(get!(grouped, key, CsvRow[]), row)
    end

    by_formulation_seed = NamedTuple[]
    for (key, selected) in sort!(collect(grouped); by = first)
        accepted_rows =
            filter(row -> _reference_accepted(row, tolerance), selected)
        push!(by_formulation_seed, (;
            formulation = key[1],
            scenario_seed = key[2],
            accepted =
                _fraction(length(accepted_rows), length(selected)),
            accepted_steps = sort!([
                _required_int(row, "step") for row in accepted_rows
            ]),
            solver_statuses = _status_counts(selected, "solver_status"),
        ))
    end

    by_formulation = NamedTuple[]
    for formulation in sort!(unique(get.(rows, "formulation", "")))
        selected =
            filter(row -> get(row, "formulation", "") == formulation, rows)
        accepted = count(
            row -> _reference_accepted(row, tolerance),
            selected,
        )
        push!(by_formulation, (;
            formulation,
            accepted = _fraction(accepted, length(selected)),
        ))
    end

    predicate_disagreements = [
        get(row, "case_id", "") for row in rows if
        _is_true(row, "reference_accepted") !=
        _reference_accepted(row, tolerance)
    ]
    focus = [
        (;
            step = _required_int(row, "step"),
            source = get(row, "source", ""),
            solver_status = get(row, "solver_status", ""),
            accepted = _reference_accepted(row, tolerance),
            direct_residual_norm2 =
                _finite_float(row, "direct_residual_norm2"),
            total_inner_iters = _field_int(row, "total_inner_iters"),
        ) for row in rows if
        get(row, "formulation", "") == "reduced" &&
        _required_int(row, "scenario_seed") == 202 &&
        4 <= _required_int(row, "step") <= 8
    ]
    sort!(focus; by = row -> row.step)

    acceptance_tolerances = sort!(unique([
        value for row in rows
        for value in (_finite_float(row, "acceptance_tol"),)
        if !isnothing(value)
    ]))
    requested_tolerances = sort!(unique([
        value for row in rows
        for value in (_finite_float(row, "requested_tol"),)
        if !isnothing(value)
    ]))
    accepted = count(row -> _reference_accepted(row, tolerance), rows)
    (;
        total = length(rows),
        accepted = _fraction(accepted, length(rows)),
        by_formulation,
        by_formulation_seed,
        predicate_disagreements,
        acceptance_tolerances,
        requested_tolerances,
        focus_reduced_seed202_steps4_to8 = focus,
    )
end

function _residual_dominance(rows)
    valid_rows = NamedTuple[]
    for row in rows
        _is_true(row, "valid_reference_pair") || continue
        total = _finite_float(row, "initial_residual_norm2")
        outer = _finite_float(row, "initial_stationarity_outer_norm2")
        inner =
            _finite_float(row, "initial_stationarity_innermost_norm2")
        equality = _finite_float(row, "initial_equality_norm2")
        any(isnothing, (total, outer, inner, equality)) && continue
        push!(valid_rows, (;
            formulation = get(row, "formulation", ""),
            scenario_seed = _required_int(row, "scenario_seed"),
            transition = _required_int(row, "transition"),
            mode = get(row, "mode", ""),
            total,
            outer,
            inner,
            equality,
        ))
    end

    outer_largest = count(
        row ->
            row.outer >= row.inner &&
            row.outer >= row.equality,
        valid_rows,
    )
    ratios = [
        row.outer / row.total for row in valid_rows if row.total > 0.0
    ]
    partition_discrepancies = [
        abs(
            row.total^2 -
            (
                row.outer^2 +
                row.inner^2 +
                row.equality^2
            ),
        ) for row in valid_rows
    ]

    transition_groups =
        Dict{Tuple{String, Int, Int}, Vector{NamedTuple}}()
    for row in valid_rows
        key = (
            row.formulation,
            row.scenario_seed,
            row.transition,
        )
        push!(get!(transition_groups, key, NamedTuple[]), row)
    end
    equality_spreads = Float64[]
    retained_inner_spreads = Float64[]
    for selected in values(transition_groups)
        equality_values = getproperty.(selected, :equality)
        length(equality_values) > 1 &&
            push!(
                equality_spreads,
                maximum(equality_values) - minimum(equality_values),
            )
        retained = filter(
            row -> row.mode != "primal_only",
            selected,
        )
        inner_values = getproperty.(retained, :inner)
        length(inner_values) > 1 &&
            push!(
                retained_inner_spreads,
                maximum(inner_values) - minimum(inner_values),
            )
    end

    grouped = Dict{Tuple{String, String}, Vector{NamedTuple}}()
    for row in valid_rows
        key = (row.formulation, row.mode)
        push!(get!(grouped, key, NamedTuple[]), row)
    end
    by_formulation_mode = NamedTuple[]
    for (key, selected) in sort!(collect(grouped); by = first)
        push!(by_formulation_mode, (;
            formulation = key[1],
            mode = key[2],
            rows = length(selected),
            median_outer = median(getproperty.(selected, :outer)),
            median_inner = median(getproperty.(selected, :inner)),
            median_equality = median(getproperty.(selected, :equality)),
        ))
    end

    (;
        rows_with_family_norms = length(valid_rows),
        outer_stationarity_largest =
            _fraction(outer_largest, length(valid_rows)),
        outer_to_total_ratio = _summary(ratios),
        squared_norm_partition_discrepancy =
            _summary(partition_discrepancies),
        maximum_equality_spread_within_transition =
            isempty(equality_spreads) ? nothing :
            maximum(equality_spreads),
        maximum_inner_spread_among_dual_retaining_modes =
            isempty(retained_inner_spreads) ? nothing :
            maximum(retained_inner_spreads),
        by_formulation_mode,
    )
end

function _replay_audit(rows, tolerance)
    modes = sort!(unique(get.(rows, "mode", "")))
    transition_groups =
        Dict{Tuple{String, Int, Int}, Vector{CsvRow}}()
    for row in rows
        key = (
            get(row, "formulation", ""),
            _required_int(row, "scenario_seed"),
            _required_int(row, "transition"),
        )
        push!(get!(transition_groups, key, CsvRow[]), row)
    end

    mixed_validity_transitions = Tuple{String, Int, Int}[]
    transition_rows = NamedTuple[]
    for (key, selected) in sort!(collect(transition_groups); by = first)
        valid_flags =
            unique(_is_true(row, "valid_reference_pair") for row in selected)
        length(valid_flags) == 1 ||
            push!(mixed_validity_transitions, key)
        valid = length(valid_flags) == 1 && first(valid_flags)
        attempted = filter(
            row -> _is_true(row, "valid_reference_pair"),
            selected,
        )
        converged_modes = sort!([
            get(row, "mode", "") for row in attempted if
            _direct_converged(row, tolerance)
        ])
        push!(transition_rows, (;
            formulation = key[1],
            scenario_seed = key[2],
            transition = key[3],
            valid,
            attempted_modes = length(attempted),
            converged_modes,
            fully_converged =
                valid &&
                length(attempted) == length(modes) &&
                length(converged_modes) == length(modes),
        ))
    end

    by_formulation_mode = NamedTuple[]
    grouped = Dict{Tuple{String, String}, Vector{CsvRow}}()
    for row in rows
        key = (get(row, "formulation", ""), get(row, "mode", ""))
        push!(get!(grouped, key, CsvRow[]), row)
    end
    for (key, selected) in sort!(collect(grouped); by = first)
        valid = filter(
            row -> _is_true(row, "valid_reference_pair"),
            selected,
        )
        converged =
            count(row -> _direct_converged(row, tolerance), valid)
        push!(by_formulation_mode, (;
            formulation = key[1],
            mode = key[2],
            planned_rows = length(selected),
            valid_rows = length(valid),
            unavailable_rows = length(selected) - length(valid),
            direct_converged =
                _fraction(converged, length(valid)),
            solver_statuses =
                _status_counts(valid, "solver_status"),
        ))
    end

    by_formulation = NamedTuple[]
    for formulation in sort!(unique(get.(rows, "formulation", "")))
        selected =
            filter(row -> row.formulation == formulation, transition_rows)
        valid = filter(row -> row.valid, selected)
        push!(by_formulation, (;
            formulation,
            valid_transitions =
                _fraction(length(valid), length(selected)),
            fully_converged_transitions = _fraction(
                count(row -> row.fully_converged, valid),
                length(valid),
            ),
        ))
    end

    valid_rows =
        filter(row -> _is_true(row, "valid_reference_pair"), rows)
    converged =
        count(row -> _direct_converged(row, tolerance), valid_rows)
    direct_predicate_disagreements = [
        get(row, "case_id", "") for row in rows if
        _is_true(row, "direct_converged") !=
        _direct_converged(row, tolerance)
    ]
    visible_valid_pair_disagreements = [
        get(row, "case_id", "") for row in rows if
        _is_true(row, "valid_reference_pair") !=
        (
            _is_true(row, "source_reference_accepted") &&
            _is_true(row, "destination_reference_accepted")
        )
    ]

    focus = NamedTuple[]
    for row in rows
        get(row, "formulation", "") == "reduced" || continue
        _required_int(row, "scenario_seed") == 202 || continue
        transition = _required_int(row, "transition")
        4 <= transition <= 7 || continue
        push!(focus, (;
            transition,
            mode = get(row, "mode", ""),
            source_reference_accepted =
                _is_true(row, "source_reference_accepted"),
            destination_reference_accepted =
                _is_true(row, "destination_reference_accepted"),
            valid_reference_pair =
                _is_true(row, "valid_reference_pair"),
            initial_residual_norm2 =
                _finite_float(row, "initial_residual_norm2"),
            initial_stationarity_outer_norm2 = _finite_float(
                row,
                "initial_stationarity_outer_norm2",
            ),
            initial_stationarity_innermost_norm2 = _finite_float(
                row,
                "initial_stationarity_innermost_norm2",
            ),
            initial_equality_norm2 =
                _finite_float(row, "initial_equality_norm2"),
            solver_status = get(row, "solver_status", ""),
            total_inner_iters =
                _field_int(row, "total_inner_iters"),
            direct_final_residual_norm2 =
                _finite_float(row, "direct_final_residual_norm2"),
            direct_converged =
                _direct_converged(row, tolerance),
            failure_reason = get(row, "failure_reason", ""),
        ))
    end
    sort!(focus; by = row -> (row.transition, row.mode))

    valid_transitions =
        count(row -> row.valid, transition_rows)
    (;
        modes,
        planned_transitions = length(transition_rows),
        valid_transitions =
            _fraction(valid_transitions, length(transition_rows)),
        planned_mode_rows = length(rows),
        valid_mode_rows = length(valid_rows),
        unavailable_mode_rows = length(rows) - length(valid_rows),
        direct_converged = _fraction(converged, length(valid_rows)),
        by_formulation,
        by_formulation_mode,
        transitions = transition_rows,
        mixed_validity_transitions,
        direct_predicate_disagreements,
        visible_valid_pair_disagreements,
        residual_dominance = _residual_dominance(rows),
        focus_reduced_seed202_transitions4_to7 = focus,
    )
end

function _linear_fit(x, y)
    length(x) >= 2 || return nothing
    x_mean = sum(x) / length(x)
    y_mean = sum(y) / length(y)
    denominator = sum((value - x_mean)^2 for value in x)
    denominator > 0.0 || return nothing
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean) for
        (x_value, y_value) in zip(x, y)
    ) / denominator
    intercept = y_mean - slope * x_mean
    total = sum((value - y_mean)^2 for value in y)
    residual = sum(
        (
            y_value -
            (intercept + slope * x_value)
        )^2 for (x_value, y_value) in zip(x, y)
    )
    r_squared = total == 0.0 ? NaN : 1.0 - residual / total
    (; slope, intercept, r_squared)
end

function _recompute_scaling_slopes(rows)
    base_reference_values =
        Dict{Tuple{String, Int, Int}, Vector{Float64}}()
    for row in rows
        get(row, "mode", "") == "all_duals" || continue
        baseline =
            _finite_float(row, "baseline_initial_residual_norm2")
        isnothing(baseline) && continue
        key = (
            get(row, "formulation", ""),
            _required_int(row, "scenario_seed"),
            _required_int(row, "direction"),
        )
        push!(
            get!(base_reference_values, key, Float64[]),
            baseline,
        )
    end
    base_reference_floor = Dict(
        key => median(values) for
        (key, values) in base_reference_values
    )

    grouped =
        Dict{Tuple{String, Int, Int, String}, Vector{CsvRow}}()
    for row in rows
        key = (
            get(row, "formulation", ""),
            _required_int(row, "scenario_seed"),
            _required_int(row, "direction"),
            get(row, "mode", ""),
        )
        push!(get!(grouped, key, CsvRow[]), row)
    end

    results = NamedTuple[]
    for (key, selected) in sort!(collect(grouped); by = first)
        base_key = key[1:3]
        base_resolution = get(base_reference_floor, base_key, 0.0)
        structural_baselines = Float64[
            value for row in selected
            for value in (
                _finite_float(
                    row,
                    "baseline_initial_residual_norm2",
                ),
            ) if !isnothing(value)
        ]
        structural_baseline =
            isempty(structural_baselines) ? NaN :
            median(structural_baselines)

        available = NamedTuple[]
        for row in selected
            epsilon = _finite_float(row, "epsilon")
            normalized =
                _finite_float(row, "initial_residual_normalized")
            residual = _finite_float(row, "initial_residual_norm2")
            perturbed_reference = something(
                _finite_float(
                    row,
                    "perturbed_reference_residual",
                ),
                0.0,
            )
            resolution = max(
                base_resolution,
                perturbed_reference,
                100 * eps(Float64),
            )
            structural_floor =
                isfinite(structural_baseline) &&
                structural_baseline > 1.25 * resolution
            reliable =
                !isnothing(residual) &&
                (
                    residual > 1.25 * resolution ||
                    structural_floor
                )
            if !isnothing(epsilon) &&
               !isnothing(normalized) &&
               !isnothing(residual) &&
               epsilon > 0.0 &&
               normalized > 0.0
                push!(available, (;
                    epsilon,
                    normalized,
                    residual,
                    reliable,
                    resolution,
                ))
            end
        end
        sort!(available; by = value -> value.epsilon)
        reliable = filter(value -> value.reliable, available)
        fit_values = first(reliable, min(3, length(reliable)))
        fit = _linear_fit(
            log.([value.epsilon for value in fit_values]),
            log.([value.normalized for value in fit_values]),
        )
        resolution_floor =
            isempty(available) ? NaN :
            maximum(value.resolution for value in available)
        structural_ratio =
            isfinite(structural_baseline) &&
            isfinite(resolution_floor) &&
            resolution_floor > 0.0 ?
            structural_baseline / resolution_floor : NaN
        structural_floor_detected =
            isfinite(structural_ratio) && structural_ratio > 1.25
        smallest_ratio =
            isempty(available) ||
            !isfinite(structural_baseline) ||
            structural_baseline <= 0.0 ?
            NaN :
            first(available).residual / structural_baseline
        skip_reason =
            length(reliable) < 2 ?
            "fewer than two points exceed the reference-resolution floor and no resolvable structural baseline is present" :
            isnothing(fit) ? "degenerate epsilon coordinates" : ""
        push!(results, (;
            case_id =
                "$(key[1])__$(key[2])__$(key[3])__$(key[4])",
            formulation = key[1],
            scenario_seed = key[2],
            direction = key[3],
            mode = key[4],
            available_points = length(available),
            reliable_points = length(reliable),
            reference_resolution_floor_norm2 = resolution_floor,
            structural_baseline_residual_norm2 =
                structural_baseline,
            structural_baseline_to_resolution_ratio =
                structural_ratio,
            structural_floor_detected,
            smallest_epsilon_residual_to_baseline_ratio =
                smallest_ratio,
            fit_points = length(fit_values),
            epsilon_min =
                isempty(fit_values) ? NaN :
                first(fit_values).epsilon,
            epsilon_max =
                isempty(fit_values) ? NaN :
                last(fit_values).epsilon,
            slope_log_r0_vs_log_epsilon =
                isnothing(fit) ? NaN : fit.slope,
            intercept = isnothing(fit) ? NaN : fit.intercept,
            r_squared = isnothing(fit) ? NaN : fit.r_squared,
            fit_performed = !isnothing(fit),
            skip_reason,
        ))
    end
    results
end

function _numbers_match(left, right)
    isnothing(left) && return isnothing(right)
    isnothing(right) && return false
    isnan(left) && return isnan(right)
    isinf(left) && return left == right
    isapprox(left, right; rtol = 1e-12, atol = 1e-12)
end

function _compare_scaling_slopes(computed, published)
    published_by_case =
        Dict(get(row, "case_id", "") => row for row in published)
    discrepancies = NamedTuple[]
    integer_fields = (
        (:available_points, "available_points"),
        (:reliable_points, "reliable_points"),
        (:fit_points, "fit_points"),
    )
    numeric_fields = (
        (
            :reference_resolution_floor_norm2,
            "reference_resolution_floor_norm2",
        ),
        (
            :structural_baseline_residual_norm2,
            "structural_baseline_residual_norm2",
        ),
        (
            :structural_baseline_to_resolution_ratio,
            "structural_baseline_to_resolution_ratio",
        ),
        (
            :slope_log_r0_vs_log_epsilon,
            "slope_log_r0_vs_log_epsilon",
        ),
    )
    for result in computed
        row = get(published_by_case, result.case_id, nothing)
        if isnothing(row)
            push!(discrepancies, (;
                case_id = result.case_id,
                fields = ["missing published row"],
            ))
            continue
        end
        fields = String[]
        for (property, column) in integer_fields
            getproperty(result, property) == _field_int(row, column) ||
                push!(fields, column)
        end
        for (property, column) in numeric_fields
            _numbers_match(
                getproperty(result, property),
                _field_float(row, column),
            ) || push!(fields, column)
        end
        getproperty(result, :structural_floor_detected) ==
        _field_bool(row, "structural_floor_detected") ||
            push!(fields, "structural_floor_detected")
        getproperty(result, :fit_performed) ==
        _field_bool(row, "fit_performed") ||
            push!(fields, "fit_performed")
        isempty(fields) || push!(discrepancies, (;
            case_id = result.case_id,
            fields,
        ))
    end
    computed_ids = Set(getproperty.(computed, :case_id))
    unexpected_published = sort!([
        get(row, "case_id", "") for row in published if
        get(row, "case_id", "") ∉ computed_ids
    ])
    (;
        recomputed_rows = length(computed),
        published_rows = length(published),
        discrepancies,
        unexpected_published,
        exact_match =
            isempty(discrepancies) &&
            isempty(unexpected_published) &&
            length(computed) == length(published),
    )
end

function _floor_audit(scaling_rows, published_slope_rows)
    slope_rows = _recompute_scaling_slopes(scaling_rows)
    modes = sort!(unique(getproperty.(slope_rows, :mode)))
    by_mode = NamedTuple[]
    for mode in modes
        selected = filter(row -> row.mode == mode, slope_rows)
        baselines =
            getproperty.(selected, :structural_baseline_residual_norm2)
        ratios = getproperty.(
            selected,
            :structural_baseline_to_resolution_ratio,
        )
        fitted_slopes = [
            row.slope_log_r0_vs_log_epsilon for row in selected if
            row.fit_performed &&
            isfinite(row.slope_log_r0_vs_log_epsilon)
        ]
        floor_count =
            count(row -> row.structural_floor_detected, selected)
        push!(by_mode, (;
            mode,
            structural_floor =
                _fraction(floor_count, length(selected)),
            structural_baseline_norm2 = _summary(baselines),
            baseline_to_resolution_ratio = _summary(ratios),
            fitted_slope = _summary(fitted_slopes),
        ))
    end
    floor_modes = Set((
        "all_except_innermost_stationarity",
        "equality_duals",
    ))
    approximately_sixty = filter(
        row -> row.mode in floor_modes,
        slope_rows,
    )
    (;
        recomputed_rows = slope_rows,
        by_mode,
        all_except_and_equality = (;
            groups = length(approximately_sixty),
            structural_floor = _fraction(
                count(
                    row -> row.structural_floor_detected,
                    approximately_sixty,
                ),
                length(approximately_sixty),
            ),
            structural_baseline_norm2 = _summary(
                getproperty.(
                    approximately_sixty,
                    :structural_baseline_residual_norm2,
                ),
            ),
        ),
        published_comparison = _compare_scaling_slopes(
            slope_rows,
            published_slope_rows,
        ),
        equation_family_decomposition_available = false,
    )
end

function _scaling_audit(rows, slope_rows, tolerance)
    all_duals =
        filter(row -> get(row, "mode", "") == "all_duals", rows)
    valid = filter(
        row -> _is_true(row, "valid_reference_pair"),
        all_duals,
    )
    converged =
        count(row -> _direct_converged(row, tolerance), valid)

    grouped = Dict{Tuple{String, Int}, Vector{CsvRow}}()
    for row in all_duals
        key = (
            get(row, "formulation", ""),
            _required_int(row, "scenario_seed"),
        )
        push!(get!(grouped, key, CsvRow[]), row)
    end
    by_formulation_seed = NamedTuple[]
    for (key, selected) in sort!(collect(grouped); by = first)
        selected_valid = filter(
            row -> _is_true(row, "valid_reference_pair"),
            selected,
        )
        push!(by_formulation_seed, (;
            formulation = key[1],
            scenario_seed = key[2],
            valid = _fraction(length(selected_valid), length(selected)),
            direct_converged = _fraction(
                count(
                    row -> _direct_converged(row, tolerance),
                    selected_valid,
                ),
                length(selected_valid),
            ),
        ))
    end
    by_formulation = NamedTuple[]
    for formulation in sort!(unique(get.(all_duals, "formulation", "")))
        selected = filter(
            row -> get(row, "formulation", "") == formulation,
            all_duals,
        )
        selected_valid = filter(
            row -> _is_true(row, "valid_reference_pair"),
            selected,
        )
        push!(by_formulation, (;
            formulation,
            valid = _fraction(length(selected_valid), length(selected)),
            direct_converged = _fraction(
                count(
                    row -> _direct_converged(row, tolerance),
                    selected_valid,
                ),
                length(selected_valid),
            ),
        ))
    end

    invalid_cases = [
        (;
            formulation = get(row, "formulation", ""),
            scenario_seed = _required_int(row, "scenario_seed"),
            direction = _required_int(row, "direction"),
            epsilon = _finite_float(row, "epsilon"),
            base_reference_accepted =
                _is_true(row, "base_reference_accepted"),
            perturbed_reference_accepted =
                _is_true(row, "perturbed_reference_accepted"),
        ) for row in all_duals if
        !_is_true(row, "valid_reference_pair")
    ]
    sort!(invalid_cases; by = row -> (
        row.formulation,
        row.scenario_seed,
        row.direction,
        something(row.epsilon, Inf),
    ))

    direct_predicate_disagreements = [
        get(row, "case_id", "") for row in rows if
        _is_true(row, "direct_converged") !=
        _direct_converged(row, tolerance)
    ]
    visible_valid_pair_disagreements = [
        get(row, "case_id", "") for row in rows if
        _is_true(row, "valid_reference_pair") !=
        (
            _is_true(row, "base_reference_accepted") &&
            _is_true(row, "perturbed_reference_accepted")
        )
    ]
    final_residuals = Float64[
        value for row in valid
        for value in (
            _finite_float(row, "direct_final_residual_norm2"),
        ) if !isnothing(value)
    ]
    iterations = Int[
        value for row in valid
        for value in (_field_int(row, "total_inner_iters"),)
        if !isnothing(value)
    ]

    (;
        all_rows = length(rows),
        all_duals = (;
            planned = length(all_duals),
            valid = _fraction(length(valid), length(all_duals)),
            direct_converged =
                _fraction(converged, length(valid)),
            by_formulation,
            by_formulation_seed,
            invalid_cases,
            direct_final_residual_norm2 =
                _summary(final_residuals),
            total_inner_iters = isempty(iterations) ? (;
                minimum = nothing,
                maximum = nothing,
            ) : (;
                minimum = minimum(iterations),
                maximum = maximum(iterations),
            ),
        ),
        direct_predicate_disagreements,
        visible_valid_pair_disagreements,
        structural_floor = _floor_audit(rows, slope_rows),
    )
end

function _sensitivity_audit(rows)
    placeholders =
        filter(row -> get(row, "block", "") == "unavailable", rows)
    valid = filter(
        row ->
            _is_true(row, "reference_accepted") &&
            get(row, "block", "") != "unavailable",
        rows,
    )
    skipped = filter(
        row -> !_is_true(row, "near_null_computed"),
        valid,
    )
    computed = filter(
        row -> _is_true(row, "near_null_computed"),
        valid,
    )

    by_formulation = NamedTuple[]
    for formulation in sort!(unique(get.(valid, "formulation", "")))
        selected =
            filter(row -> get(row, "formulation", "") == formulation, valid)
        selected_skipped =
            filter(row -> !_is_true(row, "near_null_computed"), selected)
        push!(by_formulation, (;
            formulation,
            valid_rows = length(selected),
            near_null_skipped =
                _fraction(length(selected_skipped), length(selected)),
        ))
    end
    (;
        total_rows = length(rows),
        valid_block_amplitude_rows = length(valid),
        placeholder_rows = length(placeholders),
        near_null_computed_rows = length(computed),
        near_null_skipped =
            _fraction(length(skipped), length(valid)),
        skip_reasons = _status_counts(skipped, "near_null_skip_reason"),
        by_formulation,
        exact_numerical_nullity_available = !isempty(computed),
        row_unit =
            "one accepted reference × block × directional amplitude",
    )
end

function _exact_baseline_checks(
    dimensions,
    references,
    replay,
    scaling,
    sensitivity,
    tolerance,
)
    dimension_signature = all(
        row ->
            row.equation_rows == 1429 &&
            row.variable_columns == 2670 &&
            row.block_dimensions.z == 360 &&
            row.block_dimensions.lambda == 750 &&
            row.block_dimensions.psi_out == 720 &&
            row.block_dimensions.psi_in == 840 &&
            row.right_nullity.lower_bound == 1241 &&
            row.right_nullity.exact_value === nothing,
        dimensions.rows,
    )
    checks = (;
        fixed_tolerance_is_0_008 =
            tolerance == DEFAULT_SOLVER_TOLERANCE,
        planning_horizon_is_20 =
            dimensions.all_horizons_match &&
            dimensions.expected_planning_horizon ==
            DEFAULT_PLANNING_HORIZON,
        dimension_signature,
        block_sums_cover_all_2670_variables =
            dimensions.all_block_sums_match,
        reference_predicate_matches =
            isempty(references.predicate_disagreements),
        references_accepted_21_of_32 =
            references.accepted == _fraction(21, 32),
        replay_valid_transitions_17_of_28 =
            replay.valid_transitions == _fraction(17, 28),
        replay_valid_mode_rows_68_of_112 =
            replay.valid_mode_rows == 68 &&
            replay.planned_mode_rows == 112,
        replay_converged_62_of_68 =
            replay.direct_converged == _fraction(62, 68),
        replay_direct_predicate_matches =
            isempty(replay.direct_predicate_disagreements),
        scaling_all_duals_valid_27_of_32 =
            scaling.all_duals.valid == _fraction(27, 32),
        scaling_all_duals_converged_27_of_27 =
            scaling.all_duals.direct_converged ==
            _fraction(27, 27),
        scaling_direct_predicate_matches =
            isempty(scaling.direct_predicate_disagreements),
        scaling_slopes_recomputed_exactly =
            scaling.structural_floor.published_comparison.exact_match,
        floor_groups_16_of_16 =
            scaling.structural_floor.all_except_and_equality.structural_floor ==
            _fraction(16, 16),
        outer_stationarity_largest_68_of_68 =
            replay.residual_dominance.outer_stationarity_largest ==
            _fraction(68, 68),
        sensitivity_valid_rows_60 =
            sensitivity.valid_block_amplitude_rows == 60,
        sensitivity_svd_skips_60_of_60 =
            sensitivity.near_null_skipped == _fraction(60, 60),
        sensitivity_placeholders_3 =
            sensitivity.placeholder_rows == 3,
        sensitivity_exact_nullity_not_computed =
            !sensitivity.exact_numerical_nullity_available,
    )
    failed = String[
        String(name) for name in keys(checks) if
        !getproperty(checks, name)
    ]
    (; checks, passed = isempty(failed), failed)
end

function _toml_value(value)
    if value isa NamedTuple
        return Dict(
            String(name) => _toml_value(getproperty(value, name))
            for name in keys(value)
        )
    elseif value isa AbstractDict
        return Dict(
            string(key) => _toml_value(item)
            for (key, item) in value
        )
    elseif value isa AbstractVector || value isa Tuple
        return [_toml_value(item) for item in value]
    elseif value isa Symbol
        return String(value)
    elseif value === nothing
        # TOML has no null scalar. Preserve the distinction from a numerical
        # value with an explicit, stable sentinel.
        return "__not_computed__"
    elseif value isa AbstractString ||
           value isa Real ||
           value isa Bool
        return value
    end
    string(value)
end

"""
    write_audit(path, audit)

Serialize an `audit_baseline` result as TOML. The write is atomic; `nothing`
values use the explicit string sentinel `"__not_computed__"` because TOML has
no null scalar.
"""
function write_audit(path::AbstractString, audit)
    absolute_path = abspath(path)
    mkpath(dirname(absolute_path))
    buffer = IOBuffer()
    TOML.print(buffer, _toml_value(audit))
    temporary = "$(absolute_path).tmp.$(getpid())"
    open(temporary, "w") do io
        write(io, String(take!(buffer)))
        write(io, '\n')
        flush(io)
    end
    mv(temporary, absolute_path; force = true)
    absolute_path
end

"""
    audit_baseline(path; kwargs...)

Recompute the selective-warmstart baseline observations from the six raw CSV
tables. `path` may be either the run directory or its `raw/` directory.

Every rate is returned as `(numerator, denominator)`. The audit intentionally
reports the rectangular-Jacobian nullity as a lower bound; it never promotes
`n - m` to an exact nullity without a rank computation.

Set `enforce_expected_signature=true` to throw if this artifact no longer
matches the study baseline (T=20, tolerance=0.008, and the exact audited
denominators). This function is read-only.
"""
function audit_baseline(
    path::AbstractString;
    tolerance::Real = DEFAULT_SOLVER_TOLERANCE,
    expected_planning_horizon::Integer =
        DEFAULT_PLANNING_HORIZON,
    expected_delta_t::Real = DEFAULT_DELTA_T,
    enforce_expected_signature::Bool = false,
)
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    expected_planning_horizon > 0 ||
        throw(ArgumentError("expected_planning_horizon must be positive"))
    expected_delta_t > 0 ||
        throw(ArgumentError("expected_delta_t must be positive"))

    raw_dir = _raw_directory(path)
    tables = Dict(
        name => _read_csv(joinpath(raw_dir, "$(name).csv"))
        for name in REQUIRED_TABLES
    )
    dimensions = _dimension_audit(
        tables["kkt_systems"],
        Int(expected_planning_horizon),
    )
    references =
        _reference_audit(tables["references"], Float64(tolerance))
    replay = _replay_audit(tables["replay"], Float64(tolerance))
    scaling = _scaling_audit(
        tables["scaling"],
        tables["scaling_slopes"],
        Float64(tolerance),
    )
    sensitivity = _sensitivity_audit(tables["sensitivity"])
    signature = _exact_baseline_checks(
        dimensions,
        references,
        replay,
        scaling,
        sensitivity,
        Float64(tolerance),
    )

    if enforce_expected_signature && !signature.passed
        error(
            "Baseline audit signature failed: " *
            join(signature.failed, ", "),
        )
    end

    (;
        schema_version = 1,
        raw_directory = raw_dir,
        protocol = (;
            expected_planning_horizon =
                Int(expected_planning_horizon),
            expected_delta_t = Float64(expected_delta_t),
            solver_tolerance = Float64(tolerance),
            delta_t_recorded_in_raw_csv = false,
        ),
        predicates = (;
            reference_accepted =
                "isfinite(direct_residual_norm2) && direct_residual_norm2 <= tolerance",
            replay_direct_converged =
                "isfinite(direct_final_residual_norm2) && direct_final_residual_norm2 <= tolerance",
            scaling_direct_converged =
                "isfinite(direct_final_residual_norm2) && direct_final_residual_norm2 <= tolerance",
            visible_valid_replay_pair =
                "source_reference_accepted && destination_reference_accepted",
            visible_valid_scaling_pair =
                "base_reference_accepted && perturbed_reference_accepted",
            structural_floor =
                "median(baseline_initial_residual_norm2) / maximum(reference resolution) > 1.25",
        ),
        dimensions,
        references,
        replay,
        scaling,
        sensitivity,
        expected_baseline_signature = signature,
        corrections = (;
            nullity =
                "The CSV dimensions prove right nullity >= 1241, not exact nullity 1241; numerical SVD/rank was skipped.",
            sensitivity =
                "There are 60 valid block-amplitude SVD-skip rows plus 3 unavailable placeholders, for 63 total rows.",
            scaling =
                "All-duals convergence is 27/27 valid attempts; five additional planned rows are invalid, so it is not 32/32.",
            structural_floor =
                "The approximately-60 floor is the total norm2 for equality_duals and all_except_innermost_stationarity. Scaling CSVs contain no equation-family decomposition.",
            dimensions =
                "psi_out and psi_in are multiplier-coordinate dimensions, not equation-row counts.",
            delta_t =
                "The raw CSVs do not record Δt; the future runner must enforce and record Δt=0.1 independently.",
            transport =
                "The raw CSVs do not encode the dual index map. Flat-index copying versus stage-shifting must be verified from source or explicit transport-map output.",
            valid_pair =
                "Raw CSVs expose reference-acceptance flags but not checkpoint payload availability; stored valid_reference_pair remains authoritative for attempted-row denominators.",
        ),
    )
end

end
