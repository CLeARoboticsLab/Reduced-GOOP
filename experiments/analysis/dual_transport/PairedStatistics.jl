module PairedStatistics

using Random: MersenneTwister, rand
using Statistics: mean, median, quantile

export PAIRED_STATISTICS_COLUMNS,
    compute_paired_statistics,
    write_paired_statistics

const DEFAULT_BOOTSTRAP_SEED = 2_841_173
const DEFAULT_BOOTSTRAP_REPLICATES = 2_000
const BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

const _MODE_SPECS = (
    (label = "equality", value = "equality_duals"),
    (
        label = "all_except",
        value = "all_except_innermost_stationarity",
    ),
    (label = "all_duals", value = "all_duals"),
)

const _COMPARISON_SPECS = (
    (
        label = "identity_vs_zero",
        a = "identity_copy",
        b = "stage_shift_zero_tail",
    ),
    (
        label = "identity_vs_hold",
        a = "identity_copy",
        b = "stage_shift_hold_tail",
    ),
)

const _PERIOD_SPECS = (
    (label = "early", first_transition = 1, last_transition = 3),
    (label = "late", first_transition = 4, last_transition = 7),
    (label = "all", first_transition = 1, last_transition = 7),
)

const _DISTRIBUTION_PREFIXES = (
    "r0_a",
    "r0_b",
    "r0_difference_b_minus_a",
    "r0_ratio_b_over_a",
    "iterations_a",
    "iterations_b",
    "iterations_difference_b_minus_a",
    "iterations_ratio_b_over_a",
)

const PAIRED_STATISTICS_COLUMNS = let
    columns = String[
        "case_id",
        "formulation",
        "period",
        "first_transition",
        "last_transition",
        "mode",
        "mode_label",
        "comparison",
        "a_transport",
        "b_transport",
        "matched_pairs",
        "valid_pairs",
        "invalid_reference_pairs",
        "a_rows_without_b",
        "b_rows_without_a",
        "both_converged",
        "a_only_converged",
        "b_only_converged",
        "neither_converged",
        "a_converged_total",
        "b_converged_total",
        "a_convergence_rate",
        "b_convergence_rate",
        "convergence_rate_difference_b_minus_a",
        "convergence_rate_difference_b_minus_a_ci_lower",
        "convergence_rate_difference_b_minus_a_ci_upper",
        "r0_pairs",
        "r0_b_wins",
        "r0_ties",
        "r0_b_losses",
        "iteration_pairs",
        "iteration_pairs_excluded_non_both_converged",
        "iterations_b_wins",
        "iterations_ties",
        "iterations_b_losses",
    ]
    for prefix in _DISTRIBUTION_PREFIXES
        append!(
            columns,
            (
                "$(prefix)_n",
                "$(prefix)_median",
                "$(prefix)_q25",
                "$(prefix)_q75",
                "$(prefix)_iqr",
            ),
        )
    end
    append!(
        columns,
        (
            "r0_difference_b_minus_a_median_ci_lower",
            "r0_difference_b_minus_a_median_ci_upper",
            "r0_ratio_b_over_a_median_ci_lower",
            "r0_ratio_b_over_a_median_ci_upper",
            "iterations_difference_b_minus_a_median_ci_lower",
            "iterations_difference_b_minus_a_median_ci_upper",
            "iterations_ratio_b_over_a_median_ci_lower",
            "iterations_ratio_b_over_a_median_ci_upper",
            "bootstrap_seed",
            "bootstrap_replicates",
            "bootstrap_confidence_level",
            "difference_definition",
            "ratio_definition",
            "win_definition",
            "iteration_eligibility",
        ),
    )
    columns
end

"""
    compute_paired_statistics(
        replay_rows;
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
    )

Compute paired descriptive statistics from the dual-transport `replay.csv`
schema. Rows are paired on `(formulation, scenario_seed, transition, mode)`.
For each formulation and each early (1--3), late (4--7), and all (1--7)
stratum, the function compares identity copy (A) against zero-tail and
hold-tail transport (B) within equality, all-except-innermost, and all-dual
modes.

Only pairs for which both rows have `valid_reference_pair=true` contribute to
outcomes. Convergence uses `direct_converged`. Initial residual (`R0`)
statistics use every valid pair with two finite nonnegative
`initial_residual_norm2` values. Iteration statistics use only pairs where
both policies directly converged and both `total_inner_iters` values are
finite and nonnegative. Thus a failed solve's iteration-cap value (for
example, 1000) is never treated as a numeric performance outcome.

All paired differences are `B - A`, all paired ratios are `B / A`, and a B
win means a strictly smaller value. Ratios with a zero A denominator are
omitted. Percentile intervals are deterministic 95% paired-bootstrap
intervals for the convergence-rate difference and medians of the paired
differences and ratios. The return value is a vector of flat dictionaries
whose columns are listed in [`PAIRED_STATISTICS_COLUMNS`](@ref).
"""
function compute_paired_statistics(
    replay_rows;
    bootstrap_seed::Integer = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates::Integer = DEFAULT_BOOTSTRAP_REPLICATES,
)
    bootstrap_replicates > 0 || throw(
        ArgumentError(
            "bootstrap_replicates must be positive; got $(bootstrap_replicates).",
        ),
    )

    index, formulations = _index_replay_rows(replay_rows)
    results = Dict{String, Any}[]
    for formulation in formulations
        for mode_spec in _MODE_SPECS
            for comparison in _COMPARISON_SPECS
                for period in _PERIOD_SPECS
                    push!(
                        results,
                        _comparison_row(
                            index,
                            formulation,
                            mode_spec,
                            comparison,
                            period,
                            Int(bootstrap_seed),
                            Int(bootstrap_replicates),
                        ),
                    )
                end
            end
        end
    end
    results
end

"""
    write_paired_statistics(path, rows)

Write rows returned by [`compute_paired_statistics`](@ref) as a deterministic
CSV table. Columns use the fixed order in [`PAIRED_STATISTICS_COLUMNS`](@ref);
`nothing` and `missing` are written as empty fields. The parent directory is
created when necessary.
"""
function write_paired_statistics(path::AbstractString, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(PAIRED_STATISTICS_COLUMNS, ","))
        for row in rows
            println(
                io,
                join(
                    (
                        _csv_escape(_row_value(row, column, "")) for
                        column in PAIRED_STATISTICS_COLUMNS
                    ),
                    ",",
                ),
            )
        end
    end
    path
end

function _index_replay_rows(rows)
    allowed_modes = Set(spec.value for spec in _MODE_SPECS)
    allowed_transports = Set(
        Iterators.flatten(
            ((comparison.a, comparison.b) for comparison in _COMPARISON_SPECS),
        ),
    )
    index = Dict{Tuple{String, Int, Int, String, String}, Any}()
    formulations = Set{String}()

    for row in rows
        mode = _text_value(_row_value(row, "mode", nothing))
        transport =
            _text_value(_row_value(row, "dual_transport", nothing))
        mode in allowed_modes || continue
        transport in allowed_transports || continue

        formulation =
            _text_value(_row_value(row, "formulation", nothing))
        seed = _integer_value(_row_value(row, "scenario_seed", nothing))
        transition =
            _integer_value(_row_value(row, "transition", nothing))
        isempty(formulation) && throw(
            ArgumentError(
                "A relevant replay row is missing a formulation: $(_case_label(row)).",
            ),
        )
        isnothing(seed) && throw(
            ArgumentError(
                "A relevant replay row has an invalid scenario_seed: $(_case_label(row)).",
            ),
        )
        isnothing(transition) && throw(
            ArgumentError(
                "A relevant replay row has an invalid transition: $(_case_label(row)).",
            ),
        )

        key = (formulation, seed, transition, mode, transport)
        haskey(index, key) && throw(
            ArgumentError("Duplicate replay row for pairing key $(key)."),
        )
        index[key] = row
        push!(formulations, formulation)
    end

    ordered_formulations = sort!(
        collect(formulations);
        by = formulation -> (
            formulation == "reduced" ? 1 :
            formulation == "quasi" ? 2 : 3,
            formulation,
        ),
    )
    index, ordered_formulations
end

function _comparison_row(
    index,
    formulation,
    mode_spec,
    comparison,
    period,
    bootstrap_seed,
    bootstrap_replicates,
)
    a_keys = _pair_keys(
        index,
        formulation,
        mode_spec.value,
        comparison.a,
        period,
    )
    b_keys = _pair_keys(
        index,
        formulation,
        mode_spec.value,
        comparison.b,
        period,
    )
    matched_keys = sort!(collect(intersect(a_keys, b_keys)))
    valid = Tuple{Any, Any}[]
    for (seed, transition) in matched_keys
        a = index[(
            formulation,
            seed,
            transition,
            mode_spec.value,
            comparison.a,
        )]
        b = index[(
            formulation,
            seed,
            transition,
            mode_spec.value,
            comparison.b,
        )]
        _true_value(a, "valid_reference_pair") &&
            _true_value(b, "valid_reference_pair") &&
            push!(valid, (a, b))
    end

    both_converged = 0
    a_only_converged = 0
    b_only_converged = 0
    neither_converged = 0
    convergence_differences = Float64[]
    r0_a = Float64[]
    r0_b = Float64[]
    iterations_a = Float64[]
    iterations_b = Float64[]

    for (a, b) in valid
        a_converged = _true_value(a, "direct_converged")
        b_converged = _true_value(b, "direct_converged")
        if a_converged && b_converged
            both_converged += 1
        elseif a_converged
            a_only_converged += 1
        elseif b_converged
            b_only_converged += 1
        else
            neither_converged += 1
        end
        push!(
            convergence_differences,
            Float64(b_converged) - Float64(a_converged),
        )

        a_r0 = _nonnegative_float(a, "initial_residual_norm2")
        b_r0 = _nonnegative_float(b, "initial_residual_norm2")
        if !isnothing(a_r0) && !isnothing(b_r0)
            push!(r0_a, a_r0)
            push!(r0_b, b_r0)
        end

        # Iterations are outcomes only for jointly converged solves. This is
        # the guard that prevents a failure encoded at the cap from becoming
        # an artificial observation of 1000 iterations.
        if a_converged && b_converged
            a_iterations = _nonnegative_float(a, "total_inner_iters")
            b_iterations = _nonnegative_float(b, "total_inner_iters")
            if !isnothing(a_iterations) && !isnothing(b_iterations)
                push!(iterations_a, a_iterations)
                push!(iterations_b, b_iterations)
            end
        end
    end

    r0_differences = r0_b .- r0_a
    r0_ratios = _paired_ratios(r0_a, r0_b)
    iteration_differences = iterations_b .- iterations_a
    iteration_ratios = _paired_ratios(iterations_a, iterations_b)
    r0_wins, r0_ties, r0_losses = _win_tie_loss(r0_a, r0_b)
    iteration_wins, iteration_ties, iteration_losses =
        _win_tie_loss(iterations_a, iterations_b)

    valid_pairs = length(valid)
    a_converged_total = both_converged + a_only_converged
    b_converged_total = both_converged + b_only_converged
    a_convergence_rate = _rate(a_converged_total, valid_pairs)
    b_convergence_rate = _rate(b_converged_total, valid_pairs)
    convergence_difference =
        _rate(b_converged_total - a_converged_total, valid_pairs)

    group_key = (
        formulation,
        mode_spec.value,
        comparison.label,
        period.label,
    )
    convergence_ci = _bootstrap_interval(
        convergence_differences,
        :mean,
        bootstrap_seed,
        bootstrap_replicates,
        group_key...,
        "convergence",
    )
    r0_difference_ci = _bootstrap_interval(
        r0_differences,
        :median,
        bootstrap_seed,
        bootstrap_replicates,
        group_key...,
        "r0_difference",
    )
    r0_ratio_ci = _bootstrap_interval(
        r0_ratios,
        :median,
        bootstrap_seed,
        bootstrap_replicates,
        group_key...,
        "r0_ratio",
    )
    iteration_difference_ci = _bootstrap_interval(
        iteration_differences,
        :median,
        bootstrap_seed,
        bootstrap_replicates,
        group_key...,
        "iteration_difference",
    )
    iteration_ratio_ci = _bootstrap_interval(
        iteration_ratios,
        :median,
        bootstrap_seed,
        bootstrap_replicates,
        group_key...,
        "iteration_ratio",
    )

    row = Dict{String, Any}(
        "case_id" => join(
            (
                formulation,
                period.label,
                mode_spec.label,
                comparison.label,
            ),
            "__",
        ),
        "formulation" => formulation,
        "period" => period.label,
        "first_transition" => period.first_transition,
        "last_transition" => period.last_transition,
        "mode" => mode_spec.value,
        "mode_label" => mode_spec.label,
        "comparison" => comparison.label,
        "a_transport" => comparison.a,
        "b_transport" => comparison.b,
        "matched_pairs" => length(matched_keys),
        "valid_pairs" => valid_pairs,
        "invalid_reference_pairs" => length(matched_keys) - valid_pairs,
        "a_rows_without_b" => length(setdiff(a_keys, b_keys)),
        "b_rows_without_a" => length(setdiff(b_keys, a_keys)),
        "both_converged" => both_converged,
        "a_only_converged" => a_only_converged,
        "b_only_converged" => b_only_converged,
        "neither_converged" => neither_converged,
        "a_converged_total" => a_converged_total,
        "b_converged_total" => b_converged_total,
        "a_convergence_rate" => a_convergence_rate,
        "b_convergence_rate" => b_convergence_rate,
        "convergence_rate_difference_b_minus_a" =>
            convergence_difference,
        "convergence_rate_difference_b_minus_a_ci_lower" =>
            convergence_ci.lower,
        "convergence_rate_difference_b_minus_a_ci_upper" =>
            convergence_ci.upper,
        "r0_pairs" => length(r0_a),
        "r0_b_wins" => r0_wins,
        "r0_ties" => r0_ties,
        "r0_b_losses" => r0_losses,
        "iteration_pairs" => length(iterations_a),
        "iteration_pairs_excluded_non_both_converged" =>
            valid_pairs - both_converged,
        "iterations_b_wins" => iteration_wins,
        "iterations_ties" => iteration_ties,
        "iterations_b_losses" => iteration_losses,
        "r0_difference_b_minus_a_median_ci_lower" =>
            r0_difference_ci.lower,
        "r0_difference_b_minus_a_median_ci_upper" =>
            r0_difference_ci.upper,
        "r0_ratio_b_over_a_median_ci_lower" => r0_ratio_ci.lower,
        "r0_ratio_b_over_a_median_ci_upper" => r0_ratio_ci.upper,
        "iterations_difference_b_minus_a_median_ci_lower" =>
            iteration_difference_ci.lower,
        "iterations_difference_b_minus_a_median_ci_upper" =>
            iteration_difference_ci.upper,
        "iterations_ratio_b_over_a_median_ci_lower" =>
            iteration_ratio_ci.lower,
        "iterations_ratio_b_over_a_median_ci_upper" =>
            iteration_ratio_ci.upper,
        "bootstrap_seed" => bootstrap_seed,
        "bootstrap_replicates" => bootstrap_replicates,
        "bootstrap_confidence_level" => BOOTSTRAP_CONFIDENCE_LEVEL,
        "difference_definition" => "B - A",
        "ratio_definition" => "B / A; omitted when A == 0",
        "win_definition" => "B is a win when B < A",
        "iteration_eligibility" =>
            "both direct_converged with finite nonnegative iterations",
    )
    _add_distribution!(row, "r0_a", r0_a)
    _add_distribution!(row, "r0_b", r0_b)
    _add_distribution!(
        row,
        "r0_difference_b_minus_a",
        r0_differences,
    )
    _add_distribution!(row, "r0_ratio_b_over_a", r0_ratios)
    _add_distribution!(row, "iterations_a", iterations_a)
    _add_distribution!(row, "iterations_b", iterations_b)
    _add_distribution!(
        row,
        "iterations_difference_b_minus_a",
        iteration_differences,
    )
    _add_distribution!(
        row,
        "iterations_ratio_b_over_a",
        iteration_ratios,
    )
    row
end

function _pair_keys(index, formulation, mode, transport, period)
    Set(
        (key[2], key[3]) for key in keys(index) if
        key[1] == formulation &&
        key[4] == mode &&
        key[5] == transport &&
        period.first_transition <= key[3] <= period.last_transition
    )
end

function _paired_ratios(a, b)
    ratios = Float64[]
    for (a_value, b_value) in zip(a, b)
        a_value > 0.0 && push!(ratios, b_value / a_value)
    end
    ratios
end

function _win_tie_loss(a, b)
    wins = 0
    ties = 0
    losses = 0
    for (a_value, b_value) in zip(a, b)
        if b_value < a_value
            wins += 1
        elseif b_value > a_value
            losses += 1
        else
            ties += 1
        end
    end
    wins, ties, losses
end

function _add_distribution!(row, prefix, values)
    row["$(prefix)_n"] = length(values)
    if isempty(values)
        row["$(prefix)_median"] = NaN
        row["$(prefix)_q25"] = NaN
        row["$(prefix)_q75"] = NaN
        row["$(prefix)_iqr"] = NaN
        return row
    end
    q25 = quantile(values, 0.25)
    q75 = quantile(values, 0.75)
    row["$(prefix)_median"] = median(values)
    row["$(prefix)_q25"] = q25
    row["$(prefix)_q75"] = q75
    row["$(prefix)_iqr"] = q75 - q25
    row
end

function _bootstrap_interval(
    values,
    statistic,
    bootstrap_seed,
    bootstrap_replicates,
    key_parts...,
)
    isempty(values) && return (lower = NaN, upper = NaN)
    rng = MersenneTwister(
        _stable_seed(bootstrap_seed, key_parts...),
    )
    sample = Vector{Float64}(undef, length(values))
    estimates = Vector{Float64}(undef, bootstrap_replicates)
    for replicate in eachindex(estimates)
        for index in eachindex(sample)
            sample[index] = values[rand(rng, eachindex(values))]
        end
        estimates[replicate] =
            statistic === :mean ? mean(sample) : median(sample)
    end
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE_LEVEL
    (
        lower = quantile(estimates, alpha / 2),
        upper = quantile(estimates, 1.0 - alpha / 2),
    )
end

function _stable_seed(parts...)
    state = UInt32(0x811c9dc5)
    for byte in codeunits(join(string.(parts), "|"))
        state = (state ⊻ UInt32(byte)) * UInt32(0x01000193)
    end
    Int(mod(state, UInt32(typemax(Int32) - 1))) + 1
end

function _rate(numerator, denominator)
    denominator == 0 && return NaN
    numerator / denominator
end

function _nonnegative_float(row, name)
    value = _finite_float(_row_value(row, name, nothing))
    isnothing(value) && return nothing
    value < 0.0 && return nothing
    value
end

function _finite_float(value)
    value === nothing && return nothing
    value === missing && return nothing
    value isa Bool && return nothing
    parsed = if value isa Real
        Float64(value)
    else
        tryparse(Float64, strip(string(value)))
    end
    isnothing(parsed) || !isfinite(parsed) ? nothing : parsed
end

function _integer_value(value)
    value === nothing && return nothing
    value === missing && return nothing
    value isa Bool && return nothing
    if value isa Integer
        return Int(value)
    elseif value isa Real
        isfinite(value) && isinteger(value) || return nothing
        return Int(value)
    end
    text = strip(string(value))
    parsed = tryparse(Int, text)
    !isnothing(parsed) && return parsed
    numeric = tryparse(Float64, text)
    isnothing(numeric) && return nothing
    isfinite(numeric) && isinteger(numeric) || return nothing
    Int(numeric)
end

function _true_value(row, name)
    value = _row_value(row, name, false)
    value === true && return true
    value === false && return false
    value isa Integer && return value == 1
    value isa Real && return isfinite(value) && value == 1
    lowercase(strip(string(value))) in ("true", "t", "1", "yes")
end

function _text_value(value)
    value === nothing && return ""
    value === missing && return ""
    strip(string(value))
end

function _row_value(row, name::AbstractString, default)
    if row isa AbstractDict
        haskey(row, name) && return row[name]
        symbol = Symbol(name)
        haskey(row, symbol) && return row[symbol]
        return default
    end
    symbol = Symbol(name)
    hasproperty(row, symbol) ? getproperty(row, symbol) : default
end

function _case_label(row)
    value = _row_value(row, "case_id", "<unknown>")
    isempty(_text_value(value)) ? "<unknown>" : _text_value(value)
end

function _csv_escape(value)
    (value === nothing || value === missing) && return ""
    text = string(value)
    if occursin('"', text) ||
       occursin(',', text) ||
       occursin('\n', text) ||
       occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    text
end

end
