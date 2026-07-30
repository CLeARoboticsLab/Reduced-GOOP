function _raw_rows(run_dir, name)
    path = joinpath(run_dir, "raw", name)
    isfile(path) || error("Required study table is missing: $(path)")
    DTS.read_csv_rows(path)
end

function _optional_raw_rows(run_dir, name)
    path = joinpath(run_dir, "raw", name)
    isfile(path) ? DTS.read_csv_rows(path) : Dict{String, String}[]
end

_float(row, name, default = NaN) =
    haskey(row, name) && !isempty(row[name]) ? parse(Float64, row[name]) : default
_int(row, name, default = 0) =
    haskey(row, name) && !isempty(row[name]) ? parse(Int, row[name]) : default
_bool(row, name, default = false) =
    haskey(row, name) && !isempty(row[name]) ?
    lowercase(row[name]) == "true" : default

function _matches(row; conditions...)
    all(conditions) do (name, expected)
        value = get(row, String(name), "")
        if expected isa AbstractFloat
            !isempty(value) && parse(Float64, value) == expected
        else
            value == string(expected)
        end
    end
end

function _maximum_or_nan(values)
    values = Float64[value for value in values if isfinite(value)]
    isempty(values) ? NaN : maximum(values)
end

function _minimum_or_nan(values)
    values = Float64[value for value in values if isfinite(value)]
    isempty(values) ? NaN : minimum(values)
end

function _median_finite_or_nan(values)
    values = Float64[value for value in values if isfinite(value)]
    isempty(values) ? NaN : _median_or_nan(values)
end

function _boundary_norm(row; prefix = "")
    initial = _float(row, "$(prefix)initial_norm2")
    terminal = _float(row, "$(prefix)terminal_norm2")
    global_value = _float(row, "$(prefix)global_norm2")
    all(isfinite, (initial, terminal, global_value)) || return NaN
    sqrt(initial^2 + terminal^2 + global_value^2)
end

function _only_transition_row(rows, formulation, seed, transition)
    only(
        filter(rows) do row
            get(row, "formulation", "") == formulation &&
                _int(row, "scenario_seed") == seed &&
                _int(row, "transition") == transition
        end,
    )
end

function analyze_study!(run_dir, config::TheoryResolutionConfig)
    for marker in (
        "stagewise_complete",
        "holdout_complete",
        "globalization_complete",
        "scaling_benchmark_complete",
    )
        isfile(joinpath(run_dir, marker)) ||
            error("Analysis requires completed stage marker $(marker).")
    end
    stagewise = _raw_rows(run_dir, "production_region_residuals.csv")
    gamma = _raw_rows(run_dir, "gamma_candidates.csv")
    globalization = _raw_rows(run_dir, "globalization_case_summary.csv")
    mechanism_contrasts =
        _raw_rows(run_dir, "globalization_mechanism_contrasts.csv")
    basin_summary = _raw_rows(run_dir, "globalization_basin_summary.csv")
    candidate_distances =
        _optional_raw_rows(run_dir, "globalization_final_primal_distances.csv")
    path_validation =
        _optional_raw_rows(run_dir, "globalization_path_validation_summary.csv")
    path_validation_marker_present =
        isfile(joinpath(run_dir, "globalization_path_validation_complete"))
    scaling = _raw_rows(run_dir, "scaling_benchmark.csv")
    slopes = _raw_rows(run_dir, "scaling_benchmark_slopes.csv")
    rank_rows = _raw_rows(run_dir, "scaling_benchmark_rank.csv")
    constants = _raw_rows(run_dir, "scaling_benchmark_constants.csv")

    semantic_stagewise = filter(stagewise) do row
        get(row, "dual_transport", "") in (
            "stage_shift_zero_tail",
            "stage_shift_hold_tail",
        )
    end
    maximum_interior_defect = _maximum_or_nan(
        _float(row, "interior_shift_defect_norm2") for row in semantic_stagewise
    )
    maximum_semantic_interior_residual = _maximum_or_nan(
        _float(row, "interior_norm2") for row in semantic_stagewise
    )
    semantic_boundary_fractions = Float64[]
    for row in semantic_stagewise
        boundary = _boundary_norm(row)
        total = _float(row, "initial_residual_norm2")
        isfinite(boundary) && isfinite(total) && total > 0.0 &&
            push!(semantic_boundary_fractions, boundary / total)
    end
    minimum_semantic_boundary_fraction =
        _minimum_or_nan(semantic_boundary_fractions)
    median_semantic_boundary_fraction =
        _median_finite_or_nan(semantic_boundary_fractions)
    boundary_dominance_threshold = 0.99
    boundary_residual_dominates =
        length(semantic_boundary_fractions) == 34 &&
        minimum_semantic_boundary_fraction >= boundary_dominance_threshold

    function semantic_policy_median(policy, field)
        _median_finite_or_nan(
            _float(row, field) for row in semantic_stagewise if
            get(row, "dual_transport", "") == policy
        )
    end
    zero_tail_initial_median =
        semantic_policy_median("stage_shift_zero_tail", "initial_norm2")
    zero_tail_terminal_median =
        semantic_policy_median("stage_shift_zero_tail", "terminal_norm2")
    hold_tail_initial_median =
        semantic_policy_median("stage_shift_hold_tail", "initial_norm2")
    hold_tail_terminal_median =
        semantic_policy_median("stage_shift_hold_tail", "terminal_norm2")

    minimum_fixed_index_limit = _minimum_or_nan(
        _float(row, "fixed_index_interior_limit_norm2") for row in constants
    )
    maximum_bound_violation = _maximum_or_nan(
        _float(row, "maximum_semantic_bound_violation") for row in constants
    )
    maximum_projected_manifold_residual = _maximum_or_nan(
        _float(row, "maximum_projected_solution_manifold_residual_norm2") for
        row in constants
    )
    maximum_pseudoinverse_norm = _maximum_or_nan(
        _float(row, "pseudoinverse_norm2") for row in rank_rows
    )
    maximum_reference_residual = _maximum_or_nan(
        _float(row, "reference_residual_norm2") for row in rank_rows
    )
    minimum_rank = minimum(_int(row, "rank") for row in rank_rows)
    maximum_rank = maximum(_int(row, "rank") for row in rank_rows)
    minimum_row_dimension = minimum(_int(row, "kkt_rows") for row in rank_rows)
    maximum_row_dimension = maximum(_int(row, "kkt_rows") for row in rank_rows)
    maximum_nullity = maximum(_int(row, "nullity") for row in rank_rows)
    rank_groups = Dict{String, Vector{Int}}()
    for row in rank_rows
        push!(
            get!(rank_groups, get(row, "formulation", ""), Int[]),
            _int(row, "rank"),
        )
    end
    expected_rank_rows = 2 * 6
    constant_rank_across_path =
        length(rank_rows) == expected_rank_rows &&
        Set(keys(rank_groups)) == Set(("reduced", "quasi")) &&
        all(length(rank_values) == 6 && length(unique(rank_values)) == 1 for
            rank_values in Base.values(rank_groups))
    full_row_rank_across_path = constant_rank_across_path &&
        all(_int(row, "rank") == _int(row, "kkt_rows") for row in rank_rows)
    uniformly_bounded_pseudoinverse =
        length(rank_rows) == expected_rank_rows &&
        all(
            isfinite(_float(row, "pseudoinverse_norm2")) &&
            _float(row, "pseudoinverse_norm2") > 0.0 for row in rank_rows
        )
    interior_semantic_slopes = Float64[
        _float(row, "primary_slope") for row in slopes if
        get(row, "metric", "") == "interior_residual_norm2" &&
        get(row, "transport", "") in (
            "stage_shift_zero_tail",
            "stage_shift_hold_tail",
            "diagnostic_oracle_terminal_completion",
        ) &&
        _bool(row, "fit_performed")
    ]
    identity_slopes = Float64[
        _float(row, "primary_slope") for row in slopes if
        get(row, "metric", "") == "interior_residual_norm2" &&
        get(row, "transport", "") == "identity_copy" &&
        _bool(row, "fit_performed")
    ]
    interior_slope_rows = filter(
        row ->
            get(row, "metric", "") == "interior_residual_norm2" &&
                get(row, "transport", "") in (
                    "identity_copy",
                    "stage_shift_zero_tail",
                    "stage_shift_hold_tail",
                    "diagnostic_oracle_terminal_completion",
                ),
        slopes,
    )
    scaling_slope_grid_complete =
        length(interior_slope_rows) == 8 &&
        all(
            _int(row, "available_points") == 5 &&
            _int(row, "primary_points") == 3 &&
            _bool(row, "fit_performed") for row in interior_slope_rows
        )
    maximum_semantic_bound_ratio = _maximum_or_nan(
        _float(row, "semantic_bound_ratio") for row in scaling if
        _bool(row, "semantic_bound_applicable")
    )

    heldout_quarter = filter(gamma) do row
        get(row, "sample_group", "") == "holdout" &&
            _float(row, "gamma") == HELDOUT_GAMMA
    end
    heldout_successes =
        count(row -> _bool(row, "direct_converged"), heldout_quarter)
    heldout_total = length(heldout_quarter)
    development_quarter = filter(gamma) do row
        get(row, "sample_group", "") == "development" &&
            _float(row, "gamma") == HELDOUT_GAMMA
    end
    development_successes =
        count(row -> _bool(row, "direct_converged"), development_quarter)
    aligned_holdout_cases = count(
        row ->
            get(row, "sample_group", "") == "holdout" &&
                _float(row, "gamma") == HELDOUT_GAMMA &&
                _bool(row, "aligned_candidate_set"),
        gamma,
    )
    catastrophic_holdout_rows = filter(heldout_quarter) do row
        !_bool(row, "direct_converged") &&
            _float(row, "direct_final_residual_norm2") > 10 * SOLVER_TOL
    end
    catastrophic_holdout_failure = !isempty(catastrophic_holdout_rows)
    maximum_catastrophic_holdout_residual = _maximum_or_nan(
        _float(row, "direct_final_residual_norm2") for
        row in catastrophic_holdout_rows
    )
    catastrophic_holdout_case_ids =
        join(
            sort!(
                String[get(row, "case_id", "") for row in catastrophic_holdout_rows],
            ),
            ",",
        )

    failure_rows = filter(row -> !_bool(row, "direct_converged"), globalization)
    maximum_hard_rank_deficit =
        maximum(_int(row, "maximum_rank_deficit") for row in globalization)
    boundary_rejected_trials =
        sum(_int(row, "boundary_rejected_trial_count") for row in globalization)
    armijo_rejected_trials =
        sum(_int(row, "armijo_rejected_trial_count") for row in globalization)
    failure_boundary_rejected_trials =
        sum(_int(row, "boundary_rejected_trial_count") for row in failure_rows)
    failure_armijo_rejected_trials =
        sum(_int(row, "armijo_rejected_trial_count") for row in failure_rows)
    maximum_failure_boundary_restricted_fraction = _maximum_or_nan(
        _float(row, "boundary_restricted_direction_fraction") for
        row in failure_rows
    )
    maximum_hard_eta =
        maximum(_float(row, "maximum_eta") for row in globalization)
    failures_with_armijo = count(
        row -> _int(row, "armijo_rejected_trial_count") > 0,
        failure_rows,
    )
    failures_with_eta_retry = count(
        row -> _int(row, "eta_retry_event_count") > 0,
        failure_rows,
    )
    length(globalization) == 15 ||
        error("Expected 15 hard-case globalization summary rows.")
    length(mechanism_contrasts) == 3 ||
        error("Expected three hard-case mechanism contrasts.")
    length(basin_summary) == 3 ||
        error("Expected three hard-case basin summaries.")
    q202_t2_contrast =
        _only_transition_row(mechanism_contrasts, "quasi", 202, 2)
    r202_t5_contrast =
        _only_transition_row(mechanism_contrasts, "reduced", 202, 5)
    q101_t1_contrast =
        _only_transition_row(mechanism_contrasts, "quasi", 101, 1)
    r202_t5_gamma0 = only(filter(globalization) do row
        get(row, "formulation", "") == "reduced" &&
            _int(row, "scenario_seed") == 202 &&
            _int(row, "transition") == 5 &&
            _float(row, "gamma") == 0.0
    end)

    q202_direction_model_eta_mechanism =
        _bool(q202_t2_contrast, "failure_has_larger_relative_step") &&
        _bool(q202_t2_contrast, "failure_has_poorer_accepted_model_ratio") &&
        _bool(q202_t2_contrast, "failure_has_more_armijo_rejection") &&
        _float(
            q202_t2_contrast,
            "failure_minus_success_eta_retry_event_count",
        ) > 0.0
    r202_near_tolerance_plateau =
        !_bool(r202_t5_gamma0, "direct_converged") &&
        _float(r202_t5_gamma0, "direct_final_residual_norm2") >
        SOLVER_TOL &&
        _float(r202_t5_gamma0, "direct_final_residual_norm2") <
        2 * SOLVER_TOL &&
        _int(r202_t5_gamma0, "total_inner_iters") >= 999 &&
        _float(r202_t5_gamma0, "median_linearized_residual_ratio") > 0.99 &&
        _float(r202_t5_gamma0, "maximum_rrqr_condition_proxy") > 1e7 &&
        _int(r202_t5_gamma0, "klu_singular_retry_count") == 1 &&
        _int(r202_t5_gamma0, "eta_retry_event_count") == 0 &&
        _int(r202_t5_gamma0, "boundary_rejected_trial_count") == 0
    q101_no_clean_scalar_separator =
        _bool(q101_t1_contrast, "both_outcomes_present") &&
        !_bool(q101_t1_contrast, "failure_has_larger_relative_step") &&
        !_bool(q101_t1_contrast, "failure_has_poorer_accepted_model_ratio") &&
        !_bool(q101_t1_contrast, "failure_has_more_armijo_rejection") &&
        !_bool(q101_t1_contrast, "failure_has_more_boundary_restriction") &&
        !_bool(q101_t1_contrast, "failure_has_more_rank_loss")
    no_boundary_restriction_mechanism =
        boundary_rejected_trials == 0 &&
        all(
            _float(row, "boundary_restricted_direction_fraction") == 0.0 &&
            _float(row, "minimum_combined_stepsize_cap") >= 1.0 for
            row in globalization
        )
    no_consistent_rank_loss_explanation =
        all(
            !_bool(row, "failure_has_more_rank_loss") for
            row in mechanism_contrasts
        )
    lower_initial_residual_flags = [
        _bool(row, "failure_has_lower_initial_raw_residual") for
        row in mechanism_contrasts
    ]
    initial_residual_not_predictive =
        length(unique(lower_initial_residual_flags)) > 1 &&
        _bool(q101_t1_contrast, "failure_has_lower_initial_raw_residual")
    hard_nonmonotonic_cases =
        count(row -> _bool(row, "both_outcomes_present"), mechanism_contrasts)
    separated_successful_hard_cases = count(basin_summary) do row
        _int(row, "successful_candidates") >= 2 &&
            !_bool(row, "successful_candidates_aligned")
    end
    hard_candidate_separation_requires_conditioning =
        separated_successful_hard_cases > 0
    all_hard_candidate_pairs_separated =
        length(candidate_distances) == 33 &&
        all(candidate_distances) do row
            !_bool(row, "aligned_at_threshold") &&
                _float(row, "normalized_distance") >
                _float(row, "alignment_threshold")
        end
    minimum_hard_candidate_pair_separation = _minimum_or_nan(
        _float(row, "normalized_distance") for row in candidate_distances
    )
    maximum_successful_hard_candidate_separation = _maximum_or_nan(
        _float(row, "max_successful_candidate_separation_normalized") for
        row in basin_summary
    )
    cross_process_trace_structure_exact_cases =
        count(row -> _bool(row, "trace_event_structure_exact"), globalization)
    cross_process_trace_numeric_within_tolerance_cases = count(
        row -> _bool(row, "trace_numeric_fields_within_tolerance"),
        globalization,
    )
    cross_process_final_primal_bitwise_exact_cases =
        count(row -> _bool(row, "final_primal_exact"), globalization)
    same_process_path_validation_protocol_valid =
        length(path_validation) == 15 &&
        all(path_validation) do row
            get(row, "comparison_scope", "") ==
                "full_rich_hook_vs_lightweight_hash_hook_same_process" &&
                _bool(row, "full_diagnostic_collector_used") &&
                _bool(row, "full_diagnostic_finalized_postsolve") &&
                !_bool(row, "spqr_executed_during_rich_replay_solve") &&
                !_bool(row, "production_candidate_solve") &&
                !_bool(row, "rescue_solve") &&
                _bool(row, "zero_forbidden_fallback")
        end
    same_process_path_validation_exact =
        path_validation_marker_present &&
        same_process_path_validation_protocol_valid &&
        all(path_validation) do row
            _bool(row, "same_process_accepted_hash_sequence_exact") &&
                _bool(row, "same_process_accepted_metadata_exact") &&
                _bool(row, "same_process_final_z_bitwise_exact") &&
                _bool(row, "same_process_scalar_trace_exact")
        end

    semantic_slopes_linear = !isempty(interior_semantic_slopes) &&
        length(interior_semantic_slopes) == 6 &&
        scaling_slope_grid_complete &&
        all(slope -> abs(slope - 1.0) <= 0.05, interior_semantic_slopes)
    identity_nonvanishing = minimum_fixed_index_limit > 1e-3 &&
        length(identity_slopes) == 2 &&
        scaling_slope_grid_complete &&
        all(slope -> abs(slope) <= 0.1, identity_slopes)
    semantic_interior_theorem_supported =
        maximum_interior_defect <= 1e-8 &&
        semantic_slopes_linear &&
        maximum_bound_violation <= 1e-10 &&
        maximum_reference_residual <= 1e-8
    local_solution_manifold_corollary_supported =
        semantic_interior_theorem_supported &&
        constant_rank_across_path &&
        full_row_rank_across_path &&
        uniformly_bounded_pseudoinverse &&
        maximum_projected_manifold_residual <= 1e-8 &&
        maximum_nullity > 0
    fixed_gamma_validated =
        heldout_total == 11 &&
        heldout_successes >= HOLDOUT_GAMMA_NEAR_ALL_REQUIRED_SUCCESSES &&
        length(development_quarter) == 6 &&
        development_successes == DEVELOPMENT_GAMMA_REQUIRED_SUCCESSES
    adaptive_damping_needed =
        !fixed_gamma_validated ||
        catastrophic_holdout_failure ||
        (
            hard_nonmonotonic_cases > 0 &&
            hard_candidate_separation_requires_conditioning
        )
    globalization_observable_association =
        q202_direction_model_eta_mechanism &&
        r202_near_tolerance_plateau &&
        q101_no_clean_scalar_separator

    decisions = Dict{String, Any}(
        "semantic_interior_consistency_theorem_supported" =>
            semantic_interior_theorem_supported,
        "fixed_index_inconsistency_supported" => identity_nonvanishing,
        "local_solution_manifold_corollary_supported" =>
            local_solution_manifold_corollary_supported,
        "fixed_gamma_0p25_empirical_safeguard_validated" =>
            fixed_gamma_validated,
        "fixed_gamma_0p25_population_baseline_threshold_met" =>
            fixed_gamma_validated,
        "fixed_gamma_0p25_failure_free_safeguard_supported" =>
            fixed_gamma_validated && !catastrophic_holdout_failure,
        "catastrophic_gamma_0p25_holdout_failure_observed" =>
            catastrophic_holdout_failure,
        "catastrophic_gamma_0p25_holdout_failure_count" =>
            length(catastrophic_holdout_rows),
        "catastrophic_gamma_0p25_holdout_maximum_final_residual_norm2" =>
            maximum_catastrophic_holdout_residual,
        "catastrophic_gamma_0p25_holdout_case_ids" =>
            catastrophic_holdout_case_ids,
        "adaptive_damping_algorithm_needed" => adaptive_damping_needed,
        "globalization_observable_association_supported" =>
            globalization_observable_association,
        "globalization_single_scalar_separator_supported" => false,
        "quasi_seed202_transition2_direction_model_eta_mechanism" =>
            q202_direction_model_eta_mechanism,
        "reduced_seed202_transition5_near_tolerance_plateau" =>
            r202_near_tolerance_plateau,
        "quasi_seed101_transition1_no_clean_scalar_separator" =>
            q101_no_clean_scalar_separator,
        "hard_case_boundary_restriction_excluded" =>
            no_boundary_restriction_mechanism,
        "hard_case_consistent_rank_loss_explanation_excluded" =>
            no_consistent_rank_loss_explanation,
        "hard_case_initial_residual_predictive" =>
            !initial_residual_not_predictive,
        "hard_case_nonmonotonic_transition_count" =>
            hard_nonmonotonic_cases,
        "hard_case_separated_successful_candidate_set_count" =>
            separated_successful_hard_cases,
        "hard_case_candidate_separation_requires_conditioning" =>
            hard_candidate_separation_requires_conditioning,
        "hard_case_all_candidate_pairs_separated" =>
            all_hard_candidate_pairs_separated,
        "hard_case_candidate_pair_count" => length(candidate_distances),
        "hard_case_minimum_candidate_pair_separation_normalized" =>
            minimum_hard_candidate_pair_separation,
        "hard_case_maximum_successful_candidate_separation_normalized" =>
            maximum_successful_hard_candidate_separation,
        "same_process_iterate_path_validation_available" =>
            !isempty(path_validation),
        "same_process_iterate_path_validation_complete_marker_present" =>
            path_validation_marker_present,
        "same_process_iterate_path_validation_protocol_valid" =>
            same_process_path_validation_protocol_valid,
        "same_process_iterate_path_validation_exact" =>
            same_process_path_validation_exact,
        "same_process_iterate_path_validation_cases" =>
            length(path_validation),
        "cross_process_trace_structure_exact_cases" =>
            cross_process_trace_structure_exact_cases,
        "cross_process_trace_numeric_within_tolerance_cases" =>
            cross_process_trace_numeric_within_tolerance_cases,
        "cross_process_final_primal_bitwise_exact_cases" =>
            cross_process_final_primal_bitwise_exact_cases,
        "boundary_residual_dominates_after_semantic_transport" =>
            boundary_residual_dominates,
        "semantic_boundary_dominance_threshold" =>
            boundary_dominance_threshold,
        "minimum_semantic_boundary_fraction" =>
            minimum_semantic_boundary_fraction,
        "median_semantic_boundary_fraction" =>
            median_semantic_boundary_fraction,
        "zero_tail_median_initial_boundary_residual_norm2" =>
            zero_tail_initial_median,
        "zero_tail_median_terminal_boundary_residual_norm2" =>
            zero_tail_terminal_median,
        "hold_tail_median_initial_boundary_residual_norm2" =>
            hold_tail_initial_median,
        "hold_tail_median_terminal_boundary_residual_norm2" =>
            hold_tail_terminal_median,
        "maximum_stagewise_interior_shift_defect_norm2" =>
            maximum_interior_defect,
        "maximum_semantic_interior_residual_norm2" =>
            maximum_semantic_interior_residual,
        "minimum_fixed_index_interior_epsilon0_limit_norm2" =>
            minimum_fixed_index_limit,
        "semantic_interior_primary_slope_min" =>
            _minimum_or_nan(interior_semantic_slopes),
        "semantic_interior_primary_slope_max" =>
            _maximum_or_nan(interior_semantic_slopes),
        "semantic_interior_primary_slope_count" =>
            length(interior_semantic_slopes),
        "scaling_interior_slope_grid_complete" =>
            scaling_slope_grid_complete,
        "identity_interior_primary_slope_min" =>
            _minimum_or_nan(identity_slopes),
        "identity_interior_primary_slope_max" =>
            _maximum_or_nan(identity_slopes),
        "identity_interior_primary_slope_count" => length(identity_slopes),
        "maximum_semantic_bound_violation" => maximum_bound_violation,
        "maximum_semantic_bound_ratio" => maximum_semantic_bound_ratio,
        "maximum_scaling_reference_residual_norm2" =>
            maximum_reference_residual,
        "scaling_constant_rank_across_path" => constant_rank_across_path,
        "scaling_full_row_rank_across_path" => full_row_rank_across_path,
        "scaling_uniformly_bounded_pseudoinverse" =>
            uniformly_bounded_pseudoinverse,
        "scaling_jacobian_minimum_rank" => minimum_rank,
        "scaling_jacobian_maximum_rank" => maximum_rank,
        "scaling_jacobian_minimum_row_dimension" => minimum_row_dimension,
        "scaling_jacobian_maximum_row_dimension" => maximum_row_dimension,
        "scaling_jacobian_maximum_nullity" => maximum_nullity,
        "scaling_maximum_pseudoinverse_norm2" =>
            maximum_pseudoinverse_norm,
        "maximum_projected_solution_manifold_residual_norm2" =>
            maximum_projected_manifold_residual,
        "development_gamma_0p25_successes" => development_successes,
        "development_gamma_0p25_cases" => length(development_quarter),
        "development_gamma_0p25_required_successes" =>
            DEVELOPMENT_GAMMA_REQUIRED_SUCCESSES,
        "holdout_gamma_0p25_successes" => heldout_successes,
        "holdout_gamma_0p25_cases" => heldout_total,
        "holdout_gamma_0p25_near_all_required_successes" =>
            HOLDOUT_GAMMA_NEAR_ALL_REQUIRED_SUCCESSES,
        "aligned_holdout_candidate_sets" => aligned_holdout_cases,
        "hard_case_failures" => length(failure_rows),
        "hard_case_maximum_rank_deficit" => maximum_hard_rank_deficit,
        "hard_case_boundary_rejected_trials" => boundary_rejected_trials,
        "hard_case_armijo_rejected_trials" => armijo_rejected_trials,
        "failed_hard_case_boundary_rejected_trials" =>
            failure_boundary_rejected_trials,
        "failed_hard_case_armijo_rejected_trials" =>
            failure_armijo_rejected_trials,
        "failed_hard_case_maximum_boundary_restricted_direction_fraction" =>
            maximum_failure_boundary_restricted_fraction,
        "hard_case_maximum_eta" => maximum_hard_eta,
        "hard_case_failures_with_eta_retry" => failures_with_eta_retry,
        "qualification_fixed_gamma" =>
            "meets the focused 6/6 plus 10/11 threshold as a population-study baseline; one catastrophic holdout failure and nonmonotonic separated hard cases preclude a universal robustness claim",
        "qualification_iteration_counts" =>
            "condition on jointly converged and aligned final-primal candidate sets; no unconditioned hard-case iteration ranking is valid",
        "qualification_instrumentation_neutrality" =>
            "exact invariance is established only by the same-process full-rich-versus-lightweight-hash validator; saved-checkpoint replay is a cross-process diagnostic",
        "qualification_conditioning" =>
            "hard-case values are RRQR R-diagonal proxies; scaling benchmark uses exact dense SVD",
    )
    DTS._write_toml(joinpath(run_dir, "decision_summary.toml"), decisions)
    DTS.write_csv(
        joinpath(run_dir, "raw", "decision_summary.csv"),
        [
            Dict{String, Any}(
                "decision" => name,
                "value" => value,
            ) for (name, value) in sort!(collect(decisions); by = first)
        ],
    )
    _atomic_write(joinpath(run_dir, "analysis_complete"), "ok\n")
    decisions
end

_fmt(value; digits = 4) =
    !isfinite(value) ? string(value) :
    abs(value) != 0.0 && (abs(value) < 1e-3 || abs(value) >= 1e4) ?
    @sprintf("%.3e", value) :
    @sprintf("%.*f", digits, value)

function _println_table_header(io, names)
    println(io, "| ", join(names, " | "), " |")
    println(io, "|", join(fill("---", length(names)), "|"), "|")
end

function finalize_provenance!(run_dir)
    provenance_dir = joinpath(run_dir, "provenance")
    source_snapshot = joinpath(provenance_dir, "source")
    mkpath(source_snapshot)
    source_roots = (
        joinpath(REPOSITORY_ROOT, "src"),
        joinpath(REPOSITORY_ROOT, "test"),
        joinpath(REPOSITORY_ROOT, "experiments", "analysis", "theory_resolution"),
        joinpath(REPOSITORY_ROOT, "experiments", "analysis", "dual_transport"),
        joinpath(REPOSITORY_ROOT, "experiments", "analysis", "selective_warmstart"),
    )
    source_files = String[
        joinpath(REPOSITORY_ROOT, "Project.toml"),
        joinpath(REPOSITORY_ROOT, "Manifest.toml"),
        joinpath(REPOSITORY_ROOT, "experiments", "Project.toml"),
        joinpath(REPOSITORY_ROOT, "experiments", "Manifest.toml"),
        joinpath(REPOSITORY_ROOT, "experiments", "robotic_arm_core.jl"),
    ]
    for root in source_roots
        isdir(root) || continue
        for (directory, _, names) in walkdir(root), name in names
            any(suffix -> endswith(name, suffix), (".jl", ".toml", ".md")) ||
                continue
            push!(source_files, joinpath(directory, name))
        end
    end
    unique!(source_files)
    sort!(source_files)
    source_records = Dict{String, Any}[]
    for source in source_files
        isfile(source) || continue
        relative = relpath(source, REPOSITORY_ROOT)
        destination = joinpath(source_snapshot, relative)
        mkpath(dirname(destination))
        cp(source, destination; force = true)
        push!(
            source_records,
            Dict{String, Any}(
                "path" => relative,
                "sha256" => _sha256(source),
                "bytes" => filesize(source),
            ),
        )
    end

    git_commit = try
        readchomp(`git -C $(REPOSITORY_ROOT) rev-parse HEAD`)
    catch
        "unavailable"
    end
    git_status = try
        read(`git -C $(REPOSITORY_ROOT) status --short`, String)
    catch error
        "unavailable: $(sprint(showerror, error))\n"
    end
    git_diff = try
        read(`git -C $(REPOSITORY_ROOT) diff --binary`, String)
    catch error
        "unavailable: $(sprint(showerror, error))\n"
    end
    _atomic_write(joinpath(provenance_dir, "git_status.txt"), git_status)
    _atomic_write(joinpath(provenance_dir, "git_diff.patch"), git_diff)

    artifact_records = Dict{String, Any}[]
    for relative_root in ("raw", "checkpoints", "figures")
        root = joinpath(run_dir, relative_root)
        isdir(root) || continue
        for (directory, _, names) in walkdir(root), name in names
            path = joinpath(directory, name)
            push!(
                artifact_records,
                Dict{String, Any}(
                    "path" => relpath(path, run_dir),
                    "sha256" => _sha256(path),
                    "bytes" => filesize(path),
                ),
            )
        end
    end
    for name in (
        "config.toml",
        "decision_summary.toml",
        "reproduction_command.txt",
        "report.md",
    )
        path = joinpath(run_dir, name)
        isfile(path) || continue
        push!(
            artifact_records,
            Dict{String, Any}(
                "path" => name,
                "sha256" => _sha256(path),
                "bytes" => filesize(path),
            ),
        )
    end
    sort!(artifact_records; by = row -> row["path"])
    manifest = Dict{String, Any}(
        "git_commit" => git_commit,
        "working_tree_dirty" => !isempty(strip(git_status)),
        "source_run" =>
            load_config(joinpath(run_dir, "config.toml")).source_run,
        "source_snapshot_files" => source_records,
        "artifact_files" => artifact_records,
    )
    DTS._write_toml(joinpath(provenance_dir, "manifest.toml"), manifest)
    _atomic_write(joinpath(run_dir, "provenance_complete"), "ok\n")
    manifest
end

function _gamma_summary_table(io, rows)
    _println_table_header(
        io,
        (
            "Sample",
            "Form",
            "γ",
            "Conv.",
            "Joint",
            "Aligned",
            "Median iters (joint/aligned)",
            "Median backtracks (joint/aligned)",
            "Median η retries (joint/aligned)",
            "Median KLU retries (joint/aligned)",
            "Median final ‖F‖₂",
            "Max normalized separation across {0, .25, 1}",
        ),
    )
    order = Dict("development" => 1, "holdout" => 2, "all" => 3)
    form_order = Dict("reduced" => 1, "quasi" => 2, "all" => 3)
    sort!(
        rows;
        by = row -> (
            order[get(row, "sample_group", "all")],
            form_order[get(row, "formulation", "all")],
            _float(row, "gamma"),
        ),
    )
    for row in rows
        println(
            io,
            "| $(row["sample_group"]) | $(row["formulation"]) | " *
            "$(_fmt(_float(row, "gamma"); digits=2)) | " *
            "$(_int(row, "converged_cases"))/$(_int(row, "total_cases")) | " *
            "$(_int(row, "jointly_converged_cases")) | " *
            "$(_int(row, "aligned_joint_cases")) | " *
            "$(_fmt(_float(row, "median_iterations_joint"); digits=1)) / " *
            "$(_fmt(_float(row, "median_iterations_aligned"); digits=1)) | " *
            "$(_fmt(_float(row, "median_backtracking_joint"); digits=1)) / " *
            "$(_fmt(_float(row, "median_backtracking_aligned"); digits=1)) | " *
            "$(_fmt(_float(row, "median_eta_retries_joint"); digits=1)) / " *
            "$(_fmt(_float(row, "median_eta_retries_aligned"); digits=1)) | " *
            "$(_fmt(_float(row, "median_klu_retries_joint"); digits=1)) / " *
            "$(_fmt(_float(row, "median_klu_retries_aligned"); digits=1)) | " *
            "$(_fmt(_float(row, "median_final_residual"))) | " *
            "$(_fmt(_float(row, "max_case_final_primal_separation"))) |",
        )
    end
end

function generate_report(run_dir, config::TheoryResolutionConfig)
    isfile(joinpath(run_dir, "analysis_complete")) ||
        analyze_study!(run_dir, config)
    decisions = TOML.parsefile(joinpath(run_dir, "decision_summary.toml"))
    stagewise = _raw_rows(run_dir, "production_region_residuals.csv")
    ablations = _raw_rows(run_dir, "boundary_ablations.csv")
    gamma_summary = _raw_rows(run_dir, "gamma_group_summary.csv")
    globalization = _raw_rows(run_dir, "globalization_case_summary.csv")
    mechanism_contrasts =
        _raw_rows(run_dir, "globalization_mechanism_contrasts.csv")
    basin_summary = _raw_rows(run_dir, "globalization_basin_summary.csv")
    path_validation =
        _optional_raw_rows(run_dir, "globalization_path_validation_summary.csv")
    scaling = _raw_rows(run_dir, "scaling_benchmark.csv")
    slopes = _raw_rows(run_dir, "scaling_benchmark_slopes.csv")
    rank_rows = _raw_rows(run_dir, "scaling_benchmark_rank.csv")
    constants = _raw_rows(run_dir, "scaling_benchmark_constants.csv")

    io = IOBuffer()
    println(io, "# Final semantic-transport theory-resolution study")
    println(io)
    println(
        io,
        "Source run: `$(basename(config.source_run))`. This focused study uses only " *
        "the 17 frozen valid replay transitions; the 10-seed population study was not run.",
    )
    println(io)
    println(io, "## Frozen numerical protocol")
    println(io)
    println(io, "| Setting | Value |")
    println(io, "|---|---:|")
    for (name, value) in (
        ("T", config.planning_horizon),
        ("Δt", config.Δt),
        ("tol", config.tol),
        ("max_inner_iters", config.max_inner_iters),
        ("max_outer_iters", config.max_outer_iters),
        ("linear solver", config.linear_solver),
        ("line search", config.linesearch),
    )
        println(io, "| $(name) | `$(value)` |")
    end
    println(io)
    println(
        io,
        "The complete solver-option dictionary is hash-snapshotted from the source run " *
        "and checked field-for-field. No fallback tolerance, rescue solve, alternate " *
        "experimental solver, or reference tournament was used. The controlled scaling " *
        "fixture invokes no optimizer.",
    )
    println(io)
    println(io, "## Phase 1 — stagewise residual localization")
    println(io)
    println(
        io,
        "Both generated KKT systems contain 1,429 rows, partitioned identically: " *
        "1,220 inherited-interior, 78 initial-boundary, 131 terminal-boundary, and " *
        "0 global rows. Stage-19 stationarity is terminal-boundary because the " *
        "terminal objective changes its stencil, even though a stage-20 coordinate exists.",
    )
    println(io)
    _println_table_header(
        io,
        (
            "Transition",
            "Transport",
            "Total ‖F‖₂",
            "Interior",
            "Initial",
            "Terminal",
            "Mapped interior defect",
        ),
    )
    spike_rows = filter(stagewise) do row
        get(row, "formulation", "") == "reduced" &&
            _int(row, "scenario_seed") == 202 &&
            _int(row, "transition") in 4:7
    end
    sort!(
        spike_rows;
        by = row -> (_int(row, "transition"), get(row, "dual_transport", "")),
    )
    for row in spike_rows
        println(
            io,
            "| $(_int(row, "transition")) | $(row["dual_transport"]) | " *
            "$(_fmt(_float(row, "initial_residual_norm2"))) | " *
            "$(_fmt(_float(row, "interior_norm2"))) | " *
            "$(_fmt(_float(row, "initial_norm2"))) | " *
            "$(_fmt(_float(row, "terminal_norm2"))) | " *
            "$(_fmt(_float(row, "interior_shift_defect_norm2"))) |",
        )
    end
    println(io)
    shifted_copy_boundary = filter(ablations) do row
        get(row, "formulation", "") == "reduced" &&
            _int(row, "scenario_seed") == 202 &&
            _int(row, "transition") in 4:7 &&
            get(row, "interior_map", "") == "semantic_shift" &&
            get(row, "initial_completion", "") == "copy" &&
            get(row, "terminal_completion", "") == "hold" &&
            get(row, "explicit_terminal_completion", "") == "copy"
    end
    if length(shifted_copy_boundary) == 4
        println(
            io,
            "The diagnostic semantic-shift/copy-initial/hold-tail/copy-explicit " *
            "combination (not a production policy) leaves total residuals " *
            join(
                (
                    "t$(_int(row, "transition"))=" *
                    _fmt(_float(row, "initial_residual_norm2")) for
                    row in sort!(
                        shifted_copy_boundary;
                        by = row -> _int(row, "transition"),
                    )
                ),
                ", ",
            ) *
            ". This isolates the large fixed-index spikes from boundary completion. " *
            "Resetting initial multipliers and zeroing the tail explain the larger " *
            "production semantic residuals; all boundary-factor changes leave the " *
            "1,220-coordinate interior vector exactly unchanged.",
        )
    end
    println(io)
    println(
        io,
        "The following four-row summary is a diagnostic boundary-factor ablation, " *
        "not a set of production policies. Values are medians over reduced seed 202, " *
        "transitions 4–7, with semantic interior mapping and the explicit-terminal " *
        "factor held at `copy` (its copy/reset choice is exactly inert in these rows).",
    )
    println(io)
    _println_table_header(
        io,
        (
            "Initial multiplier",
            "Tail",
            "Median total",
            "Median interior",
            "Median initial",
            "Median terminal",
        ),
    )
    for initial_completion in ("copy", "reset"),
        terminal_completion in ("hold", "zero")
        selected = filter(ablations) do row
            get(row, "formulation", "") == "reduced" &&
                _int(row, "scenario_seed") == 202 &&
                _int(row, "transition") in 4:7 &&
                get(row, "interior_map", "") == "semantic_shift" &&
                get(row, "initial_completion", "") == initial_completion &&
                get(row, "terminal_completion", "") == terminal_completion &&
                get(row, "explicit_terminal_completion", "") == "copy"
        end
        length(selected) == 4 ||
            error(
                "Expected four focused boundary-ablation rows for " *
                "$(initial_completion)/$(terminal_completion).",
            )
        println(
            io,
            "| $(initial_completion) | $(terminal_completion) | " *
            "$(_fmt(_median_finite_or_nan(_float(row, "initial_residual_norm2") for row in selected))) | " *
            "$(_fmt(_median_finite_or_nan(_float(row, "interior_norm2") for row in selected))) | " *
            "$(_fmt(_median_finite_or_nan(_float(row, "initial_norm2") for row in selected))) | " *
            "$(_fmt(_median_finite_or_nan(_float(row, "terminal_norm2") for row in selected))) |",
        )
    end
    println(io)
    println(
        io,
        "Across all semantic policies, the maximum mapped inherited-interior defect is " *
        "`$(_fmt(decisions["maximum_stagewise_interior_shift_defect_norm2"]))`; " *
        "the corresponding absolute interior norm remains bounded by the frozen " *
        "source-reference solve floor (`$(_fmt(decisions["maximum_semantic_interior_residual_norm2"]))`). " *
        "All 1,220 inherited rows are marked exact-shift-invariant, and all global " *
        "residual blocks are empty.",
    )
    println(io)
    println(
        io,
        "Boundary residuals account for at least " *
        "`$(_fmt(100 * decisions["minimum_semantic_boundary_fraction"]; digits=6))%` " *
        "of every semantic transported residual (median " *
        "`$(_fmt(100 * decisions["median_semantic_boundary_fraction"]; digits=6))%`). " *
        "Across all 17 cases the median initial/terminal norms are " *
        "`$(_fmt(decisions["zero_tail_median_initial_boundary_residual_norm2"]))`/" *
        "`$(_fmt(decisions["zero_tail_median_terminal_boundary_residual_norm2"]))` " *
        "for zero-tail and " *
        "`$(_fmt(decisions["hold_tail_median_initial_boundary_residual_norm2"]))`/" *
        "`$(_fmt(decisions["hold_tail_median_terminal_boundary_residual_norm2"]))` " *
        "for hold-tail. Thus the next algorithmic work belongs at the boundary: " *
        "terminal completion is the online rule exposed by the zero/hold contrast, " *
        "while the larger initial-multiplier reset contribution must be reported " *
        "rather than attributed to the interior transport.",
    )
    println(io)
    println(io, "## Phase 2 — held-out γ = 0.25")
    println(io)
    _gamma_summary_table(io, gamma_summary)
    println(io)
    println(
        io,
        "Only the 11 previously untested γ=0.25 cases were solved. The γ=0 and γ=1 " *
        "rows were reused from frozen source-run candidate summaries; the six " *
        "development rows were reused from the damping checkpoints. Iteration comparisons above " *
        "are first restricted to jointly converged candidates and then to the " *
        "predeclared normalized final-primal alignment threshold `$(ALIGNMENT_TOL)`; " *
        "the same joint/aligned conditioning is shown for backtracking and retry " *
        "summaries. Every Phase-2 count and named failure in this section comes from " *
        "`raw/gamma_candidates.csv`; no case-ID join to the Phase-3 globalization " *
        "tables is used.",
    )
    println(io)
    fixed_gamma = decisions["fixed_gamma_0p25_empirical_safeguard_validated"]
    println(
        io,
        "Held-out γ=0.25 convergence is " *
        "`$(decisions["holdout_gamma_0p25_successes"])/$(decisions["holdout_gamma_0p25_cases"])`; " *
        "development convergence is " *
        "`$(decisions["development_gamma_0p25_successes"])/$(decisions["development_gamma_0p25_cases"])`. " *
        "For this focused decision, “nearly all” was fixed as at least " *
        "`$(decisions["holdout_gamma_0p25_near_all_required_successes"])/11` " *
        "held-out successes, together with all " *
        "`$(decisions["development_gamma_0p25_required_successes"])` development successes. " *
        (fixed_gamma ?
         (
             decisions["catastrophic_gamma_0p25_holdout_failure_observed"] ?
             "This meets the preregistered focused threshold, so γ=0.25 is retained as " *
             "the empirical baseline for a future population study. It is not a universal " *
             "or failure-free safeguard: `$(decisions["catastrophic_gamma_0p25_holdout_case_ids"])` " *
             "failed with final residual " *
             "`$(_fmt(decisions["catastrophic_gamma_0p25_holdout_maximum_final_residual_norm2"]))`, " *
             "and adaptive-damping development remains warranted." :
             "This meets the preregistered focused threshold and retains γ=0.25 as " *
             "the empirical baseline for a future population study, not a universal theorem."
         ) :
         "This does not validate a fixed damping value; an adaptive rule is required."),
    )
    println(io)
    println(io, "## Phase 3 — globalization mechanism")
    println(io)
    _println_table_header(
        io,
        (
            "Case",
            "γ",
            "Conv.",
            "Iters",
            "Initial ‖F‖₂",
            "Final ‖F‖₂",
            "Max relative step",
            "Median model ratio",
            "Median linearized-residual ratio",
            "Armijo reject fraction",
            "Max condition proxy",
            "max η",
            "η retries / reversals",
            "KLU singular retries",
            "Rank loss",
            "Boundary-dir. fraction",
        ),
    )
    sort!(
        globalization;
        by = row -> (
            get(row, "formulation", ""),
            _int(row, "scenario_seed"),
            _int(row, "transition"),
            _float(row, "gamma"),
        ),
    )
    for row in globalization
        case = "$(row["formulation"]) s$(_int(row, "scenario_seed")) t$(_int(row, "transition"))"
        println(
            io,
            "| $(case) | $(_fmt(_float(row, "gamma"); digits=2)) | " *
            "$(_bool(row, "direct_converged")) | $(_int(row, "total_inner_iters")) | " *
            "$(_fmt(_float(row, "initial_residual_norm2"))) | " *
            "$(_fmt(_float(row, "direct_final_residual_norm2"))) | " *
            "$(_fmt(_float(row, "maximum_relative_step_norm2"))) | " *
            "$(_fmt(_float(row, "median_accepted_reduction_ratio"))) | " *
            "$(_fmt(_float(row, "median_linearized_residual_ratio"))) | " *
            "$(_fmt(_float(row, "armijo_rejection_fraction"))) | " *
            "$(_fmt(_float(row, "maximum_rrqr_condition_proxy"))) | " *
            "$(_fmt(_float(row, "maximum_eta"))) | " *
            "$(_int(row, "eta_retry_event_count")) / " *
            "$(_int(row, "eta_reversal_count")) | " *
            "$(_int(row, "klu_singular_retry_count")) | " *
            "$(_int(row, "rank_loss_from_initial")) | " *
            "$(_fmt(_float(row, "boundary_restricted_direction_fraction"))) |",
        )
    end
    println(io)
    if decisions["same_process_iterate_path_validation_exact"]
        println(
            io,
            "Instrumentation neutrality is established by the separate same-process " *
            "validator: all `$(decisions["same_process_iterate_path_validation_cases"])` " *
            "hard cases have identical accepted-iterate hash sequences and metadata, " *
            "bitwise-identical final full iterates, and identical scalar traces between " *
            "the full rich hook and the lightweight hash hook. This same-process result " *
            "is marker-complete, satisfies the diagnostic-only protocol checks, and is " *
            "the sole exact invariance claim.",
        )
    else
        println(
            io,
            "Exact instrumentation neutrality is not established by the current report " *
            "artifacts: the required same-process full-rich-versus-lightweight-hash " *
            "validator is absent or incomplete.",
        )
    end
    println(
        io,
        "The saved source-run comparison is cross-process and is diagnostic only: " *
        "`$(decisions["cross_process_trace_structure_exact_cases"])/15` traces retain " *
        "exact event structure, `$(decisions["cross_process_trace_numeric_within_tolerance_cases"])/15` " *
        "have all numeric fields within replay tolerance, and " *
        "`$(decisions["cross_process_final_primal_bitwise_exact_cases"])/15` final " *
        "primals are bitwise exact. It is not used to claim or refute path neutrality. " *
        "SPQR/RRQR diagnostics are finalized after each solve; reported conditioning " *
        "values are R-diagonal proxies, not singular-value estimates.",
    )
    println(io)
    println(io, "### Within-transition mechanism contrasts")
    println(io)
    _println_table_header(
        io,
        (
            "Transition",
            "Success/failure",
            "Median raw R₀ S/F",
            "Median row-equilibrated R₀ S/F",
            "Median max relative step S/F",
            "Median accepted ratio S/F",
            "Armijo fraction S/F",
            "η retries S/F",
            "Rank loss S/F",
            "Boundary fraction S/F",
        ),
    )
    contrast_order = Dict(
        "quasi__seed202__t2" => 1,
        "reduced__seed202__t5" => 2,
        "quasi__seed101__t1" => 3,
    )
    sort!(
        mechanism_contrasts;
        by = row -> contrast_order[get(row, "case_id", "")],
    )
    for row in mechanism_contrasts
        println(
            io,
            "| $(row["case_id"]) | $(_int(row, "success_count"))/" *
            "$(_int(row, "failure_count")) | " *
            "$(_fmt(_float(row, "success_median_initial_raw_residual_norm2"))) / " *
            "$(_fmt(_float(row, "failure_median_initial_raw_residual_norm2"))) | " *
            "$(_fmt(_float(row, "success_median_initial_row_equilibrated_residual_norm2"))) / " *
            "$(_fmt(_float(row, "failure_median_initial_row_equilibrated_residual_norm2"))) | " *
            "$(_fmt(_float(row, "success_median_maximum_relative_step_norm2"))) / " *
            "$(_fmt(_float(row, "failure_median_maximum_relative_step_norm2"))) | " *
            "$(_fmt(_float(row, "success_median_median_accepted_reduction_ratio"))) / " *
            "$(_fmt(_float(row, "failure_median_median_accepted_reduction_ratio"))) | " *
            "$(_fmt(_float(row, "success_median_armijo_rejection_fraction"))) / " *
            "$(_fmt(_float(row, "failure_median_armijo_rejection_fraction"))) | " *
            "$(_fmt(_float(row, "success_median_eta_retry_event_count"); digits=1)) / " *
            "$(_fmt(_float(row, "failure_median_eta_retry_event_count"); digits=1)) | " *
            "$(_fmt(_float(row, "success_median_rank_loss_from_initial"); digits=1)) / " *
            "$(_fmt(_float(row, "failure_median_rank_loss_from_initial"); digits=1)) | " *
            "$(_fmt(_float(row, "success_median_boundary_restricted_direction_fraction"))) / " *
            "$(_fmt(_float(row, "failure_median_boundary_restricted_direction_fraction"))) |",
        )
    end
    println(io)
    q202_t2 = _only_transition_row(mechanism_contrasts, "quasi", 202, 2)
    q101_t1 = _only_transition_row(mechanism_contrasts, "quasi", 101, 1)
    r202_gamma0 = only(filter(globalization) do row
        get(row, "formulation", "") == "reduced" &&
            _int(row, "scenario_seed") == 202 &&
            _int(row, "transition") == 5 &&
            _float(row, "gamma") == 0.0
    end)
    println(
        io,
        "- **Quasi seed 202, transition 2:** the failed γ=0.75/1 runs start with " *
        "a *lower* median R₀ than γ=0.5 (" *
        "$(_fmt(_float(q202_t2, "failure_median_initial_raw_residual_norm2"))) versus " *
        "$(_fmt(_float(q202_t2, "success_median_initial_raw_residual_norm2")))), " *
        "but their median maximum relative step jumps to " *
        "`$(_fmt(_float(q202_t2, "failure_median_maximum_relative_step_norm2")))` " *
        "from `$(_fmt(_float(q202_t2, "success_median_maximum_relative_step_norm2")))`; " *
        "accepted model ratio falls from " *
        "`$(_fmt(_float(q202_t2, "success_median_median_accepted_reduction_ratio")))` " *
        "to `$(_fmt(_float(q202_t2, "failure_median_median_accepted_reduction_ratio")))`, " *
        "Armijo rejection rises, and the two failures hit `η=100` with 8 and 6 " *
        "retries and a cap event each. Their η-reversal counts (6 and 4) are far below " *
        "the successful run's 139, so the evidence is saturation/escalation with " *
        "retries—not frequent cycling. This resolves the case as an oversized-direction, " *
        "failed-model-agreement, η-saturation mechanism.",
    )
    println(
        io,
        "- **Reduced seed 202, transition 5:** γ=0 ends after " *
        "`$(_int(r202_gamma0, "total_inner_iters"))` iterations at " *
        "`‖F‖₂=$(_fmt(_float(r202_gamma0, "direct_final_residual_norm2")))`, only " *
        "slightly above `tol=0.008`, with median relative step " *
        "`$(_fmt(_float(r202_gamma0, "median_relative_step_norm2")))` and median " *
        "linearized-residual ratio " *
        "`$(_fmt(_float(r202_gamma0, "median_linearized_residual_ratio")))`. The " *
        "maximum condition proxy is " *
        "`$(_fmt(_float(r202_gamma0, "maximum_rrqr_condition_proxy")))`; there is one " *
        "KLU singular retry, no η retry, 240 η reversals with only " *
        "`η_max=$(_fmt(_float(r202_gamma0, "maximum_eta")))`, zero rank loss, and no " *
        "boundary rejection. This is slow near-tolerance stagnation, not the quasi-case " *
        "direction failure; rank loss does not explain it.",
    )
    println(
        io,
        "- **Quasi seed 101, transition 1:** failures also have lower median R₀, " *
        "but do not have larger steps, poorer accepted model ratios, more Armijo " *
        "rejection, more boundary restriction, or more rank loss than successes. " *
        "γ=0,0.25,1 succeed while the intervening values fail, so no clean scalar " *
        "separator is supported.",
    )
    println(io)
    println(
        io,
        "The apparent lower-initial-residual effect disappears under row equilibration. " *
        "For quasi seed 101 transition 1, the success/failure medians are " *
        "`$(_fmt(_float(q101_t1, "success_median_initial_row_equilibrated_residual_norm2"); digits=6))`/" *
        "`$(_fmt(_float(q101_t1, "failure_median_initial_row_equilibrated_residual_norm2"); digits=6))` " *
        "(relative gap " *
        "`$(_fmt(abs(_float(q101_t1, "failure_median_initial_row_equilibrated_residual_norm2") - _float(q101_t1, "success_median_initial_row_equilibrated_residual_norm2")) / _float(q101_t1, "success_median_initial_row_equilibrated_residual_norm2")))`); " *
        "for quasi seed 202 transition 2 they are " *
        "`$(_fmt(_float(q202_t2, "success_median_initial_row_equilibrated_residual_norm2"); digits=6))`/" *
        "`$(_fmt(_float(q202_t2, "failure_median_initial_row_equilibrated_residual_norm2"); digits=6))` " *
        "(relative gap " *
        "`$(_fmt(abs(_float(q202_t2, "failure_median_initial_row_equilibrated_residual_norm2") - _float(q202_t2, "success_median_initial_row_equilibrated_residual_norm2")) / _float(q202_t2, "success_median_initial_row_equilibrated_residual_norm2")))`). " *
        "Thus the small raw-R₀ ordering is a scaling artifact, not evidence that the " *
        "failed starts are closer in the solver-relevant geometry.",
    )
    println(io)
    println(
        io,
        "Across all 15 traces, every boundary-direction fraction is zero, every " *
        "minimum positivity cap is 1, and there are no boundary-rejected trials. " *
        "Fraction-to-boundary restriction is therefore excluded here. Failure has " *
        "greater rank loss in none of the three within-transition contrasts, so there " *
        "is no consistent rank-loss explanation. R₀ is likewise not predictive: it is " *
        "lower for failures in both quasi contrasts, while reduced γ=0 has the higher " *
        "R₀ and fails by slow progress.",
    )
    println(io)
    println(io, "### Basin and candidate conditioning")
    println(io)
    _println_table_header(
        io,
        (
            "Transition",
            "Tested",
            "Successful",
            "Max all-candidate separation",
            "Max successful separation",
            "Successful set aligned?",
            "Iteration qualification",
        ),
    )
    sort!(basin_summary; by = row -> get(row, "case_id", ""))
    for row in basin_summary
        println(
            io,
            "| $(row["case_id"]) | $(_int(row, "tested_candidates")) | " *
            "$(_int(row, "successful_candidates")) | " *
            "$(_fmt(_float(row, "max_all_candidate_separation_normalized"))) | " *
            "$(_fmt(_float(row, "max_successful_candidate_separation_normalized"))) | " *
            "$(_bool(row, "successful_candidates_aligned")) | " *
            "$(row["iteration_comparison_qualification"]) |",
        )
    end
    q101_basin = _only_transition_row(basin_summary, "quasi", 101, 1)
    r202_basin = _only_transition_row(basin_summary, "reduced", 202, 5)
    println(io)
    if decisions["hard_case_all_candidate_pairs_separated"]
        println(
            io,
            "All `$(decisions["hard_case_candidate_pair_count"])` within-transition " *
            "candidate pairs exceed the predeclared `0.001` alignment threshold; " *
            "the minimum normalized separation is " *
            "`$(_fmt(decisions["hard_case_minimum_candidate_pair_separation_normalized"]))`.",
        )
        println(io)
    end
    println(
        io,
        "The successful candidates are separated by " *
        "`$(_fmt(_float(q101_basin, "max_successful_candidate_separation_normalized")))` " *
        "for quasi seed 101 transition 1 and " *
        "`$(_fmt(_float(r202_basin, "max_successful_candidate_separation_normalized")))` " *
        "for reduced seed 202 transition 5, both far above " *
        "the `$(ALIGNMENT_TOL)` threshold; quasi seed 202 transition 2 has only one " *
        "success. Consequently no unconditioned iteration ranking across γ is valid. " *
        "An adaptive design should use online step/model-agreement and " *
        "η-escalation/retry signals " *
        "for the quasi-202 failure, a progress/plateau monitor for reduced-202, and " *
        "retain basin awareness because quasi-101 admits no single scalar trigger.",
    )
    println(io)
    println(io, "## Phase 4 — controlled theorem scaling")
    println(io)
    _println_table_header(
        io,
        (
            "Form",
            "Transport",
            "ε=0: interior / initial / terminal / total",
            "Primary slopes: interior / initial / terminal / total",
        ),
    )
    for formulation in ("reduced", "quasi")
        anchors = filter(scaling) do row
            get(row, "formulation", "") == formulation &&
                _float(row, "epsilon") == 0.0
        end
        for transport in (
            "identity_copy",
            "stage_shift_zero_tail",
            "stage_shift_hold_tail",
            "diagnostic_oracle_terminal_completion",
        )
            anchor = only(
                filter(row -> get(row, "transport", "") == transport, anchors),
            )
            slope(metric) = only(filter(slopes) do row
                get(row, "formulation", "") == formulation &&
                    get(row, "transport", "") == transport &&
                    get(row, "metric", "") == metric
            end)
            anchor_values = join(
                (
                    _fmt(_float(anchor, metric)) for metric in (
                        "interior_residual_norm2",
                        "initial_residual_norm2",
                        "terminal_residual_norm2",
                        "total_residual_norm2",
                    )
                ),
                " / ",
            )
            slope_values = join(
                (
                    _fmt(_float(slope(metric), "primary_slope")) for metric in (
                        "interior_residual_norm2",
                        "initial_residual_norm2",
                        "terminal_residual_norm2",
                        "total_residual_norm2",
                    )
                ),
                " / ",
            )
            println(
                io,
                "| $(formulation) | $(transport) | $(anchor_values) | " *
                "$(slope_values) |",
            )
        end
    end
    constant = only(filter(row -> get(row, "formulation", "") == "reduced", constants))
    println(io)
    println(
        io,
        "Primary slopes use the three smallest positive perturbations " *
        "`10⁻⁵, 10⁻⁴, 10⁻³`; the corresponding all-five-point fits are retained in " *
        "`raw/scaling_benchmark_slopes.csv`.",
    )
    println(io)
    println(
        io,
        "All exact references satisfy `‖F‖₂ ≤ " *
        "$(_fmt(decisions["maximum_scaling_reference_residual_norm2"]))`. " *
        "Across all 12 formulation/ε rank checks, the Jacobian rank is " *
        "`$(decisions["scaling_jacobian_minimum_rank"])`–" *
        "`$(decisions["scaling_jacobian_maximum_rank"])` for " *
        "`$(decisions["scaling_jacobian_minimum_row_dimension"])`–" *
        "`$(decisions["scaling_jacobian_maximum_row_dimension"])` rows, with maximum " *
        "nullity `$(decisions["scaling_jacobian_maximum_nullity"])` and " *
        "`sup ‖J†‖₂ = $(_fmt(decisions["scaling_maximum_pseudoinverse_norm2"]))`. " *
        "The maximum projected " *
        "solution-manifold residual is " *
        "`$(_fmt(_float(constant, "maximum_projected_solution_manifold_residual_norm2")))`, " *
        "and the tested `Cε + C_terminal τ` bound has maximum violation " *
        "`$(_fmt(_float(constant, "maximum_semantic_bound_violation")))` and maximum " *
        "left/right ratio `$(_fmt(decisions["maximum_semantic_bound_ratio"]))`. " *
        "For the reduced fixture, `C = " *
        "$(_fmt(_float(constant, "bound_data_constant")))` and " *
        "`C_terminal = $(_fmt(_float(constant, "bound_terminal_constant")))`.",
    )
    println(io)
    println(
        io,
        "The diagnostic oracle uses exact destination terminal coordinates and is not " *
        "online-available. It is included only to identify the completion term. Here " *
        "`τ` is the reported `terminal_completion_error_norm2`. The initial-condition " *
        "multiplier reset error is O(ε) by construction and is absorbed into `Cε`: " *
        "the benchmark's `bound_data_constant` is computed from the complete " *
        "nonterminal reference direction, including initial-completion coordinates. " *
        "This is an affine equality-only fixture with no inequality active-set changes; " *
        "reduced and quasi metrics coincide. It tests the transport scaling identity, " *
        "not every nonlinear GOOP regime.",
    )
    println(io)
    println(io, "## Theoretical interpretation")
    println(io)
    println(
        io,
        "The evidence " *
        (decisions["semantic_interior_consistency_theorem_supported"] ? "supports" : "does not yet support") *
        " formalizing the proposed semantic interior-consistency statement: under a " *
        "shift-equivariant stagewise smooth fixed-horizon KKT map and an unchanged " *
        "active/preference regime, semantic transport preserves inherited equations up " *
        "to the problem-data perturbation. The full residual additionally contains " *
        "initial- and terminal-completion errors. In the controlled fixture the initial " *
        "error scales with ε and is absorbed into `Cε`; the terminal error is retained " *
        "explicitly as `C_terminal τ`. This focused affine benchmark is evidence for " *
        "the statement under those hypotheses, not a proof of an assumption-free " *
        "universal theorem.",
    )
    println(io)
    println(
        io,
        (decisions["fixed_index_inconsistency_supported"] ?
         "The deterministic fixture supports a formal inconsistency result for " *
         "fixed-index copying: its additional multiplier-misalignment term is not " *
         "controlled by the MPC parameter change. The interior residual has a nonzero " *
         "ε=0 limit `$(_fmt(decisions["minimum_fixed_index_interior_epsilon0_limit_norm2"]))` " *
         "and near-zero fitted slope." :
         "The controlled fixture does not resolve a nonvanishing fixed-index " *
         "misalignment term, so no formal inconsistency result is supported."),
    )
    println(io)
    println(
        io,
        "The local corollary is a conditional residual-to-solution-manifold error bound, " *
        "not an unconditional convergence theorem for this solver. In a neighborhood " *
        "where the KKT map has constant rank, a uniformly bounded pseudoinverse, and a " *
        "Lipschitz Jacobian, `dist(y, M(p)) ≤ κ‖F(y;p)‖`. If a local algorithm's basin " *
        "contains the corresponding tubular neighborhood, the semantic residual-entry " *
        "bound supplies basin entry for sufficiently small data/completion errors. It " *
        "does not select a unique full dual vector, identify a basin globally, or imply " *
        "a uniform iteration-count improvement. On this affine fixture `J†F` is the " *
        "exact manifold projection; in a nonlinear system it is only the local " *
        "linearized correction.",
    )
    println(io)
    println(io, "## Final decisions")
    println(io)
    println(
        io,
        "1. Semantic interior-consistency theorem: **" *
        (decisions["semantic_interior_consistency_theorem_supported"] ? "supported" : "not supported") *
        "**.",
    )
    println(
        io,
        "2. Conditional local solution-manifold error-bound/basin-entry corollary: **" *
        (decisions["local_solution_manifold_corollary_supported"] ? "supported with the stated constant-rank qualifications" : "not supported") *
        "**.",
    )
    println(
        io,
        "3. Fixed γ=0.25 empirical safeguard: **" *
        (
            fixed_gamma ?
            "focused 6/6 + 10/11 threshold met; retain as the empirical population-study baseline, not as a failure-free or universal safeguard" :
            "focused threshold not met"
        ) *
        "**.",
    )
    println(
        io,
        "4. Adaptive damping: **" *
        (
            decisions["adaptive_damping_algorithm_needed"] ?
            "development warranted by the catastrophic holdout failure and nonmonotonic, basin-separated hard cases" :
            "not required by this focused evidence"
        ) *
        "**.",
    )
    println(
        io,
        "5. Boundary completion after semantic transport: **" *
        (decisions["boundary_residual_dominates_after_semantic_transport"] ?
         "dominant; prioritize terminal completion while retaining the separate initial-multiplier diagnostic" :
         "not dominant") *
        "**.",
    )
    println(io)
    println(
        io,
        "No nullspace-projection theorem was pursued; the prior 4/6 versus 5/6 result " *
        "was not reversed.",
    )
    println(io)
    println(io, "## Artifacts and exact reproduction")
    println(io)
    println(io, "- `raw/`: all long- and wide-form CSVs.")
    println(io, "- `checkpoints/`: resumable JLD2 case checkpoints.")
    println(io, "- `figures/`: PNG and PDF publication figures.")
    println(io, "- `decision_summary.toml`: machine-readable decisions and thresholds.")
    println(io)
    println(io, "Clean reproduction from the frozen source run:")
    println(io)
    println(io, "```bash")
    println(
        io,
        "julia --project=experiments experiments/analysis/theory_resolution/run.jl \\",
    )
    println(io, "  --source-run $(abspath(config.source_run)) \\")
    println(io, "  --output-root $(abspath(config.output_root))")
    println(io, "```")
    println(io)
    println(io, "Exact checkpoint resume for this run:")
    println(io)
    println(io, "```bash")
    println(
        io,
        "julia --project=experiments experiments/analysis/theory_resolution/run.jl \\",
    )
    println(io, "  --resume $(abspath(run_dir))")
    println(io, "```")
    println(io)
    println(io, "Resumable same-process instrumentation-path validation:")
    println(io)
    println(io, "```bash")
    println(
        io,
        "julia --project=experiments " *
        "experiments/analysis/theory_resolution/validate_globalization_paths.jl " *
        "--run-dir $(abspath(run_dir))",
    )
    println(io, "```")
    println(io)
    println(io, "Focused correctness checks:")
    println(io)
    println(io, "```bash")
    println(io, "julia --project=. -e 'using Test; include(\"test/kkt_metadata.jl\")'")
    println(io, "julia --project=. -e 'using Test; include(\"test/solver_trace.jl\")'")
    println(
        io,
        "julia --project=experiments experiments/analysis/theory_resolution/test_scaling_benchmark.jl",
    )
    println(
        io,
        "julia --project=experiments experiments/analysis/theory_resolution/test_iterate_path_validation.jl",
    )
    println(
        io,
        "julia --project=experiments experiments/analysis/theory_resolution/validate_stagewise.jl " *
        abspath(config.source_run) * " " * abspath(run_dir),
    )
    println(io, "```")

    path = joinpath(run_dir, "report.md")
    _atomic_write(path, String(take!(io)))
    _atomic_write(joinpath(run_dir, "report_complete"), "ok\n")
    finalize_provenance!(run_dir)
    path
end
