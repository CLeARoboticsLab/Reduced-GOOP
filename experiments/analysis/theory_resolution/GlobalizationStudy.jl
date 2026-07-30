const GLOBALIZATION_RANK_RTOL = 1e-8
const GLOBALIZATION_CHECKPOINT_SCHEMA = "rich_globalization_trace_v3"
const GLOBALIZATION_DIAGNOSTIC_SCHEMA =
    "postsolve_rrqr_row_equilibrated_v2"
# Cross-process FastDifferentiation/code-generation order can perturb otherwise
# identical sparse evaluations below the solver's decision scale.  Structural
# trace fields remain exact; these bounds apply only to floating telemetry when
# comparing against the independently generated source-run checkpoints.
const GLOBALIZATION_REPLAY_RTOL = 1e-3
const GLOBALIZATION_REPLAY_ATOL = 1e-5
const GLOBALIZATION_FINAL_REPLAY_RTOL = 1e-3

function _globalization_diagnostic_code_files()
    files = String[]
    source_root = joinpath(REPOSITORY_ROOT, "src")
    for (directory, _, names) in walkdir(source_root), name in names
        endswith(name, ".jl") || continue
        push!(files, joinpath(directory, name))
    end
    for relative in (
        joinpath(
            "experiments",
            "analysis",
            "dual_transport",
            "DualTransportStudy.jl",
        ),
        joinpath(
            "experiments",
            "analysis",
            "selective_warmstart",
            "SelectiveWarmstartStudy.jl",
        ),
        joinpath(
            "experiments",
            "analysis",
            "theory_resolution",
            "GlobalizationStudy.jl",
        ),
        joinpath(
            "experiments",
            "analysis",
            "theory_resolution",
            "globalization_case.jl",
        ),
        joinpath("experiments", "robotic_arm_core.jl"),
        joinpath("experiments", "dynamics.jl"),
    )
        path = joinpath(REPOSITORY_ROOT, relative)
        isfile(path) ||
            error("Globalization diagnostic dependency is missing: $(path)")
        push!(files, path)
    end
    sort!(unique(files))
end

function _globalization_diagnostic_code_manifest()
    records = Dict{String, Any}[
        Dict(
            "path" => relpath(path, REPOSITORY_ROOT),
            "sha256" => _sha256(path),
            "bytes" => filesize(path),
        ) for path in _globalization_diagnostic_code_files()
    ]
    payload = join(
        (
            "$(record["path"])\0$(record["sha256"])\0$(record["bytes"])" for
            record in records
        ),
        "\n",
    )
    Dict{String, Any}(
        "schema" => "globalization_diagnostic_code_manifest_v1",
        "sha256" => bytes2hex(
            SHA.sha256(Vector{UInt8}(codeunits(payload))),
        ),
        "files" => records,
    )
end

function _globalization_diagnostic_parameters()
    Dict{String, Any}(
        "schema" => GLOBALIZATION_DIAGNOSTIC_SCHEMA,
        "rank_rtol" => GLOBALIZATION_RANK_RTOL,
        "rank_tolerance_rule" =>
            "rank_rtol_times_maximum_column_norm_with_unit_floor",
        "rank_factorization" => "SuiteSparse_SPQR",
        "rank_factorization_execution" =>
            "postsolve_on_callback_owned_sparse_copies",
        "rank_retention_rule" => "positive_R_diagonal_after_SPQR_tolerance",
        "row_scaling_rule" => "max_one_and_jacobian_row_l2_norm",
        "accepted_residual_scaling" => "pre_direction_jacobian_row_scaling",
        "residual_grouping" => "equation_family_and_horizon_role_v1",
        "source_replay_rtol" => GLOBALIZATION_REPLAY_RTOL,
        "source_replay_atol" => GLOBALIZATION_REPLAY_ATOL,
        "source_final_replay_rtol" => GLOBALIZATION_FINAL_REPLAY_RTOL,
    )
end

function _globalization_resume_identity(run_dir)
    solver_options_path = joinpath(
        run_dir,
        "inputs",
        "dual_transport",
        "solver_options.toml",
    )
    isfile(solver_options_path) ||
        error("Snapshotted solver options are missing: $(solver_options_path)")
    code_manifest = _globalization_diagnostic_code_manifest()
    Dict{String, Any}(
        "solver_options_hash" => _sha256(solver_options_path),
        "diagnostic_parameters" => _globalization_diagnostic_parameters(),
        "diagnostic_code_digest" => code_manifest["sha256"],
        "diagnostic_code_manifest" => code_manifest,
    )
end

function _hard_input_path(run_dir, form, seed, transition, gamma)
    source_name = basename(
        DTS._damping_path(run_dir, form, seed, transition, gamma),
    )
    joinpath(
        run_dir,
        "inputs",
        "dual_transport",
        "checkpoints",
        "hard_case_damping",
        String(form),
        "seed_$(seed)",
        source_name,
    )
end

function _globalization_checkpoint_path(
    run_dir,
    form,
    seed,
    transition,
    gamma,
)
    source_name = basename(
        DTS._damping_path(run_dir, form, seed, transition, gamma),
    )
    joinpath(
        run_dir,
        "checkpoints",
        "globalization",
        String(form),
        "seed_$(seed)",
        source_name,
    )
end

function _row_scaling_and_rank(jacobian)
    row_squared_norms = zeros(Float64, size(jacobian, 1))
    maximum_column_norm_squared = 0.0
    rows = rowvals(jacobian)
    values = nonzeros(jacobian)
    for column in axes(jacobian, 2)
        column_norm_squared = 0.0
        for pointer in nzrange(jacobian, column)
            value_squared = abs2(values[pointer])
            row_squared_norms[rows[pointer]] += value_squared
            column_norm_squared += value_squared
        end
        maximum_column_norm_squared =
            max(maximum_column_norm_squared, column_norm_squared)
    end
    row_scales = max.(1.0, sqrt.(row_squared_norms))
    maximum_column_norm = sqrt(maximum_column_norm_squared)
    tolerance = GLOBALIZATION_RANK_RTOL * max(maximum_column_norm, 1.0)
    factorization = nothing
    factorization_time_sec = @elapsed factorization = qr(
        jacobian;
        tol = tolerance,
    )
    diagonal = abs.(diag(factorization.R))
    retained = filter(>(0.0), diagonal)
    numerical_rank = length(retained)
    largest = isempty(retained) ? 0.0 : maximum(retained)
    smallest = isempty(retained) ? 0.0 : minimum(retained)
    condition_proxy =
        smallest > 0.0 ? largest / smallest : Inf
    pseudoinverse_norm_proxy =
        smallest > 0.0 ? inv(smallest) : Inf
    (;
        row_scales,
        numerical_rank,
        rank_tolerance_absolute = tolerance,
        rrqr_diagonal_max = largest,
        rrqr_diagonal_min = smallest,
        rrqr_condition_proxy = condition_proxy,
        rrqr_pseudoinverse_norm_proxy = pseudoinverse_norm_proxy,
        rrqr_time_sec = factorization_time_sec,
        jacobian_nnz = nnz(jacobian),
    )
end

function _scaled_residual_norm(residual, row_scales, indices)
    isempty(indices) && return 0.0
    accumulator = 0.0
    @inbounds for index in indices
        value = residual[index] / row_scales[index]
        accumulator += value * value
    end
    sqrt(accumulator)
end

function _residual_group_indices(kkt)
    equation_rows = DTS._equation_rows(kkt)
    metadata = kkt.metadata.equations
    groups = Pair{Tuple{Symbol, Symbol}, Vector{Int}}[
        (:equation_family, :outer_stationarity) => equation_rows.outer,
        (:equation_family, :innermost_stationarity) => equation_rows.inner,
        (:equation_family, :equalities) => equation_rows.equality,
        (:equation_family, :other) => equation_rows.other,
    ]
    for role in (
        :inherited_interior,
        :initial_boundary,
        :terminal_boundary,
        :global,
    )
        push!(
            groups,
            (:horizon_role, role) =>
                Int[record.row for record in metadata if record.horizon_role === role],
        )
    end
    groups
end

mutable struct GlobalizationDiagnosticCollector
    base::Dict{String, Any}
    groups::Vector{Pair{Tuple{Symbol, Symbol}, Vector{Int}}}
    direction_rows::Vector{Dict{String, Any}}
    accepted_snapshot_rows::Vector{Dict{String, Any}}
    family_rows::Vector{Dict{String, Any}}
    rank_inputs::Dict{Int, Any}
    finalized::Bool
end

function _snapshot_base(collector, snapshot)
    row = copy(collector.base)
    row["total_iter"] = snapshot.total_iter
    row["inner_iter"] = snapshot.inner_iter
    row["direction_attempt"] = snapshot.direction_attempt
    row["eta_retry_count"] = get(snapshot, :eta_retry_count, 0)
    row
end

function _append_residual_groups!(
    collector,
    residual,
    row_scales,
    phase,
    total_iter,
    direction_attempt,
)
    for ((group_kind, group), indices) in collector.groups
        row = copy(collector.base)
        row["phase"] = phase
        row["total_iter"] = total_iter
        row["direction_attempt"] = direction_attempt
        row["group_kind"] = group_kind
        row["group"] = group
        row["coordinate_count"] = length(indices)
        row["raw_residual_norm2"] =
            isempty(indices) ? 0.0 : norm(view(residual, indices))
        row["raw_residual_rms"] =
            isempty(indices) ? 0.0 :
            row["raw_residual_norm2"] / sqrt(length(indices))
        row["row_equilibrated_residual_norm2"] =
            _scaled_residual_norm(residual, row_scales, indices)
        push!(collector.family_rows, row)
    end
    nothing
end

function (collector::GlobalizationDiagnosticCollector)(snapshot)
    collector.finalized &&
        error("A finalized diagnostic collector cannot accept snapshots.")
    if snapshot.event === :direction_snapshot
        residual_norm2 = norm(snapshot.residual)
        row = _snapshot_base(collector, snapshot)
        row["epsilon"] = snapshot.epsilon
        row["eta"] = snapshot.eta
        row["raw_residual_norm2"] = residual_norm2
        row["raw_residual_norm_inf"] = norm(snapshot.residual, Inf)
        row["residual_rms"] =
            residual_norm2 / sqrt(length(snapshot.residual))
        row["row_dimension"] = size(snapshot.jacobian, 1)
        row["column_dimension"] = size(snapshot.jacobian, 2)
        row["step_norm2"] = snapshot.step_norm2
        row["step_norm_inf"] = snapshot.step_norm_inf
        row["relative_step_norm2"] =
            snapshot.step_norm2 / max(1.0, norm(snapshot.z))
        row["jacobian_step_norm2"] = norm(snapshot.jacobian_step)
        row["linearized_residual_norm2"] =
            snapshot.linearized_residual_norm2
        row["linearized_residual_ratio"] =
            snapshot.linearized_residual_norm2 / max(residual_norm2, eps())
        row["full_step_predicted_reduction"] =
            snapshot.full_step_predicted_reduction
        row["armijo_slope_raw"] = snapshot.armijo_slope_raw
        row["armijo_slope"] = snapshot.armijo_slope
        row["slack_stepsize_cap"] = snapshot.slack_stepsize_cap
        row["dual_stepsize_cap"] = snapshot.dual_stepsize_cap
        row["combined_stepsize_cap"] = snapshot.combined_stepsize_cap
        row["klu_singular_retry_count"] =
            snapshot.klu_singular_retry_count
        row["svd_fallback_count"] = snapshot.svd_fallback_count
        push!(collector.direction_rows, row)
        get!(
            collector.rank_inputs,
            snapshot.total_iter,
            (
                jacobian = snapshot.jacobian,
                residual = snapshot.residual,
            ),
        )
    elseif snapshot.event === :accepted_snapshot
        haskey(collector.rank_inputs, snapshot.total_iter) ||
            error("Accepted snapshot has no matching direction diagnostics.")
        residual_norm2 = norm(snapshot.residual)
        row = copy(collector.base)
        row["total_iter"] = snapshot.total_iter
        row["inner_iter"] = snapshot.inner_iter
        row["direction_attempt"] = snapshot.direction_attempt
        row["accepted_alpha"] = snapshot.accepted_alpha
        row["eta_used"] = snapshot.eta_used
        row["eta_next"] = snapshot.eta_next
        row["raw_residual_norm2"] = residual_norm2
        row["raw_residual_norm_inf"] = norm(snapshot.residual, Inf)
        row["residual_rms"] =
            residual_norm2 / sqrt(length(snapshot.residual))
        row["iterate_sha256"] =
            bytes2hex(SHA.sha256(reinterpret(UInt8, snapshot.z)))
        row["_postsolve_residual"] = snapshot.residual
        push!(collector.accepted_snapshot_rows, row)
    else
        error("Unexpected diagnostic snapshot $(snapshot.event).")
    end
    nothing
end

function _append_rank_diagnostics!(row, diagnostics, residual)
    row["row_equilibrated_residual_norm2"] =
        norm(residual ./ diagnostics.row_scales)
    row["numerical_rank_estimate"] = diagnostics.numerical_rank
    row["rank_tolerance_absolute"] =
        diagnostics.rank_tolerance_absolute
    row["rrqr_diagonal_max"] = diagnostics.rrqr_diagonal_max
    row["rrqr_diagonal_min"] = diagnostics.rrqr_diagonal_min
    row["rrqr_condition_proxy"] = diagnostics.rrqr_condition_proxy
    row["rrqr_pseudoinverse_norm_proxy"] =
        diagnostics.rrqr_pseudoinverse_norm_proxy
    row["rrqr_time_sec"] = diagnostics.rrqr_time_sec
    row["jacobian_nnz"] = diagnostics.jacobian_nnz
    row
end

"""
    _finalize_globalization_diagnostics!(collector)

Compute SPQR rank and row-equilibration diagnostics only after the optimizer
has returned. The solver callback retains callback-owned copies and performs no
SuiteSparse factorization, so diagnostic SPQR state cannot feed back into KLU
or line-search decisions.
"""
function _finalize_globalization_diagnostics!(
    collector::GlobalizationDiagnosticCollector,
)
    collector.finalized && return collector
    diagnostics_by_iteration = Dict{Int, Any}()
    for total_iter in sort!(collect(keys(collector.rank_inputs)))
        input = collector.rank_inputs[total_iter]
        diagnostics_by_iteration[total_iter] =
            _row_scaling_and_rank(input.jacobian)
    end
    for row in collector.direction_rows
        total_iter = Int(row["total_iter"])
        input = collector.rank_inputs[total_iter]
        diagnostics = diagnostics_by_iteration[total_iter]
        _append_rank_diagnostics!(row, diagnostics, input.residual)
        _append_residual_groups!(
            collector,
            input.residual,
            diagnostics.row_scales,
            :before_direction,
            total_iter,
            Int(row["direction_attempt"]),
        )
    end
    for row in collector.accepted_snapshot_rows
        total_iter = Int(row["total_iter"])
        diagnostics = diagnostics_by_iteration[total_iter]
        residual = pop!(row, "_postsolve_residual")
        row["row_equilibrated_residual_norm2"] =
            norm(residual ./ diagnostics.row_scales)
        _append_residual_groups!(
            collector,
            residual,
            diagnostics.row_scales,
            :accepted,
            total_iter,
            Int(row["direction_attempt"]),
        )
    end
    empty!(collector.rank_inputs)
    collector.finalized = true
    collector
end

function _comparable_trace(events)
    filter(event -> event.event !== :direction, events)
end

function _assert_trace_sequence_unchanged(reference_events, rich_events)
    comparable = _comparable_trace(rich_events)
    maximum_absolute_difference = 0.0
    maximum_scaled_difference = 0.0
    numeric_out_of_tolerance_fields = 0
    event_type_mismatches = 0
    missing_reference_fields = 0
    discrete_field_mismatches = 0
    for (index, (reference, observed)) in
        enumerate(zip(reference_events, comparable))
        if reference.event !== observed.event
            event_type_mismatches += 1
            continue
        end
        for name in propertynames(reference)
            if !hasproperty(observed, name)
                missing_reference_fields += 1
                continue
            end
            reference_value = getproperty(reference, name)
            observed_value = getproperty(observed, name)
            if reference_value isa AbstractFloat &&
               observed_value isa AbstractFloat &&
               isfinite(reference_value) &&
               isfinite(observed_value)
                absolute_difference =
                    abs(reference_value - observed_value)
                scaled_difference = absolute_difference / max(
                    1.0,
                    abs(reference_value),
                    abs(observed_value),
                )
                maximum_absolute_difference =
                    max(maximum_absolute_difference, absolute_difference)
                maximum_scaled_difference =
                    max(maximum_scaled_difference, scaled_difference)
                numeric_match = isapprox(
                    reference_value,
                    observed_value;
                    rtol = GLOBALIZATION_REPLAY_RTOL,
                    atol = GLOBALIZATION_REPLAY_ATOL,
                )
                numeric_out_of_tolerance_fields += !numeric_match
                continue
            end
            discrete_field_mismatches +=
                !isequal(reference_value, observed_value)
        end
    end
    event_count_exact = length(comparable) == length(reference_events)
    event_structure_exact =
        event_count_exact &&
        event_type_mismatches == 0 &&
        missing_reference_fields == 0 &&
        discrete_field_mismatches == 0
    (;
        event_structure_exact,
        event_count_exact,
        reference_event_count = length(reference_events),
        observed_event_count = length(comparable),
        event_type_mismatches,
        missing_reference_fields,
        discrete_field_mismatches,
        numeric_fields_within_tolerance =
            numeric_out_of_tolerance_fields == 0,
        numeric_out_of_tolerance_fields,
        maximum_absolute_difference,
        maximum_scaled_difference,
        rtol = GLOBALIZATION_REPLAY_RTOL,
        atol = GLOBALIZATION_REPLAY_ATOL,
    )
end

function _event_csv_row(base, event)
    row = copy(base)
    for name in propertynames(event)
        value = getproperty(event, name)
        value isa Number || value isa Symbol || value isa Bool ||
            error("Scalar trace event contains a non-scalar field $(name).")
        row[string(name)] = value
    end
    row
end

function _solve_globalization_case!(
    run_dir,
    pair,
    gamma,
    built,
    options,
    resume_identity,
)
    destination = _globalization_checkpoint_path(
        run_dir,
        pair.form,
        pair.seed,
        pair.transition,
        gamma,
    )
    source_path = _hard_input_path(
        run_dir,
        pair.form,
        pair.seed,
        pair.transition,
        gamma,
    )
    source = JLD2.load_object(source_path)
    warmstart = DTS._build_warmstart(
        pair.source["shifted_primal"],
        pair.source["z"],
        built.kkt,
        :all_duals,
        :stage_shift_zero_tail,
    )
    warmstart[built.blocks.ψ_in] .*= gamma
    warmstart == source["warmstart"] ||
        error("Hard-case warm start differs from the completed run.")
    source_reference_hash = _sha256(pair.source_path)
    destination_reference_hash = _sha256(pair.destination_path)
    if isfile(destination)
        checkpoint = JLD2.load_object(destination)
        get(checkpoint, "schema_version", "") ==
            GLOBALIZATION_CHECKPOINT_SCHEMA ||
            error("Globalization checkpoint schema mismatch: $(destination)")
        get(checkpoint, "protocol", "") == String(PROTOCOL) ||
            error("Globalization checkpoint protocol mismatch: $(destination)")
        row = checkpoint["row"]
        Symbol(string(row["formulation"])) === pair.form ||
            error("Globalization checkpoint formulation mismatch.")
        Int(row["scenario_seed"]) == pair.seed ||
            error("Globalization checkpoint seed mismatch.")
        Int(row["transition"]) == pair.transition ||
            error("Globalization checkpoint transition mismatch.")
        Float64(row["gamma"]) == gamma ||
            error("Globalization checkpoint gamma mismatch.")
        checkpoint["warmstart"] == warmstart ||
            error("Globalization checkpoint warm start drifted.")
        checkpoint["source_checkpoint_hash"] == _sha256(source_path) ||
            error("Globalization source checkpoint drifted.")
        checkpoint["source_reference_hash"] == source_reference_hash ||
            error("Globalization source reference drifted.")
        checkpoint["destination_reference_hash"] == destination_reference_hash ||
            error("Globalization destination reference drifted.")
        checkpoint["solver_options_hash"] ==
            resume_identity["solver_options_hash"] ||
            error("Globalization solver options drifted.")
        checkpoint["diagnostic_parameters"] ==
            resume_identity["diagnostic_parameters"] ||
            error("Globalization diagnostic parameters drifted.")
        checkpoint["diagnostic_code_digest"] ==
            resume_identity["diagnostic_code_digest"] ||
            error("Globalization diagnostic code drifted.")
        checkpoint["diagnostic_code_manifest"] ==
            resume_identity["diagnostic_code_manifest"] ||
            error("Globalization diagnostic code manifest drifted.")
        return checkpoint
    end

    base = Dict{String, Any}(
        "case_id" =>
            "$(pair.form)__seed$(pair.seed)__t$(pair.transition)__gamma$(gamma)",
        "formulation" => pair.form,
        "scenario_seed" => pair.seed,
        "transition" => pair.transition,
        "gamma" => gamma,
    )
    collector = GlobalizationDiagnosticCollector(
        base,
        _residual_group_indices(built.kkt),
        Dict{String, Any}[],
        Dict{String, Any}[],
        Dict{String, Any}[],
        Dict{Int, Any}(),
        false,
    )
    events = Any[]
    output = nothing
    solve_elapsed = @elapsed output = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        built.kkt,
        pair.destination["parameters"].θ;
        z₀ = copy(warmstart),
        options,
        trace_hook = event -> push!(events, event),
        diagnostic_hook = collector,
    )
    postprocess_elapsed =
        @elapsed _finalize_globalization_diagnostics!(collector)
    trace_replay =
        _assert_trace_sequence_unchanged(source["events"], events)
    final_primal = copy(output.z[built.blocks.z])
    final_primal_difference =
        norm(final_primal .- source["final_primal"])
    final_primal_scaled_difference = final_primal_difference / max(
        1.0,
        norm(final_primal),
        norm(source["final_primal"]),
    )
    direct = SWS._residual(
        built.kkt,
        output.z,
        pair.destination["parameters"].θ;
        epsilon = output.ϵ,
    )
    source_row = source["row"]
    source_final_residual =
        Float64(source_row["direct_final_residual_norm2"])
    final_residual_absolute_difference =
        abs(direct.norm2 - source_final_residual)
    final_residual_scaled_difference =
        final_residual_absolute_difference /
        max(1.0, abs(direct.norm2), abs(source_final_residual))
    converged = isfinite(direct.norm2) && direct.norm2 <= SOLVER_TOL
    source_converged = Bool(source["converged"])
    converged == source_converged ||
        error("Rich instrumentation changed convergence classification.")
    if converged && source_converged
        final_primal_scaled_difference <= GLOBALIZATION_FINAL_REPLAY_RTOL ||
            error(
                "Jointly converged replay reached a source-separated primal " *
                "candidate: $(final_primal_scaled_difference).",
            )
    end
    output.svd_fallback_count == 0 ||
        error("A hard case invoked the forbidden alternate dense fallback.")
    row = copy(base)
    row["direct_converged"] = converged
    row["solver_status"] = output.status
    row["total_inner_iters"] = output.total_iters
    row["direct_final_residual_norm2"] = direct.norm2
    row["direct_final_residual_norm_inf"] = direct.norm_inf
    row["source_initial_residual_norm2"] =
        Float64(source_row["initial_residual_norm2"])
    row["solve_time_sec"] = solve_elapsed
    row["diagnostic_postprocess_time_sec"] = postprocess_elapsed
    row["solve_and_diagnostics_time_sec"] =
        solve_elapsed + postprocess_elapsed
    row["spqr_executed_during_solve"] = false
    row["direction_attempts"] = length(collector.direction_rows)
    row["accepted_steps"] = length(collector.accepted_snapshot_rows)
    row["trace_event_structure_exact"] =
        trace_replay.event_structure_exact
    row["trace_event_count_exact"] =
        trace_replay.event_count_exact
    row["trace_reference_event_count"] =
        trace_replay.reference_event_count
    row["trace_observed_event_count"] =
        trace_replay.observed_event_count
    row["trace_event_type_mismatches"] =
        trace_replay.event_type_mismatches
    row["trace_missing_reference_fields"] =
        trace_replay.missing_reference_fields
    row["trace_discrete_field_mismatches"] =
        trace_replay.discrete_field_mismatches
    row["trace_numeric_fields_within_tolerance"] =
        trace_replay.numeric_fields_within_tolerance
    row["trace_numeric_out_of_tolerance_fields"] =
        trace_replay.numeric_out_of_tolerance_fields
    row["trace_numeric_max_absolute_difference"] =
        trace_replay.maximum_absolute_difference
    row["trace_numeric_max_scaled_difference"] =
        trace_replay.maximum_scaled_difference
    row["trace_replay_rtol"] = trace_replay.rtol
    row["trace_replay_atol"] = trace_replay.atol
    row["final_primal_exact"] =
        final_primal_difference == 0.0
    row["final_primal_difference_norm2"] =
        final_primal_difference
    row["final_primal_scaled_difference"] =
        final_primal_scaled_difference
    row["final_primal_within_source_alignment_tolerance"] =
        final_primal_scaled_difference <= GLOBALIZATION_FINAL_REPLAY_RTOL
    row["final_replay_rtol"] =
        GLOBALIZATION_FINAL_REPLAY_RTOL
    row["final_residual_absolute_difference"] =
        final_residual_absolute_difference
    row["final_residual_scaled_difference"] =
        final_residual_scaled_difference
    row["source_checkpoint"] = abspath(source_path)
    row["source_sha256"] = _sha256(source_path)
    checkpoint = Dict{String, Any}(
        "schema_version" => GLOBALIZATION_CHECKPOINT_SCHEMA,
        "protocol" => String(PROTOCOL),
        "row" => row,
        "events" => events,
        "direction_rows" => collector.direction_rows,
        "accepted_snapshot_rows" => collector.accepted_snapshot_rows,
        "family_rows" => collector.family_rows,
        "warmstart" => warmstart,
        "final_z" => copy(output.z),
        "final_primal" => final_primal,
        "source_checkpoint_hash" => _sha256(source_path),
        "source_reference_hash" => source_reference_hash,
        "destination_reference_hash" => destination_reference_hash,
        "solver_options_hash" => resume_identity["solver_options_hash"],
        "diagnostic_parameters" => resume_identity["diagnostic_parameters"],
        "diagnostic_code_digest" => resume_identity["diagnostic_code_digest"],
        "diagnostic_code_manifest" =>
            resume_identity["diagnostic_code_manifest"],
    )
    _atomic_save(destination, checkpoint)
    checkpoint
end

function _safe_median(values)
    values = Float64[value for value in values if isfinite(value)]
    _median_or_nan(values)
end

function _safe_minimum(values)
    values = Float64[value for value in values if isfinite(value)]
    isempty(values) ? NaN : minimum(values)
end

function _safe_maximum(values)
    values = Float64[value for value in values if isfinite(value)]
    isempty(values) ? NaN : maximum(values)
end

function _eta_reversal_count(events)
    signs = Int[]
    for event in events
        event.changed || continue
        change = event.eta_after - event.eta_before
        iszero(change) || push!(signs, signbit(change) ? -1 : 1)
    end
    count(index -> signs[index] != signs[index-1], 2:length(signs))
end

function _globalization_summary(checkpoint, options)
    row = copy(checkpoint["row"])
    directions = checkpoint["direction_rows"]
    isempty(directions) &&
        error("A globalization checkpoint has no direction diagnostics.")
    first_direction = first(directions)
    accepted = filter(
        event -> event.event === :accepted_step,
        checkpoint["events"],
    )
    trials = filter(
        event -> event.event === :line_search_trial,
        checkpoint["events"],
    )
    eta_events = filter(
        event -> event.event === :eta_change,
        checkpoint["events"],
    )
    armijo_rejected_trials =
        filter(event -> event.armijo_rejected, trials)
    rejected_trials = filter(event -> !event.accepted, trials)

    row["initial_raw_residual_norm2"] =
        first_direction["raw_residual_norm2"]
    row["initial_residual_norm2"] =
        row["initial_raw_residual_norm2"]
    row["initial_row_equilibrated_residual_norm2"] =
        first_direction["row_equilibrated_residual_norm2"]
    row["initial_residual_rms"] =
        first_direction["residual_rms"]
    row["source_initial_residual_absolute_difference"] = abs(
        row["source_initial_residual_norm2"] -
        row["initial_raw_residual_norm2"],
    )
    row["initial_rank_estimate"] =
        first_direction["numerical_rank_estimate"]
    row["minimum_rank_estimate"] =
        minimum(direction["numerical_rank_estimate"] for direction in directions)
    row["rank_loss_from_initial"] =
        row["initial_rank_estimate"] - row["minimum_rank_estimate"]
    row["maximum_rank_deficit"] =
        maximum(
            direction["row_dimension"] - direction["numerical_rank_estimate"] for
            direction in directions
        )
    row["initial_rrqr_condition_proxy"] =
        first_direction["rrqr_condition_proxy"]
    row["maximum_rrqr_condition_proxy"] = maximum(
        direction["rrqr_condition_proxy"] for direction in directions
    )
    row["initial_rrqr_pseudoinverse_norm_proxy"] =
        first_direction["rrqr_pseudoinverse_norm_proxy"]
    row["maximum_rrqr_pseudoinverse_norm_proxy"] = maximum(
        direction["rrqr_pseudoinverse_norm_proxy"] for direction in directions
    )
    row["median_linearized_residual_ratio"] = _safe_median(
        direction["linearized_residual_ratio"] for direction in directions
    )
    row["maximum_relative_step_norm2"] = maximum(
        direction["relative_step_norm2"] for direction in directions
    )
    row["median_relative_step_norm2"] = _safe_median(
        direction["relative_step_norm2"] for direction in directions
    )
    row["median_step_norm2"] = _safe_median(
        direction["step_norm2"] for direction in directions
    )
    row["maximum_step_norm2"] = maximum(
        direction["step_norm2"] for direction in directions
    )
    row["minimum_combined_stepsize_cap"] = minimum(
        direction["combined_stepsize_cap"] for direction in directions
    )
    row["boundary_restricted_direction_fraction"] =
        count(direction -> direction["combined_stepsize_cap"] < 1.0, directions) /
        length(directions)
    row["boundary_rejected_trial_count"] = count(
        event -> event.slack_boundary_violated || event.dual_boundary_violated,
        trials,
    )
    row["armijo_rejected_trial_count"] =
        length(armijo_rejected_trials)
    row["trial_count"] = length(trials)
    row["rejected_trial_count"] = length(rejected_trials)
    row["armijo_rejection_fraction"] =
        row["armijo_rejected_trial_count"] / max(length(trials), 1)
    row["median_accepted_alpha"] =
        _safe_median(event.accepted_alpha for event in accepted)
    row["minimum_accepted_alpha"] =
        _safe_minimum(event.accepted_alpha for event in accepted)
    row["median_accepted_reduction_ratio"] =
        _safe_median(event.reduction_ratio for event in accepted)
    row["minimum_accepted_reduction_ratio"] =
        _safe_minimum(event.reduction_ratio for event in accepted)
    row["median_accepted_predicted_reduction"] =
        _safe_median(event.predicted_reduction for event in accepted)
    row["median_accepted_actual_reduction"] =
        _safe_median(event.actual_reduction for event in accepted)
    row["median_armijo_rejected_reduction_ratio"] =
        _safe_median(
            event.reduction_ratio for event in armijo_rejected_trials
        )
    row["nonpositive_reduction_trial_fraction"] =
        count(
            event ->
                isfinite(event.reduction_ratio) &&
                event.reduction_ratio <= 0.0,
            trials,
        ) / max(length(trials), 1)
    row["poor_model_agreement_fraction"] =
        count(event -> event.reduction_ratio <= 0.75, accepted) /
        max(length(accepted), 1)
    row["eta_change_count"] = count(event -> event.changed, eta_events)
    row["eta_retry_event_count"] =
        count(event -> event.reason === :line_search_retry, eta_events)
    row["eta_increase_event_count"] = count(
        event -> event.changed && event.eta_after > event.eta_before,
        eta_events,
    )
    row["eta_decrease_event_count"] = count(
        event -> event.changed && event.eta_after < event.eta_before,
        eta_events,
    )
    row["eta_unchanged_event_count"] =
        count(event -> !event.changed, eta_events)
    row["eta_reversal_count"] = _eta_reversal_count(eta_events)
    row["eta_cap_event_count"] = count(
        event ->
            event.eta_after == options.η_max &&
                event.eta_before < options.η_max,
        eta_events,
    )
    row["eta_max_setting"] = options.η_max
    row["maximum_eta"] = maximum(
        vcat(
            Float64[get(event, :eta_after, -Inf) for event in eta_events],
            Float64[get(event, :eta_used, -Inf) for event in accepted],
        ),
    )
    row["klu_singular_retry_count"] = sum(
        Int(get(event, :klu_singular_retry_count, 0)) for event in eta_events
    )
    row["regularization_reversal_observed"] =
        row["eta_reversal_count"] > 0
    row
end

function _globalization_basin_diagnostics!(
    checkpoints,
    alignment_tolerance,
)
    grouped = Dict{Tuple{Symbol, Int, Int}, Vector{Any}}()
    for checkpoint in checkpoints
        row = checkpoint["row"]
        key = (
            Symbol(string(row["formulation"])),
            Int(row["scenario_seed"]),
            Int(row["transition"]),
        )
        push!(get!(grouped, key, Any[]), checkpoint)
    end

    distance_rows = Dict{String, Any}[]
    basin_rows = Dict{String, Any}[]
    for (key, group) in sort!(collect(grouped); by = first)
        sort!(group; by = checkpoint -> Float64(checkpoint["row"]["gamma"]))
        all_distances = Float64[]
        successful_distances = Float64[]
        for first_index in eachindex(group)
            for second_index in (first_index+1):length(group)
                first_checkpoint = group[first_index]
                second_checkpoint = group[second_index]
                first_row = first_checkpoint["row"]
                second_row = second_checkpoint["row"]
                first_primal = first_checkpoint["final_primal"]
                second_primal = second_checkpoint["final_primal"]
                distance = norm(first_primal .- second_primal)
                normalized_distance = distance / max(
                    1.0,
                    norm(first_primal),
                    norm(second_primal),
                )
                both_converged =
                    Bool(first_row["direct_converged"]) &&
                    Bool(second_row["direct_converged"])
                push!(all_distances, normalized_distance)
                both_converged &&
                    push!(successful_distances, normalized_distance)
                push!(
                    distance_rows,
                    Dict{String, Any}(
                        "case_id" =>
                            "$(key[1])__seed$(key[2])__t$(key[3])__gamma$(first_row["gamma"])__gamma$(second_row["gamma"])",
                        "formulation" => key[1],
                        "scenario_seed" => key[2],
                        "transition" => key[3],
                        "gamma_a" => first_row["gamma"],
                        "gamma_b" => second_row["gamma"],
                        "converged_a" => first_row["direct_converged"],
                        "converged_b" => second_row["direct_converged"],
                        "both_converged" => both_converged,
                        "distance" => distance,
                        "normalized_distance" => normalized_distance,
                        "aligned_at_threshold" =>
                            normalized_distance <= alignment_tolerance,
                        "alignment_threshold" => alignment_tolerance,
                    ),
                )
            end
        end
        successful_count =
            count(checkpoint -> checkpoint["row"]["direct_converged"], group)
        maximum_successful_separation =
            isempty(successful_distances) ? NaN :
            maximum(successful_distances)
        successful_candidates_aligned =
            successful_count >= 2 &&
            maximum_successful_separation <= alignment_tolerance
        basin = Dict{String, Any}(
            "case_id" => "$(key[1])__seed$(key[2])__t$(key[3])",
            "formulation" => key[1],
            "scenario_seed" => key[2],
            "transition" => key[3],
            "tested_candidates" => length(group),
            "successful_candidates" => successful_count,
            "all_candidates_jointly_converged" =>
                successful_count == length(group),
            "max_all_candidate_separation_normalized" =>
                isempty(all_distances) ? NaN : maximum(all_distances),
            "max_successful_candidate_separation_normalized" =>
                maximum_successful_separation,
            "successful_candidates_aligned" =>
                successful_candidates_aligned,
            "alignment_threshold" => alignment_tolerance,
            "iteration_comparison_qualification" =>
                successful_candidates_aligned ?
                :aligned_successful_candidates :
                :separated_candidates_no_unconditioned_comparison,
        )
        push!(basin_rows, basin)
        for checkpoint in group
            row = checkpoint["row"]
            for name in (
                "tested_candidates",
                "successful_candidates",
                "all_candidates_jointly_converged",
                "max_all_candidate_separation_normalized",
                "max_successful_candidate_separation_normalized",
                "successful_candidates_aligned",
                "alignment_threshold",
                "iteration_comparison_qualification",
            )
                row[name] = basin[name]
            end
        end
    end
    (; distance_rows, basin_rows)
end

const GLOBALIZATION_CONTRAST_METRICS = (
    "initial_raw_residual_norm2",
    "initial_row_equilibrated_residual_norm2",
    "minimum_rank_estimate",
    "rank_loss_from_initial",
    "maximum_rrqr_condition_proxy",
    "maximum_rrqr_pseudoinverse_norm_proxy",
    "median_relative_step_norm2",
    "maximum_relative_step_norm2",
    "minimum_combined_stepsize_cap",
    "boundary_restricted_direction_fraction",
    "boundary_rejected_trial_count",
    "median_accepted_reduction_ratio",
    "poor_model_agreement_fraction",
    "nonpositive_reduction_trial_fraction",
    "armijo_rejection_fraction",
    "maximum_eta",
    "eta_retry_event_count",
    "eta_reversal_count",
    "eta_cap_event_count",
    "klu_singular_retry_count",
    "total_inner_iters",
    "direct_final_residual_norm2",
)

function _globalization_mechanism_contrasts(summaries, basin_rows)
    basin_index = Dict(
        (
            Symbol(string(row["formulation"])),
            Int(row["scenario_seed"]),
            Int(row["transition"]),
        ) => row for row in basin_rows
    )
    grouped = Dict{Tuple{Symbol, Int, Int}, Vector{Dict{String, Any}}}()
    for row in summaries
        key = (
            Symbol(string(row["formulation"])),
            Int(row["scenario_seed"]),
            Int(row["transition"]),
        )
        push!(get!(grouped, key, Dict{String, Any}[]), row)
    end

    rows = Dict{String, Any}[]
    for (key, group) in sort!(collect(grouped); by = first)
        successes =
            filter(row -> Bool(row["direct_converged"]), group)
        failures =
            filter(row -> !Bool(row["direct_converged"]), group)
        basin = basin_index[key]
        row = Dict{String, Any}(
            "case_id" => "$(key[1])__seed$(key[2])__t$(key[3])",
            "formulation" => key[1],
            "scenario_seed" => key[2],
            "transition" => key[3],
            "success_count" => length(successes),
            "failure_count" => length(failures),
            "both_outcomes_present" =>
                !isempty(successes) && !isempty(failures),
            "successful_candidates_aligned" =>
                basin["successful_candidates_aligned"],
            "max_successful_candidate_separation_normalized" =>
                basin["max_successful_candidate_separation_normalized"],
            "alignment_threshold" => basin["alignment_threshold"],
        )
        for metric in GLOBALIZATION_CONTRAST_METRICS
            success_median = _safe_median(
                candidate[metric] for candidate in successes
            )
            failure_median = _safe_median(
                candidate[metric] for candidate in failures
            )
            row["success_median_$(metric)"] = success_median
            row["failure_median_$(metric)"] = failure_median
            row["failure_minus_success_$(metric)"] =
                failure_median - success_median
            row["failure_to_success_ratio_$(metric)"] =
                isfinite(success_median) &&
                isfinite(failure_median) &&
                !iszero(success_median) ?
                failure_median / success_median : NaN
        end
        row["failure_has_lower_initial_raw_residual"] =
            row[
                "failure_minus_success_initial_raw_residual_norm2"
            ] < 0.0
        row["failure_has_lower_scaled_initial_residual"] =
            row[
                "failure_minus_success_initial_row_equilibrated_residual_norm2"
            ] < 0.0
        row["failure_has_more_rank_loss"] =
            row["failure_minus_success_rank_loss_from_initial"] > 0.0
        row["failure_has_larger_relative_step"] =
            row["failure_minus_success_maximum_relative_step_norm2"] > 0.0
        row["failure_has_more_boundary_restriction"] =
            row[
                "failure_minus_success_boundary_restricted_direction_fraction"
            ] > 0.0
        row["failure_has_more_armijo_rejection"] =
            row["failure_minus_success_armijo_rejection_fraction"] > 0.0
        row["failure_has_poorer_accepted_model_ratio"] =
            row[
                "failure_minus_success_median_accepted_reduction_ratio"
            ] < 0.0
        row["failure_has_more_eta_reversals"] =
            row["failure_minus_success_eta_reversal_count"] > 0.0
        push!(rows, row)
    end
    rows
end

function _run_single_globalization_case!(
    run_dir,
    form::Symbol,
    seed::Int,
    transition::Int,
    gamma::Float64,
)
    config = load_config(joinpath(run_dir, "config.toml"))
    validate_protocol(config)
    snapshot_inputs!(run_dir, config)
    pairs = DTS.valid_pairs(run_dir)
    pair = only(filter(pairs) do candidate
        candidate.form === form &&
            candidate.seed == seed &&
            candidate.transition == transition
    end)
    source_config = _source_config(config)
    cache = Dict{Symbol, Any}()
    built = DTS._build_system(source_config, form, cache)
    options = _source_solver_options(config)
    resume_identity = _globalization_resume_identity(run_dir)
    _solve_globalization_case!(
        run_dir,
        pair,
        gamma,
        built,
        options,
        resume_identity,
    )
end

function _run_globalization_case_child!(
    run_dir,
    form,
    seed,
    transition,
    gamma,
)
    script = joinpath(@__DIR__, "globalization_case.jl")
    isfile(script) ||
        error("Globalization child entry point is missing: $(script)")
    project = joinpath(REPOSITORY_ROOT, "experiments")
    julia = Base.julia_cmd()
    command = `$(julia) --project=$(project) $(script) --run-dir $(run_dir) --form $(String(form)) --seed $(seed) --transition $(transition) --gamma $(repr(gamma))`
    println(
        "[theory-resolution] isolated globalization case: ",
        form,
        " seed ",
        seed,
        " transition ",
        transition,
        " gamma ",
        gamma,
    )
    flush(stdout)
    run(command)
    nothing
end

function run_globalization_study!(
    run_dir,
    config::TheoryResolutionConfig,
    cache = Dict{Symbol, Any}(),
)
    validate_protocol(config)
    stored = load_config(joinpath(run_dir, "config.toml"))
    _config_dict(stored) == _config_dict(config) ||
        error("Run-directory configuration differs from the active configuration.")
    snapshot_inputs!(run_dir, config)
    pairs = DTS.valid_pairs(run_dir)
    pair_index = Dict(_case_key(pair) => pair for pair in pairs)
    source_config = _source_config(config)
    options = _source_solver_options(config)
    resume_identity = _globalization_resume_identity(run_dir)
    checkpoints = Any[]
    for case in GLOBALIZATION_CASES
        key = (case.form, case.seed, case.transition)
        haskey(pair_index, key) || error("Hard-case pair $(key) is unavailable.")
        pair = pair_index[key]
        built = DTS._build_system(source_config, pair.form, cache)
        for gamma in case.gammas
            checkpoint_path = _globalization_checkpoint_path(
                run_dir,
                pair.form,
                pair.seed,
                pair.transition,
                gamma,
            )
            if !isfile(checkpoint_path)
                _run_globalization_case_child!(
                    run_dir,
                    pair.form,
                    pair.seed,
                    pair.transition,
                    gamma,
                )
            end
            push!(
                checkpoints,
                _solve_globalization_case!(
                    run_dir,
                    pair,
                    gamma,
                    built,
                    options,
                    resume_identity,
                ),
            )
        end
    end
    length(checkpoints) == 15 || error("Expected 15 rich hard-case traces.")
    basin = _globalization_basin_diagnostics!(
        checkpoints,
        config.alignment_tol,
    )
    direction_rows = reduce(
        vcat,
        (checkpoint["direction_rows"] for checkpoint in checkpoints);
        init = Dict{String, Any}[],
    )
    accepted_snapshot_rows = reduce(
        vcat,
        (checkpoint["accepted_snapshot_rows"] for checkpoint in checkpoints);
        init = Dict{String, Any}[],
    )
    family_rows = reduce(
        vcat,
        (checkpoint["family_rows"] for checkpoint in checkpoints);
        init = Dict{String, Any}[],
    )
    trial_rows = Dict{String, Any}[]
    accepted_rows = Dict{String, Any}[]
    eta_rows = Dict{String, Any}[]
    for checkpoint in checkpoints
        base = Dict(
            key => checkpoint["row"][key] for key in (
                "case_id",
                "formulation",
                "scenario_seed",
                "transition",
                "gamma",
            )
        )
        for event in checkpoint["events"]
            destination = event.event === :line_search_trial ? trial_rows :
                          event.event === :accepted_step ? accepted_rows :
                          event.event === :eta_change ? eta_rows : nothing
            isnothing(destination) ||
                push!(destination, _event_csv_row(base, event))
        end
    end
    summaries = [
        _globalization_summary(checkpoint, options) for
        checkpoint in checkpoints
    ]
    contrasts =
        _globalization_mechanism_contrasts(summaries, basin.basin_rows)
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_directions.csv"),
        direction_rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_trials.csv"),
        trial_rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_iterations.csv"),
        accepted_rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_accepted_snapshots.csv"),
        accepted_snapshot_rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_eta_events.csv"),
        eta_rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_family_residuals.csv"),
        family_rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_case_summary.csv"),
        summaries,
    )
    DTS.write_csv(
        joinpath(
            run_dir,
            "raw",
            "globalization_final_primal_distances.csv",
        ),
        basin.distance_rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_basin_summary.csv"),
        basin.basin_rows,
    )
    DTS.write_csv(
        joinpath(
            run_dir,
            "raw",
            "globalization_mechanism_contrasts.csv",
        ),
        contrasts,
    )
    _atomic_save(
        joinpath(run_dir, "checkpoints", "globalization_summary.jld2"),
        Dict(
            "case_summaries" => summaries,
            "basin_summaries" => basin.basin_rows,
            "mechanism_contrasts" => contrasts,
            "case_checkpoints" => [
                checkpoint["row"]["case_id"] for checkpoint in checkpoints
            ],
        ),
    )
    _atomic_write(joinpath(run_dir, "globalization_complete"), "ok\n")
    summaries
end
