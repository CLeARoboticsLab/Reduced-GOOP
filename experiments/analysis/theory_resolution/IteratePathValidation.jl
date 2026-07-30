module GlobalizationIteratePathValidation

using JLD2: JLD2
using LinearAlgebra: BLAS, norm
using SHA: SHA
using TOML: TOML
using ReducedGOOP

if !isdefined(Main, :TheoryResolutionStudy)
    Base.include(Main, joinpath(@__DIR__, "TheoryResolutionStudy.jl"))
end
const TRS = Main.TheoryResolutionStudy
const DTS = TRS.DTS

export PATH_VALIDATION_SCHEMA,
    PATH_VALIDATION_SUMMARY_SCHEMA,
    validate_iterate_path_inputs,
    run_iterate_path_validation!

const PATH_VALIDATION_SCHEMA = "globalization_iterate_path_validation_v5"
const PATH_VALIDATION_SUMMARY_SCHEMA =
    "globalization_iterate_path_validation_summary_v5"

"""
Hash the in-memory IEEE-754 representation used by the production solve.

The rich globalization checkpoint uses this same byte-level definition for each
accepted full `z` snapshot. Restricting the helper to `Float64` prevents an
implicit numeric conversion from making two different iterate representations
look equal.
"""
function _full_vector_sha256(values::AbstractVector)
    eltype(values) === Float64 ||
        error("Expected a Float64 full iterate, got $(eltype(values)).")
    contiguous = values isa Vector{Float64} ? values : collect(values)
    bytes2hex(SHA.sha256(reinterpret(UInt8, contiguous)))
end

function _full_vector_exact(expected::AbstractVector, observed::AbstractVector)
    eltype(expected) === Float64 || return false
    eltype(observed) === Float64 || return false
    length(expected) == length(observed) || return false
    expected_values =
        expected isa Vector{Float64} ? expected : collect(expected)
    observed_values =
        observed isa Vector{Float64} ? observed : collect(observed)
    reinterpret(UInt64, expected_values) ==
        reinterpret(UInt64, observed_values)
end

function _hash_sequence_sha256(hashes)
    payload = join(String.(hashes), "\n")
    bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(payload))))
end

_is_scalar_trace_value(value) =
    value isa Number || value isa Symbol || value isa Bool

function _scalar_exact(expected, observed)
    typeof(expected) === typeof(observed) || return false
    if expected isa Union{Float16, Float32, Float64}
        return bitstring(expected) == bitstring(observed)
    end
    isequal(expected, observed)
end

"""
Assert exact equality of the complete scalar trace.

This intentionally compares the rich `:direction` events too. It is stricter
than the legacy-to-rich compatibility check, which omitted those newly added
events because they did not exist in the legacy checkpoints.
"""
function _assert_scalar_trace_exact(expected_events, observed_events)
    length(expected_events) == length(observed_events) ||
        error(
            "Scalar trace event count changed: $(length(expected_events)) " *
            "expected, $(length(observed_events)) observed.",
        )
    for (event_index, (expected, observed)) in
        enumerate(zip(expected_events, observed_events))
        expected_names = propertynames(expected)
        observed_names = propertynames(observed)
        expected_names == observed_names ||
            error(
                "Scalar trace event $(event_index) field schema changed: " *
                "$(expected_names) versus $(observed_names).",
            )
        for name in expected_names
            expected_value = getproperty(expected, name)
            observed_value = getproperty(observed, name)
            _is_scalar_trace_value(expected_value) ||
                error(
                    "Rich trace event $(event_index) field $(name) is not scalar.",
                )
            _is_scalar_trace_value(observed_value) ||
                error(
                    "Replay trace event $(event_index) field $(name) is not scalar.",
                )
            _scalar_exact(expected_value, observed_value) ||
                error(
                    "Scalar trace event $(event_index) field $(name) changed: " *
                    "$(repr(expected_value)) versus $(repr(observed_value)).",
                )
        end
    end
    true
end

function _compare_scalar_trace_diagnostic(
    saved_events,
    new_events;
    rtol,
    atol,
)
    saved_event_count = length(saved_events)
    new_event_count = length(new_events)
    overlap_event_count = min(saved_event_count, new_event_count)
    event_type_match_count = 0
    event_type_mismatch_count = 0
    event_schema_exact_count = 0
    event_schema_mismatch_count = 0
    saved_only_field_count = 0
    new_only_field_count = 0
    discrete_field_comparison_count = 0
    discrete_field_mismatch_count = 0
    numeric_field_comparison_count = 0
    numeric_out_of_tolerance_count = 0
    maximum_absolute_difference = 0.0
    maximum_scaled_difference = 0.0
    for event_index in 1:overlap_event_count
        saved = saved_events[event_index]
        new = new_events[event_index]
        if hasproperty(saved, :event) &&
           hasproperty(new, :event) &&
           isequal(saved.event, new.event)
            event_type_match_count += 1
        else
            event_type_mismatch_count += 1
        end
        saved_names = propertynames(saved)
        new_names = propertynames(new)
        if saved_names == new_names
            event_schema_exact_count += 1
        else
            event_schema_mismatch_count += 1
        end
        saved_set = Set(saved_names)
        new_set = Set(new_names)
        saved_only_field_count += length(setdiff(saved_set, new_set))
        new_only_field_count += length(setdiff(new_set, saved_set))
        for name in intersect(saved_set, new_set)
            saved_value = getproperty(saved, name)
            new_value = getproperty(new, name)
            if saved_value isa AbstractFloat &&
               new_value isa AbstractFloat &&
               isfinite(saved_value) &&
               isfinite(new_value)
                numeric_field_comparison_count += 1
                absolute_difference =
                    abs(saved_value - new_value)
                scaled_difference = absolute_difference / max(
                    1.0,
                    abs(saved_value),
                    abs(new_value),
                )
                maximum_absolute_difference =
                    max(maximum_absolute_difference, absolute_difference)
                maximum_scaled_difference =
                    max(maximum_scaled_difference, scaled_difference)
                numeric_out_of_tolerance_count += !isapprox(
                    saved_value,
                    new_value;
                    rtol,
                    atol,
                )
            else
                discrete_field_comparison_count += 1
                discrete_field_mismatch_count +=
                    !isequal(saved_value, new_value)
            end
        end
    end
    (;
        saved_event_count,
        new_event_count,
        event_count_difference = new_event_count - saved_event_count,
        overlap_event_count,
        saved_unpaired_event_count =
            saved_event_count - overlap_event_count,
        new_unpaired_event_count = new_event_count - overlap_event_count,
        event_type_match_count,
        event_type_mismatch_count,
        event_schema_exact_count,
        event_schema_mismatch_count,
        saved_only_field_count,
        new_only_field_count,
        discrete_field_comparison_count,
        discrete_field_mismatch_count,
        numeric_field_comparison_count,
        numeric_out_of_tolerance_count,
        maximum_absolute_difference,
        maximum_scaled_difference,
        rtol = Float64(rtol),
        atol = Float64(atol),
    )
end

function _trace_diagnostic_fields(comparison)
    Dict{String, Any}(
        "saved_rich_trace_comparison_role" =>
            :cross_process_diagnostic_only,
        "saved_rich_trace_saved_event_count" =>
            comparison.saved_event_count,
        "saved_rich_trace_new_event_count" =>
            comparison.new_event_count,
        "saved_rich_trace_event_count_difference" =>
            comparison.event_count_difference,
        "saved_rich_trace_overlap_event_count" =>
            comparison.overlap_event_count,
        "saved_rich_trace_saved_unpaired_event_count" =>
            comparison.saved_unpaired_event_count,
        "saved_rich_trace_new_unpaired_event_count" =>
            comparison.new_unpaired_event_count,
        "saved_rich_trace_event_type_match_count" =>
            comparison.event_type_match_count,
        "saved_rich_trace_event_type_mismatch_count" =>
            comparison.event_type_mismatch_count,
        "saved_rich_trace_event_schema_exact_count" =>
            comparison.event_schema_exact_count,
        "saved_rich_trace_event_schema_mismatch_count" =>
            comparison.event_schema_mismatch_count,
        "saved_rich_trace_saved_only_field_count" =>
            comparison.saved_only_field_count,
        "saved_rich_trace_new_only_field_count" =>
            comparison.new_only_field_count,
        "saved_rich_trace_discrete_field_comparison_count" =>
            comparison.discrete_field_comparison_count,
        "saved_rich_trace_discrete_field_mismatch_count" =>
            comparison.discrete_field_mismatch_count,
        "saved_rich_trace_numeric_field_comparison_count" =>
            comparison.numeric_field_comparison_count,
        "saved_rich_trace_numeric_out_of_tolerance_count" =>
            comparison.numeric_out_of_tolerance_count,
        "saved_rich_trace_max_absolute_difference" =>
            comparison.maximum_absolute_difference,
        "saved_rich_trace_max_scaled_difference" =>
            comparison.maximum_scaled_difference,
    )
end

function _compare_final_primal_basin(expected, observed; threshold)
    eltype(expected) === Float64 ||
        error("Saved final primal is not Float64.")
    eltype(observed) === Float64 ||
        error("Replay final primal is not Float64.")
    length(expected) == length(observed) ||
        error("Saved-rich and new-rich final-primal dimensions differ.")
    difference_norm2 = norm(observed .- expected)
    scale = max(1.0, norm(expected), norm(observed))
    scaled_difference = difference_norm2 / scale
    within_basin = scaled_difference <= threshold
    (;
        within_basin,
        difference_norm2,
        scaled_difference,
        threshold = Float64(threshold),
    )
end

function _scalar_trace_sha256(events)
    io = IOBuffer()
    for event in events
        for name in propertynames(event)
            value = getproperty(event, name)
            _is_scalar_trace_value(value) ||
                error("Trace field $(name) is not scalar.")
            print(io, name, '\0', typeof(value), '\0')
            if value isa Union{Float16, Float32, Float64}
                print(io, bitstring(value))
            else
                print(io, repr(value))
            end
            write(io, UInt8(0xff))
        end
        write(io, UInt8(0xfe))
    end
    bytes2hex(SHA.sha256(take!(io)))
end

mutable struct AcceptedIterateHashCollector
    accepted_rows::Vector{Dict{String, Any}}
    direction_snapshot_count::Int
    direction_svd_fallback_count::Int
end

AcceptedIterateHashCollector() =
    AcceptedIterateHashCollector(Dict{String, Any}[], 0, 0)

function (collector::AcceptedIterateHashCollector)(snapshot)
    if snapshot.event === :direction_snapshot
        collector.direction_snapshot_count += 1
        collector.direction_svd_fallback_count +=
            Int(get(snapshot, :svd_fallback_count, 0))
    elseif snapshot.event === :accepted_snapshot
        push!(
            collector.accepted_rows,
            Dict{String, Any}(
                "accepted_step" => length(collector.accepted_rows) + 1,
                "total_iter" => Int(snapshot.total_iter),
                "inner_iter" => Int(snapshot.inner_iter),
                "direction_attempt" => Int(snapshot.direction_attempt),
                "accepted_alpha" => Float64(snapshot.accepted_alpha),
                "eta_used" => Float64(snapshot.eta_used),
                "eta_next" => Float64(snapshot.eta_next),
                "iterate_sha256" => _full_vector_sha256(snapshot.z),
            ),
        )
    else
        error("Unexpected diagnostic snapshot $(snapshot.event).")
    end
    nothing
end

function _validation_code_identity()
    paths = sort!([
        abspath(@__FILE__),
        joinpath(@__DIR__, "TheoryResolutionStudy.jl"),
        joinpath(@__DIR__, "HoldoutValidation.jl"),
    ])
    all(isfile, paths) ||
        error("An iterate-path validation code dependency is missing.")
    files = Dict{String, Any}[
        Dict(
            "path" => relpath(path, TRS.REPOSITORY_ROOT),
            "sha256" => TRS._sha256(path),
            "bytes" => filesize(path),
        ) for path in paths
    ]
    payload = join(
        (
            "$(record["path"])\0$(record["sha256"])\0$(record["bytes"])" for
            record in files
        ),
        "\n",
    )
    Dict{String, Any}(
        "schema" => "iterate_path_validation_code_manifest_v5",
        "sha256" => bytes2hex(
            SHA.sha256(Vector{UInt8}(codeunits(payload))),
        ),
        "files" => files,
    )
end

function _runtime_identity()
    project_path = Base.active_project()
    isnothing(project_path) &&
        error("Iterate-path validation requires an active Julia project.")
    expected_project = joinpath(
        TRS.REPOSITORY_ROOT,
        "experiments",
        "Project.toml",
    )
    abspath(project_path) == abspath(expected_project) ||
        error(
            "Run iterate-path validation with --project=experiments; active " *
            "project is $(project_path).",
        )
    manifest_path = joinpath(dirname(project_path), "Manifest.toml")
    isfile(project_path) ||
        error("Active Julia project is missing: $(project_path)")
    isfile(manifest_path) ||
        error("Active Julia manifest is missing: $(manifest_path)")
    Dict{String, Any}(
        "schema" => "iterate_path_validation_runtime_v1",
        "julia_version" => string(VERSION),
        "machine" => String(Sys.MACHINE),
        "kernel" => String(Sys.KERNEL),
        "cpu_name" => String(Sys.CPU_NAME),
        "julia_threads" => Base.Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "blas_config" => string(BLAS.get_config()),
        "project_path" => abspath(project_path),
        "project_sha256" => TRS._sha256(project_path),
        "manifest_path" => abspath(manifest_path),
        "manifest_sha256" => TRS._sha256(manifest_path),
    )
end

function _assert_identity_unchanged(validation_code, runtime)
    _validation_code_identity() == validation_code ||
        error(
            "Validation code changed on disk while loaded methods were running.",
        )
    _runtime_identity() == runtime ||
        error("Julia runtime or environment changed during path validation.")
    nothing
end

function _case_id(form, seed, transition, gamma)
    "$(form)__seed$(seed)__t$(transition)__gamma$(gamma)"
end

function _case_checkpoint_path(run_dir, form, seed, transition, gamma)
    rich_name = basename(
        TRS._globalization_checkpoint_path(
            run_dir,
            form,
            seed,
            transition,
            gamma,
        ),
    )
    joinpath(
        run_dir,
        "checkpoints",
        "globalization_path_validation",
        String(form),
        "seed_$(seed)",
        rich_name,
    )
end

function _hard_cases()
    [
        (
            form = case.form,
            seed = case.seed,
            transition = case.transition,
            gamma = Float64(gamma),
        ) for case in TRS.GLOBALIZATION_CASES for gamma in case.gammas
    ]
end

function _finish_event(events)
    finishes = filter(event -> event.event === :finish, events)
    length(finishes) == 1 ||
        error("Expected exactly one scalar :finish event, got $(length(finishes)).")
    only(finishes)
end

function _rich_expected_hashes(rich)
    rows = rich["accepted_snapshot_rows"]
    hashes = String[String(row["iterate_sha256"]) for row in rows]
    accepted_events =
        filter(event -> event.event === :accepted_step, rich["events"])
    length(rows) == length(accepted_events) ||
        error(
            "Rich checkpoint accepted-snapshot/event counts disagree: " *
            "$(length(rows)) versus $(length(accepted_events)).",
        )
    for (accepted_index, (row, event)) in
        enumerate(zip(rows, accepted_events))
        Int(row["total_iter"]) == Int(event.total_iter) ||
            error("Rich accepted record $(accepted_index) has a total-iteration mismatch.")
        Int(row["inner_iter"]) == Int(event.inner_iter) ||
            error("Rich accepted record $(accepted_index) has an inner-iteration mismatch.")
        Int(row["direction_attempt"]) == Int(event.direction_attempt) ||
            error("Rich accepted record $(accepted_index) has a direction-attempt mismatch.")
    end
    hashes
end

function _warmstart(pair, built, gamma)
    warmstart = DTS._build_warmstart(
        pair.source["shifted_primal"],
        pair.source["z"],
        built.kkt,
        :all_duals,
        :stage_shift_zero_tail,
    )
    warmstart[built.blocks.ψ_in] .*= gamma
    warmstart
end

function _assert_rich_checkpoint(
    run_dir,
    pair,
    gamma,
    built,
    resume_identity,
    validation_code,
    runtime,
)
    rich_path = TRS._globalization_checkpoint_path(
        run_dir,
        pair.form,
        pair.seed,
        pair.transition,
        gamma,
    )
    isfile(rich_path) ||
        error("Rich v3 globalization checkpoint is missing: $(rich_path)")
    rich = JLD2.load_object(rich_path)
    get(rich, "schema_version", "") == TRS.GLOBALIZATION_CHECKPOINT_SCHEMA ||
        error("Expected a rich v3 globalization checkpoint: $(rich_path)")
    get(rich, "protocol", "") == String(TRS.PROTOCOL) ||
        error("Rich checkpoint protocol mismatch: $(rich_path)")
    row = rich["row"]
    Symbol(string(row["formulation"])) === pair.form ||
        error("Rich checkpoint formulation mismatch: $(rich_path)")
    Int(row["scenario_seed"]) == pair.seed ||
        error("Rich checkpoint seed mismatch: $(rich_path)")
    Int(row["transition"]) == pair.transition ||
        error("Rich checkpoint transition mismatch: $(rich_path)")
    Float64(row["gamma"]) == gamma ||
        error("Rich checkpoint gamma mismatch: $(rich_path)")
    get(row, "spqr_executed_during_solve", true) === false ||
        error("Saved rich checkpoint did not defer SPQR until post-solve.")

    hard_input_path = TRS._hard_input_path(
        run_dir,
        pair.form,
        pair.seed,
        pair.transition,
        gamma,
    )
    isfile(hard_input_path) ||
        error("Frozen hard-case checkpoint is missing: $(hard_input_path)")
    hard_input = JLD2.load_object(hard_input_path)
    warmstart = _warmstart(pair, built, gamma)
    _full_vector_exact(warmstart, hard_input["warmstart"]) ||
        error("Reconstructed warm start differs bitwise from the frozen hard case.")
    _full_vector_exact(warmstart, rich["warmstart"]) ||
        error("Reconstructed warm start differs bitwise from the rich checkpoint.")

    rich["source_checkpoint_hash"] == TRS._sha256(hard_input_path) ||
        error("Rich hard-case input hash drifted.")
    rich["source_reference_hash"] == TRS._sha256(pair.source_path) ||
        error("Rich source-reference hash drifted.")
    rich["destination_reference_hash"] ==
        TRS._sha256(pair.destination_path) ||
        error("Rich destination-reference hash drifted.")
    rich["solver_options_hash"] == resume_identity["solver_options_hash"] ||
        error("Rich solver-options hash drifted.")
    rich["diagnostic_parameters"] ==
        resume_identity["diagnostic_parameters"] ||
        error("Rich diagnostic parameters drifted.")
    rich["diagnostic_code_digest"] ==
        resume_identity["diagnostic_code_digest"] ||
        error("Rich diagnostic code digest drifted.")
    rich["diagnostic_code_manifest"] ==
        resume_identity["diagnostic_code_manifest"] ||
        error("Rich diagnostic code manifest drifted.")
    haskey(rich, "final_z") ||
        error("Rich checkpoint has no full final iterate: $(rich_path)")
    _full_vector_sha256(rich["final_z"])
    saved_hashes = _rich_expected_hashes(rich)
    rich_finish = _finish_event(rich["events"])
    Int(rich_finish.svd_fallback_count) == 0 ||
        error("Rich checkpoint used the forbidden dense SVD fallback.")
    diagnostic_parameters = rich["diagnostic_parameters"]
    for name in (
        "trace_replay_rtol",
        "trace_replay_atol",
        "final_replay_rtol",
    )
        haskey(row, name) ||
            error("Rich v3 checkpoint row did not record $(name).")
    end
    trace_replay_rtol = Float64(row["trace_replay_rtol"])
    trace_replay_atol = Float64(row["trace_replay_atol"])
    final_replay_rtol = Float64(row["final_replay_rtol"])
    trace_replay_rtol >= 0.0 ||
        error("Saved-rich trace replay rtol is negative.")
    trace_replay_atol >= 0.0 ||
        error("Saved-rich trace replay atol is negative.")
    final_replay_rtol >= 0.0 ||
        error("Saved-rich final replay rtol is negative.")
    get(diagnostic_parameters, "source_replay_rtol", trace_replay_rtol) ==
        trace_replay_rtol ||
        error("Rich row and diagnostic trace rtol disagree.")
    get(diagnostic_parameters, "source_replay_atol", trace_replay_atol) ==
        trace_replay_atol ||
        error("Rich row and diagnostic trace atol disagree.")
    get(
        diagnostic_parameters,
        "source_final_replay_rtol",
        final_replay_rtol,
    ) == final_replay_rtol ||
        error("Rich row and diagnostic final replay rtol disagree.")

    identity = Dict{String, Any}(
        "case_id" => _case_id(
            pair.form,
            pair.seed,
            pair.transition,
            gamma,
        ),
        "rich_checkpoint_hash" => TRS._sha256(rich_path),
        "hard_input_hash" => TRS._sha256(hard_input_path),
        "source_reference_hash" => TRS._sha256(pair.source_path),
        "destination_reference_hash" =>
            TRS._sha256(pair.destination_path),
        "solver_options_hash" => resume_identity["solver_options_hash"],
        "rich_diagnostic_code_digest" =>
            resume_identity["diagnostic_code_digest"],
        "validation_code" => validation_code,
        "runtime" => runtime,
        "warmstart_sha256" => _full_vector_sha256(warmstart),
        "saved_hash_sequence_sha256" =>
            _hash_sequence_sha256(saved_hashes),
        "saved_final_z_sha256" => _full_vector_sha256(rich["final_z"]),
        "saved_final_primal_sha256" =>
            _full_vector_sha256(rich["final_primal"]),
        "saved_scalar_trace_sha256" =>
            _scalar_trace_sha256(rich["events"]),
        "saved_trace_replay_rtol" => trace_replay_rtol,
        "saved_trace_replay_atol" => trace_replay_atol,
        "saved_final_replay_rtol" => final_replay_rtol,
    )
    (;
        rich_path,
        rich,
        hard_input_path,
        hard_input,
        warmstart,
        saved_hashes,
        trace_replay_rtol,
        trace_replay_atol,
        final_replay_rtol,
        identity,
    )
end

function _assert_accepted_metadata_exact(rich_rows, observed_rows)
    length(rich_rows) == length(observed_rows) ||
        error("Accepted snapshot count changed.")
    names = (
        "total_iter",
        "inner_iter",
        "direction_attempt",
        "accepted_alpha",
        "eta_used",
        "eta_next",
    )
    for (accepted_index, (expected, observed)) in
        enumerate(zip(rich_rows, observed_rows))
        for name in names
            _scalar_exact(expected[name], observed[name]) ||
                error(
                    "Accepted snapshot $(accepted_index) field $(name) changed: " *
                    "$(repr(expected[name])) versus $(repr(observed[name])).",
                )
        end
    end
    true
end

function _accepted_validation_rows(
    base,
    saved_rows,
    lightweight_rows,
    rich_replay_rows,
)
    rows = Dict{String, Any}[]
    length(lightweight_rows) == length(rich_replay_rows) ||
        error("Same-process accepted snapshot counts differ.")
    row_count = max(length(saved_rows), length(lightweight_rows))
    for accepted_index in 1:row_count
        saved_present = accepted_index <= length(saved_rows)
        same_process_present =
            accepted_index <= length(lightweight_rows)
        row = copy(base)
        row["accepted_step"] = accepted_index
        row["saved_rich_snapshot_present"] = saved_present
        row["same_process_snapshot_present"] = same_process_present
        if saved_present
            saved = saved_rows[accepted_index]
            row["saved_rich_total_iter"] = Int(saved["total_iter"])
            row["saved_rich_iterate_sha256"] =
                String(saved["iterate_sha256"])
        end
        if same_process_present
            lightweight = lightweight_rows[accepted_index]
            rich_replay = rich_replay_rows[accepted_index]
            row["lightweight_total_iter"] =
                Int(lightweight["total_iter"])
            row["rich_replay_total_iter"] =
                Int(rich_replay["total_iter"])
            row["lightweight_iterate_sha256"] =
                String(lightweight["iterate_sha256"])
            row["rich_replay_iterate_sha256"] =
                String(rich_replay["iterate_sha256"])
            row["same_process_iterate_exact"] =
                row["lightweight_iterate_sha256"] ==
                row["rich_replay_iterate_sha256"]
        end
        if saved_present && same_process_present
            row["saved_rich_hash_exact_diagnostic"] =
                row["saved_rich_iterate_sha256"] ==
                row["rich_replay_iterate_sha256"]
        end
        push!(rows, row)
    end
    rows
end

function _accepted_hash_diagnostic_fields(saved_hashes, new_hashes)
    overlap_count = min(length(saved_hashes), length(new_hashes))
    exact_hash_count = count(
        pair -> pair[1] == pair[2],
        zip(saved_hashes, new_hashes),
    )
    Dict{String, Any}(
        "saved_rich_accepted_comparison_role" =>
            :cross_process_diagnostic_only,
        "saved_rich_accepted_saved_count" => length(saved_hashes),
        "saved_rich_accepted_new_count" => length(new_hashes),
        "saved_rich_accepted_count_difference" =>
            length(new_hashes) - length(saved_hashes),
        "saved_rich_accepted_overlap_count" => overlap_count,
        "saved_rich_accepted_saved_unpaired_count" =>
            length(saved_hashes) - overlap_count,
        "saved_rich_accepted_new_unpaired_count" =>
            length(new_hashes) - overlap_count,
        "saved_rich_exact_hash_count_diagnostic" => exact_hash_count,
    )
end

function _assert_resumed_checkpoint(
    checkpoint,
    expected_identity,
    rich,
    built,
    pair,
)
    get(checkpoint, "schema_version", "") == PATH_VALIDATION_SCHEMA ||
        error("Iterate-path validation checkpoint schema mismatch.")
    get(checkpoint, "protocol", "") == String(TRS.PROTOCOL) ||
        error("Iterate-path validation checkpoint protocol mismatch.")
    checkpoint["identity"] == expected_identity ||
        error("Iterate-path validation checkpoint inputs or code drifted.")
    saved_hashes = _rich_expected_hashes(rich)
    checkpoint["saved_rich_accepted_hashes"] == saved_hashes ||
        error("Stored saved-rich accepted-iterate hashes drifted.")
    lightweight_hashes = checkpoint["lightweight_accepted_hashes"]
    rich_replay_hashes = checkpoint["rich_replay_accepted_hashes"]
    lightweight_hashes == rich_replay_hashes ||
        error("Stored same-process accepted-iterate hashes are not exact.")
    _full_vector_exact(
        checkpoint["lightweight_final_z"],
        checkpoint["rich_replay_final_z"],
    ) || error("Stored same-process final full iterates are not bitwise exact.")
    _full_vector_exact(
        checkpoint["rich_replay_final_primal"],
        checkpoint["rich_replay_final_z"][built.blocks.z],
    ) || error("Stored rich-replay final primal is not its final-z primal slice.")
    _assert_scalar_trace_exact(
        checkpoint["lightweight_events"],
        checkpoint["rich_replay_events"],
    )
    trace_replay_rtol =
        Float64(expected_identity["saved_trace_replay_rtol"])
    trace_replay_atol =
        Float64(expected_identity["saved_trace_replay_atol"])
    final_replay_rtol =
        Float64(expected_identity["saved_final_replay_rtol"])
    trace_comparison = _compare_scalar_trace_diagnostic(
        rich["events"],
        checkpoint["rich_replay_events"];
        rtol = trace_replay_rtol,
        atol = trace_replay_atol,
    )
    final_primal_comparison = _compare_final_primal_basin(
        rich["final_primal"],
        checkpoint["rich_replay_final_primal"];
        threshold = final_replay_rtol,
    )
    row = checkpoint["row"]
    rich_row = rich["row"]
    expected_base = Dict{String, Any}(
        "case_id" => expected_identity["case_id"],
        "formulation" => Symbol(string(rich_row["formulation"])),
        "scenario_seed" => Int(rich_row["scenario_seed"]),
        "transition" => Int(rich_row["transition"]),
        "gamma" => Float64(rich_row["gamma"]),
        "validation_role" => :diagnostic_validation_replay,
        "comparison_scope" =>
            :full_rich_hook_vs_lightweight_hash_hook_same_process,
        "production_candidate_solve" => false,
        "rescue_solve" => false,
    )
    for (name, expected) in expected_base
        haskey(row, name) ||
            error("Stored validation summary lost field $(name).")
        isequal(row[name], expected) ||
            error("Stored validation summary field $(name) drifted.")
    end
    get(row, "rich_checkpoint_sha256", "") ==
        expected_identity["rich_checkpoint_hash"] ||
        error("Stored validation summary rich-checkpoint hash drifted.")
    for (name, expected) in (
        ("saved_rich_trace_replay_rtol", trace_replay_rtol),
        ("saved_rich_trace_replay_atol", trace_replay_atol),
        ("saved_rich_final_replay_rtol", final_replay_rtol),
    )
        _scalar_exact(row[name], expected) ||
            error("Stored validation tolerance $(name) drifted.")
    end
    replay_finish = _finish_event(checkpoint["lightweight_events"])
    get(row, "scalar_trace_events", -1) ==
        length(checkpoint["lightweight_events"]) ||
        error("Stored scalar-trace event count drifted.")
    get(row, "total_inner_iters", -1) == Int(replay_finish.total_iters) ||
        error("Stored total-iteration count drifted.")
    get(row, "solver_status", :unknown) === replay_finish.status ||
        error("Stored solver status drifted.")
    get(row, "accepted_steps", -1) == length(lightweight_hashes) ||
        error("Stored accepted-step count drifted.")
    get(row, "lightweight_hash_sequence_sha256", "") ==
        _hash_sequence_sha256(lightweight_hashes) ||
        error("Stored lightweight iterate-sequence digest drifted.")
    get(row, "rich_replay_hash_sequence_sha256", "") ==
        _hash_sequence_sha256(rich_replay_hashes) ||
        error("Stored rich-replay iterate-sequence digest drifted.")
    get(row, "lightweight_final_z_sha256", "") ==
        _full_vector_sha256(checkpoint["lightweight_final_z"]) ||
        error("Stored lightweight final-iterate digest drifted.")
    get(row, "rich_replay_final_z_sha256", "") ==
        _full_vector_sha256(checkpoint["rich_replay_final_z"]) ||
        error("Stored rich-replay final-iterate digest drifted.")
    get(row, "rich_replay_final_primal_sha256", "") ==
        _full_vector_sha256(checkpoint["rich_replay_final_primal"]) ||
        error("Stored rich-replay final-primal digest drifted.")
    get(row, "lightweight_scalar_trace_sha256", "") ==
        _scalar_trace_sha256(checkpoint["lightweight_events"]) ||
        error("Stored lightweight scalar-trace digest drifted.")
    get(row, "rich_replay_scalar_trace_sha256", "") ==
        _scalar_trace_sha256(checkpoint["rich_replay_events"]) ||
        error("Stored rich-replay scalar-trace digest drifted.")
    get(row, "lightweight_svd_fallback_count", -1) == 0 ||
        error("Stored lightweight replay reported a forbidden fallback.")
    get(row, "rich_replay_svd_fallback_count", -1) == 0 ||
        error("Stored rich replay reported a forbidden fallback.")
    for (name, expected) in _trace_diagnostic_fields(trace_comparison)
        haskey(row, name) ||
            error("Stored saved-rich trace diagnostic lost field $(name).")
        _scalar_exact(row[name], expected) ||
            error("Stored saved-rich trace diagnostic $(name) drifted.")
    end
    _scalar_exact(
        row["saved_rich_final_primal_difference_norm2"],
        final_primal_comparison.difference_norm2,
    ) || error("Stored final-primal difference norm drifted.")
    _scalar_exact(
        row["saved_rich_final_primal_scaled_difference"],
        final_primal_comparison.scaled_difference,
    ) || error("Stored final-primal scaled difference drifted.")
    saved_final_residual =
        Float64(rich_row["direct_final_residual_norm2"])
    direct = TRS.SWS._residual(
        built.kkt,
        checkpoint["rich_replay_final_z"],
        pair.destination["parameters"].θ;
        epsilon = Float64(replay_finish.epsilon),
    )
    stored_replay_final_residual =
        Float64(row["rich_replay_direct_final_residual_norm2"])
    fresh_residual_difference =
        abs(direct.norm2 - stored_replay_final_residual)
    fresh_residual_scaled_difference = fresh_residual_difference / max(
        1.0,
        abs(direct.norm2),
        abs(stored_replay_final_residual),
    )
    fresh_residual_consistent =
        if isfinite(direct.norm2) &&
           isfinite(stored_replay_final_residual)
            fresh_residual_scaled_difference <= final_replay_rtol
        else
            isequal(direct.norm2, stored_replay_final_residual)
        end
    fresh_residual_consistent ||
        error(
            "Fresh residual reevaluation differs from the stored validation " *
            "replay beyond its final replay tolerance.",
        )
    replay_final_residual = stored_replay_final_residual
    residual_absolute_difference =
        abs(replay_final_residual - saved_final_residual)
    residual_scaled_difference = residual_absolute_difference / max(
        1.0,
        abs(replay_final_residual),
        abs(saved_final_residual),
    )
    _scalar_exact(
        row["saved_rich_final_residual_absolute_difference"],
        residual_absolute_difference,
    ) || error("Stored final-residual absolute difference drifted.")
    _scalar_exact(
        row["saved_rich_final_residual_scaled_difference"],
        residual_scaled_difference,
    ) || error("Stored final-residual scaled difference drifted.")
    final_residual_within_tolerance =
        residual_scaled_difference <= final_replay_rtol
    saved_converged = Bool(rich_row["direct_converged"])
    new_converged =
        isfinite(replay_final_residual) &&
        replay_final_residual <= TRS.SOLVER_TOL
    saved_converged == new_converged ||
        error(
            "Saved-rich and resumed new-rich convergence classifications " *
            "differ.",
        )
    jointly_converged = saved_converged && new_converged
    primal_alignment_required = jointly_converged
    primal_alignment_passed =
        !primal_alignment_required || final_primal_comparison.within_basin
    primal_alignment_passed ||
        error(
            "Jointly converged saved-rich and resumed new-rich endpoints are " *
            "outside the recorded primal-alignment threshold.",
        )
    expected_saved_comparison_fields = Dict{String, Any}(
        "saved_rich_direct_converged" => saved_converged,
        "new_rich_direct_converged" => new_converged,
        "saved_rich_convergence_classification_match" => true,
        "saved_rich_jointly_converged" => jointly_converged,
        "saved_rich_primal_alignment_required" =>
            primal_alignment_required,
        "saved_rich_final_primal_within_replay_basin_diagnostic" =>
            final_primal_comparison.within_basin,
        "saved_rich_primal_alignment_passed_when_required" =>
            primal_alignment_passed,
        "saved_rich_final_residual_within_replay_tolerance_diagnostic" =>
            final_residual_within_tolerance,
    )
    for (name, expected) in expected_saved_comparison_fields
        haskey(row, name) ||
            error("Stored saved-rich comparison lost field $(name).")
        _scalar_exact(row[name], expected) ||
            error("Stored saved-rich comparison $(name) drifted.")
    end
    for (name, expected) in _accepted_hash_diagnostic_fields(
        saved_hashes,
        rich_replay_hashes,
    )
        haskey(row, name) ||
            error("Stored saved-rich accepted diagnostic lost field $(name).")
        _scalar_exact(row[name], expected) ||
            error("Stored saved-rich accepted diagnostic $(name) drifted.")
    end
    get(row, "spqr_executed_during_rich_replay_solve", true) === false ||
        error("Stored validation claims in-solve SPQR execution.")
    direction_event_count = count(
        event -> event.event === :direction,
        checkpoint["rich_replay_events"],
    )
    get(row, "direction_snapshots", -1) == direction_event_count ||
        error("Stored direction-snapshot count drifted.")
    get(row, "rich_replay_direction_rows", -1) == direction_event_count ||
        error("Stored rich direction-row count drifted.")
    get(row, "rich_replay_accepted_snapshot_rows", -1) ==
        length(lightweight_hashes) ||
        error("Stored rich accepted-snapshot count drifted.")
    group_count = length(TRS._residual_group_indices(built.kkt))
    expected_family_rows =
        group_count * (direction_event_count + length(lightweight_hashes))
    get(row, "rich_replay_family_rows", -1) == expected_family_rows ||
        error("Stored residual-family row count drifted.")
    get(row, "full_diagnostic_schema", "") ==
        TRS.GLOBALIZATION_DIAGNOSTIC_SCHEMA ||
        error("Stored full-diagnostic schema drifted.")
    get(row, "rich_replay_rrqr_time_sec", -1.0) >= 0.0 ||
        error("Stored rich replay has an invalid RRQR time.")
    minimum_rank = get(row, "rich_replay_minimum_rank_estimate", -1)
    0 <= minimum_rank <= built.kkt.kkt_dimension ||
        error("Stored rich replay has an invalid numerical-rank estimate.")
    for name in (
        "same_process_accepted_hash_sequence_exact",
        "same_process_accepted_metadata_exact",
        "same_process_final_z_bitwise_exact",
        "same_process_scalar_trace_exact",
        "saved_rich_convergence_classification_match",
        "saved_rich_primal_alignment_passed_when_required",
        "full_diagnostic_collector_used",
        "full_diagnostic_finalized_postsolve",
        "zero_forbidden_fallback",
    )
        get(row, name, false) === true ||
            error("Resumed path validation did not pass $(name).")
    end
    accepted_rows = checkpoint["accepted_rows"]
    expected_accepted_row_count =
        max(length(saved_hashes), length(lightweight_hashes))
    length(accepted_rows) == expected_accepted_row_count ||
        error("Stored accepted validation rows are incomplete.")
    for (accepted_index, accepted_row) in enumerate(accepted_rows)
        for (name, expected) in expected_base
            isequal(accepted_row[name], expected) ||
                error(
                    "Stored accepted row $(accepted_index) field $(name) drifted.",
                )
        end
        Int(accepted_row["accepted_step"]) == accepted_index ||
            error("Stored accepted validation row order drifted.")
        saved_present = accepted_index <= length(saved_hashes)
        same_process_present =
            accepted_index <= length(lightweight_hashes)
        get(accepted_row, "saved_rich_snapshot_present", nothing) ===
            saved_present ||
            error("Stored saved-rich presence flag drifted.")
        get(accepted_row, "same_process_snapshot_present", nothing) ===
            same_process_present ||
            error("Stored same-process presence flag drifted.")
        if saved_present
            accepted_row["saved_rich_iterate_sha256"] ==
                saved_hashes[accepted_index] ||
                error(
                    "Stored saved-rich hash drifted at step " *
                    "$(accepted_index).",
                )
        else
            !haskey(accepted_row, "saved_rich_iterate_sha256") ||
                error("Stored unpaired saved-rich hash is unexpected.")
        end
        if same_process_present
            accepted_row["lightweight_iterate_sha256"] ==
                lightweight_hashes[accepted_index] ||
                error(
                    "Stored lightweight hash drifted at step " *
                    "$(accepted_index).",
                )
            accepted_row["rich_replay_iterate_sha256"] ==
                rich_replay_hashes[accepted_index] ||
                error(
                    "Stored rich-replay hash drifted at step " *
                    "$(accepted_index).",
                )
            accepted_row["same_process_iterate_exact"] === true ||
                error("Stored accepted step $(accepted_index) is not exact.")
        else
            !haskey(accepted_row, "lightweight_iterate_sha256") ||
                error("Stored unpaired lightweight hash is unexpected.")
            !haskey(accepted_row, "rich_replay_iterate_sha256") ||
                error("Stored unpaired rich-replay hash is unexpected.")
            !haskey(accepted_row, "same_process_iterate_exact") ||
                error("Stored unpaired same-process comparison is unexpected.")
        end
        if saved_present && same_process_present
            expected_hash_match =
                saved_hashes[accepted_index] ==
                rich_replay_hashes[accepted_index]
            get(
                accepted_row,
                "saved_rich_hash_exact_diagnostic",
                nothing,
            ) === expected_hash_match ||
                error(
                    "Stored saved-rich hash diagnostic drifted at step " *
                    "$(accepted_index).",
                )
        else
            !haskey(
                accepted_row,
                "saved_rich_hash_exact_diagnostic",
            ) || error("Stored unpaired saved-rich hash comparison is unexpected.")
        end
    end
    checkpoint
end

function _validate_case!(
    run_dir,
    pair,
    gamma,
    built,
    options,
    resume_identity,
    validation_code,
    runtime,
)
    inputs = _assert_rich_checkpoint(
        run_dir,
        pair,
        gamma,
        built,
        resume_identity,
        validation_code,
        runtime,
    )
    destination = _case_checkpoint_path(
        run_dir,
        pair.form,
        pair.seed,
        pair.transition,
        gamma,
    )
    if isfile(destination)
        return _assert_resumed_checkpoint(
            JLD2.load_object(destination),
            inputs.identity,
            inputs.rich,
            built,
            pair,
        )
    end

    base = Dict{String, Any}(
        "case_id" => inputs.identity["case_id"],
        "formulation" => pair.form,
        "scenario_seed" => pair.seed,
        "transition" => pair.transition,
        "gamma" => gamma,
        "validation_role" => :diagnostic_validation_replay,
        "comparison_scope" =>
            :full_rich_hook_vs_lightweight_hash_hook_same_process,
        "production_candidate_solve" => false,
        "rescue_solve" => false,
    )

    lightweight_collector = AcceptedIterateHashCollector()
    lightweight_events = Any[]
    lightweight_output = nothing
    lightweight_elapsed = @elapsed lightweight_output = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        built.kkt,
        pair.destination["parameters"].θ;
        z₀ = copy(inputs.warmstart),
        options,
        trace_hook = event -> push!(lightweight_events, event),
        diagnostic_hook = lightweight_collector,
    )

    rich_collector = TRS.GlobalizationDiagnosticCollector(
        base,
        TRS._residual_group_indices(built.kkt),
        Dict{String, Any}[],
        Dict{String, Any}[],
        Dict{String, Any}[],
        Dict{Int, Any}(),
        false,
    )
    rich_replay_events = Any[]
    rich_replay_output = nothing
    rich_replay_solve_elapsed =
        @elapsed rich_replay_output = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        built.kkt,
        pair.destination["parameters"].θ;
        z₀ = copy(inputs.warmstart),
        options,
        trace_hook = event -> push!(rich_replay_events, event),
        diagnostic_hook = rich_collector,
    )
    rich_replay_postprocess_elapsed =
        @elapsed TRS._finalize_globalization_diagnostics!(rich_collector)
    rich_collector.finalized ||
        error("Full rich diagnostics were not finalized post-solve.")

    _assert_scalar_trace_exact(lightweight_events, rich_replay_events)
    lightweight_hashes = String[
        String(row["iterate_sha256"]) for
        row in lightweight_collector.accepted_rows
    ]
    rich_replay_hashes = String[
        String(row["iterate_sha256"]) for
        row in rich_collector.accepted_snapshot_rows
    ]
    lightweight_hashes == rich_replay_hashes ||
        error(
            "Full SPQR instrumentation changed the accepted iterate path for " *
            "$(inputs.identity["case_id"]).",
        )
    _assert_accepted_metadata_exact(
        lightweight_collector.accepted_rows,
        rich_collector.accepted_snapshot_rows,
    )
    _full_vector_exact(lightweight_output.z, rich_replay_output.z) ||
        error(
            "Full SPQR instrumentation changed the final full iterate for " *
            "$(inputs.identity["case_id"]).",
        )
    lightweight_collector.direction_snapshot_count ==
        length(rich_collector.direction_rows) ||
        error("Same-process direction-snapshot counts differ.")

    saved_trace_comparison = _compare_scalar_trace_diagnostic(
        inputs.rich["events"],
        rich_replay_events;
        rtol = inputs.trace_replay_rtol,
        atol = inputs.trace_replay_atol,
    )
    rich_replay_final_primal =
        copy(rich_replay_output.z[built.blocks.z])
    saved_final_primal_comparison = _compare_final_primal_basin(
        inputs.rich["final_primal"],
        rich_replay_final_primal;
        threshold = inputs.final_replay_rtol,
    )
    direct = TRS.SWS._residual(
        built.kkt,
        rich_replay_output.z,
        pair.destination["parameters"].θ;
        epsilon = rich_replay_output.ϵ,
    )
    saved_final_residual =
        Float64(inputs.rich["row"]["direct_final_residual_norm2"])
    final_residual_absolute_difference =
        abs(direct.norm2 - saved_final_residual)
    final_residual_scaled_difference =
        final_residual_absolute_difference /
        max(1.0, abs(direct.norm2), abs(saved_final_residual))
    final_residual_within_tolerance =
        final_residual_scaled_difference <= inputs.final_replay_rtol
    saved_converged = Bool(inputs.rich["row"]["direct_converged"])
    new_converged =
        isfinite(direct.norm2) && direct.norm2 <= TRS.SOLVER_TOL
    saved_converged == new_converged ||
        error(
            "Saved-rich and new-rich convergence classifications differ for " *
            "$(inputs.identity["case_id"]).",
        )
    jointly_converged = saved_converged && new_converged
    primal_alignment_required = jointly_converged
    primal_alignment_passed =
        !primal_alignment_required ||
        saved_final_primal_comparison.within_basin
    primal_alignment_passed ||
        error(
            "Jointly converged saved-rich and new-rich endpoints are outside " *
            "the recorded primal-alignment threshold.",
        )

    rich_direction_fallback_count = sum(
        Int(row["svd_fallback_count"]) for row in rich_collector.direction_rows
    )
    lightweight_finish = _finish_event(lightweight_events)
    rich_replay_finish = _finish_event(rich_replay_events)
    lightweight_collector.direction_svd_fallback_count == 0 ||
        error("Lightweight replay direction snapshots reported a dense fallback.")
    rich_direction_fallback_count == 0 ||
        error("Rich replay direction snapshots reported a dense fallback.")
    lightweight_output.svd_fallback_count == 0 ||
        error("Lightweight replay invoked the forbidden dense fallback.")
    rich_replay_output.svd_fallback_count == 0 ||
        error("Rich replay invoked the forbidden dense fallback.")
    Int(lightweight_finish.svd_fallback_count) == 0 ||
        error("Lightweight replay finish event reported a dense fallback.")
    Int(rich_replay_finish.svd_fallback_count) == 0 ||
        error("Rich replay finish event reported a dense fallback.")

    accepted_rows = _accepted_validation_rows(
        base,
        inputs.rich["accepted_snapshot_rows"],
        lightweight_collector.accepted_rows,
        rich_collector.accepted_snapshot_rows,
    )
    row = copy(base)
    row["accepted_steps"] = length(lightweight_hashes)
    row["direction_snapshots"] =
        lightweight_collector.direction_snapshot_count
    row["scalar_trace_events"] = length(lightweight_events)
    row["total_inner_iters"] = lightweight_output.total_iters
    row["solver_status"] = lightweight_output.status
    row["lightweight_solve_and_hash_time_sec"] = lightweight_elapsed
    row["rich_replay_solve_time_sec"] = rich_replay_solve_elapsed
    row["rich_replay_diagnostic_postprocess_time_sec"] =
        rich_replay_postprocess_elapsed
    row["rich_replay_solve_and_diagnostics_time_sec"] =
        rich_replay_solve_elapsed + rich_replay_postprocess_elapsed
    row["spqr_executed_during_rich_replay_solve"] = false
    row["lightweight_hash_sequence_sha256"] =
        _hash_sequence_sha256(lightweight_hashes)
    row["rich_replay_hash_sequence_sha256"] =
        _hash_sequence_sha256(rich_replay_hashes)
    row["saved_rich_hash_sequence_sha256"] =
        _hash_sequence_sha256(inputs.saved_hashes)
    row["lightweight_final_z_sha256"] =
        _full_vector_sha256(lightweight_output.z)
    row["rich_replay_final_z_sha256"] =
        _full_vector_sha256(rich_replay_output.z)
    row["rich_replay_final_primal_sha256"] =
        _full_vector_sha256(rich_replay_final_primal)
    row["lightweight_scalar_trace_sha256"] =
        _scalar_trace_sha256(lightweight_events)
    row["rich_replay_scalar_trace_sha256"] =
        _scalar_trace_sha256(rich_replay_events)
    row["same_process_accepted_hash_sequence_exact"] = true
    row["same_process_accepted_metadata_exact"] = true
    row["same_process_final_z_bitwise_exact"] = true
    row["same_process_scalar_trace_exact"] = true
    merge!(
        row,
        _accepted_hash_diagnostic_fields(
            inputs.saved_hashes,
            rich_replay_hashes,
        ),
    )
    merge!(row, _trace_diagnostic_fields(saved_trace_comparison))
    row["saved_rich_direct_converged"] = saved_converged
    row["new_rich_direct_converged"] = new_converged
    row["saved_rich_convergence_classification_match"] = true
    row["saved_rich_jointly_converged"] = jointly_converged
    row["saved_rich_primal_alignment_required"] =
        primal_alignment_required
    row["saved_rich_final_primal_within_replay_basin_diagnostic"] =
        saved_final_primal_comparison.within_basin
    row["saved_rich_primal_alignment_passed_when_required"] =
        primal_alignment_passed
    row["saved_rich_final_primal_difference_norm2"] =
        saved_final_primal_comparison.difference_norm2
    row["saved_rich_final_primal_scaled_difference"] =
        saved_final_primal_comparison.scaled_difference
    row["saved_rich_final_residual_within_replay_tolerance_diagnostic"] =
        final_residual_within_tolerance
    row["rich_replay_direct_final_residual_norm2"] = direct.norm2
    row["saved_rich_final_residual_absolute_difference"] =
        final_residual_absolute_difference
    row["saved_rich_final_residual_scaled_difference"] =
        final_residual_scaled_difference
    row["saved_rich_trace_replay_rtol"] =
        inputs.trace_replay_rtol
    row["saved_rich_trace_replay_atol"] =
        inputs.trace_replay_atol
    row["saved_rich_final_replay_rtol"] =
        inputs.final_replay_rtol
    row["full_diagnostic_collector_used"] = true
    row["full_diagnostic_finalized_postsolve"] = true
    row["full_diagnostic_schema"] = TRS.GLOBALIZATION_DIAGNOSTIC_SCHEMA
    row["rich_replay_direction_rows"] =
        length(rich_collector.direction_rows)
    row["rich_replay_family_rows"] = length(rich_collector.family_rows)
    row["rich_replay_accepted_snapshot_rows"] =
        length(rich_collector.accepted_snapshot_rows)
    row["rich_replay_rrqr_time_sec"] = sum(
        Float64(direction["rrqr_time_sec"]) for
        direction in rich_collector.direction_rows
    )
    row["rich_replay_minimum_rank_estimate"] = minimum(
        Int(direction["numerical_rank_estimate"]) for
        direction in rich_collector.direction_rows
    )
    row["zero_forbidden_fallback"] = true
    row["lightweight_svd_fallback_count"] =
        lightweight_output.svd_fallback_count
    row["rich_replay_svd_fallback_count"] =
        rich_replay_output.svd_fallback_count
    row["rich_checkpoint"] = abspath(inputs.rich_path)
    row["rich_checkpoint_sha256"] =
        inputs.identity["rich_checkpoint_hash"]

    checkpoint = Dict{String, Any}(
        "schema_version" => PATH_VALIDATION_SCHEMA,
        "protocol" => String(TRS.PROTOCOL),
        "identity" => inputs.identity,
        "row" => row,
        "accepted_rows" => accepted_rows,
        "saved_rich_accepted_hashes" => inputs.saved_hashes,
        "lightweight_accepted_hashes" => lightweight_hashes,
        "rich_replay_accepted_hashes" => rich_replay_hashes,
        "lightweight_final_z" => copy(lightweight_output.z),
        "rich_replay_final_z" => copy(rich_replay_output.z),
        "rich_replay_final_primal" => rich_replay_final_primal,
        "lightweight_events" => lightweight_events,
        "rich_replay_events" => rich_replay_events,
    )
    TRS._atomic_save(destination, checkpoint)
    checkpoint
end

function _write_progress!(
    run_dir,
    checkpoints,
    resume_identity,
    validation_code,
    runtime;
    complete,
)
    _assert_identity_unchanged(validation_code, runtime)
    summary_rows =
        Dict{String, Any}[checkpoint["row"] for checkpoint in checkpoints]
    accepted_rows = reduce(
        vcat,
        (checkpoint["accepted_rows"] for checkpoint in checkpoints);
        init = Dict{String, Any}[],
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "globalization_path_validation.csv"),
        accepted_rows,
    )
    DTS.write_csv(
        joinpath(
            run_dir,
            "raw",
            "globalization_path_validation_summary.csv",
        ),
        summary_rows,
    )
    summary_path = joinpath(
        run_dir,
        "checkpoints",
        "globalization_path_validation_summary.jld2",
    )
    TRS._atomic_save(
        summary_path,
        Dict{String, Any}(
            "schema_version" => PATH_VALIDATION_SUMMARY_SCHEMA,
            "protocol" => String(TRS.PROTOCOL),
            "complete" => complete,
            "completed_case_count" => length(checkpoints),
            "expected_case_count" => length(_hard_cases()),
            "case_ids" =>
                String[checkpoint["row"]["case_id"] for checkpoint in checkpoints],
            "case_rows" => summary_rows,
            "solver_options_hash" =>
                resume_identity["solver_options_hash"],
            "rich_diagnostic_code_digest" =>
                resume_identity["diagnostic_code_digest"],
            "validation_code" => validation_code,
            "runtime" => runtime,
        ),
    )
    if complete
        summary_sha256 = TRS._sha256(summary_path)
        TRS._atomic_write(
            joinpath(run_dir, "globalization_path_validation_complete"),
            "schema=$(PATH_VALIDATION_SUMMARY_SCHEMA)\n" *
            "summary_sha256=$(summary_sha256)\n" *
            "completed_case_count=$(length(checkpoints))\n",
        )
    end
    summary_rows
end

function _load_frozen_transport_config(path)
    values = TOML.parsefile(path)
    config = DTS.TransportStudyConfig(;
        baseline_dir = abspath(String(values["baseline_dir"])),
        output_root = abspath(String(values["output_root"])),
        profile = Symbol(values["profile"]),
        protocol = Symbol(values["protocol"]),
        planning_horizon = Int(values["planning_horizon"]),
        Δt = Float64(values["dt"]),
        tol = Float64(values["tol"]),
        max_inner_iters = Int(values["max_inner_iters"]),
        max_outer_iters = Int(values["max_outer_iters"]),
        linear_solver = Symbol(values["linear_solver"]),
        linesearch = Symbol(values["linesearch"]),
        fd_codegen_chunk_size = Int(values["fd_codegen_chunk_size"]),
        order_seed = Int(values["order_seed"]),
        bootstrap_seed = Int(values["bootstrap_seed"]),
        bootstrap_replicates = Int(values["bootstrap_replicates"]),
        gammas = Float64.(values["gammas"]),
        projection_rtols = Float64.(values["projection_rtols"]),
        projection_solve = Bool(values["projection_solve"]),
        save_full_solutions = Bool(values["save_full_solutions"]),
    )
    actual = (
        config.protocol,
        config.planning_horizon,
        config.Δt,
        config.tol,
        config.max_inner_iters,
        config.max_outer_iters,
        config.linear_solver,
        config.linesearch,
        config.gammas,
        config.projection_rtols,
    )
    expected = (
        DTS.PROTOCOL,
        DTS.PLANNING_HORIZON,
        DTS.DELTA_T,
        DTS.SOLVER_TOL,
        DTS.MAX_INNER_ITERS,
        DTS.MAX_OUTER_ITERS,
        DTS.LINEAR_SOLVER,
        DTS.LINESEARCH,
        collect(DTS.GAMMAS),
        collect(DTS.PROJECTION_RTOLS),
    )
    actual == expected ||
        error("Frozen dual-transport numerical protocol drifted.")
    config
end

function _frozen_source_config_and_options(run_dir)
    snapshot_root = joinpath(
        run_dir,
        "inputs",
        "dual_transport",
    )
    config_path = joinpath(snapshot_root, "config.toml")
    options_path = joinpath(snapshot_root, "solver_options.toml")
    isfile(config_path) ||
        error("Frozen dual-transport config is missing: $(config_path)")
    isfile(options_path) ||
        error("Frozen solver-options snapshot is missing: $(options_path)")
    source_config = _load_frozen_transport_config(config_path)
    options = DTS._solver_options(source_config)
    frozen = TOML.parsefile(options_path)["options"]
    actual = Dict(
        string(name) => begin
            value = getproperty(options, name)
            value isa Symbol ? String(value) : value
        end for name in propertynames(options)
    )
    Set(keys(actual)) == Set(keys(frozen)) ||
        error("Frozen solver-option schema differs from the replay options.")
    for name in keys(frozen)
        isequal(actual[name], frozen[name]) ||
            error(
                "Frozen replay option $(name) drifted: expected " *
                "$(repr(frozen[name])), got $(repr(actual[name])).",
            )
    end
    expected_protocol_options = (
        tol = TRS.SOLVER_TOL,
        max_inner_iters = TRS.MAX_INNER_ITERS,
        max_outer_iters = TRS.MAX_OUTER_ITERS,
        linear_solver = TRS.LINEAR_SOLVER,
        linesearch = TRS.LINESEARCH,
        record_condition_number = false,
        tsvd_threshold = 0.0,
        use_marquardt_scaling = false,
        reuse_factorization_iters = 0,
    )
    for name in propertynames(expected_protocol_options)
        getproperty(options, name) ==
            getproperty(expected_protocol_options, name) ||
            error(
                "Frozen solver option $(name) violates the theory-resolution " *
                "protocol.",
            )
    end
    (; source_config, options)
end

function _load_run_config(path)
    values = TOML.parsefile(path)
    TRS.TheoryResolutionConfig(;
        source_run = abspath(String(values["source_run"])),
        output_root = abspath(String(values["output_root"])),
        profile = Symbol(values["profile"]),
        protocol = Symbol(values["protocol"]),
        planning_horizon = Int(values["planning_horizon"]),
        Δt = Float64(values["dt"]),
        tol = Float64(values["tol"]),
        max_inner_iters = Int(values["max_inner_iters"]),
        max_outer_iters = Int(values["max_outer_iters"]),
        linear_solver = Symbol(values["linear_solver"]),
        linesearch = Symbol(values["linesearch"]),
        heldout_gamma = Float64(values["heldout_gamma"]),
        alignment_tol = Float64(values["alignment_tol"]),
    )
end

function _validate_frozen_protocol(config)
    actual = (
        config.protocol,
        config.planning_horizon,
        config.Δt,
        config.tol,
        config.max_inner_iters,
        config.max_outer_iters,
        config.linear_solver,
        config.linesearch,
        config.heldout_gamma,
        config.alignment_tol,
    )
    expected = (
        TRS.PROTOCOL,
        TRS.PLANNING_HORIZON,
        TRS.DELTA_T,
        TRS.SOLVER_TOL,
        TRS.MAX_INNER_ITERS,
        TRS.MAX_OUTER_ITERS,
        TRS.LINEAR_SOLVER,
        TRS.LINESEARCH,
        TRS.HELDOUT_GAMMA,
        TRS.ALIGNMENT_TOL,
    )
    actual == expected ||
        error(
            "Iterate-path validation protocol is frozen at $(expected); " *
            "got $(actual).",
        )
    nothing
end

function _validated_context(run_dir, config::TRS.TheoryResolutionConfig)
    run_dir = abspath(run_dir)
    validation_code = _validation_code_identity()
    runtime = _runtime_identity()
    _validate_frozen_protocol(config)
    stored = _load_run_config(joinpath(run_dir, "config.toml"))
    TRS._config_dict(stored) == TRS._config_dict(config) ||
        error("Run-directory configuration differs from the active configuration.")
    isfile(joinpath(run_dir, "inputs_complete")) ||
        error("Theory-resolution input snapshot is incomplete.")
    TRS.snapshot_inputs!(run_dir, config)
    isfile(joinpath(run_dir, "globalization_complete")) ||
        error("Run the rich globalization phase before path validation.")
    pairs = DTS.valid_pairs(run_dir)
    length(pairs) == 17 ||
        error("Expected 17 frozen reference pairs, got $(length(pairs)).")
    pair_index = Dict(TRS._case_key(pair) => pair for pair in pairs)
    frozen = _frozen_source_config_and_options(run_dir)
    source_config = frozen.source_config
    options = frozen.options
    options.linear_solver === :klu ||
        error("Path validation is frozen to the KLU production path.")
    options.linesearch === :backtracking ||
        error("Path validation is frozen to backtracking.")
    resume_identity = TRS._globalization_resume_identity(run_dir)
    (;
        run_dir,
        pair_index,
        source_config,
        options,
        resume_identity,
        validation_code,
        runtime,
    )
end

"""
Read-only preflight for the 15 rich checkpoints.

This validates frozen configuration, hashes, rich-v3 schema, warm starts, and
the absence of a rich-run SVD fallback. It constructs the same KKT systems used
by the replay but does not invoke the optimizer.
"""
function validate_iterate_path_inputs(
    run_dir::AbstractString;
    config = _load_run_config(joinpath(abspath(run_dir), "config.toml")),
)
    context = _validated_context(run_dir, config)
    cache = Dict{Symbol, Any}()
    validated = String[]
    for case in _hard_cases()
        key = (case.form, case.seed, case.transition)
        haskey(context.pair_index, key) ||
            error("Hard-case pair $(key) is unavailable.")
        pair = context.pair_index[key]
        built = DTS._build_system(context.source_config, case.form, cache)
        inputs = _assert_rich_checkpoint(
            context.run_dir,
            pair,
            case.gamma,
            built,
            context.resume_identity,
            context.validation_code,
            context.runtime,
        )
        push!(validated, String(inputs.identity["case_id"]))
    end
    length(validated) == 15 ||
        error("Expected 15 rich hard-case checkpoints, got $(length(validated)).")
    _assert_identity_unchanged(
        context.validation_code,
        context.runtime,
    )
    validated
end

"""
Validate instrumentation neutrality on all 15 preregistered hard cases.

Each successful case is atomically checkpointed, so an interrupted invocation
resumes without repeating completed validation pairs. Each case performs two
diagnostic validation replays in the same Julia process: a lightweight
accepted-iterate hash replay followed by the full rich/SPQR collector replay.
Their scalar events, accepted iterates, and final full `z` must agree exactly.
The saved-rich checkpoint is a cross-process diagnostic reference: only its
convergence classification, and its final primal when jointly converged, are
validation gates. These are instrumentation validations, not candidate, rescue,
or tuning solves.
"""
function run_iterate_path_validation!(
    run_dir::AbstractString;
    config = nothing,
    cache = Dict{Symbol, Any}(),
)
    resolved_run_dir = abspath(run_dir)
    complete_marker = joinpath(
        resolved_run_dir,
        "globalization_path_validation_complete",
    )
    isfile(complete_marker) && rm(complete_marker)
    active_config = if isnothing(config)
        _load_run_config(joinpath(resolved_run_dir, "config.toml"))
    else
        config
    end
    context = _validated_context(resolved_run_dir, active_config)
    checkpoints = Any[]
    for case in _hard_cases()
        key = (case.form, case.seed, case.transition)
        haskey(context.pair_index, key) ||
            error("Hard-case pair $(key) is unavailable.")
        pair = context.pair_index[key]
        built = DTS._build_system(context.source_config, case.form, cache)
        push!(
            checkpoints,
            _validate_case!(
                context.run_dir,
                pair,
                case.gamma,
                built,
                context.options,
                context.resume_identity,
                context.validation_code,
                context.runtime,
            ),
        )
        _write_progress!(
            context.run_dir,
            checkpoints,
            context.resume_identity,
            context.validation_code,
            context.runtime;
            complete = false,
        )
    end
    length(checkpoints) == 15 ||
        error("Expected 15 iterate-path validations.")
    rows = _write_progress!(
        context.run_dir,
        checkpoints,
        context.resume_identity,
        context.validation_code,
        context.runtime;
        complete = true,
    )
    all(
        row ->
            row["same_process_accepted_hash_sequence_exact"] &&
                row["same_process_accepted_metadata_exact"] &&
                row["same_process_final_z_bitwise_exact"] &&
                row["same_process_scalar_trace_exact"] &&
                row["saved_rich_convergence_classification_match"] &&
                row["saved_rich_primal_alignment_passed_when_required"] &&
                row["full_diagnostic_collector_used"] &&
                row["full_diagnostic_finalized_postsolve"] &&
                row["zero_forbidden_fallback"],
        rows,
    ) || error("At least one hard-case iterate-path validation failed.")
    rows
end

end # module GlobalizationIteratePathValidation
