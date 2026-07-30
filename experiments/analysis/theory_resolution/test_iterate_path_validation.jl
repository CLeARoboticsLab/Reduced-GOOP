using Test
using SparseArrays

include(joinpath(@__DIR__, "IteratePathValidation.jl"))
const IPV = GlobalizationIteratePathValidation

@testset "globalization iterate-path validation helpers" begin
    validation_code = IPV._validation_code_identity()
    runtime = IPV._runtime_identity()
    @test validation_code["schema"] ==
          "iterate_path_validation_code_manifest_v5"
    @test length(validation_code["files"]) == 3
    @test runtime["julia_version"] == string(VERSION)
    @test isnothing(IPV._assert_identity_unchanged(validation_code, runtime))
    frozen_theory_config = IPV.TRS.TheoryResolutionConfig(;
        source_run = "/definitely/not/a/live/source/run",
    )
    @test isnothing(IPV._validate_frozen_protocol(frozen_theory_config))
    mktempdir() do directory
        transport_config = IPV.DTS.TransportStudyConfig(;
            baseline_dir = joinpath(directory, "missing_baseline"),
        )
        path = joinpath(directory, "config.toml")
        IPV.DTS._write_toml(path, IPV.DTS.config_dict(transport_config))
        loaded = IPV._load_frozen_transport_config(path)
        @test loaded.baseline_dir == transport_config.baseline_dir
        @test !isdir(loaded.baseline_dir)
    end

    a = [0.0, -0.0, 1.25, Inf, NaN]
    b = copy(a)
    @test IPV._full_vector_sha256(a) == IPV._full_vector_sha256(b)
    @test IPV._full_vector_exact(a, b)
    b[1] = -0.0
    @test !IPV._full_vector_exact(a, b)
    @test IPV._full_vector_sha256(a) != IPV._full_vector_sha256(b)

    expected = Any[
        (event = :initial_residual, total_iter = 0, residual_norm2 = 1.0),
        (
            event = :finish,
            status = :solved,
            total_iters = 1,
            svd_fallback_count = 0,
            residual_norm2 = NaN,
        ),
    ]
    observed = deepcopy(expected)
    @test IPV._assert_scalar_trace_exact(expected, observed)
    @test IPV._scalar_trace_sha256(expected) ==
          IPV._scalar_trace_sha256(observed)
    changed = deepcopy(expected)
    changed[1] = (
        event = :initial_residual,
        total_iter = 0,
        residual_norm2 = nextfloat(1.0),
    )
    @test_throws ErrorException IPV._assert_scalar_trace_exact(expected, changed)
    @test IPV._scalar_trace_sha256(expected) !=
          IPV._scalar_trace_sha256(changed)
    nearby = deepcopy(expected)
    nearby[1] = (
        event = :initial_residual,
        total_iter = 0,
        residual_norm2 = 1.0 + 5e-11,
    )
    comparison = IPV._compare_scalar_trace_diagnostic(
        expected,
        nearby;
        rtol = 1e-10,
        atol = 1e-10,
    )
    @test comparison.saved_event_count == 2
    @test comparison.new_event_count == 2
    @test comparison.event_count_difference == 0
    @test comparison.event_schema_exact_count == 2
    @test comparison.numeric_out_of_tolerance_count == 0
    diagnostic_difference = Any[
        (
            event = :different_event,
            total_iter = 0,
            residual_norm2 = 2.0,
            extra_field = true,
        ),
    ]
    divergent_comparison = IPV._compare_scalar_trace_diagnostic(
        expected,
        diagnostic_difference;
        rtol = 0.0,
        atol = 0.0,
    )
    @test divergent_comparison.saved_event_count == 2
    @test divergent_comparison.new_event_count == 1
    @test divergent_comparison.event_count_difference == -1
    @test divergent_comparison.overlap_event_count == 1
    @test divergent_comparison.saved_unpaired_event_count == 1
    @test divergent_comparison.new_unpaired_event_count == 0
    @test divergent_comparison.event_type_mismatch_count == 1
    @test divergent_comparison.event_schema_mismatch_count == 1
    @test divergent_comparison.new_only_field_count == 1
    @test divergent_comparison.discrete_field_mismatch_count == 1
    @test divergent_comparison.numeric_out_of_tolerance_count == 1
    diagnostic_fields =
        IPV._trace_diagnostic_fields(divergent_comparison)
    @test diagnostic_fields["saved_rich_trace_comparison_role"] ===
          :cross_process_diagnostic_only
    basin_comparison = IPV._compare_final_primal_basin(
        ones(4),
        ones(4) .+ 1e-4;
        threshold = 1e-3,
    )
    @test basin_comparison.within_basin
    separated_basin_comparison = IPV._compare_final_primal_basin(
        ones(4),
        fill(3.0, 4);
        threshold = 1e-3,
    )
    @test !separated_basin_comparison.within_basin

    collector = IPV.AcceptedIterateHashCollector()
    collector((
        event = :direction_snapshot,
        svd_fallback_count = 0,
    ))
    collector((
        event = :accepted_snapshot,
        total_iter = 1,
        inner_iter = 1,
        direction_attempt = 1,
        accepted_alpha = 0.5,
        eta_used = 1e-4,
        eta_next = 2e-4,
        z = copy(a),
    ))
    @test collector.direction_snapshot_count == 1
    @test collector.direction_svd_fallback_count == 0
    @test length(collector.accepted_rows) == 1
    @test collector.accepted_rows[1]["iterate_sha256"] ==
          IPV._full_vector_sha256(a)
    saved_row = Dict{String, Any}(
        "total_iter" => 1,
        "iterate_sha256" => "saved",
    )
    rich_row = copy(collector.accepted_rows[1])
    validation_rows = IPV._accepted_validation_rows(
        Dict{String, Any}("case_id" => "synthetic"),
        [saved_row],
        collector.accepted_rows,
        [rich_row],
    )
    @test validation_rows[1]["same_process_iterate_exact"]
    @test !validation_rows[1]["saved_rich_hash_exact_diagnostic"]
    unpaired_validation_rows = IPV._accepted_validation_rows(
        Dict{String, Any}("case_id" => "synthetic"),
        [
            saved_row,
            Dict{String, Any}(
                "total_iter" => 2,
                "iterate_sha256" => "saved-only",
            ),
        ],
        collector.accepted_rows,
        [rich_row],
    )
    @test length(unpaired_validation_rows) == 2
    @test unpaired_validation_rows[2]["saved_rich_snapshot_present"]
    @test !unpaired_validation_rows[2]["same_process_snapshot_present"]
    @test !haskey(
        unpaired_validation_rows[2],
        "same_process_iterate_exact",
    )
    second_same_process_row = copy(rich_row)
    second_same_process_row["total_iter"] = 2
    second_same_process_row["iterate_sha256"] = "new-only"
    reverse_unpaired_validation_rows = IPV._accepted_validation_rows(
        Dict{String, Any}("case_id" => "synthetic"),
        [saved_row],
        [collector.accepted_rows[1], second_same_process_row],
        [rich_row, second_same_process_row],
    )
    @test length(reverse_unpaired_validation_rows) == 2
    @test !reverse_unpaired_validation_rows[2][
        "saved_rich_snapshot_present"
    ]
    @test reverse_unpaired_validation_rows[2][
        "same_process_snapshot_present"
    ]
    @test reverse_unpaired_validation_rows[2][
        "same_process_iterate_exact"
    ]
    @test !haskey(
        reverse_unpaired_validation_rows[2],
        "saved_rich_iterate_sha256",
    )
    accepted_diagnostics = IPV._accepted_hash_diagnostic_fields(
        ["same", "saved-only"],
        ["same", "new-only", "new-extra"],
    )
    @test accepted_diagnostics[
        "saved_rich_accepted_comparison_role"
    ] === :cross_process_diagnostic_only
    @test accepted_diagnostics["saved_rich_accepted_saved_count"] == 2
    @test accepted_diagnostics["saved_rich_accepted_new_count"] == 3
    @test accepted_diagnostics[
        "saved_rich_accepted_count_difference"
    ] == 1
    @test accepted_diagnostics[
        "saved_rich_accepted_overlap_count"
    ] == 2
    @test accepted_diagnostics[
        "saved_rich_exact_hash_count_diagnostic"
    ] == 1
    full_collector = IPV.TRS.GlobalizationDiagnosticCollector(
        Dict{String, Any}(),
        Pair{Tuple{Symbol, Symbol}, Vector{Int}}[],
        Dict{String, Any}[],
        Dict{String, Any}[],
        Dict{String, Any}[],
        Dict{Int, Any}(),
        false,
    )
    IPV.TRS._finalize_globalization_diagnostics!(full_collector)
    @test full_collector.finalized
    postsolve_collector = IPV.TRS.GlobalizationDiagnosticCollector(
        Dict{String, Any}(),
        Pair{Tuple{Symbol, Symbol}, Vector{Int}}[],
        Dict{String, Any}[
            Dict(
                "total_iter" => 1,
                "direction_attempt" => 1,
            ),
        ],
        Dict{String, Any}[
            Dict(
                "total_iter" => 1,
                "direction_attempt" => 1,
                "_postsolve_residual" => [0.25, -0.5],
            ),
        ],
        Dict{String, Any}[],
        Dict{Int, Any}(
            1 => (
                jacobian = sparse([1.0 0.0; 0.0 2.0]),
                residual = [0.25, -0.5],
            ),
        ),
        false,
    )
    IPV.TRS._finalize_globalization_diagnostics!(postsolve_collector)
    @test postsolve_collector.finalized
    @test isempty(postsolve_collector.rank_inputs)
    @test haskey(
        postsolve_collector.direction_rows[1],
        "numerical_rank_estimate",
    )
    @test !haskey(
        postsolve_collector.accepted_snapshot_rows[1],
        "_postsolve_residual",
    )
    @test length(IPV._hard_cases()) == 15
    @test length(unique(
        IPV._case_id(
            case.form,
            case.seed,
            case.transition,
            case.gamma,
        ) for case in IPV._hard_cases()
    )) == 15
end
