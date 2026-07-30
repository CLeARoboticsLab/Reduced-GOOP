using Test
using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "SelectiveWarmstartStudy.jl"))
const SWS = Main.SelectiveWarmstartStudy
include(joinpath(@__DIR__, "..", "Figures.jl"))
const SWF = Main.SelectiveWarmstartFigures

struct ToyResidual end

function (::ToyResidual)(output, z; θ, ϵ, η)
    output[1] = z[1] + θ[1] + ϵ
    output[2] = 2z[2] - θ[2] + η
    output
end

@testset "Selective warm-start study metrics" begin
    @test SWF._symmetric_offsets(0) == Float64[]
    @test SWF._symmetric_offsets(1) == [0.0]
    @test SWF._symmetric_offsets(4) ==
          collect(range(-0.12, 0.12; length = 4))
    @test_throws ArgumentError SWF._symmetric_offsets(-1)

    singleton_shift_rows = [
        Dict{String, String}(
            "formulation" => formulation,
            "valid_reference_pair" => "true",
            "scenario_seed" => "101",
            "transition" => "1",
            "shift_quality_lambda" => "0.9",
            "shift_quality_psi_out" => "1.1",
            "shift_quality_psi_in" => "1.3",
        ) for formulation in ("reduced", "quasi")
    ]
    singleton_shift_figure =
        SWF._shift_quality_figure(singleton_shift_rows)
    singleton_figure_dir = mktempdir()
    SWF._save_versions(
        singleton_figure_dir,
        "singleton_shift_quality",
        singleton_shift_figure,
    )
    @test isfile(
        joinpath(singleton_figure_dir, "singleton_shift_quality.png"),
    )
    @test isfile(
        joinpath(singleton_figure_dir, "singleton_shift_quality.pdf"),
    )

    initial = [4.0, 2.0, 10.0, 14.0]
    destination = [3.0, 4.0, 8.0, 10.0]

    @test SWS._normalized_block_error(initial, destination, [1, 2]) ≈
          sqrt(5.0) / 5.0
    @test SWS._normalized_block_error(initial, destination, Int[]) == 0.0

    transported = [100.0, 100.0, 9.0, 10.0]
    @test SWS._shift_quality(transported, destination, [3, 4]) ≈
          1.0 / (sqrt(164.0) + eps(Float64))
    @test SWS._shift_quality(transported, destination, Int[]) == 0.0

    kkt = (; kkt_dimension = 2, F! = ToyResidual())
    warmstart = [1.0, 2.0]
    θ = [3.0, 1.0]
    metrics =
        SWS._initial_residual_metrics(kkt, warmstart, θ; epsilon = 0.5)
    expected_values = [4.5, 3.0]
    @test metrics.residual.values == expected_values
    @test metrics.initial_residual_norm2 ≈ norm(expected_values)
    @test metrics.initial_residual_normalized ≈
          norm(expected_values) / sqrt(2)
    @test metrics.initial_residual_norm_inf == 4.5

    # These are the exact fields copied into every replay CSV row.
    replay_fields = (
        initial_residual_norm2 = metrics.initial_residual_norm2,
        initial_residual_normalized = metrics.initial_residual_normalized,
        initial_residual_norm_inf = metrics.initial_residual_norm_inf,
    )
    @test replay_fields.initial_residual_norm2 ==
          norm(metrics.residual.values)
    @test replay_fields.initial_residual_normalized ==
          replay_fields.initial_residual_norm2 /
          sqrt(kkt.kkt_dimension)
    @test replay_fields.initial_residual_norm_inf ==
          norm(metrics.residual.values, Inf)

    config = SWS.preset_config(:smoke)
    @test config.reference_initialization ==
          :cold_default_each_step
    @test config.comparability_protocol ==
          :uniform_t20_dt0p1_tol0p008_max1000_v1
    @test SWS.config_dict(config)["reference_initialization"] ==
          "cold_default_each_step"
    @test SWS.config_dict(config)["comparability_protocol"] ==
          "uniform_t20_dt0p1_tol0p008_max1000_v1"
    @test SWS.config_dict(
        SWS.config_from_dict(SWS.config_dict(config)),
    ) == SWS.config_dict(config)
    legacy_comparability_config = SWS.config_dict(config)
    delete!(legacy_comparability_config, "comparability_protocol")
    @test_throws ErrorException SWS.config_from_dict(
        legacy_comparability_config,
    )
    unsupported_comparability_config = SWS.config_dict(config)
    unsupported_comparability_config["comparability_protocol"] =
        "legacy_variable_grid_v0"
    @test_throws ArgumentError SWS.config_from_dict(
        unsupported_comparability_config,
    )
    legacy_config = SWS.config_dict(config)
    delete!(legacy_config, "reference_initialization")
    @test_throws ErrorException SWS.config_from_dict(legacy_config)
    unsupported_config = SWS.config_dict(config)
    unsupported_config["reference_initialization"] =
        "all_dual_continuation"
    @test_throws ArgumentError SWS.config_from_dict(unsupported_config)
    pilot_config = SWS.preset_config(:pilot)
    full_config = SWS.preset_config(:full)
    for profile_config in (config, pilot_config, full_config)
        @test profile_config.planning_horizon == 20
        @test profile_config.Δt == 0.1
        @test profile_config.requested_reference_tol == 8e-3
        @test profile_config.reference_acceptance_tol == 8e-3
        @test profile_config.replay_tol == 8e-3
        @test profile_config.reference_max_inner_iters == 1000
        @test profile_config.replay_max_inner_iters == 1000
        options = SWS._solver_options(profile_config)
        @test options.tol == 8e-3
        @test options.max_inner_iters == 1000
        @test options.linear_solver == :klu
        @test options.linesearch == :backtracking
        @test options.η₀ == 1e-6
        @test options.η_max == 1e2
        @test options.armijo_constant == 1e-4
        @test options.reuse_factorization_iters == 0
    end
    @test SWS._solver_options_dict(config) ==
          SWS._solver_options_dict(pilot_config) ==
          SWS._solver_options_dict(full_config)
    invalid_grid_config = SWS.config_dict(config)
    invalid_grid_config["planning_horizon"] = 3
    @test_throws ArgumentError SWS.config_from_dict(invalid_grid_config)
    invalid_tolerance_config = SWS.config_dict(config)
    invalid_tolerance_config["requested_reference_tol"] = 1e-2
    @test_throws ArgumentError SWS.config_from_dict(
        invalid_tolerance_config,
    )
    disabled_warmup_config = SWS.config_dict(config)
    disabled_warmup_config["warmup"] = false
    @test_throws ArgumentError SWS.config_from_dict(
        disabled_warmup_config,
    )
    @test !hasproperty(config, :reference_fallback_enabled)
    @test SWS._config_fingerprint(config) !=
          SWS._config_fingerprint(pilot_config)
    direction = SWS._initial_state_direction(config, 101, 1).direction
    base_instance = (;
        initial_state1 = [1.0, 2.0, 3.0],
        initial_state2 = [4.0, 5.0, 6.0],
        initial_state3 = [7.0, 8.0, 9.0],
        initial_control1 = zeros(3),
        initial_control2 = zeros(3),
        initial_control3 = zeros(3),
    )
    scenario = SWS.RAC.demo_scenario_config(;
        planning_horizon = config.planning_horizon,
        Δt = config.Δt,
        use_running_goal_cost = false,
    )
    cold_step2 =
        SWS._reference_initialization(base_instance, scenario, 2)
    cold_step2_again =
        SWS._reference_initialization(base_instance, scenario, 2)
    expected_cold = SWS.RAC.build_default_warmstart(
        base_instance,
        scenario,
    ).warmstart_solution
    @test cold_step2.source == :cold_default
    @test cold_step2.source != :all_duals
    @test cold_step2.warmstart == expected_cold
    @test cold_step2.warmstart !== cold_step2_again.warmstart
    study_source = read(
        joinpath(@__DIR__, "..", "SelectiveWarmstartStudy.jl"),
        String,
    )
    @test !occursin("all_dual_continuation", study_source)
    @test !occursin("continuation_z", study_source)
    @test !occursin("max_inner_iters = 2", study_source)
    @test !occursin("_solver_options(config;", study_source)
    epsilon = 1e-2
    perturbed =
        SWS._perturb_initial_instance(base_instance, direction, epsilon)
    full_state_delta = vcat(
        perturbed.initial_state1 .- base_instance.initial_state1,
        perturbed.initial_state2 .- base_instance.initial_state2,
        perturbed.initial_state3 .- base_instance.initial_state3,
    )
    @test norm(full_state_delta) ≈ epsilon
    @test (
        perturbed.initial_state2 .- perturbed.initial_state1
    ) ≈ (
        base_instance.initial_state2 .- base_instance.initial_state1
    )
    placeholder = SWS._reference_placeholder(
        config,
        :quasi,
        101,
        2,
        :canonical_sequence_unavailable,
        "driver failed";
        sequence_driver = :reduced,
        instance = base_instance,
    )
    @test placeholder["sequence_driver"] == :reduced
    @test placeholder["instance_digest"] ==
          SWS._instance_digest(base_instance)
    @test placeholder["row"]["instance_digest"] ==
          placeholder["instance_digest"]

    csv_path = joinpath(mktempdir(), "incremental.csv")
    SWS._atomic_write(
        csv_path,
        "case_id,value\nvalid,1\ntruncated,\"unterminated",
    )
    recovered = SWS.read_csv_rows(csv_path)
    @test length(recovered) == 1
    @test recovered[1]["case_id"] == "valid"
    @test read(csv_path, String) == "case_id,value\nvalid,1\n"

    SWS._atomic_write(
        csv_path,
        "case_id,value\nbad,\"unterminated\nvalid,1\n",
    )
    @test_throws ErrorException SWS.read_csv_rows(csv_path)

    expected_head = readchomp(
        Cmd(["git", "-C", SWS.REPOSITORY_ROOT, "rev-parse", "HEAD"]),
    )
    @test SWS._git_read("rev-parse", "HEAD") == expected_head
    @test occursin(r"^[0-9a-f]{40}$", expected_head)

    function replay_row(mode, transition; valid, converged, value, churn)
        Dict{String, String}(
            "formulation" => "reduced",
            "scenario_seed" => "1",
            "transition" => string(transition),
            "mode" => mode,
            "preference_stratum" => "satisfied",
            "valid_reference_pair" => string(valid),
            "direct_converged" => string(converged),
            "solver_status" => converged ? "solved" : "failed",
            "initial_residual_normalized" => string(value),
            "initial_residual_norm2" => string(value),
            "total_inner_iters" => "2",
            "total_outer_iters" => "1",
            "first_accepted_residual_norm2" => string(value / 2),
            "first_iteration_backtracking_count" => "0",
            "regularization_change_count" => string(churn),
            "total_backtracking_count" => "0",
            "eta_retry_count" => "0",
            "full_step_fraction" => "1",
            "first_accepted_alpha" => "1",
            "solve_time_sec" => "0.01",
            "direct_final_residual_norm2" => "1e-4",
        )
    end
    paired_input = Dict{String, String}[]
    for (transition, valid, converged_a, converged_b) in (
        (1, false, false, false),
        (2, true, false, true),
        (3, true, true, true),
    )
        push!(
            paired_input,
            replay_row(
                "all_except_innermost_stationarity",
                transition;
                valid,
                converged = converged_a,
                value = 1.0,
                churn = 3,
            ),
        )
        push!(
            paired_input,
            replay_row(
                "equality_duals",
                transition;
                valid,
                converged = converged_b,
                value = 2.0,
                churn = 1,
            ),
        )
    end
    paired_summary = SWS._paired_summary_rows(paired_input, config)
    initial_summary = only(filter(
        row ->
            row["formulation"] == :reduced &&
            row["comparison"] == "all_except_vs_equality" &&
            row["preference_stratum"] == "all" &&
            row["metric"] == "initial_residual_normalized",
        paired_summary,
    ))
    @test initial_summary["mode_a_failures"] == 1
    @test initial_summary["mode_b_failures"] == 0
    churn_summary = only(filter(
        row ->
            row["formulation"] == :reduced &&
            row["comparison"] == "all_except_vs_equality" &&
            row["preference_stratum"] == "all" &&
            row["metric"] == "regularization_change_count",
        paired_summary,
    ))
    @test startswith(
        churn_summary["directional_scoring"],
        "none_",
    )
    @test churn_summary["wins_a"] == ""
    @test churn_summary["losses_a"] == ""

    scaling_input = Dict{String, String}[]
    for epsilon in (1e-4, 1e-3, 1e-2, 1e-1)
        for (mode, baseline, R0, r0) in (
            ("all_duals", 1e-3, 1e-3, 1e-4),
            (
                "all_except_innermost_stationarity",
                1e-1,
                1.01e-1,
                1.01e-2,
            ),
        )
            push!(
                scaling_input,
                Dict{String, String}(
                    "formulation" => "reduced",
                    "scenario_seed" => "1",
                    "direction" => "1",
                    "mode" => mode,
                    "epsilon" => string(epsilon),
                    "baseline_initial_residual_norm2" =>
                        string(baseline),
                    "initial_residual_norm2" => string(R0),
                    "initial_residual_normalized" => string(r0),
                    "perturbed_reference_residual" => "1e-3",
                ),
            )
        end
    end
    slopes = SWS._scaling_slope_rows(scaling_input, config)
    structural = only(filter(
        row ->
            row["formulation"] == "reduced" &&
            row["mode"] ==
            "all_except_innermost_stationarity",
        slopes,
    ))
    @test structural["structural_floor_detected"] == true
    @test structural["fit_performed"] == true
    @test abs(structural["slope_log_r0_vs_log_epsilon"]) < 1e-12

    report_run = mktempdir()
    report_raw = joinpath(report_run, "raw")
    mkpath(report_raw)
    SWS._write_rows(
        joinpath(report_raw, "replay.csv"),
        [
            "case_id",
            "formulation",
            "scenario_seed",
            "transition",
            "valid_reference_pair",
        ],
        [
            Dict(
                "case_id" => "r1",
                "formulation" => "reduced",
                "scenario_seed" => 101,
                "transition" => 1,
                "valid_reference_pair" => true,
            ),
            Dict(
                "case_id" => "r2",
                "formulation" => "reduced",
                "scenario_seed" => 101,
                "transition" => 2,
                "valid_reference_pair" => true,
            ),
            Dict(
                "case_id" => "q1",
                "formulation" => "quasi",
                "scenario_seed" => 101,
                "transition" => 1,
                "valid_reference_pair" => false,
            ),
            Dict(
                "case_id" => "q2",
                "formulation" => "quasi",
                "scenario_seed" => 101,
                "transition" => 2,
                "valid_reference_pair" => true,
            ),
        ],
    )
    SWS._write_rows(
        joinpath(report_raw, "root_spread.csv"),
        [
            "case_id",
            "formulation",
            "scenario_seed",
            "transition",
            "converged_mode_count",
            "max_pairwise_primal_distance_normalized",
            "materially_different_roots",
        ],
        [
            Dict(
                "case_id" => "root_r1",
                "formulation" => "reduced",
                "scenario_seed" => 101,
                "transition" => 1,
                "converged_mode_count" => 2,
                "max_pairwise_primal_distance_normalized" => 0.25,
                "materially_different_roots" => true,
            ),
            Dict(
                "case_id" => "root_r2",
                "formulation" => "reduced",
                "scenario_seed" => 101,
                "transition" => 2,
                "converged_mode_count" => 1,
                "max_pairwise_primal_distance_normalized" => 0.0,
                "materially_different_roots" => false,
            ),
            Dict(
                "case_id" => "root_q1",
                "formulation" => "quasi",
                "scenario_seed" => 101,
                "transition" => 1,
                "converged_mode_count" => 0,
                "max_pairwise_primal_distance_normalized" => 0.0,
                "materially_different_roots" => false,
            ),
            Dict(
                "case_id" => "root_q2",
                "formulation" => "quasi",
                "scenario_seed" => 101,
                "transition" => 2,
                "converged_mode_count" => 3,
                "max_pairwise_primal_distance_normalized" => 5e-4,
                "materially_different_roots" => false,
            ),
        ],
    )
    function scaling_report_row(
        case_id,
        formulation,
        epsilon,
        mode;
        reference_status,
        reference_accepted,
        valid,
        solver_status,
        converged,
    )
        Dict(
            "case_id" => case_id,
            "formulation" => formulation,
            "scenario_seed" => 101,
            "direction" => 1,
            "epsilon" => epsilon,
            "mode" => mode,
            "perturbed_reference_status" => reference_status,
            "perturbed_reference_accepted" => reference_accepted,
            "valid_reference_pair" => valid,
            "solver_status" => solver_status,
            "direct_converged" => converged,
        )
    end
    scaling_report_rows = [
        scaling_report_row(
            "sr1a",
            "reduced",
            1e-3,
            "all_except_innermost_stationarity";
            reference_status = "solved",
            reference_accepted = true,
            valid = true,
            solver_status = "solved",
            converged = true,
        ),
        scaling_report_row(
            "sr1b",
            "reduced",
            1e-3,
            "all_duals";
            reference_status = "solved",
            reference_accepted = true,
            valid = true,
            solver_status = "failed",
            converged = false,
        ),
        scaling_report_row(
            "sr2a",
            "reduced",
            1e-2,
            "all_except_innermost_stationarity";
            reference_status = "failed",
            reference_accepted = false,
            valid = false,
            solver_status = "not_run_reference_unavailable",
            converged = false,
        ),
        scaling_report_row(
            "sr2b",
            "reduced",
            1e-2,
            "all_duals";
            reference_status = "failed",
            reference_accepted = false,
            valid = false,
            solver_status = "not_run_reference_unavailable",
            converged = false,
        ),
        scaling_report_row(
            "sq1",
            "quasi",
            1e-3,
            "primal_only";
            reference_status = "not_run_base_reference_unavailable",
            reference_accepted = false,
            valid = false,
            solver_status = "not_run_reference_unavailable",
            converged = false,
        ),
        scaling_report_row(
            "sq2a",
            "quasi",
            1e-2,
            "primal_only";
            reference_status = "solved",
            reference_accepted = true,
            valid = true,
            solver_status = "not_run_configured",
            converged = false,
        ),
        scaling_report_row(
            "sq2b",
            "quasi",
            1e-2,
            "equality_duals";
            reference_status = "solved",
            reference_accepted = true,
            valid = true,
            solver_status = "solved",
            converged = true,
        ),
    ]
    SWS._write_rows(
        joinpath(report_raw, "scaling.csv"),
        [
            "case_id",
            "formulation",
            "scenario_seed",
            "direction",
            "epsilon",
            "mode",
            "perturbed_reference_status",
            "perturbed_reference_accepted",
            "valid_reference_pair",
            "solver_status",
            "direct_converged",
        ],
        scaling_report_rows,
    )
    root_summary =
        SWS._root_evaluability_summary(report_run, config)
    @test root_summary.planned == 4
    @test root_summary.recorded == 4
    @test root_summary.evaluable == 2
    @test root_summary.flagged == 1
    @test root_summary.invalid_reference_count == 1
    @test root_summary.insufficient_mode_count == 1
    @test root_summary.maximum_spread == 0.25

    scaling_summary =
        SWS._scaling_failure_summary(report_run, config)
    @test scaling_summary.reference_totals == (
        cases = 4,
        attempts = 3,
        accepted = 2,
        failed = 1,
        not_attempted = 1,
    )
    @test scaling_summary.mode_totals == (
        valid_reference_rows = 4,
        solve_attempts = 3,
        direct_converged = 2,
        failed = 1,
        valid_not_run = 1,
    )
    failure_io = IOBuffer()
    SWS._report_failures_and_roots(
        failure_io,
        report_run,
        config,
    )
    failure_report = String(take!(failure_io))
    @test occursin("**1/2** evaluable transitions", failure_report)
    @test occursin(
        "**1** lacked a valid source/destination reference pair",
        failure_report,
    )
    @test occursin(
        "**1** had valid references but fewer than two converged modes",
        failure_report,
    )
    @test occursin(
        "**3** perturbed references were attempted",
        failure_report,
    )
    @test occursin(
        "**2** direct-converged and **1** failed",
        failure_report,
    )

    accuracy_run = mktempdir()
    mkpath(joinpath(accuracy_run, "raw"))
    SWS._atomic_write(
        joinpath(accuracy_run, "raw", "references.csv"),
        "case_id,reference_accepted,direct_residual_norm2\nr1,true,0.0079\nr2,true,0.007\n",
    )
    SWS._atomic_write(
        joinpath(accuracy_run, "raw", "replay.csv"),
        "case_id,formulation,scenario_seed,transition,valid_reference_pair,source_reference_residual,destination_reference_residual\np1a,reduced,1,1,true,0.007,0.0079\np1b,reduced,1,1,true,0.007,0.0079\n",
    )
    SWS._atomic_write(
        joinpath(accuracy_run, "raw", "scaling.csv"),
        "case_id,formulation,scenario_seed,direction,epsilon,perturbed_reference_accepted,perturbed_reference_residual\ns1a,reduced,1,1,0.001,true,0.0075\ns1b,reduced,1,1,0.001,true,0.0075\n",
    )
    accuracy =
        SWS._reference_accuracy_summary(accuracy_run, pilot_config)
    @test accuracy["canonical_reference_protocol"] ==
          "uniform_comparability_tolerance"
    @test accuracy["accepted_canonical_references"] == 2
    @test accuracy[
        "accepted_canonical_references_above_replay_tol"
    ] == 0
    @test accuracy[
        "valid_replay_pairs_with_source_or_destination_above_replay_tol"
    ] == 0
    @test accuracy[
        "accepted_perturbed_scaling_references_above_replay_tol"
    ] == 0
    pilot_manifest = SWS._study_manifest(
        pilot_config,
        Dict{String, Any}[];
        reference_accuracy_summary = accuracy,
    )
    @test pilot_manifest["study"]["comparability_protocol"] ==
          "uniform_t20_dt0p1_tol0p008_max1000_v1"
    @test occursin(
        "not higher-accuracy ground truth",
        pilot_manifest["study"]["canonical_reference_protocol"],
    )
    @test occursin(
        "717 iterations",
        pilot_manifest["study"]["baseline_choice"],
    )
    @test occursin(
        "0.007991129782359604",
        pilot_manifest["study"]["baseline_choice"],
    )
    @test pilot_manifest["study"]["reference_initialization"] ==
          "cold_default_each_step"
    @test occursin(
        "fresh destination-specific cold/default solve",
        pilot_manifest["study"]["reference_initialization_rule"],
    )
    @test occursin(
        "fresh run is required",
        pilot_manifest["study"][
            "reference_initialization_change_disclosure"
        ],
    )
    @test pilot_manifest["study"]["uniform_solver_options"][
        "tol"
    ] == 8e-3
    @test pilot_manifest["study"]["uniform_solver_options"][
        "max_inner_iters"
    ] == 1000
    @test !haskey(
        pilot_manifest["study"],
        "reference_solver_options",
    )
    @test !haskey(pilot_manifest["study"], "replay_solver_options")
    @test pilot_manifest["study"]["reference_accuracy_counts"][
        "accepted_canonical_references_above_replay_tol"
    ] == 0
    uniform_report =
        SWS.generate_report(accuracy_run; config = pilot_config)
    uniform_report_text = read(uniform_report, String)
    @test !occursin("Relaxed-oracle caveat", uniform_report_text)
    @test occursin(
        "uniform_t20_dt0p1_tol0p008_max1000_v1",
        uniform_report_text,
    )
    @test occursin(
        "All solver calls",
        uniform_report_text,
    )
    @test occursin("717 iterations", uniform_report_text)
    @test occursin(
        "0.007991129782359604",
        uniform_report_text,
    )
    @test occursin(
        "Reference-initialization disclosure",
        uniform_report_text,
    )
    @test occursin(
        "reference_initialization = \"cold_default_each_step\"",
        uniform_report_text,
    )
    @test occursin(
        "not higher-accuracy ground truth",
        uniform_report_text,
    )

    provenance_run = mktempdir()
    provenance =
        SWS._snapshot_environment_and_provenance!(provenance_run)
    @test isfile(
        joinpath(provenance_run, "environment", "Project.toml"),
    )
    @test isfile(
        joinpath(provenance_run, "environment", "Manifest.toml"),
    )
    @test isfile(
        joinpath(
            provenance_run,
            "provenance",
            "git_diff_binary.patch",
        ),
    )
    @test occursin(
        r"^[0-9a-f]{64}$",
        provenance["measurement_code_fingerprint"],
    )
    drift = SWS._record_provenance_drift!(provenance_run)
    @test drift["measurement_code_drift_detected"] == false
    @test drift["environment_drift_detected"] == false

    legacy_io = IOBuffer()
    SWS._report_provenance(
        legacy_io,
        mktempdir(),
        Dict{String, Any}(),
    )
    legacy_text = String(take!(legacy_io))
    @test occursin("Legacy-run provenance warning", legacy_text)
    @test !occursin("provenance/manifest.toml", legacy_text)
end
