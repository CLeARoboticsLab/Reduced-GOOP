const HOLDOUT_GAMMAS = (0.0, HELDOUT_GAMMA, 1.0)
const HOLDOUT_CHECKPOINT_SCHEMA = "heldout_gamma_0p25_v1"
const DEVELOPMENT_GAMMA_REQUIRED_SUCCESSES = 6
const HOLDOUT_GAMMA_NEAR_ALL_REQUIRED_SUCCESSES = 10

_case_key(pair) = (pair.form, pair.seed, pair.transition)
_sample_group(pair) = _case_key(pair) in DEVELOPMENT_CASES ? :development : :holdout

function _replay_input_path(run_dir, pair, gamma)
    label = gamma == 0.0 ? "gamma_0" :
            gamma == 1.0 ? "gamma_1" :
            error("Only frozen endpoint replay checkpoints may be reused.")
    joinpath(
        run_dir,
        "inputs",
        "dual_transport",
        "checkpoints",
        "replay",
        String(pair.form),
        "seed_$(pair.seed)",
        "transition_$(pair.transition)__$(label).jld2",
    )
end

function _development_input_path(run_dir, pair, gamma)
    source_name = basename(
        DTS._damping_path(
            run_dir,
            pair.form,
            pair.seed,
            pair.transition,
            gamma,
        ),
    )
    joinpath(
        run_dir,
        "inputs",
        "dual_transport",
        "checkpoints",
        "development_damping",
        String(pair.form),
        "seed_$(pair.seed)",
        source_name,
    )
end

function _heldout_checkpoint_path(run_dir, pair)
    joinpath(
        run_dir,
        "checkpoints",
        "heldout_gamma",
        String(pair.form),
        "seed_$(pair.seed)",
        "transition_$(pair.transition)__gamma_0p25.jld2",
    )
end

_row_value(row, key, default = missing) = get(row, key, default)

function _normalized_candidate(
    pair,
    gamma,
    checkpoint,
    source_kind,
    source_path,
)
    row = checkpoint["row"]
    haskey(checkpoint, "final_primal") ||
        error("Frozen candidate has no final primal: $(source_path)")
    converged = Bool(_row_value(row, "direct_converged", false))
    Dict{String, Any}(
        "case_id" =>
            "$(pair.form)__seed$(pair.seed)__t$(pair.transition)__gamma$(gamma)",
        "formulation" => pair.form,
        "scenario_seed" => pair.seed,
        "transition" => pair.transition,
        "sample_group" => _sample_group(pair),
        "gamma" => gamma,
        "source_kind" => source_kind,
        "source_path" => abspath(source_path),
        "source_sha256" => _sha256(source_path),
        "direct_converged" => converged,
        "solver_status" => _row_value(row, "solver_status", :unknown),
        "total_inner_iters" => Int(_row_value(row, "total_inner_iters", 0)),
        "total_backtracking_count" =>
            Int(_row_value(row, "total_backtracking_count", 0)),
        "eta_retry_count" => Int(_row_value(row, "eta_retry_count", 0)),
        "klu_singular_retries" =>
            Int(_row_value(row, "klu_singular_retries", 0)),
        "svd_fallback_count" =>
            Int(_row_value(row, "svd_fallback_count", 0)),
        "direct_final_residual_norm2" =>
            Float64(_row_value(row, "direct_final_residual_norm2", Inf)),
        "direct_final_residual_norm_inf" =>
            Float64(_row_value(row, "direct_final_residual_norm_inf", Inf)),
        "initial_residual_norm2" =>
            Float64(_row_value(row, "initial_residual_norm2", NaN)),
        "first_accepted_alpha" =>
            Float64(_row_value(row, "first_accepted_alpha", NaN)),
        "full_step_fraction" =>
            Float64(_row_value(row, "full_step_fraction", NaN)),
        "failure" => _row_value(row, "failure", ""),
        "final_primal" => copy(checkpoint["final_primal"]),
    )
end

function _assert_development_endpoint_matches_replay!(
    run_dir,
    pair,
    gamma,
    development,
)
    replay_path = _replay_input_path(run_dir, pair, gamma)
    replay = JLD2.load_object(replay_path)
    replay_row = replay["row"]
    development_row = development["row"]
    for key in (
        "initial_residual_norm2",
        "direct_final_residual_norm2",
        "total_inner_iters",
        "total_backtracking_count",
        "eta_retry_count",
        "direct_converged",
        "solver_status",
    )
        isequal(replay_row[key], development_row[key]) ||
            error(
                "Development damping endpoint $(key) differs from its frozen " *
                "replay endpoint for $(_case_key(pair)), gamma=$(gamma).",
            )
    end
    replay["final_primal"] == development["final_primal"] ||
        error("Development endpoint final primal differs from frozen replay.")
    nothing
end

function _trace_value(event, name, default)
    isnothing(event) ? default : get(event, name, default)
end

function _solve_heldout_gamma!(
    run_dir,
    pair,
    built,
    options,
    config::TheoryResolutionConfig,
)
    path = _heldout_checkpoint_path(run_dir, pair)
    _sample_group(pair) === :holdout ||
        error("Attempted a new gamma solve on a development case.")

    warmstart = DTS._build_warmstart(
        pair.source["shifted_primal"],
        pair.source["z"],
        built.kkt,
        :all_duals,
        :stage_shift_zero_tail,
    )
    warmstart[built.blocks.ψ_in] .*= config.heldout_gamma
    if isfile(path)
        checkpoint = JLD2.load_object(path)
        get(checkpoint, "schema_version", "") == HOLDOUT_CHECKPOINT_SCHEMA ||
            error("Held-out checkpoint schema mismatch: $(path)")
        get(checkpoint, "protocol", "") == String(PROTOCOL) ||
            error("Held-out checkpoint protocol mismatch: $(path)")
        row = checkpoint["row"]
        Symbol(string(row["formulation"])) === pair.form ||
            error("Held-out checkpoint formulation mismatch: $(path)")
        Int(row["scenario_seed"]) == pair.seed ||
            error("Held-out checkpoint seed mismatch: $(path)")
        Int(row["transition"]) == pair.transition ||
            error("Held-out checkpoint transition mismatch: $(path)")
        Float64(row["gamma"]) == config.heldout_gamma ||
            error("Held-out checkpoint gamma mismatch: $(path)")
        Symbol(string(row["dual_transport"])) === :stage_shift_zero_tail ||
            error("Held-out checkpoint transport mismatch: $(path)")
        checkpoint["warmstart"] == warmstart ||
            error("Held-out checkpoint warm start drifted: $(path)")
        checkpoint["source_reference_hash"] == _sha256(pair.source_path) ||
            error("Held-out checkpoint source reference drifted: $(path)")
        checkpoint["destination_reference_hash"] ==
            _sha256(pair.destination_path) ||
            error("Held-out checkpoint destination reference drifted: $(path)")
        checkpoint["solver_options_hash"] ==
            _sha256(joinpath(config.source_run, "solver_options.toml")) ||
            error("Held-out checkpoint solver options drifted: $(path)")
        return checkpoint
    end
    θ = pair.destination["parameters"].θ
    initial = DTS._residual_metrics(built.kkt, warmstart, θ)
    events = Any[]
    output = nothing
    elapsed = @elapsed output = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        built.kkt,
        θ;
        z₀ = copy(warmstart),
        options,
        trace_hook = event -> push!(events, event),
    )
    trace = SWS._trace_summary(events)
    direct = SWS._residual(
        built.kkt,
        output.z,
        θ;
        epsilon = output.ϵ,
    )
    converged = isfinite(direct.norm2) && direct.norm2 <= config.tol
    finish = trace.finish
    fallback_count = Int(_trace_value(
        finish,
        :svd_fallback_count,
        output.svd_fallback_count,
    ))
    fallback_count == 0 ||
        error(
            "The frozen KLU path invoked an alternate dense fallback for " *
            "$(_case_key(pair)); this study forbids alternate solvers.",
        )
    row = Dict{String, Any}(
        "case_id" =>
            "$(pair.form)__seed$(pair.seed)__t$(pair.transition)__gamma0.25",
        "formulation" => pair.form,
        "scenario_seed" => pair.seed,
        "transition" => pair.transition,
        "sample_group" => :holdout,
        "gamma" => config.heldout_gamma,
        "dual_transport" => :stage_shift_zero_tail,
        "direct_converged" => converged,
        "solver_status" => output.status,
        "total_inner_iters" =>
            Int(_trace_value(finish, :total_iters, output.total_iters)),
        "total_backtracking_count" =>
            Int(_trace_value(finish, :total_backtracking_count, 0)),
        "eta_retry_count" =>
            Int(_trace_value(finish, :total_eta_retry_count, 0)),
        "klu_singular_retries" =>
            Int(_trace_value(finish, :klu_singular_retries, output.klu_singular_retries)),
        "svd_fallback_count" => fallback_count,
        "direct_final_residual_norm2" => direct.norm2,
        "direct_final_residual_norm_inf" => direct.norm_inf,
        "reported_final_residual_norm2" => output.kkt_error,
        "initial_residual_norm2" => initial.residual.norm2,
        "initial_residual_normalized" => initial.residual.normalized,
        "initial_outer_stationarity_norm2" => initial.outer,
        "initial_innermost_stationarity_norm2" => initial.inner,
        "initial_equality_norm2" => initial.equality,
        "first_accepted_alpha" =>
            Float64(_trace_value(trace.first_step, :accepted_alpha, NaN)),
        "full_step_fraction" =>
            Float64(_trace_value(finish, :full_step_fraction, NaN)),
        "solve_time_sec" => elapsed,
        "failure" => converged ? "" : :residual_above_tolerance,
    )
    checkpoint = Dict{String, Any}(
        "schema_version" => HOLDOUT_CHECKPOINT_SCHEMA,
        "protocol" => String(PROTOCOL),
        "row" => row,
        "warmstart" => warmstart,
        "events" => events,
        "converged" => converged,
        "final_primal" => copy(output.z[built.blocks.z]),
        "final_z" => copy(output.z),
        "source_reference_paths" => Dict(
            "source" => pair.source_path,
            "destination" => pair.destination_path,
        ),
        "source_reference_hash" => _sha256(pair.source_path),
        "destination_reference_hash" => _sha256(pair.destination_path),
        "solver_options_hash" =>
            _sha256(joinpath(config.source_run, "solver_options.toml")),
    )
    _atomic_save(path, checkpoint)
    checkpoint
end

function _candidate_records!(run_dir, config, pairs, cache)
    source_config = _source_config(config)
    options = _source_solver_options(config)
    records = Dict{Tuple, Dict{String, Any}}()
    for pair in pairs
        for gamma in HOLDOUT_GAMMAS
            checkpoint, source_kind, source_path = if _sample_group(pair) === :development
                path = _development_input_path(run_dir, pair, gamma)
                loaded = JLD2.load_object(path)
                if gamma in (0.0, 1.0)
                    _assert_development_endpoint_matches_replay!(
                        run_dir,
                        pair,
                        gamma,
                        loaded,
                    )
                end
                (loaded, :reused_development_damping, path)
            elseif gamma in (0.0, 1.0)
                path = _replay_input_path(run_dir, pair, gamma)
                (JLD2.load_object(path), :reused_replay_endpoint, path)
            else
                built = DTS._build_system(source_config, pair.form, cache)
                path = _heldout_checkpoint_path(run_dir, pair)
                (
                    _solve_heldout_gamma!(
                        run_dir,
                        pair,
                        built,
                        options,
                        config,
                    ),
                    :new_preregistered_holdout_solve,
                    path,
                )
            end
            candidate = _normalized_candidate(
                pair,
                gamma,
                checkpoint,
                source_kind,
                source_path,
            )
            candidate["svd_fallback_count"] == 0 ||
                error("A reused candidate used an alternate SVD fallback.")
            records[(pair.form, pair.seed, pair.transition, gamma)] = candidate
        end
    end
    records
end

function _final_primal_distances(records, pairs, alignment_tol)
    rows = Dict{String, Any}[]
    case_summary = Dict{Tuple, NamedTuple}()
    for pair in pairs
        key = _case_key(pair)
        candidates = Dict(
            gamma => records[(key..., gamma)] for gamma in HOLDOUT_GAMMAS
        )
        distances = Float64[]
        for i in eachindex(HOLDOUT_GAMMAS)
            for j in (i+1):length(HOLDOUT_GAMMAS)
                gamma_a = HOLDOUT_GAMMAS[i]
                gamma_b = HOLDOUT_GAMMAS[j]
                primal_a = candidates[gamma_a]["final_primal"]
                primal_b = candidates[gamma_b]["final_primal"]
                distance = norm(primal_a .- primal_b)
                normalized =
                    distance / max(1.0, norm(primal_a), norm(primal_b))
                push!(distances, normalized)
                push!(
                    rows,
                    Dict{String, Any}(
                        "case_id" =>
                            "$(pair.form)__seed$(pair.seed)__t$(pair.transition)__gamma$(gamma_a)__gamma$(gamma_b)",
                        "formulation" => pair.form,
                        "scenario_seed" => pair.seed,
                        "transition" => pair.transition,
                        "sample_group" => _sample_group(pair),
                        "gamma_a" => gamma_a,
                        "gamma_b" => gamma_b,
                        "distance" => distance,
                        "normalized_distance" => normalized,
                        "aligned_at_threshold" => normalized <= alignment_tol,
                        "alignment_threshold" => alignment_tol,
                    ),
                )
            end
        end
        joint = all(gamma -> candidates[gamma]["direct_converged"], HOLDOUT_GAMMAS)
        max_distance = maximum(distances)
        case_summary[key] = (;
            jointly_converged = joint,
            max_normalized_separation = max_distance,
            aligned_candidate_set = joint && max_distance <= alignment_tol,
        )
    end
    (; rows, case_summary)
end

function _csv_candidate(candidate, case_info)
    row = Dict{String, Any}(
        key => value for (key, value) in candidate if key != "final_primal"
    )
    row["jointly_converged"] = case_info.jointly_converged
    row["max_final_primal_separation_normalized"] =
        case_info.max_normalized_separation
    row["aligned_candidate_set"] = case_info.aligned_candidate_set
    row["alignment_threshold"] = ALIGNMENT_TOL
    row
end

function _mean_or_nan(values)
    isempty(values) ? NaN : sum(values) / length(values)
end

function _median_or_nan(values)
    isempty(values) && return NaN
    ordered = sort!(Float64.(values))
    n = length(ordered)
    isodd(n) ? ordered[(n+1)÷2] :
    0.5 * (ordered[n÷2] + ordered[n÷2+1])
end

function _summary_rows(records, pairs, case_info)
    rows = Dict{String, Any}[]
    group_specs = Tuple{Symbol, Union{Symbol, Nothing}}[]
    for sample_group in (:development, :holdout, :all)
        for form in (:reduced, :quasi, :all)
            push!(
                group_specs,
                (
                    sample_group,
                    form === :all ? nothing : form,
                ),
            )
        end
    end
    for (sample_group, form_filter) in group_specs
        selected = filter(pairs) do pair
            sample_ok =
                sample_group === :all || _sample_group(pair) === sample_group
            form_ok = isnothing(form_filter) || pair.form === form_filter
            sample_ok && form_ok
        end
        isempty(selected) && continue
        joint_pairs = filter(
            pair -> case_info[_case_key(pair)].jointly_converged,
            selected,
        )
        aligned_pairs = filter(
            pair -> case_info[_case_key(pair)].aligned_candidate_set,
            selected,
        )
        separations = [
            case_info[_case_key(pair)].max_normalized_separation for pair in selected
        ]
        for gamma in HOLDOUT_GAMMAS
            candidates = [
                records[(_case_key(pair)..., gamma)] for pair in selected
            ]
            joint_candidates = [
                records[(_case_key(pair)..., gamma)] for pair in joint_pairs
            ]
            aligned_candidates = [
                records[(_case_key(pair)..., gamma)] for pair in aligned_pairs
            ]
            field_values(values, field) =
                Float64[candidate[field] for candidate in values]
            push!(
                rows,
                Dict{String, Any}(
                    "case_id" =>
                        "$(sample_group)__$(isnothing(form_filter) ? "all" : form_filter)__gamma$(gamma)",
                    "sample_group" => sample_group,
                    "formulation" =>
                        isnothing(form_filter) ? :all : form_filter,
                    "gamma" => gamma,
                    "total_cases" => length(selected),
                    "converged_cases" =>
                        count(candidate -> candidate["direct_converged"], candidates),
                    "convergence_rate" =>
                        count(candidate -> candidate["direct_converged"], candidates) /
                        length(selected),
                    "jointly_converged_cases" => length(joint_pairs),
                    "aligned_joint_cases" => length(aligned_pairs),
                    "median_iterations_joint" => _median_or_nan(
                        field_values(joint_candidates, "total_inner_iters"),
                    ),
                    "mean_iterations_joint" => _mean_or_nan(
                        field_values(joint_candidates, "total_inner_iters"),
                    ),
                    "median_iterations_aligned" => _median_or_nan(
                        field_values(aligned_candidates, "total_inner_iters"),
                    ),
                    "mean_iterations_aligned" => _mean_or_nan(
                        field_values(aligned_candidates, "total_inner_iters"),
                    ),
                    "median_backtracking_joint" => _median_or_nan(
                        field_values(joint_candidates, "total_backtracking_count"),
                    ),
                    "mean_backtracking_joint" => _mean_or_nan(
                        field_values(joint_candidates, "total_backtracking_count"),
                    ),
                    "median_backtracking_aligned" => _median_or_nan(
                        field_values(
                            aligned_candidates,
                            "total_backtracking_count",
                        ),
                    ),
                    "mean_backtracking_aligned" => _mean_or_nan(
                        field_values(
                            aligned_candidates,
                            "total_backtracking_count",
                        ),
                    ),
                    "median_eta_retries_joint" => _median_or_nan(
                        field_values(joint_candidates, "eta_retry_count"),
                    ),
                    "mean_eta_retries_joint" => _mean_or_nan(
                        field_values(joint_candidates, "eta_retry_count"),
                    ),
                    "median_eta_retries_aligned" => _median_or_nan(
                        field_values(aligned_candidates, "eta_retry_count"),
                    ),
                    "mean_eta_retries_aligned" => _mean_or_nan(
                        field_values(aligned_candidates, "eta_retry_count"),
                    ),
                    "median_klu_retries_joint" => _median_or_nan(
                        field_values(joint_candidates, "klu_singular_retries"),
                    ),
                    "mean_klu_retries_joint" => _mean_or_nan(
                        field_values(joint_candidates, "klu_singular_retries"),
                    ),
                    "median_klu_retries_aligned" => _median_or_nan(
                        field_values(
                            aligned_candidates,
                            "klu_singular_retries",
                        ),
                    ),
                    "mean_klu_retries_aligned" => _mean_or_nan(
                        field_values(
                            aligned_candidates,
                            "klu_singular_retries",
                        ),
                    ),
                    "median_final_residual" => _median_or_nan(
                        field_values(candidates, "direct_final_residual_norm2"),
                    ),
                    "max_final_residual" => maximum(
                        field_values(candidates, "direct_final_residual_norm2"),
                    ),
                    "median_case_max_final_primal_separation" =>
                        _median_or_nan(separations),
                    "max_case_final_primal_separation" => maximum(separations),
                    "alignment_threshold" => ALIGNMENT_TOL,
                ),
            )
        end
    end
    rows
end

function run_holdout_validation!(
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
    length(pairs) == 17 || error("Expected 17 snapshotted valid pairs.")
    count(pair -> _sample_group(pair) === :development, pairs) == 6 ||
        error("Expected six preregistered development cases.")
    count(pair -> _sample_group(pair) === :holdout, pairs) == 11 ||
        error("Expected eleven held-out cases.")

    records = _candidate_records!(run_dir, config, pairs, cache)
    length(records) == 51 || error("Expected 51 endpoint/safeguard candidates.")
    distances = _final_primal_distances(records, pairs, config.alignment_tol)
    candidate_rows = [
        _csv_candidate(
            records[(_case_key(pair)..., gamma)],
            distances.case_summary[_case_key(pair)],
        ) for pair in pairs for gamma in HOLDOUT_GAMMAS
    ]
    summary_rows = _summary_rows(records, pairs, distances.case_summary)
    DTS.write_csv(
        joinpath(run_dir, "raw", "gamma_candidates.csv"),
        candidate_rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "gamma_final_primal_distances.csv"),
        distances.rows,
    )
    DTS.write_csv(
        joinpath(run_dir, "raw", "gamma_group_summary.csv"),
        summary_rows,
    )
    aggregate = Dict{String, Any}(
        "candidate_rows" => candidate_rows,
        "distance_rows" => distances.rows,
        "summary_rows" => summary_rows,
        "case_summary" => distances.case_summary,
    )
    _atomic_save(
        joinpath(run_dir, "checkpoints", "holdout_validation_summary.jld2"),
        aggregate,
    )
    _atomic_write(joinpath(run_dir, "holdout_complete"), "ok\n")
    aggregate
end

analyze_holdout_validation!(run_dir, config::TheoryResolutionConfig) =
    JLD2.load_object(
        joinpath(run_dir, "checkpoints", "holdout_validation_summary.jld2"),
    )
