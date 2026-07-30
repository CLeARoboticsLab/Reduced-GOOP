module StagewiseDiagnostics

using JLD2: JLD2
using LinearAlgebra: norm
using SHA: SHA

include(
    normpath(
        joinpath(
            @__DIR__,
            "..",
            "dual_transport",
            "DualTransportStudy.jl",
        ),
    ),
)

const DTS = DualTransportStudy
const ReducedGOOP = DTS.ReducedGOOP

export run_stagewise_diagnostics, validate_stagewise_diagnostics

const SCHEMA_VERSION = "stagewise_diagnostics_v1"
const HORIZON_ROLES = (
    :inherited_interior,
    :initial_boundary,
    :terminal_boundary,
    :global,
)
const EXPECTED_ROW_COUNTS = Dict(
    :inherited_interior => 1_220,
    :initial_boundary => 78,
    :terminal_boundary => 131,
    :global => 0,
)
const DUAL_FACTORS = (
    :inherited_successor,
    :initial_condition,
    :missing_successor_tail,
    :explicit_terminal_reset,
    :global_identity,
)
const EXPECTED_DUAL_FACTOR_COUNTS = Dict(
    :inherited_successor => 2_136,
    :initial_condition => 60,
    :missing_successor_tail => 75,
    :explicit_terminal_reset => 39,
    :global_identity => 0,
)
const PRODUCTION_TRANSPORTS = (
    :identity_copy,
    :stage_shift_zero_tail,
    :stage_shift_hold_tail,
)
const INTERIOR_MAPS = (:fixed_index, :semantic_shift)
const INITIAL_COMPLETIONS = (:copy, :reset)
const TERMINAL_COMPLETIONS = (:hold, :zero)
const EXPLICIT_COMPLETIONS = (:copy, :reset)

_field(value, name, default = nothing) =
    hasproperty(value, name) ? getproperty(value, name) : default

function _source_config(source_run)
    source_run = abspath(source_run)
    config_path = joinpath(source_run, "config.toml")
    isfile(config_path) ||
        error("Dual-transport source config is missing: $(config_path)")
    config = DTS.load_config(config_path)
    DTS.validate_protocol(config)
    config
end

function _metadata_signature(kkt)
    lines = String[]
    for row in sort(kkt.metadata.equations; by = coordinate -> coordinate.row)
        push!(
            lines,
            join(
                string.(
                    (
                        row.row,
                        row.family,
                        row.scope,
                        row.player,
                        row.level,
                        row.equation_class,
                        row.equation_type,
                        row.primal_variable,
                        row.stage,
                        row.successor_stage,
                        row.component,
                        row.horizon_role,
                        row.exact_shift_invariant,
                    ),
                ),
                "|",
            ),
        )
    end
    bytes2hex(SHA.sha256(join(lines, "\n")))
end

function _equation_key(row; stage = row.stage)
    (
        row.family,
        row.scope,
        row.player,
        row.level,
        row.equation_class,
        row.equation_type,
        row.primal_variable,
        row.component,
        stage,
    )
end

"""
Map every destination inherited-interior residual row to the semantically
identical source row one physical stage later. The source row can itself carry
a source-horizon boundary label; row identity is determined by equation
semantics and stage, not by the direction-dependent horizon role.
"""
function _residual_shift_map(kkt)
    rows = kkt.metadata.equations
    lookup = Dict{Tuple, Any}()
    for row in rows
        key = _equation_key(row)
        haskey(lookup, key) &&
            error("Residual metadata is not unique for semantic key $(key).")
        lookup[key] = row
    end

    source_rows = Int[]
    destination_rows = Int[]
    for destination in sort(rows; by = row -> row.row)
        destination.exact_shift_invariant || continue
        destination.horizon_role === :inherited_interior || error(
            "Row $(destination.row) is exact-shift invariant but has role " *
            "$(destination.horizon_role).",
        )
        isnothing(destination.stage) && error(
            "Inherited residual row $(destination.row) has no physical stage.",
        )
        source_key = _equation_key(
            destination;
            stage = something(destination.stage) + 1,
        )
        haskey(lookup, source_key) || error(
            "No stage-$(something(destination.stage) + 1) source residual for " *
            "destination row $(destination.row), key $(source_key).",
        )
        push!(source_rows, lookup[source_key].row)
        push!(destination_rows, destination.row)
    end

    length(destination_rows) == EXPECTED_ROW_COUNTS[:inherited_interior] ||
        error(
            "Expected $(EXPECTED_ROW_COUNTS[:inherited_interior]) inherited " *
            "residual rows, found $(length(destination_rows)).",
        )
    length(unique(source_rows)) == length(source_rows) ||
        error("The inherited residual source map is not injective.")
    length(unique(destination_rows)) == length(destination_rows) ||
        error("The inherited residual destination map is not injective.")
    (; source_rows, destination_rows)
end

function _validate_equation_metadata(kkt, form)
    isnothing(kkt.metadata) &&
        error("$(form) KKT system has no semantic metadata.")
    rows = sort(kkt.metadata.equations; by = row -> row.row)
    length(rows) == kkt.kkt_dimension || error(
        "$(form) equation metadata has $(length(rows)) rows; expected " *
        "$(kkt.kkt_dimension).",
    )
    getfield.(rows, :row) == collect(1:kkt.kkt_dimension) ||
        error("$(form) equation metadata is not ordered and exhaustive.")

    for row in rows
        row.horizon_role in HORIZON_ROLES || error(
            "$(form) row $(row.row) has unclassified/unknown horizon role " *
            "$(row.horizon_role).",
        )
        row.exact_shift_invariant ==
        (row.horizon_role === :inherited_interior) || error(
            "$(form) row $(row.row) has inconsistent horizon role and " *
            "exact-shift flag.",
        )
        row.horizon_role !== :global && isnothing(row.stage) && error(
            "$(form) non-global row $(row.row) has no physical stage.",
        )
    end

    counts = Dict(
        role => count(row -> row.horizon_role === role, rows) for
        role in HORIZON_ROLES
    )
    counts == EXPECTED_ROW_COUNTS || error(
        "$(form) horizon-role counts drifted: expected " *
        "$(EXPECTED_ROW_COUNTS), got $(counts).",
    )
    shift_map = _residual_shift_map(kkt)
    (; rows, counts, shift_map, signature = _metadata_signature(kkt))
end

function _dual_key(row; stage = row.stage)
    (
        row.family,
        row.scope,
        row.player,
        row.owner_level,
        row.target_level,
        row.equation_class,
        row.equation_type,
        row.primal_variable,
        row.component,
        stage,
    )
end

function _dual_factor(row)
    row.family === :equality_multiplier &&
        row.equation_class === :initial_condition &&
        return :initial_condition
    row.shift_rule === :reset && return :explicit_terminal_reset
    row.shift_rule === :successor && row.successor_exists &&
        return :inherited_successor
    row.shift_rule === :successor && !row.successor_exists &&
        return :missing_successor_tail
    row.shift_rule === :identity && return :global_identity
    error(
        "Dual coordinate $(row.index) cannot be assigned to a diagnostic " *
        "transport factor (rule=$(row.shift_rule), " *
        "successor_exists=$(row.successor_exists)).",
    )
end

function _dual_metadata_signature(duals)
    lines = String[]
    for row in duals
        push!(
            lines,
            join(
                string.(
                    (
                        row.index,
                        row.family,
                        row.scope,
                        row.player,
                        row.owner_level,
                        row.target_level,
                        row.equation_class,
                        row.equation_type,
                        row.primal_variable,
                        row.stage,
                        row.successor_stage,
                        row.component,
                        row.shift_rule,
                        row.successor_exists,
                        row.tail_role,
                        _dual_factor(row),
                    ),
                ),
                "|",
            ),
        )
    end
    bytes2hex(SHA.sha256(join(lines, "\n")))
end

function _validate_dual_factors(kkt, form)
    duals = sort!(
        filter(
            row ->
                row.family in
                (:equality_multiplier, :stationarity_multiplier),
            kkt.metadata.variables,
        );
        by = row -> row.index,
    )
    counts = Dict(
        factor => count(row -> _dual_factor(row) === factor, duals) for
        factor in DUAL_FACTORS
    )
    counts == EXPECTED_DUAL_FACTOR_COUNTS || error(
        "$(form) diagnostic dual-factor counts drifted: expected " *
        "$(EXPECTED_DUAL_FACTOR_COUNTS), got $(counts).",
    )
    length(duals) == sum(values(EXPECTED_DUAL_FACTOR_COUNTS)) ||
        error("$(form) dual-factor classification is not exhaustive.")

    lookup = Dict{Tuple, Any}()
    for row in duals
        key = _dual_key(row)
        haskey(lookup, key) &&
            error("Dual metadata is not unique for semantic key $(key).")
        lookup[key] = row
    end
    for row in duals
        _dual_factor(row) === :inherited_successor || continue
        key = _dual_key(row; stage = something(row.stage) + 1)
        haskey(lookup, key) || error(
            "Inherited dual coordinate $(row.index) has no semantic " *
            "stage-$(something(row.stage) + 1) source.",
        )
    end
    (; duals, counts, lookup, signature = _dual_metadata_signature(duals))
end

function _ablation_specs()
    specs = NamedTuple[]
    for interior_map in INTERIOR_MAPS,
        initial_completion in INITIAL_COMPLETIONS,
        terminal_completion in TERMINAL_COMPLETIONS,
        explicit_completion in EXPLICIT_COMPLETIONS

        id = join(
            (
                "interior_$(interior_map)",
                "initial_$(initial_completion)",
                "tail_$(terminal_completion)",
                "explicit_$(explicit_completion)",
            ),
            "__",
        )
        production_alias =
            interior_map === :fixed_index &&
            initial_completion === :copy &&
            terminal_completion === :hold &&
            explicit_completion === :copy ? :identity_copy :
            interior_map === :semantic_shift &&
            initial_completion === :reset &&
            terminal_completion === :zero &&
            explicit_completion === :reset ? :stage_shift_zero_tail :
            interior_map === :semantic_shift &&
            initial_completion === :reset &&
            terminal_completion === :hold &&
            explicit_completion === :reset ? :stage_shift_hold_tail :
            :diagnostic_only
        push!(
            specs,
            (;
                id,
                interior_map,
                initial_completion,
                terminal_completion,
                explicit_completion,
                production_alias,
            ),
        )
    end
    length(specs) == 16 || error("Expected the full private 2^4 ablation.")
    count(spec -> spec.production_alias !== :diagnostic_only, specs) == 3 ||
        error("Production transports are not uniquely embedded in the ablation.")
    specs
end

function _diagnostic_warmstart(pair, kkt, dual_validation, spec)
    source_z = pair.source["z"]
    transported = copy(source_z)
    for row in dual_validation.duals
        factor = _dual_factor(row)
        if factor === :inherited_successor
            if spec.interior_map === :semantic_shift
                source_key = _dual_key(
                    row;
                    stage = something(row.stage) + 1,
                )
                transported[row.index] =
                    source_z[dual_validation.lookup[source_key].index]
            else
                @assert spec.interior_map === :fixed_index
                transported[row.index] = source_z[row.index]
            end
        elseif factor === :initial_condition
            transported[row.index] =
                spec.initial_completion === :reset ?
                zero(eltype(transported)) : source_z[row.index]
        elseif factor === :missing_successor_tail
            transported[row.index] =
                spec.terminal_completion === :zero ?
                zero(eltype(transported)) : source_z[row.index]
        elseif factor === :explicit_terminal_reset
            transported[row.index] =
                spec.explicit_completion === :reset ?
                zero(eltype(transported)) : source_z[row.index]
        else
            @assert factor === :global_identity
            transported[row.index] = source_z[row.index]
        end
    end
    DTS._build_warmstart(
        pair.source["shifted_primal"],
        transported,
        kkt,
        :all_duals,
        :identity_copy,
    )
end

function _assert_exact_vector(actual, expected, label)
    length(actual) == length(expected) ||
        error("$(label) length mismatch: $(length(actual)) != $(length(expected)).")
    isequal(actual, expected) || begin
        discrepancy = maximum(abs, actual .- expected; init = 0.0)
        error("$(label) is not an exact alias (max discrepancy $(discrepancy)).")
    end
    nothing
end

function _vector_metrics(values)
    count = length(values)
    count == 0 &&
        return (;
            count = 0,
            norm2 = 0.0,
            norm_inf = 0.0,
            rms = 0.0,
            mean_abs = 0.0,
        )
    norm2 = norm(values)
    (;
        count,
        norm2,
        norm_inf = maximum(abs, values),
        rms = norm2 / sqrt(count),
        mean_abs = sum(abs, values) / count,
    )
end

function _role_indices(metadata_validation)
    Dict(
        role => Int[
            row.row for row in metadata_validation.rows if
            row.horizon_role === role
        ] for role in HORIZON_ROLES
    )
end

function _case_fields(pair)
    Dict{String, Any}(
        "formulation" => pair.form,
        "scenario_seed" => pair.seed,
        "transition" => pair.transition,
        "period" => pair.transition <= 3 ? :early : :late,
        "source_instance_digest" => pair.source["instance_digest"],
        "destination_instance_digest" =>
            pair.destination["instance_digest"],
        "source_reference_residual" =>
            pair.source["direct_residual_norm2"],
        "destination_reference_residual" =>
            pair.destination["direct_residual_norm2"],
    )
end

function _merge_fields(parts...)
    row = Dict{String, Any}()
    for part in parts, (key, value) in pairs(part)
        row[string(key)] = value
    end
    row
end

function _region_row(
    pair,
    residual,
    source_residual,
    metadata_validation;
    descriptor,
)
    role_indices = _role_indices(metadata_validation)
    row = _merge_fields(
        _case_fields(pair),
        descriptor,
        Dict(
            "initial_residual_norm2" => norm(residual),
            "initial_residual_norm_inf" =>
                maximum(abs, residual; init = 0.0),
        ),
    )
    squared_partition = 0.0
    for role in HORIZON_ROLES
        metrics = _vector_metrics(view(residual, role_indices[role]))
        prefix =
            role === :inherited_interior ? "interior" :
            role === :initial_boundary ? "initial" :
            role === :terminal_boundary ? "terminal" : "global"
        row["$(prefix)_count"] = metrics.count
        row["$(prefix)_norm2"] = metrics.norm2
        row["$(prefix)_norm_inf"] = metrics.norm_inf
        row["$(prefix)_rms"] = metrics.rms
        row["$(prefix)_mean_abs"] = metrics.mean_abs
        squared_partition += metrics.norm2^2
    end
    total_squared = norm(residual)^2
    abs(total_squared - squared_partition) <=
    2_000 * eps(Float64) * max(1.0, total_squared) || error(
        "Residual horizon-role norms do not form an orthogonal partition for " *
        "$(pair.form) seed $(pair.seed) transition $(pair.transition).",
    )

    shift_map = metadata_validation.shift_map
    defect =
        residual[shift_map.destination_rows] .-
        source_residual[shift_map.source_rows]
    defect_metrics = _vector_metrics(defect)
    row["interior_shift_defect_norm2"] = defect_metrics.norm2
    row["interior_shift_defect_norm_inf"] = defect_metrics.norm_inf
    row["interior_shift_defect_rms"] = defect_metrics.rms
    row
end

function _stage_group_key(row)
    (
        row.horizon_role,
        row.family,
        row.scope,
        row.player,
        row.level,
        row.equation_class,
        row.equation_type,
        row.primal_variable,
        row.stage,
    )
end

function _stage_rows(
    pair,
    residual,
    source_residual,
    metadata_validation;
    descriptor,
)
    grouped = Dict{Tuple, Vector{Any}}()
    for equation in metadata_validation.rows
        push!(
            get!(grouped, _stage_group_key(equation), Any[]),
            equation,
        )
    end
    source_by_destination = Dict(
        destination => source for
        (source, destination) in zip(
            metadata_validation.shift_map.source_rows,
            metadata_validation.shift_map.destination_rows,
        )
    )

    rows = Dict{String, Any}[]
    for (key, equations) in sort!(collect(grouped); by = item -> string(first(item)))
        (
            horizon_role,
            family,
            scope,
            player,
            level,
            equation_class,
            equation_type,
            primal_variable,
            stage,
        ) = key
        indices = sort!(Int[equation.row for equation in equations])
        metrics = _vector_metrics(view(residual, indices))
        defect_metrics = if horizon_role === :inherited_interior
            source_indices = Int[source_by_destination[index] for index in indices]
            _vector_metrics(
                residual[indices] .- source_residual[source_indices],
            )
        else
            nothing
        end
        push!(
            rows,
            _merge_fields(
                _case_fields(pair),
                descriptor,
                Dict(
                    "horizon_role" => horizon_role,
                    "equation_family" => family,
                    "scope" => scope,
                    "player" => player,
                    "preference_level" => level,
                    "equation_class" => equation_class,
                    "equation_type" => equation_type,
                    "primal_variable" => primal_variable,
                    "physical_stage" => stage,
                    "coordinate_count" => metrics.count,
                    "residual_norm2" => metrics.norm2,
                    "residual_norm_inf" => metrics.norm_inf,
                    "residual_rms" => metrics.rms,
                    "residual_mean_abs" => metrics.mean_abs,
                    "shift_defect_norm2" =>
                        isnothing(defect_metrics) ? NaN :
                        defect_metrics.norm2,
                    "shift_defect_norm_inf" =>
                        isnothing(defect_metrics) ? NaN :
                        defect_metrics.norm_inf,
                    "shift_defect_rms" =>
                        isnothing(defect_metrics) ? NaN :
                        defect_metrics.rms,
                ),
            ),
        )
    end
    rows
end

function _equation_metadata_rows(form, metadata_validation)
    [
        Dict{String, Any}(
            "case_id" => "$(form)__row$(row.row)",
            "formulation" => form,
            "row" => row.row,
            "equation_family" => row.family,
            "scope" => row.scope,
            "player" => row.player,
            "preference_level" => row.level,
            "equation_class" => row.equation_class,
            "equation_type" => row.equation_type,
            "primal_variable" => row.primal_variable,
            "physical_stage" => row.stage,
            "successor_stage" => row.successor_stage,
            "component" => row.component,
            "horizon_role" => row.horizon_role,
            "exact_shift_invariant" => row.exact_shift_invariant,
        ) for row in metadata_validation.rows
    ]
end

function _diagnostic_checkpoint_path(run_dir, pair)
    joinpath(
        run_dir,
        "checkpoints",
        "stagewise",
        String(pair.form),
        "seed_$(pair.seed)",
        "transition_$(pair.transition).jld2",
    )
end

function _source_checkpoint(source_run, pair)
    path = DTS._diagnostic_path(source_run, pair)
    isfile(path) || error(
        "Completed source diagnostic checkpoint is missing: $(path)",
    )
    checkpoint = JLD2.load_object(path)
    for transport in PRODUCTION_TRANSPORTS
        key = (:all_duals, transport)
        haskey(checkpoint["warmstarts"], key) ||
            error("Source checkpoint $(path) lacks warm start $(key).")
        haskey(checkpoint["residual_values"], key) ||
            error("Source checkpoint $(path) lacks residual $(key).")
    end
    checkpoint
end

function _checkpoint_matches(
    checkpoint,
    pair,
    metadata_validation,
    dual_validation,
    source_run,
)
    get(checkpoint, "schema_version", "") == SCHEMA_VERSION || return false
    get(checkpoint, "formulation", nothing) == pair.form || return false
    get(checkpoint, "scenario_seed", nothing) == pair.seed || return false
    get(checkpoint, "transition", nothing) == pair.transition || return false
    get(checkpoint, "source_instance_digest", "") ==
        pair.source["instance_digest"] || return false
    get(checkpoint, "destination_instance_digest", "") ==
        pair.destination["instance_digest"] || return false
    get(checkpoint, "metadata_signature", "") ==
        metadata_validation.signature || return false
    get(checkpoint, "dual_metadata_signature", "") ==
        dual_validation.signature || return false
    source_path = DTS._diagnostic_path(source_run, pair)
    isfile(source_path) || return false
    get(checkpoint, "source_diagnostic_sha256", "") ==
        DTS._sha256(source_path) || return false
    length(get(checkpoint, "production_region_rows", Any[])) == 3 ||
        return false
    length(get(checkpoint, "ablation_region_rows", Any[])) == 16 ||
        return false
    true
end

function _compute_pair_checkpoint(
    source_run,
    pair,
    built,
    metadata_validation,
    dual_validation,
)
    source_checkpoint = _source_checkpoint(source_run, pair)
    source_residual = DTS._residual_metrics(
        built.kkt,
        pair.source["z"],
        pair.source["parameters"].θ,
    ).residual.values
    length(source_residual) == built.kkt.kkt_dimension ||
        error("Source residual dimension drifted.")

    production_warmstarts = Dict{Symbol, Vector{Float64}}()
    production_residuals = Dict{Symbol, Vector{Float64}}()
    for transport in PRODUCTION_TRANSPORTS
        current = DTS._build_warmstart(
            pair.source["shifted_primal"],
            pair.source["z"],
            built.kkt,
            :all_duals,
            transport,
        )
        stored =
            source_checkpoint["warmstarts"][(:all_duals, transport)]
        _assert_exact_vector(
            current,
            stored,
            "Stored production $(transport) warm start",
        )
        production_warmstarts[transport] = current
        residual =
            source_checkpoint["residual_values"][(:all_duals, transport)]
        length(residual) == built.kkt.kkt_dimension ||
            error("Stored $(transport) residual dimension drifted.")
        production_residuals[transport] = residual
    end

    ablation_warmstarts = Dict{String, Vector{Float64}}()
    ablation_residuals = Dict{String, Vector{Float64}}()
    ablation_specs = _ablation_specs()
    for spec in ablation_specs
        warmstart =
            _diagnostic_warmstart(pair, built.kkt, dual_validation, spec)
        if spec.production_alias !== :diagnostic_only
            _assert_exact_vector(
                warmstart,
                production_warmstarts[spec.production_alias],
                "Diagnostic $(spec.id) / $(spec.production_alias)",
            )
            residual = production_residuals[spec.production_alias]
        else
            residual = DTS._residual_metrics(
                built.kkt,
                warmstart,
                pair.destination["parameters"].θ,
            ).residual.values
        end
        ablation_warmstarts[spec.id] = warmstart
        ablation_residuals[spec.id] = residual
    end

    production_region_rows = Dict{String, Any}[]
    production_stage_rows = Dict{String, Any}[]
    for transport in PRODUCTION_TRANSPORTS
        descriptor = Dict{String, Any}(
            "case_id" =>
                "$(pair.form)__seed$(pair.seed)__t$(pair.transition)__$(transport)",
            "record_kind" => "production_policy",
            "dual_transport" => transport,
            "ablation_id" => "",
        )
        residual = production_residuals[transport]
        push!(
            production_region_rows,
            _region_row(
                pair,
                residual,
                source_residual,
                metadata_validation;
                descriptor,
            ),
        )
        append!(
            production_stage_rows,
            _stage_rows(
                pair,
                residual,
                source_residual,
                metadata_validation;
                descriptor,
            ),
        )
    end

    ablation_region_rows = Dict{String, Any}[]
    ablation_stage_rows = Dict{String, Any}[]
    for spec in ablation_specs
        descriptor = Dict{String, Any}(
            "case_id" =>
                "$(pair.form)__seed$(pair.seed)__t$(pair.transition)__$(spec.id)",
            "record_kind" => "diagnostic_ablation",
            "dual_transport" =>
                spec.production_alias === :diagnostic_only ? "" :
                spec.production_alias,
            "ablation_id" => spec.id,
            "interior_map" => spec.interior_map,
            "initial_completion" => spec.initial_completion,
            "terminal_completion" => spec.terminal_completion,
            "explicit_terminal_completion" =>
                spec.explicit_completion,
            "production_alias" => spec.production_alias,
        )
        residual = ablation_residuals[spec.id]
        push!(
            ablation_region_rows,
            _region_row(
                pair,
                residual,
                source_residual,
                metadata_validation;
                descriptor,
            ),
        )
        append!(
            ablation_stage_rows,
            _stage_rows(
                pair,
                residual,
                source_residual,
                metadata_validation;
                descriptor,
            ),
        )
    end

    # Boundary completion factors cannot contaminate inherited interior rows.
    # For a fixed interior-map choice, every boundary-factor combination must
    # produce the same inherited-interior residual vector exactly.
    interior_rows = metadata_validation.shift_map.destination_rows
    for interior_map in INTERIOR_MAPS
        subset = filter(
            spec -> spec.interior_map === interior_map,
            ablation_specs,
        )
        reference =
            ablation_residuals[first(subset).id][interior_rows]
        for spec in subset[2:end]
            _assert_exact_vector(
                ablation_residuals[spec.id][interior_rows],
                reference,
                "Boundary-factor isolation for $(interior_map), $(spec.id)",
            )
        end
    end

    Dict{String, Any}(
        "schema_version" => SCHEMA_VERSION,
        "formulation" => pair.form,
        "scenario_seed" => pair.seed,
        "transition" => pair.transition,
        "source_instance_digest" => pair.source["instance_digest"],
        "destination_instance_digest" =>
            pair.destination["instance_digest"],
        "metadata_signature" => metadata_validation.signature,
        "dual_metadata_signature" => dual_validation.signature,
        "source_diagnostic_sha256" =>
            DTS._sha256(DTS._diagnostic_path(source_run, pair)),
        "source_residual" => source_residual,
        "residual_shift_source_rows" =>
            metadata_validation.shift_map.source_rows,
        "residual_shift_destination_rows" =>
            metadata_validation.shift_map.destination_rows,
        "production_warmstarts" => production_warmstarts,
        "production_residuals" => production_residuals,
        "ablation_warmstarts" => ablation_warmstarts,
        "ablation_residuals" => ablation_residuals,
        "production_region_rows" => production_region_rows,
        "production_stage_rows" => production_stage_rows,
        "ablation_region_rows" => ablation_region_rows,
        "ablation_stage_rows" => ablation_stage_rows,
    )
end

function _load_or_compute_pair!(
    run_dir,
    source_run,
    pair,
    built,
    metadata_validation,
    dual_validation,
)
    path = _diagnostic_checkpoint_path(run_dir, pair)
    if isfile(path)
        checkpoint = JLD2.load_object(path)
        _checkpoint_matches(
            checkpoint,
            pair,
            metadata_validation,
            dual_validation,
            source_run,
        ) || error(
            "Stagewise checkpoint is stale or incompatible: $(path). " *
            "Move it aside and rerun; it will not be overwritten silently.",
        )
        return checkpoint
    end
    checkpoint = _compute_pair_checkpoint(
        source_run,
        pair,
        built,
        metadata_validation,
        dual_validation,
    )
    DTS._atomic_save(path, checkpoint)
    checkpoint
end

function _validation_context(source_run)
    source_run = abspath(source_run)
    isdir(source_run) ||
        error("Dual-transport source run does not exist: $(source_run)")
    for marker in ("inputs/complete", "metadata_complete", "diagnostics_complete")
        isfile(joinpath(source_run, marker)) ||
            error("Source run is incomplete; missing $(marker).")
    end
    config = _source_config(source_run)
    DTS._assert_input_hashes(source_run)
    pairs = DTS.valid_pairs(source_run)
    length(pairs) == 17 ||
        error("Expected 17 valid source/destination pairs, found $(length(pairs)).")
    form_counts = Dict(
        form => count(pair -> pair.form === form, pairs) for
        form in DTS.FORMS
    )
    form_counts == Dict(:reduced => 8, :quasi => 9) || error(
        "Valid-pair formulation counts drifted: $(form_counts).",
    )

    cache = Dict{Symbol, Any}()
    built_by_form = Dict{Symbol, Any}()
    metadata_by_form = Dict{Symbol, Any}()
    duals_by_form = Dict{Symbol, Any}()
    for form in DTS.FORMS
        built = DTS._build_system(config, form, cache)
        built_by_form[form] = built
        metadata_by_form[form] =
            _validate_equation_metadata(built.kkt, form)
        duals_by_form[form] = _validate_dual_factors(built.kkt, form)
    end
    metadata_by_form[:reduced].signature ==
        metadata_by_form[:quasi].signature || error(
        "Reduced and quasi equation metadata differ.",
    )

    for pair in pairs
        _source_checkpoint(source_run, pair)
    end
    (;
        source_run,
        config,
        pairs,
        form_counts,
        built_by_form,
        metadata_by_form,
        duals_by_form,
    )
end

"""
    validate_stagewise_diagnostics(source_run; run_dir = nothing)

Validate the frozen dual-transport source run, all 17 reusable diagnostic
checkpoints, both generated KKT metadata layouts, the exact 1,220-row residual
shift map, and the 2,310-coordinate diagnostic dual factorization. No solver is
called. If `run_dir` is supplied, also validate all resumable stagewise
checkpoints and finalized CSV markers in that output directory.
"""
function validate_stagewise_diagnostics(source_run; run_dir = nothing)
    context = _validation_context(source_run)
    output_checkpoint_count = 0
    if !isnothing(run_dir)
        run_dir = abspath(run_dir)
        isfile(joinpath(run_dir, "stagewise_complete")) ||
            error("Stagewise output is incomplete: $(run_dir)")
        for filename in (
            "equation_metadata.csv",
            "production_region_residuals.csv",
            "production_stagewise_residuals.csv",
            "boundary_ablations.csv",
            "boundary_ablation_stagewise_residuals.csv",
            "spike_localization.csv",
        )
            isfile(joinpath(run_dir, "raw", filename)) ||
                error("Stagewise output is missing raw/$(filename).")
        end
        for pair in context.pairs
            path = _diagnostic_checkpoint_path(run_dir, pair)
            isfile(path) || error("Stagewise checkpoint is missing: $(path)")
            checkpoint = JLD2.load_object(path)
            _checkpoint_matches(
                checkpoint,
                pair,
                context.metadata_by_form[pair.form],
                context.duals_by_form[pair.form],
                context.source_run,
            ) || error("Stagewise checkpoint failed validation: $(path)")
            output_checkpoint_count += 1
        end
    end
    (
        schema_version = SCHEMA_VERSION,
        source_run = context.source_run,
        valid_pairs = length(context.pairs),
        reduced_pairs = context.form_counts[:reduced],
        quasi_pairs = context.form_counts[:quasi],
        equation_rows =
            length(context.metadata_by_form[:reduced].rows),
        interior_rows =
            context.metadata_by_form[:reduced].counts[:inherited_interior],
        initial_rows =
            context.metadata_by_form[:reduced].counts[:initial_boundary],
        terminal_rows =
            context.metadata_by_form[:reduced].counts[:terminal_boundary],
        global_rows = context.metadata_by_form[:reduced].counts[:global],
        transported_duals =
            length(context.duals_by_form[:reduced].duals),
        output_checkpoint_count,
    )
end

"""
    run_stagewise_diagnostics(run_dir, source_run)

Run the solver-free Phase-1 stagewise localization study. The function reads
the completed dual-transport run, validates and exactly reproduces its three
production all-dual warm starts, evaluates the private 2^4 diagnostic
factorization, saves one atomic JLD2 checkpoint per valid transition, and
writes wide region and long stage/family/player/level CSVs.
"""
function run_stagewise_diagnostics(run_dir, source_run)
    run_dir = abspath(run_dir)
    context = _validation_context(source_run)
    mkpath(joinpath(run_dir, "raw"))
    mkpath(joinpath(run_dir, "checkpoints", "stagewise"))

    equation_rows = Dict{String, Any}[]
    for form in DTS.FORMS
        append!(
            equation_rows,
            _equation_metadata_rows(
                form,
                context.metadata_by_form[form],
            ),
        )
    end
    DTS.write_csv(
        joinpath(run_dir, "raw", "equation_metadata.csv"),
        equation_rows,
    )

    production_region_rows = Dict{String, Any}[]
    production_stage_rows = Dict{String, Any}[]
    ablation_region_rows = Dict{String, Any}[]
    ablation_stage_rows = Dict{String, Any}[]
    for pair in context.pairs
        checkpoint = _load_or_compute_pair!(
            run_dir,
            context.source_run,
            pair,
            context.built_by_form[pair.form],
            context.metadata_by_form[pair.form],
            context.duals_by_form[pair.form],
        )
        append!(
            production_region_rows,
            checkpoint["production_region_rows"],
        )
        append!(
            production_stage_rows,
            checkpoint["production_stage_rows"],
        )
        append!(
            ablation_region_rows,
            checkpoint["ablation_region_rows"],
        )
        append!(
            ablation_stage_rows,
            checkpoint["ablation_stage_rows"],
        )
    end

    length(production_region_rows) == 51 || error(
        "Expected 51 production region rows, got " *
        "$(length(production_region_rows)).",
    )
    length(ablation_region_rows) == 272 || error(
        "Expected 272 boundary-ablation rows, got " *
        "$(length(ablation_region_rows)).",
    )
    spike_rows = filter(production_region_rows) do row
        row["formulation"] in (:reduced, "reduced") &&
            row["scenario_seed"] == 202 &&
            row["transition"] in 4:7
    end
    length(spike_rows) == 12 ||
        error("Expected 12 reduced seed-202 spike-localization rows.")

    raw_dir = joinpath(run_dir, "raw")
    outputs = Dict(
        "production_region_residuals.csv" => production_region_rows,
        "production_stagewise_residuals.csv" => production_stage_rows,
        "boundary_ablations.csv" => ablation_region_rows,
        "boundary_ablation_stagewise_residuals.csv" =>
            ablation_stage_rows,
        "spike_localization.csv" => spike_rows,
    )
    for (filename, rows) in outputs
        DTS.write_csv(joinpath(raw_dir, filename), rows)
    end

    manifest = Dict{String, Any}(
        "schema_version" => SCHEMA_VERSION,
        "source_run" => context.source_run,
        "valid_pairs" => length(context.pairs),
        "production_cases" => length(production_region_rows),
        "diagnostic_ablation_cases" => length(ablation_region_rows),
        "equation_metadata_rows" => length(equation_rows),
        "interior_rows_per_form" =>
            EXPECTED_ROW_COUNTS[:inherited_interior],
        "initial_rows_per_form" =>
            EXPECTED_ROW_COUNTS[:initial_boundary],
        "terminal_rows_per_form" =>
            EXPECTED_ROW_COUNTS[:terminal_boundary],
        "global_rows_per_form" => EXPECTED_ROW_COUNTS[:global],
        "transported_duals_per_form" =>
            sum(values(EXPECTED_DUAL_FACTOR_COUNTS)),
    )
    for filename in (
        "equation_metadata.csv",
        keys(outputs)...,
    )
        manifest["sha256_$(filename)"] =
            DTS._sha256(joinpath(raw_dir, filename))
    end
    DTS._write_toml(
        joinpath(run_dir, "stagewise_manifest.toml"),
        manifest,
    )
    DTS._atomic_write(
        joinpath(run_dir, "stagewise_complete"),
        "$(SCHEMA_VERSION)\n",
    )
    validate_stagewise_diagnostics(
        context.source_run;
        run_dir,
    )
end

end # module
