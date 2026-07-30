module TheoryResolutionStudy

using Dates: Dates
using JLD2: JLD2
using LinearAlgebra: diag, dot, norm, qr
using Printf: @sprintf
using SHA: SHA
using SparseArrays: nnz, nonzeros, nzrange, rowvals
using TOML: TOML
using ReducedGOOP

if !isdefined(Main, :DualTransportStudy)
    Base.include(
        Main,
        normpath(joinpath(@__DIR__, "..", "dual_transport", "DualTransportStudy.jl")),
    )
end
const DTS = Main.DualTransportStudy
const SWS = DTS.SWS

export TheoryResolutionConfig,
    default_config,
    load_config,
    prepare_run,
    snapshot_inputs!,
    run_holdout_validation!,
    analyze_holdout_validation!,
    run_globalization_study!,
    analyze_study!,
    generate_report

const STUDY_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(STUDY_DIR, "..", "..", ".."))
const DEFAULT_SOURCE_RUN = joinpath(
    REPOSITORY_ROOT,
    "data",
    "dual_transport",
    "pilot",
    "2026-07-30_103743_dual_transport",
)
const PROTOCOL = :theory_resolution_t20_dt0p1_tol0p008_max1000_v1
const PLANNING_HORIZON = 20
const DELTA_T = 0.1
const SOLVER_TOL = 8e-3
const MAX_INNER_ITERS = 1000
const MAX_OUTER_ITERS = 1
const LINEAR_SOLVER = :klu
const LINESEARCH = :backtracking
const HELDOUT_GAMMA = 0.25
const ALIGNMENT_TOL = 1e-3
const FORMS = (:reduced, :quasi)
const DEVELOPMENT_CASES = Set((
    (:reduced, 202, 1),
    (:reduced, 202, 4),
    (:reduced, 202, 5),
    (:quasi, 101, 1),
    (:quasi, 202, 2),
    (:quasi, 202, 6),
))
const GLOBALIZATION_CASES = (
    (
        form = :quasi,
        seed = 202,
        transition = 2,
        gammas = (0.5, 0.75, 1.0),
    ),
    (
        form = :reduced,
        seed = 202,
        transition = 5,
        gammas = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    ),
    (
        form = :quasi,
        seed = 101,
        transition = 1,
        gammas = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    ),
)

Base.@kwdef struct TheoryResolutionConfig
    source_run::String = DEFAULT_SOURCE_RUN
    output_root::String = joinpath(REPOSITORY_ROOT, "data", "theory_resolution")
    profile::Symbol = :pilot
    protocol::Symbol = PROTOCOL
    planning_horizon::Int = PLANNING_HORIZON
    Δt::Float64 = DELTA_T
    tol::Float64 = SOLVER_TOL
    max_inner_iters::Int = MAX_INNER_ITERS
    max_outer_iters::Int = MAX_OUTER_ITERS
    linear_solver::Symbol = LINEAR_SOLVER
    linesearch::Symbol = LINESEARCH
    heldout_gamma::Float64 = HELDOUT_GAMMA
    alignment_tol::Float64 = ALIGNMENT_TOL
end

default_config(;
    source_run = DEFAULT_SOURCE_RUN,
    output_root = joinpath(REPOSITORY_ROOT, "data", "theory_resolution"),
) = TheoryResolutionConfig(;
    source_run = abspath(source_run),
    output_root = abspath(output_root),
)

function _config_dict(config::TheoryResolutionConfig)
    Dict{String, Any}(
        "source_run" => abspath(config.source_run),
        "output_root" => abspath(config.output_root),
        "profile" => String(config.profile),
        "protocol" => String(config.protocol),
        "planning_horizon" => config.planning_horizon,
        "dt" => config.Δt,
        "tol" => config.tol,
        "max_inner_iters" => config.max_inner_iters,
        "max_outer_iters" => config.max_outer_iters,
        "linear_solver" => String(config.linear_solver),
        "linesearch" => String(config.linesearch),
        "heldout_gamma" => config.heldout_gamma,
        "alignment_tol" => config.alignment_tol,
    )
end

function _config_from_dict(values)
    config = TheoryResolutionConfig(;
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
    validate_protocol(config)
    config
end

load_config(path::AbstractString) = _config_from_dict(TOML.parsefile(path))

function _source_config(config::TheoryResolutionConfig)
    path = joinpath(config.source_run, "config.toml")
    isfile(path) || error("Completed dual-transport config is missing: $(path)")
    DTS.load_config(path)
end

function _source_solver_options(config::TheoryResolutionConfig)
    source_config = _source_config(config)
    options = DTS._solver_options(source_config)
    frozen_path = joinpath(config.source_run, "solver_options.toml")
    isfile(frozen_path) ||
        error("Completed run has no frozen solver-options snapshot: $(frozen_path)")
    frozen = TOML.parsefile(frozen_path)["options"]
    actual = Dict(
        string(name) => begin
            value = getproperty(options, name)
            value isa Symbol ? String(value) : value
        end for name in propertynames(options)
    )
    Set(keys(actual)) == Set(keys(frozen)) ||
        error("The current solver-option schema differs from the completed run.")
    for name in keys(frozen)
        isequal(actual[name], frozen[name]) ||
            error(
                "Frozen solver option $(name) drifted: expected " *
                "$(frozen[name]), got $(actual[name]).",
            )
    end
    expected = (
        tol = SOLVER_TOL,
        max_inner_iters = MAX_INNER_ITERS,
        max_outer_iters = MAX_OUTER_ITERS,
        linear_solver = LINEAR_SOLVER,
        linesearch = LINESEARCH,
        record_condition_number = false,
        tsvd_threshold = 0.0,
        use_marquardt_scaling = false,
        reuse_factorization_iters = 0,
    )
    for name in propertynames(expected)
        getproperty(options, name) == getproperty(expected, name) ||
            error(
                "Frozen solver option $(name) drifted: expected " *
                "$(getproperty(expected, name)), got $(getproperty(options, name)).",
            )
    end
    options
end

function validate_protocol(config::TheoryResolutionConfig)
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
    )
    expected = (
        PROTOCOL,
        PLANNING_HORIZON,
        DELTA_T,
        SOLVER_TOL,
        MAX_INNER_ITERS,
        MAX_OUTER_ITERS,
        LINEAR_SOLVER,
        LINESEARCH,
        HELDOUT_GAMMA,
    )
    actual == expected ||
        error("The theory-resolution numerical protocol is frozen at $(expected); got $(actual).")
    config.alignment_tol == ALIGNMENT_TOL ||
        error("Candidate-alignment threshold is preregistered at $(ALIGNMENT_TOL).")
    isdir(config.source_run) ||
        error("Completed dual-transport run does not exist: $(config.source_run)")
    _source_solver_options(config)
    pairs = DTS.valid_pairs(config.source_run)
    length(pairs) == 17 ||
        error("Expected 17 valid frozen transition pairs, got $(length(pairs)).")
    nothing
end

_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function _atomic_write(path, contents)
    mkpath(dirname(path))
    temporary = path * ".tmp." * string(getpid())
    open(temporary, "w") do io
        write(io, contents)
        flush(io)
    end
    mv(temporary, path; force = true)
    path
end

function _atomic_save(path, object)
    mkpath(dirname(path))
    temporary = path * ".tmp." * string(getpid())
    JLD2.save_object(temporary, object)
    mv(temporary, path; force = true)
    path
end

function _copy_with_record!(records, source, destination, root)
    isfile(source) || error("Required frozen input is missing: $(source)")
    mkpath(dirname(destination))
    cp(source, destination; force = true)
    source_hash = _sha256(source)
    destination_hash = _sha256(destination)
    source_hash == destination_hash || error("Input snapshot hash mismatch: $(source)")
    push!(
        records,
        Dict{String, Any}(
            "path" => relpath(destination, root),
            "source_path" => abspath(source),
            "sha256" => source_hash,
            "bytes" => filesize(source),
        ),
    )
    destination
end

function _copy_tree_with_records!(records, source_root, destination_root, run_dir)
    isdir(source_root) || error("Required input directory is missing: $(source_root)")
    for (directory, _, names) in walkdir(source_root), name in names
        source = joinpath(directory, name)
        relative = relpath(source, source_root)
        _copy_with_record!(
            records,
            source,
            joinpath(destination_root, relative),
            run_dir,
        )
    end
    nothing
end

function prepare_run(
    config::TheoryResolutionConfig;
    run_dir::Union{Nothing, AbstractString} = nothing,
    command::AbstractString = "",
)
    validate_protocol(config)
    resolved = if isnothing(run_dir)
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
        joinpath(
            config.output_root,
            String(config.profile),
            "$(timestamp)_theory_resolution",
        )
    else
        abspath(run_dir)
    end
    mkpath(resolved)
    for directory in (
        "inputs",
        "raw",
        "checkpoints",
        "figures",
        "provenance",
    )
        mkpath(joinpath(resolved, directory))
    end
    config_path = joinpath(resolved, "config.toml")
    if isfile(config_path)
        stored = load_config(config_path)
        _config_dict(stored) == _config_dict(config) ||
            error("Resume configuration differs from $(config_path).")
    else
        DTS._write_toml(config_path, _config_dict(config))
    end
    if !isempty(command)
        _atomic_write(joinpath(resolved, "reproduction_command.txt"), command * "\n")
    end
    resolved
end

function snapshot_inputs!(run_dir, config::TheoryResolutionConfig)
    complete = joinpath(run_dir, "inputs_complete")
    manifest_path = joinpath(run_dir, "inputs", "manifest.toml")
    if isfile(complete)
        manifest = TOML.parsefile(manifest_path)
        for record in manifest["files"]
            path = joinpath(run_dir, String(record["path"]))
            isfile(path) || error("Snapshotted input is missing: $(path)")
            _sha256(path) == record["sha256"] ||
                error("Snapshotted input drifted: $(path)")
        end
        return manifest
    end

    records = Dict{String, Any}[]
    for name in ("config.toml", "solver_options.toml", "analysis_manifest.toml")
        source = joinpath(config.source_run, name)
        isfile(source) || continue
        _copy_with_record!(
            records,
            source,
            joinpath(run_dir, "inputs", "dual_transport", name),
            run_dir,
        )
    end
    for name in (
        "replay.csv",
        "damping.csv",
        "residual_diagnostics.csv",
        "dual_metadata.csv",
    )
        source = joinpath(config.source_run, "raw", name)
        isfile(source) || continue
        _copy_with_record!(
            records,
            source,
            joinpath(run_dir, "inputs", "dual_transport", "raw", name),
            run_dir,
        )
    end
    _copy_tree_with_records!(
        records,
        joinpath(config.source_run, "inputs", "references"),
        joinpath(run_dir, "inputs", "references"),
        run_dir,
    )

    pairs = DTS.valid_pairs(run_dir)
    length(pairs) == 17 ||
        error("Reference snapshot produced $(length(pairs)) pairs instead of 17.")
    source_pairs = DTS.valid_pairs(config.source_run)
    source_index = Dict(
        (pair.form, pair.seed, pair.transition) => pair for pair in source_pairs
    )
    for pair in pairs
        key = (pair.form, pair.seed, pair.transition)
        source_pair = source_index[key]
        for (mode, label) in (
            (:all_except_innermost_stationarity, "gamma_0"),
            (:all_duals, "gamma_1"),
        )
            source = DTS._solve_checkpoint_path(
                config.source_run,
                source_pair,
                mode,
                :stage_shift_zero_tail,
            )
            destination = joinpath(
                run_dir,
                "inputs",
                "dual_transport",
                "checkpoints",
                "replay",
                String(pair.form),
                "seed_$(pair.seed)",
                "transition_$(pair.transition)__$(label).jld2",
            )
            _copy_with_record!(records, source, destination, run_dir)
        end
    end
    for key in DEVELOPMENT_CASES
        form, seed, transition = key
        for gamma in (0.0, HELDOUT_GAMMA, 1.0)
            source = DTS._damping_path(
                config.source_run,
                form,
                seed,
                transition,
                gamma,
            )
            destination = joinpath(
                run_dir,
                "inputs",
                "dual_transport",
                "checkpoints",
                "development_damping",
                String(form),
                "seed_$(seed)",
                basename(source),
            )
            _copy_with_record!(records, source, destination, run_dir)
        end
    end
    for case in GLOBALIZATION_CASES, gamma in case.gammas
        source = DTS._damping_path(
            config.source_run,
            case.form,
            case.seed,
            case.transition,
            gamma,
        )
        destination = joinpath(
            run_dir,
            "inputs",
            "dual_transport",
            "checkpoints",
            "hard_case_damping",
            String(case.form),
            "seed_$(case.seed)",
            basename(source),
        )
        _copy_with_record!(records, source, destination, run_dir)
    end
    manifest = Dict{String, Any}(
        "source_run" => abspath(config.source_run),
        "source_run_name" => basename(config.source_run),
        "valid_pairs" => length(pairs),
        "development_cases" => length(DEVELOPMENT_CASES),
        "heldout_cases" => length(pairs) - length(DEVELOPMENT_CASES),
        "files" => records,
    )
    DTS._write_toml(manifest_path, manifest)
    _atomic_write(complete, "ok\n")
    manifest
end

include("HoldoutValidation.jl")
include("GlobalizationStudy.jl")
include("AnalysisReport.jl")

end # module TheoryResolutionStudy
