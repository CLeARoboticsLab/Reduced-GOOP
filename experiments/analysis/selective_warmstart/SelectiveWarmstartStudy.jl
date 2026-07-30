module SelectiveWarmstartStudy

using BlockArrays: BlockArray
using Dates: Dates
using JLD2: JLD2
using LinearAlgebra: BLAS, norm, svd
using Pkg: Pkg
using Printf: @sprintf
using Random: MersenneTwister, rand, randn, shuffle!
using SHA: SHA
using SparseArrays: SparseMatrixCSC
using Statistics: cor, median, quantile
using TOML: TOML
using ReducedGOOP

export StudyConfig,
    preset_config,
    load_config,
    run_study,
    analyze_study,
    generate_figures,
    generate_report,
    read_csv_rows

const STUDY_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(STUDY_DIR, "..", "..", ".."))
const ROBOTIC_ARM_CORE_PATH =
    normpath(joinpath(STUDY_DIR, "..", "..", "robotic_arm_core.jl"))
const PROVENANCE_ROOTS = [
    "src",
    joinpath("experiments", "analysis", "selective_warmstart"),
]
const PROVENANCE_FILES = [
    joinpath("experiments", "Robotic_arm_mpc.jl"),
    joinpath("experiments", "Robotic_arm_receding.jl"),
    joinpath("experiments", "robotic_arm_core.jl"),
    joinpath("experiments", "Project.toml"),
    joinpath("experiments", "Manifest.toml"),
    joinpath("test", "Project.toml"),
    joinpath("test", "runtests.jl"),
    joinpath("test", "selective_warmstart.jl"),
    joinpath("test", "solver_trace.jl"),
    joinpath("test", "robotic_arm_receding_warmstart.jl"),
]

if !isdefined(Main, :RoboticArmCore)
    Base.include(Main, ROBOTIC_ARM_CORE_PATH)
end
Main.RoboticArmCore isa Module ||
    error("Main.RoboticArmCore exists but is not a module.")
const RAC = Main.RoboticArmCore

const MODES = ReducedGOOP.SELECTIVE_WARMSTART_MODES
const FORMULATIONS = (:reduced, :quasi)
const EPSILON0 = 0.1
const REFERENCE_INITIALIZATION = :cold_default_each_step
const COMPARABILITY_PROTOCOL =
    :uniform_t20_dt0p1_tol0p008_max1000_v1
const PROTOCOL_PLANNING_HORIZON = 20
const PROTOCOL_DELTA_T = 0.1
const PROTOCOL_SOLVER_TOL = 8e-3
const PROTOCOL_MAX_INNER_ITERS = 1000

Base.@kwdef struct StudyConfig
    profile::Symbol
    scenario_seeds::Vector{Int}
    sensitivity_seeds::Vector{Int}
    scaling_seeds::Vector{Int}
    formulations::Vector{Symbol} = collect(FORMULATIONS)
    modes::Vector{Symbol} = collect(MODES)
    num_mpc_steps::Int
    planning_horizon::Int
    Δt::Float64 = PROTOCOL_DELTA_T
    scenario_jitter::Float64 = 0.15
    order_seed::Int = 730_241
    bootstrap_seed::Int = 991_827
    bootstrap_replicates::Int
    sensitivity_steps::Vector{Int}
    sensitivity_amplitudes::Vector{Float64} = [1e-8, 1e-6, 1e-4]
    scaling_directions::Int
    scaling_epsilons::Vector{Float64} = [1e-4, 1e-3, 1e-2, 1e-1]
    reference_initialization::Symbol = REFERENCE_INITIALIZATION
    comparability_protocol::Symbol = COMPARABILITY_PROTOCOL
    requested_reference_tol::Float64 = PROTOCOL_SOLVER_TOL
    reference_acceptance_tol::Float64 = PROTOCOL_SOLVER_TOL
    reference_max_inner_iters::Int = PROTOCOL_MAX_INNER_ITERS
    replay_tol::Float64 = PROTOCOL_SOLVER_TOL
    replay_max_inner_iters::Int = PROTOCOL_MAX_INNER_ITERS
    η₀::Float64 = 1e-6
    η_max::Float64 = 1e2
    fd_codegen_chunk_size::Int = 128
    linear_solver::Symbol = :klu
    boundary_tol::Float64 = 1e-6
    material_root_rtol::Float64 = 1e-3
    svd_max_variable_dimension::Int = 800
    warmup::Bool = true
    solve_scaling_cases::Bool = true
    save_full_solutions::Bool = false
    output_root::String = joinpath("data", "selective_warmstart")
end

function preset_config(profile::Symbol)
    if profile === :smoke
        return StudyConfig(;
            profile,
            scenario_seeds = [101],
            sensitivity_seeds = [101],
            scaling_seeds = [101],
            num_mpc_steps = 3,
            planning_horizon = PROTOCOL_PLANNING_HORIZON,
            bootstrap_replicates = 500,
            sensitivity_steps = [2],
            scaling_directions = 1,
        )
    elseif profile === :pilot
        return StudyConfig(;
            profile,
            scenario_seeds = [101, 202],
            sensitivity_seeds = [101, 202],
            scaling_seeds = [101, 202],
            num_mpc_steps = 8,
            planning_horizon = PROTOCOL_PLANNING_HORIZON,
            bootstrap_replicates = 2_000,
            sensitivity_steps = [3, 6],
            scaling_directions = 2,
        )
    elseif profile === :full
        seeds = collect(1001:1010)
        return StudyConfig(;
            profile,
            scenario_seeds = seeds,
            sensitivity_seeds = seeds[1:2],
            scaling_seeds = seeds[1:3],
            num_mpc_steps = 20,
            planning_horizon = PROTOCOL_PLANNING_HORIZON,
            bootstrap_replicates = 10_000,
            sensitivity_steps = [5, 10, 15],
            scaling_directions = 3,
        )
    end
    throw(ArgumentError("Unknown profile $(profile); use smoke, pilot, or full."))
end

function config_dict(config::StudyConfig)
    Dict(
        "profile" => String(config.profile),
        "scenario_seeds" => config.scenario_seeds,
        "sensitivity_seeds" => config.sensitivity_seeds,
        "scaling_seeds" => config.scaling_seeds,
        "formulations" => String.(config.formulations),
        "modes" => String.(config.modes),
        "num_mpc_steps" => config.num_mpc_steps,
        "planning_horizon" => config.planning_horizon,
        "dt" => config.Δt,
        "scenario_jitter" => config.scenario_jitter,
        "order_seed" => config.order_seed,
        "bootstrap_seed" => config.bootstrap_seed,
        "bootstrap_replicates" => config.bootstrap_replicates,
        "sensitivity_steps" => config.sensitivity_steps,
        "sensitivity_amplitudes" => config.sensitivity_amplitudes,
        "scaling_directions" => config.scaling_directions,
        "scaling_epsilons" => config.scaling_epsilons,
        "reference_initialization" =>
            String(config.reference_initialization),
        "comparability_protocol" =>
            String(config.comparability_protocol),
        "requested_reference_tol" => config.requested_reference_tol,
        "reference_acceptance_tol" => config.reference_acceptance_tol,
        "reference_max_inner_iters" => config.reference_max_inner_iters,
        "replay_tol" => config.replay_tol,
        "replay_max_inner_iters" => config.replay_max_inner_iters,
        "eta0" => config.η₀,
        "eta_max" => config.η_max,
        "fd_codegen_chunk_size" => config.fd_codegen_chunk_size,
        "linear_solver" => String(config.linear_solver),
        "boundary_tol" => config.boundary_tol,
        "material_root_rtol" => config.material_root_rtol,
        "svd_max_variable_dimension" => config.svd_max_variable_dimension,
        "warmup" => config.warmup,
        "solve_scaling_cases" => config.solve_scaling_cases,
        "save_full_solutions" => config.save_full_solutions,
        "output_root" => config.output_root,
    )
end

function _int_vector(value)
    Int[x for x in value]
end

function _float_vector(value)
    Float64[x for x in value]
end

function config_from_dict(d)
    haskey(d, "comparability_protocol") || error(
        "Configuration predates the uniform T=20, dt=0.1, tol=0.008, " *
        "max_inner_iters=1000 comparability protocol. Start a fresh run; " *
        "legacy checkpoints cannot be resumed.",
    )
    comparability_protocol = Symbol(d["comparability_protocol"])
    comparability_protocol === COMPARABILITY_PROTOCOL || throw(
        ArgumentError(
            "Unsupported comparability protocol $(comparability_protocol); " *
            "expected $(COMPARABILITY_PROTOCOL).",
        ),
    )
    haskey(d, "reference_initialization") || error(
        "Configuration predates the cold-default-each-step reference protocol. " *
        "Start a fresh run; legacy all-dual-continuation checkpoints cannot be resumed.",
    )
    reference_initialization = Symbol(d["reference_initialization"])
    reference_initialization === REFERENCE_INITIALIZATION || throw(
        ArgumentError(
            "Unsupported reference initialization $(reference_initialization); " *
            "expected $(REFERENCE_INITIALIZATION).",
        ),
    )
    config = StudyConfig(;
        profile = Symbol(d["profile"]),
        scenario_seeds = _int_vector(d["scenario_seeds"]),
        sensitivity_seeds = _int_vector(d["sensitivity_seeds"]),
        scaling_seeds = _int_vector(d["scaling_seeds"]),
        formulations = Symbol.(d["formulations"]),
        modes = Symbol.(d["modes"]),
        num_mpc_steps = Int(d["num_mpc_steps"]),
        planning_horizon = Int(d["planning_horizon"]),
        Δt = Float64(d["dt"]),
        scenario_jitter = Float64(d["scenario_jitter"]),
        order_seed = Int(d["order_seed"]),
        bootstrap_seed = Int(d["bootstrap_seed"]),
        bootstrap_replicates = Int(d["bootstrap_replicates"]),
        sensitivity_steps = _int_vector(d["sensitivity_steps"]),
        sensitivity_amplitudes = _float_vector(d["sensitivity_amplitudes"]),
        scaling_directions = Int(d["scaling_directions"]),
        scaling_epsilons = _float_vector(d["scaling_epsilons"]),
        reference_initialization,
        comparability_protocol,
        requested_reference_tol = Float64(d["requested_reference_tol"]),
        reference_acceptance_tol = Float64(d["reference_acceptance_tol"]),
        reference_max_inner_iters = Int(d["reference_max_inner_iters"]),
        replay_tol = Float64(d["replay_tol"]),
        replay_max_inner_iters = Int(d["replay_max_inner_iters"]),
        η₀ = Float64(d["eta0"]),
        η_max = Float64(d["eta_max"]),
        fd_codegen_chunk_size = Int(d["fd_codegen_chunk_size"]),
        linear_solver = Symbol(d["linear_solver"]),
        boundary_tol = Float64(d["boundary_tol"]),
        material_root_rtol = Float64(d["material_root_rtol"]),
        svd_max_variable_dimension = Int(d["svd_max_variable_dimension"]),
        warmup = Bool(d["warmup"]),
        solve_scaling_cases = Bool(d["solve_scaling_cases"]),
        save_full_solutions = Bool(d["save_full_solutions"]),
        output_root = String(d["output_root"]),
    )
    _validate_study_protocol(config)
    config
end

load_config(path::AbstractString) = config_from_dict(TOML.parsefile(path))

function _config_text(config)
    io = IOBuffer()
    TOML.print(io, config_dict(config); sorted = true)
    String(take!(io))
end

_config_fingerprint(config) = bytes2hex(SHA.sha256(_config_text(config)))

function _validate_study_protocol(config)
    config.reference_initialization === REFERENCE_INITIALIZATION || throw(
        ArgumentError(
            "Unsupported reference initialization $(config.reference_initialization); " *
            "expected $(REFERENCE_INITIALIZATION).",
        ),
    )
    config.comparability_protocol === COMPARABILITY_PROTOCOL || throw(
        ArgumentError(
            "Unsupported comparability protocol $(config.comparability_protocol); " *
            "expected $(COMPARABILITY_PROTOCOL).",
        ),
    )
    actual = (;
        planning_horizon = config.planning_horizon,
        Δt = config.Δt,
        requested_reference_tol = config.requested_reference_tol,
        reference_acceptance_tol = config.reference_acceptance_tol,
        reference_max_inner_iters = config.reference_max_inner_iters,
        replay_tol = config.replay_tol,
        replay_max_inner_iters = config.replay_max_inner_iters,
        linear_solver = config.linear_solver,
        η₀ = config.η₀,
        η_max = config.η_max,
        warmup = config.warmup,
    )
    expected = (;
        planning_horizon = PROTOCOL_PLANNING_HORIZON,
        Δt = PROTOCOL_DELTA_T,
        requested_reference_tol = PROTOCOL_SOLVER_TOL,
        reference_acceptance_tol = PROTOCOL_SOLVER_TOL,
        reference_max_inner_iters = PROTOCOL_MAX_INNER_ITERS,
        replay_tol = PROTOCOL_SOLVER_TOL,
        replay_max_inner_iters = PROTOCOL_MAX_INNER_ITERS,
        linear_solver = :klu,
        η₀ = 1e-6,
        η_max = 1e2,
        warmup = true,
    )
    actual == expected || throw(
        ArgumentError(
            "Configuration violates $(COMPARABILITY_PROTOCOL). " *
            "Expected $(expected), got $(actual).",
        ),
    )
    nothing
end

function _stable_seed(parts...)
    h = UInt32(0x811c9dc5)
    for byte in codeunits(join(string.(parts), "|"))
        h = (h ⊻ UInt32(byte)) * UInt32(0x01000193)
    end
    Int(mod(h, UInt32(typemax(Int32) - 1))) + 1
end

function _atomic_write(path, text)
    mkpath(dirname(path))
    temporary = path * ".tmp." * string(getpid())
    open(temporary, "w") do io
        write(io, text)
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

function _git_read(args...)
    try
        command = Cmd(
            vcat(
                ["git", "-C", REPOSITORY_ROOT],
                string.(collect(args)),
            ),
        )
        readchomp(command)
    catch error
        "unavailable: " * sprint(showerror, error)
    end
end

_file_sha256(path) =
    isfile(path) ? bytes2hex(SHA.sha256(read(path))) : "missing"

function _provenance_relative_files()
    files = Set{String}()
    for relative_root in PROVENANCE_ROOTS
        absolute_root = joinpath(REPOSITORY_ROOT, relative_root)
        isdir(absolute_root) || continue
        for (directory, _, names) in walkdir(absolute_root)
            for name in names
                path = joinpath(directory, name)
                isfile(path) || continue
                push!(files, relpath(path, REPOSITORY_ROOT))
            end
        end
    end
    for relative_path in PROVENANCE_FILES
        isfile(joinpath(REPOSITORY_ROOT, relative_path)) &&
            push!(files, relative_path)
    end
    sort(collect(files))
end

function _measurement_code_state()
    relative_files = _provenance_relative_files()
    records = Dict{String, Any}[]
    fingerprint_input = IOBuffer()
    write(fingerprint_input, _git_read("rev-parse", "HEAD"), '\n')
    for relative_path in relative_files
        absolute_path = joinpath(REPOSITORY_ROOT, relative_path)
        digest = _file_sha256(absolute_path)
        push!(
            records,
            Dict(
                "path" => relative_path,
                "sha256" => digest,
                "bytes" => filesize(absolute_path),
            ),
        )
        write(fingerprint_input, relative_path, '\0', digest, '\n')
    end
    (;
        files = records,
        fingerprint =
            bytes2hex(SHA.sha256(take!(fingerprint_input))),
    )
end

function _git_diff_binary()
    try
        read(
            Cmd([
                "git",
                "-C",
                REPOSITORY_ROOT,
                "diff",
                "--binary",
                "HEAD",
                "--",
                ".",
            ]),
            String,
        )
    catch error
        "unavailable: " * sprint(showerror, error)
    end
end

function _snapshot_environment_and_provenance!(run_dir)
    environment_dir = joinpath(run_dir, "environment")
    provenance_dir = joinpath(run_dir, "provenance")
    mkpath(environment_dir)
    mkpath(provenance_dir)
    experiment_project =
        joinpath(REPOSITORY_ROOT, "experiments", "Project.toml")
    experiment_manifest =
        joinpath(REPOSITORY_ROOT, "experiments", "Manifest.toml")
    project_snapshot = joinpath(environment_dir, "Project.toml")
    manifest_snapshot = joinpath(environment_dir, "Manifest.toml")
    cp(experiment_project, project_snapshot; force = true)
    isfile(experiment_manifest) &&
        cp(experiment_manifest, manifest_snapshot; force = true)

    state = _measurement_code_state()
    for record in state.files
        relative_path = record["path"]
        source = joinpath(REPOSITORY_ROOT, relative_path)
        destination =
            joinpath(provenance_dir, "files", relative_path)
        mkpath(dirname(destination))
        cp(source, destination; force = true)
    end
    diff_text = _git_diff_binary()
    diff_path = joinpath(provenance_dir, "git_diff_binary.patch")
    _atomic_write(diff_path, diff_text)
    provenance = Dict(
        "created_at" => string(Dates.now()),
        "git_commit" => _git_read("rev-parse", "HEAD"),
        "git_branch" =>
            _git_read("rev-parse", "--abbrev-ref", "HEAD"),
        "measurement_code_fingerprint" => state.fingerprint,
        "git_diff_binary_sha256" => _file_sha256(diff_path),
        "project_snapshot_sha256" =>
            _file_sha256(project_snapshot),
        "manifest_snapshot_sha256" =>
            _file_sha256(manifest_snapshot),
        "files" => state.files,
    )
    _write_toml(
        joinpath(provenance_dir, "manifest.toml"),
        provenance,
    )
    provenance
end

function _record_provenance_drift!(run_dir)
    manifest_path =
        joinpath(run_dir, "provenance", "manifest.toml")
    isfile(manifest_path) || return nothing
    creation = TOML.parsefile(manifest_path)
    current_state = _measurement_code_state()
    project_path =
        joinpath(REPOSITORY_ROOT, "experiments", "Project.toml")
    manifest_path_current =
        joinpath(REPOSITORY_ROOT, "experiments", "Manifest.toml")
    current_project_sha = _file_sha256(project_path)
    current_manifest_sha = _file_sha256(manifest_path_current)
    creation_project_sha =
        get(creation, "project_snapshot_sha256", "missing")
    creation_manifest_sha =
        get(creation, "manifest_snapshot_sha256", "missing")
    drift = Dict(
        "checked_at" => string(Dates.now()),
        "creation_git_commit" =>
            get(creation, "git_commit", "unavailable"),
        "current_git_commit" => _git_read("rev-parse", "HEAD"),
        "creation_measurement_code_fingerprint" => get(
            creation,
            "measurement_code_fingerprint",
            "missing",
        ),
        "current_measurement_code_fingerprint" =>
            current_state.fingerprint,
        "measurement_code_drift_detected" =>
            current_state.fingerprint !=
            get(
                creation,
                "measurement_code_fingerprint",
                "missing",
            ),
        "creation_project_sha256" => creation_project_sha,
        "current_project_sha256" => current_project_sha,
        "creation_manifest_sha256" => creation_manifest_sha,
        "current_manifest_sha256" => current_manifest_sha,
        "environment_drift_detected" =>
            creation_project_sha != current_project_sha ||
            creation_manifest_sha != current_manifest_sha,
        "interpretation" =>
            "Raw measurements use the creation fingerprint; derived analysis/report regeneration uses the current fingerprint shown here.",
    )
    _write_toml(
        joinpath(run_dir, "provenance", "drift.toml"),
        drift,
    )
    drift
end

function _environment_dict(config, command)
    status = _git_read("status", "--porcelain=v1")
    package_status = IOBuffer()
    try
        Pkg.status(; io = package_status)
    catch error
        println(package_status, "Pkg.status failed: ", sprint(showerror, error))
    end
    Dict(
        "created_at" => string(Dates.now()),
        "command" => command,
        "git_commit" => _git_read("rev-parse", "HEAD"),
        "git_branch" => _git_read("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty" => !isempty(status),
        "git_status_porcelain" => status,
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_config" => sprint(show, BLAS.get_config()),
        "blas_threads" => BLAS.get_num_threads(),
        "platform" => string(Sys.KERNEL, "-", Sys.ARCH),
        "machine" => Sys.MACHINE,
        "cpu_threads" => Sys.CPU_THREADS,
        "word_size" => Sys.WORD_SIZE,
        "active_project" => something(Base.active_project(), ""),
        "project_sha256" => bytes2hex(
            SHA.sha256(read(joinpath(REPOSITORY_ROOT, "experiments", "Project.toml"))),
        ),
        "manifest_sha256" => begin
            path = joinpath(REPOSITORY_ROOT, "experiments", "Manifest.toml")
            isfile(path) ? bytes2hex(SHA.sha256(read(path))) : "missing"
        end,
        "package_status" => String(take!(package_status)),
        "config_sha256" => _config_fingerprint(config),
    )
end

function _write_toml(path, dictionary)
    io = IOBuffer()
    TOML.print(io, dictionary; sorted = true)
    _atomic_write(path, String(take!(io)))
end

function _prepare_run(config; run_dir = nothing, command = "")
    _validate_study_protocol(config)
    if isnothing(run_dir)
        run_id =
            Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS") * "_" * String(config.profile)
        run_dir = abspath(joinpath(config.output_root, String(config.profile), run_id))
        mkpath(run_dir)
        _atomic_write(joinpath(run_dir, "config.toml"), _config_text(config))
        provenance =
            _snapshot_environment_and_provenance!(run_dir)
        environment = _environment_dict(config, command)
        environment["project_snapshot"] =
            joinpath("environment", "Project.toml")
        environment["manifest_snapshot"] =
            joinpath("environment", "Manifest.toml")
        environment["measurement_code_fingerprint"] =
            provenance["measurement_code_fingerprint"]
        environment["provenance_manifest"] =
            joinpath("provenance", "manifest.toml")
        environment["git_diff_binary"] =
            joinpath("provenance", "git_diff_binary.patch")
        _write_toml(
            joinpath(run_dir, "environment.toml"),
            environment,
        )
    else
        run_dir = abspath(run_dir)
        isfile(joinpath(run_dir, "config.toml")) ||
            error("Resume directory has no config.toml: $(run_dir)")
        stored = load_config(joinpath(run_dir, "config.toml"))
        _config_fingerprint(stored) == _config_fingerprint(config) ||
            error("Resume configuration differs from the stored configuration.")
        _record_provenance_drift!(run_dir)
    end
    for subdir in (
        "raw",
        "checkpoints/references",
        "checkpoints/replay",
        "checkpoints/sequences",
        "checkpoints/scaling",
        "checkpoints/sensitivity",
        "figures",
    )
        mkpath(joinpath(run_dir, subdir))
    end
    run_dir
end

function _csv_escape(value)
    if value === nothing || value === missing
        return ""
    elseif value isa AbstractFloat
        return isfinite(value) ? repr(value) :
               isnan(value) ? "NaN" :
               value > 0 ? "Inf" : "-Inf"
    end
    text = value isa Symbol ? String(value) : string(value)
    # Incremental resume parsing is intentionally one physical line per row.
    text = replace(text, '\r' => "\\r", '\n' => "\\n")
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) ||
       occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    text
end

function _split_csv_line(line)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    i = firstindex(line)
    while i <= lastindex(line)
        char = line[i]
        if quoted
            if char == '"'
                next_i = nextind(line, i)
                if next_i <= lastindex(line) && line[next_i] == '"'
                    write(buffer, '"')
                    i = next_i
                else
                    quoted = false
                end
            else
                write(buffer, char)
            end
        elseif char == '"'
            quoted = true
        elseif char == ','
            push!(fields, String(take!(buffer)))
        else
            write(buffer, char)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buffer)))
    fields
end

function _csv_line_complete(line)
    quoted = false
    index = firstindex(line)
    while index <= lastindex(line)
        if line[index] == '"'
            next_index = nextind(line, index)
            if quoted &&
               next_index <= lastindex(line) &&
               line[next_index] == '"'
                index = next_index
            else
                quoted = !quoted
            end
        end
        index = nextind(line, index)
    end
    !quoted
end

function read_csv_rows(path)
    isfile(path) || return Dict{String, String}[]
    lines = readlines(path)
    isempty(lines) && return Dict{String, String}[]
    header = _split_csv_line(first(lines))
    rows = Dict{String, String}[]
    repaired_tail = false
    for (line_index, line) in enumerate(Iterators.drop(lines, 1))
        isempty(line) && continue
        values = _split_csv_line(line)
        valid =
            _csv_line_complete(line) &&
            length(values) == length(header)
        if !valid
            physical_index = line_index + 1
            if physical_index == length(lines)
                repaired_tail = true
                break
            end
            error("Malformed nonfinal CSV row $(physical_index) in $(path).")
        end
        push!(rows, Dict(zip(header, values)))
    end
    if repaired_tail
        valid_lines = lines[1:(length(rows)+1)]
        _atomic_write(path, join(valid_lines, "\n") * "\n")
    end
    rows
end

mutable struct IncrementalTable
    path::String
    columns::Vector{String}
    completed::Set{String}
end

function IncrementalTable(path, columns)
    rows = read_csv_rows(path)
    completed = Set(get(row, "case_id", "") for row in rows)
    IncrementalTable(path, columns, completed)
end

function append_row!(table::IncrementalTable, row)
    case_id = string(get(row, "case_id", ""))
    case_id in table.completed && return false
    mkpath(dirname(table.path))
    new_file = !isfile(table.path) || filesize(table.path) == 0
    open(table.path, "a") do io
        new_file && println(io, join(table.columns, ","))
        println(io, join((_csv_escape(get(row, column, "")) for column in table.columns), ","))
        flush(io)
    end
    push!(table.completed, case_id)
    true
end

const REFERENCE_COLUMNS = [
    "case_id",
    "profile",
    "formulation",
    "scenario_seed",
    "step",
    "sequence_driver",
    "instance_digest",
    "source",
    "solver_status",
    "reference_accepted",
    "requested_tol",
    "acceptance_tol",
    "initial_residual_norm2",
    "direct_residual_norm2",
    "direct_residual_normalized",
    "direct_residual_norm_inf",
    "solve_time_sec",
    "instrumented_solve_time_sec",
    "timing_solver_status",
    "total_inner_iters",
    "total_outer_iters",
    "final_epsilon",
    "error",
]

const REPLAY_COLUMNS = [
    "case_id",
    "profile",
    "formulation",
    "scenario_seed",
    "transition",
    "source_step",
    "destination_step",
    "sequence_driver",
    "source_instance_digest",
    "destination_instance_digest",
    "mode",
    "execution_order",
    "mode_order",
    "order_seed",
    "source_reference_status",
    "destination_reference_status",
    "source_reference_accepted",
    "destination_reference_accepted",
    "valid_reference_pair",
    "source_reference_residual",
    "destination_reference_residual",
    "innermost_margin",
    "preference_stratum",
    "initial_residual_norm2",
    "initial_residual_normalized",
    "initial_residual_norm_inf",
    "initial_residual_direct_discrepancy",
    "initial_stationarity_outer_norm2",
    "initial_stationarity_innermost_norm2",
    "initial_equality_norm2",
    "error_z",
    "error_lambda",
    "error_psi_out",
    "error_psi_in",
    "shift_quality_z",
    "shift_quality_lambda",
    "shift_quality_psi_out",
    "shift_quality_psi_in",
    "first_trial_residual_norm2",
    "first_accepted_residual_norm2",
    "first_attempt_alpha",
    "first_accepted_alpha",
    "first_iteration_backtracking_count",
    "regularization_change_count",
    "eta_retry_count",
    "full_step_fraction",
    "total_backtracking_count",
    "total_inner_iters",
    "total_outer_iters",
    "solve_time_sec",
    "instrumented_solve_time_sec",
    "solver_status",
    "timing_solver_status",
    "reported_final_residual_norm2",
    "direct_final_residual_norm2",
    "direct_final_residual_norm_inf",
    "direct_converged",
    "final_primal_distance",
    "final_primal_distance_normalized",
    "materially_different_from_reference",
    "klu_singular_retries",
    "svd_fallback_count",
    "failure_reason",
    "error",
]

const ROOT_COLUMNS = [
    "case_id",
    "profile",
    "formulation",
    "scenario_seed",
    "transition",
    "converged_mode_count",
    "max_pairwise_primal_distance",
    "max_pairwise_primal_distance_normalized",
    "mode_a",
    "mode_b",
    "materially_different_roots",
]

const SENSITIVITY_COLUMNS = [
    "case_id",
    "profile",
    "formulation",
    "scenario_seed",
    "step",
    "reference_status",
    "reference_accepted",
    "preference_stratum",
    "innermost_margin",
    "block",
    "direction_seed",
    "amplitude",
    "directional_residual_change_per_amplitude",
    "baseline_residual_norm2",
    "block_columns",
    "raw_frobenius_norm",
    "raw_frobenius_per_sqrt_column",
    "column_scaled_frobenius_norm",
    "column_scaled_frobenius_per_sqrt_column",
    "maximum_column_norm",
    "near_null_computed",
    "near_null_vector_count",
    "near_null_psi_in_energy_mean",
    "near_null_psi_in_energy_median",
    "near_null_psi_in_energy_max",
    "near_null_relative_threshold",
    "near_null_skip_reason",
    "error",
]

const SCALING_COLUMNS = [
    "case_id",
    "profile",
    "formulation",
    "scenario_seed",
    "direction",
    "direction_seed",
    "epsilon",
    "mode",
    "execution_order",
    "mode_order",
    "order_seed",
    "base_reference_status",
    "perturbed_reference_status",
    "base_reference_accepted",
    "perturbed_reference_accepted",
    "valid_reference_pair",
    "perturbed_reference_residual",
    "baseline_initial_residual_norm2",
    "initial_residual_norm2",
    "initial_residual_normalized",
    "initial_residual_norm_inf",
    "parameter_induced_residual_change_norm2",
    "parameter_induced_residual_change_normalized",
    "numerical_floor",
    "above_numerical_floor",
    "first_accepted_residual_norm2",
    "first_accepted_alpha",
    "total_inner_iters",
    "total_backtracking_count",
    "eta_retry_count",
    "full_step_fraction",
    "solve_time_sec",
    "instrumented_solve_time_sec",
    "solver_status",
    "timing_solver_status",
    "direct_final_residual_norm2",
    "direct_converged",
    "final_primal_distance_normalized",
    "error",
]

const KKT_COLUMNS = [
    "case_id",
    "profile",
    "formulation",
    "planning_horizon",
    "kkt_dimension",
    "variable_dimension",
    "z_dimension",
    "lambda_dimension",
    "psi_out_dimension",
    "psi_in_dimension",
    "preference_slack_dimension",
    "interior_point_slack_dimension",
    "inequality_dual_dimension",
    "unclassified_dimension",
    "build_time_sec",
]

function _scenario_and_problem(config)
    scenario = RAC.demo_scenario_config(;
        planning_horizon = config.planning_horizon,
        Δt = config.Δt,
        use_running_goal_cost = false,
    )
    setup = RAC.get_setup(scenario)
    primal_dimensions = [
        (dynamics.state_dimension + dynamics.control_dimension) *
        config.planning_horizon for dynamics in scenario.dynamics
    ]
    (; scenario, setup.problem, setup.flatten_parameters, primal_dimensions)
end

function _build_kkt(problem, formulation, config)
    generator =
        formulation === :reduced ?
        ReducedGOOP.generate_slacked_reduced_kkt_system :
        formulation === :quasi ?
        ReducedGOOP.generate_slacked_quasi_kkt_system :
        throw(ArgumentError("Unsupported formulation $(formulation)."))
    backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend()
    build_time = @elapsed kkt = generator(
        problem;
        backend,
        backend_options = (;),
        codegen = :fast_differentiation,
        fd_codegen_chunk_size = config.fd_codegen_chunk_size,
    )
    blocks = ReducedGOOP.kkt_variable_blocks(kkt)
    (; kkt, blocks, build_time)
end

function _solver_options(config)
    _validate_study_protocol(config)
    ReducedGOOP.InteriorPointOptions(;
        tol = PROTOCOL_SOLVER_TOL,
        η₀ = config.η₀,
        η_max = config.η_max,
        ϵ₀ = EPSILON0,
        max_inner_iters = PROTOCOL_MAX_INNER_ITERS,
        max_outer_iters = 1,
        tightening_rate = 1.2,
        loosening_rate = 3.0,
        min_stepsize = 1e-20,
        linesearch = :backtracking,
        record_convergence = false,
        record_condition_number = false,
        eta_retry_growth = 2.0,
        ρ_low = 0.75,
        ρ_high = 0.75,
        tsvd_threshold = 0.0,
        use_marquardt_scaling = false,
        linear_solver = config.linear_solver,
        klu_singularity_eta_growth = 100.0,
        armijo_constant = 1e-4,
        reuse_factorization_iters = 0,
        verbose = false,
    )
end

function _solver_options_dict(config)
    options = _solver_options(config)
    Dict(
        String(name) => begin
            value = getproperty(options, name)
            value isa Symbol ? String(value) : value
        end for name in propertynames(options)
    )
end

function _residual(kkt, z, θ; epsilon = EPSILON0)
    values = zeros(kkt.kkt_dimension)
    kkt.F!(values, z; θ, ϵ = epsilon, η = 0.0)
    (;
        values,
        norm2 = norm(values),
        normalized = norm(values) / sqrt(length(values)),
        norm_inf = norm(values, Inf),
    )
end

function _initial_residual_metrics(kkt, warmstart, θ; epsilon = EPSILON0)
    direct = _residual(kkt, warmstart, θ; epsilon)
    (;
        residual = direct,
        initial_residual_norm2 = direct.norm2,
        initial_residual_normalized = direct.normalized,
        initial_residual_norm_inf = direct.norm_inf,
    )
end

function _trace_summary(events)
    initial = findfirst(event -> event.event === :initial_residual, events)
    trials = filter(event -> event.event === :line_search_trial, events)
    accepted = filter(event -> event.event === :accepted_step, events)
    eta_changes = filter(
        event -> event.event === :eta_change && get(event, :changed, false),
        events,
    )
    finish_index = findlast(event -> event.event === :finish, events)
    failure_index = findlast(event -> event.event === :failure, events)
    first_trial = isempty(trials) ? nothing : first(trials)
    first_step = isempty(accepted) ? nothing : first(accepted)
    finish = isnothing(finish_index) ? nothing : events[finish_index]
    failure = isnothing(failure_index) ? nothing : events[failure_index]
    (;
        initial = isnothing(initial) ? nothing : events[initial],
        first_trial,
        first_step,
        finish,
        failure,
        regularization_change_count = length(eta_changes),
    )
end

mutable struct TraceCollector
    events::Vector{Any}
end

(collector::TraceCollector)(event) = push!(collector.events, event)

function _solve_with_trace(kkt, θ, warmstart, options)
    collector = TraceCollector(Any[])
    output = nothing
    instrumented_solve_time = @elapsed output = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀ = copy(warmstart),
        options,
        trace_hook = collector,
    )
    timing_output = nothing
    solve_time = @elapsed timing_output = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀ = copy(warmstart),
        options,
        trace_hook = nothing,
    )
    (;
        output,
        timing_output,
        events = collector.events,
        trace = _trace_summary(collector.events),
        solve_time,
        instrumented_solve_time,
    )
end

_event_value(event, name, default = NaN) =
    isnothing(event) ? default : get(event, name, default)

function _result_metrics(result)
    output = result.output
    trace = result.trace
    finish = trace.finish
    failure = trace.failure
    (;
        initial_residual_norm2 =
            _event_value(trace.initial, :residual_norm2),
        initial_residual_norm_inf =
            _event_value(trace.initial, :residual_norm_inf),
        first_trial_residual_norm2 =
            _event_value(trace.first_trial, :residual_norm2),
        first_accepted_residual_norm2 =
            _event_value(trace.first_step, :residual_after_norm2),
        first_attempt_alpha =
            _event_value(trace.first_step, :first_attempt_alpha),
        first_accepted_alpha =
            _event_value(trace.first_step, :accepted_alpha),
        first_iteration_backtracking_count =
            _event_value(trace.first_step, :backtracking_count, 0),
        regularization_change_count = trace.regularization_change_count,
        eta_retry_count =
            _event_value(finish, :total_eta_retry_count, 0),
        full_step_fraction =
            _event_value(finish, :full_step_fraction, 0.0),
        total_backtracking_count =
            _event_value(finish, :total_backtracking_count, 0),
        total_inner_iters =
            _event_value(finish, :total_iters, output.total_iters),
        total_outer_iters =
            _event_value(finish, :actual_outer_iters, max(output.outer_iters - 1, 1)),
        solve_time_sec = result.solve_time,
        instrumented_solve_time_sec = result.instrumented_solve_time,
        solver_status = output.status,
        timing_solver_status = result.timing_output.status,
        reported_final_residual_norm2 = output.kkt_error,
        direct_final_residual_norm2 =
            _event_value(finish, :final_residual_norm2, output.kkt_error),
        direct_final_residual_norm_inf =
            _event_value(finish, :final_residual_norm_inf),
        klu_singular_retries = output.klu_singular_retries,
        svd_fallback_count = output.svd_fallback_count,
        failure_reason =
            isnothing(failure) ? "" : string(get(failure, :reason, "")),
    )
end

function _initial_instance(scenario, scenario_seed, config)
    rng = MersenneTwister(
        _stable_seed(config.order_seed, :scenario, scenario_seed),
    )
    # The two grippers must receive the same rigid translation; independent
    # perturbations would violate the hard handle-separation equality at t = 1.
    direction = randn(rng, 5)
    direction ./= norm(direction)
    perturbation = config.scenario_jitter .* direction
    (;
        initial_state1 =
            copy(scenario.base_initial_state1) .+ perturbation[1:3],
        initial_state2 =
            copy(scenario.base_initial_state2) .+ perturbation[1:3],
        initial_state3 =
            copy(scenario.initial_state3) .+ [perturbation[4], perturbation[5], 0.0],
        initial_control1 = zeros(3),
        initial_control2 = zeros(3),
        initial_control3 = zeros(3),
    )
end

function _instance_digest(instance)
    values = Float64[]
    for name in (
        :initial_state1,
        :initial_state2,
        :initial_state3,
        :initial_control1,
        :initial_control2,
        :initial_control3,
    )
        append!(values, Float64.(getproperty(instance, name)))
    end
    bytes2hex(
        SHA.sha256(
            join((@sprintf("%.17g", value) for value in values), "|"),
        ),
    )
end

function _next_instance(strategies)
    combined_state = collect(strategies[1].xs[2])
    combined_control = collect(strategies[1].us[2])
    (;
        initial_state1 = combined_state[1:3],
        initial_state2 = combined_state[4:6],
        initial_state3 = collect(strategies[2].xs[2]),
        initial_control1 = combined_control[1:3],
        initial_control2 = combined_control[4:6],
        initial_control3 = collect(strategies[2].us[2]),
    )
end

function _shifted_primal(strategies, scenario, config)
    shifted = ReducedGOOP.shift_receding_trajectories(
        strategies,
        scenario.dynamics,
        config.planning_horizon,
    )
    RAC.flatten_warmstart_solution(
        config.planning_horizon,
        [strategy.xs for strategy in shifted],
        [strategy.us for strategy in shifted],
    )
end

function _warmup!(kkt, θ, primal_warmstart, full_warmstart, config)
    config.warmup || return
    options = _solver_options(config)
    for warmstart in (primal_warmstart, full_warmstart)
        try
            collector = TraceCollector(Any[])
            ReducedGOOP.solve(
                ReducedGOOP.InteriorPoint(),
                kkt,
                θ;
                z₀ = copy(warmstart),
                options,
                trace_hook = collector,
            )
            ReducedGOOP.solve(
                ReducedGOOP.InteriorPoint(),
                kkt,
                θ;
                z₀ = copy(warmstart),
                options,
                trace_hook = nothing,
            )
        catch error
            @warn "Discarded warm-up solve failed" exception = error
        end
    end
    nothing
end

function _reference_case_id(formulation, scenario_seed, step)
    "reference__$(formulation)__seed$(scenario_seed)__step$(step)"
end

function _reference_path(run_dir, formulation, scenario_seed, step)
    joinpath(
        run_dir,
        "checkpoints",
        "references",
        String(formulation),
        "seed_$(scenario_seed)",
        "step_$(step).jld2",
    )
end

function _sequence_path(run_dir, scenario_seed)
    joinpath(
        run_dir,
        "checkpoints",
        "sequences",
        "seed_$(scenario_seed).jld2",
    )
end

function _persist_canonical_sequence!(
    run_dir,
    scenario_seed,
    sequence_driver,
    references,
)
    instances = Any[
        haskey(reference, "instance") ? reference["instance"] : nothing for
        reference in references
    ]
    digests = String[
        isnothing(instance) ? "" : _instance_digest(instance) for
        instance in instances
    ]
    checkpoint = Dict{String, Any}(
        "scenario_seed" => scenario_seed,
        "sequence_driver" => sequence_driver,
        "instances" => instances,
        "instance_digests" => digests,
        "sequence_digest" =>
            bytes2hex(SHA.sha256(join(digests, "|"))),
        "driver_reference_accepted" => [
            get(reference, "accepted", false) for reference in references
        ],
    )
    _atomic_save(_sequence_path(run_dir, scenario_seed), checkpoint)
    checkpoint
end

function _reference_placeholder(
    config,
    formulation,
    scenario_seed,
    step,
    source,
    error,
    ;
    sequence_driver = formulation,
    instance = nothing,
)
    instance_digest =
        isnothing(instance) ? "" : _instance_digest(instance)
    row = Dict{String, Any}(
        "case_id" => _reference_case_id(formulation, scenario_seed, step),
        "profile" => config.profile,
        "formulation" => formulation,
        "scenario_seed" => scenario_seed,
        "step" => step,
        "sequence_driver" => sequence_driver,
        "instance_digest" => instance_digest,
        "source" => source,
        "solver_status" => :not_run,
        "reference_accepted" => false,
        "requested_tol" => config.requested_reference_tol,
        "acceptance_tol" => config.reference_acceptance_tol,
        "error" => error,
    )
    checkpoint = Dict{String, Any}(
        "row" => row,
        "accepted" => false,
        "status" => :not_run,
        "sequence_driver" => sequence_driver,
        "instance_digest" => instance_digest,
        "error" => error,
    )
    !isnothing(instance) && (checkpoint["instance"] = instance)
    checkpoint
end

function _reference_initialization(current_instance, scenario, step)
    step >= 1 ||
        throw(ArgumentError("Reference step must be positive; got $(step)."))
    (;
        source = :cold_default,
        warmstart = RAC.build_default_warmstart(
            current_instance,
            scenario,
        ).warmstart_solution,
    )
end

function _ensure_reference_sequence!(
    reference_table,
    run_dir,
    formulation,
    scenario_seed,
    scenario,
    problem,
    flatten_parameters,
    primal_dimensions,
    kkt,
    config,
    ;
    canonical_instances = nothing,
    sequence_driver = formulation,
)
    references = Vector{Dict{String, Any}}(undef, config.num_mpc_steps)
    current_instance =
        isnothing(canonical_instances) ?
        _initial_instance(scenario, scenario_seed, config) :
        nothing
    blocked_error = nothing
    for step in 1:config.num_mpc_steps
        path = _reference_path(run_dir, formulation, scenario_seed, step)
        if !isnothing(canonical_instances)
            if step > length(canonical_instances) ||
               isnothing(canonical_instances[step])
                error_text =
                    "canonical sequence unavailable because the $(sequence_driver) driver did not produce step $(step)"
                checkpoint = _reference_placeholder(
                    config,
                    formulation,
                    scenario_seed,
                    step,
                    :canonical_sequence_unavailable,
                    error_text;
                    sequence_driver,
                )
                !isfile(path) && _atomic_save(path, checkpoint)
                references[step] = checkpoint
                append_row!(reference_table, checkpoint["row"])
                continue
            end
            current_instance = canonical_instances[step]
        end
        if isfile(path)
            checkpoint = JLD2.load_object(path)
            expected_digest = _instance_digest(current_instance)
            actual_digest = get(checkpoint, "instance_digest", "")
            (isempty(actual_digest) || actual_digest == expected_digest) ||
                error(
                    "Reference checkpoint $(path) has instance digest $(actual_digest), expected $(expected_digest).",
                )
            if !haskey(checkpoint, "sequence_driver") ||
               isempty(actual_digest)
                checkpoint["sequence_driver"] = sequence_driver
                checkpoint["instance_digest"] = expected_digest
                checkpoint["row"]["sequence_driver"] = sequence_driver
                checkpoint["row"]["instance_digest"] = expected_digest
                _atomic_save(path, checkpoint)
            end
            references[step] = checkpoint
            append_row!(reference_table, checkpoint["row"])
            if isnothing(canonical_instances) &&
               get(checkpoint, "accepted", false) &&
               haskey(checkpoint, "next_instance")
                current_instance = checkpoint["next_instance"]
            end
            checkpoint_usable =
                get(checkpoint, "accepted", false) &&
                haskey(checkpoint, "z") &&
                haskey(checkpoint, "shifted_primal") &&
                (
                    !isnothing(canonical_instances) ||
                    haskey(checkpoint, "next_instance")
                )
            if !checkpoint_usable
                blocked_error = get(checkpoint, "error", "reference unavailable")
            end
            continue
        end

        if !isnothing(blocked_error)
            checkpoint = _reference_placeholder(
                config,
                formulation,
                scenario_seed,
                step,
                :blocked,
                "blocked by earlier reference failure: $(blocked_error)",
                ;
                sequence_driver,
                instance =
                    isnothing(canonical_instances) ? nothing :
                    current_instance,
            )
            _atomic_save(path, checkpoint)
            references[step] = checkpoint
            append_row!(reference_table, checkpoint["row"])
            continue
        end

        parameters =
            RAC.build_instance_parameters(flatten_parameters, current_instance, scenario)
        initialization =
            _reference_initialization(current_instance, scenario, step)
        source = initialization.source
        warmstart = initialization.warmstart

        try
            result = _solve_with_trace(
                kkt,
                parameters.θ,
                warmstart,
                _solver_options(config),
            )
            output = result.output
            direct = _residual(kkt, output.z, parameters.θ; epsilon = output.ϵ)
            accepted =
                isfinite(direct.norm2) &&
                direct.norm2 <= config.reference_acceptance_tol
            strategies = RAC.extract_player_strategies(
                output.z[kkt.primal_dims],
                primal_dimensions,
                scenario.dynamics,
            )
            shifted_primal = _shifted_primal(strategies, scenario, config)
            next_instance = _next_instance(strategies)
            summary = _trace_summary(result.events)
            row = Dict{String, Any}(
                "case_id" => _reference_case_id(formulation, scenario_seed, step),
                "profile" => config.profile,
                "formulation" => formulation,
                "scenario_seed" => scenario_seed,
                "step" => step,
                "sequence_driver" => sequence_driver,
                "instance_digest" => _instance_digest(current_instance),
                "source" => source,
                "solver_status" => output.status,
                "reference_accepted" => accepted,
                "requested_tol" => config.requested_reference_tol,
                "acceptance_tol" => config.reference_acceptance_tol,
                "initial_residual_norm2" =>
                    _event_value(summary.initial, :residual_norm2),
                "direct_residual_norm2" => direct.norm2,
                "direct_residual_normalized" => direct.normalized,
                "direct_residual_norm_inf" => direct.norm_inf,
                "solve_time_sec" => result.solve_time,
                "instrumented_solve_time_sec" =>
                    result.instrumented_solve_time,
                "timing_solver_status" => result.timing_output.status,
                "total_inner_iters" => output.total_iters,
                "total_outer_iters" => _event_value(
                    summary.finish,
                    :actual_outer_iters,
                    max(output.outer_iters - 1, 1),
                ),
                "final_epsilon" => output.ϵ,
                "error" => "",
            )
            checkpoint = Dict{String, Any}(
                "row" => row,
                "accepted" => accepted,
                "status" => output.status,
                "sequence_driver" => sequence_driver,
                "instance_digest" => _instance_digest(current_instance),
                "direct_residual_norm2" => direct.norm2,
                "instance" => current_instance,
                "parameters" => parameters,
                "z" => copy(output.z),
                "x" => copy(output.z[kkt.primal_dims]),
                "strategies" => strategies,
                "shifted_primal" => shifted_primal,
                "next_instance" => next_instance,
                "events" => result.events,
            )
            _atomic_save(path, checkpoint)
            append_row!(reference_table, row)
            references[step] = checkpoint
            if accepted
                isnothing(canonical_instances) &&
                    (current_instance = next_instance)
            else
                blocked_error =
                    "reference direct residual $(direct.norm2) exceeds acceptance tolerance $(config.reference_acceptance_tol)"
            end
        catch error
            error_text = sprint(showerror, error, catch_backtrace())
            checkpoint = _reference_placeholder(
                config,
                formulation,
                scenario_seed,
                step,
                source,
                error_text,
                ;
                sequence_driver,
                instance = current_instance,
            )
            _atomic_save(path, checkpoint)
            append_row!(reference_table, checkpoint["row"])
            references[step] = checkpoint
            blocked_error = error_text
        end
    end
    references
end

function _preference_stratum(problem, x, θ, config)
    x_block = BlockArray(collect(x), problem.primal_dims)
    θ_block = BlockArray(collect(θ), problem.parameter_dims)
    margins = Float64[]
    for player in 1:problem.num_players
        isempty(problem.preferences[player]) && continue
        if problem.is_prioritized_constraint[player][end]
            values = problem.preferences[player][end](x_block, θ_block)
            append!(margins, Float64.(values))
        end
    end
    isempty(margins) && return (; margin = NaN, stratum = :not_applicable)
    margin = minimum(margins)
    stratum =
        margin < -config.boundary_tol ? :violated :
        margin <= config.boundary_tol ? :boundary : :satisfied
    (; margin, stratum)
end

function _equation_block_indices(problem, kkt)
    ordinary_inequalities =
        sum(problem.inequality_dims) + problem.shared_inequality_dims
    ordinary_inequalities == 0 || return nothing
    stationarity_outer = Int[]
    stationarity_innermost = Int[]
    equality = Int[]
    offset = 0
    for player in 1:problem.num_players
        primal_dimension = problem.primal_dims[player]
        num_levels = length(problem.preferences[player])
        for level in 1:num_levels
            rows = collect((offset+1):(offset+primal_dimension))
            append!(
                level == num_levels ? stationarity_innermost : stationarity_outer,
                rows,
            )
            offset += primal_dimension
        end
        equality_dimension = problem.equality_dims[player]
        append!(equality, (offset+1):(offset+equality_dimension))
        offset += equality_dimension
    end
    if problem.shared_equality_dims > 0
        append!(equality, (offset+1):(offset+problem.shared_equality_dims))
        offset += problem.shared_equality_dims
    end
    offset == kkt.kkt_dimension || return nothing
    (; stationarity_outer, stationarity_innermost, equality)
end

function _equation_residual_metrics(problem, kkt, residual_values)
    blocks = _equation_block_indices(problem, kkt)
    isnothing(blocks) && return (;
        stationarity_outer = NaN,
        stationarity_innermost = NaN,
        equality = NaN,
    )
    (;
        stationarity_outer = norm(view(residual_values, blocks.stationarity_outer)),
        stationarity_innermost =
            norm(view(residual_values, blocks.stationarity_innermost)),
        equality = norm(view(residual_values, blocks.equality)),
    )
end

function _normalized_block_error(initial, destination, dims)
    isempty(dims) && return 0.0
    norm(view(initial, dims) .- view(destination, dims)) /
    max(1.0, norm(view(destination, dims)))
end

function _shift_quality(transported, destination, dims)
    isempty(dims) && return 0.0
    norm(view(transported, dims) .- view(destination, dims)) /
    (norm(view(destination, dims)) + eps(Float64))
end

function _transported_source(source_z, shifted_primal, blocks)
    transported = copy(source_z)
    transported[blocks.z] .= shifted_primal
    transported
end

function _mode_order(config, formulation, scenario_seed, transition, experiment)
    seed = _stable_seed(
        config.order_seed,
        experiment,
        formulation,
        scenario_seed,
        transition,
    )
    order = copy(config.modes)
    shuffle!(MersenneTwister(seed), order)
    (; seed, order)
end

function _replay_case_id(formulation, scenario_seed, transition, mode)
    "replay__$(formulation)__seed$(scenario_seed)__transition$(transition)__$(mode)"
end

function _unavailable_replay_row(
    config,
    formulation,
    scenario_seed,
    transition,
    mode,
    execution_order,
    mode_order,
    order_seed,
    source,
    destination,
)
    Dict{String, Any}(
        "case_id" => _replay_case_id(formulation, scenario_seed, transition, mode),
        "profile" => config.profile,
        "formulation" => formulation,
        "scenario_seed" => scenario_seed,
        "transition" => transition,
        "source_step" => transition,
        "destination_step" => transition + 1,
        "sequence_driver" =>
            get(source, "sequence_driver", formulation),
        "source_instance_digest" =>
            get(source, "instance_digest", ""),
        "destination_instance_digest" =>
            get(destination, "instance_digest", ""),
        "mode" => mode,
        "execution_order" => execution_order,
        "mode_order" => mode_order,
        "order_seed" => order_seed,
        "source_reference_status" => get(source, "status", :unavailable),
        "destination_reference_status" =>
            get(destination, "status", :unavailable),
        "source_reference_accepted" => get(source, "accepted", false),
        "destination_reference_accepted" =>
            get(destination, "accepted", false),
        "valid_reference_pair" => false,
        "solver_status" => :not_run_reference_unavailable,
        "direct_converged" => false,
        "error" => "source or destination reference unavailable",
    )
end

function _replay_path(run_dir, formulation, scenario_seed, transition, mode)
    joinpath(
        run_dir,
        "checkpoints",
        "replay",
        String(formulation),
        "seed_$(scenario_seed)",
        "transition_$(transition)__$(mode).jld2",
    )
end

function _replay_case!(
    replay_table,
    run_dir,
    formulation,
    scenario_seed,
    transition,
    mode,
    execution_order,
    mode_order_text,
    order_seed,
    source,
    destination,
    warmstart,
    transported_source,
    problem,
    kkt,
    blocks,
    config,
)
    path = _replay_path(run_dir, formulation, scenario_seed, transition, mode)
    if isfile(path)
        checkpoint = JLD2.load_object(path)
        append_row!(replay_table, checkpoint["row"])
        return checkpoint
    end

    source_z = source["z"]
    destination_z = destination["z"]
    destination_primal = view(destination_z, blocks.z)
    θ = destination["parameters"].θ
    initial_metrics = _initial_residual_metrics(kkt, warmstart, θ)
    initial = initial_metrics.residual
    equation_residuals =
        _equation_residual_metrics(problem, kkt, initial.values)
    preference = _preference_stratum(
        problem,
        destination_primal,
        θ,
        config,
    )

    base_row = Dict{String, Any}(
        "case_id" =>
            _replay_case_id(formulation, scenario_seed, transition, mode),
        "profile" => config.profile,
        "formulation" => formulation,
        "scenario_seed" => scenario_seed,
        "transition" => transition,
        "source_step" => transition,
        "destination_step" => transition + 1,
        "sequence_driver" => source["sequence_driver"],
        "source_instance_digest" => source["instance_digest"],
        "destination_instance_digest" =>
            destination["instance_digest"],
        "mode" => mode,
        "execution_order" => execution_order,
        "mode_order" => mode_order_text,
        "order_seed" => order_seed,
        "source_reference_status" => source["status"],
        "destination_reference_status" => destination["status"],
        "source_reference_accepted" => source["accepted"],
        "destination_reference_accepted" => destination["accepted"],
        "valid_reference_pair" => true,
        "source_reference_residual" => source["direct_residual_norm2"],
        "destination_reference_residual" =>
            destination["direct_residual_norm2"],
        "innermost_margin" => preference.margin,
        "preference_stratum" => preference.stratum,
        "initial_residual_norm2" =>
            initial_metrics.initial_residual_norm2,
        "initial_residual_normalized" =>
            initial_metrics.initial_residual_normalized,
        "initial_residual_norm_inf" =>
            initial_metrics.initial_residual_norm_inf,
        "initial_stationarity_outer_norm2" =>
            equation_residuals.stationarity_outer,
        "initial_stationarity_innermost_norm2" =>
            equation_residuals.stationarity_innermost,
        "initial_equality_norm2" => equation_residuals.equality,
        "error_z" =>
            _normalized_block_error(warmstart, destination_z, blocks.z),
        "error_lambda" =>
            _normalized_block_error(warmstart, destination_z, blocks.λ),
        "error_psi_out" =>
            _normalized_block_error(warmstart, destination_z, blocks.ψ_out),
        "error_psi_in" =>
            _normalized_block_error(warmstart, destination_z, blocks.ψ_in),
        "shift_quality_z" =>
            _shift_quality(transported_source, destination_z, blocks.z),
        "shift_quality_lambda" =>
            _shift_quality(transported_source, destination_z, blocks.λ),
        "shift_quality_psi_out" =>
            _shift_quality(transported_source, destination_z, blocks.ψ_out),
        "shift_quality_psi_in" =>
            _shift_quality(transported_source, destination_z, blocks.ψ_in),
    )

    try
        result = _solve_with_trace(
            kkt,
            θ,
            warmstart,
            _solver_options(config),
        )
        metrics = _result_metrics(result)
        direct =
            _residual(kkt, result.output.z, θ; epsilon = result.output.ϵ)
        final_primal = copy(view(result.output.z, blocks.z))
        primal_distance = norm(final_primal .- destination_primal)
        primal_distance_normalized =
            primal_distance / max(1.0, norm(destination_primal))
        row = copy(base_row)
        for name in propertynames(metrics)
            row[string(name)] = getproperty(metrics, name)
        end
        row["initial_residual_normalized"] =
            metrics.initial_residual_norm2 / sqrt(kkt.kkt_dimension)
        row["initial_residual_direct_discrepancy"] =
            abs(metrics.initial_residual_norm2 - initial.norm2)
        row["direct_final_residual_norm2"] = direct.norm2
        row["direct_final_residual_norm_inf"] = direct.norm_inf
        direct_converged =
            isfinite(direct.norm2) && direct.norm2 <= config.replay_tol
        row["direct_converged"] = direct_converged
        row["final_primal_distance"] = primal_distance
        row["final_primal_distance_normalized"] =
            primal_distance_normalized
        row["materially_different_from_reference"] =
            primal_distance_normalized > config.material_root_rtol
        row["error"] = ""
        checkpoint = Dict{String, Any}(
            "row" => row,
            "events" => result.events,
            "final_primal" => final_primal,
            "converged" => direct_converged,
            "direct_final_residual_norm2" => direct.norm2,
        )
        config.save_full_solutions &&
            (checkpoint["final_z"] = copy(result.output.z))
        _atomic_save(path, checkpoint)
        append_row!(replay_table, row)
        return checkpoint
    catch error
        row = copy(base_row)
        row["solver_status"] = :exception
        row["direct_converged"] = false
        row["failure_reason"] = :exception
        row["error"] = sprint(showerror, error, catch_backtrace())
        checkpoint = Dict{String, Any}(
            "row" => row,
            "converged" => false,
            "error" => row["error"],
        )
        _atomic_save(path, checkpoint)
        append_row!(replay_table, row)
        return checkpoint
    end
end

function _root_case_id(formulation, scenario_seed, transition)
    "roots__$(formulation)__seed$(scenario_seed)__transition$(transition)"
end

function _root_spread_row(
    config,
    formulation,
    scenario_seed,
    transition,
    replay_checkpoints,
)
    available = Tuple{Symbol, Vector{Float64}}[]
    for mode in config.modes
        checkpoint = get(replay_checkpoints, mode, nothing)
        if !isnothing(checkpoint) &&
           get(checkpoint, "converged", false) &&
           haskey(checkpoint, "final_primal")
            push!(available, (mode, checkpoint["final_primal"]))
        end
    end
    maximum_distance = 0.0
    maximum_normalized = 0.0
    maximum_modes = ("", "")
    for i in eachindex(available), j in (i+1):length(available)
        mode_a, primal_a = available[i]
        mode_b, primal_b = available[j]
        distance = norm(primal_a .- primal_b)
        normalized =
            distance / max(1.0, norm(primal_a), norm(primal_b))
        if normalized > maximum_normalized
            maximum_distance = distance
            maximum_normalized = normalized
            maximum_modes = (String(mode_a), String(mode_b))
        end
    end
    Dict{String, Any}(
        "case_id" => _root_case_id(formulation, scenario_seed, transition),
        "profile" => config.profile,
        "formulation" => formulation,
        "scenario_seed" => scenario_seed,
        "transition" => transition,
        "converged_mode_count" => length(available),
        "max_pairwise_primal_distance" => maximum_distance,
        "max_pairwise_primal_distance_normalized" => maximum_normalized,
        "mode_a" => maximum_modes[1],
        "mode_b" => maximum_modes[2],
        "materially_different_roots" =>
            maximum_normalized > config.material_root_rtol,
    )
end

function _run_replay_transitions!(
    replay_table,
    root_table,
    run_dir,
    formulation,
    scenario_seed,
    references,
    scenario,
    problem,
    kkt,
    blocks,
    config,
)
    for transition in 1:(config.num_mpc_steps-1)
        source = references[transition]
        destination = references[transition+1]
        ordering = _mode_order(
            config,
            formulation,
            scenario_seed,
            transition,
            :replay,
        )
        mode_order_text = join(String.(ordering.order), ";")
        valid =
            get(source, "accepted", false) &&
            get(destination, "accepted", false) &&
            all(
                key -> haskey(source, key),
                ("z", "shifted_primal"),
            ) &&
            all(key -> haskey(destination, key), ("z", "parameters"))
        if !valid
            for (execution_order, mode) in enumerate(ordering.order)
                row = _unavailable_replay_row(
                    config,
                    formulation,
                    scenario_seed,
                    transition,
                    mode,
                    execution_order,
                    mode_order_text,
                    ordering.seed,
                    source,
                    destination,
                )
                append_row!(replay_table, row)
            end
            append_row!(
                root_table,
                _root_spread_row(
                    config,
                    formulation,
                    scenario_seed,
                    transition,
                    Dict{Symbol, Any}(),
                ),
            )
            continue
        end

        source_z = source["z"]
        shifted_primal = source["shifted_primal"]
        transported_source =
            _transported_source(source_z, shifted_primal, blocks)
        warmstarts = Dict(
            mode => ReducedGOOP.build_selective_warmstart(
                shifted_primal,
                source_z,
                kkt,
                mode,
            ) for mode in config.modes
        )
        replay_checkpoints = Dict{Symbol, Any}()
        for (execution_order, mode) in enumerate(ordering.order)
            replay_checkpoints[mode] = _replay_case!(
                replay_table,
                run_dir,
                formulation,
                scenario_seed,
                transition,
                mode,
                execution_order,
                mode_order_text,
                ordering.seed,
                source,
                destination,
                warmstarts[mode],
                transported_source,
                problem,
                kkt,
                blocks,
                config,
            )
        end
        append_row!(
            root_table,
            _root_spread_row(
                config,
                formulation,
                scenario_seed,
                transition,
                replay_checkpoints,
            ),
        )
    end
    nothing
end

function _sensitivity_path(run_dir, formulation, scenario_seed, step)
    joinpath(
        run_dir,
        "checkpoints",
        "sensitivity",
        String(formulation),
        "seed_$(scenario_seed)__step_$(step).jld2",
    )
end

function _sensitivity_case_id(
    formulation,
    scenario_seed,
    step,
    block,
    amplitude,
)
    amplitude_label = replace(@sprintf("%.0e", amplitude), "+" => "")
    "sensitivity__$(formulation)__seed$(scenario_seed)__step$(step)__$(block)__$(amplitude_label)"
end

function _jacobian_block_statistics(jacobian, z, dims)
    columns = collect(dims)
    if isempty(columns)
        return (;
            count = 0,
            raw = 0.0,
            raw_per_sqrt_column = 0.0,
            scaled = 0.0,
            scaled_per_sqrt_column = 0.0,
            maximum_column = 0.0,
        )
    end
    column_norms =
        Float64[norm(view(jacobian, :, column)) for column in columns]
    raw = norm(column_norms)
    # A relative-unit column scaling: a unit direction in the reported statistic
    # corresponds to max(1, |z_j|) in coordinate j.
    coordinate_scales = max.(1.0, abs.(view(z, columns)))
    scaled = norm(column_norms .* coordinate_scales)
    (;
        count = length(columns),
        raw,
        raw_per_sqrt_column = raw / sqrt(length(columns)),
        scaled,
        scaled_per_sqrt_column = scaled / sqrt(length(columns)),
        maximum_column = maximum(column_norms),
    )
end

function _near_null_summary(jacobian, blocks, config)
    rows, columns = size(jacobian)
    if columns > config.svd_max_variable_dimension
        return (;
            computed = false,
            count = 0,
            energy_mean = NaN,
            energy_median = NaN,
            energy_maximum = NaN,
            relative_threshold = 1e-10,
            skip_reason =
                "variable dimension $(columns) exceeds cap $(config.svd_max_variable_dimension)",
        )
    end
    if rows == 0 || columns == 0
        return (;
            computed = false,
            count = 0,
            energy_mean = NaN,
            energy_median = NaN,
            energy_maximum = NaN,
            relative_threshold = 1e-10,
            skip_reason = "empty Jacobian",
        )
    end
    try
        decomposition = svd(Matrix(jacobian); full = true)
        relative_threshold = 1e-10
        absolute_threshold =
            isempty(decomposition.S) ? 0.0 :
            relative_threshold * maximum(decomposition.S)
        numerical_rank = count(>(absolute_threshold), decomposition.S)
        V = decomposition.V
        indices = (numerical_rank+1):size(V, 2)
        energies = Float64[]
        for index in indices
            vector = view(V, :, index)
            denominator = sum(abs2, vector)
            numerator =
                isempty(blocks.ψ_in) ? 0.0 :
                sum(abs2, view(vector, blocks.ψ_in))
            push!(
                energies,
                denominator == 0.0 ? 0.0 : numerator / denominator,
            )
        end
        return (;
            computed = true,
            count = length(energies),
            energy_mean = isempty(energies) ? NaN : sum(energies) / length(energies),
            energy_median = isempty(energies) ? NaN : median(energies),
            energy_maximum = isempty(energies) ? NaN : maximum(energies),
            relative_threshold,
            skip_reason = "",
        )
    catch error
        return (;
            computed = false,
            count = 0,
            energy_mean = NaN,
            energy_median = NaN,
            energy_maximum = NaN,
            relative_threshold = 1e-10,
            skip_reason = sprint(showerror, error),
        )
    end
end

function _run_sensitivity_case!(
    sensitivity_table,
    run_dir,
    formulation,
    scenario_seed,
    step,
    reference,
    problem,
    kkt,
    blocks,
    config,
)
    path = _sensitivity_path(run_dir, formulation, scenario_seed, step)
    if isfile(path)
        checkpoint = JLD2.load_object(path)
        for row in checkpoint["rows"]
            append_row!(sensitivity_table, row)
        end
        return checkpoint
    end

    if !get(reference, "accepted", false) ||
       !all(key -> haskey(reference, key), ("z", "parameters"))
        row = Dict{String, Any}(
            "case_id" =>
                "sensitivity__$(formulation)__seed$(scenario_seed)__step$(step)__unavailable",
            "profile" => config.profile,
            "formulation" => formulation,
            "scenario_seed" => scenario_seed,
            "step" => step,
            "reference_status" => get(reference, "status", :unavailable),
            "reference_accepted" => false,
            "block" => :unavailable,
            "error" => "reference unavailable",
        )
        checkpoint = Dict{String, Any}("rows" => [row])
        _atomic_save(path, checkpoint)
        append_row!(sensitivity_table, row)
        return checkpoint
    end

    z = reference["z"]
    θ = reference["parameters"].θ
    preference =
        _preference_stratum(problem, view(z, blocks.z), θ, config)
    baseline = _residual(kkt, z, θ)
    jacobian = copy(kkt.∇F_z!.result_buffer)
    kkt.∇F_z!(jacobian, z; θ, ϵ = EPSILON0, η = 0.0)
    near_null = _near_null_summary(jacobian, blocks, config)
    block_dimensions = (
        z = blocks.z,
        lambda = blocks.λ,
        psi_out = blocks.ψ_out,
        psi_in = blocks.ψ_in,
    )
    rows = Dict{String, Any}[]
    for (block, dimensions) in pairs(block_dimensions)
        statistics = _jacobian_block_statistics(jacobian, z, dimensions)
        direction_seed = _stable_seed(
            config.order_seed,
            :sensitivity,
            formulation,
            scenario_seed,
            step,
            block,
        )
        direction = zeros(kkt.variable_dimension)
        if !isempty(dimensions)
            local_values =
                randn(MersenneTwister(direction_seed), length(dimensions))
            local_values ./= norm(local_values)
            direction[dimensions] .= local_values
        end
        for amplitude in config.sensitivity_amplitudes
            perturbed = z .+ amplitude .* direction
            perturbed_residual = _residual(kkt, perturbed, θ)
            response =
                norm(perturbed_residual.values .- baseline.values) / amplitude
            row = Dict{String, Any}(
                "case_id" => _sensitivity_case_id(
                    formulation,
                    scenario_seed,
                    step,
                    block,
                    amplitude,
                ),
                "profile" => config.profile,
                "formulation" => formulation,
                "scenario_seed" => scenario_seed,
                "step" => step,
                "reference_status" => reference["status"],
                "reference_accepted" => reference["accepted"],
                "preference_stratum" => preference.stratum,
                "innermost_margin" => preference.margin,
                "block" => block,
                "direction_seed" => direction_seed,
                "amplitude" => amplitude,
                "directional_residual_change_per_amplitude" => response,
                "baseline_residual_norm2" => baseline.norm2,
                "block_columns" => statistics.count,
                "raw_frobenius_norm" => statistics.raw,
                "raw_frobenius_per_sqrt_column" =>
                    statistics.raw_per_sqrt_column,
                "column_scaled_frobenius_norm" => statistics.scaled,
                "column_scaled_frobenius_per_sqrt_column" =>
                    statistics.scaled_per_sqrt_column,
                "maximum_column_norm" => statistics.maximum_column,
                "near_null_computed" => near_null.computed,
                "near_null_vector_count" => near_null.count,
                "near_null_psi_in_energy_mean" => near_null.energy_mean,
                "near_null_psi_in_energy_median" =>
                    near_null.energy_median,
                "near_null_psi_in_energy_max" =>
                    near_null.energy_maximum,
                "near_null_relative_threshold" =>
                    near_null.relative_threshold,
                "near_null_skip_reason" => near_null.skip_reason,
                "error" => "",
            )
            push!(rows, row)
        end
    end
    checkpoint = Dict{String, Any}(
        "rows" => rows,
        "near_null" => near_null,
    )
    _atomic_save(path, checkpoint)
    for row in rows
        append_row!(sensitivity_table, row)
    end
    checkpoint
end

_epsilon_label(epsilon) =
    replace(replace(@sprintf("%.0e", epsilon), "+" => ""), "-" => "m")

function _scaling_reference_path(
    run_dir,
    formulation,
    scenario_seed,
    direction_index,
    epsilon,
)
    joinpath(
        run_dir,
        "checkpoints",
        "scaling",
        String(formulation),
        "seed_$(scenario_seed)",
        "direction_$(direction_index)",
        "reference__epsilon_$(_epsilon_label(epsilon)).jld2",
    )
end

function _scaling_mode_path(
    run_dir,
    formulation,
    scenario_seed,
    direction_index,
    epsilon,
    mode,
)
    joinpath(
        run_dir,
        "checkpoints",
        "scaling",
        String(formulation),
        "seed_$(scenario_seed)",
        "direction_$(direction_index)",
        "epsilon_$(_epsilon_label(epsilon))__$(mode).jld2",
    )
end

function _scaling_case_id(
    formulation,
    scenario_seed,
    direction_index,
    epsilon,
    mode,
)
    "scaling__$(formulation)__seed$(scenario_seed)__direction$(direction_index)__epsilon$(_epsilon_label(epsilon))__$(mode)"
end

function _initial_state_direction(config, scenario_seed, direction_index)
    seed = _stable_seed(
        config.order_seed,
        :scaling_direction,
        scenario_seed,
        direction_index,
    )
    direction = randn(MersenneTwister(seed), 5)
    # The 3-D rigid translation appears in both arm state blocks, so normalize
    # in the induced norm of the full concatenated initial-state perturbation.
    induced_norm = sqrt(
        2 * sum(abs2, view(direction, 1:3)) +
        sum(abs2, view(direction, 4:5)),
    )
    direction ./= induced_norm
    (; seed, direction)
end

function _perturb_initial_instance(instance, direction, epsilon)
    translation = epsilon .* direction[1:3]
    child_translation = epsilon .* [direction[4], direction[5], 0.0]
    (;
        initial_state1 = copy(instance.initial_state1) .+ translation,
        initial_state2 = copy(instance.initial_state2) .+ translation,
        initial_state3 = copy(instance.initial_state3) .+ child_translation,
        initial_control1 = copy(instance.initial_control1),
        initial_control2 = copy(instance.initial_control2),
        initial_control3 = copy(instance.initial_control3),
    )
end

function _ensure_scaling_reference!(
    run_dir,
    formulation,
    scenario_seed,
    direction_index,
    epsilon,
    instance,
    warmstart,
    scenario,
    flatten_parameters,
    kkt,
    config,
)
    path = _scaling_reference_path(
        run_dir,
        formulation,
        scenario_seed,
        direction_index,
        epsilon,
    )
    isfile(path) && return JLD2.load_object(path)
    parameters =
        RAC.build_instance_parameters(flatten_parameters, instance, scenario)
    try
        result = _solve_with_trace(
            kkt,
            parameters.θ,
            warmstart,
            _solver_options(config),
        )
        direct = _residual(
            kkt,
            result.output.z,
            parameters.θ;
            epsilon = result.output.ϵ,
        )
        accepted =
            isfinite(direct.norm2) &&
            direct.norm2 <= config.reference_acceptance_tol
        checkpoint = Dict{String, Any}(
            "accepted" => accepted,
            "status" => result.output.status,
            "source" => :cold_default,
            "reference_initialization" =>
                config.reference_initialization,
            "direct_residual_norm2" => direct.norm2,
            "direct_residual_normalized" => direct.normalized,
            "instance" => instance,
            "parameters" => parameters,
            "z" => copy(result.output.z),
            "events" => result.events,
            "solve_time_sec" => result.solve_time,
            "instrumented_solve_time_sec" =>
                result.instrumented_solve_time,
            "timing_solver_status" => result.timing_output.status,
        )
        _atomic_save(path, checkpoint)
        return checkpoint
    catch error
        checkpoint = Dict{String, Any}(
            "accepted" => false,
            "status" => :exception,
            "source" => :cold_default,
            "reference_initialization" =>
                config.reference_initialization,
            "direct_residual_norm2" => NaN,
            "instance" => instance,
            "parameters" => parameters,
            "error" => sprint(showerror, error, catch_backtrace()),
        )
        _atomic_save(path, checkpoint)
        return checkpoint
    end
end

function _unavailable_scaling_row(
    config,
    formulation,
    scenario_seed,
    direction_index,
    epsilon,
    mode,
    execution_order,
    mode_order_text,
    order_seed,
    base_reference,
    perturbed_reference,
)
    Dict{String, Any}(
        "case_id" => _scaling_case_id(
            formulation,
            scenario_seed,
            direction_index,
            epsilon,
            mode,
        ),
        "profile" => config.profile,
        "formulation" => formulation,
        "scenario_seed" => scenario_seed,
        "direction" => direction_index,
        "direction_seed" => _initial_state_direction(
            config,
            scenario_seed,
            direction_index,
        ).seed,
        "epsilon" => epsilon,
        "mode" => mode,
        "execution_order" => execution_order,
        "mode_order" => mode_order_text,
        "order_seed" => order_seed,
        "base_reference_status" =>
            get(base_reference, "status", :unavailable),
        "perturbed_reference_status" =>
            get(perturbed_reference, "status", :unavailable),
        "base_reference_accepted" =>
            get(base_reference, "accepted", false),
        "perturbed_reference_accepted" =>
            get(perturbed_reference, "accepted", false),
        "valid_reference_pair" => false,
        "solver_status" => :not_run_reference_unavailable,
        "direct_converged" => false,
        "error" => get(
            perturbed_reference,
            "error",
            "base or perturbed reference unavailable",
        ),
    )
end

function _run_scaling_mode!(
    scaling_table,
    run_dir,
    formulation,
    scenario_seed,
    direction_index,
    epsilon,
    mode,
    execution_order,
    mode_order_text,
    order_seed,
    base_reference,
    perturbed_reference,
    warmstart,
    kkt,
    blocks,
    config,
)
    path = _scaling_mode_path(
        run_dir,
        formulation,
        scenario_seed,
        direction_index,
        epsilon,
        mode,
    )
    if isfile(path)
        checkpoint = JLD2.load_object(path)
        append_row!(scaling_table, checkpoint["row"])
        return checkpoint
    end

    θ_base = base_reference["parameters"].θ
    θ_perturbed = perturbed_reference["parameters"].θ
    baseline = _residual(kkt, warmstart, θ_base)
    initial = _residual(kkt, warmstart, θ_perturbed)
    induced =
        norm(initial.values .- baseline.values)
    induced_normalized = induced / sqrt(kkt.kkt_dimension)
    reference_floor = maximum(
        (
            get(base_reference, "direct_residual_norm2", 0.0),
            get(perturbed_reference, "direct_residual_norm2", 0.0),
            100 * eps(Float64) * sqrt(kkt.kkt_dimension),
        ),
    )
    # For a total-residual log-log slope, the zero-perturbation residual of the
    # same mode is an empirical floor rather than evidence for O(epsilon).
    numerical_floor = max(reference_floor, baseline.norm2)
    above_floor = initial.norm2 > 1.25 * numerical_floor
    row = Dict{String, Any}(
        "case_id" => _scaling_case_id(
            formulation,
            scenario_seed,
            direction_index,
            epsilon,
            mode,
        ),
        "profile" => config.profile,
        "formulation" => formulation,
        "scenario_seed" => scenario_seed,
        "direction" => direction_index,
        "direction_seed" => _initial_state_direction(
            config,
            scenario_seed,
            direction_index,
        ).seed,
        "epsilon" => epsilon,
        "mode" => mode,
        "execution_order" => execution_order,
        "mode_order" => mode_order_text,
        "order_seed" => order_seed,
        "base_reference_status" => base_reference["status"],
        "perturbed_reference_status" => perturbed_reference["status"],
        "base_reference_accepted" => base_reference["accepted"],
        "perturbed_reference_accepted" => perturbed_reference["accepted"],
        "valid_reference_pair" => true,
        "perturbed_reference_residual" =>
            perturbed_reference["direct_residual_norm2"],
        "baseline_initial_residual_norm2" => baseline.norm2,
        "initial_residual_norm2" => initial.norm2,
        "initial_residual_normalized" => initial.normalized,
        "initial_residual_norm_inf" => initial.norm_inf,
        "parameter_induced_residual_change_norm2" => induced,
        "parameter_induced_residual_change_normalized" =>
            induced_normalized,
        "numerical_floor" => numerical_floor,
        "above_numerical_floor" => above_floor,
        "error" => "",
    )
    if !config.solve_scaling_cases
        row["solver_status"] = :not_run_configured
        row["direct_converged"] = false
        checkpoint = Dict{String, Any}("row" => row)
        _atomic_save(path, checkpoint)
        append_row!(scaling_table, row)
        return checkpoint
    end

    try
        result = _solve_with_trace(
            kkt,
            θ_perturbed,
            warmstart,
            _solver_options(config),
        )
        metrics = _result_metrics(result)
        direct = _residual(
            kkt,
            result.output.z,
            θ_perturbed;
            epsilon = result.output.ϵ,
        )
        destination_primal =
            view(perturbed_reference["z"], blocks.z)
        final_primal = view(result.output.z, blocks.z)
        primal_distance_normalized =
            norm(final_primal .- destination_primal) /
            max(1.0, norm(destination_primal))
        for name in (
            :first_accepted_residual_norm2,
            :first_accepted_alpha,
            :total_inner_iters,
            :total_backtracking_count,
            :eta_retry_count,
            :full_step_fraction,
            :solve_time_sec,
            :instrumented_solve_time_sec,
            :solver_status,
            :timing_solver_status,
        )
            row[string(name)] = getproperty(metrics, name)
        end
        row["direct_final_residual_norm2"] = direct.norm2
        direct_converged =
            isfinite(direct.norm2) && direct.norm2 <= config.replay_tol
        row["direct_converged"] = direct_converged
        row["final_primal_distance_normalized"] =
            primal_distance_normalized
        checkpoint = Dict{String, Any}(
            "row" => row,
            "events" => result.events,
            "converged" => direct_converged,
        )
        config.save_full_solutions &&
            (checkpoint["final_z"] = copy(result.output.z))
        _atomic_save(path, checkpoint)
        append_row!(scaling_table, row)
        return checkpoint
    catch error
        row["solver_status"] = :exception
        row["direct_converged"] = false
        row["error"] = sprint(showerror, error, catch_backtrace())
        checkpoint = Dict{String, Any}(
            "row" => row,
            "converged" => false,
        )
        _atomic_save(path, checkpoint)
        append_row!(scaling_table, row)
        return checkpoint
    end
end

function _run_scaling_cases!(
    scaling_table,
    run_dir,
    formulation,
    scenario_seed,
    base_reference,
    scenario,
    flatten_parameters,
    kkt,
    blocks,
    config,
)
    valid_base =
        get(base_reference, "accepted", false) &&
        all(
        key -> haskey(base_reference, key),
        ("z", "instance", "parameters"),
    )
    if !valid_base
        perturbed_unavailable = Dict{String, Any}(
            "status" => :not_run_base_reference_unavailable,
            "accepted" => false,
            "error" => "base reference unavailable",
        )
        for direction_index in 1:config.scaling_directions,
            epsilon in sort(config.scaling_epsilons)
            ordering = _mode_order(
                config,
                formulation,
                scenario_seed,
                "$(direction_index)_$(_epsilon_label(epsilon))",
                :scaling,
            )
            mode_order_text = join(String.(ordering.order), ";")
            for (execution_order, mode) in enumerate(ordering.order)
                path = _scaling_mode_path(
                    run_dir,
                    formulation,
                    scenario_seed,
                    direction_index,
                    epsilon,
                    mode,
                )
                if isfile(path)
                    checkpoint = JLD2.load_object(path)
                    append_row!(scaling_table, checkpoint["row"])
                else
                    row = _unavailable_scaling_row(
                        config,
                        formulation,
                        scenario_seed,
                        direction_index,
                        epsilon,
                        mode,
                        execution_order,
                        mode_order_text,
                        ordering.seed,
                        base_reference,
                        perturbed_unavailable,
                    )
                    _atomic_save(path, Dict{String, Any}("row" => row))
                    append_row!(scaling_table, row)
                end
            end
        end
        return nothing
    end
    base_z = base_reference["z"]
    base_primal = copy(view(base_z, blocks.z))
    warmstarts = Dict(
        mode => ReducedGOOP.build_selective_warmstart(
            base_primal,
            base_z,
            kkt,
            mode,
        ) for mode in config.modes
    )
    for direction_index in 1:config.scaling_directions
        direction_info =
            _initial_state_direction(config, scenario_seed, direction_index)
        for (epsilon_index, epsilon) in
            enumerate(sort(config.scaling_epsilons))
            perturbed_instance = _perturb_initial_instance(
                base_reference["instance"],
                direction_info.direction,
                epsilon,
            )
            reference_initialization = _reference_initialization(
                perturbed_instance,
                scenario,
                epsilon_index,
            )
            perturbed_reference = _ensure_scaling_reference!(
                run_dir,
                formulation,
                scenario_seed,
                direction_index,
                epsilon,
                perturbed_instance,
                reference_initialization.warmstart,
                scenario,
                flatten_parameters,
                kkt,
                config,
            )
            ordering = _mode_order(
                config,
                formulation,
                scenario_seed,
                "$(direction_index)_$(_epsilon_label(epsilon))",
                :scaling,
            )
            mode_order_text = join(String.(ordering.order), ";")
            valid =
                base_reference["accepted"] &&
                get(perturbed_reference, "accepted", false) &&
                all(
                    key -> haskey(perturbed_reference, key),
                    ("z", "parameters"),
                )
            for (execution_order, mode) in enumerate(ordering.order)
                if valid
                    _run_scaling_mode!(
                        scaling_table,
                        run_dir,
                        formulation,
                        scenario_seed,
                        direction_index,
                        epsilon,
                        mode,
                        execution_order,
                        mode_order_text,
                        ordering.seed,
                        base_reference,
                        perturbed_reference,
                        warmstarts[mode],
                        kkt,
                        blocks,
                        config,
                    )
                else
                    row = _unavailable_scaling_row(
                        config,
                        formulation,
                        scenario_seed,
                        direction_index,
                        epsilon,
                        mode,
                        execution_order,
                        mode_order_text,
                        ordering.seed,
                        base_reference,
                        perturbed_reference,
                    )
                    append_row!(scaling_table, row)
                end
            end
        end
    end
    nothing
end

function _kkt_row(config, formulation, kkt, blocks, build_time)
    classified = unique(
        vcat(
            collect(blocks.z),
            collect(blocks.λ),
            collect(blocks.ψ_out),
            collect(blocks.ψ_in),
            collect(kkt.preference_slack_dims),
            collect(kkt.interior_point_slack_dims),
            collect(kkt.inequality_constraint_dual_dims),
        ),
    )
    Dict{String, Any}(
        "case_id" => "kkt__$(formulation)",
        "profile" => config.profile,
        "formulation" => formulation,
        "planning_horizon" => config.planning_horizon,
        "kkt_dimension" => kkt.kkt_dimension,
        "variable_dimension" => kkt.variable_dimension,
        "z_dimension" => length(blocks.z),
        "lambda_dimension" => length(blocks.λ),
        "psi_out_dimension" => length(blocks.ψ_out),
        "psi_in_dimension" => length(blocks.ψ_in),
        "preference_slack_dimension" =>
            length(kkt.preference_slack_dims),
        "interior_point_slack_dimension" =>
            length(kkt.interior_point_slack_dims),
        "inequality_dual_dimension" =>
            length(kkt.inequality_constraint_dual_dims),
        "unclassified_dimension" =>
            kkt.variable_dimension - length(classified),
        "build_time_sec" => build_time,
    )
end

function _reference_accuracy_summary(run_dir, config)
    reference_rows =
        read_csv_rows(joinpath(run_dir, "raw", "references.csv"))
    accepted_references = filter(
        row -> _true_value(row, "reference_accepted"),
        reference_rows,
    )
    accepted_above_replay = count(
        row -> begin
            residual = _finite_float(row, "direct_residual_norm2")
            !isnothing(residual) &&
                residual > config.replay_tol &&
                residual <= config.reference_acceptance_tol
        end,
        accepted_references,
    )

    replay_rows = read_csv_rows(joinpath(run_dir, "raw", "replay.csv"))
    replay_pairs = Dict{String, Dict{String, String}}()
    for row in replay_rows
        _true_value(row, "valid_reference_pair") || continue
        key = join(
            (
                get(row, "formulation", ""),
                get(row, "scenario_seed", ""),
                get(row, "transition", ""),
            ),
            "|",
        )
        replay_pairs[key] = row
    end
    replay_pairs_above = count(
        row -> begin
            source = _finite_float(row, "source_reference_residual")
            destination =
                _finite_float(row, "destination_reference_residual")
            (!isnothing(source) && source > config.replay_tol) ||
                (
                    !isnothing(destination) &&
                    destination > config.replay_tol
                )
        end,
        values(replay_pairs),
    )

    scaling_rows = read_csv_rows(joinpath(run_dir, "raw", "scaling.csv"))
    scaling_references = Dict{String, Dict{String, String}}()
    for row in scaling_rows
        _true_value(row, "perturbed_reference_accepted") || continue
        key = join(
            (
                get(row, "formulation", ""),
                get(row, "scenario_seed", ""),
                get(row, "direction", ""),
                get(row, "epsilon", ""),
            ),
            "|",
        )
        scaling_references[key] = row
    end
    scaling_above = count(
        row -> begin
            residual =
                _finite_float(row, "perturbed_reference_residual")
            !isnothing(residual) && residual > config.replay_tol
        end,
        values(scaling_references),
    )
    Dict(
        "canonical_reference_protocol" =>
            "uniform_comparability_tolerance",
        "requested_reference_tol" => config.requested_reference_tol,
        "direct_acceptance_tol" => config.reference_acceptance_tol,
        "replay_tol" => config.replay_tol,
        "accepted_canonical_references" => length(accepted_references),
        "accepted_canonical_references_above_replay_tol" =>
            accepted_above_replay,
        "valid_replay_reference_pairs" => length(replay_pairs),
        "valid_replay_pairs_with_source_or_destination_above_replay_tol" =>
            replay_pairs_above,
        "accepted_perturbed_scaling_references" =>
            length(scaling_references),
        "accepted_perturbed_scaling_references_above_replay_tol" =>
            scaling_above,
    )
end

function _study_manifest(
    config,
    kkt_rows;
    sequence_driver =
        (:reduced in config.formulations ? :reduced : first(config.formulations)),
    reference_accuracy_summary = Dict{String, Any}(),
)
    Dict(
        "study" => Dict(
            "profile" => String(config.profile),
            "comparability_protocol" =>
                String(config.comparability_protocol),
            "comparability_grid" =>
                "all profiles and stages use planning_horizon=$(config.planning_horizon) and dt=$(config.Δt)",
            "comparability_solver_rule" =>
                "canonical references, replay, scaling references, scaling mode solves, and the one compilation-only warmup per formulation all use one identical InteriorPointOptions value; sensitivity uses the same KKT/scenario and accepted canonical points but invokes no solver",
            "baseline_choice" =>
                "tol=0.008 and every solver option except the iteration cap match the headless robotic-arm receding baseline in Robotic_arm_mpc.jl; max_inner_iters was raised uniformly from the 500-step baseline to 1000 after a direct reduced T20 seed101 cold solve required 717 iterations and reached residual 0.007991129782359604; the visualization-oriented Robotic_arm_receding.jl differs only by recording convergence histories; planning_horizon=20 is the explicit comparability override and dt=0.1 matches the scenario/receding default",
            "paired_unit" =>
                "formulation × scenario seed × MPC transition; all modes replay the same source/destination canonical-reference pair",
            "formulations" => String.(config.formulations),
            "sequence_driver" => String(sequence_driver),
            "common_sequence_rule" =>
                "one persisted instance sequence per scenario seed is generated by the sequence driver; every formulation solves the exact same instance digest at step t",
            "modes" => String.(config.modes),
            "mode_primal_only" =>
                "shifted primal; equality and stationarity duals reset to solver defaults",
            "mode_equality_duals" =>
                "shifted primal plus carried equality duals",
            "mode_all_except_innermost_stationarity" =>
                "shifted primal plus equality and outer stationarity duals; innermost stationarity duals reset",
            "mode_all_duals" =>
                "shifted primal plus equality, outer-stationarity, and innermost-stationarity duals",
            "epsilon0" => EPSILON0,
            "reference_initialization" =>
                String(config.reference_initialization),
            "reference_initialization_rule" =>
                "every canonical and perturbed-scaling reference makes one fresh destination-specific cold/default solve; no previous reference primal or dual is transported and no rescue tournament is used",
            "reference_initialization_change_disclosure" =>
                "cold-default-each-step was adopted after the 0.01 pilot's all-dual continuation failed; targeted diagnostics accepted cold/default starts at all four first-failure cases without changing solver options, so a fresh run is required",
            "reference_initialization_basin_caveat" =>
                "cold/default is mode-neutral relative to the four replay transport modes, but it is not basin-neutral or unbiased and can still select a numerical root basin",
            "requested_reference_tol" => config.requested_reference_tol,
            "reference_acceptance_rule" =>
                "direct ||F(z; theta, epsilon_final, eta=0)||_2 <= $(config.reference_acceptance_tol); raw solver status is retained separately",
            "reference_acceptance_tol" =>
                config.reference_acceptance_tol,
            "canonical_reference_protocol" =>
                "accepted canonical point and trajectory generator under the same tol=0.008, max_inner_iters=1000 solver protocol as replay and scaling; not higher-accuracy ground truth",
            "canonical_reference_accuracy_qualification" =>
                "block errors, candidate-root distances, sensitivity, and scaling diagnostics are conditioned on canonical points accepted at the same 0.008 direct-residual threshold used for replay",
            "reference_accuracy_counts" =>
                reference_accuracy_summary,
            "replay_tol" => config.replay_tol,
            "boundary_rule" =>
                "minimum innermost preference margin: violated < -tol, boundary in [-tol,tol], satisfied > tol",
            "boundary_tol" => config.boundary_tol,
            "material_root_rtol" => config.material_root_rtol,
            "candidate_root_qualification" =>
                "pairwise primal distances compare replay-tolerance accepted candidates; because replay_tol may exceed material_root_rtol, exceedance flags candidates for tighter follow-up and does not certify distinct exact roots",
            "scaling_raw_floor_field_rule" =>
                "legacy raw numerical_floor=max(mode zero-perturbation residual, base reference residual, perturbed reference residual, machine floor); retained for checkpoint compatibility",
            "scaling_analysis_rule" =>
                "derived scaling_slopes separates reference resolution (base all-dual and perturbed-reference residuals) from the mode structural epsilon=0 baseline; a structural baseline >1.25*resolution is retained and fit rather than discarded",
            "near_null_rule" =>
                "full right singular vectors after numerical rank at sigma <= 1e-10*sigma_max; skipped above configured variable-dimension cap",
            "near_null_dimension_cap" =>
                config.svd_max_variable_dimension,
            "randomization" =>
                "FNV-1a stable integer seeds feed MersenneTwister; mode execution order randomized independently per paired case",
            "terminal_shift_rule" =>
                "drop the executed knot, retain controls, complete the terminal state with the last retained control at planning_horizon-1, append an unused zero terminal control",
            "uniform_solver_options" =>
                _solver_options_dict(config),
        ),
        "kkt_systems" => [
            Dict(
                key => (
                    value isa Symbol ? String(value) : value
                ) for (key, value) in row
            ) for row in kkt_rows
        ],
    )
end

function _write_rows(path, columns, rows)
    io = IOBuffer()
    println(io, join(columns, ","))
    for row in rows
        println(
            io,
            join((_csv_escape(get(row, column, "")) for column in columns), ","),
        )
    end
    _atomic_write(path, String(take!(io)))
end

function _finite_float(row, name)
    value = get(row, name, "")
    isempty(value) && return nothing
    parsed = tryparse(Float64, value)
    isnothing(parsed) || isfinite(parsed) || return nothing
    parsed
end

function _true_value(row, name)
    lowercase(get(row, name, "")) == "true"
end

function _replay_key(row)
    join(
        (
            get(row, "formulation", ""),
            get(row, "scenario_seed", ""),
            get(row, "transition", ""),
        ),
        "|",
    )
end

function _mode_index(rows, mode)
    Dict(
        _replay_key(row) => row for row in rows if
        get(row, "mode", "") == String(mode)
    )
end

_solver_converged(row) = _true_value(row, "direct_converged")

function _paired_bootstrap(
    values_a,
    values_b,
    replicates,
    seed;
    positive = false,
)
    n = length(values_a)
    n == 0 && return (;
        difference_low = NaN,
        difference_high = NaN,
        ratio_low = NaN,
        ratio_high = NaN,
    )
    rng = MersenneTwister(seed)
    differences = Vector{Float64}(undef, replicates)
    log_ratios = Float64[]
    positive && sizehint!(log_ratios, replicates)
    for replicate in 1:replicates
        indices = rand(rng, 1:n, n)
        differences[replicate] =
            median(values_a[indices] .- values_b[indices])
        if positive &&
           all(>(0.0), view(values_a, indices)) &&
           all(>(0.0), view(values_b, indices))
            push!(
                log_ratios,
                median(
                    log.(view(values_a, indices)) .-
                    log.(view(values_b, indices)),
                ),
            )
        end
    end
    (;
        difference_low = quantile(differences, 0.025),
        difference_high = quantile(differences, 0.975),
        ratio_low =
            isempty(log_ratios) ? NaN : exp(quantile(log_ratios, 0.025)),
        ratio_high =
            isempty(log_ratios) ? NaN : exp(quantile(log_ratios, 0.975)),
    )
end

const PAIRED_COLUMNS = [
    "case_id",
    "formulation",
    "comparison",
    "mode_a",
    "mode_b",
    "preference_stratum",
    "metric",
    "lower_is_better",
    "directional_scoring",
    "requires_converged_pair",
    "candidate_pairs",
    "valid_pairs",
    "mode_a_failures",
    "mode_b_failures",
    "excluded_or_missing_pairs",
    "mode_a_median",
    "mode_a_q1",
    "mode_a_q3",
    "mode_b_median",
    "mode_b_q1",
    "mode_b_q3",
    "median_paired_difference_a_minus_b",
    "median_paired_ratio_a_over_b",
    "wins_a",
    "ties",
    "losses_a",
    "bootstrap_difference_low",
    "bootstrap_difference_high",
    "bootstrap_ratio_low",
    "bootstrap_ratio_high",
    "bootstrap_replicates",
]

function _paired_summary_rows(replay_rows, config)
    metric_specs = (
        (
            name = "initial_residual_normalized",
            lower = true,
            convergence = false,
            positive = true,
        ),
        (
            name = "initial_residual_norm2",
            lower = true,
            convergence = false,
            positive = true,
        ),
        (
            name = "total_inner_iters",
            lower = true,
            convergence = true,
            positive = true,
        ),
        (
            name = "total_outer_iters",
            lower = true,
            convergence = true,
            positive = true,
        ),
        (
            name = "first_accepted_residual_norm2",
            lower = true,
            convergence = true,
            positive = true,
        ),
        (
            name = "first_iteration_backtracking_count",
            lower = true,
            convergence = true,
            positive = false,
        ),
        (
            name = "regularization_change_count",
            lower = true,
            convergence = true,
            positive = false,
            directional = false,
        ),
        (
            name = "total_backtracking_count",
            lower = true,
            convergence = true,
            positive = false,
        ),
        (
            name = "eta_retry_count",
            lower = true,
            convergence = true,
            positive = false,
        ),
        (
            name = "full_step_fraction",
            lower = false,
            convergence = true,
            positive = false,
        ),
        (
            name = "first_accepted_alpha",
            lower = false,
            convergence = true,
            positive = true,
        ),
        (
            name = "solve_time_sec",
            lower = true,
            convergence = true,
            positive = true,
        ),
    )
    comparisons = (
        (
            label = "all_except_vs_equality",
            a = :all_except_innermost_stationarity,
            b = :equality_duals,
        ),
        (
            label = "all_except_vs_all_duals",
            a = :all_except_innermost_stationarity,
            b = :all_duals,
        ),
    )
    results = Dict{String, Any}[]
    for formulation in config.formulations
        formulation_rows =
            filter(row -> get(row, "formulation", "") == String(formulation), replay_rows)
        for comparison in comparisons
            index_a = _mode_index(formulation_rows, comparison.a)
            index_b = _mode_index(formulation_rows, comparison.b)
            common_keys =
                sort(collect(intersect(keys(index_a), keys(index_b))))
            stratum_counts = Dict(
                stratum => count(
                    key -> get(
                        index_a[key],
                        "preference_stratum",
                        "",
                    ) == stratum,
                    common_keys,
                ) for stratum in ("satisfied", "boundary", "violated")
            )
            strata = ["all"]
            append!(
                strata,
                [
                    stratum for stratum in
                    ("satisfied", "boundary", "violated") if
                    stratum_counts[stratum] >= 5
                ],
            )
            for stratum in strata
                selected_keys =
                    stratum == "all" ? common_keys :
                    filter(
                        key -> get(
                            index_a[key],
                            "preference_stratum",
                            "",
                        ) == stratum,
                        common_keys,
                    )
                for metric in metric_specs
                    directional = get(metric, :directional, true)
                    values_a = Float64[]
                    values_b = Float64[]
                    failures_a = 0
                    failures_b = 0
                    for key in selected_keys
                        row_a = index_a[key]
                        row_b = index_b[key]
                        valid_references =
                            _true_value(row_a, "valid_reference_pair") &&
                            _true_value(row_b, "valid_reference_pair")
                        if valid_references
                            !_solver_converged(row_a) &&
                                (failures_a += 1)
                            !_solver_converged(row_b) &&
                                (failures_b += 1)
                        end
                        converged =
                            !metric.convergence ||
                            (
                                _solver_converged(row_a) &&
                                _solver_converged(row_b)
                            )
                        value_a = _finite_float(row_a, metric.name)
                        value_b = _finite_float(row_b, metric.name)
                        if valid_references &&
                           converged &&
                           !isnothing(value_a) &&
                           !isnothing(value_b)
                            push!(values_a, value_a)
                            push!(values_b, value_b)
                        end
                    end
                    n = length(values_a)
                    differences = values_a .- values_b
                    ratios = [
                        a / b for (a, b) in zip(values_a, values_b) if
                        a > 0.0 && b > 0.0
                    ]
                    wins = 0
                    ties = 0
                    losses = 0
                    for (a, b) in zip(values_a, values_b)
                        tolerance =
                            sqrt(eps(Float64)) * max(1.0, abs(a), abs(b))
                        if !directional
                            continue
                        elseif abs(a - b) <= tolerance
                            ties += 1
                        elseif (metric.lower && a < b) ||
                               (!metric.lower && a > b)
                            wins += 1
                        else
                            losses += 1
                        end
                    end
                    bootstrap = _paired_bootstrap(
                        values_a,
                        values_b,
                        config.bootstrap_replicates,
                        _stable_seed(
                            config.bootstrap_seed,
                            formulation,
                            comparison.label,
                            stratum,
                            metric.name,
                        );
                        positive = metric.positive,
                    )
                    q = values -> (
                        isempty(values) ? (NaN, NaN, NaN) :
                        (
                            quantile(values, 0.25),
                            median(values),
                            quantile(values, 0.75),
                        )
                    )
                    q_a = q(values_a)
                    q_b = q(values_b)
                    push!(
                        results,
                        Dict{String, Any}(
                            "case_id" => join(
                                (
                                    formulation,
                                    comparison.label,
                                    stratum,
                                    metric.name,
                                ),
                                "__",
                            ),
                            "formulation" => formulation,
                            "comparison" => comparison.label,
                            "mode_a" => comparison.a,
                            "mode_b" => comparison.b,
                            "preference_stratum" => stratum,
                            "metric" => metric.name,
                            "lower_is_better" =>
                                directional ? metric.lower : "",
                            "directional_scoring" =>
                                directional ?
                                (
                                    metric.lower ?
                                    "lower_is_better" :
                                    "higher_is_better"
                                ) :
                                "none_total_eta_changes_are_nondirectional_churn",
                            "requires_converged_pair" =>
                                metric.convergence,
                            "candidate_pairs" => length(selected_keys),
                            "valid_pairs" => n,
                            "mode_a_failures" => failures_a,
                            "mode_b_failures" => failures_b,
                            "excluded_or_missing_pairs" =>
                                length(selected_keys) - n,
                            "mode_a_median" => q_a[2],
                            "mode_a_q1" => q_a[1],
                            "mode_a_q3" => q_a[3],
                            "mode_b_median" => q_b[2],
                            "mode_b_q1" => q_b[1],
                            "mode_b_q3" => q_b[3],
                            "median_paired_difference_a_minus_b" =>
                                isempty(differences) ? NaN :
                                median(differences),
                            "median_paired_ratio_a_over_b" =>
                                isempty(ratios) ? NaN : median(ratios),
                            "wins_a" => directional ? wins : "",
                            "ties" => directional ? ties : "",
                            "losses_a" => directional ? losses : "",
                            "bootstrap_difference_low" =>
                                bootstrap.difference_low,
                            "bootstrap_difference_high" =>
                                bootstrap.difference_high,
                            "bootstrap_ratio_low" =>
                                bootstrap.ratio_low,
                            "bootstrap_ratio_high" =>
                                bootstrap.ratio_high,
                            "bootstrap_replicates" =>
                                config.bootstrap_replicates,
                        ),
                    )
                end
            end
        end
    end
    results
end

function _rank_values(values)
    order = sortperm(values)
    ranks = zeros(Float64, length(values))
    start = 1
    while start <= length(order)
        stop = start
        while stop < length(order) &&
              values[order[stop+1]] == values[order[start]]
            stop += 1
        end
        rank = (start + stop) / 2
        for position in start:stop
            ranks[order[position]] = rank
        end
        start = stop + 1
    end
    ranks
end

function _safe_correlation(x, y; rank = false)
    length(x) >= 3 || return NaN
    x_values = rank ? _rank_values(x) : x
    y_values = rank ? _rank_values(y) : y
    all(==(first(x_values)), x_values) && return NaN
    all(==(first(y_values)), y_values) && return NaN
    cor(x_values, y_values)
end

const CORRELATION_COLUMNS = [
    "case_id",
    "formulation",
    "mode_scope",
    "preference_stratum",
    "outcome",
    "sample_count",
    "pearson_log_r0",
    "spearman_log_r0",
    "note",
]

function _correlation_rows(replay_rows, config)
    outcomes = (
        "total_inner_iters",
        "full_step_fraction",
        "total_backtracking_count",
        "first_accepted_alpha",
        "eta_retry_count",
        "regularization_change_count",
    )
    results = Dict{String, Any}[]
    for formulation in config.formulations
        formulation_rows = filter(
            row ->
                get(row, "formulation", "") == String(formulation) &&
                _true_value(row, "valid_reference_pair") &&
                _solver_converged(row),
            replay_rows,
        )
        scopes = [("pooled", formulation_rows)]
        append!(
            scopes,
            [
                (
                    String(mode),
                    filter(
                        row -> get(row, "mode", "") == String(mode),
                        formulation_rows,
                    ),
                ) for mode in config.modes
            ],
        )
        for (scope, rows) in scopes
            stratum_counts = Dict(
                stratum => count(
                    row ->
                        get(row, "preference_stratum", "") == stratum,
                    rows,
                ) for stratum in ("satisfied", "boundary", "violated")
            )
            strata = ["all"]
            append!(
                strata,
                [
                    stratum for stratum in
                    ("satisfied", "boundary", "violated") if
                    stratum_counts[stratum] >= 5
                ],
            )
            for stratum in strata, outcome in outcomes
                selected =
                    stratum == "all" ? rows :
                    filter(
                        row ->
                            get(row, "preference_stratum", "") == stratum,
                        rows,
                    )
                x = Float64[]
                y = Float64[]
                for row in selected
                    r0 = _finite_float(row, "initial_residual_normalized")
                    response = _finite_float(row, outcome)
                    if !isnothing(r0) &&
                       !isnothing(response) &&
                       r0 > 0.0
                        push!(x, log(r0))
                        push!(y, response)
                    end
                end
                push!(
                    results,
                    Dict{String, Any}(
                        "case_id" => join(
                            (formulation, scope, stratum, outcome),
                            "__",
                        ),
                        "formulation" => formulation,
                        "mode_scope" => scope,
                        "preference_stratum" => stratum,
                        "outcome" => outcome,
                        "sample_count" => length(x),
                        "pearson_log_r0" => _safe_correlation(x, y),
                        "spearman_log_r0" =>
                            _safe_correlation(x, y; rank = true),
                        "note" =>
                            "Descriptive association only; pooled estimates may be confounded by mode and transition.",
                    ),
                )
            end
        end
    end
    results
end

const FAILURE_COLUMNS = [
    "case_id",
    "formulation",
    "mode",
    "total_rows",
    "valid_reference_rows",
    "direct_converged_rows",
    "raw_solved_rows",
    "raw_failed_rows",
    "raw_direct_disagreements",
    "exception_rows",
    "reference_unavailable_rows",
]

function _failure_rows(replay_rows, config)
    results = Dict{String, Any}[]
    for formulation in config.formulations, mode in config.modes
        rows = filter(
            row ->
                get(row, "formulation", "") == String(formulation) &&
                get(row, "mode", "") == String(mode),
            replay_rows,
        )
        statuses = get.(rows, "solver_status", "")
        push!(
            results,
            Dict{String, Any}(
                "case_id" => "$(formulation)__$(mode)",
                "formulation" => formulation,
                "mode" => mode,
                "total_rows" => length(rows),
                "valid_reference_rows" =>
                    count(row -> _true_value(row, "valid_reference_pair"), rows),
                "direct_converged_rows" =>
                    count(row -> _solver_converged(row), rows),
                "raw_solved_rows" => count(==("solved"), statuses),
                "raw_failed_rows" => count(==("failed"), statuses),
                "raw_direct_disagreements" => count(
                    row ->
                        (
                            get(row, "solver_status", "") == "solved"
                        ) != _solver_converged(row),
                    rows,
                ),
                "exception_rows" => count(==("exception"), statuses),
                "reference_unavailable_rows" => count(
                    status -> occursin("reference_unavailable", status),
                    statuses,
                ),
            ),
        )
    end
    results
end

function _linear_fit(x, y)
    length(x) >= 2 || return nothing
    x_mean = sum(x) / length(x)
    y_mean = sum(y) / length(y)
    denominator = sum((value - x_mean)^2 for value in x)
    denominator > 0.0 || return nothing
    slope =
        sum(
            (x_value - x_mean) * (y_value - y_mean) for
            (x_value, y_value) in zip(x, y)
        ) / denominator
    intercept = y_mean - slope * x_mean
    total = sum((value - y_mean)^2 for value in y)
    residual = sum(
        (y_value - (intercept + slope * x_value))^2 for
        (x_value, y_value) in zip(x, y)
    )
    r_squared = total == 0.0 ? NaN : 1.0 - residual / total
    (; slope, intercept, r_squared)
end

const SCALING_SLOPE_COLUMNS = [
    "case_id",
    "formulation",
    "scenario_seed",
    "direction",
    "mode",
    "available_points",
    "reliable_points",
    "reference_resolution_floor_norm2",
    "structural_baseline_residual_norm2",
    "structural_baseline_to_resolution_ratio",
    "structural_floor_detected",
    "smallest_epsilon_residual_to_baseline_ratio",
    "fit_points",
    "epsilon_min",
    "epsilon_max",
    "slope_log_r0_vs_log_epsilon",
    "intercept",
    "r_squared",
    "fit_performed",
    "skip_reason",
]

function _scaling_slope_rows(scaling_rows, config)
    base_reference_values =
        Dict{String, Vector{Float64}}()
    for row in scaling_rows
        get(row, "mode", "") == "all_duals" || continue
        baseline =
            _finite_float(row, "baseline_initial_residual_norm2")
        isnothing(baseline) && continue
        base_key = join(
            (
                get(row, "formulation", ""),
                get(row, "scenario_seed", ""),
                get(row, "direction", ""),
            ),
            "|",
        )
        push!(
            get!(base_reference_values, base_key, Float64[]),
            baseline,
        )
    end
    base_reference_floor = Dict(
        key => median(values) for
        (key, values) in base_reference_values
    )
    grouped = Dict{String, Vector{Dict{String, String}}}()
    for row in scaling_rows
        key = join(
            (
                get(row, "formulation", ""),
                get(row, "scenario_seed", ""),
                get(row, "direction", ""),
                get(row, "mode", ""),
            ),
            "|",
        )
        push!(get!(grouped, key, Dict{String, String}[]), row)
    end
    results = Dict{String, Any}[]
    for (key, rows) in sort(collect(grouped); by = first)
        first_row = first(rows)
        base_key = join(
            (
                get(first_row, "formulation", ""),
                get(first_row, "scenario_seed", ""),
                get(first_row, "direction", ""),
            ),
            "|",
        )
        base_resolution = get(base_reference_floor, base_key, 0.0)
        structural_baselines = Float64[]
        for row in rows
            value = _finite_float(
                row,
                "baseline_initial_residual_norm2",
            )
            !isnothing(value) &&
                push!(structural_baselines, value)
        end
        structural_baseline =
            isempty(structural_baselines) ? NaN :
            median(structural_baselines)
        available = NamedTuple[]
        for row in rows
            epsilon = _finite_float(row, "epsilon")
            r0 = _finite_float(row, "initial_residual_normalized")
            R0 = _finite_float(row, "initial_residual_norm2")
            perturbed_reference =
                something(
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
                !isnothing(R0) &&
                (
                    R0 > 1.25 * resolution ||
                    structural_floor
                )
            if !isnothing(epsilon) &&
               !isnothing(r0) &&
               !isnothing(R0) &&
               epsilon > 0.0 &&
               r0 > 0.0
                push!(
                    available,
                    (;
                        epsilon,
                        r0,
                        R0,
                        reliable,
                        resolution,
                    ),
                )
            end
        end
        sort!(available; by = value -> value.epsilon)
        reliable = filter(value -> value.reliable, available)
        fit_values = first(reliable, min(3, length(reliable)))
        x = log.([value.epsilon for value in fit_values])
        y = log.([value.r0 for value in fit_values])
        fit = _linear_fit(x, y)
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
            NaN : first(available).R0 / structural_baseline
        skip_reason =
            length(reliable) < 2 ?
            "fewer than two points exceed the reference-resolution floor and no resolvable structural baseline is present" :
            isnothing(fit) ? "degenerate epsilon coordinates" : ""
        push!(
            results,
            Dict{String, Any}(
                "case_id" => replace(key, "|" => "__"),
                "formulation" => get(first_row, "formulation", ""),
                "scenario_seed" => get(first_row, "scenario_seed", ""),
                "direction" => get(first_row, "direction", ""),
                "mode" => get(first_row, "mode", ""),
                "available_points" => length(available),
                "reliable_points" => length(reliable),
                "reference_resolution_floor_norm2" =>
                    resolution_floor,
                "structural_baseline_residual_norm2" =>
                    structural_baseline,
                "structural_baseline_to_resolution_ratio" =>
                    structural_ratio,
                "structural_floor_detected" =>
                    structural_floor_detected,
                "smallest_epsilon_residual_to_baseline_ratio" =>
                    smallest_ratio,
                "fit_points" => length(fit_values),
                "epsilon_min" =>
                    isempty(fit_values) ? NaN :
                    first(fit_values).epsilon,
                "epsilon_max" =>
                    isempty(fit_values) ? NaN :
                    last(fit_values).epsilon,
                "slope_log_r0_vs_log_epsilon" =>
                    isnothing(fit) ? NaN : fit.slope,
                "intercept" =>
                    isnothing(fit) ? NaN : fit.intercept,
                "r_squared" =>
                    isnothing(fit) ? NaN : fit.r_squared,
                "fit_performed" => !isnothing(fit),
                "skip_reason" => skip_reason,
            ),
        )
    end
    results
end

const SENSITIVITY_SUMMARY_COLUMNS = [
    "case_id",
    "formulation",
    "block",
    "preference_stratum",
    "reference_count",
    "near_null_computed_count",
    "raw_frobenius_per_sqrt_column_median",
    "column_scaled_frobenius_per_sqrt_column_median",
    "smallest_amplitude_directional_response_median",
    "near_null_psi_in_energy_median",
]

function _sensitivity_summary_rows(sensitivity_rows, config)
    results = Dict{String, Any}[]
    for formulation in config.formulations,
        block in ("psi_out", "psi_in")
        relevant = filter(
            row ->
                get(row, "formulation", "") == String(formulation) &&
                get(row, "block", "") == block &&
                _true_value(row, "reference_accepted"),
            sensitivity_rows,
        )
        observed_strata =
            sort(unique(get.(relevant, "preference_stratum", "")))
        filter!(value -> !isempty(value), observed_strata)
        for stratum in vcat(["all"], observed_strata)
            selected =
                stratum == "all" ? relevant :
                filter(
                    row ->
                        get(row, "preference_stratum", "") == stratum,
                    relevant,
                )
            by_reference =
                Dict{String, Vector{Dict{String, String}}}()
            for row in selected
                key = join(
                    (
                        get(row, "scenario_seed", ""),
                        get(row, "step", ""),
                    ),
                    "|",
                )
                push!(
                    get!(by_reference, key, Dict{String, String}[]),
                    row,
                )
            end
            raw = Float64[]
            scaled = Float64[]
            directional = Float64[]
            near_null_energy = Float64[]
            near_null_computed = 0
            for rows in values(by_reference)
                first_row = first(rows)
                for (column, target) in (
                    ("raw_frobenius_per_sqrt_column", raw),
                    (
                        "column_scaled_frobenius_per_sqrt_column",
                        scaled,
                    ),
                    ("near_null_psi_in_energy_median", near_null_energy),
                )
                    value = _finite_float(first_row, column)
                    !isnothing(value) && push!(target, value)
                end
                _true_value(first_row, "near_null_computed") &&
                    (near_null_computed += 1)
                amplitudes = [
                    (
                        _finite_float(row, "amplitude"),
                        _finite_float(
                            row,
                            "directional_residual_change_per_amplitude",
                        ),
                    ) for row in rows
                ]
                filter!(
                    pair ->
                        !isnothing(pair[1]) && !isnothing(pair[2]),
                    amplitudes,
                )
                if !isempty(amplitudes)
                    sort!(amplitudes; by = first)
                    push!(directional, first(amplitudes)[2])
                end
            end
            push!(
                results,
                Dict{String, Any}(
                    "case_id" =>
                        "$(formulation)__$(block)__$(stratum)",
                    "formulation" => formulation,
                    "block" => block,
                    "preference_stratum" => stratum,
                    "reference_count" => length(by_reference),
                    "near_null_computed_count" => near_null_computed,
                    "raw_frobenius_per_sqrt_column_median" =>
                        isempty(raw) ? NaN : median(raw),
                    "column_scaled_frobenius_per_sqrt_column_median" =>
                        isempty(scaled) ? NaN : median(scaled),
                    "smallest_amplitude_directional_response_median" =>
                        isempty(directional) ? NaN :
                        median(directional),
                    "near_null_psi_in_energy_median" =>
                        isempty(near_null_energy) ? NaN :
                        median(near_null_energy),
                ),
            )
        end
    end
    results
end

function _analysis_counts(config, replay_rows, reference_rows, sensitivity_rows, scaling_rows)
    Dict(
        "generated_at" => string(Dates.now()),
        "planned" => Dict(
            "reference_rows" =>
                length(config.formulations) *
                length(unique(vcat(
                    config.scenario_seeds,
                    config.sensitivity_seeds,
                    config.scaling_seeds,
                ))) *
                config.num_mpc_steps,
            "replay_rows" =>
                length(config.formulations) *
                length(config.scenario_seeds) *
                max(config.num_mpc_steps - 1, 0) *
                length(config.modes),
            "sensitivity_rows" =>
                length(config.formulations) *
                length(config.sensitivity_seeds) *
                count(
                    step -> 1 <= step <= config.num_mpc_steps,
                    unique(config.sensitivity_steps),
                ) *
                4 *
                length(config.sensitivity_amplitudes),
            "scaling_rows" =>
                length(config.formulations) *
                length(config.scaling_seeds) *
                config.scaling_directions *
                length(config.scaling_epsilons) *
                length(config.modes),
        ),
        "executed" => Dict(
            "reference_rows" => length(reference_rows),
            "reference_accepted_rows" => count(
                row -> _true_value(row, "reference_accepted"),
                reference_rows,
            ),
            "reference_accepted_above_replay_tol" => count(
                row -> begin
                    residual =
                        _finite_float(row, "direct_residual_norm2")
                    _true_value(row, "reference_accepted") &&
                        !isnothing(residual) &&
                        residual > config.replay_tol
                end,
                reference_rows,
            ),
            "replay_rows" => length(replay_rows),
            "replay_valid_reference_rows" => count(
                row -> _true_value(row, "valid_reference_pair"),
                replay_rows,
            ),
            "replay_solved_rows" =>
                count(row -> _solver_converged(row), replay_rows),
            "sensitivity_rows" => length(sensitivity_rows),
            "scaling_rows" => length(scaling_rows),
        ),
    )
end

function analyze_study(
    run_dir::AbstractString;
    config = load_config(joinpath(run_dir, "config.toml")),
)
    raw_dir = joinpath(run_dir, "raw")
    replay_rows = read_csv_rows(joinpath(raw_dir, "replay.csv"))
    reference_rows = read_csv_rows(joinpath(raw_dir, "references.csv"))
    sensitivity_rows = read_csv_rows(joinpath(raw_dir, "sensitivity.csv"))
    scaling_rows = read_csv_rows(joinpath(raw_dir, "scaling.csv"))
    paired = _paired_summary_rows(replay_rows, config)
    correlations = _correlation_rows(replay_rows, config)
    failures = _failure_rows(replay_rows, config)
    slopes = _scaling_slope_rows(scaling_rows, config)
    sensitivity_summary =
        _sensitivity_summary_rows(sensitivity_rows, config)
    _write_rows(
        joinpath(raw_dir, "paired_summary.csv"),
        PAIRED_COLUMNS,
        paired,
    )
    _write_rows(
        joinpath(raw_dir, "correlation_summary.csv"),
        CORRELATION_COLUMNS,
        correlations,
    )
    _write_rows(
        joinpath(raw_dir, "failure_summary.csv"),
        FAILURE_COLUMNS,
        failures,
    )
    _write_rows(
        joinpath(raw_dir, "scaling_slopes.csv"),
        SCALING_SLOPE_COLUMNS,
        slopes,
    )
    _write_rows(
        joinpath(raw_dir, "sensitivity_summary.csv"),
        SENSITIVITY_SUMMARY_COLUMNS,
        sensitivity_summary,
    )
    _write_toml(
        joinpath(run_dir, "analysis_manifest.toml"),
        _analysis_counts(
            config,
            replay_rows,
            reference_rows,
            sensitivity_rows,
            scaling_rows,
        ),
    )
    (;
        paired,
        correlations,
        failures,
        slopes,
        sensitivity_summary,
    )
end

function generate_figures(run_dir::AbstractString)
    if !isdefined(Main, :SelectiveWarmstartFigures)
        Base.include(Main, joinpath(STUDY_DIR, "Figures.jl"))
    end
    figure_module =
        Base.invokelatest(getfield, Main, :SelectiveWarmstartFigures)
    figure_function =
        Base.invokelatest(getfield, figure_module, :generate_figures)
    Base.invokelatest(figure_function, run_dir)
end

function _markdown_escape(value)
    replace(string(value), "|" => "\\|", "\n" => " ")
end

function _display_number(value; digits = 4)
    value === nothing && return "—"
    parsed =
        value isa Number ? Float64(value) :
        tryparse(Float64, string(value))
    isnothing(parsed) && return _markdown_escape(value)
    !isfinite(parsed) && return "—"
    abs(parsed) != 0.0 &&
    (abs(parsed) < 1e-3 || abs(parsed) >= 1e4) &&
        return @sprintf("%.3e", parsed)
    @sprintf("%.*f", digits, parsed)
end

function _report_table(io, headers, rows)
    println(io, "| ", join(headers, " | "), " |")
    println(io, "| ", join(fill("---", length(headers)), " | "), " |")
    for row in rows
        println(
            io,
            "| ",
            join((_markdown_escape(value) for value in row), " | "),
            " |",
        )
    end
    isempty(rows) &&
        println(io, "| ", join(vcat(["No observations"], fill("", length(headers)-1)), " | "), " |")
    println(io)
end

function _report_counts(io, run_dir)
    path = joinpath(run_dir, "analysis_manifest.toml")
    if !isfile(path)
        println(io, "Analysis counts are unavailable because `analysis_manifest.toml` is missing.\n")
        return
    end
    counts = TOML.parsefile(path)
    planned = get(counts, "planned", Dict())
    executed = get(counts, "executed", Dict())
    rows = Any[]
    for (label, key) in (
        ("Reference solves", "reference_rows"),
        ("Replay mode runs", "replay_rows"),
        ("Sensitivity rows", "sensitivity_rows"),
        ("Scaling mode runs", "scaling_rows"),
    )
        push!(
            rows,
            (
                label,
                get(planned, key, "—"),
                get(executed, key, "—"),
            ),
        )
    end
    _report_table(io, ["Study component", "Planned", "Recorded"], rows)
    println(
        io,
        "Accepted references: **",
        get(executed, "reference_accepted_rows", 0),
        "**; replay rows with valid source/destination references: **",
        get(executed, "replay_valid_reference_rows", 0),
        "**; replay rows reporting convergence: **",
        get(executed, "replay_solved_rows", 0),
        "**.\n",
    )
end

function _report_solver_options(io, config)
    options = _solver_options_dict(config)
    names = sort(collect(keys(options)))
    _report_table(
        io,
        ["Option", "All solver calls"],
        [
            (
                name,
                get(options, name, "—"),
            ) for name in names
        ],
    )
end

function _report_provenance(io, run_dir, environment)
    provenance_manifest =
        joinpath(run_dir, "provenance", "manifest.toml")
    if !isfile(provenance_manifest)
        println(
            io,
            "**Legacy-run provenance warning:** this run predates exact environment/source snapshotting. Its `environment.toml` hashes and Git dirty-status record are informative, but absent `environment/` and `provenance/` artifacts mean the dirty measurement state cannot be reconstructed exactly. Do not treat regenerated analysis from this legacy run as exactly reproducible.\n",
        )
        return
    end
    drift_path = joinpath(run_dir, "provenance", "drift.toml")
    drift = isfile(drift_path) ? TOML.parsefile(drift_path) : Dict()
    println(
        io,
        "- Creation measurement-code fingerprint: `",
        get(
            environment,
            "measurement_code_fingerprint",
            "unavailable",
        ),
        "`.\n",
        "- Measurement-code drift at report generation: `",
        get(drift, "measurement_code_drift_detected", "unavailable"),
        "`; Project/Manifest drift: `",
        get(drift, "environment_drift_detected", "unavailable"),
        "`.\n",
    )
    println(
        io,
        "Raw measurements are attributed to the creation fingerprint. If drift is true, regenerated analysis and this report use the current fingerprint recorded in [`provenance/drift.toml`](provenance/drift.toml); this distinction prevents later diagnostic-definition edits from being mistaken for measurement code. Exact environment snapshots are [`environment/Project.toml`](environment/Project.toml) and [`environment/Manifest.toml`](environment/Manifest.toml). The dirty tracked patch is [`provenance/git_diff_binary.patch`](provenance/git_diff_binary.patch), and exact relevant source copies plus hashes are indexed by [`provenance/manifest.toml`](provenance/manifest.toml).\n",
    )
end

function _report_kkt_mapping(io, run_dir)
    rows =
        read_csv_rows(joinpath(run_dir, "raw", "kkt_systems.csv"))
    display_rows = [
        (
            get(row, "formulation", ""),
            get(row, "kkt_dimension", ""),
            get(row, "variable_dimension", ""),
            get(row, "z_dimension", ""),
            get(row, "lambda_dimension", ""),
            get(row, "psi_out_dimension", ""),
            get(row, "psi_in_dimension", ""),
            get(row, "unclassified_dimension", ""),
        ) for row in rows
    ]
    _report_table(
        io,
        [
            "Formulation",
            "KKT rows",
            "Variables",
            "z",
            "λ",
            "ψout",
            "ψin",
            "Other",
        ],
        display_rows,
    )
end

function _report_paired_summaries(io, run_dir)
    rows =
        read_csv_rows(joinpath(run_dir, "raw", "paired_summary.csv"))
    display_rows = Any[]
    for row in rows
        a_summary = string(
            _display_number(get(row, "mode_a_median", "")),
            " [",
            _display_number(get(row, "mode_a_q1", "")),
            ", ",
            _display_number(get(row, "mode_a_q3", "")),
            "]",
        )
        b_summary = string(
            _display_number(get(row, "mode_b_median", "")),
            " [",
            _display_number(get(row, "mode_b_q1", "")),
            ", ",
            _display_number(get(row, "mode_b_q3", "")),
            "]",
        )
        ci = string(
            "[",
            _display_number(get(row, "bootstrap_difference_low", "")),
            ", ",
            _display_number(get(row, "bootstrap_difference_high", "")),
            "]",
        )
        wtl =
            startswith(get(row, "directional_scoring", ""), "none_") ?
            "not scored (churn)" :
            join(
                (
                    get(row, "wins_a", "0"),
                    get(row, "ties", "0"),
                    get(row, "losses_a", "0"),
                ),
                "/",
            )
        push!(
            display_rows,
            (
                get(row, "formulation", ""),
                get(row, "comparison", ""),
                get(row, "preference_stratum", ""),
                get(row, "metric", ""),
                get(row, "valid_pairs", "0"),
                a_summary,
                b_summary,
                _display_number(
                    get(
                        row,
                        "median_paired_difference_a_minus_b",
                        "",
                    ),
                ),
                _display_number(
                    get(row, "median_paired_ratio_a_over_b", ""),
                ),
                ci,
                wtl,
                string(
                    get(row, "mode_a_failures", "0"),
                    "/",
                    get(row, "mode_b_failures", "0"),
                ),
            ),
        )
    end
    _report_table(
        io,
        [
            "Form.",
            "Comparison",
            "Stratum",
            "Metric",
            "n",
            "A median [IQR]",
            "B median [IQR]",
            "Median A−B",
            "Median A/B",
            "95% paired-bootstrap CI (A−B)",
            "A W/T/L",
            "Failures A/B",
        ],
        display_rows,
    )
    println(
        io,
        "Here A is `all_except_innermost_stationarity`; B is `equality_duals` or `all_duals` as named. Wins respect the metric direction. Total regularization-change count pools beneficial decreases with adverse increases and is therefore reported only as nondirectional churn; η retries remain the adverse regularization metric. Solver-behavior metrics require both runs to converge; initialization-residual metrics require valid references but retain runs whose later solve failed. Failure counts include only attempted solves with valid source/destination references. The deterministic bootstrap resamples paired transitions, not scenario clusters, so serial within-seed dependence can make its intervals anti-conservative. Smoke/pilot profiles with only one or two seeds do not support robust scenario-level inference. Runtime is secondary and may still reflect cache effects.\n",
    )
end

function _report_correlations(io, run_dir)
    rows = read_csv_rows(
        joinpath(run_dir, "raw", "correlation_summary.csv"),
    )
    selected = filter(
        row ->
            get(row, "mode_scope", "") == "pooled" &&
            get(row, "preference_stratum", "") == "all",
        rows,
    )
    display_rows = [
        (
            get(row, "formulation", ""),
            get(row, "outcome", ""),
            get(row, "sample_count", ""),
            _display_number(get(row, "pearson_log_r0", "")),
            _display_number(get(row, "spearman_log_r0", "")),
        ) for row in selected
    ]
    _report_table(
        io,
        ["Formulation", "Outcome", "n", "Pearson", "Spearman"],
        display_rows,
    )
    println(
        io,
        "These are descriptive associations between `log(r₀)` and the outcome, not causal estimates. `eta_retry_count` is adverse; `regularization_change_count` is nondirectional churn because it pools increases, decreases, and other changed-η events. Pooled estimates can be confounded by mode, transition difficulty, or preference stratum; mode-specific rows are preserved in [`raw/correlation_summary.csv`](raw/correlation_summary.csv).\n",
    )
end

function _report_scaling(io, run_dir)
    rows =
        read_csv_rows(joinpath(run_dir, "raw", "scaling_slopes.csv"))
    display_rows = [
        (
            get(row, "formulation", ""),
            get(row, "scenario_seed", ""),
            get(row, "direction", ""),
            get(row, "mode", ""),
            string(
                get(row, "fit_points", "0"),
                "/",
                get(row, "available_points", "0"),
            ),
            get(row, "structural_floor_detected", ""),
            _display_number(
                get(
                    row,
                    "structural_baseline_residual_norm2",
                    "",
                ),
            ),
            _display_number(
                get(
                    row,
                    "reference_resolution_floor_norm2",
                    "",
                ),
            ),
            _display_number(
                get(
                    row,
                    "structural_baseline_to_resolution_ratio",
                    "",
                ),
            ),
            _display_number(
                get(row, "slope_log_r0_vs_log_epsilon", ""),
            ),
            _display_number(get(row, "r_squared", "")),
            get(row, "skip_reason", ""),
        ) for row in rows
    ]
    _report_table(
        io,
        [
            "Form.",
            "Seed",
            "Direction",
            "Mode",
            "Fit/available",
            "Structural floor?",
            "Baseline R₀",
            "Resolution floor",
            "Baseline/resolution",
            "Slope",
            "R²",
            "Skip reason",
        ],
        display_rows,
    )
    println(
        io,
        "The post-smoke diagnostic separates two quantities that the raw `numerical_floor` field originally combined: the reference/numerical resolution floor (estimated from the base all-dual residual and perturbed-reference residual) and the mode's actual ε=0 structural baseline. A structural baseline more than 1.25× the resolution is treated as signal, not noise, so a stable nonvanishing floor is retained and should produce a near-zero log–log slope. Fits use at most the three smallest resolved points. A blank slope means neither a resolved perturbation response nor a resolvable structural floor supplied two points; no power law is forced.\n",
    )
end

function _report_sensitivity(io, run_dir)
    rows = read_csv_rows(
        joinpath(run_dir, "raw", "sensitivity_summary.csv"),
    )
    display_rows = [
        (
            get(row, "formulation", ""),
            get(row, "block", ""),
            get(row, "preference_stratum", ""),
            get(row, "reference_count", ""),
            _display_number(
                get(
                    row,
                    "raw_frobenius_per_sqrt_column_median",
                    "",
                ),
            ),
            _display_number(
                get(
                    row,
                    "column_scaled_frobenius_per_sqrt_column_median",
                    "",
                ),
            ),
            _display_number(
                get(
                    row,
                    "smallest_amplitude_directional_response_median",
                    "",
                ),
            ),
            _display_number(
                get(row, "near_null_psi_in_energy_median", ""),
            ),
            get(row, "near_null_computed_count", ""),
        ) for row in rows
    ]
    _report_table(
        io,
        [
            "Form.",
            "Block",
            "Stratum",
            "Refs",
            "Raw ‖Jb‖F/√d",
            "Column-scaled/√d",
            "Small-a response",
            "Near-null ψin energy",
            "SVD refs",
        ],
        display_rows,
    )
    println(
        io,
        "The `all` rows provide pooled context; satisfied, boundary, and violated rows make the requested feasibility stratification explicit. Column scaling uses `max(1, |y_j|)` per coordinate and is reported alongside—not in place of—the raw norm. Near-null energy is a property of the full Jacobian and is repeated across block rows only to keep each stratum table self-contained.\n",
    )
end

function _root_evaluability_summary(run_dir, config)
    roots =
        read_csv_rows(joinpath(run_dir, "raw", "root_spread.csv"))
    replay = read_csv_rows(joinpath(run_dir, "raw", "replay.csv"))
    valid_reference_keys = Set(
        _replay_key(row) for row in replay if
        _true_value(row, "valid_reference_pair")
    )
    reference_valid(row) = _replay_key(row) in valid_reference_keys
    enough_converged_modes(row) =
        something(_finite_float(row, "converged_mode_count"), 0.0) >=
        2.0
    evaluable = filter(
        row -> reference_valid(row) && enough_converged_modes(row),
        roots,
    )
    flagged = count(
        row -> _true_value(row, "materially_different_roots"),
        evaluable,
    )
    invalid_reference_count =
        count(row -> !reference_valid(row), roots)
    insufficient_mode_count = count(
        row ->
            reference_valid(row) && !enough_converged_modes(row),
        roots,
    )
    maximum_spread = maximum(
        (
            something(
                _finite_float(
                    row,
                    "max_pairwise_primal_distance_normalized",
                ),
                0.0,
            ) for row in evaluable
        );
        init = 0.0,
    )
    planned =
        length(config.formulations) *
        length(config.scenario_seeds) *
        max(config.num_mpc_steps - 1, 0)
    (;
        planned,
        recorded = length(roots),
        evaluable = length(evaluable),
        flagged,
        unevaluable = length(roots) - length(evaluable),
        invalid_reference_count,
        insufficient_mode_count,
        maximum_spread,
    )
end

function _attempted_status(value)
    status = lowercase(strip(string(value)))
    !isempty(status) &&
        !startswith(status, "not_run") &&
        status != "unavailable"
end

function _scaling_failure_summary(run_dir, config)
    scaling =
        read_csv_rows(joinpath(run_dir, "raw", "scaling.csv"))
    reference_cases = Dict{String, Dict{String, String}}()
    for row in scaling
        key = join(
            (
                get(row, "formulation", ""),
                get(row, "scenario_seed", ""),
                get(row, "direction", ""),
                get(row, "epsilon", ""),
            ),
            "|",
        )
        reference_cases[key] = row
    end
    cases = collect(values(reference_cases))
    reference_accepted(row) =
        _true_value(row, "perturbed_reference_accepted")
    reference_attempted(row) =
        reference_accepted(row) ||
        _attempted_status(
            get(row, "perturbed_reference_status", ""),
        )
    reference_rows = NamedTuple[]
    for formulation in String.(config.formulations)
        formulation_cases = filter(
            row -> get(row, "formulation", "") == formulation,
            cases,
        )
        isempty(formulation_cases) && continue
        attempts = count(reference_attempted, formulation_cases)
        accepted = count(reference_accepted, formulation_cases)
        push!(
            reference_rows,
            (;
                formulation,
                cases = length(formulation_cases),
                attempts,
                accepted,
                failed = attempts - accepted,
                not_attempted =
                    length(formulation_cases) - attempts,
            ),
        )
    end

    solve_attempted(row) =
        _true_value(row, "direct_converged") ||
        _attempted_status(get(row, "solver_status", ""))
    mode_rows = NamedTuple[]
    for formulation in String.(config.formulations),
        mode in String.(config.modes)
        group = filter(
            row ->
                get(row, "formulation", "") == formulation &&
                get(row, "mode", "") == mode,
            scaling,
        )
        isempty(group) && continue
        valid_rows = filter(
            row -> _true_value(row, "valid_reference_pair"),
            group,
        )
        attempted_rows = filter(solve_attempted, valid_rows)
        direct_converged = count(
            row -> _true_value(row, "direct_converged"),
            attempted_rows,
        )
        push!(
            mode_rows,
            (;
                formulation,
                mode,
                valid_reference_rows = length(valid_rows),
                solve_attempts = length(attempted_rows),
                direct_converged,
                failed =
                    length(attempted_rows) - direct_converged,
                valid_not_run = length(valid_rows) -
                                length(attempted_rows),
            ),
        )
    end
    reference_totals = (;
        cases = sum(row.cases for row in reference_rows; init = 0),
        attempts = sum(
            row.attempts for row in reference_rows;
            init = 0,
        ),
        accepted = sum(
            row.accepted for row in reference_rows;
            init = 0,
        ),
        failed = sum(row.failed for row in reference_rows; init = 0),
        not_attempted = sum(
            row.not_attempted for row in reference_rows;
            init = 0,
        ),
    )
    mode_totals = (;
        valid_reference_rows = sum(
            row.valid_reference_rows for row in mode_rows;
            init = 0,
        ),
        solve_attempts =
            sum(row.solve_attempts for row in mode_rows; init = 0),
        direct_converged = sum(
            row.direct_converged for row in mode_rows;
            init = 0,
        ),
        failed = sum(row.failed for row in mode_rows; init = 0),
        valid_not_run = sum(
            row.valid_not_run for row in mode_rows;
            init = 0,
        ),
    )
    (; reference_rows, reference_totals, mode_rows, mode_totals)
end

function _report_failures_and_roots(io, run_dir, config)
    println(io, "### Replay mode-run failures\n")
    failures =
        read_csv_rows(joinpath(run_dir, "raw", "failure_summary.csv"))
    failure_rows = [
        (
            get(row, "formulation", ""),
            get(row, "mode", ""),
            get(row, "total_rows", ""),
            get(row, "valid_reference_rows", ""),
            get(row, "direct_converged_rows", ""),
            get(row, "raw_solved_rows", ""),
            get(row, "raw_failed_rows", ""),
            get(row, "raw_direct_disagreements", ""),
            get(row, "exception_rows", ""),
            get(row, "reference_unavailable_rows", ""),
        ) for row in failures
    ]
    _report_table(
        io,
        [
            "Form.",
            "Mode",
            "Rows",
            "Valid refs",
            "Direct converged",
            "Raw solved",
            "Raw failed",
            "Raw/direct disagree",
            "Exceptions",
            "Ref unavailable",
        ],
        failure_rows,
    )
    println(io, "### Scaling reference and mode-run failures\n")
    scaling = _scaling_failure_summary(run_dir, config)
    reference_totals = scaling.reference_totals
    println(
        io,
        "The raw scaling rows contain **",
        reference_totals.cases,
        "** unique perturbed-reference cases. **",
        reference_totals.attempts,
        "** perturbed references were attempted: **",
        reference_totals.accepted,
        "** passed the direct acceptance criterion and **",
        reference_totals.failed,
        "** failed it; **",
        reference_totals.not_attempted,
        "** cases were not attempted because a prerequisite reference was unavailable.\n",
    )
    _report_table(
        io,
        [
            "Form.",
            "Unique cases",
            "Attempts",
            "Accepted",
            "Failed",
            "Not attempted",
        ],
        [
            (
                row.formulation,
                row.cases,
                row.attempts,
                row.accepted,
                row.failed,
                row.not_attempted,
            ) for row in scaling.reference_rows
        ],
    )
    mode_totals = scaling.mode_totals
    println(
        io,
        "Among **",
        mode_totals.valid_reference_rows,
        "** scaling mode rows with valid base and perturbed references, **",
        mode_totals.solve_attempts,
        "** mode solves were attempted: **",
        mode_totals.direct_converged,
        "** direct-converged and **",
        mode_totals.failed,
        "** failed the direct replay-tolerance criterion; **",
        mode_totals.valid_not_run,
        "** reference-valid rows were not run. Reference-unavailable and configured non-runs are excluded from the failure count.\n",
    )
    _report_table(
        io,
        [
            "Form.",
            "Mode",
            "Valid reference rows",
            "Solve attempts",
            "Direct converged",
            "Failed attempts",
            "Valid not run",
        ],
        [
            (
                row.formulation,
                row.mode,
                row.valid_reference_rows,
                row.solve_attempts,
                row.direct_converged,
                row.failed,
                row.valid_not_run,
            ) for row in scaling.mode_rows
        ],
    )

    println(io, "### Candidate-solution separation\n")
    roots = _root_evaluability_summary(run_dir, config)
    println(
        io,
        "Of **",
        roots.planned,
        "** planned replay-transition root records, **",
        roots.recorded,
        "** were recorded. A recorded transition is evaluable only when its source/destination references are valid and at least two modes converged with stored final primals. **",
        roots.evaluable,
        "** recorded transitions were evaluable, and **",
        roots.flagged,
        "/",
        roots.evaluable,
        "** evaluable transitions exceeded the preregistered relative primal-distance flag `",
        config.material_root_rtol,
        "`. The other **",
        roots.unevaluable,
        "** recorded transitions were unevaluable: **",
        roots.invalid_reference_count,
        "** lacked a valid source/destination reference pair and **",
        roots.insufficient_mode_count,
        "** had valid references but fewer than two converged modes. The largest normalized pairwise spread among evaluable transitions was **",
        _display_number(roots.maximum_spread),
        "**. This is an approximate candidate-root diagnostic, not evidence of distinct exact roots: replay solutions are accepted at direct residual tolerance `",
        config.replay_tol,
        "`, while candidate separation uses a normalized primal-distance flag `",
        config.material_root_rtol,
        "`. These are different quantities, and no error bound links them at the current solve tolerance. The legacy raw field `materially_different_roots` should therefore be read as “materially separated accepted candidate solutions.” Per-transition mode pairs are in [`raw/root_spread.csv`](raw/root_spread.csv).\n",
    )
    sensitivity =
        read_csv_rows(joinpath(run_dir, "raw", "sensitivity.csv"))
    skipped = count(
        row ->
            lowercase(get(row, "near_null_computed", "")) != "true" &&
            !isempty(get(row, "near_null_skip_reason", "")),
        sensitivity,
    )
    println(
        io,
        "Near-null SVD diagnostics were skipped or failed in **",
        skipped,
        "** sensitivity rows (the reason and dimension cap are recorded row by row). Raw Jacobian block norms and finite perturbation responses remain available even when SVD is skipped.\n",
    )
end

function _report_observations(io, run_dir)
    rows =
        read_csv_rows(joinpath(run_dir, "raw", "paired_summary.csv"))
    primary = filter(
        row ->
            get(row, "preference_stratum", "") == "all" &&
            get(row, "metric", "") in
            ("initial_residual_normalized", "total_inner_iters") &&
            something(_finite_float(row, "valid_pairs"), 0.0) > 0.0,
        rows,
    )
    if isempty(primary)
        println(
            io,
            "No valid replay pairs were recorded, so there is no empirical ordering of warm-start modes from replay.\n",
        )
    else
        println(
            io,
            "The following are compact empirical effect estimates; they are not proofs and should be read with the failure and candidate-solution-spread audits above:\n",
        )
        for row in primary
            println(
                io,
                "- ",
                uppercasefirst(get(row, "formulation", "")),
                ", `",
                get(row, "comparison", ""),
                "`, ",
                get(row, "metric", ""),
                ": n=",
                get(row, "valid_pairs", "0"),
                ", median A−B=",
                _display_number(
                    get(row, "median_paired_difference_a_minus_b", ""),
                ),
                ", 95% paired-bootstrap CI ",
                "[",
                _display_number(
                    get(row, "bootstrap_difference_low", ""),
                ),
                ", ",
                _display_number(
                    get(row, "bootstrap_difference_high", ""),
                ),
                "], W/T/L=",
                join(
                    (
                        get(row, "wins_a", "0"),
                        get(row, "ties", "0"),
                        get(row, "losses_a", "0"),
                    ),
                    "/",
                ),
                ".",
            )
        end
        println(io)
    end
    scaling = read_csv_rows(
        joinpath(run_dir, "raw", "scaling_slopes.csv"),
    )
    scaling_observations = String[]
    for formulation in ("reduced", "quasi"),
        mode in (
            "all_except_innermost_stationarity",
            "all_duals",
        )
        selected = filter(
            row ->
                get(row, "formulation", "") == formulation &&
                get(row, "mode", "") == mode,
            scaling,
        )
        isempty(selected) && continue
        slopes = Float64[]
        for row in selected
            slope = _finite_float(
                row,
                "slope_log_r0_vs_log_epsilon",
            )
            !isnothing(slope) && push!(slopes, slope)
        end
        structural_count = count(
            row ->
                lowercase(
                    get(row, "structural_floor_detected", ""),
                ) == "true",
            selected,
        )
        push!(
            scaling_observations,
            string(
                "- ",
                uppercasefirst(formulation),
                ", `",
                mode,
                "`: ",
                length(slopes),
                "/",
                length(selected),
                " direction/seed groups had a resolved slope; median slope ",
                isempty(slopes) ? "—" : _display_number(median(slopes)),
                "; ",
                structural_count,
                "/",
                length(selected),
                " groups had an ε=0 structural baseline above reference resolution.",
            ),
        )
    end
    if !isempty(scaling_observations)
        println(
            io,
            "Scaling diagnostics relevant to the preregistered `O(ε)` hypothesis:\n",
        )
        for observation in scaling_observations
            println(io, observation)
        end
        println(
            io,
            "\nA slope near 1 without a structural-floor flag is consistent with `O(ε)` over the resolved range; a near-zero slope with a flag is consistent with a nonvanishing floor. Neither pattern proves an asymptotic law. Smoke/pilot direction and seed counts are descriptive and too small for robust scenario-level inference.\n",
        )
    end
end

function generate_report(
    run_dir::AbstractString;
    config = load_config(joinpath(run_dir, "config.toml")),
)
    environment_path = joinpath(run_dir, "environment.toml")
    environment =
        isfile(environment_path) ? TOML.parsefile(environment_path) : Dict()
    report_path = joinpath(run_dir, "report.md")
    io = IOBuffer()
    println(io, "# Selective primal–dual warm-start study\n")
    println(
        io,
        "This report is generated from the checkpointed, incrementally persisted row-level files in this run directory. It is deliberately written for independent interpretation and does not treat an empirical pattern as a theorem.\n",
    )
    println(io, "## Research questions and preregistered hypotheses\n")
    println(
        io,
        "1. Does retaining non-innermost hierarchical stationarity multipliers improve initialization relative to primal plus equality duals?\n",
        "2. Is resetting innermost-stationarity multipliers beneficial, neutral, or harmful relative to retaining them?\n",
        "3. Does the mechanism appear in both reduced and quasi GOOP?\n",
        "4. Is a smaller initial KKT residual associated with earlier full Newton steps, less globalization/regularization, and fewer iterations?\n",
        "5. Under a small initial-state perturbation, is the `all_except` residual consistent with `O(ε)`, while resetting identifiable nonzero blocks can create a residual floor?\n",
    )
    println(
        io,
        "The two primary paired comparisons were fixed in advance: `all_except_innermost_stationarity` versus `equality_duals`, and versus `all_duals`. The suite also preregisters the alternative outcomes that all-dual retention helps, that transported innermost duals hurt first-step behavior, that iteration changes occur without residual changes, or that accepted candidate solutions appear materially separated (which motivates, but does not by itself establish, different exact primal roots).\n",
    )
    println(io, "### Conditional outcome interpretation framework\n")
    println(
        io,
        "- If `all_except` beats `equality_duals` in both initial residual and iterations, that is evidence—within this protocol—that retaining `ψout` improves KKT tracking.\n",
        "- If `all_except` and `all_duals` have paired effects small relative to the observed transition variability and intervals spanning zero, the study detects no clear benefit from `ψin`; this is not a formal equivalence result or evidence that resetting it is better.\n",
        "- If `all_except` beats `all_duals` and all-dual runs also accept shorter first steps, backtrack more, or retry regularization more, transported `ψin` is implicated in worse globalization/conditioning behavior.\n",
        "- If `all_duals` beats `all_except`, the innermost multipliers contain useful tracking information and the selective-reset claim is contradicted for those cases.\n",
        "- If iterations improve without a smaller initial residual, the mechanism is more consistent with globalization or regularization state than with local KKT distance alone.\n",
        "- If accepted mode solutions remain materially separated, iteration comparisons alone cannot support the proposed tracking explanation; tighter follow-up solves are required before treating root identity as a competing explanation.\n",
    )

    println(io, "## Code-to-mathematics mapping\n")
    println(
        io,
        "- `z` is `kkt.primal_dims`, exposed as `kkt_variable_blocks(kkt).z`.\n",
        "- `λ` is `kkt.equality_constraint_dual_dims`, exposed as `.λ`; it includes dynamics costates and all equality multipliers packed by the generated KKT system.\n",
        "- `ψout` is the subset of `kkt.stationarity_dual_dims` whose owning stationarity equation is not innermost, exposed as `.ψ_out`.\n",
        "- `ψin` is `kkt.innermost_stationarity_dual_dims`, exposed as `.ψ_in`. These are multipliers *attached to* innermost stationarity equations, not the equations themselves.\n",
        "- Preference slacks, interior-point slacks, and ordinary inequality duals are not members of the four mathematical blocks. The production constructor starts every mode from the same solver-default full vector, then writes exactly the selected blocks, so those other coordinates remain identical.\n",
    )
    println(
        io,
        "The repository's existing spelling is `:primal_only` (singular), which is retained. Only trajectory primals receive a receding-horizon time shift. Dual coordinates are transported by their fixed generated-KKT coordinate identity; no unverified time-index permutation is imposed. Terminal completion drops the executed knot, retains the remaining controls, advances the last retained state with the last retained control at `planning_horizon - 1`, and appends the unused zero terminal control.\n",
    )
    println(
        io,
        "For cross-formulation control, reduced GOOP (or the first configured formulation if reduced is absent) is the sole sequence driver. Its accepted cold-reference primal determines `pₜ₊₁`, and the resulting instance sequence is persisted under `checkpoints/sequences/`; both formulations solve the exact same per-step instance digest. Every numerical reference step in both formulations starts from a fresh destination-specific default primal, so neither formulation transports a previous reference solution while constructing the canonical sequence. Previous reference `z` and shifted primals remain stored solely as replay sources. If the driver reference is not directly accepted, later common-sequence steps are marked unavailable rather than generated from a different formulation.\n",
    )
    _report_kkt_mapping(io, run_dir)

    println(io, "## Implementation changes\n")
    println(
        io,
        "- Added production KKT block metadata, isolated full-vector warm-start construction for all four nested modes, and the `:all_duals` mode.\n",
        "- Centralized the robotic-arm trajectory shift and its terminal completion.\n",
        "- Added an optional solver trace hook that observes the initial residual, line-search trials, accepted steps, regularization changes, failures, and completion counters without changing the uninstrumented return type or decisions.\n",
        "- Canonical and perturbed-scaling references use one mode-neutral `cold_default` initialization per destination, without a continuation tournament or rescue; this post-pilot rule is versioned in `config.toml`.\n",
        "- Added the serialized `uniform_t20_dt0p1_tol0p008_max1000_v1` comparability guard. Every profile and stage uses `T=20`, `Δt=0.1`, and the same solver options, including compilation warmups; legacy configurations cannot resume into this protocol.\n",
        "- Added fixed-sequence canonical replay, sensitivity, perturbation-scaling, atomic JLD2 checkpoints, incremental one-line CSV output, deterministic randomization, paired bootstrap analysis, six figures, and this report. Every measured case also performs an identical copied-warm-start uninstrumented duplicate; `solve_time_sec` comes from that duplicate and `instrumented_solve_time_sec` is retained separately.\n",
    )

    println(io, "## Configuration and reproduction\n")
    println(
        io,
        "- Profile: `",
        config.profile,
        "`; horizon: ",
        config.planning_horizon,
        "; Δt: ",
        config.Δt,
        "; MPC steps: ",
        config.num_mpc_steps,
        "; formulations: `",
        join(String.(config.formulations), "`, `"),
        "`; comparability protocol: `",
        config.comparability_protocol,
        "`.\n",
        "- Git commit recorded at run creation: `",
        get(environment, "git_commit", "unavailable"),
        "`; dirty tree: `",
        get(environment, "git_dirty", "unavailable"),
        "`.\n",
        "- Julia: `",
        get(environment, "julia_version", "unavailable"),
        "`; threads: ",
        get(environment, "julia_threads", "unavailable"),
        "; platform: `",
        get(environment, "platform", "unavailable"),
        "`.\n",
    )
    _report_provenance(io, run_dir, environment)
    println(io, "From the repository root:\n")
    println(io, "```bash")
    println(
        io,
        "julia --project=experiments experiments/analysis/selective_warmstart/run.jl --profile ",
        config.profile,
    )
    println(
        io,
        "julia --project=experiments experiments/analysis/selective_warmstart/run.jl --resume \"",
        abspath(run_dir),
        "\"",
    )
    println(
        io,
        "julia --project=experiments experiments/analysis/selective_warmstart/run.jl --resume \"",
        abspath(run_dir),
        "\" --stages analysis,figures,report",
    )
    println(io, "```\n")
    println(
        io,
        "The first command creates a timestamped run. The second resumes missing checkpoint cases without re-running completed rows. Exact configuration and environment snapshots are [`config.toml`](config.toml), [`environment.toml`](environment.toml), [`manifest.toml`](manifest.toml), and [`analysis_manifest.toml`](analysis_manifest.toml).\n",
    )
    println(io, "### Solver options\n")
    println(
        io,
        "The single option set below is used unchanged by canonical-reference, replay, perturbed-scaling-reference, scaling-mode, and compilation-warmup solves. The warmup is compilation-only, excluded from measurements, and runs once per formulation rather than once per seed. Sensitivity invokes no solver; it uses the same `T=20`, `Δt=0.1` KKT system and accepted canonical points. `tol=0.008` and every solver option except the iteration cap match the headless robotic-arm receding baseline in `Robotic_arm_mpc.jl`. The cap is uniformly `max_inner_iters=1000`, raised from that baseline's 500 after a direct reduced T20 seed101 cold solve required 717 iterations and reached residual `0.007991129782359604`. The visualization-oriented `Robotic_arm_receding.jl` differs only by setting `record_convergence=true`. `T=20` is this study's explicit comparability override (the scenario/receding default is `T=30`), while `Δt=0.1` matches that baseline.\n",
    )
    _report_solver_options(io, config)

    println(io, "## Canonical-reference qualification\n")
    reference_accuracy =
        _reference_accuracy_summary(run_dir, config)
    println(
        io,
        "**Reference-initialization disclosure:** every canonical and perturbed-scaling reference makes one fresh destination-specific `cold_default` solve. This rule is mode-neutral relative to the four replay transport modes: it does not transport a previous reference primal or dual, and it is not a tournament or rescue rule. It was adopted after the 0.01 pilot's all-dual continuation failed at the first unavailable reference in four formulation/seed sequences; targeted diagnostics accepted fresh cold/default starts in all four cases without changing any solver option. The protocol fields `reference_initialization = \"cold_default_each_step\"` and `comparability_protocol = \"uniform_t20_dt0p1_tol0p008_max1000_v1\"` require a fresh run and prevent legacy checkpoints from being mixed with this rule. Mode-neutral does not mean basin-neutral or unbiased: a cold initializer can still select a numerical root basin.\n",
    )
    println(
        io,
        "Canonical-reference solves use the same solver tolerance `",
        config.requested_reference_tol,
        "` and `max_inner_iters = ",
        config.reference_max_inner_iters,
        "` as replay and scaling. Acceptance is independently audited by directly evaluating `‖K(y; p, εfinal, η=0)‖₂ ≤ ",
        config.reference_acceptance_tol,
        "`, exactly the replay/scaling threshold. Raw solver status and the direct 2-/infinity-norm residuals are stored separately. For replay/scaling, `direct_converged`—the independently evaluated direct residual against the same tolerance—is authoritative for converged-only analysis; raw status is never overwritten and raw/direct disagreements are counted. A failed or inaccurate source/destination canonical point invalidates that replay pair but remains in the CSV. The fixed barrier value at initialization is `ε₀ = ",
        EPSILON0,
        "`.\n",
    )
    println(
        io,
        "These accepted points generate the canonical trajectory and provide common comparison endpoints; they are **not higher-accuracy ground truth**. Block errors, candidate-root distances, sensitivity evaluations, and scaling conclusions are therefore conditioned on canonical points accepted at residual `≤ ",
        config.reference_acceptance_tol,
        "`. Recorded accepted canonical points: ",
        get(reference_accuracy, "accepted_canonical_references", 0),
        "; accepted perturbed-scaling points: ",
        get(
            reference_accuracy,
            "accepted_perturbed_scaling_references",
            0,
        ),
        ".\n",
    )

    println(io, "## Executed versus planned sample counts\n")
    _report_counts(io, run_dir)

    println(io, "## Complete paired summaries\n")
    _report_paired_summaries(io, run_dir)

    println(io, "## Residual–solver associations\n")
    _report_correlations(io, run_dir)

    println(io, "## Small-perturbation slopes\n")
    _report_scaling(io, run_dir)

    println(io, "## Stationarity-block sensitivity by preference stratum\n")
    _report_sensitivity(io, run_dir)

    println(io, "## Failures, candidate-solution separation, anomalies, and numerical limitations\n")
    _report_failures_and_roots(io, run_dir, config)
    println(
        io,
        "Block error to one destination `ψin` is diagnostic but cannot establish usefulness because innermost multipliers may be nonunique. Raw and column-scaled Jacobian norms are reported together with finite-direction responses; no degeneracy claim is made from one unscaled norm. Equation-block residual classification is emitted only when the generated no-ordinary-inequality packing can be verified exactly. Solve times exclude explicit KKT build/warm-up and callback overhead by timing an identical untraced duplicate, but remain secondary because cache and machine state can contaminate timing.\n",
    )

    println(io, "## Figures\n")
    figure_info = (
        (
            "paired_initial_residual",
            "Paired normalized initial residuals. Thin lines join identical canonical-reference transitions; points and bars show medians and interquartile ranges.",
        ),
        (
            "paired_iterations",
            "Paired Newton iteration counts among converged mode pairs.",
        ),
        (
            "shift_quality",
            "Transported-block shift quality. The dashed line q=1 marks parity with resetting to zero.",
        ),
        (
            "small_perturbation_scaling",
            "Log–log initial-residual scaling. Squares denote a resolvable ε=0 structural baseline retained in the fit; open circles do not clear reference/numerical resolution and are excluded.",
        ),
        (
            "residual_vs_iterations",
            "Descriptive relationship between initial KKT distance and Newton iterations.",
        ),
        (
            "stationarity_sensitivity",
            "Dimension-scaled Jacobian block norms and deterministic finite-direction responses for ψout versus ψin.",
        ),
    )
    for (stem, caption) in figure_info
        println(
            io,
            "### ",
            replace(stem, "_" => " ") |> titlecase,
            "\n\n",
            "![",
            caption,
            "](figures/",
            stem,
            ".png)\n\n",
            caption,
            " [Vector PDF](figures/",
            stem,
            ".pdf).\n",
        )
    end

    println(io, "## What the experiments establish\n")
    _report_observations(io, run_dir)
    println(
        io,
        "The protocol itself establishes a controlled paired comparison: every mode in a transition is constructed from the same stored source solution and evaluated on the same stored destination problem, with deterministic randomized execution order. The raw initial residual is independently evaluated, so any reported initialization ordering is not inferred from iteration count alone.\n",
    )

    println(io, "## What the experiments do not establish\n")
    println(
        io,
        "- No empirical result proves a warm-start theorem, multiplier uniqueness, or global convergence.\n",
        "- Correlation between `log(r₀)` and solver behavior does not establish that residual reduction causes iteration reduction.\n",
        "- A result for this robotic-arm scenario, horizon, barrier, tolerance, and solver configuration need not generalize to other games.\n",
        "- Similar iteration counts do not establish equivalent initializations; separated accepted candidates or failure patterns can invalidate a simple iteration comparison, and the current replay tolerance cannot certify distinct exact roots.\n",
        "- The fresh cold/default canonical-reference initialization is independent of the four replay transport modes, but it is not basin-neutral or unbiased; conclusions remain conditional on the root basin selected by that initializer.\n",
        "- Skipped SVDs, resolution-limited scaling points, inaccurate references, or incomplete planned runs reduce the scope of any conclusion and are never extrapolated; resolved structural floors are reported as signal rather than discarded.\n",
    )

    println(io, "## Data and source index\n")
    println(
        io,
        "- Row-level data: [`raw/references.csv`](raw/references.csv), [`raw/replay.csv`](raw/replay.csv), [`raw/root_spread.csv`](raw/root_spread.csv), [`raw/sensitivity.csv`](raw/sensitivity.csv), [`raw/scaling.csv`](raw/scaling.csv).\n",
        "- Statistical outputs: [`raw/paired_summary.csv`](raw/paired_summary.csv), [`raw/correlation_summary.csv`](raw/correlation_summary.csv), [`raw/failure_summary.csv`](raw/failure_summary.csv), [`raw/scaling_slopes.csv`](raw/scaling_slopes.csv), [`raw/sensitivity_summary.csv`](raw/sensitivity_summary.csv).\n",
        "- Checkpoints: [`checkpoints/`](checkpoints/); figures: [`figures/`](figures/).\n",
        "- Experiment source: `experiments/analysis/selective_warmstart/`; production warm-start and solver instrumentation: `src/`; focused tests: `test/` and `experiments/analysis/selective_warmstart/test/`.\n",
    )
    if isfile(joinpath(run_dir, "provenance", "manifest.toml"))
        println(
            io,
            "- Exact measurement provenance: [`provenance/manifest.toml`](provenance/manifest.toml), [`provenance/drift.toml`](provenance/drift.toml), [`provenance/git_diff_binary.patch`](provenance/git_diff_binary.patch), and [`environment/`](environment/).\n",
        )
    end
    _atomic_write(report_path, String(take!(io)))
    report_path
end

function run_study(
    config::StudyConfig = preset_config(:smoke);
    run_dir = nothing,
    only = [
        :references,
        :replay,
        :sensitivity,
        :scaling,
        :analysis,
        :figures,
        :report,
    ],
    command = "",
)
    stages = Set(Symbol.(only))
    valid_stages = Set([
        :references,
        :replay,
        :sensitivity,
        :scaling,
        :analysis,
        :figures,
        :report,
    ])
    invalid = setdiff(stages, valid_stages)
    isempty(invalid) ||
        throw(ArgumentError("Unknown study stages: $(collect(invalid))"))
    run_dir = _prepare_run(config; run_dir, command)

    reference_table = IncrementalTable(
        joinpath(run_dir, "raw", "references.csv"),
        REFERENCE_COLUMNS,
    )
    replay_table = IncrementalTable(
        joinpath(run_dir, "raw", "replay.csv"),
        REPLAY_COLUMNS,
    )
    root_table = IncrementalTable(
        joinpath(run_dir, "raw", "root_spread.csv"),
        ROOT_COLUMNS,
    )
    sensitivity_table = IncrementalTable(
        joinpath(run_dir, "raw", "sensitivity.csv"),
        SENSITIVITY_COLUMNS,
    )
    scaling_table = IncrementalTable(
        joinpath(run_dir, "raw", "scaling.csv"),
        SCALING_COLUMNS,
    )
    kkt_table = IncrementalTable(
        joinpath(run_dir, "raw", "kkt_systems.csv"),
        KKT_COLUMNS,
    )

    measurement_stages =
        intersect(stages, Set([:references, :replay, :sensitivity, :scaling]))
    kkt_rows = Dict{String, Any}[]
    if !isempty(measurement_stages)
        setup = _scenario_and_problem(config)
        required_seeds = Int[]
        (:references in stages || :replay in stages) &&
            append!(required_seeds, config.scenario_seeds)
        :sensitivity in stages &&
            append!(required_seeds, config.sensitivity_seeds)
        :scaling in stages && append!(required_seeds, config.scaling_seeds)
        required_seeds = sort(unique(required_seeds))

        sequence_driver =
            :reduced in config.formulations ?
            :reduced : first(config.formulations)
        formulation_order = vcat(
            [sequence_driver],
            filter(!=(sequence_driver), config.formulations),
        )
        canonical_sequences = Dict{Int, Dict{String, Any}}()
        for formulation in formulation_order
            @info "Building KKT system" formulation profile = config.profile
            built = _build_kkt(setup.problem, formulation, config)
            kkt_row = _kkt_row(
                config,
                formulation,
                built.kkt,
                built.blocks,
                built.build_time,
            )
            push!(kkt_rows, kkt_row)
            append_row!(kkt_table, kkt_row)
            if config.warmup && !isempty(required_seeds)
                warmup_instance = _initial_instance(
                    setup.scenario,
                    first(required_seeds),
                    config,
                )
                warmup_parameters = RAC.build_instance_parameters(
                    setup.flatten_parameters,
                    warmup_instance,
                    setup.scenario,
                )
                warmup_primal = _reference_initialization(
                    warmup_instance,
                    setup.scenario,
                    1,
                ).warmstart
                warmup_full = ReducedGOOP.build_selective_warmstart(
                    warmup_primal,
                    zeros(built.kkt.variable_dimension),
                    built.kkt,
                    :primal_only,
                )
                _warmup!(
                    built.kkt,
                    warmup_parameters.θ,
                    warmup_primal,
                    warmup_full,
                    config,
                )
            end
            references_by_seed = Dict{Int, Any}()
            for scenario_seed in required_seeds
                @info "Ensuring canonical reference sequence" formulation scenario_seed
                canonical_instances =
                    formulation === sequence_driver ?
                    nothing :
                    canonical_sequences[scenario_seed]["instances"]
                references_by_seed[scenario_seed] =
                    _ensure_reference_sequence!(
                        reference_table,
                        run_dir,
                        formulation,
                        scenario_seed,
                        setup.scenario,
                        setup.problem,
                        setup.flatten_parameters,
                        setup.primal_dimensions,
                        built.kkt,
                        config,
                        ;
                        canonical_instances,
                        sequence_driver,
                    )
                if formulation === sequence_driver
                    canonical_sequences[scenario_seed] =
                        _persist_canonical_sequence!(
                            run_dir,
                            scenario_seed,
                            sequence_driver,
                            references_by_seed[scenario_seed],
                        )
                end
            end
            if :replay in stages
                for scenario_seed in config.scenario_seeds
                    @info "Running fixed-sequence replay" formulation scenario_seed
                    _run_replay_transitions!(
                        replay_table,
                        root_table,
                        run_dir,
                        formulation,
                        scenario_seed,
                        references_by_seed[scenario_seed],
                        setup.scenario,
                        setup.problem,
                        built.kkt,
                        built.blocks,
                        config,
                    )
                end
            end
            if :sensitivity in stages
                for scenario_seed in config.sensitivity_seeds
                    references = references_by_seed[scenario_seed]
                    for step in sort(unique(config.sensitivity_steps))
                        1 <= step <= length(references) || continue
                        @info "Running local sensitivity case" formulation scenario_seed step
                        _run_sensitivity_case!(
                            sensitivity_table,
                            run_dir,
                            formulation,
                            scenario_seed,
                            step,
                            references[step],
                            setup.problem,
                            built.kkt,
                            built.blocks,
                            config,
                        )
                    end
                end
            end
            if :scaling in stages
                for scenario_seed in config.scaling_seeds
                    @info "Running initial-state scaling cases" formulation scenario_seed
                    _run_scaling_cases!(
                        scaling_table,
                        run_dir,
                        formulation,
                        scenario_seed,
                        references_by_seed[scenario_seed][1],
                        setup.scenario,
                        setup.flatten_parameters,
                        built.kkt,
                        built.blocks,
                        config,
                    )
                end
            end
        end
        _write_toml(
            joinpath(run_dir, "manifest.toml"),
            _study_manifest(
                config,
                kkt_rows;
                sequence_driver,
                reference_accuracy_summary =
                    _reference_accuracy_summary(run_dir, config),
            ),
        )
    elseif !isfile(joinpath(run_dir, "manifest.toml"))
        _write_toml(
            joinpath(run_dir, "manifest.toml"),
            _study_manifest(
                config,
                kkt_rows;
                reference_accuracy_summary =
                    _reference_accuracy_summary(run_dir, config),
            ),
        )
    end

    _record_provenance_drift!(run_dir)
    :analysis in stages && analyze_study(run_dir; config)
    :figures in stages && generate_figures(run_dir)
    :report in stages && generate_report(run_dir; config)
    run_dir
end

end
