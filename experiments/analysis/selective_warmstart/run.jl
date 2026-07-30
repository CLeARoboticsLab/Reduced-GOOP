#!/usr/bin/env julia

include(joinpath(@__DIR__, "SelectiveWarmstartStudy.jl"))
using .SelectiveWarmstartStudy

function usage(io = stdout)
    println(
        io,
        """
        Usage:
          julia --project=experiments experiments/analysis/selective_warmstart/run.jl [options]

        Options:
          --profile smoke|pilot|full   Built-in run size (default: smoke)
          --config PATH                Load an exact StudyConfig TOML
          --resume RUN_DIR             Resume an existing run using its stored config
          --stages LIST                Comma-separated subset of:
                                       references,replay,sensitivity,scaling,
                                       analysis,figures,report
          -h, --help                   Show this help

        A new run writes its exact configuration to RUN_DIR/config.toml. Resume
        is case-level and reads atomic JLD2 checkpoints before appending CSV rows.
        """,
    )
end

function parse_arguments(arguments)
    profile = :smoke
    profile_given = false
    config_path = nothing
    resume_dir = nothing
    stages = [
        :references,
        :replay,
        :sensitivity,
        :scaling,
        :analysis,
        :figures,
        :report,
    ]
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("-h", "--help")
            usage()
            exit(0)
        elseif argument == "--profile"
            index < length(arguments) ||
                error("--profile requires a value")
            profile = Symbol(arguments[index+1])
            profile_given = true
            index += 2
        elseif argument == "--config"
            index < length(arguments) ||
                error("--config requires a path")
            config_path = arguments[index+1]
            index += 2
        elseif argument == "--resume"
            index < length(arguments) ||
                error("--resume requires a run directory")
            resume_dir = arguments[index+1]
            index += 2
        elseif argument == "--stages"
            index < length(arguments) ||
                error("--stages requires a comma-separated list")
            stages = Symbol.(
                filter(
                    value -> !isempty(value),
                    strip.(split(arguments[index+1], ",")),
                ),
            )
            index += 2
        else
            error("Unknown argument $(argument). Use --help.")
        end
    end
    (; profile, profile_given, config_path, resume_dir, stages)
end

function main(arguments = ARGS)
    parsed = parse_arguments(arguments)
    config = if !isnothing(parsed.config_path)
        load_config(abspath(parsed.config_path))
    elseif !isnothing(parsed.resume_dir) && !parsed.profile_given
        load_config(joinpath(abspath(parsed.resume_dir), "config.toml"))
    else
        preset_config(parsed.profile)
    end
    command = join(vcat([string(Base.julia_cmd())], PROGRAM_FILE, arguments), " ")
    run_dir = run_study(
        config;
        run_dir =
            isnothing(parsed.resume_dir) ? nothing :
            abspath(parsed.resume_dir),
        only = parsed.stages,
        command,
    )
    println("Selective warm-start study output: ", run_dir)
    println("Report: ", joinpath(run_dir, "report.md"))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
