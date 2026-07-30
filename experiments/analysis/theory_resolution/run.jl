#!/usr/bin/env julia

include(joinpath(@__DIR__, "TheoryResolutionStudy.jl"))
using .TheoryResolutionStudy

const ALL_STAGES = (
    :inputs,
    :stagewise,
    :holdout,
    :globalization,
    :scaling,
    :analysis,
    :figures,
    :report,
)

function usage(io = stdout)
    println(
        io,
        """
        Usage:
          julia --project=experiments experiments/analysis/theory_resolution/run.jl [options]

        Options:
          --source-run PATH   Completed 2026-07-30_103743_dual_transport run.
          --output-root PATH  Root for a new run (default: data/theory_resolution).
          --resume RUN_DIR    Resume an existing hash-validated run.
          --stages LIST       Ordered comma-separated subset of:
                              inputs,stagewise,holdout,globalization,
                              scaling,analysis,figures,report
          -h, --help          Show this help.

        Only the 11 preregistered held-out gamma=0.25 cases and the 15
        preregistered rich-trace hard cases invoke the production optimizer.
        The stagewise and controlled-scaling stages invoke no optimizer.
        """,
    )
end

function parse_arguments(arguments)
    source_run = nothing
    output_root = nothing
    resume = nothing
    stages = collect(ALL_STAGES)
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("-h", "--help")
            usage()
            exit(0)
        elseif argument == "--source-run"
            index < length(arguments) || error("--source-run requires a path.")
            source_run = abspath(arguments[index+1])
            index += 2
        elseif argument == "--output-root"
            index < length(arguments) || error("--output-root requires a path.")
            output_root = abspath(arguments[index+1])
            index += 2
        elseif argument == "--resume"
            index < length(arguments) || error("--resume requires a path.")
            resume = abspath(arguments[index+1])
            index += 2
        elseif argument == "--stages"
            index < length(arguments) ||
                error("--stages requires a comma-separated list.")
            stages = Symbol.(
                filter(!isempty, strip.(split(arguments[index+1], ","))),
            )
            index += 2
        else
            error("Unknown argument $(argument). Use --help.")
        end
    end
    unknown = setdiff(Set(stages), Set(ALL_STAGES))
    isempty(unknown) || error("Unknown stages: $(sort!(collect(unknown))).")
    (; source_run, output_root, resume, stages)
end

function _load_stagewise_module()
    isdefined(Main, :StagewiseDiagnostics) ||
        Base.include(Main, joinpath(@__DIR__, "StagewiseDiagnostics.jl"))
    Base.invokelatest(getfield, Main, :StagewiseDiagnostics)
end

function _load_scaling_module()
    isdefined(Main, :ScalingBenchmark) ||
        Base.include(Main, joinpath(@__DIR__, "ScalingBenchmark.jl"))
    Base.invokelatest(getfield, Main, :ScalingBenchmark)
end

function _load_figures_module()
    isdefined(Main, :TheoryResolutionFigures) ||
        Base.include(Main, joinpath(@__DIR__, "Figures.jl"))
    Base.invokelatest(getfield, Main, :TheoryResolutionFigures)
end

function main(arguments = ARGS)
    parsed = parse_arguments(arguments)
    config = if !isnothing(parsed.resume)
        stored = load_config(joinpath(parsed.resume, "config.toml"))
        if !isnothing(parsed.source_run) &&
           abspath(stored.source_run) != parsed.source_run
            error("--source-run differs from the stored resume configuration.")
        end
        if !isnothing(parsed.output_root) &&
           abspath(stored.output_root) != parsed.output_root
            error("--output-root differs from the stored resume configuration.")
        end
        stored
    else
        source = isnothing(parsed.source_run) ?
                 TheoryResolutionStudy.DEFAULT_SOURCE_RUN :
                 parsed.source_run
        output = isnothing(parsed.output_root) ?
                 joinpath(
            TheoryResolutionStudy.REPOSITORY_ROOT,
            "data",
            "theory_resolution",
        ) :
                 parsed.output_root
        TheoryResolutionConfig(;
            source_run = source,
            output_root = output,
        )
    end
    command = join(
        vcat(
            [string(Base.julia_cmd())],
            [abspath(@__FILE__)],
            arguments,
        ),
        " ",
    )
    run_dir = prepare_run(
        config;
        run_dir = parsed.resume,
        command,
    )
    cache = Dict{Symbol, Any}()
    for stage in parsed.stages
        println("[theory-resolution] stage: ", stage)
        flush(stdout)
        if stage === :inputs
            snapshot_inputs!(run_dir, config)
        elseif stage === :stagewise
            module_ = _load_stagewise_module()
            runner = Base.invokelatest(
                getfield,
                module_,
                :run_stagewise_diagnostics,
            )
            Base.invokelatest(
                runner,
                run_dir,
                config.source_run,
            )
        elseif stage === :holdout
            run_holdout_validation!(run_dir, config, cache)
        elseif stage === :globalization
            run_globalization_study!(run_dir, config, cache)
        elseif stage === :scaling
            module_ = _load_scaling_module()
            runner = Base.invokelatest(
                getfield,
                module_,
                :run_scaling_benchmark,
            )
            Base.invokelatest(runner, run_dir)
        elseif stage === :analysis
            analyze_study!(run_dir, config)
        elseif stage === :figures
            module_ = _load_figures_module()
            runner = Base.invokelatest(getfield, module_, :generate_figures)
            Base.invokelatest(runner, run_dir)
        elseif stage === :report
            generate_report(run_dir, config)
        end
    end
    println("Theory-resolution study output: ", run_dir)
    println("Report: ", joinpath(run_dir, "report.md"))
    run_dir
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
