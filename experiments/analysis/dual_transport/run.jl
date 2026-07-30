#!/usr/bin/env julia

include(joinpath(@__DIR__, "DualTransportStudy.jl"))
using .DualTransportStudy

function usage(io = stdout)
    println(
        io,
        """
        Usage:
          julia --project=experiments experiments/analysis/dual_transport/run.jl [options]

        Options:
          --baseline PATH       Completed 2026-07-30_002646_pilot directory.
          --output-root PATH    Root for a new run (default: data/dual_transport).
          --resume RUN_DIR      Resume an existing run with hash-validated inputs.
          --stages LIST         Comma-separated ordered subset of:
                                inputs,metadata,diagnostics,replay,damping,
                                projection,analysis,figures,report
          --no-projection-solve Compute row/null energy without diagnostic solves.
          -h, --help            Show this help.

        The default executes all stages in scientific order. Solver stages are
        gated on successful T=4 metadata/mapping tests and direct residual-action
        diagnostics. No stage generates or rescues a canonical reference.
        """,
    )
end

function parse_arguments(arguments)
    baseline = nothing
    output_root = nothing
    resume = nothing
    projection_solve = true
    stages = [
        :inputs,
        :metadata,
        :diagnostics,
        :replay,
        :damping,
        :projection,
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
        elseif argument == "--baseline"
            index < length(arguments) || error("--baseline requires a path.")
            baseline = abspath(arguments[index+1])
            index += 2
        elseif argument == "--output-root"
            index < length(arguments) || error("--output-root requires a path.")
            output_root = abspath(arguments[index+1])
            index += 2
        elseif argument == "--resume"
            index < length(arguments) || error("--resume requires a run directory.")
            resume = abspath(arguments[index+1])
            index += 2
        elseif argument == "--stages"
            index < length(arguments) || error("--stages requires a comma-separated list.")
            stages = Symbol.(filter(!isempty, strip.(split(arguments[index+1], ","))))
            index += 2
        elseif argument == "--no-projection-solve"
            projection_solve = false
            index += 1
        else
            error("Unknown argument $(argument). Use --help.")
        end
    end
    allowed = Set((
        :inputs,
        :metadata,
        :diagnostics,
        :replay,
        :damping,
        :projection,
        :analysis,
        :figures,
        :report,
    ))
    unknown = setdiff(Set(stages), allowed)
    isempty(unknown) || error("Unknown stages: $(collect(unknown)).")
    (; baseline, output_root, resume, projection_solve, stages)
end

function main(arguments = ARGS)
    parsed = parse_arguments(arguments)
    config = if !isnothing(parsed.resume)
        stored = load_config(joinpath(parsed.resume, "config.toml"))
        if !isnothing(parsed.baseline) && abspath(stored.baseline_dir) != parsed.baseline
            error("--baseline differs from the stored resume baseline.")
        end
        stored.projection_solve == parsed.projection_solve ||
            parsed.projection_solve == true ||
            error("--no-projection-solve cannot change a stored resume configuration.")
        stored
    else
        base = isnothing(parsed.baseline) ?
            DualTransportStudy.DEFAULT_BASELINE : parsed.baseline
        output = isnothing(parsed.output_root) ?
            joinpath(DualTransportStudy.REPOSITORY_ROOT, "data", "dual_transport") :
            parsed.output_root
        TransportStudyConfig(;
            baseline_dir = base,
            output_root = output,
            projection_solve = parsed.projection_solve,
        )
    end
    command = join(vcat([string(Base.julia_cmd())], PROGRAM_FILE, arguments), " ")
    run_dir = DualTransportStudy.prepare_run(
        config;
        run_dir = parsed.resume,
        command,
    )
    cache = Dict{Symbol, Any}()
    for stage in parsed.stages
        println("[dual-transport] stage: ", stage)
        if stage === :inputs
            DualTransportStudy.snapshot_inputs!(run_dir, config)
        elseif stage === :metadata
            DualTransportStudy.run_metadata!(run_dir, config, cache)
        elseif stage === :diagnostics
            DualTransportStudy.run_diagnostics!(run_dir, config, cache)
        elseif stage === :replay
            DualTransportStudy.run_replay!(run_dir, config, cache)
        elseif stage === :damping
            DualTransportStudy.run_damping!(run_dir, config, cache)
        elseif stage === :projection
            DualTransportStudy.run_projection!(run_dir, config, cache)
        elseif stage === :analysis
            analyze_study(run_dir, config)
        elseif stage === :figures
            figures_module = if isdefined(Main, :DualTransportFigures)
                Base.invokelatest(getfield, Main, :DualTransportFigures)
            else
                Base.include(Main, joinpath(@__DIR__, "Figures.jl"))
            end
            generate = Base.invokelatest(
                getfield,
                figures_module,
                :generate_figures,
            )
            Base.invokelatest(
                generate,
                run_dir,
            )
        elseif stage === :report
            generate_report(run_dir, config)
        end
    end
    println("Dual-transport study output: ", run_dir)
    println("Report: ", joinpath(run_dir, "report.md"))
    run_dir
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
