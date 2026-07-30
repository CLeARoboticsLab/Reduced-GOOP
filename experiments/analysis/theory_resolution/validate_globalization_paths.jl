#!/usr/bin/env julia

include(joinpath(@__DIR__, "IteratePathValidation.jl"))
using .GlobalizationIteratePathValidation

function usage(io = stdout)
    println(
        io,
        """
        Usage:
          julia --project=experiments \\
            experiments/analysis/theory_resolution/validate_globalization_paths.jl \\
            --run-dir RUN_DIR [--preflight]

        Options:
          --run-dir PATH  Completed theory-resolution run containing all 15
                          rich-v3 globalization checkpoints.
          --preflight     Validate inputs and checkpoint hashes without replaying
                          any solver case.
          -h, --help      Show this help.

        The default action runs two same-process diagnostic validation replays
        for each of the 15 preregistered hard cases with frozen production
        options and warm starts: one lightweight hash replay and one full
        rich/SPQR replay. Successful case pairs are resumable. These are not
        candidate, rescue, or tuning solves. Exact iterate hashes, scalar
        events, and final full-z equality are required only between those two
        same-process replays. Comparisons with the saved rich checkpoint are
        cross-process diagnostics: convergence class must match, and primal
        alignment is required only when both saved and new runs converged.
        """,
    )
end

function parse_arguments(arguments)
    run_dir = nothing
    preflight = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("-h", "--help")
            usage()
            exit(0)
        elseif argument == "--run-dir"
            index < length(arguments) || error("--run-dir requires a path.")
            run_dir = abspath(arguments[index+1])
            index += 2
        elseif argument == "--preflight"
            preflight = true
            index += 1
        else
            error("Unknown argument $(argument). Use --help.")
        end
    end
    isnothing(run_dir) && error("--run-dir is required.")
    (; run_dir, preflight)
end

function main(arguments = ARGS)
    parsed = parse_arguments(arguments)
    if parsed.preflight
        cases = validate_iterate_path_inputs(parsed.run_dir)
        println(
            "[iterate-path-validation] validated $(length(cases)) rich inputs; " *
            "no solver replays were run.",
        )
    else
        rows = run_iterate_path_validation!(parsed.run_dir)
        println(
            "[iterate-path-validation] $(length(rows))/15 complete; " *
            "same-process accepted hashes, scalar events, and final full-z " *
            "values are exact; saved/new convergence classes match, with " *
            "primal alignment enforced only for jointly converged cases.",
        )
    end
    nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
