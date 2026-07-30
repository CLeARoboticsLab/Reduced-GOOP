#!/usr/bin/env julia

include(joinpath(@__DIR__, "TheoryResolutionStudy.jl"))
using .TheoryResolutionStudy

function parse_case_arguments(arguments)
    values = Dict{String, String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("Unexpected positional argument $(arguments[index]).")
        index < length(arguments) ||
            error("Missing value for $(arguments[index]).")
        values[arguments[index]] = arguments[index+1]
        index += 2
    end
    required = (
        "--run-dir",
        "--form",
        "--seed",
        "--transition",
        "--gamma",
    )
    all(haskey(values, name) for name in required) ||
        error("Required isolated-case arguments are $(join(required, ", ")).")
    (
        run_dir = abspath(values["--run-dir"]),
        form = Symbol(values["--form"]),
        seed = parse(Int, values["--seed"]),
        transition = parse(Int, values["--transition"]),
        gamma = parse(Float64, values["--gamma"]),
    )
end

function main(arguments = ARGS)
    parsed = parse_case_arguments(arguments)
    TheoryResolutionStudy._run_single_globalization_case!(
        parsed.run_dir,
        parsed.form,
        parsed.seed,
        parsed.transition,
        parsed.gamma,
    )
    nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
