#!/usr/bin/env julia

include(joinpath(@__DIR__, "StagewiseDiagnostics.jl"))
using .StagewiseDiagnostics

function usage()
    println(
        stderr,
        "Usage: julia --project=experiments " *
        "experiments/analysis/theory_resolution/validate_stagewise.jl " *
        "SOURCE_RUN [STAGEWISE_RUN]",
    )
end

if !(length(ARGS) in (1, 2))
    usage()
    exit(2)
end

source_run = abspath(ARGS[1])
summary = if length(ARGS) == 2
    validate_stagewise_diagnostics(source_run; run_dir = abspath(ARGS[2]))
else
    validate_stagewise_diagnostics(source_run)
end

for name in propertynames(summary)
    println(name, " = ", getproperty(summary, name))
end
