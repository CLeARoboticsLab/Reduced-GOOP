# Focused semantic-transport theory resolution

This study follows the completed run
`2026-07-30_103743_dual_transport`. It uses its 17 valid frozen transition
pairs and does **not** run the 10-seed population experiment.

The production-solver protocol is frozen at:

```text
T = 20
Δt = 0.1
tol = 0.008
max_inner_iters = 1000
max_outer_iters = 1
linear solver = KLU
line search = backtracking
```

The runner verifies every remaining solver option against the completed
run's `solver_options.toml`. There is no looser tolerance, rescue solve,
reference tournament, or experimental alternate solver.

## Clean reproduction

```bash
julia --project=experiments \
  experiments/analysis/theory_resolution/run.jl \
  --source-run data/dual_transport/pilot/2026-07-30_103743_dual_transport
```

The runner records the fully expanded invocation in
`reproduction_command.txt` inside the output directory.

## Exact checkpoint resume

```bash
julia --project=experiments \
  experiments/analysis/theory_resolution/run.jl \
  --resume data/theory_resolution/pilot/RUN_NAME
```

The resume command hash-validates the frozen inputs and solver options and
reuses completed JLD2 case checkpoints. To regenerate only aggregation,
figures, and the report after a code-only reporting change:

```bash
julia --project=experiments \
  experiments/analysis/theory_resolution/run.jl \
  --resume data/theory_resolution/pilot/RUN_NAME \
  --stages holdout,scaling,analysis,figures,report
```

The `holdout` stage above does not re-solve a completed held-out case: it
validates and re-aggregates the existing checkpoint. The empirical safeguard
rule defines “nearly all” as at least 10/11 held-out successes together with
6/6 development successes. Meeting that focused threshold retains `γ=0.25`
as the preregistered population-study baseline; it does not override a
catastrophic holdout failure, establish a failure-free fixed safeguard, or
remove the need to develop adaptive damping from the hard-case telemetry.

Stages can be resumed individually with `--stages`, in this order:

```text
inputs,stagewise,holdout,globalization,scaling,analysis,figures,report
```

Only 11 new held-out `γ=0.25` solves and 15 preregistered rich-trace hard
cases invoke the optimizer. Stagewise localization and the controlled
scaling benchmark invoke no optimizer.

## Focused validation

```bash
julia --project=. -e 'using Test; include("test/kkt_metadata.jl")'
julia --project=. -e 'using Test; include("test/solver_trace.jl")'

julia --project=experiments \
  experiments/analysis/theory_resolution/test_scaling_benchmark.jl

julia --project=experiments \
  experiments/analysis/theory_resolution/validate_stagewise.jl \
  data/dual_transport/pilot/2026-07-30_103743_dual_transport
```

Every numerical case is checkpointed in JLD2. Raw CSVs, PNG/PDF figures,
the detailed report, the exact runner command, a source snapshot, and SHA-256
artifact provenance are stored in the run directory.
