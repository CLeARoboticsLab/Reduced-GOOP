# Semantic dual-transport study

This focused study reuses the accepted source/destination checkpoints from
`2026-07-30_002646_pilot`. It does not solve, rescue, or replace canonical
references.

The measured protocol is frozen at `T=20`, `Δt=0.1`, `tol=0.008`, 1000 inner
iterations, one outer iteration, KLU, and backtracking. The only experimental
factor is the warm-start coordinate construction.

Run everything:

```bash
julia --project=experiments experiments/analysis/dual_transport/run.jl \
  --baseline data/selective_warmstart/pilot/2026-07-30_002646_pilot
```

Resume atomically from case checkpoints:

```bash
julia --project=experiments experiments/analysis/dual_transport/run.jl \
  --resume data/dual_transport/pilot/<run-id>
```

Run or resume selected stages:

```bash
julia --project=experiments experiments/analysis/dual_transport/run.jl \
  --resume data/dual_transport/pilot/<run-id> \
  --stages replay,damping,projection,analysis,figures,report
```

Stages are `inputs`, `metadata`, `diagnostics`, `replay`, `damping`,
`projection`, `analysis`, `figures`, and `report`. Solver stages refuse to run
until production metadata, the T=4 sentinel map, and direct residual-action
diagnostics have completed successfully.

Important outputs:

- `baseline_audit.toml`: independent recomputation of the baseline signature;
- `t4_mapping.md` and `raw/t4_mapping.csv`: semantic source/destination map;
- `raw/residual_diagnostics.csv` and `raw/residual_action.csv`;
- `raw/replay.csv`, `raw/damping.csv`, and `raw/projection*.csv`;
- `raw/paired_statistics.csv`;
- `checkpoints/`: atomic per-pair/per-policy JLD2 records;
- `inputs/manifest.toml`: hashes of frozen baseline inputs;
- `solver_options.toml`: the complete shared solver-option record;
- `provenance/manifest.toml` and `provenance/finalization.toml`: measurement
  source snapshot, final source state, and numerical-output hashes;
- `figures/`: PNG and PDF figures;
- `report.md`: findings, qualifications, and decision table.
