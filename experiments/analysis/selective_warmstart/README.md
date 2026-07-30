# Selective primal–dual warm-start study

This directory contains the reproducible paired experiment used to determine why
selective multiplier transport changes receding-horizon solver behavior in
reduced and quasi GOOP. It uses the plotting-free robotic-arm core and the
separate `experiments/` Julia environment.

The central control is fixed-sequence canonical replay: all four modes for a
transition use the same stored source solution and solve the same stored
destination problem. A mode therefore cannot alter later MPC parameters.

Canonical-reference construction is also mode-neutral relative to those four
transport modes. Every numerical reference step makes one fresh,
destination-specific `cold_default` solve; it never transports the previous
reference primal or dual and does not run a continuation tournament or rescue.
The accepted reduced cold-reference primal drives the next MPC instance, the
same instance digest is then used by both formulations, and previous reference
solutions remain stored solely as replay sources. “Mode-neutral” does not mean
basin-neutral or unbiased: the deterministic cold initializer can still select
a numerical root basin.

## Mathematical mapping

The production helper `ReducedGOOP.kkt_variable_blocks(kkt)` maps generated KKT
coordinates as follows:

| Mathematical block | Generated-KKT coordinates |
| --- | --- |
| `z` | `kkt.primal_dims` |
| `λ` | `kkt.equality_constraint_dual_dims` |
| `ψout` | stationarity-dual coordinates not attached to an innermost stationarity equation |
| `ψin` | `kkt.innermost_stationarity_dual_dims` |

`ψin` denotes the multipliers attached to innermost stationarity equations, not
the equations. The repository’s pre-existing spelling `:primal_only` is kept.
All modes begin from the same fresh solver-default full vector, so preference
slacks, interior-point slacks, and ordinary inequality duals are identical.

Only primals are shifted in time. Duals retain their generated coordinate
identity because no independently verified time-index permutation exists for
every hierarchical multiplier block. Terminal primal completion drops the
executed knot, retains the remaining controls, advances the final retained state
with the final retained control, and appends the unused zero terminal control.

## Profiles

| Profile | Scenario seeds | MPC steps | Horizon | Sensitivity | Scaling |
| --- | ---: | ---: | ---: | --- | --- |
| `smoke` | 1 | 3 | 20 | 1 reference step | 1 direction |
| `pilot` | 2 | 8 | 20 | 2 steps/seed | 2 directions |
| `full` | 10 | 20 | 20 | 3 steps on 2 seeds | 3 directions on 3 seeds |

Every profile runs both formulations and four modes. The full replay contains
`10 × 19 × 4 × 2 = 1,520` mode runs, in addition to references,
sensitivity, scaling, and identical uninstrumented timing duplicates.

## Run and resume

From the repository root:

```bash
julia --project=experiments experiments/analysis/selective_warmstart/run.jl --profile smoke
julia --project=experiments experiments/analysis/selective_warmstart/run.jl --profile pilot
julia --project=experiments experiments/analysis/selective_warmstart/run.jl --profile full
```

The command prints the timestamped run directory under
`data/selective_warmstart/<profile>/`. That top-level `data/` directory is
already ignored by Git.

Resume an interrupted run:

```bash
julia --project=experiments experiments/analysis/selective_warmstart/run.jl \
  --resume /absolute/path/to/run
```

Regenerate only derived products:

```bash
julia --project=experiments experiments/analysis/selective_warmstart/run.jl \
  --resume /absolute/path/to/run \
  --stages analysis,figures,report
```

`--config PATH` loads an exact TOML previously written as `config.toml`.
Run `.../run.jl --help` for all CLI options.

Each numerical case first writes an atomic JLD2 checkpoint and then appends one
physical CSV row. Newlines in exception text are escaped, so the resume parser
cannot confuse one record with several lines. Existing checkpoint rows are
restored into a missing/incomplete CSV without repeating the solve. If a process
is killed during append, one malformed final physical row is removed before
checkpoint restoration; a malformed nonfinal row remains a hard error rather
than being silently discarded.

## Accuracy, convergence, and timing

All profiles and stages use the serialized
`uniform_t20_dt0p1_tol0p008_max1000_v1` protocol:

- planning horizon `T = 20` and `Δt = 0.1`;
- solver and direct-acceptance tolerance `0.008`;
- `max_inner_iters = 1000`;
- one unchanged `InteriorPointOptions` value for canonical references, replay,
  perturbed-scaling references, scaling mode solves, and compilation warmups.

Sensitivity invokes no solver; it evaluates the same `T=20`, `Δt=0.1` KKT
system at accepted canonical points. The compilation-only warmup uses that same
option set, is excluded from measurements, and runs once per formulation. The
0.008 tolerance and every solver option except the iteration cap match the
headless robotic-arm receding baseline in `Robotic_arm_mpc.jl`. The cap is
uniformly 1,000 rather than that baseline's 500: a direct reduced T20 seed101
cold solve required 717 iterations and met the 0.008 criterion at residual
`0.007991129782359604`. The visualization-oriented
`Robotic_arm_receding.jl` differs only by setting `record_convergence=true`.
`T=20` is the explicit comparability override—the scenario/receding default is
`T=30`—while `Δt=0.1` matches that baseline. KLU, backtracking, `η₀=1e-6`,
`ηmax=1e2`, Armijo `1e-4`, factorization reuse `0`, and the remaining study
options are held fixed.

Canonical and perturbed-scaling references each use one fresh, destination-
specific `cold_default` attempt. There is no fallback, continuation tournament,
or rescue. This cold-reference rule was selected after the disclosed 0.01
all-dual-continuation pilot failures, but it is applied uniformly rather than
case by case.

Accepted canonical points generate the trajectory and provide shared comparison
endpoints. They are not higher-accuracy ground truth: they use the same 0.008
solver/direct-residual threshold as replay and scaling. Block errors,
candidate-root distances, sensitivity results, and scaling results are therefore
conditioned on that canonical-point accuracy.

Both `comparability_protocol` and `reference_initialization` are serialized in
`config.toml` and its fingerprint. Configurations that predate or violate either
guard are rejected, so older runs cannot resume into a mixed protocol.

For replay and scaling, `direct_converged` is authoritative for converged-only
analysis:

```text
isfinite(||K(yfinal)||₂) && ||K(yfinal)||₂ <= replay_tol
```

The raw solver status remains in every row, and the failure summary counts
raw/direct disagreements.

Trace callbacks intentionally evaluate extra norms. Consequently each measured
case runs twice from independently copied but identical initial vectors and
options:

1. an instrumented solve supplies first-step, line-search, regularization, and
   iteration metrics;
2. an uninstrumented duplicate supplies `solve_time_sec`.

`instrumented_solve_time_sec` and both raw statuses are also retained. The same
named trace-collector type is warmed before measurement. Runtime remains
secondary because cache and operating-system state can still matter.

## Outputs

Each run contains:

- `config.toml`, `environment.toml`, `manifest.toml`, and
  `analysis_manifest.toml`;
- exact creation-time `environment/Project.toml` and
  `environment/Manifest.toml` snapshots;
- `provenance/manifest.toml`, a binary-capable dirty Git patch, and copies/hashes
  of the relevant measurement source (including untracked files);
- `provenance/drift.toml`, which distinguishes the creation-time measurement
  fingerprint from code/environment used for later analysis regeneration;
- row-level CSVs in `raw/`;
- case-level JLD2 checkpoints in `checkpoints/`;
- deterministic paired-bootstrap, failure, correlation, and scaling-slope CSVs;
- six publication-oriented figures in both PNG and PDF under `figures/`;
- `report.md`, including complete paired summaries and limitations.

Scaling postprocessing separates reference/numerical resolution from the
mode-specific ε=0 structural residual. A structural baseline that clears
reference resolution is retained as scientific signal—stable nonzero floors can
therefore produce the hypothesized near-zero slope. Fits use at most the three
smallest resolved perturbations; fewer than two resolved points produces a
disclosed skipped fit. This is a post-smoke diagnostic-definition correction:
the legacy raw `numerical_floor` column combined structural baseline and
resolution, while `raw/scaling_slopes.csv` reports them separately. Full
near-null SVD diagnostics are likewise skipped above the configured
variable-dimension cap.

## Focused test

The study-local deterministic metric test is:

```bash
julia --project=experiments experiments/analysis/selective_warmstart/test/runtests.jl
```

Core package tests cover mode block retention, non-aliasing, terminal shifting,
random-order invariance, and reduced/quasi generated-KKT block metadata.
