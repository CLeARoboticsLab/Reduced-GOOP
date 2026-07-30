# ReducedGOOP.jl

[![CI](https://github.com/CLeARoboticsLab/QuasiGOOP.jl/actions/workflows/test.yml/badge.svg)](https://github.com/CLeARoboticsLab/QuasiGOOP.jl/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD-new)](https://opensource.org/license/bsd-3-clause)

This repository contains the implementation accompanying the paper:

> [Breaking Exponential Complexity in Games of Ordered Preference: A Tractable Reformulation](https://arxiv.org/abs/2603.26950)

Games of Ordered Preference (GOOP) model strategic interactions in which each
player optimizes a hierarchy of objectives and constraints, rather than a single
scalar cost. This is useful for settings such as robotics and multi-agent
planning, where a player first satisfies safety or task constraints and only
then optimize lower-priority behavior.

`ReducedGOOP.jl` implements tractable nonlinear KKT reformulations of GOOP
problems together with a primal-dual interior-point solver. The experiment
scripts reproduce the main computational examples from the paper, including the
two-player intersection scenario and quadratic hierarchy benchmarks.

## Installation and Usage

This repository is a Julia package. The experiment drivers use the Julia
environment in `experiments/`, which depends on the local package at the
repository root.

From the repository root:

```bash
julia --project=experiments
```

Inside Julia, instantiate the experiment environment once:

```julia
import Pkg
Pkg.instantiate()
```

### Two-Player Intersection Scenario

The following reproduces the deterministic two-player intersection example used
in the paper:

```julia
using Revise
includet("experiments/Intersection.jl")
Intersection.demo(random_initial_state = false)
```

If `Revise.jl` is not installed in your local Julia setup, use `include` instead:

```julia
include("experiments/Intersection.jl")
Intersection.demo(random_initial_state = false)
```

The active script constructs a two-player open-loop trajectory game, generates a
GOOP KKT reformulation, solves it with the interior-point solver, and saves
trajectory and solution data under `data/Intersection_open_loop/`.

### Single-Player Trilevel Quadratic Program

The following runs the quadratic-program example driver:

```julia
include("experiments/ExamplesQP.jl")
```

`ExamplesQP.jl` loads `experiments/trilevel_QP.jl`, which solves a bounded
single-player quadratic hierarchy with the interior-point method and checks its
dual solution independently with `NonlinearSolve`.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/` | Core GOOP problem representation, KKT reformulation generators, and interior-point solver. |
| `experiments/` | Reproduction scripts, plotting utilities, and the experiment-specific Julia environment. |
| `test/` | Regression tests for KKT formulations, code generation, KLU solves, and warm starts. |
| `legacy/` | Older implementations, archived formulations, and historical experiments retained for reference. |
| `data/` | Generated experiment outputs and archived result artifacts. |

## Core Implementation

### `src/goop.jl`

`goop.jl` defines the main GOOP modeling interface and reformulation generators.

| Symbol | Role |
| --- | --- |
| `ParametricGOOP` | Stores player preferences, prioritized-constraint flags, player-wise equality and inequality constraints, optional shared constraints, dimensions, and number of players. |
| `ParametricGOOP(x, theta; ...)` | Convenience constructor that infers primal, parameter, equality, and inequality dimensions from template block vectors. |
| `QuasiLagrangianTerm` and helpers | Internal machinery for the quasi formulation; it builds gradients while dropping higher-order derivative terms after a bounded order. |

#### KKT-Based Formulations

These functions construct nonlinear KKT systems represented as `GOOPKKTSystem`
objects and solved by the interior-point method in `solver.jl`:

| Function | Description |
| --- | --- |
| `generate_slacked_reduced_kkt_system(...)` | Builds a reduced, slacked KKT system recursively. It introduces preference slacks, interior-point slacks, equality duals, inequality duals, lower-level policy multipliers, and complementarity-relaxation terms. |
| `generate_slacked_quasi_kkt_system(...)` | Calls the reduced generator with `drop_higher_order_terms = true`. The code implements this by propagating `QuasiLagrangianTerm` objects and truncating higher-order derivative contributions. |
| `generate_slacked_complete_kkt_system(...)` | Builds a more explicit nested KKT system by carrying inner-level stationarity, inequality rows, and decision variables into outer-level KKT conditions. |

From the source, the reduced formulation appears to encode lower-level stationarity
and complementarity information more compactly through recursively propagated
policy conditions and multipliers. The complete formulation keeps a more explicit
representation of inner KKT variables and constraints. The quasi formulation is
a reduced formulation with selected higher-order derivative terms omitted.

### `src/goop_kkt_system.jl`

`goop_kkt_system.jl` defines `GOOPKKTSystem`, the container used by the
interior-point solver. It stores:

- in-place residual and Jacobian evaluators for the KKT residual and its
  derivative with respect to the decision vector;
- index sets for primal variables, preference slacks, interior-point slacks,
  inequality duals, and the equality/stationarity dual subsets used by selective
  warm starts;
- KKT and variable dimensions;
- the symbolic residual and symbolic variable vector used to build the system.

`BuildGOOPKKTSystem(...)` selects either the Symbolics or FastDifferentiation
backend, builds an in-place residual function, constructs a sparse Jacobian with
respect to the full decision vector, and records constant sparse entries for
efficient repeated evaluation.

### `src/solver.jl`

`solver.jl` provides the `InteriorPoint` solver front end:

| Solver | Description |
| --- | --- |
| `InteriorPoint` | Solves a `GOOPKKTSystem` by Newton steps on the relaxed primal-dual residual. It initializes preference slacks, interior-point slacks, and inequality duals to positive values, supports backtracking and fraction-to-boundary line search, and can record KKT-error histories. |

Solver options are configured through `InteriorPointOptions`.

## Experiments

| File | Description |
| --- | --- |
| `experiments/Intersection.jl` | Two-player open-loop intersection example with trajectory dynamics, prioritized preferences, an interior-point solve, and result plotting. |
| `experiments/ExamplesQP.jl` | Lightweight entry point for the quadratic-program example. |
| `experiments/Robotic_arm_receding.jl` | Receding-horizon robotic-arm demo with primal-only and selective dual warm-start strategies. |
| `experiments/Plotting.jl` | Plotting utilities used by the intersection experiments. |

## Tests

The current tests in `test/runtests.jl` exercise the complete and reduced
slacked KKT formulations with the interior-point solver.

| Benchmark family | What is tested |
| --- | --- |
| Known-solution benchmarks | Single-player and coupled three-player cases with quadratic/linear and nonlinear/nonlinear variants. |
| Complete KKT smoke test | Agreement between complete and reduced formulations on an unconstrained quadratic problem. |
| Code-generation parity | Agreement between Symbolics and FastDifferentiation residual/Jacobian evaluators. |
| KLU solver tests | Augmented-system direction accuracy, factorization reuse, singular-retry behavior, and agreement with dense SVD. |
| Warm-start tests | Full-vector solver warm starts and the robotic-arm selective-dual warm-start policies. |

The tests verify convergence status, residual tolerances, known primal
solutions, active/inactive constraint behavior, and linear-solver robustness.

Run the tests from the repository root with:

```bash
julia --project=. -e 'import Pkg; Pkg.test()'
```

## Legacy Code

The `legacy/` directory contains older implementations, experimental code,
archived formulations, and historical experiments. These files are not part of
the active code path, but they may be useful for understanding earlier modeling
choices or for future development.

## Developer Notes

The high-level workflow is:

1. Define a `ParametricGOOP` problem from player preferences and constraints.
2. Generate a complete, reduced, or quasi KKT reformulation.
3. Solve the reformulated system with the interior-point solver in `solver.jl`.
4. Extract the primal strategies and analyze the resulting equilibrium.

For new experiments, prefer the `experiments/` environment and the existing
problem-construction patterns in `Intersection.jl` and `test/runtests.jl`.
