# QuasiGOOP.jl

[![CI](https://github.com/CLeARoboticsLab/QuasiGOOP.jl/actions/workflows/test.yml/badge.svg)](https://github.com/CLeARoboticsLab/QuasiGOOP.jl/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD-new)](https://opensource.org/license/bsd-3-clause)

This repository contains the implementation accompanying the paper:

> [Breaking Exponential Complexity in Games of Ordered Preference: A Tractable Reformulation](https://arxiv.org/abs/2603.26950)

Games of Ordered Preference (GOOP) model strategic interactions in which each
player optimizes a hierarchy of objectives and constraints, rather than a single
scalar cost. This is useful for settings such as robotics and multi-agent
planning, where a player first satisfies safety or task constraints and only
then optimize lower-priority behavior.

`ReducedGOOP.jl` implements tractable reformulations of GOOP problems and provides
both nonlinear KKT systems for a primal-dual interior-point method and mixed
complementarity problem (MiCP/MCP) formulations for PATH. The experiment scripts
reproduce the main computational examples from the paper, including the
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
GOOP reformulation, solves it with PATH by default, and saves trajectory,
velocity, control, warm-start, and solution data under `data/Intersection_open_loop/`.

### Single-Player Trilevel Quadratic Program

The following runs the quadratic-program example driver:

```julia
include("experiments/ExamplesQP.jl")
```

`ExamplesQP.jl` loads `experiments/trilevel_QP_PATH.jl`, which compares complete
and reduced MCP formulations on a bounded single-player quadratic hierarchy with
linear equality and inequality constraints. The script reports solve status,
residuals, formulation sizes, and primal-solution agreement between the complete
and reduced systems.

Note: the active `trilevel_QP_PATH.jl` file creates three quadratic objectives
but currently passes two objective levels to `ParametricGOOP` for debugging purpose.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/` | Core GOOP problem representation, KKT/MCP reformulation generators, and solver wrappers. |
| `experiments/` | Reproduction scripts, plotting utilities, and the experiment-specific Julia environment. |
| `test/` | PATH-based regression tests for complete and reduced MCP formulations. |
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

#### MCP-Based Formulations

These functions construct `ParametricMCP` objects intended for PATH via
`ParametricMCPs.jl`:

| Function | Description |
| --- | --- |
| `generate_mcp_reduced_kkt_system(...)` | Builds the reduced MCP analogue of the reduced KKT formulation. Inequality constraints are represented through MCP bounds rather than interior-point slack variables. The relaxation parameter is appended to the parameter vector. |
| `generate_mcp_quasi_kkt_system(...)` | Calls the reduced MCP generator with higher-order terms dropped. |
| `generate_mcp_complete_kkt_system(...)` | Builds the complete MCP analogue, stacking player-wise KKT equations and lower-bounded complementarity variables for PATH. |

The MCP formulations correspond to the KKT-based formulations structurally, but
encode complementarity using lower and upper variable bounds instead of the
slacked primal-dual equations used by the interior-point solver.

### `src/goop_kkt_system.jl`

`goop_kkt_system.jl` defines `GOOPKKTSystem`, the container used by the
interior-point solver. It stores:

- in-place residual and Jacobian evaluators for the KKT residual and its
  derivative with respect to the decision vector;
- index sets for primal variables, preference slacks, interior-point slacks, and
  inequality duals;
- KKT and variable dimensions;
- the symbolic residual and symbolic variable vector used to build the system.

`BuildGOOPKKTSystem(...)` selects either the Symbolics or FastDifferentiation
backend, builds an in-place residual function, constructs a sparse Jacobian with
respect to the full decision vector, and records constant sparse entries for
efficient repeated evaluation.

### `src/solver.jl`

`solver.jl` provides two solver front ends:

| Solver | Description |
| --- | --- |
| `InteriorPoint` | Solves a `GOOPKKTSystem` by Newton steps on the relaxed primal-dual residual. It initializes preference slacks, interior-point slacks, and inequality duals to positive values, supports backtracking and fraction-to-boundary line search, and can record KKT-error histories. |
| `PATHSolver` | Solves a `ParametricMCP` by calling `ParametricMCPs.solve`, passing the GOOP parameters together with the complementarity relaxation parameter. |

The solver options are configured through `InteriorPointOptions` and `PATHOptions`.

## Experiments

| File | Description |
| --- | --- |
| `experiments/Intersection.jl` | Two-player open-loop intersection example with trajectory dynamics, prioritized preferences, PATH/interior-point solver selection, continuation over relaxation values, and result plotting. |
| `experiments/ExamplesQP.jl` | Lightweight entry point for the quadratic-program example. |
| `experiments/trilevel_QP_PATH.jl` | PATH-based comparison between complete and reduced MCP formulations for a bounded quadratic hierarchy. |
| `experiments/Plotting.jl` | Plotting utilities used by the intersection experiments. |

## Tests

The current tests in `test/runtests.jl` focus on PATH MCP formulations. The
test file defines:

```julia
complete_mcp_system(problem) =
    QuasiGOOP.generate_mcp_complete_kkt_system(problem)

reduced_mcp_system(problem) =
    QuasiGOOP.generate_mcp_reduced_kkt_system(problem)
```

The test suite checks the complete MCP generator directly and compares the
reduced MCP generator against complete-MCP primal solutions.

| Benchmark family | What is tested |
| --- | --- |
| Baseline nonnegative problem | A degenerate single-level problem `min 0` subject to `x >= 0`, used as a basic PATH sanity check. |
| Baseline bilevel PSD problem | A two-level problem with a rank-deficient upper quadratic and a degenerate lower objective under nonnegativity constraints. |
| Single-player benchmarks | Single-level, bilevel, and trilevel problems in dimension 5 with known box-constrained solutions. Both quadratic/linear and nonlinear/nonlinear variants are tested. |
| Three-player benchmarks | Coupled generalized Nash problems with three players, known primal solutions, active resource constraints, and the same hierarchy depths and objective/constraint variants as the single-player tests. |

The tests verify PATH success status, residual tolerances, agreement with known
primal solutions, active/inactive constraint behavior, and agreement between
reduced and complete MCP primal solutions.

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
2. Generate either a KKT or MCP reformulation.
3. Solve the reformulated system with the interior-point solver in `solver.jl` or
   the PATH interface through `ParametricMCPs.jl`.
4. Extract the primal strategies and analyze the resulting equilibrium.

For new experiments, prefer the `experiments/` environment and the existing
problem-construction patterns in `Intersection.jl` and `test/runtests.jl`.
