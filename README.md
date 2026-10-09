# ReducedGOOP.jl

[!\[CI\](https://github.com/CLeARoboticsLab/QuasiGOOP.jl/actions/workflows/test.yml/badge.svg)](https://github.com/CLeARoboticsLab/QuasiGOOP.jl/actions/workflows/test.yml)
[!\[License\](https://img.shields.io/badge/license-BSD-new)](https://opensource.org/license/bsd-3-clause)

This repository contains the implementation accompanying the paper:

> [Breaking Exponential Complexity in Games of Ordered Preference: A Tractable Reformulation](https://arxiv.org/abs/2603.26950)

Games of Ordered Preference (GOOP) model strategic interactions in which each
player optimizes a hierarchy of objectives and constraints, rather than a single
scalar cost. This is useful for settings such as robotics and multi-agent
planning, where a player first satisfies safety or task constraints and only
then optimize lower-priority behavior.

`ReducedGOOP.jl` implements tractable nonlinear KKT reformulations of GOOP
problems together with two solvers: a primal-dual interior-point method and a
Scholtes-relaxation ρ homotopy with a projected Newton step. The experiment
scripts reproduce the main computational examples from the paper, including the
two-player intersection scenario and quadratic hierarchy benchmarks, plus a
two-arm robotic pot-carrying game solved with the Scholtes path.

The Scholtes relaxation scheme (the explicit φ formulation, projected two-metric
step and ρ homotopy, with the certificate and multistart layers) was designed and
first implemented by **Jingqi Li** in ScholtesReducedGOOP.jl, and is ported here.

## Installation and Usage

This repository is a Julia package. The experiment drivers use the Julia
environment in `experiments/`, which depends on the local package at the
repository root.

From the repository root:

```bash
julia --project=experiments
```

Inside Julia, instantiate the experiment environment once:

```javascript
import Pkg
Pkg.instantiate()
```

### Two-Player Intersection Scenario

The following reproduces the deterministic two-player intersection example used
in the paper:

```javascript
using Revise
includet("experiments/Intersection.jl")
Intersection.demo(random_initial_state = false)
```

If `Revise.jl` is not installed in your local Julia setup, use `include` instead:

```javascript
include("experiments/Intersection.jl")
Intersection.demo(random_initial_state = false)
```

The active script constructs a two-player open-loop trajectory game, generates a
GOOP KKT reformulation, solves it with the interior-point solver, and saves
trajectory and solution data under `data/Intersection_open_loop/`.

### Single-Player Trilevel Quadratic Program

The following runs the quadratic-program example driver:

```javascript
include("experiments/ExamplesQP.jl")
```

`ExamplesQP.jl` loads `experiments/trilevel_QP.jl`, which solves a bounded
single-player quadratic hierarchy with the interior-point method and checks its
dual solution independently with `NonlinearSolve`.

## Repository Structure

| Path           | Purpose                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `src/`         | Core GOOP problem representation, KKT reformulation generators, and interior-point solver.       |
| `experiments/` | Reproduction scripts, plotting utilities, and the experiment-specific Julia environment.         |
| `test/`        | Regression tests for KKT formulations, code generation, KLU solves, and warm starts.             |
| `legacy/`      | Older implementations, archived formulations, and historical experiments retained for reference. |
| `data/`        | Generated experiment outputs and archived result artifacts.                                      |

## Core Implementation

### `src/goop.jl`

`goop.jl` defines the main GOOP modeling interface and reformulation generators.

| Symbol                            | Role                                                                                                                                                                                                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ParametricGOOP`                  | Stores player preferences, prioritized-constraint flags, player-wise equality and inequality constraints, dimensions, and number of players. (Shared constraints were removed; write a shared constraint into each player's own constraints.) |
| `ParametricGOOP(x, theta; ...)`   | Convenience constructor that infers primal, parameter, equality, and inequality dimensions from template block vectors.                                                                                                                       |
| `QuasiLagrangianTerm` and helpers | Internal machinery for the quasi formulation; it builds gradients while dropping higher-order derivative terms after a bounded order.                                                                                                         |

#### KKT-Based Formulations

These functions construct nonlinear KKT systems represented as `GOOPKKTSystem`
objects and solved by the interior-point method in `solver.jl`. The reduced
generators also take `complementarity = :scholtes`, which returns a
`ScholtesKKTSystem` for the Scholtes solver instead (see below):

| Function                                    | Description                                                                                                                                                                                                       |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_slacked_reduced_kkt_system(...)`  | Builds a reduced, slacked KKT system recursively. It introduces preference slacks, interior-point slacks, equality duals, inequality duals, lower-level policy multipliers, and complementarity-relaxation terms. |
| `generate_slacked_quasi_kkt_system(...)`    | Calls the reduced generator with `drop_higher_order_terms = true`. The code implements this by propagating `QuasiLagrangianTerm` objects and truncating higher-order derivative contributions.                    |
| `generate_slacked_complete_kkt_system(...)` | Builds a more explicit nested KKT system by carrying inner-level stationarity, inequality rows, and decision variables into outer-level KKT conditions.                                                           |

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

| Solver          | Description                                                                                                                                                                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InteriorPoint` | Solves a `GOOPKKTSystem` by Newton steps on the relaxed primal-dual residual. It initializes preference slacks, interior-point slacks, and inequality duals to positive values, supports backtracking and fraction-to-boundary line search, and can record KKT-error histories. |

Solver options are configured through `InteriorPointOptions`. Its sparse linear
solvers are `:normal`, `:klu_sqrt_eta` (alias `:klu`; augmented diagonals ±√η) and
`:klu_eta` (diagonals ±η, the Scholtes source's convention), with an SVD fallback.

### Scholtes relaxation (`src/scholtes_*.jl`)

| File                                       | Contents                                                                                                                                                                                                                                                      |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scholtes_kkt.jl`                          | `ScholtesKKTSystem`: the reduced KKT system as three blocks — stationarity and equalities `F_nc`, the sign-constrained functions `a` (g per level, then the φ slacks ρ𝟙 − g ⊙ γ) and their duals `b`. Only the explicit formulation (φ = true) is supported. |
| `scholtes_residual.jl`                     | The relaxed residual `R(w; ρ)` (`s ⊙ γ + u = ρ𝟙`, exact `σ ⊙ φ = 0`), the true residual ‖K₀‖ and the sign box.                                                                                                                                               |
| `scholtes_linsolve.jl`, `scholtes_step.jl` | The regularized least-squares Newton step `δ = −Jᵀ(η²I + JJᵀ)⁻¹r` (`:normal`, `:klu_eta`, `:klu_sqrt_eta`, `:svd`) and the projected two-metric bound rule.                                                                                                   |
| `scholtes_solver.jl`                       | `Scholtes`, `ScholtesOptions`, `solve`, `geometric_schedule`, `scholtes_warm_start`.                                                                                                                                                                          |
| `scholtes_certify.jl`                      | `stat_feas`, `solve_certified`: judge a fixed-ρ solve on the hypotheses of Scholtes' theorem (stationarity/feasibility, complementarity shortfall ≈ ρ, constraint margin).                                                                                    |
| `scholtes_multistart.jl`                   | `solve_multi`, `random_starts`, `grid_starts`, `is_feasible`.                                                                                                                                                                                                 |

```javascript
using ReducedGOOP: generate_slacked_reduced_kkt_system, solve, Scholtes, ScholtesOptions
kkt = generate_slacked_reduced_kkt_system(goop; complementarity = :scholtes)
result = solve(Scholtes(), kkt, θ; z₀, options = ScholtesOptions(linear_solver = :normal))
result.z, result.converged, result.residual   # primal answer, status, ‖K₀‖
```

The Scholtes path requires `projected_step = true` (the default); `false` throws.
Factorization reuse (`reuse_factorization_iters > 0`) is interior-point only.

## Experiments

| File                                   | Description                                                                                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiments/Intersection.jl`          | Two-player open-loop intersection example with trajectory dynamics, prioritized preferences, an interior-point solve, and result plotting.                          |
| `experiments/ExamplesQP.jl`            | Lightweight entry point for the quadratic-program example.                                                                                                          |
| `experiments/robotic_arm_core.jl`      | Two-arm pot-carrying game (Scholtes scenario, x₀ as the parameter θ), the ρ sweep and plan metrics.                                                                 |
| `experiments/Robotic_arm_final.jl`     | The robotic arm, open loop or receding horizon: `Robotic_arm_final.demo(; receding_horizon = 1, plot_fig = true, …)`.                                               |
| `experiments/Robotic_arm_receding.jl`  | Python/juliacall entry points (`build_mpc_context`, `create_planner_from_context`).                                                                                 |
| `experiments/Robotic_arm_plotting.jl`  | Robotic-arm figures (Plots.jl): a PDF and an interactive HTML per initial guess and per solve, ported from ScholtesReducedGOOP.jl's `examples/robotic_arm_plot.jl`. |
| `experiments/Intersection_plotting.jl` | Figures of the intersection scenario.                                                                                                                               |

## Tests

`test/runtests.jl` exercises the complete and reduced slacked KKT formulations
with the interior-point solver, and includes `test/scholtes.jl`, the Scholtes
solver's tests ported from ScholtesReducedGOOP.jl.

| Benchmark family              | What is tested                                                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Known-solution benchmarks     | Single-player and coupled three-player cases with quadratic/linear and nonlinear/nonlinear variants.                                                                                 |
| Complete KKT smoke test       | Agreement between complete and reduced formulations on an unconstrained quadratic problem.                                                                                           |
| Code-generation parity        | Agreement between Symbolics and FastDifferentiation residual/Jacobian evaluators.                                                                                                    |
| KLU solver tests              | Augmented-system direction accuracy, factorization reuse, singular-retry behavior, and agreement with dense SVD.                                                                     |
| Warm-start tests              | Full-vector solver warm starts.                                                                                                                                                      |
| Scholtes (`test/scholtes.jl`) | KKT blocks, residual and sign box, every linear backend, the projected bound rule, ρ schedules, certificate, multistart, non-unique answers, and exact inequalities at the solution. |

`test/compare_with_scholtes.jl` compares every Scholtes solve in the suite against
a recording of ScholtesReducedGOOP.jl's own tests (a development tool, not run by
`Pkg.test()`).

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
3. Solve the reformulated system with the interior-point solver in `solver.jl`,
   or, with `complementarity = :scholtes`, with `solve(Scholtes(), …)`.
4. Extract the primal strategies and analyze the resulting equilibrium.

For new experiments, prefer the `experiments/` environment and the existing
problem-construction patterns in `Intersection.jl` and `test/runtests.jl`.
