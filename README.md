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
problems together with two solvers: a primal-dual interior-point method (`PDIP`,
Algorithm 5.1 of the paper) and a projected Levenberg–Marquardt method
(`ProjectedLM`), which solves the reduced system in Scholtes form down a ρ
homotopy. The experiment scripts reproduce the main computational examples from
the paper, including the two-player intersection scenario and quadratic hierarchy
benchmarks, plus a two-arm robotic pot-carrying game solved in Scholtes form.

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

The active script constructs a two-player open-loop trajectory game, generates the
reduced GOOP KKT system in Scholtes form, solves it down a ρ schedule with
`ProjectedLM`, and saves trajectory and solution data under
`data/Intersection_open_loop/`.

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
| `src/`         | Core GOOP problem representation, KKT reformulation generators, and the solvers.                 |
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

These functions construct nonlinear KKT systems, each a `GOOPKKTSystem` for the
solvers in `solver.jl`:

| Function                                    | Description                                                                                                                                                                                                                                    |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_slacked_reduced_kkt_system(...)`  | Builds a reduced, slacked KKT system recursively. It introduces preference slacks, interior-point slacks, equality duals, inequality duals, lower-level policy multipliers, and complementarity-relaxation terms.                              |
| `generate_slacked_quasi_kkt_system(...)`    | Calls the reduced generator with `drop_higher_order_terms = true`. The code implements this by propagating `QuasiLagrangianTerm` objects and truncating higher-order derivative contributions.                                                 |
| `generate_slacked_complete_kkt_system(...)` | Builds the complete KKT system, carrying inner-level stationarity, inequality rows and multipliers into the outer levels: `complementarity = :exact` (equations (2.12)–(2.17)) or `:perturbed` (equation (6.1), complementarity relaxed to ρ). |
| `generate_slacked_scholtes_reduced_kkt_system(...)` | Builds the reduced system in Scholtes form: the same recursion as the reduced generator, with the relaxation σ ⊙ γ ≤ ρ𝟙 written as σ ⊙ γ + u − ρ𝟙 = 0 (u ≥ 0) and the lower levels' relaxed complementarity carried as an exact product σ̂ ⊙ ϕ = 0 (see below). |
| `initialize_scholtes!(...)` | Starts the Scholtes system at a primal guess and a level ρ: a cold start (every inequality strictly satisfied) or a warm, continuation start. |

From the source, the reduced formulation appears to encode lower-level stationarity
and complementarity information more compactly through recursively propagated
policy conditions and multipliers. The complete formulation keeps a more explicit
representation of inner KKT variables and constraints. The quasi formulation is
a reduced formulation with selected higher-order derivative terms omitted.

### `src/goop_kkt_system.jl`

`goop_kkt_system.jl` defines `GOOPKKTSystem`, the container every solver in
`solver.jl` reads. It stores:

- in-place evaluators of the residual F(z; ρ), its sparse Jacobian with respect to
  the decision vector z, the unrelaxed residual K₀ (F at ρ = 0), and the values the
  auxiliary slacks stand for;
- index sets for primal variables, preference slacks, interior-point slacks,
  inequality duals, the equality/stationarity dual subsets used by selective warm
  starts, and the sign box (the coordinates kept positive);
- KKT and variable dimensions, and tags naming the system (`kind`,
  `complementarity`);
- the symbolic residual and symbolic variable vector used to build the system, and,
  for the Scholtes system, its complementarity pairs.

`BuildGOOPKKTSystem(...)` selects either the Symbolics or FastDifferentiation
backend, builds the in-place residual function, and constructs the sparse Jacobian
with respect to the full decision vector.

### `src/solver.jl`

`solver.jl` provides the solvers, each called as `solve(Solver(), kkt, θ; z₀, options)`:

| Solver          | Description                                                                                                                                                                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PDIP`          | Algorithm 5.1: Newton's method on F(z; ρ) = 0 for each relaxation ρ, with the minimum-norm step −J⁺F (`:svd`, the default) or a sparse QR, KLU or LDLᵀ step, and a backtracking (or fraction-to-boundary) line search that keeps the sign box positive. Options: `PDIPOptions`. For the perturbed reduced (5.2) and complete (6.1) systems. |
| `ProjectedLM`   | The projected Levenberg–Marquardt method: the Levenberg–Marquardt step on the coordinates that are free, a scaled gradient step on those held at their floor (a two-metric projection), an Armijo line search along the projected arc, and an η schedule from the gain ratio. Options: `ProjectedLMOptions`; linear solvers `:normal` (default), `:klu`, `:ldl`. For the Scholtes system and the complete systems. |

With `record_convergence = true`, `PDIP` and `ProjectedLM` also return per-iteration
histories (‖F(z; ρ)‖, ‖K₀‖, step sizes, and for `ProjectedLM` η).

### The Scholtes system and `ProjectedLM`

```julia
using ReducedGOOP: generate_slacked_scholtes_reduced_kkt_system, initialize_scholtes!, solve,
    ProjectedLM, ProjectedLMOptions
kkt = generate_slacked_scholtes_reduced_kkt_system(goop)
z₀ = initialize_scholtes!(zeros(kkt.variable_dimension), kkt, θ, x₀, ρ; warm = false) # g(x₀) > 0
result = solve(ProjectedLM(), kkt, θ; z₀, options = ProjectedLMOptions(; ρ_schedule = [ρ]))
result.x, result.status, result.residual   # primal answer, status, ‖K₀‖
```

A level marked `is_prioritized_constraint = true` introduces one
`preference_slack` component per preference entry. The level minimizes
`sum(preference_slack)` subject to `h + preference_slack ≥ 0` and
`preference_slack ≥ 0`. Upper levels retain these same slack variables and both
inequality blocks, with their own multipliers and the lower-level Scholtes KKT
conditions. As in the reduced formulation, stationarity differentiates with
respect to the player's original variables and that level's own preference slack;
inherited preference slacks and lower-level multipliers are held fixed.
`initialize_scholtes!` sets the slacks from the values they stand for;
`result.x` contains the original primal variables and
`result.z[kkt.preference_slack_dims]` the preference slacks. Pass the latter to
`initialize_scholtes!(...; preference_slacks = ...)` to keep them during
continuation.

### The Scholtes engine (`src/scholtes_*.jl`)

The first port of the Scholtes solver, `solve(Scholtes(), …)` on its own
`ScholtesKKTSystem`. It is kept, with its own tests, while it remains; new code
uses `ProjectedLM` on `generate_slacked_scholtes_reduced_kkt_system`.

| File                                       | Contents                                                                                                                                                                                                                                                      |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scholtes_kkt.jl`                          | `ScholtesKKTSystem`: stationarity and equalities `F_nc`, level inequalities `a` (including prioritized preference slacks, then φ slacks ρ𝟙 − G ⊙ γ), and their duals `b`. Only the explicit formulation (φ = true) is supported. |
| `scholtes_residual.jl`                     | The relaxed residual `R(w; ρ)` (`s ⊙ γ + u = ρ𝟙`, exact `σ ⊙ φ = 0`), the true residual ‖K₀‖ and the sign box.                                                                                                                                               |
| `scholtes_linsolve.jl`, `scholtes_step.jl` | The regularized least-squares Newton step `δ = −Jᵀ(η²I + JJᵀ)⁻¹r` (`:normal`, `:klu_eta`, `:klu_sqrt_eta`, `:svd`) and the projected two-metric bound rule.                                                                                                   |
| `scholtes_solver.jl`                       | `Scholtes`, `ScholtesOptions`, `solve`, `geometric_schedule`, `scholtes_warm_start`.                                                                                                                                                                          |
| `scholtes_certify.jl`                      | `stat_feas`, `solve_certified`: judge a fixed-ρ solve on the hypotheses of Scholtes' theorem (stationarity/feasibility, complementarity shortfall ≈ ρ, constraint margin).                                                                                    |
| `scholtes_multistart.jl`                   | `solve_multi`, `random_starts`, `grid_starts`, `is_feasible`.                                                                                                                                                                                                 |

## Experiments

| File                                   | Description                                                                                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiments/Intersection.jl`          | Two-player open-loop intersection example with trajectory dynamics, prioritized preferences, the reduced system in Scholtes form solved by `ProjectedLM`, and result plotting. |
| `experiments/ExamplesQP.jl`            | Lightweight entry point for the quadratic-program example.                                                                                                          |
| `experiments/robotic_arm_core.jl`      | Two-arm pot-carrying game (in Scholtes form, x₀ as the parameter θ), the ρ sweep and plan metrics.                                                                  |
| `experiments/Robotic_arm_final.jl`     | The robotic arm, open loop or receding horizon: `Robotic_arm_final.demo(; receding_horizon = 1, plot_fig = true, …)`.                                               |
| `experiments/Robotic_arm_receding.jl`  | Python/juliacall entry points (`build_mpc_context`, `create_planner_from_context`).                                                                                 |
| `experiments/Robotic_arm_plotting.jl`  | Robotic-arm figures (Plots.jl): a PDF and an interactive HTML per initial guess and per solve, ported from ScholtesReducedGOOP.jl's `examples/robotic_arm_plot.jl`. |
| `experiments/Intersection_plotting.jl` | Figures of the intersection scenario.                                                                                                                               |

## Tests

`test/runtests.jl` exercises the reduced and quasi KKT formulations with `PDIP`
(Algorithm 5.1, the regularized `:klu` step), and includes `test/solver.jl` (the two solvers, `PDIP` and
`ProjectedLM`), `test/complete_kkt.jl` (the complete systems) and
`test/scholtes.jl` (the Scholtes system, from ScholtesReducedGOOP.jl's tests, and
the Scholtes engine's own tests).

| Benchmark family              | What is tested                                                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Known-solution benchmarks     | Single-player and coupled three-player cases with quadratic/linear and nonlinear/nonlinear variants.                                                                                 |
| Complete KKT smoke test       | Agreement between complete and reduced formulations on an unconstrained quadratic problem.                                                                                           |
| Code-generation parity        | Agreement between Symbolics and FastDifferentiation residual/Jacobian evaluators.                                                                                                    |
| Warm-start tests              | Full-vector `PDIP` warm starts.                                                                                                                                                            |
| Solvers (`test/solver.jl`)    | The Newton step of every linear solver, the η schedule, Algorithm 5.1 on the perturbed reduced and complete systems, termination reasons, option checks.                         |
| Scholtes (`test/scholtes.jl`) | The Scholtes system's rows, prioritized preference slacks, the quasi system, starts, solves with every linear solver, and exact inequalities at the solution; the engine's certificate and multistart. |

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
2. Generate a complete, reduced, or quasi KKT reformulation, or the reduced system in Scholtes form.
3. Solve it with a solver in `solver.jl`: `solve(ProjectedLM(), …)` for the
   Scholtes system and the complete systems, `solve(PDIP(), …)` for the perturbed
   reduced and complete systems.
4. Extract the primal strategies and analyze the resulting equilibrium.

For new experiments, prefer the `experiments/` environment and the existing
problem-construction patterns in `Intersection.jl` and `test/runtests.jl`.
