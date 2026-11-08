# ReducedGOOP.jl

[!\[CI\](https://github.com/CLeARoboticsLab/Reduced-GOOP/actions/workflows/test.yml/badge.svg)](https://github.com/CLeARoboticsLab/Reduced-GOOP/actions/workflows/test.yml)
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
step and ρ homotopy, with the multistart layer) was designed and
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
| `is_feasible(goop, z, θ)`         | The worst (most negative) inequality value at a primal point, over all players, and whether it is within `tol` of feasible. |
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
| `initialize_scholtes!(...)` | Starts the Scholtes system at a primal guess and a stage ρ: a cold start (every inequality strictly satisfied) or a warm, continuation start. |

The reduced formulation carries the lower levels' stationarity upward as policy
constraints with their own multipliers; the complete formulation keeps every lower
level's KKT conditions and multipliers explicitly; the quasi formulation is the
reduced one without the terms differentiated more than `quasi_order` times.

### `src/goop_kkt_system.jl`

`goop_kkt_system.jl` defines `GOOPKKTSystem`, the container every solver in
`solver.jl` reads. It stores:

- in-place evaluators of the residual K_ρ(z) (`K_ρ!`), its sparse Jacobian with respect to
  the decision vector z (`∇K_ρ!`), the unrelaxed residual K₀ (K_ρ at ρ = 0), and the values the
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
| `PDIP`          | Algorithm 5.1: Newton's method on K_ρ(z) = 0 for each relaxation ρ, with the minimum-norm step −J⁺K_ρ (`:svd`, the default) or a sparse QR, KLU or LDLᵀ step, and a backtracking (or fraction-to-boundary) line search that keeps the sign box positive. Options: `PDIPOptions`. For the perturbed reduced (5.2) and complete (6.1) systems. |
| `ProjectedLM`   | The projected Levenberg–Marquardt method: the Levenberg–Marquardt step on the coordinates that are free, a scaled gradient step on those held at their floor (a two-metric projection), an Armijo line search along the projected arc, and an η schedule from the gain ratio. Options: `ProjectedLMOptions`; linear solvers `:normal` (default), `:klu`, `:ldl`. For the Scholtes system and the complete systems. |

With `record_convergence = true`, `PDIP` and `ProjectedLM` also return per-iteration
histories (‖K_ρ(z)‖, ‖K₀‖, step sizes, and for `ProjectedLM` η). Both report `norm_K₀` and `norm_K_ρ` at
the returned point.

### `src/multistart.jl`

`solve_multi(solver, kkt, θ, starts; options, accept, dedupe_tol)` solves one system from many starts with
`PDIP` or `ProjectedLM` and merges the primal points that coincide: the different points of a GOOP's
answer when it is a set, or a local convergence study around one solution. The starts come from
`ball_starts` (uniform in a ball around a full iterate, in the primal coordinates only or in all of them),
`random_starts` (strictly feasible primal points in a box) or `grid_starts`.

### The Scholtes system and `ProjectedLM`

```julia
using ReducedGOOP: generate_slacked_scholtes_reduced_kkt_system, initialize_scholtes!, solve,
    ProjectedLM, ProjectedLMOptions
kkt = generate_slacked_scholtes_reduced_kkt_system(goop)
z₀ = initialize_scholtes!(zeros(kkt.variable_dimension), kkt, θ, x₀, ρ; warm = false) # g(x₀) > 0
result = solve(ProjectedLM(), kkt, θ; z₀, options = ProjectedLMOptions(; ρ_schedule = [ρ]))
result.x, result.status, result.norm_K₀   # primal answer, status, ‖K₀‖
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

## Experiments

| File                                   | Description                                                                                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiments/Intersection.jl`          | Two-player open-loop intersection example with trajectory dynamics, prioritized preferences, the reduced system in Scholtes form solved by `ProjectedLM`, and result plotting. |
| `experiments/robotic_arm_core.jl`      | Two-arm pot-carrying game (in Scholtes form, x₀ as the parameter θ), the ρ sweep and plan metrics.                                                                  |
| `experiments/Robotic_arm_final.jl`     | The robotic arm, open loop or receding horizon: `Robotic_arm_final.demo(; receding_horizon = 1, plot_fig = true, …)`.                                               |
| `experiments/Robotic_arm_receding.jl`  | Python/juliacall entry points (`build_mpc_context`, `create_planner_from_context`).                                                                                 |
| `experiments/Robotic_arm_plotting.jl`  | Robotic-arm figures (Plots.jl): a PDF and an interactive HTML per initial guess and per solve, ported from ScholtesReducedGOOP.jl's `examples/robotic_arm_plot.jl`. |
| `experiments/Intersection_plotting.jl` | Figures of the intersection scenario.                                                                                                                               |
| `experiments/dynamics.jl`              | Dynamics constraints (unicycle, bicycle, planar double integrator, single integrators) and `unflatten_trajectory`. |
| `experiments/MonteCarlo_Intersection.jl`| Monte Carlo study of the intersection scenario: reduced solves, complete-system checks of their solutions, figures. |
| `experiments/MonteCarlo_RandomGOOP.jl` | The §6 random GOOP study: reduced and complete systems by `PDIP`, Tables 2–3 and Figs 2 and 4. |
| `experiments/RandomGOOP_local_convergence.jl`| Fig 3: `PDIP` at a fixed ρ from starts around a solution (`solve_multi`, `ball_starts`). |
| `experiments/RandomGOOP_plotting.py`   | The figures and tables of both RandomGOOP drivers. |

## Tests

`test/runtests.jl` holds the suite, one testset per behavior; `test/problems.jl` holds the test problems
(known-solution benchmark families, the small problems of ScholtesReducedGOOP.jl's tests, a coupled game
with a rank-deficient J, one-variable complete-system problems) and the helpers that build and solve them.

| Testset                    | What is tested                                                                                     |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| Reduced and quasi systems  | Known solutions by `PDIP` (one or three players, one to three levels), dual-coordinate metadata, the quasi order. |
| Scholtes system            | Rows, unknowns and prioritized preference slacks, the quasi form, `initialize_scholtes!`, `ProjectedLM` solves and exact inequalities at the solution. |
| Complete systems           | The manuscript's equations (2.12)–(2.17) and (6.1), K ≥ 2, the sign box, `ProjectedLM` on `:exact`, agreement with the reduced system. |
| Solvers                    | The Newton step of every linear solver, the η schedule, stages, termination reasons, starts and option checks, warm starts. |
| Code generation            | Symbolics and FastDifferentiation give the same residuals and Jacobians.                            |
| Multistart                 | `solve_multi` on a non-unique answer and from `ball_starts`; strictly feasible random starts.      |

The tests verify convergence status, residual tolerances, known primal
solutions, active/inactive constraint behavior, and linear-solver robustness.

Run the tests from the repository root with:

```bash
julia --project=. -e 'import Pkg; Pkg.test()'
```

`Pkg.test()` builds the test environment from `test/Project.toml` itself; `julia --project=test` does not work
without a `test/Manifest.toml`, which the repository deliberately does not keep.

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
