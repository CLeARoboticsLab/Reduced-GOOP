# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the full test suite
julia --project=. -e 'import Pkg; Pkg.test()'

# Run a single test file directly
julia --project=test test/runtests.jl

# Instantiate the experiment environment (first time or after adding deps)
julia --project=experiments -e 'import Pkg; Pkg.instantiate()'

# Format source code
julia --project=. -e 'using JuliaFormatter; format(".")'
```

Run the intersection demo from a `julia --project=experiments` REPL:

```javascript
using Revise
includet("experiments/Intersection.jl")
Intersection.demo(random_initial_state = false)

# Or run the QP benchmark
include("experiments/ExamplesQP.jl")
```

## Architecture

The package models and solves **Games of Ordered Preference (GOOP)**: multi-player problems where each player optimizes a *hierarchy* of objectives and constraints rather than a single scalar cost. It accompanies the paper "Breaking Exponential Complexity in Games of Ordered Preference: A Tractable Reformulation" (arXiv:2603.26950).

### Source files

| File                 | Role                                                                                            |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `goop.jl`            | `ParametricGOOP` struct and the complete, reduced, and quasi KKT generators                     |
| `goop_kkt_system.jl` | `GOOPKKTSystem` container; `BuildGOOPKKTSystem` compiles symbolic equations to sparse Jacobians |
| `solver.jl`          | `InteriorPoint` solver and its options                                                          |
| `ReducedGOOP.jl`     | Module boundary: imports, includes, exports only                                                |

### Core data flow

1. **Define** a `ParametricGOOP` (in `src/goop.jl`) — the central struct that holds player preferences, constraint flags, equality/inequality constraints (per-player and shared), and all dimension metadata.
2. **Reformulate** with the complete, reduced, or quasi generator into a KKT system.
3. **Solve** with `InteriorPoint` in `src/solver.jl`.
4. **Extract** the primal equilibrium strategies.

### Solver path

| Reformulation                        | Solver                                                                   | Key type        |
| ------------------------------------ | ------------------------------------------------------------------------ | --------------- |
| `generate_slacked_*_kkt_system(...)` | `InteriorPoint` (Newton, fraction-to-boundary or backtracking line search, outer ϵ loop) | `GOOPKKTSystem` |

### Three formulation variants

| Variant    | Description                                                                                                                                                 |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `complete` | Explicit nested KKT — keeps inner-level stationarity, constraints, and decision variables fully expanded in outer levels                                    |
| `reduced`  | Recursive compact encoding — lower-level stationarity is propagated through policy multipliers; smaller system                                              |
| `quasi`    | Same as `reduced` with `drop_higher_order_terms = true`; implemented via `QuasiLagrangianTerm` objects that truncate derivative contributions above order 2 |

### `GOOPKKTSystem` (`src/goop_kkt_system.jl`)

Built by `BuildGOOPKKTSystem(...)`, which selects either the `Symbolics` or `FastDifferentiation` backend to compile symbolic residuals into in-place functions and build a static sparse Jacobian. Stores index sets for primal variables, preference slacks, interior-point slacks, inequality duals, and selective equality/stationarity dual warm starts — used by `InteriorPoint` to initialize positive slack variables and perform bounded Newton steps.

### Non-obvious conventions

- **Preference ordering**: preferences are stored `[outermost, ..., innermost]` with innermost being the highest priority.
- `x[Block(i)][j]` is the j-th decision variable of player i (uses `BlockArrays.jl`).

### Variable nomenclature

| Symbol | Meaning                                       | Sign constraint |
| ------ | --------------------------------------------- | --------------- |
| `x`    | Primal decision variables (blocked by player) | —               |
| `θ`    | Parameters (e.g. targets/goals)               | —               |
| `s`    | Preference slack variables                    | `s ≥ 0`         |
| `σ`    | Interior-point slack variables                | `σ ≥ 0`         |
| `λ`    | Equality constraint duals                     | —               |
| `γ`    | Inequality and preference constraint duals    | `γ ≥ 0`         |
| `ψ`    | Policy constraint duals (policy gradient)     | —               |
| `ϵ`    | Complementarity relaxation parameter          | `ϵ > 0`         |
| `η`    | Newton regularization parameter               | `η ≥ 0`         |

## Julia Environments

Three separate environments exist in this repo:

| Environment    | Activate                                   | Purpose                       |
| -------------- | ------------------------------------------ | ----------------------------- |
| Root           | `julia --project=.`                        | Package development           |
| `experiments/` | `julia --project=experiments`              | Demo and reproduction scripts |
| `test/`        | `julia --project=test` or via `Pkg.test()` | Test suite                    |

Both `experiments/Project.toml` and `test/Project.toml` contain:

```javascript
[sources]
ReducedGOOP = {path = ".."}
```

Always preserve these entries — they route imports to the local `src/` rather than a registered version.

`test/Manifest.toml` must not exist or be committed. When present it drifts from the root `Manifest.toml` and causes `Pkg.test()` to emit version-conflict warnings for every shared package.

## Symbolics `ifelse`

This keeps preferences in a continuous, differentiable (i.e., no kink at origin) format, which the solver needs.

## Experiment Logging

For every execution of `intersection.jl`, save inputs, outputs, and intermediate results under `runs/intersection/<timestamp>/`. Use descriptive filenames and never overwrite previous runs.

## Pre-Completion Checklist

Before finishing any task that touches `src/`:

- `julia --project=. -e 'using Pkg; Pkg.test()'` passes with no new failures
- `[sources]` entries in `experiments/Project.toml` and `test/Project.toml` still point to `..`

## Repository Style Rules

- Whenever the user asks you to implement something, do not begin implementation if any part of the request is unclear, if you have concerns that the proposed approach may not work, or if you anticipate any technical issues, risks, or other concerns. Instead, ask clarifying questions and discuss the concerns with the user until the requirements are fully understood and the issues are resolved. Only then should you proceed with implementation.
- Keep package code in `src/` only.
- Keep `src/PackageName.jl` thin: imports, includes, exports, and module boundary only.
- Split implementation files by concept, e.g. `problem.jl`, `solver.jl`, `game.jl`
- Use `CamelCase` for public types and modules.
- Use lowercase or snake\_case for functions.
- Use `!` suffix for mutating functions and in-place callbacks.
- Use `_` prefix for private helpers.
- Use explicit keyword arguments for function options and complex constructors.
- Write docstrings for exported types and functions.
- Keep plotting, demos, and visualization code in `experiments/`, not `src/`.
- Give `experiments/` and `test/` their own `Project.toml` when they need extra dependencies.
- In those sub-environments, use `[sources] PackageName = {path = ".."}` to point to the local package.
- Format code with JuliaFormatter using the project `.JuliaFormatter.toml`.
