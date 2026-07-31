# Penalty exponent study: `(level + 2)` vs `(level)`

What the `+ 2` in `smooth_piecewise_preference_objective` buys, and what it costs.

```julia
smooth_piecewise_preference_objective(preference, level; ϵ = 0.0) =
    ifelse(preference ≥ ϵ, 0.0, (ϵ - preference)^(level + 2))
```

## Setup

Driver: `test/quasi_vs_reduced_preference_inequality.jl`, unmodified except for
`COMPARISON_TOLERANCE`. Problem: the 3-player, 3-level benchmark families from
`test/benchmark_problems.jl` with each player's hard inequality `g(x, θ) ≥ 0`
appended as the innermost prioritized preference — the `robotic_arm_core.jl`
layout. The prioritized level index is therefore `K = 4`, so the penalty power is

| variant | power `p` at the prioritized level |
| --- | --- |
| `(level + 2)` | `p = 6` |
| `(level)` | `p = 4` |

Solver held fixed across all runs: sparse `:klu`, `:backtracking` line search,
`ϵ₀ = 0.0`, `max_outer_iters = 1` (inner iterations only), `max_inner_iters =
5000`. Three deterministic starts (`expected`, `zeros`, `offset`) × two
formulations (`:reduced`, `:quasi`) × two families (`:quadratic`, `:nonlinear`).

Four runs: `p ∈ {6, 4}` × `COMPARISON_TOLERANCE ∈ {1e-4, 1e-5}`. Both methods of
`smooth_piecewise_preference_objective` were swept together, though only the
generic (Symbolics) one is traced — the `FD.Node` method is stale and unused.

`:reduced` and `:quasi` agreed to ≤ 3.7e-6 (usually ≤ 1e-11) on every row of
every run, so the tables below report `:reduced`; the exponent question is
orthogonal to the quasi/reduced question.

Throughout, `:failed` means the iteration cap was hit above tolerance — a
lower-optimality iterate, not a divergence. Every run returned finite,
sensible primals.

## Headline result

**Dropping the `+ 2` improves everything the solver is measured on.** It is
roughly a 5–6× reduction in both distance-to-ground-truth and constraint
violation, at every tolerance, on both families. The single thing that breaks is
a *test assertion* about the two formulations agreeing on the feasible side —
and that assertion turns out to be exactly what the `+ 2` was silently buying.

### Distance to ground truth and feasibility, `tol = 1e-4`

`x*` is the benchmark's closed-form constrained solution. Both runs converged
(`:solved`, `‖F‖ ≈ 9.2e-5`–`1.0e-4`) on all six rows.

| family | z₀ | `‖x − x*‖∞` p=6 | p=4 | max violation p=6 | p=4 |
| --- | --- | --- | --- | --- | --- |
| quadratic | expected | 0.0607 | **0.0117** | 0.0910 | **0.0175** |
| quadratic | zeros | 0.7273 | 0.6784 | 0.0909 | **0.0176** |
| quadratic | offset | 0.0606 | **0.0120** | 0.0909 | **0.0180** |
| nonlinear | expected | 0.0626 | **0.0108** | 0.0896 | **0.0160** |
| nonlinear | zeros | 0.7301 | **0.0153** | 0.0907 | **0.0227** |
| nonlinear | offset | 0.0624 | **0.0108** | 0.0894 | **0.0160** |

### Distance to ground truth and feasibility, `tol = 1e-5`

| family | z₀ | `‖x − x*‖∞` p=6 | p=4 | max violation p=6 | p=4 |
| --- | --- | --- | --- | --- | --- |
| quadratic | expected | 0.0463 | **0.00701** | 0.0694 | **0.01051** |
| quadratic | zeros | 0.7141 | 0.6738 | 0.0712 | **0.01076** |
| quadratic | offset | 0.0463 | **0.00704** | 0.0694 | **0.01056** |
| nonlinear | expected | 0.0456 | **0.00704** | 0.0661 | **0.01050** |
| nonlinear | zeros | 0.7136 | **0.00704** | 0.0680 | **0.01050** |
| nonlinear | offset | 0.0456 | **0.00704** | 0.0661 | **0.01050** |

Two effects are visible and they are different in kind:

1. **Within a basin**, `p = 4` is uniformly ~6× closer to `x*` and ~6× less
   infeasible. This is the quantitative effect explained below.
2. **Across basins**, `p = 4` fixes the `nonlinear`/`zeros` case outright:
   `0.730 → 0.0153` at `tol = 1e-4` and `0.714 → 0.00704` at `1e-5`, i.e. it
   lands in the correct basin instead of a spurious one. The
   `quadratic`/`zeros` case stays at ~0.68 under both exponents — that stationary
   point is a property of the problem from that start, not of the exponent,
   though even there the violation still drops 5×.

## Did `(level)` fail to reach `‖F‖ < 1e-5`?

The hypothesis was that it would. **The measurements say the reverse: `p = 6` is
the one that cannot get there.**

`tol = 1e-5`, `:reduced` (`:quasi` identical):

| family | z₀ | p=6 `‖F‖` | iters | p=4 `‖F‖` | iters |
| --- | --- | --- | --- | --- | --- |
| quadratic | expected | 2.13e-5 | 4999 (cap) | 1.02e-5 | 4999 (cap) |
| quadratic | zeros | 2.85e-5 | 4999 (cap) | 1.34e-5 | 4999 (cap) |
| quadratic | offset | 2.13e-5 | 4999 (cap) | 1.04e-5 | 4999 (cap) |
| nonlinear | expected | 1.55e-5 | 4999 (cap) | **1.00e-5 → `:solved`** | 1548 |
| nonlinear | zeros | 2.09e-5 | 4999 (cap) | **1.00e-5 → `:solved`** | 3044 |
| nonlinear | offset | 1.55e-5 | 4999 (cap) | **1.00e-5 → `:solved`** | 1513 |

At `p = 6` all six runs exhaust the 5000-iteration cap and return iterates at
1.5e-5–2.9e-5. At `p = 4` the nonconvex family converges outright, and the
quadratic family stalls only 2–34% above the threshold instead of 113–185%.

At `tol = 1e-4` both exponents reach `:solved` everywhere. Iteration counts are
comparable (43–56 vs 48–76 on quadratic; 34–69 vs 33–66 on nonlinear) with one
outlier: `nonlinear`/`zeros` costs 526 inner iterations at `p = 4` versus 39 at
`p = 6` — that is the price of actually crossing into the correct basin.

## What does break at `(level)`

One assertion, in both `p = 4` runs (`quasi_vs_reduced_preference_inequality.jl:600`):

```
Three-level quadratic: Test Failed
  Expression: gaps.at_feasible_probe.absolute <= 1.0e-12
   Evaluated: 2.999999999999999 <= 1.0e-12
```

This is the claim that on the **feasible** side of a quadratic-objective problem
the `:reduced` and `:quasi` residual maps coincide exactly. It holds at `p = 6`
(gap `0.0`) and fails at `p = 4` (gap `3.0`). The relative residual-map gaps:

| probe | quad p=6 | quad p=4 | nonlin p=6 | nonlin p=4 |
| --- | --- | --- | --- | --- |
| feasible | **0.0** | **0.750** | 0.2117 | 0.2117 |
| violating | 0.798 | 0.591 | 0.25 | 0.25 |
| at z₀ = zeros | 0.864 | 0.677 | 0.179 | 0.241 |

The nonlinear rows are unchanged by the exponent — there the gap is dominated by
the `cosh` objectives' own third derivatives, which the penalty cannot mask.

### Why 3.0, and why `+ 2` is the right amount

`max(−h, 0)^p` is `C^(p−1)` at `h = 0`: value and the first `p − 1` derivatives
all vanish there, the `p`-th is the constant `p!`.

A `K`-level reduced formulation differentiates the innermost Lagrangian once per
level as it recurses outward — level `K` contributes `∇L`, level `K−1` needs
`∇²L`, …, level 1 needs `∇^K L`. Here `K = 4`. `:quasi` truncates above order 2,
so the terms it drops are precisely orders 3…`K`.

| | order-`K` derivative of `(−h)^p` at `h = 0` |
| --- | --- |
| `p = 6 = K + 2` | `∝ (−h)^2 = 0` — nothing survives, the two maps coincide |
| `p = 4 = K` | `= 4! = 24` — a nonzero constant leaks into the outermost row |

So `3.0` is not noise: at `p = K` the highest-order term the recursion takes is a
constant that does not switch off on the feasible side, and `:quasi` drops it
while `:reduced` keeps it. **The minimum safe power is `p = K + 1`**; `level + 2`
is that bound with one derivative to spare. `level + 1` is untested here and
worth a run — it should close the leak while keeping the penalty less flat than
`+ 2`.

### Why the violation shrinks: the flatness/accuracy trade

Stationarity at the prioritized level balances the objective gradient against the
penalty gradient, `p · v^(p−1) · ‖∇h‖ ≈ ‖∇J‖`, so the equilibrium violation is

```
v ≈ (‖∇J‖ / (p ‖∇h‖))^(1/(p−1))
```

Calibrating the constant from the `p = 6` measurement (`v = 0.0910` ⇒
`c = 6 · 0.0910^5 ≈ 3.7e-5`) predicts `v = (c/4)^(1/3) = 0.021` at `p = 4`,
against a measured `0.0175`. The prediction tracks the data across the whole
table, which is the point: **the exponent, not the solver tolerance, sets the
feasibility floor.** A higher power makes the constraint invisible to Newton near
its own boundary, and no tolerance can recover what the encoding threw away.

That is the whole trade:

| | large `p` | small `p` |
| --- | --- | --- |
| smoothness at `h = 0` | `C^(p−1)`, safe for Newton | `C^(p−1)`, rough at `p ≤ 2` |
| higher-order leak in the `K`-level recursion | none once `p ≥ K + 1` | present at `p ≤ K` |
| equilibrium violation `v` | large (`c^(1/(p−1))`, `c ≪ 1`) | small |
| attainable `‖F‖` | floors higher | floors lower |

## Caveat before generalizing

`p = level + offset` is per-level, so a global offset change hits the shallow
prioritized levels hardest:

| site | prioritized level | p at `+2` | p at `+0` |
| --- | --- | --- | --- |
| `benchmark_problems.jl` (this study) | 4 | 6 | 4 |
| `runtests.jl` FD parity | 3 | 5 | 3 |
| `robotic_arm_core.jl` robot | 4 | 6 | 4 |
| `robotic_arm_core.jl` child | 2 | 4 | **2** ⚠ |
| `Intersection.jl` player 1 | 2 | 4 | **2** ⚠ |
| `Intersection.jl` player 2 | 3 | 5 | 3 |

At `p = 2` the penalty is only `C¹`: the Hessian jumps at the active boundary, so
the Newton linearization is discontinuous there. The two ⚠ sites should be
watched separately from the rest — if they get noisier at offset 0 while the
level-3/4 sites improve, that is the `C¹` effect, not a regression. `p ≥ 3` is
the guard, i.e. `max(level + offset, 3)`.

Also note that a global exponent change is not neutral for the *encoding*: the
relative weight between a level-2 and a level-4 prioritized constraint changes
with the offset, so multi-level prioritized problems are not simply "the same
problem, solved better."

## Bottom line

- On this benchmark, `(level)` dominates `(level + 2)` on every solver-facing
  metric: ~6× closer to `x*`, ~6× less infeasible, a lower `‖F‖` floor, and it
  recovers the correct basin on `nonlinear`/`zeros`.
- The `+ 2` is not arbitrary: at a prioritized level of index `K`, it is what
  keeps the penalty flat through all `K` derivatives the reduced recursion takes,
  which is why `:reduced` and `:quasi` agree exactly on the feasible side there
  and disagree by `O(1)` without it.
- Those two facts point opposite ways, and the choice is a modelling one. If the
  goal is accurate constrained solutions, lower is better. If the goal is that
  `:quasi` be a faithful truncation of `:reduced` near feasibility, `p ≥ K + 1`
  is required.
- `level + 1` sits at exactly that boundary and was not tested. It is the obvious
  next run.

## Reproducing

```bash
# edit the exponent in src/goop.jl:143 and COMPARISON_TOLERANCE in the test, then
julia --project=. test/quasi_vs_reduced_preference_inequality.jl
```

Each run takes ~60 s including precompilation. Raw logs for the four
configurations are not checked in; the tables above are transcribed from the
`GROUND-TRUTH primal distances` and `SUMMARY` blocks the script prints.
