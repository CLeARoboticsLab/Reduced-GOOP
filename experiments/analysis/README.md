# 3-level vs 4-level preference analysis (robotic arm)

Post-hoc analysis of the two-player robotic-arm GOOP (two-arm robot carrying a
pot vs a curious child), comparing the 3-level hierarchy
`[goal, load_balance, robot_ineq]` against the 4-level hierarchy with
min-control prepended at the outermost (lowest-priority) slot. All experiments
were run on 2026-07-14 against the converged instances in
`data/robotic_arm_open_loop/runs/Robotic_arm_single_robot_agent_{3,4}_levels/`
(planning horizon T = 10, `tol = 0.01`, `ϵ₀ = 0.1`, identical initial states
and default warmstart).

## Headline finding

The 3-level solve converges to an equilibrium in which the pot **swerves
toward the child** (path bows down-and-left toward the ground-bound child,
grazing the 2 m safety sphere into violation). The 4-level solve, from the
same warmstart, produces a straight pursuit line with a wide safety margin.
The swerve is a **spurious near-stationary branch that the 3-level system
admits and the 4-level system excludes** — it is not preferred by any robot
objective, and it cannot be removed by tightening the solver tolerance.

| | 3-level (swerve) | 4-level (direct) |
|---|---|---|
| goal value | 55.28 | 49.48 |
| control cost | 451.16 | 451.86 |
| min safety margin (dist² − 4) | **−0.046** (violated) | +6.09 |
| worst step-vs-goal angle | 30° | 0.6° |
| pot x-drift toward child | −0.48 | 0.00 |
| pot z profile | dips 1.5 → 1.24 → 1.68 | climbs 1.5 → 2.85 |
| child pot_approach | 239.6 | 283.4 |

The swerve is adverse to *every* robot preference (worse goal, worse safety,
equal load balance, equal control — speed saturates in both) and benefits only
the child's pot-approach objective.

## Why the 3-level system admits the swerve

At the swerving arrangement, the measurable restoring force from each robot
level is below the solver tolerance (`tol = 0.01`; the run passed at 0.0098):

- **load_balance** is identically 0: it penalizes the *height difference*
  between the grippers (max |z₁−z₂| = 0.012, inside the 0.1 allowance
  deadzone); a symmetric dip of both grippers toward the ground is invisible.
- **safety** penalty `max(−h,0)^(level+2)` has slope ≈ 2×10⁻⁵ at the grazing
  point — flat until deeply violated, so it cannot steer *away from* approach.
- **goal** pulls hard (≈1.5/step; ≈0.7 perpendicular at 30° misalignment), but
  with the speed cap saturated the along-track component is absorbed
  legitimately, and the turning component is balanced by the policy-multiplier
  terms ψᵀ∂(inner conditions)/∂u. Direct measurement
  (`analyze_policy_duals.jl`) shows this balancing is *not* a multiplier
  pathology: ψ is moderate at the swerve (max |ψ| ≈ 31) and comparable to the
  direct-basin point (max |ψ| ≈ 10), and zeroing ψ blows up the residual
  similarly at both points (‖F‖∞ ≈ 6). The inner penalty levels are flat in
  value/gradient (~10⁻⁴) but their *curvature* enters the policy-constraint
  Jacobian at O(1), giving ordinary-sized ψ legitimate directions to cancel
  the outer pull. The swerve is therefore a bona fide near-stationary
  arrangement of the reduced 3-level system — consistent with the tight-tol
  finding — not a tolerance trick hidden in large multipliers.

The child's chase gradient is sharp, so the joint Newton iterates get dragged
into the mutually consistent interception arrangement. This is **not** a
proximity effect (`warmstart_distances.jl`): the default warmstart is *farther*
from the swerve (‖Δ‖ = 10.8) than from the direct equilibrium (6.6), and its
robot block nearly coincides with the 4-level solution's robot block
(‖Δ‖ = 0.15). The 3-level flow drags the robot 8.9 to the swerve past an
admissible direct point 3.2 away — the basin is shaped by the flat inner-level
valley in the Newton map, not by Euclidean distance.

## Experiments (scripts in this directory)

1. **`cross_warmstart.jl`** — solve the 3-level problem warmstarted at the
   4-level solution. Result: stays in the direct basin (goal 50.24, safety
   +4.38; moved 3.1 from warmstart vs 6.5 to the swerving equilibrium). The
   direct trajectory **is admissible for the 3-level system**; the default
   warmstart's basin simply never finds it. The residual drift (speed
   overshoot 0.104 → 0.053) is the penalty-exponent mismatch (see below).
2. **`cross_warmstart_reverse.jl`** — solve the 4-level problem warmstarted at
   the swerving 3-level solution. Result: **escapes** (102 iters, moved 7.0 of
   the 8.8 inter-equilibrium gap): z-dip eliminated, safety −0.046 → +4.44,
   goal 55.28 → 51.11. The swerving arrangement is **not a solution of the
   4-level system**. The escape is not driven by control cost (451.16 → 452.10,
   slightly up); it is the added optimality conditions.
3. **`tight_tol_3level.jl`** — 3-level solve from the default warmstart with
   tol = 1e-3 / 1e-4. Result: `:failed` after 2000 iters, stalled at
   kkt_error 1.57×10⁻³ at a point that swerves *more* (x-drift −0.75, angle
   35°, goal 56.17, safety still violated). Tolerance strictness **entrenches**
   the swerve; it does not remove it. The deficiency is structural.
4. **`compare_preference_levels.jl`** — side-by-side level values (in the
   solver's own penalty form), feasibility margins, and path-shape metrics for
   the two saved runs.
5. **`check_constraints.jl`** — per-timestep raw inequality residuals for any
   saved run (pass the run id as ARGS[1]).
6. **`analyze_policy_duals.jl`** — decomposes saved solution vectors into
   named blocks via `kkt.z_symbolic`, prints the policy duals ψ, and
   re-evaluates the KKT residual with ψ zeroed at both the swerving and
   direct-basin 3-level points (requires `results/cross_warmstart_solution.jld2`).
7. **`warmstart_distances.jl`** — primal distances from the default warmstart
   to the three saved equilibria. Result: the warmstart is closer to the
   direct equilibrium than to the swerve (6.6 vs 10.8; robot block 3.2 vs 8.9,
   and only 0.15 from the 4-level robot block) — the swerve basin is a Newton-
   flow phenomenon, not warmstart proximity.

Converged solutions from (1)–(3) are stored in `results/`.

## Conclusion

The fourth (lowest-priority) preference level is needed not to trade off
control effort — speed saturates either way — but because its stationarity
equations couple the control gradient through the policy multipliers of all
inner levels, restoring rank along exactly the directions where the three
inner levels are numerically silent. This renders the unsafe interception
equilibrium inadmissible, which neither the 3-level hierarchy nor any solver
tolerance setting achieves.

## Caveats

- **Penalty exponent artifact**: the KKT generator uses exponent `level + 2`
  indexed from the outermost level, so adding a level re-indexes the innermost
  constraint from `violation⁵` to `violation⁶` (flatter near the boundary).
  This fully accounts for the 4-level run's slightly larger speed overshoot
  (control 451.86 vs 451.16 = 450 + Σ overshoot, exactly). For clean level-count
  comparisons, index the exponent from the innermost level instead.
- With no plain `inequality_constraints` in this experiment, the ϵ
  complementarity relaxation plays **no role** in these residuals; small
  constraint violations come from penalty flatness + `tol`.
- Single instance, default initial states; a perturbed-initial-state sweep
  would establish whether the exclusion is systematic.
- `cross_warmstart*.jl` and `tight_tol_3level.jl` require
  `experiments/Robotic_arm.jl` to be in its **3-level** configuration
  (`control_objective` commented out of `goop_preferences`).
