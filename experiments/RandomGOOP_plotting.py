"""
Figures and tables for the manuscript's §6.1–6.2 from one MonteCarlo_RandomGOOP run directory.

Reads the CSV files that MonteCarlo_RandomGOOP.export_csv writes (settings, sizes, setup, solves, stages,
traces) and writes into <run_dir>/figures/:
  fig2_<formulation>.{png,pdf}         convergence traces, styled like manuscript/trace.png: one row per
                                       objective class; reduced panels at ρ = 1.0 and 0.1, then the complete
                                       panels (:perturbed and :aggregate at ρ = 1.0 and 0.1, :exact at ρ = 0);
                                       traces aligned at the iteration where each reaches its floor
  fig3_<formulation>.{png,pdf}         histograms of log10 ‖z_complete − z_reduced‖₂, one panel per class,
  fig3_<formulation>_<class>.pdf       styled like manuscript/exp4_kkt_equiv.pdf; and one file per class
  table2_<formulation>.tex             Table 2 (tab:kkt-compare-levels) in the manuscript's layout
  timing_table.tex, timing_table.md    build, compile and solve time of the reduced system and each
                                       complete formulation, per class and K
  timing_per_rho.{png,pdf}             solve seconds and Newton iterations per ρ stage, per class

Table 2 cells: sizes from the manuscript's formulas (Props. thm:quadratic-growth-reduced-kkt and
thm:exponential-growth-complete-kkt); solve time as mean ± standard deviation of the accepted solves after
trimming the top and bottom 2.5 %. A K the class does not run is blank; a system that was not built, or
that has no accepted solve, is "Failed"; a cell with fewer than all solves accepted gets a dagger, and
the counts are listed in a comment below the table.

Usage (MonteCarlo_RandomGOOP.run calls this at the end, through plot_results):
    uv run --no-project --isolated --with matplotlib --with numpy python -I experiments/RandomGOOP_plotting.py \
        <run_dir> [--fig-levels quadratic=4,nonquadratic=3] [--all-solves]
"""

import argparse
import csv
import math
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

# Colors sampled from manuscript/trace.png and manuscript/exp4_kkt_equiv.pdf. Blue/crimson: OKLab ΔE×100
# 27 (normal vision), 18 (protan), 19 (deutan), 33 (tritan), and every panel names its series in a legend.
REDUCED_COLOR = "#4775a9"
COMPLETE_COLOR = "#d0283a"
HISTOGRAM_COLOR = "#74a1c7"
INK = "#1a1a1a"
CLASS_ORDER = ("quadratic", "nonquadratic")
REDUCED_SYSTEMS = ("reduced_scholtes", "reduced_interior_point")  # :scholtes, and (5.2) as :interior_point
FORMULATIONS = ("perturbed", "exact", "aggregate")  # complete formulations
SHORT = {"reduced_scholtes": "scholtes", "reduced_interior_point": "interior_point"}
# Table 2, Fig 2 and Fig 3 of the manuscript: R = the paper's (5.2), C = the exact complete system (your call).
MAIN_PAIR = ("reduced_interior_point", "perturbed")

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.edgecolor": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fancybox": False,
        "legend.edgecolor": INK,
        "legend.framealpha": 1.0,
        "savefig.dpi": 200,
    }
)


# ── Reading the run ──────────────────────────────────────────────────────────


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def number(text):
    return float(text) if text not in ("", None) else math.nan


def read_run(run_dir):
    run_dir = pathlib.Path(run_dir)
    settings = {r["name"]: r["value"] for r in read_rows(run_dir / "settings.csv")}
    return {
        "dir": run_dir,
        "settings": settings,
        "sizes": read_rows(run_dir / "sizes.csv"),
        "setup": read_rows(run_dir / "setup.csv"),
        "solves": read_rows(run_dir / "solves.csv"),
        "stages": read_rows(run_dir / "stages.csv"),
        "traces": read_rows(run_dir / "traces.csv"),
    }


def system_key(row):
    """'reduced_scholtes', 'reduced_interior_point', 'exact' or 'aggregate' (older runs: reduced rows unlabeled)."""
    return "reduced_" + (row["formulation"] or "scholtes") if row["system"] == "reduced" else row["formulation"]


def levels_run(run, class_name):
    """K values with reduced solves for this class (the run's num_levels)."""
    return sorted({int(r["K"]) for r in run["solves"] if r["class"] == class_name and r["system"] == "reduced"})


def solve_rows(run, class_name, K, system):
    return [r for r in run["solves"] if r["class"] == class_name and int(r["K"]) == K and system_key(r) == system]


# ── Statistics ───────────────────────────────────────────────────────────────


def trimmed(values, fraction=0.025):
    """Values without the top and bottom `fraction` (the manuscript's 2.5 % trimming)."""
    values = np.sort(np.asarray(values, dtype=float))
    k = int(math.floor(fraction * len(values)))
    return values[k : len(values) - k] if len(values) > 2 * k else values


def format_seconds(value):
    return f"{value:.2f}" if abs(value) >= 0.1 else f"{value:.3f}"


# ── Table 2 ──────────────────────────────────────────────────────────────────


def manuscript_sizes(K, settings):
    """System and variable sizes of both systems from the manuscript's formulas (all players)."""
    N, n = int(settings["num_players"]), int(settings["primal_dimension"])
    mE, mI = int(settings["equality_dimension"]), int(settings["inequality_dimension"])
    reduced_F, reduced_G = K * n + mE + K * mI, (K + 1) * mI
    complete_F, complete_G = 2 ** (K - 1) * (n + mE + K * mI), 2**K * mI
    return {
        "reduced_system": N * (reduced_F + reduced_G),
        "complete_system": N * (complete_F + complete_G),
        "reduced_variables": N * ((1 + K * (K - 1) // 2) * n + K * mE + K * (K + 1) // 2 * mI),
        "complete_variables": N * complete_F,
    }


def time_cell(run, class_name, K, system, notes):
    """Table 2 solve-time cell: blank, "Failed", or trimmed mean ± std of the accepted solves."""
    if K not in levels_run(run, class_name):
        return ""
    rows = solve_rows(run, class_name, K, system)
    accepted = [number(r["seconds"]) for r in rows if r["accepted"] == "true"]
    if not accepted:
        notes.append(f"{class_name} K = {K} {system}: " + ("not built" if not rows else f"0/{len(rows)} accepted"))
        return "Failed"
    values = trimmed(accepted)
    cell = f"${format_seconds(values.mean())} \\pm {format_seconds(values.std())}$"
    if len(accepted) < len(rows):
        notes.append(f"{class_name} K = {K} {system}: {len(accepted)}/{len(rows)} accepted")
        cell = cell[:-1] + "^{\\dagger}$"
    return cell


def setup_cell(run, class_name, K, system):
    """Table 2 setup-time cell: one-time build + compile seconds of the system, blank if not built."""
    if K not in levels_run(run, class_name):
        return ""
    build, compile_ = setup_value(run, class_name, K, system, "build_seconds"), setup_value(run, class_name, K, system, "compile_seconds")
    if not (np.isfinite(build) and np.isfinite(compile_)):
        return "Failed" if not solve_rows(run, class_name, K, system) else ""
    total = build + compile_
    return f"${total:.1f}$" if total < 10 else f"${total:.0f}$"


def write_table2(run, reduced, formulation, out_dir):
    settings = run["settings"]
    levels = sorted({K for c in CLASS_ORDER for K in levels_run(run, c)})
    notes = []
    lines = [
        "% Generated by experiments/RandomGOOP_plotting.py from " + str(run["dir"]),
        f"% Reduced formulation: :{SHORT[reduced]}; complete formulation: :{formulation}.",
        "% Solve time: mean ± std of the accepted solves after trimming the",
        "% top and bottom 2.5 %. Setup times are in Table 3 (write_table3).",
        "% Blank: K not run for that class. Failed: not built, or no accepted solve.",
        "\\begin{table}[ht]",
        "\\caption{Comparison of reduced and complete KKT systems across preference levels. \\textnormal{R} denotes "
        "the reduced KKT system, and \\textnormal{C} denotes the complete KKT system. }",
        "\\label{tab:kkt-compare-levels}",
        "\\centering",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{c c c c c c c c c}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\shortstack{System\\\\size}}",
        "& \\multicolumn{2}{c}{\\shortstack{Variable\\\\size}}",
        "& \\multicolumn{2}{c}{\\shortstack{Solve time (s)\\\\for Quad.\\ GOOP}}",
        "& \\multicolumn{2}{c}{\\shortstack{Solve time (s)\\\\for Nonquad.\\ GOOP}} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "Level ($K$) & R & C & R & C & R & C & R & C \\\\",
        "\\midrule",
    ]
    for K in levels:
        s = manuscript_sizes(K, settings)
        cells = [
            str(K),
            str(s["reduced_system"]),
            str(s["complete_system"]),
            str(s["reduced_variables"]),
            str(s["complete_variables"]),
        ]
        for class_name in CLASS_ORDER:
            cells.append(time_cell(run, class_name, K, reduced, notes))
            cells.append(time_cell(run, class_name, K, formulation, notes))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    if notes:
        lines.append("% Notes (\\dagger: not every solve accepted):")
        lines += [f"%   {note}" for note in notes]
    path = out_dir / f"table2_{SHORT[reduced]}_{formulation}.tex"
    path.write_text("\n".join(lines) + "\n")
    if (reduced, formulation) == MAIN_PAIR:  # the manuscript's Table 2
        (out_dir / "table2.tex").write_text("\n".join(lines) + "\n")
    return path


def write_table3(run, reduced, formulation, out_dir):
    """Table 3: the one-time setup time (build + compile, setup.csv) of R and C per class and K."""
    levels = sorted({K for c in CLASS_ORDER for K in levels_run(run, c)})
    lines = [
        "% Generated by experiments/RandomGOOP_plotting.py from " + str(run["dir"]),
        f"% Reduced formulation: :{SHORT[reduced]}; complete formulation: :{formulation}. Blank: K not run for that class.",
        "\\begin{table}[ht]",
        "\\caption{One-time setup time (s), i.e., construction and compilation, of the reduced KKT system (\\textnormal{R}) "
        "and the complete KKT system (\\textnormal{C}).}",
        "\\label{tab:setup-time}",
        "\\centering",
        "\\begin{tabular}{c c c c c}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Quad.\\ GOOP} & \\multicolumn{2}{c}{Nonquad.\\ GOOP} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Level ($K$) & R & C & R & C \\\\",
        "\\midrule",
    ]
    for K in levels:
        cells = [str(K)]
        for class_name in CLASS_ORDER:
            cells += [setup_cell(run, class_name, K, reduced), setup_cell(run, class_name, K, formulation)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path = out_dir / f"table3_setup_{SHORT[reduced]}_{formulation}.tex"
    path.write_text("\n".join(lines) + "\n")
    if (reduced, formulation) == MAIN_PAIR:
        (out_dir / "table3.tex").write_text("\n".join(lines) + "\n")
    return path


# ── Fig 2: convergence traces ────────────────────────────────────────────────


def trace_curves(run, class_name, K, system, rho, accepted_only):
    """log10 ‖K_ρ‖₂ per Newton iteration at stage ρ, one array per solve."""
    accepted = {
        int(r["trial"])
        for r in run["solves"]
        if r["class"] == class_name and int(r["K"]) == K and system_key(r) == system and r["accepted"] == "true"
    }
    by_trial = defaultdict(list)
    for r in run["traces"]:
        if r["class"] != class_name or int(r["K"]) != K or system_key(r) != system:
            continue
        if not math.isclose(number(r["rho"]), rho, rel_tol=1e-9, abs_tol=1e-15):
            continue
        if accepted_only and int(r["trial"]) not in accepted:
            continue
        by_trial[int(r["trial"])].append((int(r["iteration"]), number(r["raw"])))
    curves = [np.log10(np.maximum([v for _, v in sorted(points)], 1e-300)) for points in by_trial.values() if points]
    return curves or None


# Fig 2 aligns each trace at the iteration where it first comes within FLOOR_DECADES of its minimum (its
# floor), so the local convergence of every solve lines up; x runs over ALIGNED_ITERATIONS from there.
FLOOR_DECADES = 0.5
ALIGNED_ITERATIONS = np.arange(-7, 3)


def floor_iteration(curve):
    """Index of the trace's first iterate within FLOOR_DECADES of its minimum."""
    return int(np.argmax(curve <= curve.min() + FLOOR_DECADES))


def plot_trace_panel(ax, curves, color, label, rho, ylabel):
    # Mean ± std, at each offset from the floor iteration, over the solves that have an iterate there;
    # offsets where fewer than a tenth of the solves (and fewer than 3, or all if fewer) do are dropped.
    columns = [[c[floor_iteration(c) + o] for c in curves if 0 <= floor_iteration(c) + o < len(c)] for o in ALIGNED_ITERATIONS]
    keep = [len(v) >= min(len(curves), max(3, len(curves) // 10)) for v in columns]
    x = ALIGNED_ITERATIONS[keep]
    mean = np.array([np.mean(v) for v, k in zip(columns, keep) if k])
    std = np.array([np.std(v) for v, k in zip(columns, keep) if k])
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, linewidth=0.6, edgecolor=color)
    ax.plot(x, mean, color=color, linewidth=2.0, marker="o", markersize=4, label=label)
    # Quadratic-rate reference: from the mean at offset -3, the digits double each iteration
    # (log10 r_{k+1} = 2 log10 r_k), drawn while it stays above the panel's floor.
    start = int(np.argmax(x >= -3))
    if mean[start] < 0:
        reference = [mean[start]]
        while len(reference) < 4 and 2 * reference[-1] > mean.min() - 1:
            reference.append(2 * reference[-1])
        steps = x[start : start + len(reference)]
        if len(steps) >= 2:
            ax.plot(steps, reference[: len(steps)], color="#555555", linestyle="--", linewidth=1.2, label="digits double")
    ax.set_xlim(x[0], x[-1])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(f"iterations from convergence  (ρ={rho:.1f})" if rho >= 0.1 else f"iterations from convergence  (ρ={rho:g})", fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, color="#b0b0b0", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower left", fontsize=12, handlelength=1.6)


def write_fig2(run, reduced, formulation, fig_levels, accepted_only, out_dir):
    reduced_label = r"$\log_{10}(\|\mathcal{K}_\rho(y)\|_2)$"
    complete_label = r"$\log_{10}(\|\bar{\mathcal{K}}_\rho(\bar{y})\|_2)$"
    panels = [(reduced, 1.0), (reduced, 0.1)]
    panels += [(formulation, 1.0), (formulation, 0.1)] if formulation in ("perturbed", "aggregate") else [(formulation, 0.0)]
    classes = [c for c in CLASS_ORDER if fig_levels.get(c) in levels_run(run, c)]
    if not classes:
        print(f"Fig 2 ({SHORT[reduced]}, {formulation}): no class was run at the levels {fig_levels}; skipped")
        return []
    fig, axes = plt.subplots(len(classes), len(panels), figsize=(4.55 * len(panels), 3.2 * len(classes)), squeeze=False)
    for row, class_name in enumerate(classes):
        K = fig_levels[class_name]
        for col, (system, rho) in enumerate(panels):
            ax = axes[row][col]
            curves = trace_curves(run, class_name, K, system, rho, accepted_only)
            if curves is None:
                ax.text(0.5, 0.5, f"no {system} data at ρ = {rho:g}", ha="center", va="center", transform=ax.transAxes)
                continue
            is_reduced = system.startswith("reduced")
            first_of_kind = col == 0 or (not is_reduced and panels[col - 1][0].startswith("reduced"))
            plot_trace_panel(
                ax,
                curves,
                REDUCED_COLOR if is_reduced else COMPLETE_COLOR,
                "Reduced KKT" if is_reduced else "Complete KKT",
                rho,
                (reduced_label if is_reduced else complete_label) if first_of_kind else None,
            )
        constraint = run["settings"].get("nonquadratic_constraint_type", "nonlinear") if class_name == "nonquadratic" else "linear"
        label = ("Quad." if class_name == "quadratic" else "Nonquad.") + f" obj. with\n{constraint} const."
        axes[row][0].annotate(
            label,
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-78, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            rotation=90,
            fontsize=17,
        )
    fig.tight_layout(w_pad=1.2, h_pad=1.6)
    paths = [out_dir / f"fig2_{SHORT[reduced]}_{formulation}.{ext}" for ext in ("png", "pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


# ── Fig 3: distance histograms ───────────────────────────────────────────────


def distances(run, class_name, K, reduced, formulation):
    """log10 ‖z_complete − z_reduced‖₂ of the accepted pairs, against the :scholtes or the :interior_point z."""
    column = "distance" if reduced == "reduced_scholtes" else "distance_interior_point"
    values = [
        number(r[column])
        for r in run["solves"]
        if r["class"] == class_name
        and (K is None or int(r["K"]) == K)
        and r["system"] == "complete"
        and r["formulation"] == formulation
        and r.get(column, "") != ""
    ]
    return np.log10(np.maximum(np.array([v for v in values if np.isfinite(v)]), 1e-300))


def plot_histogram(ax, values):
    ax.hist(values, bins=min(22, max(6, len(values) // 4)), color=HISTOGRAM_COLOR, edgecolor=INK, linewidth=1.2)
    ax.set_xlabel(r"$\log_{10}(\|\,z_{\mathrm{complete}} - z_{\mathrm{reduced}}\|_2)$", fontsize=15)
    ax.set_ylabel("Count", fontsize=18)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color="#b0b0b0", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def write_fig3(run, reduced, formulation, fig_levels, out_dir):
    classes = [c for c in CLASS_ORDER if fig_levels.get(c) in levels_run(run, c)]
    data = {c: distances(run, c, fig_levels[c], reduced, formulation) for c in classes}
    classes = [c for c in classes if len(data[c])]
    if not classes:
        return []
    fig, axes = plt.subplots(1, len(classes), figsize=(5.6 * len(classes), 2.9), squeeze=False)
    for ax, class_name in zip(axes[0], classes):
        plot_histogram(ax, data[class_name])
        ax.set_title(f"{'Quadratic' if class_name == 'quadratic' else 'Nonquadratic'}, K = {fig_levels[class_name]}", fontsize=13)
    fig.tight_layout()
    paths = [out_dir / f"fig3_{SHORT[reduced]}_{formulation}.{ext}" for ext in ("png", "pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    for class_name in classes:
        fig, ax = plt.subplots(figsize=(5.6, 2.9))
        plot_histogram(ax, data[class_name])
        fig.tight_layout()
        path = out_dir / f"fig3_{SHORT[reduced]}_{formulation}_{class_name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


# ── Timing comparison: build, compile and solve ──────────────────────────────


def setup_value(run, class_name, K, system, column):
    for r in run["setup"]:
        if r["class"] == class_name and int(r["K"]) == K and system_key(r) == system:
            return number(r[column])
    if column == "build_seconds":  # runs without setup.csv: the build time is also in sizes.csv
        for r in run["sizes"]:
            if r["class"] == class_name and int(r["K"]) == K and system_key(r) == system:
                return number(r["construction_seconds"])
    return math.nan


def timing_rows(run):
    rows = []
    for class_name in CLASS_ORDER:
        for K in levels_run(run, class_name):
            for system in REDUCED_SYSTEMS + FORMULATIONS:
                solves = solve_rows(run, class_name, K, system)
                if not solves:
                    continue
                accepted = [number(r["seconds"]) for r in solves if r["accepted"] == "true"]
                values = trimmed(accepted) if accepted else np.array([])
                rows.append(
                    {
                        "class": class_name,
                        "K": K,
                        "system": system,
                        "build": setup_value(run, class_name, K, system, "build_seconds"),
                        "compile": setup_value(run, class_name, K, system, "compile_seconds"),
                        "first_solve": setup_value(run, class_name, K, system, "first_solve_seconds"),
                        "mean": values.mean() if len(values) else math.nan,
                        "std": values.std() if len(values) else math.nan,
                        "median": float(np.median(accepted)) if accepted else math.nan,
                        "iterations": float(np.median([number(r["iterations"]) for r in solves if r["accepted"] == "true"]))
                        if accepted
                        else math.nan,
                        "accepted": f"{len(accepted)}/{len(solves)}",
                    }
                )
    return rows


def write_timing_tables(run, out_dir):
    rows = timing_rows(run)
    show = lambda v: "—" if not np.isfinite(v) else format_seconds(v)  # noqa: E731
    md = [
        "| class | K | system | build (s) | compile (s) | solve, trimmed mean ± std (s) | solve median (s) | Newton iterations (median) | accepted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tex = [
        "% Generated by experiments/RandomGOOP_plotting.py: one-time build and compile, then the solve time of the",
        "% accepted solves (trimmed 2.5 % each side; compilation excluded).",
        "\\begin{tabular}{l c l r r c r r c}",
        "\\toprule",
        "Class & $K$ & System & Build (s) & Compile (s) & Solve (s) & Median (s) & Newton its. & Accepted \\\\",
        "\\midrule",
    ]
    for r in rows:
        solve = "—" if not np.isfinite(r["mean"]) else f"{format_seconds(r['mean'])} ± {format_seconds(r['std'])}"
        md.append(
            f"| {r['class']} | {r['K']} | {r['system']} | {show(r['build'])} | {show(r['compile'])} | {solve} | "
            f"{show(r['median'])} | {'—' if not np.isfinite(r['iterations']) else int(r['iterations'])} | {r['accepted']} |"
        )
        solve_tex = "--" if not np.isfinite(r["mean"]) else f"${format_seconds(r['mean'])} \\pm {format_seconds(r['std'])}$"
        tex.append(
            f"{r['class']} & {r['K']} & {r['system']} & {show(r['build']).replace('—', '--')} & "
            f"{show(r['compile']).replace('—', '--')} & {solve_tex} & {show(r['median']).replace('—', '--')} & "
            f"{'--' if not np.isfinite(r['iterations']) else int(r['iterations'])} & {r['accepted']} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "timing_table.md").write_text("\n".join(md) + "\n")
    (out_dir / "timing_table.tex").write_text("\n".join(tex) + "\n")
    return [out_dir / "timing_table.md", out_dir / "timing_table.tex"]


def stage_profile(run, class_name, K, system):
    """Mean seconds and Newton iterations per ρ stage over the accepted solves (all tries of a complete solve)."""
    accepted = {
        int(r["trial"])
        for r in run["solves"]
        if r["class"] == class_name and int(r["K"]) == K and system_key(r) == system and r["accepted"] == "true"
    }
    seconds, iterations = defaultdict(list), defaultdict(list)
    per_trial = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in run["stages"]:
        if r["class"] != class_name or int(r["K"]) != K or system_key(r) != system or int(r["trial"]) not in accepted:
            continue
        cell = per_trial[int(r["trial"])][number(r["rho"])]
        cell[0] += number(r["seconds"])
        cell[1] += number(r["iterations"])
    for by_rho in per_trial.values():
        for rho, (s, i) in by_rho.items():
            seconds[rho].append(s)
            iterations[rho].append(i)
    rhos = sorted(seconds, reverse=True)
    return rhos, [np.mean(seconds[r]) for r in rhos], [np.mean(iterations[r]) for r in rhos]


def rho_label(rho):
    if rho == 0:
        return "0"
    exponent = round(math.log10(rho))
    return "1" if exponent == 0 else f"$10^{{{exponent}}}$"


def write_timing_per_rho(run, fig_levels, out_dir):
    classes = [c for c in CLASS_ORDER if fig_levels.get(c) in levels_run(run, c)]
    if not classes:
        return []
    fig, axes = plt.subplots(len(classes), 2, figsize=(11.5, 3.3 * len(classes)), squeeze=False)
    styles = {
        "reduced_scholtes": dict(color=REDUCED_COLOR, marker="o", label="Reduced KKT, :scholtes"),
        "reduced_interior_point": dict(color=REDUCED_COLOR, marker="^", linestyle="--", label="Reduced KKT, (5.2) :interior_point"),
        "perturbed": dict(color=COMPLETE_COLOR, marker="v", linestyle="--", label="Complete KKT, :perturbed"),
        "aggregate": dict(color=COMPLETE_COLOR, marker="s", label="Complete KKT, :aggregate"),
        "exact": dict(color=COMPLETE_COLOR, marker="D", linestyle="none", markerfacecolor="white", label="Complete KKT, :exact (ρ = 0)"),
    }
    for row, class_name in enumerate(classes):
        K = fig_levels[class_name]
        profiles = {s: stage_profile(run, class_name, K, s) for s in REDUCED_SYSTEMS + FORMULATIONS}
        all_rhos = sorted({rho for rhos, _, _ in profiles.values() for rho in rhos if rho > 0}, reverse=True)
        positions = {rho: i for i, rho in enumerate(all_rhos)}
        positions[0.0] = len(all_rhos) + 0.5
        for col, (index, ylabel) in enumerate(((1, "Seconds per stage (mean)"), (2, "Newton iterations per stage (mean)"))):
            ax = axes[row][col]
            for system, (rhos, secs, its) in profiles.items():
                if not rhos:
                    continue
                values = secs if index == 1 else its
                ax.plot([positions[r] for r in rhos], values, linewidth=2.0, markersize=7, **styles[system])
            # Every other ρ is labeled, so the 10⁻ᵏ labels don't collide; ρ = 0 (:exact) always is.
            ax.set_xticks(list(positions.values()), [rho_label(r) if (i % 2 == 0 or r == 0) else "" for (i, r) in enumerate(positions)])
            ax.set_xlabel("ρ stage", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_yscale("log" if index == 1 else "linear")
            ax.grid(True, color="#b0b0b0", linewidth=0.6, alpha=0.6)
            ax.set_title(f"{'Quadratic' if class_name == 'quadratic' else 'Nonquadratic'}, K = {K}", fontsize=12)
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    paths = [out_dir / f"timing_per_rho.{ext}" for ext in ("png", "pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_levels(text):
    levels = {}
    for item in text.split(","):
        name, K = item.split("=")
        levels[name.strip()] = int(K)
    return levels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--fig-levels", default="quadratic=4,nonquadratic=3", help="K of Fig 2, Fig 3 and the per-ρ plot, per class")
    parser.add_argument("--all-solves", action="store_true", help="Fig 2: include rejected solves (default: accepted only)")
    args = parser.parse_args()
    run = read_run(args.run_dir)
    fig_levels = parse_levels(args.fig_levels)
    out_dir = run["dir"] / "figures"
    out_dir.mkdir(exist_ok=True)
    written = []
    present = {system_key(r) for r in run["solves"]}
    for reduced in [s for s in REDUCED_SYSTEMS if s in present]:
        for formulation in [f for f in FORMULATIONS if f in present]:
            written.append(write_table2(run, reduced, formulation, out_dir))
            written.append(write_table3(run, reduced, formulation, out_dir))
            written += write_fig2(run, reduced, formulation, fig_levels, not args.all_solves, out_dir)
            written += write_fig3(run, reduced, formulation, fig_levels, out_dir)
    written += write_timing_tables(run, out_dir)
    written += write_timing_per_rho(run, fig_levels, out_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
