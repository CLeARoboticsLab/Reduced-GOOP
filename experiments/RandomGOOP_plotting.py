"""
Figures and tables for the manuscript's §6 from one MonteCarlo_RandomGOOP run directory.

Reads the CSV files that MonteCarlo_RandomGOOP.export_csv writes (settings, sizes, setup, solves, stages)
and writes into <run_dir>/figures/, with <pair> = interior_point_perturbed (the reduced (5.2) and
the complete :perturbed system):
  timing_per_rho_<pair>.{png,pdf}      manuscript Fig 2: Newton steps (top row) and solve seconds (bottom
                                       row) per ρ stage, mean ± std after the 2.5 % trimming, one column per class
  fig3_<pair>.{png,pdf}                manuscript Fig 4: histograms of log10 ‖z_complete − z_reduced‖₂,
  fig3_<pair>_<class>.pdf              one panel per class; and one file per class
  table2_<pair>.tex, table2.tex        Table 2 (tab:kkt-compare-levels) in the manuscript's layout
  table3_setup_<pair>.tex, table3.tex  Table 3 (tab:setup-time)
  timing_per_rho_sec6.pdf, fig3_sec6.pdf
                                       copies of Figs 2 and 4 under the manuscript's file names (MANUSCRIPT_FIGURES)
  timing_table.tex, timing_table.md    build, compile and solve time of each system, per class and K

With --local-convergence, <run_dir> is instead a RandomGOOP_local_convergence run (its traces.csv), and the
script writes into <run_dir>/figures/:
  local_convergence_<mode>.{png,pdf}   manuscript Fig 3: ‖K_ρ‖₂ of the runs from the starts around y*_ρ, at
                                       ρ = 1.0 and 0.1, one row per class, against the Newton step from the
                                       start (every run averaged at every step, a converged run held at its last residual), with the quadratic
                                       reference from step 2; one figure per sampling mode
  fig2_sec6.pdf                        the copy of --mode's figure under the manuscript's file name

Table 2 cells: sizes from the manuscript's formulas (Props. thm:quadratic-growth-reduced-kkt and
thm:exponential-growth-complete-kkt); solve time as mean ± standard deviation of the accepted solves after
trimming the top and bottom 2.5 %. A K the class does not run is blank; a system that was not built
(complete_levels) is "--", and one with no accepted solve "Failed"; a cell with fewer than all solves
accepted gets a dagger, and the counts are listed in a comment below the table.

Usage (MonteCarlo_RandomGOOP.run and RandomGOOP_local_convergence.run call this at the end, through
plot_results):
    uv run --no-project --isolated --with matplotlib --with numpy python -I experiments/RandomGOOP_plotting.py \
        <run_dir> [--fig-levels quadratic=4,nonquadratic=3]
    uv run --no-project --isolated --with matplotlib --with numpy python -I experiments/RandomGOOP_plotting.py \
        --local-convergence <run_dir> [--mode whole_iterate]
"""

import argparse
import csv
import math
import pathlib
import shutil
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402

# Colors sampled from manuscript/trace.png and manuscript/exp4_kkt_equiv.pdf. Blue/crimson: OKLab ΔE×100
# 27 (normal vision), 18 (protan), 19 (deutan), 33 (tritan), and every panel names its series in a legend.
REDUCED_COLOR = "#4775a9"
COMPLETE_COLOR = "#d0283a"
HISTOGRAM_COLOR = "#74a1c7"
INK = "#1a1a1a"
CLASS_ORDER = ("quadratic", "nonquadratic")
REDUCED_SYSTEMS = ("reduced_interior_point",)  # the paper's (5.2)
FORMULATIONS = ("perturbed",)  # the ρ-perturbed complete system (6.1)
SHORT = {"reduced_interior_point": "interior_point"}
# The manuscript's tables and figures: R = the paper's (5.2), C = the ρ-perturbed complete system (6.1).
MAIN_PAIR = ("reduced_interior_point", "perturbed")
# The manuscript's file names for MAIN_PAIR's figures (manuscript/6_experiments.tex): Fig 2 is the per-ρ
# figure; Fig 4 keeps the file name of its earlier number. Fig 3 (fig2_sec6.pdf, its earlier number too)
# comes from a local-convergence run (--local-convergence, LOCAL_FIGURE).
MANUSCRIPT_FIGURES = {"timing_per_rho": "timing_per_rho_sec6.pdf", "fig3": "fig3_sec6.pdf"}
LOCAL_FIGURE = "fig2_sec6.pdf"

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
    }


def system_key(row):
    """'reduced_interior_point' or 'perturbed'."""
    return "reduced_" + row["formulation"] if row["system"] == "reduced" else row["formulation"]


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


def format_seconds(value, like=None):
    """Seconds to 2 decimals from 0.1 s up and 3 below; with `like`, to the decimals of `like` (a std takes its mean's)."""
    return f"{value:.2f}" if abs(value if like is None else like) >= 0.1 else f"{value:.3f}"


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
    """Table 2 solve-time cell: blank, "--", "Failed", or trimmed mean ± std of the accepted solves."""
    if K not in levels_run(run, class_name):
        return ""
    rows = solve_rows(run, class_name, K, system)
    accepted = [number(r["seconds"]) for r in rows if r["accepted"] == "true"]
    if not rows:
        notes.append(f"{class_name} K = {K} {system}: not built")
        return "--"
    if not accepted:
        notes.append(f"{class_name} K = {K} {system}: 0/{len(rows)} accepted")
        return "Failed"
    values = trimmed(accepted)
    cell = f"${format_seconds(values.mean())} \\pm {format_seconds(values.std(), like=values.mean())}$"
    if len(accepted) < len(rows):
        notes.append(f"{class_name} K = {K} {system}: {len(accepted)}/{len(rows)} accepted")
        cell = cell[:-1] + "^{\\dagger}$"
    return cell


def setup_cell(run, class_name, K, system):
    """Table 3 setup-time cell: one-time build + compile seconds of the system, "--" if not built."""
    if K not in levels_run(run, class_name):
        return ""
    build, compile_ = setup_value(run, class_name, K, system, "build_seconds"), setup_value(run, class_name, K, system, "compile_seconds")
    if not (np.isfinite(build) and np.isfinite(compile_)):
        return "--" if not solve_rows(run, class_name, K, system) else ""
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
        "% Blank: K not run for that class. --: system not built. Failed: no accepted solve.",
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


# ── Fig 3: local convergence around a solution (RandomGOOP_local_convergence.jl) ──

# Legend label of the dashed reference; the caption gives its formula, ‖K_ρ(y + Δy_ρ)‖₂ = ‖K_ρ(y)‖₂², which
# is too long for the legend row and would name only the reduced system.
QUADRATIC_REFERENCE = "quadratic reference"


# Figs 3 and 4 are drawn at their printed width, the manuscript's text width (svjour3 smallextended,
# 119 mm), so their text prints at 6.5–7.5 pt (your calls, 2026-10-29: Fig 3 in a 2 × 4 layout). A Fig 3
# panel is about 1 in wide, hence one y-label per row, column titles in place of x-labels, and one legend.
PRINT_WIDTH = 4.55  # inches; the tight bounding box adds ≈ 0.12 in, so the PDF is ≈ 4.68 in wide
PRINT_STYLE = {
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}


# The panels: the reduced and the complete system at ρ = 1 and 0.1, one row per class.
LOCAL_PANELS = (("reduced", 1.0), ("reduced", 0.1), ("complete", 1.0), ("complete", 0.1))
# The step of the mean the quadratic reference starts from (your call, 2026-11-08: iteration 2 in every panel).
REFERENCE_START = 2


def local_curves(rows, mode, class_name, system, rho):
    """log10 ‖K_ρ‖₂ at every Newton step from the start, one array per accepted run."""
    by_start = defaultdict(list)
    for r in rows:
        if r["mode"] != mode or r["class"] != class_name or r["system"] != system:
            continue
        if not math.isclose(number(r["rho"]), rho, rel_tol=1e-9):
            continue
        by_start[int(r["start"])].append((int(r["step"]), number(r["norm_K_ρ"])))
    return [np.log10(np.maximum([v for _, v in sorted(points)], 1e-300)) for points in by_start.values()]


def plot_local_panel(ax, curves, color, label):
    # Mean ± std at each step from the start over ALL the runs: a run stops once it converges and is held at its
    # last residual from then on, so the runs that finish early do not leave the average (which would otherwise
    # rise or stall at the steps where they drop out).
    longest = max(len(c) for c in curves)
    x = np.arange(longest)
    columns = [[c[min(s, len(c) - 1)] for c in curves] for s in x]
    assert all(len(v) == len(curves) for v in columns)
    mean = np.array([np.mean(v) for v in columns])
    std = np.array([np.std(v) for v in columns])
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, linewidth=0.4, edgecolor=color)
    ax.plot(x, mean, color=color, linewidth=1.1, marker="o", markersize=2.2, label=label)
    # Quadratic-rate reference r_{k+1} = r_k² from the mean at step REFERENCE_START: the order of magnitude doubles
    # each step, drawn to the last step; added as an artist, so the shared y axis clips it.
    start = REFERENCE_START
    if start < len(x) - 1 and mean[start] < 0:
        reference = mean[start] * 2.0 ** np.arange(len(x) - start)
        ax.add_artist(Line2D(x[start:], reference, color="#555555", linestyle="--", linewidth=0.8, label=QUADRATIC_REFERENCE))
    ax.set_xlim(0, x[-1])
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, color="#b0b0b0", linewidth=0.4, alpha=0.6)


def write_local_convergence(run_dir, manuscript_mode):
    """One figure per sampling mode in the run's traces.csv; `manuscript_mode`'s is also copied as LOCAL_FIGURE."""
    run_dir = pathlib.Path(run_dir)
    rows = read_rows(run_dir / "traces.csv")
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    levels = {r["class"]: int(r["K"]) for r in rows}
    classes = [c for c in CLASS_ORDER if c in levels]
    written = []
    for mode in sorted({r["mode"] for r in rows}):
        with plt.rc_context(PRINT_STYLE):
            fig, axes = plt.subplots(
                len(classes), len(LOCAL_PANELS), figsize=(PRINT_WIDTH, 0.6 + 1.15 * len(classes)), sharey=True, squeeze=False, layout="constrained"
            )
            for row, class_name in enumerate(classes):
                for col, (system, rho) in enumerate(LOCAL_PANELS):
                    ax = axes[row][col]
                    is_reduced = system == "reduced"
                    if row == 0:
                        ax.set_title(f"{'Reduced' if is_reduced else 'Complete'}, ρ = {rho:g}")
                    curves = local_curves(rows, mode, class_name, system, rho)
                    if not curves:
                        ax.text(0.5, 0.5, f"no {system} runs at ρ = {rho:g}", ha="center", va="center", transform=ax.transAxes)
                        continue
                    plot_local_panel(ax, curves, REDUCED_COLOR if is_reduced else COMPLETE_COLOR, "Reduced KKT" if is_reduced else "Complete KKT")
                K = levels[class_name]
                axes[row][0].set_ylabel(f"{'Quadratic' if class_name == 'quadratic' else 'Nonquadratic'}, K = {K}\n" + r"$\log_{10}$ residual")
            axes[0][0].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            fig.supxlabel("Newton steps from the start", fontsize=7.5)
            handles = {}
            for ax in axes.flat:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    handles.setdefault(label, handle)
            order = [label for label in ("Reduced KKT", "Complete KKT", QUADRATIC_REFERENCE) if label in handles]
            fig.legend([handles[label] for label in order], order, loc="outside upper center", ncol=len(order), frameon=False)
            paths = [out_dir / f"local_convergence_{mode}.{ext}" for ext in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
        written += paths
        if mode == manuscript_mode:
            written.append(shutil.copyfile(out_dir / f"local_convergence_{mode}.pdf", out_dir / LOCAL_FIGURE))
    return written


# ── Fig 4: distance histograms (fig3_*, the file names of its earlier number) ──


def distances(run, class_name, K, formulation):
    """log10 ‖z_complete − z_reduced‖₂ of the accepted pairs: distance_fig3, the two iterates at the end of the
    same ρ stage (settings fig3_rho)."""
    values = [
        number(r["distance_fig3"])
        for r in run["solves"]
        if r["class"] == class_name
        and (K is None or int(r["K"]) == K)
        and r["system"] == "complete"
        and r["formulation"] == formulation
        and r.get("distance_fig3", "") != ""
    ]
    return np.log10(np.maximum(np.array([v for v in values if np.isfinite(v)]), 1e-300))


DISTANCE_LABEL = r"$\log_{10}\|z_{\mathrm{complete}} - z_{\mathrm{reduced}}\|_2$"


def plot_histogram(ax, values):
    ax.hist(values, bins=min(22, max(6, len(values) // 4)), color=HISTOGRAM_COLOR, edgecolor=INK, linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.grid(True, color="#b0b0b0", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)


def write_fig3(run, reduced, formulation, fig_levels, out_dir):
    classes = [c for c in CLASS_ORDER if fig_levels.get(c) in levels_run(run, c)]
    data = {c: distances(run, c, fig_levels[c], formulation) for c in classes}
    classes = [c for c in classes if len(data[c])]
    if not classes:
        return []
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(1, len(classes), figsize=(PRINT_WIDTH, 1.75), squeeze=False, layout="constrained")
        for ax, class_name in zip(axes[0], classes):
            plot_histogram(ax, data[class_name])
            ax.set_title(f"{'Quadratic' if class_name == 'quadratic' else 'Nonquadratic'}, K = {fig_levels[class_name]}")
        axes[0][0].set_ylabel("Count")
        fig.supxlabel(DISTANCE_LABEL, fontsize=7.5)
        paths = [out_dir / f"fig3_{SHORT[reduced]}_{formulation}.{ext}" for ext in ("png", "pdf")]
        for path in paths:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        # One panel per class, at half the text width.
        for class_name in classes:
            fig, ax = plt.subplots(figsize=(PRINT_WIDTH / 2, 1.75), layout="constrained")
            plot_histogram(ax, data[class_name])
            ax.set_xlabel(DISTANCE_LABEL)
            ax.set_ylabel("Count")
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
                        # Runs before 2026-11-02 record only residual evaluations: Newton steps plus one per stage.
                        "iterations": float(np.median([number(r["newton_steps"]) if "newton_steps" in r else number(r["iterations"]) - 1 for r in solves if r["accepted"] == "true"]))
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
        "| class | K | system | build (s) | compile (s) | solve, trimmed mean ± std (s) | solve median (s) | Newton steps (median) | accepted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tex = [
        "% Generated by experiments/RandomGOOP_plotting.py: one-time build and compile, then the solve time of the",
        "% accepted solves (trimmed 2.5 % each side; compilation excluded).",
        "\\begin{tabular}{l c l r r c r r c}",
        "\\toprule",
        "Class & $K$ & System & Build (s) & Compile (s) & Solve (s) & Median (s) & Newton steps & Accepted \\\\",
        "\\midrule",
    ]
    for r in rows:
        solve = "—" if not np.isfinite(r["mean"]) else f"{format_seconds(r['mean'])} ± {format_seconds(r['std'], like=r['mean'])}"
        md.append(
            f"| {r['class']} | {r['K']} | {r['system']} | {show(r['build'])} | {show(r['compile'])} | {solve} | "
            f"{show(r['median'])} | {'—' if not np.isfinite(r['iterations']) else int(r['iterations'])} | {r['accepted']} |"
        )
        solve_tex = "--" if not np.isfinite(r["mean"]) else f"${format_seconds(r['mean'])} \\pm {format_seconds(r['std'], like=r['mean'])}$"
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
    """Seconds and Newton steps per ρ stage, one value per accepted solve. The stages a solve runs after its
    ‖K₀‖ < epsilon_0 stop take no step and are left out."""
    accepted = {
        int(r["trial"])
        for r in run["solves"]
        if r["class"] == class_name and int(r["K"]) == K and system_key(r) == system and r["accepted"] == "true"
    }
    by_trial = defaultdict(list)
    for r in run["stages"]:
        if r["class"] != class_name or int(r["K"]) != K or system_key(r) != system or int(r["trial"]) not in accepted:
            continue
        by_trial[int(r["trial"])].append(r)
    per_trial = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for trial, rows in by_trial.items():
        rows.sort(key=lambda r: int(r["stage"]))
        # Runs before 2026-11-02 record only a stage's residual evaluations, one more than its Newton steps.
        steps = [int(r["newton_steps"]) if "newton_steps" in r else int(r["iterations"]) - 1 for r in rows]
        last = max((i for i, s in enumerate(steps) if s > 0), default=-1)
        for r, s in zip(rows[: last + 1], steps):
            cell = per_trial[trial][number(r["rho"])]
            cell[0] += number(r["seconds"])
            cell[1] += s
    seconds, steps = defaultdict(list), defaultdict(list)
    for by_rho in per_trial.values():
        for rho, (s, n) in by_rho.items():
            seconds[rho].append(s)
            steps[rho].append(n)
    rhos = sorted(seconds, reverse=True)
    return rhos, [seconds[r] for r in rhos], [steps[r] for r in rhos]


def rho_label(rho):
    exponent = round(math.log10(rho))
    # \mathdefault sets 10⁻ᵏ in the tick font, as matplotlib's own log ticks do, not in the serif math font.
    return "1" if exponent == 0 else f"$\\mathdefault{{10^{{{exponent}}}}}$"


def write_timing_per_rho(run, reduced, formulation, fig_levels, out_dir):
    # Drawn at the printed width like Fig 3: Newton steps (top row) and seconds (bottom row), one column per
    # class, each row on one y axis; mean ± std curves in Fig 3's style (your calls, 2026-11-01).
    classes = [c for c in CLASS_ORDER if fig_levels.get(c) in levels_run(run, c)]
    if not classes:
        return []
    profiles = {c: {s: stage_profile(run, c, fig_levels[c], s) for s in (reduced, formulation)} for c in classes}
    all_rhos = sorted({rho for by_system in profiles.values() for rhos, _, _ in by_system.values() for rho in rhos if rho > 0}, reverse=True)
    positions = {rho: i for i, rho in enumerate(all_rhos)}
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(
            2, len(classes), figsize=(PRINT_WIDTH, 0.5 + 1.35 * 2), sharex=True, sharey="row", squeeze=False, layout="constrained"
        )
        for col, class_name in enumerate(classes):
            for row in range(2):
                ax = axes[row][col]
                for system, (rhos, seconds, steps) in profiles[class_name].items():
                    if not rhos:
                        continue
                    is_reduced = system.startswith("reduced")
                    color = REDUCED_COLOR if is_reduced else COMPLETE_COLOR
                    x = [positions[r] for r in rhos]
                    # Mean ± std after Table 2's trimming; on the log axis, a nonpositive lower edge is clipped.
                    values = [trimmed(v) for v in (steps if row == 0 else seconds)]
                    mean, std = np.array([np.mean(v) for v in values]), np.array([np.std(v) for v in values])
                    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, linewidth=0.4, edgecolor=color)
                    ax.plot(x, mean, color=color, linewidth=1.1, marker="o", markersize=2.2, label="Reduced KKT" if is_reduced else "Complete KKT")
                # Every ρ gets a tick, every fourth a label, so the 10⁻ᵏ labels don't collide.
                ax.set_xticks(list(positions.values()), [rho_label(r) if i % 4 == 0 else "" for (i, r) in enumerate(positions)])
                ax.set_yscale("linear" if row == 0 else "log")
                if row == 0:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                ax.grid(True, color="#b0b0b0", linewidth=0.4, alpha=0.6)
            axes[0][col].set_title(f"{'Quadratic' if class_name == 'quadratic' else 'Nonquadratic'}, K = {fig_levels[class_name]}")
        axes[0][0].set_ylabel("Newton steps per stage")
        axes[1][0].set_ylabel("seconds per stage")
        # The rows' tick labels differ in width (8 against 10⁻²); align the two labels on one vertical line.
        fig.align_ylabels(axes[:, 0])
        fig.supxlabel("ρ stage", fontsize=7.5)
        handles = {}
        for ax in axes.flat:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, handle)
        order = [label for label in ("Reduced KKT", "Complete KKT") if label in handles]
        fig.legend([handles[label] for label in order], order, loc="outside upper center", ncol=len(order), frameon=False)
        paths = [out_dir / f"timing_per_rho_{SHORT[reduced]}_{formulation}.{ext}" for ext in ("png", "pdf")]
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
    parser.add_argument("--fig-levels", default="quadratic=4,nonquadratic=3", help="K of Fig 2 (per ρ stage) and Fig 4, per class")
    parser.add_argument("--local-convergence", action="store_true", help="run_dir is a RandomGOOP_local_convergence run")
    parser.add_argument("--mode", default="whole_iterate", help="--local-convergence: the sampling mode copied as the manuscript's Fig 3")
    args = parser.parse_args()
    if args.local_convergence:
        for path in write_local_convergence(args.run_dir, args.mode):
            print(path)
        return
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
            written += write_fig3(run, reduced, formulation, fig_levels, out_dir)
            written += write_timing_per_rho(run, reduced, formulation, fig_levels, out_dir)
            if (reduced, formulation) == MAIN_PAIR:
                for prefix, name in MANUSCRIPT_FIGURES.items():
                    source = out_dir / f"{prefix}_{SHORT[reduced]}_{formulation}.pdf"
                    if source.exists():
                        written.append(shutil.copyfile(source, out_dir / name))
    written += write_timing_tables(run, out_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
