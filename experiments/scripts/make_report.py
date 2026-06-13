"""Render the full benchmark results to a landscape PDF (experiments/results_all.pdf).

Page 1: win-count summary (by budget, by indicator) + a factual methods/parameters table.
Pages 2..: one per indicator, three budget sub-tables (5k/10k/20k), all problems x methods,
best per problem in bold/green. Run from the experiments directory:  python scripts/make_report.py
"""

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

BUDGETS = [5000, 10000, 20000]
METH = [
    ("pymoo_lhs", "pymoo"),
    ("desdeo_greedy", "SMS-grdy"),
    ("desdeo_batched", "SMS-bat"),
    ("desdeo_nsga3", "NSGA-III"),
    ("desdeo_rvea", "RVEA"),
]
MLAB = [m for _, m in METH]
PROBS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "dtlz1", "dtlz2", "dtlz3", "dtlz4", "inv_dtlz2", "inv_dtlz4"]
# (key, full name, short code, direction)
IND = [
    ("hv", "HV", "HV", "hi"),
    ("igd", "IGD", "IGD", "lo"),
    ("igd_plus", "IGD+", "IGD+", "lo"),
    ("gd", "GD", "GD", "lo"),
    ("delta_p", "Δp  (averaged Hausdorff)", "Δp", "lo"),
    ("hausdorff", "Hausdorff distance", "Hdf", "lo"),
    ("coverage_error", "Coverage error", "Cov", "lo"),
    ("eps_add", "ε+  (additive epsilon)", "ε+", "lo"),
    ("r2", "R2", "R2", "hi"),
]

data = {b: pl.read_csv(f"summary_b{b}.csv") for b in BUDGETS}


def med(b, a, p, m):
    v = data[b].filter((pl.col("algorithm") == a) & (pl.col("problem") == p)).select(m).to_numpy().ravel()
    return float(np.nanmedian(v)) if len(v) and not np.all(np.isnan(v)) else np.nan


def winners(vals, direction):
    arr = np.array(vals, dtype=float)
    if np.all(np.isnan(arr)) or np.ptp(arr[~np.isnan(arr)]) < 1e-9:
        return set()
    best = np.nanmax(arr) if direction == "hi" else np.nanmin(arr)
    return {j for j, v in enumerate(arr) if not np.isnan(v) and abs(v - best) < 1e-9}


win_bud = {m: {b: 0 for b in BUDGETS} for m in MLAB}
win_ind = {m: {i: 0 for i, _, _, _ in IND} for m in MLAB}
for b in BUDGETS:
    for key, _, _, d in IND:
        for p in PROBS:
            for j in winners([med(b, a, p, key) for a, _ in METH], d):
                win_bud[MLAB[j]][b] += 1
                win_ind[MLAB[j]][key] += 1


def draw_num(ax, matrix, col_labels, row_labels, winset, title, fs=8.5, intfmt=False):
    ax.axis("off")
    nR, nC = len(row_labels), len(col_labels)
    ax.set_xlim(0, nC)
    ax.set_ylim(0, nR + 1)
    ax.invert_yaxis()
    ax.text(nC / 2, -0.25, title, ha="center", va="bottom", fontsize=fs + 2.5, fontweight="bold")
    for j, c in enumerate(col_labels):
        ax.text(j + 0.5, 0.5, c, ha="center", va="center", fontsize=fs, fontweight="bold")
    for i, rl in enumerate(row_labels):
        y = i + 1
        ax.text(-0.06, y + 0.5, rl, ha="right", va="center", fontsize=fs, fontweight="bold")
        for j in range(nC):
            win = j in winset[i]
            v = matrix[i][j]
            if win:
                ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor="#cfe8cf", edgecolor="white", lw=1))
            txt = "—" if (isinstance(v, float) and np.isnan(v)) else (f"{int(v)}" if intfmt else f"{v:.3f}")
            ax.text(
                j + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if win else "normal", color="#0a5d0a" if win else "0.15",
            )
    for k in range(nR + 2):
        ax.plot([0, nC], [k, k], color="0.85", lw=0.6)
    ax.plot([0, nC], [1, 1], color="0.4", lw=1.1)


MCOLS = ["Framework", "Selection / survival", "Scheme", "μ (pop)", "SBX  η, p", "PM  η, p", "Init"]
MROWS = {
    "pymoo": ["pymoo 0.6.1.6", "greedy least-HV contribution\n(ref point 1+ε, ε=10)", "(μ+μ)", "100", "15,  0.9", "20,  1/n", "LHS"],
    "SMS-grdy": ["DESDEO", "greedy least-HV, pure HV\n(ref = nadir+1; use_dom_pts=F)", "(μ+μ)", "100", "15,  0.9", "20,  1/n", "LHS"],
    "SMS-bat": ["DESDEO", "batched least-HV + dominating-\npoints (greedy_reduction=F)", "(μ+μ)", "100", "15,  0.9", "20,  1/n", "LHS"],
    "NSGA-III": ["DESDEO", "reference vectors\n(das-dennis, planar)", "(μ+μ)", "100 / 84*", "30,  0.5", "20,  1/n", "LHS"],
    "RVEA": ["DESDEO", "ref vectors (spherical) + APD\n(α=2, FE-adaptive, freq=100)", "(μ+μ)", "100 / 84*", "30,  0.5", "20,  1/n", "LHS"],
}

with PdfPages("results_all.pdf") as pdf:
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("SMS-EMOA benchmark — full results", fontsize=20, fontweight="bold", y=0.975)
    fig.text(0.5, 0.935, "5 methods · 11 problems · 9 indicators · 3 budgets · median of 11 seeds.  "
             "Best per cell in bold/green (ties shared; all-equal rows score no win).", ha="center", fontsize=9.5, color="0.3")
    axA = fig.add_axes([0.055, 0.60, 0.36, 0.26])
    mat = [[win_bud[m][b] for b in BUDGETS] + [sum(win_bud[m].values())] for m in MLAB]
    bi = int(np.argmax([r[-1] for r in mat]))
    draw_num(axA, mat, ["5k", "10k", "20k", "TOTAL"], MLAB, [{3} if i == bi else set() for i in range(len(MLAB))],
             "Win count by budget   (max 99 / budget)", fs=9.5, intfmt=True)
    axB = fig.add_axes([0.49, 0.555, 0.48, 0.305])
    matI = [[win_ind[m][k] for k, _, _, _ in IND] + [sum(win_ind[m].values())] for m in MLAB]
    cols = [s for _, _, s, _ in IND] + ["TOT"]
    wsI = [set() for _ in MLAB]
    for jc in range(len(cols)):
        cb = max(matI[i][jc] for i in range(len(MLAB)))
        if cb > 0:
            for i in range(len(MLAB)):
                if matI[i][jc] == cb:
                    wsI[i].add(jc)
    draw_num(axB, matI, cols, MLAB, wsI, "Win count by indicator   (overall, max 33 each)", fs=9, intfmt=True)

    # ---- problems table (transposed: problems as columns, compact) ----
    fig.text(0.055, 0.50, "Problems", ha="left", va="bottom", fontsize=13, fontweight="bold", color="0.1")
    PINFO = {
        "zdt1": (30, 2, "min"), "zdt2": (30, 2, "min"), "zdt3": (30, 2, "min"), "zdt4": (10, 2, "min"),
        "zdt6": (10, 2, "min"), "dtlz1": (8, 4, "min"), "dtlz2": (13, 4, "min"), "dtlz3": (13, 4, "min"),
        "dtlz4": (13, 4, "min"), "inv_dtlz2": (13, 4, "max"), "inv_dtlz4": (13, 4, "max"),
    }
    axP = fig.add_axes([0.055, 0.40, 0.90, 0.085])
    axP.axis("off")
    ptbl = axP.table(
        cellText=[[str(PINFO[p][0]) for p in PROBS], [str(PINFO[p][1]) for p in PROBS], [PINFO[p][2] for p in PROBS]],
        colLabels=PROBS, rowLabels=["variables", "objectives", "sense"], cellLoc="center", rowLoc="center", loc="center",
    )
    ptbl.auto_set_font_size(False)
    ptbl.set_fontsize(8.5)
    ptbl.scale(1, 1.6)
    for (r, c), cell in ptbl.get_celld().items():
        cell.set_edgecolor("0.85")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eef3ee")
        if c == -1:
            cell.set_text_props(fontweight="bold")

    fig.text(0.055, 0.355, "Methods & parameters", ha="left", va="bottom", fontsize=13, fontweight="bold", color="0.1")
    axM = fig.add_axes([0.055, 0.12, 0.90, 0.225])
    axM.axis("off")
    tbl = axM.table(cellText=[MROWS[m] for m in MLAB], colLabels=MCOLS, rowLabels=MLAB,
                    cellLoc="center", rowLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eef3ee")
        if c == -1:
            cell.set_text_props(fontweight="bold")
    fig.text(0.055, 0.025,
             "All methods evaluate the same pymoo problem objects.  *NSGA-III/RVEA population = number of reference vectors:\n"
             "100 at 2 objectives, 84 at 4 objectives (simplex lattice).  Common to all runs: budgets 5 000 / 10 000 / 20 000 function\n"
             "evaluations; seeds 1-11; metrics on the non-dominated final population, normalized per problem by the true-front ideal/nadir.",
             ha="left", va="bottom", fontsize=7.8, color="0.35")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    for key, name, _, d in IND:
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f"{name}    ({'↑ higher = better' if d == 'hi' else '↓ lower = better'})",
                     fontsize=18, fontweight="bold", y=0.96)
        fig.text(0.5, 0.915, "median of 11 seeds · best per problem in bold/green · — = not available",
                 ha="center", fontsize=9.5, color="0.35")
        for k, b in enumerate(BUDGETS):
            ax = fig.add_axes([0.05 + k * 0.322, 0.08, 0.28, 0.78])
            mat = [[med(b, a, p, key) for a, _ in METH] for p in PROBS]
            draw_num(ax, mat, MLAB, PROBS, [winners(mat[i], d) for i in range(len(PROBS))], f"{b // 1000}k evals", fs=8.2)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
print("saved results_all.pdf")
