"""Summarize the multi-seed benchmark: per-(algorithm, problem) median + IQR, and a Mann-Whitney U test
of each DESDEO variant against the pymoo reference.

Standalone (not a Snakemake rule). Run after a sweep with an explicit summary file, e.g.
    python experiments/scripts/report_stats.py experiments/summary_b20000.csv
If no path is given, the newest summary_b*.csv (largest budget) is used.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parents[1]
if len(sys.argv) > 1:
    summary_path = Path(sys.argv[1])
else:
    candidates = sorted(HERE.glob("summary_b*.csv"), key=lambda p: int(p.stem.split("b")[-1]))
    summary_path = candidates[-1] if candidates else HERE / "summary.csv"
print(f"Reading {summary_path.name}")
df = pl.read_csv(summary_path)

PROBLEMS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "dtlz1", "dtlz2", "dtlz3", "dtlz4"]
REFERENCE = "pymoo_lhs"
DESDEO = [a for a in df["algorithm"].unique().to_list() if a != REFERENCE]
METRICS = [
    ("hv", "higher"),
    ("igd", "lower"),
    ("igd_plus", "lower"),
    ("gd", "lower"),
    ("delta_p", "lower"),
    ("hausdorff", "lower"),
    ("coverage_error", "lower"),
    ("eps_add", "lower"),
    ("r2", "higher"),
]

n_seeds = df.select(pl.col("seed").n_unique()).item()
print(f"Seeds per cell: {n_seeds}   reference: {REFERENCE}\n")


def vals(algo: str, problem: str, metric: str) -> np.ndarray:
    return (
        df.filter((pl.col("algorithm") == algo) & (pl.col("problem") == problem))
        .select(metric)
        .to_numpy()
        .ravel()
    )


for metric, direction in METRICS:
    better = "↑" if direction == "higher" else "↓"
    print(f"=== {metric.upper()} ({better} better) — median [IQR] ===")
    header = f"{'problem':<7} " + "".join(f"{a:<26}" for a in [REFERENCE, *DESDEO])
    print(header)
    for p in PROBLEMS:
        cells = []
        ref = vals(REFERENCE, p, metric)
        for a in [REFERENCE, *DESDEO]:
            v = vals(a, p, metric)
            med = np.median(v)
            iqr = np.subtract(*np.percentile(v, [75, 25]))
            cell = f"{med:.4f} [{iqr:.4f}]"
            if a != REFERENCE and len(v) == len(ref) and len(v) > 1:
                # Mann-Whitney U: is `a` better than the reference on this metric?
                alt = "greater" if direction == "higher" else "less"
                try:
                    _, pval = mannwhitneyu(v, ref, alternative=alt)
                    star = "*" if pval < 0.05 else " "  # noqa: PLR2004
                    cell += f" p={pval:.3f}{star}"
                except ValueError:
                    cell += " p=n/a"
            cells.append(f"{cell:<26}")
        print(f"{p:<7} " + "".join(cells))
    print()

# Overall win tally (median-based) of each DESDEO variant vs reference.
print("=== Median win tally vs reference (across 9 problems x 3 metrics) ===")
for a in DESDEO:
    wins = ties = losses = 0
    for p in PROBLEMS:
        for metric, direction in METRICS:
            ma, mr = np.median(vals(a, p, metric)), np.median(vals(REFERENCE, p, metric))
            if ma == mr:
                ties += 1
            elif (ma > mr) == (direction == "higher"):
                wins += 1
            else:
                losses += 1
    print(f"  {a:<16} wins {wins}, losses {losses}, ties {ties}")
