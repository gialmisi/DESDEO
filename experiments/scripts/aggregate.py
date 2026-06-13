"""Collect all per-run metrics and metadata into a single tidy CSV summary.

Metric columns are taken verbatim from each evaluate JSON (so adding indicators needs no change here);
run metadata (runtime, reported evaluations, final size) is joined in by (algorithm, problem, seed).

Run by Snakemake (uses the global ``snakemake`` object).
"""

import json

import polars as pl

# Index run metadata (runtime, reported evaluations, ...) by (algorithm, problem, seed).
meta_by_key = {}
for meta_path in snakemake.input.meta:  # noqa: F821
    with open(meta_path) as fh:  # noqa: PTH123
        d = json.load(fh)
    meta_by_key[(d["algorithm"], d["problem"], d["seed"])] = d

rows = []
for metric_path in snakemake.input.metrics:  # noqa: F821
    with open(metric_path) as fh:  # noqa: PTH123
        d = json.load(fh)
    meta = meta_by_key.get((d["algorithm"], d["problem"], d["seed"]), {})
    rows.append(
        {
            **d,  # all indicator columns + algorithm/problem/seed/n_nondominated
            "runtime_s": meta.get("runtime_s"),
            "n_eval_reported": meta.get("n_eval_reported"),
            "n_final": meta.get("n_final"),
        }
    )

df = pl.DataFrame(rows).sort(["problem", "algorithm", "seed"])
df.write_csv(snakemake.output[0])  # noqa: F821

with pl.Config(tbl_rows=-1, tbl_cols=-1):
    print(df)
