# SMS-EMOA benchmark: DESDEO batched vs. pymoo

Reproducible comparison of the DESDEO batched SMS-EMOA variant against the canonical pymoo
SMS-EMOA, driven by [Snakemake](https://snakemake.readthedocs.io/en/stable/). Both algorithms solve
the same pymoo test problems (ZDT, DTLZ) under an equal function-evaluation budget.

See `smsemoa_vs_pymoo_notes.md` and `snakemake_reference.md` for background.

## Setup

Snakemake is installed with pip into the project's environment (it is intentionally not a
project dependency in `pyproject.toml`):

```bash
pip install snakemake          # or: uv pip install snakemake
```

Everything else (DESDEO, pymoo, moocore) comes from the project environment.

## Run

From this directory (so generated files land under `experiments/`):

```bash
cd experiments
snakemake -n                   # dry run: show the planned jobs
snakemake --cores 4            # run the full sweep
```

The final result is `summary.csv` (one row per algorithm × problem × seed, with HV / IGD / R2 and
runtime). Intermediate artifacts: `reference/`, `results/`, `metrics/` (all git-ignored).

## Configure

All parameters live in `config.yaml`: budget, population size, seeds, algorithms, problems, and
metric settings. To increase replications, extend `seeds`, e.g. `seeds: [1, 2, 3, 4, 5]`, and re-run.

## Pipeline

| Rule | Output | Purpose |
|---|---|---|
| `reference` | `reference/{problem}.npz` | true Pareto front, ideal/nadir, R2 weight vectors |
| `run_algo`  | `results/{algo}/{problem}/seed{seed}.parquet` (+ `.meta.json`) | final population objective vectors + runtime |
| `evaluate`  | `metrics/{algo}/{problem}/seed{seed}.json` | HV, IGD, R2 for that run |
| `aggregate` | `summary.csv` | tidy table over the whole sweep |

## Notes

- Budgets are exact in both frameworks (DESDEO after the initial-population FE double-count fix). With
  `budget=5000, mu=100`, the batched DESDEO runs ~49 generations and pymoo ~4900 steady-state steps.
- Metrics are computed on the non-dominated subset of each final population, normalized per problem
  with a common ideal/nadir from the true Pareto front, so they are comparable across runs.
- Hard problems (DTLZ1, DTLZ3) may not converge within 5k evaluations at 4 objectives; expect HV≈0 /
  large IGD there; not an error.
