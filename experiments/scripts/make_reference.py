"""Precompute per-problem reference data for the benchmark.

Produces, for one problem, an ``.npz`` with the true Pareto front (``pf``), per-objective ideal/nadir
(``z_ideal``/``z_nadir``, for normalization), das-dennis ``weights`` (R2 + many-objective PF sampling),
and ``n_obj``.

Everything is stored in minimization space. For an inverted problem (the objectives are maximized, the
runner stores y = -f) the reference front is the negated, scaled standard front: maximizing a DTLZ2/DTLZ4
objective converges at g = g_max, i.e. on the sphere of radius (1 + g_max), so the front is
-(1 + g_max) * (unit sphere). The scale is read off the problem by evaluating it at the all-zeros point.

Run by Snakemake (uses the global ``snakemake`` object).
"""

import numpy as np
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

problem_name = snakemake.wildcards.problem  # noqa: F821
problem_cfg = snakemake.params.problem_cfg or {}  # noqa: F821
metrics = snakemake.params.metrics  # noqa: F821
out_path = snakemake.output[0]  # noqa: F821

pymoo_name = problem_cfg.get("pymoo_name", problem_name)
invert = bool(problem_cfg.get("invert", False))
kwargs = {k: v for k, v in problem_cfg.items() if k in ("n_var", "n_obj") and v is not None}

problem = get_problem(pymoo_name, **kwargs)
n_obj = int(problem.n_obj)

# Reference directions: R2 weight vectors, and (for many objectives) the Pareto-front sampling directions.
n_partitions = int(metrics.get("ref_dirs_partitions_2d", 99)) if n_obj == 2 else int(  # noqa: PLR2004
    metrics.get("ref_dirs_partitions_md", 12)
)
weights = np.asarray(get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions), dtype=float)

# Standard (minimization) Pareto front.
pf = problem.pareto_front() if n_obj == 2 else problem.pareto_front(weights)  # noqa: PLR2004
pf = np.atleast_2d(np.asarray(pf, dtype=float))

if invert:
    # Maximizing the objectives converges at g = g_max; the front is (1 + g_max) * unit-sphere. Read the scale
    # by evaluating the problem at the all-zeros point (g = g_max, single non-zero objective = 1 + g_max).
    out = {}
    problem._evaluate(np.zeros((1, problem.n_var)), out)
    scale = float(np.max(out["F"]))
    pf = -scale * pf  # negated (maximization stored as y = -f) and scaled to the outer sphere

z_ideal = pf.min(axis=0)
z_nadir = pf.max(axis=0)

np.savez(out_path, pf=pf, z_ideal=z_ideal, z_nadir=z_nadir, weights=weights, n_obj=n_obj)
