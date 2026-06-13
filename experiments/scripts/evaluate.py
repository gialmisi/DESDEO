"""Compute a suite of quality indicators for one run's final population vs. the problem's reference data.

Everything is handled in *minimization* space (lower = better). For inverted/maximization problems the runner
stores negated objectives (y = -f) and the reference front is likewise in that space, so this script is agnostic
to the optimization sense. Fronts are normalized per problem with a common ideal/nadir (from the reference front)
so metrics are comparable across algorithms, seeds, and budgets. Only the non-dominated subset of the final
population is scored.

Indicators (against the normalized reference front, unless noted):
- hv             : hypervolume (moocore), reference point ``hv_ref_component`` per objective. Higher better.
- igd            : inverted generational distance (mean of nearest-front distance over reference points). Lower.
- igd_plus       : dominance-compliant IGD+. Lower better.
- gd             : generational distance (mean nearest-reference distance over the front). Lower better.
- delta_p        : averaged Hausdorff distance Delta_p = max(GD_p, IGD_p), p=2 (Schuetze). Lower better.
- hausdorff      : standard Hausdorff distance = max(directed front->ref, directed ref->front). Lower better.
- coverage_error : max over reference points of nearest-front distance (worst-covered reference point). Lower.
- eps_add        : additive epsilon indicator I(front, reference). Lower better.
- r2             : R2 indicator (augmented Tchebycheff over das-dennis weights, z* = ideal). Higher better.

Run by Snakemake (uses the global ``snakemake`` object).
"""

import json

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

from desdeo.tools.indicators_binary import epsilon_indicator
from desdeo.tools.indicators_unary import distance_indicators, hv, igd_plus_indicator, r2_indicator
from desdeo.tools.non_dominated_sorting import non_dominated

metrics = snakemake.params.metrics  # noqa: F821
hv_ref = float(metrics.get("hv_ref_component", 1.1))
rho = float(metrics.get("r2_rho", 0.05))

front = pl.read_parquet(snakemake.input.front).to_numpy()  # noqa: F821
ref = np.load(snakemake.input.reference)  # noqa: F821
pf, z_ideal, z_nadir, weights = ref["pf"], ref["z_ideal"], ref["z_nadir"], ref["weights"]

# Avoid division by zero for degenerate objectives.
span = np.where((z_nadir - z_ideal) > 0, z_nadir - z_ideal, 1.0)


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - z_ideal) / span


# Non-dominated subset of the final population, normalized.
nd_front = front[non_dominated(front)]
front_n = normalize(nd_front)
pf_n = normalize(pf)
z_star = np.zeros(front_n.shape[1])

# HV: only points strictly inside the reference box contribute; if none, HV is 0.
inside = np.all(front_n < hv_ref, axis=1)
hv_val = hv(front_n[inside], hv_ref) if inside.sum() > 0 else 0.0

# Distance-based bundle (igd, igd_p, gd, gd_p, ahd) in one call.
di = distance_indicators(front_n, pf_n)

# Hausdorff distances from the pairwise distance matrix (rows: front, cols: reference).
# NOTE: DESDEO's distance_indicators.ahd uses a set-size-scaled IGD_p/GD_p that is not the standard
# averaged Hausdorff distance and is not comparable across problems, so Delta_p is computed here directly.
dmat = cdist(front_n, pf_n)
p = 2.0
gd_to_ref = dmat.min(axis=1)  # each front point -> nearest reference point
ref_to_front = dmat.min(axis=0)  # each reference point -> nearest front point
coverage_error = float(ref_to_front.max())  # worst-covered reference point (directed ref->front)
hausdorff = float(max(gd_to_ref.max(), coverage_error))  # standard (worst-case) Hausdorff distance
delta_p = float(max(np.mean(gd_to_ref**p) ** (1 / p), np.mean(ref_to_front**p) ** (1 / p)))  # averaged Hausdorff

igd_plus_val = igd_plus_indicator(front_n, pf_n).igd_plus
eps_add = float(epsilon_indicator(front_n, pf_n, kind="additive"))
r2_val = r2_indicator(front_n, weights, z_star, rho).r2_value

with open(snakemake.output[0], "w") as fh:  # noqa: F821, PTH123
    json.dump(
        {
            "algorithm": snakemake.wildcards.algo,  # noqa: F821
            "problem": snakemake.wildcards.problem,  # noqa: F821
            "seed": int(snakemake.wildcards.seed),  # noqa: F821
            "hv": float(hv_val),
            "igd": float(di.igd),
            "igd_plus": float(igd_plus_val),
            "gd": float(di.gd),
            "delta_p": delta_p,
            "hausdorff": hausdorff,
            "coverage_error": float(coverage_error),
            "eps_add": float(eps_add),
            "r2": float(r2_val),
            "n_nondominated": int(nd_front.shape[0]),
        },
        fh,
        indent=2,
    )
