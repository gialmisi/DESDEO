"""Defines a forest management scenario problem with multiple stakeholders.

Three forest owners each manage a cluster of 8 stands, and a supervisor
oversees the entire landscape of 24 stands.  Each stand has 6 management
regimes.  Objectives (NPV, CHSI, CO2) are summed per-cluster for owners
and landscape-wide for the supervisor.
"""

import json
from pathlib import Path

import numpy as np

from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    TensorConstant,
    TensorVariable,
    VariableTypeEnum,
)

_DEFAULT_DATA = Path(__file__).parent.parent.parent.parent / "experiments" / "forest_scenario_data.csv"
_DEFAULT_MAPPING = Path(__file__).parent.parent.parent.parent / "experiments" / "cluster_mapping.json"


def forest_scenario_problem(
    data_path: str | Path = _DEFAULT_DATA,
    cluster_mapping_path: str | Path = _DEFAULT_MAPPING,
) -> Problem:
    """Build a scenario-based forest management problem.

    Args:
        data_path: Path to the CSV with columns
            ``cluster, stand_id, regime_id, regime_name, npv, chsi, co2``.
        cluster_mapping_path: Path to a JSON file mapping owner keys
            (``owner_1``, ``owner_2``, ``owner_3``) to lists of stand IDs.

    Returns:
        A :class:`Problem` with 4 scenarios (``supervisor``, ``owner_1``,
        ``owner_2``, ``owner_3``), 12 objectives (3 per scenario), shared
        binary decision variables, and one-regime-per-stand equality
        constraints.
    """
    data_path = Path(data_path)
    cluster_mapping_path = Path(cluster_mapping_path)

    # --- load data ---
    raw = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=None, encoding="utf-8")

    with open(cluster_mapping_path) as f:
        cluster_mapping: dict[str, list[int]] = json.load(f)

    # Organise per-stand arrays: stand_id -> {npv: [...], chsi: [...], co2: [...]}
    stand_data: dict[int, dict[str, list[float]]] = {}
    for row in raw:
        stand_id = int(row[1])
        if stand_id not in stand_data:
            stand_data[stand_id] = {"npv": [], "chsi": [], "co2": []}
        stand_data[stand_id]["npv"].append(float(row[4]))
        stand_data[stand_id]["chsi"].append(float(row[5]))
        stand_data[stand_id]["co2"].append(float(row[6]))

    all_stand_ids = sorted(stand_data.keys())
    n_regimes = len(stand_data[all_stand_ids[0]]["npv"])

    # --- build constants, variables, constraints ---
    constants: list[TensorConstant] = []
    variables: list[TensorVariable] = []
    constraints: list[Constraint] = []

    for sid in all_stand_ids:
        for obj_key in ("NPV", "CHSI", "CO2"):
            constants.append(
                TensorConstant(
                    name=f"{obj_key}_{sid}",
                    symbol=f"{obj_key}_{sid}",
                    shape=[n_regimes],
                    values=stand_data[sid][obj_key.lower()],
                )
            )

        variables.append(
            TensorVariable(
                name=f"X_{sid}",
                symbol=f"X_{sid}",
                variable_type=VariableTypeEnum.binary,
                shape=[n_regimes],
                lowerbounds=[0] * n_regimes,
                upperbounds=[1] * n_regimes,
                initial_values=[0] * n_regimes,
            )
        )

        constraints.append(
            Constraint(
                name=f"one_regime_{sid}",
                symbol=f"one_regime_{sid}",
                cons_type=ConstraintTypeEnum.EQ,
                func=f"Sum(X_{sid}) - 1",
                is_linear=True,
                is_convex=False,
                is_twice_differentiable=True,
                scenario_keys=None,  # shared across all scenarios
            )
        )

    # --- compute per-stand min/max for ideal/nadir ---
    # For maximize objectives: ideal = sum of max per stand, nadir = sum of min per stand
    def _compute_bounds(obj_key: str, stand_ids: list[int]) -> tuple[float, float]:
        ideal = sum(max(stand_data[sid][obj_key.lower()]) for sid in stand_ids)
        nadir = sum(min(stand_data[sid][obj_key.lower()]) for sid in stand_ids)
        return ideal, nadir

    # --- build objectives ---
    objectives: list[Objective] = []

    # Helper: build a sum-of-dot-products expression for a set of stands
    def _sum_expr(obj_key: str, stand_ids: list[int]) -> str:
        return " + ".join(f"{obj_key}_{sid}@X_{sid}" for sid in stand_ids)

    obj_meta = [
        ("NPV", "Net present value", True),
        ("CHSI", "Combined habitat suitability index", True),
        ("CO2", "Carbon dioxide storage", True),
    ]

    # Per-owner objectives
    for owner_key, owner_stands in cluster_mapping.items():
        for obj_key, obj_name, maximize in obj_meta:
            ideal, nadir = _compute_bounds(obj_key, owner_stands)
            objectives.append(
                Objective(
                    name=f"{obj_name} ({owner_key})",
                    symbol=f"{obj_key}_{owner_key}",
                    func=_sum_expr(obj_key, owner_stands),
                    maximize=maximize,
                    ideal=ideal,
                    nadir=nadir,
                    objective_type=ObjectiveTypeEnum.analytical,
                    is_linear=True,
                    is_convex=False,
                    is_twice_differentiable=True,
                    scenario_keys=[owner_key],
                )
            )

    # Supervisor objectives (all stands)
    for obj_key, obj_name, maximize in obj_meta:
        ideal, nadir = _compute_bounds(obj_key, all_stand_ids)
        objectives.append(
            Objective(
                name=f"{obj_name} (total)",
                symbol=f"{obj_key}_total",
                func=_sum_expr(obj_key, all_stand_ids),
                maximize=maximize,
                ideal=ideal,
                nadir=nadir,
                objective_type=ObjectiveTypeEnum.analytical,
                is_linear=True,
                is_convex=False,
                is_twice_differentiable=True,
                scenario_keys=["supervisor"],
            )
        )

    return Problem(
        name="Forest Scenario Problem",
        description=(
            "A collaborative forest management problem with 24 stands across 3 clusters. "
            "Three owners optimise their own clusters while a supervisor optimises the whole landscape."
        ),
        constants=constants,
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        scenario_keys=["supervisor", "owner_1", "owner_2", "owner_3"],
    )
