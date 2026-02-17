"""Options model and constructors for the XLEMOO learning mode operators.

Provides ``XLEMOOSelectorOptions`` for configuring both the
:class:`~desdeo.emo.operators.xlemoo_selection.XLEMOOInstantiator` and
:class:`~desdeo.emo.operators.xlemoo_selection.XLEMOOSelector`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from moocore import Hypervolume
from pydantic import BaseModel, Field

from desdeo.emo.operators.xlemoo_selection import MLModelType, XLEMOOInstantiator, XLEMOOSelector

if TYPE_CHECKING:
    from desdeo.emo.hooks.archivers import Archive
    from desdeo.problem import Problem
    from desdeo.tools.patterns import Publisher


class XLEMOOSelectorOptions(BaseModel):
    """Options for the XLEMOO learning mode operators (instantiator + selector)."""

    name: Literal["XLEMOOSelector"] = Field(
        default="XLEMOOSelector", frozen=True, description="The name of the selection operator."
    )
    ml_model_type: MLModelType = Field(
        default=MLModelType.DECISION_TREE,
        description="The type of ML classifier to use.",
    )
    ml_model_kwargs: dict = Field(
        default_factory=dict,
        description="Additional kwargs for the ML model constructor.",
    )
    h_split: float = Field(
        default=0.1,
        gt=0,
        description="Fraction (<1) or count (>=1) of best solutions for H-group.",
    )
    l_split: float = Field(
        default=0.1,
        gt=0,
        description="Fraction (<1) or count (>=1) of worst solutions for L-group.",
    )
    instantiation_factor: float = Field(
        default=2.0,
        gt=0,
        description="Multiplier for the number of new solutions to instantiate.",
    )
    generation_lookback: int = Field(
        default=0,
        ge=0,
        description="Number of recent generations to consider (0 = all history).",
    )
    ancestral_recall: int = Field(
        default=0,
        ge=0,
        description="Number of earliest generations to always include.",
    )
    unique_only: bool = Field(
        default=False,
        description="Whether to deduplicate solutions before training.",
    )
    fitness_indicator: Literal["naive_sum", "asf", "hypervolume"] = Field(
        default="naive_sum",
        description="Which fitness indicator to use for H/L ranking.",
    )
    asf_reference_point: list[float] | None = Field(
        default=None,
        description="Reference point for ASF indicator (required if fitness_indicator='asf').",
    )
    asf_weights: list[float] | None = Field(
        default=None,
        description="Weights for ASF indicator. If None, equal weights are used.",
    )
    hypervolume_ref_point: list[float] | None = Field(
        default=None,
        description="Reference point for hypervolume indicator.",
    )


def _build_indicator(options: XLEMOOSelectorOptions):
    """Build the fitness indicator callable from the options.

    Args:
        options: The XLEMOOSelector options.

    Returns:
        A callable that takes targets (n, m) and returns fitness (n,). Lower is better.
    """
    if options.fitness_indicator == "naive_sum":

        def naive_sum_indicator(targets: np.ndarray) -> np.ndarray:
            return np.sum(targets, axis=1)

        return naive_sum_indicator

    if options.fitness_indicator == "asf":
        ref = np.array(options.asf_reference_point) if options.asf_reference_point is not None else None
        w = np.array(options.asf_weights) if options.asf_weights is not None else None
        rho = 1e-6

        def asf_indicator(targets: np.ndarray, ref=ref, w=w, rho=rho) -> np.ndarray:
            if ref is None:
                z = np.zeros(targets.shape[1])
            else:
                z = ref
            if w is None:
                weights = np.ones(targets.shape[1])
            else:
                weights = w
            weighted = (targets - z) * weights
            return np.max(weighted, axis=1) + rho * np.sum(weighted, axis=1)

        return asf_indicator

    if options.fitness_indicator == "hypervolume":
        ref_point = np.array(options.hypervolume_ref_point) if options.hypervolume_ref_point is not None else None

        def hv_contribution_indicator(targets: np.ndarray, ref_point=ref_point) -> np.ndarray:
            if ref_point is None:
                rp = np.max(targets, axis=0) * 1.1 + 0.1
            else:
                rp = ref_point
            hv_calc = Hypervolume(rp)
            hv_baseline = hv_calc(targets)
            contributions = np.zeros(targets.shape[0])
            mask = np.ones(targets.shape[0], dtype=bool)
            for i in range(targets.shape[0]):
                mask[i] = False
                remaining = targets[mask]
                if len(remaining) > 0:
                    hv_without = hv_calc(remaining)
                else:
                    hv_without = 0.0
                contributions[i] = hv_baseline - hv_without
                mask[i] = True
            # Negate: higher contribution = better, but we want lower = better
            return -contributions

        return hv_contribution_indicator

    msg = f"Unknown fitness indicator: {options.fitness_indicator}"
    raise ValueError(msg)


def xlemoo_instantiator_constructor(
    problem: Problem,
    options: XLEMOOSelectorOptions,
    archive: Archive,
    publisher: Publisher,
    verbosity: int,
    seed: int,
) -> XLEMOOInstantiator:
    """Construct an XLEMOOInstantiator from options.

    Args:
        problem: The optimization problem.
        options: XLEMOO learning mode options.
        archive: The Archive that accumulates population history.
        publisher: The publisher for pub-sub.
        verbosity: Verbosity level.
        seed: Random seed.

    Returns:
        An initialized XLEMOOInstantiator.
    """
    indicator = _build_indicator(options)

    return XLEMOOInstantiator(
        problem=problem,
        verbosity=verbosity,
        publisher=publisher,
        archive=archive,
        indicator=indicator,
        ml_model_type=options.ml_model_type,
        ml_model_kwargs=options.ml_model_kwargs,
        h_split=options.h_split,
        l_split=options.l_split,
        instantiation_factor=options.instantiation_factor,
        generation_lookback=options.generation_lookback,
        ancestral_recall=options.ancestral_recall,
        unique_only=options.unique_only,
        seed=seed,
    )


def xlemoo_selector_constructor(
    problem: Problem,
    options: XLEMOOSelectorOptions,
    publisher: Publisher,
    verbosity: int,
    seed: int,
) -> XLEMOOSelector:
    """Construct an XLEMOOSelector from options.

    Args:
        problem: The optimization problem.
        options: XLEMOO learning mode options.
        publisher: The publisher for pub-sub.
        verbosity: Verbosity level.
        seed: Random seed.

    Returns:
        An initialized XLEMOOSelector.
    """
    indicator = _build_indicator(options)

    return XLEMOOSelector(
        problem=problem,
        verbosity=verbosity,
        publisher=publisher,
        indicator=indicator,
        seed=seed,
    )
