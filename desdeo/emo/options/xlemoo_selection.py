"""Options model and constructor for the XLEMOO learning mode instantiator.

Provides ``XLEMOOSelectorOptions`` for configuring the
:class:`~desdeo.emo.operators.xlemoo_selection.XLEMOOInstantiator`.

The fitness indicator logic has been removed; the instantiator now reads the
scalarization column directly from the archive (computed by the evaluator via
a ``ScalarizationFunction`` added to the ``Problem``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from desdeo.emo.operators.xlemoo_selection import MLModelType, XLEMOOInstantiator

if TYPE_CHECKING:
    from desdeo.emo.hooks.archivers import Archive
    from desdeo.problem import Problem
    from desdeo.tools.patterns import Publisher


class XLEMOOSelectorOptions(BaseModel):
    """Options for the XLEMOO learning mode instantiator."""

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
        problem: The optimization problem (should already have a scalarization
            function so that ``target_symbols`` points to the scalarization column).
        options: XLEMOO learning mode options.
        archive: The Archive that accumulates population history.
        publisher: The publisher for pub-sub.
        verbosity: Verbosity level.
        seed: Random seed.

    Returns:
        An initialized XLEMOOInstantiator.
    """
    return XLEMOOInstantiator(
        problem=problem,
        verbosity=verbosity,
        publisher=publisher,
        archive=archive,
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
