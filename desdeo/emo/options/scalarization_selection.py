"""Options for the ScalarizationSelector and ScalarizationSpec.

``ScalarizationSelectorOptions`` configures the
:class:`~desdeo.emo.operators.scalarization_selection.ScalarizationSelector`.

``ScalarizationSpec`` describes a scalarization function to add to the Problem
before creating the evaluator, so that both the Darwinian selector and the
XLEMOO learning mode share the same fitness criterion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScalarizationSelectorOptions(BaseModel):
    """Options for the ScalarizationSelector."""

    name: Literal["ScalarizationSelector"] = Field(
        default="ScalarizationSelector",
        frozen=True,
        description="The name of the selection operator.",
    )
    population_size: int = Field(gt=0, default=100, description="The population size.")


class ScalarizationSpec(BaseModel):
    """Specification for a scalarization function to add to the Problem.

    Used by ``emo_constructor`` to add the scalarization to the Problem before
    creating the evaluator.  The evaluator then computes this value for every
    solution, and both the ``ScalarizationSelector`` and the
    ``XLEMOOInstantiator`` read the same column from the archive / outputs.
    """

    type: Literal["weighted_sums", "asf"] = Field(
        default="weighted_sums",
        description="Type of scalarization function.",
    )
    symbol: str = Field(
        default="scal_fitness",
        description="Column name for the scalarization value.",
    )
    weights: list[float] | None = Field(
        default=None,
        description="Weights per objective for weighted_sums. If None, equal weights.",
    )
    reference_point: list[float] | None = Field(
        default=None,
        description="Reference point for ASF (required when type='asf').",
    )
    delta: float = Field(default=0.000001, description="Delta parameter for ASF.")
    rho: float = Field(default=0.000001, description="Rho parameter for ASF.")
