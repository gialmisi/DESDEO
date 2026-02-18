"""Selection operator that ranks solutions by a scalarization column.

The ``ScalarizationSelector`` reads the scalarization value(s) produced by the
evaluator (via a ``ScalarizationFunction`` added to the ``Problem``) and performs
truncation selection by sorting on those values.  This ensures that both the
Darwinian selector and the XLEMOO learning mode use the **same** fitness criterion.

Follows the ``NSGA2Selector`` pattern for pub-sub topics and state messages.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import polars as pl

from desdeo.problem import Problem
from desdeo.tools.message import (
    DictMessage,
    Message,
    NumpyArrayMessage,
    PolarsDataFrameMessage,
    SelectorMessageTopics,
)
from desdeo.tools.patterns import Publisher

from .selection import BaseSelector, SolutionType


class ScalarizationSelector(BaseSelector):
    """Truncation selector that ranks solutions by a scalarization column.

    Expects the ``Problem`` to have one or more ``scalarization_funcs`` so that
    ``BaseSelector.__init__`` populates ``self.target_symbols`` with the
    corresponding column name(s).  The evaluator computes these columns for
    every solution; this selector simply sorts on the summed scalarization
    value and keeps the best ``population_size`` solutions.
    """

    @property
    def provided_topics(self) -> dict[int, Sequence[SelectorMessageTopics]]:
        """Topics provided at each verbosity level."""
        return {
            0: [],
            1: [SelectorMessageTopics.STATE],
            2: [
                SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                SelectorMessageTopics.SELECTED_FITNESS,
            ],
        }

    @property
    def interested_topics(self) -> Sequence:
        """No external messages needed."""
        return []

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        population_size: int,
        seed: int = 0,
    ):
        super().__init__(problem=problem, verbosity=verbosity, publisher=publisher, seed=seed)
        self.population_size = population_size
        self.selection: list[int] | None = None
        self.selected_individuals: SolutionType | None = None
        self.selected_targets: pl.DataFrame | None = None
        self.fitness: np.ndarray | None = None

    def do(
        self,
        parents: tuple[SolutionType, pl.DataFrame],
        offsprings: tuple[SolutionType, pl.DataFrame],
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Select the best ``population_size`` solutions by scalarization value."""
        # First iteration: offspring is empty
        if offsprings[0].is_empty() and offsprings[1].is_empty():
            targets = parents[1][self.target_symbols].to_numpy()
            fitness = targets[:, 0] if targets.shape[1] == 1 else np.sum(targets, axis=1)
            sorted_idx = np.argsort(fitness)

            self.fitness = fitness[sorted_idx]
            self.selection = sorted_idx.tolist()
            self.selected_individuals = parents[0][sorted_idx.tolist()]
            self.selected_targets = parents[1][sorted_idx.tolist()]

            self.notify()
            return self.selected_individuals, self.selected_targets

        # Combine parents and offspring
        all_solutions = parents[0].vstack(offsprings[0])
        all_outputs = parents[1].vstack(offsprings[1])

        targets = all_outputs[self.target_symbols].to_numpy()
        fitness = targets[:, 0] if targets.shape[1] == 1 else np.sum(targets, axis=1)
        sorted_idx = np.argsort(fitness)[: self.population_size]

        self.fitness = fitness[sorted_idx]
        self.selection = sorted_idx.tolist()
        self.selected_individuals = all_solutions[sorted_idx.tolist()]
        self.selected_targets = all_outputs[sorted_idx.tolist()]

        self.notify()
        return self.selected_individuals, self.selected_targets

    def state(self) -> Sequence[Message]:
        """Return the current state of the selector."""
        if self.verbosity == 0 or self.selection is None or self.selected_targets is None:
            return []
        if self.verbosity == 1:
            return [
                DictMessage(
                    topic=SelectorMessageTopics.STATE,
                    value={
                        "population_size": self.population_size,
                        "selected_individuals": self.selection,
                    },
                    source=self.__class__.__name__,
                )
            ]
        # verbosity == 2
        if isinstance(self.selected_individuals, pl.DataFrame):
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=pl.concat([self.selected_individuals, self.selected_targets], how="horizontal"),
                source=self.__class__.__name__,
            )
        else:
            warnings.warn("Population is not a Polars DataFrame. Defaulting to providing OUTPUTS only.", stacklevel=2)
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=self.selected_targets,
                source=self.__class__.__name__,
            )
        return [
            DictMessage(
                topic=SelectorMessageTopics.STATE,
                value={
                    "population_size": self.population_size,
                    "selected_individuals": self.selection,
                },
                source=self.__class__.__name__,
            ),
            message,
            NumpyArrayMessage(
                topic=SelectorMessageTopics.SELECTED_FITNESS,
                value=self.fitness,
                source=self.__class__.__name__,
            ),
        ]

    def update(self, message: Message) -> None:
        """No-op: this selector does not consume any messages."""
