"""Explanation providers: what the experiment harness shows the subjects of some arms after the anchor.

The harness knows nothing about what an explanation means. It asks the provider named in the experiment definition to
explain each solution a subject finds after the anchor, and stores and returns the payload with its content hash.

A provider is created from the options in the experiment definition. It checks, when a run is created, that it can
explain solutions of the run's problem, and it explains a solution as a JSON-serializable payload. The same solution
must always give the same payload, so that the content hash identifies what was shown.

Providers are looked up by name in `EXPLANATION_PROVIDERS`, so an experiment definition cannot make the server import
arbitrary code.
"""

from functools import cache
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from desdeo.problem import Problem
from desdeo.problem.testproblems.forest_landscape_problem import (
    ForestLandscape,
    forest_landscape_data,
    forest_landscape_problem,
    generalize,
)
from desdeo.tools import SolverResults


class ExplanationProvider(Protocol):
    """The interface of explanation providers."""

    def check(self, problem: Problem) -> None:
        """Check that solutions of a problem can be explained.

        Raises:
            ValueError: solutions of the problem cannot be explained.
        """

    def explain(self, problem: Problem, solution: SolverResults) -> dict:
        """Explain a solution of a problem, returning a JSON-serializable payload."""


@cache
def _forest_landscape(stands_per_lot: int, seed: int) -> ForestLandscape:
    """Return the forest landscape, generated once per process."""
    return forest_landscape_data(stands_per_lot=stands_per_lot, seed=seed)


class ForestGeneralization(BaseModel):
    """Explains a solution of a lot of the forest landscape by what happens to the lot if every owner did likewise.

    The payload holds, for the owner's own lot only, the baseline value, the generalized value, and the difference of
    each objective and property (see `generalize`). The landscape is how the effect comes about, but the landscape-wide
    figures and the other lots' regimes are left out.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stands_per_lot: int = Field(default=4, description="The number of stands in each lot of the landscape.")
    seed: int = Field(default=0, description="The seed of the landscape.")
    lot: str = Field(default="A", description="The lot of the subject.")

    @property
    def landscape(self) -> ForestLandscape:
        """The landscape."""
        return _forest_landscape(self.stands_per_lot, self.seed)

    def check(self, problem: Problem) -> None:
        """Check that the problem is the problem of the lot: the same variables, and objectives with the same ranges.

        Raises:
            ValueError: the problem is not the problem of the lot.
        """
        expected = forest_landscape_problem(self.landscape, self.lot)

        def signature(p: Problem) -> tuple:
            return (
                sorted(variable.symbol for variable in p.variables),
                [(o.symbol, round(o.ideal, 9), round(o.nadir, 9)) for o in p.objectives],
            )

        if signature(problem) != signature(expected):
            msg = (
                f"Problem '{problem.name}' is not the problem of lot '{self.lot}' of the forest landscape with "
                f"{self.stands_per_lot} stands per lot and seed {self.seed}."
            )
            raise ValueError(msg)

    def explain(self, problem: Problem, solution: SolverResults) -> dict:
        """Explain a solution by its generalization, for the owner's own lot only."""
        landscape = self.landscape
        regimes = {
            stand.id: landscape.regimes[int(np.argmax(solution.optimal_variables[f"X_{stand.id}"]))]
            for stand in landscape.lot_stands(self.lot)
        }
        return generalize(landscape, self.lot, regimes).model_dump(mode="json", include={"objectives", "properties"})


EXPLANATION_PROVIDERS: dict[str, type[ExplanationProvider]] = {"forest_generalization": ForestGeneralization}
"""The explanation providers experiment definitions can name, by name."""


def explanation_provider(name: str, options: dict) -> ExplanationProvider:
    """Create the explanation provider of a name with options.

    Args:
        name (str): the name of the provider, a key of `EXPLANATION_PROVIDERS`.
        options (dict): the options of the provider.

    Raises:
        ValueError: there is no provider of the name, or the options are invalid for it.

    Returns:
        ExplanationProvider: the provider.
    """
    if name not in EXPLANATION_PROVIDERS:
        msg = f"There is no explanation provider '{name}'; the providers are {sorted(EXPLANATION_PROVIDERS)}."
        raise ValueError(msg)
    return EXPLANATION_PROVIDERS[name](**options)
