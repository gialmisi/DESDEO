import pytest

from desdeo.method.NIMBUS import NIMBUS
from desdeo.optimization.OptimizationMethod import SciPyDE
from desdeo.problem.Problem import MOProblem
from examples.NarulaWeistroffer import RiverPollution


@pytest.fixture(scope="function")
def method():
    return NIMBUS(RiverPollution(), SciPyDE)


moproblem_params = {
    "nobj": 5,
    "nconst": 2,
    "ideal": [2.0, -10.7, 6.2, 0.00001, -99.1],
    "nadir": [1.0, -5.2, 4.1, 0.0001, -56.1],
    "maximized": [True, False, True, False, False],
    "objectives": ["velocity", "tension", "mass", "factor", "morality"],
    "name": "test problem",
    "points": None}


class moproblem_specialized(MOProblem):
    """Documentation for moproblem_specialized

    """
    def __init__(self, args):
        super().__init__(**args)

    def evaluate(self, population):
        return population


@pytest.fixture
def moproblem():
    return moproblem_specialized(moproblem_params)


@pytest.fixture
def moproblem_no_objectives_names():
    params = moproblem_params
    params["objectives"] = None
    return moproblem_specialized(params)
