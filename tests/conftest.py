import pytest

from desdeo.method.NIMBUS import NIMBUS
from desdeo.optimization.OptimizationMethod import SciPyDE
from desdeo.problem.Problem import MOProblem, Variable, PreGeneratedProblem
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

variables = [
    Variable([-5.0, 5.0], 0.0, "test_var_1"),
    Variable([-14.2, -2.3], -12.22, "test_var_2"),
    Variable([4.2, 99.99], 34.1, "test_var_3")]

variable_params = {
    "bounds": [-2.5, 6.2],
    "starting_point": -0.001,
    "name": "test_var"}

test_file_path = "/home/kilo/workspace/DESDEO/tests/test_data.dat"

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


@pytest.fixture
def moproblem_bare():
    return moproblem_specialized({"nobj": 0})


@pytest.fixture
def moproblem_with_vars():
    res = moproblem_specialized(moproblem_params)
    res.add_variables(variables)
    return res


@pytest.fixture
def variable():
    return Variable(**variable_params)


@pytest.fixture
def pregeneratedproblem_file():
    return PreGeneratedProblem(
        filename=test_file_path)


@pytest.fixture
def pregeneratedproblem_param():
    points = []
    with open(test_file_path) as handle:
        for line in handle:
            points.append(list(map(float, map(str.strip, line.split(',')))))
    return PreGeneratedProblem(points=points)
