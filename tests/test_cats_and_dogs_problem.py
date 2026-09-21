"""Tests related to the cat and dog breed problems."""

import pytest

from desdeo.mcdm import rpm_solve_solutions
from desdeo.problem.testproblems import (
    cat_breed_group_names,
    cat_breed_problem,
    dog_breed_group_names,
    dog_breed_problem,
)

# The problem, the breed group names, and the number of breed groups the data
# is known to contain.
BREED_PROBLEMS = [
    (cat_breed_problem, cat_breed_group_names, 26),
    (dog_breed_problem, dog_breed_group_names, 52),
]


@pytest.mark.testproblem
@pytest.mark.parametrize(("build_problem", "group_names", "num_groups"), BREED_PROBLEMS)
def test_breed_problem_scales(build_problem, group_names, num_groups):
    """Every objective is on a 0 to 100 scale, with its direction given by `maximize`."""
    problem = build_problem()

    assert len(problem.objectives) == 7
    assert problem.discrete_representation is not None
    assert len(group_names()) == num_groups

    for objective in problem.objectives:
        best, worst = (100.0, 0.0) if objective.maximize else (0.0, 100.0)
        assert objective.ideal == pytest.approx(best)
        assert objective.nadir == pytest.approx(worst)


@pytest.mark.testproblem
@pytest.mark.parametrize(("build_problem", "group_names", "num_groups"), BREED_PROBLEMS)
def test_breed_problem_breed_ids(build_problem, group_names, num_groups):
    """The `breed_id` variable of every row points at one of the breed groups."""
    problem = build_problem()

    breed_ids = problem.discrete_representation.variable_values["breed_id"]
    row_indices = problem.discrete_representation.variable_values["index"]

    assert len(breed_ids) == len(row_indices)
    assert row_indices == list(range(len(row_indices)))
    assert set(breed_ids) == set(range(num_groups))


@pytest.mark.rpm
@pytest.mark.proximal
@pytest.mark.parametrize(("build_problem", "group_names", "num_groups"), BREED_PROBLEMS)
def test_breed_problem_with_reference_point_method(build_problem, group_names, num_groups):
    """The reference point method finds breeds, and every one of them can be named."""
    problem = build_problem()
    names = group_names()

    # A decision maker who wants the best of everything.
    reference_point = {objective.symbol: objective.ideal for objective in problem.objectives}

    results = rpm_solve_solutions(problem, reference_point)

    assert len(results) == len(problem.objectives) + 1

    for result in results:
        assert result.success

        breed_id = int(result.optimal_variables["breed_id"])
        row_index = int(result.optimal_variables["index"])

        assert 0 <= breed_id < num_groups
        assert 0 <= row_index < len(problem.discrete_representation.variable_values["index"])

        # The breed group of the solution matches the row it came from.
        assert breed_id == problem.discrete_representation.variable_values["breed_id"][row_index]
        assert names[breed_id]
