"""Tests for the synthetic forest landscape problem."""

import pytest

from desdeo.problem.testproblems.forest_landscape_problem import LOTS, REGIMES, forest_landscape_data


@pytest.mark.testproblem
@pytest.mark.forest_problem
@pytest.mark.parametrize("stands_per_lot", [2, 4, 5, 20])
def test_landscape_structure(stands_per_lot: int):
    """Test that the landscape has the requested size and a symmetric, cross-owner adjacency."""
    landscape = forest_landscape_data(stands_per_lot=stands_per_lot)

    assert len(landscape.stands) == len(LOTS) * stands_per_lot
    assert all(len(landscape.lot_stands(lot)) == stands_per_lot for lot in LOTS)
    assert all(stand.id == i for i, stand in enumerate(landscape.stands))

    # stands occupy distinct grid cells
    assert len({(stand.row, stand.col) for stand in landscape.stands}) == len(landscape.stands)

    # adjacency is symmetric
    for stand_id, neighbours in landscape.neighbours.items():
        assert stand_id not in neighbours
        for n in neighbours:
            assert stand_id in landscape.neighbours[n]

    # every lot borders exactly two other lots
    for lot in LOTS:
        bordering = {
            landscape.stands[n].lot
            for stand in landscape.lot_stands(lot)
            for n in landscape.cross_owner_neighbours(stand.id)
        }
        assert len(bordering) == 2
        assert lot not in bordering


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_landscape_values():
    """Test that every stand has a value per regime and that the regimes keep their characteristic ordering."""
    landscape = forest_landscape_data()

    for stand in landscape.stands:
        assert 0.3 <= stand.area <= 4.8
        assert all(len(values) == len(REGIMES) for values in stand.values.values())
        assert all(0.0 <= value <= 1.0 for value in stand.values["habitat"])

        npv = dict(zip(REGIMES, stand.values["npv"], strict=True))
        carbon = dict(zip(REGIMES, stand.values["carbon"], strict=True))
        habitat = dict(zip(REGIMES, stand.values["habitat"], strict=True))

        assert npv["set_aside"] == 0.0
        assert max(npv, key=npv.get) == "clearfell"
        assert max(carbon, key=carbon.get) == "set_aside"
        assert max(habitat, key=habitat.get) == "set_aside"
        assert min(habitat, key=habitat.get) == "clearfell"


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_landscape_is_reproducible():
    """Test that the same seed gives the same landscape and a different seed a different one."""
    assert forest_landscape_data(seed=1) == forest_landscape_data(seed=1)
    assert forest_landscape_data(seed=1) != forest_landscape_data(seed=2)


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_landscape_rejects_single_stand_lots():
    """Test that a lot must have at least two stands."""
    with pytest.raises(ValueError, match="at least two stands"):
        forest_landscape_data(stands_per_lot=1)
