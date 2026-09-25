"""Tests for the synthetic forest landscape problem."""

import itertools

import numpy as np
import pytest

from desdeo.problem import PolarsEvaluator
from desdeo.problem.testproblems.forest_landscape_problem import (
    LOTS,
    REGIMES,
    CouplingParameters,
    forest_landscape_data,
    forest_landscape_problem,
    realized_values,
)
from desdeo.tools import available_solvers, payoff_table_method
from desdeo.tools.scalarization import add_asf_diff


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
        assert max(habitat, key=habitat.get) == "selection_cut"
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


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_problem_structure():
    """Test that the problem has the three objectives, and that the properties are not among them."""
    landscape = forest_landscape_data()
    problem = forest_landscape_problem(landscape, lot="B")

    objective_symbols = [objective.symbol for objective in problem.objectives]
    property_symbols = [extra.symbol for extra in problem.extra_funcs]

    assert objective_symbols == ["npv", "habitat", "carbon"]
    assert all(objective.maximize for objective in problem.objectives)
    assert set(property_symbols) == {"bilberry", "mushroom", "scenic", "deadwood"}
    assert not set(property_symbols) & set(objective_symbols)

    # only the stands of the lot are decision variables
    assert {variable.symbol for variable in problem.variables} == {
        f"X_{stand.id}" for stand in landscape.lot_stands("B")
    }
    assert problem.is_linear

    with pytest.raises(ValueError, match="not in the landscape"):
        forest_landscape_problem(landscape, lot="E")


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_problem_evaluates_over_regime_space():
    """Test that objectives and properties evaluate over every regime assignment of the lot, and match the data."""
    landscape = forest_landscape_data()
    stands = landscape.lot_stands("A")
    problem = forest_landscape_problem(landscape, lot="A")

    assignments = list(itertools.product(range(len(REGIMES)), repeat=len(stands)))
    one_hot = np.eye(len(REGIMES)).tolist()
    xs = {f"X_{stand.id}": [one_hot[assignment[j]] for assignment in assignments] for j, stand in enumerate(stands)}

    result = PolarsEvaluator(problem).evaluate(xs)

    assert len(result) == len(assignments)
    for symbol in ["npv", "habitat", "carbon", "bilberry", "mushroom", "scenic", "deadwood"]:
        assert result[symbol].is_finite().all()

    # check one assignment by hand: the regime of stand j is the j-th regime, cycling
    assignment = tuple(j % len(REGIMES) for j in range(len(stands)))
    row = result.row(assignments.index(assignment), named=True)
    lot_area = sum(stand.area for stand in stands)
    chosen = list(zip(stands, assignment, strict=True))

    assert np.isclose(row["npv"], sum(stand.area * stand.values["npv"][i] for stand, i in chosen))
    assert np.isclose(row["carbon"], sum(stand.area * stand.values["carbon"][i] for stand, i in chosen))
    realized = realized_values(landscape, "A")
    assert np.isclose(
        row["habitat"], sum(stand.area * realized[stand.id]["habitat"][i] for stand, i in chosen) / lot_area
    )
    assert np.isclose(row["scenic"], sum(stand.area * stand.values["scenic"][i] for stand, i in chosen) / lot_area)


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_problem_objectives_conflict():
    """Test that each objective, optimized alone, prefers a different regime on every stand."""
    landscape = forest_landscape_data()
    stands = landscape.lot_stands("A")
    problem = forest_landscape_problem(landscape, lot="A")
    solver = available_solvers["pyomo_cbc"]["constructor"](problem)

    for objective, regime in [("npv", "clearfell"), ("habitat", "selection_cut"), ("carbon", "set_aside")]:
        result = solver.solve(f"{objective}_min")

        assert result.success
        for stand in stands:
            chosen = int(np.argmax(result.optimal_variables[f"X_{stand.id}"]))
            assert REGIMES[chosen] == regime


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_problem_ideal_and_nadir():
    """Test that the stored ideal and nadir match the payoff table computed with a solver, and span a range."""
    problem = forest_landscape_problem(forest_landscape_data(), lot="C")

    ideal, nadir = payoff_table_method(problem, solver=available_solvers["pyomo_cbc"]["constructor"])

    for objective in problem.objectives:
        assert np.isclose(objective.ideal, ideal[objective.symbol])
        assert np.isclose(objective.nadir, nadir[objective.symbol])
        assert objective.nadir < objective.ideal


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_problem_solves_with_reference_point():
    """Test that an achievement scalarizing function can be built on the problem and solved."""
    problem = forest_landscape_problem(forest_landscape_data(), lot="A")

    reference_point = {objective.symbol: (objective.ideal + objective.nadir) / 2 for objective in problem.objectives}
    problem_w_asf, target = add_asf_diff(problem, "asf", reference_point)
    result = available_solvers["pyomo_cbc"]["constructor"](problem_w_asf).solve(target)

    assert result.success
    for objective in problem.objectives:
        assert objective.nadir - 1e-6 <= result.optimal_objectives[objective.symbol] <= objective.ideal + 1e-6
    for variable in problem.variables:
        assert sorted(np.round(result.optimal_variables[variable.symbol], 6)) == [0, 0, 0, 1]


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_neighbouring_lot_changes_realized_objectives():
    """Test that changing the regimes of a neighbouring lot changes the lot's realized objective values.

    This is the cross-owner coupling. Without it, the problem is not fit for studying cross-owner consequences.
    """
    landscape = forest_landscape_data()
    stands = landscape.lot_stands("A")
    neighbours_in_b = {n for stand in stands for n in landscape.cross_owner_neighbours(stand.id)} & {
        stand.id for stand in landscape.lot_stands("B")
    }
    assert neighbours_in_b

    xs = {f"X_{stand.id}": [[0.0, 1.0, 0.0, 0.0]] for stand in stands}  # selection cut everywhere

    def evaluate(others: dict[int, str] | None) -> dict[str, float]:
        return PolarsEvaluator(forest_landscape_problem(landscape, "A", others)).evaluate(xs).row(0, named=True)

    baseline = evaluate(None)
    clearfelled = evaluate(dict.fromkeys(neighbours_in_b, "clearfell"))
    selection_cut = evaluate(dict.fromkeys(neighbours_in_b, "selection_cut"))

    # clear-felling next to the boundary degrades the habitat of lot A, while selection cut, the best
    # regime for hazel grouse, improves it over the baseline of thinning
    assert clearfelled["habitat"] < baseline["habitat"] < selection_cut["habitat"]


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_coupling_is_local_to_the_boundary():
    """Test that only stands bordering another lot are affected, and only by the lots they border."""
    landscape = forest_landscape_data()

    # lot D does not border lot A, so its management does not matter to A
    all_of_d_clearfelled = {stand.id: "clearfell" for stand in landscape.lot_stands("D")}
    assert realized_values(landscape, "A", all_of_d_clearfelled) == realized_values(landscape, "A")

    everything_else_clearfelled = {stand.id: "clearfell" for stand in landscape.stands if stand.lot != "A"}
    realized = realized_values(landscape, "A", everything_else_clearfelled)
    for stand in landscape.lot_stands("A"):
        if landscape.cross_owner_neighbours(stand.id):
            assert realized[stand.id]["habitat"] != stand.values["habitat"]
        else:
            assert realized[stand.id] == stand.values


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_coupling_weight_zero_disables_coupling():
    """Test that with a zero neighbourhood weight the realized values are the stand-wise values."""
    landscape = forest_landscape_data(coupling=CouplingParameters(habitat_neighbourhood_weight=0.0))
    others = {stand.id: "clearfell" for stand in landscape.stands if stand.lot != "A"}

    realized = realized_values(landscape, "A", others)
    for stand in landscape.lot_stands("A"):
        assert realized[stand.id] == stand.values


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_ideal_and_nadir_do_not_depend_on_other_lots():
    """Test that the ideal and nadir are those of the baseline, however the other lots are managed."""
    landscape = forest_landscape_data()
    others = {stand.id: "clearfell" for stand in landscape.stands if stand.lot != "A"}

    baseline = forest_landscape_problem(landscape, "A")
    changed = forest_landscape_problem(landscape, "A", others)

    assert baseline.get_ideal_point() == changed.get_ideal_point()
    assert baseline.get_nadir_point() == changed.get_nadir_point()


@pytest.mark.testproblem
@pytest.mark.forest_problem
def test_other_lots_configuration_is_validated():
    """Test that the configuration of the other lots cannot set the lot's own stands or unknown regimes."""
    landscape = forest_landscape_data()
    own_stand = landscape.lot_stands("A")[0].id
    other_stand = landscape.lot_stands("B")[0].id

    with pytest.raises(ValueError, match="not a stand of another lot"):
        realized_values(landscape, "A", {own_stand: "clearfell"})
    with pytest.raises(ValueError, match="Unknown regime"):
        realized_values(landscape, "A", {other_stand: "burn"})
    with pytest.raises(ValueError, match="Unknown baseline regime"):
        forest_landscape_data(baseline_regime="burn")
