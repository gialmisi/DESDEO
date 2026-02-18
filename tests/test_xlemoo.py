"""Tests for the XLEMOO learning mode and template3.

Tests are structured to reflect the original XLEMOO implementation
(Misitano, 2024) while adapting to the DESDEO 2 framework.  We use DTLZ2
as the test problem (the original used Vehicle Crash Worthiness, which is
not yet available in DESDEO 2).
"""

import numpy as np
import polars as pl
import pytest

from desdeo.emo.options.algorithms import xlemoo_options
from desdeo.emo.options.scalarization_selection import ScalarizationSelectorOptions, ScalarizationSpec
from desdeo.emo.options.templates import emo_constructor
from desdeo.emo.options.xlemoo_selection import XLEMOOSelectorOptions
from desdeo.explanations.rule_interpreters import (
    find_all_paths,
    instantiate_rules,
    instantiate_ruleset_rules,
    instantiate_tree_rules,
)
from desdeo.problem.testproblems import dtlz2

# --- Rule Interpreter Unit Tests ---


@pytest.mark.ea
def test_find_all_paths():
    """Test that tree paths are correctly extracted from a DecisionTreeClassifier."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(42)
    X = rng.random((100, 3))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    paths = find_all_paths(tree)

    assert len(paths) > 0
    for p in paths:
        assert "rules" in p
        assert "samples" in p
        assert "impurity" in p
        assert "classification" in p
        assert p["classification"] in tree.classes_


@pytest.mark.ea
def test_instantiate_tree_rules():
    """Test that tree rule instantiation produces valid samples within bounds."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(42)
    X = rng.random((200, 4))
    y = np.where(X.sum(axis=1) > 2.0, 1, -1)

    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X, y)

    paths = find_all_paths(tree)
    limits = [(0.0, 1.0)] * 4

    instantiated = instantiate_tree_rules(paths, 4, limits, 10, 1, rng)

    # Should be 3D: (n_matching_paths, n_samples, n_features)
    assert instantiated.ndim == 3
    if instantiated.size > 0:
        assert instantiated.shape[1] == 10
        assert instantiated.shape[2] == 4
        # All within bounds
        assert np.all(instantiated >= 0.0)
        assert np.all(instantiated <= 1.0)


@pytest.mark.ea
def test_instantiate_rules():
    """Test single rule instantiation respects constraints."""
    rng = np.random.default_rng(42)
    rules = {("x_0", "<="): "0.5", ("x_1", ">="): "0.3"}
    limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    samples = instantiate_rules(rules, 3, limits, 50, rng)

    assert samples.shape == (50, 3)
    # x_0 should be <= 0.5
    assert np.all(samples[:, 0] <= 0.5)
    # x_1 should be >= 0.3
    assert np.all(samples[:, 1] >= 0.3)
    # x_2 unconstrained, just within bounds
    assert np.all(samples[:, 2] >= 0.0)
    assert np.all(samples[:, 2] <= 1.0)


@pytest.mark.ea
def test_instantiate_ruleset_rules():
    """Test ruleset instantiation distributes samples by weight."""
    rng = np.random.default_rng(42)
    rules = [
        {("x_0", "<="): "0.5"},
        {("x_0", ">="): "0.5"},
    ]
    weights = [3.0, 1.0]
    limits = [(0.0, 1.0), (0.0, 1.0)]

    samples = instantiate_ruleset_rules(rules, weights, 2, limits, 100, rng)

    assert samples.shape[1] == 2
    # Total should be approximately 100
    assert 90 <= samples.shape[0] <= 110


# --- ScalarizationSelector Unit Test ---


@pytest.mark.ea
def test_scalarization_selector_basic():
    """Unit test: ScalarizationSelector ranks by scalarization column."""
    from desdeo.emo.operators.scalarization_selection import ScalarizationSelector
    from desdeo.tools.patterns import Publisher
    from desdeo.tools.scalarization import add_weighted_sums

    problem = dtlz2(n_objectives=3, n_variables=6)
    # Add a scalarization function so target_symbols picks it up
    weights = {obj.symbol: 1.0 for obj in problem.objectives}
    problem, _ = add_weighted_sums(problem, "scal_fitness", weights)

    publisher = Publisher()
    selector = ScalarizationSelector(
        problem=problem,
        verbosity=0,
        publisher=publisher,
        population_size=5,
        seed=42,
    )

    # target_symbols should point to the scalarization column
    assert selector.target_symbols == ["scal_fitness"]

    # Create mock data: 10 solutions with known scalarization values
    rng = np.random.default_rng(42)
    n = 10
    var_data = {f"x_{i}": rng.random(n).tolist() for i in range(1, 7)}
    solutions = pl.DataFrame(var_data)

    # Mock outputs with the scalarization column (lower is better)
    scal_values = list(range(n))  # 0, 1, 2, ..., 9
    rng.shuffle(scal_values)
    output_data = {"scal_fitness": [float(v) for v in scal_values]}
    outputs = pl.DataFrame(output_data)

    # First call (empty offspring)
    empty_sol = pl.DataFrame(schema=solutions.schema)
    empty_out = pl.DataFrame(schema=outputs.schema)
    sel_sol, sel_out = selector.do((solutions, outputs), (empty_sol, empty_out))

    # Should return all 10 (first iteration returns all parents)
    assert len(sel_sol) == n

    # Second call with offspring — should truncate to 5
    offsprings_sol = pl.DataFrame({f"x_{i}": rng.random(5).tolist() for i in range(1, 7)})
    offsprings_out = pl.DataFrame({"scal_fitness": [10.0, 11.0, 12.0, 13.0, 14.0]})

    sel_sol2, sel_out2 = selector.do((solutions, outputs), (offsprings_sol, offsprings_out))
    assert len(sel_sol2) == 5

    # The selected scalarization values should be the 5 lowest
    selected_scal = sel_out2["scal_fitness"].to_numpy()
    assert np.all(selected_scal == np.sort(selected_scal))
    assert selected_scal[-1] <= 4.0  # worst of top 5 from original 0..9


# --- Integration Tests ---


@pytest.mark.ea
def test_xlemoo_weighted_sums_dtlz2():
    """Integration test: XLEMOO on DTLZ2 using weighted sums scalarization."""
    problem = dtlz2(n_objectives=3, n_variables=12)

    options = xlemoo_options()
    options.template.termination.max_generations = 50
    options.template.learning_instantiator = XLEMOOSelectorOptions(
        ml_model_type="DecisionTree",
        h_split=0.2,
        l_split=0.2,
        instantiation_factor=10.0,
        unique_only=True,
    )
    options.template.darwin_iterations_per_cycle = 19
    options.template.learning_iterations_per_cycle = 1

    solver, extras = emo_constructor(emo_options=options, problem=problem)
    results = solver()

    assert results.optimal_variables is not None
    assert results.optimal_outputs is not None
    assert len(results.optimal_variables) > 0

    # Check solutions approximate the unit sphere (DTLZ2 Pareto front)
    obj_cols = [f"f_{i}" for i in range(1, 4)]
    obj_values = results.optimal_outputs[obj_cols].to_numpy()
    norms = np.sqrt(np.sum(obj_values**2, axis=1))
    assert np.median(norms) < 1.5, f"Median norm {np.median(norms)} too far from unit sphere"


@pytest.mark.ea
def test_xlemoo_asf_scalarization():
    """Integration test: XLEMOO on DTLZ2 using ASF scalarization.

    Verifies that the ScalarizationSpec(type='asf') path works end-to-end.
    """
    problem = dtlz2(n_objectives=3, n_variables=6)

    options = xlemoo_options()
    options.template.termination.max_generations = 30
    options.template.scalarization = ScalarizationSpec(
        type="asf",
        symbol="scal_fitness",
        reference_point=[0.5, 0.5, 0.5],
    )
    options.template.learning_instantiator = XLEMOOSelectorOptions(
        ml_model_type="DecisionTree",
        h_split=0.2,
        l_split=0.2,
        instantiation_factor=10.0,
        unique_only=True,
    )

    solver, extras = emo_constructor(emo_options=options, problem=problem)
    results = solver()

    assert results.optimal_variables is not None
    assert len(results.optimal_variables) > 0

    # Archive should contain the scalarization column
    archive_df = extras.learning_archive.solutions
    assert "scal_fitness" in archive_df.columns

    # The learning_instantiator should be accessible and have last_classifier/last_rules
    assert extras.learning_instantiator is not None
    assert extras.learning_instantiator.last_classifier is not None
    assert extras.learning_instantiator.last_rules is not None


@pytest.mark.ea
def test_xlemoo_with_skope_rules():
    """Integration test using SkopeRulesClassifier, the primary ML model in the paper."""
    problem = dtlz2(n_objectives=3, n_variables=6)

    options = xlemoo_options()
    options.template.termination.max_generations = 30
    options.template.learning_instantiator = XLEMOOSelectorOptions(
        ml_model_type="SkopeRules",
        ml_model_kwargs={
            "precision_min": 0.1,
            "n_estimators": 30,
            "max_depth": None,
        },
        h_split=0.2,
        l_split=0.2,
        instantiation_factor=10.0,
        unique_only=True,
    )

    solver, extras = emo_constructor(emo_options=options, problem=problem)
    results = solver()

    assert results.optimal_variables is not None
    assert len(results.optimal_variables) > 0

    # SkopeRules should produce (ruleset, precisions) in last_rules
    ruleset, precisions = extras.learning_instantiator.last_rules
    assert len(ruleset) > 0
    assert len(precisions) == len(ruleset)
    # Each rule should be a dict mapping (feature_name, op) -> threshold
    for rule in ruleset:
        assert isinstance(rule, dict)


@pytest.mark.ea
def test_xlemoo_with_slipper():
    """Integration test using SlipperClassifier."""
    problem = dtlz2(n_objectives=3, n_variables=6)

    options = xlemoo_options()
    options.template.termination.max_generations = 30
    options.template.learning_instantiator = XLEMOOSelectorOptions(
        ml_model_type="Slipper",
        ml_model_kwargs={"n_estimators": 5},
        h_split=0.2,
        l_split=0.2,
        instantiation_factor=10.0,
    )

    solver, extras = emo_constructor(emo_options=options, problem=problem)
    results = solver()

    assert results.optimal_variables is not None
    assert len(results.optimal_variables) > 0


@pytest.mark.ea
def test_xlemoo_with_boosted_rules():
    """Integration test using BoostedRulesClassifier."""
    problem = dtlz2(n_objectives=3, n_variables=6)

    options = xlemoo_options()
    options.template.termination.max_generations = 30
    options.template.learning_instantiator = XLEMOOSelectorOptions(
        ml_model_type="BoostedRules",
        ml_model_kwargs={"n_estimators": 5},
        h_split=0.2,
        l_split=0.2,
        instantiation_factor=10.0,
    )

    solver, extras = emo_constructor(emo_options=options, problem=problem)
    results = solver()

    assert results.optimal_variables is not None
    assert len(results.optimal_variables) > 0


@pytest.mark.ea
def test_xlemoo_learning_archive_populated():
    """Verify that the learning archive accumulates solutions and contains the scalarization column."""
    problem = dtlz2(n_objectives=3, n_variables=6)

    options = xlemoo_options()
    options.template.termination.max_generations = 30
    options.template.learning_instantiator = XLEMOOSelectorOptions(
        h_split=0.2,
        l_split=0.2,
    )

    solver, extras = emo_constructor(emo_options=options, problem=problem)
    results = solver()

    # The learning archive should contain solutions from multiple generations
    archive_df = extras.learning_archive.solutions
    assert archive_df is not None
    assert len(archive_df) > 0

    # Archive should span multiple generations
    generations = archive_df["generation"].unique().to_list()
    assert len(generations) > 1, "Archive should contain solutions from multiple generations"

    # Archive should contain variable columns, and the scalarization column
    var_cols = [f"x_{i}" for i in range(1, 7)]
    for col in var_cols:
        assert col in archive_df.columns, f"Missing column: {col}"

    # The scalarization column should be present (added by emo_constructor)
    assert "scal_fitness" in archive_df.columns, "Missing scalarization column: scal_fitness"


@pytest.mark.ea
def test_xlemoo_rule_extraction_from_archive():
    """Test post-hoc rule extraction from the learning archive.

    Uses the scalarization column from the archive (computed by the evaluator)
    to rank solutions into H/L groups, then trains a DecisionTreeClassifier.
    """
    from sklearn.tree import DecisionTreeClassifier

    from desdeo.explanations.rule_interpreters import find_all_paths

    problem = dtlz2(n_objectives=3, n_variables=6)

    options = xlemoo_options()
    options.template.termination.max_generations = 50
    options.template.learning_instantiator = XLEMOOSelectorOptions(
        h_split=0.2,
        l_split=0.2,
    )

    solver, extras = emo_constructor(emo_options=options, problem=problem)
    results = solver()

    # Extract data from the learning archive
    archive_df = extras.learning_archive.solutions
    var_cols = [f"x_{i}" for i in range(1, 7)]

    X = archive_df[var_cols].to_numpy()
    # Use the scalarization column for ranking (same as selector and instantiator)
    fitness = archive_df["scal_fitness"].to_numpy()

    sorted_idx = np.argsort(fitness)
    n = len(sorted_idx)
    h_count = max(1, int(0.2 * n))
    l_count = max(1, int(0.2 * n))

    h_idx = sorted_idx[:h_count]
    l_idx = sorted_idx[-l_count:]

    X_train = np.vstack((X[h_idx], X[l_idx]))
    y_train = np.hstack((np.ones(h_count, dtype=int), -np.ones(l_count, dtype=int)))

    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)

    paths = find_all_paths(tree)
    assert len(paths) > 0

    # At least one path should be classified as H-group (good)
    h_paths = [p for p in paths if p["classification"] == 1]
    assert len(h_paths) > 0, "Expected at least one path classified as H-group"

    # Each H-group path should have rules
    for p in h_paths:
        assert len(p["rules"]) > 0, "H-group paths should have at least one rule"
