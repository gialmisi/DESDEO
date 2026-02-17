"""Tests for the XLEMOO learning mode and template3."""

import numpy as np
import pytest

from desdeo.emo.options.algorithms import xlemoo_rvea_options
from desdeo.emo.options.templates import emo_constructor
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
    """Test that tree rule instantiation produces valid samples."""
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
    """Test single rule instantiation."""
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


# --- XLEMOOSelector Unit Tests ---


@pytest.mark.ea
def test_xlemoo_selector_indicator_naive_sum():
    """Test that the naive_sum indicator works correctly."""
    from desdeo.emo.options.xlemoo_selection import XLEMOOSelectorOptions, _build_indicator

    options = XLEMOOSelectorOptions(fitness_indicator="naive_sum")
    indicator = _build_indicator(options)

    targets = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    result = indicator(targets)

    assert result.shape == (2,)
    assert result[0] == pytest.approx(6.0)
    assert result[1] == pytest.approx(1.5)


@pytest.mark.ea
def test_xlemoo_selector_indicator_asf():
    """Test that the ASF indicator works correctly."""
    from desdeo.emo.options.xlemoo_selection import XLEMOOSelectorOptions, _build_indicator

    options = XLEMOOSelectorOptions(
        fitness_indicator="asf",
        asf_reference_point=[0.0, 0.0],
        asf_weights=[1.0, 1.0],
    )
    indicator = _build_indicator(options)

    targets = np.array([[1.0, 2.0], [0.5, 0.5]])
    result = indicator(targets)

    assert result.shape == (2,)
    # max(1,2) + rho*sum(1,2) should be > max(0.5,0.5) + rho*sum(0.5,0.5)
    assert result[0] > result[1]


# --- Integration Test ---


@pytest.mark.ea
def test_xlemoo_rvea_dtlz2():
    """Integration test: run XLEMOO with RVEA on DTLZ2(3 obj, 12 vars)."""
    problem = dtlz2(n_objectives=3, n_variables=12)

    options = xlemoo_rvea_options()
    # Reduce generations for faster test
    options.template.termination.max_generations = 50

    solver, extras = emo_constructor(emo_options=options, problem=problem)

    results = solver()

    # Verify we got results
    assert results.optimal_variables is not None
    assert results.optimal_outputs is not None
    assert len(results.optimal_variables) > 0

    # Check that solutions approximate the unit sphere
    outputs = results.optimal_outputs
    obj_cols = [f"f_{i}" for i in range(1, 4)]
    obj_values = outputs[obj_cols].to_numpy()

    norms = np.sqrt(np.sum(obj_values**2, axis=1))
    # Median norm should be reasonably close to 1 (the true Pareto front)
    assert norms.min() > 0, "Some solutions have zero norm"
    assert np.median(norms) < 1.5, f"Median norm {np.median(norms)} too far from unit sphere"
