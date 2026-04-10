"""Tests for the LEM learning-mode abstract interface and SkopeRules implementation.

These tests operate on synthetic numpy data only -- no DESDEO problems or
evolutionary infrastructure involved. This isolates rule extraction and
solution instantiation so Phase 1 can be validated before the pub/sub
integration lands.
"""

import numpy as np
import pytest

from desdeo.emo.operators.learning_mode import (
    BaseLearningModel,
    SkopeRulesModel,
    _instantiate_rules,
    _instantiate_ruleset_rules,
)

# Skip the whole file if the optional lemoo extras aren't installed.
imodels = pytest.importorskip("imodels")


# Known high-performing rectangle for the 2D fixture.
HIGH_BOX_2D = ((2.0, 4.0), (3.0, 6.0))


def _make_rectangle_dataset(
    box: tuple[tuple[float, float], tuple[float, float]] = HIGH_BOX_2D,
    outer_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 10.0),
        (0.0, 10.0),
    ),
    n_pos: int = 500,
    n_neg: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a balanced 2D dataset where class 1 fills a known rectangle.

    Positives are sampled uniformly inside ``box``. Negatives are sampled
    uniformly inside ``outer_bounds`` but filtered so none fall inside
    ``box``.
    """
    rng = np.random.default_rng(seed)
    (x0_lo, x0_hi), (x1_lo, x1_hi) = box
    (ox0_lo, ox0_hi), (ox1_lo, ox1_hi) = outer_bounds

    x_pos = rng.uniform([x0_lo, x1_lo], [x0_hi, x1_hi], size=(n_pos, 2))

    # Rejection-sample negatives outside the positive box.
    neg_samples: list[np.ndarray] = []
    while sum(arr.shape[0] for arr in neg_samples) < n_neg:
        cand = rng.uniform([ox0_lo, ox1_lo], [ox0_hi, ox1_hi], size=(n_neg * 3, 2))
        outside = ~((cand[:, 0] >= x0_lo) & (cand[:, 0] <= x0_hi) & (cand[:, 1] >= x1_lo) & (cand[:, 1] <= x1_hi))
        neg_samples.append(cand[outside])
    x_neg = np.vstack(neg_samples)[:n_neg]

    x = np.vstack([x_pos, x_neg]).astype(float)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
    return x, y


def _fit_2d_model(seed: int = 42, **overrides) -> SkopeRulesModel:
    """Fit a ``SkopeRulesModel`` on the 2D rectangle dataset.

    ``max_samples_features=1.0`` is passed so the SkopeRules base estimator
    uses *both* features per tree. Without this, ``imodels`` has an internal
    guard that returns zero rules when base estimators happen to use only a
    single feature (see ``imodels.util.score.score_precision_recall``).
    """
    x, y = _make_rectangle_dataset(seed=seed)
    kwargs = {
        "min_precision": 0.5,
        "max_estimators": 30,
        "bootstrap": True,
        "random_state": seed,
        "max_samples_features": 1.0,
    }
    kwargs.update(overrides)
    model = SkopeRulesModel(**kwargs)
    model.fit(x, y)
    return model


def test_skope_rules_is_a_base_learning_model():
    """SkopeRulesModel must implement the abstract interface."""
    model = SkopeRulesModel(random_state=0)
    assert isinstance(model, BaseLearningModel)


def test_base_learning_model_cannot_be_instantiated():
    """BaseLearningModel is abstract and must not be directly instantiable."""
    with pytest.raises(TypeError):
        BaseLearningModel()  # type: ignore[abstract]


def test_fit_on_known_rectangle_does_not_crash():
    """Fitting on a well-separated rectangle should not raise."""
    model = _fit_2d_model()
    # With a well-separated rectangle SkopeRules should find at least one rule.
    assert len(model._classifier.rules_) > 0


def test_instantiate_before_fit_raises():
    """Calling ``instantiate`` before ``fit`` must raise a clear error."""
    model = SkopeRulesModel(random_state=0)
    with pytest.raises(RuntimeError):
        model.instantiate(2, [(0.0, 10.0), (0.0, 10.0)], n_samples=10)


def test_instantiate_shape_and_bounds_are_respected():
    """All instantiated samples must lie inside the outer variable bounds."""
    model = _fit_2d_model()
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    samples = model.instantiate(n_variables=2, variable_bounds=bounds, n_samples=500)

    # Shape: right number of columns and at least one row.
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert samples.shape[0] > 0

    # Output count may differ slightly from n_samples due to per-rule rounding.
    assert abs(samples.shape[0] - 500) <= 100

    # All samples within variable bounds.
    assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 10.0)
    assert np.all(samples[:, 1] >= 0.0) and np.all(samples[:, 1] <= 10.0)


def test_instantiate_concentrates_samples_in_high_region():
    """Most instantiated samples should fall in (or very near) the high-performing box.

    A uniform baseline over the [0,10]^2 box would yield roughly
    ``(2 * 3) / 100 = 0.06`` inside-rate. We require the model to beat this
    by a wide margin.
    """
    model = _fit_2d_model()
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    samples = model.instantiate(2, bounds, n_samples=1000)

    (x0_lo, x0_hi), (_, x1_hi) = HIGH_BOX_2D
    in_box_x0 = (samples[:, 0] >= x0_lo) & (samples[:, 0] <= x0_hi)
    # Rules tend to only bound x_1 from above, so we relax to "x_1 <= upper".
    in_box_x1_upper = samples[:, 1] <= x1_hi

    in_region_rate = float(np.mean(in_box_x0 & in_box_x1_upper))
    assert in_region_rate > 0.5, f"Expected most samples in the high-performing region, got {in_region_rate:.2%}"


def test_get_rules_description_returns_list_of_dicts():
    """``get_rules_description`` must return a non-empty list of dicts with expected keys."""
    model = _fit_2d_model()
    descriptions = model.get_rules_description()

    assert isinstance(descriptions, list)
    assert len(descriptions) > 0
    for desc in descriptions:
        assert isinstance(desc, dict)
        assert "rule" in desc
        assert "conditions" in desc
        assert "precision" in desc
        assert isinstance(desc["conditions"], list)
        assert all(isinstance(c, str) for c in desc["conditions"])
        assert desc["precision"] is None or 0.0 <= desc["precision"] <= 1.0


def test_get_rules_description_empty_before_fit():
    """Before fitting, rule descriptions should be an empty list."""
    model = SkopeRulesModel(random_state=0)
    assert model.get_rules_description() == []


def test_single_class_data_raises():
    """SkopeRules requires both classes -- fitting on a single class should raise."""
    x = np.random.default_rng(0).uniform(0, 10, size=(50, 2))
    y = np.ones(50, dtype=int)
    model = SkopeRulesModel(random_state=0, max_samples_features=1.0)
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_fit_with_very_few_samples_does_not_crash():
    """The model should not raise on tiny datasets even if it learns no rules."""
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.uniform([2.0, 3.0], [4.0, 6.0], size=(5, 2)),
            rng.uniform([6.0, 7.0], [9.0, 9.0], size=(5, 2)),
        ]
    )
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    model = SkopeRulesModel(
        min_precision=0.1,
        max_estimators=5,
        bootstrap=False,
        random_state=0,
        max_samples=1.0,
        max_samples_features=1.0,
    )
    model.fit(x, y)  # must not raise

    # instantiate must still produce a valid numpy array of the right shape.
    samples = model.instantiate(2, [(0.0, 10.0), (0.0, 10.0)], n_samples=50)
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 10.0)
    assert np.all(samples[:, 1] >= 0.0) and np.all(samples[:, 1] <= 10.0)


def test_instantiate_fallback_when_no_rules_learned():
    """If no rules were learned the model must gracefully fall back to uniform sampling."""
    # Fit normally, then wipe the learned rules to simulate "no usable rules".
    model = _fit_2d_model()
    model._classifier.rules_ = []

    bounds = [(0.0, 10.0), (0.0, 10.0)]
    samples = model.instantiate(2, bounds, n_samples=200)

    assert samples.shape == (200, 2)
    assert np.all(samples >= 0.0) and np.all(samples <= 10.0)


def test_instantiate_rules_single_feature_bounds():
    """Given a hand-crafted rule ``2 < x_0 < 5`` samples must fall in [2, 5].

    The other feature has no rule so it should be sampled from the full bounds.
    """
    rule = {("x_0", "<"): "5.0", ("x_0", ">"): "2.0"}
    feature_limits = [(0.0, 10.0), (0.0, 10.0)]
    samples = _instantiate_rules(
        rule,
        n_features=2,
        feature_limits=feature_limits,
        n_samples=1000,
        rng=np.random.default_rng(0),
    )

    assert samples.shape == (1000, 2)
    # x_0 must be strictly inside [2, 5] (uniform is inclusive of low exclusive of high).
    assert np.all(samples[:, 0] >= 2.0)
    assert np.all(samples[:, 0] <= 5.0)
    # x_1 should use the full bounds.
    assert np.all(samples[:, 1] >= 0.0)
    assert np.all(samples[:, 1] <= 10.0)
    # x_1 should genuinely span the full bound range (crude coverage check).
    assert samples[:, 1].min() < 2.0
    assert samples[:, 1].max() > 8.0


def test_instantiate_rules_rule_bounds_are_clipped_to_feature_limits():
    """Rule bounds that escape the variable bounds must be clipped to them."""
    # Rule wants x_0 in [-5, 20] -- should be clipped to [0, 10].
    rule = {("x_0", ">"): "-5.0", ("x_0", "<"): "20.0"}
    feature_limits = [(0.0, 10.0)]
    samples = _instantiate_rules(
        rule,
        n_features=1,
        feature_limits=feature_limits,
        n_samples=500,
        rng=np.random.default_rng(1),
    )
    assert np.all(samples[:, 0] >= 0.0)
    assert np.all(samples[:, 0] <= 10.0)


def test_instantiate_rules_unknown_feature_uses_full_bounds():
    """Features not mentioned in the rule must be sampled from their full bounds."""
    rule = {("x_0", "<"): "5.0"}
    feature_limits = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    samples = _instantiate_rules(
        rule,
        n_features=3,
        feature_limits=feature_limits,
        n_samples=2000,
        rng=np.random.default_rng(2),
    )
    assert samples.shape == (2000, 3)
    assert np.all(samples[:, 0] <= 5.0)
    # Features 1 and 2 should cover a wide range.
    for i in (1, 2):
        assert samples[:, i].min() < 2.0
        assert samples[:, i].max() > 8.0


def test_instantiate_ruleset_distributes_samples_by_weight():
    """Ruleset instantiation should produce roughly n_samples total rows, partitioned by weight."""
    rules = [
        {("x_0", "<"): "5.0", ("x_0", ">"): "2.0"},
        {("x_0", "<"): "9.0", ("x_0", ">"): "7.0"},
    ]
    weights = [0.75, 0.25]
    feature_limits = [(0.0, 10.0)]
    samples = _instantiate_ruleset_rules(
        rules,
        weights,
        n_features=1,
        feature_limits=feature_limits,
        n_samples=400,
        rng=np.random.default_rng(3),
    )
    # Total should be close to n_samples.
    assert abs(samples.shape[0] - 400) <= 4
    # Split samples back into their rule regions.
    in_first = ((samples[:, 0] >= 2.0) & (samples[:, 0] <= 5.0)).sum()
    in_second = ((samples[:, 0] >= 7.0) & (samples[:, 0] <= 9.0)).sum()
    assert in_first + in_second == samples.shape[0]
    # First rule should get ~3x the samples of the second.
    assert in_first > in_second * 2
