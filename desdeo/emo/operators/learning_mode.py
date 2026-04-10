"""Abstract learning model interface and concrete implementations for LEM learning mode.

This module defines the :class:`BaseLearningModel` abstract base class that any
learnable evolutionary multiobjective optimization (LEMOO) learning-mode model
must implement, plus a :class:`SkopeRulesModel` concrete implementation that
wraps :class:`imodels.SkopeRulesClassifier`.

The rule-extraction and rule-instantiation logic is ported from the reference
XLEMOO implementation (``XLEMOO/ruleset_interpreter.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Type alias for a rule: dict keyed by (feature_name, comparison_op) -> value-as-str.
Rules = dict[tuple[str, str], str]


class BaseLearningModel(ABC):
    """Abstract interface for ML models used in LEM learning mode.

    Designed to be swappable; BRB will implement this interface later.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Train the model on labeled data.

        Args:
            X: ``(n_samples, n_features)`` decision variable values.
            y: ``(n_samples,)`` binary labels -- 1 for high-performing, 0 for
                low-performing.
        """
        ...

    @abstractmethod
    def instantiate(
        self,
        n_variables: int,
        variable_bounds: list[tuple[float, float]],
        n_samples: int,
    ) -> np.ndarray:
        """Generate new solution candidates based on the learned model.

        Args:
            n_variables: Number of decision variables.
            variable_bounds: List of ``(lower, upper)`` bounds per variable.
            n_samples: How many new candidates to generate.

        Returns:
            ``(n_generated, n_variables)`` array of new candidate solutions.
            ``n_generated`` may differ slightly from ``n_samples`` due to
            rounding across rules.
        """
        ...

    @abstractmethod
    def get_rules_description(self) -> list[dict]:
        """Return human-readable description of learned rules for explainability.

        Returns:
            List of dicts, each describing one rule with keys like
            ``conditions`` (list of str), ``precision`` (float), etc.
        """
        ...


def _extract_skoped_rules(classifier) -> tuple[list[Rules], list[float]]:
    """Extract ``agg_dict`` rules and their precisions from a fitted SkopeRules classifier.

    Ported from ``XLEMOO/ruleset_interpreter.py::extract_skoped_rules``.
    """
    precisions = [rule.args[0] for rule in classifier.rules_]
    rules = [rule.agg_dict for rule in classifier.rules_]
    return rules, precisions


def _instantiate_rules(
    rule: Rules,
    n_features: int,
    feature_limits: list[tuple[float, float]],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Instantiate ``n_samples`` decision-variable vectors from a single rule.

    For each feature the rule's bounds are intersected with ``feature_limits``
    and samples are drawn uniformly inside the resulting interval. Features
    not constrained by the rule are sampled uniformly across their full bounds.

    Ported from ``XLEMOO/ruleset_interpreter.py::instantiate_rules``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_samples <= 0:
        return np.zeros((0, n_features))

    # Convert each rule entry into (feature_index, op, value).
    index_op_value = [(int(key[0].split("_")[-1]), key[1], float(rule[key])) for key in rule]

    # Group operator/value tuples by feature index.
    op_value_per_index: dict[int, list[tuple[str, float]]] = {}
    for i, op, val in index_op_value:
        op_value_per_index.setdefault(i, []).append((op, val))

    new_samples = np.zeros((n_samples, n_features))

    for feature_i in range(n_features):
        current_min = feature_limits[feature_i][0]
        current_max = feature_limits[feature_i][1]

        if feature_i not in op_value_per_index:
            new_samples[:, feature_i] = rng.uniform(current_min, current_max, n_samples)
            continue

        for op, value in op_value_per_index[feature_i]:
            if op in ("<", "<="):
                if current_min < value < current_max:
                    current_max = value
            elif op in (">", ">="):
                if current_min < value < current_max:
                    current_min = value
            elif op in ("=", "=="):
                current_min = value
                current_max = value
            # Unknown ops are silently ignored (matches reference behavior).

        if current_min == current_max:
            new_samples[:, feature_i] = current_min
        else:
            new_samples[:, feature_i] = rng.uniform(current_min, current_max, n_samples)

    return new_samples


def _instantiate_ruleset_rules(
    rules: list[Rules],
    weights: list[float],
    n_features: int,
    feature_limits: list[tuple[float, float]],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Distribute ``n_samples`` across rules proportional to their weights and instantiate.

    Rules with non-positive weight are dropped. The total number of generated
    samples may differ slightly from ``n_samples`` due to per-rule rounding.

    Ported from ``XLEMOO/ruleset_interpreter.py::instantiate_ruleset_rules``.
    """
    if len(rules) == 0 or len(weights) == 0:
        return np.zeros((0, n_features))

    # Truncate rules to match weights length (mirrors reference behavior).
    if len(weights) < len(rules):
        rules = rules[: len(weights)]

    w_arr = np.asarray(weights, dtype=float)
    positive_mask = w_arr >= 0

    if not np.any(positive_mask) or np.sum(w_arr[positive_mask]) <= 0:
        return np.zeros((0, n_features))

    fractions = w_arr[positive_mask] / np.sum(w_arr[positive_mask])
    n_per_rule = np.round(fractions * n_samples).astype(int)

    rules_pos_w = [r for r, keep in zip(rules, positive_mask, strict=True) if keep]

    instantiated = [
        _instantiate_rules(rule, n_features, feature_limits, int(n_per_rule[i]), rng=rng)
        for i, rule in enumerate(rules_pos_w)
    ]
    instantiated = [arr for arr in instantiated if arr.shape[0] > 0]

    if not instantiated:
        return np.zeros((0, n_features))

    return np.vstack(instantiated)


class SkopeRulesModel(BaseLearningModel):
    """Concrete :class:`BaseLearningModel` backed by ``imodels.SkopeRulesClassifier``.

    Trains a SkopeRules classifier on labeled decision-variable data, then
    extracts the learned rules and uses them to generate new candidate
    solutions inside the rule-defined regions.
    """

    def __init__(
        self,
        min_precision: float = 0.1,
        max_estimators: int = 30,
        bootstrap: bool = True,
        random_state: int | None = None,
        **skope_kwargs,
    ) -> None:
        """Construct an unfitted SkopeRules learning model.

        Args:
            min_precision: Minimum precision a rule must achieve to be retained.
            max_estimators: Maximum number of base estimators built by SkopeRules.
            bootstrap: Whether base estimators use bootstrap sampling.
            random_state: Optional random seed forwarded to SkopeRules and used
                for sampling during instantiation.
            **skope_kwargs: Additional kwargs forwarded to
                :class:`imodels.SkopeRulesClassifier`.
        """
        # Imported lazily so that the rest of desdeo doesn't pay the import cost.
        from imodels import SkopeRulesClassifier  # noqa: PLC0415

        self._classifier = SkopeRulesClassifier(
            precision_min=min_precision,
            n_estimators=max_estimators,
            bootstrap=bootstrap,
            random_state=random_state,
            **skope_kwargs,
        )
        self._fitted = False
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Train the wrapped SkopeRules classifier.

        Args:
            X: ``(n_samples, n_features)`` decision-variable matrix.
            y: ``(n_samples,)`` binary labels.
        """
        x = np.asarray(X)
        y = np.asarray(y)
        # SkopeRules expects feature names of the form ``x_i`` so the rule
        # extractor downstream can recover the feature index.
        feature_names = [f"x_{i}" for i in range(X.shape[1])]
        self._classifier.fit(x, y, feature_names=feature_names)
        self._fitted = True

    def instantiate(
        self,
        n_variables: int,
        variable_bounds: list[tuple[float, float]],
        n_samples: int,
    ) -> np.ndarray:
        """Generate new candidates by sampling inside the learned rules' regions.

        If the classifier did not learn any usable rules (e.g. degenerate
        single-class training data), candidates are sampled uniformly across
        the variable bounds as a graceful fallback.
        """
        if not self._fitted:
            raise RuntimeError("SkopeRulesModel.instantiate called before fit().")

        rules, precisions = _extract_skoped_rules(self._classifier)

        if not rules:
            # Graceful fallback: uniform sampling across the box.
            lows = np.array([b[0] for b in variable_bounds])
            highs = np.array([b[1] for b in variable_bounds])
            return self._rng.uniform(lows, highs, size=(n_samples, n_variables))

        return _instantiate_ruleset_rules(
            rules,
            precisions,
            n_variables,
            variable_bounds,
            n_samples,
            rng=self._rng,
        )

    def get_rules_description(self) -> list[dict]:
        """Return a list of dicts describing each learned rule.

        Each dict has:
            - ``rule``: the original SkopeRules string representation
            - ``conditions``: list of human-readable condition strings
            - ``precision``: the rule's precision on the training data
            - ``recall``: the rule's recall (if reported by SkopeRules)
            - ``n_points``: support count (if reported by SkopeRules)
        """
        if not self._fitted:
            return []

        descriptions: list[dict] = []
        for rule in self._classifier.rules_:
            args = getattr(rule, "args", ())
            description = {
                "rule": str(rule),
                "conditions": [c.strip() for c in str(rule).split(" and ")],
                "precision": float(args[0]) if len(args) > 0 else None,
                "recall": float(args[1]) if len(args) > 1 else None,
                "n_points": int(args[2]) if len(args) > 2 else None,  # noqa: PLR2004
            }
            descriptions.append(description)
        return descriptions
