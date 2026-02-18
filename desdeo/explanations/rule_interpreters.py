"""Rule extraction and instantiation from interpretable ML classifiers.

Provides utilities for extracting decision rules from trained classifiers
(DecisionTreeClassifier, SlipperClassifier, BoostedRulesClassifier, SkopeRulesClassifier)
and instantiating new decision variable vectors based on those rules.

Reference:
    Misitano, G. (2024). Towards Explainable Multiobjective Optimization:
    XLEMOO. ACM Trans. Evol. Learn. Optim., 4(1).
    https://doi.org/10.1145/3626104
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.random import Generator


class TreePath(TypedDict):
    """A typed dictionary describing a decision path in a decision tree.

    Attributes:
        rules: A list where each entry is [feature_index, comparison_op, threshold].
            comparison_op is either 'lte' (less than or equal) or 'gt' (greater than).
        samples: The number of samples that reached the leaf node.
        impurity: Impurity of the leaf node.
        classification: The predicted class at the leaf node.
    """

    rules: list
    samples: int
    impurity: float
    classification: int


def find_all_paths(tree) -> list[TreePath]:
    """Extract all decision paths from a trained sklearn DecisionTreeClassifier.

    Args:
        tree: A trained sklearn DecisionTreeClassifier instance.

    Returns:
        A list of TreePath dictionaries, one per leaf node.
    """
    paths: list[TreePath] = []

    def traverse_tree(tree, rules: list, node_id: int) -> None:
        # Leaf node: children_left == children_right
        if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
            classification = tree.classes_[np.argmax(tree.tree_.value[node_id])]
            paths.append(
                TreePath(
                    rules=rules,
                    samples=tree.tree_.weighted_n_node_samples[node_id],
                    impurity=tree.tree_.impurity[node_id],
                    classification=classification,
                )
            )
            return

        threshold = tree.tree_.threshold[node_id]
        feature = tree.tree_.feature[node_id]
        rule_left = [feature, "lte", threshold]
        rule_right = [feature, "gt", threshold]

        left_id = tree.tree_.children_left[node_id]
        right_id = tree.tree_.children_right[node_id]

        if node_id == 0:
            left_rules = [rule_left]
            right_rules = [rule_right]
        else:
            left_rules = rules + [rule_left]
            right_rules = rules + [rule_right]

        traverse_tree(tree, left_rules, left_id)
        traverse_tree(tree, right_rules, right_id)

    traverse_tree(tree, [], 0)
    return paths


def instantiate_tree_rules(
    paths: list[TreePath],
    n_features: int,
    feature_limits: list[tuple[float, float]],
    n_samples: int,
    desired_classification: int,
    rng: Generator,
) -> np.ndarray:
    """Instantiate new samples from decision tree paths matching a desired classification.

    For each matching path, generates n_samples new solutions where features mentioned
    in rules are sampled within rule-constrained bounds and unconstrained features are
    sampled uniformly within their limits.

    Args:
        paths: Decision tree paths from find_all_paths().
        n_features: Number of decision variables.
        feature_limits: Lower and upper bounds per feature as (low, high) tuples.
        n_samples: Number of samples to instantiate per matching path.
        desired_classification: Only paths predicting this class are used.
        rng: A numpy random Generator for reproducibility.

    Returns:
        A 3D array of shape (n_matching_paths, n_samples, n_features).
        Returns an empty 3D array if no paths match.
    """
    n_matching_paths = sum(1 for p in paths if p["classification"] == desired_classification)

    if n_matching_paths == 0:
        return np.atleast_3d([])

    limits = np.array(feature_limits)
    instantiated = np.atleast_3d(rng.uniform(limits[:, 0], limits[:, 1], (n_matching_paths, n_samples, n_features)))

    path_i = 0
    for p in paths:
        if p["classification"] == desired_classification:
            for rule in p["rules"]:
                feature = rule[0]
                comp = rule[1]
                threshold = rule[2]

                if comp == "gt":
                    instantiated[path_i, :, feature] = rng.uniform(threshold, feature_limits[feature][1], n_samples)
                else:
                    instantiated[path_i, :, feature] = rng.uniform(feature_limits[feature][0], threshold, n_samples)
            path_i += 1

    return instantiated


Rules = dict[tuple[str, str], str]


def extract_slipper_rules(classifier) -> tuple[list[Rules], list[float]]:
    """Extract rules and weights from a trained SlipperClassifier.

    Supports the current imodels API where rules are stored in
    ``classifier.estimators_[i].rule`` as lists of dicts with
    ``feature``, ``operator``, and ``pivot`` keys.

    Args:
        classifier: A trained SlipperClassifier instance.

    Returns:
        A tuple of (rules_list, weights) where each rule is a dict mapping
        (feature_name, comparison_op) to threshold value strings.
    """
    weights = list(classifier.estimator_weights_)

    if not weights:
        weights = [1.0] * len(classifier.estimators_)

    rules: list[Rules] = []
    for estimator in classifier.estimators_:
        rule_dict: Rules = {}
        for condition in estimator.rule:
            feature_name = f"X{condition['feature']}"
            op = condition["operator"]
            pivot = str(condition["pivot"])
            rule_dict[(feature_name, op)] = pivot
        rules.append(rule_dict)

    return rules, weights


def extract_boosted_rules(classifier) -> tuple[list[Rules], list[float]]:
    """Extract rules and weights from a trained BoostedRulesClassifier.

    BoostedRulesClassifier uses ``DecisionTreeClassifier(max_depth=1)`` stumps
    as its base estimators.  Each stump splits on a single feature, which we
    convert to the same ``Rules`` dict format used by the other extractors.

    Args:
        classifier: A trained BoostedRulesClassifier instance.

    Returns:
        A tuple of (rules_list, weights) where each rule is a dict mapping
        (feature_name, comparison_op) to threshold value strings.
    """
    weights = list(classifier.estimator_weights_)

    if not weights:
        weights = [1.0] * len(classifier.estimators_)

    rules: list[Rules] = []
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        feature_idx = tree.feature[0]
        threshold = tree.threshold[0]
        feature_name = f"X{feature_idx}"

        # A depth-1 stump has one split: left child (<=) and right child (>).
        # Determine which child is the positive class.
        left_class = np.argmax(tree.value[tree.children_left[0]])
        if left_class == 1:
            rule_dict: Rules = {(feature_name, "<="): str(threshold)}
        else:
            rule_dict = {(feature_name, ">"): str(threshold)}
        rules.append(rule_dict)

    return rules, weights


def extract_skoped_rules(classifier) -> tuple[list[Rules], list[float]]:
    """Extract rules and precisions from a trained SkopeRulesClassifier.

    Args:
        classifier: A trained SkopeRulesClassifier instance.

    Returns:
        A tuple of (rules_list, precisions).
    """
    precisions = [rule.args[0] for rule in classifier.rules_]
    rules = [rule.agg_dict for rule in classifier.rules_]
    return rules, precisions


def instantiate_rules(
    rules: Rules,
    n_features: int,
    feature_limits: list[tuple[float, float]],
    n_samples: int,
    rng: Generator,
) -> np.ndarray:
    """Instantiate samples from a single rule dictionary.

    For each feature, narrows bounds based on applicable rules, then samples uniformly.
    Features without rules are sampled between their natural limits.

    Args:
        rules: A dict mapping (feature_name, comparison_op) to threshold value.
            Feature names are expected as "x_i" (zero-indexed).
        n_features: Number of decision variables.
        feature_limits: Lower and upper bounds per feature.
        n_samples: How many samples to generate.
        rng: A numpy random Generator for reproducibility.

    Returns:
        A 2D array of shape (n_samples, n_features).
    """

    def _parse_feature_index(name: str) -> int:
        """Parse feature index from names like 'x_0', 'X0', 'X10', 'x_12'."""
        # Try underscore-separated format first: "x_0", "x_12"
        if "_" in name:
            return int(name.split("_")[-1])
        # Default sklearn/imodels format: "X0", "X10"
        import re

        match = re.search(r"(\d+)$", name)
        if match:
            return int(match.group(1))
        msg = f"Cannot parse feature index from name: {name!r}"
        raise ValueError(msg)

    index_op_value = [(_parse_feature_index(key[0]), key[1], float(rules[key])) for key in rules]

    op_value_per_index: dict[int, list[tuple[str, float]]] = {}
    for i, op, val in index_op_value:
        if i in op_value_per_index:
            op_value_per_index[i].append((op, val))
        else:
            op_value_per_index[i] = [(op, val)]

    new_samples = np.zeros((n_samples, n_features))

    for feature_i in range(n_features):
        current_min = feature_limits[feature_i][0]
        current_max = feature_limits[feature_i][1]

        if feature_i not in op_value_per_index:
            new_samples[:, feature_i] = rng.uniform(current_min, current_max, n_samples)
            continue

        for _rule_i, (op, value) in enumerate(op_value_per_index[feature_i]):
            if op in ["<", "<="]:
                if current_min < value < current_max:
                    current_max = value
            elif op in [">", ">="]:
                if current_min < value < current_max:
                    current_min = value
            elif op in ["=", "=="]:
                current_min = value
                current_max = value

        new_samples[:, feature_i] = rng.uniform(current_min, current_max, n_samples)

    return new_samples


def instantiate_ruleset_rules(
    rules: list[Rules],
    weights: list[float],
    n_features: int,
    feature_limits: list[tuple[float, float]],
    n_samples: int,
    rng: Generator,
) -> np.ndarray:
    """Instantiate samples from a weighted set of rules.

    Distributes n_samples across rules proportionally to their (non-negative) weights,
    then instantiates each rule's share. Returns all instantiated samples combined.

    Args:
        rules: List of rule dictionaries.
        weights: Weight for each rule (rules with negative weight are ignored).
        n_features: Number of decision variables.
        feature_limits: Lower and upper bounds per feature.
        n_samples: Approximate total number of samples to generate.
        rng: A numpy random Generator for reproducibility.

    Returns:
        A 2D array of all generated samples stacked vertically.
    """
    if len(weights) < len(rules):
        rules = rules[: len(weights)]

    w_arr = np.array(weights)
    pos_mask = w_arr >= 0
    fractions = w_arr[pos_mask] / np.sum(w_arr[pos_mask])
    n_per_rule = np.round(fractions * n_samples).astype(int)

    instantiated = []
    rules_pos_w = [r for r, m in zip(rules, pos_mask, strict=False) if m]

    for rule_i, rule in enumerate(rules_pos_w):
        if n_per_rule[rule_i] > 0:
            instantiated.append(instantiate_rules(rule, n_features, feature_limits, int(n_per_rule[rule_i]), rng))

    if not instantiated:
        return np.zeros((0, n_features))

    return np.vstack(instantiated)
