"""Operators for the XLEMOO learning mode.

Implements the :class:`XLEMOOInstantiator`, which reads population history from
an Archive, ranks solutions by a scalarization column (computed by the evaluator),
trains an interpretable ML classifier to distinguish high-performing (H-group)
from low-performing (L-group) solutions, extracts rules, and instantiates new
candidate solutions.  Returns candidate decision variable vectors for evaluation
by the shared EMOEvaluator.

The scalarization function is added to the Problem (via ``add_weighted_sums`` or
``add_asf_nondiff``) before the evaluator is created, so every solution gets the
same fitness value.  Both the Darwinian selector (``ScalarizationSelector``) and
this instantiator read the same column, ensuring consistent fitness pressure.

Reference:
    Misitano, G. (2024). Towards Explainable Multiobjective Optimization:
    XLEMOO. ACM Trans. Evol. Learn. Optim., 4(1).
    https://doi.org/10.1145/3626104
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier

from desdeo.emo.hooks.archivers import Archive
from desdeo.explanations.rule_interpreters import (
    extract_boosted_rules,
    extract_skoped_rules,
    extract_slipper_rules,
    find_all_paths,
    instantiate_ruleset_rules,
    instantiate_tree_rules,
)
from desdeo.problem import Problem
from desdeo.tools.message import (
    Message,
    MessageTopics,
    TerminatorMessageTopics,
)
from desdeo.tools.patterns import Publisher, Subscriber


class MLModelType(StrEnum):
    """Supported ML model types for the XLEMOO learning mode."""

    DECISION_TREE = "DecisionTree"
    SLIPPER = "Slipper"
    BOOSTED_RULES = "BoostedRules"
    SKOPE_RULES = "SkopeRules"


class XLEMOOInstantiator(Subscriber):
    """Instantiator operator for the XLEMOO learning mode.

    Reads population history from an Archive, trains an interpretable ML
    classifier to distinguish high-performing (H-group) from low-performing
    (L-group) solutions, extracts rules, and instantiates new candidate
    solutions. The candidates (H-group + newly instantiated) are returned
    as a polars DataFrame of decision variable vectors for external evaluation.
    """

    @property
    def provided_topics(self) -> dict[int, Sequence[MessageTopics]]:
        """The topics provided by the XLEMOOInstantiator."""
        return {0: [], 1: [], 2: []}

    @property
    def interested_topics(self) -> Sequence[MessageTopics]:
        """The topics the XLEMOOInstantiator is interested in."""
        return [TerminatorMessageTopics.GENERATION]

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        archive: Archive,
        ml_model_type: MLModelType = MLModelType.DECISION_TREE,
        ml_model_kwargs: dict | None = None,
        h_split: float = 0.1,
        l_split: float = 0.1,
        instantiation_factor: float = 2.0,
        generation_lookback: int = 0,
        ancestral_recall: int = 0,
        unique_only: bool = False,
        seed: int = 0,
    ):
        """Initialize the XLEMOOInstantiator.

        Args:
            problem: The optimization problem.
            verbosity: Verbosity level for pub-sub.
            publisher: The publisher for the pub-sub system.
            archive: The Archive containing population history.
            ml_model_type: Which ML classifier to use.
            ml_model_kwargs: Additional kwargs passed to the ML model constructor.
            h_split: Fraction (if < 1) or count (if >= 1) of best solutions for H-group.
            l_split: Fraction (if < 1) or count (if >= 1) of worst solutions for L-group.
            instantiation_factor: Multiplier for how many new solutions to instantiate
                relative to the history size.
            generation_lookback: How many recent generations to consider (0 = all).
            ancestral_recall: How many of the earliest generations to always include.
            unique_only: Whether to deduplicate solutions before training.
            seed: Random seed.
        """
        super().__init__(publisher=publisher, verbosity=verbosity)
        self.problem = problem
        self.archive = archive
        self.ml_model_type = ml_model_type
        self.ml_model_kwargs = ml_model_kwargs or {}
        self.h_split = h_split
        self.l_split = l_split
        self.instantiation_factor = instantiation_factor
        self.generation_lookback = generation_lookback
        self.ancestral_recall = ancestral_recall
        self.unique_only = unique_only
        self.rng = np.random.default_rng(seed)

        # The most recently trained ML classifier (set by do())
        self.last_classifier = None
        # The most recently extracted rules (set by do())
        self.last_rules = None

        # Derive feature/target symbols from the problem
        flat_vars = problem.get_flattened_variables()
        self.variable_symbols = [v.symbol for v in flat_vars]
        self._feature_limits = [(v.lowerbound, v.upperbound) for v in flat_vars]
        self._n_features = len(flat_vars)

        if problem.scalarization_funcs:
            self.target_symbols = [f.symbol for f in problem.scalarization_funcs if f.symbol is not None]
        else:
            self.target_symbols = [f"{obj.symbol}_min" for obj in problem.objectives]

        # Track current generation (updated via pub-sub)
        self._current_generation: int = 1

    def _create_ml_model(self):
        """Create a fresh ML model instance based on ml_model_type."""
        if self.ml_model_type == MLModelType.DECISION_TREE:
            return DecisionTreeClassifier(**self.ml_model_kwargs)
        if self.ml_model_type == MLModelType.SLIPPER:
            from imodels import SlipperClassifier

            return SlipperClassifier(**self.ml_model_kwargs)
        if self.ml_model_type == MLModelType.BOOSTED_RULES:
            from imodels import BoostedRulesClassifier

            return BoostedRulesClassifier(**self.ml_model_kwargs)
        if self.ml_model_type == MLModelType.SKOPE_RULES:
            from imodels import SkopeRulesClassifier

            return SkopeRulesClassifier(**self.ml_model_kwargs)
        msg = f"Unsupported ML model type: {self.ml_model_type}"
        raise ValueError(msg)

    def _get_history_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract variable and target arrays from the archive, applying lookback/recall filters.

        Returns:
            Tuple of (individuals, targets) as numpy arrays.
        """
        solutions_df = self.archive.solutions
        if solutions_df is None or solutions_df.is_empty():
            msg = "Archive has no solutions. Cannot execute learning mode."
            raise RuntimeError(msg)

        # Apply generation filtering
        if self.generation_lookback > 0:
            min_gen = max(1, self._current_generation - self.generation_lookback)
            recent = solutions_df.filter(pl.col("generation") >= min_gen)

            if self.ancestral_recall > 0:
                ancestral = solutions_df.filter(pl.col("generation") <= self.ancestral_recall)
                filtered = pl.concat([ancestral, recent]).unique()
            else:
                filtered = recent
        else:
            filtered = solutions_df

        individuals = filtered[self.variable_symbols].to_numpy()
        targets = filtered[self.target_symbols].to_numpy()

        if self.unique_only:
            _, unique_inds = np.unique(individuals, return_index=True, axis=0)
            individuals = individuals[unique_inds]
            targets = targets[unique_inds]

        return individuals, targets

    def _compute_split_indices(self, n_total: int) -> tuple[int, int]:
        """Compute H and L group sizes from split parameters.

        Args:
            n_total: Total number of solutions.

        Returns:
            Tuple of (h_count, l_count).
        """
        if self.h_split < 1.0:
            h_count = int(self.h_split * n_total)
        else:
            h_count = min(int(self.h_split), n_total // 2)

        if self.l_split < 1.0:
            l_count = int(self.l_split * n_total)
        else:
            l_count = min(int(self.l_split), n_total // 2)

        # Ensure at least 1 in each group
        h_count = max(1, h_count)
        l_count = max(1, l_count)

        return h_count, l_count

    def do(self, population: pl.DataFrame) -> pl.DataFrame:
        """Execute one learning mode instantiation step.

        Steps:
        1. Read population history from archive, apply filters.
        2. Compute scalar fitness from scalarization column(s).
        3. Split into H-group (best) and L-group (worst).
        4. Train ML classifier (H=good, L=bad).
        5. Extract rules and instantiate new solutions.
        6. Combine with H-group, clip to bounds, return as DataFrame.

        The returned DataFrame contains decision variable vectors that should
        be evaluated by the shared EMOEvaluator before being passed to the
        XLEMOOSelector.

        Args:
            population: Current population variables DataFrame (used for schema
                and population size).

        Returns:
            A polars DataFrame of candidate variable vectors (H-group + instantiated).
        """
        population_size = len(population)

        # 1. Get history data
        individuals, targets = self._get_history_data()

        # 2. Compute scalar fitness from scalarization column(s)
        if targets.shape[1] == 1:
            fitness_values = targets[:, 0]
        else:
            fitness_values = np.sum(targets, axis=1)

        # 3. Sort and split into H/L groups
        sorted_indices = np.argsort(fitness_values)
        h_count, l_count = self._compute_split_indices(len(sorted_indices))

        h_indices = sorted_indices[:h_count]
        l_indices = sorted_indices[-l_count:]

        h_group = individuals[h_indices]
        l_group = individuals[l_indices]

        # 4. Train ML classifier
        x_train = np.vstack((h_group, l_group))
        ml_model = self._create_ml_model()

        if self.ml_model_type == MLModelType.DECISION_TREE:
            y_train = np.hstack(
                (
                    np.ones(len(h_group), dtype=int),
                    -1 * np.ones(len(l_group), dtype=int),
                )
            )
        else:
            y_train = np.hstack(
                (
                    np.ones(len(h_group), dtype=int),
                    np.zeros(len(l_group), dtype=int),
                )
            )

        ml_model.fit(x_train, y_train)
        classifier = ml_model
        self.last_classifier = classifier

        # 5. Extract rules and instantiate
        # Use population_size (not archive size) to keep instantiation count stable,
        # since the archive grows with each learning mode cycle.
        n_to_instantiate = int(population_size * self.instantiation_factor)

        if self.ml_model_type == MLModelType.DECISION_TREE:
            paths = find_all_paths(classifier)
            self.last_rules = paths
            instantiated = instantiate_tree_rules(
                paths,
                self._n_features,
                self._feature_limits,
                n_to_instantiate,
                1,  # desired classification = 1 (H-group)
                self.rng,
            )
            # Reshape from 3D (n_paths, n_samples, n_features) to 2D
            if instantiated.size > 0:
                instantiated = instantiated.reshape((-1, instantiated.shape[2]))
            else:
                instantiated = np.zeros((0, self._n_features))
        elif self.ml_model_type == MLModelType.SLIPPER:
            ruleset, weights = extract_slipper_rules(classifier)
            self.last_rules = (ruleset, weights)
            instantiated = instantiate_ruleset_rules(
                ruleset, weights, self._n_features, self._feature_limits, n_to_instantiate, self.rng
            )
        elif self.ml_model_type == MLModelType.BOOSTED_RULES:
            ruleset, weights = extract_boosted_rules(classifier)
            self.last_rules = (ruleset, weights)
            instantiated = instantiate_ruleset_rules(
                ruleset, weights, self._n_features, self._feature_limits, n_to_instantiate, self.rng
            )
        elif self.ml_model_type == MLModelType.SKOPE_RULES:
            ruleset, weights = extract_skoped_rules(classifier)
            self.last_rules = (ruleset, weights)
            instantiated = instantiate_ruleset_rules(
                ruleset, weights, self._n_features, self._feature_limits, n_to_instantiate, self.rng
            )
        else:
            msg = f"Unsupported ML model type: {self.ml_model_type}"
            raise ValueError(msg)

        # 6. Combine with H-group and clip to variable bounds
        if instantiated.shape[0] > 0:
            combined = np.vstack((h_group, instantiated))
        else:
            combined = h_group

        limits = np.array(self._feature_limits)
        combined = np.clip(combined, limits[:, 0], limits[:, 1])

        # Build polars DataFrame with same schema as population
        return pl.DataFrame(
            {sym: combined[:, i].tolist() for i, sym in enumerate(self.variable_symbols)},
            schema=population.schema,
        )

    def state(self) -> Sequence[Message]:
        """Return the current state (no messages published)."""
        return []

    def update(self, message: Message) -> None:
        """Update internal state from pub-sub messages.

        Tracks current generation number for lookback filtering.

        Args:
            message: A message from the publisher.
        """
        if message.topic == TerminatorMessageTopics.GENERATION and isinstance(message.value, int):
            self._current_generation = message.value
