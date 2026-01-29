"""The base class for selection operators.

Some operators should be rewritten.
TODO:@light-weaver
"""

import warnings
from abc import abstractmethod
from collections.abc import Callable, Sequence
from enum import StrEnum
from itertools import combinations
from typing import Literal, TypeVar

import numpy as np
import polars as pl
from numba import njit
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import comb
from scipy.stats.qmc import LatinHypercube

from desdeo.problem import Problem
from desdeo.tools import get_corrected_ideal_and_nadir
from desdeo.tools.indicators_binary import self_epsilon
from desdeo.tools.message import (
    Array2DMessage,
    DictMessage,
    Message,
    NumpyArrayMessage,
    PolarsDataFrameMessage,
    SelectorMessageTopics,
    TerminatorMessageTopics,
)
from desdeo.tools.non_dominated_sorting import fast_non_dominated_sort
from desdeo.tools.patterns import Publisher, Subscriber

SolutionType = TypeVar("SolutionType", list, pl.DataFrame)


class BaseSelector(Subscriber):
    """A base class for selection operators."""

    def __init__(self, problem: Problem, verbosity: int, publisher: Publisher, seed: int = 0):
        """Initialize a selection operator."""
        super().__init__(verbosity=verbosity, publisher=publisher)
        self.problem = problem
        self.variable_symbols = [x.symbol for x in problem.get_flattened_variables()]
        self.objective_symbols = [x.symbol for x in problem.objectives]
        self.maximization_mult = {x.symbol: -1 if x.maximize else 1 for x in problem.objectives}

        if problem.scalarization_funcs is None:
            self.target_symbols = [f"{x.symbol}_min" for x in problem.objectives]
            try:
                ideal, nadir = get_corrected_ideal_and_nadir(problem)  # This is for the minimized problem
                self.ideal = np.array([ideal[x.symbol] for x in problem.objectives])
                self.nadir = np.array([nadir[x.symbol] for x in problem.objectives]) if nadir is not None else None
            except ValueError:  # in case the ideal and nadir are not provided
                self.ideal = None
                self.nadir = None
        else:
            self.target_symbols = [x.symbol for x in problem.scalarization_funcs if x.symbol is not None]
            self.ideal: np.ndarray | None = None
            self.nadir: np.ndarray | None = None
        if problem.constraints is None:
            self.constraints_symbols = None
        else:
            self.constraints_symbols = [x.symbol for x in problem.constraints]
        self.num_dims = len(self.target_symbols)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def do(
        self,
        parents: tuple[SolutionType, pl.DataFrame],
        offsprings: tuple[SolutionType, pl.DataFrame],
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Perform the selection operation.

        Args:
            parents (tuple[SolutionType, pl.DataFrame]): the decision variables as the first element.
                The second element is the objective values, targets, and constraint violations.
            offsprings (tuple[SolutionType, pl.DataFrame]): the decision variables as the first element.
                The second element is the objective values, targets, and constraint violations.

        Returns:
            tuple[SolutionType, pl.DataFrame]: The selected decision variables and their objective values,
                targets, and constraint violations.
        """


class ReferenceVectorOptions(BaseModel):
    """Pydantic model for Reference Vector arguments."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    adaptation_frequency: int = Field(default=0)
    """Number of generations between reference vector adaptation. If set to 0, no adaptation occurs. Defaults to 0.
    Only used if no preference is provided."""
    creation_type: Literal["simplex", "s_energy"] = Field(default="simplex")
    """The method for creating reference vectors. Defaults to "simplex".
    Currently only "simplex" is implemented. Future versions will include "s_energy".

    If set to "simplex", the reference vectors are created using the simplex lattice design method.
    This method is generates distributions with specific numbers of reference vectors.
    Check: https://www.itl.nist.gov/div898/handbook/pri/section5/pri542.htm for more information.
    If set to "s_energy", the reference vectors are created using the Riesz s-energy criterion. This method is used to
    distribute an arbitrary number of reference vectors in the objective space while minimizing the s-energy.
    Currently not implemented.
    """
    vector_type: Literal["spherical", "planar"] = Field(default="spherical")
    """The method for normalizing the reference vectors. Defaults to "spherical"."""
    lattice_resolution: int | None = None
    """Number of divisions along an axis when creating the simplex lattice. This is not required/used for the "s_energy"
    method. If not specified, the lattice resolution is calculated based on the `number_of_vectors`. If "spherical" is
    selected as the `vector_type`, this value overrides the `number_of_vectors`.
    """
    number_of_vectors: int = 200
    """Number of reference vectors to be created. If "simplex" is selected as the `creation_type`, then the closest
    `lattice_resolution` is calculated based on this value. If "s_energy" is selected, then this value is used directly.
    Note that if neither `lattice_resolution` nor `number_of_vectors` is specified, the number of vectors defaults to
    200. Overridden if "spherical" is selected as the `vector_type` and `lattice_resolution` is provided.
    """
    adaptation_distance: float = Field(default=0.2)
    """Distance parameter for the interactive adaptation methods. Defaults to 0.2."""
    reference_point: dict[str, float] | None = Field(default=None)
    """The reference point for interactive adaptation."""
    preferred_solutions: dict[str, list[float]] | None = Field(default=None)
    """The preferred solutions for interactive adaptation."""
    non_preferred_solutions: dict[str, list[float]] | None = Field(default=None)
    """The non-preferred solutions for interactive adaptation."""
    preferred_ranges: dict[str, list[float]] | None = Field(default=None)
    """The preferred ranges for interactive adaptation."""


class BaseDecompositionSelector(BaseSelector):
    """Base class for decomposition based selection operators."""

    def __init__(
        self,
        problem: Problem,
        reference_vector_options: ReferenceVectorOptions,
        verbosity: int,
        publisher: Publisher,
        invert_reference_vectors: bool = False,
        seed: int = 0,
    ):
        super().__init__(problem, verbosity=verbosity, publisher=publisher, seed=seed)
        self.reference_vector_options = reference_vector_options
        self.invert_reference_vectors = invert_reference_vectors
        self.reference_vectors: np.ndarray
        self.reference_vectors_initial: np.ndarray

        if self.reference_vector_options.creation_type == "s_energy":
            raise NotImplementedError("Riesz s-energy criterion is not yet implemented.")

        self._create_simplex()

        if self.reference_vector_options.reference_point:
            corrected_rp = np.array(
                [
                    self.reference_vector_options.reference_point[x] * self.maximization_mult[x]
                    for x in self.objective_symbols
                ]
            )
            self.interactive_adapt_3(
                corrected_rp,
                translation_param=self.reference_vector_options.adaptation_distance,
            )
        elif self.reference_vector_options.preferred_solutions:
            corrected_sols = np.array(
                [
                    np.array(self.reference_vector_options.preferred_solutions[x]) * self.maximization_mult[x]
                    for x in self.objective_symbols
                ]
            ).T
            self.interactive_adapt_1(
                corrected_sols,
                translation_param=self.reference_vector_options.adaptation_distance,
            )
        elif self.reference_vector_options.non_preferred_solutions:
            corrected_sols = np.array(
                [
                    np.array(self.reference_vector_options.non_preferred_solutions[x]) * self.maximization_mult[x]
                    for x in self.objective_symbols
                ]
            ).T
            self.interactive_adapt_2(
                corrected_sols,
                predefined_distance=self.reference_vector_options.adaptation_distance,
                ord=2 if self.reference_vector_options.vector_type == "spherical" else 1,
            )
        elif self.reference_vector_options.preferred_ranges:
            corrected_ranges = np.array(
                [
                    np.array(self.reference_vector_options.preferred_ranges[x]) * self.maximization_mult[x]
                    for x in self.objective_symbols
                ]
            ).T
            self.interactive_adapt_4(
                corrected_ranges,
            )

    def _create_simplex(self):
        """Create the reference vectors using simplex lattice design."""

        def approx_lattice_resolution(number_of_vectors: int, num_dims: int) -> int:
            """Approximate the lattice resolution based on the number of vectors."""
            temp_lattice_resolution = 0
            while True:
                temp_lattice_resolution += 1
                temp_number_of_vectors = comb(
                    temp_lattice_resolution + num_dims - 1,
                    num_dims - 1,
                    exact=True,
                )
                if temp_number_of_vectors > number_of_vectors:
                    break
            return temp_lattice_resolution - 1

        if self.reference_vector_options.lattice_resolution:
            lattice_resolution = self.reference_vector_options.lattice_resolution
        else:
            lattice_resolution = approx_lattice_resolution(
                self.reference_vector_options.number_of_vectors, num_dims=self.num_dims
            )

        number_of_vectors: int = comb(
            lattice_resolution + self.num_dims - 1,
            self.num_dims - 1,
            exact=True,
        )

        self.reference_vector_options.number_of_vectors = number_of_vectors
        self.reference_vector_options.lattice_resolution = lattice_resolution

        temp1 = range(1, self.num_dims + lattice_resolution)
        temp1 = np.array(list(combinations(temp1, self.num_dims - 1)))
        temp2 = np.array([range(self.num_dims - 1)] * number_of_vectors)
        temp = temp1 - temp2 - 1
        weight = np.zeros((number_of_vectors, self.num_dims), dtype=int)
        weight[:, 0] = temp[:, 0]
        for i in range(1, self.num_dims - 1):
            weight[:, i] = temp[:, i] - temp[:, i - 1]
        weight[:, -1] = lattice_resolution - temp[:, -1]
        if not self.invert_reference_vectors:  # todo, this currently only exists for nsga3
            self.reference_vectors = weight / lattice_resolution
        else:
            self.reference_vectors = 1 - (weight / lattice_resolution)
        self.reference_vectors_initial = np.copy(self.reference_vectors)
        self._normalize_rvs()

    def _normalize_rvs(self):
        """Normalize the reference vectors to a unit hypersphere."""
        if self.reference_vector_options.vector_type == "spherical":
            norm = np.linalg.norm(self.reference_vectors, axis=1).reshape(-1, 1)
            norm[norm == 0] = np.finfo(float).eps
            self.reference_vectors = np.divide(self.reference_vectors, norm)
            return
        if self.reference_vector_options.vector_type == "planar":
            if not self.invert_reference_vectors:
                norm = np.sum(self.reference_vectors, axis=1).reshape(-1, 1)
                self.reference_vectors = np.divide(self.reference_vectors, norm)
                return
            norm = np.sum(1 - self.reference_vectors, axis=1).reshape(-1, 1)
            self.reference_vectors = 1 - np.divide(1 - self.reference_vectors, norm)
            return
        # Not needed due to pydantic validation
        raise ValueError("Invalid vector type. Must be either 'spherical' or 'planar'.")

    def interactive_adapt_1(self, z: np.ndarray, translation_param: float) -> None:
        """Adapt reference vectors using the information about prefererred solution(s) selected by the Decision maker.

        Args:
            z (np.ndarray): Preferred solution(s).
            translation_param (float): Parameter determining how close the reference vectors are to the central vector
            **v** defined by using the selected solution(s) z.
        """
        if z.shape[0] == 1:
            # single preferred solution
            # calculate new reference vectors
            self.reference_vectors = translation_param * self.reference_vectors_initial + ((1 - translation_param) * z)

        else:
            # multiple preferred solutions
            # calculate new reference vectors for each preferred solution
            values = [translation_param * self.reference_vectors_initial + ((1 - translation_param) * z_i) for z_i in z]

            # combine arrays of reference vectors into a single array and update reference vectors
            self.reference_vectors = np.concatenate(values)

        self._normalize_rvs()
        self.add_edge_vectors()

    def interactive_adapt_2(self, z: np.ndarray, predefined_distance: float, ord: int) -> None:
        """Adapt reference vectors by using the information about non-preferred solution(s) selected by the Decision maker.

        After the Decision maker has specified non-preferred solution(s), Euclidian distance between normalized solution
        vector(s) and each of the reference vectors are calculated. Those reference vectors that are **closer** than a
        predefined distance are either **removed** or **re-positioned** somewhere else.

        Note:
            At the moment, only the **removal** of reference vectors is supported. Repositioning of the reference
            vectors is **not** supported.

        Note:
            In case the Decision maker specifies multiple non-preferred solutions, the reference vector(s) for which the
            distance to **any** of the non-preferred solutions is less than predefined distance are removed.

        Note:
            Future developer should implement a way for a user to say: "Remove some percentage of
            objecive space/reference vectors" rather than giving a predefined distance value.

        Args:
            z (np.ndarray): Non-preferred solution(s).
            predefined_distance (float): The reference vectors that are closer than this distance are either removed or
                re-positioned somewhere else. Default value: 0.2
            ord (int): Order of the norm. Default is 2, i.e., Euclidian distance.
        """
        # calculate L1 norm of non-preferred solution(s)
        z = np.atleast_2d(z)
        norm = np.linalg.norm(z, ord=ord, axis=1).reshape(np.shape(z)[0], 1)

        # non-preferred solutions normalized
        v_c = np.divide(z, norm)

        # distances from non-preferred solution(s) to each reference vector
        distances = np.array(
            [
                list(
                    map(
                        lambda solution: np.linalg.norm(solution - value, ord=2),
                        v_c,
                    )
                )
                for value in self.reference_vectors
            ]
        )

        # find out reference vectors that are not closer than threshold value to any non-preferred solution
        mask = [all(d >= predefined_distance) for d in distances]

        # set those reference vectors that met previous condition as new reference vectors, drop others
        self.reference_vectors = self.reference_vectors[mask]

        self._normalize_rvs()
        self.add_edge_vectors()

    def interactive_adapt_3(self, ref_point, translation_param):
        """Adapt reference vectors linearly towards a reference point. Then normalize.

        The details can be found in the following paper: Hakanen, Jussi &
        Chugh, Tinkle & Sindhya, Karthik & Jin, Yaochu & Miettinen, Kaisa.
        (2016). Connections of Reference Vectors and Different Types of
        Preference Information in Interactive Multiobjective Evolutionary
        Algorithms.

        Parameters
        ----------
        ref_point :

        translation_param :
            (Default value = 0.2)

        """
        self.reference_vectors = self.reference_vectors_initial * translation_param + (
            (1 - translation_param) * ref_point
        )
        self._normalize_rvs()
        self.add_edge_vectors()

    def interactive_adapt_4(self, preferred_ranges: np.ndarray) -> None:
        """Adapt reference vectors by using the information about the Decision maker's preferred range for each of the objective.

        Using these ranges, Latin hypercube sampling is applied to generate m number of samples between
        within these ranges, where m is the number of reference vectors. Normalized vectors constructed of these samples
        are then set as new reference vectors.

        Args:
            preferred_ranges (np.ndarray): Preferred lower and upper bound for each of the objective function values.
        """
        # bounds
        lower_limits = np.min(preferred_ranges, axis=0)
        upper_limits = np.max(preferred_ranges, axis=0)

        # generate samples using Latin hypercube sampling
        lhs = LatinHypercube(d=self.num_dims, seed=self.rng)
        w = lhs.random(n=self.reference_vectors_initial.shape[0])

        # scale between bounds
        w = w * (upper_limits - lower_limits) + lower_limits

        # set new reference vectors and normalize them
        self.reference_vectors = w
        self._normalize_rvs()
        self.add_edge_vectors()

    def add_edge_vectors(self):
        """Add edge vectors to the list of reference vectors.

        Used to cover the entire orthant when preference information is
        provided.

        """
        edge_vectors = np.eye(self.reference_vectors.shape[1])
        self.reference_vectors = np.vstack([self.reference_vectors, edge_vectors])
        self._normalize_rvs()


class ParameterAdaptationStrategy(StrEnum):
    """The parameter adaptation strategies for the RVEA selector."""

    GENERATION_BASED = "GENERATION_BASED"  # Based on the current generation and the maximum generation.
    FUNCTION_EVALUATION_BASED = (
        "FUNCTION_EVALUATION_BASED"  # Based on the current function evaluation and the maximum function evaluation.
    )
    OTHER = "OTHER"  # As of yet undefined strategies.


@njit
def _rvea_selection(
    fitness: np.ndarray, reference_vectors: np.ndarray, ideal: np.ndarray, partial_penalty: float, gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select individuals based on their fitness and their distance to the reference vectors.

    Args:
        fitness (np.ndarray): The fitness values of the individuals.
        reference_vectors (np.ndarray): The reference vectors.
        ideal (np.ndarray): The ideal point.
        partial_penalty (float): The partial penalty in APD.
        gamma (np.ndarray): The angle between current and closest reference vector.

    Returns:
        tuple[np.ndarray, np.ndarray]: The selected individuals and their APD fitness values.
    """
    tranlated_fitness = fitness - ideal
    num_vectors = reference_vectors.shape[0]
    num_solutions = fitness.shape[0]

    cos_matrix = np.zeros((num_solutions, num_vectors))

    for i in range(num_solutions):
        solution = tranlated_fitness[i]
        norm = np.linalg.norm(solution)
        for j in range(num_vectors):
            cos_matrix[i, j] = np.dot(solution, reference_vectors[j]) / max(1e-10, norm)  # Avoid division by zero

    assignment_matrix = np.zeros((num_solutions, num_vectors), dtype=np.bool_)

    for i in range(num_solutions):
        assignment_matrix[i, np.argmax(cos_matrix[i])] = True

    selection = np.zeros(num_solutions, dtype=np.bool_)
    apd_fitness = np.zeros(num_solutions, dtype=np.float64)

    for j in range(num_vectors):
        min_apd = np.inf
        select = -1
        for i in np.where(assignment_matrix[:, j])[0]:
            solution = tranlated_fitness[i]
            apd = (1 + (partial_penalty * np.arccos(cos_matrix[i, j]) / gamma[j])) * np.linalg.norm(solution)
            apd_fitness[i] = apd
            if apd < min_apd:
                min_apd = apd
                select = i
        selection[select] = True

    return selection, apd_fitness


@njit
def _rvea_selection_constrained(
    fitness: np.ndarray,
    constraints: np.ndarray,
    reference_vectors: np.ndarray,
    ideal: np.ndarray,
    partial_penalty: float,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select individuals based on their fitness and their distance to the reference vectors.

    Args:
        fitness (np.ndarray): The fitness values of the individuals.
        constraints (np.ndarray): The constraint violations of the individuals.
        reference_vectors (np.ndarray): The reference vectors.
        ideal (np.ndarray): The ideal point.
        partial_penalty (float): The partial penalty in APD.
        gamma (np.ndarray): The angle between current and closest reference vector.

    Returns:
        tuple[np.ndarray, np.ndarray]: The selected individuals and their APD fitness values.
    """
    tranlated_fitness = fitness - ideal
    num_vectors = reference_vectors.shape[0]
    num_solutions = fitness.shape[0]

    violations = np.maximum(0, constraints)

    cos_matrix = np.zeros((num_solutions, num_vectors))

    for i in range(num_solutions):
        solution = tranlated_fitness[i]
        norm = np.linalg.norm(solution)
        for j in range(num_vectors):
            cos_matrix[i, j] = np.dot(solution, reference_vectors[j]) / max(1e-10, norm)  # Avoid division by zero

    assignment_matrix = np.zeros((num_solutions, num_vectors), dtype=np.bool_)

    for i in range(num_solutions):
        assignment_matrix[i, np.argmax(cos_matrix[i])] = True

    selection = np.zeros(num_solutions, dtype=np.bool_)
    apd_fitness = np.zeros(num_solutions, dtype=np.float64)

    for j in range(num_vectors):
        min_apd = np.inf
        min_violation = np.inf
        select = -1
        select_violation = -1
        for i in np.where(assignment_matrix[:, j])[0]:
            solution = tranlated_fitness[i]
            apd = (1 + (partial_penalty * np.arccos(cos_matrix[i, j]) / gamma[j])) * np.linalg.norm(solution)
            apd_fitness[i] = apd
            feasible = np.all(violations[i] == 0)
            current_violation = np.sum(violations[i])
            if feasible:
                if apd < min_apd:
                    min_apd = apd
                    select = i
            elif current_violation < min_violation:
                min_violation = current_violation
                select_violation = i
        if select != -1:
            selection[select] = True
        else:
            selection[select_violation] = True

    return selection, apd_fitness


class RVEASelector(BaseDecompositionSelector):
    @property
    def provided_topics(self):
        return {
            0: [],
            1: [
                SelectorMessageTopics.STATE,
            ],
            2: [
                SelectorMessageTopics.REFERENCE_VECTORS,
                SelectorMessageTopics.STATE,
                SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
            ],
        }

    @property
    def interested_topics(self):
        return [
            TerminatorMessageTopics.GENERATION,
            TerminatorMessageTopics.MAX_GENERATIONS,
            TerminatorMessageTopics.EVALUATION,
            TerminatorMessageTopics.MAX_EVALUATIONS,
        ]

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        alpha: float = 2.0,
        parameter_adaptation_strategy: ParameterAdaptationStrategy = ParameterAdaptationStrategy.GENERATION_BASED,
        reference_vector_options: ReferenceVectorOptions | dict | None = None,
        seed: int = 0,
    ):
        if parameter_adaptation_strategy not in ParameterAdaptationStrategy:
            raise TypeError(f"Parameter adaptation strategy must be of Type {type(ParameterAdaptationStrategy)}")
        if parameter_adaptation_strategy == ParameterAdaptationStrategy.OTHER:
            raise ValueError("Other parameter adaptation strategies are not yet implemented.")

        if reference_vector_options is None:
            reference_vector_options = ReferenceVectorOptions()

        if isinstance(reference_vector_options, dict):
            reference_vector_options = ReferenceVectorOptions.model_validate(reference_vector_options)

        # Just asserting correct options for RVEA
        reference_vector_options.vector_type = "spherical"
        if reference_vector_options.adaptation_frequency == 0:
            warnings.warn(
                "Adaptation frequency was set to 0. Setting it to 100 for RVEA selector. "
                "Set it to 0 only if you provide preference information.",
                UserWarning,
                stacklevel=2,
            )
            reference_vector_options.adaptation_frequency = 100

        super().__init__(
            problem=problem,
            reference_vector_options=reference_vector_options,
            verbosity=verbosity,
            publisher=publisher,
            seed=seed,
        )

        self.reference_vectors_gamma: np.ndarray
        self.numerator: float | None = None
        self.denominator: float | None = None
        self.alpha = alpha
        self.selected_individuals: list | pl.DataFrame
        self.selected_targets: pl.DataFrame
        self.selection: list[int]
        self.penalty = None
        self.parameter_adaptation_strategy = parameter_adaptation_strategy
        self.adapted_reference_vectors = None

    def do(
        self,
        parents: tuple[SolutionType, pl.DataFrame],
        offsprings: tuple[SolutionType, pl.DataFrame],
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Perform the selection operation.

        Args:
            parents (tuple[SolutionType, pl.DataFrame]): the decision variables as the first element.
                The second element is the objective values, targets, and constraint violations.
            offsprings (tuple[SolutionType, pl.DataFrame]): the decision variables as the first element.
                The second element is the objective values, targets, and constraint violations.

        Returns:
            tuple[SolutionType, pl.DataFrame]: The selected decision variables and their objective values,
                targets, and constraint violations.
        """
        if isinstance(parents[0], pl.DataFrame) and isinstance(offsprings[0], pl.DataFrame):
            solutions = parents[0].vstack(offsprings[0])
        elif isinstance(parents[0], list) and isinstance(offsprings[0], list):
            solutions = parents[0] + offsprings[0]
        else:
            raise TypeError("The decision variables must be either a list or a polars DataFrame, not both")
        if len(parents[0]) == 0:
            raise RuntimeError(
                "The parents population is empty. Cannot perform selection. This is a known unresolved issue."
            )
        alltargets = parents[1].vstack(offsprings[1])
        targets = alltargets[self.target_symbols].to_numpy()
        if self.constraints_symbols is None or len(self.constraints_symbols) == 0:
            # No constraints :)
            if self.ideal is None:
                self.ideal = np.min(targets, axis=0)
            else:
                self.ideal = np.min(np.vstack((self.ideal, np.min(targets, axis=0))), axis=0)
            self.nadir = np.max(targets, axis=0) if self.nadir is None else self.nadir
            if self.adapted_reference_vectors is None:
                self._adapt()
            selection, _ = _rvea_selection(
                fitness=targets,
                reference_vectors=self.adapted_reference_vectors,
                ideal=self.ideal,
                partial_penalty=self._partial_penalty_factor(),
                gamma=self.reference_vectors_gamma,
            )
        else:
            # Yes constraints :(
            constraints = (
                parents[1][self.constraints_symbols].vstack(offsprings[1][self.constraints_symbols]).to_numpy()
            )
            feasible = (constraints <= 0).all(axis=1)
            # Note that
            if self.ideal is None:
                # TODO: This breaks if there are no feasible solutions in the initial population
                self.ideal = np.min(targets[feasible], axis=0)
            else:
                self.ideal = np.min(np.vstack((self.ideal, np.min(targets[feasible], axis=0))), axis=0)
            try:
                nadir = np.max(targets[feasible], axis=0)
                self.nadir = nadir
            except ValueError:  # No feasible solution in current population
                pass  # Use previous nadir
            if self.adapted_reference_vectors is None:
                self._adapt()
            selection, _ = _rvea_selection_constrained(
                fitness=targets,
                constraints=constraints,
                reference_vectors=self.adapted_reference_vectors,
                ideal=self.ideal,
                partial_penalty=self._partial_penalty_factor(),
                gamma=self.reference_vectors_gamma,
            )

        self.selection = np.where(selection)[0].tolist()
        self.selected_individuals = solutions[self.selection]
        self.selected_targets = alltargets[self.selection]
        self.notify()
        return self.selected_individuals, self.selected_targets

    def _partial_penalty_factor(self) -> float:
        """Calculate and return the partial penalty factor for APD calculation.

            This calculation does not include the angle related terms, hence the name.
            If the calculated penalty is outside [0, 1], it will round it up/down to 0/1

        Returns:
            float: The partial penalty factor
        """
        if self.numerator is None or self.denominator is None or self.denominator == 0:
            raise RuntimeError("Numerator and denominator must be set before calculating the partial penalty factor.")
        penalty = self.numerator / self.denominator
        penalty = float(np.clip(penalty, 0, 1))
        self.penalty = (penalty**self.alpha) * self.reference_vectors.shape[1]
        return self.penalty

    def update(self, message: Message) -> None:
        """Update the parameters of the RVEA APD calculation.

        Args:
            message (Message): The message to update the parameters. The message should be coming from the
                Terminator operator (via the Publisher).
        """
        if not isinstance(message.topic, TerminatorMessageTopics):
            return
        if not isinstance(message.value, int):
            return
        if self.parameter_adaptation_strategy == ParameterAdaptationStrategy.GENERATION_BASED:
            if message.topic == TerminatorMessageTopics.GENERATION:
                self.numerator = message.value
                if (
                    self.reference_vector_options.adaptation_frequency > 0
                    and self.numerator % self.reference_vector_options.adaptation_frequency == 0
                ):
                    self._adapt()
            if message.topic == TerminatorMessageTopics.MAX_GENERATIONS:
                self.denominator = message.value
        elif self.parameter_adaptation_strategy == ParameterAdaptationStrategy.FUNCTION_EVALUATION_BASED:
            if message.topic == TerminatorMessageTopics.EVALUATION:
                self.numerator = message.value
            if message.topic == TerminatorMessageTopics.MAX_EVALUATIONS:
                self.denominator = message.value
        return

    def state(self) -> Sequence[Message]:
        if self.verbosity == 0 or self.selection is None:
            return []
        if self.verbosity == 1:
            return [
                Array2DMessage(
                    topic=SelectorMessageTopics.REFERENCE_VECTORS,
                    value=self.reference_vectors.tolist(),
                    source=self.__class__.__name__,
                ),
                DictMessage(
                    topic=SelectorMessageTopics.STATE,
                    value={
                        "ideal": self.ideal,
                        "nadir": self.nadir,
                        "partial_penalty_factor": self._partial_penalty_factor(),
                    },
                    source=self.__class__.__name__,
                ),
            ]  # verbosity == 2
        if isinstance(self.selected_individuals, pl.DataFrame):
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=pl.concat([self.selected_individuals, self.selected_targets], how="horizontal"),
                source=self.__class__.__name__,
            )
        else:
            warnings.warn("Population is not a Polars DataFrame. Defaulting to providing OUTPUTS only.", stacklevel=2)
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=self.selected_targets,
                source=self.__class__.__name__,
            )
        state_verbose = [
            Array2DMessage(
                topic=SelectorMessageTopics.REFERENCE_VECTORS,
                value=self.reference_vectors.tolist(),
                source=self.__class__.__name__,
            ),
            DictMessage(
                topic=SelectorMessageTopics.STATE,
                value={
                    "ideal": self.ideal,
                    "nadir": self.nadir,
                    "partial_penalty_factor": self._partial_penalty_factor(),
                },
                source=self.__class__.__name__,
            ),
            # DictMessage(
            #     topic=SelectorMessageTopics.SELECTED_INDIVIDUALS,
            #     value=self.selection[0].tolist(),
            #     source=self.__class__.__name__,
            # ),
            message,
        ]
        return state_verbose

    def _adapt(self):
        self.adapted_reference_vectors = self.reference_vectors
        if self.ideal is not None and self.nadir is not None:
            for i in range(self.reference_vectors.shape[0]):
                self.adapted_reference_vectors[i] = self.reference_vectors[i] * (self.nadir - self.ideal)
        self.adapted_reference_vectors = (
            self.adapted_reference_vectors / np.linalg.norm(self.adapted_reference_vectors, axis=1)[:, None]
        )

        self.reference_vectors_gamma = np.zeros(self.adapted_reference_vectors.shape[0])
        for i in range(self.adapted_reference_vectors.shape[0]):
            closest_angle = np.inf
            for j in range(self.adapted_reference_vectors.shape[0]):
                if i != j:
                    angle = np.arccos(
                        np.clip(np.dot(self.adapted_reference_vectors[i], self.adapted_reference_vectors[j]), -1.0, 1.0)
                    )
                    if angle < closest_angle and angle > 0:
                        # In cases with extreme differences in obj func ranges
                        # sometimes, the closest reference vectors are so close that
                        # the angle between them is 0 according to arccos (literally 0)
                        closest_angle = angle
            self.reference_vectors_gamma[i] = closest_angle


@njit
def jitted_calc_perpendicular_distance(
    solutions: np.ndarray, ref_dirs: np.ndarray, invert_reference_vectors: bool
) -> np.ndarray:
    """Calculate the perpendicular distance between solutions and reference directions.

    Args:
        solutions (np.ndarray): The normalized solutions.
        ref_dirs (np.ndarray): The reference directions.
        invert_reference_vectors (bool): Whether to invert the reference vectors.

    Returns:
        np.ndarray: The perpendicular distance matrix.
    """
    matrix = np.zeros((solutions.shape[0], ref_dirs.shape[0]))
    for i in range(ref_dirs.shape[0]):
        for j in range(solutions.shape[0]):
            if invert_reference_vectors:
                unit_vector = 1 - ref_dirs[i]
                unit_vector = -unit_vector / np.linalg.norm(unit_vector)
            else:
                unit_vector = ref_dirs[i] / np.linalg.norm(ref_dirs[i])
            component = ref_dirs[i] - solutions[j] - np.dot(ref_dirs[i] - solutions[j], unit_vector) * unit_vector
            matrix[j, i] = np.linalg.norm(component)
    return matrix


class NSGA3Selector(BaseDecompositionSelector):
    """The NSGA-III selection operator, heavily based on the version of nsga3 in the pymoo package by msu-coinlab."""

    @property
    def provided_topics(self):
        return {
            0: [],
            1: [
                SelectorMessageTopics.STATE,
            ],
            2: [
                SelectorMessageTopics.REFERENCE_VECTORS,
                SelectorMessageTopics.STATE,
                SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
            ],
        }

    @property
    def interested_topics(self):
        return []

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        reference_vector_options: ReferenceVectorOptions | None = None,
        invert_reference_vectors: bool = False,
        seed: int = 0,
    ):
        """Initialize the NSGA-III selection operator.

        Args:
            problem (Problem): The optimization problem to be solved.
            verbosity (int): The verbosity level of the operator.
            publisher (Publisher): The publisher to use for communication.
            reference_vector_options (ReferenceVectorOptions | None, optional): Options for the reference vectors. Defaults to None.
            invert_reference_vectors (bool, optional): Whether to invert the reference vectors. Defaults to False.
            seed (int, optional): The random seed to use. Defaults to 0.
        """
        if reference_vector_options is None:
            reference_vector_options = ReferenceVectorOptions()
        elif isinstance(reference_vector_options, dict):
            reference_vector_options = ReferenceVectorOptions.model_validate(reference_vector_options)

        # Just asserting correct options for NSGA-III
        reference_vector_options.vector_type = "planar"
        super().__init__(
            problem,
            reference_vector_options=reference_vector_options,
            verbosity=verbosity,
            publisher=publisher,
            seed=seed,
            invert_reference_vectors=invert_reference_vectors,
        )
        if self.constraints_symbols is not None:
            raise NotImplementedError("NSGA3 selector does not support constraints. Please use a different selector.")

        self.adapted_reference_vectors = None
        self.worst_fitness: np.ndarray | None = None
        self.extreme_points: np.ndarray | None = None
        self.n_survive = self.reference_vectors.shape[0]
        self.selection: list[int] | None = None
        self.selected_individuals: SolutionType | None = None
        self.selected_targets: pl.DataFrame | None = None

    def do(
        self,
        parents: tuple[SolutionType, pl.DataFrame],
        offsprings: tuple[SolutionType, pl.DataFrame],
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Perform the selection operation.

        Args:
            parents (tuple[SolutionType, pl.DataFrame]): the decision variables as the first element.
                The second element is the objective values, targets, and constraint violations.
            offsprings (tuple[SolutionType, pl.DataFrame]): the decision variables as the first element.
                The second element is the objective values, targets, and constraint violations.

        Returns:
            tuple[SolutionType, pl.DataFrame]: The selected decision variables and their objective values,
                targets, and constraint violations.
        """
        if isinstance(parents[0], pl.DataFrame) and isinstance(offsprings[0], pl.DataFrame):
            solutions = parents[0].vstack(offsprings[0])
        elif isinstance(parents[0], list) and isinstance(offsprings[0], list):
            solutions = parents[0] + offsprings[0]
        else:
            raise TypeError("The decision variables must be either a list or a polars DataFrame, not both")
        alltargets = parents[1].vstack(offsprings[1])
        targets = alltargets[self.target_symbols].to_numpy()
        if self.constraints_symbols is None:
            constraints = None
        else:
            constraints = (
                parents[1][self.constraints_symbols].vstack(offsprings[1][self.constraints_symbols]).to_numpy()
            )
        ref_dirs = self.reference_vectors

        if self.ideal is None:
            self.ideal = np.min(targets, axis=0)
        else:
            self.ideal = np.min(np.vstack((self.ideal, np.min(targets, axis=0))), axis=0)
        fitness = targets
        # Calculating fronts and ranks
        # fronts, dl, dc, rank = nds(fitness)
        fronts = fast_non_dominated_sort(fitness)
        fronts = [np.where(fronts[i])[0] for i in range(len(fronts))]
        non_dominated = fronts[0]

        if self.worst_fitness is None:
            self.worst_fitness = np.max(fitness, axis=0)
        else:
            self.worst_fitness = np.amax(np.vstack((self.worst_fitness, fitness)), axis=0)

        # Calculating worst points
        worst_of_population = np.amax(fitness, axis=0)
        worst_of_front = np.max(fitness[non_dominated, :], axis=0)
        self.extreme_points = self.get_extreme_points_c(
            fitness[non_dominated, :], self.ideal, extreme_points=self.extreme_points
        )
        self.nadir_point = nadir_point = self.get_nadir_point(
            self.extreme_points,
            self.ideal,
            self.worst_fitness,
            worst_of_population,
            worst_of_front,
        )

        # Finding individuals in first 'n' fronts
        selection = np.asarray([], dtype=int)
        for front_id in range(len(fronts)):
            if len(np.concatenate(fronts[: front_id + 1])) < self.n_survive:
                continue
            fronts = fronts[: front_id + 1]
            selection = np.concatenate(fronts)
            break
        F = fitness[selection]

        last_front = fronts[-1]

        # Selecting individuals from the last acceptable front.
        if len(selection) > self.n_survive:
            niche_of_individuals, dist_to_niche = self.associate_to_niches(F, ref_dirs, self.ideal, nadir_point)
            # if there is only one front
            if len(fronts) == 1:
                n_remaining = self.n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                id_until_last_front = list(range(len(until_last_front)))
                niche_count = self.calc_niche_count(len(ref_dirs), niche_of_individuals[id_until_last_front])
                n_remaining = self.n_survive - len(until_last_front)

            last_front_selection_id = list(range(len(until_last_front), len(selection)))
            if np.any(selection[last_front_selection_id] != last_front):
                print("error!!!")
            selected_from_last_front = self.niching(
                fitness[last_front, :],
                n_remaining,
                niche_count,
                niche_of_individuals[last_front_selection_id],
                dist_to_niche[last_front_selection_id],
            )
            final_selection = np.concatenate((until_last_front, last_front[selected_from_last_front]))
            if self.extreme_points is None:
                print("Error")
            if final_selection is None:
                print("Error")
        else:
            final_selection = selection

        self.selection = final_selection.tolist()
        if isinstance(solutions, pl.DataFrame) and self.selection is not None:
            self.selected_individuals = solutions[self.selection]
        elif isinstance(solutions, list) and self.selection is not None:
            self.selected_individuals = [solutions[i] for i in self.selection]
        else:
            raise RuntimeError("Something went wrong with the selection")
        self.selected_targets = alltargets[self.selection]

        self.notify()
        return self.selected_individuals, self.selected_targets

    def get_extreme_points_c(self, F, ideal_point, extreme_points=None):
        """Taken from pymoo"""
        # calculate the asf which is used for the extreme point decomposition
        asf = np.eye(F.shape[1])
        asf[asf == 0] = 1e6

        # add the old extreme points to never loose them for normalization
        _F = F
        if extreme_points is not None:
            _F = np.concatenate([extreme_points, _F], axis=0)

        # use __F because we substitute small values to be 0
        __F = _F - ideal_point
        __F[__F < 1e-3] = 0

        # update the extreme points for the normalization having the highest asf value
        # each
        F_asf = np.max(__F * asf[:, None, :], axis=2)
        I = np.argmin(F_asf, axis=1)
        extreme_points = _F[I, :]
        return extreme_points

    def get_nadir_point(
        self,
        extreme_points,
        ideal_point,
        worst_point,
        worst_of_front,
        worst_of_population,
    ):
        LinAlgError = np.linalg.LinAlgError
        try:
            # find the intercepts using gaussian elimination
            M = extreme_points - ideal_point
            b = np.ones(extreme_points.shape[1])
            plane = np.linalg.solve(M, b)
            intercepts = 1 / plane

            nadir_point = ideal_point + intercepts

            if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6) or np.any(nadir_point > worst_point):
                raise LinAlgError()

        except LinAlgError:
            nadir_point = worst_of_front

        b = nadir_point - ideal_point <= 1e-6
        nadir_point[b] = worst_of_population[b]
        return nadir_point

    def niching(self, F, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
        survivors = []

        # boolean array of elements that are considered for each iteration
        mask = np.full(F.shape[0], True)

        while len(survivors) < n_remaining:
            # all niches where new individuals can be assigned to
            next_niches_list = np.unique(niche_of_individuals[mask])

            # pick a niche with minimum assigned individuals - break tie if necessary
            next_niche_count = niche_count[next_niches_list]
            next_niche = np.where(next_niche_count == next_niche_count.min())[0]
            next_niche = next_niches_list[next_niche]
            next_niche = next_niche[self.rng.integers(0, len(next_niche))]

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            self.rng.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            mask[next_ind] = False
            survivors.append(int(next_ind))

            niche_count[next_niche] += 1

        return survivors

    def associate_to_niches(self, F, ref_dirs, ideal_point, nadir_point, utopian_epsilon=0.0):
        utopian_point = ideal_point - utopian_epsilon

        denom = nadir_point - utopian_point
        denom[denom == 0] = 1e-12

        # normalize by ideal point and intercepts
        N = (F - utopian_point) / denom
        # dist_matrix = self.calc_perpendicular_distance(N, ref_dirs)
        dist_matrix = jitted_calc_perpendicular_distance(N, ref_dirs, self.invert_reference_vectors)

        niche_of_individuals = np.argmin(dist_matrix, axis=1)
        dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

        return niche_of_individuals, dist_to_niche

    def calc_niche_count(self, n_niches, niche_of_individuals):
        niche_count = np.zeros(n_niches, dtype=int)
        index, count = np.unique(niche_of_individuals, return_counts=True)
        niche_count[index] = count
        return niche_count

    def calc_perpendicular_distance(self, N, ref_dirs):
        if self.invert_reference_vectors:
            u = np.tile(-ref_dirs, (len(N), 1))
            v = np.repeat(1 - N, len(ref_dirs), axis=0)
        else:
            u = np.tile(ref_dirs, (len(N), 1))
            v = np.repeat(N, len(ref_dirs), axis=0)

        norm_u = np.linalg.norm(u, axis=1)

        scalar_proj = np.sum(v * u, axis=1) / norm_u
        proj = scalar_proj[:, None] * u / norm_u[:, None]
        val = np.linalg.norm(proj - v, axis=1)
        matrix = np.reshape(val, (len(N), len(ref_dirs)))

        return matrix

    def state(self) -> Sequence[Message]:
        if self.verbosity == 0 or self.selection is None or self.selected_targets is None:
            return []
        if self.verbosity == 1:
            return [
                Array2DMessage(
                    topic=SelectorMessageTopics.REFERENCE_VECTORS,
                    value=self.reference_vectors.tolist(),
                    source=self.__class__.__name__,
                ),
                DictMessage(
                    topic=SelectorMessageTopics.STATE,
                    value={
                        "ideal": self.ideal,
                        "nadir": self.worst_fitness,
                        "extreme_points": self.extreme_points,
                        "n_survive": self.n_survive,
                    },
                    source=self.__class__.__name__,
                ),
            ]
        # verbosity == 2
        if isinstance(self.selected_individuals, pl.DataFrame):
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=pl.concat([self.selected_individuals, self.selected_targets], how="horizontal"),
                source=self.__class__.__name__,
            )
        else:
            warnings.warn("Population is not a Polars DataFrame. Defaulting to providing OUTPUTS only.", stacklevel=2)
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=self.selected_targets,
                source=self.__class__.__name__,
            )
        state_verbose = [
            Array2DMessage(
                topic=SelectorMessageTopics.REFERENCE_VECTORS,
                value=self.reference_vectors.tolist(),
                source=self.__class__.__name__,
            ),
            DictMessage(
                topic=SelectorMessageTopics.STATE,
                value={
                    "ideal": self.ideal,
                    "nadir": self.worst_fitness,
                    "extreme_points": self.extreme_points,
                    "n_survive": self.n_survive,
                },
                source=self.__class__.__name__,
            ),
            # Array2DMessage(
            #     topic=SelectorMessageTopics.SELECTED_INDIVIDUALS,
            #     value=self.selected_individuals,
            #     source=self.__class__.__name__,
            # ),
            message,
        ]
        return state_verbose

    def update(self, message: Message) -> None:
        pass


@njit
def _ibea_fitness(fitness_components: np.ndarray, kappa: float) -> np.ndarray:
    """Calculates the IBEA fitness for each individual based on pairwise fitness components.

    Args:
        fitness_components (np.ndarray): The pairwise fitness components of the individuals.
        kappa (float): The kappa value for the IBEA selection.

    Returns:
        np.ndarray: The IBEA fitness values for each individual.
    """
    num_individuals = fitness_components.shape[0]
    fitness = np.zeros(num_individuals)
    for i in range(num_individuals):
        for j in range(num_individuals):
            if i != j:
                fitness[i] -= np.exp(-fitness_components[j, i] / kappa)
    return fitness


@njit
def _ibea_select(fitness_components: np.ndarray, bad_sols: np.ndarray, kappa: float) -> int:
    """Selects the worst individual based on the IBEA indicator.

    Args:
        fitness_components (np.ndarray): The pairwise fitness components of the individuals.
        bad_sols (np.ndarray): A boolean array indicating which individuals are considered "bad".
        kappa (float): The kappa value for the IBEA selection.

    Returns:
        int: The index of the selected individual.
    """
    fitness = np.zeros(len(fitness_components))
    for i in range(len(fitness_components)):
        if bad_sols[i]:
            continue
        for j in range(len(fitness_components)):
            if bad_sols[j] or i == j:
                continue
            fitness[i] -= np.exp(-fitness_components[j, i] / kappa)
    choice = np.argmin(fitness)
    if fitness[choice] >= 0:
        if sum(bad_sols) == len(fitness_components) - 1:
            # If all but one individual is chosen, select the last one
            return np.where(~bad_sols)[0][0]
        raise RuntimeError("All individuals have non-negative fitness. Cannot select a new individual.")
    return choice


@njit
def _ibea_select_all(fitness_components: np.ndarray, population_size: int, kappa: float) -> np.ndarray:
    """Selects all individuals based on the IBEA indicator.

    Args:
        fitness_components (np.ndarray): The pairwise fitness components of the individuals.
        population_size (int): The desired size of the population after selection.
        kappa (float): The kappa value for the IBEA selection.

    Returns:
        list[int]: The list of indices of the selected individuals.
    """
    current_pop_size = len(fitness_components)
    bad_sols = np.zeros(current_pop_size, dtype=np.bool_)
    fitness = np.zeros(len(fitness_components))
    mod_fit_components = np.exp(-fitness_components / kappa)
    for i in range(len(fitness_components)):
        for j in range(len(fitness_components)):
            if i == j:
                continue
            fitness[i] -= mod_fit_components[j, i]
    while current_pop_size - sum(bad_sols) > population_size:
        selected = np.argmin(fitness)
        if fitness[selected] >= 0:
            if sum(bad_sols) == len(fitness_components) - 1:
                # If all but one individual is chosen, select the last one
                selected = np.where(~bad_sols)[0][0]
            raise RuntimeError("All individuals have non-negative fitness. Cannot select a new individual.")
        fitness[selected] = np.inf  # Make sure that this individual is not selected again
        bad_sols[selected] = True
        for i in range(len(mod_fit_components)):
            if bad_sols[i]:
                continue
            # Update fitness of the remaining individuals
            fitness[i] += mod_fit_components[selected, i]
    return ~bad_sols


class IBEASelector(BaseSelector):
    """The adaptive IBEA selection operator.

    Reference: Zitzler, E., Künzli, S. (2004). Indicator-Based Selection in Multiobjective Search. In: Yao, X., et al.
    Parallel Problem Solving from Nature - PPSN VIII. PPSN 2004. Lecture Notes in Computer Science, vol 3242.
    Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-30217-9_84
    """

    @property
    def provided_topics(self):
        return {
            0: [],
            1: [SelectorMessageTopics.STATE],
            2: [SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS, SelectorMessageTopics.SELECTED_FITNESS],
        }

    @property
    def interested_topics(self):
        return []

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        population_size: int,
        kappa: float = 0.05,
        binary_indicator: Callable[[np.ndarray], np.ndarray] = self_epsilon,
        seed: int = 0,
    ):
        """Initialize the IBEA selector.

        Args:
            problem (Problem): The problem to solve.
            verbosity (int): The verbosity level of the selector.
            publisher (Publisher): The publisher to send messages to.
            population_size (int): The size of the population to select.
            kappa (float, optional): The kappa value for the IBEA selection. Defaults to 0.05.
            binary_indicator (Callable[[np.ndarray], np.ndarray], optional): The binary indicator function to use.
                Defaults to self_epsilon with uses binary addaptive epsilon indicator.
        """
        # TODO(@light-weaver): IBEA doesn't perform as good as expected
        # The distribution of solutions found isn't very uniform
        # Update 21st August, tested against jmetalpy IBEA. Our version is both faster and better
        # What is happening???
        # Results are similar to this https://github.com/Xavier-MaYiMing/IBEA/
        super().__init__(problem=problem, verbosity=verbosity, publisher=publisher, seed=seed)
        self.selection: list[int] | None = None
        self.selected_individuals: SolutionType | None = None
        self.selected_targets: pl.DataFrame | None = None
        self.binary_indicator = binary_indicator
        self.kappa = kappa
        self.population_size = population_size
        if self.constraints_symbols is not None:
            raise NotImplementedError("IBEA selector does not support constraints. Please use a different selector.")

    def do(
        self, parents: tuple[SolutionType, pl.DataFrame], offsprings: tuple[SolutionType, pl.DataFrame]
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Perform the selection operation.

        Args:
            parents (tuple[SolutionType, pl.DataFrame]): the decision variables as the first element.
                The second element is the objective values, targets, and constraint violations.
            offsprings (tuple[SolutionType, pl.DataFrame]): the decision variables as the first element.
                The second element is the objective values, targets, and constraint violations.

        Returns:
            tuple[SolutionType, pl.DataFrame]: The selected decision variables and their objective values,
                targets, and constraint violations.
        """
        if self.constraints_symbols is not None:
            raise NotImplementedError("IBEA selector does not support constraints. Please use a different selector.")
        if isinstance(parents[0], pl.DataFrame) and isinstance(offsprings[0], pl.DataFrame):
            solutions = parents[0].vstack(offsprings[0])
        elif isinstance(parents[0], list) and isinstance(offsprings[0], list):
            solutions = parents[0] + offsprings[0]
        else:
            raise TypeError("The decision variables must be either a list or a polars DataFrame, not both")
        if len(parents[0]) < self.population_size:
            return parents[0], parents[1]
        alltargets = parents[1].vstack(offsprings[1])

        # Adaptation
        target_vals = alltargets[self.target_symbols].to_numpy()
        target_min = np.min(target_vals, axis=0)
        target_max = np.max(target_vals, axis=0)
        # Scale the targets to the range [0, 1]
        target_vals = (target_vals - target_min) / (target_max - target_min)
        fitness_components = self.binary_indicator(target_vals)
        kappa_mult = np.max(np.abs(fitness_components))

        chosen = _ibea_select_all(
            fitness_components, population_size=self.population_size, kappa=kappa_mult * self.kappa
        )
        self.selected_individuals = solutions.filter(chosen)
        self.selected_targets = alltargets.filter(chosen)
        self.selection = chosen

        fitness_components = fitness_components[chosen][:, chosen]
        self.fitness = _ibea_fitness(fitness_components, kappa=self.kappa * np.abs(fitness_components).max())

        self.notify()
        return self.selected_individuals, self.selected_targets

    def state(self) -> Sequence[Message]:
        """Return the state of the selector."""
        if self.verbosity == 0 or self.selection is None or self.selected_targets is None:
            return []
        if self.verbosity == 1:
            return [
                DictMessage(
                    topic=SelectorMessageTopics.STATE,
                    value={
                        "population_size": self.population_size,
                        "selected_individuals": self.selection,
                    },
                    source=self.__class__.__name__,
                )
            ]
        # verbosity == 2
        if isinstance(self.selected_individuals, pl.DataFrame):
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=pl.concat([self.selected_individuals, self.selected_targets], how="horizontal"),
                source=self.__class__.__name__,
            )
        else:
            warnings.warn("Population is not a Polars DataFrame. Defaulting to providing OUTPUTS only.", stacklevel=2)
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=self.selected_targets,
                source=self.__class__.__name__,
            )
        return [
            DictMessage(
                topic=SelectorMessageTopics.STATE,
                value={
                    "population_size": self.population_size,
                    "selected_individuals": self.selection,
                },
                source=self.__class__.__name__,
            ),
            message,
            NumpyArrayMessage(
                topic=SelectorMessageTopics.SELECTED_FITNESS,
                value=self.fitness,
                source=self.__class__.__name__,
            ),
        ]

    def update(self, message: Message) -> None:
        pass


@njit
def _nsga2_crowding_distance_assignment(
    non_dominated_front: np.ndarray, f_mins: np.ndarray, f_maxs: np.ndarray
) -> np.ndarray:
    """Computes the crowding distance as pecified in the definition of NSGA2.

    This function computed the crowding distances for a non-dominated set of solutions.
    A smaller value means that a solution is more crowded (worse), while a larger value means
    it is less crowded (better).

    Note:
        The boundary point in `non_dominated_front` will be assigned a non-crowding
            distance value of `np.inf` indicating, that they shouls always be included
            in later sorting.

    Args:
        non_dominated_front (np.ndarray): a 2D numpy array (size n x m = number
            of vectors x number of targets (obejctive funcitons)) containing
            mutually non-dominated vectors. The values of the vectors correspond to
            the optimization 'target' (usually the minimized objective function
            values.)
        f_mins (np.ndarray): a 1D numpy array of size m containing the minimum objective function
            values in `non_dominated_front`.
        f_maxs (np.ndarray): a 1D numpy array of size m containing the maximum objective function
            values in `non_dominated_front`.

    Returns:
        np.ndarray: a numpy array of size m containing the crowding distances for each vector
            in `non_dominated_front`.

    Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T.
        (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE
        transactions on evolutionary computation, 6(2), 182-197.
    """
    vectors = non_dominated_front  # I
    num_vectors = vectors.shape[0]  # l
    num_objectives = vectors.shape[1]

    crowding_distances = np.zeros(num_vectors)  # I[i]_distance

    for m in range(num_objectives):
        # sort by column (objective)
        m_order = vectors[:, m].argsort()
        # inlcude boundary points
        crowding_distances[m_order[0]], crowding_distances[m_order[-1]] = np.inf, np.inf

        for i in range(1, num_vectors - 1):
            crowding_distances[m_order[i]] = crowding_distances[m_order[i]] + (
                vectors[m_order[i + 1], m] - vectors[m_order[i - 1], m]
            ) / (f_maxs[m] - f_mins[m])

    return crowding_distances


class NSGA2Selector(BaseSelector):
    """Implements the selection operator defined for NSGA2.

    Implements the selection operator defined for NSGA2, which included the crowding
    distance calculation.

    Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T.
        (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE
        transactions on evolutionary computation, 6(2), 182-197.
    """

    @property
    def provided_topics(self):
        """The topics provided for the NSGA2 method."""
        return {
            0: [],
            1: [SelectorMessageTopics.STATE],
            2: [SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS, SelectorMessageTopics.SELECTED_FITNESS],
        }

    @property
    def interested_topics(self):
        """The topics the NSGA2 method is interested in."""
        return []

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        population_size: int,
        seed: int = 0,
    ):
        super().__init__(problem=problem, verbosity=verbosity, publisher=publisher, seed=seed)
        if self.constraints_symbols is not None:
            print(
                "NSGA2 selector does not currently support constraints. "
                "Results may vary if used to solve constrainted problems."
            )
        self.population_size = population_size
        self.seed = seed
        self.selection: list[int] | None = None
        self.selected_individuals: SolutionType | None = None
        self.selected_targets: pl.DataFrame | None = None

    def do(
        self, parents: tuple[SolutionType, pl.DataFrame], offsprings: tuple[SolutionType, pl.DataFrame]
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Perform the selection operation."""
        # First iteration, offspring is empty
        # Do basic binary tournament selection, recombination, and mutation
        # In practice, just compute the non-dom ranks and provide them as fitness

        # Off-spring empty (first iteration, compute only non-dominated ranks and provide them as fitness)
        if offsprings[0].is_empty() and offsprings[1].is_empty():
            # just compute non-dominated ranks of population and be done
            parents_a = parents[1][self.target_symbols].to_numpy()
            fronts = fast_non_dominated_sort(parents_a)

            # assign fitness according to non-dom rank (lower better)
            scores = np.arange(len(fronts))
            fitness_values = scores @ fronts
            self.fitness = fitness_values

            # all selected in first iteration
            self.selection = list(range(len(parents[1])))
            self.selected_individuals = parents[0]
            self.selected_targets = parents[1]

            self.notify()

            return self.selected_individuals, self.selected_targets

        # #Actual selection operator for NSGA2

        # Combine parent and offspring R_t = P_t U Q_t
        r_solutions = parents[0].vstack(offsprings[0])
        r_population = parents[1].vstack(offsprings[1])
        r_targets_arr = r_population[self.target_symbols].to_numpy()

        # the minimum and maximum target values in the whole current population
        f_mins, f_maxs = np.min(r_targets_arr, axis=0), np.max(r_targets_arr, axis=0)

        # Do fast non-dominated sorting on R_t -> F
        fronts = fast_non_dominated_sort(r_targets_arr)
        crowding_distances = np.ones(self.population_size) * np.nan
        rankings = np.ones(self.population_size) * np.nan
        fitness_values = np.ones(self.population_size) * np.nan

        # Set the new parent population to P_t+1 = empty and i=1
        new_parents = np.ones((self.population_size, parents[1].shape[1])) * np.nan
        new_parents_solutions = np.ones((self.population_size, parents[0].shape[1])) * np.nan
        parents_ptr = 0  # keep track where stuff was last added

        # the -1 is here because searchsorted returns the index where we can insert the population size to preserve the
        # order, hence, the previous index of this will be the last element in the cumsum that is less than
        # the population size
        last_whole_front_idx = (
            np.searchsorted(np.cumsum(np.sum(fronts, axis=1)), self.population_size, side="right") - 1
        )

        last_ranking = 0  # in case first front is larger th population size
        for i in range(last_whole_front_idx + 1):  # inclusive
            # The looped front here will result in a new population with size <= 100.

            # Compute the crowding distances for F_i
            distances = _nsga2_crowding_distance_assignment(r_targets_arr[fronts[i]], f_mins, f_maxs)
            crowding_distances[parents_ptr : parents_ptr + distances.shape[0]] = (
                distances  # distances will have same number of elements as in front[i]
            )

            # keep track of the rankings as well (best = 0, larger worse). First
            # non-dom front will have a rank fitness of 0.
            rankings[parents_ptr : parents_ptr + distances.shape[0]] = i

            #   P_t+1 = P_t+1 U F_i
            new_parents[parents_ptr : parents_ptr + distances.shape[0]] = r_population.filter(fronts[i])
            new_parents_solutions[parents_ptr : parents_ptr + distances.shape[0]] = r_solutions.filter(fronts[i])

            # compute fitness
            # infs are checked since boundary points are assigned this value when computing the crowding distance
            finite_distances = distances[distances != np.inf]
            max_no_inf = np.nanmax(finite_distances) if finite_distances.size > 0 else np.ones(fronts[i].sum())
            distances_no_inf = np.nan_to_num(distances, posinf=max_no_inf * 1.1)

            # Distances for the current front normalized between 0 and 1.
            # The small scalar we add in the nominator and denominator is to
            # ensure that no distance value would result in exactly 0 after
            # normalizing, which would increase the corresponding solution
            # ranking, once reversed, which we do not want to.
            normalized_distances = (distances_no_inf - (distances_no_inf.min() - 1e-6)) / (
                distances_no_inf.max() - (distances_no_inf.min() - 1e-6)
            )

            # since higher is better for the crowded distance, we substract the normalized distances from 1 so that
            # lower is better, which allows us to combine them with the ranking
            # No value here should be 1.0 or greater.
            reversed_distances = 1.0 - normalized_distances

            front_fitness = reversed_distances + rankings[parents_ptr : parents_ptr + distances.shape[0]]
            fitness_values[parents_ptr : parents_ptr + distances.shape[0]] = front_fitness

            # increment parent pointer
            parents_ptr += distances.shape[0]

            # keep track of last given rank
            last_ranking = i

        # deal with last (partial) front, if needed
        trimmed_and_sorted_indices = None
        if parents_ptr < self.population_size:
            distances = _nsga2_crowding_distance_assignment(
                r_targets_arr[fronts[last_whole_front_idx + 1]], f_mins, f_maxs
            )

            # Sort F_i in descending order according to crowding distance
            # This makes picking the selected part of the partial front easier
            trimmed_and_sorted_indices = distances.argsort()[::-1][: self.population_size - parents_ptr]

            crowding_distances[parents_ptr : self.population_size] = distances[trimmed_and_sorted_indices]
            rankings[parents_ptr : self.population_size] = last_ranking + 1

            # P_t+1 = P_t+1 U F_i[1: (N - |P_t+1|)]
            new_parents[parents_ptr : self.population_size] = r_population.filter(fronts[last_whole_front_idx + 1])[
                trimmed_and_sorted_indices
            ]
            new_parents_solutions[parents_ptr : self.population_size] = r_solutions.filter(
                fronts[last_whole_front_idx + 1]
            )[trimmed_and_sorted_indices]

            # compute fitness (see above for details)
            finite_distances = distances[trimmed_and_sorted_indices][distances[trimmed_and_sorted_indices] != np.inf]
            max_no_inf = (
                np.nanmax(finite_distances)
                if finite_distances.size > 0
                else np.ones(len(trimmed_and_sorted_indices))  # we have only boundary points
            )
            distances_no_inf = np.nan_to_num(distances[trimmed_and_sorted_indices], posinf=max_no_inf * 1.1)

            normalized_distances = (distances_no_inf - (distances_no_inf.min() - 1e-6)) / (
                distances_no_inf.max() - (distances_no_inf.min() - 1e-6)
            )

            reversed_distances = 1.0 - normalized_distances

            front_fitness = reversed_distances + rankings[parents_ptr : self.population_size]
            fitness_values[parents_ptr : parents_ptr + self.population_size] = front_fitness

        # back to polars, return values
        solutions = pl.DataFrame(new_parents_solutions, schema=parents[0].schema)
        outputs = pl.DataFrame(new_parents, schema=parents[1].schema)

        self.fitness = fitness_values

        whole_fronts = fronts[: last_whole_front_idx + 1]
        whole_indices = [np.where(row)[0].tolist() for row in whole_fronts]

        if trimmed_and_sorted_indices is not None:
            # partial front considered
            partial_front = fronts[last_whole_front_idx + 1]
            partial_indices = np.where(partial_front)[0][trimmed_and_sorted_indices].tolist()
        else:
            partial_indices = []

        self.selection = [index for indices in whole_indices for index in indices] + partial_indices
        self.selected_individuals = solutions
        self.selected_targets = outputs

        self.notify()
        return solutions, outputs

    def state(self) -> Sequence[Message]:
        """Return the state of the selector."""
        if self.verbosity == 0 or self.selection is None or self.selected_targets is None:
            return []
        if self.verbosity == 1:
            return [
                DictMessage(
                    topic=SelectorMessageTopics.STATE,
                    value={
                        "population_size": self.population_size,
                        "selected_individuals": self.selection,
                    },
                    source=self.__class__.__name__,
                )
            ]
        # verbosity == 2
        if isinstance(self.selected_individuals, pl.DataFrame):
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=pl.concat([self.selected_individuals, self.selected_targets], how="horizontal"),
                source=self.__class__.__name__,
            )
        else:
            warnings.warn("Population is not a Polars DataFrame. Defaulting to providing OUTPUTS only.", stacklevel=2)
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=self.selected_targets,
                source=self.__class__.__name__,
            )
        return [
            DictMessage(
                topic=SelectorMessageTopics.STATE,
                value={
                    "population_size": self.population_size,
                    "selected_individuals": self.selection,
                },
                source=self.__class__.__name__,
            ),
            message,
            NumpyArrayMessage(
                topic=SelectorMessageTopics.SELECTED_FITNESS,
                value=self.fitness,
                source=self.__class__.__name__,
            ),
        ]

    def update(self, message: Message) -> None:
        pass


class NSGA2ShadowSelector(BaseSelector):
    """Implements the selection operator defined for NSGA2.

    Implements the selection operator defined for NSGA2, which included the crowding
    distance calculation.

    Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T.
        (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE
        transactions on evolutionary computation, 6(2), 182-197.
    """

    @property
    def provided_topics(self):
        """The topics provided for the NSGA2 method."""
        return {
            0: [],
            1: [SelectorMessageTopics.STATE],
            2: [SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS, SelectorMessageTopics.SELECTED_FITNESS],
        }

    @property
    def interested_topics(self):
        """The topics the NSGA2 method is interested in."""
        return []

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        population_size: int,
        relaxed_constraint_symbol: str,
        constraint_threshold: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(problem=problem, verbosity=verbosity, publisher=publisher, seed=seed)
        if self.constraints_symbols is not None:
            print(
                "NSGA2 selector does not currently support constraints. "
                "Results may vary if used to solve constrainted problems."
            )
        self.population_size = population_size
        self.seed = seed
        self.relaxed_constraint_symbol = relaxed_constraint_symbol
        self.constraint_threshold = constraint_threshold
        self.selection: list[int] | None = None
        self.selected_individuals: SolutionType | None = None
        self.selected_targets: pl.DataFrame | None = None

    def do(
        self, parents: tuple[SolutionType, pl.DataFrame], offsprings: tuple[SolutionType, pl.DataFrame]
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Perform the selection operation."""
        # First iteration, offspring is empty
        # Do basic binary tournament selection, recombination, and mutation
        # In practice, just compute the non-dom ranks and provide them as fitness

        # Off-spring empty (first iteration, compute only non-dominated ranks and provide them as fitness)
        if offsprings[0].is_empty() and offsprings[1].is_empty():
            # just compute non-dominated ranks of population and be done
            parents_a = parents[1][self.target_symbols].to_numpy()
            fronts = fast_non_dominated_sort(parents_a)

            # assign fitness according to non-dom rank (lower better)
            scores = np.arange(len(fronts))
            fitness_values = scores @ fronts
            self.fitness = fitness_values

            # all selected in first iteration
            self.selection = list(range(len(parents[1])))
            self.selected_individuals = parents[0]
            self.selected_targets = parents[1]

            self.notify()

            return self.selected_individuals, self.selected_targets

        # #Actual selection operator for NSGA2

        # Combine parent and offspring R_t = P_t U Q_t
        r_solutions = parents[0].vstack(offsprings[0])
        r_population = parents[1].vstack(offsprings[1])
        r_targets_arr = r_population[self.target_symbols].to_numpy()

        # the minimum and maximum target values in the whole current population
        f_mins, f_maxs = np.min(r_targets_arr, axis=0), np.max(r_targets_arr, axis=0)

        # Do fast non-dominated sorting on R_t -> F
        fronts = fast_non_dominated_sort(r_targets_arr)
        crowding_distances = np.ones(self.population_size) * np.nan
        rankings = np.ones(self.population_size) * np.nan
        fitness_values = np.ones(self.population_size) * np.nan

        # Set the new parent population to P_t+1 = empty and i=1
        new_parents = np.ones(parents[1].shape) * np.nan
        new_parents_solutions = np.ones(parents[0].shape) * np.nan
        parents_ptr = 0  # keep track where stuff was last added

        # the -1 is here because searchsorted returns the index where we can insert the population size to preserve the
        # order, hence, the previous index of this will be the last element in the cumsum that is less than
        # the population size
        last_whole_front_idx = (
            np.searchsorted(np.cumsum(np.sum(fronts, axis=1)), self.population_size, side="right") - 1
        )

        last_ranking = 0  # in case first front is larger th population size
        for i in range(last_whole_front_idx + 1):  # inclusive
            # The looped front here will result in a new population with size <= 100.

            # Compute the crowding distances for F_i
            distances = _nsga2_crowding_distance_assignment(r_targets_arr[fronts[i]], f_mins, f_maxs)
            crowding_distances[parents_ptr : parents_ptr + distances.shape[0]] = (
                distances  # distances will have same number of elements as in front[i]
            )

            # keep track of the rankings as well (best = 0, larger worse). First
            # non-dom front will have a rank fitness of 0.
            rankings[parents_ptr : parents_ptr + distances.shape[0]] = i

            #   P_t+1 = P_t+1 U F_i
            new_parents[parents_ptr : parents_ptr + distances.shape[0]] = r_population.filter(fronts[i])
            new_parents_solutions[parents_ptr : parents_ptr + distances.shape[0]] = r_solutions.filter(fronts[i])

            # compute fitness
            # If fronts[i].sum() == 2 ([inf, inf]) will result is zero-size array here, hence the if else
            max_no_inf = np.nanmax(distances[distances != np.inf]) if fronts[i].sum() > 2 else np.ones(fronts[i].sum())
            distances_no_inf = np.nan_to_num(distances, posinf=max_no_inf * 1.1)

            # Distances for the current front normalized between 0 and 1.
            # The small scalar we add in the nominator and denominator is to
            # ensure that no distance value would result in exactly 0 after
            # normalizing, which would increase the corresponding solution
            # ranking, once reversed, which we do not want to.
            normalized_distances = (distances_no_inf - (distances_no_inf.min() - 1e-6)) / (
                distances_no_inf.max() - (distances_no_inf.min() - 1e-6)
            )

            # since higher is better for the crowded distance, we substract the normalized distances from 1 so that
            # lower is better, which allows us to combine them with the ranking
            # No value here should be 1.0 or greater.
            reversed_distances = 1.0 - normalized_distances

            front_fitness = reversed_distances + rankings[parents_ptr : parents_ptr + distances.shape[0]]
            fitness_values[parents_ptr : parents_ptr + distances.shape[0]] = front_fitness

            # increment parent pointer
            parents_ptr += distances.shape[0]

            # keep track of last given rank
            last_ranking = i

        # deal with last (partial) front, if needed
        if parents_ptr < self.population_size:
            distances = _nsga2_crowding_distance_assignment(
                r_targets_arr[fronts[last_whole_front_idx + 1]], f_mins, f_maxs
            )

            # Sort F_i in descending order according to crowding distance
            trimmed_and_sorted_indices = distances.argsort()[::-1][: self.population_size - parents_ptr]

            crowding_distances[parents_ptr : self.population_size] = distances[trimmed_and_sorted_indices]
            rankings[parents_ptr : self.population_size] = last_ranking + 1

            # P_t+1 = P_t+1 U F_i[1: (N - |P_t+1|)]
            new_parents[parents_ptr : self.population_size] = r_population.filter(fronts[last_whole_front_idx + 1])[
                trimmed_and_sorted_indices
            ]
            new_parents_solutions[parents_ptr : self.population_size] = r_solutions.filter(
                fronts[last_whole_front_idx + 1]
            )[trimmed_and_sorted_indices]

            # compute fitness (see above for details)
            max_no_inf = (
                np.nanmax(distances[trimmed_and_sorted_indices][distances[trimmed_and_sorted_indices] != np.inf])
                if len(trimmed_and_sorted_indices) > 2
                else np.ones(len(trimmed_and_sorted_indices))  # we have 1 or 2 boundary points
            )
            distances_no_inf = np.nan_to_num(distances[trimmed_and_sorted_indices], posinf=max_no_inf * 1.1)

            normalized_distances = (distances_no_inf - (distances_no_inf.min() - 1e-6)) / (
                distances_no_inf.max() - (distances_no_inf.min() - 1e-6)
            )

            reversed_distances = 1.0 - normalized_distances

            front_fitness = reversed_distances + rankings[parents_ptr : self.population_size]
            fitness_values[parents_ptr : parents_ptr + self.population_size] = front_fitness

        # back to polars, return values
        solutions = pl.DataFrame(new_parents_solutions, schema=parents[0].schema)
        outputs = pl.DataFrame(new_parents, schema=parents[1].schema)

        # check feasible/infeasible solutions
        feasible_mask = outputs[self.relaxed_constraint_symbol] <= self.constraint_threshold
        max_feasible_fitness = np.max(fitness_values[feasible_mask]) if feasible_mask.sum() > 0 else 0.0

        # Penalize non-feasible
        fitness_values[~feasible_mask] += max_feasible_fitness

        self.fitness = fitness_values

        self.notify()
        return solutions, outputs

    def state(self) -> Sequence[Message]:
        """Return the state of the selector."""
        if self.verbosity == 0 or self.selection is None or self.selected_targets is None:
            return []
        if self.verbosity == 1:
            return [
                DictMessage(
                    topic=SelectorMessageTopics.STATE,
                    value={
                        "population_size": self.population_size,
                        "selected_individuals": self.selection,
                    },
                    source=self.__class__.__name__,
                )
            ]
        # verbosity == 2
        if isinstance(self.selected_individuals, pl.DataFrame):
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=pl.concat([self.selected_individuals, self.selected_targets], how="horizontal"),
                source=self.__class__.__name__,
            )
        else:
            warnings.warn("Population is not a Polars DataFrame. Defaulting to providing OUTPUTS only.", stacklevel=2)
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=self.selected_targets,
                source=self.__class__.__name__,
            )
        return [
            DictMessage(
                topic=SelectorMessageTopics.STATE,
                value={
                    "population_size": self.population_size,
                    "selected_individuals": self.selection,
                },
                source=self.__class__.__name__,
            ),
            message,
            NumpyArrayMessage(
                topic=SelectorMessageTopics.SELECTED_FITNESS,
                value=self.fitness,
                source=self.__class__.__name__,
            ),
        ]

    def update(self, message: Message) -> None:
        pass


class _SingleObjectiveConstrainedRankingSelector(BaseSelector):
    """Implements a single-objective selector.

    This operator ranks solutions according to a single objective. Solutions
    with the best value are chosen. Considers also infeasible solutions by
    ranking them according to their constraint violation value. The selected
    population will consists of both feasible and infeasible solutions, which
    are picked alternatively from the original population based on the two
    rankings.
    """

    @property
    def provided_topics(self):
        """The topics provided for the operator."""
        return {
            0: [],
            1: [SelectorMessageTopics.STATE],
            2: [SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS, SelectorMessageTopics.SELECTED_FITNESS],
        }

    @property
    def interested_topics(self):
        """The topics the operator is interested in."""
        return []

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        population_size: int,
        target_objective_symbol: str,
        mode: str = "alternate",
        target_constraint_symbol: str | None = None,
        seed: int = 0,
        constraint_threshold: float = 0,
    ):
        super().__init__(problem=problem, verbosity=verbosity, publisher=publisher, seed=seed)
        self.population_size = population_size
        self.seed = seed
        self.selection: list[int] | None = None
        self.selected_individuals: SolutionType | None = None
        self.selected_targets: pl.DataFrame | None = None
        self.mode = mode
        self.target_objective_symbol = target_objective_symbol
        self.constraint_threshold = constraint_threshold
        self.target_constraint_symbol = target_constraint_symbol

    def do(
        self, parents: tuple[SolutionType, pl.DataFrame], offsprings: tuple[SolutionType, pl.DataFrame]
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Perform the selection operation."""
        # combine the population
        target = self.target_objective_symbol + "_min"
        solutions = parents[0].vstack(offsprings[0])
        population = parents[1].vstack(offsprings[1])

        target_arr = population[target].to_numpy()
        constraint_arr = (
            population[self.target_constraint_symbol].to_numpy() if self.target_constraint_symbol is not None else None
        )

        # rank feasible solutions according to their fitness
        target_ranks = target_arr.argsort()[
            constraint_arr[target_arr.argsort()] <= 0 if constraint_arr is not None else True
        ].squeeze()

        # rank infeasible solutions according to their constraint violation value
        # sort from lowest to high, filter positive values, then lowest value is better
        constraint_ranks = (
            constraint_arr.argsort()[constraint_arr[constraint_arr.argsort()] > 0].squeeze()
            if constraint_arr is not None
            else np.array([], dtype=int)
        )

        # form a new population by picking solutions alternating between the rankings
        order = np.ones(self.population_size, dtype=int) * -1

        if self.mode == "alternate":
            # do alternatve picking
            for i in range(self.population_size):
                # pick alternating, even i feasible rank, uneven infeasible rank
                if (remaining_targets := ~np.isin(target_ranks, order)).any() and i % 2 == 0:
                    # feasible rank
                    order[i] = target_ranks[remaining_targets][0]
                elif (remaining_constraints := ~np.isin(constraint_ranks, order)).any():
                    # infeasible rank
                    order[i] = constraint_ranks[remaining_constraints][0]
                else:
                    # feasible rank
                    remaining_targets = ~np.isin(target_ranks, order)
                    order[i] = target_ranks[remaining_targets][0]

        elif self.mode == "baseline":
            # do baseline fitness
            # feasbibles always better, cut at population size
            order = np.concat(
                (
                    np.atleast_1d(target_ranks),
                    np.atleast_1d(constraint_ranks),
                ),
            )[: self.population_size]

        elif self.mode == "baseline2":
            target_arr = population[target].to_numpy()

            tol = 1e-8
            target_arr_rounded = np.round(target_arr / tol) * tol

            _, unique_idx = np.unique(target_arr_rounded, return_index=True)

            order = np.concat(
                (
                    np.atleast_1d(target_ranks),
                    np.atleast_1d(constraint_ranks),
                ),
            )

            # Drop out duplicates
            order = order[np.isin(order, unique_idx)][: self.population_size]

        elif self.mode == "relaxed":
            relaxed_target_ranks = target_arr.argsort()[
                constraint_arr[target_arr.argsort()] <= self.constraint_threshold
                if constraint_arr is not None
                else True
            ].squeeze()
            relaxed_constraint_ranks = (
                constraint_arr.argsort()[constraint_arr[constraint_arr.argsort()] > self.constraint_threshold].squeeze()
                if constraint_arr is not None
                else []
            )
            order_original = np.concat(
                (
                    np.atleast_1d(target_ranks),
                    np.atleast_1d(constraint_ranks),
                )
            )
            order_relaxed = np.concat(
                (
                    np.atleast_1d(relaxed_target_ranks),
                    np.atleast_1d(relaxed_constraint_ranks),
                )
            )
            for i in range(self.population_size):
                # pick alternating, even i feasible rank, uneven infeasible rank
                if (remaining_original := ~np.isin(order_original, order)).any() and i % 2 == 0:
                    # feasible rank
                    order[i] = order_original[remaining_original][0]
                elif (remaining_relaxed := ~np.isin(order_relaxed, order)).any():
                    # infeasible rank
                    order[i] = order_relaxed[remaining_relaxed][0]
                else:
                    # feasible rank
                    remaining_original = ~np.isin(order_original, order)
                    order[i] = order_original[remaining_original][0]

        elif self.mode == "ranking":
            # 1. compute two ranking for each solution, one with the original constraint, one with the relaxed one
            #   - in both cases, rank feasble solutions according to objective function value, and rank infeasible
            #     solutions accordin to constraint violation value.
            # 2. use the resulting fitness value to do non-dominated sorting + crowding distance
            # 3. based on 2. assign fitness values to each solution {front}.{crowding distance}

            # 1.
            relaxed_target_ranks = target_arr.argsort()[
                constraint_arr[target_arr.argsort()] <= self.constraint_threshold
                if constraint_arr is not None
                else True
            ].squeeze()
            relaxed_constraint_ranks = (
                constraint_arr.argsort()[constraint_arr[constraint_arr.argsort()] > self.constraint_threshold].squeeze()
                if constraint_arr is not None
                else []
            )
            order_original = np.concat(
                (
                    np.atleast_1d(target_ranks),
                    np.atleast_1d(constraint_ranks),
                )
            )
            order_relaxed = np.concat(
                (
                    np.atleast_1d(relaxed_target_ranks),
                    np.atleast_1d(relaxed_constraint_ranks),
                )
            )
            # 2.
            fitness_original = np.arange(population.shape[0])[order_original.argsort()]
            fitness_relaxed = np.arange(population.shape[0])[order_relaxed.argsort()]

            # non-dominated sorting, get fronts
            pseudo_objectives = np.stack((fitness_original, fitness_relaxed), axis=-1)
            fronts = fast_non_dominated_sort(pseudo_objectives)

            # compute front ranks
            front_ranks = [[i] * np.sum(row) for i, row in enumerate(fronts)]

            # crowding distances, not normalized between 0 and <1, ascending order
            f_mins = pseudo_objectives.min(axis=0)
            f_maxs = pseudo_objectives.max(axis=0)
            distances_raw = [
                _nsga2_crowding_distance_assignment(pseudo_objectives[front], f_mins, f_maxs) for front in fronts
            ]

            # normalized between 0 and <1, descending order, preserves ordering, but not relative differences
            distance_ranks = [
                1
                - ((unique_vals_and_inv := np.unique(front_distances, return_inverse=True))[1] + 1)
                / (len(unique_vals_and_inv[0]) + 1)
                for front_distances in distances_raw
            ]

            # combine ranks into a fitness value
            fitness_combined = np.inf * np.ones(population.shape[0])
            for i, front in enumerate(fronts):
                fitness_combined[front] = front_ranks[i] + distance_ranks[i]

            # ordering
            order = fitness_combined.argsort()[: self.population_size]

        else:
            pass

        # remember to compute fitness
        # just 0, 1, 2, etc.. because of order
        self.fitness = np.arange(0, self.population_size)

        # new population
        new_solutions = pl.DataFrame(solutions[order], schema=solutions.schema)
        new_outputs = pl.DataFrame(population[order], schema=population.schema)

        self.selection = order
        self.selected_individuals = new_solutions
        self.selected_targets = new_outputs

        self.notify()
        return new_solutions, new_outputs

    def state(self) -> Sequence[Message]:
        """Return the state of the selector."""
        if self.verbosity == 0 or self.selection is None or self.selected_targets is None:
            return []
        if self.verbosity == 1:
            return [
                DictMessage(
                    topic=SelectorMessageTopics.STATE,
                    value={
                        "population_size": self.population_size,
                        "selected_individuals": self.selection,
                    },
                    source=self.__class__.__name__,
                )
            ]
        # verbosity == 2
        if isinstance(self.selected_individuals, pl.DataFrame):
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=pl.concat([self.selected_individuals, self.selected_targets], how="horizontal"),
                source=self.__class__.__name__,
            )
        else:
            warnings.warn("Population is not a Polars DataFrame. Defaulting to providing OUTPUTS only.", stacklevel=2)
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=self.selected_targets,
                source=self.__class__.__name__,
            )
        return [
            DictMessage(
                topic=SelectorMessageTopics.STATE,
                value={
                    "population_size": self.population_size,
                    "selected_individuals": self.selection,
                },
                source=self.__class__.__name__,
            ),
            message,
            NumpyArrayMessage(
                topic=SelectorMessageTopics.SELECTED_FITNESS,
                value=self.fitness,
                source=self.__class__.__name__,
            ),
        ]

    def update(self, message: Message) -> None:
        pass


class SingleObjectiveConstrainedRankingSelector(BaseSelector):
    """Single-objective selector for constrained problems.

    Implements a selector suitable for single-objective optimization of constrained problems.
    Currently supports three modes:

    1.  Baseline: in the baseline mode, selection first happens based on the
        objective function value of feasible solutions (all constraint violations
        are 0 or less). If this is not enough to form a new population, then
        solutions are picked based on the number of violated constraints. If
        multiple solutions breach the same number, then these are ranked based on
        their total constraint violation until a large enough population is formed.
        Total constraint violation is computed as the sum of the constraint
        violations of all breached constraints, where each value is first normalized
        to reside between 0 and 1 (based on the minimum and maximum value of the
        constraint in the current population). Selection will filter for unique
        solutions (based on their objective function value).
    2.  Relaxed: in the relaxed mode, each solution is ranked c+1 times, where
        c is the number of constraints. These ranking are collected into pools.
        The first pool will always contain rankings based on the same selection
        scheme as in the baseline mode. In the remaining c pools, a selection
        scheme similar to the baseline is also applied, but now a constraint
        is relaxed based on a supplied threshold value before ranking solutions
        based on it. After all the pools are formed, a new population is created
        by picking alternatively from each pool the currently best ranked solution
        (after picking, that solution is removed from all the pools). Picking
        is follows a round robin scheme, where we first pick from pool 1, then
        pool 2, and so forth until pool c+1. After that, picking starts again from
        pool 1.
    3.  Ranking: solutions are ranked as done in the relaxed mode, but now these
        ranking are considered as pseudo-objectives. Therefore, each solution will
        be represented by an objective vector, where each element corresponds to the
        ranking given to it in each of the pools. These vectors are then sorted
        based on non-dominated sorting and their crowding distances are computed
        (like in NSGA-II). After that, utilizing the non-dominated ranking and the
        distances, each solution is ranked and a new population is created based on
        these rankings.
    """

    @property
    def provided_topics(self):
        """The topics provided by the operator."""
        return {
            0: [],
            1: [SelectorMessageTopics.STATE],
            2: [SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS, SelectorMessageTopics.SELECTED_FITNESS],
        }

    @property
    def interested_topics(self):
        """Topics the operator is interested in."""
        return []

    def __init__(
        self,
        problem: Problem,
        verbosity: int,
        publisher: Publisher,
        population_size: int,
        target_objective_symbol: str,
        mode: str = "baseline",
        constraints: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        """Initializes the operator.

        Args:
            problem (Problem): the problem to be solved.
            verbosity (int): verbosity level.
            publisher (Publisher): publisher the operator subscribes to.
            population_size (int): the population size. This will be the size of the newly created population.
            target_objective_symbol (str): the symbol of the objective function that should be optimized.
            mode (str, optional): the mode the operator works in. Defaults to "baseline".
            constraints (dict[str, float] | None, optional): A dict that
                provides threshold values for each constraint present in `Problem`.
                The keys should match the constraint symbols in `Problem`. Ignored
                in the 'baseline' mode. Defaults to None.
            seed (int, optional): the seed utilized in random number generation (currently not used). Defaults to 0.
        """
        super().__init__(problem=problem, verbosity=verbosity, publisher=publisher, seed=seed)
        self.population_size = int(population_size)
        self.seed = seed
        self.mode = mode
        self.target_objective_symbol = target_objective_symbol

        # constraints: dict symbol -> threshold
        self.constraint_dict = dict(constraints) if constraints is not None else {}
        self.constraint_symbols = list(self.constraint_dict.keys())
        self.constraint_thresholds = np.asarray(list(self.constraint_dict.values()), dtype=float)

        self.selection: list[int] | None = None
        self.selected_individuals: SolutionType | None = None
        self.selected_targets: pl.DataFrame | None = None
        self.fitness: np.ndarray | None = None

    @staticmethod
    def _normalize_minmax(v: np.ndarray) -> np.ndarray:
        """Min-max normalize per column. v shape (N, K). Returns (N, K)."""
        vmin = v.min(axis=0)
        vmax = v.max(axis=0)
        denom = vmax - vmin
        denom_safe = np.where(denom == 0.0, 1.0, denom)
        out = (v - vmin) / denom_safe
        out[:, denom == 0.0] = 0.0
        return out

    @staticmethod
    def _breach_count(v: np.ndarray) -> np.ndarray:
        """Count breached constraints per row. v is nonnegative violation matrix (N, K)."""
        return (v > 0.0).sum(axis=1)

    @staticmethod
    def _objective_unique_mask(target_arr: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Return boolean mask for indices that are the first occurrence of each rounded objective value."""
        rounded = np.round(target_arr / tol) * tol
        _, first_idx = np.unique(rounded, return_index=True)
        mask = np.zeros(target_arr.shape[0], dtype=bool)
        mask[first_idx] = True
        return mask

    def _build_violation_all_original(self, c: np.ndarray) -> np.ndarray:
        """Original violation for all constraints: max(0, constraint violation value)."""
        return np.maximum(0.0, c)

    def _build_violation_single_threshold(self, cj: np.ndarray, thr: float) -> np.ndarray:
        """Thresholded violation for one constraint: max(0, constraint violation value - threshold)."""
        return np.maximum(0.0, cj - thr)

    def _order_by_objective_then_violation(
        self,
        target_arr: np.ndarray,
        violation: np.ndarray,
    ) -> np.ndarray:
        """Create an ordering for a set of constraints represented by a violation matrix.

        Rules:
          - feasible (no breach) ranked first by objective
          - infeasible ranked by (breach_count, sum(normalized_violation)), then objective
        """
        v = violation[:, None] if violation.ndim == 1 else violation

        breach = self._breach_count(v)
        feas_mask = breach == 0

        # rank feasible solutions
        feas_idx = np.where(feas_mask)[0]
        feas_sorted = feas_idx[np.argsort(target_arr[feas_idx], kind="mergesort")]

        # infeasible: by (breach_count, sum_norm_violation), then objective
        infeas_idx = np.where(~feas_mask)[0]
        if infeas_idx.size == 0:
            # no breaches
            return feas_sorted.astype(int)

        # rank infeasible solutions first by breach count, then sum of normalized violation, and then objective value
        v_infeas = v[infeas_idx, :]
        vn_infeas = self._normalize_minmax(v_infeas)
        sum_vn = vn_infeas.sum(axis=1)
        breach_infeas = breach[infeas_idx]

        keys = (target_arr[infeas_idx], sum_vn, breach_infeas)
        infeas_sorted = infeas_idx[np.lexsort(keys)]

        return np.concatenate([feas_sorted, infeas_sorted]).astype(int)

    def _round_robin_from_pools(self, pool_orders: list[np.ndarray], m: int) -> np.ndarray:
        """Round-robin pick across pool orderings, removing chosen indices from each pool."""
        if not pool_orders:
            return np.array([], dtype=int)

        n = pool_orders[0].shape[0]

        # keeps track of already chosen solutions
        chosen = np.zeros(n, dtype=bool)

        # pointer for each pool, points to elements in each pool
        cursors = [0] * len(pool_orders)

        out = np.ones(m, dtype=int) * -1

        # keep track of populated pools
        active = [True] * len(pool_orders)

        p = 0  # pool 0 is always the feasible pool, start from it
        i = 0
        while i < m:
            # iterate until population is full or all pools are exhausted
            if not any(active):
                # no more populated pools
                break

            # find next active pool
            tries = 0
            while tries < len(pool_orders) and not active[p]:
                p = (p + 1) % len(pool_orders)
                tries += 1
            if tries == len(pool_orders) and not active[p]:
                break

            order = pool_orders[p]
            k = cursors[p]
            while k < len(order) and chosen[order[k]]:
                k += 1
            cursors[p] = k

            if k >= len(order):
                active[p] = False
                p = (p + 1) % len(pool_orders)
                continue

            idx = int(order[k])
            out[i] = idx
            chosen[idx] = True
            cursors[p] += 1

            p = (p + 1) % len(pool_orders)
            i += 1

        return out[out >= 0]

    def do(
        self, parents: tuple[SolutionType, pl.DataFrame], offsprings: tuple[SolutionType, pl.DataFrame]
    ) -> tuple[SolutionType, pl.DataFrame]:
        """Run the operator.

        Args:
            parents (tuple[SolutionType, pl.DataFrame]): parent population.
            offsprings (tuple[SolutionType, pl.DataFrame]): offspring population.

        Raises:
            ValueError: unsupported `mode`.

        Returns:
            tuple[SolutionType, pl.DataFrame]: a new population selected from the parent and offspring populations.
        """
        target_col = self.target_objective_symbol + "_min"

        solutions = parents[0].vstack(offsprings[0])
        population = parents[1].vstack(offsprings[1])

        target_arr = population[target_col].to_numpy()
        n = population.shape[0]

        # Constraints matrix (n, k). If no constraints, treat as empty.
        if len(self.constraint_symbols) > 0:
            c_mat = population.select(self.constraint_symbols).to_numpy()
        else:
            c_mat = np.zeros((n, 0), dtype=float)

        if self.mode == "baseline":
            # Full ordering: feasible-by-objective first, then infeasible by (breach count, sum normalized violation)
            if c_mat.shape[1] == 0:
                order_full = np.argsort(target_arr, kind="mergesort")
            else:
                v0 = self._build_violation_all_original(c_mat)  # (n, k)
                order_full = self._order_by_objective_then_violation(target_arr, v0)

            # Filter unique solutions
            unique_mask = self._objective_unique_mask(target_arr, tol=1e-8)
            order_full = order_full[unique_mask[order_full]]

            order = order_full[: self.population_size]

        elif self.mode == "relaxed":
            # first pool, ignores thresholds
            pool_orders: list[np.ndarray] = []
            if c_mat.shape[1] == 0:
                # no constraints
                pool0 = np.argsort(target_arr, kind="mergesort")
                pool_orders.append(pool0.astype(int))
            else:
                # one or more constraints
                v0 = self._build_violation_all_original(c_mat)
                pool0 = self._order_by_objective_then_violation(target_arr, v0)
                pool_orders.append(pool0)

            # rest of pools, per-constraint threshold, feasibility based only on that constraint
            for j in range(c_mat.shape[1]):
                cj = c_mat[:, j]
                thr = float(self.constraint_thresholds[j])
                vj = self._build_violation_single_threshold(cj, thr)
                poolj = self._order_by_objective_then_violation(target_arr, vj)
                pool_orders.append(poolj)

            order = self._round_robin_from_pools(pool_orders, self.population_size)

        elif self.mode == "ranking":
            if c_mat.shape[1] == 0:
                order = np.argsort(target_arr, kind="mergesort")[: self.population_size]
            else:
                # c + 1 rankings (orderings)
                orderings: list[np.ndarray] = []

                # First ranking with original constraints
                v0 = self._build_violation_all_original(c_mat)
                ord0 = self._order_by_objective_then_violation(target_arr, v0)
                orderings.append(ord0)

                # Remaining ranking, each constraint with its threshold (single constraint)
                for j in range(c_mat.shape[1]):
                    cj = c_mat[:, j]
                    thr = float(self.constraint_thresholds[j])
                    vj = self._build_violation_single_threshold(cj, thr)
                    orderings.append(self._order_by_objective_then_violation(target_arr, vj))

                # Convert to rank vectors (pseudo objective vectors)
                rank_vectors = np.empty((n, len(orderings)), dtype=float)
                for k, ordk in enumerate(orderings):
                    r = np.empty(n, dtype=float)
                    r[ordk] = np.arange(n, dtype=float)
                    rank_vectors[:, k] = r

                # Non-dominated sorting + crowding distance (NSGA-II style)
                fronts = fast_non_dominated_sort(rank_vectors)

                front_ranks = [[i] * np.sum(row) for i, row in enumerate(fronts)]

                f_mins = rank_vectors.min(axis=0)
                f_maxs = rank_vectors.max(axis=0)
                distances_raw = [
                    _nsga2_crowding_distance_assignment(rank_vectors[front], f_mins, f_maxs) for front in fronts
                ]

                distance_ranks = [
                    1
                    - ((unique_vals_and_inv := np.unique(front_distances, return_inverse=True))[1] + 1)
                    / (len(unique_vals_and_inv[0]) + 1)
                    for front_distances in distances_raw
                ]

                fitness_combined = np.inf * np.ones(n)
                for i, front in enumerate(fronts):
                    fitness_combined[front] = front_ranks[i] + distance_ranks[i]

                order = fitness_combined.argsort(kind="mergesort")[: self.population_size]

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'. Expected one of: baseline2, relaxed, ranking.")

        order = np.atleast_1d(order).astype(int)
        self.fitness = np.arange(0, len(order))

        new_solutions = pl.DataFrame(solutions[order], schema=solutions.schema)
        new_outputs = pl.DataFrame(population[order], schema=population.schema)

        self.selection = order.tolist()
        self.selected_individuals = new_solutions
        self.selected_targets = new_outputs

        self.notify()
        return new_solutions, new_outputs

    def state(self) -> Sequence[Message]:
        """Return the state of the operator.

        Returns:
            Sequence[Message]: the current state.
        """
        if self.verbosity == 0 or self.selection is None or self.selected_targets is None:
            return []

        if self.verbosity == 1:
            return [
                DictMessage(
                    topic=SelectorMessageTopics.STATE,
                    value={"population_size": self.population_size, "selected_individuals": self.selection},
                    source=self.__class__.__name__,
                )
            ]

        if isinstance(self.selected_individuals, pl.DataFrame):
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=pl.concat([self.selected_individuals, self.selected_targets], how="horizontal"),
                source=self.__class__.__name__,
            )
        else:
            warnings.warn("Population is not a Polars DataFrame. Defaulting to providing OUTPUTS only.", stacklevel=2)
            message = PolarsDataFrameMessage(
                topic=SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
                value=self.selected_targets,
                source=self.__class__.__name__,
            )

        return [
            DictMessage(
                topic=SelectorMessageTopics.STATE,
                value={"population_size": self.population_size, "selected_individuals": self.selection},
                source=self.__class__.__name__,
            ),
            message,
            NumpyArrayMessage(
                topic=SelectorMessageTopics.SELECTED_FITNESS,
                value=self.fitness,
                source=self.__class__.__name__,
            ),
        ]

    def update(self, message: Message) -> None:
        """Update the state of the operator. Not used.

        Args:
            message (Message): message used to update the state.
        """
