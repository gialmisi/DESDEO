"""This module contains the basic functional implementations for the EMO methods.

This can be used as a template for the implementation of the EMO methods.
"""

from collections.abc import Callable

import polars as pl

from desdeo.emo.operators.crossover import BaseCrossover
from desdeo.emo.operators.evaluator import EMOEvaluator
from desdeo.emo.operators.generator import BaseGenerator
from desdeo.emo.operators.mutation import BaseMutation
from desdeo.emo.operators.scalar_selection import BaseScalarSelector
from desdeo.emo.operators.selection import BaseSelector
from desdeo.emo.operators.termination import BaseTerminator
from desdeo.emo.operators.xlemoo_selection import XLEMOOInstantiator
from desdeo.tools.generics import EMOResult


def template1(
    evaluator: EMOEvaluator,
    crossover: BaseCrossover,
    mutation: BaseMutation,
    generator: BaseGenerator,
    selection: BaseSelector,
    terminator: BaseTerminator,
    repair: Callable[[pl.DataFrame], pl.DataFrame] = lambda x: x,  # Default to identity function if no repair is needed
) -> EMOResult:
    """Implements a template that many EMO methods, such as RVEA and NSGA-III, follow.

    Args:
        evaluator (EMOEvaluator): A class that evaluates the solutions and provides the objective vectors, constraint
            vectors, and targets.
        crossover (BaseCrossover): The crossover operator.
        mutation (BaseMutation): The mutation operator.
        generator (BaseGenerator): A class that generates the initial population.
        selection (BaseSelector): The selection operator.
        terminator (BaseTerminator): The termination operator.
        repair (Callable, optional): A function that repairs the offspring if they go out of bounds. Defaults to an
            identity function, meaning no repair is done. See :py:func:`desdeo.tools.utils.repair` as an example of a
            repair function.

    Returns:
        EMOResult: The final population and their objective vectors, constraint vectors, and targets
    """
    solutions, outputs = generator.do()

    while not terminator.check():
        offspring = crossover.do(population=solutions)
        offspring = mutation.do(offspring, solutions)
        # Repair offspring if they go out of bounds
        offspring = repair(offspring)
        offspring_outputs = evaluator.evaluate(offspring)
        solutions, outputs = selection.do(parents=(solutions, outputs), offsprings=(offspring, offspring_outputs))

    return EMOResult(optimal_variables=solutions, optimal_outputs=outputs)


def template2(
    evaluator: EMOEvaluator,
    crossover: BaseCrossover,
    mutation: BaseMutation,
    generator: BaseGenerator,
    selection: BaseSelector,
    mate_selection: BaseScalarSelector,
    terminator: BaseTerminator,
    repair: Callable[[pl.DataFrame], pl.DataFrame] = lambda x: x,  # Default to identity function if no repair is needed
) -> EMOResult:
    """Implements a template that many EMO methods, such as IBEA, follow.

    Args:
        evaluator (EMOEvaluator): A class that evaluates the solutions and provides the objective vectors, constraint
            vectors, and targets.
        crossover (BaseCrossover): The crossover operator.
        mutation (BaseMutation): The mutation operator.
        generator (BaseGenerator): A class that generates the initial population.
        selection (BaseSelector): The selection operator.
        mate_selection (BaseScalarSelector): The mating selection operator, which selects parents for mating.
            This is typically a scalar selector that selects parents based on their fitness.
        terminator (BaseTerminator): The termination operator.
        repair (Callable, optional): A function that repairs the offspring if they go out of bounds. Defaults to an
            identity function, meaning no repair is done. See :py:func:`desdeo.tools.utils.repair` as an example of a
            repair function.

    Returns:
        EMOResult: The final population and their objective vectors, constraint vectors, and targets
    """
    solutions, outputs = generator.do()
    # This is just a hack to make all selection operators work (they require offsprings to be passed separately rn)
    offspring = pl.DataFrame(
        schema=solutions.schema,
    )
    offspring_outputs = pl.DataFrame(
        schema=outputs.schema,
    )

    while True:
        solutions, outputs = selection.do(parents=(solutions, outputs), offsprings=(offspring, offspring_outputs))
        if terminator.check():
            # Weird way to do looping, but IBEA does environmental selection before the loop check, and...
            # does mating afterwards.
            break
        parents, _ = mate_selection.do((solutions, outputs))
        offspring = crossover.do(population=parents)
        offspring = mutation.do(offspring, solutions)
        # Repair offspring if they go out of bounds
        offspring = repair(offspring)
        offspring_outputs = evaluator.evaluate(offspring)

    return EMOResult(optimal_variables=solutions, optimal_outputs=outputs)


def template3(
    evaluator: EMOEvaluator,
    crossover: BaseCrossover,
    mutation: BaseMutation,
    generator: BaseGenerator,
    selection: BaseSelector,
    learning_instantiator: XLEMOOInstantiator,
    terminator: BaseTerminator,
    darwin_iterations_per_cycle: int = 10,
    learning_iterations_per_cycle: int = 1,
    mate_selection: BaseScalarSelector | None = None,
    repair: Callable[[pl.DataFrame], pl.DataFrame] = lambda x: x,
) -> EMOResult:
    """Template for the XLEMOO hybrid evolutionary-ML method.

    Alternates between Darwinian mode (standard EA using ``selection``) and
    Learning mode (ML-based rule extraction and instantiation).

    The Darwinian mode supports two styles:

    - **Template1-style** (default): crossover is applied to the full population.
      Used with selectors like RVEA and NSGA-III.
    - **Template2-style** (when ``mate_selection`` is provided): environmental
      selection runs first, then mate selection (e.g., tournament) picks parents
      for crossover. Used with selectors like IBEA.

    In Learning mode, the ``learning_instantiator`` reads population history,
    trains an ML classifier, extracts rules, and returns candidate variable
    vectors. These candidates are evaluated by the shared ``evaluator``
    (ensuring terminator/archives get notified), and then the Darwinian
    ``selection`` operator picks the next population from both the current
    population and the ML candidates.

    Args:
        evaluator: Evaluator for objective/constraint computation.
        crossover: The crossover operator (Darwinian mode).
        mutation: The mutation operator (Darwinian mode).
        generator: Initial population generator.
        selection: The selector used in both Darwinian and Learning mode
            (e.g., RVEA, NSGA-III, IBEA).
        learning_instantiator: The XLEMOOInstantiator for candidate generation.
        terminator: The termination operator.
        darwin_iterations_per_cycle: Darwinian iterations per cycle.
        learning_iterations_per_cycle: Learning iterations per cycle.
        mate_selection: Optional mating selection operator (e.g., tournament
            selection for IBEA-style Darwinian mode). When provided, the
            Darwinian mode follows template2's pattern.
        repair: Repair function for out-of-bounds offspring.

    Returns:
        EMOResult: The final population and outputs.
    """
    solutions, outputs = generator.do()

    if mate_selection is not None:
        # Template2-style: environmental selection before crossover (IBEA pattern)
        # Initialize empty offspring for the first selection call
        offspring = pl.DataFrame(schema=solutions.schema)
        offspring_outputs = pl.DataFrame(schema=outputs.schema)

        while True:
            # Darwinian mode (template2-style)
            for _ in range(darwin_iterations_per_cycle):
                solutions, outputs = selection.do(
                    parents=(solutions, outputs), offsprings=(offspring, offspring_outputs)
                )
                if terminator.check():
                    return EMOResult(optimal_variables=solutions, optimal_outputs=outputs)
                parents, _ = mate_selection.do((solutions, outputs))
                offspring = crossover.do(population=parents)
                offspring = mutation.do(offspring, solutions)
                offspring = repair(offspring)
                offspring_outputs = evaluator.evaluate(offspring)

            # Learning mode: self-contained — selection happens inside the loop
            for _ in range(learning_iterations_per_cycle):
                solutions, outputs = selection.do(
                    parents=(solutions, outputs), offsprings=(offspring, offspring_outputs)
                )
                if terminator.check():
                    return EMOResult(optimal_variables=solutions, optimal_outputs=outputs)
                candidates = learning_instantiator.do(solutions)
                candidates_outputs = evaluator.evaluate(candidates)
                offspring = candidates
                offspring_outputs = candidates_outputs

            # Final selection from last learning iteration before returning
            # to Darwinian mode, so the population reflects ML candidates.
            solutions, outputs = selection.do(
                parents=(solutions, outputs), offsprings=(offspring, offspring_outputs)
            )
            # Re-initialize offspring for the next Darwinian cycle's first
            # selection call (same as template2 does at startup).
            offspring = pl.DataFrame(schema=solutions.schema)
            offspring_outputs = pl.DataFrame(schema=outputs.schema)
    else:
        # Template1-style: crossover on full population (RVEA/NSGA-III pattern)
        while True:
            # Darwinian mode
            for _ in range(darwin_iterations_per_cycle):
                if terminator.check():
                    return EMOResult(optimal_variables=solutions, optimal_outputs=outputs)
                offspring = crossover.do(population=solutions)
                offspring = mutation.do(offspring, solutions)
                offspring = repair(offspring)
                offspring_outputs = evaluator.evaluate(offspring)
                solutions, outputs = selection.do(
                    parents=(solutions, outputs), offsprings=(offspring, offspring_outputs)
                )

            # Learning mode
            for _ in range(learning_iterations_per_cycle):
                if terminator.check():
                    return EMOResult(optimal_variables=solutions, optimal_outputs=outputs)
                candidates = learning_instantiator.do(solutions)
                candidates_outputs = evaluator.evaluate(candidates)
                solutions, outputs = selection.do(
                    parents=(solutions, outputs), offsprings=(candidates, candidates_outputs)
                )
