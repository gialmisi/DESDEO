"""Cat and dog breed problems based on Finnish feline and canine survey studies.

The problems are discrete representations of the Pareto optimal breed groups
found in two survey studies that charted the behaviour and personality traits
of cat and dog breeds in Finland. Each row of the underlying data corresponds
to one surveyed animal belonging to a breed group, and the objectives are the
behavioural trait scores measured for that animal.

The data files shipped with DESDEO contain only the non-dominated rows of the
original datasets, with each trait score min-max normalized to the range
`[0, 100]` across the non-dominated set. Traits that are desirable were
negated before the non-dominated sorting was carried out, which leaves them in
the range `[-100, 0]` in the data files. They are shifted back to `[0, 100]`
here so that every objective is reported on a common `0` to `100` scale, and
the desired direction of each objective is carried by its `maximize` flag
alone.

The problems are intended for demonstration purposes. The original modelling
was done for a Researchers' Night demo, and the problem formulation should not
be taken as a serious model of breed selection.
"""

from pathlib import Path

import polars as pl

from desdeo.problem.schema import (
    DiscreteRepresentation,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    Variable,
    VariableTypeEnum,
)

_DATA_DIR = Path(__file__).parents[3] / "datasets"
_CAT_DATA_PATH = _DATA_DIR / "pf_cat_data.csv"
_DOG_DATA_PATH = _DATA_DIR / "pf_dog_data.csv"

_ROW_VARIABLE = "index"
_BREED_VARIABLE = "breed_id"

# The name of the column in the data files that holds the breed group of each row.
_CAT_BREED_COLUMN = "BREEDGROUP"
_DOG_BREED_COLUMN = "breed_group"

# Objective symbol, display name, and whether the objective is to be maximized.
# The display names carry an emoji on purpose: these problems back a playful
# science outreach demo, and the emoji is part of how the traits are presented
# to the decision maker.
_CAT_OBJECTIVES: list[tuple[str, str, bool]] = [
    ("fearfulness", "Fearfulness 🙀", False),
    ("human_aggression", "Aggression towards people 😾", False),
    ("activity_playfulness", "Activity and playfulness 🧶", True),
    ("cat_sociability", "Sociability towards cats 😻", True),
    ("human_sociability", "Sociability towards people 🤝", True),
    ("litterbox_issues", "Litterbox issues 🚽", False),
    ("excessive_grooming", "Excessive grooming 💈", False),
]

_DOG_OBJECTIVES: list[tuple[str, str, bool]] = [
    ("insecurity_score", "Insecurity 😨", False),
    ("training_focus_score", "Trainability 🎓", True),
    ("activity_playfulness_score", "Activity and playfulness 🎾", True),
    ("aggressiveness_dominance_score", "Aggressiveness and dominance 😤", False),
    ("human_sociability_score", "Sociability towards people 🤝", True),
    ("dog_sociability_score", "Sociability towards dogs 🐶", True),
    ("perseverance_score", "Perseverance 🦴", True),
]


def _breed_problem(
    data_path: Path,
    breed_column: str,
    objective_spec: list[tuple[str, str, bool]],
    name: str,
    description: str,
) -> Problem:
    """Build a discrete breed problem from one of the shipped data files.

    Args:
        data_path (Path): path to the CSV file with the non-dominated survey data.
        breed_column (str): name of the column holding the breed group of each row.
        objective_spec (list[tuple[str, str, bool]]): the symbol, display name, and
            maximization flag of each objective, in the order they appear in the problem.
        name (str): the name of the resulting problem.
        description (str): the description of the resulting problem.

    Returns:
        Problem: the breed problem with a discrete representation of its Pareto front.
    """
    symbols = [symbol for symbol, _, _ in objective_spec]

    data = pl.read_csv(data_path, columns=[breed_column, *symbols])

    # Breed groups are given by name in the data. They are numbered here so that
    # the breed group of a solution can be read straight off the decision
    # variables, and the numbering is exposed by `breed_group_names`.
    breed_names = sorted(set(data[breed_column].to_list()))
    breed_ids = {breed_name: breed_id for breed_id, breed_name in enumerate(breed_names)}

    # Objectives to be maximized are stored negated in the data files. Shifting
    # them by 100 puts every objective on a common 0 to 100 scale without
    # changing which rows are non-dominated.
    data = data.with_columns(
        [(pl.col(symbol) + 100.0).alias(symbol) for symbol, _, maximize in objective_spec if maximize]
    )

    objectives = [
        Objective(
            name=objective_name,
            symbol=symbol,
            objective_type=ObjectiveTypeEnum.data_based,
            ideal=data[symbol].max() if maximize else data[symbol].min(),
            nadir=data[symbol].min() if maximize else data[symbol].max(),
            maximize=maximize,
        )
        for symbol, objective_name, maximize in objective_spec
    ]

    variables = [
        Variable(
            name="Survey response index",
            symbol=_ROW_VARIABLE,
            variable_type=VariableTypeEnum.integer,
            lowerbound=0,
            upperbound=len(data) - 1,
            initial_value=0,
        ),
        Variable(
            name="Breed group",
            symbol=_BREED_VARIABLE,
            variable_type=VariableTypeEnum.integer,
            lowerbound=0,
            upperbound=len(breed_names) - 1,
            initial_value=0,
        ),
    ]

    discrete_representation = DiscreteRepresentation(
        variable_values={
            _ROW_VARIABLE: list(range(len(data))),
            _BREED_VARIABLE: [breed_ids[breed_name] for breed_name in data[breed_column]],
        },
        objective_values=data[symbols].to_dict(as_series=False),
        non_dominated=True,
    )

    return Problem(
        name=name,
        description=description,
        variables=variables,
        objectives=objectives,
        discrete_representation=discrete_representation,
        is_twice_differentiable=False,
    )


def cat_breed_problem() -> Problem:
    """Defines a cat breed problem with seven behavioural objectives.

    Each row of the discrete representation describes one surveyed cat. The
    `index` decision variable points to the row, and the `breed_id` decision
    variable gives the breed group the cat belongs to. The name of a breed
    group can be looked up with `cat_breed_group_names`.

    Returns:
        Problem: the cat breed problem.
    """
    return _breed_problem(
        data_path=_CAT_DATA_PATH,
        breed_column=_CAT_BREED_COLUMN,
        objective_spec=_CAT_OBJECTIVES,
        name="Cat breeds",
        description=(
            "Find a cat breed group that suits your preferences. The seven objectives are behavioural "
            "trait scores from a Finnish survey study on the characteristics of cat breeds, normalized "
            "to a scale from 0 to 100. Only the non-dominated survey responses are included."
        ),
    )


def dog_breed_problem() -> Problem:
    """Defines a dog breed problem with seven behavioural objectives.

    Each row of the discrete representation describes one surveyed dog. The
    `index` decision variable points to the row, and the `breed_id` decision
    variable gives the breed group the dog belongs to. The name of a breed
    group can be looked up with `dog_breed_group_names`.

    Returns:
        Problem: the dog breed problem.
    """
    return _breed_problem(
        data_path=_DOG_DATA_PATH,
        breed_column=_DOG_BREED_COLUMN,
        objective_spec=_DOG_OBJECTIVES,
        name="Dog breeds",
        description=(
            "Find a dog breed group that suits your preferences. The seven objectives are behavioural "
            "trait scores from a Finnish survey study on the characteristics of dog breeds, normalized "
            "to a scale from 0 to 100. Only the non-dominated survey responses are included."
        ),
    )


def cat_breed_group_names() -> list[str]:
    """Returns the names of the cat breed groups, ordered by their breed id.

    The name at position `i` is the name of the breed group whose `breed_id`
    decision variable value is `i` in the problem returned by
    `cat_breed_problem`.

    Returns:
        list[str]: the breed group names, ordered by breed id.
    """
    return sorted(set(pl.read_csv(_CAT_DATA_PATH, columns=[_CAT_BREED_COLUMN])[_CAT_BREED_COLUMN].to_list()))


def dog_breed_group_names() -> list[str]:
    """Returns the names of the dog breed groups, ordered by their breed id.

    The name at position `i` is the name of the breed group whose `breed_id`
    decision variable value is `i` in the problem returned by
    `dog_breed_problem`.

    Returns:
        list[str]: the breed group names, ordered by breed id.
    """
    return sorted(set(pl.read_csv(_DOG_DATA_PATH, columns=[_DOG_BREED_COLUMN])[_DOG_BREED_COLUMN].to_list()))
