"""Contains forest problem models."""

from pathlib import Path

import numpy as np
import polars as pl

from desdeo.problem import (
    Constraint,
    ConstraintTypeEnum,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    TensorConstant,
    TensorVariable,
    VariableTypeEnum,
)


def simple_forest_problem() -> Problem:
    number_of_stands = 5  # I
    number_of_years = 3  # J
    number_of_regimes = 4  # K

    # Net present value (NPV) matrices for each stand
    # These tell us what is the NPV for each stand for a selected regime and year
    # The first index is the selected year, the second is the regime.
    # E.g., NPVs[1][1, 2] is the NPV value for stand 1, second year (zero indexing) and third regime.

    NPVs = {
        1: [[250, 300, 420, 1], [330, 280, 410, 1], [280, 290, 390, 1]],
        2: [[312, 405, 478, 1], [287, 331, 462, 1], [344, 298, 411, 1]],
        3: [[276, 418, 395, 1], [392, 284, 487, 1], [315, 367, 442, 1]],
        4: [[329, 401, 356, 1], [245, 372, 498, 1], [408, 293, 475, 1]],
        5: [[351, 289, 437, 1], [304, 426, 389, 1], [267, 338, 452, 1]],
    }

    # Deadwood volume matrices for each stand
    # Same as NPV, but the value is now the deadwood volume per regime and year

    DWVs = {
        1: [[280, 350, 410, 490], [320, 290, 380, 390], [270, 330, 400, 420]],
        2: [[310, 270, 390, 480], [250, 340, 420, 470], [300, 360, 330, 360]],
        3: [[290, 370, 320, 370], [340, 260, 400, 460], [280, 350, 310, 350]],
        4: [[260, 380, 330, 380], [320, 290, 330, 370], [350, 310, 340, 350]],
        5: [[300, 270, 360, 460], [280, 340, 390, 440], [330, 310, 350, 390]],
    }

    # Constants
    constants = []

    for i in range(number_of_stands):
        constant_npv_i = TensorConstant(
            name=f"NPV for stand {i+1}",
            symbol=f"NPV_{i+1}",
            shape=[len(NPVs[i + 1]), len(NPVs[i + 1][0])],
            values=NPVs[i + 1],
        )

        constants.append(constant_npv_i)

        constant_dwv_i = TensorConstant(
            name=f"DWV for stand {i+1}",
            symbol=f"DWV_{i+1}",
            shape=[len(DWVs[i + 1]), len(DWVs[i + 1][0])],
            values=DWVs[i + 1],
        )

        constants.append(constant_dwv_i)

    # Decision variables
    # Each variable represents a stand.
    # Each variable is a matrix of binary values.
    # The first index is a year and the second a selected regime.
    # E.g., X_1[2, 0] = 1 means that for stand 1, the first regime has been selected on the third year (zero indexing).
    # A value of 0 indicates that for a particular year, a regime is not selected.

    variables = []

    initial_values = np.full((number_of_years, number_of_regimes), [1, 0, 0, 0]).tolist()

    for i in range(number_of_stands):
        var_i = TensorVariable(
            name=f"Stand {i+1}",
            symbol=f"X_{i+1}",
            variable_type=VariableTypeEnum.binary,
            shape=[number_of_years, number_of_regimes],
            lowerbounds=0,
            upperbounds=1,
            initial_values=initial_values,
        )

        variables.append(var_i)

    # Constraints
    # For each stand, one regime, and only one, must be selected for each year.

    constraints = []

    for i in range(number_of_stands):
        for j in range(number_of_years):
            constraint_ij = Constraint(
                name=f"Row {j+1} of stand {i+1} must sum to one.",
                symbol=f"row_constraint_{i}{j}",
                # minus 1 because constraint must equal 0
                func=" + ".join(f"X_{i+1}[{j+1}, {r+1}]" for r in range(number_of_regimes)) + " - 1",
                cons_type=ConstraintTypeEnum.LTE,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )

            constraints.append(constraint_ij)

            constraint_ij = Constraint(
                name=f"Row {j+1} of stand {i+1} must sum to one.",
                symbol=f"row_constraint_{i}{j}_",
                # minus 1 because constraint must equal 0
                func="-1*(" + " + ".join(f"X_{i+1}[{j+1}, {r+1}]" for r in range(number_of_regimes)) + ") + 1",
                cons_type=ConstraintTypeEnum.LTE,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )

            constraints.append(constraint_ij)

    # Objectives
    objectives = []

    ## NPV sum
    npv_sum_expr = "Sum(" + " + ".join([f"X_{i+1} * NPV_{i+1}" for i in range(number_of_stands)]) + ")"
    npv_objective = Objective(
        name="NPV",
        symbol="NPV",
        func=npv_sum_expr,
        is_convex=True,
        is_linear=True,
        is_twice_differentiable=True,
        maximize=True,
        objective_type=ObjectiveTypeEnum.analytical,
        unit="Million euros",
    )
    objectives.append(npv_objective)

    ## DWV
    dwv_sum_expr = "Sum(" + " + ".join([f"X_{i+1} * DWV_{i+1}" for i in range(number_of_stands)]) + ")"
    dwv_objective = Objective(
        name="DWV",
        symbol="DWV",
        func=dwv_sum_expr,
        is_convex=True,
        is_linear=True,
        is_twice_differentiable=True,
        maximize=True,
        objective_type=ObjectiveTypeEnum.analytical,
        unit="Metric tons",
    )

    objectives.append(dwv_objective)

    # Construct and return problem
    return Problem(
        name="Simple forest problem",
        description=(
            f"Simple forest problem with {number_of_stands} stands, "
            f"{number_of_regimes} regimes, over a time horizon of {number_of_years}"
        ),
        constants=constants,
        variables=variables,
        constraints=constraints,
        objectives=objectives,
    )


def forest_problem_vaaler(
    input_file_name: str = "./data/Vaaler_DATA_50.csv", output_name: str = "./data/out.json"
) -> Problem:
    """Construct the forest problem for data stored in a CSV file."""

    # load data
    df = pl.read_csv(input_file_name, separator=",", has_header=True)

    # Sort the dataframe for efficient access
    df_sorted = df.sort(["id", "year", "branch"])

    unique_stands = df["id"].unique().to_list()
    n_stands = len(unique_stands)
    unique_regimes = df["branch"].unique().to_list()
    n_regimes = len(unique_regimes)
    years = [df["year"].unique().to_list()[0]]
    n_years = len(years)

    NPVs = {}
    DWVs = {}

    for i_, stand_id in enumerate(unique_stands):
        print(f"Processing stand {i_+1}/{n_stands+1}")
        stand_data = df_sorted.filter(pl.col("id") == stand_id)

        # Convert to numpy for faster indexing
        stand_array = stand_data.select(["year", "branch", "NPV", "Harvested_V"]).to_numpy()

        # missing values will be set to -1
        npv_data = np.full((len(years), len(unique_regimes)), -1)
        dwv_data = np.full((len(years), len(unique_regimes)), -1)

        for i, year in enumerate(years):
            for j, regime in enumerate(unique_regimes):
                match = stand_array[(stand_array[:, 0] == year) & (stand_array[:, 1] == regime)]
                if match.size > 0:
                    npv_data[i, j] = match[0, 2]  # NPV
                    dwv_data[i, j] = match[0, 3]  # Harvested_V

        NPVs[stand_id] = npv_data.tolist()
        DWVs[stand_id] = dwv_data.tolist()

    # Constants
    constants = []

    for stand_id in unique_stands:
        constant_npv_i = TensorConstant(
            name=f"NPV for stand id={stand_id}",
            symbol=f"NPV_{stand_id}",
            shape=[n_years, n_regimes],
            values=NPVs[stand_id],
        )

        constants.append(constant_npv_i)

        constant_dwv_i = TensorConstant(
            name=f"DWV for stand id={stand_id}",
            symbol=f"DWV_{stand_id}",
            shape=[n_years, n_regimes],
            values=DWVs[stand_id],
        )

        constants.append(constant_dwv_i)

    variables = []

    for stand_id in unique_stands:
        var_i = TensorVariable(
            name=f"Stand id={stand_id}",
            symbol=f"X_{stand_id}",
            variable_type=VariableTypeEnum.binary,
            shape=[n_years, n_regimes],
            lowerbounds=0,
            upperbounds=1,
            initial_values=0,
        )

        variables.append(var_i)

    # Constraints
    # For each stand, one regime, and only one, must be selected for each year.

    constraints = []

    for i, stand_id in enumerate(unique_stands):
        print(f"Processing stand {i+1}/{n_stands+1}")
        for j, year in enumerate(years):
            constraint_ij = Constraint(
                name=f"Year {year} of stand id={stand_id} sum to one.",
                symbol=f"row_sum_{stand_id}_{year}",
                # minus 1 because constraint must equal 0
                func=" + ".join(f"X_{stand_id}[{j+1}, {r+1}]" for r in range(n_regimes)) + " - 1",
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )

            constraints.append(constraint_ij)

            # if a regime has no value for an objective, it can never be active for the stand and year
            # check NPV
            #
            # maybe needed later, but missing values are just set to be -1 because maximization
            """
            for k, regime in enumerate(unique_regimes):
                if NPVs[stand_id][j][k] is None:
                    # print(f"Found note for stand id={stand_id}, year={year} and regime={regime}")
                    constraint_null = Constraint(
                        name="Null value",
                        symbol=f"null_{stand_id}_{year}_{regime}",
                        func=f"X_{stand_id}{[j+1, k+1]}",  # must equal zero
                        cons_type=ConstraintTypeEnum.EQ,
                        is_linear=True,
                        is_convex=True,
                        is_twice_differentiable=True,
                    )

                    constraints.append(constraint_null)
            """
    # Objectives
    objectives = []

    ## NPV sum
    npv_sum_expr = "Sum(" + " + ".join([f"X_{stand_id} * NPV_{stand_id}" for stand_id in unique_stands]) + ")"
    npv_objective = Objective(
        name="NPV",
        symbol="NPV",
        func=npv_sum_expr,
        is_convex=True,
        is_linear=True,
        is_twice_differentiable=True,
        maximize=True,
        objective_type=ObjectiveTypeEnum.analytical,
        unit="Million euros",
    )
    objectives.append(npv_objective)

    ## DWV
    dwv_sum_expr = "Sum(" + " + ".join([f"X_{stand_id} * DWV_{stand_id}" for stand_id in unique_stands]) + ")"
    dwv_objective = Objective(
        name="DWV",
        symbol="DWV",
        func=dwv_sum_expr,
        is_convex=True,
        is_linear=True,
        is_twice_differentiable=True,
        maximize=True,
        objective_type=ObjectiveTypeEnum.analytical,
        unit="Metric tons",
    )

    objectives.append(dwv_objective)

    # Construct and return problem
    problem = Problem(
        name="Simple forest problem",
        description=(
            f"Simple forest problem with {n_stands} stands, " f"{n_regimes} regimes, over a time horizon of {n_years}"
        ),
        constants=constants,
        variables=variables,
        constraints=constraints,
        objectives=objectives,
    )

    # save to file for quicker access
    json_data = problem.model_dump_json()

    with Path(output_name).open("w") as f:
        f.write(json_data)

    return problem
