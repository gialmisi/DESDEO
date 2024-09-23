from desdeo.problem import (
    Problem,
    TensorConstant,
    TensorVariable,
    Objective,
    ObjectiveTypeEnum,
    VariableTypeEnum,
    Constraint,
    ConstraintTypeEnum,
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
        1: [[250, 300, 420, 0], [330, 280, 410, 0], [280, 290, 390, 0]],
        2: [[312, 405, 478, 0], [287, 331, 462, 0], [344, 298, 411, 0]],
        3: [[276, 418, 395, 0], [392, 284, 487, 0], [315, 367, 442, 0]],
        4: [[329, 401, 356, 0], [245, 372, 498, 0], [408, 293, 475, 0]],
        5: [[351, 289, 437, 0], [304, 426, 389, 0], [267, 338, 452, 0]],
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
    # Helper constants to select rows and columns of matrices
    constants = []
    for j in range(number_of_years):
        constant_j = TensorConstant(
            name=f"Select row {j+1} of a square matrix",
            symbol=f"row_{j+1}",
            shape=[1, number_of_years],
            values=[1 if jj == j else 0 for jj in range(number_of_years)],
        )
        constants.append(constant_j)

    for k in range(number_of_regimes):
        constant_k = TensorConstant(
            name=f"Select column {k+1} of a square matrix",
            symbol=f"col_{k+1}",
            shape=[number_of_regimes],
            values=[[1] if kk == k else [0] for kk in range(number_of_regimes)],
        )
        constants.append(constant_k)

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

    lower_bounds = [[0 for _ in range(number_of_regimes)] for _ in range(number_of_years)]
    upper_bounds = [[0 for _ in range(number_of_regimes)] for _ in range(number_of_years)]

    for i in range(number_of_stands):
        var_i = TensorVariable(
            name=f"Stand {i+1}",
            symbol=f"X_{i+1}",
            variable_type=VariableTypeEnum.binary,
            shape=[number_of_years, number_of_regimes],
            lowerbounds=lower_bounds,
            upperbounds=upper_bounds,
            initial_values=None,
        )

        variables.append(var_i)

    # Constraints
    # For each stand, one regime, and only one, must be selected for each year.

    constraints = []

    for i in range(number_of_stands):
        for j in range(number_of_years):
            expr_ij = f"Sum(row_{j+1} @ X_{i+1}) - 1"  # minus 1 because must equal zero

            constraint_ij = Constraint(
                name=f"Row {j+1} of stand {i+1} sum to one.",
                symbol=f"row_constraint_{i}{j}",
                func=expr_ij,
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )

            constraints.append(constraint_ij)

        for k in range(number_of_regimes):
            expr_ik = f"Sum(X_{i+1} @ col_{k+1}) - 1"  # minus 1 because must equal zero

            constraint_ik = Constraint(
                name=f"Column {k+1} of stand {i+1} sum to one.",
                symbol=f"column_constraint_{i}{k}",
                func=expr_ik,
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )

            constraints.append(constraint_ik)

    # Objectives
    objectives = []

    ## NPV sum
    npv_sum_expr = " + ".join([f"X_{i+1} * NPV_{i+1}" for i in range(number_of_stands)])
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
    dwv_sum_expr = " + ".join([f"X_{i+1} * DWV_{i+1}" for i in range(number_of_stands)])
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


problem = simple_forest_problem()
print(problem)
