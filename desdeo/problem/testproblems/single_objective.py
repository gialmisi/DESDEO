"""Here a variety of single-objective optimization problems are defined."""

import math

from desdeo.problem import Constant, Constraint, ConstraintTypeEnum, Objective, Problem, Variable, VariableTypeEnum


def mystery_function() -> Problem:
    """_summary_

    Returns:
        Problem: _description_
    """
    pi = Constant(name="Pi", symbol="PI", value=math.pi)
    x_1 = Variable(
        name="x_1", symbol="x_1", variable_type=VariableTypeEnum.real, lowerbound=0.0, upperbound=5.0, initial_value=0.1
    )
    x_2 = Variable(
        name="x_2", symbol="x_2", variable_type=VariableTypeEnum.real, lowerbound=0.0, upperbound=5.0, initial_value=0.1
    )

    f_1_def = "2 + 0.01*(x_2 - x_1**2)**2 + (1 - x_1)**2 + 2*(2 - x_2)**2 + 7*Sin(0.5*x_1)*Sin(0.7*x_1*x_2)"
    f_1 = Objective(
        name="f_1",
        symbol="f_1",
        func=f_1_def,
        maximize=False,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    )

    c_1_def = "-Sin(x_1 - x_2 - PI/8.0)"
    c_1 = Constraint(
        name="c_1",
        symbol="c_1",
        cons_type=ConstraintTypeEnum.LTE,
        func=c_1_def,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    )

    return Problem(
        name="Mystery function",
        description="The single-objective mystery function.",
        constants=[pi],
        variables=[x_1, x_2],
        objectives=[f_1],
        constraints=[c_1],
    )


if __name__ == "__main__":
    problem = mystery_function()

    print(problem)
