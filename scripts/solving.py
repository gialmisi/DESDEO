from desdeo.tools import PyomoBonminSolver, add_objective_as_scalarization

from problem_model import simple_forest_problem

problem = simple_forest_problem()

# scalarized_problem = add_objective_as_scalarization(problem, symbol="f1", objective_symbol="NPV")

solver = PyomoBonminSolver(problem)
