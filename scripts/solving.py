from desdeo.tools import PyomoBonminSolver, BonminOptions, PyomoCBCSolver, add_objective_as_scalarization

from problem_model import simple_forest_problem

problem = simple_forest_problem()

scalarized_problem, target = add_objective_as_scalarization(problem, symbol="f1", objective_symbol="DWV")

sol_options = BonminOptions(tol=1e-6)
# solver = PyomoBonminSolver(scalarized_problem, sol_options)
solver = PyomoCBCSolver(scalarized_problem)

result = solver.solve(target)

print()
