from pathlib import Path
from desdeo.tools import PyomoBonminSolver, BonminOptions, PyomoCBCSolver, add_objective_as_scalarization
from desdeo.problem import Problem


# load problem
with Path("./data/Vaaler_NPV_and_DWV.json").open("r") as f:
    json_data = f.read()

problem = Problem.model_validate_json(json_data)

scalarized_problem, target = add_objective_as_scalarization(problem, symbol="f1", objective_symbol="DWV")

sol_options = BonminOptions(tol=1e-6)
# solver = PyomoBonminSolver(scalarized_problem, sol_options)
solver = PyomoCBCSolver(scalarized_problem)

result = solver.solve(target)

print()
