from problem_model import forest_problem_vaaler, simple_forest_problem
from desdeo.problem import simple_knapsack
from desdeo.emo.hooks.archivers import FeasibleArchive

from methods import binary_nsga3, binary_rvea

problem = simple_forest_problem()

solver, publisher = binary_rvea(problem=problem)

archive = FeasibleArchive(problem=problem, publisher=publisher)
publisher.auto_subscribe(archive)

results = solver()

print()
