from problem_model import forest_problem_vaaler, simple_forest_problem

from methods import binary_nsga3

problem = simple_forest_problem()

solver, publisher = binary_nsga3(problem=problem)

result = solver()
