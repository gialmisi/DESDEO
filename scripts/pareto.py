from problem_model import forest_problem_vaaler, simple_forest_problem
from desdeo.problem import simple_knapsack, Problem
from desdeo.emo.hooks.archivers import FeasibleArchive
from pathlib import Path

from methods import binary_nsga3, binary_rvea

problem = forest_problem_vaaler()
problem.save_to_json(Path("./data/vaaler.json"))
