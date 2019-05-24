#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:13:46 2019

@author: yuezhou
An example: how to use DESDEO to solve multiobjective optimization problems
 with decision uncertainty.
Decision uncertainty: the realization of x is subject to perturbation in the
 neighborhood. Due to this, the objective function values can be very different
 from the computed ones.
Here, we use a robustness measure to quantify the change in the objective
 functions and use this information to help the decision maker in making
 informed decisions.
We use the river pollution toy problem(see desdeo documentation about this problem)
 to construct the example: we extend the river pollution problem with the
 robustness measure as an additonal objective- RiverPollutionRobust.
The solution method demonstrated here is published in
Zhou-Kangas, Y., Miettinen, K., & Sindhya, K. (2019).
Solving multiobjective optimization problems with decision uncertainty : an interactive approach.
Journal of Business Economics, 89 (1), 25-51. doi:10.1007/s11573-018-0900-1
"""

import numpy as np

from typing import List, Optional

from desdeo.method.NIMBUS import NIMBUS
from desdeo.optimization import SciPyDE
from desdeo.problem.toy import RiverPollutionRobust


def simple_asf(objectives : List[float],
               reference_p : List[float],
               nadir_p : List[float],
               utopian_p : List[float],
               rho : Optional[float] = 0.1,
               weights : Optional[float] = None):
    """Implements a simple achievement scalarizing function (ASF) as
    specified in _[MIETTINEN2010], equation (2).

    ...

    Parameters
    ----------
    objectives : List[float]
        Objective function values at observable point.
    reference_p : List[float]
        Reference point given mby a decision maker.
    nadir_p : List[float]
        The nadir point.
    utopian_p : List[float]
        The utopian point.
    rho : Optional[float]
        A small positive number. Default: 0.1
    weights : Optional[List[float]]
        Weighst used in the ASF.

    Returns
    -------
    float
        Result of the ASF.

    References
    ----------
    .. [MIETTINEN2010] Miettinen, K.; Eskelinen, P.; Ruiz, F. & Luque, M.
        NAUTILUS method: An interactive technique in multiobjective optimization
        based on the nadir point
        European Journal of Operational Research, 2010, 206, 426-434
    """
    # Check correct dimensions (remove if unnecessary)
    assert len(objectives) == len(reference_p)
    assert len(objectives) == len(nadir_p)
    assert len(objectives) == len(utopian_p)

    if not weights:
        weights = [1.0 for e in range(len(objectives))]
    else:
        assert len(objectives) == len(weights)

    first_term = max([weights[i] * (objectives[i] - reference_p[i])
                      for i in range(len(objectives))])

    second_term = sum([(objectives[i] - reference_p[i]) / (nadir_p[i] - utopian_p[i])
                       for i in range(len(objectives))]) * rho

    return first_term + second_term


if __name__ == "__main__":
    problem = RiverPollutionRobust()

    method = NIMBUS(problem, SciPyDE)
    # the method takes problem and optimization method as input
    results = method.init_iteration()
    obj_fn = results.objective_vars[0]
    obj_fn[0:3] = np.multiply(obj_fn[0:3], -1)
    print(obj_fn)

    # ASF usage example (with dummy variabels)
    reference_p = [1.0 for _ in range(len(obj_fn))]
    utopian_p = [0.1 - problem.nadir[i] for i in range(len(problem.nadir))]
    rho = 0.25
    weights = [i for i in range(len(obj_fn))]

    asf_result = simple_asf(obj_fn,
                            reference_p,
                            problem.nadir,
                            utopian_p,
                            rho,
                            weights)

    print(asf_result)
