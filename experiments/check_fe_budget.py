"""Sanity-check: are function-evaluation (FE) budgets respected by DESDEO SMS-EMOA and pymoo SMSEMOA?

We count actual objective evaluations at the objective-function level in both frameworks (not the
internal counters), so the measurement is implementation-agnostic.

- pymoo: wrap the pymoo problem's ``_evaluate`` and count rows.
- DESDEO: wrap ``PymooProvider.evaluate`` and count points; DESDEO is driven by the SAME pymoo
  problem (via ``create_pymoo_problem``) so both sides evaluate the identical objective.

Run:  python experiments/check_fe_budget.py
"""

import numpy as np

from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.problems import get_problem as pymoo_get_problem


def run_pymoo(name: str, budget: int, pop_size: int, seed: int = 1):
    problem = pymoo_get_problem(name)

    count = {"n": 0}
    orig_evaluate = problem._evaluate

    def counting_evaluate(x, out, *args, **kwargs):
        count["n"] += int(np.atleast_2d(x).shape[0])
        return orig_evaluate(x, out, *args, **kwargs)

    problem._evaluate = counting_evaluate  # type: ignore[method-assign]

    res = minimize(
        problem,
        SMSEMOA(pop_size=pop_size),
        termination=("n_eval", budget),
        seed=seed,
        verbose=False,
    )
    reported = int(res.algorithm.evaluator.n_eval)
    return {"actual": count["n"], "reported": reported, "final_pop": len(res.X)}


from desdeo.emo.options.algorithms import smsemoa_options
from desdeo.emo.options.templates import emo_constructor
from desdeo.problem.external import PymooProblemParams, create_pymoo_problem
from desdeo.problem.external import pymoo_provider


def run_desdeo(name: str, budget: int, pop_size: int, n_offspring: int, greedy: bool, seed: int = 1):
    # Count at the lowest level: wrap the pymoo problem's ``_evaluate`` (ground truth, catches both the
    # initial population and every offspring batch, regardless of provider/resolver binding).
    count = {"n": 0}
    orig_get = pymoo_provider.pymoo_get_problem

    def counting_get(*a, **k):
        prob = orig_get(*a, **k)
        o_eval = prob._evaluate

        def ce(x, out, *aa, **kk):
            count["n"] += int(np.atleast_2d(x).shape[0])
            return o_eval(x, out, *aa, **kk)

        prob._evaluate = ce  # type: ignore[method-assign]
        return prob

    pymoo_provider._get_cached_pymoo_problem.cache_clear()
    pymoo_provider.pymoo_get_problem = counting_get  # type: ignore[assignment]
    try:
        params = PymooProblemParams(name=name)
        problem = create_pymoo_problem(params)
        opts = smsemoa_options(
            population_size=pop_size,
            n_offspring=n_offspring,
            max_evaluations=budget,
            greedy_reduction=greedy,
            seed=seed,
        )
        solver, _ = emo_constructor(opts, problem)
        res = solver()
        final_pop = len(res.optimal_outputs)
    finally:
        pymoo_provider.pymoo_get_problem = orig_get  # type: ignore[assignment]
        pymoo_provider._get_cached_pymoo_problem.cache_clear()

    return {"actual": count["n"], "final_pop": final_pop}


if __name__ == "__main__":
    BUDGET = 5000
    MU = 100
    PROBLEM = "zdt1"

    print(f"=== FE budget sanity check (problem={PROBLEM}, requested budget={BUDGET}, mu={MU}) ===\n")

    print("-- pymoo SMSEMOA (canonical, steady-state mu+1) --")
    r = run_pymoo(PROBLEM, BUDGET, MU)
    print(f"  actual FE = {r['actual']}, pymoo-reported n_eval = {r['reported']}, "
          f"overshoot = {r['actual'] - BUDGET}, final pop = {r['final_pop']}\n")

    # Note: n_offspring must be > 1 (TournamentSelectionOptions.winner_size > 1), so true steady-state
    # (lambda=1) is NOT expressible through smsemoa_options; the smallest batch is lambda=2.
    print("-- DESDEO SMS-EMOA, near-faithful (n_offspring=2, greedy=True) --")
    r = run_desdeo(PROBLEM, BUDGET, MU, n_offspring=2, greedy=True)
    print(f"  actual FE = {r['actual']}, overshoot = {r['actual'] - BUDGET}, final pop = {r['final_pop']}\n")

    print("-- DESDEO SMS-EMOA, BATCHED (n_offspring=mu, greedy=False) --")
    r = run_desdeo(PROBLEM, BUDGET, MU, n_offspring=MU, greedy=False)
    print(f"  actual FE = {r['actual']}, overshoot = {r['actual'] - BUDGET}, final pop = {r['final_pop']}\n")

    print("-- DESDEO SMS-EMOA, batched with n_offspring=20 --")
    r = run_desdeo(PROBLEM, BUDGET, MU, n_offspring=20, greedy=False)
    print(f"  actual FE = {r['actual']}, overshoot = {r['actual'] - BUDGET}, final pop = {r['final_pop']}\n")
