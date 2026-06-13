"""Run one (algorithm, problem, seed) SMS-EMOA job.

Both frameworks evaluate the same pymoo problem object (DESDEO wraps it via ``create_pymoo_problem``),
so the objective evaluations are identical. We save the final population's objective vectors (columns
``f_1..f_m``) and a small metadata JSON (runtime, reported evaluations, final size).

Run by Snakemake (uses the global ``snakemake`` object).
"""

import json
import time

import numpy as np
import polars as pl

algo_cfg = snakemake.params.algo_cfg  # noqa: F821
problem_cfg = snakemake.params.problem_cfg or {}  # noqa: F821
budget = int(snakemake.params.budget)  # noqa: F821
mu = int(snakemake.params.population_size)  # noqa: F821
seed = int(snakemake.wildcards.seed)  # noqa: F821
problem_name = snakemake.wildcards.problem  # noqa: F821
algo_name = snakemake.wildcards.algo  # noqa: F821
framework = algo_cfg["framework"]

# A problem entry may rename the underlying pymoo problem (``pymoo_name``) and/or be inverted (objectives
# maximized). Inverted runs are stored in minimization space as y = -f, matching the reference front.
pymoo_name = problem_cfg.get("pymoo_name", problem_name)
invert = bool(problem_cfg.get("invert", False))
kwargs = {k: problem_cfg[k] for k in ("n_var", "n_obj") if problem_cfg.get(k) is not None}

t0 = time.perf_counter()
if framework == "pymoo":
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.optimize import minimize
    from pymoo.problems import get_problem

    sms_kwargs = {"pop_size": mu}
    if str(algo_cfg.get("sampling", "")).lower() == "lhs":
        from pymoo.operators.sampling.lhs import LHS

        sms_kwargs["sampling"] = LHS()
    problem = get_problem(pymoo_name, **kwargs)
    if invert:
        # Maximize f by minimizing -f: negate the objective output (the stored front is then y = -f).
        _orig_evaluate = problem._evaluate

        def _negated_evaluate(x, out, *a, **k):
            _orig_evaluate(x, out, *a, **k)
            out["F"] = -np.asarray(out["F"])

        problem._evaluate = _negated_evaluate
    res = minimize(problem, SMSEMOA(**sms_kwargs), termination=("n_eval", budget), seed=seed, verbose=False)
    objectives = np.atleast_2d(res.pop.get("F"))  # already y = -f when inverted
    n_eval_reported = int(res.algorithm.evaluator.n_eval)

elif framework == "desdeo":
    from desdeo.emo.options.algorithms import nsga3_options, rvea_options, smsemoa_options
    from desdeo.emo.options.templates import emo_constructor
    from desdeo.emo.options.termination import MaxEvaluationsTerminatorOptions
    from desdeo.problem.external import PymooProblemParams, create_pymoo_problem

    params = PymooProblemParams(name=pymoo_name, n_var=kwargs.get("n_var"), n_obj=kwargs.get("n_obj"))
    problem = create_pymoo_problem(params)
    if invert:
        # Flip the (frozen) problem's objectives to maximization (model_copy bypasses the freeze).
        max_objs = [obj.model_copy(update={"maximize": True}) for obj in problem.objectives]
        problem = problem.model_copy(update={"objectives": max_objs})

    method = str(algo_cfg.get("method", "smsemoa")).lower()
    if method == "smsemoa":
        n_offspring = int(algo_cfg.get("n_offspring") or mu)
        opts = smsemoa_options(
            population_size=mu,
            n_offspring=n_offspring,
            max_evaluations=budget,
            use_dominating_points=bool(algo_cfg.get("use_dominating_points", True)),
            greedy_reduction=bool(algo_cfg.get("greedy_reduction", False)),
            seed=seed,
        )
        # Optional override of the HV-reduction reference-point offset (not exposed by smsemoa_options).
        ref_offset = algo_cfg.get("reference_point_offset")
        if ref_offset is not None:
            opts.template.selection.reference_point_offset = float(ref_offset)
    elif method in ("nsga3", "rvea"):
        # Reference-vector methods (Template1). They default to a generation-based terminator, so switch to the
        # shared evaluation budget and match the population/seed. NOTE: at >2 objectives the simplex lattice
        # fixes the population to the nearest lattice size (e.g. 84 for 4 objectives), not exactly ``mu``.
        opts = (nsga3_options if method == "nsga3" else rvea_options)()
        opts.template.termination = MaxEvaluationsTerminatorOptions(
            name="MaxEvaluationsTerminator", max_evaluations=budget
        )
        opts.template.seed = seed
        opts.template.generator.n_points = mu
        opts.template.selection.reference_vector_options.number_of_vectors = mu
        if method == "rvea":
            # RVEA's angle-penalized distance adapts with progress; under an evaluation budget it must use the
            # function-evaluation-based schedule (the default is generation-based and errors without max_generations).
            from desdeo.emo.options.selection import ParameterAdaptationStrategy

            opts.template.selection.parameter_adaptation_strategy = ParameterAdaptationStrategy.FUNCTION_EVALUATION_BASED
    else:
        raise ValueError(f"Unknown desdeo method: {method!r}")

    solver, extras = emo_constructor(opts, problem)
    res = solver()
    obj_symbols = [obj.symbol for obj in problem.objectives]
    objectives = res.optimal_outputs.select(obj_symbols).to_numpy()
    if invert:
        objectives = -objectives  # optimal_outputs reports f; store minimization-space y = -f

    # Recover the function-evaluation count from the terminator (it holds ``current_evaluations``).
    # The terminator is registered with the publisher; scan its subscribers for it.
    subscribers = list(getattr(extras.publisher, "global_subscribers", []))
    for subs in getattr(extras.publisher, "subscribers", {}).values():
        subscribers.extend(subs)
    n_eval_reported = None
    for sub in {id(s): s for s in subscribers}.values():
        if hasattr(sub, "current_evaluations"):
            value = int(sub.current_evaluations)
            n_eval_reported = value if n_eval_reported is None else max(n_eval_reported, value)

else:
    raise ValueError(f"Unknown framework: {framework!r}")
runtime_s = time.perf_counter() - t0

objectives = np.atleast_2d(objectives)
n_obj = objectives.shape[1]
pl.DataFrame({f"f_{i + 1}": objectives[:, i] for i in range(n_obj)}).write_parquet(snakemake.output.front)  # noqa: F821

with open(snakemake.output.meta, "w") as fh:  # noqa: F821, PTH123
    json.dump(
        {
            "algorithm": algo_name,
            "problem": problem_name,
            "seed": seed,
            "framework": framework,
            "runtime_s": runtime_s,
            "n_eval_reported": n_eval_reported,
            "n_final": int(objectives.shape[0]),
        },
        fh,
        indent=2,
    )
