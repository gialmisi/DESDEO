"""Assigning subjects to arms at the anchor.

At the anchor, the subject's stratification value (the anchor's distance to the compliant region of the policy) and
binding status are computed, for every subject in every arm. The subject is then assigned the arm furthest behind in
their stratum, counting only subjects who have completed the experiment, so that abandoned subjects do not count and
the balance corrects itself. Ties are broken at random with a logged seed.
"""

import bisect
import math
import random
import secrets
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlmodel import Session, select

from desdeo.api.models import (
    CommitStageData,
    ExperimentDefinition,
    ExperimentEnrolment,
    ExperimentStageProgress,
    Policy,
    ProblemDB,
    StateDB,
)
from desdeo.problem import Problem


def committed_solution(
    session: Session, enrolment: ExperimentEnrolment, data: CommitStageData
) -> tuple[ProblemDB, dict[str, float]]:
    """Return the problem and the objective values of a solution a subject commits to.

    Args:
        session (Session): the database session.
        enrolment (ExperimentEnrolment): the subject's enrolment.
        data (CommitStageData): the data of the commit stage.

    Raises:
        HTTPException: 422 if the state is not in the subject's interactive session, is not of a problem of the
            subject's run, or has no solution at the index.

    Returns:
        tuple[ProblemDB, dict[str, float]]: the problem and the objective values of the solution.
    """
    state_db = session.exec(select(StateDB).where(StateDB.state_id == data.state_id)).first()
    if state_db is None or state_db.session_id != enrolment.session_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"State {data.state_id} is not in the subject's interactive session.",
        )
    if state_db.problem_id not in enrolment.run.problems.values():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"State {data.state_id} is not of a problem of the experiment.",
        )

    solver_results = getattr(state_db.state, "solver_results", None) or []
    if data.solution_index >= len(solver_results):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"State {data.state_id} has no solution {data.solution_index}.",
        )

    return state_db.problem, solver_results[data.solution_index].optimal_objectives


def check_policy(policy: Policy, problem: Problem) -> None:
    """Check that a policy's thresholds bound objectives of the problem that have an ideal and a nadir.

    Raises:
        ValueError: a threshold is of an objective the problem does not have, or that has no ideal or nadir.
    """
    objectives = {objective.symbol: objective for objective in problem.objectives}
    for symbol in policy.thresholds:
        if symbol not in objectives:
            msg = f"The policy bounds '{symbol}', which is not an objective of problem '{problem.name}'."
            raise ValueError(msg)
        if objectives[symbol].ideal is None or objectives[symbol].nadir is None:
            msg = f"Objective '{symbol}' of problem '{problem.name}' needs an ideal and a nadir to be bounded."
            raise ValueError(msg)


def anchor_distance(policy: Policy, problem: Problem, objective_values: dict[str, float]) -> float:
    """Compute the distance of a solution to the compliant region of a policy.

    The distance is Euclidean, in objective space normalized by the problem's ideal and nadir, from the solution to the
    nearest compliant point. It is zero for a compliant solution.

    Args:
        policy (Policy): the policy.
        problem (Problem): the problem, with the ideal and nadir of the bounded objectives.
        objective_values (dict[str, float]): the objective values of the solution.

    Returns:
        float: the distance.
    """
    check_policy(policy, problem)
    shortfalls = []
    for symbol, threshold in policy.thresholds.items():
        objective = problem.get_objective(symbol)
        value = objective_values[symbol]
        shortfall = threshold - value if objective.maximize else value - threshold
        shortfalls.append(max(0.0, shortfall) / abs(objective.nadir - objective.ideal))

    return math.hypot(*shortfalls)


def stratum_of(value: float, cut_points: list[float]) -> int:
    """Return the stratum of a value, counted from zero; a value on a cut point belongs to the upper stratum."""
    return bisect.bisect_right(cut_points, value)


def is_complete(session: Session, enrolment: ExperimentEnrolment, definition: ExperimentDefinition) -> bool:
    """Return whether a subject is complete: every required stage of the experiment has stored data."""
    with_data = set(
        session.exec(
            select(ExperimentStageProgress.stage_ordinal).where(
                ExperimentStageProgress.enrolment_id == enrolment.id, ExperimentStageProgress.data.is_not(None)
            )
        ).all()
    )
    return all(ordinal in with_data for ordinal, stage in enumerate(definition.stages) if stage.required)


def completed_counts(
    session: Session, enrolment: ExperimentEnrolment, definition: ExperimentDefinition, stratum: int
) -> dict[str, int]:
    """Count the complete subjects of each arm in a stratum of the subject's run."""
    counts = dict.fromkeys(definition.arms, 0)
    others = session.exec(
        select(ExperimentEnrolment).where(
            ExperimentEnrolment.run_id == enrolment.run_id,
            ExperimentEnrolment.stratum == stratum,
            ExperimentEnrolment.arm.is_not(None),
            ExperimentEnrolment.id != enrolment.id,
        )
    ).all()
    for other in others:
        if other.arm in counts and is_complete(session, other, definition):
            counts[other.arm] += 1
    return counts


def assign_arm(
    session: Session,
    enrolment: ExperimentEnrolment,
    definition: ExperimentDefinition,
    problem: Problem,
    anchor_objectives: dict[str, float],
    *,
    seed: int | None = None,
) -> ExperimentEnrolment:
    """Stratify a subject at the anchor and assign them an arm.

    Args:
        session (Session): the database session.
        enrolment (ExperimentEnrolment): the subject's enrolment, not yet assigned.
        definition (ExperimentDefinition): the experiment definition, with a policy.
        problem (Problem): the problem the anchor is a solution of.
        anchor_objectives (dict[str, float]): the objective values of the anchor.
        seed (int | None, optional): the seed of the random tiebreak. Defaults to None, a new random seed.

    Returns:
        ExperimentEnrolment: the enrolment with the assignment stored.
    """
    distance = anchor_distance(definition.policy, problem, anchor_objectives)
    stratum = stratum_of(distance, definition.stratification.cut_points)
    counts = completed_counts(session, enrolment, definition, stratum)

    fewest = min(counts.values())
    candidates = [arm for arm in definition.arms if counts[arm] == fewest]
    seed = secrets.randbits(32) if seed is None else seed
    arm = random.Random(seed).choice(candidates)  # noqa: S311, not for security

    enrolment.stratification_value = distance
    enrolment.stratum = stratum
    enrolment.is_binding = distance > 0
    enrolment.arm = arm
    enrolment.assignment_details = {
        "anchor_objectives": anchor_objectives,
        "completed_counts": counts,
        "candidates": candidates,
        "seed": seed,
    }
    enrolment.assigned_at = datetime.now(UTC)
    session.add(enrolment)
    session.commit()
    session.refresh(enrolment)

    return enrolment
