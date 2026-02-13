"""Endpoints for the Scenario RPM method.

Allows multiple stakeholders (owners + supervisor) to solve a shared
forest management problem from their own perspectives using the
reference point method, then compare solutions at the cluster level.
"""

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from desdeo.api.db import get_session
from desdeo.api.models import (
    ProblemDB,
    StateDB,
)
from desdeo.api.models.scenario_rpm import (
    ClusterComparison,
    ScenarioComparisonRequest,
    ScenarioComparisonResponse,
    ScenarioRPMSolveRequest,
)
from desdeo.api.models.state import ScenarioRPMState
from desdeo.api.routers.problem import check_solver
from desdeo.mcdm import rpm_solve_solutions
from desdeo.problem import Problem
from desdeo.problem.evaluator import PolarsEvaluator
from desdeo.tools import SolverResults

from .utils import ContextField, SessionContext, SessionContextGuard

router = APIRouter(prefix="/method/scenario-rpm")


@router.post("/solve")
def solve(
    request: ScenarioRPMSolveRequest,
    context: Annotated[
        SessionContext,
        Depends(SessionContextGuard(require=[ContextField.PROBLEM])),
    ],
) -> ScenarioRPMState:
    """Solve one scenario of the problem using the reference point method."""
    db_session = context.db_session
    problem_db = context.problem_db
    interactive_session = context.interactive_session
    parent_state = context.parent_state

    if problem_db is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Problem context missing.",
        )

    solver = check_solver(problem_db=problem_db)
    problem = Problem.from_problemdb(problem_db)

    # Extract the sub-problem for the requested scenario
    try:
        scenario_problem = problem.get_scenario_problem(request.scenario_key)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scenario_key '{request.scenario_key}': {exc}",
        ) from exc

    solver_results: list[SolverResults] = rpm_solve_solutions(
        scenario_problem,
        request.preference.aspiration_levels,
        request.scalarization_options,
        solver,
        request.solver_options,
    )

    scenario_state = ScenarioRPMState(
        scenario_key=request.scenario_key,
        preferences=request.preference,
        scalarization_options=request.scalarization_options,
        solver=request.solver,
        solver_options=request.solver_options,
        solver_results=solver_results,
    )

    state_db = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=interactive_session.id if interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=scenario_state,
    )

    db_session.add(state_db)
    db_session.commit()
    db_session.refresh(state_db)
    db_session.refresh(scenario_state)

    return scenario_state


@router.post("/compare")
def compare(
    request: ScenarioComparisonRequest,
    db_session: Annotated[Session, Depends(get_session)],
) -> ScenarioComparisonResponse:
    """Compare supervisor's solution against all owners at the cluster level.

    For each owner, evaluates the supervisor's decision variables on the
    owner's scenario sub-problem to get the supervisor's cluster-level
    objective values, then compares with the owner's solved objectives.
    """
    # Load supervisor state
    sup_state_db = db_session.exec(select(StateDB).where(StateDB.id == request.supervisor_state_id)).first()
    if sup_state_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Supervisor state not found.")

    sup_state = sup_state_db.state
    if not isinstance(sup_state, ScenarioRPMState):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Supervisor state is not a ScenarioRPMState."
        )

    # Load problem
    problem_db = db_session.exec(select(ProblemDB).where(ProblemDB.id == request.problem_id)).first()
    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Problem not found.")

    problem = Problem.from_problemdb(problem_db)
    sup_vars = sup_state.solver_results[0].optimal_variables

    clusters: list[ClusterComparison] = []

    for owner_key, owner_state_id in request.owner_states.items():
        # Load owner state
        own_state_db = db_session.exec(select(StateDB).where(StateDB.id == owner_state_id)).first()
        if own_state_db is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Owner state {owner_state_id} not found."
            )

        own_state = own_state_db.state
        if not isinstance(own_state, ScenarioRPMState):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"State {owner_state_id} is not a ScenarioRPMState."
            )

        # Get the owner's scenario sub-problem
        scenario_problem = problem.get_scenario_problem(owner_key)

        # Evaluate the supervisor's variables on the owner's scenario problem
        evaluator = PolarsEvaluator(scenario_problem)
        # Filter to only variables defined in the scenario sub-problem
        var_symbols = {v.symbol for v in scenario_problem.variables}
        eval_input = {k: [v] for k, v in sup_vars.items() if k in var_symbols}
        result_df = evaluator.evaluate(eval_input)

        # Extract supervisor's objective values for this cluster
        sup_obj: dict[str, float] = {}
        for obj in scenario_problem.objectives:
            if obj.scenario_keys and owner_key in obj.scenario_keys:
                sup_obj[obj.symbol] = float(result_df[obj.symbol][0])

        # Owner's objective values come directly from their solver results
        own_obj: dict[str, float] = {}
        for obj in scenario_problem.objectives:
            if obj.scenario_keys and owner_key in obj.scenario_keys:
                own_obj[obj.symbol] = float(own_state.solver_results[0].optimal_objectives.get(obj.symbol, 0.0))

        # Normalized Euclidean distance using ideal/nadir ranges
        dist_sq = 0.0
        for obj in scenario_problem.objectives:
            if obj.scenario_keys and owner_key in obj.scenario_keys:
                ideal = obj.ideal if obj.ideal is not None else 0.0
                nadir = obj.nadir if obj.nadir is not None else 0.0
                rng = abs(ideal - nadir)
                if rng > 0:
                    diff = (sup_obj[obj.symbol] - own_obj[obj.symbol]) / rng
                    dist_sq += diff**2

        distance = math.sqrt(dist_sq)

        clusters.append(
            ClusterComparison(
                cluster_key=owner_key,
                supervisor_objectives=sup_obj,
                owner_objectives=own_obj,
                distance=distance,
            )
        )

    max_distance = max((c.distance for c in clusters), default=0.0)

    return ScenarioComparisonResponse(clusters=clusters, max_distance=max_distance)


@router.get("/get-state/{state_id}")
def get_state(
    state_id: int,
    db_session: Annotated[Session, Depends(get_session)],
) -> ScenarioRPMState:
    """Retrieve a stored ScenarioRPMState."""
    state_db = db_session.exec(select(StateDB).where(StateDB.id == state_id)).first()

    if state_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"State {state_id} not found.")

    state = state_db.state
    if not isinstance(state, ScenarioRPMState):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The requested state is not a ScenarioRPMState.",
        )

    return state
