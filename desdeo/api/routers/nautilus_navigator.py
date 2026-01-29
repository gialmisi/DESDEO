"""Endpoints for NAUTILUS Navigator."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select

from desdeo.api.models import (
    NautilusNavigatorInitializeRequest,
    NautilusNavigatorInitializeResponse,
    NautilusNavigatorNavigateRequest,
    NautilusNavigatorNavigateResponse,
    NautilusNavigatorState,
    State,
    StateDB,
)
from desdeo.api.models.generic_states import StateKind
from desdeo.api.routers.utils import SessionContext, get_session_context
from desdeo.mcdm.nautilus_navigator import (
    NAUTILUS_Response,
    get_current_path,
    navigator_all_steps,
    navigator_init,
    step_back_index,
)
from desdeo.problem import Problem

router = APIRouter(prefix="/method/nautilus_navigator")


@router.post("/initialize")
def initialize_navigator(
    request: NautilusNavigatorInitializeRequest,
    context: Annotated[SessionContext, Depends(get_session_context)],
) -> NautilusNavigatorInitializeResponse:
    """Initialize the NAUTILUS Navigator method."""
    db_session = context.db_session
    problem_db = context.problem_db

    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Problem not found.")

    problem = Problem.from_problemdb(problem_db)

    response = navigator_init(problem)

    nautilus_state = NautilusNavigatorState(
        total_steps=request.total_steps,
        nautilus_response=response,
    )

    state_db = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=context.interactive_session.id if context.interactive_session is not None else None,
        parent_id=context.parent_state.id if context.parent_state is not None else None,
        state=nautilus_state,
    )

    db_session.add(state_db)
    db_session.commit()
    db_session.refresh(state_db)

    steps_remaining = max(request.total_steps - response.step_number, 0)

    return NautilusNavigatorInitializeResponse(
        state_id=state_db.id,
        total_steps=request.total_steps,
        steps_remaining=steps_remaining,
        step_number=response.step_number,
        distance_to_front=response.distance_to_front,
        navigation_point=response.navigation_point,
        reachable_solution=response.reachable_solution,
        reachable_bounds=response.reachable_bounds,
        reference_point=response.reference_point,
        bounds=response.bounds,
    )


@router.post("/navigate")
def navigate_navigator(
    request: NautilusNavigatorNavigateRequest,
    context: Annotated[SessionContext, Depends(get_session_context)],
) -> NautilusNavigatorNavigateResponse:
    """Navigate the NAUTILUS Navigator method."""
    db_session = context.db_session
    problem_db = context.problem_db

    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Problem not found.")

    problem = Problem.from_problemdb(problem_db)

    session_id = (
        context.interactive_session.id
        if context.interactive_session is not None
        else context.user.active_session_id
    )

    statement = (
        select(StateDB)
        .join(State, StateDB.state_id == State.id)
        .where(
            State.kind == StateKind.NAUTILUS_NAVIGATOR_STEP,
            StateDB.problem_id == problem_db.id,
            StateDB.session_id == session_id,
        )
        .order_by(StateDB.id)
    )

    state_rows = db_session.exec(statement).all()

    if not state_rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="NAUTILUS Navigator not initialized.")

    responses: list[NAUTILUS_Response] = []
    state_ids: list[int] = []
    for row in state_rows:
        if not isinstance(row.state, NautilusNavigatorState):
            continue
        responses.append(row.state.nautilus_response)
        state_ids.append(row.id)

    if not responses:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="NAUTILUS Navigator states missing.")

    back_index = step_back_index(responses, request.go_back_step)
    responses.append(responses[back_index])
    state_ids.append(state_ids[back_index])

    try:
        new_responses = navigator_all_steps(
            problem=problem,
            steps_remaining=request.steps_remaining,
            reference_point=request.reference_point,
            previous_responses=responses,
            bounds=request.bounds,
        )
    except IndexError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Possible reason for error: bounds are too restrictive.",
        ) from exc

    parent_state_id = state_ids[-1]
    for response in new_responses:
        nautilus_state = NautilusNavigatorState(
            total_steps=request.total_steps,
            nautilus_response=response,
        )
        state_db = StateDB.create(
            database_session=db_session,
            problem_id=problem_db.id,
            session_id=session_id,
            parent_id=parent_state_id,
            state=nautilus_state,
        )
        db_session.add(state_db)
        db_session.flush()
        parent_state_id = state_db.id
        state_ids.append(state_db.id)

    db_session.commit()

    responses = [*responses, *new_responses]
    current_path = get_current_path(responses)
    active_responses = [responses[i] for i in current_path]
    active_state_ids = [state_ids[i] for i in current_path]

    lower_bounds: dict[str, list[float]] = {}
    upper_bounds: dict[str, list[float]] = {}
    preferences: dict[str, list[float]] = {}
    bounds: dict[str, list[float]] = {}
    navigation_points: dict[str, list[float]] = {}

    for obj in problem.objectives:
        symbol = obj.symbol
        lower_bounds[symbol] = [
            response.reachable_bounds["lower_bounds"][symbol] for response in active_responses
        ]
        upper_bounds[symbol] = [
            response.reachable_bounds["upper_bounds"][symbol] for response in active_responses
        ]
        navigation_points[symbol] = [response.navigation_point[symbol] for response in active_responses]
        preferences[symbol] = [
            response.reference_point[symbol]
            for response in active_responses[1:]
            if response.reference_point is not None
        ]
        bounds[symbol] = [
            response.bounds[symbol]
            for response in active_responses[1:]
            if response.bounds is not None
        ]

    return NautilusNavigatorNavigateResponse(
        objective_symbols=[obj.symbol for obj in problem.objectives],
        objective_long_names=[obj.name for obj in problem.objectives],
        units=[obj.unit or "" for obj in problem.objectives],
        is_maximized=[obj.maximize for obj in problem.objectives],
        ideal=[obj.ideal for obj in problem.objectives],
        nadir=[obj.nadir for obj in problem.objectives],
        total_steps=request.total_steps,
        current_step=active_responses[-1].step_number,
        step_numbers=[response.step_number for response in active_responses],
        state_ids=active_state_ids,
        distance_to_front=[response.distance_to_front for response in active_responses],
        navigation_points=navigation_points,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
        preferences=preferences,
        reachable_solution=active_responses[-1].reachable_solution,
    )
