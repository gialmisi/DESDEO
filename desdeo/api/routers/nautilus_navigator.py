"""Defines end-points to access functionalities related to the NAUTILUS Navigator method."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from desdeo.api.models import (
    InteractiveSessionDB,
    NautilusNavigatorInitRequest,
    NautilusNavigatorInitState,
    NautilusNavigatorRecomputeRequest,
    NautilusNavigatorRecomputeState,
    NautilusNavigatorSegmentResponse,
    StateDB,
)
from desdeo.api.routers.utils import SessionContext, get_session_context
from desdeo.mcdm.nautilus_navigator import (
    NAUTILUS_Response,
    navigator_all_steps,
    navigator_init,
    step_back_index,
)
from desdeo.problem import Problem

router = APIRouter(prefix="/method/nautilus_navigator")


def _ensure_interactive_session(context: SessionContext) -> InteractiveSessionDB:
    db_session = context.db_session
    user = context.user
    interactive_session = context.interactive_session

    if interactive_session is not None:
        return interactive_session

    interactive_session = InteractiveSessionDB(user_id=user.id, info="NAUTILUS Navigator session")
    db_session.add(interactive_session)
    db_session.commit()
    db_session.refresh(interactive_session)

    user.active_session_id = interactive_session.id
    db_session.add(user)
    db_session.commit()

    return interactive_session


def _normalize_bounds(problem: Problem, bounds: dict[str, float | None] | None) -> dict[str, float] | None:
    if bounds is None:
        return None

    normalized: dict[str, float] = {}
    for obj in problem.objectives:
        value = bounds.get(obj.symbol) if bounds is not None else None
        normalized[obj.symbol] = value if value is not None else obj.nadir

    return normalized


def _serialize_responses(responses: list[NAUTILUS_Response]) -> list[dict]:
    return [response.model_dump(mode="json") for response in responses]


def _segment_arrays(
    objective_symbols: list[str], responses: list[NAUTILUS_Response]
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], list[float]]:
    lower_bounds: dict[str, list[float]] = {}
    upper_bounds: dict[str, list[float]] = {}
    navigation_points: dict[str, list[float]] = {}

    for symbol in objective_symbols:
        lower_bounds[symbol] = [response.reachable_bounds["lower_bounds"][symbol] for response in responses]
        upper_bounds[symbol] = [response.reachable_bounds["upper_bounds"][symbol] for response in responses]
        navigation_points[symbol] = [response.navigation_point[symbol] for response in responses]

    distance = [response.distance_to_front for response in responses]

    return lower_bounds, upper_bounds, navigation_points, distance


@router.post("/initialize")
def initialize(
    request: NautilusNavigatorInitRequest,
    context: Annotated[SessionContext, Depends(get_session_context)],
) -> NautilusNavigatorSegmentResponse:
    """Initialize NAUTILUS Navigator."""
    db_session = context.db_session
    problem_db = context.problem_db

    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Problem not found.")

    problem = Problem.from_problemdb(problem_db)

    interactive_session = _ensure_interactive_session(context)
    parent_state = context.parent_state

    initial_response = navigator_init(problem)
    responses = [initial_response]

    objective_symbols = [objective.symbol for objective in problem.objectives]
    lower_bounds, upper_bounds, navigation_points, distance = _segment_arrays(objective_symbols, responses)

    navigator_state = NautilusNavigatorInitState(
        total_steps=request.total_steps,
        segment_start_step=0,
        segment_steps=0,
        responses=_serialize_responses(responses),
        segment_responses=_serialize_responses(responses),
    )

    state_db = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=interactive_session.id,
        parent_id=parent_state.id if parent_state is not None else None,
        state=navigator_state,
    )
    db_session.add(state_db)
    db_session.commit()
    db_session.refresh(state_db)

    return NautilusNavigatorSegmentResponse(
        state_id=state_db.id,
        session_id=interactive_session.id,
        total_steps=request.total_steps,
        segment_start_step=0,
        segment_steps=0,
        objective_symbols=objective_symbols,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        navigation_points=navigation_points,
        distance=distance,
    )


@router.post("/recompute")
def recompute(
    request: NautilusNavigatorRecomputeRequest,
    context: Annotated[SessionContext, Depends(get_session_context)],
) -> NautilusNavigatorSegmentResponse:
    """Recompute a NAUTILUS Navigator segment after a DM action."""
    db_session = context.db_session
    problem_db = context.problem_db

    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Problem not found.")

    problem = Problem.from_problemdb(problem_db)
    interactive_session = _ensure_interactive_session(context)
    parent_state = context.parent_state

    if parent_state is None or not isinstance(
        parent_state.state, (NautilusNavigatorInitState, NautilusNavigatorRecomputeState)
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="NAUTILUS Navigator history not found for recompute.",
        )

    parent_nav_state = parent_state.state
    previous_responses = [
        NAUTILUS_Response.model_validate(response) for response in parent_nav_state.responses
    ]

    try:
        go_back_index = step_back_index(previous_responses, request.go_back_step)
    except IndexError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid go_back_step.") from exc

    truncated_history = previous_responses[: go_back_index + 1]
    normalized_bounds = _normalize_bounds(problem, request.bounds)

    try:
        new_responses = navigator_all_steps(
            problem,
            steps_remaining=request.steps,
            reference_point=request.reference_point,
            previous_responses=truncated_history,
            bounds=normalized_bounds,
        )
    except IndexError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Possible reason for error: bounds are too restrictive.",
        ) from exc

    segment_responses = [truncated_history[-1], *new_responses]
    full_responses = [*truncated_history, *new_responses]

    objective_symbols = [objective.symbol for objective in problem.objectives]
    lower_bounds, upper_bounds, navigation_points, distance = _segment_arrays(objective_symbols, segment_responses)

    segment_start_step = truncated_history[-1].step_number
    segment_steps = len(segment_responses) - 1

    navigator_state = NautilusNavigatorRecomputeState(
        total_steps=parent_nav_state.total_steps,
        segment_start_step=segment_start_step,
        segment_steps=segment_steps,
        go_back_step=request.go_back_step,
        reference_point=request.reference_point,
        bounds=request.bounds,
        responses=_serialize_responses(full_responses),
        segment_responses=_serialize_responses(segment_responses),
    )

    state_db = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=interactive_session.id,
        parent_id=parent_state.id,
        state=navigator_state,
    )
    db_session.add(state_db)
    db_session.commit()
    db_session.refresh(state_db)

    return NautilusNavigatorSegmentResponse(
        state_id=state_db.id,
        session_id=interactive_session.id,
        total_steps=parent_nav_state.total_steps,
        segment_start_step=segment_start_step,
        segment_steps=segment_steps,
        objective_symbols=objective_symbols,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        navigation_points=navigation_points,
        distance=distance,
    )
