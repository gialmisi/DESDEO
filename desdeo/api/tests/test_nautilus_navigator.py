"""Tests for NAUTILUS Navigator models and endpoints."""

from fastapi import status

from desdeo.api.models import (
    NautilusNavigatorInitRequest,
    NautilusNavigatorRecomputeRequest,
    NautilusNavigatorSegmentResponse,
    ProblemDB,
    StateDB,
)
from desdeo.problem import Problem

from .conftest import login, post_json


def test_nautilus_navigator_models():
    """Model sanity checks for NAUTILUS Navigator."""
    init_request = NautilusNavigatorInitRequest(problem_id=1, total_steps=10)
    assert init_request.problem_id == 1
    assert init_request.total_steps == 10

    recompute_request = NautilusNavigatorRecomputeRequest(
        problem_id=1,
        go_back_step=0,
        steps=5,
        reference_point={"f_1": 1.0},
    )
    assert recompute_request.reference_point["f_1"] == 1.0


def test_nautilus_navigator_initialize_and_recompute(client, session_and_user):
    """Initialize and recompute a NAUTILUS Navigator segment."""
    access_token = login(client)
    session = session_and_user["session"]

    init_request = NautilusNavigatorInitRequest(problem_id=1, total_steps=5)
    init_response = post_json(
        client,
        "/method/nautilus_navigator/initialize",
        init_request.model_dump(),
        access_token,
    )

    assert init_response.status_code == status.HTTP_200_OK

    init_payload = NautilusNavigatorSegmentResponse.model_validate(init_response.json())
    assert init_payload.segment_steps == 0
    assert init_payload.segment_start_step == 0
    assert init_payload.state_id is not None

    for symbol in init_payload.objective_symbols:
        assert len(init_payload.lower_bounds[symbol]) == 1
        assert len(init_payload.upper_bounds[symbol]) == 1

    problem_db = session.get(ProblemDB, 1)
    problem = Problem.from_problemdb(problem_db)
    reference_point = {
        obj.symbol: (obj.nadir if obj.nadir is not None else 1.0) for obj in problem.objectives
    }

    recompute_request = NautilusNavigatorRecomputeRequest(
        problem_id=1,
        parent_state_id=init_payload.state_id,
        go_back_step=0,
        steps=3,
        reference_point=reference_point,
    )

    recompute_response = post_json(
        client,
        "/method/nautilus_navigator/recompute",
        recompute_request.model_dump(),
        access_token,
    )

    assert recompute_response.status_code == status.HTTP_200_OK

    segment = NautilusNavigatorSegmentResponse.model_validate(recompute_response.json())
    assert segment.segment_steps == 3
    assert segment.segment_start_step == 0
    assert segment.state_id is not None

    for symbol in segment.objective_symbols:
        assert len(segment.lower_bounds[symbol]) == 4
        assert len(segment.upper_bounds[symbol]) == 4

    state = session.get(StateDB, segment.state_id)
    assert state.parent_id == init_payload.state_id
