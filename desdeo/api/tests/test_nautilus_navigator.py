"""Tests related to NAUTILUS Navigator models and routes."""

import json

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from desdeo.api.models import (
    NautilusNavigatorInitializeRequest,
    NautilusNavigatorInitializeResponse,
    NautilusNavigatorNavigateRequest,
    NautilusNavigatorNavigateResponse,
    ProblemDB,
)
from desdeo.problem import Problem

from .conftest import login, post_json


def test_nautilus_navigator_initialize_and_navigate(
    client: TestClient, session_and_user: dict[str, Session]
):
    """Test the NAUTILUS Navigator initialize and navigate endpoints."""
    session = session_and_user["session"]
    access_token = login(client)

    problem_db = session.exec(select(ProblemDB).where(ProblemDB.name == "dtlz2")).first()
    assert problem_db is not None

    total_steps = 3
    init_request = NautilusNavigatorInitializeRequest(
        problem_id=problem_db.id,
        total_steps=total_steps,
    )

    raw_response = post_json(
        client,
        "/method/nautilus_navigator/initialize",
        init_request.model_dump(),
        access_token,
    )
    assert raw_response.status_code == 200

    init_response = NautilusNavigatorInitializeResponse.model_validate(json.loads(raw_response.content))

    assert init_response.step_number == 0
    assert init_response.steps_remaining == total_steps

    problem = Problem.from_problemdb(problem_db)
    reference_point = {obj.symbol: (obj.ideal + obj.nadir) / 2 for obj in problem.objectives}

    navigate_request = NautilusNavigatorNavigateRequest(
        problem_id=problem_db.id,
        total_steps=total_steps,
        go_back_step=0,
        steps_remaining=total_steps,
        reference_point=reference_point,
        bounds=None,
    )

    raw_response = post_json(
        client,
        "/method/nautilus_navigator/navigate",
        navigate_request.model_dump(),
        access_token,
    )
    assert raw_response.status_code == 200

    navigate_response = NautilusNavigatorNavigateResponse.model_validate(json.loads(raw_response.content))

    assert navigate_response.total_steps == total_steps
    assert navigate_response.current_step == total_steps
    assert len(navigate_response.step_numbers) == total_steps + 1

    for symbol in navigate_response.objective_symbols:
        assert len(navigate_response.lower_bounds[symbol]) == total_steps + 1
        assert len(navigate_response.upper_bounds[symbol]) == total_steps + 1
