"""Tests for the flow of subjects through the stages of an experiment, consent, and events."""

from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from desdeo.api.models import (
    ExperimentEnrolment,
    ExperimentEvent,
    ExperimentRun,
    ExperimentStageProgress,
    ReferencePoint,
    RPMSolveRequest,
    StateDB,
)

from .conftest import get_json, login, post_json
from .test_experiment_enrolment import _create_run

T0, T1, T2, T3 = (f"2026-10-12T09:0{i}:00.000Z" for i in range(4))


@pytest.fixture
def run(session_and_user: dict, tmp_path: Path) -> ExperimentRun:
    """A run of the test experiment: consent, anchoring (RPM), anchor commit, and an optional transition."""
    return _create_run(session_and_user["session"], tmp_path, "pilot")


@pytest.fixture
def subject(client: TestClient, run: ExperimentRun) -> dict:
    """An enrolled subject: their access and resume tokens."""
    return client.post("/experiment/pilot/enrol", json={}).json()


def _post(client: TestClient, endpoint: str, json: dict | list, subject: dict):
    """Helper: post as the subject."""
    return post_json(client, f"/experiment/{endpoint}", json, subject["access_token"])


def _consent(client: TestClient, subject: dict):
    """Helper: enter the consent stage, consent, and advance to the anchoring stage."""
    _post(client, "stage/enter", {"client_timestamp": T0}, subject)
    _post(client, "stage/data", {"stage_ordinal": 0, "data": {"acknowledged": True}, "client_timestamp": T1}, subject)
    return _post(client, "stage/advance", {"stage_ordinal": 0, "client_timestamp": T2}, subject)


def _solve(client: TestClient, run: ExperimentRun, access_token: str) -> int:
    """Helper: solve the run's problem with the RPM, returning the id of the state."""
    request = RPMSolveRequest(
        problem_id=run.problems["study"],
        preference=ReferencePoint(aspiration_levels={"f_1": 0.5, "f_2": 0.3, "f_3": 0.4}),
        include_perturbed=False,
    )
    return post_json(client, "/method/rpm/solve", request.model_dump(), access_token).json()["id"]


def _progress(db_session: Session) -> list[ExperimentStageProgress]:
    """Helper: all stage progress rows, in stage order."""
    db_session.expire_all()
    return db_session.exec(select(ExperimentStageProgress).order_by(ExperimentStageProgress.stage_ordinal)).all()


def test_entering_starts_with_consent_once(client: TestClient, session_and_user: dict, subject: dict):
    """Test that the first stage is consent, and that entering again returns it without entering it again."""
    first = _post(client, "stage/enter", {"client_timestamp": T0}, subject).json()
    again = _post(client, "stage/enter", {"client_timestamp": T1}, subject).json()

    assert first == again
    assert not first["finished"]
    stage = first["stage"]
    assert (stage["ordinal"], stage["id"], stage["type"], stage["required"]) == (0, "consent", "text", True)
    assert stage["payload"]["body"] == "Do you consent?\n"
    assert stage["data"] is None
    assert [(p.stage_ordinal, p.entered_at, p.exited_at) for p in _progress(session_and_user["session"])] == [
        (0, T0, None)
    ]


def test_no_stage_is_reachable_before_consent(client: TestClient, session_and_user: dict, subject: dict):
    """Test that a subject cannot advance or store data for later stages before consenting."""
    _post(client, "stage/enter", {"client_timestamp": T0}, subject)

    response = _post(client, "stage/advance", {"stage_ordinal": 0, "client_timestamp": T1}, subject)
    assert response.status_code == status.HTTP_409_CONFLICT
    for endpoint, body in (
        ("stage/advance", {"stage_ordinal": 1, "client_timestamp": T1}),
        ("stage/data", {"stage_ordinal": 1, "data": {}, "client_timestamp": T1}),
    ):
        assert _post(client, endpoint, body, subject).status_code == status.HTTP_409_CONFLICT

    # not consenting is not consent
    response = _post(
        client, "stage/data", {"stage_ordinal": 0, "data": {"acknowledged": False}, "client_timestamp": T1}, subject
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert [p.stage_ordinal for p in _progress(session_and_user["session"])] == [0]


def test_advancing_records_exit_and_enters_the_next_stage(client: TestClient, session_and_user: dict, subject: dict):
    """Test that advancing with the required data exits the stage and enters the next one at the same moment."""
    response = _consent(client, subject).json()

    assert response["stage"]["id"] == "anchoring"
    assert response["stage"]["payload"] == {"method": "rpm", "problem": "study", "instructions": None}
    consent, anchoring = _progress(session_and_user["session"])
    assert (consent.entered_at, consent.data_submitted_at, consent.exited_at) == (T0, T1, T2)
    assert consent.data == {"acknowledged": True}
    assert (anchoring.entered_at, anchoring.exited_at) == (T2, None)

    # a stale request about the left stage is rejected
    response = _post(client, "stage/advance", {"stage_ordinal": 0, "client_timestamp": T3}, subject)
    assert response.status_code == status.HTTP_409_CONFLICT


def test_commit_needs_a_state_of_the_subjects_session(
    client: TestClient, session_and_user: dict, run: ExperimentRun, subject: dict
):
    """Test that a commit stage accepts only a state in the subject's own interactive session."""
    # offset the ids of the state rows from the ids of the method states the method end-points return
    session_and_user["session"].add(StateDB())
    session_and_user["session"].commit()

    _consent(client, subject)
    state_id = _solve(client, run, subject["access_token"])
    _post(client, "stage/data", {"stage_ordinal": 1, "data": {"state_id": state_id}, "client_timestamp": T3}, subject)
    _post(client, "stage/advance", {"stage_ordinal": 1, "client_timestamp": T3}, subject)

    other_state_id = _solve(client, run, login(client))  # the analyst's state, in no subject's session
    for data in ({"state_id": other_state_id}, {"state_id": 10_000}, {"state_id": state_id, "solution_index": 1}, {}):
        response = _post(client, "stage/data", {"stage_ordinal": 2, "data": data, "client_timestamp": T3}, subject)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    response = _post(
        client, "stage/data", {"stage_ordinal": 2, "data": {"state_id": state_id}, "client_timestamp": T3}, subject
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"] == {"state_id": state_id, "solution_index": 0}


def _walk_to_transition(client: TestClient, run: ExperimentRun, subject: dict) -> dict:
    """Helper: walk a subject from consent to the transition stage, returning the transition response."""
    _consent(client, subject)
    state_id = _solve(client, run, subject["access_token"])
    for ordinal, data in ((1, {}), (2, {"state_id": state_id})):
        _post(client, "stage/data", {"stage_ordinal": ordinal, "data": data, "client_timestamp": T3}, subject)
        response = _post(client, "stage/advance", {"stage_ordinal": ordinal, "client_timestamp": T3}, subject)
    return response


@pytest.mark.parametrize("arm_and_body", [("control", "Neutral."), ("treatment", "You will see more.\n")])
def test_texts_are_resolved_for_the_arm_without_revealing_it(
    client: TestClient, session_and_user: dict, run: ExperimentRun, subject: dict, arm_and_body: tuple
):
    """Test that a text that differs per arm shows the subject's arm's text, and never the arm or other texts."""
    arm, body = arm_and_body
    db_session = session_and_user["session"]
    enrolment = db_session.exec(select(ExperimentEnrolment)).one()
    enrolment.arm = arm
    db_session.add(enrolment)
    db_session.commit()

    response = _walk_to_transition(client, run, subject)
    payload = response.json()["stage"]["payload"]

    assert payload["body"] == body
    assert payload["arm_bodies"] == {}
    assert "treatment" not in response.text
    assert "arm" not in response.json()["stage"]

    entered = _post(client, "stage/enter", {"client_timestamp": T3}, subject)
    assert entered.json()["stage"]["payload"]["body"] == body
    assert "treatment" not in entered.text


def test_optional_last_stage_finishes_without_data(
    client: TestClient, session_and_user: dict, run: ExperimentRun, subject: dict
):
    """Test that an optional stage can be left without data, and that leaving the last stage finishes the flow."""
    transition = _walk_to_transition(client, run, subject).json()["stage"]
    assert (transition["required"], transition["is_last"]) == (False, True)

    response = _post(client, "stage/advance", {"stage_ordinal": 3, "client_timestamp": T3}, subject).json()
    assert response == {"stage": None, "finished": True}
    assert _post(client, "stage/enter", {"client_timestamp": T3}, subject).json() == {"stage": None, "finished": True}
    assert all(p.exited_at is not None for p in _progress(session_and_user["session"]))


def test_consent_record_has_no_identifiers(client: TestClient, session_and_user: dict, subject: dict):
    """Test that the consent record contains the consent text and time, and nothing identifying the subject."""
    assert get_json(client, "/experiment/consent", subject["access_token"]).status_code == status.HTTP_404_NOT_FOUND

    _consent(client, subject)
    response = get_json(client, "/experiment/consent", subject["access_token"])

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-disposition"] == 'attachment; filename="consent.md"'
    assert response.text == (
        "# Consent\n\nDo you consent?\n\n---\n\nConsent given on " + T1 + " to the study 'A test experiment'.\n"
    )
    enrolment = session_and_user["session"].exec(select(ExperimentEnrolment)).one()
    for identifier in (
        enrolment.user.username,
        subject["resume_token"],
        subject["access_token"],
        enrolment.resume_token_hash,
    ):
        assert identifier not in response.text


def test_events_are_stored_as_received(client: TestClient, session_and_user: dict, subject: dict):
    """Test that a batch of events is stored with the client timestamps unmodified, and invalid events rejected."""
    events = [
        {"stage_ordinal": 0, "event_type": "stage_entered", "client_timestamp": "2026-10-12T09:00:00.123Z"},
        {
            "stage_ordinal": 0,
            "event_type": "visibility_changed",
            "payload": {"visible": False},
            "client_timestamp": "2026-10-12T12:00:01.5+03:00",
        },
    ]
    response = _post(client, "events", events, subject)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == len(events)

    stored = session_and_user["session"].exec(select(ExperimentEvent).order_by(ExperimentEvent.id)).all()
    assert [(e.event_type, e.payload, e.client_timestamp) for e in stored] == [
        ("stage_entered", {}, "2026-10-12T09:00:00.123Z"),
        ("visibility_changed", {"visible": False}, "2026-10-12T12:00:01.5+03:00"),
    ]

    for invalid in (
        {"stage_ordinal": 0, "event_type": "no_such_event", "client_timestamp": T0},
        {"stage_ordinal": 0, "event_type": "stage_entered", "client_timestamp": "2026-10-12T09:00:00"},  # no zone
        {"stage_ordinal": 0, "event_type": "stage_entered", "client_timestamp": "yesterday"},
    ):
        assert _post(client, "events", [invalid], subject).status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert len(session_and_user["session"].exec(select(ExperimentEvent)).all()) == len(events)


def test_only_enrolled_subjects_take_part(client: TestClient, session_and_user: dict, subject: dict):
    """Test that users who are not enrolled, and subjects whose tokens are revoked, cannot use the flow."""
    analyst = {"access_token": login(client)}
    assert _post(client, "stage/enter", {"client_timestamp": T0}, analyst).status_code == status.HTTP_403_FORBIDDEN

    db_session = session_and_user["session"]
    enrolment = db_session.exec(select(ExperimentEnrolment)).one()
    enrolment.token_revoked = True
    db_session.add(enrolment)
    db_session.commit()
    assert _post(client, "stage/enter", {"client_timestamp": T0}, subject).status_code == status.HTTP_403_FORBIDDEN
