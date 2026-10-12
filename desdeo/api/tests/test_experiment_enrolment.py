"""Tests for enrolling subjects in experiment runs through the campaign link, and resuming enrolments."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from jose import jwt
from sqlmodel import Session, select

from desdeo.api import AuthConfig
from desdeo.api.models import (
    ExperimentEnrolment,
    ExperimentRun,
    ProblemDB,
    ReferencePoint,
    RPMSolveRequest,
    StateDB,
    UserRole,
)
from desdeo.api.routers.experiment import SUBJECT_TOKEN_LIFETIME
from desdeo.api.utils.experiment_definition import load_experiment_definition, store_experiment_definition
from desdeo.api.utils.experiment_run import create_experiment_run
from desdeo.problem.testproblems import dtlz2

from .conftest import get_json, post_json
from .test_experiment_definition import EXPERIMENT_TOML


def _create_run(db_session: Session, folder: Path, slug: str) -> ExperimentRun:
    """Helper: create a run of the test experiment, solving the first problem of the fixture."""
    (folder / "experiment.toml").write_text(EXPERIMENT_TOML, encoding="utf-8")
    (folder / "consent.md").write_text("Do you consent?\n", encoding="utf-8")
    (folder / "treatment.md").write_text("You will see more.\n", encoding="utf-8")
    snapshot = store_experiment_definition(db_session, load_experiment_definition(folder / "experiment.toml"))
    problem = db_session.exec(select(ProblemDB).order_by(ProblemDB.id)).first()
    return create_experiment_run(db_session, snapshot, slug, slug, {"study": problem})


@pytest.fixture
def run(session_and_user: dict, tmp_path: Path) -> ExperimentRun:
    """A run of the test experiment, with the slug 'pilot'."""
    return _create_run(session_and_user["session"], tmp_path, "pilot")


def _enrol(client: TestClient, slug: str = "pilot", resume_token: str | None = None):
    """Helper: open the campaign link, with or without a stored resume token."""
    return client.post(f"/experiment/{slug}/enrol", json={"resume_token": resume_token})


def _username(client: TestClient, access_token: str) -> str:
    """Helper: the username of the user an access token belongs to."""
    return get_json(client, "/user_info", access_token).json()["username"]


def test_first_visit_enrols_a_subject(client: TestClient, session_and_user: dict, run: ExperimentRun):
    """Test that a first visit creates the subject, the enrolment, and the interactive session together."""
    db_session = session_and_user["session"]

    response = _enrol(client)
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert not body["resumed"]

    enrolment = db_session.exec(select(ExperimentEnrolment)).one()
    subject = enrolment.user
    assert subject.role == UserRole.subject
    assert _username(client, body["access_token"]) == subject.username
    assert enrolment.run_id == run.id
    assert (enrolment.experiment_id, enrolment.content_hash) == ("test-experiment", run.definition.content_hash)
    assert enrolment.session.user_id == subject.id
    assert subject.active_session_id == enrolment.session_id
    assert subject in run.group.members
    assert enrolment.resume_token_hash != body["resume_token"]  # only the hash is stored


def test_resume_token_returns_the_same_subject(client: TestClient, session_and_user: dict, run: ExperimentRun):
    """Test that a stored resume token returns the existing subject instead of enrolling a new one."""
    first = _enrol(client).json()
    second = _enrol(client, resume_token=first["resume_token"]).json()

    assert second["resumed"]
    assert second["resume_token"] == first["resume_token"]
    assert _username(client, second["access_token"]) == _username(client, first["access_token"])
    assert len(session_and_user["session"].exec(select(ExperimentEnrolment)).all()) == 1


def test_unknown_resume_token_enrols_a_new_subject(
    client: TestClient, session_and_user: dict, run: ExperimentRun, tmp_path: Path
):
    """Test that a resume token of no enrolment in the run, e.g., of another run, enrols a new subject."""
    db_session = session_and_user["session"]
    other_folder = tmp_path / "other"
    other_folder.mkdir()
    _create_run(db_session, other_folder, "live")
    in_live = _enrol(client, slug="live").json()

    for resume_token in ("not-a-token", in_live["resume_token"]):
        response = _enrol(client, resume_token=resume_token).json()
        assert not response["resumed"]
        assert response["resume_token"] != resume_token

    assert len(db_session.exec(select(ExperimentEnrolment).where(ExperimentEnrolment.run_id == run.id)).all()) == 2


def test_closed_run_resumes_but_does_not_enrol(client: TestClient, session_and_user: dict, run: ExperimentRun):
    """Test that a closed run refuses new subjects but lets enrolled subjects resume."""
    db_session = session_and_user["session"]
    enrolled = _enrol(client).json()

    run.is_open = False
    db_session.add(run)
    db_session.commit()

    assert _enrol(client).status_code == status.HTTP_403_FORBIDDEN
    assert _enrol(client, resume_token=enrolled["resume_token"]).json()["resumed"]


def test_finished_subject_cannot_resume(client: TestClient, session_and_user: dict, run: ExperimentRun):
    """Test that a subject whose tokens have been revoked cannot resume."""
    db_session = session_and_user["session"]
    enrolled = _enrol(client).json()

    enrolment = db_session.exec(select(ExperimentEnrolment)).one()
    enrolment.token_revoked = True
    db_session.add(enrolment)
    db_session.commit()

    assert _enrol(client, resume_token=enrolled["resume_token"]).status_code == status.HTTP_403_FORBIDDEN


def test_unknown_run_is_not_found(client: TestClient, run: ExperimentRun):
    """Test that the campaign link of a run that does not exist is not found."""
    assert _enrol(client, slug="no-such-run").status_code == status.HTTP_404_NOT_FOUND


def test_access_token_lasts_a_whole_session(client: TestClient, run: ExperimentRun):
    """Test that the subject's access token outlasts a full experiment session."""
    access_token = _enrol(client).json()["access_token"]
    claims = jwt.decode(access_token, AuthConfig.authjwt_secret_key, algorithms=[AuthConfig.authjwt_algorithm])

    lifetime = claims["exp"] - datetime.now(UTC).timestamp()
    assert lifetime > SUBJECT_TOKEN_LIFETIME.total_seconds() - 60
    assert SUBJECT_TOKEN_LIFETIME.total_seconds() >= 3 * 60 * 60


def test_subject_solves_the_run_problem_in_their_session(
    client: TestClient, session_and_user: dict, run: ExperimentRun
):
    """Test that a subject can solve the run's problem right away, with the states stored in their session."""
    db_session = session_and_user["session"]
    access_token = _enrol(client).json()["access_token"]
    enrolment = db_session.exec(select(ExperimentEnrolment)).one()

    request = RPMSolveRequest(
        problem_id=run.problems["study"],
        preference=ReferencePoint(aspiration_levels={"f_1": 0.5, "f_2": 0.3, "f_3": 0.4}),
        include_perturbed=False,
    )
    response = post_json(client, "/method/rpm/solve", request.model_dump(), access_token)
    assert response.status_code == status.HTTP_200_OK

    states = db_session.exec(select(StateDB).where(StateDB.session_id == enrolment.session_id)).all()
    assert len(states) == 1


def test_subject_cannot_add_problems(client: TestClient, session_and_user: dict, run: ExperimentRun):
    """Test that a subject's token cannot be used to add problems to the database."""
    db_session = session_and_user["session"]
    access_token = _enrol(client).json()["access_token"]
    problem_json = dtlz2(5, 3).model_dump_json()

    response = client.post(
        "/problem/add",
        content=problem_json,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

    response = client.post(
        "/problem/add_json",
        files={"json_file": ("problem.json", problem_json, "application/json")},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

    subject_id = db_session.exec(select(ExperimentEnrolment)).one().user_id
    assert db_session.exec(select(ProblemDB).where(ProblemDB.user_id == subject_id)).all() == []
