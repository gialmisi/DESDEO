"""Tests for experiment runs and enrolments."""

import hashlib
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from desdeo.api.models import (
    ExperimentDefinitionDB,
    ExperimentEnrolment,
    ExperimentRun,
    InteractiveSessionDB,
    ProblemDB,
    User,
    UserRole,
)
from desdeo.api.routers.user_authentication import UNUSABLE_PASSWORD_HASH
from desdeo.api.routers.utils import fetch_problem_with_role_check
from desdeo.api.utils.experiment_definition import load_experiment_definition, store_experiment_definition
from desdeo.api.utils.experiment_run import create_experiment_run

from .test_experiment_definition import EXPERIMENT_TOML


@pytest.fixture
def snapshot(session_and_user: dict, tmp_path: Path) -> ExperimentDefinitionDB:
    """A stored experiment definition whose method stage solves the problem named 'study'."""
    (tmp_path / "experiment.toml").write_text(EXPERIMENT_TOML, encoding="utf-8")
    (tmp_path / "consent.md").write_text("Do you consent?\n", encoding="utf-8")
    (tmp_path / "treatment.md").write_text("You will see more.\n", encoding="utf-8")
    definition = load_experiment_definition(tmp_path / "experiment.toml")
    return store_experiment_definition(session_and_user["session"], definition)


def _problem(db_session: Session) -> ProblemDB:
    """Helper: the first problem created by the fixture."""
    return db_session.exec(select(ProblemDB).order_by(ProblemDB.id)).first()


def _enrol(db_session: Session, run: ExperimentRun, username: str) -> ExperimentEnrolment:
    """Helper: enrol a new subject in a run, as the campaign link will."""
    subject = User(username=username, password_hash=UNUSABLE_PASSWORD_HASH, role=UserRole.subject)
    interactive_session = InteractiveSessionDB(user=subject, info=f"Experiment run {run.slug}")
    run.group.members.append(subject)
    enrolment = ExperimentEnrolment(
        user=subject,
        run=run,
        experiment_id=run.definition.experiment_id,
        content_hash=run.definition.content_hash,
        session=interactive_session,
        resume_token_hash=hashlib.sha256(username.encode()).hexdigest(),
    )
    db_session.add(enrolment)
    db_session.commit()
    db_session.refresh(enrolment)
    return enrolment


def test_run_creates_a_group_granting_its_problems(session_and_user: dict, snapshot: ExperimentDefinitionDB):
    """Test that a run gets its own group, which grants access to the run's problems."""
    db_session = session_and_user["session"]
    problem = _problem(db_session)

    run = create_experiment_run(
        db_session, snapshot, "pilot-1", "pilot", {"study": problem}, owner=session_and_user["user"]
    )

    assert run.is_open
    assert run.definition.content_hash == snapshot.content_hash
    assert run.problems == {"study": problem.id}
    assert run.group.name == "experiment-run:pilot-1"
    assert [p.id for p in run.group.problems] == [problem.id]
    assert run.group.members == []


@pytest.mark.parametrize("problems", [{}, {"study": None, "other": None}, {"stdy": None}])
def test_run_needs_exactly_the_problems_of_the_definition(
    session_and_user: dict, snapshot: ExperimentDefinitionDB, problems: dict
):
    """Test that a run is rejected when its problems do not match the problem names of the definition."""
    db_session = session_and_user["session"]
    problem = _problem(db_session)

    with pytest.raises(ValueError, match="exactly the names"):
        create_experiment_run(db_session, snapshot, "run", "pilot", dict.fromkeys(problems, problem))
    assert db_session.exec(select(ExperimentRun)).all() == []


def test_pilot_and_live_runs_have_separate_groups(session_and_user: dict, snapshot: ExperimentDefinitionDB):
    """Test that runs of the same experiment keep their subjects apart."""
    db_session = session_and_user["session"]
    problem = _problem(db_session)
    pilot = create_experiment_run(db_session, snapshot, "pilot", "pilot", {"study": problem})
    live = create_experiment_run(db_session, snapshot, "live", "live", {"study": problem})

    _enrol(db_session, pilot, "subject-1")
    _enrol(db_session, live, "subject-2")

    assert pilot.group_id != live.group_id
    assert [m.username for m in pilot.group.members] == ["subject-1"]
    assert [m.username for m in live.group.members] == ["subject-2"]


def test_enrolment_links_subject_run_group_and_session(session_and_user: dict, snapshot: ExperimentDefinitionDB):
    """Test that an enrolment links its parts, with all assignment fields null before the anchor."""
    db_session = session_and_user["session"]
    problem = _problem(db_session)
    run = create_experiment_run(db_session, snapshot, "pilot", "pilot", {"study": problem})

    enrolment = _enrol(db_session, run, "subject")

    assert enrolment.user.role == UserRole.subject
    assert enrolment.run.group.members == [enrolment.user]
    assert enrolment.session.user_id == enrolment.user_id
    assert (enrolment.experiment_id, enrolment.content_hash) == ("test-experiment", snapshot.content_hash)
    assert enrolment.arm is None
    assert enrolment.stratum is None
    assert enrolment.stratification_value is None
    assert enrolment.is_binding is None
    assert enrolment.assignment_details is None
    assert enrolment.assigned_at is None
    assert not enrolment.token_revoked

    # membership of the run's group gives the subject access to the run's problem
    assert fetch_problem_with_role_check(enrolment.user, problem.id, db_session).id == problem.id


def test_a_subject_is_enrolled_once(session_and_user: dict, snapshot: ExperimentDefinitionDB):
    """Test that a user cannot have two enrolments."""
    db_session = session_and_user["session"]
    run = create_experiment_run(db_session, snapshot, "pilot", "pilot", {"study": _problem(db_session)})
    enrolment = _enrol(db_session, run, "subject")

    db_session.add(
        ExperimentEnrolment(
            user_id=enrolment.user_id,
            run_id=run.id,
            experiment_id=enrolment.experiment_id,
            content_hash=enrolment.content_hash,
            session_id=enrolment.session_id,
            resume_token_hash="another",  # noqa: S106
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
