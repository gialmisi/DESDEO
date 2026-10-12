"""Tests for stratifying subjects and assigning them to arms at the anchor."""

import random
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from desdeo.api.models import (
    ExperimentDefinition,
    ExperimentEnrolment,
    ExperimentRun,
    ExperimentStageProgress,
    Policy,
    ProblemDB,
)
from desdeo.api.utils.experiment_assignment import anchor_distance, assign_arm, is_complete, stratum_of
from desdeo.api.utils.experiment_definition import load_experiment_definition, store_experiment_definition
from desdeo.api.utils.experiment_run import create_experiment_run, enrol_subject
from desdeo.problem.testproblems import dtlz2

from .test_experiment_definition import EXPERIMENT_TOML
from .test_experiment_enrolment import _create_run
from .test_experiment_flow import _consent, _post, _solve

# the test experiment bounds f_1 and f_2 (minimized, ideal 0, nadir 1) at 0.5, with cut points 0.1 and 0.3
COMPLIANT = {"f_1": 0.2, "f_2": 0.3, "f_3": 0.9}  # distance 0, stratum 0
FAR = {"f_1": 0.8, "f_2": 0.9, "f_3": 0.0}  # distance hypot(0.3, 0.4) = 0.5, stratum 2
T = "2026-10-12T09:00:00.000Z"


@pytest.fixture
def run(session_and_user: dict, tmp_path: Path) -> ExperimentRun:
    """A run of the test experiment, with the arms 'control' and 'treatment'."""
    return _create_run(session_and_user["session"], tmp_path, "pilot")


def _definition(run: ExperimentRun) -> ExperimentDefinition:
    return run.definition.to_definition()


def _subject(db_session: Session, run: ExperimentRun, arm: str, stratum: int, *, complete: bool) -> None:
    """Helper: a subject assigned an arm in a stratum, who has completed the experiment or abandoned it."""
    enrolment, _ = enrol_subject(db_session, run)
    enrolment.arm, enrolment.stratum = arm, stratum
    required = [i for i, stage in enumerate(_definition(run).stages) if stage.required]
    for ordinal in required if complete else required[:1]:
        db_session.add(
            ExperimentStageProgress(
                enrolment_id=enrolment.id, stage_ordinal=ordinal, stage_id="s", entered_at=T, data={"x": 1}
            )
        )
    db_session.add(enrolment)
    db_session.commit()


def _assign(db_session: Session, run: ExperimentRun, objectives: dict, seed: int | None = None) -> ExperimentEnrolment:
    """Helper: enrol a new subject and assign them at an anchor with the objective values."""
    enrolment, _ = enrol_subject(db_session, run)
    return assign_arm(db_session, enrolment, _definition(run), dtlz2(5, 3), objectives, seed=seed)


def test_anchor_distance_is_normalized_euclidean_shortfall():
    """Test the distance to the compliant region, for minimized and maximized objectives."""
    policy = Policy(thresholds={"f_1": 0.5, "f_2": 0.5})
    problem = dtlz2(5, 3)

    assert anchor_distance(policy, problem, COMPLIANT) == 0
    assert anchor_distance(policy, problem, {"f_1": 0.5, "f_2": 0.5, "f_3": 1.0}) == 0  # on the boundary
    assert anchor_distance(policy, problem, FAR) == pytest.approx(0.5)
    assert anchor_distance(policy, problem, {"f_1": 0.8, "f_2": 0.1, "f_3": 0.0}) == pytest.approx(0.3)

    maximized = problem.model_copy(
        update={
            "objectives": [
                o.model_copy(update={"maximize": True, "ideal": 10.0, "nadir": 0.0}) for o in problem.objectives
            ]
        }
    )
    policy = Policy(thresholds={"f_1": 6.0})
    assert anchor_distance(policy, maximized, {"f_1": 4.0, "f_2": 0.0, "f_3": 0.0}) == pytest.approx(0.2)
    assert anchor_distance(policy, maximized, {"f_1": 7.0, "f_2": 0.0, "f_3": 0.0}) == 0


def test_strata_follow_the_cut_points():
    """Test that values map to strata, with a value on a cut point in the upper stratum."""
    assert [stratum_of(v, [0.1, 0.3]) for v in (0.0, 0.05, 0.1, 0.2, 0.3, 0.9)] == [0, 0, 1, 1, 2, 2]


def test_empty_study_assigns_any_arm_and_records_the_counts(session_and_user: dict, run: ExperimentRun):
    """Test that the first subject may get any arm, with level counts and the seed recorded."""
    enrolment = _assign(session_and_user["session"], run, FAR)

    assert enrolment.arm in {"control", "treatment"}
    assert (enrolment.stratum, enrolment.is_binding) == (2, True)
    assert enrolment.stratification_value == pytest.approx(0.5)
    assert enrolment.assigned_at is not None
    details = enrolment.assignment_details
    assert details["completed_counts"] == {"control": 0, "treatment": 0}
    assert details["candidates"] == ["control", "treatment"]
    assert details["anchor_objectives"] == FAR
    assert isinstance(details["seed"], int)


def test_lopsided_stratum_gets_the_arm_behind(session_and_user: dict, run: ExperimentRun):
    """Test that the next subject in a stratum is assigned the arm furthest behind there, whatever the seed."""
    db_session = session_and_user["session"]
    for arm in ("control", "control", "treatment"):
        _subject(db_session, run, arm, 2, complete=True)
    for arm in ("treatment", "treatment", "treatment"):  # other strata do not count
        _subject(db_session, run, arm, 0, complete=True)

    for seed in range(10):
        enrolment = _assign(db_session, run, FAR, seed=seed)
        assert enrolment.arm == "treatment"
        assert enrolment.assignment_details["completed_counts"] == {"control": 2, "treatment": 1}
        # make the assigned subject abandon, so the next one sees the same counts
        enrolment.arm = None
        db_session.add(enrolment)
        db_session.commit()


def test_tie_is_reproducible_from_the_logged_seed(session_and_user: dict, run: ExperimentRun):
    """Test that with level cells the choice is random, and reproducible from the logged seed."""
    db_session = session_and_user["session"]
    _subject(db_session, run, "control", 0, complete=True)
    _subject(db_session, run, "treatment", 0, complete=True)

    arms = set()
    for _ in range(20):
        enrolment = _assign(db_session, run, COMPLIANT)
        details = enrolment.assignment_details
        assert details["candidates"] == ["control", "treatment"]
        assert random.Random(details["seed"]).choice(details["candidates"]) == enrolment.arm  # noqa: S311
        assert _assign(db_session, run, COMPLIANT, seed=details["seed"]).arm == enrolment.arm
        arms.add(enrolment.arm)
    assert arms == {"control", "treatment"}  # both arms happen, with probability 1 - 2**-19


def test_abandoned_subjects_do_not_count(session_and_user: dict, run: ExperimentRun):
    """Test that assigned but incomplete subjects do not count, so the balance corrects toward abandoned arms."""
    db_session = session_and_user["session"]
    _subject(db_session, run, "control", 1, complete=True)
    for _ in range(3):
        _subject(db_session, run, "treatment", 1, complete=False)

    middle = {"f_1": 0.7, "f_2": 0.5, "f_3": 0.0}  # distance 0.2, stratum 1
    enrolment = _assign(db_session, run, middle)
    assert enrolment.stratum == 1
    assert enrolment.assignment_details["completed_counts"] == {"control": 1, "treatment": 0}
    assert enrolment.arm == "treatment"


@pytest.mark.parametrize(("objectives", "binding"), [(FAR, True), (COMPLIANT, False)])
def test_control_subjects_carry_binding_status(
    session_and_user: dict, run: ExperimentRun, objectives: dict, binding: bool
):
    """Test that a subject assigned to control still has the stratification value and binding status stored."""
    db_session = session_and_user["session"]
    stratum = stratum_of(anchor_distance(_definition(run).policy, dtlz2(5, 3), objectives), [0.1, 0.3])
    _subject(db_session, run, "treatment", stratum, complete=True)  # so that control is behind

    enrolment = _assign(db_session, run, objectives)
    db_session.expire_all()
    stored = db_session.get(ExperimentEnrolment, enrolment.id)

    assert stored.arm == "control"
    assert stored.is_binding is binding
    assert stored.stratification_value is not None
    assert stored.stratum == stratum


def test_completion_needs_every_required_stage(session_and_user: dict, run: ExperimentRun):
    """Test that a subject is complete only when every required stage has data."""
    db_session = session_and_user["session"]
    _subject(db_session, run, "control", 0, complete=True)
    _subject(db_session, run, "control", 0, complete=False)
    complete, abandoned = db_session.exec(select(ExperimentEnrolment).order_by(ExperimentEnrolment.id)).all()

    assert is_complete(db_session, complete, _definition(run))
    assert not is_complete(db_session, abandoned, _definition(run))


def test_anchor_commit_assigns_the_arm_once(client: TestClient, session_and_user: dict, run: ExperimentRun):
    """Test that leaving the anchor commit through the API assigns the arm from the committed solution, once."""
    db_session = session_and_user["session"]
    subject = client.post("/experiment/pilot/enrol", json={}).json()
    _consent(client, subject)
    state_id = _solve(client, run, subject["access_token"])
    _post(client, "stage/data", {"stage_ordinal": 1, "data": {}, "client_timestamp": T}, subject)
    _post(client, "stage/advance", {"stage_ordinal": 1, "client_timestamp": T}, subject)
    _post(client, "stage/data", {"stage_ordinal": 2, "data": {"state_id": state_id}, "client_timestamp": T}, subject)

    enrolment = db_session.exec(select(ExperimentEnrolment)).one()
    assert enrolment.arm is None  # storing the commit does not assign; leaving the anchor commit does

    response = _post(client, "stage/advance", {"stage_ordinal": 2, "client_timestamp": T}, subject)
    assert response.json()["stage"]["id"] == "transition"
    db_session.expire_all()
    enrolment = db_session.exec(select(ExperimentEnrolment)).one()
    assert enrolment.arm in {"control", "treatment"}
    assert enrolment.is_binding is not None
    anchor = enrolment.assignment_details["anchor_objectives"]
    assert enrolment.stratification_value == pytest.approx(
        anchor_distance(_definition(run).policy, dtlz2(5, 3), anchor)
    )
    assert "arm" not in response.json()["stage"]


def test_run_rejects_a_policy_on_unknown_objectives(session_and_user: dict, tmp_path: Path):
    """Test that a run cannot be created when the policy bounds objectives its problem does not have."""
    db_session = session_and_user["session"]
    (tmp_path / "experiment.toml").write_text(EXPERIMENT_TOML.replace("f_2 = 0.5", "g = 0.5"), encoding="utf-8")
    (tmp_path / "consent.md").write_text("Do you consent?\n", encoding="utf-8")
    (tmp_path / "treatment.md").write_text("You will see more.\n", encoding="utf-8")
    snapshot = store_experiment_definition(db_session, load_experiment_definition(tmp_path / "experiment.toml"))
    problem = db_session.exec(select(ProblemDB).order_by(ProblemDB.id)).first()

    with pytest.raises(ValueError, match="not an objective"):
        create_experiment_run(db_session, snapshot, "bad", "pilot", {"study": problem})
