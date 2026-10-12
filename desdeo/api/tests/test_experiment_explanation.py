"""Tests for the explanation shown after the anchor, and for the explanation providers."""

from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session, select

from desdeo.api.models import ExperimentEnrolment, ExperimentExplanation, ExperimentRun, ProblemDB
from desdeo.api.utils.experiment_definition import (
    content_hash,
    load_experiment_definition,
    store_experiment_definition,
)
from desdeo.api.utils.experiment_run import create_experiment_run
from desdeo.api.utils.explanation_providers import EXPLANATION_PROVIDERS, ForestGeneralization
from desdeo.problem import Problem
from desdeo.problem.testproblems import dtlz2
from desdeo.problem.testproblems.forest_landscape_problem import (
    forest_landscape_data,
    forest_landscape_problem,
    generalize,
    nearest_regimes,
)
from desdeo.tools import SolverResults

from .conftest import login
from .test_experiment_definition import EXPERIMENT_TOML
from .test_experiment_flow import _consent, _post, _solve

T = "2026-10-12T09:00:00.000Z"

# the test experiment with a required transition followed by a method stage, and an explanation for 'treatment'
EXPLAINED_TOML = (
    EXPERIMENT_TOML.replace('type = "text"\nrequired = false', 'type = "text"')
    + """
[[stages]]
id = "final"
type = "method"
[stages.payload]
method = "rpm"
problem = "study"

[explanation]
provider = "sum"
arms = ["treatment"]
"""
)


class SumOfObjectives(BaseModel):
    """A test provider: explains a solution by the sum of its objective values."""

    model_config = ConfigDict(extra="forbid")

    objective: str = "f_1"

    def check(self, problem: Problem) -> None:
        """Check that the problem has the objective."""
        if self.objective not in [o.symbol for o in problem.objectives]:
            msg = f"No objective {self.objective}."
            raise ValueError(msg)

    def explain(self, problem: Problem, solution: SolverResults) -> dict:
        """Explain a solution by the sum of its objective values."""
        return {"sum": sum(solution.optimal_objectives.values())}


@pytest.fixture(autouse=True)
def sum_provider(monkeypatch: pytest.MonkeyPatch):
    """Register the test provider."""
    monkeypatch.setitem(EXPLANATION_PROVIDERS, "sum", SumOfObjectives)


def _write(folder: Path, toml: str) -> Path:
    """Helper: write an experiment folder, returning the path of the TOML file."""
    (folder / "experiment.toml").write_text(toml, encoding="utf-8")
    (folder / "consent.md").write_text("Do you consent?\n", encoding="utf-8")
    (folder / "treatment.md").write_text("You will see more.\n", encoding="utf-8")
    return folder / "experiment.toml"


def _create_run(db_session: Session, folder: Path, toml: str = EXPLAINED_TOML) -> ExperimentRun:
    """Helper: create a run 'pilot' of an experiment, solving the first problem of the fixture."""
    snapshot = store_experiment_definition(db_session, load_experiment_definition(_write(folder, toml)))
    problem = db_session.exec(select(ProblemDB).order_by(ProblemDB.id)).first()
    return create_experiment_run(db_session, snapshot, "pilot", "pilot", {"study": problem})


@pytest.mark.parametrize(
    "toml",
    [
        EXPLAINED_TOML.replace('provider = "sum"', 'provider = "no-such-provider"'),
        EXPLAINED_TOML + "[explanation.options]\nno_such_option = 1\n",
        EXPLAINED_TOML.replace('arms = ["treatment"]', 'arms = ["nobody"]'),
        EXPLAINED_TOML.replace("anchor = true", "anchor = false"),
    ],
)
def test_definition_rejects_invalid_explanations(tmp_path: Path, toml: str):
    """Test that unknown providers and options, unknown arms, and an explanation without an anchor are rejected."""
    with pytest.raises(ValueError):
        load_experiment_definition(_write(tmp_path, toml))


def test_run_checks_the_provider_against_the_problem(session_and_user: dict, tmp_path: Path):
    """Test that a run cannot be created when the provider cannot explain solutions of a problem after the anchor."""
    toml = EXPLAINED_TOML + '[explanation.options]\nobjective = "g"\n'
    with pytest.raises(ValueError, match="No objective g"):
        _create_run(session_and_user["session"], tmp_path, toml)


@pytest.fixture
def run(session_and_user: dict, tmp_path: Path) -> ExperimentRun:
    """A run of the explained test experiment."""
    return _create_run(session_and_user["session"], tmp_path)


def _explain(client: TestClient, subject: dict, stage_ordinal: int, state_id: int):
    """Helper: request the explanation of a solution."""
    body = {"stage_ordinal": stage_ordinal, "state_id": state_id, "client_timestamp": T}
    return _post(client, "explanation", body, subject)


def _walk_to_final(client: TestClient, db_session: Session, run: ExperimentRun, arm: str) -> tuple[dict, int]:
    """Helper: enrol a subject in an arm and walk them to the final method stage, returning them and their anchor."""
    subject = client.post("/experiment/pilot/enrol", json={}).json()
    enrolment = db_session.exec(select(ExperimentEnrolment).order_by(ExperimentEnrolment.id.desc())).first()
    enrolment.arm = arm  # an arm set before the anchor is kept, so the arm is known from the start
    db_session.add(enrolment)
    db_session.commit()

    _consent(client, subject)
    anchor = _solve(client, run, subject["access_token"])
    assert _explain(client, subject, 1, anchor).status_code == status.HTTP_404_NOT_FOUND  # not before the anchor

    for ordinal, data in ((1, {}), (2, {"state_id": anchor}), (3, {"acknowledged": True})):
        _post(client, "stage/data", {"stage_ordinal": ordinal, "data": data, "client_timestamp": T}, subject)
        response = _post(client, "stage/advance", {"stage_ordinal": ordinal, "client_timestamp": T}, subject)
    assert response.json()["stage"]["id"] == "final"
    return subject, anchor


def test_explanation_is_shown_after_the_anchor_in_the_arm(
    client: TestClient, session_and_user: dict, run: ExperimentRun
):
    """Test that a subject of the explained arm gets the explanation after the anchor, stored with its hash."""
    db_session = session_and_user["session"]
    subject, _ = _walk_to_final(client, db_session, run, "treatment")
    state_id = _solve(client, run, subject["access_token"])

    response = _explain(client, subject, 4, state_id)
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert set(body["payload"]) == {"sum"}
    assert body["content_hash"] == content_hash(body["payload"])

    stored = db_session.exec(select(ExperimentExplanation)).one()
    assert (stored.stage_ordinal, stored.state_id, stored.solution_index) == (4, state_id, 0)
    assert (stored.provider, stored.payload, stored.client_timestamp) == ("sum", body["payload"], T)


def test_explanation_is_never_shown_to_other_arms(client: TestClient, session_and_user: dict, run: ExperimentRun):
    """Test that a subject of an arm not explained gets no explanation, and nothing naming an arm."""
    db_session = session_and_user["session"]
    subject, _ = _walk_to_final(client, db_session, run, "control")
    state_id = _solve(client, run, subject["access_token"])

    response = _explain(client, subject, 4, state_id)
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "treatment" not in response.text
    assert "control" not in response.text
    assert db_session.exec(select(ExperimentExplanation)).all() == []


def test_explanation_is_computed_once_per_solution(client: TestClient, session_and_user: dict, run: ExperimentRun):
    """Test that explaining a solution again returns the stored explanation, and that only own solutions count."""
    db_session = session_and_user["session"]
    subject, _ = _walk_to_final(client, db_session, run, "treatment")
    state_id = _solve(client, run, subject["access_token"])

    first = _explain(client, subject, 4, state_id).json()
    assert _explain(client, subject, 4, state_id).json() == first
    assert len(db_session.exec(select(ExperimentExplanation)).all()) == 1

    assert _explain(client, subject, 3, state_id).status_code == status.HTTP_409_CONFLICT  # not the current stage
    other_state_id = _solve(client, run, login(client))  # the analyst's state
    assert _explain(client, subject, 4, other_state_id).status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def _one_hot(regimes: dict[int, str], all_regimes: list[str]) -> SolverResults:
    """Helper: a solution of the forest problem choosing the given regime on each stand."""
    variables = {f"X_{stand}": [int(r == regime) for r in all_regimes] for stand, regime in regimes.items()}
    return SolverResults(optimal_variables=variables, optimal_objectives={}, success=True, message="")


def test_forest_provider_checks_the_problem():
    """Test that the forest provider accepts the problem of its lot only."""
    provider = ForestGeneralization()
    provider.check(forest_landscape_problem())

    for other in (ForestGeneralization(seed=1), ForestGeneralization(lot="B")):
        with pytest.raises(ValueError, match="is not the problem of lot"):
            other.check(forest_landscape_problem())
    with pytest.raises(ValueError, match="is not the problem of lot"):
        provider.check(dtlz2(5, 3))


def test_forest_provider_explains_the_own_lot_only():
    """Test that the forest explanation is the generalization of the own lot, the same for the same solution."""
    provider = ForestGeneralization()
    landscape = forest_landscape_data()
    problem = forest_landscape_problem(landscape)
    regimes = nearest_regimes(landscape, "A", {"npv": 1.0, "habitat": 0.0, "carbon": 0.0})  # a timber-heavy solution

    payload = provider.explain(problem, _one_hot(regimes, landscape.regimes))

    assert set(payload) == {"objectives", "properties"}
    assert set(payload["objectives"]) == {o.symbol for o in problem.objectives}
    assert set(payload["properties"]) == {f.symbol for f in problem.extra_funcs}
    expected = generalize(landscape, "A", regimes)
    assert payload["objectives"] == expected.model_dump(mode="json")["objectives"]
    assert content_hash(provider.explain(problem, _one_hot(regimes, landscape.regimes))) == content_hash(payload)
