"""Tests for experiment definitions: loading, hashing, storing, and the subject role."""

from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlmodel import Session

from desdeo.api.models import CommitStage, ExperimentDefinitionDB, TextStage, User, UserRole
from desdeo.api.routers.user_authentication import UNUSABLE_PASSWORD_HASH, verify_password
from desdeo.api.utils.experiment_definition import (
    ExperimentDefinitionConflictError,
    definition_hash,
    load_experiment_definition,
    store_experiment_definition,
)

EXPERIMENT_TOML = """
# A comment
id = "test-experiment"
name = "A test experiment"
version = "1"
arms = ["control", "treatment"]

[stratification]
variable = "anchor_distance"
cut_points = [0.1, 0.3]

[[stages]]
id = "consent"
type = "text"
[stages.payload]
title = "Consent"
body_file = "consent.md"

[[stages]]
id = "anchoring"
type = "method"
[stages.payload]
method = "rpm"
problem = "study"

[[stages]]
id = "anchor-commit"
type = "commit"
[stages.payload]
confirmation = "Commit?"
anchor = true

[[stages]]
id = "transition"
type = "text"
required = false
[stages.payload]
title = "Transition"
body = "Neutral."
arm_bodies_file = { treatment = "treatment.md" }
"""


@pytest.fixture
def experiment_folder(tmp_path: Path) -> Path:
    """An experiment folder with a TOML file and the texts it refers to."""
    (tmp_path / "experiment.toml").write_text(EXPERIMENT_TOML, encoding="utf-8")
    (tmp_path / "consent.md").write_text("Do you *consent*?\n", encoding="utf-8")
    (tmp_path / "treatment.md").write_text("You will see more.\n", encoding="utf-8")
    return tmp_path


def test_load_inlines_texts_and_types_stages(experiment_folder: Path):
    """Test that texts are read from files and that each stage gets the payload model of its type."""
    definition = load_experiment_definition(experiment_folder / "experiment.toml")

    assert [stage.type for stage in definition.stages] == ["text", "method", "commit", "text"]
    consent, _, commit, transition = definition.stages
    assert isinstance(consent, TextStage)
    assert consent.payload.body == "Do you *consent*?\n"
    assert isinstance(commit, CommitStage)
    assert commit.payload.anchor
    assert transition.payload.arm_bodies == {"treatment": "You will see more.\n"}
    assert not transition.required
    assert definition.stratification.cut_points == [0.1, 0.3]


def test_hash_is_stable_and_ignores_formatting(experiment_folder: Path):
    """Test that the hash is the same across loads and does not change with comments or whitespace."""
    path = experiment_folder / "experiment.toml"
    first = definition_hash(load_experiment_definition(path))
    assert definition_hash(load_experiment_definition(path)) == first

    path.write_text("# Another comment\n\n" + EXPERIMENT_TOML.replace("[[stages]]", "\n[[stages]]"), encoding="utf-8")
    assert definition_hash(load_experiment_definition(path)) == first


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('name = "A test experiment"', 'name = "Another name"'),
        ("cut_points = [0.1, 0.3]", "cut_points = [0.1, 0.4]"),
        ('confirmation = "Commit?"', 'confirmation = "Commit now?"'),
        ('id = "transition"\ntype = "text"\nrequired = false', 'id = "transition"\ntype = "text"'),
    ],
)
def test_hash_changes_with_any_value(experiment_folder: Path, old: str, new: str):
    """Test that changing a value in the TOML file changes the hash."""
    path = experiment_folder / "experiment.toml"
    original = definition_hash(load_experiment_definition(path))

    assert old in EXPERIMENT_TOML
    path.write_text(EXPERIMENT_TOML.replace(old, new), encoding="utf-8")
    assert definition_hash(load_experiment_definition(path)) != original


def test_hash_changes_with_a_text_file(experiment_folder: Path):
    """Test that changing a referred text file changes the hash."""
    path = experiment_folder / "experiment.toml"
    original = definition_hash(load_experiment_definition(path))

    (experiment_folder / "treatment.md").write_text("You will see more!\n", encoding="utf-8")
    assert definition_hash(load_experiment_definition(path)) != original


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('type = "method"', 'type = "survey"'),  # unknown stage type
        ('method = "rpm"', 'method = "rpm"\nproblme = "typo"'),  # unknown payload field
        ('arms = ["control", "treatment"]', 'arms = ["control", "control"]'),  # duplicate arms
        ('id = "anchoring"', 'id = "consent"'),  # duplicate stage ids
        ("{ treatment = ", "{ placebo = "),  # text for an unknown arm
        ("cut_points = [0.1, 0.3]", "cut_points = [0.3, 0.1]"),  # cut points not increasing
        ('body = "Neutral."', 'body = "Neutral."\nbody_file = "treatment.md"'),  # both a text and a file
    ],
)
def test_invalid_definitions_are_rejected(experiment_folder: Path, old: str, new: str):
    """Test that mistakes in the definition fail when it is loaded."""
    path = experiment_folder / "experiment.toml"
    assert old in EXPERIMENT_TOML
    path.write_text(EXPERIMENT_TOML.replace(old, new, 1), encoding="utf-8")

    with pytest.raises((ValidationError, ValueError)):
        load_experiment_definition(path)


def test_definition_round_trips_through_storage(session_and_user: dict, experiment_folder: Path):
    """Test that a stored definition is read back unchanged and with the same hash."""
    db_session: Session = session_and_user["session"]
    definition = load_experiment_definition(experiment_folder / "experiment.toml")

    snapshot = store_experiment_definition(db_session, definition)
    db_session.expire_all()
    stored = db_session.get(ExperimentDefinitionDB, snapshot.id)

    assert stored.experiment_id == "test-experiment"
    assert stored.version == "1"
    assert stored.to_definition() == definition
    assert definition_hash(stored.to_definition()) == stored.content_hash == definition_hash(definition)


def test_same_version_is_stored_once_and_conflicts_are_rejected(session_and_user: dict, experiment_folder: Path):
    """Test that storing an identical definition again returns the stored one, and changed content needs a version."""
    db_session: Session = session_and_user["session"]
    path = experiment_folder / "experiment.toml"
    snapshot = store_experiment_definition(db_session, load_experiment_definition(path))

    assert store_experiment_definition(db_session, load_experiment_definition(path)).id == snapshot.id

    path.write_text(EXPERIMENT_TOML.replace('body = "Neutral."', 'body = "Changed."'), encoding="utf-8")
    with pytest.raises(ExperimentDefinitionConflictError):
        store_experiment_definition(db_session, load_experiment_definition(path))

    path.write_text(
        EXPERIMENT_TOML.replace('body = "Neutral."', 'body = "Changed."').replace('version = "1"', 'version = "2"'),
        encoding="utf-8",
    )
    assert store_experiment_definition(db_session, load_experiment_definition(path)).id != snapshot.id


def test_subject_cannot_log_in_with_a_password(client: TestClient, session_and_user: dict):
    """Test that a subject with the unusable password hash cannot authenticate by password."""
    db_session: Session = session_and_user["session"]
    db_session.add(User(username="subject", password_hash=UNUSABLE_PASSWORD_HASH, role=UserRole.subject))
    db_session.commit()

    assert not verify_password("", UNUSABLE_PASSWORD_HASH)
    assert not verify_password("!", UNUSABLE_PASSWORD_HASH)

    for password in ("!", "subject"):
        response = client.post(
            "/login",
            data={"username": "subject", "password": password, "grant_type": "password"},
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
