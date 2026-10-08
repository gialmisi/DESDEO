"""Loading, hashing, and storing experiment definitions.

An experiment definition is a TOML file. Any text field can instead be given as a file next to the TOML file by
appending `_file` to its key, e.g., `body_file = "consent.md"` instead of `body = "..."`, and a table of texts, e.g.,
`arm_bodies_file = { control = "control.md" }`. The files are read when the definition is loaded, so their contents
are part of the definition and its hash.
"""

import hashlib
import json
import tomllib
from pathlib import Path

from sqlmodel import Session, select

from desdeo.api.models.experiment import ExperimentDefinition, ExperimentDefinitionDB

FILE_SUFFIX = "_file"


class ExperimentDefinitionConflictError(ValueError):
    """Raised when a definition is stored with the identifier and version of a stored one, but different content."""


def _inline_files(data: dict | list, folder: Path) -> dict | list:
    """Replace the `<key>_file` entries in the data with `<key>` entries holding the contents of the files."""
    if isinstance(data, list):
        return [_inline_files(item, folder) if isinstance(item, dict | list) else item for item in data]

    inlined = {}
    for key, value in data.items():
        if not key.endswith(FILE_SUFFIX):
            inlined[key] = _inline_files(value, folder) if isinstance(value, dict | list) else value
            continue

        target = key.removesuffix(FILE_SUFFIX)
        if target in data:
            msg = f"Both '{target}' and '{key}' are given; give only one of them."
            raise ValueError(msg)
        if isinstance(value, dict):
            inlined[target] = {name: (folder / path).read_text(encoding="utf-8") for name, path in value.items()}
        else:
            inlined[target] = (folder / value).read_text(encoding="utf-8")

    return inlined


def load_experiment_definition(path: str | Path) -> ExperimentDefinition:
    """Load and validate an experiment definition from a TOML file.

    Args:
        path (str | Path): the path of the TOML file. Files referred to with `_file` keys are relative to its folder.

    Returns:
        ExperimentDefinition: the validated definition, with the referred files inlined.
    """
    path = Path(path)
    with path.open("rb") as file:
        data = tomllib.load(file)

    return ExperimentDefinition.model_validate(_inline_files(data, path.parent))


def definition_hash(definition: ExperimentDefinition) -> str:
    """Compute the content hash of an experiment definition.

    The hash is computed over the canonical JSON of the validated definition, so it does not depend on how the TOML
    file is formatted, but changes when any value or text changes.

    Args:
        definition (ExperimentDefinition): the definition.

    Returns:
        str: the SHA-256 hash, as a hexadecimal string.
    """
    canonical = json.dumps(
        definition.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_experiment_definition(session: Session, definition: ExperimentDefinition) -> ExperimentDefinitionDB:
    """Store a snapshot of an experiment definition, or return the stored one if it is identical.

    Args:
        session (Session): the database session.
        definition (ExperimentDefinition): the definition.

    Raises:
        ExperimentDefinitionConflictError: a definition with the same identifier and version but different content
            is already stored. Its version must be changed.

    Returns:
        ExperimentDefinitionDB: the stored snapshot.
    """
    content_hash = definition_hash(definition)
    stored = session.exec(
        select(ExperimentDefinitionDB).where(
            ExperimentDefinitionDB.experiment_id == definition.id, ExperimentDefinitionDB.version == definition.version
        )
    ).first()

    if stored is not None:
        if stored.content_hash != content_hash:
            msg = (
                f"Experiment '{definition.id}' version '{definition.version}' is already stored with different "
                "content. Change the version of the definition."
            )
            raise ExperimentDefinitionConflictError(msg)
        return stored

    snapshot = ExperimentDefinitionDB(
        experiment_id=definition.id,
        version=definition.version,
        content_hash=content_hash,
        definition=definition.model_dump(mode="json"),
    )
    session.add(snapshot)
    session.commit()
    session.refresh(snapshot)

    return snapshot
