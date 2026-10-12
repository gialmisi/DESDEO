"""Defines the models of experiments run with the experiment harness.

An experiment definition is authored as a TOML file (see `desdeo.api.utils.experiment_definition`), validated with the
models below, and stored as a snapshot together with a hash of its content. The hash ties collected data to the exact
definition it was collected with.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlmodel import JSON, Column, Relationship, SQLModel, UniqueConstraint
from sqlmodel import Field as SQLField

if TYPE_CHECKING:
    from .session import InteractiveSessionDB
    from .user import User
    from .user_group import UserGroup


class TextPayload(BaseModel):
    """The payload of a text stage, e.g., consent, transition, or debrief."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="The title shown on the stage.")
    body: str = Field(description="The text of the stage, in Markdown.")
    arm_bodies: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Texts in Markdown shown instead of `body` to subjects in the given arms, keyed by arm label. "
            "Used for stages that differ per arm, e.g., the transition."
        ),
    )


class MethodPayload(BaseModel):
    """The payload of a method stage, where the subject iterates with an interactive method."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["rpm"] = Field(description="The interactive method used in the stage.")
    problem: str = Field(
        description="The name of the problem solved in the stage. Runs of the experiment map names to stored problems."
    )
    instructions: str | None = Field(default=None, description="Instructions shown on the stage, in Markdown.")


class CommitPayload(BaseModel):
    """The payload of a commit stage, where the subject commits to a solution of the preceding method stage."""

    model_config = ConfigDict(extra="forbid")

    confirmation: str = Field(description="The text asking the subject to confirm the commitment, in Markdown.")
    anchor: bool = Field(
        default=False, description="Whether the committed solution is the anchor, at which arms are assigned."
    )


class StageBase(BaseModel):
    """The fields common to all stages."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="An identifier of the stage, unique within the experiment.")
    required: bool = Field(
        default=True, description="Whether the stage must have stored data for the subject to count as complete."
    )


class TextStage(StageBase):
    """A stage showing text."""

    type: Literal["text"]
    payload: TextPayload


class MethodStage(StageBase):
    """A stage with an interactive method."""

    type: Literal["method"]
    payload: MethodPayload


class CommitStage(StageBase):
    """A stage committing to a solution."""

    type: Literal["commit"]
    payload: CommitPayload


Stage = Annotated[TextStage | MethodStage | CommitStage, Field(discriminator="type")]


class Stratification(BaseModel):
    """How subjects are stratified when arms are assigned."""

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(description="The name of the quantity the strata are cut from.")
    cut_points: list[float] = Field(
        description="Strictly increasing cut points; `n` cut points give `n + 1` strata.", min_length=1
    )

    @model_validator(mode="after")
    def check_increasing(self) -> "Stratification":
        """Check that the cut points are strictly increasing."""
        if any(a >= b for a, b in zip(self.cut_points, self.cut_points[1:], strict=False)):
            msg = f"The cut points must be strictly increasing, got {self.cut_points}."
            raise ValueError(msg)
        return self


class ExperimentDefinition(BaseModel):
    """A validated experiment definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="The identifier of the experiment.")
    name: str = Field(description="A human-readable name of the experiment.")
    version: str = Field(description="The version of the definition. Content changes require a new version.")
    arms: list[str] = Field(description="The labels of the arms.", min_length=1)
    stratification: Stratification
    stages: list[Stage] = Field(description="The stages in the order subjects pass through them.", min_length=1)

    @model_validator(mode="after")
    def check_consistency(self) -> "ExperimentDefinition":
        """Check that labels and identifiers are unique and that stages refer to existing arms."""
        if len(set(self.arms)) != len(self.arms):
            msg = f"The arm labels must be unique, got {self.arms}."
            raise ValueError(msg)

        stage_ids = [stage.id for stage in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            msg = f"The stage identifiers must be unique, got {stage_ids}."
            raise ValueError(msg)

        for stage in self.stages:
            if isinstance(stage, TextStage) and (unknown := set(stage.payload.arm_bodies) - set(self.arms)):
                msg = f"Stage '{stage.id}' has texts for unknown arms {sorted(unknown)}."
                raise ValueError(msg)

        anchors = [stage.id for stage in self.stages if isinstance(stage, CommitStage) and stage.payload.anchor]
        if len(anchors) > 1:
            msg = f"At most one commit stage can be the anchor, got {anchors}."
            raise ValueError(msg)

        return self


class ExperimentDefinitionDB(SQLModel, table=True):
    """The table model of a stored snapshot of an experiment definition."""

    __table_args__ = (UniqueConstraint("experiment_id", "version"),)

    id: int | None = SQLField(primary_key=True, default=None)
    experiment_id: str = SQLField(index=True)
    version: str = SQLField()
    content_hash: str = SQLField(index=True)
    definition: dict = SQLField(sa_column=Column(JSON), description="The validated definition, as JSON.")
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))

    def to_definition(self) -> ExperimentDefinition:
        """Return the stored definition as a validated model."""
        return ExperimentDefinition.model_validate(self.definition)


class ExperimentRun(SQLModel, table=True):
    """The table model of a run of an experiment, e.g., its pilot or its live data collection.

    Subjects enrolled in a run are members of its group, which grants them access to the run's problems. Runs are
    kept apart, so that the data of one run is exactly the data of the subjects in its group.
    """

    id: int | None = SQLField(primary_key=True, default=None)
    slug: str = SQLField(index=True, unique=True, description="The identifier of the run in its campaign link.")
    label: str = SQLField(description="A human-readable label of the run, e.g., 'pilot' or 'live'.")
    definition_id: int = SQLField(foreign_key="experimentdefinitiondb.id")
    group_id: int = SQLField(foreign_key="usergroup.id")
    problems: dict[str, int] = SQLField(
        sa_column=Column(JSON), default_factory=dict, description="The stored problem of each problem name."
    )
    is_open: bool = SQLField(default=True, description="Whether the campaign link still enrols new subjects.")
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))

    definition: ExperimentDefinitionDB = Relationship()
    group: "UserGroup" = Relationship()


class ExperimentEnrolment(SQLModel, table=True):
    """The table model of the enrolment of a subject in an experiment run.

    The assignment fields are null until the subject commits to the anchor. The interactive session holds the
    subject's method states; it is created at enrolment, not at the first solve.
    """

    id: int | None = SQLField(primary_key=True, default=None)
    user_id: int = SQLField(foreign_key="user.id", unique=True)
    run_id: int = SQLField(foreign_key="experimentrun.id", index=True)
    experiment_id: str = SQLField(description="The identifier of the experiment, at enrolment.")
    content_hash: str = SQLField(description="The content hash of the experiment definition, at enrolment.")
    session_id: int = SQLField(foreign_key="interactivesessiondb.id")
    resume_token_hash: str = SQLField(
        index=True, unique=True, description="The SHA-256 hash of the resume token stored in the browser."
    )
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))

    arm: str | None = SQLField(default=None, description="The assigned arm.")
    stratum: int | None = SQLField(default=None, description="The stratum, counted from zero.")
    stratification_value: float | None = SQLField(
        default=None, description="The value of the stratification variable the stratum was cut from."
    )
    is_binding: bool | None = SQLField(
        default=None, description="Whether the compliant region binds the anchor. Computed in every arm."
    )
    assignment_details: dict | None = SQLField(
        sa_column=Column(JSON), default=None, description="How the arm was assigned, e.g., the counts balanced on."
    )
    assigned_at: datetime | None = SQLField(default=None)

    token_revoked: bool = SQLField(default=False, description="Whether the subject's tokens have been revoked.")

    user: "User" = Relationship()
    run: ExperimentRun = Relationship()
    session: "InteractiveSessionDB" = Relationship()
