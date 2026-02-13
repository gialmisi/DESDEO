"""Request and response models for the Scenario RPM method."""

from sqlmodel import JSON, Column, Field, SQLModel

from .preference import ReferencePoint


class ScenarioRPMSolveRequest(SQLModel):
    """Request to solve one scenario with the reference point method."""

    problem_id: int
    session_id: int | None = Field(default=None)
    parent_state_id: int | None = Field(default=None)

    scenario_key: str = Field(description="Which scenario to solve (e.g. 'supervisor', 'owner_1').")
    preference: ReferencePoint = Field(Column(JSON))
    scalarization_options: dict[str, float | str | bool] | None = Field(sa_column=Column(JSON), default=None)
    solver: str | None = Field(default=None)
    solver_options: dict[str, float | str | bool] | None = Field(sa_column=Column(JSON), default=None)


class ScenarioComparisonRequest(SQLModel):
    """Request to compare a supervisor's solution against all owners at once."""

    problem_id: int
    supervisor_state_id: int
    owner_states: dict[str, int] = Field(
        sa_column=Column(JSON),
        description="Mapping of owner scenario key to state ID, e.g. {'owner_1': 5, 'owner_2': 7}.",
    )


class ClusterComparison(SQLModel):
    """Cluster-level comparison between supervisor and one owner."""

    cluster_key: str
    supervisor_objectives: dict[str, float] = Field(sa_column=Column(JSON))
    owner_objectives: dict[str, float] = Field(sa_column=Column(JSON))
    distance: float


class ScenarioComparisonResponse(SQLModel):
    """Full comparison result across all owner clusters."""

    clusters: list[ClusterComparison] = Field(sa_column=Column(JSON))
    max_distance: float
