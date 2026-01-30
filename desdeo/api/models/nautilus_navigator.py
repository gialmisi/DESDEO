"""Models specific to the NAUTILUS Navigator method."""

from sqlmodel import Field, SQLModel


class NautilusNavigatorInitRequest(SQLModel):
    """Request to initialize the NAUTILUS Navigator."""

    problem_id: int
    session_id: int | None = Field(default=None)
    parent_state_id: int | None = Field(default=None)
    total_steps: int = Field(default=100, description="Total steps for the whole navigation horizon.")


class NautilusNavigatorRecomputeRequest(SQLModel):
    """Request to recompute a NAUTILUS Navigator segment."""

    problem_id: int
    session_id: int | None = Field(default=None)
    parent_state_id: int | None = Field(default=None)

    go_back_step: int = Field(default=0, description="Step index to rewind to within the active path.")
    steps: int = Field(default=20, description="How many steps to compute forward from go_back_step.")

    reference_point: dict[str, float] = Field(description="Aspiration levels per objective symbol.")
    bounds: dict[str, float | None] | None = Field(
        default=None, description="Optional bounds per objective symbol."
    )


class NautilusNavigatorSegmentResponse(SQLModel):
    """Response model for an initialized or recomputed NAUTILUS Navigator segment."""

    state_id: int | None = Field(description="StateDB id created for this action.")
    session_id: int | None = None

    total_steps: int
    segment_start_step: int
    segment_steps: int
    objective_symbols: list[str]

    lower_bounds: dict[str, list[float]]
    upper_bounds: dict[str, list[float]]
    navigation_points: dict[str, list[float]] | None = None
    distance: list[float] | None = None
