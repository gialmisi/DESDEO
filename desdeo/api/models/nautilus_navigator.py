"""Models specific to the NAUTILUS Navigator method."""

from sqlmodel import Field, SQLModel


class NautilusNavigatorInitializeRequest(SQLModel):
    """Request model for initializing the NAUTILUS Navigator method."""

    problem_id: int
    session_id: int | None = Field(default=None)
    parent_state_id: int | None = Field(default=None)
    total_steps: int = Field(
        default=100,
        description="The total number of steps in the NAUTILUS Navigator.",
    )


class NautilusNavigatorInitializeResponse(SQLModel):
    """Response model for the NAUTILUS Navigator initialization."""

    state_id: int | None = Field(description="The id of the state created by the initialization.")
    total_steps: int = Field(description="The total number of steps in the NAUTILUS Navigator.")
    steps_remaining: int = Field(description="The number of steps remaining.")
    step_number: int = Field(description="The current step number.")
    distance_to_front: float = Field(description="The distance to the Pareto front as a percentage.")
    navigation_point: dict[str, float] = Field(description="The current navigation point.")
    reachable_solution: dict[str, float] | None = Field(description="The reachable solution found in this step.")
    reachable_bounds: dict[str, dict[str, float]] = Field(description="The reachable bounds for each objective.")
    reference_point: dict[str, float] | None = Field(description="The reference point used in the step.")
    bounds: dict[str, float] | None = Field(description="The user provided bounds.")


class NautilusNavigatorNavigateRequest(SQLModel):
    """Request model for navigating the NAUTILUS Navigator method."""

    problem_id: int
    session_id: int | None = Field(default=None)
    parent_state_id: int | None = Field(default=None)
    total_steps: int = Field(description="The total number of steps in the NAUTILUS Navigator.")
    go_back_step: int = Field(description="The step number to go back to before navigating forward.")
    steps_remaining: int = Field(description="The number of steps remaining after the go-back step.")
    reference_point: dict[str, float] = Field(description="The preference (reference point) of the DM.")
    bounds: dict[str, float] | None = Field(
        default=None, description="Optional bounds preference of the DM."
    )


class NautilusNavigatorNavigateResponse(SQLModel):
    """Response model for navigating the NAUTILUS Navigator method."""

    objective_symbols: list[str] = Field(description="The symbols of the objectives.")
    objective_long_names: list[str] = Field(description="Long/descriptive names of the objectives.")
    units: list[str] | None = Field(description="The units of the objectives.")
    is_maximized: list[bool] = Field(description="Whether the objectives are to be maximized or minimized.")
    ideal: list[float] = Field(description="The ideal values of the objectives.")
    nadir: list[float] = Field(description="The nadir values of the objectives.")
    total_steps: int = Field(description="The total number of steps in the NAUTILUS Navigator.")
    current_step: int = Field(description="The current step number.")
    step_numbers: list[int] = Field(description="The step numbers along the active path.")
    state_ids: list[int] = Field(description="The state ids along the active path.")
    distance_to_front: list[float] = Field(description="The distance to the Pareto front at each step.")
    navigation_points: dict[str, list[float]] = Field(description="The navigation points along the path.")
    lower_bounds: dict[str, list[float]] = Field(description="Lower bounds of the reachable region.")
    upper_bounds: dict[str, list[float]] = Field(description="Upper bounds of the reachable region.")
    preferences: dict[str, list[float]] = Field(description="The preferences used in each step.")
    bounds: dict[str, list[float]] = Field(description="The bounds preference of the DM at each step.")
    reachable_solution: dict[str, float] | None = Field(description="The solution reached at the end of navigation.")
