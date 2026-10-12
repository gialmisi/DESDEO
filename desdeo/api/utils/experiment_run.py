"""Creating runs of experiments."""

from sqlmodel import Session

from desdeo.api.models import ExperimentDefinitionDB, ExperimentRun, MethodStage, ProblemDB, User, UserGroup


def problem_names(snapshot: ExperimentDefinitionDB) -> set[str]:
    """Return the names of the problems the stages of a stored experiment definition refer to."""
    return {stage.payload.problem for stage in snapshot.to_definition().stages if isinstance(stage, MethodStage)}


def create_experiment_run(
    session: Session,
    snapshot: ExperimentDefinitionDB,
    slug: str,
    label: str,
    problems: dict[str, ProblemDB],
    *,
    owner: User | None = None,
) -> ExperimentRun:
    """Create a run of an experiment, with a new group that grants access to the run's problems.

    Args:
        session (Session): the database session.
        snapshot (ExperimentDefinitionDB): the stored definition of the experiment.
        slug (str): the identifier of the run in its campaign link.
        label (str): a human-readable label of the run, e.g., 'pilot' or 'live'.
        problems (dict[str, ProblemDB]): the stored problem of each problem name in the definition.
        owner (User | None, optional): the owner of the run's group, e.g., the analyst running the experiment.
            Defaults to None.

    Raises:
        ValueError: the problems do not match the problem names in the definition.

    Returns:
        ExperimentRun: the created run.
    """
    expected = problem_names(snapshot)
    if set(problems) != expected:
        msg = f"The problems must be given for exactly the names {sorted(expected)}, got {sorted(problems)}."
        raise ValueError(msg)

    group = UserGroup(
        name=f"experiment-run:{slug}",
        description=f"The subjects of the {label} run of experiment '{snapshot.experiment_id}'.",
        owner_id=owner.id if owner is not None else None,
        problems=list({problem.id: problem for problem in problems.values()}.values()),
    )
    run = ExperimentRun(
        slug=slug,
        label=label,
        definition=snapshot,
        group=group,
        problems={name: problem.id for name, problem in problems.items()},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    return run
