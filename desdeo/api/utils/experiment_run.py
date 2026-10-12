"""Creating runs of experiments and enrolling subjects in them."""

import hashlib
import secrets

from sqlmodel import Session, select

from desdeo.api.models import (
    CommitStage,
    ExperimentDefinition,
    ExperimentDefinitionDB,
    ExperimentEnrolment,
    ExperimentRun,
    InteractiveSessionDB,
    MethodStage,
    ProblemDB,
    User,
    UserGroup,
    UserRole,
)
from desdeo.api.routers.user_authentication import UNUSABLE_PASSWORD_HASH
from desdeo.api.utils.experiment_assignment import check_policy
from desdeo.api.utils.explanation_providers import explanation_provider
from desdeo.problem import Problem


def anchor_ordinal(definition: ExperimentDefinition) -> int | None:
    """Return the position of the anchor commit stage, or None if there is no anchor."""
    return next(
        (i for i, stage in enumerate(definition.stages) if isinstance(stage, CommitStage) and stage.payload.anchor),
        None,
    )


def explained_problem_names(definition: ExperimentDefinition) -> set[str]:
    """Return the names of the problems of the method stages after the anchor, where explanations are shown."""
    after = definition.stages[anchor_ordinal(definition) + 1 :]
    return {stage.payload.problem for stage in after if isinstance(stage, MethodStage)}


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
        ValueError: the problems do not match the problem names in the definition, the policy of the definition
            bounds objectives a problem does not have or that have no ideal or nadir, or the explanation provider
            cannot explain solutions of a problem of a stage after the anchor.

    Returns:
        ExperimentRun: the created run.
    """
    expected = problem_names(snapshot)
    if set(problems) != expected:
        msg = f"The problems must be given for exactly the names {sorted(expected)}, got {sorted(problems)}."
        raise ValueError(msg)

    definition = snapshot.to_definition()
    if definition.policy is not None:
        for problem in problems.values():
            check_policy(definition.policy, Problem.from_problemdb(problem))
    if definition.explanation is not None:
        provider = explanation_provider(definition.explanation.provider, definition.explanation.options)
        for name in explained_problem_names(definition):
            provider.check(Problem.from_problemdb(problems[name]))

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


def hash_resume_token(resume_token: str) -> str:
    """Return the hash of a resume token, which is stored instead of the token."""
    return hashlib.sha256(resume_token.encode("utf-8")).hexdigest()


def enrol_subject(session: Session, run: ExperimentRun) -> tuple[ExperimentEnrolment, str]:
    """Enrol a new subject in an experiment run.

    Creates the subject, a user who cannot log in with a password, adds them to the run's group, which grants access
    to the run's problems, and creates their interactive session and enrolment record.

    Args:
        session (Session): the database session.
        run (ExperimentRun): the run.

    Returns:
        tuple[ExperimentEnrolment, str]: the enrolment, and the resume token to be stored in the subject's browser.
    """
    resume_token = secrets.token_urlsafe(32)

    subject = User(
        username=f"subject-{secrets.token_hex(8)}", password_hash=UNUSABLE_PASSWORD_HASH, role=UserRole.subject
    )
    interactive_session = InteractiveSessionDB(user=subject, info=f"Experiment run '{run.slug}'")
    run.group.members.append(subject)
    enrolment = ExperimentEnrolment(
        user=subject,
        run=run,
        experiment_id=run.definition.experiment_id,
        content_hash=run.definition.content_hash,
        session=interactive_session,
        resume_token_hash=hash_resume_token(resume_token),
    )
    session.add(enrolment)
    session.commit()

    subject.active_session_id = interactive_session.id
    session.add(subject)
    session.commit()
    session.refresh(enrolment)

    return enrolment, resume_token


def find_enrolment(session: Session, run: ExperimentRun, resume_token: str) -> ExperimentEnrolment | None:
    """Find the enrolment in a run that a resume token belongs to.

    Args:
        session (Session): the database session.
        run (ExperimentRun): the run.
        resume_token (str): the resume token stored in the subject's browser.

    Returns:
        ExperimentEnrolment | None: the enrolment, or None if the token belongs to no enrolment in the run.
    """
    return session.exec(
        select(ExperimentEnrolment).where(
            ExperimentEnrolment.run_id == run.id,
            ExperimentEnrolment.resume_token_hash == hash_resume_token(resume_token),
        )
    ).first()
