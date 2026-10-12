"""The flow of subjects through the stages of an experiment.

The server owns the flow: a subject is always in exactly one stage, enters the stages in order, and leaves a stage
only when its required data is stored. Leaving the anchor commit assigns the subject's arm. Texts that differ per arm
are resolved for the subject's arm here, so the arm itself is never sent to the subject.
"""

from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlmodel import Session, select

from desdeo.api.models import (
    STAGE_DATA_MODELS,
    CommitStage,
    CommitStageData,
    ExperimentDefinition,
    ExperimentDefinitionDB,
    ExperimentEnrolment,
    ExperimentStageProgress,
    ExperimentStageResponse,
    TextStage,
)
from desdeo.api.utils.experiment_assignment import assign_arm, committed_solution
from desdeo.problem import Problem


def enrolment_definition(session: Session, enrolment: ExperimentEnrolment) -> ExperimentDefinition:
    """Return the experiment definition the subject was enrolled with."""
    snapshot = session.exec(
        select(ExperimentDefinitionDB).where(
            ExperimentDefinitionDB.experiment_id == enrolment.experiment_id,
            ExperimentDefinitionDB.content_hash == enrolment.content_hash,
        )
    ).one()
    return snapshot.to_definition()


def stage_response(
    definition: ExperimentDefinition, enrolment: ExperimentEnrolment, progress: ExperimentStageProgress
) -> ExperimentStageResponse:
    """Return a stage as shown to the subject, with its text resolved for the subject's arm."""
    stage = definition.stages[progress.stage_ordinal]
    payload = stage.payload
    if isinstance(stage, TextStage):
        body = payload.arm_bodies.get(enrolment.arm, payload.body) if enrolment.arm is not None else payload.body
        payload = payload.model_copy(update={"body": body, "arm_bodies": {}})

    return ExperimentStageResponse(
        ordinal=progress.stage_ordinal,
        id=stage.id,
        type=stage.type,
        required=stage.required,
        payload=payload,
        data=progress.data,
        is_last=progress.stage_ordinal == len(definition.stages) - 1,
    )


def latest_progress(session: Session, enrolment: ExperimentEnrolment) -> ExperimentStageProgress | None:
    """Return the progress of the last stage the subject entered, or None if the subject has entered no stage."""
    return session.exec(
        select(ExperimentStageProgress)
        .where(ExperimentStageProgress.enrolment_id == enrolment.id)
        .order_by(ExperimentStageProgress.stage_ordinal.desc())
    ).first()


def _enter(
    session: Session, definition: ExperimentDefinition, enrolment: ExperimentEnrolment, ordinal: int, at: str
) -> ExperimentStageProgress:
    """Enter a stage."""
    progress = ExperimentStageProgress(
        enrolment_id=enrolment.id, stage_ordinal=ordinal, stage_id=definition.stages[ordinal].id, entered_at=at
    )
    session.add(progress)
    session.commit()
    session.refresh(progress)
    return progress


def _current(session: Session, enrolment: ExperimentEnrolment, stage_ordinal: int) -> ExperimentStageProgress:
    """Return the progress of the current stage, checking that it is the stage the request is about."""
    progress = latest_progress(session, enrolment)
    if progress is None or progress.exited_at is not None or progress.stage_ordinal != stage_ordinal:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Stage {stage_ordinal} is not current.")
    return progress


def enter_current_stage(
    session: Session, enrolment: ExperimentEnrolment, client_timestamp: str
) -> ExperimentStageResponse | None:
    """Return the subject's current stage, entering the first stage if the subject has entered none.

    Args:
        session (Session): the database session.
        enrolment (ExperimentEnrolment): the subject's enrolment.
        client_timestamp (str): the client timestamp, recorded if the first stage is entered.

    Returns:
        ExperimentStageResponse | None: the current stage, or None if the subject has finished the experiment.
    """
    definition = enrolment_definition(session, enrolment)
    progress = latest_progress(session, enrolment)

    if progress is None:
        progress = _enter(session, definition, enrolment, 0, client_timestamp)
    elif progress.exited_at is not None:
        return None  # the last stage has been left

    return stage_response(definition, enrolment, progress)


def store_stage_data(
    session: Session, enrolment: ExperimentEnrolment, stage_ordinal: int, data: dict, client_timestamp: str
) -> ExperimentStageResponse:
    """Store the data of the current stage, replacing any data stored before.

    Args:
        session (Session): the database session.
        enrolment (ExperimentEnrolment): the subject's enrolment.
        stage_ordinal (int): the position of the stage the data is for, which must be the current stage.
        data (dict): the data, validated against the model of the stage's type.
        client_timestamp (str): the client timestamp of submitting the data.

    Raises:
        HTTPException: 409 if the stage is not current, 422 if the data is invalid for the stage.

    Returns:
        ExperimentStageResponse: the stage with the stored data.
    """
    definition = enrolment_definition(session, enrolment)
    progress = _current(session, enrolment, stage_ordinal)
    stage = definition.stages[stage_ordinal]

    try:
        validated = STAGE_DATA_MODELS[stage.type].model_validate(data)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors()) from e

    if stage.type == "commit":
        committed_solution(session, enrolment, validated)

    progress.data = validated.model_dump()
    progress.data_submitted_at = client_timestamp
    session.add(progress)
    session.commit()
    session.refresh(progress)

    return stage_response(definition, enrolment, progress)


def advance(
    session: Session, enrolment: ExperimentEnrolment, stage_ordinal: int, client_timestamp: str
) -> ExperimentStageResponse | None:
    """Leave the current stage and enter the next one.

    Args:
        session (Session): the database session.
        enrolment (ExperimentEnrolment): the subject's enrolment.
        stage_ordinal (int): the position of the stage to leave, which must be the current stage.
        client_timestamp (str): the client timestamp, recorded as the exit of the stage and the entry of the next.

    Raises:
        HTTPException: 409 if the stage is not current or is required and has no data.

    Returns:
        ExperimentStageResponse | None: the next stage, or None if the left stage was the last one.
    """
    definition = enrolment_definition(session, enrolment)
    progress = _current(session, enrolment, stage_ordinal)

    if definition.stages[stage_ordinal].required and progress.data is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"Stage {stage_ordinal} is required and has no data."
        )

    stage = definition.stages[stage_ordinal]
    if isinstance(stage, CommitStage) and stage.payload.anchor and enrolment.arm is None:
        problem_db, objectives = committed_solution(session, enrolment, CommitStageData.model_validate(progress.data))
        assign_arm(session, enrolment, definition, Problem.from_problemdb(problem_db), objectives)

    progress.exited_at = client_timestamp
    session.add(progress)
    session.commit()

    if stage_ordinal == len(definition.stages) - 1:
        return None

    return stage_response(
        definition, enrolment, _enter(session, definition, enrolment, stage_ordinal + 1, client_timestamp)
    )


def consent_record(session: Session, enrolment: ExperimentEnrolment) -> str:
    """Return a record of the subject's consent: the consent text and the time of agreement, without any identifier.

    Args:
        session (Session): the database session.
        enrolment (ExperimentEnrolment): the subject's enrolment.

    Raises:
        HTTPException: 404 if the subject has not consented.

    Returns:
        str: the record, in Markdown.
    """
    definition = enrolment_definition(session, enrolment)
    progress = session.exec(
        select(ExperimentStageProgress).where(
            ExperimentStageProgress.enrolment_id == enrolment.id, ExperimentStageProgress.stage_ordinal == 0
        )
    ).first()
    if progress is None or progress.data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No consent has been recorded.")

    consent = definition.stages[0].payload
    return (
        f"# {consent.title}\n\n{consent.body.rstrip()}\n\n---\n\n"
        f"Consent given on {progress.data_submitted_at} to the study '{definition.name}'.\n"
    )
