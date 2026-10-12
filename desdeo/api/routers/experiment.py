"""Defines end-points of the experiment harness, through which subjects take part in experiments."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlmodel import Session, select

from desdeo.api.db import get_session
from desdeo.api.models import (
    ExperimentAdvanceRequest,
    ExperimentEnrolment,
    ExperimentEnrolRequest,
    ExperimentEnrolResponse,
    ExperimentEnterRequest,
    ExperimentEvent,
    ExperimentEventIn,
    ExperimentFlowResponse,
    ExperimentRun,
    ExperimentStageDataRequest,
    ExperimentStageResponse,
    User,
)
from desdeo.api.routers.user_authentication import create_jwt_token, get_current_user
from desdeo.api.utils.experiment_flow import advance, consent_record, enter_current_stage, store_stage_data
from desdeo.api.utils.experiment_run import enrol_subject, find_enrolment

router = APIRouter(prefix="/experiment", tags=["experiment"])

# Subjects never log in or refresh their tokens, so an access token must outlast a whole experiment session.
SUBJECT_TOKEN_LIFETIME = timedelta(hours=4)


@router.post("/{slug}/enrol")
def enrol(
    slug: str,
    request: ExperimentEnrolRequest,
    session: Annotated[Session, Depends(get_session)],
) -> ExperimentEnrolResponse:
    """Enrol a new subject in an experiment run, or resume an earlier enrolment.

    This is the end-point the campaign link of a run opens; it needs no credentials. A resume token of an enrolment
    in the run returns a new access token for the same subject. Otherwise a new subject is enrolled, if the run is
    open.

    Args:
        slug (str): the identifier of the run in its campaign link.
        request (ExperimentEnrolRequest): the resume token stored in the browser, if any.
        session (Annotated[Session, Depends(get_session)]): the database session.

    Raises:
        HTTPException: 404 if there is no such run, 403 if the subject has finished the experiment or the run is closed
            to new subjects.

    Returns:
        ExperimentEnrolResponse: the subject's access token and resume token.
    """
    run = session.exec(select(ExperimentRun).where(ExperimentRun.slug == slug)).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No experiment run '{slug}'.")

    enrolment = find_enrolment(session, run, request.resume_token) if request.resume_token is not None else None

    if enrolment is not None:
        if enrolment.token_revoked:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="The experiment has been finished.")
        resume_token, resumed = request.resume_token, True
    else:
        if not run.is_open:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="The experiment run is closed.")
        enrolment, resume_token = enrol_subject(session, run)
        resumed = False

    access_token = create_jwt_token({"id": enrolment.user.id, "sub": enrolment.user.username}, SUBJECT_TOKEN_LIFETIME)

    return ExperimentEnrolResponse(access_token=access_token, resume_token=resume_token, resumed=resumed)


def get_enrolment(
    user: Annotated[User, Depends(get_current_user)], session: Annotated[Session, Depends(get_session)]
) -> ExperimentEnrolment:
    """Return the enrolment of the current user, who must be an enrolled subject whose tokens are not revoked."""
    enrolment = session.exec(select(ExperimentEnrolment).where(ExperimentEnrolment.user_id == user.id)).first()
    if enrolment is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enrolled in an experiment.")
    if enrolment.token_revoked:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="The experiment has been finished.")
    return enrolment


@router.post("/stage/enter")
def enter_stage(
    request: ExperimentEnterRequest,
    enrolment: Annotated[ExperimentEnrolment, Depends(get_enrolment)],
    session: Annotated[Session, Depends(get_session)],
) -> ExperimentFlowResponse:
    """Return the subject's current stage, entering the first stage on the first call.

    Calling this again, e.g., after a page reload, returns the same stage without entering it again.
    """
    stage = enter_current_stage(session, enrolment, request.client_timestamp)
    return ExperimentFlowResponse(stage=stage, finished=stage is None)


@router.post("/stage/data")
def post_stage_data(
    request: ExperimentStageDataRequest,
    enrolment: Annotated[ExperimentEnrolment, Depends(get_enrolment)],
    session: Annotated[Session, Depends(get_session)],
) -> ExperimentStageResponse:
    """Store the data of the current stage, replacing any data stored for it before."""
    return store_stage_data(session, enrolment, request.stage_ordinal, request.data, request.client_timestamp)


@router.post("/stage/advance")
def advance_stage(
    request: ExperimentAdvanceRequest,
    enrolment: Annotated[ExperimentEnrolment, Depends(get_enrolment)],
    session: Annotated[Session, Depends(get_session)],
) -> ExperimentFlowResponse:
    """Leave the current stage, if its required data is stored, and enter the next one."""
    stage = advance(session, enrolment, request.stage_ordinal, request.client_timestamp)
    return ExperimentFlowResponse(stage=stage, finished=stage is None)


@router.get("/consent", response_class=PlainTextResponse)
def download_consent(
    enrolment: Annotated[ExperimentEnrolment, Depends(get_enrolment)],
    session: Annotated[Session, Depends(get_session)],
) -> PlainTextResponse:
    """Download a record of the subject's consent. It contains no identifier of the subject."""
    return PlainTextResponse(
        consent_record(session, enrolment),
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="consent.md"'},
    )


@router.post("/events")
def post_events(
    events: list[ExperimentEventIn],
    enrolment: Annotated[ExperimentEnrolment, Depends(get_enrolment)],
    session: Annotated[Session, Depends(get_session)],
) -> int:
    """Store a batch of events, with their client timestamps as received. Returns the number of events stored."""
    session.add_all(ExperimentEvent(enrolment_id=enrolment.id, **event.model_dump(mode="json")) for event in events)
    session.commit()
    return len(events)
