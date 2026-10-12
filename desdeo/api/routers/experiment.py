"""Defines end-points of the experiment harness, through which subjects take part in experiments."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from desdeo.api.db import get_session
from desdeo.api.models import ExperimentEnrolRequest, ExperimentEnrolResponse, ExperimentRun
from desdeo.api.routers.user_authentication import create_jwt_token
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
