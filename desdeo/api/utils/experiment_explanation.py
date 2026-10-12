"""Explaining the solutions subjects find after the anchor, in the arms shown an explanation.

The explanation is computed by the provider named in the experiment definition (see
`desdeo.api.utils.explanation_providers`), and stored with the content hash of its payload, once per solution.
Subjects of other arms, and every subject before the anchor, get no explanation, and cannot tell from the response
which arm would.
"""

from fastapi import HTTPException, status
from sqlmodel import Session, select

from desdeo.api.models import (
    ExperimentEnrolment,
    ExperimentExplanation,
    ExperimentExplanationRequest,
    MethodStage,
)
from desdeo.api.utils.experiment_assignment import subject_solution
from desdeo.api.utils.experiment_definition import content_hash
from desdeo.api.utils.experiment_flow import current_progress, enrolment_definition
from desdeo.api.utils.experiment_run import anchor_ordinal
from desdeo.api.utils.explanation_providers import explanation_provider
from desdeo.problem import Problem


def explain_solution(
    session: Session, enrolment: ExperimentEnrolment, request: ExperimentExplanationRequest
) -> ExperimentExplanation:
    """Return the explanation of a solution the subject has found in the current method stage, computing it once.

    Args:
        session (Session): the database session.
        enrolment (ExperimentEnrolment): the subject's enrolment.
        request (ExperimentExplanationRequest): the solution to explain, and the current stage.

    Raises:
        HTTPException: 409 if the stage is not current; 404 if the subject is not shown explanations in the stage, i.e.,
            the stage is not a method stage, the subject has not passed the anchor, or the subject's arm is not shown
            the explanation; 422 if the solution is not one the subject has found in the stage.

    Returns:
        ExperimentExplanation: the stored explanation.
    """
    definition = enrolment_definition(session, enrolment)
    current_progress(session, enrolment, request.stage_ordinal)
    stage = definition.stages[request.stage_ordinal]

    config = definition.explanation
    if (
        config is None
        or enrolment.arm not in config.arms
        or not isinstance(stage, MethodStage)
        or request.stage_ordinal < anchor_ordinal(definition)
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="There is no explanation.")

    problem_db, solution = subject_solution(
        session,
        enrolment,
        request.state_id,
        request.solution_index,
        problem_id=enrolment.run.problems[stage.payload.problem],
    )

    stored = session.exec(
        select(ExperimentExplanation).where(
            ExperimentExplanation.enrolment_id == enrolment.id,
            ExperimentExplanation.state_id == request.state_id,
            ExperimentExplanation.solution_index == request.solution_index,
        )
    ).first()
    if stored is not None:
        return stored

    provider = explanation_provider(config.provider, config.options)
    payload = provider.explain(Problem.from_problemdb(problem_db), solution)
    explanation = ExperimentExplanation(
        enrolment_id=enrolment.id,
        stage_ordinal=request.stage_ordinal,
        state_id=request.state_id,
        solution_index=request.solution_index,
        provider=config.provider,
        content_hash=content_hash(payload),
        payload=payload,
        client_timestamp=request.client_timestamp,
    )
    session.add(explanation)
    session.commit()
    session.refresh(explanation)

    return explanation
