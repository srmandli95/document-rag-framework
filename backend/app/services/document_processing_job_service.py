from datetime import datetime

from sqlalchemy.orm import Session

from app.models.document_processing_job import DocumentProcessingJob


def _save_job(db: Session, job: DocumentProcessingJob) -> DocumentProcessingJob:
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def create_processing_job(
    db: Session,
    document_id: str,
    user_id: str,
    force: bool = False,
) -> DocumentProcessingJob:
    job = DocumentProcessingJob(
        document_id=document_id,
        user_id=user_id,
        force=force,
        status="pending",
        steps=[],
    )
    return _save_job(db, job)


def mark_job_running(
    db: Session,
    job: DocumentProcessingJob,
    current_step: str | None = None,
) -> DocumentProcessingJob:
    job.status = "running"
    job.current_step = current_step
    job.started_at = job.started_at or datetime.utcnow()
    return _save_job(db, job)


def update_job_step(
    db: Session,
    job: DocumentProcessingJob,
    step: dict,
    current_step: str | None = None,
) -> DocumentProcessingJob:
    job.steps = [*(job.steps or []), step]
    job.current_step = current_step
    return _save_job(db, job)


def mark_job_completed(
    db: Session,
    job: DocumentProcessingJob,
    steps: list[dict],
) -> DocumentProcessingJob:
    job.status = "completed"
    job.steps = list(steps)
    job.current_step = None
    job.error_message = None
    job.completed_at = datetime.utcnow()
    return _save_job(db, job)


def mark_job_failed(
    db: Session,
    job: DocumentProcessingJob,
    steps: list[dict],
    error_message: str,
) -> DocumentProcessingJob:
    job.status = "failed"
    job.steps = list(steps)
    job.current_step = None
    job.error_message = error_message
    job.completed_at = datetime.utcnow()
    return _save_job(db, job)


def mark_job_skipped(
    db: Session,
    job: DocumentProcessingJob,
    steps: list[dict],
    message: str,
) -> DocumentProcessingJob:
    job.status = "skipped"
    job.steps = list(steps)
    job.current_step = None
    job.error_message = None
    job.completed_at = datetime.utcnow()
    return _save_job(db, job)


def get_processing_jobs_by_document(
    db: Session,
    document_id: str,
    user_id: str,
) -> list[DocumentProcessingJob]:
    return (
        db.query(DocumentProcessingJob)
        .filter(
            DocumentProcessingJob.document_id == document_id,
            DocumentProcessingJob.user_id == user_id,
        )
        .order_by(DocumentProcessingJob.created_at.desc())
        .all()
    )


def get_processing_job_by_id(
    db: Session,
    job_id: str,
    user_id: str,
) -> DocumentProcessingJob | None:
    return (
        db.query(DocumentProcessingJob)
        .filter(
            DocumentProcessingJob.id == job_id,
            DocumentProcessingJob.user_id == user_id,
        )
        .first()
    )


def get_latest_processing_job_for_document(
    db: Session,
    document_id: str,
    user_id: str,
) -> DocumentProcessingJob | None:
    return (
        db.query(DocumentProcessingJob)
        .filter(
            DocumentProcessingJob.document_id == document_id,
            DocumentProcessingJob.user_id == user_id,
        )
        .order_by(DocumentProcessingJob.created_at.desc())
        .first()
    )
