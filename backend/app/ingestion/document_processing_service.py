from typing import Any

from sqlalchemy.orm import Session

from app.ingestion.chunking_service import chunk_and_store_document_text
from app.ingestion.embedding_indexing_service import embed_document_chunks
from app.ingestion.extraction_service import extract_and_store_document_text
from app.models.document import Document
from app.models.document_processing_job import DocumentProcessingJob
from app.schemas.document_schema import (
    DocumentProcessingResponse,
    DocumentProcessingStep,
)
from app.services.document_processing_job_service import (
    create_processing_job,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
    mark_job_skipped,
    update_job_step,
)
from app.utils.logger import get_logger


SUCCESS_STATUSES = {"completed", "success", "extracted", "chunked", "embedded"}
logger = get_logger(__name__)


def _step_result(name: str, status: str, message: str) -> DocumentProcessingStep:
    """Build the standard status payload for a processing step."""
    return DocumentProcessingStep(name=name, status=status, message=message)


def _serialized_steps(steps: list[DocumentProcessingStep]) -> list[dict]:
    """Serialize processing step results for job metadata."""
    return [step.model_dump() for step in steps]


def _get_response_value(response: Any, field_name: str, default: Any = None) -> Any:
    """Read a value from a response object or dictionary."""
    if isinstance(response, dict):
        return response.get(field_name, default)
    return getattr(response, field_name, default)


def _is_successful_step(response: Any) -> bool:
    """Return whether a processing step completed successfully."""
    return str(_get_response_value(response, "status", "")).lower() in SUCCESS_STATUSES


def _message_from_response(response: Any, default: str) -> str:
    """Extract a message from a processing step response."""
    return _get_response_value(response, "message", None) or default


def _is_deleted_document(document: Document) -> bool:
    """Return whether a SQLAlchemy error indicates a deleted document."""
    return (
        getattr(document, "status", None) == "deleted"
        or getattr(document, "is_deleted", False)
        or getattr(document, "deleted_at", None) is not None
    )


def _mark_document_failed(db: Session, document: Document, message: str) -> None:
    """Mark a document as failed and flush the change."""
    document.status = "failed"
    db.add(document)
    db.commit()
    db.refresh(document)


def _refresh_document(db: Session, document: Document) -> Document:
    """Refresh a document model from the database when possible."""
    db.refresh(document)
    return document


def _response(
    document: Document,
    job: DocumentProcessingJob,
    status: str,
    steps: list[DocumentProcessingStep],
    message: str,
) -> DocumentProcessingResponse:
    """Build the final document processing response payload."""
    return DocumentProcessingResponse(
        document_id=str(document.id),
        user_id=document.user_id,
        status=status,
        steps=steps,
        message=message,
        job_id=str(job.id),
    )


def process_document(
    db: Session,
    document: Document,
    force: bool = False,
    job: DocumentProcessingJob | None = None,
) -> DocumentProcessingResponse:
    """Synchronously run extract -> chunk -> embed while tracking the attempt."""
    steps: list[DocumentProcessingStep] = []

    if document is None:
        logger.error("Document processing requested with no document")
        return DocumentProcessingResponse(
            document_id="",
            user_id="",
            status="failed",
            steps=[_step_result("validate", "failed", "Document does not exist.")],
            message="Document does not exist.",
            job_id=None,
        )

    if job is None:
        logger.debug(
            "Creating document processing job: document_id=%s user_id=%s force=%s",
            document.id,
            document.user_id,
            force,
        )
        job = create_processing_job(
            db=db,
            document_id=str(document.id),
            user_id=document.user_id,
            force=force,
        )

    logger.info(
        "Document processing started: document_id=%s user_id=%s job_id=%s force=%s status=%s",
        document.id,
        document.user_id,
        job.id,
        force,
        document.status,
    )
    mark_job_running(db=db, job=job, current_step="validate")

    if _is_deleted_document(document):
        message = "Deleted documents cannot be processed."
        logger.warning(
            "Document processing skipped for deleted document: document_id=%s user_id=%s job_id=%s",
            document.id,
            document.user_id,
            job.id,
        )
        steps.append(_step_result("validate", "failed", message))
        mark_job_failed(db=db, job=job, steps=_serialized_steps(steps), error_message=message)
        return _response(document, job, "deleted", steps, message)

    if document.status == "embedded" and not force:
        message = "Document is already processed and ready for questions."
        logger.info(
            "Document processing skipped because document is already embedded: document_id=%s job_id=%s",
            document.id,
            job.id,
        )
        steps.append(_step_result("process", "skipped", message))
        mark_job_skipped(db=db, job=job, steps=_serialized_steps(steps), message=message)
        return _response(
            document,
            job,
            document.status,
            steps,
            f"{message} Use /reprocess if you want to rerun extraction, chunking, and embedding.",
        )

    try:
        logger.debug("Document processing step started: document_id=%s step=extract", document.id)
        mark_job_running(db=db, job=job, current_step="extract")
        extraction_response = extract_and_store_document_text(db, document)
        document = _refresh_document(db, document)
        extraction_message = _message_from_response(extraction_response, "Text extraction completed.")
        extraction_status = "completed" if _is_successful_step(extraction_response) else "failed"
        steps.append(_step_result("extract", extraction_status, extraction_message))
        update_job_step(db=db, job=job, step=steps[-1].model_dump(), current_step="extract")

        if extraction_status == "failed":
            logger.error(
                "Document processing failed during extraction: document_id=%s job_id=%s message=%s",
                document.id,
                job.id,
                extraction_message,
            )
            _mark_document_failed(db, document, extraction_message)
            mark_job_failed(db, job, _serialized_steps(steps), extraction_message)
            return _response(
                document, job, document.status, steps, "Document processing failed during extraction."
            )

        logger.debug("Document processing step started: document_id=%s step=chunk", document.id)
        mark_job_running(db=db, job=job, current_step="chunk")
        chunking_response = chunk_and_store_document_text(db, document)
        document = _refresh_document(db, document)
        chunking_message = _message_from_response(chunking_response, "Document chunking completed.")
        chunking_status = "completed" if _is_successful_step(chunking_response) else "failed"
        steps.append(_step_result("chunk", chunking_status, chunking_message))
        update_job_step(db=db, job=job, step=steps[-1].model_dump(), current_step="chunk")

        if chunking_status == "failed":
            logger.error(
                "Document processing failed during chunking: document_id=%s job_id=%s message=%s",
                document.id,
                job.id,
                chunking_message,
            )
            _mark_document_failed(db, document, chunking_message)
            mark_job_failed(db, job, _serialized_steps(steps), chunking_message)
            return _response(
                document, job, document.status, steps, "Document processing failed during chunking."
            )

        logger.debug("Document processing step started: document_id=%s step=embed", document.id)
        mark_job_running(db=db, job=job, current_step="embed")
        embedding_response = embed_document_chunks(db, document)
        document = _refresh_document(db, document)
        embedding_message = _message_from_response(embedding_response, "Document embedding completed.")
        embedding_status = "completed" if _is_successful_step(embedding_response) else "failed"
        steps.append(_step_result("embed", embedding_status, embedding_message))
        update_job_step(db=db, job=job, step=steps[-1].model_dump(), current_step="embed")

        if embedding_status == "failed":
            logger.error(
                "Document processing failed during embedding: document_id=%s job_id=%s message=%s",
                document.id,
                job.id,
                embedding_message,
            )
            _mark_document_failed(db, document, embedding_message)
            mark_job_failed(db, job, _serialized_steps(steps), embedding_message)
            return _response(
                document, job, document.status, steps, "Document processing failed during embedding."
            )

        document = _refresh_document(db, document)
        mark_job_completed(db=db, job=job, steps=_serialized_steps(steps))
        logger.info(
            "Document processing completed: document_id=%s user_id=%s job_id=%s status=%s steps=%s",
            document.id,
            document.user_id,
            job.id,
            document.status,
            len(steps),
        )
        return _response(
            document,
            job,
            document.status,
            steps,
            "Document processed successfully and is ready for questions.",
        )

    except Exception as exc:
        error_message = f"Document processing failed: {exc}"
        logger.exception(
            "Document processing failed unexpectedly: document_id=%s user_id=%s job_id=%s",
            document.id,
            document.user_id,
            job.id,
        )
        db.rollback()
        _mark_document_failed(db, document, error_message)
        failed_step_name = job.current_step or "process"
        steps.append(_step_result(failed_step_name, "failed", error_message))
        mark_job_failed(db=db, job=job, steps=_serialized_steps(steps), error_message=error_message)
        return _response(document, job, document.status, steps, error_message)
