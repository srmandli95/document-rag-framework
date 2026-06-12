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


SUCCESS_STATUSES = {"completed", "success", "extracted", "chunked", "embedded"}


def _step_result(name: str, status: str, message: str) -> DocumentProcessingStep:
    return DocumentProcessingStep(name=name, status=status, message=message)


def _serialized_steps(steps: list[DocumentProcessingStep]) -> list[dict]:
    return [step.model_dump() for step in steps]


def _get_response_value(response: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(field_name, default)
    return getattr(response, field_name, default)


def _is_successful_step(response: Any) -> bool:
    return str(_get_response_value(response, "status", "")).lower() in SUCCESS_STATUSES


def _message_from_response(response: Any, default: str) -> str:
    return _get_response_value(response, "message", None) or default


def _is_deleted_document(document: Document) -> bool:
    return (
        getattr(document, "status", None) == "deleted"
        or getattr(document, "is_deleted", False)
        or getattr(document, "deleted_at", None) is not None
    )


def _mark_document_failed(db: Session, document: Document, message: str) -> None:
    document.status = "failed"
    db.add(document)
    db.commit()
    db.refresh(document)


def _refresh_document(db: Session, document: Document) -> Document:
    db.refresh(document)
    return document


def _response(
    document: Document,
    job: DocumentProcessingJob,
    status: str,
    steps: list[DocumentProcessingStep],
    message: str,
) -> DocumentProcessingResponse:
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
        return DocumentProcessingResponse(
            document_id="",
            user_id="",
            status="failed",
            steps=[_step_result("validate", "failed", "Document does not exist.")],
            message="Document does not exist.",
            job_id=None,
        )

    if job is None:
        job = create_processing_job(
            db=db,
            document_id=str(document.id),
            user_id=document.user_id,
            force=force,
        )

    mark_job_running(db=db, job=job, current_step="validate")

    if _is_deleted_document(document):
        message = "Deleted documents cannot be processed."
        steps.append(_step_result("validate", "failed", message))
        mark_job_failed(db=db, job=job, steps=_serialized_steps(steps), error_message=message)
        return _response(document, job, "deleted", steps, message)

    if document.status == "embedded" and not force:
        message = "Document is already processed and ready for questions."
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
        mark_job_running(db=db, job=job, current_step="extract")
        extraction_response = extract_and_store_document_text(db, document)
        document = _refresh_document(db, document)
        extraction_message = _message_from_response(extraction_response, "Text extraction completed.")
        extraction_status = "completed" if _is_successful_step(extraction_response) else "failed"
        steps.append(_step_result("extract", extraction_status, extraction_message))
        update_job_step(db=db, job=job, step=steps[-1].model_dump(), current_step="extract")

        if extraction_status == "failed":
            _mark_document_failed(db, document, extraction_message)
            mark_job_failed(db, job, _serialized_steps(steps), extraction_message)
            return _response(
                document, job, document.status, steps, "Document processing failed during extraction."
            )

        mark_job_running(db=db, job=job, current_step="chunk")
        chunking_response = chunk_and_store_document_text(db, document)
        document = _refresh_document(db, document)
        chunking_message = _message_from_response(chunking_response, "Document chunking completed.")
        chunking_status = "completed" if _is_successful_step(chunking_response) else "failed"
        steps.append(_step_result("chunk", chunking_status, chunking_message))
        update_job_step(db=db, job=job, step=steps[-1].model_dump(), current_step="chunk")

        if chunking_status == "failed":
            _mark_document_failed(db, document, chunking_message)
            mark_job_failed(db, job, _serialized_steps(steps), chunking_message)
            return _response(
                document, job, document.status, steps, "Document processing failed during chunking."
            )

        mark_job_running(db=db, job=job, current_step="embed")
        embedding_response = embed_document_chunks(db, document)
        document = _refresh_document(db, document)
        embedding_message = _message_from_response(embedding_response, "Document embedding completed.")
        embedding_status = "completed" if _is_successful_step(embedding_response) else "failed"
        steps.append(_step_result("embed", embedding_status, embedding_message))
        update_job_step(db=db, job=job, step=steps[-1].model_dump(), current_step="embed")

        if embedding_status == "failed":
            _mark_document_failed(db, document, embedding_message)
            mark_job_failed(db, job, _serialized_steps(steps), embedding_message)
            return _response(
                document, job, document.status, steps, "Document processing failed during embedding."
            )

        document = _refresh_document(db, document)
        mark_job_completed(db=db, job=job, steps=_serialized_steps(steps))
        return _response(
            document,
            job,
            document.status,
            steps,
            "Document processed successfully and is ready for questions.",
        )

    except Exception as exc:
        error_message = f"Document processing failed: {exc}"
        _mark_document_failed(db, document, error_message)
        failed_step_name = job.current_step or "process"
        steps.append(_step_result(failed_step_name, "failed", error_message))
        mark_job_failed(db=db, job=job, steps=_serialized_steps(steps), error_message=error_message)
        return _response(document, job, document.status, steps, error_message)
