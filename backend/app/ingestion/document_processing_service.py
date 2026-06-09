from typing import Any

from sqlalchemy.orm import Session

from app.ingestion.chunking_service import chunk_and_store_document_text
from app.ingestion.embedding_indexing_service import embed_document_chunks
from app.ingestion.extraction_service import extract_and_store_document_text
from app.models.document import Document
from app.schemas.document_schema import (
    DocumentProcessingResponse,
    DocumentProcessingStep,
)


SUCCESS_STATUSES = {"completed", "success", "extracted", "chunked", "embedded"}


def _step_result(name: str, status: str, message: str) -> DocumentProcessingStep:
    return DocumentProcessingStep(
        name=name,
        status=status,
        message=message,
    )


def _get_response_value(response: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(field_name, default)

    return getattr(response, field_name, default)


def _is_successful_step(response: Any) -> bool:
    status = _get_response_value(response, "status", "")
    return str(status).lower() in SUCCESS_STATUSES


def _message_from_response(response: Any, default: str) -> str:
    message = _get_response_value(response, "message", None)
    return message or default


def _is_deleted_document(document: Document) -> bool:
    if getattr(document, "status", None) == "deleted":
        return True

    if getattr(document, "is_deleted", False):
        return True

    if getattr(document, "deleted_at", None) is not None:
        return True

    return False


def _mark_document_failed(db: Session, document: Document, message: str) -> None:
    document.status = "failed"
    db.add(document)
    db.commit()
    db.refresh(document)


def _refresh_document(db: Session, document: Document) -> Document:
    db.refresh(document)
    return document


def process_document(
    db: Session,
    document: Document,
    force: bool = False,
) -> DocumentProcessingResponse:
    """
    Run the full document processing pipeline:

    extract -> chunk -> embed

    If the document is already embedded:
    - force=False returns an already-processed response.
    - force=True reruns the full pipeline.

    The chunking service is expected to remove old chunks before creating
    new ones, so force reprocessing should not duplicate chunks.
    """
    steps: list[DocumentProcessingStep] = []

    if document is None:
        return DocumentProcessingResponse(
            document_id="",
            user_id="",
            status="failed",
            steps=[
                _step_result(
                    name="validate",
                    status="failed",
                    message="Document does not exist.",
                )
            ],
            message="Document does not exist.",
        )

    if _is_deleted_document(document):
        return DocumentProcessingResponse(
            document_id=str(document.id),
            user_id=document.user_id,
            status="deleted",
            steps=[
                _step_result(
                    name="validate",
                    status="failed",
                    message="Deleted documents cannot be processed.",
                )
            ],
            message="Deleted documents cannot be processed.",
        )

    if document.status == "embedded" and not force:
        return DocumentProcessingResponse(
            document_id=str(document.id),
            user_id=document.user_id,
            status=document.status,
            steps=[
                _step_result(
                    name="process",
                    status="skipped",
                    message="Document is already processed and ready for questions.",
                )
            ],
            message="Document is already processed and ready for questions. Use /reprocess if you want to rerun extraction, chunking, and embedding.",
        )

    try:
        extraction_response = extract_and_store_document_text(db, document)
        document = _refresh_document(db, document)

        extraction_message = _message_from_response(
            extraction_response,
            "Text extraction completed.",
        )

        if not _is_successful_step(extraction_response):
            steps.append(
                _step_result(
                    name="extract",
                    status="failed",
                    message=extraction_message,
                )
            )
            _mark_document_failed(db, document, extraction_message)

            return DocumentProcessingResponse(
                document_id=str(document.id),
                user_id=document.user_id,
                status=document.status,
                steps=steps,
                message="Document processing failed during extraction.",
            )

        steps.append(
            _step_result(
                name="extract",
                status="completed",
                message=extraction_message,
            )
        )

        chunking_response = chunk_and_store_document_text(db, document)
        document = _refresh_document(db, document)

        chunking_message = _message_from_response(
            chunking_response,
            "Document chunking completed.",
        )

        if not _is_successful_step(chunking_response):
            steps.append(
                _step_result(
                    name="chunk",
                    status="failed",
                    message=chunking_message,
                )
            )
            _mark_document_failed(db, document, chunking_message)

            return DocumentProcessingResponse(
                document_id=str(document.id),
                user_id=document.user_id,
                status=document.status,
                steps=steps,
                message="Document processing failed during chunking.",
            )

        steps.append(
            _step_result(
                name="chunk",
                status="completed",
                message=chunking_message,
            )
        )

        embedding_response = embed_document_chunks(db, document)
        document = _refresh_document(db, document)

        embedding_message = _message_from_response(
            embedding_response,
            "Document embedding completed.",
        )

        if not _is_successful_step(embedding_response):
            steps.append(
                _step_result(
                    name="embed",
                    status="failed",
                    message=embedding_message,
                )
            )
            _mark_document_failed(db, document, embedding_message)

            return DocumentProcessingResponse(
                document_id=str(document.id),
                user_id=document.user_id,
                status=document.status,
                steps=steps,
                message="Document processing failed during embedding.",
            )

        steps.append(
            _step_result(
                name="embed",
                status="completed",
                message=embedding_message,
            )
        )

        document = _refresh_document(db, document)

        return DocumentProcessingResponse(
            document_id=str(document.id),
            user_id=document.user_id,
            status=document.status,
            steps=steps,
            message="Document processed successfully and is ready for questions.",
        )

    except Exception as exc:
        error_message = f"Document processing failed: {exc}"
        _mark_document_failed(db, document, error_message)

        steps.append(
            _step_result(
                name="process",
                status="failed",
                message=error_message,
            )
        )

        return DocumentProcessingResponse(
            document_id=str(document.id),
            user_id=document.user_id,
            status=document.status,
            steps=steps,
            message=error_message,
        )