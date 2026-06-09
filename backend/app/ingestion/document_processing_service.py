from typing import Any

from sqlalchemy.orm import Session

from app.ingestion.extraction_service import extract_and_store_document_text
from app.ingestion.chunking_service import chunk_and_store_document_text
from app.ingestion.embedding_indexing_service import embed_document_chunks


def _step_result(name: str, status: str, message: str) -> dict[str, str]:
    return {
        "name": name,
        "status": status,
        "message": message,
    }


def _mark_document_failed(db: Session, document: Any) -> None:
    """
    Best-effort failure status update.

    Existing extraction/chunking/embedding services may already mark failure.
    This helper keeps Day 17 orchestration safe if a service raises before
    updating the document status.
    """
    try:
        document.status = "failed"
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception:
        db.rollback()


def _refresh_document(db: Session, document: Any) -> Any:
    """
    Refresh the SQLAlchemy document instance after each processing step.

    This avoids stale local status when the underlying service updates
    document.status and commits.
    """
    db.refresh(document)
    return document


def process_document(
    db: Session,
    document: Any,
) -> dict[str, Any]:
    """
    Run the synchronous document processing pipeline:

        extract -> chunk -> embed

    This function intentionally does not duplicate extraction, chunking,
    or embedding logic. It only orchestrates existing ingestion services.
    """
    if document is None:
        return {
            "document_id": None,
            "user_id": None,
            "status": "failed",
            "steps": [],
            "message": "Document not found.",
        }

    if getattr(document, "status", None) == "deleted":
        return {
            "document_id": str(document.id),
            "user_id": document.user_id,
            "status": "deleted",
            "steps": [],
            "message": "Deleted documents cannot be processed.",
        }

    if getattr(document, "status", None) == "embedded":
        return {
            "document_id": str(document.id),
            "user_id": document.user_id,
            "status": "embedded",
            "steps": [],
            "message": "Document is already processed.",
        }

    steps: list[dict[str, str]] = []

    # Step 1: Extract
    try:
        extract_and_store_document_text(db, document)
        document = _refresh_document(db, document)

        if document.status != "extracted":
            raise RuntimeError(
                f"Extraction completed but document status is '{document.status}', expected 'extracted'."
            )

        steps.append(
            _step_result(
                name="extract",
                status="completed",
                message="Document text extracted successfully.",
            )
        )
    except Exception as exc:
        _mark_document_failed(db, document)
        steps.append(
            _step_result(
                name="extract",
                status="failed",
                message=f"Document text extraction failed: {exc}",
            )
        )
        return {
            "document_id": str(document.id),
            "user_id": document.user_id,
            "status": "failed",
            "steps": steps,
            "message": "Document processing failed during extraction.",
        }

    # Step 2: Chunk
    try:
        chunk_and_store_document_text(db, document)
        document = _refresh_document(db, document)

        if document.status != "chunked":
            raise RuntimeError(
                f"Chunking completed but document status is '{document.status}', expected 'chunked'."
            )

        steps.append(
            _step_result(
                name="chunk",
                status="completed",
                message="Document text chunked successfully.",
            )
        )
    except Exception as exc:
        _mark_document_failed(db, document)
        steps.append(
            _step_result(
                name="chunk",
                status="failed",
                message=f"Document text chunking failed: {exc}",
            )
        )
        return {
            "document_id": str(document.id),
            "user_id": document.user_id,
            "status": "failed",
            "steps": steps,
            "message": "Document processing failed during chunking.",
        }

    # Step 3: Embed
    try:
        embed_document_chunks(db, document)
        document = _refresh_document(db, document)

        if document.status != "embedded":
            raise RuntimeError(
                f"Embedding completed but document status is '{document.status}', expected 'embedded'."
            )

        steps.append(
            _step_result(
                name="embed",
                status="completed",
                message="Document chunks embedded successfully.",
            )
        )
    except Exception as exc:
        _mark_document_failed(db, document)
        steps.append(
            _step_result(
                name="embed",
                status="failed",
                message=f"Document chunk embedding failed: {exc}",
            )
        )
        return {
            "document_id": str(document.id),
            "user_id": document.user_id,
            "status": "failed",
            "steps": steps,
            "message": "Document processing failed during embedding.",
        }

    return {
        "document_id": str(document.id),
        "user_id": document.user_id,
        "status": document.status,
        "steps": steps,
        "message": "Document processed successfully.",
    }