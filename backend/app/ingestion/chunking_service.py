from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.ingestion.chunker import chunk_text
from app.models.document import Document
from app.repositories.document_chunk_repository import (
    create_document_chunks,
    delete_chunks_by_document,
)
from app.repositories.document_repository import update_document_status
from app.utils.logger import get_logger


logger = get_logger(__name__)


def chunk_and_store_document_text(
    db: Session,
    document: Document,
) -> dict:
    """
    Read extracted text, split into chunks, store chunks in PostgreSQL,
    and update document status to chunked.
    """

    if document.status != "extracted":
        logger.warning(
            "Document chunking rejected: document_id=%s user_id=%s status=%s",
            document.id,
            document.user_id,
            document.status,
        )
        raise ValueError("Document must be in extracted status before chunking")

    extracted_text_path = (
        Path(settings.EXTRACTED_TEXT_DIR)
        / document.user_id
        / document.id
        / "extracted_text.txt"
    )

    if not extracted_text_path.exists():
        logger.error(
            "Document chunking failed because extracted text is missing: document_id=%s path=%s",
            document.id,
            extracted_text_path,
        )
        raise FileNotFoundError(
            f"Extracted text file not found: {extracted_text_path}"
        )

    try:
        logger.info(
            "Document chunking started: document_id=%s user_id=%s",
            document.id,
            document.user_id,
        )
        update_document_status(
            db=db,
            document_id=document.id,
            status="processing",
        )

        extracted_text = extracted_text_path.read_text(encoding="utf-8")

        chunks = chunk_text(extracted_text)

        delete_chunks_by_document(
            db=db,
            document_id=document.id,
            user_id=document.user_id,
        )

        create_document_chunks(
            db=db,
            document_id=document.id,
            user_id=document.user_id,
            chunks=chunks,
        )

        update_document_status(
            db=db,
            document_id=document.id,
            status="chunked",
        )

        logger.info(
            "Document chunking completed: document_id=%s user_id=%s chunks=%s",
            document.id,
            document.user_id,
            len(chunks),
        )
        return {
            "document_id": document.id,
            "user_id": document.user_id,
            "status": "chunked",
            "chunk_count": len(chunks),
            "message": "Document text chunked successfully",
        }

    except Exception:
        logger.exception(
            "Document chunking failed: document_id=%s user_id=%s",
            document.id,
            document.user_id,
        )
        rollback = getattr(db, "rollback", None)
        if rollback is not None:
            rollback()
        update_document_status(
            db=db,
            document_id=document.id,
            status="failed",
        )
        raise
