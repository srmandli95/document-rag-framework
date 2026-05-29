from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.ingestion.chunker import chunk_text
from app.models.document import Document
from app.services.document_chunk_service import (
    create_document_chunks,
    delete_chunks_by_document,
)
from app.services.document_service import update_document_status


def chunk_and_store_document_text(
    db: Session,
    document: Document,
) -> dict:
    """
    Read extracted text, split into chunks, store chunks in PostgreSQL,
    and update document status to chunked.
    """

    if document.status != "extracted":
        raise ValueError("Document must be in extracted status before chunking")

    extracted_text_path = (
        Path(settings.EXTRACTED_TEXT_DIR)
        / document.user_id
        / document.id
        / "extracted_text.txt"
    )

    if not extracted_text_path.exists():
        raise FileNotFoundError(
            f"Extracted text file not found: {extracted_text_path}"
        )

    try:
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

        return {
            "document_id": document.id,
            "user_id": document.user_id,
            "status": "chunked",
            "chunk_count": len(chunks),
            "message": "Document text chunked successfully",
        }

    except Exception:
        update_document_status(
            db=db,
            document_id=document.id,
            status="failed",
        )
        raise