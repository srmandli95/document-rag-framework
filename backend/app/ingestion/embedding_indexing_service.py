from typing import Dict

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.embeddings.embedding_service import get_embedding_service
from app.models.document import Document
from app.services.document_chunk_service import (
    get_created_chunks_by_document,
    update_chunks_with_embeddings,
)
from app.services.document_service import update_document_status


def embed_document_chunks(
    db: Session,
    document: Document,
) -> Dict:
    if document.status != "chunked":
        raise ValueError(
            f"Document must be in chunked status before embedding. Current status: {document.status}"
        )

    try:
        chunks = get_created_chunks_by_document(
            db=db,
            document_id=document.id,
            user_id=document.user_id,
        )

        if not chunks:
            raise ValueError("No created chunks found for this document.")

        update_document_status(
            db=db,
            document_id=document.id,
            status="processing",
        )

        embedding_service = get_embedding_service()

        chunk_texts = [chunk.chunk_text for chunk in chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)

        update_chunks_with_embeddings(
            db=db,
            chunks=chunks,
            embeddings=embeddings,
        )

        update_document_status(
            db=db,
            document_id=document.id,
            status="embedded",
        )

        return {
            "document_id": document.id,
            "user_id": document.user_id,
            "status": "embedded",
            "embedded_chunk_count": len(chunks),
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model_name": settings.EMBEDDING_MODEL_NAME,
            "message": "Document chunks embedded successfully.",
        }

    except Exception:
        update_document_status(
            db=db,
            document_id=document.id,
            status="failed",
        )
        raise