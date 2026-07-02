from typing import Dict

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.embeddings.embedding_service import get_embedding_service
from app.models.document import Document
from app.repositories.document_chunk_repository import (
    get_created_chunks_by_document,
    update_chunks_with_embeddings,
)
from app.repositories.document_repository import update_document_status
from app.utils.logger import get_logger


logger = get_logger(__name__)


def embed_document_chunks(
    db: Session,
    document: Document,
) -> Dict:
    """Embed created chunks for a document and persist the vectors."""
    if document.status != "chunked":
        logger.warning(
            "Document embedding rejected: document_id=%s user_id=%s status=%s",
            document.id,
            document.user_id,
            document.status,
        )
        raise ValueError(
            f"Document must be in chunked status before embedding. Current status: {document.status}"
        )

    try:
        logger.info(
            "Document embedding started: document_id=%s user_id=%s",
            document.id,
            document.user_id,
        )
        chunks = get_created_chunks_by_document(
            db=db,
            document_id=document.id,
            user_id=document.user_id,
        )

        if not chunks:
            logger.warning(
                "Document embedding found no created chunks: document_id=%s user_id=%s",
                document.id,
                document.user_id,
            )
            raise ValueError("No created chunks found for this document.")

        update_document_status(
            db=db,
            document_id=document.id,
            status="processing",
        )

        embedding_service = get_embedding_service()

        chunk_texts = [
            getattr(chunk, "search_text", None) or chunk.chunk_text
            for chunk in chunks
        ]
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

        logger.info(
            "Document embedding completed: document_id=%s user_id=%s chunks=%s provider=%s model=%s",
            document.id,
            document.user_id,
            len(chunks),
            settings.EMBEDDING_PROVIDER,
            settings.EMBEDDING_MODEL_NAME,
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
        logger.exception(
            "Document embedding failed: document_id=%s user_id=%s",
            document.id,
            document.user_id,
        )
        update_document_status(
            db=db,
            document_id=document.id,
            status="failed",
        )
        raise
