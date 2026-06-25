import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_processing_job import DocumentProcessingJob


def create_document_record(
    db: Session,
    *,
    user_id: str,
    original_file_name: str,
    stored_file_name: str,
    category: str,
    content_type: str,
    file_size_bytes: int,
    storage_provider: str,
    storage_path: str,
    status: str = "uploaded",
) -> Document:
    """Create and persist metadata for an uploaded document."""
    document = Document(
        user_id=user_id,
        original_file_name=original_file_name,
        stored_file_name=stored_file_name,
        category=category,
        content_type=content_type,
        file_size_bytes=file_size_bytes,
        storage_provider=storage_provider,
        storage_path=storage_path,
        status=status,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


def get_documents_by_user(
    db: Session,
    *,
    user_id: str,
    status: str | None = None,
    category: str | None = None,
    search: str | None = None,
    ready_only: bool = False,
) -> list[Document]:
    """Return documents for a user with optional filters."""
    query = db.query(Document).filter(
        Document.user_id == user_id,
        Document.status != "deleted",
    )

    if status:
        query = query.filter(Document.status == status)

    if category:
        query = query.filter(Document.category == category)

    if search:
        query = query.filter(Document.original_file_name.ilike(f"%{search}%"))

    if ready_only:
        query = query.filter(Document.status == "embedded")

    return query.order_by(Document.created_at.desc()).all()


def get_document_by_id(
    db: Session,
    *,
    document_id: str,
    user_id: str,
) -> Document | None:
    """Return one document owned by a user."""
    return (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.user_id == user_id,
            Document.status != "deleted",
        )
        .first()
    )


def _remove_document_artifacts(document: Document) -> None:
    """Remove stored files and derived artifacts for a document."""
    artifact_directories = {
        Path(settings.EXTRACTED_TEXT_DIR) / str(document.user_id) / str(document.id),
        Path(settings.PROCESSED_CHUNKS_DIR) / str(document.user_id) / str(document.id),
        Path(settings.REDACTED_DOCUMENTS_DIR) / str(document.user_id) / str(document.id),
        Path(document.storage_path).parent,
    }

    for directory in artifact_directories:
        if directory.exists():
            shutil.rmtree(directory)


def delete_document_completely(
    db: Session,
    *,
    document_id: str,
    user_id: str,
) -> Document | None:
    """Delete a document and its related persisted data."""
    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    if document is None:
        return None

    response_document = document
    response_document.status = "deleted"

    try:
        _remove_document_artifacts(document)
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.user_id == user_id,
        ).delete(synchronize_session=False)
        db.query(DocumentProcessingJob).filter(
            DocumentProcessingJob.document_id == document_id,
            DocumentProcessingJob.user_id == user_id,
        ).delete(synchronize_session=False)
        db.delete(document)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return response_document


def update_document_status(
    db: Session,
    document_id: str,
    status: str,
) -> Document | None:
    """Update and persist a document status."""
    document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )

    if document is None:
        return None

    document.status = status
    db.commit()
    db.refresh(document)

    return document
