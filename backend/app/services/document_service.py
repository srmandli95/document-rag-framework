from sqlalchemy.orm import Session

from app.models.document import Document


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
) -> list[Document]:
    return (
        db.query(Document)
        .filter(
            Document.user_id == user_id,
            Document.status != "deleted",
        )
        .order_by(Document.created_at.desc())
        .all()
    )


def get_document_by_id(
    db: Session,
    *,
    document_id: str,
    user_id: str,
) -> Document | None:
    return (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.user_id == user_id,
            Document.status != "deleted",
        )
        .first()
    )


def soft_delete_document(
    db: Session,
    *,
    document_id: str,
    user_id: str,
) -> Document | None:
    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    if document is None:
        return None

    document.status = "deleted"

    db.commit()
    db.refresh(document)

    return document