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
    status: str | None = None,
    category: str | None = None,
    search: str | None = None,
    ready_only: bool = False,
) -> list[Document]:
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


def update_document_status(
    db: Session,
    document_id: str,
    status: str,
) -> Document | None:
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
