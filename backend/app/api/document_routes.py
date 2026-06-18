from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.config.settings import settings
from app.db.database import SessionLocal, get_db
from app.ingestion.document_processing_service import process_document
from app.models.user import User
from app.schemas.document_schema import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.document_service import (
    create_document_record,
    delete_document_completely,
    get_document_by_id,
    get_documents_by_user,
)
from app.services.document_processing_job_service import (
    create_processing_job,
    get_processing_job_by_id,
    get_latest_processing_job_for_document,
)
from app.services.local_storage_service import LocalStorageService


router = APIRouter(prefix="/documents", tags=["Documents"])


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/html",
}

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".html",
}

SUPPORTED_DOCUMENT_STATUSES = {
    "uploaded",
    "processing",
    "extracted",
    "chunked",
    "embedded",
    "failed",
}

SUPPORTED_DOCUMENT_CATEGORIES = {
    "health_insurance",
    "auto_insurance",
    "home_insurance",
    "mortgage",
    "hoa",
    "employer_benefits",
    "internet",
    "utility",
    "banking",
    "credit_card",
    "warranty",
    "travel",
    "general",
}


def _validate_category(category: str) -> str:
    clean_category = category.strip()

    if not clean_category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="category is required",
        )

    return clean_category


def _validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file is required",
        )

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {file_extension}",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {file.content_type}",
        )


async def _get_file_size_bytes(file: UploadFile) -> int:
    contents = await file.read()
    file_size_bytes = len(contents)

    max_upload_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if file_size_bytes > max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    await file.seek(0)

    return file_size_bytes


def _display_status(document_status: str, job=None) -> str:
    if document_status == "embedded":
        return "ready"
    if document_status == "failed" or getattr(job, "status", None) == "failed":
        return "failed"

    current_step = getattr(job, "current_step", None)
    return {
        "validate": "validating",
        "extract": "extracting",
        "chunk": "chunking",
        "embed": "embedding",
    }.get(
        current_step,
        {
            "uploaded": "uploading",
            "processing": "validating",
            "extracted": "chunking",
            "chunked": "embedding",
        }.get(document_status, document_status),
    )


def _process_document_in_background(document_id: str, user_id: str, job_id: str) -> None:
    db = SessionLocal()
    try:
        document = get_document_by_id(db=db, document_id=document_id, user_id=user_id)
        job = get_processing_job_by_id(db, job_id, user_id)
        if document is not None and job is not None:
            process_document(db=db, document=document, force=False, job=job)
    finally:
        db.close()


def _to_document_metadata(document, job=None) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document.id,
        user_id=document.user_id,
        file_name=document.stored_file_name,
        original_file_name=document.original_file_name,
        content_type=document.content_type,
        file_size_bytes=document.file_size_bytes,
        category=document.category,
        storage_provider=document.storage_provider,
        storage_path=document.storage_path,
        status=document.status,
        display_status=_display_status(document.status, job),
        failure_reason=getattr(job, "error_message", None),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    category: str = Form(...),
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),  # Backward compatible, ignored.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentUploadResponse:
    """
    Upload a document for the authenticated user.

    Day 20 auth rule:
    - JWT is required.
    - user_id from form is ignored.
    - document owner is current_user.id.
    """
    authenticated_user_id = str(current_user.id)
    clean_category = _validate_category(category)

    _validate_file(file)

    file_size_bytes = await _get_file_size_bytes(file)

    storage_service = LocalStorageService()

    from uuid import uuid4

    storage_document_id = str(uuid4())

    storage_metadata = storage_service.save_file(
        file_obj=file.file,
        user_id=authenticated_user_id,
        document_id=storage_document_id,
        original_file_name=file.filename,
    )

    try:
        document = create_document_record(
            db=db,
            user_id=authenticated_user_id,
            original_file_name=file.filename,
            stored_file_name=storage_metadata["file_name"],
            category=clean_category,
            content_type=file.content_type or "application/octet-stream",
            file_size_bytes=file_size_bytes,
            storage_provider=storage_metadata["storage_provider"],
            storage_path=storage_metadata["storage_path"],
            status="uploaded",
        )
    except Exception:
        storage_service.delete_file(storage_metadata["storage_path"])
        raise

    try:
        job = create_processing_job(
            db=db,
            document_id=str(document.id),
            user_id=authenticated_user_id,
        )
    except Exception:
        delete_document_completely(
            db=db,
            document_id=str(document.id),
            user_id=authenticated_user_id,
        )
        raise
    background_tasks.add_task(
        _process_document_in_background,
        str(document.id),
        authenticated_user_id,
        str(job.id),
    )

    metadata = _to_document_metadata(document, job)

    return DocumentUploadResponse(
        **metadata.model_dump(),
        job_id=str(job.id),
        message="Document uploaded and processing started",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
    status_filter: str | None = Query(default=None, alias="status"),
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
    ready_only: bool = Query(default=False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    """
    List documents owned by the authenticated user.

    Day 20 auth rule:
    - JWT is required.
    - query user_id is ignored.
    """
    authenticated_user_id = str(current_user.id)
    clean_status = status_filter.strip() if status_filter else None
    clean_category = category.strip() if category else None
    clean_search = search.strip() if search else None

    if clean_status and clean_status not in SUPPORTED_DOCUMENT_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported document status: {clean_status}. "
                f"Supported statuses: {', '.join(sorted(SUPPORTED_DOCUMENT_STATUSES))}"
            ),
        )

    if clean_category and clean_category not in SUPPORTED_DOCUMENT_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported document category: {clean_category}. "
                f"Supported categories: {', '.join(sorted(SUPPORTED_DOCUMENT_CATEGORIES))}"
            ),
        )

    documents = get_documents_by_user(
        db=db,
        user_id=authenticated_user_id,
        status=clean_status,
        category=clean_category,
        search=clean_search,
        ready_only=ready_only,
    )

    document_metadata = []
    for document in documents:
        latest_job = (
            get_latest_processing_job_for_document(
                db, str(document.id), authenticated_user_id
            )
            if document.status in {"processing", "failed"}
            else None
        )
        document_metadata.append(_to_document_metadata(document, latest_job))

    return DocumentListResponse(
        user_id=authenticated_user_id,
        documents=document_metadata,
        count=len(document_metadata),
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str,
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentDeleteResponse:
    """Permanently delete an owned document and all generated data."""
    authenticated_user_id = str(current_user.id)

    document = delete_document_completely(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentDeleteResponse(
        document_id=document.id,
        user_id=document.user_id,
        status=document.status,
        message="Document deleted successfully",
    )
