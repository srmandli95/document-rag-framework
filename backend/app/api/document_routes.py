from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.config.settings import settings
from app.db.database import get_db
from app.ingestion.chunking_service import chunk_and_store_document_text
from app.ingestion.document_processing_service import process_document
from app.ingestion.embedding_indexing_service import embed_document_chunks
from app.ingestion.extraction_service import extract_and_store_document_text
from app.models.user import User
from app.schemas.document_schema import (
    DocumentChunkingResponse,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentEmbeddingResponse,
    DocumentExtractionResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentProcessingResponse,
    DocumentUploadResponse,
)
from app.services.document_service import (
    create_document_record,
    get_document_by_id,
    get_documents_by_user,
    soft_delete_document,
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
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    await file.seek(0)

    return file_size_bytes


def _to_document_metadata(document) -> DocumentMetadata:
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
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _get_owned_document_or_404(
    db: Session,
    document_id: str,
    user_id: str,
):
    """
    Fetch a document only if it belongs to the authenticated user.

    Important:
    - Never trust user_id from query/body/form.
    - Always scope by current_user.id.
    - Return 404 instead of 403 so other users cannot confirm the document exists.
    """
    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if getattr(document, "status", None) == "deleted":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if getattr(document, "is_deleted", False):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if getattr(document, "deleted_at", None) is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
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

    # Create DB record first or use service-created document ID?
    # Your existing create_document_record appears to generate the ID itself.
    # So we save with a temporary UUID-like folder only if needed by storage service.
    # To preserve your current behavior, we use a generated storage document id.
    from uuid import uuid4

    storage_document_id = str(uuid4())

    storage_metadata = storage_service.save_file(
        file_obj=file.file,
        user_id=authenticated_user_id,
        document_id=storage_document_id,
        original_file_name=file.filename,
    )

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

    metadata = _to_document_metadata(document)

    return DocumentUploadResponse(
        **metadata.model_dump(),
        message="Document uploaded and metadata saved successfully",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
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

    documents = get_documents_by_user(
        db=db,
        user_id=authenticated_user_id,
    )

    document_metadata = [
        _to_document_metadata(document)
        for document in documents
    ]

    return DocumentListResponse(
        user_id=authenticated_user_id,
        documents=document_metadata,
        count=len(document_metadata),
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: str,
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentDetailResponse:
    """
    Get a single document only if it belongs to the authenticated user.
    """
    authenticated_user_id = str(current_user.id)

    document = _get_owned_document_or_404(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    metadata = _to_document_metadata(document)

    return DocumentDetailResponse(
        **metadata.model_dump(),
        message="Document found",
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str,
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentDeleteResponse:
    """
    Soft-delete a document only if it belongs to the authenticated user.
    """
    authenticated_user_id = str(current_user.id)

    document = soft_delete_document(
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
        message="Document soft deleted successfully",
    )


@router.post(
    "/{document_id}/extract",
    response_model=DocumentExtractionResponse,
)
def extract_document_text(
    document_id: str,
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentExtractionResponse:
    """
    Extract text from a document owned by the authenticated user.
    """
    authenticated_user_id = str(current_user.id)

    document = _get_owned_document_or_404(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    result = extract_and_store_document_text(
        db=db,
        document=document,
    )

    return DocumentExtractionResponse(**result)


@router.post(
    "/{document_id}/chunk",
    response_model=DocumentChunkingResponse,
)
def chunk_document_text(
    document_id: str,
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentChunkingResponse:
    """
    Chunk a document owned by the authenticated user.
    """
    authenticated_user_id = str(current_user.id)

    document = _get_owned_document_or_404(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    if document.status != "extracted":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document must be extracted before chunking",
        )

    result = chunk_and_store_document_text(
        db=db,
        document=document,
    )

    return DocumentChunkingResponse(**result)


@router.post("/{document_id}/embed", response_model=DocumentEmbeddingResponse)
def embed_document(
    document_id: str,
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentEmbeddingResponse:
    """
    Embed chunks for a document owned by the authenticated user.
    """
    authenticated_user_id = str(current_user.id)

    document = _get_owned_document_or_404(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    if document.status != "chunked":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Document must be in chunked status before embedding. "
                f"Current status: {document.status}"
            ),
        )

    try:
        result = embed_document_chunks(
            db=db,
            document=document,
        )

        return DocumentEmbeddingResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/{document_id}/process", response_model=DocumentProcessingResponse)
def process_uploaded_document(
    document_id: str,
    force: bool = Query(default=False),
    user_id: str | None = Query(default=None),  # Backward compatible, ignored.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentProcessingResponse:
    """
    Run the full document processing pipeline for a document.

    Day 20 auth rule:
    - JWT is required.
    - query user_id is ignored.
    - document must belong to current_user.id.

    Examples after Day 20:

    POST /documents/{document_id}/process
    POST /documents/{document_id}/process?force=true

    Header:
    Authorization: Bearer <token>
    """
    authenticated_user_id = str(current_user.id)

    document = _get_owned_document_or_404(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    return process_document(
        db=db,
        document=document,
        force=force,
    )