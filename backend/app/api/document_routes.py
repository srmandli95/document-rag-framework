from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.db.database import get_db
from app.schemas.document_schema import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentMetadata,
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


def _validate_user_id(user_id: str) -> str:
    clean_user_id = user_id.strip()

    if not clean_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    return clean_user_id


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


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    user_id: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    clean_user_id = _validate_user_id(user_id)
    clean_category = _validate_category(category)

    _validate_file(file)

    file_size_bytes = await _get_file_size_bytes(file)

    document_id = str(uuid4())

    storage_service = LocalStorageService()

    storage_metadata = storage_service.save_file(
        file_obj=file.file,
        user_id=clean_user_id,
        document_id=document_id,
        original_file_name=file.filename,
    )

    document = create_document_record(
        db=db,
        user_id=clean_user_id,
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
    user_id: str = Query(...),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    clean_user_id = _validate_user_id(user_id)

    documents = get_documents_by_user(
        db=db,
        user_id=clean_user_id,
    )

    document_metadata = [
        _to_document_metadata(document)
        for document in documents
    ]

    return DocumentListResponse(
        user_id=clean_user_id,
        documents=document_metadata,
        count=len(document_metadata),
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: str,
    user_id: str = Query(...),
    db: Session = Depends(get_db),
) -> DocumentDetailResponse:
    clean_user_id = _validate_user_id(user_id)

    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=clean_user_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    metadata = _to_document_metadata(document)

    return DocumentDetailResponse(
        **metadata.model_dump(),
        message="Document found",
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str,
    user_id: str = Query(...),
    db: Session = Depends(get_db),
) -> DocumentDeleteResponse:
    clean_user_id = _validate_user_id(user_id)

    document = soft_delete_document(
        db=db,
        document_id=document_id,
        user_id=clean_user_id,
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