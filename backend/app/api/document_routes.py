from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config.settings import settings
from app.schemas.document_schema import DocumentUploadResponse
from app.services.local_storage_service import LocalStorageService


router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/html",
}

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html"}

@router.post("/upload", response_model=DocumentUploadResponse)
def upload_document(
    user_id: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {file.content_type}",
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {extension}",
        )

    if not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID cannot be empty",
        )
    if not category.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category cannot be empty",
        )
    if not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File name cannot be empty",
        )

    max_upload_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    file.file.seek(0, 2)
    file_size_bytes = file.file.tell()
    file.file.seek(0)

    if file_size_bytes > max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds the maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB",
        )
    
    document_id = str(uuid4())

    storage_service = LocalStorageService()

    storage_metadata = storage_service.save_file(
        file_obj=file.file,
        user_id=user_id,
        document_id=document_id,
        original_file_name=file.filename,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        user_id=user_id,
        file_name=storage_metadata["file_name"],
        original_file_name=file.filename,
        content_type=file.content_type,
        file_size_bytes=storage_metadata["file_size_bytes"],
        category=category,
        storage_provider=storage_metadata["storage_provider"],
        storage_path=storage_metadata["storage_path"],
        status="uploaded",
        message="File uploaded successfully",
    )
