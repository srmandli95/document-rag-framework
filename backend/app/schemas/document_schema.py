from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentMetadata(BaseModel):
    """Shared API schema for document metadata."""
    document_id: str
    user_id: str
    file_name: str
    original_file_name: str
    content_type: str
    file_size_bytes: int
    category: str
    storage_provider: str
    storage_path: str
    status: str
    display_status: str
    failure_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(DocumentMetadata):
    """API response schema for document uploads."""
    job_id: str
    message: str


class DocumentDetailResponse(DocumentMetadata):
    """API response schema for document details."""
    message: str = "Document found"


class DocumentListResponse(BaseModel):
    """API response schema for listing documents."""
    user_id: str
    documents: list[DocumentMetadata]
    count: int


class DocumentDeleteResponse(BaseModel):
    """API response schema for document deletion."""
    document_id: str
    user_id: str
    status: str
    message: str

class DocumentExtractionResponse(BaseModel):
    """API response schema for document text extraction."""
    document_id: str
    user_id: str
    status: str
    extracted_text_path: str
    character_count: int
    message: str

class DocumentChunkingResponse(BaseModel):
    """API response schema for document chunking."""
    document_id: str
    user_id: str
    status: str
    chunk_count: int
    message: str


class DocumentChunkMetadata(BaseModel):
    """Shared API schema for document chunk metadata."""
    chunk_id: str
    document_id: str
    user_id: str
    chunk_index: int
    token_count: int
    section_title: str | None = None
    page_number: int | None = None
    status: str


class DocumentChunkDetailResponse(DocumentChunkMetadata):
    """API response schema for chunk details."""
    chunk_text: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentChunkListResponse(BaseModel):
    """API response schema for listing chunks."""
    document_id: str
    user_id: str
    chunks: list[DocumentChunkDetailResponse]


class DocumentEmbeddingResponse(BaseModel):
    """API response schema for document embedding."""
    document_id: str
    user_id: str
    status: str
    embedded_chunk_count: int
    embedding_provider: str
    embedding_model_name: str
    message: str

class DocumentProcessingStep(BaseModel):
    """API schema for one document processing step."""
    name: str
    status: str
    message: str


class DocumentProcessingResponse(BaseModel):
    """API response schema for processing a document."""
    document_id: str | None
    user_id: str | None
    status: str
    steps: list[DocumentProcessingStep]
    message: str
    job_id: str | None = None


class DocumentProcessingJobResponse(BaseModel):
    """API response schema for a processing job."""
    job_id: str
    document_id: str
    user_id: str
    status: str
    force: bool
    current_step: str | None = None
    steps: list[DocumentProcessingStep]
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentProcessingJobListResponse(BaseModel):
    """API response schema for listing processing jobs."""
    document_id: str
    user_id: str
    jobs: list[DocumentProcessingJobResponse]
