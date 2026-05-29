from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentMetadata(BaseModel):
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
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(DocumentMetadata):
    message: str


class DocumentDetailResponse(DocumentMetadata):
    message: str = "Document found"


class DocumentListResponse(BaseModel):
    user_id: str
    documents: list[DocumentMetadata]
    count: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    user_id: str
    status: str
    message: str

class DocumentExtractionResponse(BaseModel):
    document_id: str
    user_id: str
    status: str
    extracted_text_path: str
    character_count: int
    message: str

class DocumentChunkingResponse(BaseModel):
    document_id: str
    user_id: str
    status: str
    chunk_count: int
    message: str


class DocumentChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    user_id: str
    chunk_index: int
    token_count: int
    section_title: str | None = None
    page_number: int | None = None
    status: str