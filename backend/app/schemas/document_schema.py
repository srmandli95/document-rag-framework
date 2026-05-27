from pydantic import BaseModel

class DocumentUploadResponse(BaseModel):
    document_id: str
    user_id : str
    file_name: str
    original_file_name: str
    content_type: str
    file_size_bytes: int
    category: str
    storage_provider: str
    storage_path: str
    status: str
    message: str


class DocumentMetadata(BaseModel):
    document_id: str
    user_id : str
    file_name: str
    category: str
    status: str
    storage_provider: str
    storage_path: str