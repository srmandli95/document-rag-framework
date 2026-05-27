from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field

class Document(SQLModel, table=False):

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    original_file_name: str
    stored_file_name: str
    category: str
    content_type: str
    file_size_bytes: int
    storage_provider: str
    storage_path: str
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    