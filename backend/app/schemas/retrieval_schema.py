from pydantic import BaseModel, Field


class VectorSearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class VectorSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    user_id: str
    chunk_text: str
    chunk_index: int
    token_count: int | None = None
    page_number: int | None = None
    section_title: str | None = None
    document_name: str | None = None
    category: str | None = None
    distance: float
    similarity_score: float


class VectorSearchResponse(BaseModel):
    user_id: str
    query: str
    top_k: int
    result_count: int
    results: list[VectorSearchResult]