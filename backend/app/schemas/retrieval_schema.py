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

class BM25SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class BM25SearchResult(BaseModel):
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
    bm25_score: float


class BM25SearchResponse(BaseModel):
    user_id: str
    query: str
    top_k: int
    result_count: int
    results: list[BM25SearchResult]

class HybridSearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    vector_top_k: int = Field(default=20, ge=1, le=100)
    bm25_top_k: int = Field(default=20, ge=1, le=100)
    vector_weight: float = Field(default=0.6, ge=0)
    bm25_weight: float = Field(default=0.4, ge=0)


class HybridSearchResult(BaseModel):
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
    vector_score: float
    bm25_score: float
    normalized_vector_score: float
    normalized_bm25_score: float
    hybrid_score: float
    retrieval_sources: list[str]


class HybridSearchResponse(BaseModel):
    user_id: str
    query: str
    top_k: int
    result_count: int
    results: list[HybridSearchResult]