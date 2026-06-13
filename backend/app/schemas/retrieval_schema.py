from pydantic import BaseModel, Field


class VectorSearchRequest(BaseModel):
    user_id: str | None = None
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
    user_id: str | None = None
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
    user_id: str | None = None
    query: str
    top_k: int = Field(default=5, ge=1)
    vector_top_k: int = Field(default=20, ge=1)
    bm25_top_k: int = Field(default=20, ge=1)
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

class RerankSearchRequest(BaseModel):
    user_id: str | None = None
    query: str
    top_k: int = 8
    hybrid_top_k: int = 20
    vector_top_k: int = 20
    bm25_top_k: int = 20
    vector_weight: float = 0.6
    bm25_weight: float = 0.4


class RerankSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    user_id: str | None = None
    chunk_text: str
    chunk_index: int
    token_count: int | None = None
    page_number: int | None = None
    section_title: str | None = None
    document_name: str | None = None
    category: str | None = None

    vector_score: float = 0.0
    bm25_score: float = 0.0
    normalized_vector_score: float = 0.0
    normalized_bm25_score: float = 0.0
    hybrid_score: float = 0.0
    retrieval_sources: list[str] = []

    reranker_score: float
    reranker_model_name: str


class RerankSearchResponse(BaseModel):
    user_id: str
    query: str
    top_k: int
    result_count: int
    results: list[RerankSearchResult]


class RetrievalDiagnosticsRequest(BaseModel):
    user_id: str | None = None
    query: str
    vector_top_k: int = 10
    bm25_top_k: int = 10
    hybrid_top_k: int = 10
    rerank_top_k: int = 5


class RetrievalDiagnosticsResult(BaseModel):
    chunk_id: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    category: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    section_title: str | None = None
    token_count: int | None = None
    chunk_text_preview: str
    distance: float | None = None
    similarity_score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    normalized_vector_score: float | None = None
    normalized_bm25_score: float | None = None
    hybrid_score: float | None = None
    reranker_score: float | None = None
    retrieval_sources: list[str] | None = None


class RetrievalRankChange(BaseModel):
    chunk_id: str
    before_rank: int
    after_rank: int
    rank_delta: int
    document_name: str | None = None
    section_title: str | None = None
    hybrid_score: float | None = None
    reranker_score: float | None = None


class RetrievalDiagnosticsSummary(BaseModel):
    vector_count: int
    bm25_count: int
    hybrid_count: int
    reranked_count: int
    overlap_vector_bm25: int
    overlap_hybrid_rerank: int


class RetrievalDiagnosticsResponse(BaseModel):
    user_id: str
    query: str
    vector_results: list[RetrievalDiagnosticsResult]
    bm25_results: list[RetrievalDiagnosticsResult]
    hybrid_results: list[RetrievalDiagnosticsResult]
    reranked_results: list[RetrievalDiagnosticsResult]
    rank_changes: list[RetrievalRankChange]
    summary: RetrievalDiagnosticsSummary
