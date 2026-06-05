from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    user_id: str
    question: str
    top_k: int = Field(default=5, ge=1)
    hybrid_top_k: int = Field(default=20, ge=1)
    vector_top_k: int = Field(default=20, ge=1)
    bm25_top_k: int = Field(default=20, ge=1)


class Citation(BaseModel):
    chunk_id: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    category: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    chunk_index: int | None = None
    reranker_score: float | None = None
    hybrid_score: float | None = None


class AskResponse(BaseModel):
    user_id: str
    question: str
    answer: str
    citations: list[Citation]
    evidence_chunk_count: int
    model_name: str
    status: str
    validation_status: str
    validation_reason: str