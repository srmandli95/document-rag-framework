from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    user_id: str
    question: str

    top_k: int = Field(default=5, ge=1, le=50)
    hybrid_top_k: int = Field(default=20, ge=1, le=100)
    vector_top_k: int = Field(default=20, ge=1, le=100)
    bm25_top_k: int = Field(default=20, ge=1, le=100)

    min_reranker_score: float | None = None


class AskResponse(BaseModel):
    user_id: str
    question: str
    rewritten_question: str | None = None

    answer: str
    citations: list[dict[str, Any]] = []

    validation_status: str | None = None
    validation_reason: str | None = None

    evidence_sufficient: bool | None = None
    evidence_sufficiency_reason: str | None = None

    model_name: str | None = None
    status: str | None = None