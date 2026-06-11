from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    user_id: str | None = None
    question: str
    session_id: str | None = None

    top_k: int = 5
    hybrid_top_k: int = 20
    vector_top_k: int = 20
    bm25_top_k: int = 20
    min_reranker_score: float | None = None


class AskResponse(BaseModel):
    user_id: str
    question: str
    rewritten_question: str | None = None

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)

    evidence_chunk_count: int = 0
    model_name: str | None = None
    status: str | None = None

    validation_status: str | None = None
    validation_reason: str | None = None

    evidence_sufficient: bool | None = None
    evidence_sufficiency_reason: str | None = None

    session_id: str | None = None
    message_id: str | None = None


class ChatSessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    message_id: str
    session_id: str
    user_id: str

    question: str
    rewritten_question: str | None = None
    answer: str | None = None

    citations: list[dict[str, Any]] = Field(default_factory=list)

    evidence_chunk_count: int = 0
    model_name: str | None = None
    status: str | None = None

    validation_status: str | None = None
    validation_reason: str | None = None

    evidence_sufficient: bool | None = None
    evidence_sufficiency_reason: str | None = None

    created_at: datetime


class ChatSessionListResponse(BaseModel):
    user_id: str
    sessions: list[ChatSessionResponse]


class ChatSessionDetailResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    messages: list[ChatMessageResponse]