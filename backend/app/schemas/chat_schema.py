from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """API request schema for asking a chat question."""
    user_id: str | None = None
    question: str
    session_id: str | None = None

    top_k: int = 5
    hybrid_top_k: int = 20
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 8
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    min_reranker_score: float | None = None


class AskResponse(BaseModel):
    """API response schema for an answered chat question."""
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
    grounding_status: str | None = None
    grounding_reason: str | None = None
    unsupported_claims: list[str] = Field(default_factory=list)

    evidence_sufficient: bool | None = None
    evidence_sufficiency_reason: str | None = None

    session_id: str | None = None
    message_id: str | None = None


class ChatSessionResponse(BaseModel):
    """API response schema for a chat session summary."""
    session_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    """API response schema for a chat message."""
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


class ChatMessageEvidenceResponse(BaseModel):
    """API response schema for message evidence chunks."""
    message_id: str
    session_id: str
    user_id: str
    question: str
    answer: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    evidence_chunk_count: int = 0
    created_at: datetime


class ChatSessionListResponse(BaseModel):
    """API response schema for listing chat sessions."""
    user_id: str
    sessions: list[ChatSessionResponse]


class ChatSessionDeleteResponse(BaseModel):
    """API response schema for deleting a chat session."""
    session_id: str
    user_id: str
    message: str


class ChatSessionDetailResponse(BaseModel):
    """API response schema for a chat session with messages."""
    session_id: str
    user_id: str
    title: str
    messages: list[ChatMessageResponse]
