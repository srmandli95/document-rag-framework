from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db.database import get_async_db, get_db
from app.graph.rag_graph import run_rag_workflow
from app.schemas.chat_schema import (
    AskRequest,
    AskResponse,
    ChatMessageResponse,
    ChatSessionDetailResponse,
    ChatSessionListResponse,
    ChatSessionResponse,
)
from app.services import chat_service


router = APIRouter(prefix="/chat", tags=["Chat"])


def _to_chat_session_response(session: Any) -> ChatSessionResponse:
    return ChatSessionResponse(
        session_id=session.id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _to_chat_message_response(message: Any) -> ChatMessageResponse:
    return ChatMessageResponse(
        message_id=message.id,
        session_id=message.session_id,
        user_id=message.user_id,
        question=message.question,
        rewritten_question=message.rewritten_question,
        answer=message.answer,
        citations=message.citations or [],
        evidence_chunk_count=message.evidence_chunk_count or 0,
        model_name=message.model_name,
        status=message.status,
        validation_status=message.validation_status,
        validation_reason=message.validation_reason,
        evidence_sufficient=message.evidence_sufficient,
        evidence_sufficiency_reason=message.evidence_sufficiency_reason,
        created_at=message.created_at,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    async_db: AsyncSession = Depends(get_async_db),
    db: Session = Depends(get_db),
) -> AskResponse:
    """
    Graph-backed answer-generation endpoint with async chat persistence.

    This endpoint:
    - accepts a user question
    - creates or reuses a chat session asynchronously
    - sends the request into the LangGraph RAG workflow
    - optionally rewrites the query for better retrieval
    - retrieves and reranks evidence chunks
    - checks evidence sufficiency before answer generation
    - skips LLM answer generation when evidence is weak
    - generates a grounded answer when evidence is sufficient
    - validates citation support
    - saves the chat message asynchronously
    - returns answer, citation metadata, session_id, and message_id
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question is required",
        )

    user_id = request.user_id.strip()
    question = request.question.strip()

    safe_top_k = min(request.top_k, 8)
    safe_hybrid_top_k = min(request.hybrid_top_k, 50)
    safe_vector_top_k = min(request.vector_top_k, 50)
    safe_bm25_top_k = min(request.bm25_top_k, 50)

    try:
        chat_session = await chat_service.get_or_create_chat_session(
            db=async_db,
            user_id=user_id,
            session_id=request.session_id,
            question=question,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    try:
        result = run_rag_workflow(
            db=db,
            user_id=user_id,
            question=question,
            top_k=safe_top_k,
            hybrid_top_k=safe_hybrid_top_k,
            vector_top_k=safe_vector_top_k,
            bm25_top_k=safe_bm25_top_k,
            min_reranker_score=request.min_reranker_score,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    chat_message = await chat_service.create_chat_message(
        db=async_db,
        session_id=chat_session.id,
        user_id=user_id,
        question=question,
        answer_response=result,
    )

    return AskResponse(
        user_id=result.get("user_id") or user_id,
        question=result.get("question") or question,
        rewritten_question=result.get("rewritten_question"),
        answer=result.get("answer") or "",
        citations=result.get("citations") or [],
        evidence_chunk_count=result.get("evidence_chunk_count") or len(
            result.get("evidence_chunks") or []
        ),
        model_name=result.get("model_name"),
        status=result.get("status"),
        validation_status=result.get("validation_status"),
        validation_reason=result.get("validation_reason"),
        evidence_sufficient=result.get("evidence_sufficient"),
        evidence_sufficiency_reason=result.get("evidence_sufficiency_reason"),
        session_id=chat_session.id,
        message_id=chat_message.id,
    )


@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_chat_sessions(
    user_id: str = Query(...),
    async_db: AsyncSession = Depends(get_async_db),
) -> ChatSessionListResponse:
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    clean_user_id = user_id.strip()

    sessions = await chat_service.get_chat_sessions_by_user(
        db=async_db,
        user_id=clean_user_id,
    )

    return ChatSessionListResponse(
        user_id=clean_user_id,
        sessions=[
            _to_chat_session_response(session)
            for session in sessions
        ],
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailResponse)
async def get_chat_session_detail(
    session_id: str,
    user_id: str = Query(...),
    async_db: AsyncSession = Depends(get_async_db),
) -> ChatSessionDetailResponse:
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    clean_user_id = user_id.strip()

    chat_session = await chat_service.get_chat_session(
        db=async_db,
        session_id=session_id,
        user_id=clean_user_id,
    )

    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )

    messages = await chat_service.get_chat_messages_by_session(
        db=async_db,
        session_id=session_id,
        user_id=clean_user_id,
    )

    return ChatSessionDetailResponse(
        session_id=chat_session.id,
        user_id=chat_session.user_id,
        title=chat_session.title,
        messages=[
            _to_chat_message_response(message)
            for message in messages
        ],
    )