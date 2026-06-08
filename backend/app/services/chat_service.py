from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession


def _build_default_title(question: str | None) -> str:
    if not question:
        return "New Chat"

    cleaned_question = question.strip()
    if not cleaned_question:
        return "New Chat"

    return cleaned_question[:60]


async def create_chat_session(
    db: AsyncSession,
    user_id: str,
    title: str | None = None,
) -> ChatSession:
    session = ChatSession(
        user_id=user_id,
        title=title or "New Chat",
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)

    return session


async def get_chat_session(
    db: AsyncSession,
    session_id: str,
    user_id: str,
) -> ChatSession | None:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


async def get_or_create_chat_session(
    db: AsyncSession,
    user_id: str,
    session_id: str | None,
    question: str,
) -> ChatSession:
    if session_id:
        existing_session = await get_chat_session(
            db=db,
            session_id=session_id,
            user_id=user_id,
        )

        if existing_session is None:
            raise ValueError("Chat session not found for this user.")

        return existing_session

    title = _build_default_title(question)

    return await create_chat_session(
        db=db,
        user_id=user_id,
        title=title,
    )


async def create_chat_message(
    db: AsyncSession,
    session_id: str,
    user_id: str,
    question: str,
    answer_response: dict[str, Any],
) -> ChatMessage:
    evidence_chunks = answer_response.get("evidence_chunks") or []
    citations = answer_response.get("citations") or []

    message = ChatMessage(
        session_id=session_id,
        user_id=user_id,
        question=question,
        rewritten_question=answer_response.get("rewritten_question"),
        answer=answer_response.get("answer") or answer_response.get("final_answer"),
        citations=citations,
        retrieved_chunks=evidence_chunks,
        evidence_chunk_count=answer_response.get("evidence_chunk_count") or len(evidence_chunks),
        model_name=answer_response.get("model_name"),
        status=answer_response.get("status"),
        validation_status=answer_response.get("validation_status"),
        validation_reason=answer_response.get("validation_reason"),
        evidence_sufficient=answer_response.get("evidence_sufficient"),
        evidence_sufficiency_reason=answer_response.get("evidence_sufficiency_reason"),
    )

    db.add(message)

    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    chat_session = session_result.scalar_one_or_none()

    if chat_session is not None:
        chat_session.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(message)

    return message


async def get_chat_sessions_by_user(
    db: AsyncSession,
    user_id: str,
) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.created_at.desc())
    )

    return list(result.scalars().all())


async def get_chat_messages_by_session(
    db: AsyncSession,
    session_id: str,
    user_id: str,
) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user_id,
        )
        .order_by(ChatMessage.created_at.asc())
    )

    return list(result.scalars().all())