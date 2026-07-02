from datetime import datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _build_default_title(question: str | None) -> str:
    """Build a default chat title from the first question."""
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
    """Create and persist a new chat session."""
    session = ChatSession(
        user_id=user_id,
        title=title or "New Chat",
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)

    logger.info("Chat session created: user_id=%s session_id=%s", user_id, session.id)
    return session


async def get_chat_session(
    db: AsyncSession,
    session_id: str,
    user_id: str,
) -> ChatSession | None:
    """Return one chat session owned by a user."""
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
    """Return an existing session or create one for a new chat."""
    if session_id:
        existing_session = await get_chat_session(
            db=db,
            session_id=session_id,
            user_id=user_id,
        )

        if existing_session is None:
            logger.warning(
                "Chat session lookup failed: user_id=%s session_id=%s",
                user_id,
                session_id,
            )
            raise ValueError("Chat session not found for this user.")

        logger.debug("Chat session reused: user_id=%s session_id=%s", user_id, session_id)
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
    """Persist a chat message and its generated answer metadata."""
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

    logger.info(
        "Chat message created: user_id=%s session_id=%s message_id=%s status=%s",
        user_id,
        session_id,
        message.id,
        message.status,
    )
    return message


async def get_chat_sessions_by_user(
    db: AsyncSession,
    user_id: str,
) -> list[ChatSession]:
    """Return all chat sessions owned by a user."""
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
    """Return messages in an owned chat session."""
    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user_id,
        )
        .order_by(ChatMessage.created_at.asc())
    )

    return list(result.scalars().all())


async def get_chat_message_by_id(
    db: AsyncSession,
    message_id: str,
    user_id: str,
) -> ChatMessage | None:
    """Return one chat message owned by a user."""
    result = await db.execute(
        select(ChatMessage).where(
            ChatMessage.id == message_id,
            ChatMessage.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


async def delete_chat_session(
    db: AsyncSession,
    session_id: str,
    user_id: str,
) -> ChatSession | None:
    """Delete an owned chat session and its messages."""
    chat_session = await get_chat_session(
        db=db,
        session_id=session_id,
        user_id=user_id,
    )

    if chat_session is None:
        logger.warning(
            "Chat session delete skipped; session not found: user_id=%s session_id=%s",
            user_id,
            session_id,
        )
        return None

    await db.execute(
        delete(ChatMessage).where(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user_id,
        )
    )
    await db.delete(chat_session)
    await db.commit()

    logger.info("Chat session deleted: user_id=%s session_id=%s", user_id, session_id)
    return chat_session
