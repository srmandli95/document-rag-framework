from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.db.database import get_async_db, get_db
from app.main import app
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession
from app.services import chat_service
from conftest import override_get_current_user


class FakeScalarResult:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class FakeExecuteResult:
    def __init__(self, scalar_one=None, scalar_list=None):
        self._scalar_one = scalar_one
        self._scalar_list = scalar_list or []

    def scalar_one_or_none(self):
        return self._scalar_one

    def scalars(self):
        return FakeScalarResult(self._scalar_list)


class FakeAsyncDB:
    def __init__(self, execute_results=None):
        self.added = []
        self.committed = False
        self.refreshed = []
        self.execute_results = execute_results or []
        self.execute_calls = []
        self.deleted = []

    def add(self, record):
        self.added.append(record)

    async def commit(self):
        self.committed = True

    async def refresh(self, record):
        self.refreshed.append(record)

    async def execute(self, statement):
        self.execute_calls.append(statement)

        if self.execute_results:
            return self.execute_results.pop(0)

        return FakeExecuteResult()

    async def delete(self, record):
        self.deleted.append(record)


@pytest.mark.asyncio
async def test_create_chat_session_creates_session():
    db = FakeAsyncDB()

    session = await chat_service.create_chat_session(
        db=db,
        user_id="user-1",
        title="My first question",
    )

    assert isinstance(session, ChatSession)
    assert session.user_id == "user-1"
    assert session.title == "My first question"
    assert db.added[0] == session
    assert db.committed is True
    assert db.refreshed[0] == session


@pytest.mark.asyncio
async def test_create_chat_session_uses_new_chat_title_when_title_missing():
    db = FakeAsyncDB()

    session = await chat_service.create_chat_session(
        db=db,
        user_id="user-1",
        title=None,
    )

    assert isinstance(session, ChatSession)
    assert session.user_id == "user-1"
    assert session.title == "New Chat"
    assert db.committed is True


@pytest.mark.asyncio
async def test_get_chat_session_returns_matching_user_session():
    existing_session = ChatSession(
        id="session-1",
        user_id="user-1",
        title="Test session",
    )

    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_one=existing_session),
        ]
    )

    result = await chat_service.get_chat_session(
        db=db,
        session_id="session-1",
        user_id="user-1",
    )

    assert result == existing_session


@pytest.mark.asyncio
async def test_get_chat_session_returns_none_when_not_found():
    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_one=None),
        ]
    )

    result = await chat_service.get_chat_session(
        db=db,
        session_id="missing-session",
        user_id="user-1",
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_or_create_chat_session_creates_new_session_when_session_id_missing():
    db = FakeAsyncDB()

    session = await chat_service.get_or_create_chat_session(
        db=db,
        user_id="user-1",
        session_id=None,
        question="What is covered by my policy?",
    )

    assert isinstance(session, ChatSession)
    assert session.user_id == "user-1"
    assert session.title == "What is covered by my policy?"
    assert db.committed is True


@pytest.mark.asyncio
async def test_get_or_create_chat_session_truncates_title_to_60_chars():
    db = FakeAsyncDB()

    long_question = (
        "This is a very long question that should be truncated because "
        "chat session titles should stay short"
    )

    session = await chat_service.get_or_create_chat_session(
        db=db,
        user_id="user-1",
        session_id=None,
        question=long_question,
    )

    assert isinstance(session, ChatSession)
    assert session.title == long_question[:60]
    assert len(session.title) == 60


@pytest.mark.asyncio
async def test_get_or_create_chat_session_uses_new_chat_when_question_empty():
    db = FakeAsyncDB()

    session = await chat_service.get_or_create_chat_session(
        db=db,
        user_id="user-1",
        session_id=None,
        question="   ",
    )

    assert isinstance(session, ChatSession)
    assert session.title == "New Chat"


@pytest.mark.asyncio
async def test_get_or_create_chat_session_returns_existing_session_when_owned_by_user():
    existing_session = ChatSession(
        id="session-1",
        user_id="user-1",
        title="Existing session",
    )

    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_one=existing_session),
        ]
    )

    result = await chat_service.get_or_create_chat_session(
        db=db,
        user_id="user-1",
        session_id="session-1",
        question="Follow up question",
    )

    assert result == existing_session


@pytest.mark.asyncio
async def test_get_or_create_chat_session_raises_when_session_id_not_found_for_user():
    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_one=None),
        ]
    )

    with pytest.raises(ValueError, match="Chat session not found"):
        await chat_service.get_or_create_chat_session(
            db=db,
            user_id="user-1",
            session_id="missing-session",
            question="Question",
        )


@pytest.mark.asyncio
async def test_create_chat_message_stores_answer_citations_and_validation_fields():
    existing_session = ChatSession(
        id="session-1",
        user_id="user-1",
        title="Existing session",
    )

    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_one=existing_session),
        ]
    )

    answer_response = {
        "rewritten_question": "rewritten question",
        "answer": "Supported answer",
        "citations": [{"chunk_id": "chunk-1"}],
        "evidence_chunks": [{"chunk_id": "chunk-1", "chunk_text": "Evidence"}],
        "evidence_chunk_count": 1,
        "model_name": "gpt-4o-mini",
        "status": "answered",
        "validation_status": "supported",
        "validation_reason": "Supported by evidence",
        "evidence_sufficient": True,
        "evidence_sufficiency_reason": "Enough evidence",
    }

    message = await chat_service.create_chat_message(
        db=db,
        session_id="session-1",
        user_id="user-1",
        question="Original question",
        answer_response=answer_response,
    )

    assert isinstance(message, ChatMessage)
    assert message.session_id == "session-1"
    assert message.user_id == "user-1"
    assert message.question == "Original question"
    assert message.rewritten_question == "rewritten question"
    assert message.answer == "Supported answer"
    assert message.citations == [{"chunk_id": "chunk-1"}]
    assert message.retrieved_chunks == [{"chunk_id": "chunk-1", "chunk_text": "Evidence"}]
    assert message.evidence_chunk_count == 1
    assert message.model_name == "gpt-4o-mini"
    assert message.status == "answered"
    assert message.validation_status == "supported"
    assert message.validation_reason == "Supported by evidence"
    assert message.evidence_sufficient is True
    assert message.evidence_sufficiency_reason == "Enough evidence"
    assert db.committed is True
    assert db.refreshed[0] == message


@pytest.mark.asyncio
async def test_create_chat_message_uses_final_answer_when_answer_missing():
    existing_session = ChatSession(
        id="session-1",
        user_id="user-1",
        title="Existing session",
    )

    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_one=existing_session),
        ]
    )

    answer_response = {
        "final_answer": "Final answer fallback",
        "citations": [],
        "evidence_chunks": [],
        "status": "answered",
    }

    message = await chat_service.create_chat_message(
        db=db,
        session_id="session-1",
        user_id="user-1",
        question="Question",
        answer_response=answer_response,
    )

    assert message.answer == "Final answer fallback"


@pytest.mark.asyncio
async def test_create_chat_message_counts_evidence_chunks_when_count_missing():
    existing_session = ChatSession(
        id="session-1",
        user_id="user-1",
        title="Existing session",
    )

    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_one=existing_session),
        ]
    )

    answer_response = {
        "answer": "Answer",
        "citations": [],
        "evidence_chunks": [
            {"chunk_id": "chunk-1"},
            {"chunk_id": "chunk-2"},
        ],
        "status": "answered",
    }

    message = await chat_service.create_chat_message(
        db=db,
        session_id="session-1",
        user_id="user-1",
        question="Question",
        answer_response=answer_response,
    )

    assert message.evidence_chunk_count == 2


@pytest.mark.asyncio
async def test_get_chat_sessions_by_user_returns_sessions():
    sessions = [
        ChatSession(id="session-1", user_id="user-1", title="Session 1"),
        ChatSession(id="session-2", user_id="user-1", title="Session 2"),
    ]

    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_list=sessions),
        ]
    )

    result = await chat_service.get_chat_sessions_by_user(
        db=db,
        user_id="user-1",
    )

    assert result == sessions


@pytest.mark.asyncio
async def test_get_chat_messages_by_session_returns_messages_scoped_to_user():
    messages = [
        ChatMessage(
            id="message-1",
            session_id="session-1",
            user_id="user-1",
            question="Question 1",
            answer="Answer 1",
        ),
        ChatMessage(
            id="message-2",
            session_id="session-1",
            user_id="user-1",
            question="Question 2",
            answer="Answer 2",
        ),
    ]

    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_list=messages),
        ]
    )

    result = await chat_service.get_chat_messages_by_session(
        db=db,
        session_id="session-1",
        user_id="user-1",
    )

    assert result == messages


@pytest.mark.asyncio
async def test_delete_chat_session_removes_owned_session_and_messages():
    existing_session = ChatSession(
        id="session-1",
        user_id="user-1",
        title="Existing session",
    )
    db = FakeAsyncDB(
        execute_results=[
            FakeExecuteResult(scalar_one=existing_session),
            FakeExecuteResult(),
        ]
    )

    deleted = await chat_service.delete_chat_session(db, "session-1", "user-1")

    assert deleted == existing_session
    assert db.deleted == [existing_session]
    assert db.committed is True
    assert len(db.execute_calls) == 2


@pytest.mark.asyncio
async def test_delete_chat_session_returns_none_for_other_user():
    db = FakeAsyncDB(execute_results=[FakeExecuteResult(scalar_one=None)])

    deleted = await chat_service.delete_chat_session(db, "session-1", "other-user")

    assert deleted is None
    assert db.deleted == []
    assert db.committed is False


def override_async_db():
    async def _dependency():
        yield MagicMock()

    return _dependency


def override_sync_db():
    def _dependency():
        yield MagicMock()

    return _dependency


def test_get_chat_sessions_route(monkeypatch):
    from app.api import chat_routes

    fake_sessions = [
        SimpleNamespace(
            id="session-1",
            user_id="user-1",
            title="Session 1",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    ]

    async def fake_get_chat_sessions_by_user(db, user_id):
        assert user_id == "user-1"
        return fake_sessions

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_chat_sessions_by_user",
        fake_get_chat_sessions_by_user,
    )

    app.dependency_overrides[get_async_db] = override_async_db()

    client = TestClient(app)
    response = client.get("/chat/sessions?user_id=user-1")

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == "user-1"
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["session_id"] == "session-1"
    assert data["sessions"][0]["user_id"] == "user-1"
    assert data["sessions"][0]["title"] == "Session 1"


def test_get_chat_sessions_route_requires_authentication():
    app.dependency_overrides[get_async_db] = override_async_db()

    client = TestClient(app)
    response = client.get("/chat/sessions")

    app.dependency_overrides.clear()

    assert response.status_code == 401


def test_get_chat_session_detail_route_returns_messages(monkeypatch):
    from app.api import chat_routes

    fake_session = SimpleNamespace(
        id="session-1",
        user_id="user-1",
        title="Session 1",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    fake_messages = [
        SimpleNamespace(
            id="message-1",
            session_id="session-1",
            user_id="user-1",
            question="Question",
            rewritten_question="Rewritten question",
            answer="Answer",
            citations=[],
            evidence_chunk_count=0,
            model_name="gpt-4o-mini",
            status="answered",
            validation_status="supported",
            validation_reason=None,
            evidence_sufficient=True,
            evidence_sufficiency_reason=None,
            created_at=datetime.utcnow(),
        )
    ]

    async def fake_get_chat_session(db, session_id, user_id):
        assert session_id == "session-1"
        assert user_id == "user-1"
        return fake_session

    async def fake_get_chat_messages_by_session(db, session_id, user_id):
        assert session_id == "session-1"
        assert user_id == "user-1"
        return fake_messages

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_chat_session",
        fake_get_chat_session,
    )
    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_chat_messages_by_session",
        fake_get_chat_messages_by_session,
    )

    app.dependency_overrides[get_async_db] = override_async_db()

    client = TestClient(app)
    response = client.get("/chat/sessions/session-1?user_id=user-1")

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] == "session-1"
    assert data["user_id"] == "user-1"
    assert data["title"] == "Session 1"
    assert len(data["messages"]) == 1
    assert data["messages"][0]["message_id"] == "message-1"
    assert data["messages"][0]["question"] == "Question"
    assert data["messages"][0]["answer"] == "Answer"


def test_user_cannot_access_other_user_session(monkeypatch):
    from app.api import chat_routes

    async def fake_get_chat_session(db, session_id, user_id):
        assert session_id == "session-owned-by-b"
        assert user_id == "user-a"
        return None

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_chat_session",
        fake_get_chat_session,
    )

    app.dependency_overrides[get_async_db] = override_async_db()

    client = TestClient(app)
    response = client.get("/chat/sessions/session-owned-by-b?user_id=user-a")

    app.dependency_overrides.clear()

    assert response.status_code == 404


def test_delete_chat_session_route(monkeypatch):
    from app.api import chat_routes

    fake_session = SimpleNamespace(id="session-1", user_id="user-1")

    async def fake_delete_chat_session(db, session_id, user_id):
        assert session_id == "session-1"
        assert user_id == "user-1"
        return fake_session

    monkeypatch.setattr(chat_routes.chat_service, "delete_chat_session", fake_delete_chat_session)
    app.dependency_overrides[get_async_db] = override_async_db()

    response = TestClient(app).delete("/chat/sessions/session-1?user_id=user-1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-1"


def test_user_cannot_delete_other_user_session(monkeypatch):
    from app.api import chat_routes

    async def fake_delete_chat_session(db, session_id, user_id):
        return None

    monkeypatch.setattr(chat_routes.chat_service, "delete_chat_session", fake_delete_chat_session)
    app.dependency_overrides[get_async_db] = override_async_db()

    response = TestClient(app).delete("/chat/sessions/session-owned-by-b?user_id=user-a")
    app.dependency_overrides.clear()

    assert response.status_code == 404


def test_chat_ask_creates_session_when_no_session_id(monkeypatch):
    from app.api import chat_routes

    fake_session = SimpleNamespace(
        id="session-1",
        user_id="user-1",
        title="Question",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    fake_message = SimpleNamespace(
        id="message-1",
    )

    async def fake_get_or_create_chat_session(db, user_id, session_id, question):
        assert user_id == "user-1"
        assert session_id is None
        assert question == "Question"
        return fake_session

    async def fake_create_chat_message(db, session_id, user_id, question, answer_response):
        assert session_id == "session-1"
        assert user_id == "user-1"
        assert question == "Question"
        assert answer_response["answer"] == "Fake answer"
        return fake_message

    def fake_run_rag_workflow(**kwargs):
        assert kwargs["user_id"] == "user-1"
        assert kwargs["question"] == "Question"
        assert kwargs["top_k"] == 5

        return {
            "user_id": "user-1",
            "question": "Question",
            "rewritten_question": "Question",
            "answer": "Fake answer",
            "citations": [],
            "evidence_chunks": [],
            "evidence_chunk_count": 0,
            "model_name": "fake-model",
            "status": "answered",
            "validation_status": "supported",
            "validation_reason": None,
            "evidence_sufficient": True,
            "evidence_sufficiency_reason": None,
        }

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_or_create_chat_session",
        fake_get_or_create_chat_session,
    )
    monkeypatch.setattr(
        chat_routes.chat_service,
        "create_chat_message",
        fake_create_chat_message,
    )
    monkeypatch.setattr(
        chat_routes,
        "run_rag_workflow",
        fake_run_rag_workflow,
    )

    app.dependency_overrides[get_async_db] = override_async_db()
    app.dependency_overrides[get_db] = override_sync_db()
    app.dependency_overrides[get_current_user] = override_get_current_user("user-1")

    client = TestClient(app)
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "user-1",
            "question": "Question",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == "user-1"
    assert data["question"] == "Question"
    assert data["answer"] == "Fake answer"
    assert data["session_id"] == "session-1"
    assert data["message_id"] == "message-1"
    assert data["status"] == "answered"
    assert data["validation_status"] == "supported"


def test_chat_ask_appends_to_existing_session(monkeypatch):
    from app.api import chat_routes

    fake_session = SimpleNamespace(
        id="existing-session",
        user_id="user-1",
        title="Existing Session",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    fake_message = SimpleNamespace(
        id="message-2",
    )

    async def fake_get_or_create_chat_session(db, user_id, session_id, question):
        assert user_id == "user-1"
        assert session_id == "existing-session"
        assert question == "Follow up question"
        return fake_session

    async def fake_create_chat_message(db, session_id, user_id, question, answer_response):
        assert session_id == "existing-session"
        assert user_id == "user-1"
        assert question == "Follow up question"
        assert answer_response["answer"] == "Second fake answer"
        return fake_message

    def fake_run_rag_workflow(**kwargs):
        assert kwargs["user_id"] == "user-1"
        assert kwargs["question"] == "Follow up question"

        return {
            "user_id": "user-1",
            "question": "Follow up question",
            "rewritten_question": "Follow up question",
            "answer": "Second fake answer",
            "citations": [],
            "evidence_chunks": [],
            "evidence_chunk_count": 0,
            "model_name": "fake-model",
            "status": "answered",
            "validation_status": "supported",
            "validation_reason": None,
            "evidence_sufficient": True,
            "evidence_sufficiency_reason": None,
        }

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_or_create_chat_session",
        fake_get_or_create_chat_session,
    )
    monkeypatch.setattr(
        chat_routes.chat_service,
        "create_chat_message",
        fake_create_chat_message,
    )
    monkeypatch.setattr(
        chat_routes,
        "run_rag_workflow",
        fake_run_rag_workflow,
    )

    app.dependency_overrides[get_async_db] = override_async_db()
    app.dependency_overrides[get_db] = override_sync_db()

    client = TestClient(app)
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "user-1",
            "session_id": "existing-session",
            "question": "Follow up question",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == "user-1"
    assert data["question"] == "Follow up question"
    assert data["answer"] == "Second fake answer"
    assert data["session_id"] == "existing-session"
    assert data["message_id"] == "message-2"


def test_chat_ask_returns_404_for_invalid_session_id(monkeypatch):
    from app.api import chat_routes

    async def fake_get_or_create_chat_session(db, user_id, session_id, question):
        assert session_id == "bad-session"
        raise ValueError("Chat session not found for this user.")

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_or_create_chat_session",
        fake_get_or_create_chat_session,
    )

    app.dependency_overrides[get_async_db] = override_async_db()
    app.dependency_overrides[get_db] = override_sync_db()

    client = TestClient(app)
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "user-1",
            "session_id": "bad-session",
            "question": "Question",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 404


def test_chat_ask_returns_401_when_user_identity_missing():
    app.dependency_overrides[get_async_db] = override_async_db()
    app.dependency_overrides[get_db] = override_sync_db()

    client = TestClient(app)
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "   ",
            "question": "Question",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 401


def test_chat_ask_returns_400_when_question_missing():
    app.dependency_overrides[get_async_db] = override_async_db()
    app.dependency_overrides[get_db] = override_sync_db()

    client = TestClient(app)
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "user-1",
            "question": "   ",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400


def test_chat_ask_rejects_retrieval_limits_above_maximum(monkeypatch):
    from app.api import chat_routes

    fake_session = SimpleNamespace(
        id="session-1",
        user_id="user-1",
        title="Question",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    fake_message = SimpleNamespace(id="message-1")

    async def fake_get_or_create_chat_session(db, user_id, session_id, question):
        return fake_session

    async def fake_create_chat_message(db, session_id, user_id, question, answer_response):
        return fake_message

    def fake_run_rag_workflow(**kwargs):
        assert kwargs["top_k"] == 8
        assert kwargs["hybrid_top_k"] == 50
        assert kwargs["vector_top_k"] == 50
        assert kwargs["bm25_top_k"] == 50

        return {
            "user_id": "user-1",
            "question": "Question",
            "rewritten_question": "Question",
            "answer": "Fake answer",
            "citations": [],
            "evidence_chunks": [],
            "evidence_chunk_count": 0,
            "model_name": "fake-model",
            "status": "answered",
            "validation_status": "supported",
            "validation_reason": None,
            "evidence_sufficient": True,
            "evidence_sufficiency_reason": None,
        }

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_or_create_chat_session",
        fake_get_or_create_chat_session,
    )
    monkeypatch.setattr(
        chat_routes.chat_service,
        "create_chat_message",
        fake_create_chat_message,
    )
    monkeypatch.setattr(
        chat_routes,
        "run_rag_workflow",
        fake_run_rag_workflow,
    )

    app.dependency_overrides[get_async_db] = override_async_db()
    app.dependency_overrides[get_db] = override_sync_db()

    client = TestClient(app)
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "user-1",
            "question": "Question",
            "top_k": 100,
            "hybrid_top_k": 100,
            "vector_top_k": 100,
            "bm25_top_k": 100,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["detail"] == "top_k must be between 1 and 20"


def test_chat_ask_passes_normalized_custom_retrieval_settings(monkeypatch):
    from app.api import chat_routes

    fake_session = SimpleNamespace(
        id="session-1",
        user_id="user-1",
        title="Question",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    fake_message = SimpleNamespace(id="message-1")
    captured = {}

    async def fake_get_or_create_chat_session(db, user_id, session_id, question):
        return fake_session

    async def fake_create_chat_message(db, session_id, user_id, question, answer_response):
        return fake_message

    def fake_run_rag_workflow(**kwargs):
        captured.update(kwargs)
        return {
            "user_id": "user-1",
            "question": "Question",
            "answer": "Fake answer",
            "citations": [],
            "evidence_chunks": [],
            "status": "answered",
        }

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_or_create_chat_session",
        fake_get_or_create_chat_session,
    )
    monkeypatch.setattr(
        chat_routes.chat_service,
        "create_chat_message",
        fake_create_chat_message,
    )
    monkeypatch.setattr(chat_routes, "run_rag_workflow", fake_run_rag_workflow)

    app.dependency_overrides[get_async_db] = override_async_db()
    app.dependency_overrides[get_db] = override_sync_db()
    app.dependency_overrides[get_current_user] = override_get_current_user("user-1")

    response = TestClient(app).post(
        "/chat/ask",
        json={
            "question": "Question",
            "top_k": 4,
            "rerank_top_k": 9,
            "vector_weight": 7,
            "bm25_weight": 3,
        },
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured["top_k"] == 4
    assert captured["rerank_top_k"] == 9
    assert captured["vector_weight"] == 0.7
    assert captured["bm25_weight"] == 0.3


def test_chat_ask_returns_400_when_rag_workflow_raises_value_error(monkeypatch):
    from app.api import chat_routes

    fake_session = SimpleNamespace(
        id="session-1",
        user_id="user-1",
        title="Question",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    async def fake_get_or_create_chat_session(db, user_id, session_id, question):
        return fake_session

    def fake_run_rag_workflow(**kwargs):
        raise ValueError("RAG workflow failed")

    monkeypatch.setattr(
        chat_routes.chat_service,
        "get_or_create_chat_session",
        fake_get_or_create_chat_session,
    )
    monkeypatch.setattr(
        chat_routes,
        "run_rag_workflow",
        fake_run_rag_workflow,
    )

    app.dependency_overrides[get_async_db] = override_async_db()
    app.dependency_overrides[get_db] = override_sync_db()

    client = TestClient(app)
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "user-1",
            "question": "Question",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400
