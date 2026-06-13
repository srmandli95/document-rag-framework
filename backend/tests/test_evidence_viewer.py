from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import chat_routes, document_routes
from app.auth.dependencies import get_current_user
from app.db.database import get_async_db, get_db
from app.main import app
from app.services import chat_service, document_chunk_service


NOW = datetime.utcnow()


def fake_chunk(
    chunk_id: str = "chunk-1",
    document_id: str = "document-1",
    user_id: str = "user-1",
    chunk_index: int = 0,
    status: str = "embedded",
):
    return SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        user_id=user_id,
        chunk_text="Covered source text.",
        chunk_index=chunk_index,
        token_count=3,
        page_number=2,
        section_title="Coverage",
        status=status,
        created_at=NOW,
        updated_at=NOW,
        embedding=[0.1, 0.2],
    )


def fake_message(retrieved_chunks=None):
    chunks = retrieved_chunks if retrieved_chunks is not None else [{"chunk_id": "chunk-1"}]
    return SimpleNamespace(
        id="message-1",
        session_id="session-1",
        user_id="user-1",
        question="What is covered?",
        answer="This is covered.",
        citations=[{"chunk_id": "chunk-1"}] if chunks else [],
        retrieved_chunks=chunks,
        evidence_chunk_count=len(chunks),
        created_at=NOW,
    )


class FakeQuery:
    def __init__(self, first_value=None, all_values=None):
        self.first_value = first_value
        self.all_values = all_values or []
        self.filters = []
        self.ordering = None

    def filter(self, expression):
        self.filters.append(str(expression))
        return self

    def order_by(self, expression):
        self.ordering = str(expression)
        return self

    def first(self):
        return self.first_value

    def all(self):
        return self.all_values


class FakeDB:
    def __init__(self, query):
        self.query_result = query

    def query(self, _model):
        return self.query_result


class FakeAsyncExecuteResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeAsyncDB:
    def __init__(self, value):
        self.value = value
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return FakeAsyncExecuteResult(self.value)


@pytest.fixture
def evidence_client():
    def override_db():
        yield MagicMock()

    async def override_async_db():
        yield MagicMock()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_async_db] = override_async_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="user-1")

    client = TestClient(app)
    yield client

    client.close()
    app.dependency_overrides.clear()


def test_get_chunk_by_id_returns_owner_chunk_and_scopes_query():
    chunk = fake_chunk()
    query = FakeQuery(first_value=chunk)

    result = document_chunk_service.get_chunk_by_id(FakeDB(query), "chunk-1", "user-1")

    assert result is chunk
    assert any("document_chunks.id" in item for item in query.filters)
    assert any("document_chunks.user_id" in item for item in query.filters)
    assert any("document_chunks.status" in item for item in query.filters)


def test_get_chunk_by_id_returns_none_for_wrong_user():
    query = FakeQuery(first_value=None)

    result = document_chunk_service.get_chunk_by_id(FakeDB(query), "chunk-1", "user-2")

    assert result is None
    assert any("document_chunks.user_id" in item for item in query.filters)


def test_get_chunks_by_document_for_user_is_scoped_and_ordered():
    chunks = [fake_chunk(chunk_id="chunk-1"), fake_chunk(chunk_id="chunk-2", chunk_index=1)]
    query = FakeQuery(all_values=chunks)

    result = document_chunk_service.get_chunks_by_document_for_user(
        FakeDB(query),
        "document-1",
        "user-1",
    )

    assert result == chunks
    assert any("document_chunks.document_id" in item for item in query.filters)
    assert any("document_chunks.user_id" in item for item in query.filters)
    assert any("document_chunks.status" in item for item in query.filters)
    assert "document_chunks.chunk_index" in query.ordering


@pytest.mark.asyncio
async def test_get_chat_message_by_id_is_scoped_to_user():
    message = fake_message()
    db = FakeAsyncDB(message)

    result = await chat_service.get_chat_message_by_id(db, "message-1", "user-1")

    assert result is message
    statement = str(db.statement)
    assert "chat_messages.id" in statement
    assert "chat_messages.user_id" in statement


def test_chunk_detail_route_returns_owner_chunk_without_embedding(evidence_client, monkeypatch):
    monkeypatch.setattr(document_routes, "get_chunk_by_id", lambda db, chunk_id, user_id: fake_chunk())

    response = evidence_client.get("/documents/chunks/chunk-1")

    assert response.status_code == 200
    assert response.json()["chunk_text"] == "Covered source text."
    assert "embedding" not in response.json()


def test_chunk_detail_route_returns_404_for_other_user(evidence_client, monkeypatch):
    monkeypatch.setattr(document_routes, "get_chunk_by_id", lambda db, chunk_id, user_id: None)

    response = evidence_client.get("/documents/chunks/other-user-chunk")

    assert response.status_code == 404


def test_document_chunks_route_returns_ordered_owner_chunks(evidence_client, monkeypatch):
    chunks = [fake_chunk(), fake_chunk(chunk_id="chunk-2", chunk_index=1)]
    monkeypatch.setattr(document_routes, "_get_owned_document_or_404", lambda db, document_id, user_id: object())
    monkeypatch.setattr(document_routes, "get_chunks_by_document_for_user", lambda db, document_id, user_id: chunks)

    response = evidence_client.get("/documents/document-1/chunks")

    assert response.status_code == 200
    assert [item["chunk_index"] for item in response.json()["chunks"]] == [0, 1]
    assert all("embedding" not in item for item in response.json()["chunks"])


def test_document_chunks_route_returns_404_for_other_user(evidence_client, monkeypatch):
    def not_found(db, document_id, user_id):
        raise document_routes.HTTPException(status_code=404, detail="Document not found")

    monkeypatch.setattr(document_routes, "_get_owned_document_or_404", not_found)

    response = evidence_client.get("/documents/other-user-document/chunks")

    assert response.status_code == 404


def test_message_evidence_route_returns_stored_evidence(evidence_client, monkeypatch):
    async def get_message(db, message_id, user_id):
        assert user_id == "user-1"
        return fake_message()

    monkeypatch.setattr(chat_routes.chat_service, "get_chat_message_by_id", get_message)

    response = evidence_client.get("/chat/messages/message-1/evidence")

    assert response.status_code == 200
    assert response.json()["retrieved_chunks"] == [{"chunk_id": "chunk-1"}]
    assert response.json()["evidence_chunk_count"] == 1


def test_message_evidence_route_returns_404_for_other_user(evidence_client, monkeypatch):
    async def get_message(db, message_id, user_id):
        return None

    monkeypatch.setattr(chat_routes.chat_service, "get_chat_message_by_id", get_message)

    response = evidence_client.get("/chat/messages/other-message/evidence")

    assert response.status_code == 404


def test_message_evidence_route_supports_empty_retrieved_chunks(evidence_client, monkeypatch):
    async def get_message(db, message_id, user_id):
        return fake_message(retrieved_chunks=[])

    monkeypatch.setattr(chat_routes.chat_service, "get_chat_message_by_id", get_message)

    response = evidence_client.get("/chat/messages/message-1/evidence")

    assert response.status_code == 200
    assert response.json()["retrieved_chunks"] == []
    assert response.json()["citations"] == []
    assert response.json()["evidence_chunk_count"] == 0


def test_evidence_routes_are_registered():
    paths = {route.path for route in app.routes}

    assert "/documents/chunks/{chunk_id}" in paths
    assert "/documents/{document_id}/chunks" in paths
    assert "/chat/messages/{message_id}/evidence" in paths
