from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.main import app


@dataclass
class FakeUser:
    id: str
    email: str = "user@example.com"
    full_name: str = "Test User"
    is_active: bool = True
    auth_provider: str = "local"
    provider_user_id: str | None = None


@pytest.fixture(autouse=True)
def clear_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def override_user(user_id: str):
    def _override():
        return FakeUser(id=user_id)

    return _override


def test_get_documents_without_token_returns_401(client):
    response = client.get("/documents")

    assert response.status_code == 401


def test_chat_ask_without_token_returns_401(client):
    response = client.post(
        "/chat/ask",
        json={
            "question": "Does my policy cover urgent care?",
            "top_k": 5,
        },
    )

    assert response.status_code == 401


def test_vector_search_without_token_returns_401(client):
    response = client.post(
        "/search/vector",
        json={
            "query": "urgent care",
            "top_k": 5,
        },
    )

    assert response.status_code == 401


def test_auth_me_without_token_returns_401(client):
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_register_route_remains_public(client, monkeypatch):
    # This checks the route is not protected.
    # If your register route needs DB and fails with validation/db error,
    # it should still not be a 401 auth error.
    response = client.post(
        "/auth/register",
        json={
            "email": "public-register-test@example.com",
            "password": "password123",
            "full_name": "Public Register Test",
        },
    )

    assert response.status_code != 401


def test_login_route_remains_public(client):
    response = client.post(
        "/auth/login",
        json={
            "email": "missing@example.com",
            "password": "wrongpassword",
        },
    )

    assert response.status_code != 401


def test_chat_ask_uses_current_user_not_body_user_id(client, monkeypatch):
    app.dependency_overrides[get_current_user] = override_user("real-user-id")

    captured = {}

    def fake_run_rag_workflow(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        if kwargs:
            user_id = kwargs.get("user_id")
            question = kwargs.get("question")
        else:
            state = args[0]
            user_id = state.get("user_id")
            question = state.get("question")

        return {
            "user_id": user_id,
            "question": question,
            "answer": "Test answer",
            "session_id": "session-1",
            "citations": [],
            "metadata": {},
        }

    monkeypatch.setattr(
        "app.api.chat_routes.run_rag_workflow",
        fake_run_rag_workflow,
    )

    response = client.post(
        "/chat/ask",
        json={
            "user_id": "malicious-user-id",
            "question": "Does my policy cover urgent care?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == "real-user-id"


def test_vector_search_uses_current_user_not_body_user_id(client, monkeypatch):
    app.dependency_overrides[get_current_user] = override_user("real-user-id")

    captured = {}

    def fake_vector_search(*args, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
        raising=False,
    )

    response = client.post(
        "/search/vector",
        json={
            "user_id": "malicious-user-id",
            "query": "urgent care",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert captured["kwargs"]["user_id"] == "real-user-id"


def test_hybrid_search_ignores_malicious_body_user_id(client, monkeypatch):
    app.dependency_overrides[get_current_user] = override_user("real-user-id")

    captured = {}

    def fake_hybrid_search(*args, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.hybrid_search",
        fake_hybrid_search,
        raising=False,
    )

    response = client.post(
        "/search/hybrid",
        json={
            "user_id": "other-user-id",
            "query": "urgent care coverage",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert captured["kwargs"]["user_id"] == "real-user-id"


def test_rerank_search_ignores_malicious_body_user_id(client, monkeypatch):
    app.dependency_overrides[get_current_user] = override_user("real-user-id")

    captured = {}

    def fake_rerank_search(*args, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.rerank_search",
        fake_rerank_search,
        raising=False,
    )

    response = client.post(
        "/search/rerank",
        json={
            "user_id": "other-user-id",
            "query": "urgent care coverage",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert captured["kwargs"]["user_id"] == "real-user-id"