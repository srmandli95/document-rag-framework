"""Tests for vector search endpoint with Day 20 authorization."""

from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.main import app
from conftest import FakeDB, FakeUser, override_get_db, override_get_current_user


client = TestClient(app)


def setup_auth_overrides(user_id: str = "test-user"):
    """Setup both DB and auth overrides."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user(user_id)


def test_vector_search_endpoint_success(monkeypatch):
    """Vector search returns results for authenticated user."""
    setup_auth_overrides("local-user-123")

    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "user_id": user_id,
                "chunk_text": "Urgent care visits are covered after copay.",
                "chunk_index": 0,
                "token_count": 8,
                "page_number": None,
                "section_title": "Urgent Care",
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "distance": 0.12,
                "similarity_score": 0.88,
            }
        ]

    # Patch where the function is USED (imported), not where it's defined
    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/search/vector",
        json={
            "query": "Does my health insurance cover urgent care?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "local-user-123"
    assert data["query"] == "Does my health insurance cover urgent care?"
    assert data["top_k"] == 5
    assert data["result_count"] == 1
    assert data["results"][0]["chunk_id"] == "chunk-1"
    assert data["results"][0]["document_id"] == "doc-1"
    assert data["results"][0]["user_id"] == "local-user-123"
    assert data["results"][0]["similarity_score"] == 0.88


def test_vector_search_endpoint_empty_query_returns_400(monkeypatch):
    """Vector search rejects empty query."""
    setup_auth_overrides("test-user")

    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/search/vector",
        json={
            "query": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "query is required"


def test_vector_search_endpoint_missing_query_returns_422():
    """Vector search requires query in request."""
    setup_auth_overrides("test-user")

    response = client.post(
        "/search/vector",
        json={
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_vector_search_endpoint_rejects_top_k_greater_than_20(monkeypatch):
    """Vector request validation rejects top_k values greater than 20."""
    setup_auth_overrides("test-user")

    captured_args = {}

    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        captured_args["top_k"] = top_k
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/search/vector",
        json={
            "query": "test",
            "top_k": 100,
        },
    )

    assert response.status_code == 422
    assert "top_k" not in captured_args


def test_vector_search_endpoint_returns_result_count_correctly(monkeypatch):
    """Vector search returns correct result count."""
    setup_auth_overrides("test-user")

    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return [
            {
                "chunk_id": f"chunk-{i}",
                "document_id": "doc-1",
                "user_id": user_id,
                "chunk_text": f"Result {i}",
                "chunk_index": i,
                "distance": i * 0.1,
                "similarity_score": 0.9 - i * 0.1,
            }
            for i in range(3)
        ]

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/search/vector",
        json={"query": "test", "top_k": 5},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_count"] == 3
    assert len(data["results"]) == 3


def test_vector_search_endpoint_no_results_returns_empty_list(monkeypatch):
    """Vector search handles no results gracefully."""
    setup_auth_overrides("test-user")

    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/search/vector",
        json={"query": "nonexistent", "top_k": 5},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_count"] == 0
    assert data["results"] == []


def test_vector_search_endpoint_without_auth_returns_401():
    """Vector search requires authentication."""
    # Don't setup overrides - test unauthenticated access
    response = client.post(
        "/search/vector",
        json={"query": "test", "top_k": 5},
    )

    assert response.status_code == 401


app.dependency_overrides.clear()
