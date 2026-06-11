"""Tests for reranking endpoint with Day 20 authorization."""

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.main import app
from app.db.database import get_db
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.reranking.reranking_service import rerank_hybrid_results
from conftest import override_get_db, override_get_current_user, FakeDB


client = TestClient(app)


def setup_auth_overrides(user_id: str = "test-user"):
    """Setup both DB and auth overrides."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user(user_id)


class FakeCrossEncoderModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        scores = []

        for _query, text in pairs:
            if "best match" in text:
                scores.append(0.95)
            elif "medium match" in text:
                scores.append(0.60)
            else:
                scores.append(0.20)

        return scores


def test_cross_encoder_reranker_returns_empty_list_when_candidates_empty(monkeypatch):
    monkeypatch.setattr(
        "app.reranking.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoderModel,
    )

    reranker = CrossEncoderReranker("fake-model")

    results = reranker.rerank(
        query="urgent care copay",
        candidates=[],
        top_k=8,
    )

    assert results == []


def test_cross_encoder_reranker_raises_value_error_for_empty_query(monkeypatch):
    monkeypatch.setattr(
        "app.reranking.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoderModel,
    )

    reranker = CrossEncoderReranker("fake-model")

    with pytest.raises(ValueError, match="query is required"):
        reranker.rerank(
            query="",
            candidates=[
                {
                    "chunk_id": "chunk-1",
                    "chunk_text": "best match text",
                }
            ],
            top_k=8,
        )


def test_cross_encoder_reranker_sorts_by_reranker_score_descending(monkeypatch):
    monkeypatch.setattr(
        "app.reranking.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoderModel,
    )

    reranker = CrossEncoderReranker("fake-model")

    candidates = [
        {
            "chunk_id": "chunk-1",
            "chunk_text": "low match text",
        },
        {
            "chunk_id": "chunk-2",
            "chunk_text": "best match text",
        },
        {
            "chunk_id": "chunk-3",
            "chunk_text": "medium match text",
        },
    ]

    results = reranker.rerank(
        query="urgent care copay",
        candidates=candidates,
        top_k=3,
    )

    assert results[0]["chunk_id"] == "chunk-2"
    assert results[1]["chunk_id"] == "chunk-3"
    assert results[2]["chunk_id"] == "chunk-1"


def test_cross_encoder_reranker_adds_score_and_model_name(monkeypatch):
    monkeypatch.setattr(
        "app.reranking.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoderModel,
    )

    reranker = CrossEncoderReranker("fake-model")

    results = reranker.rerank(
        query="urgent care copay",
        candidates=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "best match text",
            }
        ],
        top_k=1,
    )

    assert "reranker_score" in results[0]
    assert "reranker_model_name" in results[0]
    assert results[0]["reranker_model_name"] == "fake-model"
    assert results[0]["reranker_score"] == 0.95


def test_rerank_hybrid_results_returns_empty_list_when_hybrid_empty(monkeypatch):
    def fake_hybrid_search(**kwargs):
        return []

    monkeypatch.setattr(
        "app.reranking.reranking_service.hybrid_search",
        fake_hybrid_search,
    )

    results = rerank_hybrid_results(
        db=FakeDB(),
        user_id="user-1",
        query="urgent care copay",
    )

    assert results == []


def test_rerank_hybrid_results_calls_hybrid_search(monkeypatch):
    called = {"hybrid_search": False}

    def fake_hybrid_search(**kwargs):
        called["hybrid_search"] = True
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "user_id": "user-1",
                "chunk_text": "best match text",
                "chunk_index": 0,
                "token_count": 10,
                "page_number": None,
                "section_title": None,
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "vector_score": 0.90,
                "bm25_score": 1.20,
                "normalized_vector_score": 1.0,
                "normalized_bm25_score": 1.0,
                "hybrid_score": 1.0,
                "retrieval_sources": ["vector", "bm25"],
            }
        ]

    class FakeReranker:
        def __init__(self, model_name):
            self.model_name = model_name

        def rerank(self, query, candidates, top_k=8):
            result = candidates[0].copy()
            result["reranker_score"] = 0.95
            result["reranker_model_name"] = self.model_name
            return [result]

    monkeypatch.setattr(
        "app.reranking.reranking_service.hybrid_search",
        fake_hybrid_search,
    )

    monkeypatch.setattr(
        "app.reranking.reranking_service.CrossEncoderReranker",
        FakeReranker,
    )

    results = rerank_hybrid_results(
        db=FakeDB(),
        user_id="user-1",
        query="urgent care copay",
    )

    assert called["hybrid_search"] is True
    assert len(results) == 1
    assert results[0]["reranker_score"] == 0.95


def test_rerank_endpoint_success(monkeypatch):
    """Rerank search returns results for authenticated user."""
    setup_auth_overrides("local-user-123")

    def fake_rerank_hybrid_results(db, user_id: str, query: str, **kwargs):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "user_id": user_id,
                "chunk_text": "Urgent care copay is $50.",
                "chunk_index": 0,
                "reranker_score": 0.98,
                "reranker_model_name": "fake-reranker",
            }
        ]

    monkeypatch.setattr(
        "app.api.retrieval_routes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    response = client.post(
        "/search/rerank",
        json={
            "query": "urgent care copay",
            "top_k": 8,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "local-user-123"
    assert data["result_count"] == 1
    assert data["results"][0]["reranker_score"] == 0.98


def test_rerank_endpoint_empty_query_validation():
    """Rerank search rejects empty query."""
    setup_auth_overrides("test-user")

    response = client.post(
        "/search/rerank",
        json={
            "query": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "query is required"


def test_rerank_endpoint_top_k_capped_at_10(monkeypatch):
    """Rerank search caps top_k at 10."""
    setup_auth_overrides("test-user")

    captured = {}

    def fake_rerank_hybrid_results(db, user_id: str, query: str, top_k: int = 5, **kwargs):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    response = client.post(
        "/search/rerank",
        json={
            "query": "test",
            "top_k": 50,
        },
    )

    assert response.status_code == 200
    assert captured["top_k"] == 10


def test_rerank_endpoint_no_results_returns_empty_list(monkeypatch):
    """Rerank search handles no results gracefully."""
    setup_auth_overrides("test-user")

    def fake_rerank_hybrid_results(db, user_id: str, query: str, **kwargs):
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    response = client.post(
        "/search/rerank",
        json={"query": "nonexistent", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["result_count"] == 0
    assert data["results"] == []


def test_rerank_endpoint_without_auth_returns_401():
    """Rerank search requires authentication."""
    # Don't setup overrides - test unauthenticated access
    response = client.post(
        "/search/rerank",
        json={"query": "test", "top_k": 5},
    )

    assert response.status_code == 401


app.dependency_overrides.clear()
