import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.reranking.reranking_service import rerank_hybrid_results


client = TestClient(app)


class FakeDB:
    pass


def override_get_db():
    yield FakeDB()


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
    app.dependency_overrides[get_db] = override_get_db

    def fake_rerank_hybrid_results(**kwargs):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "user_id": "local-user-123",
                "chunk_text": "Urgent care copay is $50.",
                "chunk_index": 0,
                "token_count": 12,
                "page_number": None,
                "section_title": "Benefits",
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "vector_score": 0.90,
                "bm25_score": 1.20,
                "normalized_vector_score": 1.0,
                "normalized_bm25_score": 1.0,
                "hybrid_score": 1.0,
                "retrieval_sources": ["vector", "bm25"],
                "reranker_score": 0.98,
                "reranker_model_name": "fake-reranker",
            }
        ]

    monkeypatch.setattr(
        "app.api.retrieval_routes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    response = client.post(
        "/retrieval/rerank-search",
        json={
            "user_id": "local-user-123",
            "query": "urgent care copay",
            "top_k": 8,
            "hybrid_top_k": 20,
            "vector_top_k": 20,
            "bm25_top_k": 20,
            "vector_weight": 0.6,
            "bm25_weight": 0.4,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "local-user-123"
    assert data["query"] == "urgent care copay"
    assert data["top_k"] == 8
    assert data["result_count"] == 1
    assert data["results"][0]["reranker_score"] == 0.98
    assert data["results"][0]["reranker_model_name"] == "fake-reranker"


def test_rerank_endpoint_empty_user_id_validation():
    response = client.post(
        "/retrieval/rerank-search",
        json={
            "user_id": "",
            "query": "urgent care copay",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "user_id is required"


def test_rerank_endpoint_empty_query_validation():
    response = client.post(
        "/retrieval/rerank-search",
        json={
            "user_id": "local-user-123",
            "query": "",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "query is required"


def test_rerank_endpoint_top_k_greater_than_10_is_capped(monkeypatch):
    app.dependency_overrides[get_db] = override_get_db

    captured = {}

    def fake_rerank_hybrid_results(**kwargs):
        captured["top_k"] = kwargs["top_k"]
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    response = client.post(
        "/retrieval/rerank-search",
        json={
            "user_id": "local-user-123",
            "query": "urgent care copay",
            "top_k": 50,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured["top_k"] == 10
    assert response.json()["top_k"] == 10


def test_rerank_endpoint_both_weights_zero_returns_400():
    response = client.post(
        "/retrieval/rerank-search",
        json={
            "user_id": "local-user-123",
            "query": "urgent care copay",
            "vector_weight": 0,
            "bm25_weight": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "vector_weight and bm25_weight cannot both be 0"