from fastapi.testclient import TestClient

from app.api import retrieval_routes
from app.db.database import get_db
from app.main import app
from app.retrieval import hybrid_retriever
from app.retrieval.hybrid_retriever import hybrid_search, normalize_scores


client = TestClient(app)


class FakeDB:
    pass


def override_get_db():
    yield FakeDB()


app.dependency_overrides[get_db] = override_get_db


def make_result(
    chunk_id: str,
    score_key: str,
    score: float,
    source_text: str = "sample chunk text",
):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "user_id": "user-1",
        "chunk_text": source_text,
        "chunk_index": 0,
        "token_count": 100,
        "page_number": None,
        "section_title": None,
        "document_name": "sample_health_policy.txt",
        "category": "health_insurance",
        score_key: score,
    }


def test_min_max_normalization_works_for_normal_scores():
    scores = [2.0, 4.0, 6.0]

    normalized = normalize_scores(scores)

    assert normalized == [0.0, 0.5, 1.0]


def test_normalization_returns_one_when_all_scores_are_equal():
    scores = [5.0, 5.0, 5.0]

    normalized = normalize_scores(scores)

    assert normalized == [1.0, 1.0, 1.0]


def test_hybrid_search_merges_vector_only_results(monkeypatch):
    def fake_vector_search(db, user_id, query, top_k):
        return [
            make_result(
                chunk_id="chunk-vector-only",
                score_key="similarity_score",
                score=0.9,
            )
        ]

    def fake_bm25_search(db, user_id, query, top_k):
        return []

    monkeypatch.setattr(hybrid_retriever, "vector_search", fake_vector_search)
    monkeypatch.setattr(hybrid_retriever, "bm25_search", fake_bm25_search)

    results = hybrid_search(
        db=FakeDB(),
        user_id="user-1",
        query="urgent care",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-vector-only"
    assert results[0]["vector_score"] == 0.9
    assert results[0]["bm25_score"] == 0.0
    assert results[0]["normalized_vector_score"] == 1.0
    assert results[0]["normalized_bm25_score"] == 0.0
    assert results[0]["retrieval_sources"] == ["vector"]


def test_hybrid_search_merges_bm25_only_results(monkeypatch):
    def fake_vector_search(db, user_id, query, top_k):
        return []

    def fake_bm25_search(db, user_id, query, top_k):
        return [
            make_result(
                chunk_id="chunk-bm25-only",
                score_key="bm25_score",
                score=3.5,
            )
        ]

    monkeypatch.setattr(hybrid_retriever, "vector_search", fake_vector_search)
    monkeypatch.setattr(hybrid_retriever, "bm25_search", fake_bm25_search)

    results = hybrid_search(
        db=FakeDB(),
        user_id="user-1",
        query="urgent care",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-bm25-only"
    assert results[0]["vector_score"] == 0.0
    assert results[0]["bm25_score"] == 3.5
    assert results[0]["normalized_vector_score"] == 0.0
    assert results[0]["normalized_bm25_score"] == 1.0
    assert results[0]["retrieval_sources"] == ["bm25"]


def test_hybrid_search_deduplicates_chunk_present_in_both(monkeypatch):
    def fake_vector_search(db, user_id, query, top_k):
        return [
            make_result(
                chunk_id="same-chunk",
                score_key="similarity_score",
                score=0.8,
            )
        ]

    def fake_bm25_search(db, user_id, query, top_k):
        return [
            make_result(
                chunk_id="same-chunk",
                score_key="bm25_score",
                score=2.0,
            )
        ]

    monkeypatch.setattr(hybrid_retriever, "vector_search", fake_vector_search)
    monkeypatch.setattr(hybrid_retriever, "bm25_search", fake_bm25_search)

    results = hybrid_search(
        db=FakeDB(),
        user_id="user-1",
        query="urgent care",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "same-chunk"
    assert results[0]["retrieval_sources"] == ["vector", "bm25"]


def test_hybrid_score_uses_vector_and_bm25_weights(monkeypatch):
    def fake_vector_search(db, user_id, query, top_k):
        return [
            make_result(
                chunk_id="chunk-a",
                score_key="similarity_score",
                score=0.2,
            ),
            make_result(
                chunk_id="chunk-b",
                score_key="similarity_score",
                score=0.8,
            ),
        ]

    def fake_bm25_search(db, user_id, query, top_k):
        return [
            make_result(
                chunk_id="chunk-a",
                score_key="bm25_score",
                score=10.0,
            ),
            make_result(
                chunk_id="chunk-b",
                score_key="bm25_score",
                score=20.0,
            ),
        ]

    monkeypatch.setattr(hybrid_retriever, "vector_search", fake_vector_search)
    monkeypatch.setattr(hybrid_retriever, "bm25_search", fake_bm25_search)

    results = hybrid_search(
        db=FakeDB(),
        user_id="user-1",
        query="urgent care",
        vector_weight=0.6,
        bm25_weight=0.4,
    )

    chunk_b = next(result for result in results if result["chunk_id"] == "chunk-b")

    assert chunk_b["normalized_vector_score"] == 1.0
    assert chunk_b["normalized_bm25_score"] == 1.0
    assert chunk_b["hybrid_score"] == 1.0


def test_retrieval_sources_show_vector_bm25_or_both(monkeypatch):
    def fake_vector_search(db, user_id, query, top_k):
        return [
            make_result(
                chunk_id="vector-only",
                score_key="similarity_score",
                score=0.9,
            ),
            make_result(
                chunk_id="both",
                score_key="similarity_score",
                score=0.7,
            ),
        ]

    def fake_bm25_search(db, user_id, query, top_k):
        return [
            make_result(
                chunk_id="bm25-only",
                score_key="bm25_score",
                score=5.0,
            ),
            make_result(
                chunk_id="both",
                score_key="bm25_score",
                score=4.0,
            ),
        ]

    monkeypatch.setattr(hybrid_retriever, "vector_search", fake_vector_search)
    monkeypatch.setattr(hybrid_retriever, "bm25_search", fake_bm25_search)

    results = hybrid_search(
        db=FakeDB(),
        user_id="user-1",
        query="urgent care",
        top_k=10,
    )

    sources_by_chunk = {
        result["chunk_id"]: result["retrieval_sources"]
        for result in results
    }

    assert sources_by_chunk["vector-only"] == ["vector"]
    assert sources_by_chunk["bm25-only"] == ["bm25"]
    assert sources_by_chunk["both"] == ["vector", "bm25"]


def test_hybrid_endpoint_success_using_monkeypatch(monkeypatch):
    def fake_hybrid_search(
        db,
        user_id,
        query,
        top_k,
        vector_top_k,
        bm25_top_k,
        vector_weight,
        bm25_weight,
    ):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "user_id": user_id,
                "chunk_text": "Urgent care copay is $40.",
                "chunk_index": 0,
                "token_count": 50,
                "page_number": None,
                "section_title": None,
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "vector_score": 0.8,
                "bm25_score": 2.5,
                "normalized_vector_score": 1.0,
                "normalized_bm25_score": 1.0,
                "hybrid_score": 1.0,
                "retrieval_sources": ["vector", "bm25"],
            }
        ]

    monkeypatch.setattr(retrieval_routes, "hybrid_search", fake_hybrid_search)

    response = client.post(
        "/retrieval/hybrid-search",
        json={
            "user_id": "user-1",
            "query": "urgent care copay",
            "top_k": 5,
            "vector_top_k": 20,
            "bm25_top_k": 20,
            "vector_weight": 0.6,
            "bm25_weight": 0.4,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "user-1"
    assert data["query"] == "urgent care copay"
    assert data["result_count"] == 1
    assert data["results"][0]["chunk_id"] == "chunk-1"
    assert data["results"][0]["retrieval_sources"] == ["vector", "bm25"]


def test_hybrid_endpoint_empty_user_id_returns_400():
    response = client.post(
        "/retrieval/hybrid-search",
        json={
            "user_id": "",
            "query": "urgent care copay",
            "top_k": 5,
        },
    )

    assert response.status_code == 400


def test_hybrid_endpoint_empty_query_returns_400():
    response = client.post(
        "/retrieval/hybrid-search",
        json={
            "user_id": "user-1",
            "query": "",
            "top_k": 5,
        },
    )

    assert response.status_code == 400


def test_hybrid_endpoint_top_k_greater_than_20_is_capped(monkeypatch):
    captured = {}

    def fake_hybrid_search(
        db,
        user_id,
        query,
        top_k,
        vector_top_k,
        bm25_top_k,
        vector_weight,
        bm25_weight,
    ):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(retrieval_routes, "hybrid_search", fake_hybrid_search)

    response = client.post(
        "/retrieval/hybrid-search",
        json={
            "user_id": "user-1",
            "query": "urgent care copay",
            "top_k": 100,
        },
    )

    assert response.status_code == 200
    assert captured["top_k"] == 20


def test_hybrid_endpoint_both_weights_zero_returns_400():
    response = client.post(
        "/retrieval/hybrid-search",
        json={
            "user_id": "user-1",
            "query": "urgent care copay",
            "top_k": 5,
            "vector_weight": 0,
            "bm25_weight": 0,
        },
    )

    assert response.status_code == 400


def test_no_vector_or_bm25_results_returns_empty_list(monkeypatch):
    def fake_vector_search(db, user_id, query, top_k):
        return []

    def fake_bm25_search(db, user_id, query, top_k):
        return []

    monkeypatch.setattr(hybrid_retriever, "vector_search", fake_vector_search)
    monkeypatch.setattr(hybrid_retriever, "bm25_search", fake_bm25_search)

    results = hybrid_search(
        db=FakeDB(),
        user_id="user-1",
        query="urgent care",
    )

    assert results == []