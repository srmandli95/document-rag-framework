from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.main import app
from app.retrieval.retrieval_diagnostics import (
    _calculate_overlap,
    _calculate_rank_changes,
    _extract_chunk_id,
    _rank_map,
    _slim_result,
    diagnose_retrieval,
)
from conftest import FakeDB, override_get_db, override_get_current_user


client = TestClient(app)


def _result(chunk_id: str, **overrides) -> dict:
    result = {
        "chunk_id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "user_id": "user-1",
        "chunk_text": f"Text for {chunk_id}",
        "chunk_index": 0,
        "document_name": "policy.txt",
        "section_title": "Coverage",
    }
    result.update(overrides)
    return result


def test_extract_chunk_id_returns_normalized_chunk_id():
    assert _extract_chunk_id({"chunk_id": " chunk-1 "}) == "chunk-1"
    assert _extract_chunk_id({"chunk_id": " "}) is None
    assert _extract_chunk_id({}) is None


def test_rank_map_returns_first_one_based_ranks():
    ranks = _rank_map([_result("a"), _result("b"), _result("a")])
    assert ranks == {"a": 1, "b": 2}


def test_calculate_overlap_counts_unique_shared_chunks():
    overlap = _calculate_overlap(
        [_result("a"), _result("a"), _result("b")],
        [_result("a"), _result("c")],
    )
    assert overlap == 1


def test_calculate_rank_changes_detects_moved_up_and_down():
    before = [
        _result("a", hybrid_score=0.9),
        _result("b", hybrid_score=0.8),
        _result("c", hybrid_score=0.7),
    ]
    after = [
        _result("c", hybrid_score=0.7, reranker_score=0.95),
        _result("a", hybrid_score=0.9, reranker_score=0.80),
    ]

    changes = _calculate_rank_changes(before, after)

    assert changes[0]["chunk_id"] == "c"
    assert changes[0]["before_rank"] == 3
    assert changes[0]["after_rank"] == 1
    assert changes[0]["rank_delta"] == 2
    assert changes[1]["chunk_id"] == "a"
    assert changes[1]["rank_delta"] == -1


def test_calculate_rank_changes_excludes_chunks_missing_before_rerank():
    assert _calculate_rank_changes([_result("a")], [_result("unknown")]) == []


def test_slim_result_truncates_preview_and_removes_sensitive_fields():
    original = _result(
        "a",
        chunk_text="x" * 350,
        vector_score=0.8,
        normalized_vector_score=1.0,
        reranker_model_name="secret-model",
    )

    slim = _slim_result(original)

    assert slim["chunk_text_preview"] == "x" * 300
    assert slim["vector_score"] == 0.8
    assert slim["normalized_vector_score"] == 1.0
    assert "chunk_text" not in slim
    assert "user_id" not in slim
    assert "reranker_model_name" not in slim


def test_diagnose_retrieval_calls_services_and_returns_summary(monkeypatch):
    calls = {}
    vector_results = [
        _result("a", distance=0.1, similarity_score=0.9),
        _result("b", distance=0.2, similarity_score=0.8),
    ]
    bm25_results = [
        _result("b", bm25_score=2.0),
        _result("c", bm25_score=1.0),
    ]
    hybrid_results = [
        _result("a", hybrid_score=0.9, retrieval_sources=["vector"]),
        _result("b", hybrid_score=0.8, retrieval_sources=["vector", "bm25"]),
    ]
    reranked_results = [
        _result(
            "b",
            hybrid_score=0.8,
            reranker_score=0.95,
            retrieval_sources=["vector", "bm25"],
        )
    ]

    def fake_vector_search(**kwargs):
        calls["vector"] = kwargs
        return vector_results

    def fake_bm25_search(**kwargs):
        calls["bm25"] = kwargs
        return bm25_results

    def fake_hybrid_search(**kwargs):
        calls["hybrid"] = kwargs
        return hybrid_results

    def fake_rerank_hybrid_results(**kwargs):
        calls["rerank"] = kwargs
        return reranked_results

    monkeypatch.setattr(
        "app.retrieval.retrieval_diagnostics.vector_search",
        fake_vector_search,
    )
    monkeypatch.setattr(
        "app.retrieval.retrieval_diagnostics.bm25_search",
        fake_bm25_search,
    )
    monkeypatch.setattr(
        "app.retrieval.retrieval_diagnostics.hybrid_search",
        fake_hybrid_search,
    )
    monkeypatch.setattr(
        "app.retrieval.retrieval_diagnostics.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    diagnostics = diagnose_retrieval(
        db=FakeDB(),
        user_id=" user-1 ",
        query=" coverage ",
        vector_top_k=7,
        bm25_top_k=8,
        hybrid_top_k=9,
        rerank_top_k=4,
    )

    assert calls["vector"]["user_id"] == "user-1"
    assert calls["vector"]["query"] == "coverage"
    assert calls["vector"]["top_k"] == 7
    assert calls["bm25"]["top_k"] == 8
    assert calls["hybrid"]["top_k"] == 9
    assert calls["hybrid"]["vector_top_k"] == 7
    assert calls["hybrid"]["bm25_top_k"] == 8
    assert calls["rerank"]["top_k"] == 4
    assert calls["rerank"]["hybrid_top_k"] == 9
    assert diagnostics["settings"]["vector_weight"] == 0.6
    assert diagnostics["settings"]["bm25_weight"] == 0.4
    assert diagnostics["summary"] == {
        "vector_count": 2,
        "bm25_count": 2,
        "hybrid_count": 2,
        "reranked_count": 1,
        "overlap_vector_bm25": 1,
        "overlap_hybrid_rerank": 1,
    }
    assert diagnostics["rank_changes"][0]["chunk_id"] == "b"
    assert diagnostics["rank_changes"][0]["rank_delta"] == 1
    assert "chunk_text" not in diagnostics["vector_results"][0]


def test_diagnose_retrieval_rejects_top_k_above_limits(monkeypatch):
    captured = {}

    def fake_search(**kwargs):
        captured.setdefault("calls", []).append(kwargs)
        return []

    monkeypatch.setattr("app.retrieval.retrieval_diagnostics.vector_search", fake_search)
    monkeypatch.setattr("app.retrieval.retrieval_diagnostics.bm25_search", fake_search)
    monkeypatch.setattr("app.retrieval.retrieval_diagnostics.hybrid_search", fake_search)
    monkeypatch.setattr(
        "app.retrieval.retrieval_diagnostics.rerank_hybrid_results",
        fake_search,
    )

    import pytest

    with pytest.raises(ValueError, match="vector_top_k must be between 1 and 50"):
        diagnose_retrieval(FakeDB(), "user-1", "query", 99, 99, 99, 99)

    assert "calls" not in captured


def test_diagnose_endpoint_uses_authenticated_user_and_returns_slim_results(
    monkeypatch,
):
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user("real-user")
    captured = {}

    def fake_diagnose_retrieval(**kwargs):
        captured.update(kwargs)
        return {
            "user_id": kwargs["user_id"],
            "query": kwargs["query"],
            "vector_results": [_slim_result(_result("a"))],
            "bm25_results": [],
            "hybrid_results": [],
            "reranked_results": [],
            "rank_changes": [],
            "summary": {
                "vector_count": 1,
                "bm25_count": 0,
                "hybrid_count": 0,
                "reranked_count": 0,
                "overlap_vector_bm25": 0,
                "overlap_hybrid_rerank": 0,
            },
            "settings": {
                "vector_top_k": kwargs["vector_top_k"],
                "bm25_top_k": kwargs["bm25_top_k"],
                "hybrid_top_k": kwargs["hybrid_top_k"],
                "rerank_top_k": kwargs["rerank_top_k"],
                "vector_weight": kwargs["vector_weight"],
                "bm25_weight": kwargs["bm25_weight"],
            },
        }

    monkeypatch.setattr(
        "app.api.retrieval_routes.diagnose_retrieval",
        fake_diagnose_retrieval,
    )

    response = client.post(
        "/search/diagnose",
        json={"user_id": "malicious-user", "query": "coverage"},
    )

    assert response.status_code == 200
    assert captured["user_id"] == "real-user"
    assert response.json()["user_id"] == "real-user"
    assert "chunk_text" not in response.json()["vector_results"][0]


def test_diagnose_endpoint_requires_auth():
    app.dependency_overrides.clear()
    response = client.post("/search/diagnose", json={"query": "coverage"})
    assert response.status_code == 401


def test_diagnose_endpoint_rejects_empty_query():
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user("user-1")
    response = client.post("/search/diagnose", json={"query": " "})
    assert response.status_code == 400
    assert response.json()["detail"] == "query is required"
