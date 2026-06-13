import pytest

from app.generation.citation_guard import check_evidence_sufficiency
from app.graph.nodes import (
    check_evidence_sufficiency_node,
    generate_answer_node,
    retrieve_and_rerank_node,
    rewrite_query_node,
)
from app.graph.rag_graph import run_rag_workflow


class FakeLLM:
    def __init__(self, response: str | None = None, should_raise: bool = False):
        self.response = response
        self.should_raise = should_raise
        self.called = False

    def generate(self, prompt: str):
        self.called = True

        if self.should_raise:
            raise RuntimeError("LLM failed")

        return self.response


def test_rewrite_query_node_stores_rewritten_question(monkeypatch):
    fake_llm = FakeLLM("late payment consequences DTE Energy bill")

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "can you explain me the consequences for the late payment of my bill for DTE Energy?"
    }

    result = rewrite_query_node(state)

    assert result["rewritten_question"] == "late payment consequences DTE Energy bill"
    assert fake_llm.called is True


def test_rewrite_query_node_falls_back_when_llm_returns_empty(monkeypatch):
    fake_llm = FakeLLM("   ")

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "Does my health insurance cover urgent care?"
    }

    result = rewrite_query_node(state)

    assert result["rewritten_question"] == "Does my health insurance cover urgent care?"


def test_rewrite_query_node_falls_back_when_llm_raises(monkeypatch):
    fake_llm = FakeLLM(should_raise=True)

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "What happens if I miss a payment?"
    }

    result = rewrite_query_node(state)

    assert result["rewritten_question"] == "What happens if I miss a payment?"
    assert "Query rewrite failed" in result["error"]


def test_retrieve_and_rerank_node_uses_rewritten_question(monkeypatch):
    captured = {}

    def fake_rerank_hybrid_results(
        db,
        user_id,
        query,
        top_k,
        hybrid_top_k,
        vector_top_k,
        bm25_top_k,
        vector_weight,
        bm25_weight,
    ):
        captured["query"] = query

        return [
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Late payments may result in a late fee.",
                "reranker_score": 0.9,
            }
        ]

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    state = {
        "db": object(),
        "user_id": "local-user-123",
        "question": "original question",
        "rewritten_question": "rewritten search query",
        "top_k": 5,
        "hybrid_top_k": 20,
        "vector_top_k": 20,
        "bm25_top_k": 20,
    }

    result = retrieve_and_rerank_node(state)

    assert captured["query"] == "rewritten search query"
    assert len(result["evidence_chunks"]) == 1


def test_retrieve_and_rerank_node_reranks_then_slices_final_evidence(monkeypatch):
    captured = {}

    def fake_rerank_hybrid_results(**kwargs):
        captured.update(kwargs)
        return [{"chunk_id": f"chunk-{index}"} for index in range(8)]

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    state = {
        "db": object(),
        "user_id": "local-user-123",
        "question": "question",
        "top_k": 3,
        "rerank_top_k": 8,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
    }

    result = retrieve_and_rerank_node(state)

    assert captured["top_k"] == 8
    assert captured["vector_weight"] == 0.7
    assert captured["bm25_weight"] == 0.3
    assert len(result["evidence_chunks"]) == 3


def test_evidence_sufficiency_fails_when_no_chunks():
    result = check_evidence_sufficiency([])

    assert result["evidence_sufficient"] is False
    assert result["reason"] == "No evidence chunks were retrieved."


def test_evidence_sufficiency_fails_when_chunk_count_below_minimum():
    result = check_evidence_sufficiency(
        evidence_chunks=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Some evidence",
            }
        ],
        min_evidence_chunks=2,
    )

    assert result["evidence_sufficient"] is False
    assert "at least 2" in result["reason"]


def test_evidence_sufficiency_passes_when_enough_chunks_exist():
    result = check_evidence_sufficiency(
        evidence_chunks=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Some evidence",
            }
        ],
        min_evidence_chunks=1,
    )

    assert result["evidence_sufficient"] is True


def test_evidence_sufficiency_fails_when_min_reranker_score_not_met():
    result = check_evidence_sufficiency(
        evidence_chunks=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Weak evidence",
                "reranker_score": 0.2,
            }
        ],
        min_evidence_chunks=1,
        min_reranker_score=0.7,
    )

    assert result["evidence_sufficient"] is False
    assert "minimum reranker score" in result["reason"]


def test_evidence_sufficiency_passes_when_at_least_one_chunk_meets_score():
    result = check_evidence_sufficiency(
        evidence_chunks=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Weak evidence",
                "reranker_score": 0.2,
            },
            {
                "chunk_id": "chunk-2",
                "chunk_text": "Strong evidence",
                "reranker_score": 0.8,
            },
        ],
        min_evidence_chunks=1,
        min_reranker_score=0.7,
    )

    assert result["evidence_sufficient"] is True


def test_check_evidence_sufficiency_node_sets_refused_status_when_weak():
    state = {
        "evidence_chunks": [],
        "min_reranker_score": None,
    }

    result = check_evidence_sufficiency_node(state)

    assert result["evidence_sufficient"] is False
    assert result["status"] == "refused"
    assert result["validation_status"] == "unsupported"
    assert result["citations"] == []
    assert "could not find enough evidence" in result["generated_answer"].lower()


def test_generate_answer_node_skips_llm_when_status_refused(monkeypatch):
    fake_llm = FakeLLM("This should not be called")

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "Does this exist?",
        "status": "refused",
        "evidence_sufficient": False,
        "generated_answer": "Refusal answer",
        "final_answer": "Refusal answer",
        "citations": [],
    }

    result = generate_answer_node(state)

    assert result["generated_answer"] == "Refusal answer"
    assert result["status"] == "refused"
    assert fake_llm.called is False


def test_run_rag_workflow_includes_rewrite_and_evidence_fields(monkeypatch):
    rewrite_llm = FakeLLM("urgent care health insurance coverage")
    answer_llm = FakeLLM("Yes, urgent care is covered according to the evidence.")

    llm_calls = [rewrite_llm, answer_llm]

    def fake_get_llm_client():
        return llm_calls.pop(0)

    def fake_rerank_hybrid_results(
        db,
        user_id,
        query,
        top_k,
        hybrid_top_k,
        vector_top_k,
        bm25_top_k,
        vector_weight,
        bm25_weight,
    ):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "page_number": None,
                "section_title": "Urgent Care",
                "chunk_index": 0,
                "chunk_text": "Urgent care visits are covered under this plan.",
                "reranker_score": 0.95,
                "hybrid_score": 0.88,
            }
        ]

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        fake_get_llm_client,
    )

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    result = run_rag_workflow(
        db=object(),
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
    )

    assert result["rewritten_question"] == "urgent care health insurance coverage"
    assert result["evidence_sufficient"] is True
    assert result["evidence_sufficiency_reason"] == "Retrieved evidence is sufficient for answer generation."
    assert result["status"] == "answered"
    assert result["validation_status"] == "supported"
    assert len(result["citations"]) == 1
