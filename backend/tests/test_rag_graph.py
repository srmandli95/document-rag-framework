import pytest
from fastapi.testclient import TestClient

from app.graph.nodes import (
    final_response_node,
    generate_answer_node,
    load_user_context_node,
    retrieve_and_rerank_node,
    validate_citations_node,
)
from app.graph.rag_graph import run_rag_workflow
from app.main import app


class FakeDB:
    pass


class FakeLLMClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.called = False

    def generate(self, prompt: str):
        self.called = True
        assert "Does my plan cover urgent care?" in prompt
        return self.response_text


def sample_evidence_chunk():
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "document_name": "sample_health_policy.txt",
        "category": "health_insurance",
        "page_number": None,
        "section_title": "Urgent Care",
        "chunk_index": 0,
        "chunk_text": "Urgent care is covered when medically necessary.",
        "reranker_score": 9.5,
        "hybrid_score": 0.91,
        "vector_score": 0.88,
        "bm25_score": 2.4,
    }


def sample_citation():
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "document_name": "sample_health_policy.txt",
        "category": "health_insurance",
        "page_number": None,
        "section_title": "Urgent Care",
        "chunk_index": 0,
        "reranker_score": 9.5,
        "hybrid_score": 0.91,
        "vector_score": 0.88,
        "bm25_score": 2.4,
    }


def assert_citation_matches_sample(citation: dict):
    assert citation["chunk_id"] == "chunk-1"
    assert citation["document_id"] == "doc-1"
    assert citation["document_name"] == "sample_health_policy.txt"
    assert citation["category"] == "health_insurance"
    assert citation["page_number"] is None
    assert citation["section_title"] == "Urgent Care"
    assert citation["chunk_index"] == 0
    assert citation["reranker_score"] == 9.5
    assert citation["hybrid_score"] == 0.91

    if "vector_score" in citation:
        assert citation["vector_score"] in {0.88, None}

    if "bm25_score" in citation:
        assert citation["bm25_score"] in {2.4, None}


def test_load_user_context_node_validates_user_id():
    state = {
        "db": FakeDB(),
        "user_id": "",
        "question": "Does my plan cover urgent care?",
    }

    with pytest.raises(ValueError, match="user_id is required"):
        load_user_context_node(state)


def test_load_user_context_node_validates_question():
    state = {
        "db": FakeDB(),
        "user_id": "local-user-123",
        "question": "",
    }

    with pytest.raises(ValueError, match="question is required"):
        load_user_context_node(state)


def test_load_user_context_node_adds_basic_user_context():
    state = {
        "db": FakeDB(),
        "user_id": " local-user-123 ",
        "question": " Does my plan cover urgent care? ",
    }

    result = load_user_context_node(state)

    assert result["user_id"] == "local-user-123"
    assert result["question"] == "Does my plan cover urgent care?"
    assert result["user_context"] == {"user_id": "local-user-123"}
    assert result["status"] == "loaded_user_context"
    assert result["error"] is None


def test_retrieve_and_rerank_node_stores_evidence_chunks(monkeypatch):
    fake_evidence = [sample_evidence_chunk()]

    def fake_rerank_hybrid_results(
        db,
        user_id,
        query,
        top_k,
        hybrid_top_k,
        vector_top_k,
        bm25_top_k,
    ):
        assert isinstance(db, FakeDB)
        assert user_id == "local-user-123"
        assert query == "Does my plan cover urgent care?"
        assert top_k == 5
        assert hybrid_top_k == 20
        assert vector_top_k == 20
        assert bm25_top_k == 20
        return fake_evidence

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    state = {
        "db": FakeDB(),
        "user_id": "local-user-123",
        "question": "Does my plan cover urgent care?",
        "top_k": 5,
        "hybrid_top_k": 20,
        "vector_top_k": 20,
        "bm25_top_k": 20,
    }

    result = retrieve_and_rerank_node(state)

    assert result["evidence_chunks"] == fake_evidence
    assert result["status"] == "retrieved_and_reranked"
    assert result["error"] is None


def test_generate_answer_node_returns_refusal_and_skips_llm_when_no_evidence(monkeypatch):
    def fail_if_called():
        raise AssertionError("LLM should not be called when evidence is empty")

    monkeypatch.setattr("app.graph.nodes.get_llm_client", fail_if_called)

    state = {
        "db": FakeDB(),
        "user_id": "local-user-123",
        "question": "Does my plan cover urgent care?",
        "evidence_chunks": [],
    }

    result = generate_answer_node(state)

    assert result["status"] == "refused"
    assert result["citations"] == []
    assert result["validation_status"] == "unsupported"
    assert result["validation_reason"] == "No evidence chunks were found."
    assert "could not find enough evidence" in result["generated_answer"].lower()
    assert "could not find enough evidence" in result["final_answer"].lower()
    assert result["model_name"]
    assert result["error"] is None


def test_generate_answer_node_calls_llm_when_evidence_exists(monkeypatch):
    fake_client = FakeLLMClient(
        "Urgent care is covered when medically necessary."
    )

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_client,
    )

    state = {
        "db": FakeDB(),
        "user_id": "local-user-123",
        "question": "Does my plan cover urgent care?",
        "evidence_chunks": [sample_evidence_chunk()],
    }

    result = generate_answer_node(state)

    assert fake_client.called is True
    assert result["generated_answer"] == "Urgent care is covered when medically necessary."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk-1"
    assert result["status"] == "generated_answer"
    assert result["model_name"]
    assert result["error"] is None


def test_validate_citations_node_returns_supported(monkeypatch):
    def fake_validate_answer_support(
        answer,
        evidence_chunks,
        citations,
        min_evidence_chunks=1,
        min_reranker_score=None,
    ):
        assert min_evidence_chunks == 1
        assert min_reranker_score is None

        return {
            "validation_status": "supported",
            "reason": "All citations reference retrieved evidence chunks.",
            "final_answer": answer,
            "citations": citations,
        }

    monkeypatch.setattr(
        "app.graph.nodes.validate_answer_support",
        fake_validate_answer_support,
    )

    state = {
        "generated_answer": "Urgent care is covered.",
        "evidence_chunks": [sample_evidence_chunk()],
        "citations": [sample_citation()],
        "min_reranker_score": None,
    }

    result = validate_citations_node(state)

    assert result["validation_status"] == "supported"
    assert result["validation_reason"] == "All citations reference retrieved evidence chunks."
    assert result["status"] == "answered"
    assert result["final_answer"] == "Urgent care is covered."
    assert result["citations"] == [sample_citation()]
    assert result["error"] is None


def test_validate_citations_node_returns_refused_when_invalid(monkeypatch):
    def fake_validate_answer_support(
        answer,
        evidence_chunks,
        citations,
        min_evidence_chunks=1,
        min_reranker_score=None,
    ):
        assert min_evidence_chunks == 1
        assert min_reranker_score is None

        return {
            "validation_status": "unsupported",
            "reason": "Citation chunk IDs are invalid.",
            "final_answer": None,
            "citations": [],
        }

    monkeypatch.setattr(
        "app.graph.nodes.validate_answer_support",
        fake_validate_answer_support,
    )

    state = {
        "generated_answer": "Urgent care is covered.",
        "evidence_chunks": [sample_evidence_chunk()],
        "citations": [
            {
                **sample_citation(),
                "chunk_id": "bad-chunk",
            }
        ],
        "min_reranker_score": None,
    }

    result = validate_citations_node(state)

    assert result["validation_status"] == "unsupported"
    assert result["validation_reason"] == "Citation chunk IDs are invalid."
    assert result["status"] == "refused"
    assert result["citations"] == []
    assert "could not find enough evidence" in result["final_answer"].lower()
    assert result["error"] is None


def test_validate_citations_node_refuses_when_no_evidence():
    state = {
        "generated_answer": "Some answer.",
        "evidence_chunks": [],
        "citations": [],
        "min_reranker_score": None,
    }

    result = validate_citations_node(state)

    assert result["validation_status"] == "unsupported"
    assert result["validation_reason"] == "No evidence chunks were found."
    assert result["status"] == "refused"
    assert result["citations"] == []
    assert "could not find enough evidence" in result["final_answer"].lower()
    assert result["error"] is None


def test_final_response_node_builds_expected_response_shape():
    state = {
        "user_id": "local-user-123",
        "question": "Does my plan cover urgent care?",
        "final_answer": "Urgent care is covered.",
        "citations": [sample_citation()],
        "evidence_chunks": [sample_evidence_chunk()],
        "model_name": "gpt-4o",
        "status": "answered",
        "validation_status": "supported",
        "validation_reason": "Valid citations.",
    }

    result = final_response_node(state)

    final_response = result["final_response"]

    assert final_response["user_id"] == "local-user-123"
    assert final_response["question"] == "Does my plan cover urgent care?"
    assert final_response["answer"] == "Urgent care is covered."
    assert final_response["evidence_chunk_count"] == 1
    assert final_response["model_name"] == "gpt-4o"
    assert final_response["status"] == "answered"
    assert final_response["validation_status"] == "supported"
    assert final_response["validation_reason"] == "Valid citations."
    assert_citation_matches_sample(final_response["citations"][0])
    assert result["error"] is None


def test_final_response_node_defaults_required_string_fields():
    state = {
        "user_id": "local-user-123",
        "question": "Does my plan cover urgent care?",
        "generated_answer": None,
        "final_answer": None,
        "citations": [],
        "evidence_chunks": [],
        "model_name": None,
        "status": None,
        "validation_status": None,
        "validation_reason": None,
    }

    result = final_response_node(state)
    final_response = result["final_response"]

    assert final_response["user_id"] == "local-user-123"
    assert final_response["question"] == "Does my plan cover urgent care?"
    assert "could not find enough evidence" in final_response["answer"].lower()
    assert final_response["citations"] == []
    assert final_response["evidence_chunk_count"] == 0
    assert isinstance(final_response["model_name"], str)
    assert final_response["model_name"]
    assert final_response["status"] == "refused"
    assert final_response["validation_status"] == "unsupported"
    assert final_response["validation_reason"] == "Validation was not completed."


def test_run_rag_workflow_returns_final_response_without_real_db_llm_or_reranker(
    monkeypatch,
):
    fake_client = FakeLLMClient(
        "Urgent care is covered when medically necessary."
    )

    def fake_rerank_hybrid_results(
        db,
        user_id,
        query,
        top_k,
        hybrid_top_k,
        vector_top_k,
        bm25_top_k,
    ):
        assert isinstance(db, FakeDB)
        assert user_id == "local-user-123"
        assert query == "Does my plan cover urgent care?"
        assert top_k == 5
        assert hybrid_top_k == 20
        assert vector_top_k == 20
        assert bm25_top_k == 20
        return [sample_evidence_chunk()]

    def fake_validate_answer_support(
        answer,
        evidence_chunks,
        citations,
        min_evidence_chunks=1,
        min_reranker_score=None,
    ):
        assert min_evidence_chunks == 1
        assert min_reranker_score is None

        return {
            "validation_status": "supported",
            "reason": "All citations are valid.",
            "final_answer": answer,
            "citations": citations,
        }

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )
    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        "app.graph.nodes.validate_answer_support",
        fake_validate_answer_support,
    )

    result = run_rag_workflow(
        db=FakeDB(),
        user_id="local-user-123",
        question="Does my plan cover urgent care?",
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
    )

    assert result["user_id"] == "local-user-123"
    assert result["question"] == "Does my plan cover urgent care?"
    assert result["answer"] == "Urgent care is covered when medically necessary."
    assert result["evidence_chunk_count"] == 1
    assert result["model_name"]
    assert result["status"] == "answered"
    assert result["validation_status"] == "supported"
    assert result["validation_reason"] == "All citations are valid."
    assert_citation_matches_sample(result["citations"][0])
    assert fake_client.called is True


def test_run_rag_workflow_passes_min_reranker_score_to_validation(monkeypatch):
    fake_client = FakeLLMClient(
        "Urgent care is covered when medically necessary."
    )

    captured = {}

    def fake_rerank_hybrid_results(
        db,
        user_id,
        query,
        top_k,
        hybrid_top_k,
        vector_top_k,
        bm25_top_k,
    ):
        return [sample_evidence_chunk()]

    def fake_validate_answer_support(
        answer,
        evidence_chunks,
        citations,
        min_evidence_chunks=1,
        min_reranker_score=None,
    ):
        captured["min_reranker_score"] = min_reranker_score

        return {
            "validation_status": "supported",
            "reason": "All citations are valid.",
            "final_answer": answer,
            "citations": citations,
        }

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )
    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        "app.graph.nodes.validate_answer_support",
        fake_validate_answer_support,
    )

    result = run_rag_workflow(
        db=FakeDB(),
        user_id="local-user-123",
        question="Does my plan cover urgent care?",
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
        min_reranker_score=0.5,
    )

    assert result["status"] == "answered"
    assert captured["min_reranker_score"] == 0.5


def test_run_rag_workflow_returns_refusal_without_real_llm_when_no_evidence(
    monkeypatch,
):
    def fake_rerank_hybrid_results(
        db,
        user_id,
        query,
        top_k,
        hybrid_top_k,
        vector_top_k,
        bm25_top_k,
    ):
        return []

    def fail_if_called():
        raise AssertionError("LLM should not be called when no evidence exists.")

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )
    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        fail_if_called,
    )

    result = run_rag_workflow(
        db=FakeDB(),
        user_id="local-user-123",
        question="Does my plan cover urgent care?",
    )

    assert result["user_id"] == "local-user-123"
    assert result["question"] == "Does my plan cover urgent care?"
    assert "could not find enough evidence" in result["answer"].lower()
    assert result["citations"] == []
    assert result["evidence_chunk_count"] == 0
    assert result["model_name"]
    assert result["status"] == "refused"
    assert result["validation_status"] == "unsupported"
    assert result["validation_reason"] == "No evidence chunks were found."


def test_chat_endpoint_still_works_without_real_db_llm_reranker(monkeypatch):
    def fake_generate_answer_from_evidence(
        db,
        user_id,
        question,
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
        min_reranker_score=None,
    ):
        assert user_id == "local-user-123"
        assert question == "Does my plan cover urgent care?"
        assert top_k == 5
        assert hybrid_top_k == 20
        assert vector_top_k == 20
        assert bm25_top_k == 20

        return {
            "user_id": user_id,
            "question": question,
            "answer": "Urgent care is covered when medically necessary.",
            "citations": [sample_citation()],
            "evidence_chunk_count": 1,
            "model_name": "gpt-4o",
            "status": "answered",
            "validation_status": "supported",
            "validation_reason": "All citations are valid.",
        }

    monkeypatch.setattr(
        "app.api.chat_routes.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )

    client = TestClient(app)

    response = client.post(
        "/chat/ask",
        json={
            "user_id": "local-user-123",
            "question": "Does my plan cover urgent care?",
            "top_k": 5,
            "hybrid_top_k": 20,
            "vector_top_k": 20,
            "bm25_top_k": 20,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "local-user-123"
    assert data["question"] == "Does my plan cover urgent care?"
    assert data["answer"] == "Urgent care is covered when medically necessary."
    assert data["evidence_chunk_count"] == 1
    assert data["model_name"] == "gpt-4o"
    assert data["status"] == "answered"
    assert data["validation_status"] == "supported"
    assert data["validation_reason"] == "All citations are valid."
    assert_citation_matches_sample(data["citations"][0])


def test_chat_endpoint_returns_400_for_missing_user_id():
    client = TestClient(app)

    response = client.post(
        "/chat/ask",
        json={
            "user_id": "",
            "question": "Does my plan cover urgent care?",
            "top_k": 5,
            "hybrid_top_k": 20,
            "vector_top_k": 20,
            "bm25_top_k": 20,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "user_id is required"


def test_chat_endpoint_returns_400_for_missing_question():
    client = TestClient(app)

    response = client.post(
        "/chat/ask",
        json={
            "user_id": "local-user-123",
            "question": "",
            "top_k": 5,
            "hybrid_top_k": 20,
            "vector_top_k": 20,
            "bm25_top_k": 20,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "question is required"