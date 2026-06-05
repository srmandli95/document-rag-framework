import pytest
from fastapi.testclient import TestClient

from app.generation.citation_guard import REFUSAL_MESSAGE
from app.generation.answer_generator import generate_answer_from_evidence
from app.main import app


class FakeLLMClient:
    def __init__(self, answer: str):
        self.answer = answer
        self.called = False

    def generate(self, prompt: str) -> str:
        self.called = True
        assert "Does my health insurance cover urgent care?" in prompt
        return self.answer


def sample_evidence_chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_name": "sample_health_policy.txt",
            "category": "health_insurance",
            "page_number": None,
            "section_title": "Urgent Care",
            "chunk_index": 0,
            "chunk_text": "Urgent care visits are covered with a $50 copay.",
            "reranker_score": 0.92,
            "hybrid_score": 0.81,
        }
    ]


def test_generate_answer_from_evidence_returns_refusal_when_no_evidence(monkeypatch):
    fake_llm = FakeLLMClient("This should not be called.")

    def fake_rerank_hybrid_results(**kwargs):
        return []

    def fake_get_llm_client():
        return fake_llm

    monkeypatch.setattr(
        "app.generation.answer_generator.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.get_llm_client",
        fake_get_llm_client,
    )

    response = generate_answer_from_evidence(
        db=None,
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert response["status"] == "refused"
    assert response["validation_status"] == "unsupported"
    assert response["answer"] == REFUSAL_MESSAGE
    assert response["citations"] == []
    assert response["evidence_chunk_count"] == 0
    assert fake_llm.called is False


def test_llm_client_is_not_called_when_no_evidence(monkeypatch):
    fake_llm = FakeLLMClient("This should not be called.")

    monkeypatch.setattr(
        "app.generation.answer_generator.rerank_hybrid_results",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.get_llm_client",
        lambda: fake_llm,
    )

    generate_answer_from_evidence(
        db=None,
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert fake_llm.called is False


def test_generated_answer_with_valid_evidence_returns_answered(monkeypatch):
    fake_llm = FakeLLMClient("Urgent care visits are covered with a $50 copay.")

    monkeypatch.setattr(
        "app.generation.answer_generator.rerank_hybrid_results",
        lambda **kwargs: sample_evidence_chunks(),
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.get_llm_client",
        lambda: fake_llm,
    )

    response = generate_answer_from_evidence(
        db=None,
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert response["status"] == "answered"
    assert response["validation_status"] == "supported"
    assert response["answer"] == "Urgent care visits are covered with a $50 copay."
    assert response["evidence_chunk_count"] == 1
    assert len(response["citations"]) == 1
    assert response["citations"][0]["chunk_id"] == "chunk-1"
    assert fake_llm.called is True


def test_generated_answer_with_invalid_citations_gets_refused(monkeypatch):
    fake_llm = FakeLLMClient("Urgent care visits are covered with a $50 copay.")

    def fake_validate_answer_support(**kwargs):
        return {
            "validation_status": "unsupported",
            "reason": "One or more citation chunk IDs are missing or not present in evidence chunks.",
            "final_answer": REFUSAL_MESSAGE,
            "citations": [],
        }

    monkeypatch.setattr(
        "app.generation.answer_generator.rerank_hybrid_results",
        lambda **kwargs: sample_evidence_chunks(),
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.get_llm_client",
        lambda: fake_llm,
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.validate_answer_support",
        fake_validate_answer_support,
    )

    response = generate_answer_from_evidence(
        db=None,
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert response["status"] == "refused"
    assert response["validation_status"] == "unsupported"
    assert response["answer"] == REFUSAL_MESSAGE
    assert response["citations"] == []


def test_generated_empty_answer_gets_refused(monkeypatch):
    fake_llm = FakeLLMClient("")

    monkeypatch.setattr(
        "app.generation.answer_generator.rerank_hybrid_results",
        lambda **kwargs: sample_evidence_chunks(),
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.get_llm_client",
        lambda: fake_llm,
    )

    response = generate_answer_from_evidence(
        db=None,
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert response["status"] == "refused"
    assert response["validation_status"] == "unsupported"
    assert response["answer"] == REFUSAL_MESSAGE
    assert response["citations"] == []


def test_response_includes_validation_status_and_validation_reason(monkeypatch):
    fake_llm = FakeLLMClient("Urgent care visits are covered with a $50 copay.")

    monkeypatch.setattr(
        "app.generation.answer_generator.rerank_hybrid_results",
        lambda **kwargs: sample_evidence_chunks(),
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.get_llm_client",
        lambda: fake_llm,
    )

    response = generate_answer_from_evidence(
        db=None,
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert "validation_status" in response
    assert "validation_reason" in response
    assert response["validation_status"] == "supported"
    assert isinstance(response["validation_reason"], str)


def test_generate_answer_from_evidence_rejects_empty_user_id():
    with pytest.raises(ValueError, match="user_id is required"):
        generate_answer_from_evidence(
            db=None,
            user_id="",
            question="Does my health insurance cover urgent care?",
        )


def test_generate_answer_from_evidence_rejects_empty_question():
    with pytest.raises(ValueError, match="question is required"):
        generate_answer_from_evidence(
            db=None,
            user_id="local-user-123",
            question="",
        )


def test_chat_endpoint_returns_validation_fields(monkeypatch):
    client = TestClient(app)

    fake_response = {
        "user_id": "local-user-123",
        "question": "Does my health insurance cover urgent care?",
        "answer": "Urgent care visits are covered with a $50 copay.",
        "citations": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "page_number": None,
                "section_title": "Urgent Care",
                "chunk_index": 0,
                "reranker_score": 0.92,
                "hybrid_score": 0.81,
            }
        ],
        "evidence_chunk_count": 1,
        "model_name": "test-model",
        "status": "answered",
        "validation_status": "supported",
        "validation_reason": "Answer is supported by provided evidence and valid citations.",
    }

    def fake_generate_answer_from_evidence(**kwargs):
        return fake_response

    monkeypatch.setattr(
        "app.api.chat_routes.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )

    response = client.post(
        "/chat/ask",
        json={
            "user_id": "local-user-123",
            "question": "Does my health insurance cover urgent care?",
            "top_k": 5,
            "hybrid_top_k": 20,
            "vector_top_k": 20,
            "bm25_top_k": 20,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "answered"
    assert data["validation_status"] == "supported"
    assert "validation_reason" in data
    assert data["citations"][0]["chunk_id"] == "chunk-1"


def test_no_real_openai_call_is_made(monkeypatch):
    fake_llm = FakeLLMClient("Urgent care visits are covered with a $50 copay.")

    monkeypatch.setattr(
        "app.generation.answer_generator.rerank_hybrid_results",
        lambda **kwargs: sample_evidence_chunks(),
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.get_llm_client",
        lambda: fake_llm,
    )

    response = generate_answer_from_evidence(
        db=None,
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert response["status"] == "answered"
    assert fake_llm.called is True


def test_no_real_reranker_model_is_loaded(monkeypatch):
    fake_llm = FakeLLMClient("Urgent care visits are covered with a $50 copay.")

    def fake_rerank_hybrid_results(**kwargs):
        return sample_evidence_chunks()

    monkeypatch.setattr(
        "app.generation.answer_generator.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )
    monkeypatch.setattr(
        "app.generation.answer_generator.get_llm_client",
        lambda: fake_llm,
    )

    response = generate_answer_from_evidence(
        db=None,
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert response["status"] == "answered"
    assert response["validation_status"] == "supported"