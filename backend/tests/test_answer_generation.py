import pytest
from fastapi.testclient import TestClient

from app.db.database import get_db
from app.generation import answer_generator
from app.generation.prompt_builder import build_answer_prompt, build_evidence_context
from app.main import app


client = TestClient(app)


class FakeDB:
    pass


class FakeLLMClient:
    def __init__(self, answer: str):
        self.answer = answer
        self.called = False

    def generate(self, prompt: str) -> str:
        self.called = True
        return self.answer


def override_get_db():
    yield FakeDB()


@pytest.fixture(autouse=True)
def override_db_dependency():
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()


def sample_evidence_chunk():
    return {
        "chunk_id": "chunk-123",
        "document_id": "doc-123",
        "document_name": "sample_health_policy.txt",
        "category": "health_insurance",
        "page_number": None,
        "section_title": "Urgent Care",
        "chunk_index": 2,
        "chunk_text": "Urgent care visits are covered after the member pays the required copay.",
        "reranker_score": 8.42,
        "hybrid_score": 0.91,
    }


def test_build_evidence_context_formats_chunk_metadata():
    evidence_context = build_evidence_context([sample_evidence_chunk()])

    assert "[Chunk ID: chunk-123]" in evidence_context
    assert "Document: sample_health_policy.txt" in evidence_context
    assert "Category: health_insurance" in evidence_context
    assert "Page: N/A" in evidence_context
    assert "Section: Urgent Care" in evidence_context
    assert "Urgent care visits are covered" in evidence_context


def test_build_answer_prompt_includes_question_and_evidence_text():
    question = "Does my health insurance cover urgent care?"

    prompt = build_answer_prompt(
        question=question,
        evidence_chunks=[sample_evidence_chunk()],
    )

    assert question in prompt
    assert "Urgent care visits are covered" in prompt
    assert "You must answer the user's question using ONLY the provided evidence chunks" in prompt


def test_generate_answer_from_evidence_returns_refusal_when_no_evidence(monkeypatch):
    llm_called = False

    def fake_rerank_hybrid_results(**kwargs):
        return []

    def fake_get_llm_client():
        nonlocal llm_called
        llm_called = True
        return FakeLLMClient("This should not be called")

    monkeypatch.setattr(
        answer_generator,
        "rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )
    monkeypatch.setattr(
        answer_generator,
        "get_llm_client",
        fake_get_llm_client,
    )

    result = answer_generator.generate_answer_from_evidence(
        db=FakeDB(),
        user_id="local-user-123",
        question="Does my policy cover dental?",
    )

    assert result["status"] == "refused"
    assert result["citations"] == []
    assert result["evidence_chunk_count"] == 0
    assert "I could not find enough evidence" in result["answer"]
    assert llm_called is False


def test_generate_answer_from_evidence_returns_answer_and_citations(monkeypatch):
    fake_llm = FakeLLMClient(
        "Based on the provided evidence, urgent care visits are covered.\n\nSources:\n- sample_health_policy.txt, Chunk ID: chunk-123"
    )

    def fake_rerank_hybrid_results(**kwargs):
        return [sample_evidence_chunk()]

    def fake_get_llm_client():
        return fake_llm

    monkeypatch.setattr(
        answer_generator,
        "rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )
    monkeypatch.setattr(
        answer_generator,
        "get_llm_client",
        fake_get_llm_client,
    )

    result = answer_generator.generate_answer_from_evidence(
        db=FakeDB(),
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
    )

    assert result["status"] == "answered"
    assert result["evidence_chunk_count"] == 1
    assert result["citations"][0]["chunk_id"] == "chunk-123"
    assert result["citations"][0]["document_name"] == "sample_health_policy.txt"
    assert result["citations"][0]["section_title"] == "Urgent Care"
    assert "urgent care visits are covered" in result["answer"]
    assert fake_llm.called is True


def test_llm_client_is_not_called_when_evidence_is_empty(monkeypatch):
    fake_llm = FakeLLMClient("This should not be called")

    def fake_rerank_hybrid_results(**kwargs):
        return []

    def fake_get_llm_client():
        return fake_llm

    monkeypatch.setattr(
        answer_generator,
        "rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )
    monkeypatch.setattr(
        answer_generator,
        "get_llm_client",
        fake_get_llm_client,
    )

    result = answer_generator.generate_answer_from_evidence(
        db=FakeDB(),
        user_id="local-user-123",
        question="What is covered?",
    )

    assert result["status"] == "refused"
    assert fake_llm.called is False


def test_chat_endpoint_success(monkeypatch):
    from app.api import chat_routes

    def fake_generate_answer_from_evidence(
        db,
        user_id,
        question,
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
    ):
        return {
            "user_id": user_id,
            "question": question,
            "answer": "Based on the provided evidence, urgent care is covered.",
            "citations": [
                {
                    "chunk_id": "chunk-123",
                    "document_id": "doc-123",
                    "document_name": "sample_health_policy.txt",
                    "category": "health_insurance",
                    "page_number": None,
                    "section_title": "Urgent Care",
                    "chunk_index": 2,
                    "reranker_score": 8.42,
                    "hybrid_score": 0.91,
                }
            ],
            "evidence_chunk_count": 1,
            "model_name": "gpt-4o-mini",
            "status": "answered",
        }

    monkeypatch.setattr(
        chat_routes,
        "generate_answer_from_evidence",
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

    assert data["user_id"] == "local-user-123"
    assert data["status"] == "answered"
    assert data["evidence_chunk_count"] == 1
    assert data["citations"][0]["chunk_id"] == "chunk-123"


def test_chat_endpoint_empty_user_id_validation():
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "   ",
            "question": "Does my policy cover urgent care?",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "user_id is required"


def test_chat_endpoint_empty_question_validation():
    response = client.post(
        "/chat/ask",
        json={
            "user_id": "local-user-123",
            "question": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "question is required"


def test_chat_endpoint_top_k_greater_than_8_is_capped(monkeypatch):
    from app.api import chat_routes

    captured_args = {}

    def fake_generate_answer_from_evidence(
        db,
        user_id,
        question,
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
    ):
        captured_args["top_k"] = top_k
        captured_args["hybrid_top_k"] = hybrid_top_k
        captured_args["vector_top_k"] = vector_top_k
        captured_args["bm25_top_k"] = bm25_top_k

        return {
            "user_id": user_id,
            "question": question,
            "answer": "Answer from evidence.",
            "citations": [],
            "evidence_chunk_count": 0,
            "model_name": "gpt-4o-mini",
            "status": "refused",
        }

    monkeypatch.setattr(
        chat_routes,
        "generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )

    response = client.post(
        "/chat/ask",
        json={
            "user_id": "local-user-123",
            "question": "Does my policy cover urgent care?",
            "top_k": 100,
            "hybrid_top_k": 100,
            "vector_top_k": 100,
            "bm25_top_k": 100,
        },
    )

    assert response.status_code == 200
    assert captured_args["top_k"] == 8
    assert captured_args["hybrid_top_k"] == 50
    assert captured_args["vector_top_k"] == 50
    assert captured_args["bm25_top_k"] == 50