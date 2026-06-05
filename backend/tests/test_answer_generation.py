from app.generation.answer_generator import generate_answer_from_evidence


class FakeDB:
    pass


def test_generate_answer_from_evidence_returns_refusal_when_no_evidence(monkeypatch):
    def fake_run_rag_workflow(
        db,
        user_id,
        question,
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
        min_reranker_score=None,
    ):
        return {
            "user_id": user_id,
            "question": question,
            "answer": (
                "I could not find enough evidence in your uploaded documents "
                "to answer this question. Please upload the relevant document "
                "or ask a question covered by your existing documents."
            ),
            "citations": [],
            "evidence_chunk_count": 0,
            "model_name": "gpt-4o",
            "status": "refused",
            "validation_status": "unsupported",
            "validation_reason": "No evidence chunks were found.",
        }

    monkeypatch.setattr(
        "app.graph.rag_graph.run_rag_workflow",
        fake_run_rag_workflow,
    )

    result = generate_answer_from_evidence(
        db=FakeDB(),
        user_id="local-user-123",
        question="Does my plan cover urgent care?",
    )

    assert result["user_id"] == "local-user-123"
    assert result["question"] == "Does my plan cover urgent care?"
    assert result["citations"] == []
    assert result["evidence_chunk_count"] == 0
    assert result["status"] == "refused"
    assert result["validation_status"] == "unsupported"
    assert result["validation_reason"] == "No evidence chunks were found."
    assert "could not find enough evidence" in result["answer"].lower()


def test_generate_answer_from_evidence_returns_answered_response(monkeypatch):
    def fake_run_rag_workflow(
        db,
        user_id,
        question,
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
        min_reranker_score=None,
    ):
        return {
            "user_id": user_id,
            "question": question,
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
                    "reranker_score": 9.5,
                    "hybrid_score": 0.91,
                }
            ],
            "evidence_chunk_count": 1,
            "model_name": "gpt-4o",
            "status": "answered",
            "validation_status": "supported",
            "validation_reason": "All citations are valid.",
        }

    monkeypatch.setattr(
        "app.graph.rag_graph.run_rag_workflow",
        fake_run_rag_workflow,
    )

    result = generate_answer_from_evidence(
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
    assert result["answer"] == "Urgent care visits are covered with a $50 copay."
    assert result["citations"][0]["chunk_id"] == "chunk-1"
    assert result["evidence_chunk_count"] == 1
    assert result["model_name"] == "gpt-4o"
    assert result["status"] == "answered"
    assert result["validation_status"] == "supported"
    assert result["validation_reason"] == "All citations are valid."


def test_generate_answer_from_evidence_passes_parameters_to_graph(monkeypatch):
    captured = {}

    def fake_run_rag_workflow(
        db,
        user_id,
        question,
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
        min_reranker_score=None,
    ):
        captured["db"] = db
        captured["user_id"] = user_id
        captured["question"] = question
        captured["top_k"] = top_k
        captured["hybrid_top_k"] = hybrid_top_k
        captured["vector_top_k"] = vector_top_k
        captured["bm25_top_k"] = bm25_top_k
        captured["min_reranker_score"] = min_reranker_score

        return {
            "user_id": user_id,
            "question": question,
            "answer": "Answer",
            "citations": [],
            "evidence_chunk_count": 0,
            "model_name": "gpt-4o",
            "status": "refused",
            "validation_status": "unsupported",
            "validation_reason": "No evidence chunks were found.",
        }

    monkeypatch.setattr(
        "app.graph.rag_graph.run_rag_workflow",
        fake_run_rag_workflow,
    )

    fake_db = FakeDB()

    generate_answer_from_evidence(
        db=fake_db,
        user_id="local-user-123",
        question="Does my plan cover urgent care?",
        top_k=7,
        hybrid_top_k=30,
        vector_top_k=25,
        bm25_top_k=15,
        min_reranker_score=0.5,
    )

    assert captured["db"] is fake_db
    assert captured["user_id"] == "local-user-123"
    assert captured["question"] == "Does my plan cover urgent care?"
    assert captured["top_k"] == 7
    assert captured["hybrid_top_k"] == 30
    assert captured["vector_top_k"] == 25
    assert captured["bm25_top_k"] == 15
    assert captured["min_reranker_score"] == 0.5


def test_generate_answer_from_evidence_does_not_call_openai_or_reranker_directly(monkeypatch):
    def fake_run_rag_workflow(
        db,
        user_id,
        question,
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
        min_reranker_score=None,
    ):
        return {
            "user_id": user_id,
            "question": question,
            "answer": "Graph handled this.",
            "citations": [],
            "evidence_chunk_count": 0,
            "model_name": "gpt-4o",
            "status": "refused",
            "validation_status": "unsupported",
            "validation_reason": "No evidence chunks were found.",
        }

    monkeypatch.setattr(
        "app.graph.rag_graph.run_rag_workflow",
        fake_run_rag_workflow,
    )

    result = generate_answer_from_evidence(
        db=FakeDB(),
        user_id="local-user-123",
        question="Does my plan cover urgent care?",
    )

    assert result["answer"] == "Graph handled this."