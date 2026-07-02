from app.graph.rag_graph import build_rag_graph, run_rag_workflow


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.called = False

    def generate(self, prompt: str):
        self.called = True
        return self.response


def test_build_rag_graph_compiles():
    graph = build_rag_graph()

    assert graph is not None


def test_run_rag_workflow_returns_final_response(monkeypatch):
    rewrite_llm = FakeLLM("late payment DTE Energy consequences")
    answer_llm = FakeLLM("A late payment may result in a late payment charge based on the evidence.")
    verifier_llm = FakeLLM(
        '{"status": "supported", "reason": "The answer is supported by the evidence.", "unsupported_claims": []}'
    )

    llm_calls = [rewrite_llm, answer_llm, verifier_llm]

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
        assert query == "late payment DTE Energy consequences"
        assert top_k == 8
        assert vector_weight == 0.6
        assert bm25_weight == 0.4

        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "document_name": "yourguide.pdf",
                "category": "utility_policy",
                "page_number": 10,
                "section_title": "Billing",
                "chunk_index": 3,
                "chunk_text": "A late payment charge may apply when payment is received after the due date.",
                "reranker_score": 0.91,
                "hybrid_score": 0.83,
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

    response = run_rag_workflow(
        db=object(),
        user_id="local-user-123",
        question="can you explain me the consequences for the late payment of my bill for DTE Energy?",
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
    )

    assert response["user_id"] == "local-user-123"
    assert response["question"] == "can you explain me the consequences for the late payment of my bill for DTE Energy?"
    assert response["rewritten_question"] == "late payment DTE Energy consequences"
    assert response["evidence_sufficient"] is True
    assert response["grounding_status"] == "supported"
    assert response["status"] == "answered"
    assert response["validation_status"] == "supported"
    assert len(response["citations"]) == 1
