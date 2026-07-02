from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    check_evidence_sufficiency_node,
    final_response_node,
    generate_answer_node,
    load_user_context_node,
    retrieve_and_rerank_node,
    rewrite_query_node,
    validate_citations_node,
    verify_answer_grounding_node,
)
from app.graph.state import RAGState
from app.retrieval.retrieval_settings import (
    RetrievalSettings,
    validate_retrieval_settings,
)
from app.utils.logger import get_logger


logger = get_logger(__name__)


def build_rag_graph():
    """Compile the LangGraph workflow used for RAG answering."""
    graph_builder = StateGraph(RAGState)

    graph_builder.add_node("load_user_context", load_user_context_node)
    graph_builder.add_node("rewrite_query", rewrite_query_node)
    graph_builder.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph_builder.add_node("check_evidence_sufficiency", check_evidence_sufficiency_node)
    graph_builder.add_node("generate_answer", generate_answer_node)
    graph_builder.add_node("verify_answer_grounding", verify_answer_grounding_node)
    graph_builder.add_node("validate_citations", validate_citations_node)
    graph_builder.add_node("final_response", final_response_node)

    graph_builder.add_edge(START, "load_user_context")
    graph_builder.add_edge("load_user_context", "rewrite_query")
    graph_builder.add_edge("rewrite_query", "retrieve_and_rerank")
    graph_builder.add_edge("retrieve_and_rerank", "check_evidence_sufficiency")
    graph_builder.add_edge("check_evidence_sufficiency", "generate_answer")
    graph_builder.add_edge("generate_answer", "verify_answer_grounding")
    graph_builder.add_edge("verify_answer_grounding", "validate_citations")
    graph_builder.add_edge("validate_citations", "final_response")
    graph_builder.add_edge("final_response", END)

    return graph_builder.compile()


def run_rag_workflow(
    db: Any,
    user_id: str,
    question: str,
    chat_history: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    rerank_top_k: int = 8,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    min_reranker_score: float | None = None,
) -> dict[str, Any]:
    """Run the RAG graph for one user question and return the response."""
    logger.info(
        "RAG workflow started: user_id=%s question_length=%s top_k=%s hybrid_top_k=%s",
        user_id,
        len(question or ""),
        top_k,
        hybrid_top_k,
    )
    retrieval_settings = validate_retrieval_settings(
        RetrievalSettings(
            top_k=top_k,
            hybrid_top_k=hybrid_top_k,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_k=rerank_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            min_reranker_score=min_reranker_score,
        ),
        require_final_top_k_within_rerank=True,
    )
    graph = build_rag_graph()

    initial_state: RAGState = {
        "db": db,
        "user_id": user_id,
        "question": question,
        "rewritten_question": None,
        "chat_history": chat_history or [],
        "top_k": retrieval_settings.top_k,
        "hybrid_top_k": retrieval_settings.hybrid_top_k,
        "vector_top_k": retrieval_settings.vector_top_k,
        "bm25_top_k": retrieval_settings.bm25_top_k,
        "rerank_top_k": retrieval_settings.rerank_top_k,
        "vector_weight": retrieval_settings.vector_weight,
        "bm25_weight": retrieval_settings.bm25_weight,
        "min_reranker_score": retrieval_settings.min_reranker_score,
        "user_context": None,
        "evidence_chunks": [],
        "evidence_sufficient": None,
        "evidence_sufficiency_reason": None,
        "generated_answer": None,
        "citations": [],
        "validation_status": None,
        "validation_reason": None,
        "grounding_status": None,
        "grounding_reason": None,
        "unsupported_claims": [],
        "final_answer": None,
        "final_response": None,
        "model_name": None,
        "status": None,
        "error": None,
    }

    final_state = graph.invoke(initial_state)

    final_response = final_state["final_response"]
    logger.info(
        "RAG workflow completed: user_id=%s status=%s evidence_chunks=%s validation=%s",
        user_id,
        final_response.get("status"),
        final_response.get("evidence_chunk_count"),
        final_response.get("validation_status"),
    )
    return final_response
