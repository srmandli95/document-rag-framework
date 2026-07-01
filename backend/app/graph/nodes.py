from typing import Any

from app.generation.citation_guard import (
    build_citations_from_evidence,
    check_evidence_sufficiency,
    validate_answer_support,
)
from app.generation.llm_client import get_llm_client
from app.generation.prompt_builder import (
    build_answer_prompt,
    build_query_rewrite_prompt,
    get_refusal_message,
    strip_generated_source_metadata,
)
from app.graph.state import RAGState
from app.reranking.reranking_service import rerank_hybrid_results
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _extract_llm_text(response: Any) -> str:
    """Extract plain text from supported LLM response shapes."""
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    if hasattr(response, "content"):
        return str(response.content)

    if isinstance(response, dict):
        if "content" in response:
            return str(response["content"])

        if "text" in response:
            return str(response["text"])

        if "answer" in response:
            return str(response["answer"])

    return str(response)


def load_user_context_node(state: RAGState) -> RAGState:
    """Attach basic user context to the RAG graph state."""
    logger.debug("RAG node load_user_context: user_id=%s", state.get("user_id"))
    state["user_context"] = {
        "user_id": state.get("user_id"),
    }

    return state


def rewrite_query_node(state: RAGState) -> RAGState:
    """Rewrite the user question before retrieval when possible."""
    original_question = state.get("question") or ""

    try:
        logger.debug(
            "RAG node rewrite_query started: user_id=%s question_length=%s",
            state.get("user_id"),
            len(original_question),
        )
        prompt = build_query_rewrite_prompt(original_question)
        llm_client = get_llm_client()
        response = llm_client.generate(prompt)
        rewritten_question = _extract_llm_text(response).strip()

        if not rewritten_question:
            rewritten_question = original_question

        state["rewritten_question"] = rewritten_question
        logger.debug(
            "RAG node rewrite_query completed: user_id=%s rewritten_length=%s",
            state.get("user_id"),
            len(rewritten_question),
        )

    except Exception as exc:
        state["rewritten_question"] = original_question
        state["error"] = f"Query rewrite failed and fell back to original question: {exc}"
        logger.warning(
            "RAG node rewrite_query failed; using original question: user_id=%s error=%s",
            state.get("user_id"),
            exc,
        )

    return state


def retrieve_and_rerank_node(state: RAGState) -> RAGState:
    """Retrieve, combine, and rerank candidate evidence chunks."""
    db = state["db"]
    user_id = state["user_id"]

    retrieval_query = state.get("rewritten_question") or state["question"]
    logger.debug(
        "RAG node retrieve_and_rerank started: user_id=%s query_length=%s",
        user_id,
        len(retrieval_query or ""),
    )

    top_k = state.get("top_k", 5)
    hybrid_top_k = state.get("hybrid_top_k", 20)
    vector_top_k = state.get("vector_top_k", 20)
    bm25_top_k = state.get("bm25_top_k", 20)
    rerank_top_k = state.get("rerank_top_k", 8)
    vector_weight = state.get("vector_weight", 0.6)
    bm25_weight = state.get("bm25_weight", 0.4)

    evidence_chunks = rerank_hybrid_results(
        db=db,
        user_id=user_id,
        query=retrieval_query,
        top_k=rerank_top_k,
        hybrid_top_k=hybrid_top_k,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    state["evidence_chunks"] = evidence_chunks[:top_k]
    logger.info(
        "RAG node retrieve_and_rerank completed: user_id=%s retrieved=%s selected=%s",
        user_id,
        len(evidence_chunks),
        len(state["evidence_chunks"]),
    )

    return state


def check_evidence_sufficiency_node(state: RAGState) -> RAGState:
    """Assess whether retrieved evidence can support an answer."""
    evidence_chunks = state.get("evidence_chunks", [])
    min_reranker_score = state.get("min_reranker_score")

    result = check_evidence_sufficiency(
        evidence_chunks=evidence_chunks,
        min_evidence_chunks=1,
        min_reranker_score=min_reranker_score,
    )

    evidence_sufficient = bool(result["evidence_sufficient"])
    reason = str(result["reason"])
    logger.info(
        "RAG node evidence sufficiency checked: user_id=%s sufficient=%s evidence_chunks=%s reason=%s",
        state.get("user_id"),
        evidence_sufficient,
        len(evidence_chunks),
        reason,
    )

    state["evidence_sufficient"] = evidence_sufficient
    state["evidence_sufficiency_reason"] = reason

    if not evidence_sufficient:
        refusal_message = get_refusal_message()

        state["generated_answer"] = refusal_message
        state["final_answer"] = refusal_message
        state["citations"] = []
        state["validation_status"] = "unsupported"
        state["validation_reason"] = reason
        state["status"] = "refused"

    return state


def generate_answer_node(state: RAGState) -> RAGState:
    """Generate or refuse an answer from the current graph state."""
    if state.get("status") == "refused" or state.get("evidence_sufficient") is False:
        refusal_message = state.get("generated_answer") or get_refusal_message()
        logger.info("RAG node generate_answer skipped with refusal: user_id=%s", state.get("user_id"))

        state["generated_answer"] = refusal_message
        state["final_answer"] = refusal_message
        state["citations"] = state.get("citations", [])
        state["status"] = "refused"

        return state

    question = state["question"]
    evidence_chunks = state.get("evidence_chunks", [])

    citations = build_citations_from_evidence(evidence_chunks)
    logger.debug(
        "RAG node generate_answer started: user_id=%s evidence_chunks=%s citations=%s",
        state.get("user_id"),
        len(evidence_chunks),
        len(citations),
    )
    prompt = build_answer_prompt(
        question=question,
        evidence_chunks=evidence_chunks,
    )

    llm_client = get_llm_client()
    generated_answer = strip_generated_source_metadata(
        _extract_llm_text(llm_client.generate(prompt))
    )

    state["generated_answer"] = generated_answer
    state["final_answer"] = generated_answer
    state["citations"] = citations
    state["status"] = "answered"
    logger.info(
        "RAG node generate_answer completed: user_id=%s answer_length=%s citations=%s",
        state.get("user_id"),
        len(generated_answer),
        len(citations),
    )

    return state


def validate_citations_node(state: RAGState) -> RAGState:
    """Validate generated citations against retrieved evidence."""
    if state.get("status") == "refused":
        state["validation_status"] = state.get("validation_status") or "unsupported"
        state["validation_reason"] = state.get("validation_reason") or "Evidence was insufficient."

        return state

    validation_result = validate_answer_support(
        answer=state.get("generated_answer"),
        citations=state.get("citations", []),
        evidence_chunks=state.get("evidence_chunks", []),
    )

    state["validation_status"] = validation_result["validation_status"]
    state["validation_reason"] = validation_result["validation_reason"]
    logger.info(
        "RAG node validate_citations completed: user_id=%s validation=%s reason=%s",
        state.get("user_id"),
        state["validation_status"],
        state["validation_reason"],
    )

    if validation_result["validation_status"] != "supported":
        refusal_message = get_refusal_message()
        logger.warning(
            "RAG answer refused after citation validation: user_id=%s reason=%s",
            state.get("user_id"),
            state["validation_reason"],
        )

        state["generated_answer"] = refusal_message
        state["final_answer"] = refusal_message
        state["citations"] = []
        state["status"] = "refused"

    return state


def final_response_node(state: RAGState) -> RAGState:
    """Build the final API response from graph state."""
    final_answer = state.get("final_answer") or state.get("generated_answer") or get_refusal_message()
    evidence_chunks = state.get("evidence_chunks", [])

    final_response = {
        "user_id": state.get("user_id"),
        "question": state.get("question"),
        "rewritten_question": state.get("rewritten_question"),
        "answer": final_answer,
        "citations": state.get("citations", []),

        "evidence_chunks": evidence_chunks,
        "evidence_chunk_count": len(evidence_chunks),

        "validation_status": state.get("validation_status"),
        "validation_reason": state.get("validation_reason"),
        "evidence_sufficient": state.get("evidence_sufficient"),
        "evidence_sufficiency_reason": state.get("evidence_sufficiency_reason"),
        "model_name": state.get("model_name"),
        "status": state.get("status"),
    }

    state["final_answer"] = final_answer
    state["final_response"] = final_response
    logger.debug(
        "RAG node final_response completed: user_id=%s status=%s",
        state.get("user_id"),
        final_response["status"],
    )

    return state
