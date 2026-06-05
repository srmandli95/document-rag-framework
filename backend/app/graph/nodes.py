from typing import Any

from app.config.settings import settings
from app.generation.citation_guard import (
    REFUSAL_MESSAGE,
    build_citations_from_evidence,
    validate_answer_support,
)
from app.generation.llm_client import get_llm_client
from app.generation.prompt_builder import build_answer_prompt
from app.graph.state import RAGState
from app.reranking.reranking_service import rerank_hybrid_results


def _get_model_name() -> str:
    provider = getattr(settings, "LLM_PROVIDER", "openai").lower().strip()

    if provider == "openai":
        return getattr(settings, "OPENAI_MODEL_NAME", "gpt-4o-mini")

    return provider


def _extract_llm_text(llm_response: Any) -> str:
    if llm_response is None:
        return ""

    if isinstance(llm_response, str):
        return llm_response

    if isinstance(llm_response, dict):
        for key in ("answer", "content", "text", "message"):
            value = llm_response.get(key)
            if isinstance(value, str):
                return value

    content = getattr(llm_response, "content", None)
    if isinstance(content, str):
        return content

    text = getattr(llm_response, "text", None)
    if isinstance(text, str):
        return text

    return str(llm_response)


def _call_llm(client: Any, prompt: str) -> str:
    if hasattr(client, "generate"):
        return _extract_llm_text(client.generate(prompt))

    if hasattr(client, "complete"):
        return _extract_llm_text(client.complete(prompt))

    if hasattr(client, "invoke"):
        return _extract_llm_text(client.invoke(prompt))

    if callable(client):
        return _extract_llm_text(client(prompt))

    raise TypeError("LLM client does not expose generate, complete, invoke, or callable interface.")


def load_user_context_node(state: RAGState) -> RAGState:
    user_id = state.get("user_id")
    question = state.get("question")

    cleaned_user_id = user_id.strip() if user_id else ""
    cleaned_question = question.strip() if question else ""

    if not cleaned_user_id:
        raise ValueError("user_id is required")

    if not cleaned_question:
        raise ValueError("question is required")

    return {
        **state,
        "user_id": cleaned_user_id,
        "question": cleaned_question,
        "user_context": {
            "user_id": cleaned_user_id,
        },
        "status": "loaded_user_context",
        "error": None,
    }


def retrieve_and_rerank_node(state: RAGState) -> RAGState:
    evidence_chunks = rerank_hybrid_results(
        db=state["db"],
        user_id=state["user_id"],
        query=state["question"],
        top_k=state.get("top_k", 5),
        hybrid_top_k=state.get("hybrid_top_k", 20),
        vector_top_k=state.get("vector_top_k", 20),
        bm25_top_k=state.get("bm25_top_k", 20),
    )

    return {
        **state,
        "evidence_chunks": evidence_chunks,
        "status": "retrieved_and_reranked",
        "error": None,
    }


def generate_answer_node(state: RAGState) -> RAGState:
    evidence_chunks = state.get("evidence_chunks", [])
    model_name = _get_model_name()

    if not evidence_chunks:
        return {
            **state,
            "generated_answer": REFUSAL_MESSAGE,
            "final_answer": REFUSAL_MESSAGE,
            "citations": [],
            "model_name": model_name,
            "status": "refused",
            "validation_status": "unsupported",
            "validation_reason": "No evidence chunks were found.",
            "error": None,
        }

    citations = build_citations_from_evidence(evidence_chunks)

    prompt = build_answer_prompt(
        question=state["question"],
        evidence_chunks=evidence_chunks,
    )

    llm_client = get_llm_client()
    generated_answer = _call_llm(llm_client, prompt)

    return {
        **state,
        "generated_answer": generated_answer,
        "citations": citations,
        "model_name": model_name,
        "status": "generated_answer",
        "error": None,
    }


def validate_citations_node(state: RAGState) -> RAGState:
    evidence_chunks = state.get("evidence_chunks", [])

    if not evidence_chunks:
        return {
            **state,
            "validation_status": "unsupported",
            "validation_reason": "No evidence chunks were found.",
            "final_answer": REFUSAL_MESSAGE,
            "citations": [],
            "status": "refused",
            "error": None,
        }

    generated_answer = state.get("generated_answer") or REFUSAL_MESSAGE
    citations = state.get("citations", [])
    min_reranker_score = state.get("min_reranker_score")

    validation_result = validate_answer_support(
        answer=generated_answer,
        evidence_chunks=evidence_chunks,
        citations=citations,
        min_evidence_chunks=1,
        min_reranker_score=min_reranker_score,
    )

    validation_status = validation_result["validation_status"]
    validation_reason = validation_result["reason"]

    if validation_status == "unsupported":
        final_answer = REFUSAL_MESSAGE
        final_citations = []
        status = "refused"
    else:
        final_answer = validation_result["final_answer"]
        final_citations = validation_result["citations"]
        status = "answered"

    return {
        **state,
        "validation_status": validation_status,
        "validation_reason": validation_reason,
        "final_answer": final_answer,
        "citations": final_citations,
        "status": status,
        "error": None,
    }


def final_response_node(state: RAGState) -> RAGState:
    model_name = state.get("model_name") or _get_model_name()
    validation_status = state.get("validation_status") or "unsupported"
    validation_reason = state.get("validation_reason") or "Validation was not completed."
    status = state.get("status") or "refused"

    final_response = {
        "user_id": state["user_id"],
        "question": state["question"],
        "answer": state.get("final_answer") or state.get("generated_answer") or REFUSAL_MESSAGE,
        "citations": state.get("citations", []),
        "evidence_chunk_count": len(state.get("evidence_chunks", [])),
        "model_name": model_name,
        "status": status,
        "validation_status": validation_status,
        "validation_reason": validation_reason,
    }

    return {
        **state,
        "final_response": final_response,
        "error": None,
    }