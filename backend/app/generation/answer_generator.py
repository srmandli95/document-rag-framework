from typing import Any

from app.config.settings import settings
from app.generation.citation_guard import (
    REFUSAL_MESSAGE,
    build_citations_from_evidence,
    validate_answer_support,
)
from app.generation.llm_client import get_llm_client
from app.generation.prompt_builder import build_answer_prompt
from app.reranking.reranking_service import rerank_hybrid_results


def _get_model_name() -> str:
    provider = getattr(settings, "LLM_PROVIDER", "openai").lower().strip()

    if provider == "openai":
        return getattr(settings, "OPENAI_MODEL_NAME", "gpt-4o-mini")

    return provider


def _refused_response(
    user_id: str,
    question: str,
    evidence_chunk_count: int,
    model_name: str,
    validation_reason: str,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "question": question,
        "answer": REFUSAL_MESSAGE,
        "citations": [],
        "evidence_chunk_count": evidence_chunk_count,
        "model_name": model_name,
        "status": "refused",
        "validation_status": "unsupported",
        "validation_reason": validation_reason,
    }


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


def generate_answer_from_evidence(
    db,
    user_id: str,
    question: str,
    top_k: int = 5,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    min_reranker_score: float | None = None,
) -> dict[str, Any]:
    cleaned_user_id = user_id.strip() if user_id else ""
    cleaned_question = question.strip() if question else ""
    model_name = _get_model_name()

    if not cleaned_user_id:
        raise ValueError("user_id is required.")

    if not cleaned_question:
        raise ValueError("question is required.")

    evidence_chunks = rerank_hybrid_results(
        db=db,
        user_id=cleaned_user_id,
        query=cleaned_question,
        top_k=top_k,
        hybrid_top_k=hybrid_top_k,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
    )

    if not evidence_chunks:
        return _refused_response(
            user_id=cleaned_user_id,
            question=cleaned_question,
            evidence_chunk_count=0,
            model_name=model_name,
            validation_reason="No evidence chunks were found.",
        )

    citations = build_citations_from_evidence(evidence_chunks)

    prompt = build_answer_prompt(
        question=cleaned_question,
        evidence_chunks=evidence_chunks,
    )

    llm_client = get_llm_client()
    generated_answer = _call_llm(llm_client, prompt)

    validation_result = validate_answer_support(
        answer=generated_answer,
        evidence_chunks=evidence_chunks,
        citations=citations,
        min_evidence_chunks=1,
        min_reranker_score=min_reranker_score,
    )

    validation_status = validation_result["validation_status"]

    if validation_status == "unsupported":
        return {
            "user_id": cleaned_user_id,
            "question": cleaned_question,
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "evidence_chunk_count": len(evidence_chunks),
            "model_name": model_name,
            "status": "refused",
            "validation_status": validation_status,
            "validation_reason": validation_result["reason"],
        }

    return {
        "user_id": cleaned_user_id,
        "question": cleaned_question,
        "answer": validation_result["final_answer"],
        "citations": validation_result["citations"],
        "evidence_chunk_count": len(evidence_chunks),
        "model_name": model_name,
        "status": "answered",
        "validation_status": validation_status,
        "validation_reason": validation_result["reason"],
    }