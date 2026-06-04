from typing import Any

from app.config.settings import settings
from app.generation.llm_client import get_llm_client
from app.generation.prompt_builder import build_answer_prompt, load_prompts
from app.reranking.reranking_service import rerank_hybrid_results


def _get_model_name() -> str:
    provider = getattr(settings, "LLM_PROVIDER", "openai").lower().strip()

    if provider == "openai":
        return getattr(settings, "OPENAI_MODEL_NAME", "gpt-4o-mini")

    return provider


def _build_citations(evidence_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []

    for chunk in evidence_chunks:
        citations.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "document_name": chunk.get("document_name"),
                "category": chunk.get("category"),
                "page_number": chunk.get("page_number"),
                "section_title": chunk.get("section_title"),
                "chunk_index": chunk.get("chunk_index"),
                "reranker_score": chunk.get("reranker_score"),
                "hybrid_score": chunk.get("hybrid_score"),
            }
        )

    return citations


def _refusal_response(user_id: str, question: str) -> dict[str, Any]:
    prompts = load_prompts()

    return {
        "user_id": user_id,
        "question": question,
        "answer": prompts["refusal_message"].strip(),
        "citations": [],
        "evidence_chunk_count": 0,
        "model_name": _get_model_name(),
        "status": "refused",
    }


def generate_answer_from_evidence(
    db,
    user_id: str,
    question: str,
    top_k: int = 5,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
) -> dict[str, Any]:
    """
    Generate a grounded answer using reranked evidence chunks.

    Flow:
    1. Validate inputs
    2. Run rerank search
    3. Build grounded prompt
    4. Call configured LLM provider
    5. Return answer with citation metadata
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")

    if not question or not question.strip():
        raise ValueError("question is required")

    safe_user_id = user_id.strip()
    safe_question = question.strip()

    safe_top_k = min(max(1, top_k), 8)
    safe_hybrid_top_k = min(max(1, hybrid_top_k), 50)
    safe_vector_top_k = min(max(1, vector_top_k), 50)
    safe_bm25_top_k = min(max(1, bm25_top_k), 50)

    evidence_chunks = rerank_hybrid_results(
        db=db,
        user_id=safe_user_id,
        query=safe_question,
        top_k=safe_top_k,
        hybrid_top_k=safe_hybrid_top_k,
        vector_top_k=safe_vector_top_k,
        bm25_top_k=safe_bm25_top_k,
    )

    if not evidence_chunks:
        return _refusal_response(
            user_id=safe_user_id,
            question=safe_question,
        )

    answer_prompt = build_answer_prompt(
        question=safe_question,
        evidence_chunks=evidence_chunks,
    )

    llm_client = get_llm_client()
    answer = llm_client.generate(answer_prompt)

    return {
        "user_id": safe_user_id,
        "question": safe_question,
        "answer": answer,
        "citations": _build_citations(evidence_chunks),
        "evidence_chunk_count": len(evidence_chunks),
        "model_name": _get_model_name(),
        "status": "answered",
    }