from typing import Any

from app.config.settings import settings
from app.generation.citation_guard import (
    REFUSAL_MESSAGE,
    build_citations_from_evidence,
    validate_answer_support,
)
from app.generation.llm_client import get_llm_client
from app.generation.prompt_builder import build_answer_prompt

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


def generate_answer_from_evidence(
    db,
    user_id: str,
    question: str,
    top_k: int = 5,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    rerank_top_k: int = 8,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    min_reranker_score: float | None = None,
) -> dict[str, Any]:
    

    from app.graph.rag_graph import run_rag_workflow

    return run_rag_workflow(
        db=db,
        user_id=user_id,
        question=question,
        top_k=top_k,
        hybrid_top_k=hybrid_top_k,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        rerank_top_k=rerank_top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        min_reranker_score=min_reranker_score,
    )
