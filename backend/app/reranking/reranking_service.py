from typing import Any

from app.config.settings import settings
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.hybrid_retriever import hybrid_search


def rerank_hybrid_results(
    db,
    user_id: str,
    query: str,
    top_k: int = 8,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Run hybrid retrieval first, then rerank candidate chunks using a local
    cross-encoder model.

    This is still a foundation/debugging service, not final answer generation.
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")

    if not query or not query.strip():
        raise ValueError("query is required")

    safe_top_k = min(max(1, top_k), 10)
    safe_hybrid_top_k = min(max(1, hybrid_top_k), 50)
    safe_vector_top_k = min(max(1, vector_top_k), 50)
    safe_bm25_top_k = min(max(1, bm25_top_k), 50)

    candidates = hybrid_search(
        db=db,
        user_id=user_id,
        query=query,
        top_k=safe_hybrid_top_k,
        vector_top_k=safe_vector_top_k,
        bm25_top_k=safe_bm25_top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    if not candidates:
        return []

    reranker = CrossEncoderReranker(
        model_name=settings.RERANKER_MODEL_NAME,
    )

    return reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=safe_top_k,
    )