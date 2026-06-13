from dataclasses import dataclass, replace
from math import isfinite


MIN_TOP_K = 1
MAX_TOP_K = 20
MAX_CANDIDATE_K = 50
MAX_RERANK_TOP_K = 20


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 5
    vector_top_k: int = 20
    bm25_top_k: int = 20
    hybrid_top_k: int = 20
    rerank_top_k: int = 8
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    min_reranker_score: float | None = None


def normalize_weights(
    vector_weight: float,
    bm25_weight: float,
) -> tuple[float, float]:
    if not isfinite(vector_weight):
        raise ValueError("vector_weight must be finite")
    if not isfinite(bm25_weight):
        raise ValueError("bm25_weight must be finite")
    if vector_weight < 0:
        raise ValueError("vector_weight must be non-negative")
    if bm25_weight < 0:
        raise ValueError("bm25_weight must be non-negative")

    total = vector_weight + bm25_weight
    if total <= 0:
        raise ValueError("At least one retrieval weight must be greater than 0")

    return vector_weight / total, bm25_weight / total


def validate_retrieval_settings(
    settings: RetrievalSettings,
    *,
    require_final_top_k_within_rerank: bool = False,
) -> RetrievalSettings:
    limits = (
        ("top_k", settings.top_k, MAX_TOP_K),
        ("vector_top_k", settings.vector_top_k, MAX_CANDIDATE_K),
        ("bm25_top_k", settings.bm25_top_k, MAX_CANDIDATE_K),
        ("hybrid_top_k", settings.hybrid_top_k, MAX_CANDIDATE_K),
        ("rerank_top_k", settings.rerank_top_k, MAX_RERANK_TOP_K),
    )
    for name, value, maximum in limits:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < MIN_TOP_K or value > maximum:
            raise ValueError(
                f"{name} must be between {MIN_TOP_K} and {maximum}"
            )

    if (
        require_final_top_k_within_rerank
        and settings.top_k > settings.rerank_top_k
    ):
        raise ValueError("top_k must be less than or equal to rerank_top_k")

    vector_weight, bm25_weight = normalize_weights(
        settings.vector_weight,
        settings.bm25_weight,
    )

    if (
        settings.min_reranker_score is not None
        and not isfinite(settings.min_reranker_score)
    ):
        raise ValueError("min_reranker_score must be finite")

    return replace(
        settings,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
