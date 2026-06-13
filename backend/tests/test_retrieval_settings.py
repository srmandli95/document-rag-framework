import math

import pytest

from app.retrieval.retrieval_settings import (
    RetrievalSettings,
    normalize_weights,
    validate_retrieval_settings,
)


def test_default_settings_are_valid():
    settings = validate_retrieval_settings(RetrievalSettings())

    assert settings == RetrievalSettings()


@pytest.mark.parametrize("top_k", [0, 21])
def test_top_k_outside_allowed_range_fails(top_k):
    with pytest.raises(ValueError, match="top_k must be between 1 and 20"):
        validate_retrieval_settings(RetrievalSettings(top_k=top_k))


@pytest.mark.parametrize("field", ["vector_top_k", "bm25_top_k", "hybrid_top_k"])
def test_candidate_top_k_above_50_fails(field):
    with pytest.raises(ValueError, match=f"{field} must be between 1 and 50"):
        validate_retrieval_settings(RetrievalSettings(**{field: 51}))


def test_rerank_top_k_above_20_fails():
    with pytest.raises(ValueError, match="rerank_top_k must be between 1 and 20"):
        validate_retrieval_settings(RetrievalSettings(rerank_top_k=21))


@pytest.mark.parametrize(
    ("vector_weight", "bm25_weight", "message"),
    [
        (-0.1, 1.0, "vector_weight must be non-negative"),
        (1.0, -0.1, "bm25_weight must be non-negative"),
        (0.0, 0.0, "At least one retrieval weight must be greater than 0"),
        (math.inf, 1.0, "vector_weight must be finite"),
        (1.0, math.nan, "bm25_weight must be finite"),
    ],
)
def test_invalid_weights_fail(vector_weight, bm25_weight, message):
    with pytest.raises(ValueError, match=message):
        normalize_weights(vector_weight, bm25_weight)


def test_weights_are_normalized_when_sum_is_not_one():
    settings = validate_retrieval_settings(
        RetrievalSettings(vector_weight=7, bm25_weight=3)
    )

    assert settings.vector_weight == pytest.approx(0.7)
    assert settings.bm25_weight == pytest.approx(0.3)


@pytest.mark.parametrize("score", [None, 0.25, -1.5])
def test_min_reranker_score_allows_none_or_finite_float(score):
    settings = validate_retrieval_settings(
        RetrievalSettings(min_reranker_score=score)
    )

    assert settings.min_reranker_score == score


def test_rag_settings_require_top_k_within_rerank_top_k():
    with pytest.raises(
        ValueError,
        match="top_k must be less than or equal to rerank_top_k",
    ):
        validate_retrieval_settings(
            RetrievalSettings(top_k=9, rerank_top_k=8),
            require_final_top_k_within_rerank=True,
        )
