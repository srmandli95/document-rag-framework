from typing import Any

from sqlalchemy.orm import Session

from app.retrieval.bm25_retriever import bm25_search
from app.retrieval.retrieval_settings import (
    RetrievalSettings,
    validate_retrieval_settings,
)
from app.retrieval.vector_retriever import vector_search


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-max normalize scores where higher is better.

    If all scores are equal, every non-empty score gets 1.0.
    """

    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def _validate_identity_and_query(user_id: str, query: str) -> None:
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")

    if not query or not query.strip():
        raise ValueError("query is required")



def hybrid_search(
    db: Session,
    user_id: str,
    query: str,
    top_k: int = 5,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Run vector retrieval and BM25 retrieval, merge results,
    deduplicate by chunk_id, normalize scores, and return top hybrid results.

    This is still a debugging/foundation retriever.
    It does not call an LLM and does not generate a final answer.
    """

    _validate_identity_and_query(
        user_id=user_id,
        query=query,
    )
    retrieval_settings = validate_retrieval_settings(
        RetrievalSettings(
            top_k=top_k,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
    )

    user_id = user_id.strip()
    query = query.strip()

    top_k = retrieval_settings.top_k
    vector_top_k = retrieval_settings.vector_top_k
    bm25_top_k = retrieval_settings.bm25_top_k
    vector_weight = retrieval_settings.vector_weight
    bm25_weight = retrieval_settings.bm25_weight

    vector_results = vector_search(
        db=db,
        user_id=user_id,
        query=query,
        top_k=vector_top_k,
    )

    bm25_results = bm25_search(
        db=db,
        user_id=user_id,
        query=query,
        top_k=bm25_top_k,
    )

    vector_scores = [
        float(result.get("similarity_score", 0.0))
        for result in vector_results
    ]

    bm25_scores = [
        float(result.get("bm25_score", 0.0))
        for result in bm25_results
    ]

    normalized_vector_scores = normalize_scores(vector_scores)
    normalized_bm25_scores = normalize_scores(bm25_scores)

    merged_results: dict[str, dict[str, Any]] = {}

    for index, result in enumerate(vector_results):
        chunk_id = str(result.get("chunk_id"))

        merged_results[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": str(result.get("document_id")),
            "user_id": result.get("user_id"),
            "chunk_text": result.get("chunk_text"),
            "chunk_index": result.get("chunk_index"),
            "token_count": result.get("token_count"),
            "page_number": result.get("page_number"),
            "section_title": result.get("section_title"),
            "document_name": result.get("document_name"),
            "category": result.get("category"),
            "vector_score": float(result.get("similarity_score", 0.0)),
            "bm25_score": 0.0,
            "normalized_vector_score": normalized_vector_scores[index],
            "normalized_bm25_score": 0.0,
            "retrieval_sources": ["vector"],
        }

    for index, result in enumerate(bm25_results):
        chunk_id = str(result.get("chunk_id"))

        if chunk_id in merged_results:
            merged_results[chunk_id]["bm25_score"] = float(
                result.get("bm25_score", 0.0)
            )
            merged_results[chunk_id]["normalized_bm25_score"] = (
                normalized_bm25_scores[index]
            )

            if "bm25" not in merged_results[chunk_id]["retrieval_sources"]:
                merged_results[chunk_id]["retrieval_sources"].append("bm25")

        else:
            merged_results[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": str(result.get("document_id")),
                "user_id": result.get("user_id"),
                "chunk_text": result.get("chunk_text"),
                "chunk_index": result.get("chunk_index"),
                "token_count": result.get("token_count"),
                "page_number": result.get("page_number"),
                "section_title": result.get("section_title"),
                "document_name": result.get("document_name"),
                "category": result.get("category"),
                "vector_score": 0.0,
                "bm25_score": float(result.get("bm25_score", 0.0)),
                "normalized_vector_score": 0.0,
                "normalized_bm25_score": normalized_bm25_scores[index],
                "retrieval_sources": ["bm25"],
            }

    final_results = []

    for result in merged_results.values():
        hybrid_score = (
            vector_weight * result["normalized_vector_score"]
            + bm25_weight * result["normalized_bm25_score"]
        )

        result["hybrid_score"] = hybrid_score
        final_results.append(result)

    final_results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return final_results[:top_k]
