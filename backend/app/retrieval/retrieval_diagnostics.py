from dataclasses import asdict
from typing import Any

from app.reranking.reranking_service import rerank_hybrid_results
from app.retrieval.bm25_retriever import bm25_search
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.retrieval_settings import (
    RetrievalSettings,
    validate_retrieval_settings,
)
from app.retrieval.vector_retriever import vector_search


_SLIM_RESULT_FIELDS = (
    "chunk_id",
    "document_id",
    "document_name",
    "category",
    "chunk_index",
    "page_number",
    "section_title",
    "token_count",
    "distance",
    "similarity_score",
    "vector_score",
    "bm25_score",
    "normalized_vector_score",
    "normalized_bm25_score",
    "hybrid_score",
    "reranker_score",
    "retrieval_sources",
)


def _extract_chunk_id(result: dict) -> str | None:
    chunk_id = result.get("chunk_id")
    if chunk_id is None:
        return None

    normalized = str(chunk_id).strip()
    return normalized or None


def _rank_map(results: list[dict]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, result in enumerate(results, start=1):
        chunk_id = _extract_chunk_id(result)
        if chunk_id and chunk_id not in ranks:
            ranks[chunk_id] = rank
    return ranks


def _calculate_overlap(results_a: list[dict], results_b: list[dict]) -> int:
    chunk_ids_a = {
        chunk_id
        for result in results_a
        if (chunk_id := _extract_chunk_id(result)) is not None
    }
    chunk_ids_b = {
        chunk_id
        for result in results_b
        if (chunk_id := _extract_chunk_id(result)) is not None
    }
    return len(chunk_ids_a & chunk_ids_b)


def _calculate_rank_changes(
    before_results: list[dict],
    after_results: list[dict],
) -> list[dict]:
    before_ranks = _rank_map(before_results)
    before_by_chunk = {
        chunk_id: result
        for result in before_results
        if (chunk_id := _extract_chunk_id(result)) is not None
    }

    changes: list[dict] = []
    seen: set[str] = set()
    for after_rank, after_result in enumerate(after_results, start=1):
        chunk_id = _extract_chunk_id(after_result)
        if not chunk_id or chunk_id in seen or chunk_id not in before_ranks:
            continue

        seen.add(chunk_id)
        before_result = before_by_chunk[chunk_id]
        before_rank = before_ranks[chunk_id]
        changes.append(
            {
                "chunk_id": chunk_id,
                "before_rank": before_rank,
                "after_rank": after_rank,
                "rank_delta": before_rank - after_rank,
                "document_name": after_result.get("document_name")
                or before_result.get("document_name"),
                "section_title": after_result.get("section_title")
                or before_result.get("section_title"),
                "hybrid_score": after_result.get("hybrid_score")
                if after_result.get("hybrid_score") is not None
                else before_result.get("hybrid_score"),
                "reranker_score": after_result.get("reranker_score"),
            }
        )
    return changes


def _slim_result(result: dict) -> dict:
    slim = {field: result.get(field) for field in _SLIM_RESULT_FIELDS}
    chunk_text = str(result.get("chunk_text") or "")
    slim["chunk_text_preview"] = chunk_text[:300]
    return slim


def diagnose_retrieval(
    db,
    user_id: str,
    query: str,
    vector_top_k: int = 10,
    bm25_top_k: int = 10,
    hybrid_top_k: int = 10,
    rerank_top_k: int = 5,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> dict[str, Any]:
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")
    if not query or not query.strip():
        raise ValueError("query is required")

    retrieval_settings = validate_retrieval_settings(
        RetrievalSettings(
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            hybrid_top_k=hybrid_top_k,
            rerank_top_k=rerank_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
    )

    clean_user_id = user_id.strip()
    clean_query = query.strip()
    safe_vector_top_k = retrieval_settings.vector_top_k
    safe_bm25_top_k = retrieval_settings.bm25_top_k
    safe_hybrid_top_k = retrieval_settings.hybrid_top_k
    safe_rerank_top_k = retrieval_settings.rerank_top_k

    vector_results = vector_search(
        db=db,
        user_id=clean_user_id,
        query=clean_query,
        top_k=safe_vector_top_k,
    )
    bm25_results = bm25_search(
        db=db,
        user_id=clean_user_id,
        query=clean_query,
        top_k=safe_bm25_top_k,
    )
    hybrid_results = hybrid_search(
        db=db,
        user_id=clean_user_id,
        query=clean_query,
        top_k=safe_hybrid_top_k,
        vector_top_k=safe_vector_top_k,
        bm25_top_k=safe_bm25_top_k,
        vector_weight=retrieval_settings.vector_weight,
        bm25_weight=retrieval_settings.bm25_weight,
    )
    reranked_results = rerank_hybrid_results(
        db=db,
        user_id=clean_user_id,
        query=clean_query,
        top_k=safe_rerank_top_k,
        hybrid_top_k=safe_hybrid_top_k,
        vector_top_k=safe_vector_top_k,
        bm25_top_k=safe_bm25_top_k,
        vector_weight=retrieval_settings.vector_weight,
        bm25_weight=retrieval_settings.bm25_weight,
    )

    return {
        "user_id": clean_user_id,
        "query": clean_query,
        "vector_results": [_slim_result(result) for result in vector_results],
        "bm25_results": [_slim_result(result) for result in bm25_results],
        "hybrid_results": [_slim_result(result) for result in hybrid_results],
        "reranked_results": [_slim_result(result) for result in reranked_results],
        "rank_changes": _calculate_rank_changes(hybrid_results, reranked_results),
        "summary": {
            "vector_count": len(vector_results),
            "bm25_count": len(bm25_results),
            "hybrid_count": len(hybrid_results),
            "reranked_count": len(reranked_results),
            "overlap_vector_bm25": _calculate_overlap(vector_results, bm25_results),
            "overlap_hybrid_rerank": _calculate_overlap(
                hybrid_results,
                reranked_results,
            ),
        },
        "settings": {
            key: value
            for key, value in asdict(retrieval_settings).items()
            if key
            in {
                "vector_top_k",
                "bm25_top_k",
                "hybrid_top_k",
                "rerank_top_k",
                "vector_weight",
                "bm25_weight",
            }
        },
    }
