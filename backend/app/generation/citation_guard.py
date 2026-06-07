from typing import Any


REFUSAL_MESSAGE = (
    "I could not find enough evidence in your uploaded documents to answer this question. "
    "Please upload the relevant document or ask a question covered by your existing documents."
)


CITATION_FIELDS = [
    "chunk_id",
    "document_id",
    "document_name",
    "category",
    "page_number",
    "section_title",
    "chunk_index",
    "reranker_score",
    "hybrid_score",
    "vector_score",
    "bm25_score",
]


def build_citations_from_evidence(
    evidence_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []

    for chunk in evidence_chunks:
        citation = {
            field: chunk.get(field)
            for field in CITATION_FIELDS
            if field in chunk
        }

        if "chunk_id" not in citation:
            citation["chunk_id"] = chunk.get("id")

        citations.append(citation)

    return citations


def citation_chunk_ids_are_valid(
    citations: list[dict[str, Any]],
    evidence_chunks: list[dict[str, Any]],
) -> bool:
    evidence_chunk_ids = {
        chunk.get("chunk_id") or chunk.get("id")
        for chunk in evidence_chunks
        if chunk.get("chunk_id") or chunk.get("id")
    }

    if not evidence_chunk_ids:
        return False

    citation_chunk_ids = {
        citation.get("chunk_id")
        for citation in citations
        if citation.get("chunk_id")
    }

    if not citation_chunk_ids:
        return False

    return citation_chunk_ids.issubset(evidence_chunk_ids)


def check_evidence_sufficiency(
    evidence_chunks: list[dict[str, Any]],
    min_evidence_chunks: int = 1,
    min_reranker_score: float | None = None,
) -> dict[str, Any]:
    if not evidence_chunks:
        return {
            "evidence_sufficient": False,
            "reason": "No evidence chunks were retrieved.",
        }

    if len(evidence_chunks) < min_evidence_chunks:
        return {
            "evidence_sufficient": False,
            "reason": (
                f"Only {len(evidence_chunks)} evidence chunk(s) were retrieved, "
                f"but at least {min_evidence_chunks} are required."
            ),
        }

    if min_reranker_score is not None:
        has_strong_chunk = any(
            chunk.get("reranker_score") is not None
            and float(chunk["reranker_score"]) >= min_reranker_score
            for chunk in evidence_chunks
        )

        if not has_strong_chunk:
            return {
                "evidence_sufficient": False,
                "reason": (
                    "Retrieved evidence did not meet the minimum reranker score "
                    f"threshold of {min_reranker_score}."
                ),
            }

    return {
        "evidence_sufficient": True,
        "reason": "Retrieved evidence is sufficient for answer generation.",
    }


def has_sufficient_evidence(
    evidence_chunks: list[dict[str, Any]],
    min_evidence_chunks: int = 1,
    min_reranker_score: float | None = None,
) -> bool:
    """
    Backward-compatible helper for older tests/code.

    Day 14 uses check_evidence_sufficiency() because it returns both:
    - evidence_sufficient
    - reason

    This wrapper preserves the old boolean behavior.
    """
    result = check_evidence_sufficiency(
        evidence_chunks=evidence_chunks,
        min_evidence_chunks=min_evidence_chunks,
        min_reranker_score=min_reranker_score,
    )

    return bool(result["evidence_sufficient"])


def _unsupported_result(reason: str) -> dict[str, Any]:
    return {
        "validation_status": "unsupported",
        "validation_reason": reason,
        "final_answer": REFUSAL_MESSAGE,
        "citations": [],
    }


def _supported_result(
    answer: str,
    citations: list[dict[str, Any]],
    reason: str = "Answer citations match retrieved evidence chunks.",
) -> dict[str, Any]:
    return {
        "validation_status": "supported",
        "validation_reason": reason,
        "final_answer": answer,
        "citations": citations,
    }


def validate_answer_support(
    answer: str | None,
    citations: list[dict[str, Any]],
    evidence_chunks: list[dict[str, Any]],
    min_reranker_score: float | None = None,
) -> dict[str, Any]:
    """
    Validate whether the generated answer is supported by retrieved evidence.

    This function intentionally returns:
    - validation_status
    - validation_reason
    - final_answer
    - citations

    Older tests and the Day 13/Day 14 graph expect this shape.
    """
    if not answer or not answer.strip():
        return _unsupported_result("No generated answer was provided.")

    cleaned_answer = answer.strip()

    if cleaned_answer == REFUSAL_MESSAGE:
        return _unsupported_result("Generated answer is already a refusal message.")

    evidence_result = check_evidence_sufficiency(
        evidence_chunks=evidence_chunks,
        min_evidence_chunks=1,
        min_reranker_score=min_reranker_score,
    )

    if not evidence_result["evidence_sufficient"]:
        return _unsupported_result(str(evidence_result["reason"]))

    if not citations:
        return _unsupported_result("No citations were provided.")

    if not citation_chunk_ids_are_valid(citations, evidence_chunks):
        return _unsupported_result(
            "One or more citations do not match retrieved evidence chunks."
        )

    return _supported_result(cleaned_answer, citations)