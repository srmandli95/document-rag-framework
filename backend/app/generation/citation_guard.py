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
]


def build_citations_from_evidence(evidence_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []

    for chunk in evidence_chunks:
        citation = {
            field: chunk.get(field)
            for field in CITATION_FIELDS
        }
        citations.append(citation)

    return citations


def citation_chunk_ids_are_valid(
    citations: list[dict[str, Any]],
    evidence_chunks: list[dict[str, Any]],
) -> bool:
    evidence_chunk_ids = {
        chunk.get("chunk_id")
        for chunk in evidence_chunks
        if chunk.get("chunk_id")
    }

    if not evidence_chunk_ids:
        return False

    for citation in citations:
        citation_chunk_id = citation.get("chunk_id")

        if not citation_chunk_id:
            return False

        if citation_chunk_id not in evidence_chunk_ids:
            return False

    return True


def has_sufficient_evidence(
    evidence_chunks: list[dict[str, Any]],
    min_evidence_chunks: int = 1,
) -> bool:
    if min_evidence_chunks < 1:
        min_evidence_chunks = 1

    return len(evidence_chunks) >= min_evidence_chunks


def _unsupported(reason: str) -> dict[str, Any]:
    return {
        "validation_status": "unsupported",
        "reason": reason,
        "final_answer": REFUSAL_MESSAGE,
        "citations": [],
    }


def _supported(
    answer: str,
    citations: list[dict[str, Any]],
    reason: str = "Answer is supported by provided evidence and valid citations.",
) -> dict[str, Any]:
    return {
        "validation_status": "supported",
        "reason": reason,
        "final_answer": answer,
        "citations": citations,
    }


def _has_passing_reranker_score(
    evidence_chunks: list[dict[str, Any]],
    min_reranker_score: float,
) -> bool:
    for chunk in evidence_chunks:
        score = chunk.get("reranker_score")

        if score is None:
            continue

        try:
            if float(score) >= min_reranker_score:
                return True
        except (TypeError, ValueError):
            continue

    return False


def validate_answer_support(
    answer: str,
    evidence_chunks: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    min_evidence_chunks: int = 1,
    min_reranker_score: float | None = None,
) -> dict[str, Any]:
    if not has_sufficient_evidence(evidence_chunks, min_evidence_chunks):
        return _unsupported("Not enough evidence chunks were provided.")

    if not answer or not answer.strip():
        return _unsupported("Generated answer was empty.")

    normalized_answer = answer.strip()

    if REFUSAL_MESSAGE in normalized_answer:
        return _unsupported("Generated answer was already a refusal.")

    if not citations:
        return _unsupported("Generated answer did not include citation metadata.")

    if not citation_chunk_ids_are_valid(citations, evidence_chunks):
        return _unsupported("One or more citation chunk IDs are missing or not present in evidence chunks.")

    if min_reranker_score is not None:
        if not _has_passing_reranker_score(evidence_chunks, min_reranker_score):
            return _unsupported(
                f"No evidence chunk met the minimum reranker score threshold: {min_reranker_score}."
            )

    return _supported(normalized_answer, citations)