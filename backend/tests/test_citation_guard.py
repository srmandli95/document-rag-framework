from app.generation.citation_guard import (
    REFUSAL_MESSAGE,
    build_citations_from_evidence,
    citation_chunk_ids_are_valid,
    has_sufficient_evidence,
    validate_answer_support,
)


def sample_evidence_chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_name": "sample_policy.pdf",
            "category": "health_insurance",
            "page_number": 2,
            "section_title": "Urgent Care",
            "chunk_index": 0,
            "chunk_text": "Urgent care is covered with a copay.",
            "reranker_score": 0.91,
            "hybrid_score": 0.82,
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "document_name": "sample_policy.pdf",
            "category": "health_insurance",
            "page_number": 3,
            "section_title": "Emergency Care",
            "chunk_index": 1,
            "chunk_text": "Emergency care may have a separate deductible.",
            "reranker_score": 0.55,
            "hybrid_score": 0.61,
        },
    ]


def test_build_citations_from_evidence_returns_expected_metadata():
    citations = build_citations_from_evidence(sample_evidence_chunks())

    assert len(citations) == 2

    first = citations[0]
    assert first["chunk_id"] == "chunk-1"
    assert first["document_id"] == "doc-1"
    assert first["document_name"] == "sample_policy.pdf"
    assert first["category"] == "health_insurance"
    assert first["page_number"] == 2
    assert first["section_title"] == "Urgent Care"
    assert first["chunk_index"] == 0
    assert first["reranker_score"] == 0.91
    assert first["hybrid_score"] == 0.82

    assert "chunk_text" not in first


def test_empty_evidence_returns_unsupported_refusal():
    result = validate_answer_support(
        answer="Urgent care is covered.",
        evidence_chunks=[],
        citations=[],
    )

    assert result["validation_status"] == "unsupported"
    assert result["final_answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []


def test_empty_answer_returns_unsupported_refusal():
    evidence = sample_evidence_chunks()
    citations = build_citations_from_evidence(evidence)

    result = validate_answer_support(
        answer="",
        evidence_chunks=evidence,
        citations=citations,
    )

    assert result["validation_status"] == "unsupported"
    assert result["final_answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []


def test_refusal_answer_returns_unsupported_with_empty_citations():
    evidence = sample_evidence_chunks()
    citations = build_citations_from_evidence(evidence)

    result = validate_answer_support(
        answer=REFUSAL_MESSAGE,
        evidence_chunks=evidence,
        citations=citations,
    )

    assert result["validation_status"] == "unsupported"
    assert result["final_answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []


def test_empty_citations_returns_unsupported_refusal():
    evidence = sample_evidence_chunks()

    result = validate_answer_support(
        answer="Urgent care is covered.",
        evidence_chunks=evidence,
        citations=[],
    )

    assert result["validation_status"] == "unsupported"
    assert result["final_answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []


def test_invalid_citation_chunk_id_returns_unsupported_refusal():
    evidence = sample_evidence_chunks()
    citations = build_citations_from_evidence(evidence)
    citations[0]["chunk_id"] = "fake-chunk-id"

    result = validate_answer_support(
        answer="Urgent care is covered.",
        evidence_chunks=evidence,
        citations=citations,
    )

    assert result["validation_status"] == "unsupported"
    assert result["final_answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []


def test_valid_citation_chunk_id_returns_supported():
    evidence = sample_evidence_chunks()
    citations = build_citations_from_evidence(evidence)

    result = validate_answer_support(
        answer="Urgent care is covered with a copay.",
        evidence_chunks=evidence,
        citations=citations,
    )

    assert result["validation_status"] == "supported"
    assert result["final_answer"] == "Urgent care is covered with a copay."
    assert result["citations"] == citations


def test_min_reranker_score_passes_when_at_least_one_chunk_score_is_high_enough():
    evidence = sample_evidence_chunks()
    citations = build_citations_from_evidence(evidence)

    result = validate_answer_support(
        answer="Urgent care is covered with a copay.",
        evidence_chunks=evidence,
        citations=citations,
        min_reranker_score=0.9,
    )

    assert result["validation_status"] == "supported"


def test_min_reranker_score_fails_when_all_scores_are_below_threshold():
    evidence = sample_evidence_chunks()
    citations = build_citations_from_evidence(evidence)

    result = validate_answer_support(
        answer="Urgent care is covered with a copay.",
        evidence_chunks=evidence,
        citations=citations,
        min_reranker_score=0.95,
    )

    assert result["validation_status"] == "unsupported"
    assert result["final_answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []


def test_unsupported_result_always_returns_refusal_message_and_empty_citations():
    result = validate_answer_support(
        answer="   ",
        evidence_chunks=[],
        citations=[{"chunk_id": "fake"}],
    )

    assert result["validation_status"] == "unsupported"
    assert result["final_answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []


def test_citation_chunk_ids_are_valid_returns_true_for_valid_ids():
    evidence = sample_evidence_chunks()
    citations = build_citations_from_evidence(evidence)

    assert citation_chunk_ids_are_valid(citations, evidence) is True


def test_citation_chunk_ids_are_valid_returns_false_for_invalid_ids():
    evidence = sample_evidence_chunks()
    citations = [{"chunk_id": "not-real"}]

    assert citation_chunk_ids_are_valid(citations, evidence) is False


def test_has_sufficient_evidence_returns_true_when_enough_chunks_exist():
    assert has_sufficient_evidence(sample_evidence_chunks(), min_evidence_chunks=1) is True


def test_has_sufficient_evidence_returns_false_when_not_enough_chunks_exist():
    assert has_sufficient_evidence([], min_evidence_chunks=1) is False