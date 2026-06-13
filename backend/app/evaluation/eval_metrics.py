from typing import Any

from app.evaluation.eval_models import EvalCase, EvalCaseResult


REFUSAL_PHRASES = (
    "could not find enough evidence",
    "cannot find enough evidence",
    "not enough evidence",
    "unable to answer",
    "cannot answer",
    "can't answer",
)


def contains_expected_terms(answer: str, expected_terms: list[str]) -> bool:
    normalized_answer = answer.casefold()
    return all(term.casefold() in normalized_answer for term in expected_terms)


def citations_contain_expected_documents(
    citations: list[dict],
    expected_document_terms: list[str],
) -> bool:
    if not expected_document_terms:
        return True

    document_values = [
        str(
            citation.get("document_name")
            or citation.get("file_name")
            or ""
        ).casefold()
        for citation in citations
    ]

    return all(
        any(term.casefold() in document_value for document_value in document_values)
        for term in expected_document_terms
    )


def is_refusal_response(answer_response: dict) -> bool:
    status = str(answer_response.get("status") or "").casefold()
    validation_status = str(
        answer_response.get("validation_status") or ""
    ).casefold()
    answer = str(answer_response.get("answer") or "").casefold()

    return (
        status == "refused"
        or validation_status in {"unsupported", "refused"}
        or any(phrase in answer for phrase in REFUSAL_PHRASES)
    )


def citations_are_present_if_needed(
    answer_response: dict,
    expected_refusal: bool,
) -> bool:
    citations = answer_response.get("citations") or []
    return not citations if expected_refusal else bool(citations)


def _citation_chunk_ids_are_valid(answer_response: dict) -> bool:
    citations = answer_response.get("citations") or []
    evidence_chunks = answer_response.get("evidence_chunks") or []

    citation_ids = {
        citation.get("chunk_id")
        for citation in citations
        if citation.get("chunk_id")
    }
    evidence_ids = {
        chunk.get("chunk_id") or chunk.get("id")
        for chunk in evidence_chunks
        if chunk.get("chunk_id") or chunk.get("id")
    }

    return bool(citation_ids) and citation_ids.issubset(evidence_ids)


def _evidence_chunk_count(answer_response: dict[str, Any]) -> int:
    count = answer_response.get("evidence_chunk_count")
    if isinstance(count, int):
        return count
    return len(answer_response.get("evidence_chunks") or [])


def evaluate_answer_response(
    case: EvalCase,
    answer_response: dict,
) -> EvalCaseResult:
    answer = str(answer_response.get("answer") or "")
    citations = answer_response.get("citations") or []
    citation_count = len(citations)
    evidence_chunk_count = _evidence_chunk_count(answer_response)
    validation_status = answer_response.get("validation_status")
    actual_refusal = is_refusal_response(answer_response)

    if case.expected_refusal:
        checks = {
            "expected_refusal_detected": actual_refusal,
            "refusal_has_no_citations": citation_count == 0,
        }
        failure_messages = {
            "expected_refusal_detected": (
                "Expected a refusal but received a supported answer."
            ),
            "refusal_has_no_citations": "Expected refusal response to contain no citations.",
        }
    else:
        validation_supported = (
            validation_status is None
            or str(validation_status).casefold() == "supported"
        )
        checks = {
            "not_refused": not actual_refusal,
            "expected_answer_terms": contains_expected_terms(
                answer,
                case.expected_answer_contains,
            ),
            "citations_present": citations_are_present_if_needed(
                answer_response,
                expected_refusal=False,
            ),
            "expected_documents_cited": citations_contain_expected_documents(
                citations,
                case.expected_citation_document_contains,
            ),
            "evidence_present": evidence_chunk_count > 0,
            "citation_chunk_ids_valid": _citation_chunk_ids_are_valid(
                answer_response
            ),
            "validation_supported": validation_supported,
        }
        failure_messages = {
            "not_refused": "Expected a supported answer but received a refusal.",
            "expected_answer_terms": "Answer did not contain all expected terms.",
            "citations_present": "Supported answer did not include citations.",
            "expected_documents_cited": (
                "Citations did not include all expected documents."
            ),
            "evidence_present": "Supported answer did not include retrieved evidence.",
            "citation_chunk_ids_valid": (
                "Citation chunk IDs did not match retrieved evidence."
            ),
            "validation_supported": "Answer validation status was not supported.",
        }

    failure_reasons = [
        failure_messages[name]
        for name, passed in checks.items()
        if not passed
    ]
    passed = all(checks.values())

    return EvalCaseResult(
        id=case.id,
        question=case.question,
        status="passed" if passed else "failed",
        passed=passed,
        answer=answer,
        validation_status=validation_status,
        expected_refusal=case.expected_refusal,
        actual_refusal=actual_refusal,
        citation_count=citation_count,
        evidence_chunk_count=evidence_chunk_count,
        checks=checks,
        failure_reasons=failure_reasons,
    )
