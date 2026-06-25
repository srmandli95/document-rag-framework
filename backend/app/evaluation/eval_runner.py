from typing import Any

from app.evaluation.eval_metrics import evaluate_answer_response
from app.evaluation.eval_models import EvalCase, EvalCaseResult, EvalRunResult
from app.generation.answer_generator import generate_answer_from_evidence
from app.retrieval.retrieval_settings import (
    RetrievalSettings,
    validate_retrieval_settings,
)


def run_rag_evaluation(
    db: Any,
    user_id: str,
    cases: list[EvalCase],
    top_k: int = 5,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    rerank_top_k: int = 8,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    min_reranker_score: float | None = None,
) -> EvalRunResult:
    """Run the configured RAG evaluation cases against the app."""
    retrieval_settings = validate_retrieval_settings(
        RetrievalSettings(
            top_k=top_k,
            hybrid_top_k=hybrid_top_k,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_k=rerank_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            min_reranker_score=min_reranker_score,
        ),
        require_final_top_k_within_rerank=True,
    )
    results: list[EvalCaseResult] = []

    for case in cases:
        try:
            answer_response = generate_answer_from_evidence(
                db=db,
                user_id=user_id,
                question=case.question,
                top_k=retrieval_settings.top_k,
                hybrid_top_k=retrieval_settings.hybrid_top_k,
                vector_top_k=retrieval_settings.vector_top_k,
                bm25_top_k=retrieval_settings.bm25_top_k,
                rerank_top_k=retrieval_settings.rerank_top_k,
                vector_weight=retrieval_settings.vector_weight,
                bm25_weight=retrieval_settings.bm25_weight,
                min_reranker_score=retrieval_settings.min_reranker_score,
            )
            results.append(evaluate_answer_response(case, answer_response))
        except Exception as exc:
            results.append(
                EvalCaseResult(
                    id=case.id,
                    question=case.question,
                    status="error",
                    passed=False,
                    expected_refusal=case.expected_refusal,
                    actual_refusal=False,
                    checks={"evaluation_completed": False},
                    failure_reasons=[f"Evaluation error: {exc}"],
                )
            )

    total = len(results)
    passed = sum(result.passed for result in results)
    failed = total - passed
    pass_rate = (passed / total * 100.0) if total else 0.0

    return EvalRunResult(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        results=results,
    )
