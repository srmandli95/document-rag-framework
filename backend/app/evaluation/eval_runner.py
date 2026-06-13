from typing import Any

from app.evaluation.eval_metrics import evaluate_answer_response
from app.evaluation.eval_models import EvalCase, EvalCaseResult, EvalRunResult
from app.generation.answer_generator import generate_answer_from_evidence


def run_rag_evaluation(
    db: Any,
    user_id: str,
    cases: list[EvalCase],
    top_k: int = 5,
) -> EvalRunResult:
    results: list[EvalCaseResult] = []

    for case in cases:
        try:
            answer_response = generate_answer_from_evidence(
                db=db,
                user_id=user_id,
                question=case.question,
                top_k=top_k,
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
