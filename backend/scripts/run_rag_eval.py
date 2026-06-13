import argparse
import sys
from datetime import datetime, timezone
from uuid import uuid4

from app.db.database import SessionLocal
from app.evaluation.eval_loader import load_eval_cases
from app.evaluation.eval_regression import compare_eval_runs
from app.evaluation.eval_reporter import (
    load_eval_result_json,
    save_eval_result_json,
    save_markdown_report,
)
from app.evaluation.eval_runner import run_rag_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic RAG evaluations for one local user.",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="User ID that owns the sample documents.",
    )
    parser.add_argument(
        "--eval-file",
        default="eval/golden_questions.json",
        help="Path to the JSON evaluation dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Final retrieval result count.",
    )
    parser.add_argument("--hybrid-top-k", type=int, default=20)
    parser.add_argument("--vector-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=8)
    parser.add_argument("--vector-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--min-reranker-score", type=float)
    parser.add_argument(
        "--output-json",
        default="eval/results/latest_eval_result.json",
        help="Path for the current evaluation result JSON.",
    )
    parser.add_argument(
        "--output-md",
        default="eval/results/latest_eval_report.md",
        help="Path for the current evaluation Markdown report.",
    )
    parser.add_argument(
        "--baseline-json",
        help="Optional previous evaluation result JSON to compare against.",
    )
    return parser.parse_args()


def _print_evaluation_summary(result) -> None:
    print("RAG Evaluation Summary")
    print(f"Total: {result.total}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Pass rate: {result.pass_rate:.2f}%")

    failed_results = [
        case_result
        for case_result in result.results
        if not case_result.passed
    ]
    if failed_results:
        print("\nFailed cases:")
        for case_result in failed_results:
            reasons = "; ".join(case_result.failure_reasons)
            print(f"- {case_result.id}: {reasons}")


def _print_regression_summary(regression) -> None:
    print("\nRAG Regression Summary")
    print(f"Baseline pass rate: {regression.baseline_pass_rate:.2f}%")
    print(f"Current pass rate: {regression.current_pass_rate:.2f}%")
    print(f"Regressed cases: {len(regression.regressed_case_ids)}")
    print(f"Improved cases: {len(regression.improved_case_ids)}")
    print(f"New cases: {len(regression.new_case_ids)}")
    print(f"Removed cases: {len(regression.removed_case_ids)}")
    print(f"Regression check: {'PASSED' if regression.passed else 'FAILED'}")

    for reason in regression.failure_reasons:
        print(f"- {reason}")


def main() -> int:
    args = parse_args()

    try:
        cases = load_eval_cases(args.eval_file)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Could not load evaluation cases: {exc}", file=sys.stderr)
        return 1

    db = SessionLocal()
    try:
        try:
            result = run_rag_evaluation(
                db=db,
                user_id=args.user_id,
                cases=cases,
                top_k=args.top_k,
                hybrid_top_k=args.hybrid_top_k,
                vector_top_k=args.vector_top_k,
                bm25_top_k=args.bm25_top_k,
                rerank_top_k=args.rerank_top_k,
                vector_weight=args.vector_weight,
                bm25_weight=args.bm25_weight,
                min_reranker_score=args.min_reranker_score,
            )
        except ValueError as exc:
            print(f"Invalid retrieval settings: {exc}", file=sys.stderr)
            return 1
    finally:
        db.close()

    result.run_id = str(uuid4())
    result.created_at = datetime.now(timezone.utc).isoformat()
    result.user_id = args.user_id
    result.eval_file = args.eval_file

    regression = None
    if args.baseline_json:
        try:
            baseline = load_eval_result_json(args.baseline_json)
            regression = compare_eval_runs(baseline, result)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            print(f"Could not compare baseline evaluation: {exc}", file=sys.stderr)
            return 1

    try:
        save_eval_result_json(result, args.output_json)
        save_markdown_report(result, args.output_md, regression)
    except OSError as exc:
        print(f"Could not save evaluation reports: {exc}", file=sys.stderr)
        return 1

    _print_evaluation_summary(result)
    if regression is not None:
        _print_regression_summary(regression)

    print(f"\nJSON result: {args.output_json}")
    print(f"Markdown report: {args.output_md}")

    regression_failed = regression is not None and not regression.passed
    return 0 if result.failed == 0 and not regression_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
