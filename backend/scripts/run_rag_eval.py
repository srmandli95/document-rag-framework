import argparse
import sys

from app.db.database import SessionLocal
from app.evaluation.eval_loader import load_eval_cases
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cases = load_eval_cases(args.eval_file)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Could not load evaluation cases: {exc}", file=sys.stderr)
        return 1

    db = SessionLocal()
    try:
        result = run_rag_evaluation(
            db=db,
            user_id=args.user_id,
            cases=cases,
            top_k=args.top_k,
        )
    finally:
        db.close()

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

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
