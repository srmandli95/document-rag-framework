import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.db.database import SessionLocal
from app.retrieval.retrieval_diagnostics import diagnose_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval methods for one user-owned query.",
    )
    parser.add_argument("--user-id", help="User ID that owns the documents.")
    parser.add_argument("--query", help="Query to inspect.")
    parser.add_argument("--vector-top-k", type=int, default=10)
    parser.add_argument("--bm25-top-k", type=int, default=10)
    parser.add_argument("--hybrid-top-k", type=int, default=10)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--output-json", help="Optional diagnostics JSON output path.")
    return parser.parse_args()


def _score_text(result: dict[str, Any]) -> str:
    score_names = (
        "reranker_score",
        "hybrid_score",
        "bm25_score",
        "similarity_score",
    )
    scores = [
        f"{name.removesuffix('_score')}={float(result[name]):.4f}"
        for name in score_names
        if result.get(name) is not None
    ]
    return " | ".join(scores) if scores else "no score"


def _print_results(title: str, results: list[dict[str, Any]]) -> None:
    print(f"\n{title}:")
    if not results:
        print("- None")
        return

    for rank, result in enumerate(results, start=1):
        document_name = result.get("document_name") or "Unknown document"
        chunk_index = result.get("chunk_index")
        chunk_label = f"Chunk {chunk_index}" if chunk_index is not None else "Chunk ?"
        print(f"{rank}. {document_name} | {chunk_label} | {_score_text(result)}")


def _print_diagnostics(diagnostics: dict[str, Any]) -> None:
    summary = diagnostics["summary"]
    print("Retrieval Diagnostics")
    print(f"Query: {diagnostics['query']}")
    print("\nSettings:")
    for name, value in diagnostics["settings"].items():
        print(f"- {name}: {value}")
    print("\nCounts:")
    print(f"- Vector: {summary['vector_count']}")
    print(f"- BM25: {summary['bm25_count']}")
    print(f"- Hybrid: {summary['hybrid_count']}")
    print(f"- Reranked: {summary['reranked_count']}")
    print("\nOverlap:")
    print(f"- Vector/BM25: {summary['overlap_vector_bm25']}")
    print(f"- Hybrid/Rerank: {summary['overlap_hybrid_rerank']}")

    _print_results("Top Vector", diagnostics["vector_results"])
    _print_results("Top BM25", diagnostics["bm25_results"])
    _print_results("Top Hybrid", diagnostics["hybrid_results"])
    _print_results("Top Reranked", diagnostics["reranked_results"])

    print("\nRank Changes:")
    if not diagnostics["rank_changes"]:
        print("- None")
    for change in diagnostics["rank_changes"]:
        direction = "moved up" if change["rank_delta"] > 0 else "moved down"
        if change["rank_delta"] == 0:
            direction = "stayed"
        print(
            f"- {change['chunk_id']} {direction} from "
            f"{change['before_rank']} to {change['after_rank']}"
        )


def _validate_args(args: argparse.Namespace) -> None:
    if not args.user_id or not args.user_id.strip():
        raise ValueError("user_id is required")
    if not args.query or not args.query.strip():
        raise ValueError("query is required")

def main() -> int:
    args = parse_args()
    try:
        _validate_args(args)
        db = SessionLocal()
        try:
            diagnostics = diagnose_retrieval(
                db=db,
                user_id=args.user_id,
                query=args.query,
                vector_top_k=args.vector_top_k,
                bm25_top_k=args.bm25_top_k,
                hybrid_top_k=args.hybrid_top_k,
                rerank_top_k=args.rerank_top_k,
                vector_weight=args.vector_weight,
                bm25_weight=args.bm25_weight,
            )
        finally:
            db.close()

        _print_diagnostics(diagnostics)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(diagnostics, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"\nJSON result: {output_path}")
    except Exception as exc:
        print(f"Retrieval diagnostics failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
