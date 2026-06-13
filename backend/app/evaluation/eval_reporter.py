import json
from pathlib import Path
from typing import Any

from app.evaluation.eval_models import EvalRegressionResult, EvalRunResult


def eval_run_to_dict(result: EvalRunResult) -> dict[str, Any]:
    return result.model_dump()


def _create_parent_directory(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create output directory for {path}: {exc}") from exc


def save_eval_result_json(result: EvalRunResult, output_path: str) -> None:
    path = Path(output_path)
    _create_parent_directory(path)
    try:
        path.write_text(
            json.dumps(eval_run_to_dict(result), indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Could not write evaluation JSON report to {path}: {exc}") from exc


def load_eval_result_json(path: str) -> dict[str, Any]:
    eval_path = Path(path)
    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Evaluation result JSON not found: {eval_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Evaluation result contains invalid JSON: {eval_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Could not read evaluation result JSON {eval_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation result JSON must contain an object: {eval_path}")
    return payload


def _display(value: str | None) -> str:
    return value if value else "Not provided"


def _format_case_ids(case_ids: list[str]) -> str:
    return ", ".join(case_ids) if case_ids else "None"


def _answer_preview(answer: str, limit: int = 300) -> str:
    normalized = " ".join(answer.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _escape_table_value(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def generate_markdown_report(
    result: EvalRunResult,
    regression: EvalRegressionResult | None = None,
) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        "## Metadata",
        f"- Run ID: {_display(result.run_id)}",
        f"- Created At: {_display(result.created_at)}",
        f"- User ID: {_display(result.user_id)}",
        f"- Eval File: {_display(result.eval_file)}",
        "",
        "## Summary",
        f"- Total: {result.total}",
        f"- Passed: {result.passed}",
        f"- Failed: {result.failed}",
        f"- Pass Rate: {result.pass_rate:.2f}%",
    ]

    if regression is not None:
        lines.extend(
            [
                "",
                "## Regression Summary",
                f"- Baseline Pass Rate: {regression.baseline_pass_rate:.2f}%",
                f"- Current Pass Rate: {regression.current_pass_rate:.2f}%",
                f"- Regressed Cases: {_format_case_ids(regression.regressed_case_ids)}",
                f"- Improved Cases: {_format_case_ids(regression.improved_case_ids)}",
                f"- New Cases: {_format_case_ids(regression.new_case_ids)}",
                f"- Removed Cases: {_format_case_ids(regression.removed_case_ids)}",
                f"- Regression Check Passed: {'Yes' if regression.passed else 'No'}",
            ]
        )

    lines.extend(["", "## Failed Cases"])
    failed_results = [case for case in result.results if not case.passed]
    if not failed_results:
        lines.append("")
        lines.append("None.")
    else:
        for case in failed_results:
            reasons = "; ".join(case.failure_reasons) or "No failure reason provided."
            lines.extend(
                [
                    "",
                    f"### {case.id}",
                    f"- Question: {case.question}",
                    f"- Failure Reasons: {reasons}",
                    f"- Validation Status: {_display(case.validation_status)}",
                    f"- Citation Count: {case.citation_count}",
                    f"- Evidence Chunk Count: {case.evidence_chunk_count}",
                ]
            )
            preview = _answer_preview(case.answer)
            if preview:
                lines.append(f"- Answer Preview: {preview}")

    lines.extend(
        [
            "",
            "## All Cases",
            "",
            "| Case ID | Passed | Status | Citation Count | Evidence Count |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for case in result.results:
        lines.append(
            "| "
            f"{_escape_table_value(case.id)} | "
            f"{'Yes' if case.passed else 'No'} | "
            f"{_escape_table_value(case.status)} | "
            f"{case.citation_count} | "
            f"{case.evidence_chunk_count} |"
        )

    return "\n".join(lines) + "\n"


def save_markdown_report(
    result: EvalRunResult,
    output_path: str,
    regression: EvalRegressionResult | None = None,
) -> None:
    path = Path(output_path)
    _create_parent_directory(path)
    try:
        path.write_text(
            generate_markdown_report(result, regression),
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Could not write Markdown report to {path}: {exc}") from exc
