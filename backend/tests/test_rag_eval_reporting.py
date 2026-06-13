import json

import pytest

from app.evaluation.eval_models import (
    EvalCaseResult,
    EvalRegressionResult,
    EvalRunResult,
)
from app.evaluation.eval_regression import compare_eval_runs
from app.evaluation.eval_reporter import (
    eval_run_to_dict,
    generate_markdown_report,
    load_eval_result_json,
    save_eval_result_json,
    save_markdown_report,
)


def make_case_result(
    case_id: str,
    passed: bool,
    answer: str = "",
) -> EvalCaseResult:
    return EvalCaseResult(
        id=case_id,
        question=f"Question for {case_id}?",
        status="passed" if passed else "failed",
        passed=passed,
        answer=answer,
        validation_status="supported" if passed else "unsupported",
        expected_refusal=False,
        actual_refusal=False,
        citation_count=1 if passed else 0,
        evidence_chunk_count=1 if passed else 0,
        checks={"example_check": passed},
        failure_reasons=[] if passed else ["Missing expected term: copay"],
    )


def make_run(*cases: EvalCaseResult) -> EvalRunResult:
    passed = sum(case.passed for case in cases)
    total = len(cases)
    return EvalRunResult(
        run_id="run-123",
        user_id="local-user-123",
        eval_file="eval/golden_questions.json",
        created_at="2026-06-13T12:00:00+00:00",
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=(passed / total * 100.0) if total else 0.0,
        results=list(cases),
    )


def make_regression() -> EvalRegressionResult:
    return EvalRegressionResult(
        baseline_total=2,
        current_total=3,
        baseline_passed=2,
        current_passed=2,
        baseline_pass_rate=100.0,
        current_pass_rate=66.67,
        regressed_case_ids=["regressed-case"],
        improved_case_ids=[],
        unchanged_failed_case_ids=[],
        unchanged_passed_case_ids=["passing-case"],
        new_case_ids=["new-case"],
        removed_case_ids=[],
        passed=False,
        failure_reasons=["Regressed cases: regressed-case"],
    )


def test_eval_run_to_dict_serializes_result():
    result = make_run(make_case_result("passing-case", True))

    payload = eval_run_to_dict(result)

    assert payload["run_id"] == "run-123"
    assert payload["results"][0]["id"] == "passing-case"
    assert payload["results"][0]["passed"] is True


def test_save_and_load_eval_result_json(tmp_path):
    output_path = tmp_path / "nested" / "result.json"
    result = make_run(make_case_result("passing-case", True))

    save_eval_result_json(result, str(output_path))
    payload = load_eval_result_json(str(output_path))

    assert output_path.is_file()
    assert payload == result.model_dump()


def test_load_eval_result_json_rejects_invalid_json(tmp_path):
    output_path = tmp_path / "invalid.json"
    output_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_eval_result_json(str(output_path))


def test_load_eval_result_json_rejects_non_object(tmp_path):
    output_path = tmp_path / "list.json"
    output_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain an object"):
        load_eval_result_json(str(output_path))


def test_generate_markdown_report_includes_metadata_and_summary():
    report = generate_markdown_report(
        make_run(
            make_case_result("passing-case", True),
            make_case_result("failing-case", False),
        )
    )

    assert "# RAG Evaluation Report" in report
    assert "- Run ID: run-123" in report
    assert "- Total: 2" in report
    assert "- Passed: 1" in report
    assert "- Failed: 1" in report
    assert "- Pass Rate: 50.00%" in report


def test_generate_markdown_report_includes_regression_summary():
    report = generate_markdown_report(
        make_run(make_case_result("passing-case", True)),
        make_regression(),
    )

    assert "## Regression Summary" in report
    assert "- Baseline Pass Rate: 100.00%" in report
    assert "- Regressed Cases: regressed-case" in report
    assert "- New Cases: new-case" in report
    assert "- Regression Check Passed: No" in report


def test_generate_markdown_report_includes_failed_case_details():
    report = generate_markdown_report(
        make_run(make_case_result("health_urgent_care_001", False, "No copay found."))
    )

    assert "### health_urgent_care_001" in report
    assert "- Question: Question for health_urgent_care_001?" in report
    assert "- Failure Reasons: Missing expected term: copay" in report
    assert "- Validation Status: unsupported" in report
    assert "- Citation Count: 0" in report
    assert "- Evidence Chunk Count: 0" in report
    assert "- Answer Preview: No copay found." in report


def test_generate_markdown_report_truncates_and_normalizes_failed_answer():
    answer = "word\n" + ("x" * 400)
    report = generate_markdown_report(
        make_run(make_case_result("failing-case", False, answer))
    )
    preview_line = next(
        line for line in report.splitlines() if line.startswith("- Answer Preview:")
    )

    assert "\n" not in preview_line
    assert preview_line.endswith("...")
    assert len(preview_line.removeprefix("- Answer Preview: ")) == 300
    assert "x" * 350 not in report


def test_generate_markdown_report_includes_all_cases_table():
    report = generate_markdown_report(
        make_run(
            make_case_result("passing-case", True),
            make_case_result("failing-case", False),
        )
    )

    assert "| Case ID | Passed | Status | Citation Count | Evidence Count |" in report
    assert "| passing-case | Yes | passed | 1 | 1 |" in report
    assert "| failing-case | No | failed | 0 | 0 |" in report


def test_save_markdown_report_writes_file(tmp_path):
    output_path = tmp_path / "nested" / "report.md"

    save_markdown_report(
        make_run(make_case_result("passing-case", True)),
        str(output_path),
    )

    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8").startswith(
        "# RAG Evaluation Report"
    )


def test_compare_eval_runs_categorizes_cases_and_sorts_ids():
    baseline = make_run(
        make_case_result("removed-case", True),
        make_case_result("unchanged-failed", False),
        make_case_result("improved-case", False),
        make_case_result("regressed-z", True),
        make_case_result("unchanged-passed", True),
        make_case_result("regressed-a", True),
    )
    current = make_run(
        make_case_result("new-case", True),
        make_case_result("regressed-z", False),
        make_case_result("unchanged-passed", True),
        make_case_result("improved-case", True),
        make_case_result("unchanged-failed", False),
        make_case_result("regressed-a", False),
    )

    regression = compare_eval_runs(baseline, current)

    assert regression.regressed_case_ids == ["regressed-a", "regressed-z"]
    assert regression.improved_case_ids == ["improved-case"]
    assert regression.unchanged_failed_case_ids == ["unchanged-failed"]
    assert regression.unchanged_passed_case_ids == ["unchanged-passed"]
    assert regression.new_case_ids == ["new-case"]
    assert regression.removed_case_ids == ["removed-case"]
    assert regression.passed is False
    assert regression.failure_reasons == [
        "Regressed cases: regressed-a, regressed-z"
    ]


def test_compare_eval_runs_accepts_dicts():
    baseline = make_run(make_case_result("case-1", False)).model_dump()
    current = make_run(make_case_result("case-1", True)).model_dump()

    regression = compare_eval_runs(baseline, current)

    assert regression.improved_case_ids == ["case-1"]
    assert regression.passed is True
    assert regression.failure_reasons == []


def test_compare_eval_runs_passes_when_only_new_failure_exists():
    baseline = make_run(make_case_result("existing-case", True))
    current = make_run(
        make_case_result("existing-case", True),
        make_case_result("new-failing-case", False),
    )

    regression = compare_eval_runs(baseline, current)

    assert regression.new_case_ids == ["new-failing-case"]
    assert regression.regressed_case_ids == []
    assert regression.passed is True


def test_compare_eval_runs_rejects_duplicate_case_ids():
    baseline = make_run(
        make_case_result("duplicate-case", True),
        make_case_result("duplicate-case", False),
    )
    current = make_run()

    with pytest.raises(ValueError, match="duplicate case id"):
        compare_eval_runs(baseline, current)
