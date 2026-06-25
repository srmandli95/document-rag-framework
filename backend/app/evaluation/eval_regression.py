from typing import Any

from app.evaluation.eval_models import EvalRegressionResult, EvalRunResult


def _run_to_dict(run: dict[str, Any] | EvalRunResult) -> dict[str, Any]:
    """Serialize an evaluation run when needed for comparison."""
    if isinstance(run, EvalRunResult):
        return run.model_dump()
    if isinstance(run, dict):
        return run
    raise TypeError("Evaluation run must be a dictionary or EvalRunResult.")


def _get_case_map(run: dict[str, Any] | EvalRunResult) -> dict[str, bool]:
    """Index evaluation case results by case id."""
    run_data = _run_to_dict(run)
    results = run_data.get("results")
    if not isinstance(results, list):
        raise ValueError("Evaluation run must contain a results list.")

    case_map: dict[str, bool] = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"Evaluation result at index {index} must be an object.")

        case_id = result.get("id")
        passed = result.get("passed")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Evaluation result at index {index} has no valid id.")
        if not isinstance(passed, bool):
            raise ValueError(
                f"Evaluation result '{case_id}' must have a boolean passed value."
            )
        if case_id in case_map:
            raise ValueError(f"Evaluation run contains duplicate case id: {case_id}")

        case_map[case_id] = passed

    return case_map


def _number(run: dict[str, Any], name: str, default: int | float) -> int | float:
    """Convert a metric value to a float for regression comparison."""
    value = run.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Evaluation run field '{name}' must be numeric.")
    return value


def compare_eval_runs(
    baseline: dict[str, Any] | EvalRunResult,
    current: dict[str, Any] | EvalRunResult,
) -> EvalRegressionResult:
    """Compare current evaluation metrics against a baseline run."""
    baseline_data = _run_to_dict(baseline)
    current_data = _run_to_dict(current)
    baseline_cases = _get_case_map(baseline_data)
    current_cases = _get_case_map(current_data)

    shared_case_ids = sorted(baseline_cases.keys() & current_cases.keys())
    regressed_case_ids = [
        case_id
        for case_id in shared_case_ids
        if baseline_cases[case_id] and not current_cases[case_id]
    ]
    improved_case_ids = [
        case_id
        for case_id in shared_case_ids
        if not baseline_cases[case_id] and current_cases[case_id]
    ]
    unchanged_failed_case_ids = [
        case_id
        for case_id in shared_case_ids
        if not baseline_cases[case_id] and not current_cases[case_id]
    ]
    unchanged_passed_case_ids = [
        case_id
        for case_id in shared_case_ids
        if baseline_cases[case_id] and current_cases[case_id]
    ]
    new_case_ids = sorted(current_cases.keys() - baseline_cases.keys())
    removed_case_ids = sorted(baseline_cases.keys() - current_cases.keys())
    passed = not regressed_case_ids
    failure_reasons = (
        [f"Regressed cases: {', '.join(regressed_case_ids)}"]
        if regressed_case_ids
        else []
    )

    return EvalRegressionResult(
        baseline_total=int(_number(baseline_data, "total", len(baseline_cases))),
        current_total=int(_number(current_data, "total", len(current_cases))),
        baseline_passed=int(
            _number(baseline_data, "passed", sum(baseline_cases.values()))
        ),
        current_passed=int(
            _number(current_data, "passed", sum(current_cases.values()))
        ),
        baseline_pass_rate=float(_number(baseline_data, "pass_rate", 0.0)),
        current_pass_rate=float(_number(current_data, "pass_rate", 0.0)),
        regressed_case_ids=regressed_case_ids,
        improved_case_ids=improved_case_ids,
        unchanged_failed_case_ids=unchanged_failed_case_ids,
        unchanged_passed_case_ids=unchanged_passed_case_ids,
        new_case_ids=new_case_ids,
        removed_case_ids=removed_case_ids,
        passed=passed,
        failure_reasons=failure_reasons,
    )
