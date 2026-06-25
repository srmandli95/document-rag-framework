import json
from pathlib import Path

from pydantic import ValidationError

from app.evaluation.eval_models import EvalCase


def load_eval_cases(path: str) -> list[EvalCase]:
    """Load evaluation cases from a JSON file."""
    eval_path = Path(path)

    if not eval_path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Evaluation file contains invalid JSON: {eval_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Could not read evaluation file: {eval_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"Evaluation file must contain a JSON list of cases: {eval_path}"
        )

    cases: list[EvalCase] = []
    for index, case_payload in enumerate(payload):
        try:
            cases.append(EvalCase.model_validate(case_payload))
        except ValidationError as exc:
            raise ValueError(
                f"Invalid evaluation case at index {index} in {eval_path}: {exc}"
            ) from exc

    return cases
