from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    category: str
    question: str
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_citation_document_contains: list[str] = Field(default_factory=list)
    expected_refusal: bool = False


class EvalCaseResult(BaseModel):
    id: str
    question: str
    status: str
    passed: bool
    answer: str = ""
    validation_status: str | None = None
    expected_refusal: bool
    actual_refusal: bool
    citation_count: int = 0
    evidence_chunk_count: int = 0
    checks: dict[str, bool] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[EvalCaseResult] = Field(default_factory=list)
