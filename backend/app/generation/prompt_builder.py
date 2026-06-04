from pathlib import Path
from typing import Any

import yaml


PROMPTS_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts.yaml"


def load_prompts() -> dict[str, str]:
    """
    Load prompt templates from backend/app/config/prompts.yaml.
    """
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Prompts file not found at: {PROMPTS_PATH}")

    with PROMPTS_PATH.open("r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file) or {}

    if "answer_generation_prompt" not in prompts:
        raise KeyError("answer_generation_prompt is missing from prompts.yaml")

    if "refusal_message" not in prompts:
        raise KeyError("refusal_message is missing from prompts.yaml")

    return prompts


def _value_or_na(value: Any) -> str:
    """
    Convert None or empty values to N/A for readable prompt formatting.
    """
    if value is None:
        return "N/A"

    value_as_string = str(value).strip()
    return value_as_string if value_as_string else "N/A"


def build_evidence_context(evidence_chunks: list[dict[str, Any]]) -> str:
    """
    Build a readable evidence context from reranked chunks.

    Each chunk includes metadata that the LLM can use for source references.
    """
    if not evidence_chunks:
        return ""

    formatted_chunks: list[str] = []

    for chunk in evidence_chunks:
        chunk_id = _value_or_na(chunk.get("chunk_id"))
        document_name = _value_or_na(chunk.get("document_name"))
        category = _value_or_na(chunk.get("category"))
        page_number = _value_or_na(chunk.get("page_number"))
        section_title = _value_or_na(chunk.get("section_title"))
        chunk_text = _value_or_na(chunk.get("chunk_text"))

        formatted_chunk = f"""[Chunk ID: {chunk_id}]
Document: {document_name}
Category: {category}
Page: {page_number}
Section: {section_title}
Text:
{chunk_text}
"""

        formatted_chunks.append(formatted_chunk)

    return "\n---\n".join(formatted_chunks)


def build_answer_prompt(question: str, evidence_chunks: list[dict[str, Any]]) -> str:
    """
    Build the final grounded answer prompt using the user question
    and reranked evidence chunks.
    """
    if not question or not question.strip():
        raise ValueError("question is required")

    prompts = load_prompts()
    evidence_context = build_evidence_context(evidence_chunks)

    return prompts["answer_generation_prompt"].format(
        question=question.strip(),
        evidence_context=evidence_context,
    )