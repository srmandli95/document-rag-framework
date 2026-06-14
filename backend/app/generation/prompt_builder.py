from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml


PROMPTS_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts.yaml"


@lru_cache
def load_prompts() -> dict[str, str]:
    with PROMPTS_PATH.open("r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file) or {}

    return prompts


def build_query_rewrite_prompt(question: str) -> str:
    prompts = load_prompts()
    template = prompts["query_rewrite_prompt"]

    return template.format(question=question)


def build_evidence_context(evidence_chunks: list[dict[str, Any]]) -> str:
    if not evidence_chunks:
        return "No evidence chunks were retrieved."

    evidence_lines: list[str] = []

    for index, chunk in enumerate(evidence_chunks, start=1):
        chunk_text = chunk.get("chunk_text") or ""
        document_name = chunk.get("document_name") or chunk.get("original_file_name") or "Unknown document"
        page_number = chunk.get("page_number")
        section_title = chunk.get("section_title")
        chunk_id = chunk.get("chunk_id") or chunk.get("id")

        metadata_parts = [
            f"Source {index}",
            f"Document: {document_name}",
        ]

        if page_number is not None:
            metadata_parts.append(f"Page: {page_number}")

        if section_title:
            metadata_parts.append(f"Section: {section_title}")

        if chunk_id:
            metadata_parts.append(f"Chunk ID: {chunk_id}")

        evidence_lines.append(
            "\n".join(
                [
                    " | ".join(metadata_parts),
                    f"Content: {chunk_text}",
                ]
            )
        )

    return "\n\n".join(evidence_lines)


def build_answer_prompt(
    question: str,
    evidence_chunks: list[dict[str, Any]],
) -> str:
    prompts = load_prompts()
    template = prompts["answer_generation_prompt"]

    evidence_context = build_evidence_context(evidence_chunks)

    return template.format(
        question=question,
        evidence_context=evidence_context,
    )


def get_refusal_message() -> str:
    prompts = load_prompts()

    return prompts.get(
        "refusal_message",
        "I could not find enough evidence in your uploaded documents to answer this question.",
    ).strip()


def strip_generated_source_metadata(answer: str) -> str:
    """Remove source metadata that belongs in the structured citations field."""
    cleaned_lines: list[str] = []

    for line in answer.strip().splitlines():
        stripped = line.strip()
        if re.match(r"^(sources?|citations?)\s*:", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r"^(sources?|citations?)\s*$", stripped, flags=re.IGNORECASE):
            continue
        if re.search(r"\bchunk\s+id\s*:", stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
