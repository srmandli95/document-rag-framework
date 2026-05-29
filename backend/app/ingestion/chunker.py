from typing import Any


def detect_section_title(words: list[str]) -> str | None:
    """
    Simple Day 5 section detection.

    For now, this is intentionally basic.
    Later, we can improve this using headings, page metadata, or document structure.
    """
    if not words:
        return None

    first_line = " ".join(words[:8]).strip()

    if first_line.endswith(":"):
        return first_line

    if first_line.istitle() and len(first_line.split()) <= 8:
        return first_line

    return None


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[dict[str, Any]]:
    """
    Split extracted document text into word-based chunks.

    Day 5 rules:
    - Simple word-based splitting
    - Approximate token_count using word count
    - Overlap between chunks
    - chunk_index starts at 0
    - page_number is None for now
    - section_title is best-effort
    """

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text or not text.strip():
        return []

    words = text.split()

    if len(words) <= chunk_size:
        return [
            {
                "chunk_text": " ".join(words),
                "chunk_index": 0,
                "token_count": len(words),
                "section_title": detect_section_title(words),
                "page_number": None,
            }
        ]

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append(
            {
                "chunk_text": " ".join(chunk_words),
                "chunk_index": chunk_index,
                "token_count": len(chunk_words),
                "section_title": detect_section_title(chunk_words),
                "page_number": None,
            }
        )

        chunk_index += 1

        if end >= len(words):
            break

        start = end - chunk_overlap

    return chunks