import re
from collections import Counter
from typing import Any


STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "between",
    "but",
    "can",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "may",
    "not",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "under",
    "with",
    "you",
    "your",
}


def _word_count(text: str) -> int:
    """Return a lightweight token approximation based on word count."""
    return len(text.split())


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


def _normalize_heading(text: str) -> str:
    """Clean markdown/list heading markers while preserving the heading text."""
    text = text.strip()
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"^\d+(\.\d+)*[\.)]\s*", "", text)
    return text.strip(" :-")


def _is_page_marker(line: str) -> bool:
    """Return whether a line is an extraction page boundary marker."""
    return bool(re.fullmatch(r"-{3}\s*Page\s+\d+\s*-{3}", line.strip(), re.IGNORECASE))


def _page_number_from_marker(line: str) -> int | None:
    match = re.search(r"Page\s+(\d+)", line, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _is_table_line(line: str) -> bool:
    """Detect simple markdown, DOCX, or extracted table rows."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.count("|") >= 1:
        return True
    if re.search(r"\S\s{2,}\S", stripped):
        return True
    return False


def _is_markdown_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line.strip()))


def _is_heading_line(line: str) -> bool:
    """Detect headings from markdown, numbering, title case, and short all-caps lines."""
    stripped = line.strip()
    if not stripped or _is_table_line(stripped):
        return False
    if stripped.startswith("#"):
        return True
    if re.match(r"^\d+(\.\d+)*[\.)]\s+\S", stripped):
        return len(stripped.split()) <= 12
    if stripped.endswith(":") and len(stripped.split()) <= 12:
        return True
    if stripped.isupper() and len(stripped.split()) <= 10 and any(ch.isalpha() for ch in stripped):
        return True
    if stripped.istitle() and len(stripped.split()) <= 8 and not stripped.endswith("."):
        return True
    return False


def parse_document_structure(text: str) -> list[dict[str, Any]]:
    """
    Parse extracted text into coarse document blocks.

    The parser is intentionally dependency-light because extraction already
    normalized each supported file type into text. It preserves tables, tracks
    page markers, and attaches paragraphs to the nearest detected heading.
    """
    if not text or not text.strip():
        return []

    blocks: list[dict[str, Any]] = []
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    current_section: str | None = None
    current_page: int | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = " ".join(paragraph_lines).strip()
        if paragraph:
            blocks.append(
                {
                    "type": "paragraph",
                    "text": paragraph,
                    "section_title": current_section,
                    "page_number": current_page,
                }
            )
        paragraph_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        table_text = "\n".join(line for line in table_lines if not _is_markdown_table_separator(line))
        if table_text.strip():
            blocks.append(
                {
                    "type": "table",
                    "text": table_text.strip(),
                    "section_title": current_section,
                    "page_number": current_page,
                }
            )
        table_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            flush_table()
            flush_paragraph()
            continue

        if _is_page_marker(stripped):
            flush_table()
            flush_paragraph()
            current_page = _page_number_from_marker(stripped)
            continue

        if _is_table_line(stripped) or (table_lines and _is_markdown_table_separator(stripped)):
            flush_paragraph()
            table_lines.append(stripped)
            continue

        flush_table()

        if _is_heading_line(stripped):
            flush_paragraph()
            current_section = _normalize_heading(stripped)
            blocks.append(
                {
                    "type": "heading",
                    "text": current_section,
                    "section_title": current_section,
                    "page_number": current_page,
                }
            )
            continue

        paragraph_lines.append(stripped)

    flush_table()
    flush_paragraph()

    return blocks


def _keywords_for_text(text: str, limit: int = 8) -> list[str]:
    words = [
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if word not in STOPWORDS
    ]
    return [word for word, _ in Counter(words).most_common(limit)]


def _summary_for_text(text: str, max_words: int = 42) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = next((sentence.strip() for sentence in sentences if sentence.strip()), text.strip())
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]).rstrip(".,;:") + "."
    return summary


def _questions_for_chunk(
    summary: str,
    keywords: list[str],
    section_title: str | None,
    limit: int = 3,
) -> list[str]:
    subject = section_title or (keywords[0].replace("-", " ") if keywords else "this policy")
    questions = [
        f"What does the document say about {subject}?",
        f"What are the key details for {subject}?",
    ]
    if keywords:
        questions.append(f"How does {keywords[0].replace('-', ' ')} apply here?")
    elif summary:
        questions.append("What information is covered in this section?")
    return questions[:limit]


def _metadata_for_chunk(
    chunk_text: str,
    section_title: str | None,
    structure_types: list[str],
) -> dict[str, Any]:
    summary = _summary_for_text(chunk_text)
    keywords = _keywords_for_text(chunk_text)
    hypothetical_questions = _questions_for_chunk(summary, keywords, section_title)
    search_text_parts = [
        chunk_text,
        f"Summary: {summary}" if summary else "",
        f"Keywords: {', '.join(keywords)}" if keywords else "",
        "Questions: " + " ".join(hypothetical_questions) if hypothetical_questions else "",
    ]
    return {
        "summary": summary,
        "keywords": keywords,
        "hypothetical_questions": hypothetical_questions,
        "structure_types": sorted(set(structure_types)),
        "search_text": "\n\n".join(part for part in search_text_parts if part),
    }


def _build_chunk(
    chunk_text: str,
    chunk_index: int,
    section_title: str | None,
    page_number: int | None,
    structure_types: list[str],
) -> dict[str, Any]:
    metadata = _metadata_for_chunk(chunk_text, section_title, structure_types)
    return {
        "chunk_text": chunk_text,
        "chunk_index": chunk_index,
        "token_count": _word_count(chunk_text),
        "section_title": section_title or detect_section_title(chunk_text.split()),
        "page_number": page_number,
        "summary": metadata["summary"],
        "keywords": metadata["keywords"],
        "hypothetical_questions": metadata["hypothetical_questions"],
        "structure_types": metadata["structure_types"],
        "search_text": metadata["search_text"],
    }


def _split_large_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    section_title: str | None,
    page_number: int | None,
    structure_type: str,
    start_index: int,
) -> list[dict[str, Any]]:
    words = text.split()
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = start_index

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(
            _build_chunk(
                chunk_text=" ".join(chunk_words),
                chunk_index=chunk_index,
                section_title=section_title,
                page_number=page_number,
                structure_types=[structure_type],
            )
        )
        chunk_index += 1

        if end >= len(words):
            break

        start = end - chunk_overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[dict[str, Any]]:
    """
    Split extracted document text into structure-aware chunks.

    The chunker preserves tables, keeps headings with nearby content, honors
    page markers produced by the PDF extractor, and enriches each chunk with
    local summaries, keywords, and hypothetical retrieval questions.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text or not text.strip():
        return []

    blocks = parse_document_structure(text)

    if not blocks:
        return []

    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_types: list[str] = []
    current_section: str | None = None
    current_page: int | None = None
    chunk_index = 0

    def flush_current() -> None:
        nonlocal current_parts, current_types, current_section, current_page, chunk_index
        if not current_parts:
            return
        chunk_text_value = "\n\n".join(current_parts).strip()
        if chunk_text_value:
            chunks.append(
                _build_chunk(
                    chunk_text=chunk_text_value,
                    chunk_index=chunk_index,
                    section_title=current_section,
                    page_number=current_page,
                    structure_types=current_types,
                )
            )
            chunk_index += 1
        current_parts = []
        current_types = []
        current_section = None
        current_page = None

    for block in blocks:
        block_text = str(block["text"]).strip()
        block_type = str(block["type"])
        block_size = _word_count(block_text)

        if not block_text:
            continue

        if block_size > chunk_size and block_type != "table":
            flush_current()
            split_chunks = _split_large_text(
                text=block_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                section_title=block.get("section_title"),
                page_number=block.get("page_number"),
                structure_type=block_type,
                start_index=chunk_index,
            )
            chunks.extend(split_chunks)
            chunk_index += len(split_chunks)
            continue

        would_exceed = current_parts and _word_count("\n\n".join(current_parts + [block_text])) > chunk_size
        section_changed = (
            current_parts
            and block.get("section_title")
            and current_section
            and block.get("section_title") != current_section
        )

        if would_exceed or section_changed:
            flush_current()

        current_parts.append(block_text)
        current_types.append(block_type)
        current_section = current_section or block.get("section_title")
        current_page = current_page if current_page is not None else block.get("page_number")

    flush_current()

    return chunks
