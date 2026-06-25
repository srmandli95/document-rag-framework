import re


def clean_text(text: str) -> str:
    """Normalize extracted document text before chunking."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Keep readable section breaks, but remove excessive blank lines.
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()