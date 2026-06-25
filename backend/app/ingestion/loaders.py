from pathlib import Path

from app.ingestion.docx_loader import extract_docx_text
from app.ingestion.html_loader import extract_html_text
from app.ingestion.pdf_loader import extract_pdf_text
from app.ingestion.text_loader import extract_text_file


def extract_text_from_file(
    file_path: str,
    content_type: str,
    file_extension: str,
) -> str:
    """Dispatch text extraction based on document content type or extension."""
    extension = file_extension.lower().strip()
    content_type = content_type.lower().strip()

    if not extension:
        extension = Path(file_path).suffix.lower()

    if content_type == "application/pdf" or extension == ".pdf":
        return extract_pdf_text(file_path)

    if (
        content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or extension == ".docx"
    ):
        return extract_docx_text(file_path)

    if content_type in {"text/plain", "text/markdown"} or extension in {
        ".txt",
        ".md",
    }:
        return extract_text_file(file_path)

    if content_type == "text/html" or extension == ".html":
        return extract_html_text(file_path)

    raise ValueError(
        f"Unsupported file type for extraction. "
        f"content_type={content_type}, extension={extension}"
    )