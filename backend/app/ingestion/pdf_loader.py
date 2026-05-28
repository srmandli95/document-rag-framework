from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(str(path))
    extracted_pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(
                f"--- Page {page_number} ---\n{page_text.strip()}"
            )

    return "\n\n".join(extracted_pages)