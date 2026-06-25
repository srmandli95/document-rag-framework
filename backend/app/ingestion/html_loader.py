from pathlib import Path

from bs4 import BeautifulSoup


def extract_html_text(file_path: str) -> str:
    """Extract readable text from an HTML file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {file_path}")

    html_content = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator="\n")