from pathlib import Path


def extract_text_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    return path.read_text(encoding="utf-8")