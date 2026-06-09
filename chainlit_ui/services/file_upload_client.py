import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BACKEND_API_URL = "http://localhost:8000"

BACKEND_API_URL = os.getenv("BACKEND_API_URL", DEFAULT_BACKEND_API_URL).rstrip("/")


class FileUploadError(Exception):
    """Friendly exception for document upload failures."""


SUPPORTED_CONTENT_TYPES_BY_EXTENSION = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
}


def _get_content_type(file_name: str) -> str:
    extension = Path(file_name).suffix.lower()

    if extension in SUPPORTED_CONTENT_TYPES_BY_EXTENSION:
        return SUPPORTED_CONTENT_TYPES_BY_EXTENSION[extension]

    guessed_content_type, _ = mimetypes.guess_type(file_name)

    if guessed_content_type:
        return guessed_content_type

    return "application/octet-stream"


def _friendly_error_from_response(response: httpx.Response) -> str:
    try:
        data = response.json()
    except Exception:
        return response.text or f"Backend returned HTTP {response.status_code}"

    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            return "; ".join(str(item) for item in detail)

        error = data.get("error")
        if isinstance(error, str):
            return error

        message = data.get("message")
        if isinstance(message, str):
            return message

    return f"Backend returned HTTP {response.status_code}"


async def upload_document(
    user_id: str,
    file_path: str,
    file_name: str,
    category: str,
) -> dict[str, Any]:
    """
    Upload a document to the FastAPI backend.

    Calls:
    POST /documents/upload

    Multipart form fields:
    - user_id
    - category
    - file
    """
    path = Path(file_path)

    if not path.exists():
        raise FileUploadError(f"File does not exist: {file_path}")

    content_type = _get_content_type(file_name)

    url = f"{BACKEND_API_URL}/documents/upload"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with path.open("rb") as file_handle:
                files = {
                    "file": (
                        file_name,
                        file_handle,
                        content_type,
                    )
                }

                data = {
                    "user_id": user_id,
                    "category": category,
                }

                response = await client.post(
                    url,
                    data=data,
                    files=files,
                )

        response.raise_for_status()
        return response.json()

    except httpx.ConnectError as exc:
        raise FileUploadError(
            "Backend is not reachable. Please make sure FastAPI is running."
        ) from exc

    except httpx.TimeoutException as exc:
        raise FileUploadError(
            "Document upload timed out. Please try again."
        ) from exc

    except httpx.HTTPStatusError as exc:
        message = _friendly_error_from_response(exc.response)
        raise FileUploadError(message) from exc

    except httpx.RequestError as exc:
        raise FileUploadError(
            f"Could not upload document: {exc}"
        ) from exc

    except ValueError as exc:
        raise FileUploadError(
            "Backend returned an invalid JSON response after upload."
        ) from exc