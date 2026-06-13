import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx


BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    os.getenv("BACKEND_API_URL", "http://backend:8000"),
).rstrip("/")


class APIClientError(Exception):
    """Raised when the backend API returns an error or cannot be reached."""


def _auth_headers(access_token: str | None = None) -> dict[str, str]:
    if not access_token:
        return {}

    return {"Authorization": f"Bearer {access_token}"}


def _format_error(response: httpx.Response) -> str:
    try:
        error_body = response.json()
    except ValueError:
        error_body = response.text

    if isinstance(error_body, dict):
        detail = error_body.get("detail")

        if isinstance(detail, list):
            return "; ".join(str(item) for item in detail)

        if detail:
            return str(detail)

    return str(error_body)


async def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    access_token: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any] | list[dict[str, Any]]:
    url = f"{BACKEND_BASE_URL}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                files=files,
                data=data,
                headers=_auth_headers(access_token),
            )
    except httpx.RequestError as exc:
        raise APIClientError(
            f"Could not connect to backend at {BACKEND_BASE_URL}. Error: {exc}"
        ) from exc

    if response.status_code >= 400:
        if response.status_code == 401:
            raise APIClientError("Authentication failed. Please login again.")
        raise APIClientError(_format_error(response))

    if not response.content:
        return {}

    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError(
            f"Backend returned non-JSON response: {response.text}"
        ) from exc


def _require_access_token(access_token: str | None) -> str:
    if not access_token:
        raise APIClientError(
            "Please login first using /login, /register, or /google."
        )

    return access_token


def _guess_content_type(file_name: str) -> str:
    content_type, _ = mimetypes.guess_type(file_name)

    if content_type:
        return content_type

    suffix = Path(file_name).suffix.lower()

    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    if suffix == ".pdf":
        return "application/pdf"

    if suffix == ".txt":
        return "text/plain"

    if suffix == ".md":
        return "text/markdown"

    if suffix == ".html":
        return "text/html"

    return "application/octet-stream"


async def register_user(
    email: str,
    password: str,
    full_name: str | None = None,
) -> dict[str, Any]:
    """
    Public endpoint.

    Registers a local user and should return app JWT from backend.
    """
    payload: dict[str, Any] = {
        "email": email,
        "password": password,
    }

    if full_name:
        payload["full_name"] = full_name

    response = await _request(
        "POST",
        "/auth/register",
        json=payload,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected register response from backend")

    return response


async def login_user(
    email: str,
    password: str,
) -> dict[str, Any]:
    """
    Public endpoint.

    Logs in local user and should return app JWT from backend.
    """
    response = await _request(
        "POST",
        "/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected login response from backend")

    return response


async def get_google_login_url() -> dict[str, Any]:
    """Return the browser URL that starts Google OAuth login."""
    try:
        response = await _request(
            "GET",
            "/auth/google/login",
        )
    except APIClientError as exc:
        if "Google OAuth is not configured" in str(exc):
            raise APIClientError(
                "Google login is not configured. Use /login or /register instead."
            ) from exc
        raise

    if not isinstance(response, dict) or not response.get("authorization_url"):
        raise APIClientError("Backend did not return a Google authorization URL")

    return response


async def get_health_status() -> dict[str, Any]:
    response = await _request("GET", "/health")

    if not isinstance(response, dict):
        raise APIClientError("Unexpected health response from backend")

    return response


async def get_current_user(
    access_token: str,
) -> dict[str, Any]:
    """
    Protected endpoint.

    Requires Authorization: Bearer <token>.
    """
    token = _require_access_token(access_token)

    response = await _request(
        "GET",
        "/auth/me",
        access_token=token,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected /auth/me response from backend")

    return response


async def upload_document(
    file_path: str,
    file_name: str | None = None,
    category: str = "general",
    access_token: str | None = None,
) -> dict[str, Any]:
    """
    Protected endpoint.

    Day 20 behavior:
    - Requires JWT.
    - Does not send user_id.
    - Backend uses current_user.id from token.
    """
    token = _require_access_token(access_token)

    path = Path(file_path)

    if not path.exists():
        raise APIClientError(f"File does not exist: {file_path}")

    upload_file_name = file_name or path.name
    content_type = _guess_content_type(upload_file_name)

    with path.open("rb") as file_obj:
        files = {
            "file": (
                upload_file_name,
                file_obj,
                content_type,
            )
        }

        data = {
            "category": category,
        }

        response = await _request(
            "POST",
            "/documents/upload",
            files=files,
            data=data,
            access_token=token,
            timeout=300.0,
        )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected upload response from backend")

    return response


async def list_documents(
    access_token: str | None = None,
    status: str | None = None,
    category: str | None = None,
    search: str | None = None,
    ready_only: bool = False,
) -> dict[str, Any]:
    """
    Protected endpoint.

    Day 20 behavior:
    - Requires JWT.
    - Does not send user_id.
    - Backend returns documents for current_user.id.
    """
    token = _require_access_token(access_token)
    params: dict[str, Any] = {}

    if status:
        params["status"] = status
    if category:
        params["category"] = category
    if search:
        params["search"] = search
    if ready_only:
        params["ready_only"] = True

    response = await _request(
        "GET",
        "/documents",
        params=params or None,
        access_token=token,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected documents response from backend")

    return response


async def process_document(
    document_id: str,
    force: bool = False,
    access_token: str | None = None,
) -> dict[str, Any]:
    """
    Protected endpoint.

    Day 20 behavior:
    - Requires JWT.
    - Does not send user_id.
    - Backend checks document ownership using current_user.id.
    """
    token = _require_access_token(access_token)

    response = await _request(
        "POST",
        f"/documents/{document_id}/process",
        params={
            "force": force,
        },
        access_token=token,
        timeout=300.0,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected process response from backend")

    return response


async def list_processing_jobs(
    document_id: str,
    access_token: str | None = None,
) -> dict[str, Any]:
    token = _require_access_token(access_token)
    response = await _request(
        "GET",
        f"/documents/{document_id}/processing-jobs",
        access_token=token,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected processing jobs response from backend")

    return response


async def get_processing_job(
    job_id: str,
    access_token: str | None = None,
) -> dict[str, Any]:
    token = _require_access_token(access_token)
    response = await _request(
        "GET",
        f"/documents/processing-jobs/{job_id}",
        access_token=token,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected processing job response from backend")

    return response


async def delete_document(
    document_id: str,
    access_token: str | None = None,
) -> dict[str, Any]:
    """
    Protected endpoint.

    Day 20 behavior:
    - Requires JWT.
    - Does not send user_id.
    - Backend checks document ownership using current_user.id.
    """
    token = _require_access_token(access_token)

    response = await _request(
        "DELETE",
        f"/documents/{document_id}",
        access_token=token,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected delete response from backend")

    return response


async def ask_question(
    question: str,
    session_id: str | None = None,
    top_k: int = 5,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    min_reranker_score: float | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    """
    Protected endpoint.

    Day 20 behavior:
    - Requires JWT.
    - Does not send user_id.
    - Backend uses current_user.id from token.
    """
    token = _require_access_token(access_token)

    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "hybrid_top_k": hybrid_top_k,
        "vector_top_k": vector_top_k,
        "bm25_top_k": bm25_top_k,
    }

    if session_id:
        payload["session_id"] = session_id

    if min_reranker_score is not None:
        payload["min_reranker_score"] = min_reranker_score

    response = await _request(
        "POST",
        "/chat/ask",
        json=payload,
        access_token=token,
        timeout=300.0,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected chat response from backend")

    return response
