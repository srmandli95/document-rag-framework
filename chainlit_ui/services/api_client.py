import os
from pathlib import Path
from typing import Any

import httpx


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://backend:8000")


class APIClientError(Exception):
    """Raised when the backend API returns an error or cannot be reached."""


def _auth_headers(access_token: str | None = None) -> dict[str, str]:
    if access_token:
        return {"Authorization": f"Bearer {access_token}"}

    return {}


def _format_error(response: httpx.Response) -> str:
    try:
        error_body = response.json()
    except ValueError:
        error_body = response.text

    if isinstance(error_body, dict):
        detail = error_body.get("detail")
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
        raise APIClientError(_format_error(response))

    if not response.content:
        return {}

    return response.json()


async def register_user(
    email: str,
    password: str,
    full_name: str | None = None,
) -> dict[str, Any]:
    response = await _request(
        "POST",
        "/auth/register",
        json={
            "email": email,
            "password": password,
            "full_name": full_name,
        },
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected register response from backend")

    return response


async def login_user(
    email: str,
    password: str,
) -> dict[str, Any]:
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


async def get_current_user(
    access_token: str,
) -> dict[str, Any]:
    response = await _request(
        "GET",
        "/auth/me",
        access_token=access_token,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected /auth/me response from backend")

    return response


async def upload_document(
    file_path: str,
    user_id: str,
    file_name: str | None = None,
    category: str = "general",
    access_token: str | None = None,
) -> dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise APIClientError(f"File does not exist: {file_path}")

    upload_file_name = file_name or path.name

    with path.open("rb") as file_obj:
        files = {
            "file": (
                upload_file_name,
                file_obj,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        }

        data = {
            "user_id": user_id,
            "category": category,
        }

        response = await _request(
            "POST",
            "/documents/upload",
            files=files,
            data=data,
            access_token=access_token,
            timeout=300.0,
        )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected upload response from backend")

    return response


async def list_documents(
    user_id: str,
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    response = await _request(
        "GET",
        "/documents",
        params={"user_id": user_id},
        access_token=access_token,
    )

    if not isinstance(response, list):
        raise APIClientError("Unexpected documents response from backend")

    return response


async def process_document(
    document_id: str,
    user_id: str,
    force: bool = False,
    access_token: str | None = None,
) -> dict[str, Any]:
    response = await _request(
        "POST",
        f"/documents/{document_id}/process",
        params={
            "user_id": user_id,
            "force": force,
        },
        access_token=access_token,
        timeout=300.0,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected process response from backend")

    return response


async def delete_document(
    document_id: str,
    user_id: str,
    access_token: str | None = None,
) -> dict[str, Any]:
    response = await _request(
        "DELETE",
        f"/documents/{document_id}",
        params={"user_id": user_id},
        access_token=access_token,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected delete response from backend")

    return response


async def ask_question(
    user_id: str,
    question: str,
    session_id: str | None = None,
    top_k: int = 5,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    min_reranker_score: float | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": user_id,
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
        access_token=access_token,
        timeout=300.0,
    )

    if not isinstance(response, dict):
        raise APIClientError("Unexpected chat response from backend")

    return response