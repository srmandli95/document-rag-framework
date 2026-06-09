import os
from typing import Any

import httpx


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://backend:8000")


class APIClientError(Exception):
    """Raised when the backend API returns an error or cannot be reached."""


def _format_error(response: httpx.Response) -> str:
    try:
        error_body = response.json()
    except ValueError:
        return response.text or f"Request failed with status {response.status_code}"

    if isinstance(error_body, dict):
        detail = error_body.get("detail")
        if isinstance(detail, str):
            return detail

        if isinstance(detail, list):
            return "; ".join(str(item) for item in detail)

        return str(error_body)

    return str(error_body)


async def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
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


async def upload_document(
    user_id: str,
    file_path: str,
    file_name: str,
    category: str = "general",
) -> dict[str, Any]:
    with open(file_path, "rb") as file:
        files = {
            "file": (
                file_name,
                file,
                "application/octet-stream",
            )
        }

        params = {
            "user_id": user_id,
            "category": category,
        }

        response = await _request(
            "POST",
            "/documents/upload",
            params=params,
            files=files,
            timeout=120.0,
        )

    return dict(response)


async def list_documents(user_id: str) -> dict[str, Any]:
    response = await _request(
        "GET",
        "/documents",
        params={"user_id": user_id},
        timeout=60.0,
    )

    return dict(response)


async def delete_document(user_id: str, document_id: str) -> dict[str, Any]:
    response = await _request(
        "DELETE",
        f"/documents/{document_id}",
        params={"user_id": user_id},
        timeout=60.0,
    )

    return dict(response)


async def process_document(
    user_id: str,
    document_id: str,
    force: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "user_id": user_id,
    }

    if force:
        params["force"] = "true"

    response = await _request(
        "POST",
        f"/documents/{document_id}/process",
        params=params,
        timeout=300.0,
    )

    return dict(response)


async def ask_question(
    user_id: str,
    question: str,
    session_id: str | None = None,
    top_k: int = 5,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    min_reranker_score: float | None = None,
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
        timeout=300.0,
    )

    return dict(response)