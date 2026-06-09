import os
from typing import Any

import httpx


DEFAULT_BACKEND_API_URL = "http://localhost:8000"

BACKEND_API_URL = os.getenv("BACKEND_API_URL", DEFAULT_BACKEND_API_URL).rstrip("/")


class BackendAPIError(Exception):
    """Friendly exception for backend API failures."""


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


async def _request_json(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any] | list[Any]:
    url = f"{BACKEND_API_URL}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                params=params,
                json=json,
            )

        response.raise_for_status()

        if not response.content:
            return {}

        return response.json()

    except httpx.ConnectError as exc:
        raise BackendAPIError(
            "Backend is not reachable. Please make sure FastAPI is running."
        ) from exc

    except httpx.TimeoutException as exc:
        raise BackendAPIError(
            "Backend request timed out. Please try again."
        ) from exc

    except httpx.HTTPStatusError as exc:
        message = _friendly_error_from_response(exc.response)
        raise BackendAPIError(message) from exc

    except httpx.RequestError as exc:
        raise BackendAPIError(
            f"Could not complete backend request: {exc}"
        ) from exc

    except ValueError as exc:
        raise BackendAPIError(
            "Backend returned an invalid JSON response."
        ) from exc


async def check_backend_health() -> dict[str, Any]:
    """
    Check FastAPI backend health.

    Calls:
    GET /health
    """
    result = await _request_json("GET", "/health", timeout=10.0)

    if isinstance(result, dict):
        return result

    return {"status": "unknown", "response": result}


async def list_documents(user_id: str) -> dict[str, Any]:
    """
    List documents for the temporary local user.

    Calls:
    GET /documents?user_id=...
    """
    result = await _request_json(
        "GET",
        "/documents",
        params={"user_id": user_id},
        timeout=30.0,
    )

    if isinstance(result, dict):
        return result

    return {"documents": result}


async def ask_question(
    user_id: str,
    question: str,
    session_id: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Ask a RAG question through the graph-backed backend endpoint.

    Calls:
    POST /chat/ask
    """
    payload: dict[str, Any] = {
        "user_id": user_id,
        "question": question,
        "top_k": top_k,
    }

    if session_id:
        payload["session_id"] = session_id

    result = await _request_json(
        "POST",
        "/chat/ask",
        json=payload,
        timeout=120.0,
    )

    if isinstance(result, dict):
        return result

    return {"answer": str(result)}


async def process_document(
    user_id: str,
    document_id: str,
) -> list[dict[str, Any]]:
    """
    Temporary Day 16 development helper.

    Runs:
    POST /documents/{document_id}/extract
    POST /documents/{document_id}/chunk
    POST /documents/{document_id}/embed

    This should later move into a proper Day 17 processing pipeline.
    """
    steps = [
        ("extract", f"/documents/{document_id}/extract"),
        ("chunk", f"/documents/{document_id}/chunk"),
        ("embed", f"/documents/{document_id}/embed"),
    ]

    results: list[dict[str, Any]] = []

    for step_name, path in steps:
        result = await _request_json(
            "POST",
            path,
            params={"user_id": user_id},
            timeout=180.0,
        )

        if isinstance(result, dict):
            results.append(
                {
                    "step": step_name,
                    "status": "completed",
                    "response": result,
                }
            )
        else:
            results.append(
                {
                    "step": step_name,
                    "status": "completed",
                    "response": {"result": result},
                }
            )

    return results