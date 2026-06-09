import os
from typing import Any

import httpx


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://backend:8000")

async def upload_document(
    user_id: str,
    file_path: str,
    file_name: str,
    category: str,
) -> dict[str, Any]:
    """
    Upload a document to the FastAPI backend.

    Backend endpoint:
        POST /documents/upload
    """
    url = f"{BACKEND_BASE_URL}/documents/upload"

    with open(file_path, "rb") as file_obj:
        files = {
            "file": (file_name, file_obj),
        }
        data = {
            "user_id": user_id,
            "category": category,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                data=data,
                files=files,
            )

    if response.status_code >= 400:
        raise RuntimeError(_format_error(response))

    return response.json()


async def list_documents(
    user_id: str,
) -> list[dict[str, Any]]:
    """
    List documents for a user.

    Backend endpoint:
        GET /documents?user_id=...
    """
    url = f"{BACKEND_BASE_URL}/documents"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(
            url,
            params={"user_id": user_id},
        )

    if response.status_code >= 400:
        raise RuntimeError(_format_error(response))

    payload = response.json()

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "documents" in payload:
            return payload["documents"]
        if "items" in payload:
            return payload["items"]

    return []


async def process_document(
    user_id: str,
    document_id: str,
) -> dict[str, Any]:
    """
    Process an uploaded document through:

        extract -> chunk -> embed

    Backend endpoint:
        POST /documents/{document_id}/process?user_id=...
    """
    url = f"{BACKEND_BASE_URL}/documents/{document_id}/process"

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            url,
            params={"user_id": user_id},
        )

    if response.status_code >= 400:
        raise RuntimeError(_format_error(response))

    return response.json()


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
    """
    Ask a question using the backend RAG workflow.

    Backend endpoint:
        POST /chat/ask
    """
    url = f"{BACKEND_BASE_URL}/chat/ask"

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

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            url,
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(_format_error(response))

    return response.json()


def _format_error(response: httpx.Response) -> str:
    """
    Convert backend error responses into readable Chainlit errors.
    """
    try:
        payload = response.json()
    except Exception:
        return f"Backend error {response.status_code}: {response.text}"

    detail = payload.get("detail")

    if isinstance(detail, str):
        return f"Backend error {response.status_code}: {detail}"

    if detail is not None:
        return f"Backend error {response.status_code}: {detail}"

    return f"Backend error {response.status_code}: {payload}"