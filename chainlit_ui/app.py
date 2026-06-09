from typing import Any

import chainlit as cl

from services.api_client import (
    BackendAPIError,
    ask_question,
    check_backend_health,
    list_documents,
    process_document,
)
from services.file_upload_client import FileUploadError, upload_document


TEMP_LOCAL_USER_ID = "local-user-123"
DEFAULT_UPLOAD_CATEGORY = "general"


def _get_documents_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Supports multiple possible backend response shapes:
    - {"documents": [...]}
    - {"items": [...]}
    - {"data": [...]}
    """
    for key in ("documents", "items", "data"):
        value = response.get(key)
        if isinstance(value, list):
            return value

    if isinstance(response, list):
        return response

    return []


def _get_document_id(document: dict[str, Any]) -> str:
    return str(
        document.get("id")
        or document.get("document_id")
        or document.get("doc_id")
        or "N/A"
    )


def _get_document_name(document: dict[str, Any]) -> str:
    return str(
        document.get("original_file_name")
        or document.get("file_name")
        or document.get("name")
        or document.get("stored_file_name")
        or "N/A"
    )


def _format_documents(response: dict[str, Any]) -> str:
    documents = _get_documents_from_response(response)

    if not documents:
        return "No documents found for this user."

    lines = ["## Uploaded Documents"]

    for index, document in enumerate(documents, start=1):
        document_id = _get_document_id(document)
        file_name = _get_document_name(document)
        category = document.get("category") or "N/A"
        status = document.get("status") or "N/A"
        created_at = document.get("created_at") or "N/A"

        lines.append(
            "\n".join(
                [
                    f"### {index}. {file_name}",
                    f"- Document ID: `{document_id}`",
                    f"- Category: `{category}`",
                    f"- Status: `{status}`",
                    f"- Created: `{created_at}`",
                ]
            )
        )

    return "\n\n".join(lines)


def _extract_answer(response: dict[str, Any]) -> str:
    if response.get("answer"):
        return str(response["answer"])

    final_response = response.get("final_response")
    if isinstance(final_response, dict):
        if final_response.get("answer"):
            return str(final_response["answer"])
        if final_response.get("final_answer"):
            return str(final_response["final_answer"])

    if response.get("final_answer"):
        return str(response["final_answer"])

    return "No answer was returned by the backend."


def _extract_citations(response: dict[str, Any]) -> list[dict[str, Any]]:
    citations = response.get("citations")

    if isinstance(citations, list):
        return [citation for citation in citations if isinstance(citation, dict)]

    final_response = response.get("final_response")
    if isinstance(final_response, dict):
        final_citations = final_response.get("citations")
        if isinstance(final_citations, list):
            return [
                citation
                for citation in final_citations
                if isinstance(citation, dict)
            ]

    return []


def _extract_session_id(response: dict[str, Any]) -> str | None:
    value = (
        response.get("session_id")
        or response.get("chat_session_id")
        or response.get("conversation_id")
    )

    if value:
        return str(value)

    final_response = response.get("final_response")
    if isinstance(final_response, dict):
        nested_value = (
            final_response.get("session_id")
            or final_response.get("chat_session_id")
            or final_response.get("conversation_id")
        )
        if nested_value:
            return str(nested_value)

    return None


def _extract_status(response: dict[str, Any]) -> str | None:
    value = response.get("status")

    if value:
        return str(value)

    final_response = response.get("final_response")
    if isinstance(final_response, dict) and final_response.get("status"):
        return str(final_response["status"])

    return None


def _extract_validation_status(response: dict[str, Any]) -> str | None:
    value = response.get("validation_status")

    if value:
        return str(value)

    final_response = response.get("final_response")
    if isinstance(final_response, dict) and final_response.get("validation_status"):
        return str(final_response["validation_status"])

    return None


def _is_refusal_or_unsupported(response: dict[str, Any]) -> bool:
    status = (_extract_status(response) or "").lower()
    validation_status = (_extract_validation_status(response) or "").lower()

    refused_statuses = {
        "refused",
        "insufficient_evidence",
        "no_evidence",
        "failed",
        "error",
    }

    unsupported_statuses = {
        "unsupported",
        "failed",
        "invalid",
    }

    return status in refused_statuses or validation_status in unsupported_statuses


def _format_score(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _format_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""

    lines = ["## Sources"]

    for index, citation in enumerate(citations, start=1):
        document_name = (
            citation.get("document_name")
            or citation.get("original_file_name")
            or citation.get("file_name")
            or "N/A"
        )
        category = citation.get("category") or "N/A"
        page_number = citation.get("page_number")
        section_title = citation.get("section_title") or "N/A"
        chunk_id = citation.get("chunk_id") or "N/A"
        reranker_score = _format_score(citation.get("reranker_score"))
        hybrid_score = _format_score(citation.get("hybrid_score"))

        page_display = page_number if page_number is not None else "N/A"

        lines.append(
            "\n".join(
                [
                    f"### {index}. {document_name}",
                    f"- Category: `{category}`",
                    f"- Page: `{page_display}`",
                    f"- Section: `{section_title}`",
                    f"- Chunk ID: `{chunk_id}`",
                    f"- Reranker Score: `{reranker_score}`",
                    f"- Hybrid Score: `{hybrid_score}`",
                ]
            )
        )

    return "\n\n".join(lines)


def _format_answer_response(response: dict[str, Any]) -> str:
    answer = _extract_answer(response)
    status = _extract_status(response)
    validation_status = _extract_validation_status(response)
    citations = _extract_citations(response)

    sections = [answer]

    metadata_lines = []

    if status:
        metadata_lines.append(f"- Status: `{status}`")

    if validation_status:
        metadata_lines.append(f"- Validation Status: `{validation_status}`")

    if metadata_lines:
        sections.append("## Response Metadata\n" + "\n".join(metadata_lines))

    if not _is_refusal_or_unsupported(response):
        citation_block = _format_citations(citations)
        if citation_block:
            sections.append(citation_block)

    return "\n\n".join(sections)


def _help_message() -> str:
    return """
## PersonalPolicyRagAssistant

Available actions:

- Upload a document using the Chainlit attachment button.
- Ask a question about your uploaded and processed documents.
- Type `/documents` to list uploaded documents.
- Type `/process {document_id}` to run extract → chunk → embed for a document.
- Type `/new` to start a new backend chat session on your next question.
- Type `/help` to show this help message.

Temporary Day 16 user:

`local-user-123`
""".strip()


async def _handle_file_uploads(message: cl.Message) -> bool:
    """
    Handles files attached to a Chainlit message.

    Returns True if files were handled.
    """
    files = []

    for element in message.elements or []:
        file_path = getattr(element, "path", None)
        file_name = getattr(element, "name", None)

        if file_path and file_name:
            files.append((file_path, file_name))

    if not files:
        return False

    user_id = cl.user_session.get("user_id") or TEMP_LOCAL_USER_ID

    for file_path, file_name in files:
        try:
            result = await upload_document(
                user_id=user_id,
                file_path=file_path,
                file_name=file_name,
                category=DEFAULT_UPLOAD_CATEGORY,
            )

            document_id = (
                result.get("id")
                or result.get("document_id")
                or result.get("doc_id")
                or "N/A"
            )
            original_file_name = (
                result.get("original_file_name")
                or result.get("file_name")
                or file_name
            )
            category = result.get("category") or DEFAULT_UPLOAD_CATEGORY
            status = result.get("status") or "uploaded"

            await cl.Message(
                content="\n".join(
                    [
                        "Document uploaded successfully.",
                        "",
                        f"- Document ID: `{document_id}`",
                        f"- File Name: `{original_file_name}`",
                        f"- Category: `{category}`",
                        f"- Status: `{status}`",
                        "",
                        "For Day 16, upload does not automatically process the document.",
                        f"Use `/process {document_id}` to run extract → chunk → embed.",
                    ]
                )
            ).send()

        except FileUploadError as exc:
            await cl.Message(
                content=f"Document upload failed: {exc}"
            ).send()

    return True


async def _handle_documents_command() -> None:
    user_id = cl.user_session.get("user_id") or TEMP_LOCAL_USER_ID

    try:
        response = await list_documents(user_id=user_id)
        await cl.Message(content=_format_documents(response)).send()

    except BackendAPIError as exc:
        await cl.Message(content=str(exc)).send()


async def _handle_process_command(content: str) -> None:
    user_id = cl.user_session.get("user_id") or TEMP_LOCAL_USER_ID

    parts = content.split(maxsplit=1)

    if len(parts) != 2 or not parts[1].strip():
        await cl.Message(
            content="Please provide a document ID.\n\nExample:\n`/process your-document-id`"
        ).send()
        return

    document_id = parts[1].strip()

    progress_message = cl.Message(
        content=f"Processing document `{document_id}`..."
    )
    await progress_message.send()

    try:
        results = await process_document(
            user_id=user_id,
            document_id=document_id,
        )

        lines = [
            f"Document processing completed for `{document_id}`.",
            "",
            "## Processing Steps",
        ]

        for result in results:
            step = result.get("step", "unknown")
            status = result.get("status", "unknown")
            lines.append(f"- `{step}`: `{status}`")

        progress_message.content = "\n".join(lines)
        await progress_message.update()

    except BackendAPIError as exc:
        progress_message.content = f"Document processing failed: {exc}"
        await progress_message.update()


async def _handle_new_command() -> None:
    cl.user_session.set("session_id", None)

    await cl.Message(
        content="Started a new UI chat session. The next question will create or use a fresh backend chat session."
    ).send()


async def _handle_question(content: str) -> None:
    user_id = cl.user_session.get("user_id") or TEMP_LOCAL_USER_ID
    session_id = cl.user_session.get("session_id")

    thinking_message = cl.Message(content="Asking the backend...")
    await thinking_message.send()

    try:
        response = await ask_question(
            user_id=user_id,
            question=content,
            session_id=session_id,
            top_k=5,
        )

        returned_session_id = _extract_session_id(response)

        if returned_session_id:
            cl.user_session.set("session_id", returned_session_id)

        thinking_message.content = _format_answer_response(response)
        await thinking_message.update()

    except BackendAPIError as exc:
        thinking_message.content = str(exc)
        await thinking_message.update()


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("user_id", TEMP_LOCAL_USER_ID)
    cl.user_session.set("session_id", None)

    try:
        health = await check_backend_health()
        backend_status = health.get("status") or "reachable"

        await cl.Message(
            content="\n".join(
                [
                    "# PersonalPolicyRagAssistant",
                    "",
                    "Backend connection successful.",
                    f"- Backend status: `{backend_status}`",
                    f"- Temporary user: `{TEMP_LOCAL_USER_ID}`",
                    "",
                    _help_message(),
                ]
            )
        ).send()

    except BackendAPIError:
        await cl.Message(
            content="\n".join(
                [
                    "# PersonalPolicyRagAssistant",
                    "",
                    "Backend is not reachable. Please make sure FastAPI is running.",
                    "",
                    "You can still open the UI, but document upload, document listing, and question answering will not work until the backend is available.",
                    "",
                    _help_message(),
                ]
            )
        ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    uploaded = await _handle_file_uploads(message)

    if uploaded:
        return

    content = (message.content or "").strip()

    if not content:
        await cl.Message(
            content="Please type a question, upload a document, or use `/help`."
        ).send()
        return

    lowered = content.lower()

    if lowered == "/help":
        await cl.Message(content=_help_message()).send()
        return

    if lowered == "/documents":
        await _handle_documents_command()
        return

    if lowered == "/new":
        await _handle_new_command()
        return

    if lowered.startswith("/process"):
        await _handle_process_command(content)
        return

    await _handle_question(content)