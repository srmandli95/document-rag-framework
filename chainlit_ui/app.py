import os
from typing import Any

import chainlit as cl

from services.api_client import (
    ask_question,
    list_documents,
    process_document,
    upload_document,
)


USER_ID = os.getenv("CHAINLIT_USER_ID", "local-user-123")
DEFAULT_CATEGORY = os.getenv("DEFAULT_DOCUMENT_CATEGORY", "general")


@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Initialize Chainlit chat session state.
    """
    cl.user_session.set("user_id", USER_ID)
    cl.user_session.set("session_id", None)

    await cl.Message(
        content=(
            "# PersonalPolicyRagAssistant\n\n"
            "Upload a document, process it, and then ask questions from it.\n\n"
            "Commands:\n\n"
            "```text\n"
            "/documents\n"
            "/process <document_id>\n"
            "```\n\n"
            "After uploading a document, run `/process <document_id>` to prepare it for search."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    """
    Main Chainlit message handler.

    Supported:
    - file uploads
    - /documents
    - /process <document_id>
    - normal RAG questions
    """
    user_id = cl.user_session.get("user_id") or USER_ID

    if await handle_file_uploads(message, user_id):
        return

    if await handle_documents_command(message, user_id):
        return

    if await handle_process_command(message, user_id):
        return

    await handle_question(message, user_id)


async def handle_file_uploads(message: cl.Message, user_id: str) -> bool:
    """
    Handle document uploads from Chainlit.

    Upload only stores the file in the backend.
    Processing is intentionally separate for Day 17.
    """
    elements = message.elements or []

    file_elements = [
        element
        for element in elements
        if getattr(element, "path", None)
    ]

    if not file_elements:
        return False

    uploaded_results: list[str] = []

    for file_element in file_elements:
        file_path = file_element.path
        file_name = getattr(file_element, "name", None) or "uploaded_document"

        try:
            result = await upload_document(
                user_id=user_id,
                file_path=file_path,
                file_name=file_name,
                category=DEFAULT_CATEGORY,
            )

            document_id = _extract_document_id(result)
            status = result.get("status", "uploaded")

            uploaded_results.append(
                (
                    "✅ Document uploaded successfully.\n\n"
                    f"**File:** `{file_name}`\n\n"
                    f"**Document ID:** `{document_id}`\n\n"
                    f"**Status:** `{status}`\n\n"
                    "To prepare this document for search, type:\n\n"
                    f"```text\n/process {document_id}\n```"
                )
            )

        except Exception as exc:
            uploaded_results.append(
                (
                    f"❌ Failed to upload `{file_name}`.\n\n"
                    f"```text\n{exc}\n```"
                )
            )

    await cl.Message(
        content="\n\n---\n\n".join(uploaded_results)
    ).send()

    return True


async def handle_documents_command(message: cl.Message, user_id: str) -> bool:
    """
    Handle:

        /documents
    """
    content = message.content.strip()

    if content != "/documents":
        return False

    try:
        documents = await list_documents(user_id=user_id)
    except Exception as exc:
        await cl.Message(
            content=(
                "❌ Failed to list documents.\n\n"
                f"```text\n{exc}\n```"
            )
        ).send()
        return True

    if not documents:
        await cl.Message(
            content="No documents found yet. Upload a document first."
        ).send()
        return True

    lines = ["## Your Documents\n"]

    for document in documents:
        document_id = document.get("id") or document.get("document_id")
        file_name = (
            document.get("original_file_name")
            or document.get("file_name")
            or document.get("stored_file_name")
            or "unknown"
        )
        category = document.get("category") or "unknown"
        status = document.get("status") or "unknown"
        readiness = _document_readiness_label(status)

        lines.append(
            (
                f"- **{file_name}**\n"
                f"  - Document ID: `{document_id}`\n"
                f"  - Category: `{category}`\n"
                f"  - Status: `{status}`\n"
                f"  - Readiness: **{readiness}**"
            )
        )

    await cl.Message(
        content="\n".join(lines)
    ).send()

    return True


async def handle_process_command(message: cl.Message, user_id: str) -> bool:
    """
    Handle:

        /process <document_id>
    """
    content = message.content.strip()

    if not content.startswith("/process"):
        return False

    parts = content.split(maxsplit=1)

    if len(parts) != 2 or not parts[1].strip():
        await cl.Message(
            content=(
                "Usage:\n\n"
                "```text\n"
                "/process <document_id>\n"
                "```"
            )
        ).send()
        return True

    document_id = parts[1].strip()

    await cl.Message(
        content=f"Processing document `{document_id}`..."
    ).send()

    try:
        result = await process_document(
            user_id=user_id,
            document_id=document_id,
        )
    except Exception as exc:
        await cl.Message(
            content=(
                "❌ Document processing failed:\n\n"
                f"```text\n{exc}\n```"
            )
        ).send()
        return True

    steps_text = _format_processing_steps(result.get("steps", []))
    final_status = result.get("status", "unknown")
    final_message = result.get("message", "")
    result_document_id = result.get("document_id", document_id)

    ready_message = ""
    if final_status == "embedded":
        ready_message = "\n\n✅ Document is ready for questions."

    await cl.Message(
        content=(
            "## Document Processing Result\n\n"
            f"**Document ID:** `{result_document_id}`\n\n"
            f"**Final Status:** `{final_status}`\n\n"
            f"{steps_text}\n\n"
            f"**Message:** {final_message}"
            f"{ready_message}"
        )
    ).send()

    return True


async def handle_question(message: cl.Message, user_id: str) -> None:
    """
    Handle normal user questions through backend /chat/ask.
    """
    question = message.content.strip()

    if not question:
        await cl.Message(
            content="Please enter a question."
        ).send()
        return

    session_id = cl.user_session.get("session_id")

    try:
        result = await ask_question(
            user_id=user_id,
            question=question,
            session_id=session_id,
        )
    except Exception as exc:
        await cl.Message(
            content=(
                "❌ Failed to get answer from backend.\n\n"
                f"```text\n{exc}\n```"
            )
        ).send()
        return

    new_session_id = result.get("session_id")
    if new_session_id:
        cl.user_session.set("session_id", new_session_id)

    answer = (
        result.get("answer")
        or result.get("final_answer")
        or result.get("message")
        or "No answer returned."
    )

    citations = result.get("citations") or []
    metadata_text = _format_response_metadata(result)
    citations_text = _format_citations(citations)

    await cl.Message(
        content=(
            f"{answer}\n\n"
            f"{metadata_text}\n\n"
            f"{citations_text}"
        )
    ).send()


def _format_processing_steps(steps: list[dict[str, Any]]) -> str:
    """
    Format processing pipeline steps for Chainlit display.
    """
    if not steps:
        return "No processing steps were run."

    lines: list[str] = []

    for step in steps:
        name = step.get("name", "unknown")
        status = step.get("status", "unknown")
        message = step.get("message", "")

        if status == "completed":
            icon = "✅"
        elif status == "failed":
            icon = "❌"
        else:
            icon = "ℹ️"

        lines.append(
            f"{icon} **{name}** — `{status}`\n{message}"
        )

    return "\n\n".join(lines)


def _document_readiness_label(status: str) -> str:
    """
    Convert backend document status into user-friendly readiness text.
    """
    if status == "embedded":
        return "Ready for questions"

    if status == "uploaded":
        return "Not ready yet — run /process"

    if status == "extracted":
        return "Partially processed — needs chunking and embedding"

    if status == "chunked":
        return "Partially processed — needs embedding"

    if status == "failed":
        return "Processing failed"

    if status == "deleted":
        return "Deleted"

    return "Unknown"


def _format_response_metadata(result: dict[str, Any]) -> str:
    """
    Format response metadata if backend returns it.
    """
    status = result.get("status")
    validation_status = result.get("validation_status")
    validation_reason = result.get("validation_reason")
    session_id = result.get("session_id")

    lines: list[str] = []

    if status:
        lines.append(f"- Status: `{status}`")

    if validation_status:
        lines.append(f"- Validation Status: `{validation_status}`")

    if validation_reason:
        lines.append(f"- Validation Reason: {validation_reason}")

    if session_id:
        lines.append(f"- Session ID: `{session_id}`")

    if not lines:
        return ""

    return "## Response Metadata\n" + "\n".join(lines)


def _format_citations(citations: list[dict[str, Any]]) -> str:
    """
    Format citations returned by backend.
    """
    if not citations:
        return "## Sources\nNo citations returned."

    lines = ["## Sources"]

    for index, citation in enumerate(citations, start=1):
        document_name = (
            citation.get("document_name")
            or citation.get("original_file_name")
            or "Unknown document"
        )
        category = citation.get("category") or "N/A"
        page_number = citation.get("page_number")
        section_title = citation.get("section_title")
        chunk_id = citation.get("chunk_id")
        chunk_index = citation.get("chunk_index")
        reranker_score = citation.get("reranker_score")
        hybrid_score = citation.get("hybrid_score")

        lines.append(f"\n### {index}. {document_name}")
        lines.append(f"- Category: `{category}`")

        if page_number is not None:
            lines.append(f"- Page: `{page_number}`")
        else:
            lines.append("- Page: `N/A`")

        if section_title:
            lines.append(f"- Section: `{section_title}`")
        else:
            lines.append("- Section: `N/A`")

        if chunk_id:
            lines.append(f"- Chunk ID: `{chunk_id}`")

        if chunk_index is not None:
            lines.append(f"- Chunk Index: `{chunk_index}`")

        if reranker_score is not None:
            lines.append(f"- Reranker Score: `{reranker_score}`")

        if hybrid_score is not None:
            lines.append(f"- Hybrid Score: `{hybrid_score}`")

    return "\n".join(lines)


def _extract_document_id(payload: dict[str, Any]) -> str:
    """
    Extract document id from different possible backend response shapes.
    """
    if payload.get("id"):
        return str(payload["id"])

    if payload.get("document_id"):
        return str(payload["document_id"])

    document = payload.get("document")
    if isinstance(document, dict):
        if document.get("id"):
            return str(document["id"])
        if document.get("document_id"):
            return str(document["document_id"])

    return "unknown"