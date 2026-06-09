import chainlit as cl

from services.api_client import (
    APIClientError,
    ask_question,
    delete_document,
    list_documents,
    process_document,
    upload_document,
)


DEFAULT_USER_ID = "local-user-123"

SUPPORTED_CATEGORIES = [
    "health_insurance",
    "auto_insurance",
    "home_insurance",
    "mortgage",
    "hoa",
    "employer_benefits",
    "internet",
    "utility",
    "banking",
    "credit_card",
    "warranty",
    "travel",
    "general",
]


def _format_processing_steps(response: dict) -> str:
    steps = response.get("steps") or []

    if not steps:
        return "No processing steps returned."

    lines = []

    for step in steps:
        name = step.get("name", "unknown")
        status = step.get("status", "unknown")
        message = step.get("message", "")

        lines.append(f"- {name}: {status} — {message}")

    return "\n".join(lines)


def _document_readiness_label(status: str | None) -> tuple[str, str]:
    normalized_status = (status or "unknown").lower()

    if normalized_status == "uploaded":
        return (
            "Not ready",
            "type /process <document_id>",
        )

    if normalized_status == "processing":
        return (
            "Processing",
            "wait or try /documents later",
        )

    if normalized_status == "extracted":
        return (
            "Partially processed",
            "type /process <document_id>",
        )

    if normalized_status == "chunked":
        return (
            "Partially processed",
            "type /process <document_id>",
        )

    if normalized_status == "embedded":
        return (
            "Ready for questions",
            "Ask a question now.",
        )

    if normalized_status == "failed":
        return (
            "Failed",
            "type /reprocess <document_id>",
        )

    if normalized_status == "deleted":
        return (
            "Deleted",
            "No action available.",
        )

    return (
        "Unknown",
        "type /documents later or try /process <document_id>",
    )


def _format_response_metadata(response: dict) -> str:
    metadata_lines = []

    status = response.get("status")
    validation_status = response.get("validation_status")
    validation_reason = response.get("validation_reason")
    model_name = response.get("model_name")
    session_id = response.get("session_id")

    if status:
        metadata_lines.append(f"- Status: `{status}`")

    if validation_status:
        metadata_lines.append(f"- Validation Status: `{validation_status}`")

    if validation_reason:
        metadata_lines.append(f"- Validation Reason: {validation_reason}")

    if model_name:
        metadata_lines.append(f"- Model: `{model_name}`")

    if session_id:
        metadata_lines.append(f"- Session ID: `{session_id}`")

    if not metadata_lines:
        return ""

    return "## Response Metadata\n" + "\n".join(metadata_lines)


def _format_citations(response: dict) -> str:
    citations = response.get("citations") or []

    if not citations:
        return ""

    lines = ["## Sources"]

    for index, citation in enumerate(citations, start=1):
        document_name = citation.get("document_name") or "Unknown document"
        category = citation.get("category") or "N/A"
        page_number = citation.get("page_number")
        section_title = citation.get("section_title") or "N/A"
        chunk_id = citation.get("chunk_id") or "N/A"
        reranker_score = citation.get("reranker_score")
        hybrid_score = citation.get("hybrid_score")

        page_display = page_number if page_number is not None else "N/A"

        lines.append("")
        lines.append(f"### {index}. {document_name}")
        lines.append(f"- Category: `{category}`")
        lines.append(f"- Page: `{page_display}`")
        lines.append(f"- Section: `{section_title}`")
        lines.append(f"- Chunk ID: `{chunk_id}`")

        if reranker_score is not None:
            lines.append(f"- Reranker Score: `{reranker_score}`")

        if hybrid_score is not None:
            lines.append(f"- Hybrid Score: `{hybrid_score}`")

    return "\n".join(lines)


def _extract_document_id(response: dict) -> str | None:
    possible_keys = [
        "document_id",
        "id",
    ]

    for key in possible_keys:
        value = response.get(key)
        if value:
            return str(value)

    document = response.get("document")
    if isinstance(document, dict):
        value = document.get("document_id") or document.get("id")
        if value:
            return str(value)

    return None


def _get_user_id() -> str:
    return cl.user_session.get("user_id", DEFAULT_USER_ID)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("user_id", DEFAULT_USER_ID)
    cl.user_session.set("session_id", None)
    cl.user_session.set("current_category", "general")

    await cl.Message(
        content=(
            "Personal Policy RAG Assistant is ready.\n\n"
            "Upload a document or type `/help` to see available commands."
        )
    ).send()


async def handle_file_uploads(files: list) -> None:
    user_id = _get_user_id()
    category = cl.user_session.get("current_category", "general")

    for file in files:
        try:
            response = await upload_document(
                user_id=user_id,
                file_path=file.path,
                file_name=file.name,
                category=category,
            )

            document_id = _extract_document_id(response)
            status = response.get("status", "uploaded")

            if not document_id:
                await cl.Message(
                    content=(
                        "Document uploaded, but I could not find the document ID "
                        "in the backend response. Type `/documents` to verify."
                    )
                ).send()
                continue

            await cl.Message(
                content=(
                    "Document uploaded successfully.\n\n"
                    f"File: {file.name}\n"
                    f"Document ID: {document_id}\n"
                    f"Category: {category}\n"
                    f"Status: {status}\n\n"
                    "Next step:\n"
                    f"Type `/process {document_id}` to prepare this document for questions.\n\n"
                    "Other actions:\n"
                    "- /documents\n"
                    f"- /delete {document_id}\n"
                    "- /category <category_name>"
                )
            ).send()

        except APIClientError as exc:
            await cl.Message(
                content=f"Upload failed: {exc}"
            ).send()
        except Exception as exc:
            await cl.Message(
                content=f"Unexpected upload error: {exc}"
            ).send()


async def handle_documents_command() -> None:
    user_id = _get_user_id()

    try:
        response = await list_documents(user_id=user_id)
    except APIClientError as exc:
        await cl.Message(content=f"Could not list documents: {exc}").send()
        return

    documents = response.get("documents") or []

    visible_documents = [
        document
        for document in documents
        if str(document.get("status", "")).lower() != "deleted"
    ]

    if not visible_documents:
        await cl.Message(
            content=(
                "No active documents found.\n\n"
                "Upload a document first, then type `/documents` again."
            )
        ).send()
        return

    lines = ["Documents:"]

    for index, document in enumerate(visible_documents, start=1):
        document_id = document.get("document_id") or document.get("id")
        file_name = document.get("file_name") or document.get("filename") or "Unknown file"
        category = document.get("category") or "general"
        status = document.get("status") or "unknown"

        readiness, action = _document_readiness_label(status)

        if action.startswith("type /process") and document_id:
            action = f"/process {document_id}"

        if action.startswith("type /reprocess") and document_id:
            action = f"/reprocess {document_id}"

        lines.append("")
        lines.append(f"{index}. {file_name}")
        lines.append(f"   ID: {document_id}")
        lines.append(f"   Category: {category}")
        lines.append(f"   Status: {status}")
        lines.append(f"   Readiness: {readiness}")
        lines.append(f"   Action: {action}")

    await cl.Message(content="\n".join(lines)).send()


async def handle_process_command(message_text: str) -> None:
    user_id = _get_user_id()
    parts = message_text.strip().split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: /process <document_id>").send()
        return

    document_id = parts[1].strip()

    try:
        response = await process_document(
            user_id=user_id,
            document_id=document_id,
            force=False,
        )
    except APIClientError as exc:
        await cl.Message(content=f"Processing failed: {exc}").send()
        return

    steps_text = _format_processing_steps(response)
    final_status = response.get("status", "unknown")
    message = response.get("message", "")

    content = (
        f"Processing result for document `{document_id}`:\n\n"
        f"{steps_text}\n\n"
        f"Final status: `{final_status}`\n"
        f"Message: {message}"
    )

    if final_status == "embedded":
        content += "\n\nDocument is ready for questions."

    await cl.Message(content=content).send()


async def handle_reprocess_command(message_text: str) -> None:
    user_id = _get_user_id()
    parts = message_text.strip().split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: /reprocess <document_id>").send()
        return

    document_id = parts[1].strip()

    await cl.Message(
        content=(
            "Reprocessing reruns extract → chunk → embed.\n\n"
            f"Document ID: `{document_id}`"
        )
    ).send()

    try:
        response = await process_document(
            user_id=user_id,
            document_id=document_id,
            force=True,
        )
    except APIClientError as exc:
        await cl.Message(content=f"Reprocessing failed: {exc}").send()
        return

    steps_text = _format_processing_steps(response)
    final_status = response.get("status", "unknown")
    message = response.get("message", "")

    content = (
        f"Reprocessing result for document `{document_id}`:\n\n"
        f"{steps_text}\n\n"
        f"Final status: `{final_status}`\n"
        f"Message: {message}"
    )

    if final_status == "embedded":
        content += "\n\nDocument is ready for questions."

    await cl.Message(content=content).send()


async def handle_delete_command(message_text: str) -> None:
    user_id = _get_user_id()
    parts = message_text.strip().split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: /delete <document_id>").send()
        return

    document_id = parts[1].strip()

    await cl.Message(
        content=(
            "This soft-deletes the document and removes it from normal search results.\n\n"
            f"Document ID: `{document_id}`"
        )
    ).send()

    try:
        response = await delete_document(
            user_id=user_id,
            document_id=document_id,
        )
    except APIClientError as exc:
        await cl.Message(content=f"Delete failed: {exc}").send()
        return

    message = response.get("message", "Document deleted.")
    status = response.get("status", "deleted")

    await cl.Message(
        content=(
            f"{message}\n\n"
            f"Status: `{status}`\n\n"
            "Type `/documents` to verify."
        )
    ).send()


async def handle_categories_command() -> None:
    lines = ["Supported categories:"]

    for category in SUPPORTED_CATEGORIES:
        lines.append(f"- {category}")

    current_category = cl.user_session.get("current_category", "general")

    lines.append("")
    lines.append(f"Current upload category: `{current_category}`")
    lines.append("")
    lines.append("To change it, type `/category <category_name>`.")

    await cl.Message(content="\n".join(lines)).send()


async def handle_category_command(message_text: str) -> None:
    parts = message_text.strip().split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(
            content=(
                "Usage: /category <category_name>\n\n"
                "Type `/categories` to see supported categories."
            )
        ).send()
        return

    category = parts[1].strip()

    if category not in SUPPORTED_CATEGORIES:
        supported = "\n".join(f"- {item}" for item in SUPPORTED_CATEGORIES)

        await cl.Message(
            content=(
                f"Unsupported category: `{category}`\n\n"
                "Supported categories:\n"
                f"{supported}"
            )
        ).send()
        return

    cl.user_session.set("current_category", category)

    await cl.Message(
        content=f"Current upload category set to: `{category}`"
    ).send()


async def handle_help_command() -> None:
    await cl.Message(
        content=(
            "Available commands:\n"
            "- /documents — list uploaded documents\n"
            "- /process <document_id> — extract, chunk, and embed a document\n"
            "- /reprocess <document_id> — force reprocess a document\n"
            "- /delete <document_id> — soft delete a document\n"
            "- /category <category_name> — set upload category\n"
            "- /categories — show supported categories\n"
            "- /new — start a new chat session\n"
            "- /help — show help"
        )
    ).send()


async def handle_new_command() -> None:
    cl.user_session.set("session_id", None)

    await cl.Message(
        content="Started a new chat session. Ask your next question when ready."
    ).send()


async def handle_question(message_text: str) -> None:
    user_id = _get_user_id()
    session_id = cl.user_session.get("session_id")

    try:
        response = await ask_question(
            user_id=user_id,
            question=message_text,
            session_id=session_id,
        )
    except APIClientError as exc:
        await cl.Message(content=f"Question failed: {exc}").send()
        return

    returned_session_id = response.get("session_id")
    if returned_session_id:
        cl.user_session.set("session_id", returned_session_id)

    answer = response.get("answer") or response.get("final_answer") or "No answer returned."
    metadata = _format_response_metadata(response)
    citations = _format_citations(response)

    sections = [answer]

    if metadata:
        sections.append(metadata)

    if citations:
        sections.append(citations)

    await cl.Message(content="\n\n".join(sections)).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    message_text = (message.content or "").strip()

    if message.elements:
        files = [
            element
            for element in message.elements
            if getattr(element, "path", None)
        ]

        if files:
            await handle_file_uploads(files)
            return

    if not message_text:
        await cl.Message(
            content="Type `/help` to see available commands."
        ).send()
        return

    normalized = message_text.lower()

    if normalized.startswith("/documents"):
        await handle_documents_command()

    elif normalized.startswith("/process"):
        await handle_process_command(message_text)

    elif normalized.startswith("/reprocess"):
        await handle_reprocess_command(message_text)

    elif normalized.startswith("/delete"):
        await handle_delete_command(message_text)

    elif normalized.startswith("/categories"):
        await handle_categories_command()

    elif normalized.startswith("/category"):
        await handle_category_command(message_text)

    elif normalized.startswith("/help"):
        await handle_help_command()

    elif normalized.startswith("/new"):
        await handle_new_command()

    else:
        await handle_question(message_text)