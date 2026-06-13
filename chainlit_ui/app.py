import chainlit as cl

from services.api_client import (
    APIClientError,
    ask_question,
    delete_document,
    get_chunk_detail,
    get_current_user,
    get_google_login_url,
    get_health_status,
    get_processing_job,
    get_message_evidence,
    list_document_chunks,
    list_documents,
    list_processing_jobs,
    login_user,
    process_document,
    register_user,
    upload_document,
)


DEFAULT_CATEGORY = "general"


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

SUPPORTED_DOCUMENT_STATUSES = [
    "uploaded",
    "processing",
    "extracted",
    "chunked",
    "embedded",
    "failed",
]


LOGIN_REQUIRED_MESSAGE = "Please login first using /login, /register, or /google."


def _get_access_token() -> str | None:
    return cl.user_session.get("access_token")


def _get_user_id() -> str | None:
    return cl.user_session.get("user_id")


def _is_authenticated() -> bool:
    return bool(_get_access_token() and _get_user_id())


def _get_current_category() -> str:
    return cl.user_session.get("current_category") or DEFAULT_CATEGORY


def _reset_document_filters() -> None:
    cl.user_session.set("document_filter_status", None)
    cl.user_session.set("document_filter_category", None)
    cl.user_session.set("document_filter_search", None)
    cl.user_session.set("document_filter_ready_only", False)


def _reset_last_answer_evidence() -> None:
    cl.user_session.set("last_message_id", None)
    cl.user_session.set("last_citations", [])
    cl.user_session.set("last_answer_evidence", [])
    cl.user_session.set("last_evidence_chunk_count", 0)


def _get_document_filters() -> dict:
    return {
        "status": cl.user_session.get("document_filter_status"),
        "category": cl.user_session.get("document_filter_category"),
        "search": cl.user_session.get("document_filter_search"),
        "ready_only": bool(cl.user_session.get("document_filter_ready_only")),
    }


def _has_active_document_filters() -> bool:
    return any(_get_document_filters().values())


def _format_active_document_filters() -> str:
    filters = _get_document_filters()
    lines = [
        f"- {name}: `{value}`"
        for name, value in filters.items()
        if value
    ]

    if not lines:
        return "Active filters: none"

    return "Active filters:\n" + "\n".join(lines)


def _format_timestamp(value) -> str:
    if value is None:
        return "N/A"

    return str(value)


def _store_auth_session(auth_response: dict) -> None:
    access_token = auth_response.get("access_token")
    user = auth_response.get("user")

    if not access_token:
        raise APIClientError("Auth response did not include access_token")

    if not isinstance(user, dict):
        raise APIClientError("Auth response did not include user")

    cl.user_session.set("access_token", access_token)
    cl.user_session.set("current_user", user)
    cl.user_session.set("user_id", user.get("id"))
    cl.user_session.set("email", user.get("email"))
    cl.user_session.set("auth_provider", user.get("auth_provider"))

    # New authenticated user context should start a new chat session.
    cl.user_session.set("session_id", None)
    _reset_last_answer_evidence()
    _reset_document_filters()


def _clear_auth_session() -> None:
    cl.user_session.set("access_token", None)
    cl.user_session.set("current_user", None)
    cl.user_session.set("user_id", None)
    cl.user_session.set("email", None)
    cl.user_session.set("auth_provider", None)
    cl.user_session.set("session_id", None)
    _reset_last_answer_evidence()
    _reset_document_filters()


async def _require_login() -> bool:
    if _is_authenticated():
        return True

    await cl.Message(content=LOGIN_REQUIRED_MESSAGE).send()
    return False


def _format_processing_steps(response: dict) -> str:
    steps = response.get("steps") or []

    if not steps:
        return "No processing steps returned."

    lines = []

    for step in steps:
        name = step.get("name", "unknown")
        status = step.get("status", "unknown")
        message = step.get("message", "")

        lines.append(f"- {name}: {status} - {message}")

    return "\n".join(lines)


def _document_readiness_label(status: str | None) -> tuple[str, str]:
    normalized_status = (status or "unknown").lower()

    if normalized_status == "uploaded":
        return "Not ready", "type /process <document_id>"

    if normalized_status == "processing":
        return "Processing", "wait or try /documents later"

    if normalized_status == "extracted":
        return "Partially processed", "type /process <document_id>"

    if normalized_status == "chunked":
        return "Partially processed", "type /process <document_id>"

    if normalized_status == "embedded":
        return "Ready for questions", "Ask a question now."

    if normalized_status == "failed":
        return "Failed", "type /process <document_id>"

    if normalized_status == "deleted":
        return "Deleted", "No action available."

    return "Unknown", "type /documents later or try /process <document_id>"


def _format_response_metadata(response: dict) -> str:
    metadata_lines = []

    status = response.get("status")
    validation_status = response.get("validation_status")
    evidence_sufficient = response.get("evidence_sufficient")
    evidence_reason = response.get("evidence_sufficiency_reason")
    rewritten_question = response.get("rewritten_question")
    session_id = response.get("session_id")

    if status:
        metadata_lines.append(f"- Status: `{status}`")

    if validation_status:
        metadata_lines.append(f"- Validation Status: `{validation_status}`")

    if evidence_sufficient is not None:
        metadata_lines.append(f"- Evidence Sufficient: `{evidence_sufficient}`")

    if evidence_reason:
        metadata_lines.append(f"- Evidence Reason: {evidence_reason}")

    if rewritten_question:
        metadata_lines.append(f"- Rewritten Question: {rewritten_question}")

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
        chunk_index = citation.get("chunk_index")
        reranker_score = citation.get("reranker_score")
        hybrid_score = citation.get("hybrid_score")
        vector_score = citation.get("vector_score")
        bm25_score = citation.get("bm25_score")

        page_display = page_number if page_number is not None else "N/A"
        chunk_display = chunk_index if chunk_index is not None else "N/A"

        lines.append("")
        lines.append(f"### {index}. {document_name}")
        lines.append(f"- Category: `{category}`")
        lines.append(f"- Page: `{page_display}`")
        lines.append(f"- Section: `{section_title}`")
        lines.append(f"- Chunk: `{chunk_display}`")
        lines.append(f"- Chunk ID: `{chunk_id}`")

        scores = []
        for label, value in [
            ("reranker", reranker_score),
            ("hybrid", hybrid_score),
            ("vector", vector_score),
            ("bm25", bm25_score),
        ]:
            if value is not None:
                try:
                    scores.append(f"{label}={float(value):.4f}")
                except (TypeError, ValueError):
                    scores.append(f"{label}={value}")

        if scores:
            lines.append(f"- Scores: `{', '.join(scores)}`")

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


def _extract_documents(response: dict | list) -> list[dict]:
    if isinstance(response, list):
        return response

    if isinstance(response, dict):
        documents = response.get("documents")
        if isinstance(documents, list):
            return documents

    return []


def _get_action_value(action: cl.Action) -> str:
    legacy_value = getattr(action, "value", None)
    if legacy_value:
        return str(legacy_value)

    payload = getattr(action, "payload", None)
    if isinstance(payload, dict):
        value = payload.get("value")
        if value:
            return str(value)

    return ""


def _action(name: str, value: str, label: str) -> cl.Action:
    return cl.Action(
        name=name,
        payload={"value": value},
        label=label,
    )


def _document_actions(document: dict) -> list[cl.Action]:
    document_id = document.get("document_id") or document.get("id")
    if not document_id:
        return []

    value = str(document_id)
    status = str(document.get("status") or "unknown").lower()

    if status in {"uploaded", "extracted", "chunked", "failed"}:
        return [
            _action("process_document_action", value, "Process"),
            _action("delete_document_action", value, "Delete"),
            _action("view_jobs_action", value, "View Jobs"),
        ]

    if status == "embedded":
        return [
            _action("reprocess_document_action", value, "Reprocess"),
            _action("delete_document_action", value, "Delete"),
            _action("view_jobs_action", value, "View Jobs"),
        ]

    if status == "processing":
        return [_action("view_jobs_action", value, "View Jobs")]

    return []


def _job_detail_action(job_id: str) -> cl.Action:
    return _action("view_job_detail_action", job_id, "View Job Detail")


def _refresh_documents_action() -> cl.Action:
    return _action("refresh_documents_action", "refresh", "Refresh Documents")


def _document_filter_actions() -> list[cl.Action]:
    return [
        _action("filter_ready_action", "ready", "Ready Only"),
        _action("filter_failed_action", "failed", "Failed"),
        _action("filter_uploaded_action", "uploaded", "Uploaded"),
        _action("clear_document_filters_action", "clear", "Clear Filters"),
        _refresh_documents_action(),
    ]


def _source_actions(citations: list[dict]) -> list[cl.Action]:
    actions = []

    for index, citation in enumerate(citations, start=1):
        chunk_id = citation.get("chunk_id")
        if chunk_id:
            actions.append(
                _action("view_chunk_action", str(chunk_id), f"View Source {index}")
            )

    for citation in citations:
        document_id = citation.get("document_id")
        category = citation.get("category")

        if document_id:
            actions.append(_action("view_jobs_action", str(document_id), "View Document Jobs"))
        if category:
            actions.append(
                _action("filter_source_category_action", str(category), "Filter by Category")
            )
        actions.append(_action("filter_ready_action", "ready", "Show Ready Documents"))
        break

    return actions


def _chunk_action(chunk_id: str, label: str = "View Chunk") -> cl.Action:
    return _action("view_chunk_action", chunk_id, label)


def _safe_code_block(text: str) -> str:
    longest_run = 0
    current_run = 0

    for character in text:
        if character == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    fence = "`" * max(3, longest_run + 1)
    return f"{fence}text\n{text}\n{fence}"


def _format_chunk_detail(chunk: dict) -> str:
    chunk_text = str(chunk.get("chunk_text") or "")
    return "\n".join(
        [
            "## Source Chunk",
            f"- Document ID: `{chunk.get('document_id') or 'N/A'}`",
            f"- Chunk ID: `{chunk.get('chunk_id') or 'N/A'}`",
            f"- Section: `{chunk.get('section_title') or 'N/A'}`",
            f"- Page: `{chunk.get('page_number') if chunk.get('page_number') is not None else 'N/A'}`",
            f"- Chunk Index: `{chunk.get('chunk_index') if chunk.get('chunk_index') is not None else 'N/A'}`",
            f"- Token Count: `{chunk.get('token_count') if chunk.get('token_count') is not None else 'N/A'}`",
            f"- Status: `{chunk.get('status') or 'N/A'}`",
            "",
            "### Excerpt",
            _safe_code_block(chunk_text),
        ]
    )


def _new_chat_action() -> cl.Action:
    return _action("new_chat_action", "new", "New Chat")


def _google_login_action() -> cl.Action:
    return _action("google_login_action", "google", "Google Login")


@cl.on_chat_start
async def on_chat_start() -> None:
    if cl.user_session.get("session_id") is None:
        cl.user_session.set("session_id", None)

    if cl.user_session.get("current_category") is None:
        cl.user_session.set("current_category", DEFAULT_CATEGORY)

    if cl.user_session.get("document_filter_ready_only") is None:
        _reset_document_filters()

    startup_lines = [
        "Personal Policy RAG Assistant is ready.",
        "",
        "Please login using /login, /register, or /google before uploading documents "
        "or asking questions.",
        "",
        "Use /help to see commands.",
    ]

    try:
        health = await get_health_status()
        if health.get("dev_auth_disabled") is True:
            startup_lines.extend(["", "Development auth bypass is enabled."])
    except APIClientError:
        pass

    await cl.Message(
        content="\n".join(startup_lines),
        actions=[_google_login_action(), _new_chat_action()],
    ).send()


async def handle_file_uploads(files: list) -> None:
    if not await _require_login():
        return

    category = _get_current_category()
    access_token = _get_access_token()

    for file in files:
        try:
            response = await upload_document(
                file_path=file.path,
                file_name=file.name,
                category=category,
                access_token=access_token,
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
                    f"- File: {file.name}\n"
                    f"- Document ID: `{document_id}`\n"
                    f"- Category: `{category}`\n"
                    f"- Status: `{status}`\n\n"
                    "Select **Process** to prepare this document for questions."
                ),
                actions=[
                    _action("process_document_action", document_id, "Process"),
                    _action("delete_document_action", document_id, "Delete"),
                    _refresh_documents_action(),
                ],
            ).send()

        except APIClientError as exc:
            await cl.Message(content=f"Upload failed: {exc}").send()
        except Exception as exc:
            await cl.Message(content=f"Unexpected upload error: {exc}").send()


async def handle_documents_command() -> None:
    if not await _require_login():
        return

    try:
        filters = _get_document_filters()
        response = await list_documents(
            access_token=_get_access_token(),
            **filters,
        )
    except APIClientError as exc:
        await cl.Message(content=f"Could not list documents: {exc}").send()
        return

    documents = _extract_documents(response)

    visible_documents = [
        document
        for document in documents
        if str(document.get("status", "")).lower() != "deleted"
    ]

    if not visible_documents:
        message = (
            "No documents match the current filters. Try `/filter clear`."
            if _has_active_document_filters()
            else "No documents found. Upload a file to get started."
        )
        await cl.Message(
            content=f"{_format_active_document_filters()}\n\n{message}",
            actions=_document_filter_actions(),
        ).send()
        return

    await cl.Message(
        content=(
            f"{_format_active_document_filters()}\n\n"
            f"Matching documents: `{len(visible_documents)}`"
        ),
        actions=_document_filter_actions(),
    ).send()

    for document in visible_documents:
        document_id = document.get("document_id") or document.get("id")
        file_name = (
            document.get("original_file_name")
            or document.get("original_filename")
            or document.get("file_name")
            or document.get("filename")
            or "Unknown file"
        )
        category = document.get("category") or "general"
        status = document.get("status") or "unknown"
        created_at = _format_timestamp(document.get("created_at"))
        updated_at = _format_timestamp(document.get("updated_at"))

        readiness, suggested_action = _document_readiness_label(status)

        if suggested_action.startswith("type /process") and document_id:
            suggested_action = f"Process this document or use `/process {document_id}`."

        if suggested_action.startswith("type /reprocess") and document_id:
            suggested_action = f"Reprocess this document or use `/reprocess {document_id}`."

        await cl.Message(
            content=(
                f"## {file_name}\n"
                f"- Document ID: `{document_id}`\n"
                f"- Category: `{category}`\n"
                f"- Status: `{status}`\n"
                f"- Readiness: **{readiness}**\n"
                f"- Uploaded: `{created_at}`\n"
                f"- Updated: `{updated_at}`\n"
                f"- Suggested action: {suggested_action}"
            ),
            actions=_document_actions(document),
        ).send()


async def handle_filter_command(message_text: str) -> None:
    if not await _require_login():
        return

    parts = message_text.strip().split(maxsplit=2)
    if len(parts) < 2:
        await cl.Message(
            content=(
                "Usage: `/filter status <status>`, `/filter category <category>`, "
                "`/filter search <text>`, `/filter ready`, or `/filter clear`."
            )
        ).send()
        return

    filter_type = parts[1].lower()

    if filter_type == "clear":
        _reset_document_filters()
    elif filter_type == "ready":
        cl.user_session.set("document_filter_status", None)
        cl.user_session.set("document_filter_ready_only", True)
    elif filter_type == "status":
        if len(parts) < 3 or parts[2].strip().lower() not in SUPPORTED_DOCUMENT_STATUSES:
            await cl.Message(
                content=f"Supported statuses: {', '.join(SUPPORTED_DOCUMENT_STATUSES)}"
            ).send()
            return
        cl.user_session.set("document_filter_status", parts[2].strip().lower())
        cl.user_session.set("document_filter_ready_only", False)
    elif filter_type == "category":
        if len(parts) < 3 or parts[2].strip().lower() not in SUPPORTED_CATEGORIES:
            await cl.Message(
                content=f"Supported categories: {', '.join(SUPPORTED_CATEGORIES)}"
            ).send()
            return
        cl.user_session.set("document_filter_category", parts[2].strip().lower())
    elif filter_type == "search":
        if len(parts) < 3 or not parts[2].strip():
            await cl.Message(content="Usage: `/filter search <text>`").send()
            return
        cl.user_session.set("document_filter_search", parts[2].strip())
    else:
        await cl.Message(content=f"Unknown document filter: `{filter_type}`").send()
        return

    await handle_documents_command()


async def _handle_process_document(document_id: str, force: bool = False) -> None:
    if not await _require_login():
        return

    if force:
        await cl.Message(
            content=(
                "Reprocessing reruns extract → chunk → embed.\n\n"
                f"Document ID: `{document_id}`"
            )
        ).send()

    try:
        response = await process_document(
            document_id=document_id,
            force=force,
            access_token=_get_access_token(),
        )
    except APIClientError as exc:
        label = "Reprocessing" if force else "Processing"
        await cl.Message(content=f"{label} failed: {exc}").send()
        return

    steps_text = _format_processing_steps(response)
    final_status = response.get("status", "unknown")
    message = response.get("message", "")
    job_id = response.get("job_id")

    title = "Reprocessing result" if force else "Processing result"

    content = (
        f"{title} for document `{document_id}`:\n\n"
        f"{steps_text}\n\n"
        f"Final status: `{final_status}`\n"
        f"Message: {message}"
    )

    if final_status == "embedded":
        content += "\n\nDocument is ready for questions."

    if job_id:
        content += (
            f"\n\nProcessing Job ID: `{job_id}`\n"
            f"To inspect later: `/job {job_id}`"
        )

    actions = [_refresh_documents_action()]
    if job_id:
        actions.insert(
            0,
            _action("view_job_detail_action", str(job_id), "View Job"),
        )

    await cl.Message(content=content, actions=actions).send()


async def handle_process_command(message_text: str, force: bool = False) -> None:
    parts = message_text.strip().split(maxsplit=1)
    command_name = "/reprocess" if force else "/process"

    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content=f"Usage: {command_name} <document_id>").send()
        return

    await _handle_process_document(parts[1].strip(), force=force)


async def _handle_document_jobs(document_id: str) -> None:
    if not await _require_login():
        return

    try:
        response = await list_processing_jobs(
            document_id=document_id,
            access_token=_get_access_token(),
        )
    except APIClientError as exc:
        await cl.Message(content=f"Could not list processing jobs: {exc}").send()
        return

    jobs = response.get("jobs") or []
    if not jobs:
        await cl.Message(content=f"No processing jobs found for `{document_id}`.").send()
        return

    await cl.Message(
        content=f"Processing jobs for `{document_id}` (newest first):"
    ).send()

    for job in jobs:
        job_id = job.get("job_id")
        lines = [
            f"## Processing Job `{job_id or 'unknown'}`",
            f"- Status: `{job.get('status', 'unknown')}`",
            f"- Force: `{job.get('force', False)}`",
            f"- Current step: `{job.get('current_step') or 'N/A'}`",
            f"- Created: `{job.get('created_at') or 'N/A'}`",
            f"- Completed: `{job.get('completed_at') or 'N/A'}`",
        ]
        if job.get("error_message"):
            lines.append(f"- Error: {job['error_message']}")

        actions = [_job_detail_action(str(job_id))] if job_id else []
        await cl.Message(content="\n".join(lines), actions=actions).send()


async def handle_jobs_command(message_text: str) -> None:
    parts = message_text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: /jobs <document_id>").send()
        return

    await _handle_document_jobs(parts[1].strip())


async def _handle_job_detail(job_id: str) -> None:
    if not await _require_login():
        return

    try:
        job = await get_processing_job(
            job_id=job_id,
            access_token=_get_access_token(),
        )
    except APIClientError as exc:
        await cl.Message(content=f"Could not get processing job: {exc}").send()
        return

    lines = [
        f"Processing job `{job_id}`:",
        "",
        f"- Document ID: `{job.get('document_id') or 'N/A'}`",
        f"- Status: `{job.get('status', 'unknown')}`",
        f"- Force: `{job.get('force', False)}`",
        f"- Current step: `{job.get('current_step') or 'N/A'}`",
        f"- Error: {job.get('error_message') or 'N/A'}",
        f"- Started: `{job.get('started_at') or 'N/A'}`",
        f"- Completed: `{job.get('completed_at') or 'N/A'}`",
        "",
        "Steps:",
        _format_processing_steps(job),
    ]

    await cl.Message(content="\n".join(lines)).send()


async def handle_job_command(message_text: str) -> None:
    parts = message_text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: /job <job_id>").send()
        return

    await _handle_job_detail(parts[1].strip())


async def _handle_delete_document(document_id: str) -> None:
    if not await _require_login():
        return

    await cl.Message(
        content=(
            "This soft-deletes the document and removes it from normal search results.\n\n"
            f"Document ID: `{document_id}`"
        )
    ).send()

    try:
        response = await delete_document(
            document_id=document_id,
            access_token=_get_access_token(),
        )
    except APIClientError as exc:
        await cl.Message(content=f"Delete failed: {exc}").send()
        return

    message = response.get("message", "Document deleted.")
    status = response.get("status", "deleted")

    await cl.Message(
        content=(
            "Document was soft deleted.\n\n"
            f"- Backend message: {message}\n"
            f"- Status: `{status}`"
        ),
        actions=[_refresh_documents_action()],
    ).send()


async def handle_delete_command(message_text: str) -> None:
    parts = message_text.strip().split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: /delete <document_id>").send()
        return

    await _handle_delete_document(parts[1].strip())


async def _handle_source_chunk(chunk_id: str) -> None:
    if not await _require_login():
        return

    try:
        chunk = await get_chunk_detail(
            chunk_id=chunk_id,
            access_token=_get_access_token(),
        )
    except APIClientError as exc:
        await cl.Message(content=f"Could not get source chunk: {exc}").send()
        return

    await cl.Message(content=_format_chunk_detail(chunk)).send()


async def handle_source_command(message_text: str) -> None:
    parts = message_text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: `/source <chunk_id>`").send()
        return

    await _handle_source_chunk(parts[1].strip())


async def handle_chunks_command(message_text: str) -> None:
    if not await _require_login():
        return

    parts = message_text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: `/chunks <document_id>`").send()
        return

    document_id = parts[1].strip()

    try:
        response = await list_document_chunks(
            document_id=document_id,
            access_token=_get_access_token(),
        )
    except APIClientError as exc:
        await cl.Message(content=f"Could not list document chunks: {exc}").send()
        return

    chunks = response.get("chunks") or []
    if not chunks:
        await cl.Message(content=f"No chunks found for document `{document_id}`.").send()
        return

    await cl.Message(
        content=f"Document `{document_id}` has `{len(chunks)}` source chunks."
    ).send()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        page = chunk.get("page_number")
        chunk_index = chunk.get("chunk_index")
        token_count = chunk.get("token_count")
        lines = [
            f"## Chunk `{chunk_index if chunk_index is not None else 'N/A'}`",
            f"- Chunk ID: `{chunk_id or 'N/A'}`",
            f"- Section: `{chunk.get('section_title') or 'N/A'}`",
            f"- Page: `{page if page is not None else 'N/A'}`",
            f"- Token Count: `{token_count if token_count is not None else 'N/A'}`",
        ]
        actions = [_chunk_action(str(chunk_id))] if chunk_id else []
        await cl.Message(content="\n".join(lines), actions=actions).send()


async def handle_evidence_command(message_text: str) -> None:
    if not await _require_login():
        return

    parts = message_text.strip().split(maxsplit=1)
    message_id = parts[1].strip() if len(parts) > 1 else cl.user_session.get("last_message_id")

    if not message_id:
        await cl.Message(content="No recent answer found. Ask a question first.").send()
        return

    try:
        evidence = await get_message_evidence(
            message_id=str(message_id),
            access_token=_get_access_token(),
        )
    except APIClientError as exc:
        await cl.Message(content=f"Could not get message evidence: {exc}").send()
        return

    answer = str(evidence.get("answer") or "")
    answer_summary = answer[:500] + ("..." if len(answer) > 500 else "")
    citations = evidence.get("citations") or []
    retrieved_chunks = evidence.get("retrieved_chunks") or []
    lines = [
        "## Answer Evidence",
        f"- Message ID: `{evidence.get('message_id') or message_id}`",
        f"- Session ID: `{evidence.get('session_id') or 'N/A'}`",
        f"- Evidence Chunk Count: `{evidence.get('evidence_chunk_count') or 0}`",
        "",
        "### Question",
        str(evidence.get("question") or "N/A"),
        "",
        "### Answer Summary",
        answer_summary or "N/A",
    ]

    citation_text = _format_citations({"citations": citations})
    if citation_text:
        lines.extend(["", citation_text])

    if retrieved_chunks:
        lines.extend(["", "## Retrieved Chunks"])
        for index, chunk in enumerate(retrieved_chunks, start=1):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            lines.append(
                f"- {index}. Chunk `{chunk_id or 'N/A'}`, "
                f"document `{chunk.get('document_id') or 'N/A'}`, "
                f"page `{chunk.get('page_number') if chunk.get('page_number') is not None else 'N/A'}`"
            )

    actions = _source_actions(citations)
    known_chunk_ids = {
        str(citation.get("chunk_id"))
        for citation in citations
        if citation.get("chunk_id")
    }
    for index, chunk in enumerate(retrieved_chunks, start=1):
        chunk_id = chunk.get("chunk_id") or chunk.get("id")
        if chunk_id and str(chunk_id) not in known_chunk_ids:
            actions.append(_chunk_action(str(chunk_id), f"View Evidence {index}"))

    await cl.Message(content="\n".join(lines), actions=actions).send()


async def handle_categories_command() -> None:
    lines = ["Supported categories:"]

    for category in SUPPORTED_CATEGORIES:
        lines.append(f"- {category}")

    current_category = _get_current_category()

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


async def handle_register_command(message_text: str) -> None:
    parts = message_text.strip().split(maxsplit=3)

    if len(parts) < 3:
        await cl.Message(
            content=(
                "Usage:\n\n"
                "`/register <email> <password> [full name]`\n\n"
                "Example:\n\n"
                "`/register test@example.com password123 Test User`"
            )
        ).send()
        return

    email = parts[1]
    password = parts[2]
    full_name = parts[3] if len(parts) >= 4 else None

    try:
        auth_response = await register_user(
            email=email,
            password=password,
            full_name=full_name,
        )
    except APIClientError as exc:
        await cl.Message(content=f"Register failed: {exc}").send()
        return

    try:
        _store_auth_session(auth_response)
    except APIClientError as exc:
        await cl.Message(content=f"Register succeeded but session setup failed: {exc}").send()
        return

    user = auth_response["user"]

    await cl.Message(
        content=(
            "Registered and logged in.\n\n"
            f"- User ID: `{user.get('id')}`\n"
            f"- Email: `{user.get('email')}`\n"
            f"- Auth provider: `{user.get('auth_provider')}`\n\n"
            "A new chat session will start for this authenticated user."
        )
    ).send()


async def handle_login_command(message_text: str) -> None:
    parts = message_text.strip().split(maxsplit=2)

    if len(parts) < 3:
        await cl.Message(
            content=(
                "Usage:\n\n"
                "`/login <email> <password>`\n\n"
                "Example:\n\n"
                "`/login test@example.com password123`"
            )
        ).send()
        return

    email = parts[1]
    password = parts[2]

    try:
        auth_response = await login_user(
            email=email,
            password=password,
        )
    except APIClientError as exc:
        await cl.Message(content=f"Login failed: {exc}").send()
        return

    try:
        _store_auth_session(auth_response)
    except APIClientError as exc:
        await cl.Message(content=f"Login succeeded but session setup failed: {exc}").send()
        return

    user = auth_response["user"]

    await cl.Message(
        content=(
            "Logged in.\n\n"
            f"- User ID: `{user.get('id')}`\n"
            f"- Email: `{user.get('email')}`\n"
            f"- Auth provider: `{user.get('auth_provider')}`\n\n"
            "A new chat session will start for this authenticated user."
        )
    ).send()


async def handle_google_command() -> None:
    try:
        response = await get_google_login_url()
    except APIClientError as exc:
        await cl.Message(content=f"Could not start Google login: {exc}").send()
        return

    authorization_url = response["authorization_url"]

    await cl.Message(
        content=(
            "Google login:\n\n"
            f"1. Open this URL in your browser:\n{authorization_url}\n\n"
            "2. Complete Google login.\n"
            "3. Copy the app token shown on the success page.\n"
            "4. Paste it here using:\n"
            "`/token <access_token>`"
        )
    ).send()


async def handle_token_command(message_text: str) -> None:
    parts = message_text.strip().split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content="Usage: `/token <access_token>`").send()
        return

    access_token = parts[1].strip()

    try:
        user = await get_current_user(access_token)
    except APIClientError as exc:
        _clear_auth_session()
        await cl.Message(content=f"Token login failed: {exc}").send()
        return

    _store_auth_session(
        {
            "access_token": access_token,
            "user": user,
        }
    )

    await cl.Message(
        content=(
            "Logged in with app token.\n\n"
            f"- User ID: `{user.get('id')}`\n"
            f"- Email: `{user.get('email')}`\n"
            f"- Auth provider: `{user.get('auth_provider')}`\n\n"
            "A new chat session will start for this authenticated user."
        )
    ).send()


async def handle_me_command() -> None:
    access_token = _get_access_token()

    if not access_token:
        await cl.Message(
            content=(
                "You are not logged in.\n\n"
                "Use `/register <email> <password> [full name]` or "
                "`/login <email> <password>`, or start Google login with `/google`."
            )
        ).send()
        return

    try:
        user = await get_current_user(access_token)
    except APIClientError as exc:
        _clear_auth_session()
        await cl.Message(
            content=(
                f"Could not validate current user: {exc}\n\n"
                "Try logging in again with `/login <email> <password>` or `/google`."
            )
        ).send()
        return

    cl.user_session.set("current_user", user)
    cl.user_session.set("user_id", user.get("id"))
    cl.user_session.set("email", user.get("email"))
    cl.user_session.set("auth_provider", user.get("auth_provider"))

    await cl.Message(
        content=(
            "Current authenticated user:\n\n"
            f"- User ID: `{user.get('id')}`\n"
            f"- Email: `{user.get('email')}`\n"
            f"- Full name: `{user.get('full_name') or 'N/A'}`\n"
            f"- Auth provider: `{user.get('auth_provider')}`\n"
            f"- Active: `{user.get('is_active')}`"
        )
    ).send()


async def handle_logout_command() -> None:
    _clear_auth_session()

    await cl.Message(
        content=(
            "Logged out.\n\n"
            "Login again with `/login <email> <password>` or `/google`, or register "
            "with `/register <email> <password> [full name]`."
        )
    ).send()


async def handle_help_command() -> None:
    await cl.Message(
        content=(
            "You can use buttons in document cards, or slash commands:\n\n"
            "Auth:\n"
            "- `/register <email> <password> [full name]`\n"
            "- `/login <email> <password>`\n"
            "- `/google`\n"
            "- `/token <access_token>`\n"
            "- `/me`\n"
            "- `/logout`\n\n"
            "Documents:\n"
            "- `/category <category_name>`\n"
            "- `/categories`\n"
            "- Upload a file\n"
            "- `/documents`\n"
            "- `/filter status <status>`\n"
            "- `/filter category <category>`\n"
            "- `/filter search <text>`\n"
            "- `/filter ready`\n"
            "- `/filter clear`\n"
            "- You can also use filter buttons in the document list.\n"
            "- `/process <document_id>`\n"
            "- `/reprocess <document_id>`\n"
            "- `/jobs <document_id>`\n"
            "- `/job <job_id>`\n"
            "- `/delete <document_id>`\n\n"
            "Chat:\n"
            "- Ask a normal question\n"
            "- `/new`\n\n"
            "Evidence:\n"
            "- `/source <chunk_id>` - view cited source chunk text\n"
            "- `/chunks <document_id>` - list chunks for a document\n"
            "- `/evidence [message_id]` - inspect evidence for the latest or specified answer\n"
            "- Citation buttons can also open source chunks.\n\n"
            "General:\n"
            "- `/help`"
        ),
        actions=[_google_login_action(), _new_chat_action()],
    ).send()


async def handle_new_command() -> None:
    cl.user_session.set("session_id", None)
    _reset_last_answer_evidence()

    await cl.Message(
        content="Started a new chat session. Ask your next question when ready."
    ).send()


async def handle_question(message_text: str) -> None:
    if not await _require_login():
        return

    session_id = cl.user_session.get("session_id")

    try:
        response = await ask_question(
            question=message_text,
            session_id=session_id,
            access_token=_get_access_token(),
        )
    except APIClientError as exc:
        await cl.Message(content=f"Question failed: {exc}").send()
        return

    returned_session_id = response.get("session_id")
    if returned_session_id:
        cl.user_session.set("session_id", returned_session_id)

    cl.user_session.set("last_message_id", response.get("message_id"))
    cl.user_session.set("last_citations", response.get("citations") or [])
    cl.user_session.set("last_answer_evidence", response.get("evidence_chunks") or [])
    cl.user_session.set(
        "last_evidence_chunk_count",
        response.get("evidence_chunk_count") or 0,
    )

    answer = response.get("answer") or response.get("final_answer") or "No answer returned."
    metadata = _format_response_metadata(response)
    citations = _format_citations(response)
    source_actions = _source_actions(response.get("citations") or [])

    sections = [answer]

    if metadata:
        sections.append(metadata)

    if citations:
        sections.append(citations)

    await cl.Message(
        content="\n\n".join(sections),
        actions=source_actions,
    ).send()


async def _send_missing_action_value(resource_name: str) -> None:
    await cl.Message(
        content=(
            f"Could not determine the {resource_name} from this action. "
            "Refresh the document list and try again."
        )
    ).send()


@cl.action_callback("process_document_action")
async def on_process_document_action(action: cl.Action) -> None:
    document_id = _get_action_value(action)
    if not document_id:
        await _send_missing_action_value("document ID")
        return

    await _handle_process_document(document_id, force=False)


@cl.action_callback("reprocess_document_action")
async def on_reprocess_document_action(action: cl.Action) -> None:
    document_id = _get_action_value(action)
    if not document_id:
        await _send_missing_action_value("document ID")
        return

    await _handle_process_document(document_id, force=True)


@cl.action_callback("delete_document_action")
async def on_delete_document_action(action: cl.Action) -> None:
    document_id = _get_action_value(action)
    if not document_id:
        await _send_missing_action_value("document ID")
        return

    await _handle_delete_document(document_id)


@cl.action_callback("view_jobs_action")
async def on_view_jobs_action(action: cl.Action) -> None:
    document_id = _get_action_value(action)
    if not document_id:
        await _send_missing_action_value("document ID")
        return

    await _handle_document_jobs(document_id)


@cl.action_callback("view_job_detail_action")
async def on_view_job_detail_action(action: cl.Action) -> None:
    job_id = _get_action_value(action)
    if not job_id:
        await _send_missing_action_value("processing job ID")
        return

    await _handle_job_detail(job_id)


@cl.action_callback("view_chunk_action")
async def on_view_chunk_action(action: cl.Action) -> None:
    chunk_id = _get_action_value(action)
    if not chunk_id:
        await _send_missing_action_value("chunk ID")
        return

    await _handle_source_chunk(chunk_id)


@cl.action_callback("refresh_documents_action")
async def on_refresh_documents_action(action: cl.Action) -> None:
    await handle_documents_command()


@cl.action_callback("filter_ready_action")
async def on_filter_ready_action(action: cl.Action) -> None:
    cl.user_session.set("document_filter_status", None)
    cl.user_session.set("document_filter_ready_only", True)
    await handle_documents_command()


@cl.action_callback("filter_failed_action")
async def on_filter_failed_action(action: cl.Action) -> None:
    cl.user_session.set("document_filter_status", "failed")
    cl.user_session.set("document_filter_ready_only", False)
    await handle_documents_command()


@cl.action_callback("filter_uploaded_action")
async def on_filter_uploaded_action(action: cl.Action) -> None:
    cl.user_session.set("document_filter_status", "uploaded")
    cl.user_session.set("document_filter_ready_only", False)
    await handle_documents_command()


@cl.action_callback("clear_document_filters_action")
async def on_clear_document_filters_action(action: cl.Action) -> None:
    _reset_document_filters()
    await handle_documents_command()


@cl.action_callback("filter_source_category_action")
async def on_filter_source_category_action(action: cl.Action) -> None:
    category = _get_action_value(action)
    if not category:
        await _send_missing_action_value("document category")
        return

    cl.user_session.set("document_filter_category", category)
    await handle_documents_command()


@cl.action_callback("new_chat_action")
async def on_new_chat_action(action: cl.Action) -> None:
    await handle_new_command()


@cl.action_callback("google_login_action")
async def on_google_login_action(action: cl.Action) -> None:
    await handle_google_command()


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

    # Keep this order.
    # /categories must be checked before /category.
    # Auth commands must stay public.
    # Protected commands call _require_login() inside their handlers.

    if normalized == "/documents":
        await handle_documents_command()
        return

    if normalized == "/filter" or normalized.startswith("/filter "):
        await handle_filter_command(message_text)
        return

    if normalized.startswith("/process "):
        await handle_process_command(message_text, force=False)
        return

    if normalized.startswith("/reprocess "):
        await handle_process_command(message_text, force=True)
        return

    if normalized.startswith("/jobs "):
        await handle_jobs_command(message_text)
        return

    if normalized.startswith("/job "):
        await handle_job_command(message_text)
        return

    if normalized.startswith("/delete "):
        await handle_delete_command(message_text)
        return

    if normalized == "/source" or normalized.startswith("/source "):
        await handle_source_command(message_text)
        return

    if normalized == "/chunks" or normalized.startswith("/chunks "):
        await handle_chunks_command(message_text)
        return

    if normalized == "/evidence" or normalized.startswith("/evidence "):
        await handle_evidence_command(message_text)
        return

    if normalized == "/categories":
        await handle_categories_command()
        return

    if normalized.startswith("/category "):
        await handle_category_command(message_text)
        return

    if normalized.startswith("/register "):
        await handle_register_command(message_text)
        return

    if normalized.startswith("/login "):
        await handle_login_command(message_text)
        return

    if normalized == "/google":
        await handle_google_command()
        return

    if normalized == "/token" or normalized.startswith("/token "):
        await handle_token_command(message_text)
        return

    if normalized == "/me":
        await handle_me_command()
        return

    if normalized == "/logout":
        await handle_logout_command()
        return

    if normalized == "/help":
        await handle_help_command()
        return

    if normalized == "/new":
        await handle_new_command()
        return

    await handle_question(message_text)
