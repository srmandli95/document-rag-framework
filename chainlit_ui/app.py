import chainlit as cl

from services.api_client import (
    APIClientError,
    ask_question,
    delete_document,
    get_current_user,
    get_google_login_url,
    list_documents,
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


LOGIN_REQUIRED_MESSAGE = (
    "Please login using `/login <email> <password>`, register using "
    "`/register <email> <password> [full name]`, or start Google login with `/google`."
)


def _get_access_token() -> str | None:
    return cl.user_session.get("access_token")


def _get_current_category() -> str:
    return cl.user_session.get("current_category") or DEFAULT_CATEGORY


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


def _clear_auth_session() -> None:
    cl.user_session.set("access_token", None)
    cl.user_session.set("current_user", None)
    cl.user_session.set("user_id", None)
    cl.user_session.set("email", None)
    cl.user_session.set("auth_provider", None)
    cl.user_session.set("session_id", None)


async def _require_login() -> bool:
    access_token = _get_access_token()

    if access_token:
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

        lines.append(f"- {name}: {status} — {message}")

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
        return "Failed", "type /reprocess <document_id>"

    if normalized_status == "deleted":
        return "Deleted", "No action available."

    return "Unknown", "type /documents later or try /process <document_id>"


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


def _extract_documents(response: dict | list) -> list[dict]:
    if isinstance(response, list):
        return response

    if isinstance(response, dict):
        documents = response.get("documents")
        if isinstance(documents, list):
            return documents

    return []


@cl.on_chat_start
async def on_chat_start() -> None:
    if cl.user_session.get("session_id") is None:
        cl.user_session.set("session_id", None)

    if cl.user_session.get("current_category") is None:
        cl.user_session.set("current_category", DEFAULT_CATEGORY)

    await cl.Message(
        content=(
            "Personal Policy RAG Assistant is ready.\n\n"
            "Please login or register before uploading documents, processing documents, "
            "listing documents, or asking questions.\n\n"
            "Auth commands:\n"
            "- `/register <email> <password> [full name]`\n"
            "- `/login <email> <password>`\n"
            "- `/google`\n"
            "- `/token <access_token>`\n"
            "- `/me`\n"
            "- `/logout`\n\n"
            "Useful commands:\n"
            "- `/help`\n"
            "- `/categories`\n"
            "- `/category <category_name>`"
        )
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
                    f"File: {file.name}\n"
                    f"Document ID: `{document_id}`\n"
                    f"Category: `{category}`\n"
                    f"Status: `{status}`\n\n"
                    "Next step:\n"
                    f"Type `/process {document_id}` to prepare this document for questions.\n\n"
                    "Other actions:\n"
                    "- `/documents`\n"
                    f"- `/delete {document_id}`\n"
                    "- `/category <category_name>`"
                )
            ).send()

        except APIClientError as exc:
            await cl.Message(content=f"Upload failed: {exc}").send()
        except Exception as exc:
            await cl.Message(content=f"Unexpected upload error: {exc}").send()


async def handle_documents_command() -> None:
    if not await _require_login():
        return

    try:
        response = await list_documents(
            access_token=_get_access_token(),
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
        file_name = (
            document.get("file_name")
            or document.get("filename")
            or document.get("original_file_name")
            or document.get("original_filename")
            or "Unknown file"
        )
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


async def handle_process_command(message_text: str, force: bool = False) -> None:
    if not await _require_login():
        return

    parts = message_text.strip().split(maxsplit=1)
    command_name = "/reprocess" if force else "/process"

    if len(parts) < 2 or not parts[1].strip():
        await cl.Message(content=f"Usage: {command_name} <document_id>").send()
        return

    document_id = parts[1].strip()

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

    title = "Reprocessing result" if force else "Processing result"

    content = (
        f"{title} for document `{document_id}`:\n\n"
        f"{steps_text}\n\n"
        f"Final status: `{final_status}`\n"
        f"Message: {message}"
    )

    if final_status == "embedded":
        content += "\n\nDocument is ready for questions."

    await cl.Message(content=content).send()


async def handle_delete_command(message_text: str) -> None:
    if not await _require_login():
        return

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
            f"{message}\n\n"
            f"Status: `{status}`\n\n"
            "Type `/documents` to verify."
        )
    ).send()


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
            "Available commands:\n\n"
            "Auth commands:\n"
            "- `/register <email> <password> [full name]` — create a local user\n"
            "- `/login <email> <password>` — login\n"
            "- `/google` — start Google OAuth login\n"
            "- `/token <access_token>` — finish OAuth login by saving the app JWT\n"
            "- `/me` — show current authenticated user\n"
            "- `/logout` — logout\n\n"
            "Document commands, login required:\n"
            "- Upload a file in the chat — upload document\n"
            "- `/documents` — list uploaded documents\n"
            "- `/process <document_id>` — extract, chunk, and embed a document\n"
            "- `/reprocess <document_id>` — force reprocess a document\n"
            "- `/delete <document_id>` — soft delete a document\n\n"
            "Category commands:\n"
            "- `/category <category_name>` — set upload category\n"
            "- `/categories` — show supported categories\n\n"
            "Chat commands, login required:\n"
            "- Ask a normal question — run RAG over your documents\n"
            "- `/new` — start a new chat session\n\n"
            "General:\n"
            "- `/help` — show this help message"
        )
    ).send()


async def handle_new_command() -> None:
    if not await _require_login():
        return

    cl.user_session.set("session_id", None)

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

    # Keep this order.
    # /categories must be checked before /category.
    # Auth commands must stay public.
    # Protected commands call _require_login() inside their handlers.

    if normalized == "/documents":
        await handle_documents_command()
        return

    if normalized.startswith("/process "):
        await handle_process_command(message_text, force=False)
        return

    if normalized.startswith("/reprocess "):
        await handle_process_command(message_text, force=True)
        return

    if normalized.startswith("/delete "):
        await handle_delete_command(message_text)
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
