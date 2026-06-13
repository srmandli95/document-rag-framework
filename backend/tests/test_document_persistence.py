from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


fake_documents = {}


def make_fake_document(
    *,
    document_id=None,
    user_id,
    original_file_name,
    stored_file_name,
    category,
    content_type,
    file_size_bytes,
    storage_provider,
    storage_path,
    status="uploaded",
):
    return SimpleNamespace(
        id=document_id or str(uuid4()),
        user_id=user_id,
        original_file_name=original_file_name,
        stored_file_name=stored_file_name,
        category=category,
        content_type=content_type,
        file_size_bytes=file_size_bytes,
        storage_provider=storage_provider,
        storage_path=storage_path,
        status=status,
        created_at=None,
        updated_at=None,
    )


def fake_create_document_record(
    db,
    *,
    user_id,
    original_file_name,
    stored_file_name,
    category,
    content_type,
    file_size_bytes,
    storage_provider,
    storage_path,
    status="uploaded",
):
    document = make_fake_document(
        user_id=user_id,
        original_file_name=original_file_name,
        stored_file_name=stored_file_name,
        category=category,
        content_type=content_type,
        file_size_bytes=file_size_bytes,
        storage_provider=storage_provider,
        storage_path=storage_path,
        status=status,
    )

    fake_documents[document.id] = document

    return document


def fake_get_documents_by_user(
    db,
    *,
    user_id,
    status=None,
    category=None,
    search=None,
    ready_only=False,
):
    documents = [
        document
        for document in fake_documents.values()
        if document.user_id == user_id and document.status != "deleted"
    ]

    if status:
        documents = [document for document in documents if document.status == status]
    if category:
        documents = [document for document in documents if document.category == category]
    if search:
        documents = [
            document
            for document in documents
            if search.lower() in document.original_file_name.lower()
        ]
    if ready_only:
        documents = [
            document for document in documents if document.status == "embedded"
        ]

    return documents


def fake_get_document_by_id(db, *, document_id, user_id):
    document = fake_documents.get(document_id)

    if document is None:
        return None

    if document.user_id != user_id:
        return None

    if document.status == "deleted":
        return None

    return document


def fake_delete_document_completely(db, *, document_id, user_id):
    document = fake_get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    if document is None:
        return None

    fake_documents.pop(document.id)
    document.status = "deleted"

    return document


def patch_document_db_functions(monkeypatch):
    fake_documents.clear()
    monkeypatch.setattr(
        "app.config.settings.settings.RAW_DOCUMENTS_DIR",
        "/tmp/test_document_persistence_raw",
    )

    monkeypatch.setattr(
        "app.api.document_routes.create_document_record",
        fake_create_document_record,
    )

    monkeypatch.setattr(
        "app.api.document_routes.get_documents_by_user",
        fake_get_documents_by_user,
    )

    monkeypatch.setattr(
        "app.api.document_routes.get_document_by_id",
        fake_get_document_by_id,
    )

    monkeypatch.setattr(
        "app.api.document_routes.delete_document_completely",
        fake_delete_document_completely,
    )
    monkeypatch.setattr(
        "app.api.document_routes.create_processing_job",
        lambda **kwargs: SimpleNamespace(
            id="job-1", status="pending", current_step=None, error_message=None
        ),
    )
    monkeypatch.setattr(
        "app.api.document_routes._process_document_in_background", lambda *args: None
    )


def test_upload_valid_document_creates_fake_db_record(monkeypatch):
    patch_document_db_functions(monkeypatch)

    sample_file = Path("sample_knowledge_base/sample_health_policy.txt")

    response = client.post(
        "/documents/upload",
        data={
            "user_id": "user-a",
            "category": "health_insurance",
        },
        files={
            "file": (
                "sample_health_policy.txt",
                b"This is a synthetic sample health policy document.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] is not None
    assert data["user_id"] == "user-a"
    assert data["file_name"] == "sample_health_policy.txt"
    assert data["original_file_name"] == "sample_health_policy.txt"
    assert data["content_type"] == "text/plain"
    assert data["category"] == "health_insurance"
    assert data["storage_provider"] == "local"
    assert data["status"] == "uploaded"
    assert data["job_id"] == "job-1"
    assert data["message"] == "Document uploaded and processing started"

    saved_file_path = Path(data["storage_path"])
    assert saved_file_path.exists()


def test_list_documents_by_user_id(monkeypatch):
    patch_document_db_functions(monkeypatch)

    upload_response = client.post(
        "/documents/upload",
        data={
            "user_id": "user-a",
            "category": "health_insurance",
        },
        files={
            "file": (
                "sample_health_policy.txt",
                b"This is a sample health policy.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    response = client.get("/documents?user_id=user-a")

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "user-a"
    assert "documents" in data
    assert data["count"] == 1
    assert data["documents"][0]["user_id"] == "user-a"
    assert data["documents"][0]["category"] == "health_insurance"


def test_get_document_by_document_id_and_user_id(monkeypatch):
    patch_document_db_functions(monkeypatch)

    upload_response = client.post(
        "/documents/upload",
        data={
            "user_id": "user-get",
            "category": "health_insurance",
        },
        files={
            "file": (
                "sample_health_policy.txt",
                b"This is a test health policy.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    get_response = client.get(
        f"/documents/{document_id}?user_id=user-get"
    )

    assert get_response.status_code == 200

    data = get_response.json()

    assert data["document_id"] == document_id
    assert data["user_id"] == "user-get"
    assert data["category"] == "health_insurance"
    assert data["status"] == "uploaded"


def test_user_a_cannot_access_user_b_document(monkeypatch):
    patch_document_db_functions(monkeypatch)

    upload_response = client.post(
        "/documents/upload",
        data={
            "user_id": "user-b",
            "category": "mortgage",
        },
        files={
            "file": (
                "sample_policy.txt",
                b"This is user B document.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    response = client.get(
        f"/documents/{document_id}?user_id=user-a"
    )

    assert response.status_code == 404


def test_complete_delete_document(monkeypatch):
    patch_document_db_functions(monkeypatch)

    upload_response = client.post(
        "/documents/upload",
        data={
            "user_id": "delete-user",
            "category": "wifi_policy",
        },
        files={
            "file": (
                "wifi_policy.txt",
                b"This is a wifi policy.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    delete_response = client.delete(
        f"/documents/{document_id}?user_id=delete-user"
    )

    assert delete_response.status_code == 200

    data = delete_response.json()

    assert data["document_id"] == document_id
    assert data["user_id"] == "delete-user"
    assert data["status"] == "deleted"
    assert data["message"] == "Document deleted successfully"


def test_deleted_document_does_not_appear_in_list(monkeypatch):
    patch_document_db_functions(monkeypatch)

    upload_response = client.post(
        "/documents/upload",
        data={
            "user_id": "deleted-list-user",
            "category": "employer_policy",
        },
        files={
            "file": (
                "employer_policy.txt",
                b"This is an employer policy.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    delete_response = client.delete(
        f"/documents/{document_id}?user_id=deleted-list-user"
    )

    assert delete_response.status_code == 200

    list_response = client.get(
        "/documents?user_id=deleted-list-user"
    )

    assert list_response.status_code == 200

    documents = list_response.json()["documents"]

    document_ids = [
        document["document_id"]
        for document in documents
    ]

    assert document_id not in document_ids
