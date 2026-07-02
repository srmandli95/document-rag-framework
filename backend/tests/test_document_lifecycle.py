from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import document_routes
from app.auth.dependencies import get_current_user
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_processing_job import DocumentProcessingJob
from app.repositories.document_repository import delete_document_completely


class UploadDB:
    pass


def test_upload_creates_job_and_starts_processing(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(document_routes.router)
    document = SimpleNamespace(
        id="doc-1",
        user_id="user-1",
        original_file_name="guide.txt",
        stored_file_name="guide.txt",
        category="general",
        content_type="text/plain",
        file_size_bytes=5,
        storage_provider="local",
        storage_path=str(tmp_path / "guide.txt"),
        status="uploaded",
        created_at=None,
        updated_at=None,
    )
    job = SimpleNamespace(
        id="job-1",
        status="pending",
        current_step=None,
        error_message=None,
    )
    processed = []

    app.dependency_overrides[document_routes.get_db] = lambda: UploadDB()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="user-1")
    monkeypatch.setattr(document_routes, "create_document_record", lambda **kwargs: document)
    monkeypatch.setattr(document_routes, "create_processing_job", lambda **kwargs: job)
    monkeypatch.setattr(
        document_routes,
        "_process_document_in_background",
        lambda *args: processed.append(args),
    )
    monkeypatch.setattr(
        document_routes.LocalStorageService,
        "save_file",
        lambda *args, **kwargs: {
            "storage_provider": "local",
            "storage_path": str(tmp_path / "guide.txt"),
            "file_name": "guide.txt",
            "file_size_bytes": 5,
        },
    )

    response = TestClient(app).post(
        "/documents/upload",
        data={"category": "general"},
        files={"file": ("guide.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-1"
    assert response.json()["job_id"] == "job-1"
    assert response.json()["display_status"] == "uploading"
    assert len(processed) == 1


def test_failed_document_metadata_exposes_useful_error():
    document = SimpleNamespace(
        id="doc-1",
        user_id="user-1",
        original_file_name="bad.txt",
        stored_file_name="bad.txt",
        category="general",
        content_type="text/plain",
        file_size_bytes=5,
        storage_provider="local",
        storage_path="/tmp/bad.txt",
        status="failed",
        created_at=None,
        updated_at=None,
    )
    job = SimpleNamespace(status="failed", current_step=None, error_message="Extraction failed")

    metadata = document_routes._to_document_metadata(document, job)

    assert metadata.display_status == "failed"
    assert metadata.failure_reason == "Extraction failed"


class DeleteQuery:
    def __init__(self, db, model):
        self.db = db
        self.model = model

    def filter(self, *conditions):
        return self

    def first(self):
        return self.db.document if self.model is Document else None

    def delete(self, synchronize_session=False):
        self.db.deleted_models.append(self.model)
        return 1


class DeleteDB:
    def __init__(self, document):
        self.document = document
        self.deleted_models = []
        self.deleted_document = None
        self.committed = False
        self.rolled_back = False

    def query(self, model):
        return DeleteQuery(self, model)

    def delete(self, document):
        self.deleted_document = document

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_complete_delete_removes_rows_jobs_vectors_and_artifacts(monkeypatch, tmp_path):
    raw_file = tmp_path / "raw" / "user-1" / "upload-id" / "guide.txt"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("source")
    document = SimpleNamespace(
        id="doc-1",
        user_id="user-1",
        storage_path=str(raw_file),
        status="embedded",
    )
    db = DeleteDB(document)

    for setting_name in ("EXTRACTED_TEXT_DIR", "PROCESSED_CHUNKS_DIR", "REDACTED_DOCUMENTS_DIR"):
        root = tmp_path / setting_name.lower()
        artifact = root / "user-1" / "doc-1" / "artifact.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("generated")
        monkeypatch.setattr(f"app.repositories.document_repository.settings.{setting_name}", str(root))

    deleted = delete_document_completely(db, document_id="doc-1", user_id="user-1")

    assert deleted.status == "deleted"
    assert db.deleted_models == [DocumentChunk, DocumentProcessingJob]
    assert db.deleted_document is document
    assert db.committed is True
    assert not raw_file.parent.exists()
    assert not (tmp_path / "extracted_text_dir" / "user-1" / "doc-1").exists()


def test_complete_delete_returns_none_for_unowned_document():
    db = DeleteDB(None)
    assert delete_document_completely(db, document_id="doc-1", user_id="other-user") is None
    assert db.committed is False
