from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ingestion.document_processing_service import process_document
from app.api import document_routes


class FakeDB:
    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self.refreshed = []

    def add(self, obj):
        pass

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def refresh(self, obj):
        self.refreshed.append(obj)


def make_document(status: str = "uploaded"):
    return SimpleNamespace(
        id="doc-123",
        user_id="local-user-123",
        status=status,
        original_file_name="sample.txt",
    )


def test_process_document_calls_extract_chunk_embed_in_order(monkeypatch):
    db = FakeDB()
    document = make_document(status="uploaded")
    calls = []

    def fake_extract(db_arg, document_arg):
        calls.append("extract")
        document_arg.status = "extracted"

    def fake_chunk(db_arg, document_arg):
        calls.append("chunk")
        document_arg.status = "chunked"

    def fake_embed(db_arg, document_arg):
        calls.append("embed")
        document_arg.status = "embedded"

    monkeypatch.setattr(
        "app.ingestion.document_processing_service.extract_and_store_document_text",
        fake_extract,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.chunk_and_store_document_text",
        fake_chunk,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.embed_document_chunks",
        fake_embed,
    )

    result = process_document(db=db, document=document)

    assert calls == ["extract", "chunk", "embed"]
    assert result["status"] == "embedded"
    assert result["message"] == "Document processed successfully."
    assert [step["name"] for step in result["steps"]] == ["extract", "chunk", "embed"]
    assert all(step["status"] == "completed" for step in result["steps"])


def test_process_document_returns_embedded_status_when_all_steps_succeed(monkeypatch):
    db = FakeDB()
    document = make_document(status="uploaded")

    def fake_extract(db_arg, document_arg):
        document_arg.status = "extracted"

    def fake_chunk(db_arg, document_arg):
        document_arg.status = "chunked"

    def fake_embed(db_arg, document_arg):
        document_arg.status = "embedded"

    monkeypatch.setattr(
        "app.ingestion.document_processing_service.extract_and_store_document_text",
        fake_extract,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.chunk_and_store_document_text",
        fake_chunk,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.embed_document_chunks",
        fake_embed,
    )

    result = process_document(db=db, document=document)

    assert result["document_id"] == "doc-123"
    assert result["user_id"] == "local-user-123"
    assert result["status"] == "embedded"
    assert len(result["steps"]) == 3


def test_process_document_stops_if_extraction_fails(monkeypatch):
    db = FakeDB()
    document = make_document(status="uploaded")
    calls = []

    def fake_extract(db_arg, document_arg):
        calls.append("extract")
        raise RuntimeError("extract failed")

    def fake_chunk(db_arg, document_arg):
        calls.append("chunk")

    def fake_embed(db_arg, document_arg):
        calls.append("embed")

    monkeypatch.setattr(
        "app.ingestion.document_processing_service.extract_and_store_document_text",
        fake_extract,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.chunk_and_store_document_text",
        fake_chunk,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.embed_document_chunks",
        fake_embed,
    )

    result = process_document(db=db, document=document)

    assert calls == ["extract"]
    assert result["status"] == "failed"
    assert result["steps"][0]["name"] == "extract"
    assert result["steps"][0]["status"] == "failed"
    assert document.status == "failed"


def test_process_document_stops_if_chunking_fails(monkeypatch):
    db = FakeDB()
    document = make_document(status="uploaded")
    calls = []

    def fake_extract(db_arg, document_arg):
        calls.append("extract")
        document_arg.status = "extracted"

    def fake_chunk(db_arg, document_arg):
        calls.append("chunk")
        raise RuntimeError("chunk failed")

    def fake_embed(db_arg, document_arg):
        calls.append("embed")

    monkeypatch.setattr(
        "app.ingestion.document_processing_service.extract_and_store_document_text",
        fake_extract,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.chunk_and_store_document_text",
        fake_chunk,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.embed_document_chunks",
        fake_embed,
    )

    result = process_document(db=db, document=document)

    assert calls == ["extract", "chunk"]
    assert result["status"] == "failed"
    assert result["steps"][-1]["name"] == "chunk"
    assert result["steps"][-1]["status"] == "failed"
    assert document.status == "failed"


def test_process_document_stops_if_embedding_fails(monkeypatch):
    db = FakeDB()
    document = make_document(status="uploaded")
    calls = []

    def fake_extract(db_arg, document_arg):
        calls.append("extract")
        document_arg.status = "extracted"

    def fake_chunk(db_arg, document_arg):
        calls.append("chunk")
        document_arg.status = "chunked"

    def fake_embed(db_arg, document_arg):
        calls.append("embed")
        raise RuntimeError("embed failed")

    monkeypatch.setattr(
        "app.ingestion.document_processing_service.extract_and_store_document_text",
        fake_extract,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.chunk_and_store_document_text",
        fake_chunk,
    )
    monkeypatch.setattr(
        "app.ingestion.document_processing_service.embed_document_chunks",
        fake_embed,
    )

    result = process_document(db=db, document=document)

    assert calls == ["extract", "chunk", "embed"]
    assert result["status"] == "failed"
    assert result["steps"][-1]["name"] == "embed"
    assert result["steps"][-1]["status"] == "failed"
    assert document.status == "failed"


def test_process_document_returns_already_processed_for_embedded_document():
    db = FakeDB()
    document = make_document(status="embedded")

    result = process_document(db=db, document=document)

    assert result["status"] == "embedded"
    assert result["steps"] == []
    assert result["message"] == "Document is already processed."


def test_process_document_rejects_deleted_document():
    db = FakeDB()
    document = make_document(status="deleted")

    result = process_document(db=db, document=document)

    assert result["status"] == "deleted"
    assert result["steps"] == []
    assert result["message"] == "Deleted documents cannot be processed."


def _make_test_client(monkeypatch, fake_document):
    app = FastAPI()
    app.include_router(document_routes.router)

    def fake_get_db():
        yield FakeDB()

    def fake_get_document_by_id(db, document_id, user_id):
        return fake_document

    def fake_process_document(db, document):
        return {
            "document_id": str(document.id),
            "user_id": document.user_id,
            "status": "embedded",
            "steps": [
                {
                    "name": "extract",
                    "status": "completed",
                    "message": "Document text extracted successfully.",
                },
                {
                    "name": "chunk",
                    "status": "completed",
                    "message": "Document text chunked successfully.",
                },
                {
                    "name": "embed",
                    "status": "completed",
                    "message": "Document chunks embedded successfully.",
                },
            ],
            "message": "Document processed successfully.",
        }

    app.dependency_overrides[document_routes.get_db] = fake_get_db

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )
    monkeypatch.setattr(
        document_routes,
        "process_document",
        fake_process_document,
    )

    return TestClient(app)


def test_process_endpoint_returns_404_for_invalid_document_id(monkeypatch):
    client = _make_test_client(monkeypatch, fake_document=None)

    response = client.post(
        "/documents/missing-doc/process",
        params={"user_id": "local-user-123"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_process_endpoint_returns_404_when_user_does_not_own_document(monkeypatch):
    client = _make_test_client(monkeypatch, fake_document=None)

    response = client.post(
        "/documents/doc-owned-by-other-user/process",
        params={"user_id": "local-user-123"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_process_endpoint_returns_already_processed_message(monkeypatch):
    document = make_document(status="embedded")
    client = _make_test_client(monkeypatch, fake_document=document)

    response = client.post(
        "/documents/doc-123/process",
        params={"user_id": "local-user-123"},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["document_id"] == "doc-123"
    assert payload["user_id"] == "local-user-123"
    assert payload["status"] == "embedded"
    assert payload["steps"] == []
    assert payload["message"] == "Document is already processed."


def test_process_endpoint_success(monkeypatch):
    document = make_document(status="uploaded")
    client = _make_test_client(monkeypatch, fake_document=document)

    response = client.post(
        "/documents/doc-123/process",
        params={"user_id": "local-user-123"},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "embedded"
    assert payload["message"] == "Document processed successfully."
    assert [step["name"] for step in payload["steps"]] == [
        "extract",
        "chunk",
        "embed",
    ]