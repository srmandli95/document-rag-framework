from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

import app.api.document_routes as document_routes
from app.embeddings.local_embedder import LocalEmbeddingService
from app.ingestion.embedding_indexing_service import embed_document_chunks
from app.main import app


client = TestClient(app)


class FakeDB:
    def query(self, model):
        return self

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def all(self):
        return []

    def first(self):
        return None

    def add(self, obj):
        return None

    def commit(self):
        return None

    def refresh(self, obj):
        return None


@dataclass
class FakeDocument:
    id: str
    user_id: str
    status: str


@dataclass
class FakeChunk:
    id: str
    document_id: str
    user_id: str
    chunk_text: str
    chunk_index: int
    token_count: int = 10
    status: str = "created"
    embedding: list[float] | None = None


@pytest.mark.slow
def test_local_embedding_service_returns_list_of_floats():
    service = LocalEmbeddingService("sentence-transformers/all-MiniLM-L6-v2")

    embedding = service.embed_text("This is a sample health policy chunk.")

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)


@pytest.mark.slow
def test_local_embedding_dimension_is_384():
    service = LocalEmbeddingService("sentence-transformers/all-MiniLM-L6-v2")

    embedding = service.embed_text("Embedding dimension test.")

    assert len(embedding) == 384


@pytest.mark.slow
def test_embed_texts_returns_same_number_of_embeddings_as_input_texts():
    service = LocalEmbeddingService("sentence-transformers/all-MiniLM-L6-v2")

    texts = [
        "First chunk text.",
        "Second chunk text.",
        "Third chunk text.",
    ]

    embeddings = service.embed_texts(texts)

    assert len(embeddings) == len(texts)
    assert all(len(embedding) == 384 for embedding in embeddings)


@pytest.mark.slow
def test_empty_text_is_handled_safely():
    service = LocalEmbeddingService("sentence-transformers/all-MiniLM-L6-v2")

    embedding = service.embed_text("")

    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_embed_endpoint_success_without_real_db_calls(monkeypatch):
    fake_document = FakeDocument(
        id="doc-123",
        user_id="user-123",
        status="chunked",
    )

    def fake_get_db():
        yield FakeDB()

    def fake_get_document_by_id(db, document_id, user_id):
        return fake_document

    def fake_embed_document_chunks(db, document):
        return {
            "document_id": document.id,
            "user_id": document.user_id,
            "status": "embedded",
            "embedded_chunk_count": 2,
            "embedding_provider": "local",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "message": "Document chunks embedded successfully.",
        }

    app.dependency_overrides[document_routes.get_db] = fake_get_db

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    monkeypatch.setattr(
        document_routes,
        "embed_document_chunks",
        fake_embed_document_chunks,
    )

    response = client.post("/documents/doc-123/embed?user_id=user-123")

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] == "doc-123"
    assert data["user_id"] == "user-123"
    assert data["status"] == "embedded"
    assert data["embedded_chunk_count"] == 2
    assert data["embedding_provider"] == "local"
    assert data["embedding_model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert data["message"] == "Document chunks embedded successfully."


def test_embed_endpoint_invalid_document_id_returns_404(monkeypatch):
    def fake_get_db():
        yield FakeDB()

    def fake_get_document_by_id(db, document_id, user_id):
        return None

    app.dependency_overrides[document_routes.get_db] = fake_get_db

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    response = client.post("/documents/missing-doc/embed?user_id=user-123")

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_user_a_cannot_embed_user_b_document(monkeypatch):
    def fake_get_db():
        yield FakeDB()

    def fake_get_document_by_id(db, document_id, user_id):
        return None

    app.dependency_overrides[document_routes.get_db] = fake_get_db

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    response = client.post("/documents/doc-user-b/embed?user_id=user-a")

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_document_not_in_chunked_status_returns_400(monkeypatch):
    fake_document = FakeDocument(
        id="doc-123",
        user_id="user-123",
        status="extracted",
    )

    def fake_get_db():
        yield FakeDB()

    def fake_get_document_by_id(db, document_id, user_id):
        return fake_document

    app.dependency_overrides[document_routes.get_db] = fake_get_db

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    response = client.post("/documents/doc-123/embed?user_id=user-123")

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "chunked status" in response.json()["detail"]


def test_embed_document_chunks_updates_statuses(monkeypatch):
    fake_db = FakeDB()

    fake_document = FakeDocument(
        id="doc-123",
        user_id="user-123",
        status="chunked",
    )

    fake_chunks = [
        FakeChunk(
            id="chunk-1",
            document_id="doc-123",
            user_id="user-123",
            chunk_text="First chunk",
            chunk_index=0,
        ),
        FakeChunk(
            id="chunk-2",
            document_id="doc-123",
            user_id="user-123",
            chunk_text="Second chunk",
            chunk_index=1,
        ),
    ]

    updated_document_statuses = []

    class FakeEmbeddingService:
        def embed_texts(self, texts):
            return [
                [0.1] * 384,
                [0.2] * 384,
            ]

    def fake_get_created_chunks_by_document(db, document_id, user_id):
        return fake_chunks

    def fake_get_embedding_service():
        return FakeEmbeddingService()

    def fake_update_chunks_with_embeddings(db, chunks, embeddings):
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk.status = "embedded"

        return chunks

    def fake_update_document_status(db, document_id, status):
        updated_document_statuses.append(status)
        fake_document.status = status
        return fake_document

    import app.ingestion.embedding_indexing_service as service_module

    monkeypatch.setattr(
        service_module,
        "get_created_chunks_by_document",
        fake_get_created_chunks_by_document,
    )

    monkeypatch.setattr(
        service_module,
        "get_embedding_service",
        fake_get_embedding_service,
    )

    monkeypatch.setattr(
        service_module,
        "update_chunks_with_embeddings",
        fake_update_chunks_with_embeddings,
    )

    monkeypatch.setattr(
        service_module,
        "update_document_status",
        fake_update_document_status,
    )

    result = embed_document_chunks(
        db=fake_db,
        document=fake_document,
    )

    assert result["document_id"] == "doc-123"
    assert result["user_id"] == "user-123"
    assert result["status"] == "embedded"
    assert result["embedded_chunk_count"] == 2
    assert result["embedding_provider"] == "local"
    assert result["embedding_model_name"] == "sentence-transformers/all-MiniLM-L6-v2"

    assert updated_document_statuses == ["processing", "embedded"]
    assert all(chunk.status == "embedded" for chunk in fake_chunks)
    assert all(chunk.embedding is not None for chunk in fake_chunks)
    assert all(len(chunk.embedding) == 384 for chunk in fake_chunks)


def test_failed_embedding_marks_document_status_as_failed(monkeypatch):
    fake_db = FakeDB()

    fake_document = FakeDocument(
        id="doc-123",
        user_id="user-123",
        status="chunked",
    )

    fake_chunks = [
        FakeChunk(
            id="chunk-1",
            document_id="doc-123",
            user_id="user-123",
            chunk_text="First chunk",
            chunk_index=0,
        )
    ]

    updated_document_statuses = []

    class FailingEmbeddingService:
        def embed_texts(self, texts):
            raise RuntimeError("Embedding failed")

    def fake_get_created_chunks_by_document(db, document_id, user_id):
        return fake_chunks

    def fake_get_embedding_service():
        return FailingEmbeddingService()

    def fake_update_document_status(db, document_id, status):
        updated_document_statuses.append(status)
        fake_document.status = status
        return fake_document

    import app.ingestion.embedding_indexing_service as service_module

    monkeypatch.setattr(
        service_module,
        "get_created_chunks_by_document",
        fake_get_created_chunks_by_document,
    )

    monkeypatch.setattr(
        service_module,
        "get_embedding_service",
        fake_get_embedding_service,
    )

    monkeypatch.setattr(
        service_module,
        "update_document_status",
        fake_update_document_status,
    )

    with pytest.raises(RuntimeError):
        embed_document_chunks(
            db=fake_db,
            document=fake_document,
        )

    assert updated_document_statuses == ["processing", "failed"]
    assert fake_document.status == "failed"