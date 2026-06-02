from fastapi.testclient import TestClient

from app.db.database import get_db
from app.main import app


client = TestClient(app)


class FakeDB:
    pass


def override_get_db():
    yield FakeDB()


app.dependency_overrides[get_db] = override_get_db


def test_vector_search_endpoint_success(monkeypatch):
    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "user_id": user_id,
                "chunk_text": "Urgent care visits are covered after copay.",
                "chunk_index": 0,
                "token_count": 8,
                "page_number": None,
                "section_title": "Urgent Care",
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "distance": 0.12,
                "similarity_score": 0.88,
            }
        ]

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/retrieval/vector-search",
        json={
            "user_id": "local-user-123",
            "query": "Does my health insurance cover urgent care?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "local-user-123"
    assert data["query"] == "Does my health insurance cover urgent care?"
    assert data["top_k"] == 5
    assert data["result_count"] == 1
    assert data["results"][0]["chunk_id"] == "chunk-1"
    assert data["results"][0]["document_id"] == "doc-1"
    assert data["results"][0]["user_id"] == "local-user-123"
    assert data["results"][0]["similarity_score"] == 0.88


def test_vector_search_endpoint_empty_user_id_returns_400(monkeypatch):
    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/retrieval/vector-search",
        json={
            "user_id": "   ",
            "query": "Does my health insurance cover urgent care?",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "user_id is required"


def test_vector_search_endpoint_missing_user_id_returns_422():
    response = client.post(
        "/retrieval/vector-search",
        json={
            "query": "Does my health insurance cover urgent care?",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_vector_search_endpoint_empty_query_returns_400(monkeypatch):
    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/retrieval/vector-search",
        json={
            "user_id": "local-user-123",
            "query": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "query is required"


def test_vector_search_endpoint_missing_query_returns_422():
    response = client.post(
        "/retrieval/vector-search",
        json={
            "user_id": "local-user-123",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_vector_search_endpoint_rejects_top_k_greater_than_20():
    response = client.post(
        "/retrieval/vector-search",
        json={
            "user_id": "local-user-123",
            "query": "Does my health insurance cover urgent care?",
            "top_k": 25,
        },
    )

    assert response.status_code == 422


def test_vector_search_endpoint_returns_result_count_correctly(monkeypatch):
    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "user_id": user_id,
                "chunk_text": "First matching chunk.",
                "chunk_index": 0,
                "token_count": 3,
                "page_number": None,
                "section_title": None,
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "distance": 0.10,
                "similarity_score": 0.90,
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-1",
                "user_id": user_id,
                "chunk_text": "Second matching chunk.",
                "chunk_index": 1,
                "token_count": 3,
                "page_number": None,
                "section_title": None,
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "distance": 0.20,
                "similarity_score": 0.80,
            },
        ]

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/retrieval/vector-search",
        json={
            "user_id": "local-user-123",
            "query": "urgent care",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_count"] == 2
    assert len(data["results"]) == 2


def test_vector_search_endpoint_no_results_returns_empty_list(monkeypatch):
    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/retrieval/vector-search",
        json={
            "user_id": "local-user-123",
            "query": "something not found",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_count"] == 0
    assert data["results"] == []


def test_vector_search_endpoint_passes_user_id_query_and_top_k(monkeypatch):
    captured = {}

    def fake_vector_search(db, user_id: str, query: str, top_k: int = 5):
        captured["user_id"] = user_id
        captured["query"] = query
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.vector_search",
        fake_vector_search,
    )

    response = client.post(
        "/retrieval/vector-search",
        json={
            "user_id": "local-user-123",
            "query": "urgent care coverage",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert captured["user_id"] == "local-user-123"
    assert captured["query"] == "urgent care coverage"
    assert captured["top_k"] == 3


def test_vector_search_validates_user_id_directly():
    from app.retrieval.vector_retriever import vector_search

    try:
        vector_search(
            db=FakeDB(),
            user_id="",
            query="urgent care",
            top_k=5,
        )
    except ValueError as exc:
        assert str(exc) == "user_id is required"
    else:
        assert False, "Expected ValueError"


def test_vector_search_validates_query_directly():
    from app.retrieval.vector_retriever import vector_search

    try:
        vector_search(
            db=FakeDB(),
            user_id="local-user-123",
            query="",
            top_k=5,
        )
    except ValueError as exc:
        assert str(exc) == "query is required"
    else:
        assert False, "Expected ValueError"


def test_vector_search_calls_embedding_service_before_db_query(monkeypatch):
    from app.retrieval import vector_retriever

    calls = {
        "embedding_called": False,
        "query_text": None,
    }

    class FakeEmbeddingService:
        def embed_text(self, text: str):
            calls["embedding_called"] = True
            calls["query_text"] = text
            return [0.1] * 384

    class FakeDistanceExpression:
        def label(self, name: str):
            return self

    class FakeEmbeddingColumn:
        def cosine_distance(self, query_embedding):
            return FakeDistanceExpression()

        def isnot(self, value):
            return True

    class FakeDocumentChunk:
        embedding = FakeEmbeddingColumn()
        user_id = "local-user-123"
        status = "embedded"
        document_id = "doc-1"

    class FakeDocument:
        id = "doc-1"
        status = "embedded"

        class original_file_name:
            @staticmethod
            def label(name: str):
                return "document_name"

        class category:
            @staticmethod
            def label(name: str):
                return "category"

    class FakeQuery:
        def join(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def limit(self, *args, **kwargs):
            return self

        def all(self):
            return []

    class FakeDBForSearch:
        def query(self, *args, **kwargs):
            return FakeQuery()

    monkeypatch.setattr(
        vector_retriever,
        "get_embedding_service",
        lambda: FakeEmbeddingService(),
    )

    monkeypatch.setattr(
        vector_retriever,
        "DocumentChunk",
        FakeDocumentChunk,
    )

    monkeypatch.setattr(
        vector_retriever,
        "Document",
        FakeDocument,
    )

    results = vector_retriever.vector_search(
        db=FakeDBForSearch(),
        user_id="local-user-123",
        query="urgent care coverage",
        top_k=5,
    )

    assert calls["embedding_called"] is True
    assert calls["query_text"] == "urgent care coverage"
    assert results == []


app.dependency_overrides.clear()