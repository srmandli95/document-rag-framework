from app.config.settings import settings
from app.embeddings.local_embedder import LocalEmbeddingService


def get_embedding_service() -> LocalEmbeddingService:
    """Return the configured embedding service implementation."""
    provider = settings.EMBEDDING_PROVIDER.lower().strip()

    if provider == "local":
        return LocalEmbeddingService(settings.EMBEDDING_MODEL_NAME)

    raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")