from typing import List

from sentence_transformers import SentenceTransformer


class LocalEmbeddingService:
    """SentenceTransformer-backed embedding service."""
    def __init__(self, model_name: str) -> None:
        """Load the local embedding model by name."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""
        safe_text = text if text and text.strip() else " "

        embedding = self.model.encode(
            safe_text,
            normalize_embeddings=True,
        )

        return embedding.astype(float).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings into vectors."""
        safe_texts = [
            text if text and text.strip() else " "
            for text in texts
        ]

        embeddings = self.model.encode(
            safe_texts,
            normalize_embeddings=True,
        )

        return [
            embedding.astype(float).tolist()
            for embedding in embeddings
        ]