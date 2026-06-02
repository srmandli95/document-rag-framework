from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.embeddings.embedding_service import get_embedding_service


def vector_search(
    db: Session,
    user_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Generate an embedding for the user question and search embedded document chunks
    using pgvector cosine distance.

    This is a foundation/debug retriever only.
    It does not generate final LLM answers.
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")

    if not query or not query.strip():
        raise ValueError("query is required")

    if top_k <= 0:
        top_k = 5

    if top_k > 20:
        top_k = 20

    clean_user_id = user_id.strip()
    clean_query = query.strip()

    embedding_service = get_embedding_service()
    query_embedding = embedding_service.embed_text(clean_query)

    distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)

    rows = (
        db.query(
            DocumentChunk,
            Document.original_file_name.label("document_name"),
            Document.category.label("category"),
            distance_expr.label("distance"),
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .filter(DocumentChunk.user_id == clean_user_id)
        .filter(DocumentChunk.status == "embedded")
        .filter(DocumentChunk.embedding.isnot(None))
        .filter(Document.status != "deleted")
        .order_by(distance_expr)
        .limit(top_k)
        .all()
    )

    results: list[dict] = []

    for row in rows:
        chunk = row.DocumentChunk
        distance = float(row.distance)
        similarity_score = 1 - distance

        results.append(
            {
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "user_id": chunk.user_id,
                "chunk_text": chunk.chunk_text,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "document_name": row.document_name,
                "category": row.category,
                "distance": distance,
                "similarity_score": similarity_score,
            }
        )

    return results