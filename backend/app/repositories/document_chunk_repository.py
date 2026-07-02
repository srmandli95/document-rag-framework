from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


def create_document_chunks(
    db: Session,
    document_id: str,
    user_id: str,
    chunks: list[dict],
) -> list[DocumentChunk]:
    """Create chunk records for extracted document text."""
    document_chunks: list[DocumentChunk] = []

    for chunk in chunks:
        document_chunk = DocumentChunk(
            document_id=document_id,
            user_id=user_id,
            chunk_text=chunk["chunk_text"],
            search_text=chunk.get("search_text") or chunk["chunk_text"],
            chunk_index=chunk["chunk_index"],
            token_count=chunk["token_count"],
            section_title=chunk.get("section_title"),
            page_number=chunk.get("page_number"),
            summary=chunk.get("summary"),
            keywords=chunk.get("keywords"),
            hypothetical_questions=chunk.get("hypothetical_questions"),
            structure_types=chunk.get("structure_types"),
            status="created",
        )

        db.add(document_chunk)
        document_chunks.append(document_chunk)

    db.commit()

    for document_chunk in document_chunks:
        db.refresh(document_chunk)

    return document_chunks


def get_chunks_by_document(
    db: Session,
    document_id: str,
    user_id: str,
) -> list[DocumentChunk]:
    """Return chunks for a document in chunk order."""
    return get_chunks_by_document_for_user(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )


def get_chunk_by_id(
    db: Session,
    chunk_id: str,
    user_id: str,
) -> DocumentChunk | None:
    """Return one chunk by id."""
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.id == chunk_id)
        .filter(DocumentChunk.user_id == user_id)
        .filter(DocumentChunk.status != "deleted")
        .first()
    )


def get_chunks_by_document_for_user(
    db: Session,
    document_id: str,
    user_id: str,
) -> list[DocumentChunk]:
    """Return chunks for a document owned by a user."""
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .filter(DocumentChunk.user_id == user_id)
        .filter(DocumentChunk.status != "deleted")
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )


def get_created_chunks_by_document(
    db: Session,
    document_id: str,
    user_id: str,
) -> list[DocumentChunk]:
    """Return chunks that are ready to be embedded."""
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .filter(DocumentChunk.user_id == user_id)
        .filter(DocumentChunk.status == "created")
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )


def update_chunk_embedding(
    db: Session,
    chunk_id: str,
    embedding: list[float],
    status: str = "embedded",
) -> DocumentChunk | None:
    """Store an embedding vector on a chunk."""
    chunk = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.id == chunk_id)
        .first()
    )

    if chunk is None:
        return None

    chunk.embedding = embedding
    chunk.status = status

    db.commit()
    db.refresh(chunk)

    return chunk


def update_chunks_with_embeddings(
    db: Session,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
) -> list[DocumentChunk]:
    """Store embedding vectors on a batch of chunks."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count must match.")

    updated_chunks: list[DocumentChunk] = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.status = "embedded"
        updated_chunks.append(chunk)

    db.commit()

    for chunk in updated_chunks:
        db.refresh(chunk)

    return updated_chunks


def delete_chunks_by_document(
    db: Session,
    document_id: str,
    user_id: str,
) -> int:
    """Delete all chunks for a document."""
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .filter(DocumentChunk.user_id == user_id)
        .all()
    )

    deleted_count = 0

    for chunk in chunks:
        chunk.status = "deleted"
        deleted_count += 1

    db.commit()

    return deleted_count
