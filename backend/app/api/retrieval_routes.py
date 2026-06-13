from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.retrieval.bm25_retriever import bm25_search
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.retrieval_diagnostics import diagnose_retrieval
from app.retrieval.retrieval_settings import (
    RetrievalSettings,
    validate_retrieval_settings,
)
from app.retrieval.vector_retriever import vector_search
from app.reranking.reranking_service import rerank_hybrid_results
from app.schemas.retrieval_schema import (
    BM25SearchRequest,
    BM25SearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    RerankSearchRequest,
    RerankSearchResponse,
    RetrievalDiagnosticsRequest,
    RetrievalDiagnosticsResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)


router = APIRouter(prefix="/search", tags=["Retrieval"])


def _validate_query(query: str | None) -> str:
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )

    return query.strip()


def _safe_positive_top_k(value: int, default: int, maximum: int) -> int:
    if value <= 0:
        return default

    return min(value, maximum)


@router.post("/vector", response_model=VectorSearchResponse)
def vector_search_endpoint(
    request: VectorSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VectorSearchResponse:
    """
    Vector search scoped to the authenticated user.

    Day 20 authorization behavior:
    - Requires JWT.
    - Ignores request.user_id if older clients still send it.
    - Uses current_user.id as the real user_id.
    """
    user_id = str(current_user.id)
    query = _validate_query(request.query)
    top_k = _safe_positive_top_k(request.top_k, default=5, maximum=20)

    try:
        results = vector_search(
            db=db,
            user_id=user_id,
            query=query,
            top_k=top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return VectorSearchResponse(
        user_id=user_id,
        query=query,
        top_k=top_k,
        result_count=len(results),
        results=results,
    )


@router.post("/bm25", response_model=BM25SearchResponse)
def search_bm25_chunks(
    request: BM25SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> BM25SearchResponse:
    """
    BM25 search scoped to the authenticated user.

    Day 20 authorization behavior:
    - Requires JWT.
    - Ignores request.user_id if older clients still send it.
    - Uses current_user.id as the real user_id.
    """
    user_id = str(current_user.id)
    query = _validate_query(request.query)
    top_k = _safe_positive_top_k(request.top_k, default=5, maximum=20)

    try:
        results = bm25_search(
            db=db,
            user_id=user_id,
            query=query,
            top_k=top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return BM25SearchResponse(
        user_id=user_id,
        query=query,
        top_k=top_k,
        result_count=len(results),
        results=results,
    )


@router.post("/hybrid", response_model=HybridSearchResponse)
def search_hybrid_chunks(
    request: HybridSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HybridSearchResponse:
    """
    Hybrid search scoped to the authenticated user.

    Day 20 authorization behavior:
    - Requires JWT.
    - Ignores request.user_id if older clients still send it.
    - Uses current_user.id as the real user_id.
    """
    user_id = str(current_user.id)
    query = _validate_query(request.query)

    try:
        retrieval_settings = validate_retrieval_settings(
            RetrievalSettings(
                top_k=request.top_k,
                vector_top_k=request.vector_top_k,
                bm25_top_k=request.bm25_top_k,
                vector_weight=request.vector_weight,
                bm25_weight=request.bm25_weight,
            )
        )
        results = hybrid_search(
            db=db,
            user_id=user_id,
            query=query,
            top_k=retrieval_settings.top_k,
            vector_top_k=retrieval_settings.vector_top_k,
            bm25_top_k=retrieval_settings.bm25_top_k,
            vector_weight=retrieval_settings.vector_weight,
            bm25_weight=retrieval_settings.bm25_weight,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return HybridSearchResponse(
        user_id=user_id,
        query=query,
        top_k=retrieval_settings.top_k,
        result_count=len(results),
        results=results,
    )


@router.post("/rerank", response_model=RerankSearchResponse)
def search_reranked_chunks(
    request: RerankSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RerankSearchResponse:
    """
    Reranked hybrid search scoped to the authenticated user.

    Day 20 authorization behavior:
    - Requires JWT.
    - Ignores request.user_id if older clients still send it.
    - Uses current_user.id as the real user_id.
    """
    user_id = str(current_user.id)
    query = _validate_query(request.query)

    try:
        retrieval_settings = validate_retrieval_settings(
            RetrievalSettings(
                hybrid_top_k=request.hybrid_top_k,
                vector_top_k=request.vector_top_k,
                bm25_top_k=request.bm25_top_k,
                rerank_top_k=request.top_k,
                vector_weight=request.vector_weight,
                bm25_weight=request.bm25_weight,
            )
        )
        results = rerank_hybrid_results(
            db=db,
            user_id=user_id,
            query=query,
            top_k=retrieval_settings.rerank_top_k,
            hybrid_top_k=retrieval_settings.hybrid_top_k,
            vector_top_k=retrieval_settings.vector_top_k,
            bm25_top_k=retrieval_settings.bm25_top_k,
            vector_weight=retrieval_settings.vector_weight,
            bm25_weight=retrieval_settings.bm25_weight,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return RerankSearchResponse(
        user_id=user_id,
        query=query,
        top_k=retrieval_settings.rerank_top_k,
        result_count=len(results),
        results=results,
    )


@router.post("/diagnose", response_model=RetrievalDiagnosticsResponse)
def diagnose_retrieval_endpoint(
    request: RetrievalDiagnosticsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RetrievalDiagnosticsResponse:
    user_id = str(current_user.id)
    query = _validate_query(request.query)
    try:
        retrieval_settings = validate_retrieval_settings(
            RetrievalSettings(
                vector_top_k=request.vector_top_k,
                bm25_top_k=request.bm25_top_k,
                hybrid_top_k=request.hybrid_top_k,
                rerank_top_k=request.rerank_top_k,
                vector_weight=request.vector_weight,
                bm25_weight=request.bm25_weight,
            )
        )
        diagnostics = diagnose_retrieval(
            db=db,
            user_id=user_id,
            query=query,
            vector_top_k=retrieval_settings.vector_top_k,
            bm25_top_k=retrieval_settings.bm25_top_k,
            hybrid_top_k=retrieval_settings.hybrid_top_k,
            rerank_top_k=retrieval_settings.rerank_top_k,
            vector_weight=retrieval_settings.vector_weight,
            bm25_weight=retrieval_settings.bm25_weight,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return RetrievalDiagnosticsResponse.model_validate(diagnostics)
