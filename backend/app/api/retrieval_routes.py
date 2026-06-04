from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.retrieval.vector_retriever import vector_search
from app.retrieval.bm25_retriever import bm25_search
from app.retrieval.hybrid_retriever import hybrid_search
from app.reranking.reranking_service import rerank_hybrid_results
from app.schemas.retrieval_schema import (
    BM25SearchRequest,
    BM25SearchResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    RerankSearchRequest,
    RerankSearchResponse
)

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post("/vector-search", response_model=VectorSearchResponse)
def vector_search_endpoint(
    request: VectorSearchRequest,
    db: Session = Depends(get_db),
) -> VectorSearchResponse:
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )

    top_k = request.top_k

    if top_k > 20:
        top_k = 20

    try:
        results = vector_search(
            db=db,
            user_id=request.user_id,
            query=request.query,
            top_k=top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return VectorSearchResponse(
        user_id=request.user_id.strip(),
        query=request.query.strip(),
        top_k=top_k,
        result_count=len(results),
        results=results,
    )

@router.post("/bm25-search", response_model=BM25SearchResponse)
def search_bm25_chunks(
    request: BM25SearchRequest,
    db: Session = Depends(get_db),
) -> BM25SearchResponse:
    if not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )

    top_k = request.top_k

    if top_k <= 0:
        top_k = 5

    if top_k > 20:
        top_k = 20

    results = bm25_search(
        db=db,
        user_id=request.user_id,
        query=request.query,
        top_k=top_k,
    )

    return BM25SearchResponse(
        user_id=request.user_id,
        query=request.query,
        top_k=top_k,
        result_count=len(results),
        results=results,
    )

@router.post("/hybrid-search", response_model=HybridSearchResponse)
def search_hybrid_chunks(
    request: HybridSearchRequest,
    db: Session = Depends(get_db),
) -> HybridSearchResponse:
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )

    if request.vector_weight < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="vector_weight must be non-negative",
        )

    if request.bm25_weight < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bm25_weight must be non-negative",
        )

    if request.vector_weight == 0 and request.bm25_weight == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one retrieval weight must be greater than 0",
        )

    top_k = min(request.top_k, 20)
    vector_top_k = min(request.vector_top_k, 50)
    bm25_top_k = min(request.bm25_top_k, 50)

    try:
        results = hybrid_search(
            db=db,
            user_id=request.user_id.strip(),
            query=request.query.strip(),
            top_k=top_k,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            vector_weight=request.vector_weight,
            bm25_weight=request.bm25_weight,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return HybridSearchResponse(
        user_id=request.user_id.strip(),
        query=request.query.strip(),
        top_k=top_k,
        result_count=len(results),
        results=results,
    )

@router.post("/rerank-search", response_model=RerankSearchResponse)
def search_reranked_chunks(
    request: RerankSearchRequest,
    db: Session = Depends(get_db),
) -> RerankSearchResponse:
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )

    if request.vector_weight < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="vector_weight must be non-negative",
        )

    if request.bm25_weight < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bm25_weight must be non-negative",
        )

    if request.vector_weight == 0 and request.bm25_weight == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="vector_weight and bm25_weight cannot both be 0",
        )

    safe_top_k = min(request.top_k, 10)
    safe_hybrid_top_k = min(request.hybrid_top_k, 50)
    safe_vector_top_k = min(request.vector_top_k, 50)
    safe_bm25_top_k = min(request.bm25_top_k, 50)

    results = rerank_hybrid_results(
        db=db,
        user_id=request.user_id,
        query=request.query,
        top_k=safe_top_k,
        hybrid_top_k=safe_hybrid_top_k,
        vector_top_k=safe_vector_top_k,
        bm25_top_k=safe_bm25_top_k,
        vector_weight=request.vector_weight,
        bm25_weight=request.bm25_weight,
    )

    return RerankSearchResponse(
        user_id=request.user_id,
        query=request.query,
        top_k=safe_top_k,
        result_count=len(results),
        results=results,
    )