from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.retrieval.vector_retriever import vector_search
from app.schemas.retrieval_schema import (
    VectorSearchRequest,
    VectorSearchResponse,
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