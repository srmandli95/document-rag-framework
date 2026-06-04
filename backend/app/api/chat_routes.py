from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.generation.answer_generator import generate_answer_from_evidence
from app.schemas.chat_schema import AskRequest, AskResponse


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db),
) -> AskResponse:
    """
    Foundation answer-generation endpoint.

    This endpoint:
    - accepts a user question
    - retrieves reranked evidence chunks
    - builds a grounded prompt
    - calls the configured LLM provider
    - returns answer and citation metadata

    LangGraph orchestration will be added later.
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question is required",
        )

    safe_top_k = min(request.top_k, 8)
    safe_hybrid_top_k = min(request.hybrid_top_k, 50)
    safe_vector_top_k = min(request.vector_top_k, 50)
    safe_bm25_top_k = min(request.bm25_top_k, 50)

    try:
        result = generate_answer_from_evidence(
            db=db,
            user_id=request.user_id,
            question=request.question,
            top_k=safe_top_k,
            hybrid_top_k=safe_hybrid_top_k,
            vector_top_k=safe_vector_top_k,
            bm25_top_k=safe_bm25_top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return AskResponse(**result)