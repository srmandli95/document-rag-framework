from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint to verify that the API is running."""
    return {
        "status": "ok",
        "service": "personal-policy-rag-assistant",
    }
