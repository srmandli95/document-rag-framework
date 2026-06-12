from fastapi import APIRouter

from app.config.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str | bool]:
    """Health check endpoint to verify that the API is running."""
    return {
        "status": "ok",
        "service": "personal-policy-rag-assistant",
        "dev_auth_disabled": settings.DEV_AUTH_DISABLED,
    }
