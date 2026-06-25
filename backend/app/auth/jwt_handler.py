from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from app.config.settings import settings


def _sanitize_token_value(value: Any) -> Any:
    """Normalize token values into JSON-serializable primitives."""
    if isinstance(value, dict):
        return {
            key: _sanitize_token_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_token_value(item) for item in value]

    return value


def _sanitize_token_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize all custom token claims before signing."""
    return _sanitize_token_value(data)


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token with an expiration claim."""
    token_data = _sanitize_token_data(data)

    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    token_data.update({"exp": expire})

    return jwt.encode(
        token_data,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except (JWTError, TypeError, ValueError) as exc:
        raise ValueError("Invalid or expired access token") from exc

    if not payload.get("sub"):
        raise ValueError("Invalid or expired access token")

    return payload
