from typing import Any

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client, OAuth2Client

from app.config.settings import settings


GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_SCOPES = "openid email profile"


class GoogleOAuthError(Exception):
    """Raised when Google OAuth configuration or provider requests fail."""


def get_google_oauth_configured() -> bool:
    return bool(
        settings.GOOGLE_CLIENT_ID
        and settings.GOOGLE_CLIENT_SECRET
        and settings.GOOGLE_REDIRECT_URI
    )


def _require_google_oauth_configured() -> None:
    if not get_google_oauth_configured():
        raise GoogleOAuthError(
            "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, "
            "GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
        )


def build_google_authorization_url(state: str) -> str:
    _require_google_oauth_configured()

    with OAuth2Client(
        client_id=settings.GOOGLE_CLIENT_ID,
        redirect_uri=settings.GOOGLE_REDIRECT_URI,
        scope=GOOGLE_SCOPES,
    ) as client:
        authorization_url, _ = client.create_authorization_url(
            GOOGLE_AUTHORIZATION_ENDPOINT,
            state=state,
            access_type="offline",
            prompt="select_account",
        )
    return authorization_url


async def exchange_google_code_for_token(code: str) -> dict[str, Any]:
    _require_google_oauth_configured()

    try:
        async with AsyncOAuth2Client(
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET,
            redirect_uri=settings.GOOGLE_REDIRECT_URI,
        ) as client:
            token = await client.fetch_token(
                GOOGLE_TOKEN_ENDPOINT,
                code=code,
                grant_type="authorization_code",
            )
    except (httpx.HTTPError, ValueError) as exc:
        raise GoogleOAuthError(f"Google token exchange failed: {exc}") from exc
    except Exception as exc:
        raise GoogleOAuthError("Google token exchange failed") from exc

    if not token.get("access_token"):
        raise GoogleOAuthError("Google token response did not include an access token")

    return dict(token)


async def fetch_google_userinfo(access_token: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                GOOGLE_USERINFO_ENDPOINT,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            userinfo = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise GoogleOAuthError(f"Google user info request failed: {exc}") from exc

    if not isinstance(userinfo, dict):
        raise GoogleOAuthError("Google user info response was invalid")

    return userinfo


def validate_google_userinfo(userinfo: dict[str, Any]) -> dict[str, str | None]:
    provider_user_id = userinfo.get("id") or userinfo.get("sub")
    email = userinfo.get("email")

    if not provider_user_id:
        raise ValueError("Google user info is missing the user ID")

    if not email or not str(email).strip():
        raise ValueError("Google user info is missing the email address")

    if userinfo.get("verified_email") is False:
        raise ValueError("Google email address is not verified")

    full_name = userinfo.get("name")

    return {
        "provider_user_id": str(provider_user_id),
        "email": str(email).strip().lower(),
        "full_name": str(full_name).strip() if full_name else None,
    }
