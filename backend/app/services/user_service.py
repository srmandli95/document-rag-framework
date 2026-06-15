from sqlalchemy.orm import Session

from app.models.user import User


VALID_AUTH_PROVIDERS = {"google"}


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_user_by_email(db: Session, email: str) -> User | None:
    normalized_email = normalize_email(email)

    return (
        db.query(User)
        .filter(User.email == normalized_email)
        .first()
    )


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )


def get_or_create_oauth_user(
    db: Session,
    provider: str,
    provider_user_id: str,
    email: str,
    full_name: str | None = None,
) -> User:
    normalized_provider = provider.strip().lower()
    normalized_email = normalize_email(email)

    if normalized_provider not in VALID_AUTH_PROVIDERS:
        raise ValueError(f"Unsupported auth provider: {provider}")

    existing_provider_user = (
        db.query(User)
        .filter(
            User.auth_provider == normalized_provider,
            User.provider_user_id == provider_user_id,
        )
        .first()
    )

    if existing_provider_user is not None:
        return existing_provider_user

    existing_email_user = get_user_by_email(db, normalized_email)
    if existing_email_user is not None:
        existing_email_user.auth_provider = normalized_provider
        existing_email_user.provider_user_id = provider_user_id
        existing_email_user.full_name = full_name or existing_email_user.full_name
        existing_email_user.is_active = True
        db.commit()
        db.refresh(existing_email_user)
        return existing_email_user

    user = User(
        email=normalized_email,
        full_name=full_name,
        auth_provider=normalized_provider,
        provider_user_id=provider_user_id,
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user
