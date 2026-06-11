from sqlalchemy.orm import Session

from app.auth.password_utils import hash_password, verify_password
from app.models.user import User


VALID_AUTH_PROVIDERS = {"local", "google", "microsoft"}


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


def create_local_user(
    db: Session,
    email: str,
    password: str,
    full_name: str | None = None,
) -> User:
    normalized_email = normalize_email(email)

    existing_user = get_user_by_email(db, normalized_email)
    if existing_user is not None:
        raise ValueError("User with this email already exists")

    user = User(
        email=normalized_email,
        full_name=full_name,
        hashed_password=hash_password(password),
        auth_provider="local",
        provider_user_id=None,
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def authenticate_local_user(
    db: Session,
    email: str,
    password: str,
) -> User | None:
    user = get_user_by_email(db, email)

    if user is None:
        return None

    if user.auth_provider != "local":
        return None

    if not verify_password(password, user.hashed_password):
        return None

    if not user.is_active:
        return None

    return user


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

    if normalized_provider == "local":
        raise ValueError("OAuth user provider cannot be local")

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
        return existing_email_user

    user = User(
        email=normalized_email,
        full_name=full_name,
        hashed_password=None,
        auth_provider=normalized_provider,
        provider_user_id=provider_user_id,
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user