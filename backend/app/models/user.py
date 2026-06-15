import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)

    auth_provider = Column(String, nullable=False, default="google")

    provider_user_id = Column(String, nullable=False)

    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, nullable=True, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=True,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
