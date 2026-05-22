from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    automcommit=False,
    autoflush=False,
    bind=engine,
)   

def get_db():
    """Dependency function to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()      