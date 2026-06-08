from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health_routes import router as health_router
from app.api.document_routes import router as document_router
from app.api.retrieval_routes import router as retrieval_router
from app.api.chat_routes import router as chat_router
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.config.settings import settings
from app.db.database import Base, engine
from app.models import Document, DocumentChunk

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Production-grade RAG assistant for personal policy and life documents.",
)

@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(document_router)
app.include_router(retrieval_router)
app.include_router(chat_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Welcome to PersonalPolicyRagAssistant backend",
        "docs": "/docs",
    }
