from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health_routes import router as health_router
from app.api.document_routes import router as document_router
from app.config.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Production-grade RAG assistant for personal policy and life documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(document_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Welcome to PersonalPolicyRagAssistant backend",
        "docs": "/docs",
    }
