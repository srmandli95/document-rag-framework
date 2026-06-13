# Document RAG
A reusable document RAG application for uploading knowledge sources and asking citation-backed questions.

## Architecture

React UI -> FastAPI Backend -> LangGraph RAG Workflow -> PostgreSQL + pgvector -> LLM

## Local development

Copy `.env.example` to `.env`, then run:

```bash
docker compose up --build
```

Open `http://localhost:3000`. The baseline frontend uses backend development-auth mode.
