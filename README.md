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
Compose and `make backend` default to `DEV_AUTH_DISABLED=true` for local use, even
when an older `.env` does not include that setting. Set
`DEV_AUTH_DISABLED=false` explicitly when authentication should be required.
