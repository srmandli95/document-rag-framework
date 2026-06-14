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

To exercise real authentication, cookie sessions, and organizations locally:

```bash
make auth-local
```

Open `http://localhost:3000`, choose **Need a local account? Create one**, then
create and select an organization from the account bar. Google login also works
locally when its client credentials and redirect URI are configured in `.env`.

## Organization authentication

Set `DEV_AUTH_DISABLED=false` outside local development. Login and OAuth callbacks
create an `HttpOnly` session cookie; bearer tokens remain supported for API clients.

Users can create or select an organization from the frontend. Documents and RAG
retrieval use the active organization's shared data scope, while chat history remains
private to each user. Organization roles are:

- `admin`: manage members and documents
- `editor`: manage documents
- `member`: search documents and chat

For production, set:

```bash
AUTH_COOKIE_SECURE=true
FRONTEND_URL=https://rag.example.com
JWT_SECRET_KEY=<strong-random-secret>
DEV_AUTH_DISABLED=false
ALLOW_LOCAL_REGISTRATION=false
```

Google OAuth is available through `/auth/google/login`. Enterprise SAML/OIDC and
SCIM can be connected later through a managed identity broker while retaining the
organization and membership model implemented here.
