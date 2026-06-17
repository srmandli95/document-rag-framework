# Document RAG

A reusable document RAG framework for uploading any document and asking citation-backed questions over its contents.

## Architecture

React UI -> FastAPI Backend -> LangGraph RAG Workflow -> PostgreSQL + pgvector -> LLM

## Authentication

Google OAuth is the only user login method.

After Google verifies a user, the backend creates or finds the local user record and
issues an application JWT inside an `HttpOnly` session cookie. Documents, retrieval,
and chat history are scoped to that authenticated user.

## Local development

Copy `.env.example` to `.env` and configure a Google OAuth client:

```bash
GOOGLE_CLIENT_ID=<google-client-id>
GOOGLE_CLIENT_SECRET=<google-client-secret>
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
FRONTEND_URL=http://localhost:3000
AUTH_COOKIE_SECURE=false
JWT_SECRET_KEY=<local-random-secret>
```

In Google Cloud Console, add this authorized redirect URI:

```text
http://localhost:8000/auth/google/callback
```

Start the application:

```bash
docker compose up --build
```

Open `http://localhost:3000` and select **Continue with Google**. After OAuth completes, the app returns you to the document workspace.

## Production

Production uses the same Google OAuth flow. Configure HTTPS URLs and secure cookies:

```bash
GOOGLE_REDIRECT_URI=https://api.example.com/auth/google/callback
FRONTEND_URL=https://rag.example.com
AUTH_COOKIE_SECURE=true
JWT_SECRET_KEY=<strong-random-secret>
```

Add the production callback URL to the Google OAuth client's authorized redirect
URIs. Keep the Google client secret and JWT secret in a secrets manager.
