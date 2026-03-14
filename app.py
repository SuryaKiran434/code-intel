"""
app.py

FastAPI web backend for Code Intel.

Auth strategy:
  - POST /auth/login   → email + password → returns a Bearer token
  - GET  /auth/me      → validates token, returns user info
  - All other routes   → require Authorization: Bearer <token> header

The token is the same UUID stored in the auth_tokens SQLite table.
api_login() is used (not login()) so the CLI's ~/.code-intel/.auth
file is never overwritten when a user logs in through the browser.
"""

import json
import logging
import time

from fastapi import Depends, FastAPI, HTTPException, Security, status

logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, QUERY_EXPANSION_ENABLED, QUERY_EXPANSION_VARIANTS
from core.db import init_db
from core.auth import api_login, get_user_by_token, register as auth_register
from core.retriever import retrieve
from core.llm import ask_stream as llm_ask_stream
from core.session import create_session, load_turns, append_turns_batch, get_session
from core.telemetry import log_query
from core.vector_store import get_or_create_collection

# Initialise SQLite schema on startup (no-op if tables already exist)
init_db()

app = FastAPI(title="Code Intel", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

_bearer = HTTPBearer(auto_error=True)


# ── Auth dependency ─────────────────────────────────────────────────────────────

def _require_user(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> dict:
    """
    FastAPI dependency — validates the Bearer token and returns the user dict.
    Raises HTTP 401 if the token is missing, invalid, or expired.
    """
    user = get_user_by_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ── Request / response models ───────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str
    password: str

class RegisterRequest(BaseModel):
    email:      str
    password:   str
    first_name: str
    last_name:  str

class QueryRequest(BaseModel):
    question:   str
    repo_name:  str | None = None
    session_id: str | None = None


# ── Auth endpoints ──────────────────────────────────────────────────────────────

@app.post("/auth/login")
def auth_login_route(req: LoginRequest):
    """
    Verify email + password and return a Bearer token.
    Store this token in localStorage and send it with every subsequent request.
    """
    try:
        token = api_login(req.email, req.password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e
    return {"token": token}


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
def auth_register_route(req: RegisterRequest):
    """
    Create a new account and return a Bearer token (auto-login after register).
    Raises 400 if the email is already registered or password is too short.
    """
    if len(req.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters.",
        )
    try:
        auth_register(req.email, req.password, req.first_name, req.last_name)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    # Auto-login: create a session token and return it
    token = api_login(req.email, req.password)
    return {"token": token}


@app.get("/auth/me")
def auth_me(user: dict = Depends(_require_user)):
    """Return the current user's profile. Used on page load to restore session."""
    return {
        "id":         user["id"],
        "email":      user["email"],
        "first_name": user["first_name"],
        "last_name":  user["last_name"],
    }


# ── Protected routes ────────────────────────────────────────────────────────────

@app.post("/query")
def query(req: QueryRequest, user: dict = Depends(_require_user)):
    """
    Stream a gpt-4.1 answer as Server-Sent Events (SSE).

    Each SSE event is a JSON object on a `data:` line:
        {"type": "token",  "text": "..."}          — streamed token
        {"type": "done",   "sources": [...],
         "tokens": N, "session_id": "..."}         — final metadata
        {"type": "error",  "message": "..."}       — on failure
    """
    # Session setup happens before the generator so it's always available
    session_id = req.session_id
    if session_id and not get_session(session_id):
        session_id = None
    if not session_id:
        session_id = create_session(user["id"], title=req.question[:120])

    history = load_turns(session_id)

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def _generate():
        t0 = time.monotonic()

        try:
            chunks = retrieve(req.question, repo_name=req.repo_name)
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            return

        if not chunks:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            msg = "No relevant code found for your question."
            append_turns_batch(session_id, req.question, msg)
            log_query(
                user_id=user["id"], session_id=session_id,
                question=req.question, repo=req.repo_name or "*",
                query_variants=QUERY_EXPANSION_VARIANTS if QUERY_EXPANSION_ENABLED else 0,
                chunks_retrieved=0, top_score=None,
                tokens_used=0, latency_ms=elapsed_ms, answer_snippet=msg,
            )
            yield _sse({"type": "token", "text": msg})
            yield _sse({"type": "done", "sources": [], "tokens": 0, "session_id": session_id})
            return

        # Stream LLM tokens — accumulate for session + telemetry writes at end
        full_answer = ""
        sources     = []
        tokens_used = 0

        try:
            for event in llm_ask_stream(req.question, chunks, history=history):
                if event["type"] == "token":
                    full_answer += event["text"]
                    yield _sse(event)
                elif event["type"] == "done":
                    sources     = event["sources"]
                    tokens_used = event["tokens"]
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            return

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        append_turns_batch(session_id, req.question, full_answer)
        log_query(
            user_id=user["id"], session_id=session_id,
            question=req.question, repo=req.repo_name or "*",
            query_variants=QUERY_EXPANSION_VARIANTS if QUERY_EXPANSION_ENABLED else 0,
            chunks_retrieved=len(chunks),
            top_score=chunks[0]["score"] if chunks else None,
            tokens_used=tokens_used, latency_ms=elapsed_ms,
            answer_snippet=full_answer[:200],
        )

        clean_sources = [
            {
                "label":  s["label"],
                "file":   s["file"],
                "symbol": s["symbol"],
                "lines":  s["lines"],
                "score":  round(s["score"], 3),
            }
            for s in sources
        ]
        yield _sse({
            "type":       "done",
            "sources":    clean_sources,
            "tokens":     tokens_used,
            "session_id": session_id,
        })

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_repos_cache: dict = {"data": None, "ts": 0.0}
_REPOS_CACHE_TTL = 60.0   # seconds


@app.get("/repos")
def list_repos(user: dict = Depends(_require_user)):
    """Return the names of all indexed repositories (cached for 60 s)."""
    now = time.monotonic()
    if _repos_cache["data"] is not None and now - _repos_cache["ts"] < _REPOS_CACHE_TTL:
        return _repos_cache["data"]
    try:
        col = get_or_create_collection()
        col.load()
        results = col.query(
            expr='chunk_type == "full"',
            output_fields=["repo_name"],
            limit=16384,
        )
        repos = sorted({r["repo_name"] for r in results if r.get("repo_name")})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    result = {"repos": repos}
    _repos_cache["data"] = result
    _repos_cache["ts"] = now
    return result


# ── Static / root ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store"})
