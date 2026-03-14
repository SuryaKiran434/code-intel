"""
core/telemetry.py

Query observability log for Code Intel.

Every `ask` invocation writes one row to the query_log table in SQLite.
This gives a full audit trail: what was asked, which repo, retrieval scores,
tokens consumed, latency, and a snippet of the answer.

Usage:
    log_query(
        user_id          = user["id"],
        session_id       = session_id,      # or None
        question         = "How does auth work?",
        repo             = "fastapi",        # or "*" for global search
        query_variants   = 4,               # how many expansion variants ran
        chunks_retrieved = len(chunks),
        top_score        = chunks[0]["score"] if chunks else None,
        tokens_used      = result["tokens_used"],
        latency_ms       = elapsed_ms,
        answer_snippet   = result["answer"][:200],
    )
"""

import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


# ── DB helper ──────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Public API ─────────────────────────────────────────────────────────────────

def log_query(
    user_id:          str | None,
    session_id:       str | None,
    question:         str,
    repo:             str,
    query_variants:   int,
    chunks_retrieved: int,
    top_score:        float | None,
    tokens_used:      int,
    latency_ms:       int,
    answer_snippet:   str,
):
    """
    Append one row to the query_log table.

    Args:
        user_id          — logged-in user ID (None if unauthenticated)
        session_id       — active conversation session ID (None if stateless)
        question         — the user's original question (before expansion)
        repo             — repo name scoped to, or "*" for global search
        query_variants   — number of expansion variants generated (0 if disabled)
        chunks_retrieved — number of chunks returned after re-ranking
        top_score        — reranker score of the best chunk (None if no chunks)
        tokens_used      — total GPT-4o tokens (prompt + completion)
        latency_ms       — wall-clock time from retrieve() call to answer
        answer_snippet   — first 200 chars of the GPT-4o answer
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO query_log (
                user_id, session_id, timestamp, question, repo,
                query_variants, chunks_retrieved, top_score,
                tokens_used, latency_ms, answer_snippet
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, session_id, now, question, repo,
                query_variants, chunks_retrieved, top_score,
                tokens_used, latency_ms, answer_snippet[:200],
            ),
        )


def get_recent_logs(user_id: str, limit: int = 20) -> list[dict]:
    """
    Return the most recent query log entries for a user.

    Returns a list of dicts with keys:
        id, timestamp, question, repo, query_variants, chunks_retrieved,
        top_score, tokens_used, latency_ms, answer_snippet, session_id
    """
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, question, repo, query_variants,
                   chunks_retrieved, top_score, tokens_used, latency_ms,
                   answer_snippet, session_id
            FROM query_log
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
