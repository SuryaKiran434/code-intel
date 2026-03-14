"""
core/session.py

Conversation session management for Code Intel.

A session is a named thread of (user question → assistant answer) turns.
Sessions are persisted in SQLite so they survive terminal restarts.

Usage pattern:
    session_id = create_session(user_id, title="How does auth work?")
    turns      = load_turns(session_id)
    # ... call llm.ask(question, chunks, history=turns) ...
    append_turn(session_id, "user",      question)
    append_turn(session_id, "assistant", answer)
"""

import sqlite3
import uuid
from datetime import datetime, timezone

from config import DB_PATH, SESSION_MAX_TURNS


# ── DB helper ──────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Public API ─────────────────────────────────────────────────────────────────

def create_session(user_id: str, title: str = "") -> str:
    """
    Create a new conversation session.

    Args:
        user_id — ID of the logged-in user
        title   — optional label (defaults to empty; CLI sets first question)

    Returns the new session_id (UUID string).
    """
    session_id = str(uuid.uuid4())
    now        = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO conv_sessions (id, user_id, title, created_at, last_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_id, title, now, now),
        )
    return session_id


def load_turns(session_id: str) -> list[dict]:
    """
    Return the last SESSION_MAX_TURNS turns for a session as a list of
    OpenAI-compatible message dicts: [{role, content}, ...].

    Oldest turns are dropped first when the history is long, keeping the
    most recent exchanges which are most relevant for follow-up questions.
    """
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM (
                SELECT role, content, id FROM conv_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (session_id, SESSION_MAX_TURNS),
        ).fetchall()

    return [{"role": r["role"], "content": r["content"]} for r in rows]


def append_turn(session_id: str, role: str, content: str):
    """
    Append a single turn (user question or assistant answer) to the session.
    Also updates last_used timestamp on the session.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO conv_turns (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        conn.execute(
            "UPDATE conv_sessions SET last_used = ? WHERE id = ?",
            (now, session_id),
        )


def append_turns_batch(session_id: str, user_content: str, assistant_content: str):
    """
    Append both the user question and assistant answer in a single transaction.
    More efficient than two separate append_turn() calls (one DB round-trip).
    """
    _sql = (
        "INSERT INTO conv_turns (session_id, role, content, timestamp)"
        " VALUES (?, ?, ?, ?)"
    )
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(_sql, (session_id, "user", user_content, now))
        conn.execute(_sql, (session_id, "assistant", assistant_content, now))
        conn.execute(
            "UPDATE conv_sessions SET last_used = ? WHERE id = ?",
            (now, session_id),
        )


def get_session(session_id: str) -> dict | None:
    """Return session metadata, or None if not found."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM conv_sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def list_sessions(user_id: str, limit: int = 10) -> list[dict]:
    """Return the most recent sessions for a user."""
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, last_used,
                   (SELECT COUNT(*) FROM conv_turns WHERE session_id = conv_sessions.id) AS turn_count
            FROM conv_sessions
            WHERE user_id = ?
            ORDER BY last_used DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
