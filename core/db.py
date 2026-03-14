"""
core/db.py

SQLite schema initialisation for Code Intel.

Call init_db() once at startup (cli.py does this). All tables are created
with IF NOT EXISTS so it is safe to call on every run.

Tables:
    users         — user accounts (email, hashed password, name)
    auth_tokens   — persistent login tokens with expiry
    conv_sessions — named conversation threads per user
    conv_turns    — individual messages within a session
    query_log     — audit log of every ask invocation
"""

import sqlite3
from config import DB_PATH


def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                first_name    TEXT NOT NULL,
                last_name     TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_tokens (
                token      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS conv_sessions (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                title      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                last_used  TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS conv_turns (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES conv_sessions(id)
            );

            CREATE TABLE IF NOT EXISTS query_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          TEXT,
                session_id       TEXT,
                timestamp        TEXT NOT NULL,
                question         TEXT NOT NULL,
                repo             TEXT NOT NULL,
                query_variants   INTEGER NOT NULL DEFAULT 0,
                chunks_retrieved INTEGER NOT NULL DEFAULT 0,
                top_score        REAL,
                tokens_used      INTEGER NOT NULL DEFAULT 0,
                latency_ms       INTEGER NOT NULL DEFAULT 0,
                answer_snippet   TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (user_id)   REFERENCES users(id),
                FOREIGN KEY (session_id) REFERENCES conv_sessions(id)
            );

            -- ── Query expansion cache ─────────────────────────────────────────
            -- Persists GPT-4o-mini expansion results across process restarts.
            -- Keyed by SHA-256 of the original query string (32-char hex).
            CREATE TABLE IF NOT EXISTS query_expansion_cache (
                query_hash TEXT PRIMARY KEY,
                variants   TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- ── Import / call graph ───────────────────────────────────────────
            -- import_edges: which symbols each file imports from which modules.
            -- call_edges:   which functions/methods each symbol calls.
            CREATE TABLE IF NOT EXISTS import_edges (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_name  TEXT NOT NULL,
                from_file  TEXT NOT NULL,
                kind       TEXT NOT NULL,
                to_module  TEXT NOT NULL,
                to_symbol  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS call_edges (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_name    TEXT NOT NULL,
                from_file    TEXT NOT NULL,
                from_symbol  TEXT NOT NULL,
                to_symbol    TEXT NOT NULL,
                kind         TEXT NOT NULL DEFAULT 'call'
            );

            -- Indexes for frequent query patterns
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id
                ON auth_tokens(user_id);

            CREATE INDEX IF NOT EXISTS idx_conv_sessions_user_last
                ON conv_sessions(user_id, last_used DESC);

            CREATE INDEX IF NOT EXISTS idx_conv_turns_session_id
                ON conv_turns(session_id);

            CREATE INDEX IF NOT EXISTS idx_query_log_user_id
                ON query_log(user_id);

            CREATE INDEX IF NOT EXISTS idx_query_log_user_timestamp
                ON query_log(user_id, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_import_edges_from_file
                ON import_edges(repo_name, from_file);

            CREATE INDEX IF NOT EXISTS idx_call_edges_from
                ON call_edges(repo_name, from_file, from_symbol);

            CREATE INDEX IF NOT EXISTS idx_call_edges_to
                ON call_edges(repo_name, to_symbol);
        """)
