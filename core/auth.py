"""
core/auth.py

User authentication for Code Intel.

Strategy:
  - Credentials stored in SQLite (DB_PATH) — no external auth service needed
  - Passwords hashed with PBKDF2-HMAC-SHA256 + random salt (stdlib only)
  - Login produces a UUID token persisted to AUTH_FILE (~/.code-intel/.auth)
  - All subsequent CLI invocations call get_current_user() to check the token

Tables owned by this module (created via init_db in core/db.py):
  users       — email, hashed password, first/last name
  auth_tokens — UUID tokens with expiry, linked to user
"""

import hashlib
import hmac
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import DB_PATH, AUTH_FILE, AUTH_TOKEN_EXPIRY_DAYS


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Password hashing ───────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 with 260,000 iterations (OWASP 2023 recommendation)."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260_000,
    )
    return dk.hex()


def _new_salt() -> str:
    return os.urandom(32).hex()


# ── Auth file (persists login across terminal sessions) ────────────────────────

def _save_auth(token: str, user: dict):
    """Write login token + user info to AUTH_FILE."""
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(
        json.dumps({
            "token":      token,
            "user_id":    user["id"],
            "email":      user["email"],
            "first_name": user["first_name"],
            "last_name":  user["last_name"],
        }, indent=2)
    )
    AUTH_FILE.chmod(0o600)   # owner read/write only


def _load_auth() -> dict | None:
    """Load the persisted auth file. Returns None if missing or malformed."""
    if not AUTH_FILE.exists():
        return None
    try:
        return json.loads(AUTH_FILE.read_text())
    except (json.JSONDecodeError, KeyError):
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def register(
    email: str,
    password: str,
    first_name: str,
    last_name: str,
) -> dict:
    """
    Create a new user account.

    Returns the new user dict on success.
    Raises ValueError if email is already registered.
    """
    email = email.strip().lower()
    salt  = _new_salt()
    pw_hash = _hash_password(password, salt)
    user_id = str(uuid.uuid4())
    now     = datetime.now(timezone.utc).isoformat()

    with _conn() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (id, email, password_hash, salt, first_name, last_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, email, pw_hash, salt, first_name.strip(), last_name.strip(), now),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"An account with '{email}' already exists.")

    return {
        "id": user_id, "email": email,
        "first_name": first_name.strip(), "last_name": last_name.strip(),
    }


def login(email: str, password: str) -> str:
    """
    Verify credentials, create a token, persist to AUTH_FILE.

    Returns the token string on success.
    Raises ValueError on bad credentials.
    """
    email = email.strip().lower()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

    if not row:
        raise ValueError("Invalid email or password.")

    expected = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected, row["password_hash"]):
        raise ValueError("Invalid email or password.")

    token      = str(uuid.uuid4())
    now        = datetime.now(timezone.utc)
    expires_at = (now + timedelta(days=AUTH_TOKEN_EXPIRY_DAYS)).isoformat()

    with _conn() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], now.isoformat(), expires_at),
        )

    user = dict(row)
    _save_auth(token, user)
    return token


def logout():
    """Delete the local auth file. Does not invalidate the DB token."""
    if AUTH_FILE.exists():
        AUTH_FILE.unlink()


def get_current_user() -> dict | None:
    """
    Return the logged-in user dict, or None if not logged in / token expired.
    Reads from AUTH_FILE then validates the token against the DB.
    """
    auth = _load_auth()
    if not auth:
        return None

    with _conn() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, u.first_name, u.last_name, t.expires_at
            FROM auth_tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token = ?
            """,
            (auth["token"],),
        ).fetchone()

    if not row:
        return None

    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        logout()
        return None

    return dict(row)


def has_any_users() -> bool:
    """Return True if at least one user account exists."""
    with _conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return count > 0


def api_login(email: str, password: str) -> str:
    """
    Verify credentials and create a token — web API variant.

    Identical to login() but does NOT write ~/.code-intel/.auth,
    so it never clobbers an active CLI session when you log in
    through the browser at the same time.

    Returns the token string on success.
    Raises ValueError on bad credentials.
    """
    email = email.strip().lower()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

    if not row:
        raise ValueError("Invalid email or password.")

    expected = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected, row["password_hash"]):
        raise ValueError("Invalid email or password.")

    token      = str(uuid.uuid4())
    now        = datetime.now(timezone.utc)
    expires_at = (now + timedelta(days=AUTH_TOKEN_EXPIRY_DAYS)).isoformat()

    with _conn() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], now.isoformat(), expires_at),
        )

    return token


def get_user_by_token(token: str) -> dict | None:
    """
    Validate a raw token string and return the user dict.
    Used by the web API to authenticate Bearer tokens from HTTP headers.
    Does not touch the .auth file.
    """
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, u.first_name, u.last_name, t.expires_at
            FROM auth_tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token = ?
            """,
            (token,),
        ).fetchone()

    if not row:
        return None

    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        return None

    return dict(row)
