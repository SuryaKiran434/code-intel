"""
core/query_expander.py

Generates semantic query variants using GPT-4o-mini to broaden retrieval
coverage. A single user query may miss relevant chunks due to vocabulary
mismatch between the question and how the code is written. Expanding to
N variants and searching each independently dramatically increases recall.

Cache hierarchy:
    L1 — in-process dict  (fastest; reset on process restart)
    L2 — SQLite           (persisted across restarts; shared between CLI + web)

Example:
    "how is auth handled"
    → ["JWT token validation logic",
       "authentication middleware implementation",
       "bearer token parsing and verification"]
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from openai import OpenAI

from config import OPENAI_API_KEY, QUERY_EXPANSION_MODEL, QUERY_EXPANSION_VARIANTS, DB_PATH

_client = OpenAI(api_key=OPENAI_API_KEY)

# L1 cache — in-process dict; values are tuples (immutable) to prevent mutation.
# Failures are NOT cached so transient errors retry next time.
_expansion_cache: dict[str, tuple[str, ...]] = {}

_EXPANSION_PROMPT = """\
You are a search query rewriter for a code intelligence tool that searches \
source code using semantic embeddings.

Given a developer's question about a codebase, generate {n} alternative search \
queries that capture the same intent using different terminology, focusing on \
the technical concepts, function names, patterns, or implementation details \
that are likely to appear in source code.

Rules:
- Each variant must express the same underlying intent as the original
- Use coding terminology: function names, class names, patterns, framework terms
- Vary vocabulary significantly — do not just rephrase word-by-word
- Output ONLY a JSON array of strings, no explanation, no markdown

Original query: {query}

Output (JSON array of {n} strings):"""


# ── SQLite L2 cache helpers ────────────────────────────────────────────────────

def _query_hash(query: str) -> str:
    """32-char hex SHA-256 of the query — used as the SQLite primary key."""
    return hashlib.sha256(query.encode()).hexdigest()[:32]


def _load_from_db(query_hash: str) -> list[str] | None:
    """Return cached variants from SQLite, or None on miss / error."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT variants FROM query_expansion_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _save_to_db(query_hash: str, variants: list[str]) -> None:
    """Persist variants to SQLite. Silently swallows errors (non-critical)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO query_expansion_cache
                   (query_hash, variants, created_at) VALUES (?, ?, ?)""",
                (query_hash, json.dumps(variants),
                 datetime.now(timezone.utc).isoformat()),
            )
    except Exception:
        pass


# ── Public API ─────────────────────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """
    Generate QUERY_EXPANSION_VARIANTS alternative phrasings of the query.

    Returns a list of variant strings (does NOT include the original query —
    the caller combines with the original).

    Cache order: L1 (in-process dict) → L2 (SQLite) → API call.
    Successful results are written to both caches.
    Failures are NOT cached so transient API errors retry on the next call.
    On any failure, returns [] so the caller falls back to single-query retrieval.
    """
    # L1 hit
    if query in _expansion_cache:
        return list(_expansion_cache[query])

    # L2 hit
    qhash = _query_hash(query)
    cached = _load_from_db(qhash)
    if cached:
        _expansion_cache[query] = tuple(cached)   # promote to L1
        return cached

    # API call
    try:
        response = _client.chat.completions.create(
            model=QUERY_EXPANSION_MODEL,
            max_tokens=256,
            temperature=0.7,    # higher temp = more diverse variants
            messages=[
                {
                    "role": "user",
                    "content": _EXPANSION_PROMPT.format(
                        n=QUERY_EXPANSION_VARIANTS,
                        query=query,
                    ),
                }
            ],
        )
        raw = response.choices[0].message.content.strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            result = [str(v) for v in variants if v and str(v) != query]
            if result:
                _expansion_cache[query] = tuple(result)   # L1
                _save_to_db(qhash, result)                # L2
            return result
    except Exception:
        # Silently degrade to single-query retrieval on any failure.
        # Not cached — next call will retry the API.
        pass
    return []
