"""
tests/test_query_expander.py

Tests for core/query_expander.py — two-level query expansion cache.

Covers:
  - L1 in-process cache hit (no DB or API call)
  - L2 SQLite cache hit (no API call, promotes to L1)
  - API call path: result saved to both L1 and L2
  - API failure: returns [] and is NOT cached (so next call retries)
  - Hash determinism
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clear_l1(qe, *queries):
    """Remove given queries from the in-process L1 cache."""
    for q in queries:
        qe._expansion_cache.pop(q, None)


# ── Hash determinism ───────────────────────────────────────────────────────────

def test_query_hash_is_deterministic():
    from core.query_expander import _query_hash
    assert _query_hash("hello world") == _query_hash("hello world")


def test_query_hash_length():
    from core.query_expander import _query_hash
    assert len(_query_hash("any query string")) == 32


def test_different_queries_produce_different_hashes():
    from core.query_expander import _query_hash
    assert _query_hash("query one") != _query_hash("query two")


# ── L1 cache hit ───────────────────────────────────────────────────────────────

def test_l1_cache_hit_skips_api(tmp_db, monkeypatch):
    """When the query is already in the L1 cache, the API is never called."""
    import core.query_expander as qe
    query = "l1_cache_test_query_unique_abc"
    qe._expansion_cache[query] = ("variant_x", "variant_y")

    with patch.object(qe._client.chat.completions, "create") as mock_api:
        result = qe.expand_query(query)
        mock_api.assert_not_called()

    assert "variant_x" in result
    _clear_l1(qe, query)


def test_l1_cache_hit_skips_db(tmp_db, monkeypatch):
    """L1 hit does not touch SQLite at all."""
    import core.query_expander as qe
    query = "l1_db_skip_test_unique_xyz"
    qe._expansion_cache[query] = ("v1",)

    with patch("core.query_expander._load_from_db") as mock_db:
        qe.expand_query(query)
        mock_db.assert_not_called()

    _clear_l1(qe, query)


# ── L2 cache hit ───────────────────────────────────────────────────────────────

def test_l2_cache_hit_skips_api(tmp_db):
    """SQLite cache hit skips the API call entirely."""
    import core.query_expander as qe
    query = "l2_cache_test_unique_mnop"
    stored = ["router dispatch", "URL pattern matching"]
    qe._save_to_db(qe._query_hash(query), stored)
    _clear_l1(qe, query)

    with patch.object(qe._client.chat.completions, "create") as mock_api:
        result = qe.expand_query(query)
        mock_api.assert_not_called()

    assert result == stored
    _clear_l1(qe, query)


def test_l2_cache_hit_promotes_to_l1(tmp_db):
    """After an L2 hit the result is stored in L1 so the next call is faster."""
    import core.query_expander as qe
    query = "l2_promote_test_unique_qrst"
    stored = ["embed search", "vector lookup"]
    qe._save_to_db(qe._query_hash(query), stored)
    _clear_l1(qe, query)

    qe.expand_query(query)

    assert query in qe._expansion_cache
    _clear_l1(qe, query)


# ── API call path ──────────────────────────────────────────────────────────────

def test_api_call_saves_to_l1_and_l2(tmp_db):
    """On full cache miss the API is called and variants are written to both caches."""
    import core.query_expander as qe
    query = "api_call_test_unique_uvwx"
    variants = ["authentication pipeline", "login validation logic"]
    _clear_l1(qe, query)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(variants)

    with patch.object(qe._client.chat.completions, "create", return_value=mock_response):
        result = qe.expand_query(query)

    assert result == variants
    # L1 populated
    assert query in qe._expansion_cache
    # L2 populated
    loaded = qe._load_from_db(qe._query_hash(query))
    assert loaded == variants

    _clear_l1(qe, query)


def test_api_call_filters_out_original_query(tmp_db):
    """The original query string should be excluded from the returned variants."""
    import core.query_expander as qe
    query = "filter_original_test_unique"
    # API returns the original query mixed in
    mock_variants = [query, "real variant one", "real variant two"]
    _clear_l1(qe, query)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(mock_variants)

    with patch.object(qe._client.chat.completions, "create", return_value=mock_response):
        result = qe.expand_query(query)

    assert query not in result
    assert "real variant one" in result

    _clear_l1(qe, query)


# ── API failure path ───────────────────────────────────────────────────────────

def test_api_failure_returns_empty(tmp_db):
    """Any API exception should return [] without raising."""
    import core.query_expander as qe
    query = "failure_test_unique_aaaa"
    _clear_l1(qe, query)

    with patch.object(qe._client.chat.completions, "create", side_effect=Exception("timeout")):
        result = qe.expand_query(query)

    assert result == []


def test_api_failure_not_cached_in_l1(tmp_db):
    """Failed API calls must NOT be written to L1 so the next call retries."""
    import core.query_expander as qe
    query = "failure_l1_skip_unique_bbbb"
    _clear_l1(qe, query)

    with patch.object(qe._client.chat.completions, "create", side_effect=Exception("err")):
        qe.expand_query(query)

    assert query not in qe._expansion_cache


def test_api_failure_not_cached_in_l2(tmp_db):
    """Failed API calls must NOT be written to L2 so the next call retries."""
    import core.query_expander as qe
    query = "failure_l2_skip_unique_cccc"
    _clear_l1(qe, query)

    with patch.object(qe._client.chat.completions, "create", side_effect=Exception("err")):
        qe.expand_query(query)

    loaded = qe._load_from_db(qe._query_hash(query))
    assert loaded is None


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def test_save_and_load_from_db(tmp_db):
    """Direct _save_to_db / _load_from_db round-trip."""
    from core.query_expander import _save_to_db, _load_from_db, _query_hash
    qhash = _query_hash("round_trip_test")
    variants = ["option one", "option two"]
    _save_to_db(qhash, variants)
    loaded = _load_from_db(qhash)
    assert loaded == variants


def test_load_from_db_miss_returns_none(tmp_db):
    """Cache miss should return None, not raise."""
    from core.query_expander import _load_from_db
    result = _load_from_db("nonexistent_hash_aaaa_bbbb_cccc_dddd")
    assert result is None
