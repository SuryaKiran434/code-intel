"""
tests/test_diff_tracker.py

Tests for core/diff_tracker.py — incremental sync state management.

Tests here are pure unit tests: no Milvus, no git, no external services.
Covers:
  - Sync state persistence (load / save / update / clear)
  - First-run: empty state returns None for any repo
  - Corrupted state file: graceful recovery
  - _is_supported(): Python files pass, others fail
"""

import json
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def dt(tmp_path, monkeypatch):
    """
    Import diff_tracker with SYNC_STATE_PATH pointing to a temp file.
    Returns the patched module so tests can call its functions directly.
    """
    import core.diff_tracker as _dt
    state_file = tmp_path / "sync_state.json"
    monkeypatch.setattr(_dt, "SYNC_STATE_PATH", state_file)
    return _dt


# ── _load_sync_state ───────────────────────────────────────────────────────────

def test_load_returns_empty_when_no_file(dt):
    assert dt._load_sync_state() == {}


def test_load_returns_empty_after_corrupted_file(tmp_path, monkeypatch):
    import core.diff_tracker as _dt
    state_file = tmp_path / "bad_sync.json"
    state_file.write_text("{invalid json{{")
    monkeypatch.setattr(_dt, "SYNC_STATE_PATH", state_file)
    result = _dt._load_sync_state()
    assert result == {}


# ── _save_sync_state / _load_sync_state ───────────────────────────────────────

def test_save_and_reload_state(dt):
    dt._save_sync_state({"repo_alpha": "abc123", "repo_beta": "def456"})
    loaded = dt._load_sync_state()
    assert loaded == {"repo_alpha": "abc123", "repo_beta": "def456"}


def test_save_creates_parent_directories(tmp_path, monkeypatch):
    import core.diff_tracker as _dt
    nested = tmp_path / "a" / "b" / "sync.json"
    monkeypatch.setattr(_dt, "SYNC_STATE_PATH", nested)
    _dt._save_sync_state({"repo": "sha"})
    assert nested.exists()


def test_save_is_atomic(dt, tmp_path, monkeypatch):
    """The tmp file should not linger after a successful save."""
    import core.diff_tracker as _dt
    state_file = tmp_path / "sync2.json"
    monkeypatch.setattr(_dt, "SYNC_STATE_PATH", state_file)
    _dt._save_sync_state({"repo": "sha"})
    tmp_file = state_file.with_suffix(".tmp")
    assert not tmp_file.exists()


# ── get_last_synced_commit ─────────────────────────────────────────────────────

def test_get_last_synced_commit_returns_none_first_run(dt):
    assert dt.get_last_synced_commit("brand_new_repo") is None


def test_get_last_synced_commit_returns_correct_sha(dt):
    dt._save_sync_state({"myrepo": "feedcafe1234"})
    assert dt.get_last_synced_commit("myrepo") == "feedcafe1234"


def test_get_last_synced_commit_ignores_other_repos(dt):
    dt._save_sync_state({"repo_a": "aaa", "repo_b": "bbb"})
    assert dt.get_last_synced_commit("repo_a") == "aaa"
    assert dt.get_last_synced_commit("repo_b") == "bbb"


# ── _update_synced_commit ──────────────────────────────────────────────────────

def test_update_synced_commit_persists(dt):
    dt._update_synced_commit("myrepo", "deadbeef")
    assert dt.get_last_synced_commit("myrepo") == "deadbeef"


def test_update_synced_commit_overwrites(dt):
    dt._update_synced_commit("myrepo", "first_sha")
    dt._update_synced_commit("myrepo", "second_sha")
    assert dt.get_last_synced_commit("myrepo") == "second_sha"


def test_update_synced_commit_preserves_other_repos(dt):
    dt._save_sync_state({"other_repo": "other_sha"})
    dt._update_synced_commit("myrepo", "my_sha")
    assert dt.get_last_synced_commit("other_repo") == "other_sha"


# ── _clear_sync_state ─────────────────────────────────────────────────────────

def test_clear_sync_state_removes_repo(dt):
    dt._save_sync_state({"repo_a": "aaa", "repo_b": "bbb"})
    dt._clear_sync_state("repo_a")
    state = dt._load_sync_state()
    assert "repo_a" not in state
    assert "repo_b" in state


def test_clear_sync_state_noop_for_missing_repo(dt):
    dt._save_sync_state({"repo_a": "aaa"})
    # Should not raise
    dt._clear_sync_state("nonexistent_repo")
    state = dt._load_sync_state()
    assert "repo_a" in state


# ── _is_supported ──────────────────────────────────────────────────────────────

def test_is_supported_python():
    from core.diff_tracker import _is_supported
    assert _is_supported("core/chunker.py") is True
    assert _is_supported("/abs/path/to/main.py") is True


def test_is_supported_rejects_non_python():
    from core.diff_tracker import _is_supported
    assert _is_supported("README.md") is False
    assert _is_supported("config.yaml") is False
    assert _is_supported("image.png") is False
    assert _is_supported("Makefile") is False


def test_is_supported_rejects_no_extension():
    from core.diff_tracker import _is_supported
    assert _is_supported("Dockerfile") is False
