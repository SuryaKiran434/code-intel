"""
tests/test_app_repos_errors.py

Regression tests for the stack-trace exposure on `GET /repos`.

Companion to tests/test_app_query_errors.py. That file covers the two SSE
handlers CodeQL flagged (`py/stack-trace-exposure`, alert #4); this one covers
the third site of the same shape, which CodeQL did not flag and PR #16 called
out as a known leftover:

    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Could not read sync state: {e}") from e

Neither exception has a generic message:

  * `OSError` stringifies as ``[Errno 21] Is a directory:
    '/home/deploy/.code-intel/sync_state.json'`` — the server's absolute path,
    which names the deploy user and the install layout, plus the errno.
  * `json.JSONDecodeError` stringifies as ``Expecting ',' delimiter: line 1
    column 24 (char 23)`` — the parse offset, and for some inputs a slice of
    the file's contents.

`/repos` sits behind `_require_user`, so the audience is authenticated users
rather than the open internet, which is why it is a smaller leak than the
`/query` stream — but every logged-in account still got the filesystem layout
from a route the web UI calls on page load.

The fix mirrors PR #16: `logger.exception(...)` server-side, a fixed generic
string to the client, and the 500 status unchanged.

Fully offline, in the established style — no httpx, no TestClient, no network,
no live services. `list_repos()` is called directly and `SYNC_STATE_PATH` is
pointed at a temp directory.
"""

import ast
import json
import logging
import tempfile
from pathlib import Path

import pytest

# `app` calls init_db() at import time, so the SQLite path is redirected into a
# temp directory for the duration of that import and then put back. Same dance
# as tests/test_app_query_errors.py — whichever module pytest collects first
# performs the real import, the other gets the cached module.
import config
import core.db

_REAL_DB_PATH = config.DB_PATH
_TEST_DB = Path(tempfile.mkdtemp(prefix="code-intel-repos-tests-")) / "code_intel.db"
config.DB_PATH = _TEST_DB
core.db.DB_PATH = _TEST_DB
try:
    import app  # noqa: E402 — must follow the DB_PATH redirect above
finally:
    config.DB_PATH = _REAL_DB_PATH
    core.db.DB_PATH = _REAL_DB_PATH

from fastapi import HTTPException  # noqa: E402 — after the guarded import above


USER = {"id": "user-1", "email": "dev@example.com"}


def call_list_repos():
    """Drive GET /repos directly.

    `list_repos()` is called rather than routed through TestClient because
    httpx is not a dependency of this project and requirements.txt stays the
    pinned runtime set — the same reason test_app_query_errors.py calls
    `query()` directly.
    """
    return app.list_repos(user=USER)


# ── Unreadable sync-state file ──────────────────────────────────────────────────

def test_os_error_does_not_leak_the_sync_state_path(monkeypatch, tmp_path):
    """A real OSError, not a fake one: reading a directory raises IsADirectoryError
    whose message embeds the absolute path we point it at."""
    secret_dir = tmp_path / "home" / "deploy" / ".code-intel"
    secret_dir.mkdir(parents=True)
    monkeypatch.setattr(app, "SYNC_STATE_PATH", secret_dir)

    with pytest.raises(HTTPException) as excinfo:
        call_list_repos()

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == app._GENERIC_REPOS_ERROR

    detail = str(excinfo.value.detail)
    assert str(secret_dir) not in detail
    assert ".code-intel" not in detail
    assert "deploy" not in detail
    assert "Errno" not in detail
    assert "Traceback" not in detail


def test_malformed_json_does_not_leak_the_parse_position(monkeypatch, tmp_path):
    state = tmp_path / "sync_state.json"
    state.write_text('{"acme_payments": {"commit": "deadbeef"')  # truncated on purpose
    monkeypatch.setattr(app, "SYNC_STATE_PATH", state)

    with pytest.raises(HTTPException) as excinfo:
        call_list_repos()

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == app._GENERIC_REPOS_ERROR

    detail = str(excinfo.value.detail)
    assert "Expecting" not in detail
    assert "char" not in detail
    assert "line 1 column" not in detail
    assert "acme_payments" not in detail


def test_the_failure_is_logged_with_its_traceback(monkeypatch, tmp_path, caplog):
    """Hiding it from the client must not hide it from the operator."""
    state = tmp_path / "sync_state.json"
    state.write_text("not json at all")
    monkeypatch.setattr(app, "SYNC_STATE_PATH", state)

    with caplog.at_level(logging.ERROR, logger="app"):
        with pytest.raises(HTTPException):
            call_list_repos()

    assert caplog.records, "the failure was hidden from the client and from the log"
    record = caplog.records[-1]
    assert record.exc_info is not None
    assert "Expecting value" in logging.Formatter().format(record)
    assert USER["id"] in record.getMessage()


def test_the_original_exception_is_still_chained(monkeypatch, tmp_path):
    """`from e` is kept, so a server-side traceback still shows the real cause."""
    state = tmp_path / "sync_state.json"
    state.write_text("{")
    monkeypatch.setattr(app, "SYNC_STATE_PATH", state)

    with pytest.raises(HTTPException) as excinfo:
        call_list_repos()

    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


# ── The working paths are untouched ─────────────────────────────────────────────

def test_missing_sync_state_returns_an_empty_list(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "SYNC_STATE_PATH", tmp_path / "nope.json")

    assert call_list_repos() == {"repos": []}


def test_repos_are_returned_sorted(monkeypatch, tmp_path):
    state = tmp_path / "sync_state.json"
    state.write_text(json.dumps({"zulu": {}, "alpha": {}, "mike": {}}))
    monkeypatch.setattr(app, "SYNC_STATE_PATH", state)

    assert call_list_repos() == {"repos": ["alpha", "mike", "zulu"]}


# ── Audit guard ─────────────────────────────────────────────────────────────────

def test_list_repos_never_raises_with_the_caught_exception_in_the_detail():
    """Stops `detail=f"...{e}"` coming back in this handler.

    Scoped to `list_repos` on purpose. Two `except ValueError as e` blocks in
    `/auth/login` and `/auth/register` do forward `str(e)`, but those come from
    `core.auth`, which raises its own fixed strings ("Invalid email or
    password.") that are meant for the user — a blanket ban would be wrong
    there. The exception being handled here is raised by the OS and by the
    stdlib json parser, and never is.

    `raise ... from e` is fine: only the raised expression is inspected, not
    the `from` clause, which is what preserves the cause for the server log.
    """
    tree = ast.parse(Path(app.__file__).read_text(), filename="app.py")

    func = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "list_repos"
    )

    offenders = []
    for handler in ast.walk(func):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        for node in ast.walk(handler):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            for inner in ast.walk(node.exc):
                if isinstance(inner, ast.Name) and inner.id == handler.name:
                    offenders.append(
                        f"app.py:{node.lineno} raises with the caught exception {handler.name!r}"
                    )

    assert offenders == [], (
        "exception detail must be logged, not returned to the client: " + "; ".join(offenders)
    )
