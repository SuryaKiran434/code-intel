"""
tests/test_app_query_errors.py

Regression tests for CodeQL py/stack-trace-exposure on the /query SSE stream.

Both `except` blocks inside `_generate()` used to forward the upstream
exception to the browser:

    except Exception as e:
        yield _sse({"type": "error", "message": str(e)})

The messages that reach those handlers are not generic. `retrieve()` raises
pymilvus errors naming the Milvus host, port and collection, and voyageai
errors naming the embedding endpoint; `ask_stream()` raises openai errors
naming the chat-completions URL, the model, the request id — and, on an
authentication failure, an echo of the API key with only its middle redacted.
The web UI renders `event.message` verbatim, so all of that landed on screen
for any logged-in user who could make the backend fail.

The constraint that shapes the fix: this is a stream. Dropping the error and
returning would leave the browser's reader waiting on a socket that never
produces another frame, which is a worse bug than the leak. So the handlers
still emit an `error` event — the same event shape the frontend already
handles — carrying a fixed generic string, and the exception with its
traceback goes to the server log instead.

Fully offline: `retrieve`, `ask_stream` and every session/telemetry write are
replaced with in-process fakes, so no Milvus, no Voyage, no OpenAI, no network
and no SQLite writes outside a temp directory.
"""

import ast
import asyncio
import json
import logging
import tempfile
from pathlib import Path

import pytest

# `app` calls init_db() at import time, so the SQLite path is redirected into a
# temp directory for the duration of that import and then put back — leaving it
# rebound would change the path every later-collected module sees. Mirrors
# tests/conftest.py's tmp_db fixture: core.db did `from config import DB_PATH`,
# so both bindings need rebinding.
import config
import core.db

_REAL_DB_PATH = config.DB_PATH
_TEST_DB = Path(tempfile.mkdtemp(prefix="code-intel-app-tests-")) / "code_intel.db"
config.DB_PATH = _TEST_DB
core.db.DB_PATH = _TEST_DB
try:
    import app  # noqa: E402 — must follow the DB_PATH redirect above
finally:
    config.DB_PATH = _REAL_DB_PATH
    core.db.DB_PATH = _REAL_DB_PATH


USER = {"id": "user-1", "email": "dev@example.com"}

# Stand-ins for what the real clients actually put in their messages.
MILVUS_ERROR = (
    "<MilvusException: (code=2, message=Fail connecting to server on "
    "localhost:19530, illegal connection params or server unavailable, "
    "collection=code_intel, partition=acme_payments)>"
)
OPENAI_ERROR = (
    "Error code: 401 - {'error': {'message': 'Incorrect API key provided: "
    "sk-proj-AbCd************************WxYz. You can find your API key at "
    "https://platform.openai.com/account/api-keys', 'type': "
    "'invalid_request_error'}} for POST https://api.openai.com/v1/chat/completions "
    "(request id req_9f2c1b7a)"
)

CHUNKS = [{"score": 0.91, "text": "def f(): ..."}]
SOURCES = [{"label": "core/f.py:f", "file": "core/f.py", "symbol": "f", "lines": "1-2", "score": 0.9123}]


@pytest.fixture()
def offline(monkeypatch):
    """Cut every dependency of /query except the two we drive per-test."""
    monkeypatch.setattr(app, "get_session", lambda session_id: None)
    monkeypatch.setattr(app, "create_session", lambda user_id, title=None: "session-1")
    monkeypatch.setattr(app, "load_turns", lambda session_id: [])
    monkeypatch.setattr(app, "append_turns_batch", lambda *a, **k: None)
    monkeypatch.setattr(app, "log_query", lambda *a, **k: None)


def run_query(question="how does auth work?", repo_name=None):
    """Drive POST /query and return the decoded SSE events.

    `query()` is called directly rather than through TestClient: httpx is not a
    dependency of this project, and the object under test is the generator
    inside the StreamingResponse. Draining `body_iterator` to exhaustion is
    also the assertion that the stream terminates rather than hanging.
    """
    response = app.query(app.QueryRequest(question=question, repo_name=repo_name), user=USER)

    async def drain():
        return [chunk async for chunk in response.body_iterator]

    events = []
    for chunk in asyncio.run(drain()):
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        for frame in text.strip().split("\n\n"):
            if frame.startswith("data: "):
                events.append(json.loads(frame[len("data: "):]))
    return events


def raiser(message):
    def _raise(*args, **kwargs):
        raise RuntimeError(message)
    return _raise


# ── Retrieval failure ───────────────────────────────────────────────────────────

def test_retrieval_failure_does_not_leak_the_exception(offline, monkeypatch):
    monkeypatch.setattr(app, "retrieve", raiser(MILVUS_ERROR))

    events = run_query()

    assert events == [{"type": "error", "message": app._GENERIC_STREAM_ERROR}]

    blob = json.dumps(events)
    assert MILVUS_ERROR not in blob
    assert "19530" not in blob
    assert "acme_payments" not in blob
    assert "MilvusException" not in blob
    assert "Traceback" not in blob


def test_retrieval_failure_is_logged_with_its_traceback(offline, monkeypatch, caplog):
    monkeypatch.setattr(app, "retrieve", raiser(MILVUS_ERROR))

    with caplog.at_level(logging.ERROR, logger="app"):
        run_query(repo_name="acme_payments")

    assert caplog.records, "the failure was hidden from the client and from the log"
    record = caplog.records[-1]
    assert record.exc_info is not None
    assert MILVUS_ERROR in logging.Formatter().format(record)
    assert "acme_payments" in record.getMessage()


# ── Mid-stream LLM failure ──────────────────────────────────────────────────────

def test_llm_failure_mid_stream_does_not_leak_the_exception(offline, monkeypatch):
    """The hard case: tokens are already on the wire when the error happens."""
    monkeypatch.setattr(app, "retrieve", lambda *a, **k: CHUNKS)

    def exploding_stream(*args, **kwargs):
        yield {"type": "token", "text": "The auth flow "}
        yield {"type": "token", "text": "starts in core/auth.py"}
        raise RuntimeError(OPENAI_ERROR)

    monkeypatch.setattr(app, "llm_ask_stream", exploding_stream)

    events = run_query()

    assert [e["type"] for e in events] == ["token", "token", "error"]
    assert events[-1]["message"] == app._GENERIC_STREAM_ERROR

    blob = json.dumps(events)
    assert OPENAI_ERROR not in blob
    assert "sk-proj-AbCd" not in blob, "a partial API key reached the client"
    assert "api.openai.com" not in blob
    assert "req_9f2c1b7a" not in blob


def test_the_stream_is_closed_rather_than_left_hanging(offline, monkeypatch):
    """Swallowing the error would leave the browser's reader waiting forever.

    The client only stops when it sees a terminal frame, so a failure must
    still produce one — draining to exhaustion above proves the generator
    ends, and this pins down that it ends with an event the frontend acts on
    (`static/index.html` throws on `event.type === "error"`).
    """
    monkeypatch.setattr(app, "retrieve", lambda *a, **k: CHUNKS)

    def exploding_stream(*args, **kwargs):
        raise RuntimeError(OPENAI_ERROR)
        yield  # pragma: no cover — makes this a generator

    monkeypatch.setattr(app, "llm_ask_stream", exploding_stream)

    events = run_query()

    assert events, "the stream produced no frames at all — the client would hang"
    assert events[-1]["type"] == "error"
    assert events[-1]["message"]


def test_llm_failure_is_logged_with_its_traceback(offline, monkeypatch, caplog):
    monkeypatch.setattr(app, "retrieve", lambda *a, **k: CHUNKS)

    def exploding_stream(*args, **kwargs):
        raise RuntimeError(OPENAI_ERROR)
        yield  # pragma: no cover

    monkeypatch.setattr(app, "llm_ask_stream", exploding_stream)

    with caplog.at_level(logging.ERROR, logger="app"):
        run_query()

    record = caplog.records[-1]
    assert record.exc_info is not None
    assert OPENAI_ERROR in logging.Formatter().format(record)


# ── The happy path is untouched ─────────────────────────────────────────────────

def test_successful_query_still_streams_tokens_and_sources(offline, monkeypatch):
    monkeypatch.setattr(app, "retrieve", lambda *a, **k: CHUNKS)

    def good_stream(*args, **kwargs):
        yield {"type": "token", "text": "It starts in core/auth.py."}
        yield {"type": "done", "sources": SOURCES, "tokens": 42}

    monkeypatch.setattr(app, "llm_ask_stream", good_stream)

    events = run_query()

    assert events[0] == {"type": "token", "text": "It starts in core/auth.py."}
    assert events[1]["type"] == "done"
    assert events[1]["tokens"] == 42
    assert events[1]["session_id"] == "session-1"
    assert events[1]["sources"] == [
        {"label": "core/f.py:f", "file": "core/f.py", "symbol": "f", "lines": "1-2", "score": 0.912}
    ]


def test_no_results_path_still_reports_cleanly(offline, monkeypatch):
    monkeypatch.setattr(app, "retrieve", lambda *a, **k: [])

    events = run_query()

    assert [e["type"] for e in events] == ["token", "done"]
    assert events[0]["text"] == "No relevant code found for your question."


# ── Audit guard ─────────────────────────────────────────────────────────────────

def test_no_sse_frame_in_app_py_carries_a_caught_exception():
    """Stops `yield _sse({... str(e) ...})` coming back anywhere in app.py.

    Walks every `except ... as <name>` block and fails if the bound name is
    reachable from a `yield` inside it.
    """
    source = Path(app.__file__).read_text()
    tree = ast.parse(source, filename="app.py")

    offenders = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        for node in ast.walk(handler):
            if not isinstance(node, ast.Yield) or node.value is None:
                continue
            for inner in ast.walk(node.value):
                if isinstance(inner, ast.Name) and inner.id == handler.name:
                    offenders.append(f"app.py:{node.lineno} yields the caught exception {handler.name!r}")

    assert offenders == [], (
        "exception detail must be logged, not streamed to the client: " + "; ".join(offenders)
    )
