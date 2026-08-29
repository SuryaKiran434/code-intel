"""
tests/test_embedder.py

Tests for core/embedder.py — batched Voyage embedding.

The Voyage client is replaced with a stub; no network calls are made.

Covers:
  - Batches run concurrently but results stay in input order
  - Single-batch and empty inputs
  - Per-batch retry with exponential backoff is preserved
"""

import threading
import time
from types import SimpleNamespace

import pytest

import core.embedder as e


class _StubClient:
    """
    Stand-in for voyageai.Client. Returns a deterministic vector per text and
    sleeps longer for texts submitted earlier, so completion order is the
    reverse of submission order.
    """

    def __init__(self, delay_scale=0.0):
        self.delay_scale = delay_scale
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()
        self.active = 0
        self.peak_active = 0

    def embed(self, texts, model, input_type, output_dimension):
        with self._lock:
            self.calls.append(list(texts))
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        if self.delay_scale:
            # Earlier texts sleep longest → completion order ≠ submission order
            time.sleep(self.delay_scale * (100 - int(texts[0])))
        with self._lock:
            self.active -= 1
        return SimpleNamespace(embeddings=[[float(t)] for t in texts])


def _embedder(client):
    """Build a VoyageEmbedder without running __init__ (no API key needed)."""
    emb = e.VoyageEmbedder.__new__(e.VoyageEmbedder)
    emb._client = client
    return emb


# ── Order preservation ────────────────────────────────────────────────────────

def test_call_api_preserves_input_order(monkeypatch):
    """
    Chunk-to-vector alignment depends on this: parallel batches must be
    reassembled by input position, never by completion order.
    """
    monkeypatch.setattr(e, "EMBEDDING_BATCH_SIZE", 2)
    client = _StubClient(delay_scale=0.002)
    texts = [str(i) for i in range(20)]

    vectors = _embedder(client)._call_api(texts, input_type="document")

    assert vectors == [[float(i)] for i in range(20)]
    assert len(client.calls) == 10           # 20 texts / batch size 2
    assert client.peak_active > 1, "batches were not embedded concurrently"


def test_call_api_bounded_concurrency(monkeypatch):
    monkeypatch.setattr(e, "EMBEDDING_BATCH_SIZE", 1)
    monkeypatch.setattr(e, "EMBED_MAX_WORKERS", 3)
    client = _StubClient(delay_scale=0.002)

    _embedder(client)._call_api([str(i) for i in range(12)], input_type="document")

    assert client.peak_active <= 3


def test_call_api_single_batch_and_empty(monkeypatch):
    monkeypatch.setattr(e, "EMBEDDING_BATCH_SIZE", 128)
    client = _StubClient()
    emb = _embedder(client)

    assert emb._call_api(["1", "2"], input_type="query") == [[1.0], [2.0]]
    assert len(client.calls) == 1

    assert emb._call_api([], input_type="query") == []
    assert len(client.calls) == 1   # nothing sent for an empty input


def test_embed_queries_and_embed_query_use_call_api(monkeypatch):
    monkeypatch.setattr(e, "EMBEDDING_BATCH_SIZE", 2)
    client = _StubClient()
    emb = _embedder(client)

    assert emb.embed_queries(["7", "8", "9"]) == [[7.0], [8.0], [9.0]]
    assert emb.embed_query("5") == [5.0]
    assert all(call for call in client.calls)


# ── Retry behaviour (unchanged by parallelisation) ────────────────────────────

class _FlakyClient(_StubClient):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self.attempts = 0

    def embed(self, texts, model, input_type, output_dimension):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("boom")
        return super().embed(texts, model, input_type, output_dimension)


def test_batch_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(e.time, "sleep", lambda s: None)
    client = _FlakyClient(failures=2)

    assert _embedder(client)._call_api(["3"], input_type="document") == [[3.0]]
    assert client.attempts == 3


def test_batch_raises_after_four_attempts(monkeypatch):
    monkeypatch.setattr(e.time, "sleep", lambda s: None)
    client = _FlakyClient(failures=99)

    with pytest.raises(RuntimeError, match="failed after 4 attempts"):
        _embedder(client)._call_api(["3"], input_type="document")
    assert client.attempts == 4


def test_parallel_batch_failure_propagates(monkeypatch):
    monkeypatch.setattr(e, "EMBEDDING_BATCH_SIZE", 1)
    monkeypatch.setattr(e.time, "sleep", lambda s: None)
    client = _FlakyClient(failures=99)

    with pytest.raises(RuntimeError, match="failed after 4 attempts"):
        _embedder(client)._call_api(["1", "2", "3"], input_type="document")
