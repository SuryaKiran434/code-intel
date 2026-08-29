"""
tests/test_retriever.py

Tests for core/retriever.py — candidate selection and graph expansion.

All external calls (Milvus, Voyage embed, Voyage rerank) are mocked; these are
pure unit tests and never touch the network.

Covers:
  - Re-rank fallback keeps the highest-scoring candidates, not the first N
  - Non-rerank branch returns the true top-k by score
  - retrieve_by_symbol: server-side summary filter, limit, quote escaping
  - Batched graph expansion returns the same chunks as the per-callee path
"""

import re

import pytest

import core.retriever as r


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _FakeHit:
    """Mimics a pymilvus search hit (entity dict + score)."""

    def __init__(self, entity: dict, score: float):
        self.entity = entity
        self.score = score


class _FakeSearchCollection:
    """Collection whose search() returns one result set of preset hits."""

    def __init__(self, hits):
        self._hits = hits

    def has_partition(self, name):
        return False

    def search(self, **kwargs):
        return [self._hits]


class _FakeQueryCollection:
    """
    Collection whose query() resolves the symbol_name clause of a Milvus
    expression against an in-memory row table. Serves both the `== "x"` form
    (old per-callee path) and the `in [...]` form (new batched path) so the
    two can be compared against identical data.
    """

    def __init__(self, rows):
        self.rows = rows
        self.exprs: list[str] = []
        self.kwargs: list[dict] = []

    def query(self, expr, output_fields=None, **kwargs):
        self.exprs.append(expr)
        self.kwargs.append(kwargs)

        in_clause = re.search(r"symbol_name in \[(.*?)\]", expr)
        if in_clause:
            wanted = set(re.findall(r'"(.*?)"', in_clause.group(1)))
        else:
            eq_clause = re.search(r'symbol_name == "(.*?)"', expr)
            wanted = {eq_clause.group(1)} if eq_clause else set()

        rows = [dict(row) for row in self.rows if row["symbol_name"] in wanted]
        if 'chunk_type != "summary"' in expr:
            rows = [row for row in rows if row["chunk_type"] != "summary"]
        limit = kwargs.get("limit")
        return rows[:limit] if limit is not None else rows


def _chunk(symbol, score, file_path=None, chunk_type="full"):
    return {
        "content": f"def {symbol}(): ...",
        "file_path": file_path or f"/repo/{symbol}.py",
        "repo_name": "myrepo",
        "symbol_name": symbol,
        "start_line": 1,
        "end_line": 2,
        "language": "python",
        "chunk_type": chunk_type,
        "parent_symbol": "",
        "score": score,
    }


@pytest.fixture()
def no_network(monkeypatch):
    """Stub the embed call so retrieve() never hits Voyage."""
    monkeypatch.setattr(r, "embed_queries", lambda texts: [[0.1, 0.2] for _ in texts])


# ── Fix 1: re-rank fallback must not slice an unsorted list ───────────────────

def test_rerank_fallback_keeps_highest_scoring_candidates(monkeypatch, no_network):
    """
    When the pre-filter keeps fewer than final_k candidates, the fallback must
    select the top-scoring candidates. `chunks` is in Milvus return order across
    query vectors — NOT score order — so slicing it drops the best matches.
    """
    # Ten weak chunks first, the two strongest last: a plain [:final_k * 2]
    # slice would take exactly the ten weak ones and discard both strong ones.
    hits = [_FakeHit(_chunk(f"weak{i}", 0.10 + i * 0.01), 0.10 + i * 0.01) for i in range(10)]
    hits.append(_FakeHit(_chunk("strong_a", 0.95), 0.95))
    hits.append(_FakeHit(_chunk("strong_b", 0.99), 0.99))

    monkeypatch.setattr(r, "_get_collection", lambda: _FakeSearchCollection(hits))
    monkeypatch.setattr(r, "RERANKER_ENABLED", True)

    captured = {}

    def _fake_rerank(query, chunks, top_k):
        captured["to_rerank"] = chunks
        return chunks[:top_k]

    monkeypatch.setattr(r, "_rerank", _fake_rerank)

    r.retrieve("parse token", repo_name=None)

    to_rerank = captured["to_rerank"]
    names = {c["symbol_name"] for c in to_rerank}
    assert "strong_b" in names, "highest-scoring candidate was dropped before reranking"
    assert "strong_a" in names, "second-highest candidate was dropped before reranking"
    # final_k = RETRIEVAL_FINAL_K (5) for a simple query → fallback keeps 10
    assert len(to_rerank) == 10
    assert [c["score"] for c in to_rerank] == sorted(
        (c["score"] for c in to_rerank), reverse=True
    )


def test_rerank_prefilter_used_when_it_has_enough_candidates(monkeypatch, no_network):
    """Existing behaviour: a large enough pre-filtered set is passed through as-is."""
    hits = [_FakeHit(_chunk(f"c{i}", 0.90 - i * 0.01), 0.90 - i * 0.01) for i in range(8)]

    monkeypatch.setattr(r, "_get_collection", lambda: _FakeSearchCollection(hits))
    monkeypatch.setattr(r, "RERANKER_ENABLED", True)

    captured = {}

    def _fake_rerank(query, chunks, top_k):
        captured["to_rerank"] = chunks
        return chunks[:top_k]

    monkeypatch.setattr(r, "_rerank", _fake_rerank)

    r.retrieve("parse token", repo_name=None)
    assert len(captured["to_rerank"]) == 8   # all within 0.35 of the best score


# ── Fix 4: non-rerank branch selects top-k without a full sort ────────────────

def test_non_rerank_branch_returns_top_k_by_score(monkeypatch, no_network):
    scores = [0.55, 0.99, 0.61, 0.72, 0.88, 0.51, 0.95, 0.66]
    hits = [_FakeHit(_chunk(f"c{i}", s), s) for i, s in enumerate(scores)]

    monkeypatch.setattr(r, "_get_collection", lambda: _FakeSearchCollection(hits))
    monkeypatch.setattr(r, "RERANKER_ENABLED", False)

    out = r.retrieve("parse token", repo_name=None)

    assert [c["score"] for c in out] == sorted(scores, reverse=True)[:r.RETRIEVAL_FINAL_K]


# ── Fix 3: retrieve_by_symbol — pushdown, limit, escaping ─────────────────────

def test_retrieve_by_symbol_filters_summaries_server_side(monkeypatch):
    rows = [
        _chunk("target", 1.0, file_path="/repo/a.py"),
        _chunk("target", 1.0, file_path="/repo/b.py", chunk_type="summary"),
    ]
    fake = _FakeQueryCollection(rows)
    monkeypatch.setattr(r, "_get_collection", lambda: fake)

    out = r.retrieve_by_symbol("target", "myrepo")

    assert 'chunk_type != "summary"' in fake.exprs[0]
    assert [c["file_path"] for c in out] == ["/repo/a.py"]
    assert all(c["score"] == 1.0 for c in out)


def test_retrieve_by_symbol_passes_limit_only_when_given(monkeypatch):
    rows = [_chunk("target", 1.0, file_path=f"/repo/{i}.py") for i in range(5)]
    fake = _FakeQueryCollection(rows)
    monkeypatch.setattr(r, "_get_collection", lambda: fake)

    assert len(r.retrieve_by_symbol("target", "myrepo", limit=2)) == 2
    assert fake.kwargs[0]["limit"] == 2

    assert len(r.retrieve_by_symbol("target", "myrepo")) == 5
    assert "limit" not in fake.kwargs[1]


def test_retrieve_by_symbol_escapes_quotes(monkeypatch):
    fake = _FakeQueryCollection([])
    monkeypatch.setattr(r, "_get_collection", lambda: fake)

    r.retrieve_by_symbol('evil" || repo_name != "', 'repo" || ""')

    expr = fake.exprs[0]
    # Every quote from user input is escaped, so the injected clauses stay
    # inside the string literals instead of becoming expression syntax.
    assert 'symbol_name == "evil\\" || repo_name != \\""' in expr
    assert 'repo_name == "repo\\" || \\"\\""' in expr


def test_escape_expr_str_escapes_backslashes_first():
    assert r._escape_expr_str(r'a\"b') == r'a\\\"b'


# ── Fix 2: batched graph expansion ────────────────────────────────────────────

_GRAPH_EDGES = {
    ("/repo/a.py", "alpha"): ["beta", "gamma"],
    ("/repo/b.py", "delta"): ["gamma", "epsilon"],
    ("/repo/beta.py", "beta"): ["zeta"],
    ("/repo/gamma.py", "gamma"): ["eta"],
}

_GRAPH_ROWS = [
    _chunk("beta", 1.0, file_path="/repo/beta.py"),
    _chunk("gamma", 1.0, file_path="/repo/gamma.py", chunk_type="summary"),
    _chunk("gamma", 1.0, file_path="/repo/gamma.py"),
    _chunk("epsilon", 1.0, file_path="/repo/epsilon.py"),
    _chunk("zeta", 1.0, file_path="/repo/zeta.py"),
    _chunk("eta", 1.0, file_path="/repo/eta.py"),
]


def _old_expand_with_graph(chunks, repo_name, collection, max_graph_chunks, depth):
    """The pre-batching implementation, kept here as the parity reference."""
    seen_ids = {f"{c['file_path']}::{c['symbol_name']}" for c in chunks}
    graph_chunks = []
    frontier = [c for c in chunks if c.get("chunk_type") != "summary"]

    for _ in range(depth):
        if len(graph_chunks) >= max_graph_chunks or not frontier:
            break
        next_frontier = []
        for chunk in frontier:
            if len(graph_chunks) >= max_graph_chunks:
                break
            callees = _GRAPH_EDGES.get((chunk["file_path"], chunk["symbol_name"]), [])
            for callee in callees:
                if len(graph_chunks) >= max_graph_chunks:
                    break
                for dep in r.retrieve_by_symbol(callee, repo_name):
                    cid = f"{dep['file_path']}::{dep['symbol_name']}"
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        dep["retrieval_source"] = "graph"
                        graph_chunks.append(dep)
                        next_frontier.append(dep)
                        break
        frontier = next_frontier

    return chunks + graph_chunks


def _ids(chunks):
    return [(c["file_path"], c["symbol_name"], c.get("retrieval_source")) for c in chunks]


@pytest.fixture()
def graph_stubs(monkeypatch):
    """Serve call edges from _GRAPH_EDGES without touching SQLite."""
    import core.graph as g

    def _batch(repo_name, pairs):
        return {p: list(_GRAPH_EDGES[p]) for p in dict.fromkeys(pairs) if p in _GRAPH_EDGES}

    monkeypatch.setattr(g, "get_callees_batch", _batch)


@pytest.mark.parametrize(("depth", "cap"), [(1, 3), (2, 5), (1, 1), (2, 2), (2, 99)])
def test_batched_graph_expansion_matches_per_callee_path(monkeypatch, graph_stubs, depth, cap):
    seed = [
        _chunk("alpha", 0.9, file_path="/repo/a.py"),
        _chunk("delta", 0.8, file_path="/repo/b.py"),
    ]

    old_collection = _FakeQueryCollection(_GRAPH_ROWS)
    monkeypatch.setattr(r, "_get_collection", lambda: old_collection)
    expected = _old_expand_with_graph(
        [dict(c) for c in seed], "myrepo", old_collection,
        max_graph_chunks=cap, depth=depth,
    )

    new_collection = _FakeQueryCollection(_GRAPH_ROWS)
    actual = r._expand_with_graph(
        [dict(c) for c in seed], "myrepo", new_collection,
        max_graph_chunks=cap, depth=depth,
    )

    assert _ids(actual) == _ids(expected)
    # …and it does so with at most one Milvus round trip per BFS hop.
    assert len(new_collection.exprs) <= depth
    assert len(new_collection.exprs) <= len(old_collection.exprs)


def test_graph_expansion_tags_source_and_respects_cap(monkeypatch, graph_stubs):
    seed = [_chunk("alpha", 0.9, file_path="/repo/a.py")]
    collection = _FakeQueryCollection(_GRAPH_ROWS)

    out = r._expand_with_graph(seed, "myrepo", collection, max_graph_chunks=1, depth=2)

    assert len(out) == 2
    assert out[0].get("retrieval_source") != "graph"
    assert out[1]["retrieval_source"] == "graph"
    assert out[1]["symbol_name"] == "beta"


def test_graph_expansion_skips_summary_chunks(monkeypatch, graph_stubs):
    """A summary chunk must never be pulled in as a graph dependency."""
    seed = [_chunk("delta", 0.9, file_path="/repo/b.py")]
    collection = _FakeQueryCollection(_GRAPH_ROWS)

    out = r._expand_with_graph(seed, "myrepo", collection, max_graph_chunks=5, depth=1)

    assert all(c["chunk_type"] != "summary" for c in out)
    assert [c["symbol_name"] for c in out] == ["delta", "gamma", "epsilon"]


def test_graph_expansion_no_query_when_no_callees(monkeypatch, graph_stubs):
    seed = [_chunk("orphan", 0.9, file_path="/repo/z.py")]
    collection = _FakeQueryCollection(_GRAPH_ROWS)

    out = r._expand_with_graph(seed, "myrepo", collection, max_graph_chunks=3, depth=2)

    assert out == seed
    assert collection.exprs == []
