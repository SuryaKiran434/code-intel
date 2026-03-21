"""
core/retriever.py

Handles semantic search against Milvus at query time.

Key design decisions:
  - Always uses embed_query() (not embed_code()) so the correct
    voyage-code-3 query prefix is applied automatically
  - Excludes "summary" chunks from results by default — summaries
    are stored for LLM context fallback, not for direct retrieval
  - Supports optional repo_name scoping for focused searches
  - Returns rich metadata alongside content so the LLM and UI
    have full source context (file, symbol, line numbers, score)
"""

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor

import voyageai

_log = logging.getLogger(__name__)

from core.vector_store import get_or_create_collection, partition_name
from core.embedder import embed_queries
from config import (
    TOP_K_RESULTS,
    VOYAGE_API_KEY,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_FINAL_K,
    MIN_RETRIEVAL_SCORE,
    QUERY_EXPANSION_ENABLED,
    COMPLEX_QUERY_CANDIDATE_K,
    COMPLEX_QUERY_FINAL_K,
    COMPLEX_QUERY_MIN_WORDS,
    COMPLEX_QUERY_KEYWORDS,
)

_voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

# Module-level collection cache — loaded once per process, reused on every query
_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        _collection = get_or_create_collection()
        _collection.load()
    return _collection

# Lazy import to avoid circular dependency at module load time
def _get_expander():
    from core.query_expander import expand_query
    return expand_query


# ── Output field set returned from Milvus ──────────────────────────────────────
_OUTPUT_FIELDS = [
    "content",
    "file_path",
    "repo_name",
    "symbol_name",
    "start_line",
    "end_line",
    "language",
    "chunk_type",
    "parent_symbol",
]


def retrieve(
    query: str,
    repo_name: str = None,
    include_summaries: bool = False,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Embed the user query and retrieve the most semantically similar
    code chunks from Milvus.

    When RERANKER_ENABLED is True (default):
      - Fetches RETRIEVAL_CANDIDATE_K candidates from Milvus
      - Re-ranks with Voyage rerank-2.5-lite
      - Returns RETRIEVAL_FINAL_K results (or top_k if explicitly set)

    When RERANKER_ENABLED is False:
      - Returns top_k results directly from Milvus cosine search

    If all scores fall below MIN_RETRIEVAL_SCORE, returns an empty list
    so the LLM layer can issue a "not enough info" response instead of
    hallucinating from weak context.

    Returns:
        List of dicts, each containing:
            content, file_path, repo_name, symbol_name,
            start_line, end_line, language, chunk_type,
            parent_symbol, score
    """
    collection = _get_collection()

    filters = _build_filter(repo_name, include_summaries)
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    # ── Adaptive Top-K ────────────────────────────────────────────────────────
    is_complex = _is_complex_query(query)
    if is_complex:
        candidate_limit = COMPLEX_QUERY_CANDIDATE_K if RERANKER_ENABLED else top_k
        final_k         = COMPLEX_QUERY_FINAL_K
    else:
        candidate_limit = RETRIEVAL_CANDIDATE_K if RERANKER_ENABLED else top_k
        final_k         = RETRIEVAL_FINAL_K

    # ── Query expansion + original embedding (run in parallel) ───────────────
    # Skip expansion for very short queries (≤ 4 words) — they are already
    # specific enough that variants add noise, not signal. Saves 2-3s.
    _run_expansion = QUERY_EXPANSION_ENABLED and len(query.split()) > 4

    t0 = _time.monotonic()
    if _run_expansion:
        # Start expansion and original-query embedding at the same time.
        # Expansion (GPT-4o-mini ~1-2s) dominates; overlapping the embed call
        # (~100ms for 1 query) hides it almost entirely.
        with ThreadPoolExecutor(max_workers=2) as pool:
            orig_future = pool.submit(embed_queries, [query])
            exp_future  = pool.submit(_get_expander(), query)
            orig_vector = orig_future.result()[0]
            variants    = exp_future.result()
        _log.info(
            "expansion+embed_orig: %.0fms  (%d variants)",
            (_time.monotonic() - t0) * 1000, len(variants),
        )
        t1 = _time.monotonic()
        variant_vectors = embed_queries(variants) if variants else []
        _log.info("embed_variants: %.0fms", (_time.monotonic() - t1) * 1000)
        vectors = [orig_vector] + variant_vectors
    else:
        vectors = embed_queries([query])
        _log.info("embedding (no expansion): %.0fms", (_time.monotonic() - t0) * 1000)

    # Single Milvus search call with all query vectors — avoids N round-trips.
    # Scope to the repo's partition when repo_name is provided for efficiency.
    t2 = _time.monotonic()
    pnames = [partition_name(repo_name)] if repo_name else None
    # Only pass partition_names when the partition actually exists
    if pnames and not collection.has_partition(pnames[0]):
        pnames = None
    all_results = collection.search(
        data=vectors,
        anns_field="embedding",
        param=search_params,
        limit=candidate_limit,
        expr=filters if filters else None,
        output_fields=_OUTPUT_FIELDS,
        partition_names=pnames,
        consistency_level="Eventually",
    )
    _log.info("milvus:    %.0fms", (_time.monotonic() - t2) * 1000)

    # Merge results from all query vectors, keep highest score per chunk
    seen_ids: dict[str, dict] = {}
    for result_set in all_results:
        for chunk in _format_result_set(result_set):
            chunk_id = chunk["file_path"] + "::" + chunk["symbol_name"]
            if chunk_id not in seen_ids or chunk["score"] > seen_ids[chunk_id]["score"]:
                seen_ids[chunk_id] = chunk

    chunks = list(seen_ids.values())

    if not chunks:
        return []

    # ── Re-ranking ────────────────────────────────────────────────────────────
    t3 = _time.monotonic()
    if RERANKER_ENABLED and len(chunks) > 1:
        # Pre-filter: drop candidates whose cosine score is far below the best
        # match before sending to the cross-encoder — reduces reranker cost and
        # latency without sacrificing recall on the candidates that matter.
        max_score  = max(c["score"] for c in chunks)
        pre_filtered = [c for c in chunks if max_score - c["score"] <= 0.35]
        # Always keep enough candidates for reranking to be meaningful
        to_rerank = pre_filtered if len(pre_filtered) >= final_k else chunks[:final_k * 2]
        chunks = _rerank(query, to_rerank, top_k=final_k)
    else:
        chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)[:final_k]
    _log.info("rerank:    %.0fms  (%d→%d chunks)", (_time.monotonic() - t3) * 1000, len(seen_ids), len(chunks))

    # ── Confidence threshold ───────────────────────────────────────────────────
    if all(c["score"] < MIN_RETRIEVAL_SCORE for c in chunks):
        return []

    # ── Graph expansion — pull in callee dependency chunks (cross-file) ───────
    # Complex queries get depth=2 (follow callees of callees) and a larger
    # graph chunk budget; simple queries stay at depth=1 to keep latency low.
    if repo_name:
        graph_depth = 2 if is_complex else 1
        graph_cap   = 5 if is_complex else 3
        chunks = _expand_with_graph(
            chunks, repo_name, collection,
            max_graph_chunks=graph_cap, depth=graph_depth,
        )

    return chunks


def retrieve_for_file(
    file_path: str,
    repo_name: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve all chunks belonging to a specific file.
    Useful for the web UI 'browse file' feature (future extension).
    """
    collection = _get_collection()

    results = collection.query(
        expr=f'repo_name == "{repo_name}" && file_path == "{file_path}"',
        output_fields=_OUTPUT_FIELDS,
        limit=top_k,
    )

    return [
        {**r, "score": 1.0}   # direct lookup, score is always 1.0
        for r in results
    ]


def retrieve_by_symbol(
    symbol_name: str,
    repo_name: str = None,
) -> list[dict]:
    """
    Exact symbol name lookup — finds a specific function or class by name.
    Useful for: 'Show me the authenticate() function'.
    """
    collection = _get_collection()

    expr = f'symbol_name == "{symbol_name}"'
    if repo_name:
        expr += f' && repo_name == "{repo_name}"'

    results = collection.query(
        expr=expr,
        output_fields=_OUTPUT_FIELDS,
    )

    return [
        {**r, "score": 1.0}
        for r in results
        if r.get("chunk_type") != "summary"
    ]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_complex_query(query: str) -> bool:
    """
    Heuristic: a query is 'complex' if it is long or contains architectural
    keywords. Complex queries get a larger Milvus candidate pool and more
    final chunks so the LLM has broader context to work with.
    """
    words = query.lower().split()
    if len(words) >= COMPLEX_QUERY_MIN_WORDS:
        return True
    return bool(set(words) & COMPLEX_QUERY_KEYWORDS)


def _rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Re-rank chunks using Voyage rerank-2.5-lite cross-encoder.
    Replaces cosine similarity scores with reranker relevance scores.
    """
    documents = [c["content"] for c in chunks]
    response = _voyage_client.rerank(
        query=query,
        documents=documents,
        model=RERANKER_MODEL,
        top_k=top_k,
    )
    reranked = []
    for result in response.results:
        chunk = dict(chunks[result.index])
        chunk["score"] = round(float(result.relevance_score), 4)
        reranked.append(chunk)
    return reranked


def _build_filter(repo_name: str | None, include_summaries: bool) -> str | None:
    """
    Construct the Milvus boolean expression for search filtering.

    Repo scoping is handled via partition_names (see retrieve()), so this
    function only handles the chunk_type filter. The repo_name expression
    clause is kept as a fallback for collections without partitions.

    chunk_type values in the index:
        "full"         — always retrieved
        "split_part"   — always retrieved
        "docstring"    — always retrieved (Phase 4.3)
        "module_level" — always retrieved (Phase 4.2)
        "summary"      — excluded by default (LLM context fallback only)
    """
    parts = []

    # Partition-based scoping handles repo filtering when a partition exists.
    # Keep the expression clause as a fallback for pre-partition legacy data.
    if repo_name:
        parts.append(f'repo_name == "{repo_name}"')

    if not include_summaries:
        parts.append('chunk_type != "summary"')

    return " && ".join(parts) if parts else None


def _format_result_set(result_set) -> list[dict]:
    """Flatten one Milvus result set (hits for a single query vector) into dicts."""
    return [
        {
            "content":          hit.entity.get("content"),
            "file_path":        hit.entity.get("file_path"),
            "repo_name":        hit.entity.get("repo_name"),
            "symbol_name":      hit.entity.get("symbol_name"),
            "start_line":       hit.entity.get("start_line"),
            "end_line":         hit.entity.get("end_line"),
            "language":         hit.entity.get("language"),
            "chunk_type":       hit.entity.get("chunk_type"),
            "parent_symbol":    hit.entity.get("parent_symbol"),
            "score":            round(float(hit.score), 4),
            "retrieval_source": "direct",
        }
        for hit in result_set
    ]


def _expand_with_graph(
    chunks: list[dict],
    repo_name: str,
    collection,
    max_graph_chunks: int = 3,
    depth: int = 1,
) -> list[dict]:
    """
    Expand direct retrieval results with dependency chunks from the call graph.

    Uses BFS up to `depth` hops. depth=1 follows direct callees; depth=2 also
    follows the callees of those callees (useful for architectural queries).
    Graph-expanded chunks are tagged `retrieval_source="graph"` so
    build_context() labels them [G1], [G2] separately from [C1], [C2].

    At most `max_graph_chunks` graph chunks are added in total.
    """
    from core.graph import get_callees  # noqa: PLC0415

    # Track all chunk identities seen so far (direct + graph)
    seen_ids: set[str] = {
        f"{c['file_path']}::{c['symbol_name']}" for c in chunks
    }

    graph_chunks: list[dict] = []
    # BFS frontier starts with the direct result set
    frontier = [c for c in chunks if c.get("chunk_type") != "summary"]

    for _ in range(depth):
        if len(graph_chunks) >= max_graph_chunks or not frontier:
            break

        next_frontier: list[dict] = []

        for chunk in frontier:
            if len(graph_chunks) >= max_graph_chunks:
                break

            callees = get_callees(repo_name, chunk["file_path"], chunk["symbol_name"])

            for callee in callees:
                if len(graph_chunks) >= max_graph_chunks:
                    break

                dep_chunks = retrieve_by_symbol(callee, repo_name)
                for dep in dep_chunks:
                    cid = f"{dep['file_path']}::{dep['symbol_name']}"
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        dep["retrieval_source"] = "graph"
                        graph_chunks.append(dep)
                        next_frontier.append(dep)
                        break   # one chunk per callee symbol is enough

        frontier = next_frontier  # next hop expands from this hop's results

    return chunks + graph_chunks
