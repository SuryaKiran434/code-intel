# Code Intel — Known Limitations

> An honest breakdown of current gaps, trade-offs, and planned improvements. Resolved items are kept for history.

---

## Table of Contents

- [Open Limitations](#open-limitations)
  - [Retrieval & Search](#retrieval--search)
  - [Chunking](#chunking)
  - [Sync & State Management](#sync--state-management)
  - [LLM & Context](#llm--context)
  - [Cross-Repo Intelligence](#cross-repo-intelligence)
  - [Operational](#operational)
    - [Docker Desktop Required](#docker-desktop-required)
    - [Repos Directory Hardcoded](#repos-directory-hardcoded)
    - [No HTTPS for Web UI](#no-https-for-web-ui)
    - [No API Rate Limiting](#no-api-rate-limiting)
    - [No Support for Jupyter Notebooks or Binary Files](#no-support-for-jupyter-notebooks-or-binary-files)
- [Resolved Limitations](#resolved-limitations)
- [Priority Roadmap](#priority-roadmap)

---

## Open Limitations

### Retrieval & Search

#### Method Calls Not Tracked in Call Graph
`core/graph.py` extracts bare function calls (e.g. `foo()`) but skips method calls on objects (e.g. `obj.bar()`, `self.helper()`). Dependency chains that flow through OOP method dispatch are invisible to graph-augmented retrieval.

**Fix:** Extend `_extract_calls()` to also resolve attribute calls — record `obj.method()` by the method name. Requires basic type inference or a naming-convention heuristic.

---

#### Graph Expansion Limited to Direct Callees
`_expand_with_graph()` only follows one hop of the call graph (direct callees of matched chunks) and caps at 3 graph chunks. Deep dependency chains are not followed.

**Fix:** Add optional depth parameter and increase `max_graph_chunks` for architectural queries. Combine with confidence filtering so low-relevance graph chunks do not crowd out high-relevance direct hits.

---

### Chunking

#### Python Only (Currently)
Java and Scala support is stubbed in `config.py` and `core/chunker.py` but not active. Any multi-language repo is partially blind.

**Fix:** Uncomment Java/Scala entries in `LANGUAGE_REGISTRY` in `config.py`, install the corresponding tree-sitter grammars, and add lazy imports in `core/chunker.py`. Zero other changes needed.

```python
# config.py — uncomment to enable:
# ".java":  {"name": "java",   "node_types": ["method_declaration", "class_declaration"]},
# ".scala": {"name": "scala",  "node_types": ["function_definition", "class_definition"]},
```

---

### Sync & State Management

#### File-Level Granularity on Sync
When a single function changes in a 50-function file, all chunks for that file are deleted and re-embedded from scratch. This is wasteful for large files with many stable functions.

**Fix:** Symbol-level diff — compare chunk hashes per symbol rather than per file, and only re-embed symbols whose content hash changed.

---

#### Single Branch Tracking
The sync system always tracks HEAD of the default branch. Feature branches and release branches are invisible.

**Fix:** Add an optional `--branch` flag to `add` and `sync`, store branch as part of the sync state key.

---

#### No Sync Automation
Keeping repos in sync requires manually running `python cli.py sync <repo>`. Stale embeddings silently accumulate as repos evolve.

**Planned fix:** A `sync --all` command plus a cron installer:
```bash
0 0 * * * cd ~/Desktop/code-intel && source .venv/bin/activate && python cli.py sync --all
```

---

### LLM & Context

#### Hard Token Cap
LLM context is capped at `LLM_CONTEXT_TOKEN_LIMIT = 4000` tokens. Complex architectural questions that would benefit from broader context are silently limited.

**Fix:** Make the cap configurable per query via a `--context-limit` CLI flag.

---

### Cross-Repo Intelligence

> **Identified as the most significant remaining architectural gap.**

Currently each repo is an isolated namespace in Milvus. Querying `--repo fastapi` strictly filters to chunks from that repo only. There is no way to ask cross-repo questions such as:

- *"How does `myapp` use `fastapi`'s dependency injection?"*
- *"Which of my services use the shared `auth-lib` token validator?"*

**Planned:**

1. **Cross-repo retrieval:** `--repos repo1,repo2` flag; no-repo default searches all indexed repos.
2. **Cross-repo import graph:** follow `import` edges across repo boundaries when both repos are indexed.

---

### Operational

#### Docker Desktop Required
Milvus runs as a Docker Compose stack (etcd + minio + milvus). If Docker Desktop crashes or is unavailable, the entire system is unavailable. There is no fallback vector store.

`dev.sh start` auto-launches Docker Desktop (polls up to 60s), but Docker Desktop must be installed.

**Fix:** Document a lightweight alternative (e.g., Milvus Lite for single-machine dev) or add a health-check startup warning with a clear error message.

---

#### Repos Directory Hardcoded
`REPOS_DIR` defaults to `~/Desktop/Repos/`. Repos located elsewhere require manually changing `config.py`, and there is no `--repos-dir` CLI override. This makes the tool non-portable across different machine setups.

**Fix:** Add a `--repo-path <absolute-path>` flag to `add` and `sync` that overrides `REPOS_DIR` per invocation, and document the config override.

---

#### No HTTPS for Web UI
The FastAPI web server is served over plain HTTP (`http://localhost:7860`). Auth tokens are transmitted in cleartext over the loopback interface. This is acceptable for purely local use but is a risk if the server is ever exposed on a LAN or public interface.

**Fix:** Document that the web UI must only be served locally, or add a TLS option via a reverse proxy (e.g., Caddy or nginx) for non-localhost deployments.

---

#### No API Rate Limiting
The FastAPI endpoints (`/query`, `/auth/login`, `/auth/register`) have no rate limiting. A misconfigured client or brute-force attempt could exhaust OpenAI / Voyage AI API quotas or hammer the local SQLite database.

**Fix:** Add a per-IP rate limiter (e.g., `slowapi`) on the `/query` endpoint and a stricter limit on `/auth/login` to prevent credential stuffing.

---

#### No Support for Jupyter Notebooks or Binary Files
Only plain `.py` source files are indexed. `.ipynb` notebooks, compiled `.pyc` files, configuration files (`.yaml`, `.toml`, `.json`), and documentation (`.md`, `.rst`) are silently skipped.

**Fix:** Add a `notebook` chunk type that extracts code cells from `.ipynb` files via the `nbformat` library, and optionally a `config_file` chunk type for structured config files.

---

## Resolved Limitations

The following limitations were addressed across all implementation phases.

| Limitation | Resolution |
|---|---|
| **Static Top-K Retrieval** | Adaptive Top-K: simple queries get 10 candidates / 5 final; complex queries get 20 / 8 |
| **No Re-Ranking** | Voyage `rerank-2.5-lite` cross-encoder re-scores all Milvus candidates |
| **No Query Expansion** | GPT-4o-mini generates 2 alternative phrasings; all searched in parallel, merged, deduped |
| **Hallucination Risk** | Confidence threshold: if all scores < `MIN_RETRIEVAL_SCORE = 0.5`, returns "not found" instead of forwarding weak context |
| **No Conversation Memory** | Conversation sessions backed by SQLite; prior turns passed as OpenAI message history to gpt-4.1 |
| **No Source Citation** | gpt-4.1 system prompt requires `[C1]`, `[C2]` inline citations for direct hits; `[G1]`, `[G2]` for graph-expanded dependency chunks |
| **No Authentication** | Email + password auth with PBKDF2-HMAC-SHA256 (260,000 iterations), UUID tokens, 30-day expiry |
| **No Observability** | `query_log` table in SQLite records every ask: question, repo, scores, latency, tokens, answer snippet |
| **Split Parts Lose Context** | Sliding window overlap: each `split_part` shares 10 lines with the previous part |
| **Only Functions/Classes Indexed** | `module_level` chunk type captures top-level constants, type aliases, and expressions per file |
| **No Docstring Awareness** | `docstring` chunks extracted separately from each function/class; embeds closer to natural language queries |
| **No Web UI** | FastAPI `app.py` + `static/index.html`: full browser UI with auth, register, markdown-rendered answers, session continuity, new-chat button, repo scoping |
| **Session tracking missing from web API** | `/query` endpoint creates/reuses sessions, appends turns to `conv_turns`, logs every request via `log_query()` |
| **Raw markdown in web answers** | Custom `renderMarkdown()` renderer handles fenced code blocks, lists, bold, italic, inline code, paragraphs |
| **High query latency (11–15s)** | Reduced to 3–7s via: single multi-vector Milvus search (one gRPC), `consistency_level="Eventually"`, batched `embed_queries()`, parallel expansion+embed via `ThreadPoolExecutor`, module-level collection caching |
| **SQLite missing indexes** | Added indexes on `auth_tokens`, `conv_sessions`, `conv_turns`, `query_log`, `import_edges`, `call_edges` |
| **No Cross-File Context** | Graph-augmented retrieval: `_expand_with_graph()` follows call edges after vector search, fetches callee chunks, labels them `[G1]`, `[G2]`, … — up to 3 dependency chunks added per query |
| **No Call Graph or Symbol Graph** | `core/graph.py` extracts `import_statement` and bare function call edges from the tree-sitter AST at index time, persists to SQLite `import_edges` and `call_edges` tables |
| **No Rename Detection** | `diff_tracker.py` detects `change_type="R"` git diff entries; fetches existing embeddings from Milvus via `fetch_chunks_for_file()`, re-inserts with updated `file_path` — zero Voyage API calls |
| **Single Milvus Collection** | Per-repo Milvus partitions via `partition_name()` and `ensure_partition()`; all search, query, and insert calls scoped to the repo's partition for query isolation and scalability |
| **Ephemeral Query Expansion Cache** | Two-level cache: L1 in-process dict + L2 SQLite `query_expansion_cache` (SHA-256 keyed) — survives restarts, shared between CLI and web server |
| **No Streaming Responses** | `ask_stream()` in `core/llm.py` uses OpenAI `stream=True`; `/query` endpoint streams via SSE (`text/event-stream`); CLI uses blocking `ask()` with immediate display |
| **No Automated Test Suite** | 85 pytest tests across 5 files covering chunker (20), graph (18), query_expander (15), diff_tracker (16), vector_store (16) — all passing in 1.3s with no external services required |

---

## Priority Roadmap

### Near term

| Priority | Item | Effort |
|---|---|---|
| High | Cross-repo retrieval (`--repos repo1,repo2`) | Low |
| Medium | `sync --all` + cron installer | Low |
| Medium | Symbol-level diff sync | Medium |
| Medium | Method call tracking in graph (`obj.method()`) | Medium |

### Future

| Priority | Item | Effort |
|---|---|---|
| Low | Java / Scala support | Low (already stubbed) |
| Low | `--branch` flag for add/sync | Low |
| Low | API rate limiting (`slowapi`) | Low |
| Low | Jupyter notebook indexing | Medium |
| Low | `--repo-path` CLI override for REPOS_DIR | Low |
| Low | Multi-hop graph expansion | Medium |
