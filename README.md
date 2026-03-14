# Code Intel

> A local code intelligence system that indexes Git repositories, understands code structure using AST-aware chunking, stores vector embeddings in Milvus, and answers natural language questions about your codebase using gpt-4.1.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Authentication](#authentication)
- [Conversation Sessions](#conversation-sessions)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Embedding Model](#embedding-model)
- [Sync & Incremental Updates](#sync--incremental-updates)
- [Observability](#observability)
- [CLI Reference](#cli-reference)
- [Web UI](#web-ui)
- [Service Manager](#service-manager)
- [Configuration](#configuration)
- [Milvus Schema](#milvus-schema)

---

## Overview

Code Intel answers questions like:

- *"How does dependency injection work in this repo?"*
- *"Where is authentication handled?"*
- *"What does the `UserService` class do?"*
- *"How does the retry logic work end-to-end?"*

It works entirely from your local machine. Your code never leaves your environment except for the embedding and LLM API calls.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interface                             │
│                                                                     │
│         CLI (cli.py)                    Web UI (app.py)             │
│    python cli.py ask "..."          http://localhost:PORT           │
└───────────────┬─────────────────────────────┬───────────────────────┘
                │                             │
                ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Auth & Session Layer                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   auth.py    │    │  session.py  │    │    telemetry.py      │  │
│  │              │    │              │    │                      │  │
│  │ PBKDF2 hash  │    │ conversation │    │ per-query audit log  │  │
│  │ UUID tokens  │    │ history      │    │ latency + scores     │  │
│  │ 30-day expiry│    │ SQLite-backed│    │ SQLite-backed        │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Core Pipeline                              │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ diff_tracker │    │   chunker    │    │      retriever       │  │
│  │              │───▶│              │    │                      │  │
│  │ git pull     │    │ tree-sitter  │    │ query_expander       │  │
│  │ diff commits │    │ 5 chunk types│    │ multi-vector search  │  │
│  │ track state  │    │ overlap split│    │ rerank-2.5-lite      │  │
│  └──────────────┘    └──────┬───────┘    └──────────┬───────────┘  │
│                             │                        │              │
│                             ▼                        │              │
│                    ┌──────────────┐                  │              │
│                    │   embedder   │                  │              │
│                    │              │                  │              │
│                    │ voyage-code-3│                  │              │
│                    │ embed_code() │                  │              │
│                    │ embed_query()│                  │              │
│                    └──────┬───────┘                  │              │
│                           │                          │              │
│                           ▼                          ▼              │
│                  ┌────────────────┐        ┌─────────────────┐     │
│                  │  vector_store  │        │      llm        │     │
│                  │                │        │                 │     │
│                  │ Milvus insert  │        │ build_context() │     │
│                  │ dedup by hash  │        │ gpt-4.1 call    │     │
│                  │ per-repo part. │        │ [C][G] labels   │     │
│                  └──────┬─────────┘        └────────┬────────┘     │
│                                                                     │
│              ┌──────────────────────┐                               │
│              │       graph.py       │                               │
│              │                      │                               │
│              │ import/call edges    │                               │
│              │ SQLite-backed        │                               │
│              │ callee expansion     │                               │
│              └──────────────────────┘                               │
└─────────────────────────│────────────────────────────│─────────────┘
                          │                            │
                          ▼                            ▼
              ┌───────────────────────┐   ┌───────────────────────┐
              │   Milvus (Docker)     │   │    OpenAI API         │
              │                       │   │                       │
              │  HNSW + COSINE        │   │  gpt-4.1              │
              │  1024-dim vectors     │   │  gpt-4o-mini (expand) │
              │  port 19530           │   │  temp = 0.1           │
              └───────────────────────┘   └───────────────────────┘

              ┌───────────────────────┐   ┌───────────────────────┐
              │   Voyage AI API       │   │   SQLite (~/.code-intel)│
              │                       │   │                       │
              │  voyage-code-3        │   │  users + tokens       │
              │  rerank-2.5-lite      │   │  sessions + turns     │
              │  1024-dim output      │   │  query_log            │
              └───────────────────────┘   │  query_expansion_cache│
                                          │  import_edges         │
                                          │  call_edges           │
                                          └───────────────────────┘
```

---

### Query Flow

```
User question
      │
      ▼
[Auth check] get_current_user()
      │
      ▼
[Session load] load_turns(session_id)   ← prior conversation history (if --session)
      │
      ▼
[Query expansion + original embed] parallel via ThreadPoolExecutor  ← QUERY_EXPANSION_ENABLED
      │  GPT-4o-mini → N variants  |  embed_queries([original]) — run at same time
      ▼
[Embed variants] embed_queries(variants)      ← voyage-code-3, input_type="query", batched
      │
      ▼
[Milvus search] single call, all vectors      ← COSINE, adaptive top-K, Eventually consistent
      │  collection.search(data=[v1,v2,...vN]) — one RPC, merged + dedup by content hash
      ▼
[Re-rank] Voyage rerank-2.5-lite             ← RERANKER_ENABLED
      │
      ▼
[Confidence threshold] score ≥ MIN_RETRIEVAL_SCORE
      │
      ▼
[Graph expansion] get_callees() → retrieve callee chunks   ← when repo_name provided
      │  tags callee chunks retrieval_source="graph" → labelled [G1], [G2], ...
      ▼
[build_context()] token budget = 4,000 tokens  ← full → truncated → skip
      │  [C] chunks first, then [G] chunks
      ▼
gpt-4.1  system prompt + history + context + question
      │
      ▼
Answer with [C1][C2] (direct) and [G1][G2] (dependency) inline citations + source list
      │
      ▼
[Session save] append_turn(user, assistant)
[Telemetry]   log_query(latency, tokens, score, ...)
```

---

### Indexing Flow

```
Repo on disk (~/Desktop/Repos/<name>)
      │
      ▼
Walk all files → filter extensions (.py only currently)
      │
      ▼
tree-sitter AST parse each file
      │
      ├──────────────────────────────────────────────────────────────►
      │                                                               │
      ▼                                                               ▼
Extract function_definition / class_definition nodes        graph.py: extract import
                                                            and call edges → SQLite
      │
      ├── ≤ 60 lines  ──────────────────────► "full" chunk
      │
      ├── 61–150 lines ────────────────────► "full" chunk
      │                                      "summary" chunk (first 15 lines)
      │
      └── 151+ lines  ──────────────────────► "split_part" chunks
                       sliding 10-line overlap  (split at blank lines)
                                               "summary" chunk (first 15 lines)

Per symbol → also extract:
      └── docstring present? ─────────────► "docstring" chunk

Per file → collect all uncovered lines:
      └── module-level code? ─────────────► "module_level" chunk
      │
      ▼
SHA-256 content hash (16 chars) → Milvus primary key (deduplication)
      │
      ▼
Batch embed via voyage-code-3 (embed_code, batch size = 128)
      │
      ▼
Insert into Milvus collection "code_intel"
      │
      ▼
Save HEAD commit hash to .sync_state.json
```

---

### Incremental Sync Flow

```
python cli.py sync <repo>
      │
      ▼
git pull
      │
      ▼
git diff <last_commit>..HEAD
      │
      ▼
For each changed file:
  ├── Re-chunk file with tree-sitter
  ├── Get old chunk IDs for this file from Milvus
  ├── Delete stale IDs (hash no longer present)
  └── Insert new chunks (only genuinely new hashes)

For each deleted file:
  └── Delete all chunk IDs for that file

For renames:
  ├── Fetch existing embeddings for old path from Milvus
  ├── Delete old path chunks by ID
  └── Re-insert with updated file_path — zero Voyage API calls
      │
      ▼
Update .sync_state.json with new HEAD commit
```

> Unchanged files are **never re-embedded**. Only modified, added, or deleted files trigger API calls.

---

## Project Structure

```
~/Desktop/code-intel/
│
├── .env                        # API keys (never commit)
├── .sync_state.json            # Auto-managed: last synced commit per repo
│
├── config.py                   # All tuneable parameters in one place
├── requirements.txt
├── docker-compose.yml          # Milvus standalone (etcd + minio + milvus)
│
├── cli.py                      # Entry point — all CLI commands
├── app.py                      # Web UI backend (FastAPI + SSE streaming)
├── dev.sh                      # Service manager (auto Docker + venv)
├── reset_collection.py         # Drop and recreate Milvus collection
├── estimate_tokens.py          # Dry-run token estimator (zero API cost)
├── pytest.ini                  # Test runner configuration
│
├── core/
│   ├── db.py               # SQLite schema (users, sessions, query log, graph, cache)
│   ├── auth.py             # User registration, login, token management
│   ├── session.py          # Conversation session persistence
│   ├── telemetry.py        # Per-query observability log
│   ├── chunker.py          # tree-sitter AST chunking, 5 chunk types
│   ├── embedder.py         # Voyage AI embedder (pluggable backend)
│   ├── graph.py            # Import/call graph extraction and SQLite persistence
│   ├── vector_store.py     # Milvus: insert, delete, search, per-repo partitions
│   ├── retriever.py        # Query-time retrieval, expansion, re-ranking, graph expansion
│   ├── query_expander.py   # GPT-4o-mini query variant generation (L1+L2 cache)
│   ├── diff_tracker.py     # Git diff → incremental sync, rename detection
│   └── llm.py              # gpt-4.1 calls, token budget, [C]/[G] context assembly
│
└── tests/
    ├── conftest.py         # Shared fixtures (tmp_db, sample_py_file)
    ├── test_chunker.py     # AST chunking — all 5 chunk types, 3-tier strategy
    ├── test_graph.py       # Import/call graph extraction, SQLite round-trips
    ├── test_query_expander.py  # L1/L2 cache hit/miss, API failure handling
    ├── test_diff_tracker.py    # Sync state persistence, file-type filtering
    └── test_vector_store.py    # partition_name, ensure_partition, reinsert
```

**Local data directory:** `~/.code-intel/`
```
~/.code-intel/
├── code_intel.db    # SQLite: users, auth_tokens, conv_sessions, conv_turns,
│                    #         query_log, query_expansion_cache, import_edges, call_edges
└── .auth            # Persisted login token (chmod 600)
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| **Embedding model** | `voyage-code-3` (Voyage AI) | Best-in-class for code, 32K context, 1024-dim |
| **Re-ranker** | `rerank-2.5-lite` (Voyage AI) | Cross-encoder re-scoring for precision |
| **LLM** | `gpt-4.1` (OpenAI) | 1M token context, strong code reasoning |
| **Query expansion** | `gpt-4o-mini` (OpenAI) | Low-cost variant generation |
| **Vector DB** | Milvus standalone (Docker) | Local, fast, production-grade, free |
| **AST parser** | tree-sitter 0.23.x | Language-aware chunking, not naive splits |
| **Index type** | HNSW + COSINE | Approximate nearest-neighbour, auto-migrated from IVF_FLAT on first run |
| **Auth** | PBKDF2-HMAC-SHA256 + SQLite | stdlib only, OWASP 2023 iterations |
| **Local DB** | SQLite | Sessions, tokens, query log, zero deps |
| **CLI** | Click + Rich | Clean terminal output, tables, progress |
| **Python** | 3.13 (pyenv) | Latest stable |
| **Platform** | macOS Apple Silicon | M-series, 8GB RAM |

---

## Setup

### Prerequisites

- Docker Desktop running
- Python 3.13 via pyenv
- OpenAI API key
- Voyage AI API key
- Repos pre-cloned under `~/Desktop/Repos/`

### Install

```bash
cd ~/Desktop/code-intel

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Environment

```bash
# .env
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...
```

### Start Services

```bash
./dev.sh start        # starts Milvus + Attu GUI
```

---

## Usage

### Create an account

```bash
python cli.py register
```

### Sign in

```bash
python cli.py login
```

### Index a repo

```bash
python cli.py add fastapi
python cli.py add fastapi --force   # re-index from scratch
```

### Ask a question

```bash
# Scope to a specific repo
python cli.py ask "How does dependency injection work?" --repo fastapi

# Show retrieved chunks alongside the answer
python cli.py ask "What does the Router class do?" --repo fastapi --show-chunks

# Start a conversation session
python cli.py ask "How is auth handled?" --repo fastapi --new-session

# Continue a session (follow-up questions)
python cli.py ask "What about error handling?" --session <session-id>

# Increase retrieval breadth
python cli.py ask "Explain the full request lifecycle" --repo fastapi --top-k 10
```

### Sync after a repo update

```bash
python cli.py sync fastapi
```

### List all indexed repos

```bash
python cli.py list
```

### Remove a repo

```bash
python cli.py remove fastapi
python cli.py remove fastapi --yes   # skip confirmation
```

### Check system status

```bash
python cli.py status
```

### View query history

```bash
python cli.py log
python cli.py log --last 10
```

### List conversation sessions

```bash
python cli.py sessions
```

### Estimate token cost before indexing

```bash
python estimate_tokens.py fastapi   # dry-run, zero API tokens consumed
python estimate_tokens.py --all     # all repos
```

---

## Authentication

Code Intel uses local email + password authentication backed by SQLite. No external auth service is required.

- Passwords are hashed with **PBKDF2-HMAC-SHA256** (260,000 iterations, random salt — OWASP 2023 recommendation)
- Login produces a UUID token persisted to `~/.code-intel/.auth` (chmod 600)
- Tokens expire after 30 days (configurable via `AUTH_TOKEN_EXPIRY_DAYS`)
- `ask`, `log`, and `sessions` commands require a valid login

```bash
python cli.py register    # create account (email + password + name)
python cli.py login       # sign in — token saved to ~/.code-intel/.auth
python cli.py logout      # delete local token
python cli.py status      # shows logged-in user
```

---

## Conversation Sessions

Sessions persist conversation history across multiple `ask` invocations so follow-up questions have context.

```bash
# Start a named session (prints session ID)
python cli.py ask "How does FastAPI handle routing?" --repo fastapi --new-session

# Continue the session
python cli.py ask "What about path parameter validation?" --session <id>
python cli.py ask "And how are dependencies resolved?" --session <id>

# View all sessions
python cli.py sessions
```

- History is kept as OpenAI message dicts and prepended to each GPT-4o call
- Maximum `SESSION_MAX_TURNS = 10` turns are loaded (oldest dropped first)
- Sessions are stored in `~/.code-intel/code_intel.db`

---

## Chunking Strategy

Code Intel uses tree-sitter to parse code at the AST level, extracting semantic units rather than naive line windows.

### Symbol chunks (functions and classes)

| Chunk size | Types produced | Description |
|---|---|---|
| ≤ 60 lines | `full` | Entire function/class |
| 61–150 lines | `full` + `summary` | Full body + first 15 lines with truncation notice |
| 151+ lines | `split_part` × N + `summary` | Split at blank line boundaries with 10-line overlap |

### Additional chunk types (Phase 4)

| Type | Source | Purpose |
|---|---|---|
| `docstring` | First string literal of any function/class | Embeds closer to natural language queries |
| `module_level` | All top-level lines not inside any function/class | Constants, type aliases, imports, module expressions |

**Summary chunks** are excluded from retrieval by default but used as LLM context fallback for large symbols.

**Split overlap:** Each `split_part` shares the last 10 lines of the previous part, preventing loss of variable bindings and setup context across splits.

### Adding more languages

In `config.py`, uncomment the relevant entries in `LANGUAGE_REGISTRY`:

```python
LANGUAGE_REGISTRY = {
    ".py":    {"name": "python", "node_types": ["function_definition", "class_definition"]},
    # ".java":  {"name": "java",   "node_types": ["method_declaration", "class_declaration"]},
    # ".scala": {"name": "scala",  "node_types": ["function_definition", "class_definition"]},
}
```

Also install the corresponding tree-sitter grammar (`tree-sitter-java`, `tree-sitter-scala`) and add the lazy import in `core/chunker.py`. Zero other changes needed.

---

## Retrieval Pipeline

The retrieval pipeline runs several stages between the user's question and the LLM:

### 1. Query Expansion (Parallel)
GPT-4o-mini generates 2 alternative phrasings of the question. Query expansion and the original query embedding run **in parallel** via `ThreadPoolExecutor`, so the GPT-4o-mini call (~1-2s) overlaps with the Voyage embed call (~100ms). All variant embeddings are then batched into a single Voyage API call. This covers vocabulary mismatch between how developers ask questions and how code is written.

```
QUERY_EXPANSION_ENABLED  = True
QUERY_EXPANSION_VARIANTS = 2
QUERY_EXPANSION_MODEL    = "gpt-4o-mini"
```

### 2. Single Multi-Vector Milvus Search
All query vectors (original + variants) are sent in **one** `collection.search(data=[v1, v2, ..., vN])` call. Milvus returns a result set per vector; results are merged and deduplicated by `file_path::symbol_name`, keeping the highest score per chunk. This avoids N sequential round-trips over the gRPC connection, which would serialize due to PyMilvus's shared connection.

```
consistency_level = "Eventually"   # fastest for single-node Docker deployment
```

### 3. Adaptive Top-K
Complex queries (long questions or those containing architectural keywords) get a larger Milvus candidate pool:

```
Simple query:  RETRIEVAL_CANDIDATE_K = 10 candidates → RETRIEVAL_FINAL_K = 5
Complex query: COMPLEX_QUERY_CANDIDATE_K = 20        → COMPLEX_QUERY_FINAL_K = 8
```

Complexity triggers: ≥ 15 words, or keywords like `architecture`, `flow`, `pipeline`, `design`, `pattern`, `overview`, `entire`, `end-to-end`, `relationship`, `interact`, `depend`, `structure`.

### 4. Re-ranking
After Milvus returns candidates, Voyage `rerank-2.5-lite` re-scores them using a cross-encoder. This replaces cosine similarity scores with deeper relevance scores and reorders results significantly.

```
RERANKER_ENABLED = True
RERANKER_MODEL   = "rerank-2.5-lite"
```

### 5. Confidence Threshold
If all final chunks score below `MIN_RETRIEVAL_SCORE = 0.5`, the pipeline returns an empty result and the LLM reports "not enough information found" rather than hallucinating from weak context.

### 6. Graph-Augmented Expansion
When `repo_name` is provided, the retriever looks up each matched symbol's callees in the SQLite call graph and fetches those dependency chunks from Milvus. Up to 3 graph-expanded chunks are added, tagged `retrieval_source="graph"`. In `build_context()`, direct chunks are labelled `[C1]`, `[C2]`, … and graph chunks are labelled `[G1]`, `[G2]`, … so the LLM can distinguish semantic matches from structural dependencies.

```
Direct vector hits  → [C1], [C2], [C3], ...
Graph-expanded deps → [G1], [G2], [G3], ...   (callee functions/classes)
```

---

## Embedding Model

**Model:** `voyage-code-3`
**Provider:** Voyage AI
**Dimensions:** 1024
**Max context:** 32,768 tokens
**Rate limits (paid tier):** 3,000,000 TPM / 2,000 RPM

Two separate embedding functions are used throughout the codebase:

| Function | Used for | Voyage `input_type` |
|---|---|---|
| `embed_code(texts)` | Indexing code chunks | `"document"` |
| `embed_query(text)` | Single query embedding | `"query"` |
| `embed_queries(texts)` | Batch query embedding (expansion variants) | `"query"` |

This asymmetry is intentional — using the wrong function for queries degrades retrieval quality significantly. `embed_queries()` sends all N variant queries in one API call instead of N sequential calls, eliminating per-request overhead.

To switch embedding backends, change `EMBEDDING_PROVIDER` in `config.py`. Nothing else changes.

---

## Sync & Incremental Updates

Code Intel tracks the last indexed Git commit per repo in `.sync_state.json`.

When `sync` is run:

1. `git pull` fetches latest changes
2. `git diff <last_commit>..HEAD` identifies changed, added, deleted, and renamed files
3. Only changed files are re-chunked and re-embedded
4. Stale chunk IDs for those files are deleted from Milvus
5. New chunks are inserted
6. Sync state is updated to the new HEAD commit

**Unchanged functions are never re-embedded** — even in a heavily modified repo, only actually changed files trigger API calls.

### Future automation

```bash
# Add to crontab — syncs nightly at midnight
0 0 * * * cd ~/Desktop/code-intel && source .venv/bin/activate && python cli.py sync <repo>
```

---

## Observability

Every `ask` invocation is logged to the `query_log` table in `~/.code-intel/code_intel.db`:

| Field | Description |
|---|---|
| `timestamp` | UTC time of the query |
| `question` | Original question (before expansion) |
| `repo` | Repo scoped to, or `*` for global |
| `query_variants` | Number of expansion variants generated |
| `chunks_retrieved` | Chunks returned after re-ranking |
| `top_score` | Reranker score of the best chunk |
| `tokens_used` | Total GPT-4o tokens (prompt + completion) |
| `latency_ms` | Wall-clock time from retrieve() to answer |
| `answer_snippet` | First 200 chars of the GPT-4o answer |

View recent queries:

```bash
python cli.py log
python cli.py log --last 50
```

---

## CLI Reference

```
Auth
  python cli.py register              Create a new account
  python cli.py login                 Sign in (token saved to ~/.code-intel/.auth)
  python cli.py logout                Sign out

Indexing
  python cli.py add <repo>            Index a repo for the first time
    --force                           Force full reindex even if already indexed
  python cli.py sync <repo>           Incremental update (git pull + re-embed changes)
  python cli.py remove <repo>         Delete all embeddings for a repo
    --yes                             Skip confirmation prompt

Search
  python cli.py ask "<question>"      Ask a natural language question
    --repo <name>                     Scope to a specific repo
    --top-k <n>                       Chunks to retrieve (default: 5)
    --show-chunks                     Print retrieved chunks alongside the answer
    --new-session                     Start a conversation session (prints ID)
    --session <id>                    Continue an existing session

Repos & Status
  python cli.py list                  Table of all indexed repos with chunk stats
  python cli.py status                Milvus health, auth, embedding config, repos

History
  python cli.py log                   View recent query history
    --last <n>                        Number of entries (default: 20)
  python cli.py sessions              List conversation sessions
    --last <n>                        Number of sessions (default: 10)
```

---

## Web UI

The web UI is available via `app.py` (FastAPI) + `static/index.html`.

### Start the web server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Open `http://localhost:7860` in your browser.

### Features

- **Register / Login** — create an account and sign in with email + password
- **Ask questions** — same pipeline as the CLI (expansion, reranking, gpt-4.1)
- **Session continuity** — each browser tab maintains a session; answers reference prior questions
- **New chat** — reset to a fresh session at any time
- **Markdown rendering** — answers with headers, bullet lists, bold, code blocks, and inline code render correctly
- **Repo scoping** — the `/repos` endpoint lists all indexed repos; the UI can scope queries to a specific one
- **Source citations** — every answer shows `[C1]`, `[C2]` labels with file path, symbol name, line range, and reranker score

### API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/login` | No | Email + password → Bearer token |
| `POST` | `/auth/register` | No | Create account → Bearer token |
| `GET` | `/auth/me` | Yes | Current user profile |
| `POST` | `/query` | Yes | Ask a question, get answer + sources |
| `GET` | `/repos` | Yes | List all indexed repositories |
| `GET` | `/` | No | Serves `static/index.html` |

---

## Service Manager

`dev.sh` manages all services in one command.

```bash
./dev.sh              # Interactive menu
./dev.sh start        # Start all services
./dev.sh stop         # Stop all services
./dev.sh status       # Check what's running
```

**`start` does:**
1. Auto-launches Docker Desktop if not running (polls up to 60s)
2. Creates `.venv` and installs `requirements.txt` if missing, otherwise activates it
3. Starts Milvus (etcd + minio + milvus) — auto-recovers from stale networks
4. Starts Attu GUI
5. Starts the Web UI (uvicorn on port 7860)

**`stop` does:**
1. Stops Web UI, Attu, Milvus
2. Deactivates the Python venv
3. Quits Docker Desktop via `osascript`

**Attu** (Milvus GUI) is available at `http://localhost:8000`
Connect to: `host.docker.internal:19530` — no auth required

---

## Configuration

All parameters live in `config.py`.

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `voyage-code-3` | Embedding model name |
| `EMBEDDING_BATCH_SIZE` | `128` | Chunks per Voyage API call |
| `RERANKER_ENABLED` | `True` | Enable Voyage re-ranking |
| `RERANKER_MODEL` | `rerank-2.5-lite` | Voyage re-ranker model |
| `RETRIEVAL_CANDIDATE_K` | `10` | Milvus candidates before re-ranking |
| `RETRIEVAL_FINAL_K` | `5` | Final results after re-ranking |
| `COMPLEX_QUERY_CANDIDATE_K` | `20` | Candidates for complex queries |
| `COMPLEX_QUERY_FINAL_K` | `8` | Final results for complex queries |
| `MIN_RETRIEVAL_SCORE` | `0.5` | Minimum score threshold |
| `QUERY_EXPANSION_ENABLED` | `True` | Enable GPT-4o-mini query expansion |
| `QUERY_EXPANSION_VARIANTS` | `2` | Number of alternative queries |
| `LLM_MODEL` | `gpt-4.1` | OpenAI model |
| `LLM_CONTEXT_TOKEN_LIMIT` | `4000` | Max code context tokens sent to gpt-4.1 |
| `LLM_MAX_TOKENS` | `1536` | Max tokens in gpt-4.1 response |
| `CHUNK_SMALL_MAX_LINES` | `60` | Small/medium chunk threshold |
| `CHUNK_MEDIUM_MAX_LINES` | `150` | Medium/large chunk threshold |
| `SPLIT_OVERLAP_LINES` | `10` | Overlap lines between split parts |
| `SESSION_MAX_TURNS` | `10` | Max prior turns in conversation context |
| `AUTH_TOKEN_EXPIRY_DAYS` | `30` | Login token lifetime |
| `REPOS_DIR` | `~/Desktop/Repos/` | Where your repos live |
| `COLLECTION_NAME` | `code_intel` | Milvus collection name |

---

## Milvus Schema

Each chunk stored in Milvus has the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | VARCHAR(16) | SHA-256 content hash (primary key, deduplication key) |
| `embedding` | FLOAT_VECTOR(1024) | voyage-code-3 output |
| `content` | VARCHAR(65535) | Raw source code of the chunk |
| `file_path` | VARCHAR(1024) | Absolute path to the source file |
| `repo_name` | VARCHAR(256) | Repository folder name |
| `symbol_name` | VARCHAR(512) | Function or class name (tree-sitter extracted) |
| `start_line` | INT64 | Start line in source file (0-indexed) |
| `end_line` | INT64 | End line in source file (0-indexed) |
| `language` | VARCHAR(64) | Programming language |
| `chunk_type` | VARCHAR(32) | `full`, `split_part`, `summary`, `docstring`, `module_level` |
| `parent_symbol` | VARCHAR(512) | Original symbol name for split/summary/docstring chunks |

**Index:** HNSW with COSINE similarity (auto-migrated from IVF_FLAT on first run)
