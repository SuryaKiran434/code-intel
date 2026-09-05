# Code Intel

> A local code intelligence system that indexes Git repositories, understands code structure using AST-aware chunking, stores vector embeddings in Milvus, and answers natural language questions about your codebase using gpt-4.1.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Run It Locally](#run-it-locally)
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
│    python cli.py ask "..."          http://localhost:7860           │
└───────────────┬─────────────────────────────┬───────────────────────┘
                │                             │
                ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Auth & Session Layer (schema: core/db.py)              │
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
│  │              │───▶│              │    │ batched graph BFS    │  │
│  │ git pull     │    │ tree-sitter  │    │ query_expander       │  │
│  │ diff commits │    │ 5 chunk types│    │ multi-vector search  │  │
│  │ track state  │    │ overlap split│    │ rerank-2.5-lite      │  │
│  └──────────────┘    └──────┬───────┘    └──────────┬───────────┘  │
│                             │                        │              │
│                             ▼                        │              │
│                    ┌──────────────┐                  │              │
│                    │   embedder   │                  │              │
│                    │ 4-worker pool│                  │              │
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
│              │ get_callees_batch()  │                               │
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
              │   Voyage AI API       │   │  SQLite ~/.code-intel │
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
[Query expansion] expand_query(question)      ← QUERY_EXPANSION_ENABLED, skipped for ≤ 4 words
      │  GPT-4o-mini → N variants (L1 dict cache → L2 SQLite cache → API)
      ▼
[Embed] embed_queries([original] + variants)  ← voyage-code-3, input_type="query", one API call
      │
      ▼
[Milvus search] single call, all vectors      ← COSINE (ef=64), adaptive top-K, Eventually
      │  collection.search(data=[v1,v2,...vN]) — one RPC, scoped to the repo's partition
      │  results merged + deduped by file_path::symbol_name, highest score wins
      ▼
[Re-rank] Voyage rerank-2.5-lite             ← RERANKER_ENABLED
      │
      ▼
[Confidence threshold] score ≥ MIN_RETRIEVAL_SCORE
      │
      ▼
[Graph expansion] BFS over the call graph, batched per hop  ← when repo_name provided
      │  get_callees_batch() = 1 SQLite query per hop, then 1 Milvus query per hop
      │  tags callee chunks retrieval_source="graph" → labelled [G1], [G2], ...
      ▼
[build_context()] token budget = 8,000 tokens  ← full → truncated → skip
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
Repo on disk (REPOS_DIR/<name> — ~/Desktop/Repos by default)
      │
      ▼
Walk all files → filter extensions (.py only currently)
      │
      ▼
tree-sitter AST parse each file (parallel — ThreadPoolExecutor, up to 8 workers)
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
      │  batches dispatched over a 4-worker pool; order preserved
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
Single Milvus query: get_ids_by_file() → all chunk IDs for repo, grouped by path
      │    (one gRPC call replaces N per-file queries)
      ▼
For each deleted file:
  └── Delete all chunk IDs for that file from the in-memory map

For renames:
  ├── Fetch existing embeddings for old path from Milvus
  ├── Delete old path chunks by ID
  └── Re-insert with updated file_path — zero Voyage API calls

For each changed/added file:
  ├── Re-chunk file with tree-sitter
  ├── Compute SHA-256 hash for each new chunk
  ├── stale_ids  = old_chunk_ids_for_file − new_chunk_hashes   (symbol-level diff)
  └── truly_new  = new_chunks whose hash doesn't exist anywhere in the repo
      │
      ▼
Batch delete stale IDs → batch insert truly_new chunks
      │
      ▼
Update .sync_state.json with new HEAD commit
```

> Unchanged files are **never re-embedded**. Unchanged symbols **within** changed files are also skipped — only symbols whose content hash is new trigger Voyage API calls.

---

## Project Structure

```
~/IdeaProjects/code-intel/
│
├── .env                        # API keys (never commit)
├── .env.example                # Template for .env — copy and fill in
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
├── pyproject.toml              # Ruff lint configuration
├── sonar-project.properties    # SonarCloud sources, tests, coverage report path
├── LIMITATIONS.md              # Known gaps and trade-offs
│
├── static/
│   └── index.html              # Web UI single-page frontend
│
├── .github/
│   ├── dependabot.yml          # Weekly grouped pip + github-actions updates
│   └── workflows/
│       ├── test.yml            # CI: pytest (3.13) + coverage + ruff lint + SonarCloud
│       ├── dependabot-auto-merge.yml  # Queues grouped minor/patch bumps to merge
│       └── slack-notify.yml
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
    ├── test_diff_tracker_git.py # Git-backed sync: baselines, unreachable SHAs, renames
    ├── test_vector_store.py    # partition_name, ensure_partition, reinsert
    ├── test_retriever.py       # Filters, adaptive top-K, rerank, graph expansion
    ├── test_embedder.py        # Batching, worker pool, order preservation
    ├── test_app_query_errors.py # /query streaming failures never leak upstream text
    └── test_app_repos_errors.py # GET /repos failures never leak the sync-state path
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
| **AST parser** | tree-sitter 0.26.x | Language-aware chunking, not naive splits |
| **Index type** | HNSW + COSINE | Approximate nearest-neighbour, auto-migrated from IVF_FLAT on first run |
| **Auth** | PBKDF2-HMAC-SHA256 + SQLite | stdlib only, OWASP 2023 iterations |
| **Local DB** | SQLite | Sessions, tokens, query log, zero deps |
| **CLI** | Click + Rich | Clean terminal output, tables, progress |
| **Python** | 3.13 (pyenv) | Latest stable |
| **Platform** | macOS Apple Silicon | M-series, 8GB RAM |

---

## Run It Locally

Everything below works from a clean machine. Steps 1–7 are the full path from
nothing installed to an answered question.

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.13** | `pyproject.toml` targets `py313` and CI runs `pytest (3.13)`. Earlier 3.x may work but is untested. |
| **Docker** | Docker Desktop on macOS, or any Docker Engine with `docker compose`. Milvus runs in containers. |
| **Git** | Used by `sync` (`git pull`, `git diff`) via GitPython. |
| **OpenAI API key** | Answers (`gpt-4.1`) and query expansion (`gpt-4o-mini`). |
| **Voyage AI API key** | Embeddings (`voyage-code-3`) and re-ranking (`rerank-2.5-lite`). |
| **~2 GB free disk** | Milvus + etcd + MinIO container volumes. |

`dev.sh` is macOS-oriented — it launches and quits Docker Desktop via `open -a Docker`
and `osascript`. On Linux, start Docker yourself and use the raw `docker compose`
and `uvicorn` commands shown below.

### 2. Clone and create a virtualenv

```bash
git clone https://github.com/SuryaKiran434/code-intel.git
cd code-intel

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

All dependencies are pinned in `requirements.txt`. Bump them intentionally
rather than with `pip install -U`.

### 3. Configure environment variables

Copy the template and fill in your two keys:

```bash
cp .env.example .env
```

```bash
# .env
OPENAI_API_KEY=sk-...          # required
VOYAGE_API_KEY=pa-...          # required

# Optional
CODE_INTEL_REPOS_DIR=/absolute/path/to/your/repos   # default: ~/Desktop/Repos
ALLOW_WEB_REGISTRATION=1                            # default: off (see Web UI)
```

`config.py` loads `.env` through `python-dotenv`. `.env` is gitignored — never commit it.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | `gpt-4.1` answers, `gpt-4o-mini` query expansion |
| `VOYAGE_API_KEY` | Yes | — | `voyage-code-3` embeddings, `rerank-2.5-lite` re-ranking |
| `CODE_INTEL_REPOS_DIR` | No | `~/Desktop/Repos` | Absolute path to the directory holding the repos you want indexed |
| `ALLOW_WEB_REGISTRATION` | No | unset (off) | Set to `1` to allow `POST /auth/register` over HTTP |

### 4. Start Milvus

```bash
docker compose up -d          # etcd + MinIO + Milvus standalone
docker compose ps             # all three should be running
```

Milvus needs a few seconds before it accepts connections on `19530`. The
collection and its HNSW/COSINE index are created automatically on first use —
there is no separate schema step.

Or let the service manager do everything (macOS):

```bash
./dev.sh start                # Docker Desktop + venv + Milvus + Attu + Web UI
```

### 5. Create an account and index your first repo

`add` indexes a repo that is **already cloned** under `REPOS_DIR` — it does not
clone for you. Point `CODE_INTEL_REPOS_DIR` at wherever your repos live, or use
the default `~/Desktop/Repos`:

```bash
mkdir -p ~/Desktop/Repos
git clone https://github.com/pallets/click.git ~/Desktop/Repos/click

python cli.py register        # email + password + name, stored in ~/.code-intel/code_intel.db
python cli.py login           # token written to ~/.code-intel/.auth

python cli.py add click       # chunk → embed → insert into Milvus
```

Only `.py` files are indexed today (see [Chunking Strategy](#chunking-strategy)).
Want a cost estimate before spending Voyage tokens?

```bash
python estimate_tokens.py click     # dry run, zero API calls
```

### 6. Ask a question

```bash
python cli.py ask "How are commands registered?" --repo click
python cli.py status                 # Milvus health, auth, embedding config, indexed repos
```

### 7. Start the web UI

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Open `http://localhost:7860` and sign in with the account you created in step 5.
Registration through the browser is disabled unless `ALLOW_WEB_REGISTRATION=1`
(the first-ever account is always allowed, so a fresh install can bootstrap itself).

### Ports

| Port | Service | Started by |
|---|---|---|
| `19530` | Milvus gRPC | `docker compose` |
| `9091` | Milvus metrics / health | `docker compose` |
| `7860` | Web UI (uvicorn) | `uvicorn` / `./dev.sh start` |
| `8000` | Attu (Milvus GUI) | `./dev.sh start` — `docker run zilliz/attu` |

etcd and MinIO run inside the compose network and publish no host ports.

### Running the tests

The suite is fully offline — every external service (Voyage, OpenAI, Milvus,
SQLite paths) is stubbed. The API-key variables only need to be *present* so the
module-level clients can be constructed at import time; they are never used to
make a call.

```bash
source .venv/bin/activate
OPENAI_API_KEY=test-openai-key VOYAGE_API_KEY=test-voyage-key pytest -v
```

**150 tests, all passing.**

| File | Tests | Covers |
|---|---|---|
| `tests/test_graph.py` | 24 | Import/call extraction, `get_callees_batch`, SQLite round-trips |
| `tests/test_diff_tracker_git.py` | 23 | Git-backed sync: baseline resolution, unreachable SHAs, rename detection |
| `tests/test_chunker.py` | 20 | All 5 chunk types, 3-tier strategy, split overlap |
| `tests/test_diff_tracker.py` | 16 | Sync state persistence, file-type filtering |
| `tests/test_vector_store.py` | 16 | `partition_name`, `ensure_partition`, reinsert |
| `tests/test_retriever.py` | 15 | Filters, adaptive top-K, rerank fallback, graph expansion |
| `tests/test_query_expander.py` | 14 | L1/L2 cache hit/miss, API failure handling |
| `tests/test_app_query_errors.py` | 8 | `/query` streaming failures return the generic message, never upstream text |
| `tests/test_embedder.py` | 7 | Batching, worker pool, order preservation |
| `tests/test_app_repos_errors.py` | 7 | `GET /repos` failures do not leak the sync-state file path |

#### Coverage

```bash
pytest --cov=core --cov=app --cov=cli --cov-report=term
```

The `--cov` list must name every path in `sonar.sources`, not just `core`. SonarCloud
scores a declared source file that is **absent** from `coverage.xml` as 0% covered
rather than as unmeasured, so measuring only `core` reported `app.py` as 0/110 while
the two `test_app_*` suites above were already exercising its error handlers — and the
quality gate failed on a number the tests had in fact already earned. `app.py` reports
81%; `cli.py` has no tests yet and so is still absent from the report.

Lint with the same rule set CI uses:

```bash
pip install ruff && ruff check .
```

CI (`.github/workflows/test.yml`) runs two required checks — **`pytest (3.13)`** and
**`lint`** — plus a SonarCloud scan that consumes `coverage.xml`. The scan is advisory:
it runs `continue-on-error`, so a SonarCloud outage never blocks a merge.

### Stopping and cleaning up

```bash
docker compose down           # stop Milvus (volumes are kept)
docker compose down -v        # also delete the indexed vectors

./dev.sh stop                 # macOS: stop Web UI + Attu + Milvus, quit Docker Desktop
python reset_collection.py    # drop and recreate the Milvus collection only
```

Local state outside Docker lives in `~/.code-intel/` (SQLite DB + auth token)
and `.sync_state.json` in the project root.

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

- History is kept as OpenAI message dicts and prepended to each gpt-4.1 call
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

**Summary chunks** are excluded from retrieval by default (`chunk_type != "summary"` is added to every search expression). They are stored so a large symbol still has a compact representation available, and can be pulled in explicitly via `include_summaries=True`. `build_context()` does not swap them in automatically — an oversized chunk is truncated to its first 30 lines instead.

**Split overlap:** Parts are `CHUNK_MEDIUM_MAX_LINES` (150) lines wide, cut at the nearest blank line within 20 lines of the boundary. Each `split_part` shares the last 10 lines of the previous part, preventing loss of variable bindings and setup context across splits.

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

### 1. Query Expansion
GPT-4o-mini generates 2 alternative phrasings of the question, covering the vocabulary mismatch between how developers ask questions and how code is written. Results are cached twice — an in-process dict (L1) and a SQLite table (L2, shared between the CLI and the web UI) — so a repeated question skips the API call entirely. Failures are not cached and degrade silently to single-query retrieval.

Expansion is **skipped for queries of 4 words or fewer**: short queries are already specific, and variants add noise rather than signal.

The original question and every variant are then embedded in a **single** Voyage call — `embed_queries([original] + variants)` — rather than one request per string.

```
QUERY_EXPANSION_ENABLED  = True
QUERY_EXPANSION_VARIANTS = 2
QUERY_EXPANSION_MODEL    = "gpt-4o-mini"
```

### 2. Single Multi-Vector Milvus Search
All query vectors (original + variants) are sent in **one** `collection.search(data=[v1, v2, ..., vN])` call. Milvus returns a result set per vector; results are merged and deduplicated by `file_path::symbol_name`, keeping the highest score per chunk. This avoids N sequential round-trips over the gRPC connection, which would serialize due to PyMilvus's shared connection.

When `--repo` is given, the search is scoped to that repo's Milvus partition (with a fallback to a `repo_name ==` expression clause for collections indexed before partitions existed). Summary chunks are excluded via `chunk_type != "summary"`.

```
search params      = {"metric_type": "COSINE", "params": {"ef": 64}}
consistency_level  = "Eventually"   # fastest for single-node Docker deployment
```

### 3. Adaptive Top-K
Complex queries (long questions or those containing architectural keywords) get a larger Milvus candidate pool:

```
Simple query:  RETRIEVAL_CANDIDATE_K = 10 candidates → RETRIEVAL_FINAL_K = 5
Complex query: COMPLEX_QUERY_CANDIDATE_K = 20        → COMPLEX_QUERY_FINAL_K = 8
```

Complexity triggers: ≥ 15 words, or keywords like `architecture`, `flow`, `pipeline`, `design`, `pattern`, `overview`, `entire`, `end-to-end`, `relationship`, `interact`, `depend`, `structure`.

### 4. Re-ranking
Before calling the cross-encoder, a score-gap pre-filter drops candidates whose cosine similarity is more than `0.35` below the best match — reducing reranker cost and latency without losing relevant results. The remaining candidates are re-scored by Voyage `rerank-2.5-lite`, which replaces cosine similarity with deeper relevance scores and reorders results significantly.

If the pre-filter leaves fewer than `final_k` candidates, the fallback takes the top `final_k * 2` **by score** (`heapq.nlargest`). The merged candidate list is in Milvus return order across query vectors, not score order, so slicing it would silently drop the best candidates.

When re-ranking is disabled, the top `final_k` chunks are selected by cosine score the same way.

```
RERANKER_ENABLED = True
RERANKER_MODEL   = "rerank-2.5-lite"
```

### 5. Confidence Threshold
If all final chunks score below `MIN_RETRIEVAL_SCORE = 0.5`, the pipeline returns an empty result and the LLM reports "not enough information found" rather than hallucinating from weak context.

### 6. Graph-Augmented Expansion
When `repo_name` is provided, the retriever expands direct results using a BFS over the call graph. Simple queries follow **1 hop** (direct callees, up to 3 graph chunks); complex queries follow **2 hops** (callees of callees, up to 5 graph chunks). The call graph tracks both bare-name calls (`foo()`) and attribute method calls (`obj.method()`).

**Each BFS hop costs exactly two round trips**, regardless of frontier size:

1. `get_callees_batch(repo, [(file, symbol), ...])` — one SQLite statement resolving every frontier chunk's callees at once, using a `(from_file, from_symbol) IN (VALUES ...)` clause with fully bound parameters.
2. One Milvus `query()` with `symbol_name in ["a", "b", ...]`, fetching every callee chunk for the hop in a single call. Interpolated values are escaped so a symbol containing a quote cannot terminate the expression literal.

The earlier implementation issued one SQLite query per chunk plus one Milvus query per callee. A depth-2 expansion went from 20+ sequential round trips to 4.

Graph-expanded chunks are tagged `retrieval_source="graph"`. In `build_context()`, direct chunks are labelled `[C1]`, `[C2]`, … and graph chunks are labelled `[G1]`, `[G2]`, … so the LLM can distinguish semantic matches from structural dependencies.

```
Direct vector hits  → [C1], [C2], [C3], ...
Graph-expanded deps → [G1], [G2], [G3], ...   (callee functions/classes, 1–2 hops)
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

### Batching

Texts are split into batches of `EMBEDDING_BATCH_SIZE` (128). When there is more than one batch, the batches are dispatched concurrently over a bounded `ThreadPoolExecutor` — `EMBED_MAX_WORKERS = 4` in `core/embedder.py`. The calls are network-bound, so a sequential loop leaves the process idle; the pool size is deliberately small so the account's rate limits still hold.

Results are reassembled in **input order** (`pool.map` yields in submission order), so chunk-to-vector alignment is preserved regardless of completion order. Each batch is attempted up to 4 times, with exponential backoff (1s, 2s, 4s) between attempts.

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
0 0 * * * cd ~/IdeaProjects/code-intel && source .venv/bin/activate && python cli.py sync <repo>
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
| `tokens_used` | Total gpt-4.1 tokens (prompt + completion) |
| `latency_ms` | Wall-clock time from retrieve() to answer |
| `answer_snippet` | First 200 chars of the gpt-4.1 answer |

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
    --context-limit <n>               Max context tokens sent to gpt-4.1 (default: 8000, range: 1000-32000)

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
python -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Open `http://localhost:7860` in your browser. `./dev.sh start` runs the same
server in the background (`nohup`, logging to `.webui.log`, PID in `.webui.pid`).

### Features

- **Register / Login** — sign in with email + password (registration gated by `ALLOW_WEB_REGISTRATION`)
- **Ask questions** — same pipeline as the CLI (expansion, reranking, graph expansion, gpt-4.1), streamed token-by-token over SSE
- **Session continuity** — each browser tab maintains a session; answers reference prior questions
- **New chat** — reset to a fresh session at any time
- **Markdown rendering** — answers with headers, bullet lists, bold, code blocks, and inline code render correctly
- **Repo scoping** — the `/repos` endpoint lists all indexed repos; the UI can scope queries to a specific one
- **Source citations** — every answer shows `[C1]`, `[C2]` labels with file path, symbol name, line range, and reranker score

### API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/login` | No | Email + password → Bearer token |
| `POST` | `/auth/register` | No | Create account → Bearer token (gated, see below) |
| `POST` | `/auth/logout` | Yes | Revoke the caller's Bearer token (idempotent) |
| `GET` | `/auth/me` | Yes | Current user profile |
| `POST` | `/query` | Yes | Ask a question — streams the answer as SSE |
| `GET` | `/repos` | Yes | List all indexed repositories (read from `.sync_state.json`) |
| `GET` | `/` | No | Serves `static/index.html` |
| `GET` | `/static/*` | No | Static assets |

**Web registration is disabled by default.** `POST /auth/register` returns 403 unless `ALLOW_WEB_REGISTRATION=1` is set — except when no users exist yet, so the very first account can always be bootstrapped. Otherwise, register with `python cli.py register`. Passwords must be at least 8 characters.

`POST /query` responds with `text/event-stream`. Each event is a JSON object on a `data:` line:

```
{"type": "token",  "text": "..."}                                  — one per streamed token
{"type": "done",   "sources": [...], "tokens": N, "session_id": "..."}
{"type": "error",  "message": "..."}
```

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

All parameters live in `config.py`, except `EMBED_MAX_WORKERS` (in `core/embedder.py`).
API keys and the two env-driven overrides come from `.env` — see [Run It Locally](#run-it-locally).

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `voyage` | Embedding backend — `voyage` or `nomic_local` |
| `EMBEDDING_MODEL` | `voyage-code-3` | Embedding model name |
| `EMBEDDING_BATCH_SIZE` | `128` | Chunks per Voyage API call |
| `EMBED_MAX_WORKERS` | `4` | Concurrent embed batches (`core/embedder.py`) |
| `VECTOR_DIM` | `1024` | Embedding dimension — must match the Milvus schema |
| `RERANKER_ENABLED` | `True` | Enable Voyage re-ranking |
| `RERANKER_MODEL` | `rerank-2.5-lite` | Voyage re-ranker model |
| `RETRIEVAL_CANDIDATE_K` | `10` | Milvus candidates before re-ranking |
| `RETRIEVAL_FINAL_K` | `5` | Final results after re-ranking |
| `COMPLEX_QUERY_CANDIDATE_K` | `20` | Candidates for complex queries |
| `COMPLEX_QUERY_FINAL_K` | `8` | Final results for complex queries |
| `MIN_RETRIEVAL_SCORE` | `0.5` | Minimum score threshold |
| `TOP_K_RESULTS` | `5` | Default `top_k` when the caller doesn't pass one |
| `COMPLEX_QUERY_MIN_WORDS` | `15` | Word count that marks a query complex |
| `QUERY_EXPANSION_ENABLED` | `True` | Enable GPT-4o-mini query expansion |
| `QUERY_EXPANSION_VARIANTS` | `2` | Number of alternative queries |
| `LLM_MODEL` | `gpt-4.1` | OpenAI model |
| `LLM_CONTEXT_TOKEN_LIMIT` | `8000` | Max code context tokens sent to gpt-4.1 (override per query with `--context-limit`) |
| `LLM_MAX_TOKENS` | `1536` | Max tokens in gpt-4.1 response |
| `CHUNK_SMALL_MAX_LINES` | `60` | Small/medium chunk threshold |
| `CHUNK_MEDIUM_MAX_LINES` | `150` | Medium/large chunk threshold |
| `SPLIT_OVERLAP_LINES` | `10` | Overlap lines between split parts |
| `SESSION_MAX_TURNS` | `10` | Max prior turns in conversation context |
| `AUTH_TOKEN_EXPIRY_DAYS` | `30` | Login token lifetime |
| `OPENAI_TIMEOUT_SECONDS` | `60.0` | Upstream timeout for OpenAI calls |
| `VOYAGE_TIMEOUT_SECONDS` | `30.0` | Upstream timeout for Voyage embed + rerank |
| `ALLOW_WEB_REGISTRATION` | `False` | Env-driven — set `ALLOW_WEB_REGISTRATION=1` to enable `POST /auth/register` |
| `REPOS_DIR` | `~/Desktop/Repos/` | Where your repos live — override with the `CODE_INTEL_REPOS_DIR` env var |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus endpoint |
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
