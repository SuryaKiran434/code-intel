import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────────
# PROJECT_DIR anchors to this file's location so the project is self-locating
# regardless of where it's checked out.
PROJECT_DIR     = Path(__file__).resolve().parent
SYNC_STATE_PATH = PROJECT_DIR / ".sync_state.json"

# REPOS_DIR defaults to ~/Desktop/Repos but can be overridden with the
# CODE_INTEL_REPOS_DIR env var (absolute path) for machines that store
# cloned repos elsewhere.
REPOS_DIR       = Path(os.getenv("CODE_INTEL_REPOS_DIR") or (Path.home() / "Desktop" / "Repos"))

# ── Local data directory ───────────────────────────────────────────────────────
#   Persists across sessions: SQLite DB (users, sessions, query log), auth token.
CODE_INTEL_DIR         = Path.home() / ".code-intel"
DB_PATH                = CODE_INTEL_DIR / "code_intel.db"
AUTH_FILE              = CODE_INTEL_DIR / ".auth"
AUTH_TOKEN_EXPIRY_DAYS = 30
SESSION_MAX_TURNS      = 10    # max prior turns kept in conversation context

# Web registration is disabled by default. Set ALLOW_WEB_REGISTRATION=1 to
# enable POST /auth/register over HTTP. When no users exist yet, registration
# is always allowed so the first account can be bootstrapped.
ALLOW_WEB_REGISTRATION = os.getenv("ALLOW_WEB_REGISTRATION") == "1"

# ── Milvus ────────────────────────────────────────────────────────────────────
MILVUS_HOST     = "localhost"
MILVUS_PORT     = 19530
COLLECTION_NAME = "code_intel"
VECTOR_DIM      = 1024          # voyage-code-3 default dimension

# ── Embedding ─────────────────────────────────────────────────────────────────
#
#   EMBEDDING_PROVIDER controls everything.
#   To add a new backend later, add a class to core/embedder.py
#   and a new value here — nothing else needs to change.
#
EMBEDDING_PROVIDER   = "voyage"
EMBEDDING_MODEL      = "voyage-code-3"
EMBEDDING_BATCH_SIZE = 128      # Voyage max batch size — voyage-code-3: 3M TPM / 2000 RPM

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL              = "gpt-4.1"
LLM_MAX_TOKENS         = 1536
LLM_CONTEXT_TOKEN_LIMIT = 8000

# ── Upstream request timeouts (seconds) ───────────────────────────────────────
# Bound how long we wait on third-party APIs so a stuck connection doesn't
# block an SSE stream (or a CLI invocation) indefinitely.
OPENAI_TIMEOUT_SECONDS = 60.0   # gpt-4.1 streaming can run long; keep generous
VOYAGE_TIMEOUT_SECONDS = 30.0   # embed + rerank — should complete well under this

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SMALL_MAX_LINES  = 60
CHUNK_MEDIUM_MAX_LINES = 150
SPLIT_OVERLAP_LINES    = 10    # lines of overlap prepended to each split part

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5               # final results returned to the user / LLM

# ── Re-ranking ────────────────────────────────────────────────────────────────
#   Voyage rerank-2 re-scores a larger candidate pool and returns the best K.
#   Set RERANKER_ENABLED = False to skip re-ranking (faster, lower quality).
RERANKER_ENABLED     = True
RERANKER_MODEL       = "rerank-2.5-lite"
RETRIEVAL_CANDIDATE_K = 10      # candidates fetched from Milvus before re-rank
RETRIEVAL_FINAL_K     = 5       # final results kept after re-ranking

# ── Confidence threshold ───────────────────────────────────────────────────────
#   If all retrieved chunks score below this, return a "not found" message
#   instead of forwarding weak context to the LLM (reduces hallucination).
MIN_RETRIEVAL_SCORE = 0.5

# ── Query expansion ────────────────────────────────────────────────────────────
#   GPT-4o-mini generates N alternative phrasings of the user's query.
#   Each variant is searched independently; results are merged and re-ranked.
#   Set QUERY_EXPANSION_ENABLED = False to disable (faster, single-vector search).
QUERY_EXPANSION_ENABLED  = True
QUERY_EXPANSION_VARIANTS = 2        # number of extra query variants to generate
QUERY_EXPANSION_MODEL    = "gpt-4o-mini"

# ── Adaptive Top-K ─────────────────────────────────────────────────────────────
#   Complex queries (architectural / multi-concept) get a larger candidate pool.
#   Complexity is detected by word count and presence of trigger keywords.
COMPLEX_QUERY_CANDIDATE_K = 20      # Milvus candidates for complex queries
COMPLEX_QUERY_FINAL_K     = 8       # kept after re-ranking for complex queries
COMPLEX_QUERY_MIN_WORDS   = 15      # word count threshold
COMPLEX_QUERY_KEYWORDS    = {
    "architecture", "flow", "pipeline",
    "design", "pattern", "overview", "entire", "end-to-end",
    "relationship", "interact", "depend", "structure",
}

# ── Language Registry ─────────────────────────────────────────────────────────
#   Add Java/Scala here when ready — nothing else needs to change
LANGUAGE_REGISTRY = {
    ".py": {
        "name": "python",
        "node_types": ["function_definition", "class_definition"],
    },
    # ── Uncomment when ready ──────────────────────────────────────────────────
    # ".java": {
    #     "name":       "java",
    #     "node_types": ["method_declaration", "class_declaration"],
    # },
    # ".scala": {
    #     "name":       "scala",
    #     "node_types": ["function_definition", "class_definition"],
    # },
}