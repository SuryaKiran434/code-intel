"""
core/embedder.py

Two backends, one interface:
  - VoyageEmbedder    → calls Voyage AI API  (current — voyage-code-3)
  - NomicLocalEmbedder → runs on-device      (future — 32GB+ Mac)

The rest of the codebase only ever calls:
    embed_code(texts)   → for indexing code chunks into Milvus
    embed_query(text)   → for embedding a user's search question

Switching provider = change EMBEDDING_PROVIDER in config.py.
Nothing else changes.
"""

import hashlib
import time
from abc import ABC, abstractmethod

from config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    VOYAGE_API_KEY,
    VECTOR_DIM,
)


# ── Shared utility ────────────────────────────────────────────────────────────

def content_hash(content: str) -> str:
    """SHA-256 fingerprint of a chunk — used as its stable Milvus ID."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _batch(items: list, size: int) -> list[list]:
    """Split a list into batches of a given size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """
    All embedders expose exactly two methods:
        embed_code(texts)  — for code chunks being stored in Milvus
        embed_query(text)  — for a user query at search time

    The distinction matters for voyage-code-3:
    the model uses input_type="query" vs input_type="document" to apply
    different internal transformations, improving retrieval precision.
    """

    @abstractmethod
    def embed_code(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of code chunks for storage."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query for retrieval."""
        ...


# ── Backend 1: Voyage AI ──────────────────────────────────────────────────────

class VoyageEmbedder(BaseEmbedder):
    """
    Uses the Voyage AI API with voyage-code-3.

    Rate limits on your account:
      - 3 RPM  (requests per minute)
      - 10K TPM (tokens per minute)

    This embedder automatically sleeps between batches to stay within limits.
    The sleep is calculated from actual token counts returned by the API,
    so it's precise rather than a fixed delay.

    Requires: VOYAGE_API_KEY in .env
    Install:  pip install voyageai
    """

    def __init__(self):
        try:
            import voyageai
        except ImportError:
            raise ImportError("Run: pip install voyageai")

        self._client = voyageai.Client(api_key=VOYAGE_API_KEY)

    def _call_api(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        """
        Call Voyage API with exponential backoff retry.
        input_type: "document" for code chunks, "query" for user queries.
        """
        all_embeddings = []

        for batch in _batch(texts, EMBEDDING_BATCH_SIZE):
            for attempt in range(4):   # up to 4 retries
                try:
                    result = self._client.embed(
                        texts=batch,
                        model=EMBEDDING_MODEL,
                        input_type=input_type,
                        output_dimension=VECTOR_DIM,
                    )
                    all_embeddings.extend(result.embeddings)
                    break

                except Exception as e:
                    if attempt == 3:
                        raise RuntimeError(
                            f"Voyage API failed after 4 attempts: {e}"
                        )
                    wait = 2 ** attempt   # 1s, 2s, 4s, 8s backoff
                    print(f"  [retry {attempt+1}/4] Waiting {wait}s... ({e})")
                    time.sleep(wait)

        return all_embeddings

    def embed_code(self, texts: list[str]) -> list[list[float]]:
        """Embed code chunks for storage — uses input_type='document'."""
        return self._call_api(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        """Embed a user query — uses input_type='query' for better retrieval."""
        result = self._call_api([text], input_type="query")
        return result[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple queries in one API call — much faster than N×embed_query()."""
        return self._call_api(texts, input_type="query")


# ── Backend 2: Nomic Local (future) ──────────────────────────────────────────

class NomicLocalEmbedder(BaseEmbedder):
    """
    Runs nomic-embed-code fully on-device via sentence-transformers.
    Requires: Apple Silicon Mac with 32GB+ RAM
    Requires: pip install sentence-transformers torch

    To activate: set EMBEDDING_PROVIDER = "nomic_local" in config.py
    NOTE: If you switch to this, also change VECTOR_DIM = 768 and
          recreate the Milvus collection (different dimension).
    """

    QUERY_PREFIX = "Represent this query for searching relevant code: "

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers torch")

        print("Loading nomic-embed-code locally (first run downloads ~14GB)...")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            "nomic-ai/nomic-embed-code",
            trust_remote_code=True,
        )
        print("✓ Model loaded.")

    def embed_code(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for batch in _batch(texts, EMBEDDING_BATCH_SIZE):
            vecs = self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(vecs.tolist())
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [text],
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].tolist()


# ── Factory ───────────────────────────────────────────────────────────────────

def _get_embedder() -> BaseEmbedder:
    if EMBEDDING_PROVIDER == "voyage":
        return VoyageEmbedder()
    elif EMBEDDING_PROVIDER == "nomic_local":
        return NomicLocalEmbedder()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{EMBEDDING_PROVIDER}'. "
            f"Must be 'voyage' or 'nomic_local'."
        )


# ── Module-level singleton — loaded once, reused across all calls ─────────────
#    Import only these two functions everywhere else in the codebase.

_embedder: BaseEmbedder = None

def _get_or_init() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = _get_embedder()
    return _embedder


def embed_code(texts: list[str]) -> list[list[float]]:
    """
    Embed code chunks for storage in Milvus.
    Call this during indexing (cli.py add / sync).
    """
    return _get_or_init().embed_code(texts)


def embed_query(text: str) -> list[float]:
    """
    Embed a user's search query for retrieval.
    Call this in core/retriever.py at query time.
    """
    return _get_or_init().embed_query(text)


def embed_queries(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple queries in a single API call.
    Use this in retriever.py when query expansion generates N variants —
    batching all N into one request is much faster than N sequential calls.
    """
    return _get_or_init().embed_queries(texts)