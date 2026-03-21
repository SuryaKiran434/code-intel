"""
core/vector_store.py

Handles all Milvus operations:
  - Collection creation with correct schema for voyage-code-3 (dim=1024)
  - Per-repo Milvus partitions for query isolation and scalability
  - Chunk indexing with deduplication via content hash
  - Repo-scoped deletion for sync operations
  - ID-level deletion for surgical diff-based updates
  - Full-record fetch (including embeddings) for zero-cost rename handling
"""

import re

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    VECTOR_DIM,
    EMBEDDING_BATCH_SIZE,
)
from core.chunker import CodeChunk
from core.embedder import embed_code, content_hash
from rich.progress import track
from rich.console import Console

console = Console()


# ── Connection ─────────────────────────────────────────────────────────────────

def connect():
    """Connect to Milvus. Safe to call multiple times — no-ops if already connected."""
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )


# ── Partition naming ────────────────────────────────────────────────────────────

def partition_name(repo_name: str) -> str:
    """
    Sanitize a repo name into a valid Milvus partition name.
    Milvus partitions: alphanumeric + underscores, cannot start with a digit.
    """
    name = re.sub(r"[^a-zA-Z0-9_]", "_", repo_name)
    if name and name[0].isdigit():
        name = "r_" + name
    return name or "_default"


def ensure_partition(collection: Collection, repo_name: str) -> str:
    """
    Create the repo's partition if it doesn't exist yet.
    Returns the sanitized partition name.
    """
    pname = partition_name(repo_name)
    if not collection.has_partition(pname):
        collection.create_partition(pname)
    return pname


# ── Schema & Collection ────────────────────────────────────────────────────────

def _migrate_to_hnsw_if_needed(collection: Collection):
    """
    If the vector index is IVF_FLAT, migrate it to HNSW in-place.
    Data is never touched — only the index is rebuilt.

    Milvus requires the collection to be released (unloaded from memory)
    before an index can be dropped. Sequence: release → drop → create → load.
    """
    try:
        index_params = collection.index().params
    except Exception:
        return  # No index yet or can't read params — nothing to migrate

    if index_params.get("index_type") != "IVF_FLAT":
        return

    console.print("[yellow]Migrating Milvus index IVF_FLAT → HNSW (one-time, data is preserved)...[/yellow]")
    try:
        collection.release()          # must unload before dropping index
        collection.drop_index()
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            },
        )
        collection.load()             # reload with the new HNSW index
        console.print("[green]✓ Index migrated to HNSW.[/green]")
    except Exception as e:
        console.print(f"[red]Index migration failed: {e}[/red]")


def get_or_create_collection() -> Collection:
    """
    Return the existing collection or create it fresh with the correct schema.

    Schema fields:
        id            — SHA-256 content hash (16 chars), primary key
        embedding     — 1024-dim float vector (voyage-code-3)
        content       — raw code text of the chunk
        file_path     — absolute path to the source file
        repo_name     — name of the parent repository
        symbol_name   — function or class name extracted by tree-sitter
        start_line    — starting line number in the source file (0-indexed)
        end_line      — ending line number in the source file (0-indexed)
        language      — programming language (e.g. "python")
        chunk_type    — "full" | "split_part" | "summary" | "docstring" | "module_level"
        parent_symbol — for split_part, summary, and docstring chunks
    """
    connect()

    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        _migrate_to_hnsw_if_needed(collection)
        return collection

    console.print(f"[cyan]Creating Milvus collection '{COLLECTION_NAME}'...[/cyan]")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=16,
            is_primary=True,
            auto_id=False,
            description="SHA-256 content hash — stable chunk identity",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIM,
            description="voyage-code-3 vector (1024 dims)",
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65_535,
            description="Raw source code of the chunk",
        ),
        FieldSchema(
            name="file_path",
            dtype=DataType.VARCHAR,
            max_length=1_024,
            description="Absolute path to the source file",
        ),
        FieldSchema(
            name="repo_name",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Repository folder name under ~/Desktop/Repos/",
        ),
        FieldSchema(
            name="symbol_name",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Function or class name (tree-sitter extracted)",
        ),
        FieldSchema(
            name="start_line",
            dtype=DataType.INT64,
            description="Start line in source file (0-indexed)",
        ),
        FieldSchema(
            name="end_line",
            dtype=DataType.INT64,
            description="End line in source file (0-indexed)",
        ),
        FieldSchema(
            name="language",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Programming language of the chunk",
        ),
        FieldSchema(
            name="chunk_type",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="full | split_part | summary | docstring | module_level",
        ),
        FieldSchema(
            name="parent_symbol",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Original symbol name for split_part chunks",
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Code intelligence — chunked repo embeddings",
    )
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        },
    )

    console.print(f"[green]✓ Collection '{COLLECTION_NAME}' created with COSINE/HNSW index.[/green]")
    return collection


# ── Read operations ────────────────────────────────────────────────────────────

def get_existing_ids(collection: Collection, repo_name: str) -> set[str]:
    """
    Return the set of chunk IDs currently stored for a given repo.
    Uses the repo's partition for efficiency — avoids a full collection scan.
    """
    pname = partition_name(repo_name)
    partition_names = [pname] if collection.has_partition(pname) else None
    results = collection.query(
        expr=f'repo_name == "{repo_name}"',
        partition_names=partition_names,
        output_fields=["id"],
        consistency_level="Strong",
    )
    return {r["id"] for r in results}


def get_ids_for_file(collection: Collection, repo_name: str, file_path: str) -> list[str]:
    """
    Return all chunk IDs stored for a specific file within a repo.
    Used during diff sync to surgically remove stale chunks.
    """
    pname = partition_name(repo_name)
    partition_names = [pname] if collection.has_partition(pname) else None
    results = collection.query(
        expr=f'repo_name == "{repo_name}" && file_path == "{file_path}"',
        partition_names=partition_names,
        output_fields=["id"],
        consistency_level="Strong",
    )
    return [r["id"] for r in results]


def get_ids_by_file(collection: Collection, repo_name: str) -> dict[str, list[str]]:
    """
    Return all chunk IDs for a repo, grouped by file_path.
    Single Milvus query replacing N per-file queries during sync.
    Scoped to the repo's partition for performance.
    """
    pname = partition_name(repo_name)
    partition_names = [pname] if collection.has_partition(pname) else None
    results = collection.query(
        expr=f'repo_name == "{repo_name}"',
        partition_names=partition_names,
        output_fields=["id", "file_path"],
        consistency_level="Strong",
    )
    by_file: dict[str, list[str]] = {}
    for r in results:
        by_file.setdefault(r["file_path"], []).append(r["id"])
    return by_file


def fetch_chunks_for_file(
    collection: Collection,
    repo_name: str,
    file_path: str,
) -> list[dict]:
    """
    Return full records (including embedding vectors) for all chunks in a file.
    Used during rename handling to reinsert with a new file_path without
    calling the embedding API — the content (and thus vectors) is unchanged.
    """
    pname = partition_name(repo_name)
    partition_names = [pname] if collection.has_partition(pname) else None
    results = collection.query(
        expr=f'repo_name == "{repo_name}" && file_path == "{file_path}"',
        partition_names=partition_names,
        output_fields=[
            "id", "embedding", "content", "file_path", "repo_name",
            "symbol_name", "start_line", "end_line",
            "language", "chunk_type", "parent_symbol",
        ],
        consistency_level="Strong",
    )
    return results


# ── Write operations ───────────────────────────────────────────────────────────

def index_chunks(
    chunks: list[CodeChunk],
    known_existing_ids: set[str] | None = None,
):
    """
    Embed and insert chunks into the repo's Milvus partition.
    Skips chunks that already exist (identified by content hash).
    This makes indexing idempotent — safe to call multiple times.

    known_existing_ids: when the caller already holds the full set of
        existing chunk IDs (e.g. sync_repo), pass them here to skip the
        redundant get_existing_ids() Milvus round-trip.
    """
    if not chunks:
        console.print("  [yellow]No chunks provided to index.[/yellow]")
        return

    collection = get_or_create_collection()
    repo_name = chunks[0].repo_name
    pname = ensure_partition(collection, repo_name)

    if known_existing_ids is not None:
        existing_ids = known_existing_ids
    else:
        existing_ids = get_existing_ids(collection, repo_name)

    # Filter to only genuinely new chunks
    new_chunks = [
        c for c in chunks
        if content_hash(c.content) not in existing_ids
    ]

    if not new_chunks:
        console.print("  [green]✓ All chunks already up to date — nothing to insert.[/green]")
        return

    console.print(
        f"  Indexing [bold]{len(new_chunks)}[/bold] new chunks "
        f"([dim]{len(chunks) - len(new_chunks)} unchanged, skipped[/dim])"
    )

    # Embed in batches, tracking progress
    all_embeddings = []
    texts = [c.content for c in new_chunks]

    for i in track(
        range(0, len(new_chunks), EMBEDDING_BATCH_SIZE),
        description="  Embedding...",
    ):
        batch_texts = texts[i : i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(embed_code(batch_texts))

    # Build Milvus insert payload — column-oriented list of lists
    data = [
        [content_hash(c.content) for c in new_chunks],   # id
        all_embeddings,                                    # embedding
        [c.content               for c in new_chunks],   # content
        [c.file_path             for c in new_chunks],   # file_path
        [c.repo_name             for c in new_chunks],   # repo_name
        [c.symbol_name           for c in new_chunks],   # symbol_name
        [c.start_line            for c in new_chunks],   # start_line
        [c.end_line              for c in new_chunks],   # end_line
        [c.language              for c in new_chunks],   # language
        [c.chunk_type            for c in new_chunks],   # chunk_type
        [c.parent_symbol         for c in new_chunks],   # parent_symbol
    ]

    collection.insert(data, partition_name=pname)
    collection.flush()

    console.print(f"  [green]✓ Inserted {len(new_chunks)} chunks into partition '{pname}'.[/green]")


def reinsert_with_new_path(
    collection: Collection,
    records: list[dict],
    new_file_path: str,
) -> None:
    """
    Reinsert fetched records with an updated file_path, reusing their
    existing embeddings — no Voyage API call required.
    Called when a file is renamed in git (content unchanged, path changed).
    """
    if not records:
        return

    repo_name = records[0]["repo_name"]
    pname = ensure_partition(collection, repo_name)

    data = [
        [r["id"]           for r in records],
        [r["embedding"]    for r in records],
        [r["content"]      for r in records],
        [new_file_path     for _ in records],   # updated path
        [r["repo_name"]    for r in records],
        [r["symbol_name"]  for r in records],
        [r["start_line"]   for r in records],
        [r["end_line"]     for r in records],
        [r["language"]     for r in records],
        [r["chunk_type"]   for r in records],
        [r["parent_symbol"] for r in records],
    ]

    collection.insert(data, partition_name=pname)
    collection.flush()


def delete_chunks_by_ids(ids: list[str]):
    """
    Delete specific chunks by their content-hash IDs.
    Applies to all partitions — Milvus scans the whole collection by ID.
    Called during diff sync to remove stale chunks for changed files.
    """
    if not ids:
        return

    collection = get_or_create_collection()

    # Milvus IN expression requires quoted, comma-separated values
    id_list = ", ".join(f'"{i}"' for i in ids)
    collection.delete(expr=f"id in [{id_list}]")
    collection.flush()

    console.print(f"  [yellow]Deleted {len(ids)} stale chunks.[/yellow]")


def delete_repo(repo_name: str):
    """
    Remove all chunks belonging to a repo by deleting within its partition.
    The partition itself is kept (empty) so it can be reused on re-index.
    Called by: cli.py remove <repo_name>
    """
    collection = get_or_create_collection()
    pname = partition_name(repo_name)

    if collection.has_partition(pname):
        collection.delete(
            expr=f'repo_name == "{repo_name}"',
            partition_name=pname,
        )
        collection.flush()
        console.print(f"  [green]✓ All chunks for '{repo_name}' removed from partition '{pname}'.[/green]")
    else:
        # Fallback: repo was indexed before partitions — delete by expression
        collection.delete(expr=f'repo_name == "{repo_name}"')
        collection.flush()
        console.print(f"  [green]✓ All chunks for '{repo_name}' removed.[/green]")


def get_repo_stats(repo_name: str) -> dict:
    """
    Return chunk counts broken down by chunk_type for a given repo.
    Scoped to the repo's partition for efficiency.
    Used by cli.py list to show a detailed summary.
    """
    collection = get_or_create_collection()
    pname = partition_name(repo_name)
    partition_names = [pname] if collection.has_partition(pname) else None

    stats = {"full": 0, "split_part": 0, "summary": 0, "docstring": 0, "module_level": 0, "total": 0}
    for chunk_type in ["full", "split_part", "summary", "docstring", "module_level"]:
        results = collection.query(
            expr=f'chunk_type == "{chunk_type}"',
            partition_names=partition_names,
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        count = results[0]["count(*)"] if results else 0
        stats[chunk_type] = count
        stats["total"] += count

    return stats
