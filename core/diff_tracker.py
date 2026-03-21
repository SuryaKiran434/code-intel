"""
core/diff_tracker.py

Manages efficient, git-native incremental sync for indexed repositories.

Strategy:
  - On first index (add):   full repo chunk + embed
  - On subsequent syncs:    git diff to find changed files only
                            re-chunk changed files
                            diff chunk hashes vs Milvus
                            delete stale, insert new
                            → unchanged files are never re-embedded

Sync state is persisted in .sync_state.json on disk so it survives
process restarts between syncs.
"""

import json
from pathlib import Path

import git
from rich.console import Console

from config import SYNC_STATE_PATH, REPOS_DIR, LANGUAGE_REGISTRY
from core.chunker import chunk_file, chunk_repository, CodeChunk
from core.embedder import content_hash
from core.vector_store import (
    get_or_create_collection,
    get_ids_by_file,
    index_chunks,
    delete_chunks_by_ids,
    fetch_chunks_for_file,
    reinsert_with_new_path,
)

console = Console()


# ── Sync state persistence ─────────────────────────────────────────────────────

def _load_sync_state() -> dict:
    """
    Load the persisted sync state from disk.
    Returns an empty dict if the file doesn't exist yet (first run).

    State structure:
    {
        "repo-name": "abc1234...",   # last synced git commit SHA
        ...
    }
    """
    if SYNC_STATE_PATH.exists():
        try:
            return json.loads(SYNC_STATE_PATH.read_text())
        except json.JSONDecodeError:
            console.print("[yellow]Warning: sync state file corrupted — resetting.[/yellow]")
            return {}
    return {}


def _save_sync_state(state: dict):
    """Persist the updated sync state to disk atomically."""
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then rename — prevents corruption on crash
    tmp = SYNC_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(SYNC_STATE_PATH)


def get_last_synced_commit(repo_name: str) -> str | None:
    """Return the last commit SHA we synced for a repo, or None if never synced."""
    return _load_sync_state().get(repo_name)


def _update_synced_commit(repo_name: str, commit_sha: str):
    """Record that we've successfully synced up to this commit."""
    state = _load_sync_state()
    state[repo_name] = commit_sha
    _save_sync_state(state)


def _clear_sync_state(repo_name: str):
    """Remove a repo from the sync state (called on remove)."""
    state = _load_sync_state()
    state.pop(repo_name, None)
    _save_sync_state(state)


# ── Supported file filter ──────────────────────────────────────────────────────

def _is_supported(file_path: str) -> bool:
    """Return True if this file's extension is in the language registry."""
    return Path(file_path).suffix in LANGUAGE_REGISTRY


# ── Initial full index ─────────────────────────────────────────────────────────

def initial_index(repo_name: str):
    """
    Full index of a repository — called once on first `cli add`.

    Steps:
        1. Chunk every supported source file via tree-sitter
        2. Embed all chunks via Voyage AI API
        3. Insert into Milvus (deduplication handled inside index_chunks)
        4. Record the current HEAD commit as the sync baseline
    """
    repo_path = REPOS_DIR / repo_name

    if not repo_path.exists():
        raise FileNotFoundError(
            f"Repo '{repo_name}' not found at {repo_path}\n"
            f"Make sure you have cloned it into ~/Desktop/Repos/ first."
        )

    try:
        repo = git.Repo(repo_path)
    except git.InvalidGitRepositoryError:
        raise ValueError(f"'{repo_path}' is not a valid git repository.")

    console.print(f"  [cyan]Chunking all files in '{repo_name}'...[/cyan]")
    chunks = chunk_repository(str(repo_path), repo_name)

    if not chunks:
        console.print(f"  [yellow]No supported source files found in '{repo_name}'.[/yellow]")
        console.print(f"  [dim]Supported extensions: {list(LANGUAGE_REGISTRY.keys())}[/dim]")
        return

    full_count       = sum(1 for c in chunks if c.chunk_type == "full")
    split_count      = sum(1 for c in chunks if c.chunk_type == "split_part")
    summary_count    = sum(1 for c in chunks if c.chunk_type == "summary")
    docstring_count  = sum(1 for c in chunks if c.chunk_type == "docstring")
    module_count     = sum(1 for c in chunks if c.chunk_type == "module_level")

    console.print(
        f"  Found [bold]{len(chunks)}[/bold] total chunks "
        f"([green]{full_count}[/green] full, "
        f"[yellow]{split_count}[/yellow] split parts, "
        f"[cyan]{docstring_count}[/cyan] docstrings, "
        f"[magenta]{module_count}[/magenta] module-level, "
        f"[dim]{summary_count}[/dim] summaries)"
    )

    index_chunks(chunks)

    # Save baseline commit so future syncs know where to diff from
    current_sha = repo.head.commit.hexsha
    _update_synced_commit(repo_name, current_sha)
    console.print(f"  [dim]Sync baseline set to commit {current_sha[:8]}[/dim]")


# ── Incremental sync ───────────────────────────────────────────────────────────

def sync_repo(repo_name: str):
    """
    Pull latest changes and surgically update only affected chunks.

    Algorithm:
        1. git pull origin (fetch latest)
        2. Compare HEAD to last synced commit → get list of changed files
        3. For each changed file:
            a. Get old chunk IDs stored in Milvus for that file
            b. Re-chunk the file with tree-sitter
            c. Compute new chunk hashes
            d. Delete chunks whose hash no longer exists (stale)
            e. Insert chunks whose hash is new
        4. Update sync state to new HEAD commit

    Files that didn't change are never touched — no re-embedding cost.
    """
    repo_path = REPOS_DIR / repo_name

    if not repo_path.exists():
        raise FileNotFoundError(
            f"Repo '{repo_name}' not found at {repo_path}.\n"
            f"Use 'add' to index it first."
        )

    try:
        repo = git.Repo(repo_path)
    except git.InvalidGitRepositoryError:
        raise ValueError(f"'{repo_path}' is not a valid git repository.")

    last_commit_sha = get_last_synced_commit(repo_name)

    # ── Step 1: Pull latest ────────────────────────────────────────────────────
    console.print(f"  [cyan]Pulling latest changes from remote...[/cyan]")
    try:
        pull_info = repo.remotes.origin.pull()
        for info in pull_info:
            console.print(f"  [dim]  {info.ref}: {info.note or 'up to date'}[/dim]")
    except git.GitCommandError as e:
        console.print(f"  [red]Git pull failed: {e}[/red]")
        raise

    new_commit_sha = repo.head.commit.hexsha

    # ── Step 2: Check if anything changed ─────────────────────────────────────
    if last_commit_sha == new_commit_sha:
        console.print(
            f"  [green]✓ Already up to date.[/green] "
            f"[dim](commit {new_commit_sha[:8]})[/dim]"
        )
        return

    if not last_commit_sha:
        # Sync state missing — fall back to full reindex
        console.print(
            f"  [yellow]No prior sync state found for '{repo_name}'. "
            f"Running full reindex...[/yellow]"
        )
        initial_index(repo_name)
        return

    console.print(
        f"  [cyan]Comparing commits "
        f"[dim]{last_commit_sha[:8]}[/dim] → "
        f"[bold]{new_commit_sha[:8]}[/bold]...[/cyan]"
    )

    # ── Step 3: Find changed files ─────────────────────────────────────────────
    try:
        old_commit = repo.commit(last_commit_sha)
        new_commit = repo.commit(new_commit_sha)
        diff       = old_commit.diff(new_commit)
    except git.BadName:
        console.print(
            f"  [yellow]Could not resolve previous commit {last_commit_sha[:8]}. "
            f"Running full reindex...[/yellow]"
        )
        initial_index(repo_name)
        return

    # Collect all affected file paths (both sides of renames/moves)
    changed_files: set[str] = set()
    deleted_files: set[str] = set()

    # Renamed files are handled separately — we reuse existing embeddings
    # instead of re-embedding identical content under a new path.
    renamed_files: list[tuple[str, str]] = []   # (old_path, new_path)

    for d in diff:
        if d.change_type == "D":
            # File deleted — remove from Milvus, don't re-chunk
            if d.a_path and _is_supported(d.a_path):
                deleted_files.add(str(repo_path / d.a_path))
        elif d.change_type == "R":
            # File renamed — handle via embedding reuse (no API call needed)
            old_supported = d.a_path and _is_supported(d.a_path)
            new_supported = d.b_path and _is_supported(d.b_path)
            if old_supported and new_supported:
                renamed_files.append(
                    (str(repo_path / d.a_path), str(repo_path / d.b_path))
                )
            elif old_supported:
                deleted_files.add(str(repo_path / d.a_path))
            elif new_supported:
                changed_files.add(str(repo_path / d.b_path))
        else:
            # Added (A) or Modified (M)
            if d.b_path and _is_supported(d.b_path):
                changed_files.add(str(repo_path / d.b_path))

    total_affected = len(changed_files) + len(deleted_files) + len(renamed_files)
    if total_affected == 0:
        console.print(
            f"  [green]✓ No supported source files changed.[/green] "
            f"[dim](commit {new_commit_sha[:8]})[/dim]"
        )
        _update_synced_commit(repo_name, new_commit_sha)
        return

    console.print(
        f"  [cyan]{len(changed_files)} file(s) changed, "
        f"{len(deleted_files)} file(s) deleted, "
        f"{len(renamed_files)} file(s) renamed.[/cyan]"
    )

    collection     = get_or_create_collection()

    # ── Step 3b: Handle renames — reuse embeddings, no API call ───────────────
    for old_path, new_path in renamed_files:
        old_records = fetch_chunks_for_file(collection, repo_name, old_path)
        if old_records:
            old_ids = [r["id"] for r in old_records]
            delete_chunks_by_ids(old_ids)
            reinsert_with_new_path(collection, old_records, new_path)
            console.print(
                f"  [dim]  Rename: {Path(old_path).name} → {Path(new_path).name} "
                f"({len(old_records)} chunks, 0 API calls)[/dim]"
            )
        else:
            # No records found (pre-partition legacy data) — reindex fresh
            if Path(new_path).exists():
                changed_files.add(new_path)
            deleted_files.add(old_path)
    all_stale_ids: list[str] = []
    all_new_chunks: list[CodeChunk] = []

    # Single Milvus query to get all IDs for the repo, grouped by file_path.
    # Replaces N per-file queries (one per changed/deleted file) with 1 query.
    ids_by_file = get_ids_by_file(collection, repo_name)
    existing_repo_ids = {id for ids in ids_by_file.values() for id in ids}

    # ── Step 4a: Handle deleted files ─────────────────────────────────────────
    from core.graph import delete_file_graph  # noqa: PLC0415
    for file_path in deleted_files:
        old_ids = ids_by_file.get(file_path, [])
        if old_ids:
            console.print(
                f"  [dim]  Deleted: {Path(file_path).name} "
                f"({len(old_ids)} chunks removed)[/dim]"
            )
            all_stale_ids.extend(old_ids)
        delete_file_graph(file_path, repo_name)

    # ── Step 4b: Handle changed / added files ─────────────────────────────────
    for file_path in changed_files:
        file_p = Path(file_path)

        # File may have been deleted on disk despite appearing in diff
        # (e.g. submodule edge cases)
        if not file_p.exists():
            all_stale_ids.extend(ids_by_file.get(file_path, []))
            continue

        # Re-chunk the file
        new_chunks = chunk_file(file_path, repo_name)
        new_hashes = {content_hash(c.content) for c in new_chunks}

        # Find which old chunks for this file are now stale
        old_id_set = set(ids_by_file.get(file_path, []))
        stale_ids  = list(old_id_set - new_hashes)
        all_stale_ids.extend(stale_ids)

        # Only queue chunks that are genuinely new
        truly_new = [
            c for c in new_chunks
            if content_hash(c.content) not in existing_repo_ids
        ]
        all_new_chunks.extend(truly_new)

        console.print(
            f"  [dim]  {file_p.name}: "
            f"+{len(truly_new)} new, "
            f"-{len(stale_ids)} stale[/dim]"
        )

    # ── Step 5: Apply changes to Milvus ───────────────────────────────────────
    if all_stale_ids:
        console.print(
            f"  [yellow]Removing {len(all_stale_ids)} stale chunks...[/yellow]"
        )
        delete_chunks_by_ids(all_stale_ids)

    if all_new_chunks:
        console.print(
            f"  [cyan]Inserting {len(all_new_chunks)} new chunks...[/cyan]"
        )
        # Pass existing IDs so index_chunks skips its own get_existing_ids()
        # call — we already have the full set from get_ids_by_file() above.
        index_chunks(all_new_chunks, known_existing_ids=existing_repo_ids)

    # ── Step 6: Update sync state ──────────────────────────────────────────────
    _update_synced_commit(repo_name, new_commit_sha)

    console.print(
        f"  [bold green]✓ Sync complete.[/bold green] "
        f"[dim]Now at commit {new_commit_sha[:8]}[/dim]"
    )


# ── Remove repo cleanup ────────────────────────────────────────────────────────

def remove_repo_state(repo_name: str):
    """
    Clear the sync state and graph edges for a removed repo.
    Called by cli.py remove after Milvus deletion.
    """
    from core.graph import delete_repo_graph  # noqa: PLC0415
    _clear_sync_state(repo_name)
    delete_repo_graph(repo_name)
    console.print(f"  [dim]Sync state and graph cleared for '{repo_name}'.[/dim]")