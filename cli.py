"""
cli.py

Command-line interface for Code Intel.

Commands:
    register            — create a new user account
    login               — sign in (persists token to ~/.code-intel/.auth)
    logout              — sign out
    add   <repo_name>   — index a repo already cloned in ~/Desktop/Repos/
    sync  <repo_name>   — pull latest changes and update embeddings
    list                — show all indexed repos with chunk stats
    remove <repo_name>  — delete a repo's embeddings from Milvus
    ask   <question>    — query the codebase from the terminal
    log                 — view recent query history
    status              — show system health (Milvus, Nomic, sync state)
"""

import json
import sys
import time
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import (
    REPOS_DIR, LANGUAGE_REGISTRY, SYNC_STATE_PATH,
    QUERY_EXPANSION_ENABLED, QUERY_EXPANSION_VARIANTS,
    EMBEDDING_MODEL, LLM_MODEL, VECTOR_DIM, MILVUS_PORT, DB_PATH,
)
from core.db import init_db
from core.auth import (
    register as auth_register,
    login as auth_login,
    logout as auth_logout,
    get_current_user,
    has_any_users,
)
from core.session import (
    create_session,
    load_turns,
    append_turn,
    get_session,
    list_sessions,
)
from core.telemetry import log_query, get_recent_logs
from core.diff_tracker import (
    initial_index,
    sync_repo,
    remove_repo_state,
    get_last_synced_commit,
)
from core.vector_store import (
    get_or_create_collection,
    delete_repo,
    get_repo_stats,
)
from core.retriever import retrieve
from core.llm import ask as llm_ask

console = Console()

# Initialise SQLite schema on every startup (no-op if tables already exist)
init_db()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _abort(message: str):
    """Print an error and exit with code 1."""
    console.print(f"\n[bold red]✗ Error:[/bold red] {message}\n")
    sys.exit(1)


def _check_repo_exists_locally(repo_name: str):
    """Abort if the repo folder doesn't exist under ~/Desktop/Repos/."""
    repo_path = REPOS_DIR / repo_name
    if not repo_path.exists():
        _abort(
            f"'{repo_name}' not found at {repo_path}\n"
            f"  Clone it first:  git clone <url> {repo_path}"
        )


def _check_milvus():
    """Abort with a helpful message if Milvus isn't reachable."""
    try:
        get_or_create_collection()
    except Exception as e:
        _abort(
            f"Cannot connect to Milvus at localhost:19530\n"
            f"  Make sure Docker is running:  cd ~/Desktop/code-intel && docker compose up -d\n"
            f"  Error: {e}"
        )


def _get_indexed_repos() -> dict[str, dict]:
    """Return {repo_name: stats} for all repos currently in Milvus."""
    # Use sync state as the authoritative list of indexed repos —
    # avoids a potentially large Milvus scan just to discover repo names.
    if not SYNC_STATE_PATH.exists():
        return {}
    state = json.loads(SYNC_STATE_PATH.read_text())
    repo_names = sorted(state.keys())
    return {name: get_repo_stats(name) for name in repo_names}


def _require_auth() -> dict:
    """
    Return the current user dict, or abort with a helpful message.
    Guides new users to register on first run.
    """
    user = get_current_user()
    if user:
        return user
    if not has_any_users():
        _abort(
            "No account found. Create one first:\n"
            "  python cli.py register"
        )
    _abort(
        "Not logged in (or session expired).\n"
        "  python cli.py login"
    )


# ── CLI group ──────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """
    \b
    ⚡ Code Intel — query your codebase with gpt-4.1
    ─────────────────────────────────────────────────
    Repos directory:  ~/Desktop/Repos/
    Project:          ~/Desktop/code-intel/
    """


# ── register ───────────────────────────────────────────────────────────────────

@cli.command()
def register():
    """
    Create a new Code Intel account.

    \b
    Example:
        python cli.py register
    """
    console.print("\n[bold cyan]Create your Code Intel account[/bold cyan]\n")

    first_name = click.prompt("  First name")
    last_name  = click.prompt("  Last name")
    email      = click.prompt("  Email")
    password   = click.prompt("  Password", hide_input=True)
    confirm    = click.prompt("  Confirm password", hide_input=True)

    if password != confirm:
        _abort("Passwords do not match.")

    if len(password) < 8:
        _abort("Password must be at least 8 characters.")

    try:
        user = auth_register(email, password, first_name, last_name)
    except ValueError as e:
        _abort(str(e))

    # Auto-login after registration
    try:
        auth_login(email, password)
    except ValueError:
        pass   # should not happen immediately after register

    console.print(
        f"\n[bold green]✓ Account created.[/bold green] "
        f"Welcome, [cyan]{user['first_name']}[/cyan]!\n"
        f"  [dim]You are now logged in.[/dim]\n"
    )


# ── login ──────────────────────────────────────────────────────────────────────

@cli.command()
def login():
    """
    Sign in to your Code Intel account.

    \b
    Example:
        python cli.py login
    """
    # If already logged in, confirm before overwriting
    existing = get_current_user()
    if existing:
        console.print(
            f"\n[dim]Already logged in as [cyan]{existing['email']}[/cyan]. "
            f"Log in as a different user?[/dim]"
        )
        if not click.confirm("  Continue?", default=False):
            console.print("  [dim]Cancelled.[/dim]\n")
            return

    console.print("\n[bold cyan]Sign in to Code Intel[/bold cyan]\n")
    email    = click.prompt("  Email")
    password = click.prompt("  Password", hide_input=True)

    try:
        auth_login(email, password)
    except ValueError as e:
        _abort(str(e))

    user = get_current_user()
    console.print(
        f"\n[bold green]✓ Logged in.[/bold green] "
        f"Welcome back, [cyan]{user['first_name']}[/cyan]!\n"
    )


# ── logout ─────────────────────────────────────────────────────────────────────

@cli.command()
def logout():
    """
    Sign out of your Code Intel account.

    \b
    Example:
        python cli.py logout
    """
    user = get_current_user()
    if not user:
        console.print("\n[dim]Not currently logged in.[/dim]\n")
        return
    auth_logout()
    console.print(f"\n[dim]Logged out. See you next time, {user['first_name']}.[/dim]\n")


# ── add ────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("repo_name")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-index from scratch even if already indexed.",
)
def add(repo_name: str, force: bool):
    """
    Index a repository that is already cloned in ~/Desktop/Repos/.

    \b
    Example:
        python cli.py add my-python-repo
        python cli.py add my-python-repo --force
    """
    _check_milvus()
    _check_repo_exists_locally(repo_name)

    last_commit = get_last_synced_commit(repo_name)
    if last_commit and not force:
        console.print(
            f"\n[yellow]'{repo_name}' is already indexed "
            f"(commit {last_commit[:8]}).[/yellow]\n"
            f"  Run [bold]sync[/bold] to update, or use [bold]--force[/bold] to reindex from scratch.\n"
        )
        return

    if force and last_commit:
        console.print(f"\n[yellow]--force: removing existing index for '{repo_name}'...[/yellow]")
        delete_repo(repo_name)
        remove_repo_state(repo_name)

    console.print(f"\n[bold cyan]Indexing '{repo_name}'...[/bold cyan]")
    console.print(
        f"  [dim]Source:  {REPOS_DIR / repo_name}[/dim]\n"
        f"  [dim]Languages: {list(LANGUAGE_REGISTRY.keys())}[/dim]\n"
    )

    try:
        initial_index(repo_name)
    except (FileNotFoundError, ValueError) as e:
        _abort(str(e))

    stats = get_repo_stats(repo_name)
    console.print(
        f"\n[bold green]✓ '{repo_name}' indexed successfully.[/bold green]\n"
        f"  [dim]{stats['full']} symbols | "
        f"{stats['split_part']} split parts | "
        f"{stats['summary']} summaries | "
        f"{stats['docstring']} docstrings | "
        f"{stats['module_level']} module-level | "
        f"{stats['total']} total chunks[/dim]\n"
    )


# ── sync ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("repo_name")
def sync(repo_name: str):
    """
    Pull latest changes and update embeddings for changed files only.

    \b
    Only files that changed since the last sync are re-embedded.
    Unchanged files are never re-processed.

    \b
    Example:
        python cli.py sync my-python-repo
    """
    _check_milvus()
    _check_repo_exists_locally(repo_name)

    last_commit = get_last_synced_commit(repo_name)
    if not last_commit:
        console.print(
            f"\n[yellow]'{repo_name}' has no sync state. "
            f"Running initial index...[/yellow]\n"
        )
        try:
            initial_index(repo_name)
        except (FileNotFoundError, ValueError) as e:
            _abort(str(e))
        return

    console.print(f"\n[bold cyan]Syncing '{repo_name}'...[/bold cyan]")
    console.print(f"  [dim]Last synced commit: {last_commit[:8]}[/dim]\n")

    try:
        sync_repo(repo_name)
    except Exception as e:
        _abort(str(e))

    stats = get_repo_stats(repo_name)
    console.print(
        f"\n  [dim]Current index: {stats['full']} symbols | "
        f"{stats['total']} total chunks[/dim]\n"
    )


# ── list ───────────────────────────────────────────────────────────────────────

@cli.command("list")
def list_repos():
    """
    Show all indexed repositories with chunk counts and sync state.

    \b
    Example:
        python cli.py list
    """
    _check_milvus()

    repos = _get_indexed_repos()

    if not repos:
        console.print(
            "\n[yellow]No repositories indexed yet.[/yellow]\n"
            "  Get started:  [bold]python cli.py add <repo_name>[/bold]\n"
            f"  Repos folder: {REPOS_DIR}\n"
        )
        return

    table = Table(
        title="\n⚡ Indexed Repositories",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Repository",   style="bold",    min_width=20)
    table.add_column("Full",         justify="right", style="green")
    table.add_column("Splits",       justify="right", style="yellow")
    table.add_column("Docstrings",   justify="right", style="cyan")
    table.add_column("Module",       justify="right", style="magenta")
    table.add_column("Summaries",    justify="right", style="dim")
    table.add_column("Total",        justify="right", style="bold")
    table.add_column("Last Commit",  style="dim",     min_width=10)

    for repo_name, stats in repos.items():
        last_commit = get_last_synced_commit(repo_name)
        commit_str  = last_commit[:8] if last_commit else "[red]unknown[/red]"
        table.add_row(
            repo_name,
            str(stats["full"]),
            str(stats["split_part"]),
            str(stats["docstring"]),
            str(stats["module_level"]),
            str(stats["summary"]),
            str(stats["total"]),
            commit_str,
        )

    console.print(table)
    console.print(
        f"\n  [dim]Repos directory: {REPOS_DIR}[/dim]\n"
        f"  [dim]Supported languages: {list(LANGUAGE_REGISTRY.keys())}[/dim]\n"
    )


# ── remove ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("repo_name")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def remove(repo_name: str, yes: bool):
    """
    Remove a repository and all its embeddings from Milvus.

    \b
    This does NOT delete the cloned repo from ~/Desktop/Repos/.
    It only removes the indexed embeddings from the vector database.

    \b
    Example:
        python cli.py remove my-python-repo
        python cli.py remove my-python-repo --yes
    """
    _check_milvus()

    last_commit = get_last_synced_commit(repo_name)
    if not last_commit:
        console.print(f"\n[yellow]'{repo_name}' is not currently indexed.[/yellow]\n")
        return

    if not yes:
        confirm = click.confirm(f"\n  Remove all embeddings for '{repo_name}'?", default=False)
        if not confirm:
            console.print("\n  [dim]Cancelled.[/dim]\n")
            return

    console.print(f"\n[cyan]Removing '{repo_name}' from index...[/cyan]")
    delete_repo(repo_name)
    remove_repo_state(repo_name)
    console.print(f"[bold green]✓ '{repo_name}' removed.[/bold green]\n")


# ── ask ────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
@click.option("--repo", "-r", default=None, help="Scope search to a specific repository.")
@click.option(
    "--top-k", "-k", default=5, show_default=True,
    help="Number of chunks to retrieve before sending to GPT-4o.",
)
@click.option(
    "--show-chunks", is_flag=True, default=False,
    help="Print the raw retrieved chunks before the answer.",
)
@click.option(
    "--session", "session_id", default=None,
    help="Continue an existing conversation session (use session ID).",
)
@click.option(
    "--new-session", is_flag=True, default=False,
    help="Start a new conversation session and print its ID.",
)
def ask(
    question:   str,
    repo:       str,
    top_k:      int,
    show_chunks: bool,
    session_id: str | None,
    new_session: bool,
):
    """
    Ask a question about your indexed codebase.

    \b
    Examples:
        python cli.py ask "How does authentication work?"
        python cli.py ask "What does the DataLoader class do?" --repo my-repo
        python cli.py ask "Explain the retry logic" --show-chunks
        python cli.py ask "How does sync work?" --new-session
        python cli.py ask "What about error handling?" --session <id>
    """
    _check_milvus()
    user = _require_auth()

    # ── Validate repo scope ────────────────────────────────────────────────────
    if repo:
        if not get_last_synced_commit(repo):
            _abort(f"'{repo}' is not indexed.\n  Run: python cli.py add {repo}")

    # ── Session setup ──────────────────────────────────────────────────────────
    history = None

    if new_session and session_id:
        _abort("Use either --new-session or --session <id>, not both.")

    if new_session:
        session_id = create_session(user["id"], title=question)
        console.print(f"\n[dim]Session started: [bold]{session_id}[/bold][/dim]")
        console.print(f"[dim]Continue with:   python cli.py ask \"...\" --session {session_id}[/dim]\n")

    elif session_id:
        sess = get_session(session_id)
        if not sess:
            _abort(f"Session '{session_id}' not found.")
        history = load_turns(session_id)
        if history:
            console.print(f"\n[dim]Continuing session ({len(history)} prior messages)[/dim]")

    # ── Retrieve ───────────────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Searching...[/bold cyan] [dim]{question}[/dim]")
    if repo:
        console.print(f"  [dim]Scoped to repo: {repo}[/dim]")

    t_start = time.monotonic()

    try:
        chunks = retrieve(question, repo_name=repo, top_k=top_k)
    except Exception as e:
        _abort(f"Retrieval failed: {e}")

    if not chunks:
        console.print(
            "\n[yellow]No relevant code found.[/yellow]\n"
            "  Try rephrasing your question, or check that the repo is indexed.\n"
        )
        return

    # ── Optionally show raw chunks ─────────────────────────────────────────────
    if show_chunks:
        console.print(f"\n[dim]── Retrieved {len(chunks)} chunks ──[/dim]")
        for i, chunk in enumerate(chunks, 1):
            console.print(
                f"\n[dim]Chunk {i} | {chunk['symbol_name']} | "
                f"score: {chunk['score']} | {chunk['file_path']}[/dim]"
            )
            console.print(
                Panel(
                    chunk["content"][:800] + ("..." if len(chunk["content"]) > 800 else ""),
                    border_style="dim",
                    expand=False,
                )
            )

    # ── Ask GPT-4o ────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Asking gpt-4.1...[/bold cyan]\n")

    try:
        result = llm_ask(question, chunks, history=history)
    except Exception as e:
        _abort(f"GPT-4o call failed: {e}")

    latency_ms = int((time.monotonic() - t_start) * 1000)

    # ── Persist session turns ──────────────────────────────────────────────────
    if session_id:
        append_turn(session_id, "user",      question)
        append_turn(session_id, "assistant", result["answer"])

    # ── Log query ─────────────────────────────────────────────────────────────
    top_score = chunks[0]["score"] if chunks else None
    log_query(
        user_id          = user["id"],
        session_id       = session_id,
        question         = question,
        repo             = repo or "*",
        query_variants   = QUERY_EXPANSION_VARIANTS if QUERY_EXPANSION_ENABLED else 0,
        chunks_retrieved = len(chunks),
        top_score        = top_score,
        tokens_used      = result["tokens_used"],
        latency_ms       = latency_ms,
        answer_snippet   = result["answer"],
    )

    # ── Print answer ──────────────────────────────────────────────────────────
    console.print(
        Panel(
            result["answer"],
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # ── Print sources ─────────────────────────────────────────────────────────
    console.print("[bold]Sources:[/bold]")
    for source in result["sources"]:
        truncated_flag = " [yellow][truncated][/yellow]" if source["truncated"] else ""
        console.print(
            f"  [dim]•[/dim] [bold dim][{source['label']}][/bold dim]"
            f" [cyan]{source['symbol']}[/cyan]"
            f"  [dim]lines {source['lines']}[/dim]"
            f"  [dim]score {source['score']}[/dim]"
            f"  [dim]{source['file']}[/dim]"
            f"{truncated_flag}"
        )

    # ── Print usage stats ─────────────────────────────────────────────────────
    session_hint = (
        f" | Session: {session_id[:8]}..."
        if session_id else ""
    )
    console.print(
        f"\n  [dim]Tokens used: {result['tokens_used']} | "
        f"Chunks in context: {result['context_chunks']} | "
        f"Retrieved: {len(chunks)} | "
        f"Latency: {latency_ms}ms"
        f"{session_hint}[/dim]\n"
    )


# ── log ────────────────────────────────────────────────────────────────────────

@cli.command("log")
@click.option(
    "--last", "-n", default=20, show_default=True,
    help="Number of recent queries to display.",
)
def query_log(last: int):
    """
    View your recent query history.

    \b
    Example:
        python cli.py log
        python cli.py log --last 10
    """
    user = _require_auth()
    rows = get_recent_logs(user["id"], limit=last)

    if not rows:
        console.print(
            f"\n[dim]No queries logged yet for {user['email']}.[/dim]\n"
            f"  Ask your first question:  [bold]python cli.py ask \"...\"[/bold]\n"
        )
        return

    table = Table(
        title=f"\n⚡ Query Log — {user['first_name']} {user['last_name']}",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("#",          justify="right", style="dim",   min_width=3)
    table.add_column("Time",       style="dim",                    min_width=19)
    table.add_column("Question",   style="bold",                   min_width=30)
    table.add_column("Repo",       style="cyan",                   min_width=10)
    table.add_column("Chunks",     justify="right", style="green", min_width=6)
    table.add_column("Top Score",  justify="right", style="yellow",min_width=9)
    table.add_column("Tokens",     justify="right", style="dim",   min_width=6)
    table.add_column("Latency",    justify="right", style="dim",   min_width=8)

    for row in rows:
        question_preview = row["question"]
        if len(question_preview) > 45:
            question_preview = question_preview[:42] + "..."
        top_score_str = f"{row['top_score']:.3f}" if row["top_score"] is not None else "—"
        # Format timestamp: trim microseconds
        ts = row["timestamp"][:19].replace("T", " ")
        table.add_row(
            str(row["id"]),
            ts,
            question_preview,
            row["repo"],
            str(row["chunks_retrieved"]),
            top_score_str,
            str(row["tokens_used"]),
            f"{row['latency_ms']}ms",
        )

    console.print(table)
    console.print()


# ── sessions ───────────────────────────────────────────────────────────────────

@cli.command("sessions")
@click.option("--last", "-n", default=10, show_default=True, help="Number of sessions to show.")
def show_sessions(last: int):
    """
    List your recent conversation sessions.

    \b
    Example:
        python cli.py sessions
        python cli.py sessions --last 5
    """
    user = _require_auth()
    sessions = list_sessions(user["id"], limit=last)

    if not sessions:
        console.print(
            "\n[dim]No sessions yet.[/dim]\n"
            "  Start one:  [bold]python cli.py ask \"...\" --new-session[/bold]\n"
        )
        return

    table = Table(
        title=f"\n⚡ Sessions — {user['first_name']} {user['last_name']}",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Session ID",  style="dim",  min_width=36)
    table.add_column("Title",       style="bold", min_width=35)
    table.add_column("Turns",       justify="right", style="green")
    table.add_column("Last Used",   style="dim",  min_width=19)

    for s in sessions:
        title     = s["title"] if s["title"] else "[dim]untitled[/dim]"
        last_used = s["last_used"][:19].replace("T", " ")
        table.add_row(s["id"], title, str(s["turn_count"]), last_used)

    console.print(table)
    console.print()


# ── status ─────────────────────────────────────────────────────────────────────

@cli.command()
def status():
    """
    Show system health: Milvus connection, indexed repos, and config summary.

    \b
    Example:
        python cli.py status
    """
    console.print()

    # ── Auth status ────────────────────────────────────────────────────────────
    user = get_current_user()
    if user:
        auth_status = f"[bold green]✓ {user['first_name']} {user['last_name']}[/bold green]"
        auth_detail = user["email"]
    else:
        auth_status = "[yellow]Not logged in[/yellow]"
        auth_detail = "python cli.py login"

    # ── Milvus health ──────────────────────────────────────────────────────────
    try:
        collection = get_or_create_collection()
        collection.load()
        milvus_status = "[bold green]✓ Connected[/bold green]"
        milvus_detail = f"localhost:{MILVUS_PORT} | collection: {collection.name}"
    except Exception as e:
        milvus_status = "[bold red]✗ Unreachable[/bold red]"
        milvus_detail = str(e)

    # ── Embedding health ───────────────────────────────────────────────────────
    embed_status = "[bold green]✓ Voyage AI[/bold green]"
    embed_detail = f"model: {EMBEDDING_MODEL} | dim: {VECTOR_DIM}"

    # ── Summary table ──────────────────────────────────────────────────────────
    table = Table(
        title="⚡ Code Intel — System Status",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Component", style="bold cyan", min_width=16)
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    table.add_row("Account",    auth_status,   auth_detail)
    table.add_row("Milvus",     milvus_status, milvus_detail)
    table.add_row("Embeddings", embed_status,  embed_detail)
    table.add_row("LLM",        "[bold green]✓ gpt-4.1[/bold green]", LLM_MODEL)
    table.add_row("Repos dir",  "[dim]─[/dim]", str(REPOS_DIR))
    table.add_row("Local DB",   "[dim]─[/dim]", str(DB_PATH))

    console.print(table)

    # ── Indexed repos summary ──────────────────────────────────────────────────
    try:
        repos = _get_indexed_repos()
        if repos:
            console.print(f"\n  [bold]Indexed repositories:[/bold] {len(repos)}")
            for repo_name, stats in repos.items():
                last_commit = get_last_synced_commit(repo_name)
                commit_str  = last_commit[:8] if last_commit else "unknown"
                console.print(
                    f"  [dim]•[/dim] [cyan]{repo_name}[/cyan] "
                    f"[dim]— {stats['total']} chunks | commit {commit_str}[/dim]"
                )
        else:
            console.print("\n  [dim]No repositories indexed yet.[/dim]")
    except Exception:
        console.print("\n  [dim]Could not load repo list (Milvus unavailable).[/dim]")

    console.print()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
