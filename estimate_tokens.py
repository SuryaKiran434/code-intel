#!/usr/bin/env python3
"""
Dry-run token estimator for voyage-code-3.
Counts tokens using Voyage's local tokenizer — zero API tokens consumed.

Usage:
    python estimate_tokens.py <repo_name>
    python estimate_tokens.py <repo_name> --all
"""

import sys
import os
import math
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config (mirrors your config.py) ──────────────────────────────────────────
REPOS_DIR        = Path.home() / "Desktop" / "Repos"
EMBEDDING_MODEL  = "voyage-code-3"
FREE_TOKENS      = 200_000_000
PRICE_PER_1M     = 0.18          # $ after free tier
TPM_LIMIT        = 10_000        # your account limit
RPM_LIMIT        = 3             # your account limit
BATCH_SIZE       = 128           # voyage max texts per request
MAX_TOKENS_PER_REQ = 120_000     # voyage-code-3 per-request limit

# Skipped directories (same as chunker.py)
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv",
    "venv", "env", "dist", "build", "target",
    ".mypy_cache", ".pytest_cache",
}
CODE_EXTENSIONS = {
    ".py", ".java", ".scala", ".js", ".ts", ".go",
    ".rs", ".cpp", ".c", ".h", ".rb", ".kt", ".swift",
    ".cs", ".php", ".sh", ".sql",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_files(repo_path: Path) -> list[Path]:
    files = []
    for f in repo_path.rglob("*"):
        if not f.is_file():
            continue
        if any(skip in f.parts for skip in SKIP_DIRS):
            continue
        if f.suffix in CODE_EXTENSIONS:
            files.append(f)
    return sorted(files)


def read_file_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def estimate_chunks_for_text(text: str) -> list[str]:
    """
    Lightweight chunk estimator that mirrors chunker.py's 3-tier logic
    without needing tree-sitter. Splits on blank lines as a proxy.
    Gives a conservative (slightly high) estimate.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks = []
    current: list[str] = []

    for line in lines:
        current.append(line)
        if len(current) >= 60 and (not line.strip()):
            chunks.append("\n".join(current))
            current = []

    if current:
        chunks.append("\n".join(current))

    # For large chunks, add a summary chunk (mirrors chunker.py behavior)
    expanded = []
    for chunk in chunks:
        chunk_lines = chunk.splitlines()
        expanded.append(chunk)
        if len(chunk_lines) > 60:
            summary = "\n".join(chunk_lines[:15]) + "\n... (truncated)"
            expanded.append(summary)

    return expanded


def format_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes/60:.1f} hr"


# ── Main estimator ────────────────────────────────────────────────────────────

def estimate_repo(repo_name: str, vo) -> dict:
    repo_path = REPOS_DIR / repo_name
    if not repo_path.exists():
        print(f"  ✗ Repo not found: {repo_path}")
        return {}

    print(f"\n  Scanning files...", end="", flush=True)
    files = collect_files(repo_path)
    print(f" {len(files)} code files found")

    all_chunks: list[str] = []
    skipped_files = 0

    for f in files:
        content = read_file_safe(f)
        if content is None or not content.strip():
            skipped_files += 1
            continue
        chunks = estimate_chunks_for_text(content)
        all_chunks.extend(chunks)

    print(f"  Estimated chunks : {format_number(len(all_chunks))}")
    print(f"  Counting tokens  : ", end="", flush=True)

    # Count tokens locally in batches (no API calls, just local tokenizer)
    total_tokens = 0
    batch_count = 0
    for i in range(0, len(all_chunks), 500):
        batch = all_chunks[i : i + 500]
        total_tokens += vo.count_tokens(batch, model=EMBEDDING_MODEL)
        batch_count += 1
        print(".", end="", flush=True)

    print(f" done")

    # ── Rate limit math ───────────────────────────────────────────────────────
    # With 3 RPM and 10K TPM, the binding constraint is usually TPM.
    # Each batch we can send = min(BATCH_SIZE, floor(TPM_LIMIT / avg_tokens_per_chunk))
    avg_tokens = total_tokens / max(len(all_chunks), 1)
    chunks_per_minute = min(
        RPM_LIMIT * BATCH_SIZE,           # RPM ceiling
        math.floor(TPM_LIMIT / max(avg_tokens, 1)) if avg_tokens > 0 else 0
    )
    # Time = chunks / chunks_per_minute
    index_minutes = len(all_chunks) / max(chunks_per_minute, 1)

    # Also factor in the per-request 120K token limit
    max_chunks_per_request = max(1, math.floor(MAX_TOKENS_PER_REQ / max(avg_tokens, 1)))
    actual_batch = min(BATCH_SIZE, max_chunks_per_request)

    # Cost
    remaining_free = max(FREE_TOKENS - total_tokens, 0)
    paid_tokens    = max(total_tokens - FREE_TOKENS, 0)
    cost_usd       = (paid_tokens / 1_000_000) * PRICE_PER_1M

    return {
        "repo":              repo_name,
        "files":             len(files),
        "skipped":           skipped_files,
        "chunks":            len(all_chunks),
        "total_tokens":      total_tokens,
        "avg_tokens_chunk":  round(avg_tokens),
        "safe_batch_size":   actual_batch,
        "index_time_min":    index_minutes,
        "remaining_free":    remaining_free,
        "paid_tokens":       paid_tokens,
        "cost_usd":          cost_usd,
    }


def print_report(results: list[dict]):
    grand_total = sum(r["total_tokens"] for r in results)
    grand_paid  = max(grand_total - FREE_TOKENS, 0)
    grand_cost  = (grand_paid / 1_000_000) * PRICE_PER_1M

    sep = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  TOKEN ESTIMATE REPORT  —  voyage-code-3")
    print(f"{'═'*60}")

    for r in results:
        if not r:
            continue
        print(f"\n  📁 {r['repo']}")
        print(f"  {sep}")
        print(f"  Code files         : {r['files']}  ({r['skipped']} skipped)")
        print(f"  Estimated chunks   : {format_number(r['chunks'])}")
        print(f"  Total tokens       : {format_number(r['total_tokens'])}")
        print(f"  Avg tokens/chunk   : {r['avg_tokens_chunk']}")
        print(f"  Safe batch size    : {r['safe_batch_size']} chunks/request")
        print(f"  Est. index time    : {format_time(r['index_time_min'] * 60)}")
        print(f"                       (at 3 RPM / 10K TPM limit)")
        print(f"  {sep}")
        if r['paid_tokens'] == 0:
            print(f"  💚 Fits in free tier  ({format_number(r['total_tokens'])} / 200M tokens)")
        else:
            print(f"  ⚠️  Exceeds free tier by {format_number(r['paid_tokens'])} tokens")
            print(f"  Estimated cost     : ${r['cost_usd']:.4f}")
        print(f"  Free tokens left   : {format_number(r['remaining_free'])}")

    if len(results) > 1:
        print(f"\n{'═'*60}")
        print(f"  GRAND TOTAL (all repos combined)")
        print(f"  {sep}")
        print(f"  Total tokens       : {format_number(grand_total)}")
        if grand_paid == 0:
            print(f"  💚 All repos fit in free tier")
        else:
            print(f"  ⚠️  Paid tokens      : {format_number(grand_paid)}")
            print(f"  Total cost         : ${grand_cost:.4f}")

    print(f"{'═'*60}\n")
    print("  ℹ️  This is a conservative OVER-estimate.")
    print("     Actual chunker (tree-sitter AST-aware) produces")
    print("     fewer, cleaner chunks than this line-split proxy.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import voyageai

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("✗ VOYAGE_API_KEY not set in .env")
        sys.exit(1)

    vo = voyageai.Client(api_key=api_key)

    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python estimate_tokens.py <repo_name>")
        print("  python estimate_tokens.py --all")
        sys.exit(0)

    if "--all" in args:
        repo_names = [d.name for d in REPOS_DIR.iterdir() if d.is_dir()]
        if not repo_names:
            print(f"No repos found in {REPOS_DIR}")
            sys.exit(1)
        print(f"Found {len(repo_names)} repos: {', '.join(repo_names)}")
    else:
        repo_names = args

    results = []
    for name in repo_names:
        print(f"\n▶ Estimating: {name}")
        result = estimate_repo(name, vo)
        results.append(result)

    print_report([r for r in results if r])


if __name__ == "__main__":
    main()