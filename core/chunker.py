"""
core/chunker.py

AST-aware code chunking using tree-sitter 0.23.x API.
Applies a 3-tier size strategy per extracted symbol:

    Tier 1  ≤ 60 lines   → store as single "full" chunk
    Tier 2  61–150 lines → store as "full" + a "summary" chunk
    Tier 3  151+ lines   → split into "split_part" chunks + a "summary" chunk

Phase 4 additions:
    - Split parts now include a SPLIT_OVERLAP_LINES sliding window so that
      each part shares context lines with the previous part.
    - Docstrings are extracted as lightweight "docstring" chunks — they embed
      closer to natural language queries than raw code.
    - Module-level code (constants, type aliases, top-level expressions) is
      captured as a "module_level" chunk per file so it can be retrieved.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tree_sitter import Language, Parser          # tree-sitter 0.23.x
from rich.progress import Progress, track
from config import (
    LANGUAGE_REGISTRY,
    CHUNK_SMALL_MAX_LINES,
    CHUNK_MEDIUM_MAX_LINES,
    SPLIT_OVERLAP_LINES,
)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class CodeChunk:
    content:       str
    file_path:     str
    repo_name:     str
    symbol_name:   str
    start_line:    int
    end_line:      int
    language:      str
    chunk_type:    str          # "full" | "split_part" | "summary"
                                # "docstring" | "module_level"
    parent_symbol: str = ""    # set for split_part, summary, docstring chunks
    line_count:    int = field(init=False)

    def __post_init__(self):
        self.line_count = self.end_line - self.start_line + 1


# ── Parser cache — one Parser per language per thread ─────────────────────────
# tree-sitter Parser objects are stateful during a parse and NOT thread-safe.
# Using threading.local() gives each thread its own cache so parallel
# chunk_repository workers never share a Parser instance.

_tls = threading.local()

def _get_parser(language_name: str) -> Parser:
    """Return a thread-local cached Parser for the given language."""
    if not hasattr(_tls, "cache"):
        _tls.cache = {}
    if language_name in _tls.cache:
        return _tls.cache[language_name]

    if language_name == "python":
        import tree_sitter_python as tspython
        lang = Language(tspython.language())

    # ── Uncomment when adding Java / Scala ────────────────────────────────────
    # elif language_name == "java":
    #     import tree_sitter_java as tsjava
    #     lang = Language(tsjava.language())
    #
    # elif language_name == "scala":
    #     import tree_sitter_scala as tsscala
    #     lang = Language(tsscala.language())

    else:
        raise ValueError(f"No tree-sitter grammar registered for: '{language_name}'")

    parser = Parser(lang)
    _tls.cache[language_name] = parser
    return parser


# ── Docstring extraction ───────────────────────────────────────────────────────

def _extract_docstring(node, lines: list[str]) -> str | None:
    """
    Return the raw docstring text of a function or class node, or None.

    In Python's tree-sitter AST the docstring is the first statement of the
    body block when that statement is an expression_statement containing a
    string node.
    """
    body = None
    for child in node.children:
        if child.type == "block":
            body = child
            break

    if not body:
        return None

    for stmt in body.children:
        if stmt.type == "expression_statement":
            for child in stmt.children:
                if child.type == "string":
                    start = child.start_point[0]
                    end   = child.end_point[0]
                    return "\n".join(lines[start : end + 1])
        # Only check the very first statement
        if stmt.type not in ("comment", "\n"):
            break

    return None


# ── 3-tier size strategy ───────────────────────────────────────────────────────

def _make_summary(chunk: "CodeChunk") -> "CodeChunk":
    """
    Produce a compact summary chunk from a large symbol.
    Keeps the signature + first 15 lines (docstring / opening logic).
    """
    lines   = chunk.content.splitlines()
    preview = lines[:15]
    preview.append("    # ... [truncated — see source file for full implementation]")
    return CodeChunk(
        content="\n".join(preview),
        file_path=chunk.file_path,
        repo_name=chunk.repo_name,
        symbol_name=f"{chunk.symbol_name}__summary",
        start_line=chunk.start_line,
        end_line=chunk.start_line + len(preview),
        language=chunk.language,
        chunk_type="summary",
        parent_symbol=chunk.symbol_name,
    )


def _split_large_chunk(chunk: "CodeChunk") -> list["CodeChunk"]:
    """
    Split a 150+ line chunk into window-sized parts with sliding overlap.

    Each part (except the first) starts SPLIT_OVERLAP_LINES before the
    previous part ended — giving the LLM enough context to understand
    variable bindings, conditionals, and setup established earlier.
    """
    lines    = chunk.content.splitlines()
    window   = CHUNK_MEDIUM_MAX_LINES
    overlap  = SPLIT_OVERLAP_LINES
    parts    = []
    i        = 0
    part_num = 0

    while i < len(lines):
        end     = min(i + window, len(lines))
        is_last = (end == len(lines))

        # Prefer splitting at a blank line near the window boundary
        if not is_last:
            for j in range(end, max(i + 20, end - 20), -1):
                if j < len(lines) and lines[j].strip() == "":
                    end = j
                    break

        part_content = "\n".join(lines[i:end])
        parts.append(CodeChunk(
            content=part_content,
            file_path=chunk.file_path,
            repo_name=chunk.repo_name,
            symbol_name=f"{chunk.symbol_name}__part{part_num}",
            start_line=chunk.start_line + i,
            end_line=chunk.start_line + end - 1,
            language=chunk.language,
            chunk_type="split_part",
            parent_symbol=chunk.symbol_name,
        ))

        if is_last:
            break

        # Overlap: next part begins SPLIT_OVERLAP_LINES before current end
        # so it re-reads the closing context of this part.
        # max(i + 1, ...) guarantees we always advance (no infinite loop).
        i = max(i + 1, end - overlap)
        part_num += 1

    return parts


def _apply_size_strategy(raw: "CodeChunk") -> list["CodeChunk"]:
    """
    Apply the 3-tier strategy to a raw extracted symbol chunk.

    Tier 1 — small  (≤ CHUNK_SMALL_MAX_LINES):  single "full" chunk.
    Tier 2 — medium (≤ CHUNK_MEDIUM_MAX_LINES):  "full" + "summary".
    Tier 3 — large  (> CHUNK_MEDIUM_MAX_LINES):  "split_part"s + "summary".
    """
    raw.chunk_type = "full"

    if raw.line_count <= CHUNK_SMALL_MAX_LINES:
        return [raw]
    elif raw.line_count <= CHUNK_MEDIUM_MAX_LINES:
        return [raw, _make_summary(raw)]
    else:
        return _split_large_chunk(raw) + [_make_summary(raw)]


# ── File-level chunking ────────────────────────────────────────────────────────

def chunk_file(file_path: str, repo_name: str) -> list[CodeChunk]:
    """
    Parse a single source file with tree-sitter and extract:

      1. All top-level functions and classes (tiered into full/split/summary)
      2. A "docstring" chunk for every function/class that has one
      3. A "module_level" chunk capturing top-level code outside any symbol
         (constants, type aliases, imports, module-level expressions)

    Returns an empty list for unsupported extensions, unreadable files,
    or files with no content.
    """
    ext         = os.path.splitext(file_path)[1]
    lang_config = LANGUAGE_REGISTRY.get(ext)
    if not lang_config:
        return []

    language   = lang_config["name"]
    node_types = set(lang_config["node_types"])

    try:
        with open(file_path, "r", errors="ignore") as f:
            source = f.read()
    except Exception:
        return []

    if not source.strip():
        return []

    parser = _get_parser(language)
    tree   = parser.parse(bytes(source, "utf-8"))
    lines  = source.splitlines()

    raw_symbol_chunks: list[CodeChunk] = []
    docstring_chunks:  list[CodeChunk] = []

    def walk(node):
        if node.type in node_types:
            start = node.start_point[0]
            end   = node.end_point[0]

            # Extract symbol name from the first identifier child
            symbol = "unknown"
            for child in node.children:
                if child.type == "identifier":
                    symbol = child.text.decode("utf-8")
                    break

            content = "\n".join(lines[start : end + 1])
            raw_symbol_chunks.append(CodeChunk(
                content=content,
                file_path=str(file_path),
                repo_name=repo_name,
                symbol_name=symbol,
                start_line=start,
                end_line=end,
                language=language,
                chunk_type="full",
            ))

            # ── 4.3: Extract docstring as a sibling chunk ──────────────────
            docstring = _extract_docstring(node, lines)
            if docstring and docstring.strip():
                docstring_chunks.append(CodeChunk(
                    content=docstring,
                    file_path=str(file_path),
                    repo_name=repo_name,
                    symbol_name=symbol,
                    start_line=start,
                    end_line=start,
                    language=language,
                    chunk_type="docstring",
                    parent_symbol=symbol,
                ))

        for child in node.children:
            walk(child)

    walk(tree.root_node)

    # ── Graph extraction — reuse the already-parsed tree (zero extra cost) ─────
    # Deferred import avoids circular dependency at module load time.
    try:
        from core.graph import extract_and_store_graph  # noqa: PLC0415
        extract_and_store_graph(file_path, repo_name, tree, lines, language)
    except Exception:  # noqa: BLE001 — graph failures must never break chunking
        pass

    # Apply 3-tier strategy (includes 4.1 overlap) to every raw symbol chunk
    result: list[CodeChunk] = []
    covered_lines: set[int] = set()

    for raw in raw_symbol_chunks:
        result.extend(_apply_size_strategy(raw))
        covered_lines.update(range(raw.start_line, raw.end_line + 1))

    # Add docstring chunks (bypasses size strategy — always small)
    result.extend(docstring_chunks)

    # ── 4.2: Module-level chunk ────────────────────────────────────────────────
    # Collect lines not covered by any extracted symbol (constants, type aliases,
    # module-level imports, decorated expressions, etc.)
    module_lines: list[tuple[int, str]] = [
        (i, line) for i, line in enumerate(lines)
        if i not in covered_lines and line.strip()
    ]

    if module_lines:
        content_parts: list[str] = []
        prev_lineno: int | None   = None
        for lineno, line in module_lines:
            # Preserve a single blank separator between non-contiguous blocks
            if prev_lineno is not None and lineno > prev_lineno + 1:
                content_parts.append("")
            content_parts.append(line)
            prev_lineno = lineno

        module_content = "\n".join(content_parts)
        if module_content.strip():
            result.append(CodeChunk(
                content=module_content,
                file_path=str(file_path),
                repo_name=repo_name,
                symbol_name="__module__",
                start_line=module_lines[0][0],
                end_line=module_lines[-1][0],
                language=language,
                chunk_type="module_level",
            ))

    return result


# ── Repository-level chunking ──────────────────────────────────────────────────

# Directories to skip — never contain meaningful source code
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules",
    ".venv", "venv", "env",
    "dist", "build", "target",
    ".mypy_cache", ".pytest_cache", ".tox",
    ".eggs", "*.egg-info",
}

def chunk_repository(repo_path: str, repo_name: str) -> list[CodeChunk]:
    """
    Walk an entire repository and chunk every supported source file.
    Skips hidden directories, virtual environments, and build artifacts.

    Files are parsed in parallel using a thread pool — tree-sitter is a C
    extension that releases the GIL, so multiple files are parsed concurrently.
    Each thread gets its own Parser instance via thread-local storage.
    """
    all_files: list[str] = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            d for d in dirs
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for file_name in files:
            all_files.append(os.path.join(root, file_name))

    if not all_files:
        return []

    max_workers = min(os.cpu_count() or 4, 8)
    all_chunks: list[CodeChunk] = []

    with Progress(transient=False) as progress:
        task = progress.add_task("  Chunking files...", total=len(all_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(chunk_file, fp, repo_name): fp
                for fp in all_files
            }
            for future in as_completed(futures):
                try:
                    all_chunks.extend(future.result())
                except Exception:  # noqa: BLE001 — never let one file break the whole index
                    pass
                progress.advance(task)

    return all_chunks
