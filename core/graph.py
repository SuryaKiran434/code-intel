"""
core/graph.py

Builds a lightweight import + call graph at index time using the same
tree-sitter AST already parsed during chunking (zero extra parsing cost).

Stored in SQLite (import_edges, call_edges tables defined in core/db.py).
Used at query time by core/retriever.py to expand direct vector-search
results with dependency chunks (cross-file context).

Supported patterns (Python):
    import os
    import os.path
    from os import path
    from os import path as p
    from os import path, getcwd
    from . import utils          (relative imports)
    from .utils import helper

Call edges:
    Direct function calls inside function/class bodies — e.g. foo() → "foo".
    Attribute method calls (obj.method()) are also tracked by method name.
"""

import sqlite3

from config import DB_PATH


# ── Per-language extractor registry ───────────────────────────────────────────
# Add a language by appending an entry: each value is (imports_fn, calls_fn).
# Mirrors the chunker's LANGUAGE_REGISTRY pattern so adding Java/Scala is
# purely additive — no edits to extract_and_store_graph required.

_GRAPH_EXTRACTORS: dict[str, tuple] = {
    # "language_name": (imports_fn(root_node) -> [(kind, module, symbol)],
    #                   calls_fn(root_node)   -> [(from_symbol, to_symbol)])
}


def register_graph_extractor(language: str, imports_fn, calls_fn) -> None:
    """Register import/call extractors for a language. Called at module load."""
    _GRAPH_EXTRACTORS[language] = (imports_fn, calls_fn)


# ── Graph extraction ───────────────────────────────────────────────────────────

def extract_and_store_graph(
    file_path: str,
    repo_name: str,
    tree,
    lines: list[str],
    language: str,
) -> None:
    """
    Extract import and call edges from an already-parsed tree-sitter AST.
    Writes results to SQLite, replacing any prior edges for this file.

    Called from core/chunker.chunk_file() after the AST is parsed so the
    tree is reused — no additional parsing cost.

    Non-critical: all exceptions are swallowed so a graph bug never breaks
    the indexing pipeline. Languages without a registered extractor are
    silently skipped.
    """
    extractors = _GRAPH_EXTRACTORS.get(language)
    if not extractors:
        return

    imports_fn, calls_fn = extractors
    try:
        import_edges = imports_fn(tree.root_node)
        call_edges   = calls_fn(tree.root_node)
        _persist(file_path, repo_name, import_edges, call_edges)
    except Exception:   # noqa: BLE001 — graph extraction must never crash indexing
        pass


def _extract_imports(root) -> list[tuple[str, str, str]]:
    """
    Walk the AST and return (kind, to_module, to_symbol) tuples.

    kind values:
        "import"      — `import X` or `import X as Y`
        "from_import" — `from X import Y` or `from X import Y as Z`
    """
    edges: list[tuple[str, str, str]] = []

    def walk(node):
        if node.type == "import_statement":
            # import X  /  import X as Y  /  import X, Y
            for child in node.children:
                if child.type == "dotted_name":
                    edges.append(("import", _text(child), "*"))
                elif child.type == "aliased_import":
                    for sub in child.children:
                        if sub.type == "dotted_name":
                            edges.append(("import", _text(sub), "*"))
                            break

        elif node.type == "import_from_statement":
            # from X import Y  /  from X import Y as Z  /  from . import Y
            module  = ""
            symbols: list[str] = []
            # Track whether we've set the module yet.  In `from X import Y`
            # both X and Y appear as dotted_name children; the first one is
            # the module, subsequent ones are imported symbols.
            module_seen = False
            for child in node.children:
                if child.type in ("dotted_name", "relative_import"):
                    if not module_seen:
                        module = _text(child)
                        module_seen = True
                    else:
                        # Single-symbol import not wrapped in import_list
                        symbols.append(_text(child))
                elif child.type == "wildcard_import":
                    symbols.append("*")
                elif child.type == "import_list":
                    for item in child.children:
                        if item.type == "identifier":
                            symbols.append(_text(item))
                        elif item.type == "aliased_import":
                            for sub in item.children:
                                if sub.type == "identifier":
                                    symbols.append(_text(sub))
                                    break
                elif child.type == "aliased_import" and module_seen:
                    # Single aliased import: `from X import Y as Z`
                    # Capture the original name (first identifier/dotted_name)
                    for sub in child.children:
                        if sub.type in ("identifier", "dotted_name"):
                            symbols.append(_text(sub))
                            break
                elif child.type == "identifier" and module_seen:
                    # Single identifier import (e.g. `from os import path`)
                    symbols.append(_text(child))

            target = module or "."
            if symbols:
                for sym in symbols:
                    edges.append(("from_import", target, sym))
            else:
                edges.append(("from_import", target, "*"))

        for child in node.children:
            walk(child)

    walk(root)
    return edges


# Python builtins we deliberately don't track as call-graph edges — they're
# noise and dominate every node in the graph. Keep this list explicit rather
# than relying on heuristics like "lowercase short name" so PascalCase
# constructors (MyClass()) and uppercase-prefixed helpers are NOT dropped.
_BUILTIN_CALLEES = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "compile", "complex", "delattr",
    "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter",
    "float", "format", "frozenset", "getattr", "globals", "hasattr",
    "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass",
    "iter", "len", "list", "locals", "map", "max", "memoryview", "min",
    "next", "object", "oct", "open", "ord", "pow", "print", "property",
    "range", "repr", "reversed", "round", "set", "setattr", "slice",
    "sorted", "staticmethod", "str", "sum", "super", "tuple", "type",
    "vars", "zip",
})


def _extract_calls(root) -> list[tuple[str, str]]:
    """
    Walk the AST and return (from_symbol, to_symbol) call pairs.

    Tracks two call shapes:
      foo()          → ("ctx", "foo")
      obj.method()   → ("ctx", "method")    — covers OOP dispatch

    Filters: skips Python builtins (see _BUILTIN_CALLEES), skips self-calls.
    PascalCase callees (e.g. MyClass()) ARE tracked — they're typically
    constructors, which is exactly the dependency edge we want to surface.
    """
    seen: set[tuple[str, str]] = set()

    def _record(ctx: str, callee: str) -> None:
        if not callee or callee == ctx or callee in _BUILTIN_CALLEES:
            return
        pair = (ctx, callee)
        if pair not in seen:
            seen.add(pair)

    def walk(node, ctx: str):
        if node.type in ("function_definition", "class_definition"):
            for child in node.children:
                if child.type == "identifier":
                    ctx = _text(child)
                    break

        if node.type == "call":
            func = node.children[0] if node.children else None
            if func and func.type == "identifier":
                _record(ctx, _text(func))
            elif func and func.type == "attribute":
                attr_node = func.child_by_field_name("attribute")
                if attr_node and attr_node.type == "identifier":
                    _record(ctx, _text(attr_node))

        for child in node.children:
            walk(child, ctx)

    walk(root, "__module__")
    return list(seen)


def _text(node) -> str:
    """Decode a tree-sitter node's text to a UTF-8 string."""
    return node.text.decode("utf-8", errors="ignore").strip()


# ── SQLite persistence ─────────────────────────────────────────────────────────

def _persist(
    file_path: str,
    repo_name: str,
    import_edges: list[tuple[str, str, str]],
    call_edges: list[tuple[str, str]],
) -> None:
    """Replace all graph edges for this file with freshly extracted ones."""
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            "DELETE FROM import_edges WHERE repo_name = ? AND from_file = ?",
            (repo_name, file_path),
        )
        conn.execute(
            "DELETE FROM call_edges WHERE repo_name = ? AND from_file = ?",
            (repo_name, file_path),
        )
        if import_edges:
            conn.executemany(
                """INSERT INTO import_edges
                   (repo_name, from_file, kind, to_module, to_symbol)
                   VALUES (?, ?, ?, ?, ?)""",
                [(repo_name, file_path, k, m, s) for k, m, s in import_edges],
            )
        if call_edges:
            conn.executemany(
                """INSERT INTO call_edges
                   (repo_name, from_file, from_symbol, to_symbol)
                   VALUES (?, ?, ?, ?)""",
                [(repo_name, file_path, frm, to) for frm, to in call_edges],
            )


# ── Query helpers (used by retriever.py) ──────────────────────────────────────

def get_callees(repo_name: str, file_path: str, symbol_name: str) -> list[str]:
    """
    Return symbol names called by `symbol_name` in `file_path`.
    Used to expand direct vector-search hits with their dependencies.
    """
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            rows = conn.execute(
                """SELECT to_symbol FROM call_edges
                   WHERE repo_name = ? AND from_file = ? AND from_symbol = ?""",
                (repo_name, file_path, symbol_name),
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:   # noqa: BLE001
        return []


def get_callers(repo_name: str, symbol_name: str) -> list[dict]:
    """
    Return (file_path, symbol_name) pairs that call `symbol_name` in this repo.
    """
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            rows = conn.execute(
                """SELECT from_file, from_symbol FROM call_edges
                   WHERE repo_name = ? AND to_symbol = ?""",
                (repo_name, symbol_name),
            ).fetchall()
        return [{"file_path": r[0], "symbol_name": r[1]} for r in rows]
    except Exception:   # noqa: BLE001
        return []


def get_imports(repo_name: str, file_path: str) -> list[dict]:
    """
    Return all import edges from a given file.
    """
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            rows = conn.execute(
                """SELECT kind, to_module, to_symbol FROM import_edges
                   WHERE repo_name = ? AND from_file = ?""",
                (repo_name, file_path),
            ).fetchall()
        return [{"kind": r[0], "to_module": r[1], "to_symbol": r[2]} for r in rows]
    except Exception:   # noqa: BLE001
        return []


# ── Cleanup helpers (used by diff_tracker.py) ─────────────────────────────────

def delete_file_graph(file_path: str, repo_name: str) -> None:
    """Remove all graph edges for a deleted file."""
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute(
                "DELETE FROM import_edges WHERE repo_name = ? AND from_file = ?",
                (repo_name, file_path),
            )
            conn.execute(
                "DELETE FROM call_edges WHERE repo_name = ? AND from_file = ?",
                (repo_name, file_path),
            )
    except Exception:   # noqa: BLE001
        pass


def delete_repo_graph(repo_name: str) -> None:
    """Remove all graph edges for a repo (called on repo removal)."""
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute(
                "DELETE FROM import_edges WHERE repo_name = ?", (repo_name,)
            )
            conn.execute(
                "DELETE FROM call_edges WHERE repo_name = ?", (repo_name,)
            )
    except Exception:   # noqa: BLE001
        pass


# ── Built-in language registrations ───────────────────────────────────────────
# To add Java/Scala: write language-specific extractors using the same return
# shapes and register them here.

register_graph_extractor("python", _extract_imports, _extract_calls)
