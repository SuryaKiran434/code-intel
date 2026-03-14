"""
tests/test_graph.py

Tests for core/graph.py — import and call graph extraction.

Covers:
  - import_statement extraction (bare and aliased)
  - import_from_statement extraction (single, multiple, relative)
  - call edge extraction (bare function calls only)
  - filtering: method calls, self-calls, builtins
  - SQLite round-trip for get_callees / get_callers / get_imports
  - Non-Python files silently skipped
"""

import textwrap

import pytest
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from core.graph import (
    _extract_imports,
    _extract_calls,
    extract_and_store_graph,
    get_callees,
    get_callers,
    get_imports,
)


# ── Tree-sitter helper ─────────────────────────────────────────────────────────

def _parse(source: str):
    """Parse Python source and return (tree, lines)."""
    lang = Language(tspython.language())
    parser = Parser(lang)
    tree = parser.parse(source.encode())
    return tree, source.splitlines()


# ── Import extraction ──────────────────────────────────────────────────────────

def test_import_statement_bare():
    tree, lines = _parse("import os\n")
    edges = _extract_imports(tree.root_node)
    assert ("import", "os", "*") in edges


def test_import_statement_multiple():
    tree, lines = _parse("import os\nimport sys\n")
    edges = _extract_imports(tree.root_node)
    modules = {e[1] for e in edges}
    assert "os" in modules
    assert "sys" in modules


def test_import_statement_dotted():
    tree, lines = _parse("import os.path\n")
    edges = _extract_imports(tree.root_node)
    modules = {e[1] for e in edges}
    assert "os.path" in modules


def test_from_import_single():
    tree, lines = _parse("from pathlib import Path\n")
    edges = _extract_imports(tree.root_node)
    assert ("from_import", "pathlib", "Path") in edges


def test_from_import_multiple():
    tree, lines = _parse("from os import getcwd, listdir\n")
    edges = _extract_imports(tree.root_node)
    symbols = {(e[1], e[2]) for e in edges}
    assert ("os", "getcwd") in symbols
    assert ("os", "listdir") in symbols


def test_from_import_aliased():
    tree, lines = _parse("from pathlib import Path as P\n")
    edges = _extract_imports(tree.root_node)
    # The original name (before 'as') should be captured
    symbols = {e[2] for e in edges}
    assert "Path" in symbols


def test_relative_import():
    tree, lines = _parse("from . import utils\n")
    edges = _extract_imports(tree.root_node)
    symbols = {e[2] for e in edges}
    assert "utils" in symbols


def test_relative_import_with_symbol():
    tree, lines = _parse("from .helpers import clean\n")
    edges = _extract_imports(tree.root_node)
    symbols = {e[2] for e in edges}
    assert "clean" in symbols


# ── Call edge extraction ───────────────────────────────────────────────────────

def test_bare_function_calls():
    source = textwrap.dedent("""\
        def foo():
            bar()
            baz()
    """)
    tree, lines = _parse(source)
    calls = _extract_calls(tree.root_node)
    callees = {to for frm, to in calls if frm == "foo"}
    assert "bar" in callees
    assert "baz" in callees


def test_method_calls_excluded():
    """obj.bar() should NOT produce a call edge — only bare identifier calls."""
    source = textwrap.dedent("""\
        def foo():
            obj.bar()
            self.helper()
    """)
    tree, lines = _parse(source)
    calls = _extract_calls(tree.root_node)
    callees = {to for _, to in calls}
    assert "bar" not in callees
    assert "helper" not in callees


def test_self_call_excluded():
    """A function calling itself should not produce a call edge."""
    source = textwrap.dedent("""\
        def recursive():
            recursive()
    """)
    tree, lines = _parse(source)
    calls = _extract_calls(tree.root_node)
    assert ("recursive", "recursive") not in calls


def test_call_edges_deduplicated():
    """Calling the same function twice should yield only one edge."""
    source = textwrap.dedent("""\
        def foo():
            helper()
            helper()
    """)
    tree, lines = _parse(source)
    calls = _extract_calls(tree.root_node)
    foo_to_helper = [(f, t) for f, t in calls if f == "foo" and t == "helper"]
    assert len(foo_to_helper) == 1


def test_calls_scoped_to_function():
    """Call edges should be attributed to the containing function, not __module__."""
    source = textwrap.dedent("""\
        def alpha():
            beta()

        def gamma():
            delta()
    """)
    tree, lines = _parse(source)
    calls = _extract_calls(tree.root_node)
    alpha_callees = {to for frm, to in calls if frm == "alpha"}
    gamma_callees = {to for frm, to in calls if frm == "gamma"}
    assert "beta" in alpha_callees
    assert "delta" in gamma_callees
    # Cross-contamination check
    assert "delta" not in alpha_callees
    assert "beta" not in gamma_callees


# ── SQLite round-trips ─────────────────────────────────────────────────────────

def test_get_callees_round_trip(tmp_db, tmp_path):
    source = textwrap.dedent("""\
        def alpha():
            beta()
            gamma()

        def beta():
            pass

        def gamma():
            pass
    """)
    f = tmp_path / "calls.py"
    f.write_text(source)
    tree, lines = _parse(source)
    extract_and_store_graph(str(f), "myrepo", tree, lines, "python")

    callees = get_callees("myrepo", str(f), "alpha")
    assert set(callees) == {"beta", "gamma"}


def test_get_callees_empty_for_leaf(tmp_db, tmp_path):
    source = textwrap.dedent("""\
        def leaf():
            pass
    """)
    f = tmp_path / "leaf.py"
    f.write_text(source)
    tree, lines = _parse(source)
    extract_and_store_graph(str(f), "myrepo", tree, lines, "python")

    callees = get_callees("myrepo", str(f), "leaf")
    assert callees == []


def test_get_callers(tmp_db, tmp_path):
    source = textwrap.dedent("""\
        def caller_a():
            shared()

        def caller_b():
            shared()

        def shared():
            pass
    """)
    f = tmp_path / "callers.py"
    f.write_text(source)
    tree, lines = _parse(source)
    extract_and_store_graph(str(f), "myrepo", tree, lines, "python")

    callers = get_callers("myrepo", "shared")
    caller_names = {r["symbol_name"] for r in callers}
    assert "caller_a" in caller_names
    assert "caller_b" in caller_names


def test_get_imports_round_trip(tmp_db, tmp_path):
    source = "import os\nfrom pathlib import Path\n"
    f = tmp_path / "imports.py"
    f.write_text(source)
    tree, lines = _parse(source)
    extract_and_store_graph(str(f), "myrepo", tree, lines, "python")

    imports = get_imports("myrepo", str(f))
    modules = {r["to_module"] for r in imports}
    assert "os" in modules
    assert "pathlib" in modules


def test_graph_replaces_stale_edges(tmp_db, tmp_path):
    """Re-indexing a file should delete old edges before inserting new ones."""
    source_v1 = textwrap.dedent("""\
        def foo():
            old_dep()
    """)
    source_v2 = textwrap.dedent("""\
        def foo():
            new_dep()
    """)
    f = tmp_path / "evolving.py"

    f.write_text(source_v1)
    tree1, lines1 = _parse(source_v1)
    extract_and_store_graph(str(f), "myrepo", tree1, lines1, "python")

    f.write_text(source_v2)
    tree2, lines2 = _parse(source_v2)
    extract_and_store_graph(str(f), "myrepo", tree2, lines2, "python")

    callees = get_callees("myrepo", str(f), "foo")
    assert "new_dep" in callees
    assert "old_dep" not in callees


# ── Non-Python files ───────────────────────────────────────────────────────────

def test_non_python_file_silently_skipped():
    """extract_and_store_graph should not raise for non-Python languages."""

    class _FakeTree:
        root_node = None

    # Should return None without raising
    result = extract_and_store_graph(
        "/fake/main.go", "myrepo", _FakeTree(), [], "go"
    )
    assert result is None
