"""
tests/test_chunker.py

Tests for core/chunker.py — AST-aware code chunking.

Covers:
  - All 5 chunk types: full, split_part, summary, docstring, module_level
  - 3-tier size strategy (small / medium / large)
  - Metadata correctness (repo_name, language, line numbers)
  - Unsupported extensions and empty files
"""

import pytest

from core.chunker import chunk_file, CodeChunk


# ── Negative cases ─────────────────────────────────────────────────────────────

def test_unsupported_extension_returns_empty(tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# Hello")
    assert chunk_file(str(f), "myrepo") == []


def test_yaml_extension_returns_empty(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("key: value\n")
    assert chunk_file(str(f), "myrepo") == []


def test_empty_file_returns_empty(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("   \n  \n")
    assert chunk_file(str(f), "myrepo") == []


def test_missing_file_returns_empty():
    assert chunk_file("/nonexistent/path/file.py", "myrepo") == []


# ── Full chunks (Tier 1 — small, ≤ 60 lines) ──────────────────────────────────

def test_small_function_produces_full_chunk(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    full_names = {c.symbol_name for c in chunks if c.chunk_type == "full"}
    assert "small_func" in full_names


def test_small_function_has_no_summary_or_split(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    small_func_subtypes = {
        c.chunk_type for c in chunks
        if c.symbol_name.startswith("small_func")
        and c.chunk_type in ("split_part", "summary")
    }
    assert not small_func_subtypes


# ── Docstring chunks (Phase 4.3) ───────────────────────────────────────────────

def test_docstring_chunks_extracted(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    docstring_symbols = {c.symbol_name for c in chunks if c.chunk_type == "docstring"}
    assert "small_func" in docstring_symbols
    assert "MyClass" in docstring_symbols


def test_docstring_chunk_contains_docstring_text(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    ds = next(c for c in chunks
              if c.chunk_type == "docstring" and c.symbol_name == "small_func")
    assert "doubles its input" in ds.content


def test_no_docstring_for_helper(sample_py_file):
    """helper() has no docstring — should produce no docstring chunk."""
    chunks = chunk_file(sample_py_file, "myrepo")
    docstring_symbols = {c.symbol_name for c in chunks if c.chunk_type == "docstring"}
    assert "helper" not in docstring_symbols


# ── Module-level chunk (Phase 4.2) ─────────────────────────────────────────────

def test_module_level_chunk_present(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    module_chunks = [c for c in chunks if c.chunk_type == "module_level"]
    assert len(module_chunks) == 1


def test_module_level_chunk_contains_constants(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    mc = next(c for c in chunks if c.chunk_type == "module_level")
    assert "MAX_SIZE" in mc.content or "DEFAULT_NAME" in mc.content


def test_module_level_symbol_name(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    mc = next(c for c in chunks if c.chunk_type == "module_level")
    assert mc.symbol_name == "__module__"


# ── Medium function (Tier 2 — 61–150 lines) ────────────────────────────────────

def test_medium_function_produces_full_and_summary(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(80))
    source = f"def medium_func():\n{body}\n"
    f = tmp_path / "medium.py"
    f.write_text(source)
    chunks = chunk_file(str(f), "testrepo")
    types = {c.chunk_type for c in chunks}
    assert "full" in types
    assert "summary" in types
    assert "split_part" not in types


def test_medium_summary_has_parent_symbol(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(80))
    source = f"def medium_func():\n{body}\n"
    f = tmp_path / "medium.py"
    f.write_text(source)
    chunks = chunk_file(str(f), "testrepo")
    summary = next(c for c in chunks if c.chunk_type == "summary")
    assert summary.parent_symbol == "medium_func"


# ── Large function (Tier 3 — 151+ lines) ──────────────────────────────────────

def test_large_function_produces_split_parts_and_summary(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(160))
    source = f"def big_func():\n{body}\n"
    f = tmp_path / "big.py"
    f.write_text(source)
    chunks = chunk_file(str(f), "testrepo")
    split_chunks = [c for c in chunks if c.chunk_type == "split_part"]
    summary_chunks = [c for c in chunks if c.chunk_type == "summary"]
    assert len(split_chunks) >= 2
    assert len(summary_chunks) == 1


def test_large_function_summary_parent_symbol(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(160))
    source = f"def big_func():\n{body}\n"
    f = tmp_path / "big.py"
    f.write_text(source)
    chunks = chunk_file(str(f), "testrepo")
    summary = next(c for c in chunks if c.chunk_type == "summary")
    assert summary.parent_symbol == "big_func"


def test_split_parts_cover_all_lines(tmp_path):
    """Split parts together should cover the full range of the function body."""
    body = "\n".join(f"    x{i} = {i}" for i in range(160))
    source = f"def big_func():\n{body}\n"
    f = tmp_path / "big.py"
    f.write_text(source)
    chunks = chunk_file(str(f), "testrepo")
    split_chunks = sorted(
        [c for c in chunks if c.chunk_type == "split_part"],
        key=lambda c: c.start_line,
    )
    # First part starts at line 0 (def big_func line)
    assert split_chunks[0].start_line == 0
    # Last part ends near the last line
    assert split_chunks[-1].end_line >= 155


# ── Metadata invariants ────────────────────────────────────────────────────────

def test_chunk_metadata_correct(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    for chunk in chunks:
        assert isinstance(chunk, CodeChunk)
        assert chunk.repo_name == "myrepo"
        assert chunk.language == "python"
        assert chunk.start_line >= 0
        assert chunk.end_line >= chunk.start_line
        assert chunk.line_count == chunk.end_line - chunk.start_line + 1


def test_chunk_file_path_matches(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    for chunk in chunks:
        assert chunk.file_path == sample_py_file


def test_class_produces_full_chunk(sample_py_file):
    chunks = chunk_file(sample_py_file, "myrepo")
    full_names = {c.symbol_name for c in chunks if c.chunk_type == "full"}
    assert "MyClass" in full_names
