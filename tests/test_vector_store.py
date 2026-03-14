"""
tests/test_vector_store.py

Tests for core/vector_store.py — Milvus operations.

All Milvus interactions are mocked — no running Milvus instance required.
Covers:
  - partition_name() sanitization rules
  - ensure_partition(): creates when missing, skips when present
  - reinsert_with_new_path(): correct column-oriented payload, updated path
  - delete_chunks_by_ids(): no-op on empty list
"""

from unittest.mock import MagicMock, call, patch

import pytest

from core.vector_store import partition_name, ensure_partition


# ── partition_name ─────────────────────────────────────────────────────────────

def test_partition_name_alphanumeric_unchanged():
    assert partition_name("myrepo") == "myrepo"
    assert partition_name("MyRepo123") == "MyRepo123"


def test_partition_name_hyphens_converted():
    assert partition_name("my-repo-name") == "my_repo_name"


def test_partition_name_dots_converted():
    assert partition_name("my.repo.v2") == "my_repo_v2"


def test_partition_name_spaces_converted():
    assert partition_name("my repo") == "my_repo"


def test_partition_name_leading_digit_prefixed():
    result = partition_name("123repo")
    assert result.startswith("r_")
    assert not result[0].isdigit()


def test_partition_name_empty_string():
    assert partition_name("") == "_default"


def test_partition_name_only_special_chars():
    result = partition_name("---")
    # All hyphens → "___", starts with underscore (not digit) → no prefix
    assert result == "___"


def test_partition_name_mixed():
    result = partition_name("repo-name.v2-final")
    assert "-" not in result
    assert "." not in result
    assert result == "repo_name_v2_final"


# ── ensure_partition ───────────────────────────────────────────────────────────

def test_ensure_partition_creates_when_missing():
    mock_col = MagicMock()
    mock_col.has_partition.return_value = False

    pname = ensure_partition(mock_col, "my-repo")

    assert pname == "my_repo"
    mock_col.create_partition.assert_called_once_with("my_repo")


def test_ensure_partition_skips_when_existing():
    mock_col = MagicMock()
    mock_col.has_partition.return_value = True

    pname = ensure_partition(mock_col, "existing-repo")

    mock_col.create_partition.assert_not_called()
    assert pname == "existing_repo"


def test_ensure_partition_returns_sanitized_name():
    mock_col = MagicMock()
    mock_col.has_partition.return_value = True

    pname = ensure_partition(mock_col, "123-my.repo")
    assert pname == "r_123_my_repo"


# ── reinsert_with_new_path ─────────────────────────────────────────────────────

def test_reinsert_with_new_path_updates_file_path():
    from core.vector_store import reinsert_with_new_path

    mock_col = MagicMock()
    mock_col.has_partition.return_value = True

    records = [
        {
            "id": "abc123",
            "embedding": [0.1] * 1024,
            "content": "def foo(): pass",
            "file_path": "/old/path/foo.py",
            "repo_name": "myrepo",
            "symbol_name": "foo",
            "start_line": 0,
            "end_line": 0,
            "language": "python",
            "chunk_type": "full",
            "parent_symbol": "",
        }
    ]

    reinsert_with_new_path(mock_col, records, "/new/path/foo.py")

    mock_col.insert.assert_called_once()
    inserted_data = mock_col.insert.call_args[0][0]

    # inserted_data is column-oriented: [ids, embeddings, contents, file_paths, ...]
    file_paths_col = inserted_data[3]
    assert file_paths_col == ["/new/path/foo.py"]


def test_reinsert_with_new_path_preserves_embeddings():
    from core.vector_store import reinsert_with_new_path

    mock_col = MagicMock()
    mock_col.has_partition.return_value = True

    embedding = [0.42] * 1024
    records = [
        {
            "id": "deadbeef",
            "embedding": embedding,
            "content": "def bar(): pass",
            "file_path": "/old/bar.py",
            "repo_name": "myrepo",
            "symbol_name": "bar",
            "start_line": 0,
            "end_line": 0,
            "language": "python",
            "chunk_type": "full",
            "parent_symbol": "",
        }
    ]

    reinsert_with_new_path(mock_col, records, "/new/bar.py")

    inserted_data = mock_col.insert.call_args[0][0]
    embeddings_col = inserted_data[1]
    assert embeddings_col[0] == embedding


def test_reinsert_with_new_path_noop_on_empty():
    from core.vector_store import reinsert_with_new_path

    mock_col = MagicMock()
    reinsert_with_new_path(mock_col, [], "/new/path.py")
    mock_col.insert.assert_not_called()


# ── delete_chunks_by_ids ───────────────────────────────────────────────────────

def test_delete_chunks_by_ids_noop_on_empty():
    """Passing an empty list should not call Milvus at all."""
    from core.vector_store import delete_chunks_by_ids

    with patch("core.vector_store.get_or_create_collection") as mock_get:
        delete_chunks_by_ids([])
        mock_get.assert_not_called()


def test_delete_chunks_by_ids_calls_delete():
    """Non-empty list should call collection.delete with correct IN expression."""
    from core.vector_store import delete_chunks_by_ids

    mock_col = MagicMock()
    with patch("core.vector_store.get_or_create_collection", return_value=mock_col):
        delete_chunks_by_ids(["id1", "id2"])

    mock_col.delete.assert_called_once()
    expr = mock_col.delete.call_args[1].get("expr") or mock_col.delete.call_args[0][0]
    assert "id1" in expr
    assert "id2" in expr
