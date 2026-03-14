"""
tests/conftest.py

Shared pytest fixtures for the Code Intel test suite.
"""

import textwrap
import pytest


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """
    Create a fresh SQLite DB with the full schema in a temp directory.

    Monkeypatches config.DB_PATH and every module that imported DB_PATH at
    module load time, so all core functions write to the isolated test DB
    rather than the real one.
    """
    db_file = tmp_path / "test_code_intel.db"

    import config
    import core.db
    import core.graph
    import core.query_expander

    monkeypatch.setattr(config, "DB_PATH", db_file)
    monkeypatch.setattr(core.db, "DB_PATH", db_file)
    monkeypatch.setattr(core.graph, "DB_PATH", db_file)
    monkeypatch.setattr(core.query_expander, "DB_PATH", db_file)

    from core.db import init_db
    init_db()

    yield db_file


@pytest.fixture()
def sample_py_source():
    """Python source with diverse constructs used across chunker / graph tests."""
    return textwrap.dedent("""\
        import os
        from pathlib import Path

        MAX_SIZE = 100
        DEFAULT_NAME = "hello"

        def small_func(x: int) -> int:
            \"\"\"A small function that doubles its input.\"\"\"
            return x * 2

        def helper(name: str) -> str:
            return name.upper()

        def caller_func(items):
            result = small_func(len(items))
            msg = helper("done")
            return result

        class MyClass:
            \"\"\"A simple class for testing.\"\"\"

            def __init__(self, value: int):
                self.value = value

            def compute(self) -> int:
                return small_func(self.value)
    """)


@pytest.fixture()
def sample_py_file(tmp_path, sample_py_source):
    """Write sample_py_source to a temp .py file and return its absolute path."""
    f = tmp_path / "sample.py"
    f.write_text(sample_py_source)
    return str(f)
