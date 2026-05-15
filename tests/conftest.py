"""Shared pytest fixtures for the co-president test suite."""

from pathlib import Path

import pytest

from co_president.data_results import _resolve_data_dir  # pyright: ignore[reportPrivateUsage]

__all__ = ["data_dir"]


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to the project's data directory.

    All test data files are relative to this directory. Session-scoped
    because the data directory path is deterministic and never changes
    during a test run.
    """
    return _resolve_data_dir(None)
