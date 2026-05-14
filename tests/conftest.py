"""Shared pytest fixtures for the co-president test suite."""

from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """Path to the project's data directory.

    All test data files are relative to this directory.
    """
    return Path(__file__).resolve().parents[1] / "data"
