"""Shared pytest fixtures for the co-president test suite."""

from pathlib import Path

import pytest


def _find_data_dir() -> Path:
    """Locate the project data directory.

    Prefers package-relative resolution (works regardless of test
    directory structure), falls back to test-file-relative resolution
    when the package is not installed (e.g., during first-time setup).
    """
    try:
        import co_president  # noqa: PLC0415 — not always importable top-level

        candidate = Path(co_president.__file__).resolve().parent.parent / "data"
        if candidate.is_dir():
            return candidate
    except (ImportError, AttributeError):
        pass
    fallback = Path(__file__).resolve().parent.parent / "data"
    if not fallback.is_dir():
        msg = f"data/ directory not found at {fallback}"
        raise FileNotFoundError(msg)
    return fallback


@pytest.fixture
def data_dir() -> Path:
    """Path to the project's data directory.

    All test data files are relative to this directory.
    """
    return _find_data_dir()
