"""Shared integration test fixtures.

These fixtures load full production CSV datasets and are module-scoped
for performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from co_president.data import (
    CleanPolls,
    RoundResult,
    load_and_clean_all,
    load_canonical_results,
)


@pytest.fixture(scope="module")
def canonical_results(data_dir: Path) -> tuple[RoundResult, RoundResult]:
    """Load canonical results once per module."""
    return load_canonical_results(data_dir)


@pytest.fixture(scope="module")
def clean_polls_fixture(data_dir: Path) -> CleanPolls:
    """Load and clean polls once per module."""
    return load_and_clean_all(data_dir)
