"""SPEC-01: Unit tests for paths.py data directory resolution."""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

from co_president.paths import resolve_data_dir


def test_resolve_data_dir_explicit_path_exists(tmp_path: Path) -> None:
    """Test data directory resolution with an explicit existing path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert resolve_data_dir(data_dir) == data_dir


def test_resolve_data_dir_explicit_path_missing() -> None:
    """Test data directory resolution with an explicit missing path."""
    with pytest.raises(FileNotFoundError, match="data/ directory not found at"):
        resolve_data_dir(Path("/non/existent/path"))


def test_resolve_data_dir_package_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test data directory resolution via package location."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pkg_dir = tmp_path / "src" / "co_president"
    pkg_dir.mkdir(parents=True)

    mock_pkg = MagicMock()
    mock_pkg.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "co_president", mock_pkg)

    assert resolve_data_dir(None) == data_dir


def test_resolve_data_dir_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test data directory resolution falls through when package path fails."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    paths_file = tmp_path / "src" / "co_president" / "paths.py"
    paths_file.parent.mkdir(parents=True)

    # Make package resolution fail: __file__ points to non-existent location
    mock_pkg = MagicMock()
    mock_pkg.__file__ = str(tmp_path / "nonexistent" / "co_president" / "__init__.py")
    monkeypatch.setitem(sys.modules, "co_president", mock_pkg)

    # Patch __file__ on the already-imported module (avoids import chain breakage)
    monkeypatch.setattr(sys.modules["co_president.paths"], "__file__", str(paths_file))

    assert resolve_data_dir(None) == data_dir


def test_resolve_data_dir_none_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test both resolution strategies fail and raise FileNotFoundError."""
    mock_pkg = MagicMock()
    mock_pkg.__file__ = str(Path("/nonexistent/co_president/__init__.py"))
    monkeypatch.setitem(sys.modules, "co_president", mock_pkg)

    with (
        patch("pathlib.Path.is_dir", return_value=False),
        pytest.raises(FileNotFoundError, match="data/ directory not found at"),
    ):
        resolve_data_dir(None)
