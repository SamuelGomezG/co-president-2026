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


# def test_resolve_data_dir_package_resolution(
#     monkeypatch: pytest.MonkeyPatch, tmp_path: Path
# ) -> None:
#     """Test data directory resolution via package location."""
#     # 1. Setup fake package structure
#     # Pkg structure: pkg/a/b/co_president/__init__.py
#     # Data dir: pkg/data/
#     
#     pkg_root = tmp_path / "pkg"
#     pkg_sub = pkg_root / "a" / "b" / "co_president"
#     pkg_sub.mkdir(parents=True)
#     init_file = pkg_sub / "__init__.py"
#     init_file.touch()
#     
#     data_dir = pkg_root / "data"
#     data_dir.mkdir()
#     
#     # 2. Mock co_president module
#     mock_pkg = MagicMock()
#     mock_pkg.__file__ = str(init_file)
#     monkeypatch.setitem(sys.modules, "co_president", mock_pkg)
#     
#     # 3. Assert
#     assert resolve_data_dir(None) == data_dir

# def test_resolve_data_dir_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
#     """Test data directory resolution with fallback."""
#     # 1. Force package resolution failure
#     monkeypatch.setitem(sys.modules, "co_president", None)
#     
#     # 2. Setup file-relative data dir
#     # The function uses Path(__file__).parent.parent.parent / "data"
#     # We need to monkeypatch pathlib.Path to return a fake location for __file__
#     
#     fake_root = tmp_path / "fake_root"
#     data_dir = fake_root / "data"
#     data_dir.mkdir(parents=True)
#     
#     # Mock __file__ to be inside fake_root/src/co_president/paths.py
#     fake_file = fake_root / "src" / "co_president" / "paths.py"
#     fake_file.parent.mkdir(parents=True)
#     
#     with patch("co_president.paths.__file__", str(fake_file)):
#         assert resolve_data_dir(None) == data_dir



def test_resolve_data_dir_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test data directory resolution with fallback."""
    # 1. Force package resolution failure
    monkeypatch.setitem(sys.modules, "co_president", None)

    # 2. Setup file-relative data dir
    # The function uses Path(__file__).parent.parent.parent / "data"
    # We need to monkeypatch pathlib.Path to return a fake location for __file__

    fake_root = tmp_path / "fake_root"
    data_dir = fake_root / "data"
    data_dir.mkdir(parents=True)

    # Mock __file__ to be inside fake_root/src/co_president/paths.py
    fake_file = fake_root / "src" / "co_president" / "paths.py"
    fake_file.parent.mkdir(parents=True)

    with patch("co_president.paths.__file__", str(fake_file)):
        assert resolve_data_dir(None) == data_dir


def test_resolve_data_dir_none_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test data directory resolution with failure on both package and fallback."""
    # 1. Force package resolution failure
    monkeypatch.setitem(sys.modules, "co_president", None)

    # 2. Force fallback check to fail
    with (
        patch("pathlib.Path.is_dir", return_value=False),
        pytest.raises(FileNotFoundError, match="data/ directory not found at"),
    ):
        resolve_data_dir(None)
