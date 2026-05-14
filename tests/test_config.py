"""SPEC-01: Test that the co_president package is properly wired."""

import importlib


def test_project_exists() -> None:
    """Verify the package can be imported and exposes a version string."""
    module = importlib.import_module("co_president")
    assert isinstance(getattr(module, "__version__", ""), str)
    assert module.__version__
