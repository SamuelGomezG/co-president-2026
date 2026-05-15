"""SPEC-03+04: Data directory path resolution.

Shared utility for locating the project data directory from either
a user-provided path, the installed package location, or a file-relative
fallback.
"""

from __future__ import annotations

from pathlib import Path


def resolve_data_dir(data_dir: Path | None) -> Path:
    """Locate the project data directory.

    When ``data_dir`` is provided, returns it directly. Otherwise resolves
    relative to the installed package, falling back to a file-relative path.

    Args:
        data_dir: User-provided path or ``None``.

    Returns:
        Absolute ``Path`` to the ``data/`` directory.

    Raises:
        FileNotFoundError: If the directory cannot be found.

    """
    if data_dir is not None:
        if not data_dir.is_dir():
            msg = f"data/ directory not found at {data_dir}"
            raise FileNotFoundError(msg)
        return data_dir
    try:
        import co_president  # noqa: PLC0415

        candidate = Path(co_president.__file__).resolve().parent.parent.parent / "data"
        if candidate.is_dir():
            return candidate
    except (ImportError, AttributeError, TypeError):
        pass
    fallback = Path(__file__).resolve().parent.parent.parent / "data"
    if not fallback.is_dir():
        msg = f"data/ directory not found at {fallback}"
        raise FileNotFoundError(msg)
    return fallback
