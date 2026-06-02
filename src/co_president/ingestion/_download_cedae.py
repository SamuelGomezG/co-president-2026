"""SPEC-12.2 annex: Bulk download CEDAE election-results CSVs from S3.

Downloads CSV files from the CEDAE electoral results page
(``cedae.datasketch.co``) by parsing the embedded file metadata and
fetching each CSV from the S3 bucket it points at.

Filters let callers select by year, ``nivel`` (Camara / Presidencia /
Senado), and format (csv or dta).
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "compress_csvs",
    "download_cedae_elections",
    "fetch_cedae_index",
    "select_cedae_files",
]

logger = logging.getLogger(__name__)

_INDEX_URL = (
    "https://cedae.datasketch.co/datos-democracia/resultados-electorales/descarga-los-datos/"
)

_DEFAULT_YEARS: tuple[int, ...] = (2002, 2006, 2010, 2014, 2018)
_DEFAULT_NIVELES: tuple[str, ...] = ("Camara", "Presidencia", "Senado")
_CHUNK_SIZE: int = 1 << 20  # 1 MiB
_MISSING_INPUT_MSG = "Could not locate data-resultados input in CEDAE page"


def _entry_sort_key(entry: dict[str, object]) -> tuple[str, str]:
    return (str(entry["ano"]), str(entry["nivel"]))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_cedae_index() -> list[dict[str, object]]:
    """Fetch file metadata from the CEDAE downloads page.

    Returns:
        Raw list of file-metadata dicts, each with keys ``ano``,
        ``format``, ``nivel``, ``path``, ``size``, ``tipo``, ``titulo``.

    Raises:
        requests.RequestException: If the page is unreachable.
        RuntimeError: If the embedded metadata cannot be located.

    """
    response = requests.get(_INDEX_URL, timeout=30, headers={"User-Agent": "co-president/1.0"})
    response.raise_for_status()
    html = response.text
    match = re.search(r"id=data-resultados value='(.*?)'", html, re.DOTALL)
    if not match:
        raise RuntimeError(_MISSING_INPUT_MSG)
    return json.loads(match.group(1))


def select_cedae_files(
    entries: list[dict[str, object]],
    years: Sequence[int] = _DEFAULT_YEARS,
    niveles: Sequence[str] = _DEFAULT_NIVELES,
    *,
    fmt: str = "csv",
) -> list[dict[str, object]]:
    """Filter CEDAE file entries by year, nivel, and format.

    Args:
        entries: Raw metadata list from ``fetch_cedae_index``.
        years: Election years to include (int values).
        niveles: ``nivel`` values to include (``"Camara"``, ``"Presidencia"``, ``"Senado"``).
        fmt: File format filter (``"csv"`` or ``"dta"``).

    Returns:
        Filtered and sorted list of metadata dicts.

    """
    year_set = {str(y) for y in years}
    nivel_set = set(niveles)
    return sorted(
        [
            e
            for e in entries
            if e.get("format") == fmt and e.get("ano") in year_set and e.get("nivel") in nivel_set
        ],
        key=_entry_sort_key,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _download_one(url: str, dest: Path, expected_size: int | None) -> Path:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with (
        requests.get(
            url, stream=True, timeout=300, headers={"User-Agent": "co-president/1.0"}
        ) as resp,
        tmp.open("wb") as fh,
    ):
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
            fh.write(chunk)
    actual_size = tmp.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        size_msg = f"Size mismatch for {dest.name}: expected {expected_size}, got {actual_size}"
        tmp.unlink(missing_ok=True)
        raise RuntimeError(size_msg)
    tmp.replace(dest)
    return dest


def download_cedae_elections(
    dest_dir: Path | None = None,
    years: Sequence[int] = _DEFAULT_YEARS,
    niveles: Sequence[str] = _DEFAULT_NIVELES,
    *,
    skip_existing: bool = True,
) -> list[Path]:
    """Download CEDAE election-result CSVs matching the given filters.

    Fetches the index page once, selects matching entries, and downloads
    each CSV to ``dest_dir``. Idempotent when ``skip_existing=True`` (a
    file whose local size matches the S3-reported size is skipped).

    Args:
        dest_dir: Target directory. If ``None``, resolves via
            ``resolve_data_dir``, writing to ``data/raw/cedae/``.
        years: Election years to download.
        niveles: ``nivel`` values to include.
        skip_existing: If ``True``, skip files that already exist with the
            correct size.

    Returns:
        List of ``Path`` objects for successfully downloaded files.

    """
    if dest_dir is None:
        dest_dir = resolve_data_dir(None) / "raw" / "cedae"
    dest_dir.mkdir(parents=True, exist_ok=True)
    entries = fetch_cedae_index()
    selected = select_cedae_files(entries, years=years, niveles=niveles)

    downloaded: list[Path] = []
    for entry in selected:
        url = str(entry["path"])
        name = Path(url).name
        dest = dest_dir / name
        size = int(entry["size"])  # type: ignore[arg-type]
        size_mb = size / (1024 * 1024)
        if skip_existing and dest.exists() and dest.stat().st_size == size:
            logger.info("Skip %s (exists, %d MiB)", name, int(size_mb))
            downloaded.append(dest)
            continue
        logger.info("Downloading %s (%6.2f MiB) ...", name, size_mb)
        _download_one(url, dest, size)
        downloaded.append(dest)
    return downloaded


def compress_csvs(source_dir: Path | None = None, dest_dir: Path | None = None) -> list[Path]:
    """Gzip-compress all CSV files in ``source_dir``.

    Writes ``<filename>.csv.gz`` files and removes the original ``.csv``.

    Args:
        source_dir: Directory containing uncompressed CSVs. If ``None``,
            defaults to ``resolve_data_dir(None) / "raw" / "cedae"``.
        dest_dir: Directory to write compressed files. If ``None``, same
            as ``source_dir``.

    Returns:
        List of ``Path`` objects for the compressed ``.csv.gz`` files.

    """
    if source_dir is None:
        source_dir = resolve_data_dir(None) / "raw" / "cedae"
    if dest_dir is None:
        dest_dir = source_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    compressed: list[Path] = []
    for csv_path in sorted(source_dir.glob("*.csv")):
        gz_path = (dest_dir / csv_path.name).with_suffix(csv_path.suffix + ".gz")
        logger.info("Compressing %s -> %s", csv_path.name, gz_path.name)
        with csv_path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
            while True:
                chunk = src.read(_CHUNK_SIZE)
                if not chunk:
                    break
                dst.write(chunk)
        csv_path.unlink()
        compressed.append(gz_path)
    return compressed
