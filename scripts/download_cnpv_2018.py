"""Download CNPV 2018 microdata from the DANE catalog.

The Colombian National Census of Population and Housing (CNPV) 2018
microdata is published by DANE as 33 departmental ZIP files, one per
department, totalling ~2.4 GB. This script downloads them into
``data/cnpv-2018/raw/`` and also fetches the PDF documentation and
DDI/XML metadata into ``data/cnpv-2018/docs/``.

Reference:
    https://microdatos.dane.gov.co/index.php/catalog/643/get-microdata

The catalog lists each file with its own numeric download id (the
``MANIFEST`` constant below pins these ids). ``parse_manifest`` is
provided so the manifest can be re-derived from the live HTML if DANE
reassigns ids.

Usage:
    python scripts/download_cnpv_2018.py
    python scripts/download_cnpv_2018.py --workers 8
    python scripts/download_cnpv_2018.py --refresh-manifest
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sys

import requests
from requests.adapters import HTTPAdapter
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from urllib3.util.retry import Retry

__all__ = [
    "CATALOG_URL",
    "DDI_XML_URL",
    "DOC_FILES",
    "MANIFEST",
    "PDF_DOC_URL",
    "CnpvFile",
    "DownloadError",
    "build_url",
    "download_file",
    "fetch_cnpv_docs",
    "fetch_cnpv_manifest_from_catalog",
    "parse_manifest",
]

logger = logging.getLogger("cnpv_downloader")

CATALOG_URL = "https://microdatos.dane.gov.co/index.php/catalog/643/get-microdata"
CATALOG_ID = 643
BASE_URL = f"https://microdatos.dane.gov.co/index.php/catalog/{CATALOG_ID}/download"
DDI_XML_URL = "https://microdatos.dane.gov.co/index.php/metadata/export/643/ddi"
PDF_DOC_URL = "https://microdatos.dane.gov.co/index.php/catalog/643/pdf-documentation"
USER_AGENT = "co-president-2026/0.1 (+https://github.com/SamuelGomezG/co-president-2026)"

_CHUNK_SIZE: int = 1 << 20  # 1 MiB
_DEFAULT_WORKERS: int = 4
_HTTP_ERROR_THRESHOLD: int = 400
_HTTP_SERVER_RETRIES: tuple[int, ...] = (500, 502, 503, 504)

DOC_FILES: tuple[tuple[str, str], ...] = (
    (PDF_DOC_URL, "CNPV-2018-documentation.pdf"),
    (DDI_XML_URL, "CNPV-2018-data-dictionary.ddi.xml"),
)


class DownloadError(RuntimeError):
    """Raised when a download fails after all retries or returns a bad status."""


@dataclass(frozen=True)
class CnpvFile:
    """A single CNPV 2018 departmental microdata file.

    Attributes:
        file_id: DANE catalog numeric id (12143-12175).
        name: Filename as published, e.g. ``05_Antioquia.zip``.

    """

    file_id: int
    name: str

    @property
    def url(self) -> str:
        """Return the HTTPS download URL for this file."""
        return build_url(self.file_id)


def build_url(file_id: int) -> str:
    """Build the catalog download URL for a given file id.

    Args:
        file_id: Numeric download id assigned by the DANE catalog.

    Returns:
        The full HTTPS URL pointing at the file's download endpoint.

    """
    return f"{BASE_URL}/{file_id}"


_MANIFEST_RE = re.compile(
    r'title="(\d+_[A-Za-z]+\.zip)"\s*[^>]*?catalog/643/download/(\d+)',
    re.DOTALL,
)


def parse_manifest(html: str) -> list[CnpvFile]:
    """Extract ``(file_id, name)`` entries from the catalog HTML page.

    Each file appears twice on the page (in the table and in a script
    handler), so duplicates are removed while preserving first-seen order.

    Args:
        html: Raw HTML content of the catalog page.

    Returns:
        Deduplicated list of ``CnpvFile`` entries in catalog order.

    """
    seen: set[int] = set()
    result: list[CnpvFile] = []
    for name, file_id_str in _MANIFEST_RE.findall(html):
        file_id = int(file_id_str)
        if file_id in seen:
            continue
        seen.add(file_id)
        result.append(CnpvFile(file_id=file_id, name=name))
    return result


# Hardcoded manifest of the 33 CNPV 2018 departmental files. Verified
# against the DANE catalog page on 2026-06-02; ids 12143-12175 map to
# the 33 Colombian departments in DIVIPOLA order.
MANIFEST: tuple[CnpvFile, ...] = (
    CnpvFile(12143, "05_Antioquia.zip"),
    CnpvFile(12144, "08_Atlantico.zip"),
    CnpvFile(12145, "11_Bogota.zip"),
    CnpvFile(12146, "13_Bolivar.zip"),
    CnpvFile(12147, "15_Boyaca.zip"),
    CnpvFile(12148, "17_Caldas.zip"),
    CnpvFile(12149, "18_Caqueta.zip"),
    CnpvFile(12150, "19_Cauca.zip"),
    CnpvFile(12151, "20_Cesar.zip"),
    CnpvFile(12152, "23_Cordoba.zip"),
    CnpvFile(12153, "25_Cundinamarca.zip"),
    CnpvFile(12154, "27_Choco.zip"),
    CnpvFile(12155, "41_Huila.zip"),
    CnpvFile(12156, "44_LaGuajira.zip"),
    CnpvFile(12157, "47_Magdalena.zip"),
    CnpvFile(12158, "50_Meta.zip"),
    CnpvFile(12159, "52_Narino.zip"),
    CnpvFile(12160, "54_NorteDeSantander.zip"),
    CnpvFile(12161, "63_Quindio.zip"),
    CnpvFile(12162, "66_Risaralda.zip"),
    CnpvFile(12163, "68_Santander.zip"),
    CnpvFile(12164, "70_Sucre.zip"),
    CnpvFile(12165, "73_Tolima.zip"),
    CnpvFile(12166, "76_ValleDelCauca.zip"),
    CnpvFile(12167, "81_Arauca.zip"),
    CnpvFile(12168, "85_Casanare.zip"),
    CnpvFile(12169, "86_Putumayo.zip"),
    CnpvFile(12170, "88_SanAndresProvidenciaYSantaCatalina.zip"),
    CnpvFile(12171, "91_Amazonas.zip"),
    CnpvFile(12172, "94_Guainia.zip"),
    CnpvFile(12173, "95_Guaviare.zip"),
    CnpvFile(12174, "97_Vaupes.zip"),
    CnpvFile(12175, "99_Vichada.zip"),
)


def _build_session() -> requests.Session:
    """Build a ``requests.Session`` with a UA header and HTTP-level retries."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry_cfg = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=_HTTP_SERVER_RETRIES,
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_TRANSIENT_EXC: tuple[type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_TRANSIENT_EXC),
)
def _head_size(session: requests.Session, url: str, *, timeout: float) -> int | None:
    """Return ``Content-Length`` for ``url`` or ``None`` if unavailable."""
    response = session.head(url, timeout=timeout, allow_redirects=True)
    response.close()
    length = response.headers.get("Content-Length")
    return int(length) if length is not None else None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_TRANSIENT_EXC),
)
def _download_one(
    session: requests.Session,
    url: str,
    dest: Path,
    expected_size: int | None,
    *,
    timeout: float,
) -> Path:
    """Download ``url`` to ``dest`` using a ``.part`` staging file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with (
        session.get(url, stream=True, timeout=timeout) as resp,
        tmp.open("wb") as fh,
    ):
        if resp.status_code >= _HTTP_ERROR_THRESHOLD:
            tmp.unlink(missing_ok=True)
            msg = f"HTTP {resp.status_code} for {url}"
            raise DownloadError(msg)
        for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
            if chunk:
                fh.write(chunk)
    actual = tmp.stat().st_size
    if expected_size is not None and actual != expected_size:
        tmp.unlink(missing_ok=True)
        msg = f"Size mismatch for {dest.name}: expected {expected_size}, got {actual}"
        raise DownloadError(msg)
    tmp.replace(dest)
    return dest


def download_file(
    session: requests.Session,
    file: CnpvFile,
    dest_dir: Path,
    *,
    skip_existing: bool = True,
    timeout: float = 120.0,
) -> Path:
    """Download a single ``CnpvFile`` into ``dest_dir`` and return its local path.

    Args:
        session: A ``requests.Session`` to use for the HTTP traffic.
        file: The CNPV file to download.
        dest_dir: Directory to write the file into (created if missing).
        skip_existing: If ``True``, do a HEAD request and skip the download
            when the local file already has the expected size.
        timeout: Per-request timeout in seconds.

    Returns:
        The local path of the downloaded (or pre-existing) file.

    Raises:
        DownloadError: If the HTTP response is 4xx/5xx or the size check
            fails after the body has been fully received.
        requests.RequestException: If a transient error persists after
            all ``tenacity`` retries.

    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.name
    expected_size: int | None = None

    if skip_existing and dest.exists():
        try:
            expected_size = _head_size(session, file.url, timeout=timeout)
        except (DownloadError, requests.RequestException) as exc:
            logger.warning("HEAD failed for %s: %s; will redownload", file.name, exc)
            expected_size = None
        if expected_size is not None and dest.stat().st_size == expected_size:
            logger.info("Skip %s (exists, %.1f MiB)", file.name, expected_size / (1 << 20))
            return dest

    logger.info("Downloading %s ...", file.name)
    try:
        return _download_one(session, file.url, dest, expected_size, timeout=timeout)
    except RetryError as exc:
        msg = f"Download failed for {file.name} after retries: {exc}"
        raise DownloadError(msg) from exc


def fetch_cnpv_manifest_from_catalog(
    session: requests.Session | None = None, *, timeout: float = 60.0
) -> list[CnpvFile]:
    """Fetch the live catalog page and return the parsed manifest.

    Args:
        session: Optional ``requests.Session``; one is built if omitted.
        timeout: Request timeout in seconds.

    Returns:
        List of ``CnpvFile`` entries as currently published on the
        DANE catalog page.

    """
    sess = session or _build_session()
    response = sess.get(CATALOG_URL, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    try:
        return parse_manifest(response.text)
    finally:
        response.close()


def fetch_cnpv_docs(dest_dir: Path, *, session: requests.Session | None = None) -> list[Path]:
    """Download the PDF documentation and DDI/XML metadata into ``dest_dir``.

    Args:
        dest_dir: Directory to write the documentation files into.
        session: Optional ``requests.Session``; one is built if omitted.

    Returns:
        List of paths successfully written or skipped.

    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _build_session()
    out: list[Path] = []
    for url, name in DOC_FILES:
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("Skip %s (exists)", name)
            out.append(dest)
            continue
        logger.info("Downloading %s ...", name)
        _download_one(sess, url, dest, expected_size=None, timeout=120.0)
        out.append(dest)
    return out


def _download_one_worker(
    file: CnpvFile, dest_dir: Path, *, session: requests.Session, skip_existing: bool
) -> tuple[CnpvFile, Path | None, BaseException | None]:
    """Worker used by ``ThreadPoolExecutor`` to download a single file.

    Returns a ``(file, path, error)`` triple so failures in one file do
    not cancel parallel siblings.

    """
    try:
        path = download_file(session, file, dest_dir, skip_existing=skip_existing)
    except (DownloadError, requests.RequestException) as exc:
        return (file, None, exc)
    else:
        return (file, path, None)


def _verify_manifest(session: requests.Session) -> int:
    """Refetch the catalog and verify MANIFEST matches the live list.

    Returns 0 if the manifest is up to date, 1 otherwise.

    """
    try:
        live = fetch_cnpv_manifest_from_catalog(session)
    except requests.RequestException:
        logger.exception("Failed to fetch catalog page")
        return 1
    if list(MANIFEST) == live:
        logger.info("Manifest verified: %d entries match the live catalog.", len(MANIFEST))
        return 0
    names_live = {e.name for e in live}
    names_local = {e.name for e in MANIFEST}
    missing = sorted(names_live - names_local)
    extra = sorted(names_local - names_live)
    logger.error(
        "MANIFEST mismatch with live catalog. Missing locally: %s; extra locally: %s",
        missing,
        extra,
    )
    return 1


def _download_all_data(
    session: requests.Session, raw_dir: Path, *, workers: int, force: bool
) -> int:
    """Download every entry in ``MANIFEST`` into ``raw_dir`` in parallel.

    Returns 0 on success, 1 if any file failed to download.

    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    skip_existing = not force
    total_bytes = 0
    failed: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_one_worker,
                entry,
                raw_dir,
                session=session,
                skip_existing=skip_existing,
            ): entry
            for entry in MANIFEST
        }
        for fut in as_completed(futures):
            entry = futures[fut]
            _file, path, err = fut.result()
            if err is not None:
                failed.append((entry.name, err))
                continue
            if path is not None:
                total_bytes += path.stat().st_size
    if failed:
        for name, err in failed:
            logger.error("FAILED %s: %s", name, err)
        return 1
    logger.info(
        "Data files: %d files, %.2f GiB on disk under %s.",
        len(MANIFEST),
        total_bytes / (1 << 30),
        raw_dir,
    )
    return 0


def _download_docs(docs_dir: Path, *, session: requests.Session) -> int:
    """Fetch the PDF and DDI/XML docs into ``docs_dir``.

    Returns 0 on success, 1 on failure.

    """
    try:
        fetch_cnpv_docs(docs_dir, session=session)
    except (DownloadError, requests.RequestException):
        logger.exception("FAILED documentation")
        return 1
    logger.info("Documentation downloaded to %s.", docs_dir)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download CNPV 2018 microdata from the DANE catalog.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "cnpv-2018",
        help="Root output directory (default: data/cnpv-2018).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=f"Parallel download workers (default: {_DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Skip the PDF and DDI/XML documentation downloads.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help=(
            "Refetch the catalog page and verify the hardcoded MANIFEST matches "
            "the live list. Exits non-zero on mismatch."
        ),
    )
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Download only the PDF and DDI/XML documentation (skip data files).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the downloader from the command line.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 if any download failed or the
        manifest check failed.

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    raw_dir = args.output_dir / "raw"
    docs_dir = args.output_dir / "docs"
    session = _build_session()

    if args.refresh_manifest:
        return _verify_manifest(session)

    rc = 0
    if not args.docs_only:
        rc |= _download_all_data(session, raw_dir, workers=args.workers, force=args.force)
    if not args.no_docs:
        rc |= _download_docs(docs_dir, session=session)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
