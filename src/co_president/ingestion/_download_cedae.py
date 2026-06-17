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

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "compress_csvs",
    "download_cedae_elections",
    "fetch_cedae_index",
    "fetch_local_cedae_results",
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


_CEDAE_CANDIDATE_MAP: dict[str, str] = {
    "GUSTAVO FRANCISCO PETRO URREGO": "gustavo_petro",
    "IVAN DUQUE MARQUEZ": "ivan_duque",
    "RODOLFO HERNANDEZ": "rodolfo_hernandez",
    "FEDERICO GUTIERREZ": "federico_gutierrez",
    "SERGIO FAJARDO": "sergio_fajardo",
    "INGRID BETANCOURT": "ingrid_betancourt",
    "JUAN MANUEL SANTOS": "juan_manuel_santos",
    "ALVARO URIBE VELEZ": "alvaro_uribe",
    "CARLOS GAVIRIA": "carlos_gaviria",
    "CLARA LOPEZ": "clara_lopez",
    "HORACIO SERPA": "horacio_serpa",
    "ANTANAS MOCKUS": "antanas_mockus",
    "NOEMI SANIN": "noemi_sanin",
    "NOEMI SANIN POSADA": "noemi_sanin",
    "ENRIQUE PENALOSA": "enrique_penalosa",
    "OSCAR IVAN ZULUAGA": "oscar_ivan_zuluaga",
    "LUIS EDUARDO GARZON": "luis_eduardo_garzon",
    "MARTA NOEMI DEL ESPIRITU SANTO SANIN POSADA": "noemi_sanin",
    "PROMOTORES VOTO EN BLANCO": "blanco",
}


def _cedae_candidate_key(row: dict[str, object]) -> str:
    """Derive a canonical candidate key from a CEDAE raw-data row.

    Concatenates ``nombres`` + ``primer_apellido`` (and optionally
    ``segundo_apellido``) and looks it up in
    ``_CEDAE_CANDIDATE_MAP``.  Falls back to ``"UNKNOWN"``
    when no match is found, distinguishable from any real canonical key.
    """
    nombres = str(row.get("nombres", "") or "").strip()
    primer = str(row.get("primer_apellido", "") or "").strip()
    segundo = str(row.get("segundo_apellido", "") or "").strip()

    full_name = f"{nombres} {primer} {segundo}".strip()
    full_name = re.sub(r"\s+", " ", full_name).upper()

    if full_name in _CEDAE_CANDIDATE_MAP:
        return _CEDAE_CANDIDATE_MAP[full_name]

    name_2 = f"{nombres} {primer}".strip().upper()
    if name_2 in _CEDAE_CANDIDATE_MAP:
        return _CEDAE_CANDIDATE_MAP[name_2]

    if full_name:
        logger.debug("Unmapped CEDAE candidate: %s", full_name)

    return f"UNKNOWN_{full_name}" if full_name else "UNKNOWN"


def _log_unknown_candidates(records: pd.DataFrame, filename: str) -> None:
    """Log a warning if any rows have unmapped (``UNKNOWN_*``) candidate keys."""
    unknown = records[records["candidate"].str.startswith("UNKNOWN", na=False)]
    if unknown.empty:
        return
    unknown_keys = unknown["candidate"].unique().tolist()
    logger.warning(
        "CEDAE %s: %d rows have unmapped candidates (%s)",
        filename,
        len(unknown),
        unknown_keys,
    )


_ROUND_FILE_PATTERNS: dict[int, list[str]] = {
    1: ["_presidencia_primera_vuelta.dta.csv.gz", "_presidencia.dta.csv.gz"],
    2: ["_presidencia_segunda_vuelta.dta.csv.gz"],
}


def _validate_round_num(round_num: int) -> None:
    """Raise ``ValueError`` if *round_num* is not a known election round."""
    if round_num not in _ROUND_FILE_PATTERNS:
        keys = list(_ROUND_FILE_PATTERNS.keys())
        msg = f"Invalid round_num={round_num!r}. Expected one of {keys}."
        raise ValueError(msg)


def _find_cedae_file(data_dir: Path, year: int, round_num: int) -> Path | None:
    """Locate a CEDAE CSV file for *year*/*round_num* under *data_dir*."""
    cedae_dir = data_dir / "raw" / "cedae"
    if not cedae_dir.is_dir():
        return None
    for suffix in _ROUND_FILE_PATTERNS[round_num]:
        candidate = cedae_dir / f"{year}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def fetch_local_cedae_results(
    year: int,
    round_num: int,
    data_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Read presidential election results from a local CEDAE CSV file.

    Tries to find a matching file in ``data/raw/cedae/`` for the given
    year and round, maps its columns to the canonical schema expected
    by downstream pipeline functions, and returns a DataFrame.

    Args:
        year: Election year (2002, 2006, …, 2022).
        round_num: Election round (1 or 2).
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns ``codigo_municipio``, ``year``, ``round``,
        ``candidate``, ``votes``, ``total_votes``, ``registered_voters``,
        or ``None`` if no matching file is found.

    Raises:
        ValueError: If *round_num* is not 1 or 2.

    """
    _validate_round_num(round_num)
    if data_dir is None:
        data_dir = resolve_data_dir(None)

    candidate_path = _find_cedae_file(data_dir, year, round_num)
    if candidate_path is None:
        return None

    try:
        raw = pd.read_csv(
            candidate_path,
            compression="gzip",
            encoding="latin-1",
            dtype={"codmpio": str, "coddpto": str},
        )
    except (pd.errors.ParserError, OSError, ValueError) as exc:
        logger.warning("Failed to read CEDAE file %s: %s", candidate_path, exc)
        return None

    required = {"codmpio", "ano", "votos"}
    missing = required - set(raw.columns)
    if missing:
        logger.warning(
            "Local CEDAE file %s is missing columns %s; falling back to remote source",
            candidate_path.name,
            missing,
        )
        return None

    records: list[dict[str, object]] = []
    grouped = raw.groupby(["codmpio", "ano"], sort=False)

    for (codmpio, ano), group in grouped:
        muni_code = str(codmpio).zfill(5)
        try:
            ano_val: int | float | str = int(float(ano))  # type: ignore[arg-type]
            records.extend(
                _aggregate_cedae_group(
                    group,
                    round_num,
                    ano_val,
                    muni_code,
                    candidate_path.name,
                )
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Failed to process group %s/%s: %s", codmpio, ano, exc)
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        _log_unknown_candidates(df, candidate_path.name)
    n_munis = df["codigo_municipio"].nunique() if not df.empty else 0
    logger.info(
        "Loaded local CEDAE %s (%d rows, %d municipalities)",
        candidate_path.name,
        len(df),
        n_munis,
    )
    return df


def _aggregate_cedae_group(
    group: pd.DataFrame,
    round_num: int,
    ano_key: float | str,
    muni_code: str,
    cedae_filename: str,
) -> list[dict[str, object]]:
    """Aggregate votes per candidate within a municipality-year group."""
    ano_int = int(float(ano_key))
    total = int(pd.to_numeric(group["votos"], errors="coerce").fillna(0).sum())

    labeled = group.copy()
    labeled["_candidate"] = [
        _cedae_candidate_key({str(k): v for k, v in row.items()}) for _, row in labeled.iterrows()
    ]
    votes_by_candidate = (
        pd.to_numeric(labeled["votos"], errors="coerce")
        .fillna(0)
        .astype(int)
        .groupby(labeled["_candidate"])
        .sum()
    )

    reg_voters: int | None = None
    for col in ("potencial", "censo", "inscritos", "mesas_potencial_sufragantes"):
        if col in group.columns:
            vals = pd.to_numeric(group[col], errors="coerce").fillna(0)
            reg_voters = int(vals.sum())
            break
    if reg_voters is None:
        logger.debug(
            "No registered-voters column found for %s; abstention rate will be unknown",
            cedae_filename,
        )

    result: list[dict[str, object]] = []
    for candidate, votes in votes_by_candidate.items():
        result.append(
            {
                "codigo_municipio": muni_code,
                "year": ano_int,
                "round": round_num,
                "candidate": candidate,
                "votes": int(votes),
                "total_votes": total,
                "registered_voters": reg_voters,
            }
        )
    return result


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
