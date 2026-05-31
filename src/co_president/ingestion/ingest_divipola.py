"""SPEC-12.1: DIVIPOLA master registry ingestion.

Creates a canonical registry of all 1 122 Colombian municipalities with
their 5-digit DIVIPOLA codes from the ``datos.gov.co`` Socrata API
(dataset ``mv2e-prx5``), with a GitHub Gist fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path
from sodapy import Socrata  # type: ignore[reportMissingTypeStubs]
from tenacity import retry, stop_after_attempt, wait_exponential

from co_president.paths import resolve_data_dir

__all__ = [
    "build_divipola_master",
    "fetch_divipola_github",
    "fetch_divipola_socrata",
    "validate_divipola",
]

logger = logging.getLogger(__name__)

_EXPECTED_MUNICIPALITIES = 1122
_SOCRATA_DATASET = "mv2e-prx5"
_GITHUB_GIST_ID = "b5848316671422b19e19bfca7f8aadcb"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def fetch_divipola_socrata() -> pd.DataFrame:
    """Fetch DIVIPOLA codes from ``datos.gov.co`` via the Socrata API.

    Returns:
        DataFrame with columns ``codigo_municipio``, ``nombre_municipio``,
        ``departamento``, ``latitud``, ``longitud``.

    Raises:
        ConnectionError: If the API is unreachable after three retries.
        ValueError: If the response cannot be parsed as JSON.

    """
    with Socrata("www.datos.gov.co", None) as client:
        results = client.get(_SOCRATA_DATASET, limit=2000)  # type: ignore[reportUnknownMemberType]
        df = pd.DataFrame.from_records(results)
    _rename_socrata_columns(df)
    return df


def fetch_divipola_github() -> pd.DataFrame:
    """Fallback: parse DIVIPOLA data from a GitHub Gist.

    Used when the Socrata API is unavailable.

    Returns:
        DataFrame with the same column schema as ``fetch_divipola_socrata``.

    """
    import requests  # noqa: PLC0415

    response = requests.get(f"https://api.github.com/gists/{_GITHUB_GIST_ID}", timeout=30)
    response.raise_for_status()
    data = response.json()
    for file_info in data.get("files", {}).values():
        raw_url = file_info.get("raw_url", "")
        if raw_url and raw_url.endswith(".csv"):
            df = pd.read_csv(raw_url)
            _rename_gist_columns(df)
            return df
    return _fallback_hardcoded()


def validate_divipola(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the DIVIPOLA DataFrame.

    Checks:
    - At least 1 122 municipalities
    - All ``codigo_municipio`` values are unique
    - No null ``codigo_municipio`` values

    Args:
        df: The DIVIPOLA DataFrame to validate.

    Returns:
        The same DataFrame (validated), for chaining.

    Raises:
        ValueError: If any validation check fails.

    """
    _raise_if_below_minimum(df)
    _raise_if_null_codes(df)
    _raise_if_duplicate_codes(df)
    return df


def build_divipola_master(data_dir: Path | None = None) -> None:
    """Fetch, validate, and save the DIVIPOLA master registry.

    Args:
        data_dir: Target data directory. If ``None``, resolves via
            ``resolve_data_dir``.

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)

    df = _fetch_with_fallback()
    df = validate_divipola(df)

    target_dir = data_dir / "fundamentals"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "divipola_master.csv"
    df.to_csv(target_path, index=False)
    logger.info("DIVIPOLA master saved to %s (%d municipalities)", target_path, len(df))


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _fetch_with_fallback() -> pd.DataFrame:
    """Attempt Socrata fetch; fall back to GitHub Gist on failure."""
    try:
        logger.info("Fetching DIVIPOLA from Socrata API ...")
        return fetch_divipola_socrata()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Socrata fetch failed (%s); trying GitHub Gist fallback", exc)
        return fetch_divipola_github()


def _raise_if_below_minimum(df: pd.DataFrame) -> None:
    """Raise if the DataFrame has fewer than the expected municipalities."""
    if len(df) < _EXPECTED_MUNICIPALITIES:
        msg = f"Expected >= {_EXPECTED_MUNICIPALITIES} municipalities, got {len(df)}"
        raise ValueError(msg)


def _raise_if_null_codes(df: pd.DataFrame) -> None:
    """Raise if any ``codigo_municipio`` value is null."""
    if not df["codigo_municipio"].notna().all():
        msg = "Null codes detected"
        raise ValueError(msg)


def _raise_if_duplicate_codes(df: pd.DataFrame) -> None:
    """Raise if ``codigo_municipio`` values are not unique."""
    if df["codigo_municipio"].nunique() != len(df):
        msg = "Duplicate codes detected"
        raise ValueError(msg)


def _rename_socrata_columns(df: pd.DataFrame) -> None:
    """Normalise Socrata API column names to canonical schema.

    The Socrata API may return snake_case or camelCase column names
    depending on the dataset version.
    """
    rename_map: dict[str, str] = {
        "codigo": "codigo_municipio",
        "municipio": "nombre_municipio",
        "depart": "departamento",
        "department": "departamento",
        "dpto": "departamento",
        "nombre_departamento": "departamento",
        "lat": "latitud",
        "lon": "longitud",
        "lng": "longitud",
    }
    # Only rename columns that actually exist and differ from target name
    existing = {k: v for k, v in rename_map.items() if k in df.columns and k != v}
    if existing:
        df.columns = [rename_map.get(c, c) if c in existing else c for c in df.columns]


def _rename_gist_columns(df: pd.DataFrame) -> None:
    """Normalise GitHub Gist column names to canonical schema."""
    rename_map: dict[str, str] = {
        "code": "codigo_municipio",
        "municipio": "nombre_municipio",
        "department": "departamento",
        "lat": "latitud",
        "lng": "longitud",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns and k != v}
    if existing:
        df.columns = [rename_map.get(c, c) if c in existing else c for c in df.columns]


def _fallback_hardcoded() -> pd.DataFrame:
    """Return a minimal hardcoded fallback when both remote sources fail.

    Provides a seed with key municipalities (Medellín and Bogotá D.C.) as a
    last-resort fallback. Validation will reject an empty or too-small
    registry, making failures visible.
    """
    records = [
        {
            "codigo_municipio": "05001",
            "nombre_municipio": "Medellin",
            "departamento": "Antioquia",
            "latitud": 6.2442,
            "longitud": -75.5812,
        },
        {
            "codigo_municipio": "11001",
            "nombre_municipio": "Bogota D.C.",
            "departamento": "Cundinamarca",
            "latitud": 4.7110,
            "longitud": -74.0721,
        },
    ]
    return pd.DataFrame(records)
