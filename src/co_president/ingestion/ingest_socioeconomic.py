"""SPEC-13a: DANE socioeconomic and demographic data ingestion.

Fetches DANE 2018 Census variables (ethnicity, education, internet),
Multidimensional Poverty Index (IPM), Unsatisfied Basic Needs (NBI),
and population projections (2018--2042) at the municipal level.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
import tempfile

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from co_president.config import EXPECTED_MUNICIPALITIES
from co_president.paths import resolve_data_dir

__all__ = [
    "build_socioeconomic_matrix",
    "calculate_features",
    "fetch_dane_csv",
    "scrape_dane_portal_playwright",
    "validate_socioeconomic",
]

logger = logging.getLogger(__name__)

# Imported from co_president.config: EXPECTED_MUNICIPALITIES

_DANE_CENSUS_URL = "https://microdatos.dane.gov.co/index.php/catalog/643"
_DANE_IPM_URL = (
    "https://www.dane.gov.co/index.php/estadisticas-por-tema/"
    "pobreza-y-condiciones-de-vida/"
    "medida-de-pobreza-multidimensional-de-fuente-censal"
)
_DANE_PROJECTIONS_URL = (
    "https://www.dane.gov.co/index.php/estadisticas-por-tema/"
    "demografia-y-poblacion/proyecciones-de-poblacion"
)

_SOCRATA_POPULATION_DATASET = "39be-6wxf"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def _fetch_dane_raw(dane_url: str) -> requests.Response:
    """Fetch DANE URL with retries — raises on network/HTTP errors."""
    response = requests.get(dane_url, timeout=30)
    response.raise_for_status()
    return response


def fetch_dane_csv(dane_url: str) -> pd.DataFrame | None:
    """Attempt to fetch a DANE CSV or Excel file from a direct URL.

    DANE sometimes provides direct download links to CSV or Excel files.
    Returns ``None`` when the response is HTML (no direct file link) or
    when the fetch fails after all retry attempts.

    Args:
        dane_url: The DANE page URL to attempt fetching from.

    Returns:
        DataFrame parsed from the response, or ``None`` if no downloadable
        file was found.

    Note:
        This function never raises.  All network errors are caught and
        logged internally; the caller should check for ``None`` to
        detect failure rather than expecting an exception.

    """
    try:
        response = _fetch_dane_raw(dane_url)
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return _parse_data_file(response)
    except (requests.RequestException, pd.errors.ParserError) as exc:
        logger.warning("Direct DANE CSV fetch failed for %s: %s", dane_url, exc)
    return None


def scrape_dane_portal_playwright() -> pd.DataFrame | None:
    """Fallback: navigate the DANE portal via Playwright to download census data.

    The DANE microdata portal is JavaScript-heavy.  Playwright handles
    dynamic page rendering, file download dialogs, and authentication
    redirects.

    Returns:
        DataFrame with census microdata, or ``None`` if the browser
        automation failed.

    """
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # noqa: PLC0415
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(_DANE_CENSUS_URL, wait_until="networkidle")
                try:
                    page.click("text=Descargar", timeout=10000)
                    with page.expect_download(timeout=30000) as download_info:
                        page.click("text=CSV", timeout=10000)
                    download = download_info.value
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        csv_path = Path(tmp_dir) / "cnpv_2018_download.csv"
                        download.save_as(str(csv_path))

                        if csv_path.stat().st_size > 0:
                            return pd.read_csv(str(csv_path), encoding="latin-1")
                except (PlaywrightTimeoutError, AttributeError) as exc:
                    logger.warning("Playwright census download failed: %s", exc)
            finally:
                browser.close()
    except ImportError:
        logger.warning("Playwright not installed; skipping browser automation fallback")
    except RuntimeError:
        logger.warning("Playwright browser launch failed; skipping browser automation")
    return None


def calculate_features(
    census_df: pd.DataFrame,
    poverty_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize census variables and optionally merge poverty indicators.

    Poverty columns (NBI, IPM) are now handled by dedicated
    ``ingest_nbi`` / ``ingest_ipm`` pipelines.  The *poverty_df*
    argument is retained for backward compatibility but is typically
    an empty DataFrame.

    Args:
        census_df: Raw census DataFrame with columns for total population,
            afro-colombian population, indigenous population, rural
            dispersed population, years of schooling, and internet access.
        poverty_df: DataFrame with ``codigo_municipio`` (may be empty).
        projections_df: DataFrame with ``codigo_municipio`` and
            ``proyeccion_2022`` (projected 2022 population).

    Returns:
        DataFrame with normalized socioeconomic features, one row per
        municipality.

    Raises:
        KeyError: If ``codigo_municipio`` is missing from any input DataFrame.
        TypeError: If input contains non-numeric data in columns expected
            to be numeric.

    Examples:
        >>> census = pd.DataFrame({
        ...     "codigo_municipio": ["11001"],
        ...     "poblacion_total": [1000],
        ...     "poblacion_afrocolombiana": [100],
        ...     "poblacion_indigena": [0],
        ...     "poblacion_rural_dispersa": [0],
        ...     "anos_escolaridad": [10],
        ...     "hogares_con_internet": [500],
        ...     "hogares_totales": [600],
        ... })
        >>> proj = pd.DataFrame({
        ...     "codigo_municipio": ["11001"],
        ...     "proyeccion_2022": [7900000],
        ... })
        >>> result = calculate_features(census, pd.DataFrame(), proj)
        >>> "pct_afro_colombian" in result.columns
        True

    """
    result = census_df.copy()
    result["pct_afro_colombian"] = _safe_ratio(
        result, "poblacion_afrocolombiana", "poblacion_total"
    )
    result["pct_indigenous"] = _safe_ratio(result, "poblacion_indigena", "poblacion_total")
    result["pct_rural_disperso"] = _safe_ratio(
        result, "poblacion_rural_dispersa", "poblacion_total"
    )
    if "anos_escolaridad" in result.columns:
        result["years_schooling"] = pd.to_numeric(  # type: ignore[reportCallIssue]
            result["anos_escolaridad"], errors="coerce"
        )
    else:
        result["years_schooling"] = pd.NA
    result["internet_access_rate"] = _safe_ratio(result, "hogares_con_internet", "hogares_totales")
    if not poverty_df.empty and "codigo_municipio" in poverty_df.columns:
        result = result.merge(poverty_df, on="codigo_municipio", how="left")
    if not projections_df.empty:
        result = result.merge(projections_df, on="codigo_municipio", how="left")
        if "proyeccion_2022" in result.columns:
            raw_pop = pd.to_numeric(result["proyeccion_2022"], errors="coerce")
            coerced_count = int(raw_pop.isna().sum() - result["proyeccion_2022"].isna().sum())
            if coerced_count > 0:
                logger.warning(
                    "%d %s coerced to NaN in proyeccion_2022",
                    coerced_count,
                    "value was" if coerced_count == 1 else "values were",
                )
            result["population_2022"] = raw_pop
    else:
        result["population_2022"] = pd.NA
    result["pct_afro_colombian"] = result["pct_afro_colombian"].clip(0.0, 1.0)
    result["pct_indigenous"] = result["pct_indigenous"].clip(0.0, 1.0)
    result["pct_rural_disperso"] = result["pct_rural_disperso"].clip(0.0, 1.0)
    # Clipping of ipm_score, internet_access_rate, and nbi_rate is performed
    # in build_socioeconomic_matrix() after validate_socioeconomic() so that
    # range checks can fire on the raw values before clamping.
    return result


_CRITICAL_COLUMNS: list[str] = ["internet_access_rate", "population_2022"]

_VALIDATION_EXPECTED_COLUMNS: set[str] = {
    "codigo_municipio",
    "pct_afro_colombian",
    "pct_indigenous",
    "pct_rural_disperso",
    "years_schooling",
    "internet_access_rate",
    "population_2022",
}


def validate_socioeconomic(df: pd.DataFrame) -> list[str]:
    """Validate a socioeconomic feature DataFrame against acceptance criteria.

    Checks for missing columns, nulls in critical columns, out-of-range
    values, and insufficient municipality count.

    Args:
        df: The socioeconomic feature DataFrame to validate.

    Returns:
        List of warning messages (empty if all checks pass).

    Examples:
        >>> df = pd.DataFrame({
        ...     "codigo_municipio": ["11001"],
        ...     "pct_afro_colombian": [0.013],
        ...     "pct_indigenous": [0.002],
        ...     "pct_rural_disperso": [0.0],
        ...     "years_schooling": [11.5],
        ...     "internet_access_rate": [0.79],
        ...     "population_2022": [7_900_000],
        ... })
        >>> validate_socioeconomic(df)
        ['Expected 1122 municipalities, got 1']

    """
    warnings: list[str] = []

    missing_cols = _VALIDATION_EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        warnings.append(f"Missing columns: {sorted(missing_cols)}")
        return warnings

    null_counts = df[_CRITICAL_COLUMNS].isna().sum()
    for col in _CRITICAL_COLUMNS:
        count = int(null_counts[col])
        if count > 0:
            warnings.append(
                f"{col}: {count} null value{'s' if count > 1 else ''} (out of {len(df)} rows)"
            )

    internet = df["internet_access_rate"].dropna()
    if not internet.between(0.0, 1.0).all():
        warnings.append("internet_access_rate: found values outside [0.0, 1.0] range")

    pop = df["population_2022"].dropna()
    if not (pop > 0).all():
        warnings.append("population_2022: found zero or negative values")

    if len(df) < EXPECTED_MUNICIPALITIES:
        warnings.append(f"Expected {EXPECTED_MUNICIPALITIES} municipalities, got {len(df)}")

    return warnings


def build_socioeconomic_matrix(data_dir: Path | None = None) -> None:
    """Fetch DANE census, IPM/NBI, and population projections; normalise; save.

    Args:
        data_dir: Target data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        ``None``.  Results are written to ``data_dir/fundamentals/socioeconomic.csv``.

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)
    census = _fetch_census_fallback()
    poverty = pd.DataFrame()
    projections = _fetch_population_projections()
    features = calculate_features(census, poverty, projections)
    validation_warnings = validate_socioeconomic(features)
    # Clip after validation so range checks can fire on raw unclipped values.
    if "internet_access_rate" in features.columns:
        features["internet_access_rate"] = features["internet_access_rate"].clip(0.0, 1.0)
    for warning in validation_warnings:
        logger.warning("Socioeconomic validation: %s", warning)
    target_dir = data_dir / "fundamentals"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "socioeconomic.csv"
    features.to_csv(target_path, index=False)
    logger.info(
        "Socioeconomic features saved to %s (%d municipalities)",
        target_path,
        len(features),
    )


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _fetch_census_fallback() -> pd.DataFrame:
    """Fetch census data with fallback chain: direct CSV -> Playwright -> hardcoded."""
    df = fetch_dane_csv(_DANE_CENSUS_URL)
    if df is not None:
        return df
    df = scrape_dane_portal_playwright()
    if df is not None:
        return df
    logger.warning("All census fetch attempts failed; using hardcoded fallback")
    return _census_hardcoded_fallback()


def _fetch_population_projections() -> pd.DataFrame:
    """Fetch DANE population projections via Socrata API with hardcoded fallback."""
    try:
        from sodapy import Socrata  # type: ignore[reportMissingTypeStubs]  # noqa: PLC0415

        with Socrata("www.datos.gov.co", None) as client:
            results = client.get(  # type: ignore[reportUnknownMemberType]
                _SOCRATA_POPULATION_DATASET, limit=2000
            )
            df = pd.DataFrame.from_records(results)
            if "codigo_municipio" in df.columns and "proyeccion_2022" in df.columns:
                return df
    except (requests.RequestException, ImportError, ValueError) as exc:
        logger.warning("Socrata population projections fetch failed: %s", exc)
    return _projections_hardcoded_fallback()


def _safe_ratio(df: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Compute ``numerator / denominator``, guarded against zero and missing columns."""
    if numerator not in df.columns or denominator not in df.columns:
        return pd.Series(0.0, index=df.index)
    num = pd.to_numeric(df[numerator], errors="coerce").fillna(0.0)
    denom = pd.to_numeric(df[denominator], errors="coerce").fillna(0.0)
    return (num / denom.replace(0, pd.NA)).fillna(0.0)


def _parse_data_file(response: requests.Response) -> pd.DataFrame | None:
    """Attempt to parse a response body as CSV or Excel.

    Returns ``None`` when the content type is not parseable.
    """
    ct = response.headers.get("Content-Type", "")
    try:
        if "csv" in ct or "text/plain" in ct:
            return pd.read_csv(io.BytesIO(response.content), encoding="latin-1")
        if "excel" in ct or "spreadsheet" in ct:
            return pd.read_excel(io.BytesIO(response.content))  # type: ignore[reportUnknownMemberType]
        if "json" in ct:
            data = response.json()
            if isinstance(data, list):
                return pd.DataFrame.from_records(data)  # type: ignore[reportUnknownArgumentType]
            return pd.DataFrame([data])
    except (pd.errors.ParserError, ValueError) as exc:
        logger.warning("Failed to parse response as data file: %s", exc)
    return None


def _census_hardcoded_fallback() -> pd.DataFrame:
    """Return hardcoded census data for key municipalities when remote sources fail.

    Provides a small seed with the largest Colombian municipalities for
    pipeline continuity.
    """
    records = [
        {
            "codigo_municipio": "11001",
            "nombre_municipio": "Bogota D.C.",
            "poblacion_total": 7_181_469,
            "poblacion_afrocolombiana": 95_000,
            "poblacion_indigena": 14_000,
            "poblacion_rural_dispersa": 0,
            "anos_escolaridad": 11.5,
            "hogares_con_internet": 2_100_000,
            "hogares_totales": 2_650_000,
        },
        {
            "codigo_municipio": "05001",
            "nombre_municipio": "Medellin",
            "poblacion_total": 2_427_129,
            "poblacion_afrocolombiana": 260_000,
            "poblacion_indigena": 3_500,
            "poblacion_rural_dispersa": 30_000,
            "anos_escolaridad": 10.2,
            "hogares_con_internet": 680_000,
            "hogares_totales": 850_000,
        },
        {
            "codigo_municipio": "76001",
            "nombre_municipio": "Cali",
            "poblacion_total": 2_172_527,
            "poblacion_afrocolombiana": 580_000,
            "poblacion_indigena": 2_100,
            "poblacion_rural_dispersa": 2_000,
            "anos_escolaridad": 10.5,
            "hogares_con_internet": 580_000,
            "hogares_totales": 750_000,
        },
    ]
    return pd.DataFrame(records)


def _projections_hardcoded_fallback() -> pd.DataFrame:
    """Return hardcoded 2022 population projections when remote fetch fails."""
    records = [
        {"codigo_municipio": "11001", "proyeccion_2022": 7_900_000},
        {"codigo_municipio": "05001", "proyeccion_2022": 2_569_007},
        {"codigo_municipio": "76001", "proyeccion_2022": 2_300_000},
        {"codigo_municipio": "08001", "proyeccion_2022": 1_300_000},
        {"codigo_municipio": "68001", "proyeccion_2022": 1_200_000},
    ]
    return pd.DataFrame(records)
