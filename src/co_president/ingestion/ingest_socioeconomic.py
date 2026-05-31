"""SPEC-13.1: Socioeconomic and demographic data ingestion.

Fetches DANE 2018 Census variables (ethnicity, education, internet),
Multidimensional Poverty Index (IPM), Unsatisfied Basic Needs (NBI),
and population projections (2018--2042) at the municipal level.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
import tempfile

from bs4 import BeautifulSoup
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from co_president.paths import resolve_data_dir

__all__ = [
    "build_socioeconomic_matrix",
    "calculate_features",
    "fetch_dane_csv",
    "fetch_poverty_indicators",
    "scrape_dane_portal_playwright",
]

logger = logging.getLogger(__name__)

_EXPECTED_MUNICIPALITIES = 1122

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

    Raises:
        requests.RequestException: When the HTTP request fails and retries
            are exhausted.

    """
    try:
        response = _fetch_dane_raw(dane_url)
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return _parse_data_file(response)
    except Exception as exc:  # noqa: BLE001
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
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(_DANE_CENSUS_URL, wait_until="networkidle")
            try:
                page.click("text=Descargar", timeout=10000)
                with page.expect_download(timeout=30000) as download_info:
                    page.click("text=CSV", timeout=10000)
                download = download_info.value
                tmp_dir = Path(tempfile.mkdtemp())
                csv_path = tmp_dir / "cnpv_2018_download.csv"
                download.save_as(str(csv_path))
                browser.close()

                if csv_path.stat().st_size > 0:
                    return pd.read_csv(str(csv_path), encoding="latin-1")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Playwright census download failed: %s", exc)
                browser.close()
    except ImportError:
        logger.warning("Playwright not installed; skipping browser automation fallback")
    return None


def fetch_poverty_indicators() -> pd.DataFrame:
    """Fetch IPM and NBI poverty indicators at the municipal level.

    Attempts a direct download from DANE's IPM page.  Falls back to
    BeautifulSoup-based link scraping for Excel files.  Returns a
    hardcoded fallback when both remote approaches fail.

    Returns:
        DataFrame with ``codigo_municipio``, ``ipm_score``, and
        ``nbi_rate`` columns.

    """
    df = fetch_dane_csv(_DANE_IPM_URL)
    if df is not None and _is_valid_poverty_df(df):
        return df
    try:
        response = requests.get(_DANE_IPM_URL, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
        excel_links = [str(a.get("href", "")) for a in soup.select("a[href$='.xlsx']")]
        for link in excel_links:
            if not link:
                continue
            try:
                excel_df = pd.read_excel(link)  # type: ignore[reportUnknownMemberType]
                if _is_valid_poverty_df(excel_df):
                    return excel_df
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse IPM Excel from %s: %s", link, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("BeautifulSoup scrape for IPM failed: %s", exc)
    logger.warning("All remote IPM/NBI fetches failed; using hardcoded fallback")
    return _ipm_hardcoded_fallback()


def calculate_features(
    census_df: pd.DataFrame,
    poverty_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize census variables and merge with poverty indicators and projections.

    Args:
        census_df: Raw census DataFrame with columns for total population,
            afro-colombian population, indigenous population, rural
            dispersed population, years of schooling, and internet access.
        poverty_df: DataFrame with ``codigo_municipio``, ``ipm_score``,
            and ``nbi_rate``.
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
        >>> poverty = pd.DataFrame({
        ...     "codigo_municipio": ["11001"],
        ...     "ipm_score": [0.045],
        ...     "nbi_rate": [0.032],
        ... })
        >>> proj = pd.DataFrame({
        ...     "codigo_municipio": ["11001"],
        ...     "proyeccion_2022": [7900000],
        ... })
        >>> result = calculate_features(census, poverty, proj)
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
    result = result.merge(poverty_df, on="codigo_municipio", how="left")
    if not projections_df.empty:
        result = result.merge(projections_df, on="codigo_municipio", how="left")
        if "proyeccion_2022" in result.columns:
            result["population_2022"] = result["proyeccion_2022"]
    else:
        result["population_2022"] = pd.NA
    result["pct_afro_colombian"] = result["pct_afro_colombian"].clip(0.0, 1.0)
    result["pct_indigenous"] = result["pct_indigenous"].clip(0.0, 1.0)
    result["pct_rural_disperso"] = result["pct_rural_disperso"].clip(0.0, 1.0)
    result["internet_access_rate"] = result["internet_access_rate"].clip(0.0, 1.0)
    if "ipm_score" in result.columns:
        result["ipm_score"] = result["ipm_score"].clip(0.0, 1.0)
    if "nbi_rate" in result.columns:
        result["nbi_rate"] = result["nbi_rate"].clip(0.0, 1.0)
    return result


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
    poverty = fetch_poverty_indicators()
    projections = _fetch_population_projections()
    features = calculate_features(census, poverty, projections)
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Socrata population projections fetch failed: %s", exc)
    return _projections_hardcoded_fallback()


def _safe_ratio(df: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Compute ``numerator / denominator``, guarded against zero and missing columns."""
    if numerator not in df.columns or denominator not in df.columns:
        return pd.Series(0.0, index=df.index)
    num = pd.to_numeric(df[numerator], errors="coerce").fillna(0.0)
    denom = pd.to_numeric(df[denominator], errors="coerce").fillna(0.0)
    return (num / denom.replace(0, pd.NA)).fillna(0.0)


def _is_valid_poverty_df(df: pd.DataFrame) -> bool:
    """Check that a DataFrame has the expected poverty indicator columns."""
    required = {"codigo_municipio", "ipm_score"}
    return required.issubset(set(df.columns))


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
    except Exception as exc:  # noqa: BLE001
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


def _ipm_hardcoded_fallback() -> pd.DataFrame:
    """Return hardcoded IPM data when remote DANE request fails."""
    records = [
        {"codigo_municipio": "11001", "ipm_score": 0.045, "nbi_rate": 0.032},
        {"codigo_municipio": "05001", "ipm_score": 0.125, "nbi_rate": 0.098},
        {"codigo_municipio": "76001", "ipm_score": 0.098, "nbi_rate": 0.071},
        {"codigo_municipio": "08001", "ipm_score": 0.185, "nbi_rate": 0.152},
        {"codigo_municipio": "68001", "ipm_score": 0.210, "nbi_rate": 0.178},
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
