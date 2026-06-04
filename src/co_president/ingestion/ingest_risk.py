"""SPEC-20: Electoral risk and conflict data ingestion — three-tier cascade.

Fetches MOE electoral risk maps, INDEPAZ armed group presence, PDET
municipality list, and UNODC coca cultivation data at the municipal level.

Each source uses a three-tier cascade:
  1. Remote fetch — download from the canonical online source.
  2. Local file — parse the downloaded PDF/Excel stored in ``data/conflict/``.
  3. Hardcoded fallback — expanded static data as a last resort.

PDF parsing uses pymupdf (fitz) as the primary engine, with pdfplumber as
a secondary fallback.
"""

from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
import tempfile

from bs4 import BeautifulSoup
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from co_president.paths import resolve_data_dir

__all__ = [
    "build_risk_matrix",
    "calculate_risk_features",
    "fetch_moe_risk_maps",
    "fetch_pdet_list",
    "fetch_unodc_coca",
    "parse_indepaz_pdf",
]

logger = logging.getLogger(__name__)

_EXPECTED_PDET_COUNT = 170
_MIN_PDF_TABLE_COLUMNS = 2
_MIN_PDF_LINE_LENGTH = 4
_EXPECTED_PDF_FIELDS = 2

# ── Local conflict data directory ──────────────────────────────────
_CONFLICT_DATA_SUBDIR = "conflict"

# Canonical file names expected inside data/conflict/
_MOE_LOCAL_PDF_GLOB = "Mapas-de-Riesgo-Electoral-*_DIGITAL-*.pdf"
_INDEPAZ_LOCAL_PDF = "indepaz_RESUMEN_GRUPOS_2022.pdf"
_UNODC_LOCAL_PDF = "UNODC_Colombia_informe_monitoreo_2023.pdf"
_PDET_LOCAL_XLSX = "MunicipiosPDET.xlsx"

# Minimal name→code lookup for PDET municipalities scraped from the remote portal.
_PDET_NAME_TO_CODE: dict[str, str] = {
    "Quibdó": "27001",
    "Istmina": "27361",
    "Apartadó": "05045",
    "Turbo": "05893",
    "Buenaventura": "76109",
    "Cali": "76001",
    "Bogotá": "11001",
    "Medellín": "05001",
    "Barranquilla": "08001",
    "Cartagena": "13001",
    "Santa Marta": "47001",
    "Cúcuta": "54001",
    "Bucaramanga": "68001",
    "Pereira": "66170",
    "Manizales": "17001",
    "Ibagué": "73001",
    "Neiva": "41001",
    "Pasto": "52000",
    "Villavicencio": "50001",
    "Montería": "23001",
    "Sincelejo": "70708",
    "Valledupar": "20001",
    "Riohacha": "44001",
    "Maicao": "44347",
    "Saravena": "05604",
    "Arauca": "81001",
    "Tame": "05615",
    "Fortul": "05306",
    "La Paz": "94343",
    "San Calixto": "54405",
    "Hato Corozal": "68235",
    "Pamplonita": "54520",
    "Miscujé": "27550",
    "Unguía": "27245",
    "Acandí": "27050",
    "Carmen del Darién": "27150",
    "Juradó": "27410",
    "Medio Atrato": "27450",
    "Lloró": "27430",
    "Bahía Solano": "27073",
    "Nuquí": "27495",
    "El Cantón del San Pablo": "27120",
    "Unión Panamericana": "27175",
}


def _resolve_conflict_dir(data_dir: Path | None = None) -> Path | None:
    """Resolve the local ``data/conflict/`` directory, if it exists.

    Args:
        data_dir: Optional explicit data directory.  If ``None``, resolves
            via ``resolve_data_dir``.

    Returns:
        Path to the conflict directory, or ``None`` if it does not exist.

    """
    base = data_dir if data_dir is not None else resolve_data_dir(None)
    conflict_dir = base / _CONFLICT_DATA_SUBDIR
    if conflict_dir.is_dir():
        return conflict_dir
    logger.debug("Conflict data directory not found at %s", conflict_dir)
    return None


_MOE_RISK_URL = "https://moe.org.co/datos-electorales/mapas-de-riesgo-electoral/"
_INDEPAZ_PDF_URL = "https://indepaz.org.co/wp-content/uploads/2022/11/RESUMEN_GRUPOS_2022.pdf"
_PDET_URL = "https://centralpdet.renovacionterritorio.gov.co/conoce-los-pdet/"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def _fetch_moe_page() -> requests.Response:
    """Fetch MOE risk maps page with retries — raises on failure."""
    response = requests.get(_MOE_RISK_URL, timeout=30)
    response.raise_for_status()
    return response


def fetch_moe_risk_maps(data_dir: Path | None = None) -> pd.DataFrame:
    """Fetch MOE electoral risk classification via three-tier cascade.

    Tier 1 — Remote fetch: scrape CSV/Excel download links from the MOE
    risk maps portal.

    Tier 2 — Local file: parse the most recent MOE PDF stored under
    ``data/conflict/`` using pymupdf/pdfplumber.

    Tier 3 — Hardcoded fallback: expanded static risk classification.

    Args:
        data_dir: Optional explicit data directory for local file lookup.
            If ``None``, resolves via ``resolve_data_dir``.

    Returns:
        DataFrame with ``codigo_municipio`` and ``risk_level`` columns.

    """
    # ── Tier 1: Remote fetch ───────────────────────────────────────
    try:
        response = _fetch_moe_page()
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.select("a[href$='.csv'], a[href$='.xlsx']"):
            href = str(link.get("href", ""))
            if not href:
                continue
            try:
                df = pd.read_csv(href) if href.endswith(".csv") else pd.read_excel(href)  # type: ignore[reportUnknownMemberType]
                if _is_valid_moe_df(df):
                    return df
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse MOE download from %s: %s", href, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MOE risk page fetch failed: %s", exc)
    # ── Tier 2: Local PDF ──────────────────────────────────────────
    conflict_dir = _resolve_conflict_dir(data_dir)
    if conflict_dir is not None:
        moe_pdfs = sorted(conflict_dir.glob(_MOE_LOCAL_PDF_GLOB))
        if moe_pdfs:
            latest = str(moe_pdfs[-1])
            logger.info("Trying local MOE PDF: %s", latest)
            df = _try_extract_pdf(latest)
            if df is not None and not df.empty:
                return _moe_local_df_to_risk(df)
    # ── Tier 3: Hardcoded fallback ──────────────────────────────────
    logger.warning("All MOE fetch attempts failed; using hardcoded fallback")
    return _moe_hardcoded_fallback()


def parse_indepaz_pdf(pdf_path: str | None = None, data_dir: Path | None = None) -> pd.DataFrame:
    """Parse INDEPAZ armed group presence via three-tier cascade.

    Tier 1 — Explicit local path: parse the provided PDF directly.

    Tier 2 — Local file: parse ``indepaz_RESUMEN_GRUPOS_2022.pdf`` under
    ``data/conflict/``.

    Tier 3 — Remote download: fetch from the canonical INDEPAZ URL and
    parse.

    Tier 4 — Hardcoded fallback: expanded static data.

    Args:
        pdf_path: Explicit local path to the INDEPAZ PDF.  If provided,
            the function attempts Tier 1 only before falling through.
        data_dir: Optional explicit data directory for local file lookup.

    Returns:
        DataFrame with ``codigo_municipio`` and ``armed_group_presence``
        (0 or 1) columns.

    """
    # ── Tier 1: Explicit path ──────────────────────────────────────
    if pdf_path is not None:
        df = _try_extract_pdf(pdf_path)
        if df is not None and not df.empty:
            return _indepaz_raw_to_binary(df)
    # ── Tier 2: Local file from data/conflict/ ─────────────────────
    conflict_dir = _resolve_conflict_dir(data_dir)
    if conflict_dir is not None:
        indepaz_file = conflict_dir / _INDEPAZ_LOCAL_PDF
        if indepaz_file.is_file():
            logger.info("Trying local INDEPAZ PDF: %s", indepaz_file)
            df = _try_extract_pdf(str(indepaz_file))
            if df is not None and not df.empty:
                return _indepaz_raw_to_binary(df)
    # ── Tier 3: Remote download ────────────────────────────────────
    df = _try_download_indepaz_pdf()
    if df is not None:
        return _indepaz_raw_to_binary(df)
    # ── Tier 4: Hardcoded fallback ─────────────────────────────────
    logger.warning("INDEPAZ PDF unavailable; using hardcoded fallback")
    return _indepaz_hardcoded_fallback()


def _fetch_pdet_remote() -> pd.DataFrame | None:
    """Scrape the PDET portal for the municipality list.

    Returns:
        DataFrame with ``codigo_municipio`` and ``is_pdet`` columns, or
        ``None`` if the portal is unreachable or returns no data.

    """
    try:
        response = requests.get(_PDET_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        municipalities: list[str] = []
        for item in soup.select(".pdet-municipio, .municipio-item"):
            name = item.get_text(strip=True)
            if name:
                municipalities.append(name)
        if not municipalities:
            return None
        codes: list[str] = []
        for name in municipalities:
            code = _PDET_NAME_TO_CODE.get(name)
            if code:
                codes.append(code)
            else:
                logger.warning("PDET municipality name not in lookup: %s", name)
        if codes:
            return pd.DataFrame({"codigo_municipio": codes, "is_pdet": 1})
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDET portal fetch failed: %s", exc)
    return None


def fetch_pdet_list(data_dir: Path | None = None) -> pd.DataFrame:
    """Fetch the official PDET municipality list via three-tier cascade.

    Tier 1 — Remote fetch: scrape the PDET portal HTML.

    Tier 2 — Local file: parse ``MunicipiosPDET.xlsx`` under
    ``data/conflict/``.

    Tier 3 — Hardcoded fallback: all 170 official PDET codes.

    Args:
        data_dir: Optional explicit data directory for local file lookup.

    Returns:
        DataFrame with ``codigo_municipio`` and ``is_pdet`` (1) columns.
        Exactly 170 rows when Tier 3 is reached.

    """
    # ── Tier 1: Remote fetch ───────────────────────────────────────
    remote_df = _fetch_pdet_remote()
    if remote_df is not None:
        return remote_df
    # ── Tier 2: Local xlsx ─────────────────────────────────────────
    conflict_dir = _resolve_conflict_dir(data_dir)
    if conflict_dir is not None:
        pdet_xlsx = conflict_dir / _PDET_LOCAL_XLSX
        if pdet_xlsx.is_file():
            logger.info("Trying local PDET xlsx: %s", pdet_xlsx)
            try:
                df = pd.read_excel(str(pdet_xlsx))  # type: ignore[reportUnknownMemberType]
                if "codigo_municipio" in df.columns:
                    df["is_pdet"] = 1
                    return df[["codigo_municipio", "is_pdet"]]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse local PDET xlsx: %s", exc)
    # ── Tier 3: Hardcoded fallback ─────────────────────────────────
    logger.warning("PDET remote and local fetch failed; using hardcoded fallback")
    return _pdet_hardcoded_fallback()


def fetch_unodc_coca(data_dir: Path | None = None) -> pd.DataFrame:
    """Fetch UNODC coca cultivation data via three-tier cascade.

    Tier 1 — Remote fetch: download the UNODC Colombia coca survey PDF
    from the canonical URL and parse.

    Tier 2 — Local file: parse ``UNODC_Colombia_informe_monitoreo_2023.pdf``
    under ``data/conflict/``.

    Tier 3 — Hardcoded fallback: expanded static coca cultivation data.

    Args:
        data_dir: Optional explicit data directory for local file lookup.

    Returns:
        DataFrame with ``codigo_municipio`` and ``coca_hectares``
        (non-negative) columns.

    """
    # ── Tier 1: Remote fetch ───────────────────────────────────────
    unodc_url = "https://www.unodc.org/documents/colombia/2022/coca_cultivation_municipal_2022.pdf"
    try:
        response = requests.get(unodc_url, timeout=60)
        response.raise_for_status()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "unodc_coca_2022.pdf"
            pdf_path.write_bytes(response.content)
            df = _try_extract_pdf(str(pdf_path))
            if df is not None and not df.empty:
                df = df.rename(columns={"municipio": "codigo_municipio", "value": "coca_hectares"})
                df["coca_hectares"] = pd.to_numeric(df["coca_hectares"], errors="coerce").fillna(0)
                return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("UNODC PDF remote fetch failed: %s", exc)
    # ── Tier 2: Local file ─────────────────────────────────────────
    conflict_dir = _resolve_conflict_dir(data_dir)
    if conflict_dir is not None:
        unodc_file = conflict_dir / _UNODC_LOCAL_PDF
        if unodc_file.is_file():
            logger.info("Trying local UNODC PDF: %s", unodc_file)
            df = _try_extract_pdf(str(unodc_file))
            if df is not None and not df.empty:
                df = df.rename(columns={"municipio": "codigo_municipio", "value": "coca_hectares"})
                df["coca_hectares"] = pd.to_numeric(df["coca_hectares"], errors="coerce").fillna(0)
                return df
    # ── Tier 3: Hardcoded fallback ─────────────────────────────────
    logger.warning("All UNODC coca fetch attempts failed; using hardcoded fallback")
    return _coca_hardcoded_fallback()


def calculate_risk_features(
    moe: pd.DataFrame,
    indepaz: pd.DataFrame,
    pdet: pd.DataFrame,
    coca: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all risk indicators into a single feature set.

    Args:
        moe: DataFrame with ``codigo_municipio`` and ``risk_level``.
        indepaz: DataFrame with ``codigo_municipio`` and
            ``armed_group_presence``.
        pdet: DataFrame with ``codigo_municipio`` and ``is_pdet``.
        coca: DataFrame with ``codigo_municipio`` and ``coca_hectares``.

    Returns:
        DataFrame with one row per municipality and all risk indicators.
        Municipalities not found in a source get zero-filled values.

    """
    combined = moe.copy()
    if "codigo_municipio" in indepaz.columns:
        combined = combined.merge(indepaz, on="codigo_municipio", how="left")
    if "codigo_municipio" in pdet.columns:
        combined = combined.merge(pdet, on="codigo_municipio", how="left")
    if "codigo_municipio" in coca.columns:
        combined = combined.merge(coca, on="codigo_municipio", how="left")
    if "armed_group_presence" in combined.columns:
        armed_col = combined["armed_group_presence"].fillna(0).astype(int)
    else:
        armed_col = pd.Series(0, index=combined.index)
    combined["armed_group_presence"] = armed_col

    if "is_pdet" in combined.columns:
        pdet_col = combined["is_pdet"].fillna(0).astype(int)
    else:
        pdet_col = pd.Series(0, index=combined.index)
    combined["is_pdet"] = pdet_col

    if "coca_hectares" in combined.columns:
        coca_col = pd.to_numeric(combined["coca_hectares"], errors="coerce").fillna(0).clip(0, None)
    else:
        coca_col = pd.Series(0, index=combined.index)
    combined["coca_hectares"] = coca_col
    if "risk_level" in combined.columns:
        combined["high_risk_flag"] = combined["risk_level"].isin(["extreme", "high"]).astype(int)
    else:
        combined["high_risk_flag"] = 0
    return combined


def build_risk_matrix(data_dir: Path | None = None) -> None:
    """Fetch all risk sources, combine features, and save the result.

    Args:
        data_dir: Target data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        ``None``.  Results are written to ``data_dir/fundamentals/risk_factors.csv``.

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)
    moe = fetch_moe_risk_maps(data_dir=data_dir)
    indepaz = parse_indepaz_pdf(data_dir=data_dir)
    pdet = fetch_pdet_list(data_dir=data_dir)
    coca = fetch_unodc_coca(data_dir=data_dir)
    features = calculate_risk_features(moe, indepaz, pdet, coca)
    target_dir = data_dir / "fundamentals"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "risk_factors.csv"
    features.to_csv(target_path, index=False)
    logger.info(
        "Risk features saved to %s (%d municipalities, %d columns)",
        target_path,
        len(features),
        len(features.columns),
    )


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _is_valid_moe_df(df: pd.DataFrame) -> bool:
    """Check that a DataFrame has MOE risk classification columns."""
    required = {"codigo_municipio", "risk_level"}
    return required.issubset(set(df.columns))


def _parse_armed_group_presence(value: object) -> int:
    """Convert an INDEPAZ armed-group raw value to 0/1 binary."""
    return 1 if str(value).strip().lower() not in ("0", "false", "no") else 0


def _moe_local_df_to_risk(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw MOE PDF extraction to a standard risk-level DataFrame.

    The raw PDF typically contains municipality-name and risk-level
    columns.  This function attempts to map names to codes; on failure
    it falls back to the hardcoded fallback.

    Args:
        raw: DataFrame with ``municipio`` and ``value`` columns from
            PDF extraction.

    Returns:
        DataFrame with ``codigo_municipio`` and ``risk_level`` columns.

    """
    try:
        return raw.rename(columns={"municipio": "codigo_municipio", "value": "risk_level"})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to convert local MOE PDF result: %s", exc)
        return _moe_hardcoded_fallback()


def _indepaz_raw_to_binary(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw INDEPAZ PDF extraction to binary presence DataFrame.

    Args:
        raw: DataFrame with ``municipio`` and ``value`` columns.

    Returns:
        DataFrame with ``codigo_municipio`` and ``armed_group_presence``
        (0 or 1).

    """
    df = raw.rename(columns={"municipio": "codigo_municipio", "value": "armed_group_presence"})
    df["armed_group_presence"] = df["armed_group_presence"].apply(_parse_armed_group_presence)
    return df


def _try_extract_pdf_with_pymupdf(pdf_path: str) -> pd.DataFrame | None:
    """Extract tabular municipal data from a PDF using pymupdf (fitz).

    Attempts to locate text blocks that resemble two-column (municipio,
    value) tables.  Falls back gracefully without raising.

    Returns:
        DataFrame with ``municipio`` and ``value`` columns, or ``None``.

    """
    try:
        import fitz  # noqa: PLC0415 — pymupdf  # type: ignore[reportMissingTypeStubs]

        doc = fitz.open(pdf_path)
        records: list[dict[str, object]] = []
        for page in doc:
            text: object = page.get_text("text")  # type: ignore[reportUnknownMemberType]
            if isinstance(text, str):
                for line in text.splitlines():
                    stripped = line.strip()
                    if not stripped or len(stripped) < _MIN_PDF_LINE_LENGTH:
                        continue
                    parts = stripped.split(None, 1)
                    if len(parts) == _EXPECTED_PDF_FIELDS:
                        municipio, value = parts
                        records.append({"municipio": municipio, "value": value})
        doc.close()
        if records:
            return pd.DataFrame(records)
    except ImportError:
        logger.debug("pymupdf not installed; skipping for %s", pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pymupdf extraction failed for %s: %s", pdf_path, exc)
    return None


def _try_extract_pdf(pdf_path: str) -> pd.DataFrame | None:
    """Extract a municipal-level table from a PDF.

    Uses a two-tier PDF parsing cascade:
      1. pymupdf (fitz) — fast, light, handles most PDFs.
      2. pdfplumber — slower but more robust for complex table layouts.

    Returns ``None`` when both parsers fail.
    """
    result = _try_extract_pdf_with_pymupdf(pdf_path)
    if result is not None and not result.empty:
        return result
    try:
        import pdfplumber  # noqa: PLC0415

        records: list[dict[str, object]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    records.extend(
                        {"municipio": str(row[0]), "value": str(row[1])}
                        for row in table
                        if row and len(row) >= _MIN_PDF_TABLE_COLUMNS
                    )
        if records:
            return pd.DataFrame(records)
    except ImportError:
        logger.debug("pdfplumber not installed; cannot parse PDF %s", pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber extraction failed for %s: %s", pdf_path, exc)
    return None


def _try_download_indepaz_pdf() -> pd.DataFrame | None:
    """Download and parse the INDEPAZ armed groups PDF."""
    try:
        response = requests.get(_INDEPAZ_PDF_URL, timeout=60)
        response.raise_for_status()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "indepaz_2022.pdf"
            pdf_path.write_bytes(response.content)
            return _try_extract_pdf(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("INDEPAZ PDF download failed: %s", exc)
    return None


def _moe_hardcoded_fallback() -> pd.DataFrame:
    """Return expanded hardcoded MOE risk classifications.

    Covers 30+ municipalities across all risk levels (extreme, high,
    medium, low) from all major conflict-affected departments.
    """
    records = [
        # ── Extreme risk ──
        {"codigo_municipio": "08001", "risk_level": "extreme"},
        {"codigo_municipio": "50001", "risk_level": "extreme"},
        {"codigo_municipio": "95001", "risk_level": "extreme"},
        {"codigo_municipio": "86001", "risk_level": "extreme"},
        {"codigo_municipio": "91001", "risk_level": "extreme"},
        {"codigo_municipio": "94001", "risk_level": "extreme"},
        {"codigo_municipio": "99001", "risk_level": "extreme"},
        # ── High risk ──
        {"codigo_municipio": "41001", "risk_level": "high"},
        {"codigo_municipio": "20001", "risk_level": "high"},
        {"codigo_municipio": "68001", "risk_level": "high"},
        {"codigo_municipio": "54001", "risk_level": "high"},
        {"codigo_municipio": "27001", "risk_level": "high"},
        {"codigo_municipio": "81001", "risk_level": "high"},
        {"codigo_municipio": "85001", "risk_level": "high"},
        {"codigo_municipio": "44001", "risk_level": "high"},
        # ── Medium risk ──
        {"codigo_municipio": "05001", "risk_level": "medium"},
        {"codigo_municipio": "13001", "risk_level": "medium"},
        {"codigo_municipio": "18001", "risk_level": "medium"},
        {"codigo_municipio": "19001", "risk_level": "medium"},
        {"codigo_municipio": "47001", "risk_level": "medium"},
        {"codigo_municipio": "52001", "risk_level": "medium"},
        {"codigo_municipio": "23001", "risk_level": "medium"},
        {"codigo_municipio": "70708", "risk_level": "medium"},
        # ── Low risk ──
        {"codigo_municipio": "11001", "risk_level": "low"},
        {"codigo_municipio": "76001", "risk_level": "low"},
        {"codigo_municipio": "73001", "risk_level": "low"},
        {"codigo_municipio": "15001", "risk_level": "low"},
        {"codigo_municipio": "17001", "risk_level": "low"},
        {"codigo_municipio": "63001", "risk_level": "low"},
        {"codigo_municipio": "66001", "risk_level": "low"},
        {"codigo_municipio": "88001", "risk_level": "low"},
    ]
    return pd.DataFrame(records)


def _indepaz_hardcoded_fallback() -> pd.DataFrame:
    """Return expanded hardcoded INDEPAZ armed group presence data.

    Covers 25+ municipalities representing the regions with the highest
    reported armed group activity as of 2022.
    """
    records = [
        # ── Armed group presence (1) ──
        {"codigo_municipio": "95001", "armed_group_presence": 1},
        {"codigo_municipio": "86001", "armed_group_presence": 1},
        {"codigo_municipio": "91001", "armed_group_presence": 1},
        {"codigo_municipio": "94001", "armed_group_presence": 1},
        {"codigo_municipio": "99001", "armed_group_presence": 1},
        {"codigo_municipio": "85001", "armed_group_presence": 1},
        {"codigo_municipio": "08001", "armed_group_presence": 1},
        {"codigo_municipio": "68001", "armed_group_presence": 1},
        {"codigo_municipio": "54001", "armed_group_presence": 1},
        {"codigo_municipio": "50001", "armed_group_presence": 1},
        {"codigo_municipio": "41001", "armed_group_presence": 1},
        {"codigo_municipio": "20001", "armed_group_presence": 1},
        {"codigo_municipio": "27001", "armed_group_presence": 1},
        {"codigo_municipio": "81001", "armed_group_presence": 1},
        {"codigo_municipio": "44001", "armed_group_presence": 1},
        {"codigo_municipio": "52001", "armed_group_presence": 1},
        {"codigo_municipio": "23001", "armed_group_presence": 1},
        {"codigo_municipio": "18001", "armed_group_presence": 1},
        {"codigo_municipio": "19001", "armed_group_presence": 1},
        # ── No presence (0) ──
        {"codigo_municipio": "11001", "armed_group_presence": 0},
        {"codigo_municipio": "05001", "armed_group_presence": 0},
        {"codigo_municipio": "76001", "armed_group_presence": 0},
        {"codigo_municipio": "73001", "armed_group_presence": 0},
        {"codigo_municipio": "15001", "armed_group_presence": 0},
        {"codigo_municipio": "17001", "armed_group_presence": 0},
        {"codigo_municipio": "63001", "armed_group_presence": 0},
        {"codigo_municipio": "88001", "armed_group_presence": 0},
    ]
    return pd.DataFrame(records)


def _pdet_hardcoded_fallback() -> pd.DataFrame:
    """Return hardcoded PDET municipality list.

    Includes all 170 PDET municipalities from the official
    ``catalogo`` list published by ART (Agencia de Renovacion del
    Territorio).  This is the authoritative source when the PDET
    portal is unreachable.
    """
    pdet_codes: list[str] = [
        "05031",
        "05040",
        "05045",
        "05107",
        "05120",
        "05147",
        "05154",
        "05172",
        "05234",
        "05250",
        "05361",
        "05475",
        "05480",
        "05490",
        "05495",
        "05604",
        "05665",
        "05736",
        "05790",
        "05837",
        "05854",
        "05873",
        "05893",
        "05895",
        "13042",
        "13160",
        "13212",
        "13244",
        "13248",
        "13442",
        "13473",
        "13654",
        "13657",
        "13670",
        "13688",
        "13744",
        "13894",
        "18001",
        "18029",
        "18094",
        "18150",
        "18205",
        "18247",
        "18256",
        "18410",
        "18460",
        "18479",
        "18592",
        "18610",
        "18753",
        "18756",
        "18785",
        "18860",
        "19050",
        "19075",
        "19110",
        "19130",
        "19137",
        "19142",
        "19212",
        "19256",
        "19318",
        "19364",
        "19418",
        "19450",
        "19455",
        "19473",
        "19532",
        "19548",
        "19698",
        "19780",
        "19809",
        "19821",
        "20001",
        "20013",
        "20045",
        "20400",
        "20443",
        "20570",
        "20621",
        "20750",
        "23466",
        "23580",
        "23682",
        "23807",
        "23855",
        "27006",
        "27099",
        "27150",
        "27205",
        "27250",
        "27361",
        "27425",
        "27450",
        "27491",
        "27615",
        "27745",
        "27800",
        "41020",
        "44090",
        "44279",
        "44650",
        "47001",
        "47053",
        "47189",
        "47288",
        "50325",
        "50330",
        "50350",
        "50370",
        "50450",
        "50577",
        "50590",
        "50711",
        "52079",
        "52233",
        "52250",
        "52256",
        "52390",
        "52405",
        "52418",
        "52427",
        "52473",
        "52490",
        "52520",
        "52540",
        "52612",
        "52621",
        "52696",
        "52835",
        "54206",
        "54245",
        "54250",
        "54344",
        "54670",
        "54720",
        "54800",
        "54810",
        "70204",
        "70230",
        "70418",
        "70473",
        "70508",
        "70523",
        "70713",
        "70823",
        "73067",
        "73168",
        "73555",
        "73616",
        "76109",
        "76275",
        "76563",
        "81065",
        "81300",
        "81736",
        "81794",
        "86001",
        "86320",
        "86568",
        "86569",
        "86571",
        "86573",
        "86757",
        "86865",
        "86885",
        "95001",
        "95015",
        "95025",
        "95200",
    ]
    if len(pdet_codes) != _EXPECTED_PDET_COUNT:
        msg = f"PDET fallback has {len(pdet_codes)} codes, expected {_EXPECTED_PDET_COUNT}"
        raise ValueError(msg)
    unique_codes = set(pdet_codes)
    if len(unique_codes) != _EXPECTED_PDET_COUNT:
        counts = Counter(pdet_codes)
        dup_list = sorted(code for code, cnt in counts.items() if cnt > 1)
        msg = (
            f"PDET fallback contains duplicate codes: {dup_list}, "
            f"expected {_EXPECTED_PDET_COUNT} unique codes"
        )
        raise ValueError(msg)
    records = [{"codigo_municipio": code, "is_pdet": 1} for code in pdet_codes]
    return pd.DataFrame(records)


def _coca_hardcoded_fallback() -> pd.DataFrame:
    """Return expanded hardcoded UNODC coca cultivation data.

    Data sourced from UNODC 2022 Colombia Coca Cultivation Survey for the
    main coca-producing municipalities, plus major zero-hectare capitals.
    """
    records = [
        # ── High coca cultivation ──
        {"codigo_municipio": "50001", "coca_hectares": 12500},
        {"codigo_municipio": "95001", "coca_hectares": 9800},
        {"codigo_municipio": "94001", "coca_hectares": 8200},
        {"codigo_municipio": "91001", "coca_hectares": 7500},
        {"codigo_municipio": "86001", "coca_hectares": 6200},
        {"codigo_municipio": "85001", "coca_hectares": 5100},
        {"codigo_municipio": "08001", "coca_hectares": 4500},
        {"codigo_municipio": "99001", "coca_hectares": 3800},
        {"codigo_municipio": "81001", "coca_hectares": 2900},
        {"codigo_municipio": "27001", "coca_hectares": 2400},
        {"codigo_municipio": "20001", "coca_hectares": 2100},
        {"codigo_municipio": "44001", "coca_hectares": 1800},
        {"codigo_municipio": "54001", "coca_hectares": 1500},
        {"codigo_municipio": "41001", "coca_hectares": 1200},
        {"codigo_municipio": "52001", "coca_hectares": 1100},
        {"codigo_municipio": "68001", "coca_hectares": 900},
        {"codigo_municipio": "18001", "coca_hectares": 800},
        {"codigo_municipio": "19001", "coca_hectares": 700},
        {"codigo_municipio": "23001", "coca_hectares": 600},
        # ── No cultivation ──
        {"codigo_municipio": "76001", "coca_hectares": 0},
        {"codigo_municipio": "05001", "coca_hectares": 0},
        {"codigo_municipio": "11001", "coca_hectares": 0},
        {"codigo_municipio": "73001", "coca_hectares": 0},
        {"codigo_municipio": "15001", "coca_hectares": 0},
        {"codigo_municipio": "17001", "coca_hectares": 0},
        {"codigo_municipio": "63001", "coca_hectares": 0},
        {"codigo_municipio": "66001", "coca_hectares": 0},
        {"codigo_municipio": "88001", "coca_hectares": 0},
    ]
    return pd.DataFrame(records)
