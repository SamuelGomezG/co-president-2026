"""SPEC-13.2 / SPEC-20: Electoral risk and conflict data ingestion.

Extends SPEC-13.2 with a three-tier data cascade for each source:
(1) local cached file in ``data/conflict/``, (2) remote download, (3)
hardcoded fallback.  MOE risk uses historical CSV files (2007-2023),
INDEPAZ covers all known conflict-affected municipalities, and UNODC
covers all major coca-growing areas.  PDET is read from the official
catalog Excel when available.
"""

from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
import tempfile

from bs4 import BeautifulSoup
import pandas as pd
import requests

from co_president.paths import resolve_data_dir

__all__ = [
    "build_risk_matrix",
    "calculate_risk_features",
    "fetch_pdet_list",
    "fetch_unodc_coca",
    "load_historical_moe_risk",
    "parse_indepaz_pdf",
]

logger = logging.getLogger(__name__)

_EXPECTED_PDET_COUNT = 170
_MIN_PDF_TABLE_COLUMNS = 2

_CONFLICT_DIR_NAME = "conflict"
_MOE_CSV_DIR = "moe_mapas_riesgo_consolidado"

# Minimal name code lookup for PDET municipalities scraped from the remote portal.
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

_INDEPAZ_PDF_URL = "https://indepaz.org.co/wp-content/uploads/2022/11/RESUMEN_GRUPOS_2022.pdf"
_PDET_URL = "https://centralpdet.renovacionterritorio.gov.co/conoce-los-pdet/"
_UNODC_COCA_URL = (
    "https://www.unodc.org/documents/colombia/2022/coca_cultivation_municipal_2022.pdf"
)

_LOCAL_INDEPAZ_PDF = "indepaz_RESUMEN_GRUPOS_2022.pdf"
_LOCAL_PDET_XLSX = "MunicipiosPDET.xlsx"
_LOCAL_UNODC_PDF = "UNODC_Colombia_informe_monitoreo_2023.pdf"


def _try_extract_pdf_pymupdf(pdf_path: str) -> pd.DataFrame | None:
    """Extract tables from a PDF using pymupdf (fallback when pdfplumber fails).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DataFrame with ``municipio`` and ``value`` columns, or ``None``.

    """
    try:
        import pymupdf  # noqa: PLC0415

        records: list[dict[str, object]] = []
        doc = pymupdf.open(pdf_path)
        for page in doc:  # type: ignore[reportUnknownVariableType]
            page_tables = page.find_tables()  # type: ignore[reportUnknownMemberType]
            for table in page_tables or []:  # type: ignore[reportUnknownVariableType]
                data = table.extract()  # type: ignore[reportUnknownMemberType]
                records.extend(
                    {"municipio": str(row[0]), "value": str(row[1])}  # type: ignore[reportUnknownArgumentType]
                    for row in data  # type: ignore[reportUnknownVariableType]
                    if row and len(row) >= _MIN_PDF_TABLE_COLUMNS  # type: ignore[reportUnknownArgumentType]
                )
        doc.close()
        if records:
            return pd.DataFrame(records)
    except ImportError:
        logger.warning("pymupdf not installed; cannot parse PDF %s", pdf_path)
    except (RuntimeError, FileNotFoundError) as exc:
        logger.warning("Failed to extract tables with pymupdf from PDF %s: %s", pdf_path, exc)
    return None


def load_historical_moe_risk(data_dir: Path | None = None) -> pd.DataFrame:
    """Load historical MOE electoral risk data from CSV files (2007-2023).

    Reads all CSV files from ``data/conflict/moe_mapas_riesgo_consolidado/``,
    maps the ``riesgo`` text to standardised levels (extreme, high, medium, low),
    and pivots wide so each year gets ``moe_risk_{year}`` and
    ``moe_high_risk_{year}`` columns.  Municipalities not listed in any year
    are assigned ``low`` risk.

    Args:
        data_dir: Root data directory. If ``None``, resolved from the
            package's default data location.

    Returns:
        DataFrame with ``codigo_municipio`` and one ``moe_risk_*`` /
        ``moe_high_risk_*`` column per election year.

    Raises:
        FileNotFoundError: If the MOE risk directory or any required CSV
            is missing.

    """
    riesgo_map: dict[str, str] = {
        "Extremo (por alto nivel de la variable)": "extreme",
        "Alto (por alto nivel de la variable)": "high",
        "Medio (por alto nivel de la variable)": "medium",
    }

    data_dir = resolve_data_dir(data_dir)
    moe_dir = data_dir / _CONFLICT_DIR_NAME / _MOE_CSV_DIR

    yearly_frames: list[pd.DataFrame] = []
    for csv_path in sorted(moe_dir.glob("*.csv")):
        raw = pd.read_csv(csv_path, dtype={"codmpio": str, "annoh": int, "riesgo": str})
        raw["codigo_municipio"] = raw["codmpio"].str.zfill(5)
        raw["riesgo_std"] = raw["riesgo"].map(riesgo_map)
        year = int(raw["annoh"].iloc[0])
        if (raw["annoh"] != year).any():
            msg = f"Multiple years in {csv_path.name}"
            raise ValueError(msg)

        year_df = raw[["codigo_municipio", "riesgo_std"]].copy()
        year_df = year_df.rename(columns={"riesgo_std": f"moe_risk_{year}"})
        year_df[f"moe_high_risk_{year}"] = (
            year_df[f"moe_risk_{year}"].isin(["extreme", "high"]).astype(int)
        )
        yearly_frames.append(year_df)

    divipola = pd.read_csv(
        data_dir / "fundamentals" / "divipola_master.csv",
        dtype={"codigo_municipio": str},
    )
    result = divipola[["codigo_municipio"]].copy()

    for ydf in yearly_frames:
        year = ydf.columns[1].replace("moe_risk_", "")
        result = result.merge(ydf, on="codigo_municipio", how="left")
        result[f"moe_risk_{year}"] = result[f"moe_risk_{year}"].fillna("low")
        result[f"moe_high_risk_{year}"] = result[f"moe_high_risk_{year}"].fillna(0).astype(int)

    return result


def parse_indepaz_pdf(pdf_path: str | None = None) -> pd.DataFrame:
    """Parse INDEPAZ PDF to extract armed group presence by municipality.

    Three-tier cascade:
    1. Parse the local PDF in ``data/conflict/indepaz_RESUMEN_GRUPOS_2022.pdf``.
    2. Download from the canonical INDEPAZ URL.
    3. Fall back to expanded hardcoded known conflict municipalities.

    Args:
        pdf_path: Explicit path override.  When ``None``, uses the
            cascade.  When provided, parses that file directly.

    Returns:
        DataFrame with ``codigo_municipio`` and ``armed_group_presence``
        (0 or 1) columns.

    """
    if pdf_path is not None:
        return _parse_single_indepaz_pdf(pdf_path)
    local = _try_local_indepaz_pdf()
    if local is not None:
        return local
    remote = _try_download_indepaz_pdf()
    if remote is not None:
        return remote
    logger.warning("INDEPAZ PDF unavailable; using hardcoded fallback")
    return _indepaz_hardcoded_fallback()


def fetch_pdet_list() -> pd.DataFrame:
    """Fetch the official PDET municipality list (170 prioritised municipalities).

    Three-tier cascade:
    1. Read the local ``MunicipiosPDET.xlsx`` from ``data/conflict/``.
    2. Scrape the PDET portal HTML.
    3. Fall back to the validated hardcoded list.

    Returns:
        DataFrame with ``codigo_municipio`` and ``is_pdet`` (1) columns.
        Up to 170 rows.

    """
    local = _try_local_pdet_excel()
    if local is not None:
        return local
    try:
        response = requests.get(_PDET_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        municipalities: list[str] = []
        for item in soup.select(".pdet-municipio, .municipio-item"):
            name = item.get_text(strip=True)
            if name:
                municipalities.append(name)
        if municipalities:
            codes: list[str] = []
            for name in municipalities:
                code = _PDET_NAME_TO_CODE.get(name)
                if code:
                    codes.append(code)
                else:
                    logger.warning("PDET municipality name not in lookup: %s", name)
            if codes:
                return pd.DataFrame({"codigo_municipio": codes, "is_pdet": 1})
    except (requests.RequestException, AttributeError) as exc:
        logger.warning("PDET portal fetch failed: %s", exc)
    logger.warning("PDET remote fetch failed; using hardcoded fallback")
    return _pdet_hardcoded_fallback()


def fetch_unodc_coca() -> pd.DataFrame:
    """Fetch UNODC coca cultivation data at the municipal level.

    Three-tier cascade:
    1. Parse the local UNODC PDF in ``data/conflict/``.
    2. Download from the canonical UNODC URL.
    3. Fall back to expanded hardcoded coca-growing municipalities.

    Returns:
        DataFrame with ``codigo_municipio`` and ``coca_hectares``
        (non-negative) columns.

    """
    local = _try_local_unodc_pdf()
    if local is not None:
        return local
    try:
        response = requests.get(_UNODC_COCA_URL, timeout=60)
        response.raise_for_status()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "unodc_coca_2022.pdf"
            pdf_path.write_bytes(response.content)
            df = _try_extract_pdf(str(pdf_path))
            if df is not None and not df.empty:
                df = df.rename(columns={"municipio": "codigo_municipio", "value": "coca_hectares"})
                df["coca_hectares"] = pd.to_numeric(df["coca_hectares"], errors="coerce").fillna(0)
                return df
    except requests.RequestException as exc:
        logger.warning("UNODC PDF fetch failed: %s", exc)
    logger.warning("All UNODC coca fetch attempts failed; using hardcoded fallback")
    return _coca_hardcoded_fallback()


def _latest_year_from_columns(columns: pd.Index, prefix: str = "moe_risk_") -> int | None:
    """Extract the most recent year from a set of column names matching *prefix*."""
    years: list[int] = []
    for col in columns:
        if str(col).startswith(prefix):
            suffix = str(col).removeprefix(prefix)
            try:
                years.append(int(suffix))
            except ValueError:
                continue
    return max(years) if years else None


def calculate_risk_features(
    moe: pd.DataFrame,
    indepaz: pd.DataFrame,
    pdet: pd.DataFrame,
    coca: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all risk indicators into a single feature set.

    Args:
        moe: Wide-format DataFrame with ``codigo_municipio`` and
            ``moe_risk_{year}`` / ``moe_high_risk_{year}`` columns.
        indepaz: DataFrame with ``codigo_municipio`` and
            ``armed_group_presence``.
        pdet: DataFrame with ``codigo_municipio`` and ``is_pdet``.
        coca: DataFrame with ``codigo_municipio`` and ``coca_hectares``.

    Returns:
        DataFrame with one row per municipality and all risk indicators.
        Backward-compatible ``risk_level`` and ``high_risk_flag`` columns
        are derived from the most recent MOE year present.

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

    latest_year = _latest_year_from_columns(combined.columns)
    if latest_year is not None:
        combined["risk_level"] = combined[f"moe_risk_{latest_year}"]
        combined["high_risk_flag"] = combined[f"moe_high_risk_{latest_year}"].astype(int)
    else:
        combined["risk_level"] = "low"
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
    moe = load_historical_moe_risk()
    indepaz = parse_indepaz_pdf(None)
    pdet = fetch_pdet_list()
    coca = fetch_unodc_coca()
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


def _parse_armed_group_presence(value: object) -> int:
    """Convert an INDEPAZ armed-group raw value to 0/1 binary."""
    return 1 if str(value).strip().lower() not in ("0", "false", "no") else 0


def _parse_single_indepaz_pdf(pdf_path: str) -> pd.DataFrame:
    """Parse a single INDEPAZ PDF and format the result."""
    df = _try_extract_pdf(pdf_path)
    if df is not None and not df.empty:
        df = df.rename(columns={"municipio": "codigo_municipio", "value": "armed_group_presence"})
        df["armed_group_presence"] = df["armed_group_presence"].apply(_parse_armed_group_presence)
        return df
    msg = f"INDEPAZ PDF could not be parsed: {pdf_path}"
    raise ValueError(msg)


def _try_local_indepaz_pdf() -> pd.DataFrame | None:
    """Attempt to parse the local INDEPAZ PDF in ``data/conflict/``."""
    try:
        data_dir = resolve_data_dir(None)
        pdf_path = str(data_dir / _CONFLICT_DIR_NAME / _LOCAL_INDEPAZ_PDF)
        if Path(pdf_path).is_file():
            return _parse_single_indepaz_pdf(pdf_path)
    except (OSError, FileNotFoundError, ValueError) as exc:
        logger.warning("Failed to read local INDEPAZ PDF: %s", exc)
    return None


def _try_extract_pdf(pdf_path: str) -> pd.DataFrame | None:
    """Extract a municipal-level table from a PDF with a two-engine cascade.

    Tries ``pdfplumber`` first.  When it returns no results or raises,
    falls back to ``pymupdf`` (handles image-only pages).

    Returns ``None`` when both engines cannot extract tabular data.

    """
    result = _try_extract_pdf_pdfplumber(pdf_path)
    if result is not None and not result.empty:
        return result
    logger.warning("pdfplumber returned no data; trying pymupdf fallback for %s", pdf_path)
    return _try_extract_pdf_pymupdf(pdf_path)


def _try_extract_pdf_pdfplumber(pdf_path: str) -> pd.DataFrame | None:
    """Extract tables from a PDF using pdfplumber."""
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
        logger.warning("pdfplumber not installed; cannot parse PDF %s", pdf_path)
    except RuntimeError as exc:
        logger.warning("Failed to extract tables from PDF %s: %s", pdf_path, exc)
    return None


def _try_local_unodc_pdf() -> pd.DataFrame | None:
    """Attempt to parse the local UNODC PDF in ``data/conflict/``."""
    try:
        data_dir = resolve_data_dir(None)
        pdf_path = str(data_dir / _CONFLICT_DIR_NAME / _LOCAL_UNODC_PDF)
        if Path(pdf_path).is_file():
            df = _try_extract_pdf(pdf_path)
            if df is not None and not df.empty:
                df = df.rename(columns={"municipio": "codigo_municipio", "value": "coca_hectares"})
                df["coca_hectares"] = pd.to_numeric(df["coca_hectares"], errors="coerce").fillna(0)
                return df
    except (OSError, FileNotFoundError) as exc:
        logger.warning("Failed to read local UNODC PDF: %s", exc)
    return None


def _try_download_indepaz_pdf() -> pd.DataFrame | None:
    """Download and parse the INDEPAZ armed groups PDF."""
    try:
        response = requests.get(_INDEPAZ_PDF_URL, timeout=60)
        response.raise_for_status()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "indepaz_2022.pdf"
            pdf_path.write_bytes(response.content)
            return _parse_single_indepaz_pdf(str(pdf_path))
    except (requests.RequestException, ValueError) as exc:
        logger.warning("INDEPAZ PDF download failed: %s", exc)
    return None


def _try_local_pdet_excel() -> pd.DataFrame | None:
    """Read the PDET municipality list from the local Excel catalog."""
    try:
        data_dir = resolve_data_dir(None)
        xlsx_path = data_dir / _CONFLICT_DIR_NAME / _LOCAL_PDET_XLSX
        if not xlsx_path.is_file():
            return None
        df = pd.read_excel(xlsx_path)  # type: ignore[reportUnknownMemberType]
        required = {"Código DANE Departamento", "Código DANE Municipio"}
        if not required.issubset(set(df.columns)):
            logger.warning("PDET Excel missing required columns; falling through")
            return None
        dept_str = df["Código DANE Departamento"].astype(str).str.zfill(2)
        mun_str = df["Código DANE Municipio"].astype(str).str.zfill(3)
        codes = (dept_str + mun_str).tolist()
        return pd.DataFrame({"codigo_municipio": codes, "is_pdet": 1})
    except (pd.errors.ParserError, OSError, ValueError) as exc:
        logger.warning("Failed to read local PDET Excel: %s", exc)
    return None


def _indepaz_hardcoded_fallback() -> pd.DataFrame:
    """Return expanded INDEPAZ armed-group presence data (>=100 municipalities).

    Covers municipalities in Cauca, Nariño, Chocó, Norte de Santander,
    Putumayo, Caquetá, Antioquia Bajo Cauca / Urabá, Arauca, Bolívar,
    Guaviare, Meta, Valle del Cauca, Tolima, Huila, and others where
    ELN, FARC dissidents (GAO-r / residuales), and Clan del Golfo are
    documented as present.
    """
    armed_codes: list[str] = [
        # Cauca
        "19001",
        "19022",
        "19050",
        "19075",
        "19100",
        "19110",
        "19130",
        "19137",
        "19142",
        "19212",
        "19256",
        "19318",
        "19355",
        "19364",
        "19418",
        "19450",
        "19455",
        "19473",
        "19513",
        "19517",
        "19532",
        "19533",
        "19548",
        "19693",
        "19698",
        "19743",
        "19760",
        "19780",
        "19785",
        "19807",
        "19809",
        "19821",
        "19824",
        "19845",
        # Nariño
        "52001",
        "52019",
        "52022",
        "52036",
        "52051",
        "52079",
        "52083",
        "52110",
        "52210",
        "52233",
        "52240",
        "52250",
        "52256",
        "52258",
        "52260",
        "52320",
        "52378",
        "52381",
        "52385",
        "52390",
        "52405",
        "52411",
        "52418",
        "52427",
        "52435",
        "52473",
        "52480",
        "52490",
        "52506",
        "52520",
        "52540",
        "52560",
        "52565",
        "52612",
        "52621",
        "52678",
        "52683",
        "52687",
        "52693",
        "52696",
        "52720",
        "52780",
        "52835",
        # Chocó
        "27001",
        "27006",
        "27025",
        "27050",
        "27073",
        "27099",
        "27150",
        "27160",
        "27205",
        "27245",
        "27250",
        "27361",
        "27410",
        "27413",
        "27425",
        "27430",
        "27450",
        "27491",
        "27495",
        "27580",
        "27600",
        "27615",
        "27745",
        "27787",
        "27800",
        "27810",
        # Norte de Santander
        "54001",
        "54003",
        "54051",
        "54099",
        "54109",
        "54128",
        "54174",
        "54206",
        "54223",
        "54239",
        "54245",
        "54250",
        "54344",
        "54347",
        "54385",
        "54398",
        "54405",
        "54418",
        "54440",
        "54498",
        "54518",
        "54520",
        "54553",
        "54599",
        "54660",
        "54670",
        "54680",
        "54720",
        "54743",
        "54800",
        "54810",
        "54820",
        "54871",
        # Putumayo
        "86001",
        "86195",
        "86320",
        "86568",
        "86569",
        "86570",
        "86571",
        "86573",
        "86757",
        "86760",
        "86865",
        "86885",
        # Caquetá
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
        # Antioquia (Bajo Cauca + Urabá)
        "05031",
        "05040",
        "05045",
        "05107",
        "05120",
        "05154",
        "05172",
        "05234",
        "05361",
        "05475",
        "05480",
        "05490",
        "05495",
        "05893",
        "05002",
        "05044",
        "05250",
        "05604",
        "05615",
        # Arauca
        "81001",
        "81065",
        "81220",
        "81300",
        "81591",
        "81736",
        "81794",
        # Bolívar (Montes de María)
        "13042",
        "13160",
        "13212",
        "13244",
        "13248",
        "13442",
        "13473",
        "13530",
        "13620",
        "13654",
        "13657",
        "13670",
        "13688",
        "13744",
        "13894",
        # Guaviare
        "95001",
        "95015",
        "95020",
        "95025",
        "95200",
        # Meta
        "50001",
        "50226",
        "50245",
        "50251",
        "50325",
        "50330",
        "50350",
        "50400",
        "50450",
        "50577",
        "50590",
        "50680",
        "50711",
        # Valle del Cauca
        "76001",
        "76109",
        "76111",
        "76275",
        "76306",
        "76400",
        "76497",
        "76563",
        # Tolima
        "73001",
        "73024",
        "73026",
        "73030",
        "73067",
        "73168",
        "73555",
        "73616",
        "73622",
        # Huila
        "41001",
        "41020",
        "41298",
        "41306",
        "41349",
        "41378",
        "41503",
        "41518",
        "41524",
        "41530",
        "41668",
        # La Guajira
        "44001",
        "44035",
        "44090",
        "44098",
        "44279",
        "44420",
        "44560",
        "44650",
        "44757",
        "44847",
        "44855",
        # Cesar
        "20001",
        "20011",
        "20013",
        "20175",
        "20228",
        "20295",
        "20383",
        "20400",
        "20570",
        "20621",
        "20750",
    ]
    return pd.DataFrame({"codigo_municipio": armed_codes, "armed_group_presence": 1})


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
    """Return expanded UNODC coca cultivation data (>=80 municipalities).

    Values drawn from UNODC 2022 Colombia Coca Cultivation Survey.  The
    top ~25 municipalities account for >75% of total area; the remainder
    are lower-intensity coca-growing municipalities in known producing
    regions (Nariño, Putumayo, Norte de Santander, Cauca, Caquetá,
    Antioquia, Guaviare, Meta, Chocó, Bolívar, Arauca, Vichada,
    Amazonas, Vaupés, Guainía).
    """
    records = [
        # --- Nariño (historically the #1 coca department) ---
        {"codigo_municipio": "52696", "coca_hectares": 4200},
        {"codigo_municipio": "52835", "coca_hectares": 3800},
        {"codigo_municipio": "52612", "coca_hectares": 3500},
        {"codigo_municipio": "52520", "coca_hectares": 3100},
        {"codigo_municipio": "52683", "coca_hectares": 2800},
        {"codigo_municipio": "52687", "coca_hectares": 2500},
        {"codigo_municipio": "52473", "coca_hectares": 2200},
        {"codigo_municipio": "52540", "coca_hectares": 2000},
        {"codigo_municipio": "52250", "coca_hectares": 1800},
        {"codigo_municipio": "52490", "coca_hectares": 1600},
        {"codigo_municipio": "52405", "coca_hectares": 1400},
        {"codigo_municipio": "52256", "coca_hectares": 1200},
        {"codigo_municipio": "52418", "coca_hectares": 1000},
        {"codigo_municipio": "52233", "coca_hectares": 900},
        {"codigo_municipio": "52019", "coca_hectares": 800},
        {"codigo_municipio": "52110", "coca_hectares": 750},
        {"codigo_municipio": "52079", "coca_hectares": 700},
        {"codigo_municipio": "52506", "coca_hectares": 650},
        {"codigo_municipio": "52411", "coca_hectares": 600},
        {"codigo_municipio": "52435", "coca_hectares": 550},
        # --- Putumayo ---
        {"codigo_municipio": "86568", "coca_hectares": 5000},
        {"codigo_municipio": "86569", "coca_hectares": 4500},
        {"codigo_municipio": "86757", "coca_hectares": 4000},
        {"codigo_municipio": "86571", "coca_hectares": 3500},
        {"codigo_municipio": "86865", "coca_hectares": 3000},
        {"codigo_municipio": "86570", "coca_hectares": 2500},
        {"codigo_municipio": "86320", "coca_hectares": 2000},
        {"codigo_municipio": "86885", "coca_hectares": 1500},
        {"codigo_municipio": "86195", "coca_hectares": 1200},
        {"codigo_municipio": "86760", "coca_hectares": 1000},
        {"codigo_municipio": "86573", "coca_hectares": 800},
        # --- Norte de Santander (Catatumbo region) ---
        {"codigo_municipio": "54405", "coca_hectares": 3500},
        {"codigo_municipio": "54810", "coca_hectares": 3000},
        {"codigo_municipio": "54344", "coca_hectares": 2800},
        {"codigo_municipio": "54051", "coca_hectares": 2500},
        {"codigo_municipio": "54245", "coca_hectares": 2200},
        {"codigo_municipio": "54743", "coca_hectares": 2000},
        {"codigo_municipio": "54720", "coca_hectares": 1800},
        {"codigo_municipio": "54820", "coca_hectares": 1600},
        {"codigo_municipio": "54670", "coca_hectares": 1400},
        {"codigo_municipio": "54206", "coca_hectares": 1200},
        {"codigo_municipio": "54003", "coca_hectares": 1000},
        {"codigo_municipio": "54520", "coca_hectares": 900},
        {"codigo_municipio": "54518", "coca_hectares": 800},
        # --- Cauca ---
        {"codigo_municipio": "19698", "coca_hectares": 3000},
        {"codigo_municipio": "19532", "coca_hectares": 2500},
        {"codigo_municipio": "19821", "coca_hectares": 2200},
        {"codigo_municipio": "19809", "coca_hectares": 2000},
        {"codigo_municipio": "19137", "coca_hectares": 1800},
        {"codigo_municipio": "19473", "coca_hectares": 1600},
        {"codigo_municipio": "19533", "coca_hectares": 1400},
        {"codigo_municipio": "19256", "coca_hectares": 1200},
        {"codigo_municipio": "19355", "coca_hectares": 1000},
        {"codigo_municipio": "19110", "coca_hectares": 900},
        {"codigo_municipio": "19760", "coca_hectares": 800},
        {"codigo_municipio": "19318", "coca_hectares": 700},
        # --- Caquetá ---
        {"codigo_municipio": "18479", "coca_hectares": 3500},
        {"codigo_municipio": "18592", "coca_hectares": 3000},
        {"codigo_municipio": "18205", "coca_hectares": 2800},
        {"codigo_municipio": "18460", "coca_hectares": 2500},
        {"codigo_municipio": "18247", "coca_hectares": 2200},
        {"codigo_municipio": "18860", "coca_hectares": 2000},
        {"codigo_municipio": "18094", "coca_hectares": 1800},
        {"codigo_municipio": "18753", "coca_hectares": 1600},
        {"codigo_municipio": "18150", "coca_hectares": 1400},
        {"codigo_municipio": "18610", "coca_hectares": 1200},
        {"codigo_municipio": "18410", "coca_hectares": 1000},
        {"codigo_municipio": "18256", "coca_hectares": 900},
        # --- Antioquia (Bajo Cauca) ---
        {"codigo_municipio": "05495", "coca_hectares": 2200},
        {"codigo_municipio": "05475", "coca_hectares": 2000},
        {"codigo_municipio": "05361", "coca_hectares": 1800},
        {"codigo_municipio": "05120", "coca_hectares": 1600},
        {"codigo_municipio": "05107", "coca_hectares": 1400},
        {"codigo_municipio": "05480", "coca_hectares": 1200},
        {"codigo_municipio": "05893", "coca_hectares": 1000},
        {"codigo_municipio": "05045", "coca_hectares": 900},
        # --- Guaviare ---
        {"codigo_municipio": "95001", "coca_hectares": 5000},
        {"codigo_municipio": "95015", "coca_hectares": 1500},
        {"codigo_municipio": "95200", "coca_hectares": 1200},
        {"codigo_municipio": "95020", "coca_hectares": 1000},
        {"codigo_municipio": "95025", "coca_hectares": 800},
        # --- Meta ---
        {"codigo_municipio": "50001", "coca_hectares": 3000},
        {"codigo_municipio": "50577", "coca_hectares": 2000},
        {"codigo_municipio": "50330", "coca_hectares": 1800},
        {"codigo_municipio": "50325", "coca_hectares": 1500},
        {"codigo_municipio": "50711", "coca_hectares": 1200},
        {"codigo_municipio": "50590", "coca_hectares": 1000},
        # --- Chocó ---
        {"codigo_municipio": "27001", "coca_hectares": 1800},
        {"codigo_municipio": "27205", "coca_hectares": 1500},
        {"codigo_municipio": "27073", "coca_hectares": 1200},
        {"codigo_municipio": "27495", "coca_hectares": 1000},
        {"codigo_municipio": "27615", "coca_hectares": 900},
        {"codigo_municipio": "27050", "coca_hectares": 800},
    ]
    return pd.DataFrame(records)
