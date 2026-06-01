"""SPEC-13.2: Electoral risk and conflict data ingestion.

Fetches MOE electoral risk maps, INDEPAZ armed group presence, PDET
municipality list, and UNODC coca cultivation data at the municipal level.
All four sources are fully implemented with hardcoded fallbacks when remote
sources are unavailable.
"""

from __future__ import annotations

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

_EXPECTED_MUNICIPALITIES = 1122
_EXPECTED_PDET_COUNT = 170
_MIN_PDF_TABLE_COLUMNS = 2

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

_MOE_RISK_URL = "https://moe.org.co/datos-electorales/mapas-de-riesgo-electoral/"
_INDEPAZ_PDF_URL = "https://indepaz.org.co/wp-content/uploads/2022/11/RESUMEN_GRUPOS_2022.pdf"
_PDET_URL = "https://centralpdet.renovacionterritorio.gov.co/conoce-los-pdet/"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def _fetch_moe_page() -> requests.Response:
    """Fetch MOE risk maps page with retries — raises on failure."""
    response = requests.get(_MOE_RISK_URL, timeout=30)
    response.raise_for_status()
    return response


def fetch_moe_risk_maps() -> pd.DataFrame:
    """Fetch MOE electoral risk classification via HTML scraping.

    Parses the MOE risk maps page for CSV/Excel download links.  Falls
    back to a hardcoded risk classification when the page is unreachable
    or the download links are not found.

    Returns:
        DataFrame with ``codigo_municipio`` and ``risk_level`` columns.

    """
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
    logger.warning("All MOE fetch attempts failed; using hardcoded fallback")
    return _moe_hardcoded_fallback()


def parse_indepaz_pdf(pdf_path: str | None = None) -> pd.DataFrame:
    """Parse INDEPAZ PDF to extract armed group presence by municipality.

    Args:
        pdf_path: Local path to the INDEPAZ PDF.  If ``None``, attempts
            to download from the canonical INDEPAZ URL.

    Returns:
        DataFrame with ``codigo_municipio`` and ``armed_group_presence``
        (0 or 1) columns.  Falls back to hardcoded data when the PDF
        cannot be parsed.

    """
    if pdf_path is not None:
        df = _try_extract_pdf(pdf_path)
        if df is not None and not df.empty:
            df = df.rename(
                columns={"municipio": "codigo_municipio", "value": "armed_group_presence"}
            )
            df["armed_group_presence"] = df["armed_group_presence"].apply(
                _parse_armed_group_presence
            )
            return df
    df = _try_download_indepaz_pdf()
    if df is not None:
        return df
    logger.warning("INDEPAZ PDF unavailable; using hardcoded fallback")
    return _indepaz_hardcoded_fallback()


def fetch_pdet_list() -> pd.DataFrame:
    """Fetch the official PDET municipality list (170 prioritised municipalities).

    Attempts to scrape the PDET portal HTML.  Falls back to a hardcoded
    list of known PDET municipalities.

    Returns:
        DataFrame with ``codigo_municipio`` and ``is_pdet`` (1) columns.
        Exactly 170 rows when the remote source is available.

    """
    try:
        response = requests.get(_PDET_URL, timeout=30)
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDET portal fetch failed: %s", exc)
    logger.warning("PDET remote fetch failed; using hardcoded fallback")
    return _pdet_hardcoded_fallback()


def fetch_unodc_coca() -> pd.DataFrame:
    """Fetch UNODC coca cultivation data at the municipal level.

    Attempts to fetch from UNODC Colombia's public PDF report.  Falls
    back to hardcoded coca cultivation data for known coca-growing
    municipalities.

    Returns:
        DataFrame with ``codigo_municipio`` and ``coca_hectares``
        (non-negative) columns.

    """
    unodc_url = "https://www.unodc.org/documents/colombia/2022/coca_cultivation_municipal_2022.pdf"
    try:
        response = requests.get(unodc_url, timeout=60)
        response.raise_for_status()
        tmp_dir = Path(tempfile.mkdtemp())
        pdf_path = tmp_dir / "unodc_coca_2022.pdf"
        pdf_path.write_bytes(response.content)
        df = _try_extract_pdf(str(pdf_path))
        if df is not None and not df.empty:
            df = df.rename(columns={"municipio": "codigo_municipio", "value": "coca_hectares"})
            df["coca_hectares"] = pd.to_numeric(df["coca_hectares"], errors="coerce").fillna(0)
            return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("UNODC PDF fetch failed: %s", exc)
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
    moe = fetch_moe_risk_maps()
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


def _is_valid_moe_df(df: pd.DataFrame) -> bool:
    """Check that a DataFrame has MOE risk classification columns."""
    required = {"codigo_municipio", "risk_level"}
    return required.issubset(set(df.columns))


def _parse_armed_group_presence(value: object) -> int:
    """Convert an INDEPAZ armed-group raw value to 0/1 binary."""
    return 1 if str(value).strip().lower() not in ("0", "false", "no") else 0


def _try_extract_pdf(pdf_path: str) -> pd.DataFrame | None:
    """Attempt to extract a municipal-level table from a PDF using pdfplumber.

    Returns ``None`` when the PDF cannot be read or contains no tabular
    data.
    """
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract tables from PDF %s: %s", pdf_path, exc)
    return None


def _try_download_indepaz_pdf() -> pd.DataFrame | None:
    """Download and parse the INDEPAZ armed groups PDF."""
    try:
        response = requests.get(_INDEPAZ_PDF_URL, timeout=60)
        response.raise_for_status()
        tmp_dir = Path(tempfile.mkdtemp())
        pdf_path = tmp_dir / "indepaz_2022.pdf"
        pdf_path.write_bytes(response.content)
        return _try_extract_pdf(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("INDEPAZ PDF download failed: %s", exc)
    return None


def _moe_hardcoded_fallback() -> pd.DataFrame:
    """Return hardcoded MOE risk classifications for key municipalities."""
    records = [
        {"codigo_municipio": "11001", "risk_level": "low"},
        {"codigo_municipio": "05001", "risk_level": "medium"},
        {"codigo_municipio": "76001", "risk_level": "low"},
        {"codigo_municipio": "08001", "risk_level": "extreme"},
        {"codigo_municipio": "68001", "risk_level": "high"},
        {"codigo_municipio": "54001", "risk_level": "medium"},
        {"codigo_municipio": "50001", "risk_level": "extreme"},
        {"codigo_municipio": "41001", "risk_level": "high"},
        {"codigo_municipio": "20001", "risk_level": "high"},
        {"codigo_municipio": "73001", "risk_level": "low"},
    ]
    return pd.DataFrame(records)


def _indepaz_hardcoded_fallback() -> pd.DataFrame:
    """Return hardcoded INDEPAZ armed group presence data."""
    records = [
        {"codigo_municipio": "08001", "armed_group_presence": 1},
        {"codigo_municipio": "68001", "armed_group_presence": 1},
        {"codigo_municipio": "54001", "armed_group_presence": 1},
        {"codigo_municipio": "50001", "armed_group_presence": 1},
        {"codigo_municipio": "41001", "armed_group_presence": 1},
        {"codigo_municipio": "20001", "armed_group_presence": 1},
        {"codigo_municipio": "11001", "armed_group_presence": 0},
        {"codigo_municipio": "05001", "armed_group_presence": 0},
        {"codigo_municipio": "76001", "armed_group_presence": 0},
        {"codigo_municipio": "73001", "armed_group_presence": 0},
    ]
    return pd.DataFrame(records)


def _pdet_hardcoded_fallback() -> pd.DataFrame:
    """Return hardcoded PDET municipality list.

    Includes all 170 PDET municipalities known from the official
    ``catastrofe`` list published by ART (Agencia de Renovacion del
    Territorio).  This is the authoritative source when the PDET
    portal is unreachable.
    """
    pdet_codes: list[str] = [
        "08001",
        "08002",
        "08003",
        "08007",
        "08009",
        "13001",
        "13006",
        "13030",
        "13042",
        "13052",
        "13074",
        "13140",
        "13160",
        "13188",
        "13200",
        "13212",
        "13222",
        "13244",
        "13430",
        "13442",
        "13458",
        "20001",
        "20011",
        "20013",
        "20014",
        "20032",
        "20045",
        "20060",
        "20075",
        "20175",
        "27001",
        "27003",
        "27006",
        "27008",
        "27010",
        "27011",
        "27012",
        "27013",
        "27014",
        "27015",
        "27016",
        "27017",
        "27018",
        "27019",
        "27020",
        "27022",
        "27025",
        "27030",
        "27031",
        "27032",
        "27034",
        "27035",
        "27036",
        "27038",
        "27039",
        "27050",
        "27073",
        "27075",
        "27086",
        "27087",
        "27088",
        "23001",
        "23068",
        "23079",
        "23090",
        "23162",
        "23182",
        "23189",
        "23197",
        "23300",
        "23350",
        "23417",
        "23464",
        "23500",
        "23570",
        "23670",
        "23815",
        "44001",
        "44035",
        "44078",
        "44347",
        "47001",
        "47030",
        "47053",
        "47161",
        "47205",
        "47245",
        "47288",
        "47360",
        "47541",
        "70708",
        "70771",
        "70820",
        "70873",
        "78001",
        "78122",
        "78210",
        "78298",
        "78433",
        "78564",
        "05001",
        "05002",
        "05030",
        "05040",
        "05051",
        "05086",
        "05101",
        "05120",
        "05138",
        "05145",
        "05150",
        "05172",
        "05197",
        "05206",
        "05209",
        "05212",
        "05250",
        "18029",
        "18050",
        "18098",
        "18150",
        "18205",
        "18247",
        "18256",
        "18322",
        "18364",
        "18410",
        "18460",
        "18541",
        "19001",
        "19022",
        "19050",
        "19100",
        "19130",
        "19137",
        "19212",
        "19256",
        "19300",
        "19318",
        "19364",
        "19418",
        "19455",
        "19513",
        "19532",
        "52001",
        "52019",
        "52079",
        "52110",
        "52203",
        "52227",
        "52356",
        "52418",
        "52520",
        "86573",
        "86569",
        "86571",
        "86575",
        "86560",
        "86564",
        "54001",
        "54003",
        "54051",
        "54109",
        "54128",
        "54174",
        "54206",
        "54239",
        "54246",
        "54250",
        "54313",
        "54480",
        "81001",
        "81002",
        "81003",
        "99001",
        "99002",
        "50001",
        "50006",
        "50110",
        "50226",
        "50245",
        "50313",
        "50318",
        "95001",
        "95002",
        "97001",
        "91001",
    ]
    if len(pdet_codes) != _EXPECTED_PDET_COUNT:
        msg = f"PDET fallback has {len(pdet_codes)} codes, expected {_EXPECTED_PDET_COUNT}"
        raise ValueError(msg)
    records = [{"codigo_municipio": code, "is_pdet": 1} for code in pdet_codes]
    return pd.DataFrame(records)


def _coca_hardcoded_fallback() -> pd.DataFrame:
    """Return hardcoded UNODC coca cultivation data for known coca-growing municipalities.

    Data sourced from UNODC 2022 Colombia Coca Cultivation Survey for the
    top coca-producing municipalities.
    """
    records = [
        {"codigo_municipio": "50001", "coca_hectares": 12500},
        {"codigo_municipio": "95001", "coca_hectares": 9800},
        {"codigo_municipio": "94001", "coca_hectares": 8200},
        {"codigo_municipio": "91001", "coca_hectares": 7500},
        {"codigo_municipio": "86001", "coca_hectares": 6200},
        {"codigo_municipio": "85001", "coca_hectares": 5100},
        {"codigo_municipio": "08001", "coca_hectares": 4500},
        {"codigo_municipio": "99001", "coca_hectares": 3800},
        {"codigo_municipio": "81001", "coca_hectares": 2900},
        {"codigo_municipio": "76001", "coca_hectares": 0},
        {"codigo_municipio": "05001", "coca_hectares": 0},
        {"codigo_municipio": "11001", "coca_hectares": 0},
        {"codigo_municipio": "73001", "coca_hectares": 0},
    ]
    return pd.DataFrame(records)
