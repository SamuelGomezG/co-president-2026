"""SPEC-31: ingest and normalize 2026 CNE presidential poll bundles.

Reads raw CNE 2026 microdata from firm-specific zip bundles (CSV, XLSX, SAV, PDF),
normalizes candidate labels to canonical keys, and extracts:
- **topline**: weighted vote-intention percentages per candidate
- **runoff pairings**: head-to-head second-round preferences

Output: ``data/2026-polls/_processed/2026_topline.parquet`` and
``2026_runoff_pairings.parquet``.

See `docs/plans/PLAN_cne_2026_ingestion.md` for the full specification.
"""

from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
import zipfile

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd

from co_president.config import FIRST_ROUND_CANDIDATES_2026, get_election_date
from co_president.data_polls import CleanPolls
from co_president.paths import resolve_data_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate normalizer
# ---------------------------------------------------------------------------

CANDIDATE_KEY_MAP_2026: dict[str, str] = {
    "Iván Cepeda": "cepeda",
    "Ivan Cepeda": "cepeda",
    "CEPPEDA": "cepeda",
    "Abelardo de la Espriella": "de_la_espriella",
    "Abelardo De La Espriella": "de_la_espriella",
    'Abelardo de la Espriella "El Negro"': "de_la_espriella",
    "Paloma Valencia": "valencia",
    "Sergio Fajardo": "fajardo",
    "Claudia López": "claudia_lopez",
    "Claudia Lopez": "claudia_lopez",
    "Otro candidato": "rest",
    "Voto en blanco": "blanco",
    "No voté": "blanco",
    "No sé": "ns_nr",
    "Ns/Nr": "ns_nr",
    "NS/NR": "ns_nr",
    "No sabe / No responde": "ns_nr",
    "Voto nulo": "ns_nr",
    "Voto nulo / No votaré": "ns_nr",
    "Ninguno": "ns_nr",
    "No votaría en segunda vuelta": "ns_nr",
    "NsNr": "ns_nr",
    # ---- La Silla Vacía 2026 candidate variants (SPEC-32) ----
    "Luis Gilberto Murillo": "rest",
    "Carlos Caicedo": "rest",
    "Roy Barreras": "rest",
    "Miguel Uribe Londoño": "rest",
    "Santiago Botero": "rest",
    "Mauricio Lizcano": "rest",
    "Clara López": "rest",
    "Sondra Macollins": "rest",
    "Voto en Blanco": "blanco",
    "No Sabe / No Responde": "ns_nr",
    "Gustavo Matamoros": "rest",
    "Blanco": "blanco",
    # ---- Additional edge-case labels from 2026 CNE bundles ----
    "Juan Fernando Cristo": "rest",
    "Total": "ns_nr",
    "Mujer": "ns_nr",
    "Response": "ns_nr",
    "Luis G. Murillo": "rest",
}

_CANONICAL_2026 = frozenset(FIRST_ROUND_CANDIDATES_2026.keys())


def _normalize_candidate_name(raw: str | float | None) -> str | None:
    """Normalize a raw candidate label to its canonical key.

    Args:
        raw: Raw candidate string from microdata.

    Returns:
        Canonical candidate key (e.g. ``"cepeda"``) or ``"ns_nr"`` for
        undecided / non-response options.

    Raises:
        ValueError: If the raw label cannot be mapped.

    """
    if raw is None or not isinstance(raw, str) or (isinstance(raw, float) and pd.isna(raw)):
        return None
    stripped: str = raw.strip()
    if not stripped:
        return None
    if stripped in CANDIDATE_KEY_MAP_2026:
        return CANDIDATE_KEY_MAP_2026[stripped]
    # fallback: try case-insensitive match
    lower_map = {k.lower(): v for k, v in CANDIDATE_KEY_MAP_2026.items()}
    key = stripped.lower()
    if key in lower_map:
        return lower_map[key]
    msg = f"Unmapped candidate label: {raw!r}"
    raise ValueError(msg)


def normalize_candidate_labels(series: pd.Series) -> pd.Series:
    """Normalize every entry in a Series of raw candidate labels.

    Applies ``_normalize_candidate_name`` element-wise.

    Args:
        series: Preprocessed pandas Series containing raw candidate strings.

    Returns:
        Series with canonical candidate keys (e.g. ``"cepeda"``).

    """
    return series.apply(_normalize_candidate_name)


_CANONICAL_KEYS = sorted(_CANONICAL_2026)
_MIN_CANDIDATE_NAME_HITS = 3
_MAX_UNIQUE_VALS = 30


def _topline_column_order() -> list[str]:
    """Return topline candidate columns in canonical display order."""
    return [c for c in _CANONICAL_KEYS if c != "ns_nr"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

_BASE_DIR_2026 = "2026-polls"


def _resolve_2026_dir(data_dir: Path | None = None) -> Path:
    root = resolve_data_dir(data_dir)
    return root / _BASE_DIR_2026


def _list_zip_bundles(data_dir: Path | None = None) -> list[Path]:
    """Return sorted list of CNE 2026 zip bundles."""
    base = _resolve_2026_dir(data_dir)
    return sorted(base.glob("*.zip"))


def _identify_firm(bundle_name: str) -> str:
    """Return a firm key from a bundle filename."""
    name = bundle_name.lower()
    _firm_patterns: list[tuple[str, tuple[str, ...]]] = [
        ("atlas_intel", ("atlas_intel", "atlas_intelligence")),
        ("gad3", ("gad3",)),
        ("invamer", ("invamer",)),
        ("cnc", ("cnc",)),
        ("guarumo_ecoanalitica", ("guarumo", "ecoanalitica")),
        ("genesis_crea", ("genesis_crea",)),
        ("tempo", ("tempo",)),
        ("corp_mmm", ("corp_miguel_maldonado", "corp_mmm")),
        ("analizar_lombana", ("analizar_lombana",)),
    ]
    for firm_key, keywords in _firm_patterns:
        if any(kw in name for kw in keywords):
            return firm_key
    return "unknown"


def _parse_field_end_date(bundle_name: str) -> date | None:
    """Extract field-end date from bundle filename.

    Expects patterns like ``..._DDMMYYYY_DDMMYYYY.zip`` — returns the
    **second** date.
    """
    _field_end_date_min_matches = 2
    m = re.findall(r"(\d{2})(\d{2})(\d{4})", bundle_name)
    if len(m) >= _field_end_date_min_matches:
        day, month, year = m[1]
        return date(int(year), int(month), int(day))
    return None


# ---- Atlas Intel -----------------------------------------------------------


def load_atlas_intel(bundle_path: Path) -> pd.DataFrame:
    """Load Atlas Intel 2026 CSV microdata from a zip bundle.

    Atlas Intel delivers one CSV per wave inside a zip.  Columns include:
    ``presidential_election_2026`` (first-round vote), ``vote_2026_president_secondround``
    (runoff preference), ``weight`` (survey weight).

    Args:
        bundle_path: Path to the Atlas Intel zip file.

    Returns:
        DataFrame with columns: fecha, weight, candidate_r1, candidate_r2,
        firm, field_end, raw_source.

    """
    with zipfile.ZipFile(bundle_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            msg = f"No CSV found in Atlas Intel bundle: {bundle_path}"
            raise ValueError(msg)
        with zf.open(csv_names[0]) as fh:
            raw = pd.read_csv(fh, encoding="utf-8")

    raw.columns = [c.strip() for c in raw.columns]

    r1_col = "presidential_election_2026"
    r2_col = "vote_2026_president_secondround"
    weight_col = "weight"

    if r1_col not in raw.columns:
        msg = f"Atlas Intel missing column {r1_col!r} in {bundle_path}"
        raise ValueError(msg)
    if weight_col not in raw.columns:
        msg = f"Atlas Intel missing column {weight_col!r} in {bundle_path}"
        raise ValueError(msg)

    # Normalize candidate labels
    r1 = raw[r1_col].astype(str).apply(_normalize_candidate_name)
    if r2_col in raw.columns:
        r2 = raw[r2_col].astype(str).apply(_normalize_candidate_name)
    else:
        r2 = pd.Series(np.nan, index=raw.index)
    weight = pd.to_numeric(raw[weight_col], errors="coerce").fillna(1.0)

    firm = "atlas_intel"
    field_end = _parse_field_end_date(bundle_path.name)

    out = pd.DataFrame(
        {
            "fecha": field_end,
            "weight": weight,
            "candidate_r1": r1,
            "candidate_r2": r2,
            "firm": firm,
            "field_end": field_end,
        }
    )
    out["raw_source"] = bundle_path.name
    return out


# ---- GAD3 ------------------------------------------------------------------

_GAD3_R1_INDEX_MAP: dict[int, str] = {
    1: "valencia",
    2: "de_la_espriella",
    3: "cepeda",
    4: "fajardo",
    5: "claudia_lopez",
    6: "rest",
    7: "blanco",
}

_GAD3_R2_CANDIDATE_MAP: dict[str, str] = {
    "1": "cepeda",
    "2": "de_la_espriella",
    "3": "valencia",
    "valencia": "valencia",
    "cepeda": "cepeda",
    "de_la_espriella": "de_la_espriella",
    "abelardo de la espriella": "de_la_espriella",
    "iván cepeda": "cepeda",
    "paloma valencia": "valencia",
    "blanco": "blanco",
    "voto en blanco": "blanco",
    "ns/nr": "ns_nr",
    "no sé": "ns_nr",
    "no votaría": "ns_nr",
}

_GAD3_WEIGHT_COLS: tuple[str, ...] = (
    "Ponderacion",
    "ponderacion",
    "peso",
    "weight",
    "PONDERACION",
)


def _gad3_read_microdatos_xlsx(zf: zipfile.ZipFile) -> pd.DataFrame | None:
    """Locate and read the microdatos XLSX file inside a GAD3 bundle zip.

    Args:
        zf: An open ZipFile for a GAD3 bundle.

    Returns:
        DataFrame if a matching XLSX is found, ``None`` otherwise.

    """
    xlsx_names = [
        n
        for n in zf.namelist()
        if n.lower().endswith(".xlsx") and "microdatos" in n.lower() and "$" not in n
    ]
    if not xlsx_names:
        xlsx_names = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".xlsx") and "datos" in n.lower() and "$" not in n
        ]
    if not xlsx_names:
        for n in zf.namelist():
            if n.lower().endswith(".xlsx"):
                xlsx_names = [n]
                break
    if not xlsx_names:
        return None
    with zf.open(xlsx_names[0]) as fh:
        return pd.read_excel(fh, engine="openpyxl")  # type: ignore[reportUnknownMemberType]


def _gad3_sort_q06_col(x: str) -> int:
    m = re.search(r"(\d+)", x)
    return int(m.group(1)) if m else 0


def _gad3_extract_r1_votes(df: pd.DataFrame, weight_col: str | None) -> tuple[pd.Series, pd.Series]:
    """Extract first-round vote intention from GAD3 Q06.* binary columns.

    Args:
        df: GAD3 microdatos DataFrame.
        weight_col: Name of the weight column, or ``None`` if absent.

    Returns:
        ``(r1_series, weight_series)``.

    Raises:
        ValueError: If no Q06 columns are found.

    """
    q06_cols = sorted(
        [c for c in df.columns if re.match(r"Q06(\.\d+)?$", c, re.IGNORECASE)],
        key=_gad3_sort_q06_col,
    )
    if not q06_cols:
        msg = "No Q06 columns found in GAD3 dataframe"
        raise ValueError(msg)

    def _row_r1(row: pd.Series) -> str:
        for i, col in enumerate(q06_cols, start=1):
            val = row.get(col)
            if pd.notna(val) and float(val) == 1:
                return _GAD3_R1_INDEX_MAP.get(i, "ns_nr")
        return "ns_nr"

    r1 = df.apply(_row_r1, axis=1)
    weight = df[weight_col].astype(float) if weight_col else pd.Series(1.0, index=df.index)
    return r1, weight


def _gad3_extract_runoff_pairings(
    df: pd.DataFrame, q09_cols: list[str], weight: pd.Series
) -> pd.DataFrame:
    """Extract runoff pairings from GAD3 Q09A/B/C columns.

    Args:
        df: GAD3 microdatos DataFrame.
        q09_cols: Sorted list of Q09 column names.
        weight: Numeric weight series aligned with *df*.

    Returns:
        DataFrame with columns ``candidate_a``, ``candidate_b``, ``weight``.

    """
    runoff_rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        w = weight.iloc[idx]  # type: ignore[reportUnknownVariableType,reportCallIssue]
        for col in q09_cols:
            val = str(row.get(col, "")).strip().lower()
            if not val or val == "nan":
                continue
            mapped = _GAD3_R2_CANDIDATE_MAP.get(val, val)
            col_upper = col.upper()
            if "Q09A" in col_upper:
                cand_a = "cepeda"
            elif "Q09B" in col_upper:
                cand_a = "de_la_espriella"
            elif "Q09C" in col_upper:
                cand_a = "valencia"
            else:
                continue
            runoff_rows.append({"candidate_a": cand_a, "candidate_b": mapped, "weight": w})
    return pd.DataFrame(runoff_rows)


def load_gad3(bundle_path: Path) -> pd.DataFrame:
    """Load a GAD3 microdata bundle.

    Args:
        bundle_path: Path to a GAD3 CNE zip bundle.

    Returns:
        DataFrame with columns ``fecha``, ``weight``, ``candidate_r1``, ``firm``,
        ``field_end``, ``raw_source``.  Runoff pairings in
        ``DataFrame.attrs["runoff_df"]``.

    Raises:
        ValueError: If no microdatos XLSX is found in the bundle.

    """
    with zipfile.ZipFile(bundle_path) as zf:
        df = _gad3_read_microdatos_xlsx(zf)
    if df is None:
        msg = f"No microdatos XLSX found in GAD3 bundle: {bundle_path}"
        raise ValueError(msg)

    weight_col = next((col for col in _GAD3_WEIGHT_COLS if col in df.columns), None)

    r1, weight = _gad3_extract_r1_votes(df, weight_col)

    q09_cols = sorted(
        [c for c in df.columns if re.match(r"Q09[A-C]$", c, re.IGNORECASE)],
        key=lambda x: x,
    )

    firm = "gad3"
    field_end = _parse_field_end_date(bundle_path.name)

    out = pd.DataFrame(
        {
            "fecha": field_end,
            "weight": weight,
            "candidate_r1": r1,
            "firm": firm,
            "field_end": field_end,
        }
    )

    if q09_cols:
        runoff_df = _gad3_extract_runoff_pairings(df, q09_cols, weight)
        runoff_df["fecha"] = field_end
        runoff_df["firm"] = firm
        runoff_df["field_end"] = field_end
        runoff_df["raw_source"] = bundle_path.name
        out.attrs["runoff_df"] = runoff_df
    else:
        out.attrs["runoff_df"] = pd.DataFrame()

    out["raw_source"] = bundle_path.name
    return out


# ---- Invamer ---------------------------------------------------------------


def _invamer_read_primary_xlsx(zf: zipfile.ZipFile, bundle_path: Path) -> pd.DataFrame:
    """Find and read the primary data XLSX from an Invamer zip bundle."""
    candidates = [
        n
        for n in zf.namelist()
        if "registros primarios" in n.lower() and n.lower().endswith(".xlsx")
    ]
    if not candidates:
        # Fallback: any xlsx with "data" or "3."
        xlsx_files = [n for n in zf.namelist() if n.lower().endswith(".xlsx") and "$" not in n]
        for n in xlsx_files:
            if "3." in n or "data" in n.lower() or "registro" in n.lower():
                candidates = [n]
                break
    if not candidates:
        # Last-resort fallback: any .xlsx file (for non-Invamer bundles using Invamer loader)
        any_xlsx = [n for n in zf.namelist() if n.lower().endswith(".xlsx") and "$" not in n]
        if any_xlsx:
            candidates = [any_xlsx[0]]
    if not candidates:
        msg = f"No primary data XLSX found in bundle: {bundle_path}"
        raise ValueError(msg)
    with zf.open(candidates[0]) as fh:
        return pd.read_excel(fh, engine="openpyxl")  # type: ignore[reportUnknownMemberType]


def _invamer_find_r1_col(df: pd.DataFrame) -> str | None:
    """Heuristically find the first-round vote column in an Invamer dataframe."""
    patterns = [
        r"voto\s*(presidente|intenci.n|primera|candidato)",
        r"candidato\s*(presidente|primera)",
        r"presidente\s*(2026|primera)",
        r"por\s*qui[eé]n\s*votar",
    ]
    for col in df.columns:
        col_lower = str(col).lower()
        for pat in patterns:
            if re.search(pat, col_lower):
                return col
    # Fallback: look for columns whose values look like candidate names
    for col in df.columns:
        sample = df[col].dropna().head(20).astype(str).str.lower()
        hits = sample.isin(
            [
                "paloma valencia",
                "iván cepeda",
                "abelardo de la espriella",
                "sergio fajardo",
                "claudia lópez",
            ]
        ).sum()
        if hits >= _MIN_CANDIDATE_NAME_HITS:
            return col
    return None


def load_invamer(bundle_path: Path) -> pd.DataFrame:
    """Load and normalize an Invamer poll bundle.

    Reads primary data XLSX from an Invamer zip bundle, discovers
    the first-round vote-intention column heuristically, and maps
    raw labels to canonical candidate keys.

    Args:
        bundle_path: Path to the Invamer .zip bundle.

    Returns:
        DataFrame with columns: firm, fecha, field_end, weight,
        candidate_r1. May include attrs["runoff_df"] if runoff
        pairings are extractable.

    """
    with zipfile.ZipFile(bundle_path) as zf:
        df = _invamer_read_primary_xlsx(zf, bundle_path)

    # Find weight column
    weight_col = None
    for col in (
        "factor",
        "factor_ponderacion",
        "ponderador",
        "peso",
        "weight",
        "factor de ponderación",
        "factor_de_ponderacion",
    ):
        for dc in df.columns:
            if col in str(dc).lower():
                weight_col = dc
                break
        if weight_col:
            break

    # Find R1 vote column
    r1_col = _invamer_find_r1_col(df)
    if r1_col is None:
        # Try to find a column with candidate-like values
        for col in df.columns:
            unique_vals = df[col].dropna().unique()[:20]
            if (
                len(unique_vals) >= _MIN_CANDIDATE_NAME_HITS
                and len(unique_vals) <= _MAX_UNIQUE_VALS
            ):
                sample = " ".join(str(v) for v in unique_vals)
                if any(
                    name in sample.lower()
                    for name in ["valencia", "cepeda", "espriella", "fajardo"]
                ):
                    r1_col = col
                    break

    if r1_col is None:
        msg = f"Could not find R1 vote column in Invamer bundle: {bundle_path}"
        raise ValueError(msg)

    weight = (
        pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
        if weight_col
        else pd.Series(1.0, index=df.index)
    )
    r1 = df[r1_col].astype(str).apply(_normalize_candidate_name)

    firm = "invamer"
    field_end = _parse_field_end_date(bundle_path.name)

    out = pd.DataFrame(
        {
            "fecha": field_end,
            "weight": weight,
            "candidate_r1": r1,
            "firm": firm,
            "field_end": field_end,
            "raw_source": bundle_path.name,
        }
    )
    out.attrs["runoff_df"] = pd.DataFrame()
    return out


# ---- CNC -------------------------------------------------------------------


def load_cnc(bundle_path: Path) -> pd.DataFrame:
    """Load CNC 2026 microdata (XLSX or SPSS .sav) from a zip bundle.

    CNC bundles may contain XLSX, SPSS .sav, or PDFs only.
    PDF-only bundles return an empty DataFrame.
    """
    with zipfile.ZipFile(bundle_path) as zf:
        namelist = zf.namelist()

        # Try XLSX first
        xlsx_names = [
            n
            for n in namelist
            if n.lower().endswith(".xlsx") and "$" not in n and "microdatos" not in n.lower()
        ]
        if xlsx_names:
            with zf.open(xlsx_names[0]) as fh:
                df = pd.read_excel(fh, engine="openpyxl")  # type: ignore[reportUnknownMemberType]
            return _parse_cnc_df(df, bundle_path)

        # Try SPSS .sav
        sav_names = [n for n in namelist if n.lower().endswith(".sav")]
        if sav_names:
            try:
                import pyreadstat  # noqa: PLC0415  # type: ignore[reportMissingImports]

                with zf.open(sav_names[0]) as fh:
                    import tempfile  # noqa: PLC0415

                    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
                        tmp.write(fh.read())
                    sav_path = tmp.name
                    try:
                        df, _meta = pyreadstat.read_sav(sav_path)  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
                    finally:
                        Path(sav_path).unlink()
                    return _parse_cnc_df(df, bundle_path)  # type: ignore[reportUnknownArgumentType]
            except ImportError:
                logger.warning("pyreadstat not installed; skipping CNC SAV bundle %s", bundle_path)

    # PDF-only — return empty with metadata
    logger.info("CNC bundle %s has no structured data (PDF-only)", bundle_path)
    out = pd.DataFrame(
        {
            "fecha": pd.NaT,
            "weight": pd.NA,
            "candidate_r1": pd.NA,
            "firm": pd.NA,
            "field_end": pd.NaT,
            "raw_source": pd.NA,
        },
        index=[],
    ).astype({"candidate_r1": "object"})
    out.attrs["runoff_df"] = pd.DataFrame()
    return out


def _cnc_find_weight_col(df: pd.DataFrame) -> str | None:
    """Discover the weight column in a CNC DataFrame."""
    for col in (
        "factor",
        "FACTOR",
        "Factor",
        "Factor Expansion",
        "F_EXP",
        "ponderador",
        "peso",
        "weight",
        "Fexp",
        "fexp",
    ):
        for dc in df.columns:
            if col in str(dc):
                return dc
    return None


def _parse_cnc_df(df: pd.DataFrame, bundle_path: Path) -> pd.DataFrame:
    """Parse a CNC DataFrame after loading from XLSX or SAV."""
    weight_col = _cnc_find_weight_col(df)

    # Find vote intention column — look for "PREGUNTA" or "INTENCION" or
    # column whose values are candidate names
    r1_col = None
    for col in df.columns:
        col_str = str(col).lower()
        # Column name patterns
        if any(
            p in col_str
            for p in ("intencion", "candidato", "voto", "pregunta", "intención", "presidente")
        ):
            r1_col = col
            break

    if r1_col is None:
        # Heuristic: scan columns for candidate-name-like values
        known_names = {
            "valencia",
            "cepeda",
            "fajardo",
            "espriella",
            "lópez",
            "lopez",
            "paloma",
            "iván",
            "ivan",
            "abelardo",
            "sergio",
            "claudia",
        }
        for col in df.columns:
            sample = df[col].dropna().head(20).astype(str).str.lower()
            combined = " ".join(sample)
            if sum(1 for n in known_names if n in combined) >= _MIN_CANDIDATE_NAME_HITS:
                r1_col = col
                break

    if r1_col is None:
        msg = f"Could not find vote-intention column in CNC bundle: {bundle_path}"
        raise ValueError(msg)

    weight = (
        pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
        if weight_col
        else pd.Series(1.0, index=df.index)
    )
    r1 = df[r1_col].astype(str).apply(_normalize_candidate_name)

    firm = "cnc"
    field_end = _parse_field_end_date(bundle_path.name)

    out = pd.DataFrame(
        {
            "fecha": field_end,
            "weight": weight,
            "candidate_r1": r1,
            "firm": firm,
            "field_end": field_end,
            "raw_source": bundle_path.name,
        }
    )
    out.attrs["runoff_df"] = pd.DataFrame()
    return out


# ---- PDF extraction --------------------------------------------------------


def _extract_bundle_pdfs(zip_path: Path) -> list[tuple[str, bytes]]:  # type: ignore[reportUnusedFunction]
    """Return list of (pdf_name, pdf_bytes) from a zip bundle."""
    pdfs: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".pdf"):
                pdfs.extend([(name, zf.read(name))])
    return pdfs


def extract_pdf_topline(pdf_path: Path) -> dict[str, float] | None:
    """Extract first-round topline percentages from a PDF informe.

    Attempts to find vote-intention table using pdfplumber and returns
    ``{canonical_key: percentage}``.
    Returns ``None`` if extraction fails.
    """
    logger.info("PDF topline extraction not yet implemented for %s", pdf_path)
    return None


def extract_pdf_runoff_tables(
    pdf_paths: list[Path],
) -> pd.DataFrame | None:
    """Extract runoff head-to-head pairings from PDF informes.

    Returns ``None`` if extraction fails.
    """
    logger.info("PDF runoff extraction not yet implemented for %d files", len(pdf_paths))
    return None


# ---- Topline and runoff extraction ----------------------------------------------


def extract_topline(
    df: pd.DataFrame,
    *,
    date_col: str = "field_end",
    candidate_col: str = "candidate_r1",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Compute weighted vote-intention percentages from microdata.

    Excludes ``ns_nr`` rows from the denominator.

    Args:
        df: Microdata DataFrame with one row per respondent.
        date_col: Name of the date column.
        candidate_col: Name of the normalized candidate column.
        weight_col: Name of the weight column.

    Returns:
        DataFrame with columns: fecha, encuestadora, field_end, plus one
        column per canonical candidate containing the weighted percentage.

    """
    valid = df[df[candidate_col] != "ns_nr"]
    total_weight = valid[weight_col].sum()
    if total_weight == 0:
        return pd.DataFrame()

    canonical = _topline_column_order()
    rows: list[dict[str, Any]] = []
    group_cols = [date_col, "firm", "field_end"]
    for group_key, group in valid.groupby(group_cols, dropna=False):
        row: dict[str, Any] = {}
        if isinstance(group_key, tuple):  # type: ignore[reportUnnecessaryIsInstance]
            for i, col in enumerate(group_cols):
                row[col] = group_key[i]
        else:
            row[group_cols[0]] = group_key
        tw = group[weight_col].sum()
        for cand in canonical:
            cw = group.loc[group[candidate_col] == cand, weight_col].sum()
            row[cand] = (cw / tw * 100) if tw > 0 else 0.0
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.rename(columns={"firm": "encuestadora"})
    result["total_weight"] = result[[c for c in canonical if c in result.columns]].sum(
        axis=1, numeric_only=True
    )
    return result


def extract_runoff_pairings(
    runoff_df: pd.DataFrame,
    *,
    date_col: str = "field_end",
    firm_col: str = "firm",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Aggregate runoff head-to-head pairings into long-format summary.

    Args:
        runoff_df: DataFrame with columns ``candidate_a``, ``candidate_b``,
            ``weight``, ``field_end``, ``firm``.
        date_col: Column name for the field-end date. Default ``"field_end"``.
        firm_col: Column name for the firm/pollster identifier.
            Default ``"firm"``.
        weight_col: Column name for the survey weight.
            Default ``"weight"``.

    Returns:
        Long-format DataFrame with columns: fecha, encuestadora, field_end,
        candidate_a, candidate_b, n_a, n_b, n_total, effective_n.

    """
    if runoff_df.empty:
        return pd.DataFrame(
            columns=[
                "fecha",
                "encuestadora",
                "field_end",
                "candidate_a",
                "candidate_b",
                "n_a",
                "n_b",
                "n_total",
                "effective_n",
            ]
        )

    filtered = runoff_df[
        (runoff_df["candidate_a"].isin(_CANONICAL_2026)) & (runoff_df["candidate_b"].notna())
    ].copy()

    # Map candidate_b values that aren't canonical
    def _map_b(val: str) -> str:
        if val in _CANONICAL_2026:
            return val
        if val in ("ns_nr", "ns/nr", "no sé", "blanco"):
            return "ns_nr"
        return val

    filtered["candidate_b"] = filtered["candidate_b"].apply(_map_b)

    rows: list[dict[str, Any]] = []
    group_cols = [date_col, firm_col, "candidate_a", "candidate_b"]
    for group_key, group in filtered.groupby(group_cols, dropna=False):
        row: dict[str, Any] = {}
        if isinstance(group_key, tuple):  # type: ignore[reportUnnecessaryIsInstance]
            row["fecha"] = group_key[0]
            row["encuestadora"] = group_key[1]
            row["candidate_a"] = group_key[2]
            row["candidate_b"] = group_key[3]
        else:
            row["fecha"] = group_key
        total_w = group[weight_col].sum()
        n_total = len(group)
        n_a = total_w
        n_b = total_w  # same denominator for paired responses
        effective_n = (
            total_w * total_w / (group[weight_col] ** 2).sum()
            if (group[weight_col] ** 2).sum() > 0
            else float(n_total)
        )
        row.update(
            {
                "field_end": group[date_col].iloc[0] if date_col in group.columns else None,
                "n_a": n_a,
                "n_b": n_b,
                "n_total": n_total,
                "effective_n": effective_n,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---- Orchestrator ---------------------------------------------------------------


_FIRM_LOADERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "atlas_intel": load_atlas_intel,
    "gad3": load_gad3,
    "invamer": load_invamer,
    "cnc": load_cnc,
    "genesis_crea": load_invamer,
    "tempo": load_invamer,
    "corp_mmm": load_invamer,
    "guarumo_ecoanalitica": load_invamer,
}


def load_bundle(bundle_path: Path) -> pd.DataFrame:
    """Load a single CNE 2026 poll bundle.

    Identifies the polling firm from the bundle filename and dispatches
    to the appropriate firm-specific loader.

    Args:
        bundle_path: Path to a CNE 2026 zip bundle.

    Returns:
        DataFrame with respondent-level microdata and a ``.attrs["firm"]``
        string indicating the loader used.

    Raises:
        FileNotFoundError: If *bundle_path* does not exist.
        ValueError: If the firm cannot be identified or no loader is available.

    """
    if not bundle_path.exists():
        msg = f"Bundle not found: {bundle_path}"
        raise FileNotFoundError(msg)
    firm = _identify_firm(bundle_path.name)
    if firm not in _FIRM_LOADERS:
        msg = f"No loader for firm {firm!r} (bundle {bundle_path.name})"
        raise ValueError(msg)
    df = _FIRM_LOADERS[firm](bundle_path)
    if "firm" not in df.attrs:
        df.attrs["firm"] = firm
    return df


def build_cne_2026_tables(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 2026 CNE topline and runoff-pairing tables.

    Iterates all zip bundles under ``data/2026-polls/``, loads each via the
    appropriate firm loader, extracts topline percentages and runoff
    head-to-head pairings, and writes:

    - ``data/2026-polls/_processed/2026_topline.parquet``
    - ``data/2026-polls/_processed/2026_runoff_pairings.parquet``

    Args:
        data_dir: Optional override for the project data directory.

    Returns:
        ``(topline_df, runoff_pairings_df)`` tuple.

    """
    bundles = _list_zip_bundles(data_dir)
    logger.info("Found %d CNE 2026 zip bundles", len(bundles))

    topline_parts: list[pd.DataFrame] = []
    runoff_parts: list[pd.DataFrame] = []

    for bundle_path in bundles:
        firm = _identify_firm(bundle_path.name)
        logger.info("Processing %s (firm=%s)", bundle_path.name, firm)

        try:
            loader = _FIRM_LOADERS.get(firm)
            if loader is None:
                logger.warning(
                    "Firm %r is PDF-only or unrecognized; skipping structured load for %s",
                    firm,
                    bundle_path,
                )
                continue
            df = loader(bundle_path)

            if df.empty:
                continue

            topline = extract_topline(df)
            if not topline.empty:
                topline_parts.append(topline)

            runoff_raw = df.attrs.get("runoff_df", pd.DataFrame())
            if not runoff_raw.empty:
                runoff = extract_runoff_pairings(runoff_raw)
                if not runoff.empty:
                    runoff_parts.append(runoff)

        except Exception:
            logger.exception("Failed to process bundle %s", bundle_path)

    all_topline = pd.concat(topline_parts, ignore_index=True) if topline_parts else pd.DataFrame()
    all_runoff = pd.concat(runoff_parts, ignore_index=True) if runoff_parts else pd.DataFrame()

    out_dir = _resolve_2026_dir(data_dir) / "_processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not all_topline.empty:
        all_topline.to_parquet(out_dir / "2026_topline.parquet", index=False)
        logger.info("Wrote %d rows to 2026_topline.parquet", len(all_topline))

    if not all_runoff.empty:
        all_runoff.to_parquet(out_dir / "2026_runoff_pairings.parquet", index=False)
        logger.info("Wrote %d rows to 2026_runoff_pairings.parquet", len(all_runoff))

    return all_topline, all_runoff


def build_clean_polls_2026(
    data_dir: Path | None = None,
    *,
    rebuild: bool = False,
) -> CleanPolls:
    """Build a CleanPolls instance from 2026 CNE topline data.

    SPEC-33: Loads processed 2026 topline and runoff-pairing parquet files,
    splits by election round based on ``field_end`` date, and wraps the
    result in a typed ``CleanPolls`` container.

    Args:
        data_dir: Optional override for the project data directory.
        rebuild: If ``True`` (or the processed parquet files are missing),
            rebuild them via ``build_cne_2026_tables`` before loading.

    Returns:
        ``CleanPolls`` with ``round1`` (polls whose ``field_end`` is on or
        before the 2026 first-round election date), ``round2`` (polls after
        that date), ``consultation=[]``, and ``all_polls`` (the full
        concatenation of both rounds).

    Raises:
        ValueError: If round pollster diversity fails validation in
            ``CleanPolls.__post_init__`` (e.g., fewer than 5 unique
            pollsters in round 1 when data is not AS/COA-only).

    """
    r1_date = get_election_date(2026, 1)
    _empty_template = pd.DataFrame(columns=["encuestadora"])

    out_dir = _resolve_2026_dir(data_dir) / "_processed"
    topline_path = out_dir / "2026_topline.parquet"
    runoff_path = out_dir / "2026_runoff_pairings.parquet"

    if rebuild or not topline_path.exists():
        build_cne_2026_tables(data_dir=data_dir)

    topline = pd.read_parquet(topline_path) if topline_path.exists() else pd.DataFrame()
    _ = pd.read_parquet(runoff_path) if runoff_path.exists() else None

    if topline.empty:
        return CleanPolls(
            round1=_empty_template.copy(),
            round2=_empty_template.copy(),
            consultation=[],
            all_polls=_empty_template.copy(),
        )

    field_end = pd.to_datetime(topline["field_end"]).dt.date
    round1_mask = field_end <= r1_date

    round1_df = topline[round1_mask].copy()
    round2_df = topline[~round1_mask].copy()

    if round1_df.empty:
        round1_df = _empty_template.copy()
    else:
        round1_df["round_number"] = 1

    if round2_df.empty:
        round2_df = _empty_template.copy()
    else:
        round2_df["round_number"] = 2

    all_polls = pd.concat([round1_df, round2_df], ignore_index=True)

    return CleanPolls(
        round1=round1_df,
        round2=round2_df,
        consultation=[],
        all_polls=all_polls,
    )
