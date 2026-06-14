"""SPEC-41: TerriData fiscal indicator extraction.

Loads year-specific fiscal indicators from DNP TerriData (``Finanzas
publicas`` dimension) for benchmark use.  Caches to parquet on first load.
"""

from __future__ import annotations

import logging
from pathlib import Path
import zipfile

import pandas as pd  # type: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_TERRIDATA_ZIP = _PROJECT_ROOT / "data/raw/TerriData/TerriData_Finanzas_Publicas.xlsx.zip"
_TERRIDATA_TXT_ZIP = _PROJECT_ROOT / "data/raw/TerriData/TerriData.txt.zip"
_CACHE_PARQUET = _PROJECT_ROOT / "data/processed/terridata_fiscal.parquet"

_ELECTION_YEARS: tuple[int, ...] = (2002, 2006, 2010, 2014, 2018, 2022)
_MUNI_CODE_LEN: int = 5
_YEAR_SPLIT_PARTS: int = 2
_TXT_FIELD_COUNT: int = 14

_TXT_DIMENSIONS: tuple[str, ...] = (
    "Economía",
    "Pobreza",
    "Medición de desempeño municipal",
    "Educación",
    "Salud",
    "Mercado laboral",
)

_FISCAL_INDICATORS: tuple[str, ...] = (
    "Ingresos totales",
    "Ingresos corrientes",
    "Ingresos tributarios",
    "Ingresos no tributarios",
    "Transferencias de los ingresos corrientes",
    "Ingresos de capital",
    "Gastos totales",
    "Gastos corrientes",
    "Funcionamiento",
    "Intereses de deuda pública",
    "Gastos de capital (Inversión)",
    "Déficit o ahorro corriente",
    "Formación bruta de capital fijo",
    "Salud",
    "Educación",
    "Agua potable",
)


def _parse_colombian_number(value: object) -> float:
    """Parse Colombian-format number (``'4.743,37'`` -> ``4743.37``).

    Args:
        value: Raw string or NaN from Excel.

    Returns:
        Float value, or NaN if parsing fails.

    """
    if pd.isna(value):  # type: ignore[reportUnknownArgumentType, reportCallIssue]
        return float("nan")
    s = str(value).strip()
    if not s:
        return float("nan")
    try:
        s = s.replace(".", "").replace(",", ".")
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def _load_terridata_excel() -> pd.DataFrame:
    """Load & concatenate both TerriData Excel sheets.

    Returns:
        DataFrame with raw TerriData rows (unfiltered).

    """
    if not _TERRIDATA_ZIP.exists():
        msg = f"TerriData zip not found: {_TERRIDATA_ZIP.resolve()}"
        raise FileNotFoundError(msg)

    with zipfile.ZipFile(str(_TERRIDATA_ZIP)) as zf:
        xl_path = "TerriData_Dim7.xlsx"
        dfs: list[pd.DataFrame] = []
        for sheet in ("Hoja01", "Hoja02"):
            with zf.open(xl_path) as f:
                dfs.append(pd.read_excel(f, sheet_name=sheet))  # type: ignore[reportUnknownMemberType]

    df_concat = pd.concat(dfs, ignore_index=True)
    logger.info("TerriData loaded: %d rows", len(df_concat))
    return df_concat


def _normalise_indicator(name: str) -> str:
    """Normalise Spanish indicator name to snake_case ASCII."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("i", "i")
        .replace("o", "o")
        .replace("e", "e")
        .replace("a", "a")
        .replace("u", "u")
        .replace("n", "n")
        .replace("/", "_")
        .replace("-", "_")
    )


def _build_fiscal_cache() -> pd.DataFrame:
    """Extract election-year fiscal indicators and cache to parquet."""
    df = _load_terridata_excel()

    df["Año"] = pd.to_numeric(df["Año"], errors="coerce")
    df = df[df["Año"].isin(_ELECTION_YEARS)].copy()

    code_str = df["Código Entidad"].astype(str).str.strip()
    df = df[code_str.str.len() == _MUNI_CODE_LEN].copy()

    df["codigo_municipio"] = pd.to_numeric(code_str, errors="coerce").astype(int)

    df["valor"] = df["Dato Numérico"].apply(_parse_colombian_number)

    df = df[df["Indicador"].isin(_FISCAL_INDICATORS)].copy()
    df = df.dropna(subset=["valor"])

    pivot = df.pivot_table(
        index="codigo_municipio",
        columns=["Indicador", "Año"],
        values="valor",
        aggfunc="first",
    )

    all_cols = pd.MultiIndex.from_product(
        [sorted(df["Indicador"].unique()), sorted(df["Año"].unique())],
        names=["Indicador", "Año"],
    )
    pivot = pivot.reindex(columns=all_cols)

    pivot.columns = [
        f"fiscal_{_normalise_indicator(str(col))}_{int(year)}" for col, year in pivot.columns
    ]
    pivot = pivot.reset_index()

    _CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_parquet(str(_CACHE_PARQUET))
    logger.info("Fiscal cache saved: %d rows -> %s", len(pivot), _CACHE_PARQUET)
    return pivot


def _build_txt_cache() -> pd.DataFrame:  # noqa: C901
    """Stream-extract additional dimensions from TerriData.txt.zip.

    Returns DataFrame indexed by ``codigo_municipio`` with columns named
    ``txt_{indicator}_{year}``.
    """
    if not _TERRIDATA_TXT_ZIP.exists():
        logger.warning("TerriData.txt.zip not found; skipping txt dimensions")
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(str(_TERRIDATA_TXT_ZIP)) as zf_txt, zf_txt.open("TerriData.txt") as txt_f:
        for line_bytes in txt_f:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            parts = line.split("|")
            if len(parts) < _TXT_FIELD_COUNT:
                continue
            dim = parts[4]
            if dim not in _TXT_DIMENSIONS:
                continue
            year_str = parts[10].strip()
            if year_str not in {str(y) for y in _ELECTION_YEARS}:
                continue
            code = parts[2].strip()
            if len(code) != _MUNI_CODE_LEN:
                continue
            if code.endswith("000"):
                continue
            val = parts[8].strip()
            if not val:
                continue
            parsed = _parse_colombian_number(val)
            if pd.isna(parsed):  # type: ignore[reportUnknownArgumentType, reportCallIssue]
                continue
            indicator = parts[6].strip()
            rows.append(
                {
                    "codigo_municipio": int(code),
                    "indicador": indicator,
                    "año": int(year_str),
                    "valor": parsed,
                }
            )

    if not rows:
        return pd.DataFrame()

    df_txt = pd.DataFrame(rows)
    logger.info("Txt streamed: %d rows across %d dimensions", len(df_txt), len(_TXT_DIMENSIONS))

    pivot = df_txt.pivot_table(
        index="codigo_municipio",
        columns=["indicador", "año"],
        values="valor",
        aggfunc="first",
    )
    all_cols = pd.MultiIndex.from_product(
        [sorted(df_txt["indicador"].unique()), sorted(df_txt["año"].unique())],
        names=["indicador", "año"],
    )
    result = pivot.reindex(columns=all_cols)
    result.columns = [
        f"txt_{_normalise_indicator(str(col))}_{int(year)}" for col, year in result.columns
    ]
    return result.reset_index()


def load_fiscal_features() -> pd.DataFrame:
    """Load cached TerriData fiscal + txt features (build cache if needed).

    Returns:
        DataFrame indexed by ``codigo_municipio`` with columns named
        ``fiscal_{indicator}_{year}`` and ``txt_{indicator}_{year}``.

    Raises:
        FileNotFoundError: If the fiscal Excel zip (``TerriData.zip``)
            is missing.
        zipfile.BadZipFile: If either zip file (fiscal or txt) is
            corrupted.

    """
    if _CACHE_PARQUET.exists():
        df_cache = pd.read_parquet(str(_CACHE_PARQUET))
        logger.info("Fiscal features loaded from cache: %d rows x %d cols", *df_cache.shape)
    else:
        logger.info("Building TerriData caches...")
        df_fiscal = _build_fiscal_cache()
        df_txt = _build_txt_cache()

        if not df_txt.empty:
            df_cache = pd.concat(
                [
                    df_fiscal.set_index("codigo_municipio"),
                    df_txt.set_index("codigo_municipio"),
                ],
                axis=1,
            ).reset_index()
        else:
            df_cache = df_fiscal

        _CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df_cache.to_parquet(str(_CACHE_PARQUET))
        logger.info(
            "Unified cache saved: %d rows x %d cols -> %s",
            *df_cache.shape,
            _CACHE_PARQUET,
        )

    df_cache = df_cache.set_index("codigo_municipio")
    df_cache.index = df_cache.index.astype(int)

    n_rows = len(df_cache)
    min_valid = max(n_rows // 2, 500)
    valid = df_cache.notna().sum() >= min_valid
    dropped = (~valid).sum()
    if dropped:
        logger.info("Dropping %d sparse columns (< %d non-null)", dropped, min_valid)
        df_cache = df_cache.loc[:, valid]

    logger.info("Final fiscal features: %d rows x %d cols", *df_cache.shape)
    return df_cache


def get_fiscal_years(df_fiscal: pd.DataFrame) -> list[int]:
    """Get available years from fiscal feature columns.

    Args:
        df_fiscal: Fiscal features DataFrame (from ``load_fiscal_features()``).

    Returns:
        Sorted list of unique years present in column suffixes.

    """
    years: set[int] = set()
    for col in df_fiscal.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == _YEAR_SPLIT_PARTS and parts[1].isdigit():
            years.add(int(parts[1]))
    return sorted(years)
