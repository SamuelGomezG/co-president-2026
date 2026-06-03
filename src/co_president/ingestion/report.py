"""SPEC-15: Fundamental data coverage report and legislative schema audit.

Generates ``data/fundamentals/COVERAGE.md`` and verifies CEDAE/MOE
legislative schema compatibility.  Acts as the gateway for all
subsequent demographics-ingestion SPECs.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING

import pandas as pd

from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "generate_coverage_markdown",
    "generate_coverage_report",
    "verify_legislative_schemas",
    "write_coverage_report",
]

logger = logging.getLogger(__name__)

_EXPECTED_MUNICIPALITIES = 1_122
_SOCIOECONOMIC_STUB_THRESHOLD = 5
_RISK_STUB_THRESHOLD = 15

_SOURCE_FILE_MAP: dict[str, str] = {
    "DIVIPOLA": "divipola_master.csv",
    "Historical results": "historical_results.csv",
    "Socioeconomic": "socioeconomic.csv",
    "Risk": "risk_factors.csv",
}


def generate_coverage_report(data_dir: Path | None = None) -> pd.DataFrame:
    """Audit every fundamental data source and return a coverage DataFrame.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns ``source``, ``file``, ``rows_expected``,
        ``rows_actual``, ``null_rate``, ``status``, ``source_of_values``,
        and ``notes``.

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)

    fundamentals = data_dir / "fundamentals"
    records: list[dict[str, object]] = []

    # DIVIPOLA
    divipola_df = _read_safe(fundamentals / "divipola_master.csv")
    municipios = (
        divipola_df["codigo_municipio"].nunique()
        if "codigo_municipio" in divipola_df.columns
        else 0
    )
    records.append(
        _coverage_row(
            source="DIVIPOLA",
            rows_expected=_EXPECTED_MUNICIPALITIES,
            df=divipola_df,
            row_note=f"{municipios} unique municipality codes" if municipios else "FILE MISSING",
        )
    )

    # Historical results
    historical_df = _read_safe(fundamentals / "historical_results.csv")
    has_placeholder = (
        historical_df["codigo_municipio"].eq("000NA").any()
        if "codigo_municipio" in historical_df.columns
        else False
    )
    years_count = historical_df["year"].nunique() if "year" in historical_df.columns else 0
    records.append(
        _coverage_row(
            source="Historical results",
            rows_expected=_EXPECTED_MUNICIPALITIES * 6 * 2,
            df=historical_df,
            row_note=f"{years_count} election years"
            + ("; CONTAINS 000NA PLACEHOLDER" if has_placeholder else ""),
        )
    )

    # Socioeconomic
    socioeconomic_df = _read_safe(fundamentals / "socioeconomic.csv")
    records.append(
        _coverage_row(
            source="Socioeconomic",
            rows_expected=_EXPECTED_MUNICIPALITIES,
            df=socioeconomic_df,
            stub_threshold=_SOCIOECONOMIC_STUB_THRESHOLD,
            row_note="Hardcoded fallback (3 rows only)"
            if len(socioeconomic_df) <= _SOCIOECONOMIC_STUB_THRESHOLD
            else "",
        )
    )

    # Risk
    risk_df = _read_safe(fundamentals / "risk_factors.csv")
    records.append(
        _coverage_row(
            source="Risk",
            rows_expected=_EXPECTED_MUNICIPALITIES,
            df=risk_df,
            stub_threshold=_RISK_STUB_THRESHOLD,
            row_note="Hardcoded fallback (10 rows only)"
            if len(risk_df) <= _RISK_STUB_THRESHOLD
            else "",
        )
    )

    # CEDAE raw directory listing (informational)
    records.append(_build_directory_row(data_dir))

    # MOE 2022 legislative files
    records.append(_build_moe_row(data_dir))

    # DANE NBI
    records.append(_build_nbi_row(data_dir))

    return pd.DataFrame(records)


def generate_coverage_markdown(data_dir: Path | None = None) -> str:
    """Generate the full ``COVERAGE.md`` markdown content.

    Args:
        data_dir: Root data directory.

    Returns:
        Markdown string ready to write to ``COVERAGE.md``.

    """
    df = generate_coverage_report(data_dir)
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Fundamentals Coverage",
        "",
        f"Generated: {now_str}",
        "",
        "| Source | File | Rows Expected | Rows Actual | Status | Notes |",
        "|---|---|---|---|---|---|",
    ]

    for _, row in df.iterrows():
        rows_exp = str(row.get("rows_expected", ""))
        rows_act = str(row.get("rows_actual", ""))
        status = str(row.get("status", ""))
        notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""
        source = str(row.get("source", ""))
        file_str = str(row.get("file", ""))
        lines.append(f"| {source} | {file_str} | {rows_exp} | {rows_act} | {status} | {notes} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Legend",
            "",
            "| Status | Meaning |",
            "|---|---|",
            "| ✅ FULL | Complete coverage (>= expected rows) |",
            "| ⚠️ PARTIAL | Partial coverage (< expected rows) |",
            "| ⚠️ STUB | Placeholder data (< threshold rows) |",
            "| ✅ READY | Files present on disk, not yet ingested |",
            "| ❌ MISSING | Source not found |",
            "| ⚠️ PLACEHOLDER | Contains 000NA sentinel rows (in notes) |",
            "",
            "---",
            "",
            "## Legislative Schema Compatibility",
            "",
        ]
    )

    schema_info = verify_legislative_schemas(data_dir)
    lines.append(f"- **CEDAE Cámara columns**: ``{schema_info.get('cedae_camara_columns', [])}``")
    lines.append(f"- **MOE Cámara columns**: ``{schema_info.get('moe_camara_columns', [])}``")
    lines.append(f"- **CEDAE Senado columns**: ``{schema_info.get('cedae_senado_columns', [])}``")
    lines.append(f"- **MOE Senado columns**: ``{schema_info.get('moe_senado_columns', [])}``")
    compat = schema_info.get("schemas_compatible", False)
    lines.append(f"- **Schemas compatible**: {'Yes' if compat else 'No (see notes)'}")
    notes = schema_info.get("notes", "")
    if notes:
        lines.append(f"- **Notes**: {notes}")

    lines.append("")
    return "\n".join(lines)


def write_coverage_report(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Write ``COVERAGE.md`` to the fundamentals directory.

    Args:
        data_dir: Root data directory.
        output_dir: Override output directory.  If ``None``, writes to
            ``data_dir / "fundamentals"``.

    Returns:
        Path to the written file.

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)
    if output_dir is None:
        output_dir = data_dir / "fundamentals"

    output_dir.mkdir(parents=True, exist_ok=True)
    content = generate_coverage_markdown(data_dir)
    dest = output_dir / "COVERAGE.md"
    dest.write_text(content, encoding="utf-8")
    logger.info("Coverage report written to %s", dest)
    return dest


def verify_legislative_schemas(data_dir: Path | None = None) -> dict[str, object]:
    """Compare CEDAE and MOE legislative column schemas for compatibility.

    Reads one CEDAE Cámara file, one CEDAE Senado file, and the MOE
    2022 legislative CSVs, then returns their column names and a
    compatibility assessment.

    Args:
        data_dir: Root data directory.

    Returns:
        Dict with keys ``cedae_camara_columns``, ``moe_camara_columns``,
        ``cedae_senado_columns``, ``moe_senado_columns``,
        ``schemas_compatible`` (bool), and ``notes`` (str).

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)

    result: dict[str, object] = {
        "cedae_camara_columns": [],
        "moe_camara_columns": [],
        "cedae_senado_columns": [],
        "moe_senado_columns": [],
        "schemas_compatible": False,
        "notes": "",
    }

    cedae_dir = data_dir / "raw" / "cedae"
    moe_dir = data_dir / "raw" / "MOE-2022-legislativas"

    _read_cedae_schema(result, cedae_dir, "camara")
    _read_cedae_schema(result, cedae_dir, "senado")
    _read_moe_schema(result, moe_dir, "camara")
    _read_moe_schema(result, moe_dir, "senado")

    _assess_schema_compatibility(result)

    return result


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _read_safe(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty DataFrame on failure."""
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"codigo_municipio": str, "year": str})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def _coverage_row(
    source: str,
    rows_expected: int,
    df: pd.DataFrame,
    stub_threshold: int | None = None,
    row_note: str = "",
) -> dict[str, object]:
    """Build a single coverage report row from a parsed DataFrame.

    The ``file`` name is derived from *source* via ``_SOURCE_FILE_MAP``.
    """
    file_name = _SOURCE_FILE_MAP.get(source, "unknown")
    if df.empty:
        status = "❌ MISSING"
        rows_actual = 0
        null_rate = "N/A"
        notes = "File not found or unparseable"
    else:
        rows_actual = len(df)
        total_cells = max(rows_actual * len(df.columns), 1)
        null_rate_val = df.isna().sum().sum() / total_cells
        null_rate = f"{null_rate_val:.1%}"

        if stub_threshold is not None and rows_actual <= stub_threshold:
            status = "⚠️ STUB"
            notes = row_note or _stub_note(source, rows_actual, stub_threshold)
        elif rows_actual < rows_expected:
            status = "⚠️ PARTIAL"
            notes = row_note or f"Only {rows_actual} of {rows_expected} rows"
        else:
            status = "✅ FULL"
            notes = row_note or ""

    source_map: dict[str, str] = {
        "DIVIPOLA": "DANE DIVIPOLA",
        "Historical results": "CEDAE / Registraduria",
        "Socioeconomic": "DANE",
        "Risk": "MOE / UNODC",
    }
    return {
        "source": source,
        "file": file_name,
        "rows_expected": rows_expected,
        "rows_actual": rows_actual,
        "null_rate": null_rate,
        "status": status,
        "notes": notes,
        "source_of_values": source_map.get(source, ""),
    }


def _stub_note(source: str, rows_actual: int, threshold: int) -> str:
    """Return a human-readable note explaining a stub detection."""
    return f"STUB DETECTED: {source} has only {rows_actual} rows (< {threshold})"


def _build_directory_row(data_dir: Path) -> dict[str, object]:
    """Build a coverage row for the CEDAE raw data directory."""
    cedae_dir = data_dir / "raw" / "cedae"
    cedae_files = (
        list(cedae_dir.glob("*_camara.dta.csv.gz"))
        + list(cedae_dir.glob("*_senado.dta.csv.gz"))
        + list(cedae_dir.glob("*_presidencia*.dta.csv.gz"))
    )
    return {
        "source": "CEDAE (raw)",
        "file": "data/raw/cedae/",
        "rows_expected": "18 files (2002-2018)",
        "rows_actual": f"{len(cedae_files)} files",
        "null_rate": "N/A",
        "status": "✅ READY" if cedae_files else "❌ MISSING",
        "source_of_values": "datasketch.co S3",
        "notes": "Legislative + presidential; primary source for historical results",
    }


def _build_moe_row(data_dir: Path) -> dict[str, object]:
    """Build a coverage row for the MOE 2022 legislative directory."""
    moe_dir = data_dir / "raw" / "MOE-2022-legislativas"
    moe_files = list(moe_dir.glob("*.csv"))
    return {
        "source": "MOE 2022 (legislative)",
        "file": "data/raw/MOE-2022-legislativas/",
        "rows_expected": "Camara + Senado",
        "rows_actual": f"{len(moe_files)} files",
        "null_rate": "N/A",
        "status": "✅ READY" if moe_files else "❌ MISSING",
        "source_of_values": "MOE Colombia",
        "notes": "Camara 13K rows, Senado 20K rows",
    }


def _build_nbi_row(data_dir: Path) -> dict[str, object]:
    """Build a coverage row for the DANE NBI data directory."""
    nbi_dir = data_dir / "raw" / "DANE-NBI"
    nbi_files = list(nbi_dir.glob("*.xlsx"))
    return {
        "source": "DANE NBI (2018)",
        "file": "data/raw/DANE-NBI/",
        "rows_expected": "Complete municipal NBI",
        "rows_actual": f"{len(nbi_files)} files",
        "null_rate": "N/A",
        "status": "✅ READY" if nbi_files else "❌ MISSING",
        "source_of_values": "DANE CNPV 2018",
        "notes": "Primary poverty feature (0% imputation needed)",
    }


def _read_cedae_schema(
    result: dict[str, object],
    cedae_dir: Path,
    chamber: str,
) -> None:
    """Read CEDAE legislative column names into *result*."""
    key = f"cedae_{chamber}_columns"
    files = sorted(cedae_dir.glob(f"*_{chamber}.dta.csv.gz"))
    if not files:
        return
    try:
        df = pd.read_csv(files[0], compression="gzip", encoding="latin-1", nrows=0)
        result[key] = list(df.columns)  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read CEDAE %s file: %s", chamber, exc)


def _read_moe_schema(
    result: dict[str, object],
    moe_dir: Path,
    chamber: str,
) -> None:
    """Read MOE 2022 legislative column names into *result*."""
    file_names = {
        "camara": "moe_camara_territorial_2022.csv",
        "senado": "moe_senado_nacional_2022.csv",
    }
    key = f"moe_{chamber}_columns"
    path = moe_dir / file_names[chamber]
    if not path.exists():
        return
    try:
        df = pd.read_csv(path, encoding="latin-1", nrows=0)
        result[key] = list(df.columns)  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read MOE %s file: %s", chamber, exc)


def _assess_schema_compatibility(result: dict[str, object]) -> None:
    """Check CEDAE and MOE Cámara schemas for shared columns."""
    cedae_raw: list[str] = result.get("cedae_camara_columns", [])  # type: ignore[assignment]
    moe_raw: list[str] = result.get("moe_camara_columns", [])  # type: ignore[assignment]
    cedae_cols: set[str] = set(cedae_raw)
    moe_cols: set[str] = set(moe_raw)

    if not cedae_cols or not moe_cols:
        result["notes"] = "Could not read one or both Cámara data sources"
        return

    shared = cedae_cols & moe_cols
    has_muni_code = "codmpio" in shared
    has_votes = "votos" in shared
    result["schemas_compatible"] = has_muni_code and has_votes

    notes_parts: list[str] = []
    if not has_muni_code:
        notes_parts.append("Missing shared municipality code column")
    if not has_votes:
        notes_parts.append("Missing shared vote column")

    cedae_party = "codigo_partido" in cedae_cols
    moe_party = "codparti" in moe_cols or "nomparti" in moe_cols
    if cedae_party and moe_party:
        notes_parts.append(
            "Party code crosswalk needed (cedae=codigo_partido, moe=codparti/nomparti)"
        )

    result["notes"] = "; ".join(notes_parts) if notes_parts else ""
