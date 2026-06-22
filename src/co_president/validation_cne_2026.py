"""SPEC-32: Cross-validate 2026 CNE toplines against La Silla Vacia published data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from co_president.data_cne_2026 import CANDIDATE_KEY_MAP_2026
from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# La Silla Vacía firm-name to CNE firm-key mapping
# ---------------------------------------------------------------------------

_LSVC_FIRM_MAP: dict[str, str] = {
    "CNC": "cnc",
    "GAD3": "gad3",
    "Invamer": "invamer",
    "Guarumo": "guarumo_ecoanalitica",
    "Atlas Intel": "atlas_intel",
    "Génesis Crea": "genesis_crea",
    "TEMPO CONSULTORIA EMPRESARIAL S.A.S.": "tempo",
    "Corporación Miguel Maldonado Manjarrez": "corp_mmm",
}

_HEADER_FILTER_VALUES = frozenset({"Candidato", "Encuestadora"})


def _normalize_lsvc_candidate(name: str) -> str | None:
    """Map an LSVC candidate name to a canonical key, returning None for headers."""
    stripped = name.strip()
    if not stripped or stripped in _HEADER_FILTER_VALUES:
        return None
    if stripped in CANDIDATE_KEY_MAP_2026:
        return CANDIDATE_KEY_MAP_2026[stripped]
    lower_map = {k.lower(): v for k, v in CANDIDATE_KEY_MAP_2026.items()}
    key = stripped.lower()
    if key in lower_map:
        return lower_map[key]
    msg = f"Unmapped LSVC candidate label: {stripped!r}"
    raise ValueError(msg)


def _normalize_lsvc_firm(name: str) -> str:
    """Normalize an LSVC pollster name to the CNE firm key."""
    stripped = name.strip()
    if stripped in _LSVC_FIRM_MAP:
        return _LSVC_FIRM_MAP[stripped]
    lower_map = {k.lower(): v for k, v in _LSVC_FIRM_MAP.items()}
    key = stripped.lower()
    if key in lower_map:
        return lower_map[key]
    msg = f"Unmapped LSVC firm name: {stripped!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

_BASE_DIR_2026 = "2026-polls"
_LSVC_DIR = "silla_vacia_ponderador"


def _resolve_lsvc_dir(data_dir: Path | None = None) -> Path:
    root = resolve_data_dir(data_dir)
    return root / _BASE_DIR_2026 / _LSVC_DIR


def load_lsvc_detalle(data_dir: Path | None = None) -> pd.DataFrame:
    """Load La Silla Vacia ``encuestas_detalle.csv`` with normalized names.

    Args:
        data_dir: Optional override for the project data directory.

    Returns:
        DataFrame with columns: ``fecha``, ``encuestadora``, ``candidate_key``,
        ``porcentaje``.  Rows where the candidate column equals the header value
        ``"Candidato"`` are silently dropped.

    Raises:
        FileNotFoundError: If ``encuestas_detalle.csv`` does not exist.

    """
    lsvc_dir = _resolve_lsvc_dir(data_dir)
    detalle_path = lsvc_dir / "encuestas_detalle.csv"
    if not detalle_path.exists():
        msg = f"LSVC detalle file not found: {detalle_path}"
        raise FileNotFoundError(msg)

    raw = pd.read_csv(detalle_path)
    raw.columns = [c.strip() for c in raw.columns]

    fecha_col = "Fecha_Fin_Campo"
    encuestadora_col = "Encuestadora"
    candidato_col = "Candidato"
    porcentaje_col = "Porcentaje"

    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(raw[fecha_col], errors="coerce").dt.date
    out["encuestadora"] = raw[encuestadora_col].apply(_normalize_lsvc_firm)
    out["porcentaje"] = pd.to_numeric(raw[porcentaje_col], errors="coerce")
    candidate_keys = raw[candidato_col].apply(_normalize_lsvc_candidate)
    out["candidate_key"] = candidate_keys

    out = out.dropna(subset=["candidate_key"])
    return out.reset_index(drop=True)


def load_lsvc_ponderacion(data_dir: Path | None = None) -> pd.DataFrame:
    """Load La Silla Vacia ``ponderacion.csv`` with normalized candidate names.

    Args:
        data_dir: Optional override for the project data directory.

    Returns:
        DataFrame with columns: ``fecha``, ``candidate_key``, ``ponderado``,
        ``ponderado_min``, ``ponderado_max``.  Rows where the candidate column
        equals the header value ``"Candidato"`` are silently dropped.

    Raises:
        FileNotFoundError: If ``ponderacion.csv`` does not exist.

    """
    lsvc_dir = _resolve_lsvc_dir(data_dir)
    ponderacion_path = lsvc_dir / "ponderacion.csv"
    if not ponderacion_path.exists():
        msg = f"LSVC ponderacion file not found: {ponderacion_path}"
        raise FileNotFoundError(msg)

    raw = pd.read_csv(ponderacion_path)
    raw.columns = [c.strip() for c in raw.columns]

    fecha_col = "Fecha"
    candidato_col = "Candidato"

    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(raw[fecha_col], errors="coerce").dt.date
    out["ponderado"] = pd.to_numeric(raw["Ponderado"], errors="coerce")
    out["ponderado_min"] = pd.to_numeric(raw["Ponderado_Min"], errors="coerce")
    out["ponderado_max"] = pd.to_numeric(raw["Ponderado_Max"], errors="coerce")
    candidate_keys = raw[candidato_col].apply(_normalize_lsvc_candidate)
    out["candidate_key"] = candidate_keys

    out = out.dropna(subset=["candidate_key"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def compare_toplines(
    cne_df: pd.DataFrame,
    lsvc_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compare CNE topline percentages against La Silla Vacia published data.

    Joins on ``(fecha, encuestadora, candidate_key)``, computes error metrics
    per candidate and per polling firm.

    Args:
        cne_df: CNE topline DataFrame as produced by
            :func:`~co_president.data_cne_2026.build_cne_2026_tables` or
            equivalent.  Must contain one column per canonical candidate key
            (e.g. ``cepeda``, ``fajardo``, …) plus ``fecha`` and
            ``encuestadora``.
        lsvc_df: LSVC detalle DataFrame as returned by
            :func:`load_lsvc_detalle`.  Columns: ``fecha``, ``encuestadora``,
            ``candidate_key``, ``porcentaje``.

    Returns:
        Dictionary with keys:

        * ``paired_rows`` (int): Number of matched (CNE, LSVC) pairs.
        * ``mae_candidate`` (DataFrame): MAE per candidate key.
        * ``rmse_candidate`` (DataFrame): RMSE per candidate key.
        * ``max_error_candidate`` (DataFrame): Max absolute error per candidate.
        * ``mae_firm`` (DataFrame): MAE per encuestadora.
        * ``rmse_firm`` (DataFrame): RMSE per encuestadora.
        * ``overall_mae`` (float): Pooled MAE.
        * ``overall_rmse`` (float): Pooled RMSE.
        * ``details`` (DataFrame): Full per-row comparison with columns
          ``fecha``, ``encuestadora``, ``candidate_key``, ``cne_pct``,
          ``lsvc_pct``, ``abs_error``.

    """
    candidate_keys = [
        c for c in cne_df.columns if c not in ("fecha", "encuestadora", "field_end", "total_weight")
    ]

    cne_long_rows: list[dict[str, Any]] = []
    for _, row in cne_df.iterrows():
        fecha = row["fecha"]
        firm = row["encuestadora"]
        for ck in candidate_keys:
            val = row.get(ck)
            if pd.notna(val):
                cne_long_rows.append(
                    {"fecha": fecha, "encuestadora": firm, "candidate_key": ck, "cne_pct": val}
                )
    cne_long = pd.DataFrame(cne_long_rows)

    merged = cne_long.merge(
        lsvc_df.rename(columns={"porcentaje": "lsvc_pct"}),
        on=["fecha", "encuestadora", "candidate_key"],
        how="inner",
    )

    if merged.empty:
        return {
            "paired_rows": 0,
            "mae_candidate": pd.DataFrame(),
            "rmse_candidate": pd.DataFrame(),
            "max_error_candidate": pd.DataFrame(),
            "mae_firm": pd.DataFrame(),
            "rmse_firm": pd.DataFrame(),
            "overall_mae": float("nan"),
            "overall_rmse": float("nan"),
            "details": pd.DataFrame(),
        }

    merged["abs_error"] = (merged["cne_pct"] - merged["lsvc_pct"]).abs()
    merged["sq_error"] = merged["abs_error"] ** 2

    mae_candidate = merged.groupby("candidate_key")["abs_error"].mean().reset_index()
    mae_candidate.columns = ["candidate_key", "mae"]

    rmse_candidate = merged.groupby("candidate_key")["sq_error"].mean().pow(0.5).reset_index()
    rmse_candidate.columns = ["candidate_key", "rmse"]

    max_error_candidate = merged.groupby("candidate_key")["abs_error"].max().reset_index()
    max_error_candidate.columns = ["candidate_key", "max_error"]

    mae_firm = merged.groupby("encuestadora")["abs_error"].mean().reset_index()
    mae_firm.columns = ["encuestadora", "mae"]

    rmse_firm = merged.groupby("encuestadora")["sq_error"].mean().pow(0.5).reset_index()
    rmse_firm.columns = ["encuestadora", "rmse"]

    overall_mae = float(merged["abs_error"].mean())
    overall_rmse = float(merged["sq_error"].mean() ** 0.5)

    details = merged[["fecha", "encuestadora", "candidate_key", "cne_pct", "lsvc_pct", "abs_error"]]

    return {
        "paired_rows": len(merged),
        "mae_candidate": mae_candidate,
        "rmse_candidate": rmse_candidate,
        "max_error_candidate": max_error_candidate,
        "mae_firm": mae_firm,
        "rmse_firm": rmse_firm,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "details": details,
    }


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown-formatted table string."""
    if df.empty:
        return "_No data._\n"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join([" --- " for _ in cols]) + "|"
    rows: list[str] = [header, sep]
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def write_validation_report(
    comparison: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write a markdown validation report from a comparison result.

    Args:
        comparison: Dictionary returned by :func:`compare_toplines`.
        output_dir: Directory in which to write the report.

    Returns:
        Path to the written ``validation_cne_2026.md`` file.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_cne_2026.md"

    lines: list[str] = []
    lines.append("# CNE 2026 Topline Cross-Validation Report")
    lines.append("")
    lines.append("Comparison of CNE microdata toplines against La Silla Vacia published data.")
    lines.append("")

    paired = comparison["paired_rows"]
    lines.append(f"**Paired rows**: {paired}")
    lines.append("")

    if paired == 0:
        lines.append("No overlapping (fecha, encuestadora, candidate_key) combinations found.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    lines.append(f"**Overall MAE**: {comparison['overall_mae']:.4f}")
    lines.append(f"**Overall RMSE**: {comparison['overall_rmse']:.4f}")
    lines.append("")

    lines.append("## MAE by Candidate")
    lines.append("")
    lines.append(_df_to_markdown(comparison["mae_candidate"]))
    lines.append("")

    lines.append("## RMSE by Candidate")
    lines.append("")
    lines.append(_df_to_markdown(comparison["rmse_candidate"]))
    lines.append("")

    lines.append("## Max Error by Candidate")
    lines.append("")
    lines.append(_df_to_markdown(comparison["max_error_candidate"]))
    lines.append("")

    lines.append("## MAE by Firm")
    lines.append("")
    lines.append(_df_to_markdown(comparison["mae_firm"]))
    lines.append("")

    lines.append("## RMSE by Firm")
    lines.append("")
    lines.append(_df_to_markdown(comparison["rmse_firm"]))
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote validation report to %s", report_path)
    return report_path
