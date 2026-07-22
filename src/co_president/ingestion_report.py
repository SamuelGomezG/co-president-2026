"""Per-bundle ingestion report for the 2026 CNE pipeline.

STATE_REPORT.md §6.A1: 18/29 bundles were silently dropped via a bare
``except Exception``. This module replaces that pattern with a structured
``BundleResult`` record, a ``DataQualityError`` that fails the pipeline when
a firm produces zero rows, and ``persist_report`` / ``write_report_markdown``
that surface the report in ``results/ingestion/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

BundleStatus = Literal["ok", "rejected", "errored"]


@dataclass(frozen=True)
class BundleResult:
    """Per-bundle ingestion outcome.

    Attributes:
        firm: Inferred firm key (e.g. ``"atlas_intel"``).
        bundle_path: Absolute path to the source zip.
        loader: Name of the loader that handled the bundle.
        rows: Number of microdata rows successfully loaded.
        status: One of ``"ok"``, ``"rejected"``, ``"errored"``.
        reason: Human-readable description when status != ``"ok"``.
        shares_sum_check: Mean of per-bundle share totals (None if not
            computed for this bundle type).

    """

    firm: str
    bundle_path: Path
    loader: str
    rows: int
    status: BundleStatus
    reason: str | None
    shares_sum_check: float | None


class DataQualityError(RuntimeError):
    """Raised when ingestion output violates a project-level invariant.

    STATE_REPORT.md §6.A1: a firm producing zero rows after a successful
    load attempt is a silent dropout and must abort the pipeline so the
    problem is fixed at the source rather than masked downstream.
    """


def check_firm_coverage(report: list[BundleResult]) -> None:
    """Raise DataQualityError if any firm produced zero rows.

    Iterates over all bundle results and asserts each distinct firm has at
    least one ``status == "ok"`` record. Otherwise the firm is silently
    absent from the topline and downstream model training would not
    notice.

    Args:
        report: Bundle results from a single ingestion run.

    Raises:
        DataQualityError: Listing the offending firms.

    """
    by_firm: dict[str, list[BundleResult]] = {}
    for result in report:
        by_firm.setdefault(result.firm, []).append(result)

    failing = [firm for firm, rows in by_firm.items() if not any(r.status == "ok" for r in rows)]
    if failing:
        msg = (
            f"DataQualityError: firms with zero successful bundles: {sorted(failing)}. "
            f"See results/ingestion/{{year}}_bundle_report.md for per-bundle detail."
        )
        raise DataQualityError(msg)


def results_to_dataframe(results: list[BundleResult]) -> pd.DataFrame:
    """Convert a list of BundleResult to a tidy DataFrame."""
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    if "bundle_path" in df.columns:
        df["bundle_path"] = df["bundle_path"].astype(str)
    return df


def persist_report(
    report_df: pd.DataFrame,
    year: int,
    output_dir: Path,
) -> Path:
    """Write *report_df* to ``{output_dir}/{year}_bundle_report.parquet``.

    Creates *output_dir* if missing.

    Args:
        report_df: DataFrame of BundleResult records.
        year: Election year (used for filename).
        output_dir: Directory to write the parquet.

    Returns:
        Path to the written parquet file.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{year}_bundle_report.parquet"
    report_df.to_parquet(path, index=False)
    logger.info("Wrote %d-row bundle report to %s", len(report_df), path)
    return path


def write_report_markdown(
    report_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write a human-readable summary of *report_df* to *output_path*.

    Args:
        report_df: DataFrame of BundleResult records.
        output_path: Where to write the markdown.

    Returns:
        Path to the written file.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(report_df)
    if total == 0:
        by_status: dict[str, int] = {}
        by_firm = pd.DataFrame()
    else:
        counts = report_df["status"].value_counts().to_dict()
        by_status = {str(k): int(v) for k, v in counts.items()}
        by_firm = report_df.pivot_table(
            index="firm",
            columns="status",
            values="bundle_path",
            aggfunc="count",
            fill_value=0,
        )

    lines = [
        "# CNE 2026 Ingestion Report",
        "",
        f"Total bundles processed: **{total}**",
        "",
        "## By status",
        "",
        "| status | count |",
        "|---|---|",
    ]
    lines.extend(f"| {s} | {by_status.get(s, 0)} |" for s in ("ok", "rejected", "errored"))

    lines.extend(
        [
            "",
            "## By firm",
            "",
            "| firm | ok | rejected | errored |",
            "|---|---|---|---|",
        ]
    )
    if not by_firm.empty:
        for firm, row in by_firm.iterrows():
            ok = row.get("ok", 0)
            rejected = row.get("rejected", 0)
            errored = row.get("errored", 0)
            lines.append(f"| {firm} | {ok} | {rejected} | {errored} |")

    failed = (
        report_df[report_df["status"] != "ok"]
        if "status" in report_df.columns and not report_df.empty
        else report_df
    )
    if not failed.empty:
        lines.extend(
            [
                "",
                "## Failures (status != ok)",
                "",
                "| firm | bundle | reason |",
                "|---|---|---|",
            ]
        )
        for _, row in failed.iterrows():
            lines.append(f"| {row['firm']} | {row['bundle_path']} | {row['reason'] or ''} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote markdown bundle report to %s", output_path)
    return output_path


__all__ = [
    "BundleResult",
    "BundleStatus",
    "DataQualityError",
    "check_firm_coverage",
    "persist_report",
    "results_to_dataframe",
    "write_report_markdown",
]
