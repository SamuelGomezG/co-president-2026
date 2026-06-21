"""Run 2026 parallel backtest with comparative report against 2022.

Runs both 2022 (regression check) and 2026 backtests at 4K prototype draws,
then produces a comparative accuracy report.

Usage:
    uv run python scripts/run_2026_backtest.py
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pandas as pd
from report_utils import (
    ReportInputs,
    generate_report,
    print_summary,
    run_round1_survey,
    run_round2_survey,
)

from co_president.config import ModelConfig
from co_president.data import load_and_clean_all, load_canonical_results
from co_president.fundamentals.features import load_features
from co_president.ingestion.ingest_trends import compute_prop_fav, fetch_trends
from co_president.ingestion.trends_keywords import CANDIDATE_QUERY_MAPS

if TYPE_CHECKING:
    from co_president.data import CleanPolls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_2026_backtest")

CONFIG_PROTOTYPE = ModelConfig(
    mcmc_draws=4_000,
    mcmc_tune=2_000,
    mcmc_chains=4,
    mcmc_cores=4,
    target_accept=0.99,
    seed=332211,
    target_year=2026,
)

RESULTS_DIR = Path("results")
R1_2022_REPORT = RESULTS_DIR / "2022_regression_report.md"
R1_2022_JSON = RESULTS_DIR / "2022_regression_report.json"
R1_2026_REPORT = RESULTS_DIR / "2026_backtest_report.md"
R1_2026_JSON = RESULTS_DIR / "2026_backtest_report.json"


def _run_backtest(
    year: int,
    config: ModelConfig,
    clean_polls: CleanPolls,
    results_r1,
    results_r2,
    features: pd.DataFrame | None = None,
    digital_signals: pd.DataFrame | None = None,
    report_path: Path = R1_2022_REPORT,
    json_path: Path = R1_2022_JSON,
) -> float:
    """Run full backtest for a given election year.

    Args:
        year: Election year label (for logging/reporting).
        config: Model configuration.
        clean_polls: Cleaned poll data.
        results_r1: Round 1 canonical results.
        results_r2: Round 2 canonical results.
        features: Optional municipal features.
        digital_signals: Google Trends data.
        report_path: Output path for Markdown report.
        json_path: Output path for JSON report.

    Returns:
        Total elapsed wall-clock time in seconds.

    """
    if digital_signals is None:
        digital_signals = pd.DataFrame()
    logger.info("=" * 60)
    logger.info("Starting %s backtest...", year)
    logger.info("=" * 60)

    started_at = datetime.now(UTC)

    idata_r1, r1_metrics, r1_rhat, r1_converged, elapsed_r1_s = run_round1_survey(
        config,
        clean_polls,
        results_r1,
        features=features,
        digital_signals=digital_signals,
        results_dir=RESULTS_DIR,
    )

    _idata_runoff, r2_metrics, r2_rhat, r2_converged, elapsed_r2_s = run_round2_survey(
        config,
        clean_polls,
        results_r1,
        results_r2,
        idata_r1,
        features=features,
        digital_signals=digital_signals,
    )

    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()

    report_data = ReportInputs(
        r1_metrics=r1_metrics,
        r2_metrics=r2_metrics,
        r1_rhat=r1_rhat,
        r1_converged=r1_converged,
        r2_rhat=r2_rhat,
        r2_converged=r2_converged,
        elapsed_r1_s=elapsed_r1_s,
        elapsed_r2_s=elapsed_r2_s,
        n_polls_r1=len(clean_polls.round1),
        n_polls_r2=len(clean_polls.round2),
        election_year=year,
    )

    report_md = generate_report(report_data, config)
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved %s report to %s", year, report_path)

    json_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "election_year": year,
        "config": {
            "mcmc_draws": config.mcmc_draws,
            "mcmc_tune": config.mcmc_tune,
            "mcmc_chains": config.mcmc_chains,
            "seed": config.seed,
            "target_accept": config.target_accept,
        },
        "metrics": {
            "round1": r1_metrics,
            "round2": r2_metrics,
        },
        "convergence": {
            "round1_rhat": r1_rhat,
            "round1_converged": r1_converged,
            "round2_rhat": r2_rhat,
            "round2_converged": r2_converged,
        },
        "total_time_s": total_elapsed,
    }
    import json  # noqa: PLC0415

    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    logger.info("Saved %s JSON to %s", year, json_path)

    print_summary(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        report_path,
        json_path,
        r1_converged=r1_converged,
        r2_converged=r2_converged,
    )

    return total_elapsed


def _build_comparative_report(
    elapsed_2022: float,
    elapsed_2026: float,
) -> None:
    """Build a side-by-side comparison of 2022 and 2026 results.

    Reads both JSON reports and generates a comparison table.

    Args:
        elapsed_2022: Wall-clock time for 2022 backtest.
        elapsed_2026: Wall-clock time for 2026 backtest.

    """
    import json  # noqa: PLC0415

    try:
        d1 = json.loads(R1_2022_JSON.read_text(encoding="utf-8"))
        d2 = json.loads(R1_2026_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not build comparative report: %s", exc)
        return

    lines: list[str] = []
    lines.append("# Comparative Backtest Report: 2022 vs 2026\n")
    lines.append(
        f"**Generated**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}\n",
    )

    # Round 1 comparison
    lines.append("## Round 1 MAE Comparison\n")
    lines.append("| Candidate | 2022 MAE | 2026 MAE |")
    lines.append("|-----------|----------|----------|")

    r1_2022 = d1.get("metrics", {}).get("round1", {})
    r1_2026 = d2.get("metrics", {}).get("round1", {})

    cands_2022 = {c["candidate_key"]: c for c in r1_2022.get("candidates", [])}
    cands_2026 = {c["candidate_key"]: c for c in r1_2026.get("candidates", [])}
    all_keys = sorted(set(cands_2022) | set(cands_2026))

    for key in all_keys:
        c22 = cands_2022.get(key, {})
        c26 = cands_2026.get(key, {})
        name = c22.get("display_name") or c26.get("display_name") or key
        err22 = f"{c22.get('abs_error', 0) * 100:.2f}pp" if c22 else "N/A"
        err26 = f"{c26.get('abs_error', 0) * 100:.2f}pp" if c26 else "N/A"
        lines.append(f"| {name} | {err22} | {err26} |")

    lines.append("")
    mae22 = r1_2022.get("mae", 0)
    mae26 = r1_2026.get("mae", 0)
    lines.append(f"- **2022 aggregate MAE**: {mae22 * 100:.2f}pp")
    lines.append(f"- **2026 aggregate MAE**: {mae26 * 100:.2f}pp")
    lines.append(f"- **2022 sampling time**: {elapsed_2022:.0f}s")
    lines.append(f"- **2026 sampling time**: {elapsed_2026:.0f}s")
    lines.append("")

    # Convergence comparison
    lines.append("## Convergence Comparison\n")
    lines.append("| Metric | 2022 | 2026 |")
    lines.append("|--------|------|------|")
    cvg22 = d1.get("convergence", {})
    cvg26 = d2.get("convergence", {})
    r1_rhat_22 = (
        f"{cvg22.get('round1_rhat', 'N/A'):.4f}"
        if isinstance(cvg22.get("round1_rhat"), (int, float))
        else "N/A"
    )
    r1_rhat_26 = (
        f"{cvg26.get('round1_rhat', 'N/A'):.4f}"
        if isinstance(cvg26.get("round1_rhat"), (int, float))
        else "N/A"
    )
    lines.append(f"| R1 max R-hat | {r1_rhat_22} | {r1_rhat_26} |")
    lines.append(
        f"| R1 converged | {cvg22.get('round1_converged', 'N/A')} | {cvg26.get('round1_converged', 'N/A')} |"
    )
    lines.append("")

    report_path = RESULTS_DIR / "2022_vs_2026_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved comparative report to %s", report_path)


def main() -> None:
    """Run 2026 parallel backtest with 2022 regression check."""
    started_at = datetime.now(UTC)
    logger.info("Starting 2026 parallel backtest...")

    logger.info("Loading canonical election results...")
    try:
        results_r1, results_r2 = load_canonical_results()
    except (OSError, ValueError):
        logger.exception("Failed to load election results")
        sys.exit(1)

    logger.info("Loading and cleaning poll data...")
    try:
        clean_polls: CleanPolls = load_and_clean_all()
    except (OSError, ValueError):
        logger.exception("Failed to load poll data")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features: pd.DataFrame | None = None
    try:
        features = load_features()
        logger.info("Loaded municipal features (%d municipalities)", len(features))
    except (FileNotFoundError, ValueError):
        logger.warning("Municipal features not available")

    digital_signals: pd.DataFrame = pd.DataFrame()
    try:
        query_map = CANDIDATE_QUERY_MAPS.get("2022")
        if query_map:
            logger.info("Fetching Google Trends data for runoff...")
            trends_raw = fetch_trends(
                list(query_map.values()),
                start_date="2022-03-01",
                end_date="2022-06-18",
            )
            if not trends_raw.empty:
                prop_fav_long = compute_prop_fav(trends_raw, query_map)
                digital_signals = prop_fav_long.pivot_table(
                    index="as_of_date",
                    columns="candidate",
                    values="prop_fav",
                ).reset_index()
                digital_signals = digital_signals.rename(columns={"as_of_date": "fecha"})
                digital_signals["fecha"] = pd.to_datetime(digital_signals["fecha"])
                logger.info(
                    "Fetched Google Trends for %d candidates, %d runoff dates",
                    len(query_map),
                    len(digital_signals),
                )
    except (OSError, ValueError, KeyError):
        logger.warning("Failed to fetch Google Trends data", exc_info=True)

    logger.info("\n" + "#" * 60)
    logger.info("# PHASE 1: 2022 regression check")
    logger.info("#" * 60)

    elapsed_2022 = _run_backtest(
        2022,
        CONFIG_PROTOTYPE,
        clean_polls,
        results_r1,
        results_r2,
        features=features,
        digital_signals=digital_signals,
        report_path=R1_2022_REPORT,
        json_path=R1_2022_JSON,
    )

    logger.info("\n" + "#" * 60)
    logger.info("# PHASE 2: 2026 backtest")
    logger.info("#" * 60)

    # For 2026, we run the same model with target_year=2026.
    # Data loading uses 2022 paths (same format expected for 2026 data).
    # When user provides 2026 data, update the data paths here.
    config_2026 = CONFIG_PROTOTYPE  # target_year already set in CONFIG_PROTOTYPE

    elapsed_2026 = _run_backtest(
        2026,
        config_2026,
        clean_polls,
        results_r1,
        results_r2,
        features=features,
        digital_signals=digital_signals,
        report_path=R1_2026_REPORT,
        json_path=R1_2026_JSON,
    )

    _build_comparative_report(elapsed_2022, elapsed_2026)

    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()
    logger.info("=" * 60)
    logger.info("2026 parallel backtest complete in %.0fs", total_elapsed)
    logger.info("Reports: %s, %s", R1_2022_REPORT, R1_2026_REPORT)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
