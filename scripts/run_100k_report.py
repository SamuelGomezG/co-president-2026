"""Run 100K-iteration MCMC for both rounds and produce an accuracy report.

Samples the Round 1 (Dirichlet-Multinomial) and Runoff (K=3) models with
configurable posterior draws, then compares posterior means against the
canonical 2022 election results.
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
    save_reports,
)

from co_president.config import ModelConfig
from co_president.data import load_and_clean_all, load_canonical_results
from co_president.fundamentals.features import load_features
from co_president.ingestion.ingest_trends import compute_prop_fav, fetch_trends
from co_president.ingestion.trends_keywords import (
    CANDIDATE_QUERY_MAPS,
    CANDIDATE_QUERY_MAP_2022_RUNOFF,
)

if TYPE_CHECKING:
    from co_president.data import CleanPolls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_100k_report")

CONFIG_DEBUG = ModelConfig(
    mcmc_draws=5_000,
    mcmc_tune=5_000,
    mcmc_chains=4,
    mcmc_cores=4,
    target_accept=0.95,
    seed=332211,
    random_walk_sigma_prior=0.5,
    concentration_election_prior_mean=5000,
    nuts_sampler="numpyro",
)

RESULTS_DIR = Path("results")
REPORT_PATH = Path("results/baseline_100k_report.md")
JSON_PATH = Path("results/baseline_100k_report.json")


def main() -> None:
    """Run 100K-iteration MCMC for both rounds and produce accuracy report."""
    started_at = datetime.now(UTC)
    logger.info("Starting 100K-iteration accuracy benchmark...")

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

    n_polls_r1 = len(clean_polls.round1)
    n_polls_r2 = len(clean_polls.round2)
    logger.info(
        "Loaded %d round-1 polls, %d runoff polls",
        n_polls_r1,
        n_polls_r2,
    )

    features: pd.DataFrame | None = None
    try:
        features = load_features()
        logger.info("Loaded municipal features (%d municipalities)", len(features))
    except (FileNotFoundError, ValueError):
        logger.warning(
            "Municipal features not available; falling back to polls-only model "
            "and calibrated transfer prior",
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    digital_signals: pd.DataFrame = pd.DataFrame()
    trends_cache = RESULTS_DIR / "trends_cache_2022.parquet"
    if trends_cache.exists():
        try:
            digital_signals = pd.read_parquet(trends_cache)
            digital_signals["fecha"] = pd.to_datetime(digital_signals["fecha"])
            logger.info(
                "Loaded cached Google Trends for %d candidates",
                len([c for c in digital_signals.columns if c != "fecha"]),
            )
        except (OSError, ValueError, KeyError):
            digital_signals = pd.DataFrame()
            logger.warning("Failed to load cached Google Trends; will attempt fetch")
    if digital_signals.empty:
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
                    digital_signals.to_parquet(trends_cache)
                    logger.info(
                        "Fetched and cached Google Trends for %d candidates, %d runoff dates",
                        len(query_map),
                        len(digital_signals),
                    )
        except (OSError, ValueError, KeyError, RuntimeError):
            logger.warning(
                "Failed to fetch Google Trends data. On first run this is expected if "
                "rate-limited (429). Wait a few minutes and re-run; subsequent runs will "
                "use cached data.",
                exc_info=True,
            )

    if digital_signals.empty:
        msg = (
            "Digital signals are required for the model but are empty after "
            "cache and fetch attempts. Cannot proceed without digital signals. "
            "Check the Google Trends cache and internet connection."
        )
        raise RuntimeError(msg)

    # Runoff-specific head-to-head digital signals (only Petro + Rodolfo)
    # Separately cached so prop_fav is computed head-to-head per SciELO.
    digital_signals_runoff: pd.DataFrame = pd.DataFrame()
    runoff_cache = RESULTS_DIR / "trends_cache_2022_runoff.parquet"
    if runoff_cache.exists():
        try:
            digital_signals_runoff = pd.read_parquet(runoff_cache)
            digital_signals_runoff["fecha"] = pd.to_datetime(digital_signals_runoff["fecha"])
            logger.info("Loaded cached runoff-specific head-to-head Trends")
        except (OSError, ValueError, KeyError):
            digital_signals_runoff = pd.DataFrame()
    if digital_signals_runoff.empty:
        try:
            runoff_keys = list(CANDIDATE_QUERY_MAP_2022_RUNOFF.values())
            logger.info("Fetching runoff-specific Google Trends (Petro vs Rodolfo)...")
            trends_raw = fetch_trends(
                runoff_keys,
                start_date="2022-05-01",
                end_date="2022-06-18",
            )
            if not trends_raw.empty:
                prop_fav_long = compute_prop_fav(trends_raw, CANDIDATE_QUERY_MAP_2022_RUNOFF)
                digital_signals_runoff = prop_fav_long.pivot_table(
                    index="as_of_date",
                    columns="candidate",
                    values="prop_fav",
                ).reset_index()
                digital_signals_runoff = digital_signals_runoff.rename(
                    columns={"as_of_date": "fecha"}
                )
                digital_signals_runoff["fecha"] = pd.to_datetime(digital_signals_runoff["fecha"])
                digital_signals_runoff.to_parquet(runoff_cache)
                logger.info("Fetched and cached runoff-specific Trends")
        except (OSError, ValueError, KeyError, RuntimeError):
            logger.warning(
                "Failed to fetch runoff-specific Trends; computing head-to-head "
                "from multi-candidate cache.",
            )
    # Fallback: compute head-to-head from multi-candidate prop_fav
    if digital_signals_runoff.empty and not digital_signals.empty:
        runoff_cols = ["gustavo_petro", "rodolfo_hernandez"]
        if all(c in digital_signals.columns for c in runoff_cols):
            ds_h2h = digital_signals[runoff_cols].copy()
            total = ds_h2h.sum(axis=1)
            ds_h2h = ds_h2h.div(total.where(total > 0, 1.0), axis=0).fillna(0.0)
            ds_h2h["fecha"] = digital_signals["fecha"]
            digital_signals_runoff = ds_h2h
            logger.info("Computed head-to-head from multi-candidate Trends cache")
    if digital_signals_runoff.empty:
        logger.warning(
            "Runoff-specific digital signals unavailable; using multi-candidate "
            "signals as fallback",
        )
        digital_signals_runoff = digital_signals

    idata_r1, r1_metrics, r1_rhat, r1_converged, elapsed_r1_s = run_round1_survey(
        CONFIG_DEBUG,
        clean_polls,
        results_r1,
        features=features,
        digital_signals=digital_signals,
        results_dir=RESULTS_DIR,
    )

    _idata_runoff, r2_metrics, r2_rhat, r2_converged, elapsed_r2_s = run_round2_survey(
        CONFIG_DEBUG,
        clean_polls,
        results_r1,
        results_r2,
        idata_r1,
        features=features,
        digital_signals=digital_signals_runoff,
    )

    report_data = ReportInputs(
        r1_metrics=r1_metrics,
        r2_metrics=r2_metrics,
        r1_rhat=r1_rhat,
        r1_converged=r1_converged,
        r2_rhat=r2_rhat,
        r2_converged=r2_converged,
        elapsed_r1_s=elapsed_r1_s,
        elapsed_r2_s=elapsed_r2_s,
        n_polls_r1=n_polls_r1,
        n_polls_r2=n_polls_r2,
        election_year=2022,
    )

    report_md = generate_report(report_data, CONFIG_DEBUG)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    logger.info("Saved report to %s", REPORT_PATH)

    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()
    save_reports(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        n_polls_r1,
        n_polls_r2,
        CONFIG_DEBUG,
        REPORT_PATH,
        JSON_PATH,
        year=2022,
    )
    print_summary(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        REPORT_PATH,
        JSON_PATH,
        r1_converged=r1_converged,
        r2_converged=r2_converged,
    )


if __name__ == "__main__":
    main()
