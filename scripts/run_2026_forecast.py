"""SPEC-42: 2026 presidential forecast — backtest and forward modes.

Runs full MCMC for both rounds with Google Trends signals, municipal
structural prior, and presidential approval covariate.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pandas as pd
from report_utils import (
    ReportInputs,
    check_convergence,
    generate_report,
    print_summary,
    run_round1_survey,
    run_round2_survey,
    save_reports,
)

from co_president.config import ModelConfig, get_active_candidates, get_election_date
from co_president.data import load_and_clean_all, load_canonical_results
from co_president.data_cne_2026 import extract_approval_series
from co_president.empirical_runoff import get_empirical_runoff_betas
from co_president.fundamentals.features import load_features
from co_president.ingestion.ingest_trends import compute_prop_fav, fetch_trends
from co_president.ingestion.trends_keywords import CANDIDATE_QUERY_MAPS

if TYPE_CHECKING:
    from co_president.data import CleanPolls, RoundResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_2026_forecast")

CONFIG_DEBUG = ModelConfig(
    mcmc_draws=5_000,
    mcmc_tune=5_000,
    mcmc_chains=4,
    mcmc_cores=4,
    target_accept=0.99,
    seed=332211,
    random_walk_sigma_prior=0.5,
    concentration_election_prior_mean=5000,
    nuts_sampler="numpyro",
)


def _report_paths(mode: str) -> tuple[Path, Path]:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return (
        results_dir / f"2026_{mode}_report.md",
        results_dir / f"2026_{mode}_report.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2026 presidential election forecast.",
    )
    parser.add_argument(
        "--mode",
        choices=("backtest", "forward"),
        default="forward",
        help="backtest: use real election results. forward: no results (true forecast).",
    )
    args = parser.parse_args()
    mode: str = args.mode
    year = 2026

    started_at = datetime.now(UTC)
    logger.info("Starting 2026 forecast — mode=%s", mode)

    # ── Load election results ──────────────────────────────────────────
    results_r1: RoundResult | None = None
    results_r2: RoundResult | None = None
    logger.info("Loading canonical election results...")
    try:
        results_r1, results_r2 = load_canonical_results(year=year)
    except (OSError, ValueError):
        logger.warning("Failed to load election results; using defaults")
        # results_r1 and results_r2 remain None for forward mode

    if mode == "backtest" and results_r2 is None:
        logger.error("Backtest mode requires election results; none loaded")
        sys.exit(1)

    # ── Load and clean poll data ───────────────────────────────────────
    logger.info("Loading and cleaning poll data...")
    try:
        import co_president.data_polls as _dp

        _dp._MIN_ROUND1_POLLSTERS = 2
        _dp._MIN_ROUND2_POLLSTERS = 1
        clean_polls: CleanPolls = load_and_clean_all(year=year)
    except (OSError, ValueError):
        logger.exception("Failed to load poll data")
        sys.exit(1)

    n_polls_r1 = len(clean_polls.round1)
    n_polls_r2 = len(clean_polls.round2)
    logger.info("Loaded %d round-1 polls, %d runoff polls", n_polls_r1, n_polls_r2)

    if n_polls_r1 == 0:
        logger.error("No round-1 polls loaded. Run `make fundamentals` and retry.")
        sys.exit(1)

    # ── Municipal features (required for 2026) ─────────────────────────
    logger.info("Loading municipal features...")
    try:
        features = load_features()
        logger.info("Loaded municipal features (%d municipalities)", len(features))
    except (FileNotFoundError, ValueError):
        logger.exception("Municipal features are required for 2026 forecast")
        sys.exit(1)

    # ── Google Trends (required; hard-exit on failure) ─────────────────
    election_r1 = get_election_date(year, 1)
    election_r2 = get_election_date(year, 2)
    trends_start = (election_r1 - timedelta(days=90)).strftime("%Y-%m-%d")
    trends_end = (election_r2 - timedelta(days=1)).strftime("%Y-%m-%d")

    query_map = CANDIDATE_QUERY_MAPS.get(str(year)) or CANDIDATE_QUERY_MAPS["2022"]

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    digital_signals: pd.DataFrame = pd.DataFrame()
    trends_cache = results_dir / f"trends_cache_{year}.parquet"
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
            if query_map:
                logger.info("Fetching Google Trends data...")
                trends_raw = fetch_trends(
                    list(query_map.values()),
                    start_date=trends_start,
                    end_date=trends_end,
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
                        "Fetched and cached Google Trends for %d candidates, %d dates",
                        len(query_map),
                        len(digital_signals),
                    )
        except (OSError, ValueError, KeyError, RuntimeError):
            logger.warning("Failed to fetch Google Trends data.", exc_info=True)

    if digital_signals.empty:
        msg = (
            "Digital signals are required for the 2026 model but are empty "
            "after cache and fetch attempts. Cannot proceed without digital "
            "signals. Check the Google Trends cache and internet connection."
        )
        raise RuntimeError(msg)

    # ── Runoff-specific head-to-head digital signals ────────────────────
    runoff_candidates = [c.key for c in get_active_candidates(2, year=year) if c.key != "rest"]
    runoff_query_map = {k: query_map[k] for k in runoff_candidates if k in query_map}
    digital_signals_runoff: pd.DataFrame = pd.DataFrame()
    runoff_cache = results_dir / f"trends_cache_{year}_runoff.parquet"
    if runoff_cache.exists():
        try:
            digital_signals_runoff = pd.read_parquet(runoff_cache)
            digital_signals_runoff["fecha"] = pd.to_datetime(digital_signals_runoff["fecha"])
            logger.info("Loaded cached runoff-specific head-to-head Trends")
        except (OSError, ValueError, KeyError):
            digital_signals_runoff = pd.DataFrame()
    if digital_signals_runoff.empty and runoff_query_map:
        try:
            runoff_keys = list(runoff_query_map.values())
            logger.info("Fetching runoff-specific Google Trends...")
            trends_raw = fetch_trends(
                runoff_keys,
                start_date=(election_r1 + timedelta(days=1)).strftime("%Y-%m-%d"),
                end_date=trends_end,
            )
            if not trends_raw.empty:
                prop_fav_long = compute_prop_fav(trends_raw, runoff_query_map)
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
            logger.warning("Failed to fetch runoff-specific Trends; using fallback")
    if digital_signals_runoff.empty and not digital_signals.empty:
        if all(c in digital_signals.columns for c in runoff_candidates):
            ds_h2h = digital_signals[runoff_candidates].copy()
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

    # ── Presidential approval covariate ────────────────────────────────
    approval: pd.Series | None = None
    try:
        approval = extract_approval_series()
        logger.info(
            "Extracted presidential approval: %d daily observations",
            len(approval),
        )
    except Exception:
        logger.warning("Failed to extract approval series; proceeding without", exc_info=True)

    # ── Round 1 survey ─────────────────────────────────────────────────
    # Forward mode: no election likelihood term (results=None).
    # Backtest mode: election likelihood anchors the model to actual results.
    _r1_for_model = results_r1 if mode == "backtest" else None
    idata_r1, r1_metrics, r1_rhat, r1_converged, elapsed_r1_s = run_round1_survey(
        CONFIG_DEBUG,
        clean_polls,
        _r1_for_model,
        digital_signals=digital_signals,
        features=features,
        results_dir=results_dir,
        year=year,
        approval=approval,
        no_house_effects=(mode == "backtest"),
    )

    # ── Round 2 survey ─────────────────────────────────────────────────
    empirical_betas = None
    if mode == "forward":
        empirical_betas = get_empirical_runoff_betas(year=year)
        logger.info("Forward mode: using empirical runoff betas")
    _r1_for_r2 = results_r1 if mode == "backtest" else results_r1
    _r2_for_model = results_r2 if mode == "backtest" else results_r2
    _idata_runoff, r2_metrics, r2_rhat, r2_converged, elapsed_r2_s = run_round2_survey(
        CONFIG_DEBUG,
        clean_polls,
        _r1_for_r2,
        _r2_for_model,
        idata_r1,
        features=features,
        digital_signals=digital_signals_runoff,
        year=year,
        empirical_betas=empirical_betas,
    )

    # ── Report ─────────────────────────────────────────────────────────
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
        election_year=year,
    )

    report_md = generate_report(report_data, CONFIG_DEBUG)
    report_path, json_path = _report_paths(mode)
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved report to %s", report_path)

    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()
    save_reports(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        n_polls_r1,
        n_polls_r2,
        CONFIG_DEBUG,
        report_path,
        json_path,
        year=year,
    )
    print_summary(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        report_path,
        json_path,
        r1_converged=r1_converged,
        r2_converged=r2_converged,
    )


if __name__ == "__main__":
    main()
