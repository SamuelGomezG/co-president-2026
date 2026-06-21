"""True forward backtest: R1 without election result, then honest runoff.

Saves report to ``results/forward_backtest_2022_report.{md,json}``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path

import pandas as pd
from report_utils import (
    ReportInputs,
    check_convergence,
    compute_r1_accuracy,
    generate_report,
    print_summary,
    run_round2_survey,
    save_reports,
)

from co_president.config import ModelConfig, get_active_candidates, get_election_date
from co_president.data import load_and_clean_all, load_canonical_results
from co_president.fundamentals.features import load_features
from co_president.ingestion.ingest_trends import compute_prop_fav, fetch_trends
from co_president.ingestion.trends_keywords import CANDIDATE_QUERY_MAPS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forward")

CONFIG = ModelConfig(
    mcmc_draws=5000,
    mcmc_tune=5000,
    mcmc_chains=4,
    mcmc_cores=4,
    target_accept=0.95,
    seed=332211,
    random_walk_sigma_prior=0.5,
    concentration_election_prior_mean=5000,
    nuts_sampler="numpyro",
)


def _load_digital_signals(
    year: int,
    query_map: dict[str, str],
    start_date: str,
    end_date: str,
    results_dir: Path,
) -> pd.DataFrame:
    """Load or fetch Google Trends digital signals.

    Args:
        year: Election year.
        query_map: Candidate→query map.
        start_date: Trends start date (YYYY-MM-DD).
        end_date: Trends end date (YYYY-MM-DD).
        results_dir: Directory for cache files.

    Returns:
        DataFrame with ``fecha`` column and per-candidate proportion-favorable
        columns, or empty ``DataFrame`` if unavailable.

    """
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
    if digital_signals.empty and query_map:
        try:
            logger.info("Fetching Google Trends data...")
            trends_raw = fetch_trends(
                list(query_map.values()),
                start_date=start_date,
                end_date=end_date,
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
                    "Fetched and cached Google Trends for %d candidates",
                    len(query_map),
                )
        except (OSError, ValueError, KeyError, RuntimeError):
            logger.warning(
                "Failed to fetch Google Trends data",
                exc_info=True,
            )
    return digital_signals


def _load_digital_signals_runoff(
    year: int,
    digital_signals: pd.DataFrame,
    query_map: dict[str, str],
    runoff_query_map: dict[str, str],
    start_date: str,
    end_date: str,
    results_dir: Path,
) -> pd.DataFrame:
    """Load or compute head-to-head runoff digital signals.

    Args:
        year: Election year.
        digital_signals: Multi-candidate trends DataFrame.
        query_map: Full candidate→query map (fallback).
        runoff_query_map: Runoff candidate→query map.
        start_date: Trends start date (YYYY-MM-DD).
        end_date: Trends end date (YYYY-MM-DD).
        results_dir: Directory for cache files.

    Returns:
        DataFrame with head-to-head runoff proportions.

    """
    digital_signals_runoff: pd.DataFrame = pd.DataFrame()
    runoff_cache = results_dir / f"trends_cache_{year}_runoff.parquet"
    if runoff_cache.exists():
        try:
            digital_signals_runoff = pd.read_parquet(runoff_cache)
            digital_signals_runoff["fecha"] = pd.to_datetime(digital_signals_runoff["fecha"])
            logger.info("Loaded cached runoff-specific head-to-head Trends")
            return digital_signals_runoff
        except (OSError, ValueError, KeyError):
            digital_signals_runoff = pd.DataFrame()
    if digital_signals_runoff.empty and runoff_query_map:
        try:
            runoff_keys = list(runoff_query_map.values())
            logger.info("Fetching runoff-specific Google Trends...")
            trends_raw = fetch_trends(
                runoff_keys,
                start_date=start_date,
                end_date=end_date,
            )
            if not trends_raw.empty:
                prop_fav_long = compute_prop_fav(trends_raw, runoff_query_map)
                digital_signals_runoff = prop_fav_long.pivot_table(
                    index="as_of_date",
                    columns="candidate",
                    values="prop_fav",
                ).reset_index()
                digital_signals_runoff = digital_signals_runoff.rename(
                    columns={"as_of_date": "fecha"},
                )
                digital_signals_runoff["fecha"] = pd.to_datetime(digital_signals_runoff["fecha"])
                digital_signals_runoff.to_parquet(runoff_cache)
                logger.info("Fetched and cached runoff-specific Trends")
                return digital_signals_runoff
        except (OSError, ValueError, KeyError, RuntimeError):
            logger.warning("Failed to fetch runoff-specific Trends")

    runoff_candidates = list(runoff_query_map.keys()) if runoff_query_map else []
    if digital_signals_runoff.empty and not digital_signals.empty:
        if all(c in digital_signals.columns for c in runoff_candidates):
            ds_h2h = digital_signals[runoff_candidates].copy()
            total = ds_h2h.sum(axis=1)
            ds_h2h = ds_h2h.div(total.where(total > 0, 1.0), axis=0).fillna(0.0)
            ds_h2h["fecha"] = digital_signals["fecha"]
            digital_signals_runoff = ds_h2h
            logger.info("Computed head-to-head from multi-candidate Trends cache")
    if digital_signals_runoff.empty and not digital_signals.empty:
        logger.warning("Using multi-candidate signals as runoff fallback")
        digital_signals_runoff = digital_signals
    return digital_signals_runoff


def _extract_posterior_summaries(
    idata_r1: object,
    features: pd.DataFrame | None,
) -> dict:
    """Extract model parameter posteriors for the report."""
    import arviz as az

    summaries: dict = {}

    try:
        phi_elec_samples = idata_r1.posterior["phi_elec"].to_numpy().flatten()  # type: ignore[union-attr]
        ci = az.hdi(phi_elec_samples, prob=0.95)  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        summaries["phi_elec"] = {
            "mean": round(float(phi_elec_samples.mean()), 1),
            "std": round(float(phi_elec_samples.std()), 1),
            "ci_95_lower": round(float(ci[0]), 1),
            "ci_95_upper": round(float(ci[1]), 1),
        }
    except (KeyError, ValueError, TypeError):
        summaries["phi_elec"] = {}

    try:
        phi_digital_samples = idata_r1.posterior["phi_digital_base"].to_numpy().flatten()  # type: ignore[union-attr]
        ci = az.hdi(phi_digital_samples, prob=0.95)  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        summaries["phi_digital_base"] = {
            "mean": round(float(phi_digital_samples.mean()), 1),
            "std": round(float(phi_digital_samples.std()), 1),
            "ci_95_lower": round(float(ci[0]), 1),
            "ci_95_upper": round(float(ci[1]), 1),
        }
    except (KeyError, ValueError, TypeError):
        summaries["phi_digital_base"] = {}

    if features is not None and not features.empty:
        from co_president.model_round1 import _compute_national_internet_rate  # noqa: PLC0415

        summaries["internet_rate"] = round(
            _compute_national_internet_rate(features),
            4,
        )
    else:
        summaries["internet_rate"] = None

    return summaries


def main() -> None:
    """Run true forward backtest (R1 without election result)."""
    parser = argparse.ArgumentParser(description="Run true forward backtest.")
    parser.add_argument("--year", type=int, default=2022, help="Election year (default 2022)")
    args = parser.parse_args()
    year = args.year

    started_at = datetime.now(UTC)
    logger.info("Starting true forward backtest for %d...", year)

    logger.info("Loading canonical election results...")
    results_r1, results_r2 = load_canonical_results()

    logger.info("Loading and cleaning poll data...")
    clean_polls = load_and_clean_all(year=year)

    n_polls_r1 = len(clean_polls.round1)
    n_polls_r2 = len(clean_polls.round2)
    logger.info("Loaded %d round-1 polls, %d runoff polls", n_polls_r1, n_polls_r2)

    features: pd.DataFrame | None = None
    try:
        features = load_features()
        logger.info("Loaded municipal features (%d municipalities)", len(features))
    except (FileNotFoundError, ValueError):
        logger.warning(
            "Municipal features not available; falling back to polls-only model",
        )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    election_r1 = get_election_date(year, 1)
    election_r2 = get_election_date(year, 2)
    trends_start = (election_r1 - timedelta(days=90)).strftime("%Y-%m-%d")
    trends_end = (election_r2 - timedelta(days=1)).strftime("%Y-%m-%d")

    query_map = CANDIDATE_QUERY_MAPS.get(str(year)) or CANDIDATE_QUERY_MAPS["2022"]
    runoff_candidates = [c.key for c in get_active_candidates(2, year=year) if c.key != "rest"]
    runoff_query_map = {k: query_map[k] for k in runoff_candidates if k in query_map}

    digital_signals = _load_digital_signals(
        year,
        query_map,
        trends_start,
        trends_end,
        results_dir,
    )
    if digital_signals.empty:
        logger.warning(
            "Digital signals unavailable; model will proceed without trends data",
        )

    digital_signals_runoff = _load_digital_signals_runoff(
        year,
        digital_signals,
        query_map,
        runoff_query_map,
        (election_r1 + timedelta(days=1)).strftime("%Y-%m-%d"),
        trends_end,
        results_dir,
    )

    # ── R1 model WITHOUT election result (true forward) ──
    import co_president.model_round1 as m1  # noqa: PLC0415

    logger.info("Building R1 model without election result...")
    model_r1 = m1.build_round1_model(
        clean_polls.round1,
        results=None,
        config=CONFIG,
        features=features,
        digital_signals=digital_signals,
        year=year,
    )

    logger.info(
        "Sampling R1 model (%d draws x %d chains)...",
        CONFIG.mcmc_draws,
        CONFIG.mcmc_chains,
    )
    t0 = datetime.now(UTC)
    idata_r1 = m1.sample_round1(model_r1, CONFIG)
    elapsed_r1_s = (datetime.now(UTC) - t0).total_seconds()
    logger.info("R1 sampling complete in %.0fs", elapsed_r1_s)

    # R1 convergence and accuracy
    r1_rhat, r1_converged = check_convergence(idata_r1, "R1")
    r1_metrics = compute_r1_accuracy(
        idata_r1,
        results_r1,
        poll_columns=set(clean_polls.round1.columns),
    )
    r1_metrics["posterior_summaries"] = _extract_posterior_summaries(idata_r1, features)

    # ── Runoff via estimate_runoff_matrix ──
    logger.info("Computing runoff matrix...")
    _idata_runoff, r2_metrics, r2_rhat, r2_converged, elapsed_r2_s = run_round2_survey(
        CONFIG,
        clean_polls,
        results_r1,
        results_r2,
        idata_r1,
        features=features,
        digital_signals=digital_signals_runoff,
        year=year,
    )

    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()

    # ── Report ──
    if year == 2022:
        report_path = results_dir / "forward_backtest_2022_report.md"
        json_path = results_dir / "forward_backtest_2022_report.json"
    else:
        report_path = results_dir / f"forward_backtest_{year}_report.md"
        json_path = results_dir / f"forward_backtest_{year}_report.json"

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

    report_md = generate_report(report_data, CONFIG)
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved report to %s", report_path)

    save_reports(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        n_polls_r1,
        n_polls_r2,
        CONFIG,
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
