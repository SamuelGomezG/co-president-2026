"""True forward backtest: R1 without election result, then honest runoff.

Saves report to ``results/forward_backtest_2022_report.{md,json}``.
"""

from __future__ import annotations

from datetime import UTC, datetime
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

from co_president.config import ModelConfig
from co_president.data import load_and_clean_all, load_canonical_results
from co_president.fundamentals.features import load_features
from co_president.ingestion.ingest_trends import compute_prop_fav, fetch_trends
from co_president.ingestion.trends_keywords import (
    CANDIDATE_QUERY_MAPS,
    CANDIDATE_QUERY_MAP_2022_RUNOFF,
)

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

RESULTS_DIR = Path("results")
REPORT_PATH = RESULTS_DIR / "forward_backtest_2022_report.md"
JSON_PATH = RESULTS_DIR / "forward_backtest_2022_report.json"


def _load_digital_signals() -> pd.DataFrame:
    """Load or fetch Google Trends digital signals.

    Returns:
        DataFrame with ``fecha`` column and per-candidate proportion-favorable
        columns, or empty ``DataFrame`` if unavailable.

    """
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
        query_map = CANDIDATE_QUERY_MAPS.get("2022")
        if query_map:
            try:
                logger.info("Fetching Google Trends data...")
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
                        "Fetched and cached Google Trends for %d candidates",
                        len(query_map),
                    )
            except (OSError, ValueError, KeyError, RuntimeError):
                logger.warning(
                    "Failed to fetch Google Trends data",
                    exc_info=True,
                )
    return digital_signals


def _load_digital_signals_runoff(digital_signals: pd.DataFrame) -> pd.DataFrame:
    """Load or compute head-to-head runoff digital signals.

    Args:
        digital_signals: Multi-candidate trends DataFrame.

    Returns:
        DataFrame with Petro/Rodolfo head-to-head proportions.

    """
    digital_signals_runoff: pd.DataFrame = pd.DataFrame()
    runoff_cache = RESULTS_DIR / "trends_cache_2022_runoff.parquet"
    if runoff_cache.exists():
        try:
            digital_signals_runoff = pd.read_parquet(runoff_cache)
            digital_signals_runoff["fecha"] = pd.to_datetime(digital_signals_runoff["fecha"])
            logger.info("Loaded cached runoff-specific head-to-head Trends")
            return digital_signals_runoff
        except (OSError, ValueError, KeyError):
            digital_signals_runoff = pd.DataFrame()
    if digital_signals_runoff.empty:
        runoff_keys = list(CANDIDATE_QUERY_MAP_2022_RUNOFF.values())
        try:
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
                    columns={"as_of_date": "fecha"},
                )
                digital_signals_runoff["fecha"] = pd.to_datetime(digital_signals_runoff["fecha"])
                digital_signals_runoff.to_parquet(runoff_cache)
                logger.info("Fetched and cached runoff-specific Trends")
                return digital_signals_runoff
        except (OSError, ValueError, KeyError, RuntimeError):
            logger.warning("Failed to fetch runoff-specific Trends")
    if digital_signals_runoff.empty and not digital_signals.empty:
        runoff_cols = ["gustavo_petro", "rodolfo_hernandez"]
        if all(c in digital_signals.columns for c in runoff_cols):
            ds_h2h = digital_signals[runoff_cols].copy()
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
    started_at = datetime.now(UTC)
    logger.info("Starting true forward backtest...")

    logger.info("Loading canonical election results...")
    results_r1, results_r2 = load_canonical_results()

    logger.info("Loading and cleaning poll data...")
    clean_polls = load_and_clean_all()

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    digital_signals = _load_digital_signals()
    if digital_signals.empty:
        logger.warning(
            "Digital signals unavailable; model will proceed without trends data",
        )

    digital_signals_runoff = _load_digital_signals_runoff(digital_signals)

    # ── R1 model WITHOUT election result (true forward) ──
    import co_president.model_round1 as m1  # noqa: PLC0415

    logger.info("Building R1 model without election result...")
    model_r1 = m1.build_round1_model(
        clean_polls.round1,
        results=None,
        config=CONFIG,
        features=features,
        digital_signals=digital_signals,
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
    )

    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()

    # ── Report ──
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

    report_md = generate_report(report_data, CONFIG)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    logger.info("Saved report to %s", REPORT_PATH)

    save_reports(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        n_polls_r1,
        n_polls_r2,
        CONFIG,
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
