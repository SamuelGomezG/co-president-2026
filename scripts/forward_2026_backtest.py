"""SPEC-42: True forward backtest for 2026 Colombian presidential election.

Runs R1 model without election result (true forward), then empirical
runoff matrix from SPEC-34.  Reporting follows the same pattern as
``run_100k_report.py`` via ``report_utils``.

No Google Trends.  No municipal features.  Simple, fast, forward.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys

import pandas as pd
from report_utils import (
    ReportInputs,
    _extract_posterior_summaries,
    check_convergence,
    compute_r1_accuracy,
    generate_report,
    print_summary,
    run_round2_survey,
    save_reports,
)

from co_president.config import ModelConfig, get_active_candidates, get_election_date
from co_president.data import CandidateResult, RoundResult
from co_president.data_polls import CleanPolls
from co_president.empirical_runoff import get_empirical_runoff_betas

# Relax pollster minimums for 2026 (only 3 firms provide machine-readable CNE microdata)
import co_president.data_polls as _dp  # noqa: E402

_dp._MIN_ROUND1_POLLSTERS = 3  # noqa: SLF001
_dp._MIN_ROUND2_POLLSTERS = 1  # noqa: SLF001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forward_2026")

CONFIG = ModelConfig(
    mcmc_draws=2000,
    mcmc_tune=2000,
    mcmc_chains=4,
    mcmc_cores=4,
    target_accept=0.95,
    seed=332211,
    random_walk_sigma_prior=0.5,
    concentration_election_prior_mean=5000,
    nuts_sampler="numpyro",
)


def _build_synthetic_result(round_number: int, year: int) -> RoundResult:
    """Construct a minimal ``RoundResult`` for a forward forecast.

    All candidate vote shares are zero — no canonical results exist for
    the target year.  Used for structural fields (date, registered_voters)
    and to satisfy ``run_round2_survey`` / ``compute_r1_accuracy`` signatures.
    """
    active = get_active_candidates(round_number, year=year)
    candidate_results = tuple(
        CandidateResult(candidate_key=c.key, votes=0, vote_share=0.0) for c in active
    )
    return RoundResult(
        round_number=round_number,
        date=get_election_date(year, round_number),
        total_valid_votes=1,
        total_votes_incl_blank=1,
        registered_voters=1,
        polling_stations=1,
        candidates=candidate_results,
        blank_votes=0,
        null_votes=0,
        unmarked_votes=0,
    )


def main() -> None:
    """Run true forward backtest for 2026 with empirical runoff matrix."""
    parser = argparse.ArgumentParser(
        description="Run 2026 true forward backtest with empirical runoff matrix.",
    )
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()
    year = args.year

    started_at = datetime.now(UTC)
    logger.info("Starting %d true forward backtest...", year)

    # ── Load data ──
    topline_path = Path("data/2026-polls/_processed/2026_topline.parquet")
    topline = pd.read_parquet(topline_path)
    r1_date = get_election_date(year, 1)
    field_end = pd.to_datetime(topline["field_end"]).dt.date
    round1_df = topline[field_end <= r1_date].copy()
    round2_df = topline[field_end > r1_date].copy()
    round1_df = round1_df.rename(columns={"field_end": "fecha"})
    round2_df = round2_df.rename(columns={"field_end": "fecha"})
    for df in (round1_df, round2_df):
        if "muestra" not in df.columns and "total_weight" in df.columns:
            df["muestra"] = df["total_weight"]

    n_polls_r1 = len(round1_df)
    n_polls_r2 = len(round2_df)
    logger.info(
        "Loaded %d R1 polls (%d pollsters), %d R2 polls (%d pollsters)",
        n_polls_r1,
        round1_df["encuestadora"].nunique(),
        n_polls_r2,
        round2_df["encuestadora"].nunique(),
    )

    clean_polls = CleanPolls(
        round1=round1_df,
        round2=round2_df,
        consultation=[],
        all_polls=topline.copy(),
    )

    synthetic_r1 = _build_synthetic_result(1, year)
    synthetic_r2 = _build_synthetic_result(2, year)
    empty_ds = pd.DataFrame()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_registry = {c.key: c for c in get_active_candidates(1, year=year)}

    # ── Round 1 (true forward — no election result likelihood) ──
    import co_president.model_round1 as m1  # noqa: PLC0415

    logger.info("Building R1 model without election result (true forward)...")
    model_r1 = m1.build_round1_model(
        round1_df,
        None,
        CONFIG,
        features=None,
        digital_signals=empty_ds,
        year=year,
    )
    logger.info(
        "Sampling R1 (%d draws x %d chains)...",
        CONFIG.mcmc_draws,
        CONFIG.mcmc_chains,
    )
    t0 = datetime.now(UTC)
    idata_r1 = m1.sample_round1(model_r1, CONFIG)
    elapsed_r1_s = (datetime.now(UTC) - t0).total_seconds()
    logger.info("R1 sampling done in %.0fs", elapsed_r1_s)

    r1_rhat, r1_converged = check_convergence(idata_r1, "R1")

    # Compute R1 metrics against synthetic zero-actuals so the
    # candidate table shows forecasted vote shares (error = prediction).
    r1_metrics = compute_r1_accuracy(
        idata_r1,
        synthetic_r1,
        poll_columns=set(round1_df.columns),
        candidate_registry=candidate_registry,
    )
    r1_metrics["posterior_summaries"] = _extract_posterior_summaries(idata_r1, None)

    # ── Round 2 (empirical runoff matrix) ──
    empirical_betas = get_empirical_runoff_betas(year=2026, force_rebuild=False)
    logger.info("Loaded %d empirical Beta posteriors", len(empirical_betas))

    _idata_runoff, r2_metrics, r2_rhat, r2_converged, elapsed_r2_s = run_round2_survey(
        CONFIG,
        clean_polls,
        synthetic_r1,
        synthetic_r2,
        idata_r1,
        empty_ds,
        features=None,
        year=year,
        empirical_betas=empirical_betas,
    )

    # ── Reports ──
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
    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()

    forward_md = results_dir / f"forward_{year}_report.md"
    forward_json = results_dir / f"forward_{year}_report.json"
    forward_md.write_text(report_md, encoding="utf-8")

    save_reports(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        n_polls_r1,
        n_polls_r2,
        CONFIG,
        forward_md,
        forward_json,
        year=year,
    )

    print_summary(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        forward_md,
        forward_json,
        r1_converged=r1_converged,
        r2_converged=r2_converged,
    )

    # ── Runoff forecast JSON (extra output for 2026 forward forecast) ──
    runoff_json_path = results_dir / "runoff_forecast_2026_report.json"
    runoff_data: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "election_year": year,
        "config": {
            "mcmc_draws": CONFIG.mcmc_draws,
            "mcmc_tune": CONFIG.mcmc_tune,
            "mcmc_chains": CONFIG.mcmc_chains,
            "seed": CONFIG.seed,
            "target_accept": CONFIG.target_accept,
            "nuts_sampler": CONFIG.nuts_sampler,
        },
        "runoff_metrics": r2_metrics,
        "n_empirical_betas": len(empirical_betas),
        "convergence": {
            "r1_rhat": round(r1_rhat, 6),
            "r1_converged": r1_converged,
            "r2_rhat": round(r2_rhat, 6) if pd.notna(r2_rhat) else None,
            "r2_converged": r2_converged,
        },
        "timing": {
            "elapsed_r1_s": round(elapsed_r1_s, 1),
            "elapsed_r2_s": round(elapsed_r2_s, 1),
            "total_s": round(total_elapsed, 1),
        },
    }
    runoff_json_path.write_text(
        json.dumps(runoff_data, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved runoff forecast JSON to %s", runoff_json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Forward backtest failed")
        sys.exit(1)
