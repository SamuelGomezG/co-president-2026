"""SPEC-42: True forward backtest for 2026 Colombian presidential election.

Runs R1 model without election result, then empirical runoff matrix from
SPEC-34, producing final forecasts with statistical significance.

Output files
------------
``results/forward_2026_report.{md,json}``
    Standard forward backtest report: R1 posterior means, convergence
    diagnostics, MCMC configuration, and model parameter summaries.
``results/runoff_forecast_2026_report.{md,json}``
    Final runoff forecast with 95% credible intervals, top-two candidate
    identification, per-pairing win probabilities, and a statistical
    significance statement ("Candidate X has posterior probability >95%
    of winning the runoff" when applicable).

Methodology
-----------
1. **R1 model**: Dirichlet-Multinomial with reverse-time random walk,
   **without** election result likelihood — true forward forecast with
   no look-ahead bias.
2. **Runoff matrix**: Empirical Beta posteriors (Jeffreys prior) from
   CNE 2026 head-to-head polling data (SPEC-34), combined with the R1
   posterior via :func:`~co_president.model_runoff_matrix.estimate_runoff_matrix`.
3. **Top-two identification**: :func:`~co_president.model_runoff_matrix.compute_top_two_probabilities`
   ranks candidates by posterior probability of finishing 1st and 2nd
   (no canonical results used — this is a true forward forecast).
4. **95% credible intervals**: Computed via Monte Carlo sampling
   (n = 100 000) from the empirical Beta posteriors for each runoff
   pairing.
5. **Statistical significance**: A candidate is declared statistically
   significant when their posterior win probability exceeds 95% for
   a given runoff pairing.

Design notes
------------
- Google Trends data is optional; the script warns and continues if
  fetch fails (2026 trends may not be fully available yet).
- `estimate_runoff_matrix` is called directly (not through
  ``run_round2_survey``) to pass ``empirical_betas``.
- A minimal synthetic ``RoundResult`` is constructed for structural
  fields (date, registered_voters, etc.) since no canonical 2026
  election results exist.
- Top two candidates are identified from the R1 posterior, not from
  historical data — this is a genuine forward forecast.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from co_president.config import ModelConfig, get_active_candidates, get_election_date
from co_president.data import RoundResult, load_and_clean_all
from co_president.empirical_runoff import get_empirical_runoff_betas
from co_president.fundamentals.features import load_features
from co_president.ingestion.ingest_trends import compute_prop_fav, fetch_trends
from co_president.ingestion.trends_keywords import CANDIDATE_QUERY_MAPS
from co_president.model_runoff_matrix import (
    _get_candidate_order,
    compute_top_two_probabilities,
    estimate_runoff_matrix,
)
from report_utils import check_convergence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forward_2026")

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

_EMPIRICAL_N_SAMPLES: int = 100_000
_EMPIRICAL_SEED: int = 998877


# ═══════════════════════════════════════════════════════════════════
# Helpers (same patterns as forward_backtest.py)
# ═══════════════════════════════════════════════════════════════════


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
                "Failed to fetch Google Trends for %d; continuing without digital signals",
                year,
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
    """Extract model parameter posteriors (phi_elec, phi_digital_base) for report.

    Args:
        idata_r1: Round 1 posterior ``DataTree``.
        features: Optional municipal feature matrix.

    Returns:
        Dict with optional ``phi_elec``, ``phi_digital_base``, and
        ``internet_rate`` keys.

    """
    summaries: dict = {}

    try:
        phi_elec_samples = idata_r1.posterior["phi_elec"].to_numpy().flatten()
        ci = az.hdi(phi_elec_samples, prob=0.95)
        summaries["phi_elec"] = {
            "mean": round(float(phi_elec_samples.mean()), 1),
            "std": round(float(phi_elec_samples.std()), 1),
            "ci_95_lower": round(float(ci[0]), 1),
            "ci_95_upper": round(float(ci[1]), 1),
        }
    except (KeyError, ValueError, TypeError):
        summaries["phi_elec"] = {}

    try:
        phi_digital_samples = idata_r1.posterior["phi_digital_base"].to_numpy().flatten()
        ci = az.hdi(phi_digital_samples, prob=0.95)
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


def _extract_r1_posterior_means(
    idata_r1: object,
    year: int,
) -> dict[str, float]:
    """Extract round 1 posterior mean vote shares.

    Args:
        idata_r1: Round 1 posterior ``DataTree``.
        year: Election year.

    Returns:
        Dict mapping candidate key to posterior mean vote share.

    """
    from co_president.model_utils import get_election_day_array  # noqa: PLC0415

    election_day = get_election_day_array(idata_r1)
    candidate_keys = _get_candidate_order(year)

    means: dict[str, float] = {}
    n_posterior = election_day.shape[-1]
    for i, key in enumerate(candidate_keys[:n_posterior]):
        samples = election_day[:, :, i].to_numpy().flatten()
        means[key] = float(samples.mean())

    return means


# ═══════════════════════════════════════════════════════════════════
# Runoff CI computation
# ═══════════════════════════════════════════════════════════════════


def _compute_runoff_ci(
    empirical_betas: dict[tuple[str, str], tuple[float, float]],
    first: str,
    second: str,
    n_samples: int = _EMPIRICAL_N_SAMPLES,
    seed: int = _EMPIRICAL_SEED,
) -> tuple[float, float, float, np.ndarray]:
    """Compute runoff win probability and 95% CI from empirical Beta.

    Resolves the canonical ``(candidate_a, candidate_b)`` key from
    ``empirical_betas`` and draws from the Beta(alpha, beta) posterior
    to estimate win probability and a 95% percentile interval for the
    vote share of ``first``.

    Falls back to Jeffreys Beta(0.5, 0.5) when the pairing is not in
    ``empirical_betas`` (effectively a uniform prior with no data).

    Args:
        empirical_betas: Dict mapping ``(candidate_a, candidate_b)`` to
            ``(alpha, beta)``.
        first: First candidate key.
        second: Second candidate key.
        n_samples: Number of Monte Carlo samples.
        seed: RNG seed.

    Returns:
        Tuple of ``(prob_first_wins, ci_lower, ci_upper, samples)``.
        *ci_lower* and *ci_upper* are the 2.5th and 97.5th percentiles
        of the Beta posterior for ``first``'s vote share.

    """
    key = (first, second) if first <= second else (second, first)
    alpha, beta_val = empirical_betas.get(key, (0.5, 0.5))

    rng = np.random.default_rng(seed=seed)
    samples = rng.beta(alpha, beta_val, size=n_samples)
    prob_first_wins = float((samples > 0.5).mean())
    ci_lower = float(np.percentile(samples, 2.5))
    ci_upper = float(np.percentile(samples, 97.5))
    return prob_first_wins, ci_lower, ci_upper, samples


# ═══════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════


def _format_pct(value: float) -> str:
    """Format a proportion as a percentage string."""
    if not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


# ═══════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════


def _generate_runoff_forecast_report(
    r1_means: dict[str, float],
    matrix: object,
    top_two_probs: dict[tuple[str, str], float],
    empirical_betas: dict[tuple[str, str], tuple[float, float]],
    r1_rhat: float,
    r1_converged: bool,
    elapsed_r1_s: float,
    elapsed_r2_s: float,
    n_polls_r1: int,
    n_polls_r2: int,
    posterior_summaries: dict,
    config: ModelConfig,
    year: int,
) -> str:
    """Generate the runoff forecast Markdown report.

    Includes: R1 posterior means, top-two identification, per-pairing
    win probabilities with 95% credible intervals, statistical
    significance statement, convergence diagnostics, MCMC configuration,
    and model parameter summaries.

    Args:
        r1_means: Round 1 posterior mean vote shares per candidate.
        matrix: ``RunoffMatrix`` from ``estimate_runoff_matrix``.
        top_two_probs: Top-two pairing probabilities from posterior.
        empirical_betas: Empirical Beta posteriors dict.
        r1_rhat: Round 1 max R-hat.
        r1_converged: Whether Round 1 converged.
        elapsed_r1_s: Round 1 sampling wall-clock time.
        elapsed_r2_s: Runoff matrix computation time.
        n_polls_r1: Number of round 1 polls.
        n_polls_r2: Number of runoff polls.
        posterior_summaries: Model parameter posterior summaries.
        config: Model configuration.
        year: Election year.

    Returns:
        Markdown-formatted report string.

    """
    candidate_registry = {c.key: c for c in get_active_candidates(1, year=year)}

    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    total_samples = config.mcmc_draws * config.mcmc_chains
    lines: list[str] = []
    lines.append(f"# {year} Runoff Forecast Report\n")
    lines.append(f"**Generated**: {now_str}\n")
    lines.append(
        f"**Configuration**: {config.mcmc_draws} draws x {config.mcmc_chains} "
        f"chains = {total_samples} total samples  |  seed={config.seed}  |  "
        f"target_accept={config.target_accept}\n",
    )

    # ── Round 1 Posterior Summary ──
    lines.append("## Round 1 Posterior Summary\n")
    lines.append("| Candidate | Posterior Mean |")
    lines.append("|-----------|---------------:|")
    for key, mean_val in sorted(r1_means.items(), key=lambda x: x[1], reverse=True):
        name = candidate_registry.get(key)
        display = name.display_name if name else key
        lines.append(f"| {display} | {_format_pct(mean_val)} |")
    lines.append("")

    # ── Top Two Identification ──
    lines.append("## Top Two Candidate Identification\n")
    sorted_pairs = sorted(top_two_probs.items(), key=lambda x: x[1], reverse=True)
    if sorted_pairs:
        (first, second), top_prob = sorted_pairs[0]
        name_first = candidate_registry.get(first)
        name_second = candidate_registry.get(second)
        display_first = name_first.display_name if name_first else first
        display_second = name_second.display_name if name_second else second
        lines.append(
            f"The posterior identifies **{display_first}** and **{display_second}** "
            f"as the most likely runoff pairing "
            f"({_format_pct(top_prob)} probability).\n",
        )

        lines.append("### Top-Two Pairing Probabilities\n")
        lines.append("| Pairing | Probability |")
        lines.append("|---------|------------:|")
        for (c1, c2), prob in sorted_pairs[:5]:
            n1 = candidate_registry.get(c1)
            n2 = candidate_registry.get(c2)
            d1 = n1.display_name if n1 else c1
            d2 = n2.display_name if n2 else c2
            lines.append(f"| {d1} vs {d2} | {_format_pct(prob)} |")
        lines.append("")
    else:
        lines.append("No top-two pairings identified (insufficient posterior data).\n")

    # ── Runoff Win Probabilities ──
    lines.append("## Runoff Win Probabilities\n")
    lines.append(
        "| Pairing | P(Pairing) | P(First Wins) | P(Second Wins) | 95% CI (First Share) |",
    )
    lines.append(
        "|---------|-----------:|-------------:|---------------:|----------------------:|",
    )

    for pf in matrix.pairings:
        n_first = candidate_registry.get(pf.candidate_first)
        n_second = candidate_registry.get(pf.candidate_second)
        d_first = n_first.display_name if n_first else pf.candidate_first
        d_second = n_second.display_name if n_second else pf.candidate_second

        # Compute 95% CI: prefer empirical betas, then K=3 model posterior
        if empirical_betas:
            _, ci_low, ci_high, _ = _compute_runoff_ci(
                empirical_betas,
                pf.candidate_first,
                pf.candidate_second,
            )
            ci_str = f"[{_format_pct(ci_low)}, {_format_pct(ci_high)}]"
        elif pf.idata is not None:
            try:
                p_time = pf.idata.posterior["p_time"]
                first_samples = p_time[:, :, 0, :].to_numpy().flatten()
                ci = az.hdi(first_samples, prob=0.95)
                ci_str = f"[{_format_pct(float(ci[0]))}, {_format_pct(float(ci[1]))}]"
            except (KeyError, ValueError, TypeError):
                ci_str = "N/A"
        else:
            ci_str = "N/A"

        lines.append(
            f"| {d_first} vs {d_second} | "
            f"{_format_pct(pf.prob_pairing)} | "
            f"{_format_pct(pf.prob_first_wins)} | "
            f"{_format_pct(pf.prob_second_wins)} | "
            f"{ci_str} |",
        )
    lines.append("")

    # ── Statistical Significance ──
    lines.append("## Statistical Significance\n")
    sig_found = False
    for pf in matrix.pairings:
        if pf.prob_first_wins > 0.95:
            n_first = candidate_registry.get(pf.candidate_first)
            n_second = candidate_registry.get(pf.candidate_second)
            d_first = n_first.display_name if n_first else pf.candidate_first
            d_second = n_second.display_name if n_second else pf.candidate_second
            lines.append(
                f"**{d_first}** has posterior probability "
                f"**>95%** of winning the runoff against **{d_second}**.\n",
            )
            sig_found = True
        elif pf.prob_second_wins > 0.95:
            n_first = candidate_registry.get(pf.candidate_first)
            n_second = candidate_registry.get(pf.candidate_second)
            d_first = n_first.display_name if n_first else pf.candidate_first
            d_second = n_second.display_name if n_second else pf.candidate_second
            lines.append(
                f"**{d_second}** has posterior probability "
                f"**>95%** of winning the runoff against **{d_first}**.\n",
            )
            sig_found = True
    if not sig_found:
        lines.append(
            "No candidate reaches the 95% statistical significance threshold "
            "for any plausible runoff pairing. The forecast should be "
            "interpreted with appropriate uncertainty.\n",
        )
    lines.append("")

    # ── Data Summary ──
    lines.append("## Data Summary\n")
    lines.append(f"- **Round 1 polls**: {n_polls_r1}")
    lines.append(f"- **Runoff polls**: {n_polls_r2}")
    n_betas = len(empirical_betas)
    if n_betas > 0:
        lines.append(f"- **Empirical Beta posteriors**: {n_betas} pairings from CNE 2026 data")
    else:
        lines.append("- **Empirical Beta posteriors**: None (CNE 2026 data unavailable)")
    lines.append("")

    # ── Convergence Diagnostics ──
    lines.append("## Convergence Diagnostics\n")
    lines.append(
        f"- **R1 max R-hat**: {r1_rhat:.4f}  (converged: {'Y' if r1_converged else 'N'})",
    )
    if r1_converged:
        lines.append("- **R1 status**: Chains converged (R-hat < 1.10)")
    else:
        lines.append(
            "- **R1 status**: **WARNING** — chains did not fully converge"
            " (R-hat >= 1.10). Results should be interpreted with caution.",
        )
    lines.append("- **Runoff**: N/A (empirical path — no MCMC sampling required)")
    lines.append("")

    # ── MCMC Configuration ──
    lines.append("## MCMC Configuration\n")
    lines.append(f"- **Sampler**: {config.nuts_sampler or 'pymc'}")
    lines.append(f"- **Draws per chain**: {config.mcmc_draws}")
    lines.append(f"- **Tune steps**: {config.mcmc_tune}")
    lines.append(f"- **Chains**: {config.mcmc_chains}")
    lines.append(f"- **Target accept**: {config.target_accept}")
    lines.append(f"- **Seed**: {config.seed}")
    lines.append(f"- **R1 sampling time**: {elapsed_r1_s:.0f}s")
    lines.append(f"- **Runoff matrix time**: {elapsed_r2_s:.0f}s")
    lines.append("")

    # ── Model Parameters ──
    ps = posterior_summaries
    phi_elec = ps.get("phi_elec", {})
    phi_digital = ps.get("phi_digital_base", {})
    internet_rate = ps.get("internet_rate")

    if phi_elec or phi_digital or internet_rate is not None:
        lines.append("## Model Parameter Summaries\n")
        if phi_elec:
            lines.append(
                f"- **φ_elec**: mean={phi_elec['mean']:.1f},"
                f" std={phi_elec['std']:.1f},"
                f" 95% HDI=[{phi_elec['ci_95_lower']:.1f},"
                f" {phi_elec['ci_95_upper']:.1f}]",
            )
        if phi_digital:
            lines.append(
                f"- **φ_digital_base**: mean={phi_digital['mean']:.1f},"
                f" std={phi_digital['std']:.1f},"
                f" 95% HDI=[{phi_digital['ci_95_lower']:.1f},"
                f" {phi_digital['ci_95_upper']:.1f}]",
            )
        if internet_rate is not None:
            lines.append(f"- **Internet access rate**: {internet_rate:.2%}")
        lines.append("")

    # ── Methodology ──
    lines.append("## Methodology\n")
    lines.append(
        "1. **R1 model**: Dirichlet-Multinomial with reverse-time random walk"
        " (no election result likelihood — true forward forecast).\n",
    )
    lines.append(
        "2. **Runoff matrix**: Empirical Beta posteriors (Jeffreys Beta(0.5, 0.5)"
        " prior) from CNE 2026 head-to-head polling data (SPEC-34), combined"
        " with the R1 posterior via `estimate_runoff_matrix`.\n",
    )
    lines.append(
        "3. **Top-two identification**: `compute_top_two_probabilities` from"
        " the R1 posterior ranks candidates by probability of finishing 1st"
        " and 2nd (no canonical results used).\n",
    )
    lines.append(
        f"4. **95% credible intervals**: Computed via Monte Carlo sampling"
        f" (n={_EMPIRICAL_N_SAMPLES:,}) from the empirical Beta posteriors"
        f" for each pairing.\n",
    )
    lines.append(
        "5. **Statistical significance**: A candidate is declared statistically"
        " significant when their posterior win probability exceeds 95% for a"
        " given runoff pairing.\n",
    )
    lines.append("")

    return "\n".join(lines)


def _build_runoff_forecast_json(
    r1_means: dict[str, float],
    matrix: object,
    top_two_probs: dict[tuple[str, str], float],
    empirical_betas: dict[tuple[str, str], tuple[float, float]],
    r1_rhat: float,
    r1_converged: bool,
    elapsed_r1_s: float,
    elapsed_r2_s: float,
    n_polls_r1: int,
    n_polls_r2: int,
    posterior_summaries: dict,
    config: ModelConfig,
    year: int,
) -> dict:
    """Build JSON-serializable runoff forecast report.

    Args:
        r1_means: Round 1 posterior mean vote shares per candidate.
        matrix: ``RunoffMatrix`` from ``estimate_runoff_matrix``.
        top_two_probs: Top-two pairing probabilities.
        empirical_betas: Empirical Beta posteriors.
        r1_rhat: Round 1 max R-hat.
        r1_converged: Whether Round 1 converged.
        elapsed_r1_s: Round 1 sampling wall-clock time.
        elapsed_r2_s: Runoff matrix computation time.
        n_polls_r1: Number of round 1 polls.
        n_polls_r2: Number of runoff polls.
        posterior_summaries: Model parameter posterior summaries.
        config: Model configuration.
        year: Election year.

    Returns:
        JSON-serializable dict.

    """
    sorted_pairs = sorted(top_two_probs.items(), key=lambda x: x[1], reverse=True)
    top_pair = sorted_pairs[0] if sorted_pairs else None

    pairings_data: list[dict] = []
    sig_candidates: list[dict] = []
    for pf in matrix.pairings:
        if empirical_betas:
            _, ci_low, ci_high, _ = _compute_runoff_ci(
                empirical_betas,
                pf.candidate_first,
                pf.candidate_second,
            )
        elif pf.idata is not None:
            try:
                p_time = pf.idata.posterior["p_time"]
                first_samples = p_time[:, :, 0, :].to_numpy().flatten()
                ci = az.hdi(first_samples, prob=0.95)
                ci_low = float(ci[0])
                ci_high = float(ci[1])
            except (KeyError, ValueError, TypeError):
                ci_low = float("nan")
                ci_high = float("nan")
        else:
            ci_low = float("nan")
            ci_high = float("nan")

        if pf.prob_first_wins > 0.95:
            sig_candidates.append(
                {
                    "candidate": pf.candidate_first,
                    "against": pf.candidate_second,
                    "probability": round(pf.prob_first_wins, 6),
                },
            )
        elif pf.prob_second_wins > 0.95:
            sig_candidates.append(
                {
                    "candidate": pf.candidate_second,
                    "against": pf.candidate_first,
                    "probability": round(pf.prob_second_wins, 6),
                },
            )

        pairings_data.append(
            {
                "candidate_first": pf.candidate_first,
                "candidate_second": pf.candidate_second,
                "prob_pairing": round(pf.prob_pairing, 6),
                "prob_first_wins": round(pf.prob_first_wins, 6),
                "prob_second_wins": round(pf.prob_second_wins, 6),
                "mean_margin": round(pf.mean_margin, 6),
                "mean_share_first": round(pf.mean_share_first, 6),
                "mean_share_second": round(pf.mean_share_second, 6),
                "ci_95_lower": round(ci_low, 6),
                "ci_95_upper": round(ci_high, 6),
            },
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "election_year": year,
        "config": {
            "mcmc_draws": config.mcmc_draws,
            "mcmc_tune": config.mcmc_tune,
            "mcmc_chains": config.mcmc_chains,
            "seed": config.seed,
            "target_accept": config.target_accept,
            "nuts_sampler": config.nuts_sampler,
        },
        "round1": {
            "posterior_means": {k: round(v, 6) for k, v in r1_means.items()},
            "top_pairing": list(top_pair[0]) if top_pair else [],
            "top_pairing_probability": round(top_pair[1], 6) if top_pair else 0.0,
        },
        "top_two_probabilities": {f"{c1}__vs__{c2}": round(p, 6) for (c1, c2), p in sorted_pairs},
        "runoff_pairings": pairings_data,
        "statistical_significance": sig_candidates,
        "data": {
            "n_polls_round1": n_polls_r1,
            "n_polls_round2": n_polls_r2,
            "n_empirical_betas": len(empirical_betas),
        },
        "convergence": {
            "r1_rhat": round(r1_rhat, 6),
            "r1_converged": r1_converged,
            "runoff": "N/A (empirical path)",
        },
        "model_parameters": posterior_summaries,
        "timing": {
            "elapsed_r1_s": round(elapsed_r1_s, 1),
            "elapsed_r2_s": round(elapsed_r2_s, 1),
        },
    }


def _write_forward_report(
    md_path: Path,
    json_path: Path,
    r1_means: dict[str, float],
    r1_rhat: float,
    r1_converged: bool,
    elapsed_r1_s: float,
    elapsed_r2_s: float,
    n_polls_r1: int,
    n_polls_r2: int,
    posterior_summaries: dict,
    config: ModelConfig,
    year: int,
) -> None:
    """Write standard forward backtest report (MD + JSON).

    Args:
        md_path: Path for Markdown report.
        json_path: Path for JSON report.
        r1_means: Round 1 posterior means.
        r1_rhat: Round 1 max R-hat.
        r1_converged: Whether Round 1 converged.
        elapsed_r1_s: Round 1 sampling time.
        elapsed_r2_s: Runoff matrix computation time.
        n_polls_r1: Number of round 1 polls.
        n_polls_r2: Number of runoff polls.
        posterior_summaries: Model parameter posterior summaries.
        config: Model configuration.
        year: Election year.

    """
    candidate_registry = {c.key: c for c in get_active_candidates(1, year=year)}

    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    total_samples = config.mcmc_draws * config.mcmc_chains

    lines: list[str] = []
    lines.append(f"# {year} Forward Backtest Report\n")
    lines.append(f"**Generated**: {now_str}\n")
    lines.append(
        f"**Configuration**: {config.mcmc_draws} draws x {config.mcmc_chains} "
        f"chains = {total_samples} total samples  |  seed={config.seed}\n",
    )
    lines.append("")

    lines.append("## Data Summary\n")
    lines.append(f"- **Round 1 polls**: {n_polls_r1}")
    lines.append(f"- **Runoff polls**: {n_polls_r2}")
    lines.append(f"- **R1 sampling time**: {elapsed_r1_s:.0f}s")
    lines.append(f"- **Runoff matrix time**: {elapsed_r2_s:.0f}s")
    lines.append(
        f"- **R1 R-hat**: {r1_rhat:.4f}  (converged: {'Y' if r1_converged else 'N'})",
    )
    lines.append("")

    lines.append("## Round 1 Posterior Means\n")
    lines.append("| Candidate | Posterior Mean |")
    lines.append("|-----------|---------------:|")
    for key, mean_val in sorted(r1_means.items(), key=lambda x: x[1], reverse=True):
        name = candidate_registry.get(key)
        display = name.display_name if name else key
        lines.append(f"| {display} | {_format_pct(mean_val)} |")
    lines.append("")

    ps = posterior_summaries
    phi_elec = ps.get("phi_elec", {})
    phi_digital = ps.get("phi_digital_base", {})
    internet_rate = ps.get("internet_rate")

    if phi_elec or phi_digital or internet_rate is not None:
        lines.append("## Model Parameters\n")
        if phi_elec:
            lines.append(
                f"- **φ_elec**: mean={phi_elec['mean']:.1f},"
                f" std={phi_elec['std']:.1f},"
                f" 95% HDI=[{phi_elec['ci_95_lower']:.1f},"
                f" {phi_elec['ci_95_upper']:.1f}]",
            )
        if phi_digital:
            lines.append(
                f"- **φ_digital_base**: mean={phi_digital['mean']:.1f},"
                f" std={phi_digital['std']:.1f},"
                f" 95% HDI=[{phi_digital['ci_95_lower']:.1f},"
                f" {phi_digital['ci_95_upper']:.1f}]",
            )
        if internet_rate is not None:
            lines.append(f"- **Internet access rate**: {internet_rate:.2%}")
        lines.append("")

    lines.append("## MCMC Configuration\n")
    lines.append(f"- **Sampler**: {config.nuts_sampler or 'pymc'}")
    lines.append(f"- **Draws**: {config.mcmc_draws} x {config.mcmc_chains} chains")
    lines.append(f"- **Tune**: {config.mcmc_tune}")
    lines.append(f"- **Target accept**: {config.target_accept}")
    lines.append(f"- **Seed**: {config.seed}")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved forward report to %s", md_path)

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
        "round1_posterior_means": {k: round(v, 6) for k, v in r1_means.items()},
        "convergence": {
            "r1_rhat": round(r1_rhat, 6),
            "r1_converged": r1_converged,
        },
        "data": {
            "n_polls_round1": n_polls_r1,
            "n_polls_round2": n_polls_r2,
        },
        "model_parameters": posterior_summaries,
        "timing": {
            "elapsed_r1_s": round(elapsed_r1_s, 1),
            "elapsed_r2_s": round(elapsed_r2_s, 1),
        },
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    logger.info("Saved forward JSON to %s", json_path)


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Run true forward backtest for 2026 with empirical runoff matrix."""
    parser = argparse.ArgumentParser(
        description="Run 2026 true forward backtest with empirical runoff matrix.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Election year (default 2026)",
    )
    args = parser.parse_args()
    year = args.year

    started_at = datetime.now(UTC)
    logger.info("Starting %d true forward backtest...", year)

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

    query_map = CANDIDATE_QUERY_MAPS.get(str(year), {})
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
            "Digital signals unavailable for %d; model will proceed without Google Trends data",
            year,
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

    # R1 convergence
    r1_rhat, r1_converged = check_convergence(idata_r1, "R1")

    # Extract R1 posterior means and parameter summaries
    r1_means = _extract_r1_posterior_means(idata_r1, year)
    posterior_summaries = _extract_posterior_summaries(idata_r1, features)

    # ── Empirical runoff betas ──
    logger.info("Loading empirical runoff betas...")
    empirical_betas = get_empirical_runoff_betas(year=year, force_rebuild=False)
    logger.info("Loaded %d empirical Beta posteriors", len(empirical_betas))

    # ── Runoff matrix ──
    logger.info("Computing runoff pairing matrix with empirical betas...")

    # Construct minimal synthetic RoundResult for structural fields.
    # estimate_runoff_matrix uses transfer-implied shares (always loaded),
    # so get_share() is never called on this object. Only date,
    # registered_voters, and polling_stations are read.
    synthetic_r1 = RoundResult(
        round_number=1,
        date=election_r1,
        total_valid_votes=1,
        total_votes_incl_blank=1,
        registered_voters=1,
        polling_stations=1,
        candidates=(),
        blank_votes=0,
        null_votes=0,
        unmarked_votes=0,
    )

    t0 = datetime.now(UTC)
    matrix = estimate_runoff_matrix(
        idata_r1,
        (synthetic_r1, synthetic_r1),
        clean_polls.round2,
        CONFIG,
        candidate_keys=_get_candidate_order(year),
        features=features,
        digital_signals=digital_signals_runoff,
        year=year,
        empirical_betas=empirical_betas,
    )
    elapsed_r2_s = (datetime.now(UTC) - t0).total_seconds()
    logger.info("Runoff matrix computed in %.0fs", elapsed_r2_s)

    # Identify top two from posterior (no canonical results)
    candidate_keys = _get_candidate_order(year)
    candidates_for_top_two = [k for k in candidate_keys if k not in ("rest", "blanco")]
    top_two_probs = compute_top_two_probabilities(
        idata_r1,
        candidates_for_top_two,
        candidate_keys=candidate_keys,
        year=year,
    )

    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()

    # ── Output files ──
    forward_md_path = results_dir / f"forward_{year}_report.md"
    forward_json_path = results_dir / f"forward_{year}_report.json"
    runoff_md_path = results_dir / "runoff_forecast_2026_report.md"
    runoff_json_path = results_dir / "runoff_forecast_2026_report.json"

    # 1. Forward backtest report (standard format)
    _write_forward_report(
        forward_md_path,
        forward_json_path,
        r1_means,
        r1_rhat,
        r1_converged,
        elapsed_r1_s,
        elapsed_r2_s,
        n_polls_r1,
        n_polls_r2,
        posterior_summaries,
        CONFIG,
        year,
    )

    # 2. Runoff forecast report (with statistical significance)
    runoff_md = _generate_runoff_forecast_report(
        r1_means,
        matrix,
        top_two_probs,
        empirical_betas,
        r1_rhat,
        r1_converged,
        elapsed_r1_s,
        elapsed_r2_s,
        n_polls_r1,
        n_polls_r2,
        posterior_summaries,
        CONFIG,
        year,
    )
    runoff_md_path.write_text(runoff_md, encoding="utf-8")
    logger.info("Saved runoff forecast report to %s", runoff_md_path)

    runoff_json = _build_runoff_forecast_json(
        r1_means,
        matrix,
        top_two_probs,
        empirical_betas,
        r1_rhat,
        r1_converged,
        elapsed_r1_s,
        elapsed_r2_s,
        n_polls_r1,
        n_polls_r2,
        posterior_summaries,
        CONFIG,
        year,
    )
    runoff_json_path.write_text(
        json.dumps(runoff_json, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved runoff forecast JSON to %s", runoff_json_path)

    # Summary
    logger.info("=" * 60)
    logger.info("2026 forward backtest complete in %.0fs", total_elapsed)
    logger.info("Forward report: %s", forward_md_path)
    logger.info("Forward JSON:   %s", forward_json_path)
    logger.info("Runoff report:  %s", runoff_md_path)
    logger.info("Runoff JSON:    %s", runoff_json_path)
    logger.info(
        "R1 polls=%d, R2 polls=%d, Empirical betas=%d, R1 R-hat=%.4f (converged=%s)",
        n_polls_r1,
        n_polls_r2,
        len(empirical_betas),
        r1_rhat,
        r1_converged,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
