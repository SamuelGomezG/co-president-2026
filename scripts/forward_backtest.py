"""True forward backtest: R1 without election result, then honest runoff."""

import warnings

warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, arviz as az, logging
from pathlib import Path

logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("forward")

from co_president.config import ModelConfig, FIRST_ROUND_CANDIDATES, ELECTION_DATE_ROUND2
from co_president.data import (
    load_and_clean_all,
    load_canonical_results,
    CandidateResult,
    RoundResult,
)
from co_president.model_round1 import build_round1_model, sample_round1
from co_president.model_runoff_simple import (
    build_runoff_simple_model,
    sample_runoff,
    forecast_runoff_simple,
)
from co_president.model_runoff_matrix import _filter_polls_for_pairing, _compute_transfer_shares
from co_president.model_transfer import sample_transfer_rates
from co_president.model_utils import get_election_day_array

CONFIG = ModelConfig(
    mcmc_draws=2000,
    mcmc_tune=1000,
    mcmc_chains=4,
    mcmc_cores=4,
    target_accept=0.95,
    seed=332211,
    nuts_sampler="numpyro",
)


def main():
    logger.info("Loading data...")
    cp = load_and_clean_all()
    r1, r2 = load_canonical_results()
    ds = pd.read_parquet("results/trends_cache_2022.parquet")
    ds["fecha"] = pd.to_datetime(ds["fecha"])

    # ── R1 model WITHOUT election result (true forward) ──
    logger.info("Building R1 model without election result...")
    model_r1 = build_round1_model(cp.round1, results=None, config=CONFIG, digital_signals=ds)
    logger.info(f"Sampling R1 {CONFIG.mcmc_draws}x{CONFIG.mcmc_chains}...")
    idata_r1 = sample_round1(model_r1, CONFIG)

    # R1 accuracy
    from report_utils import compute_r1_accuracy  # type: ignore[import-unused]

    r1_metrics = compute_r1_accuracy(idata_r1, r1, poll_columns=set(cp.round1.columns))
    rhat = list(az.rhat(idata_r1.posterior).values())
    r1_max_rhat = max(float(np.max(v.values)) for v in rhat)
    logger.info(f"R1 forward MAE: {r1_metrics.get('mae', 0) * 100:.2f}pp")
    logger.info(f"R1 forward max R-hat: {r1_max_rhat:.4f}")

    # ── Honest Runoff ──
    logger.info("Computing runoff pairing matrix...")
    tr = sample_transfer_rates(features=None, config=CONFIG)
    ed = get_election_day_array(idata_r1)
    n_dim = ed.shape[-1]
    all_keys = sorted(set(FIRST_ROUND_CANDIDATES.keys()) & set(cp.round1.columns))[:n_dim]
    sf, ss = _compute_transfer_shares(ed, all_keys, "gustavo_petro", "rodolfo_hernandez", tr)
    f1, f2 = float(sf.mean()), float(ss.mean())
    logger.info(f"Transfer prior: Petro={f1:.4f} Rodolfo={f2:.4f}")

    sc = (
        CandidateResult("gustavo_petro", int(f1 * 100000), f1),
        CandidateResult("rodolfo_hernandez", int(f2 * 100000), f2),
    )
    sr = RoundResult(
        1,
        r1.date,
        int((f1 + f2) * 100000),
        int((f1 + f2) * 100000),
        r1.registered_voters,
        r1.polling_stations,
        sc,
        0,
        0,
        0,
    )
    polls = _filter_polls_for_pairing(cp.round2, "gustavo_petro", "rodolfo_hernandez")
    model = build_runoff_simple_model(
        polls, sr, idata_r1, CONFIG, digital_signals=pd.DataFrame(), round2_result=None
    )
    logger.info(f"Sampling runoff {CONFIG.mcmc_draws}x{CONFIG.mcmc_chains}...")
    idata_r2 = sample_runoff(model, CONFIG)
    fc = forecast_runoff_simple(idata_r2, "gustavo_petro", "rodolfo_hernandez")

    r2_rhat_vals = list(az.rhat(idata_r2.posterior).values())
    r2_rhat = max(float(np.max(v.values)) for v in r2_rhat_vals)
    r2_ess_vals = list(az.ess(idata_r2.posterior).values())
    r2_ess_bulk = min(float(np.min(v.values)) for v in r2_ess_vals)

    print(f"\n{'=' * 60}")
    print(f"  TRUE FORWARD BACKTEST (R1 without election result)")
    print(f"{'=' * 60}")
    print(f"\n-- Round 1 --")
    r1_mae = r1_metrics.get("mae", 0) * 100
    r1_rmse = r1_metrics.get("rmse", 0) * 100
    print(f"  MAE:            {r1_mae:.2f}pp")
    print(f"  RMSE:           {r1_rmse:.2f}pp")
    print(f"  Max R-hat:      {r1_max_rhat:.4f}")
    print(f"\n-- Runoff --")
    print(
        f"  Petro:          {fc.mean_share_a * 100:.2f}%  (actual 50.42%)  error {fc.mean_share_a - 0.5042:+.2%}"
    )
    print(
        f"  Rodolfo:        {fc.mean_share_b * 100:.2f}%  (actual 47.35%)  error {fc.mean_share_b - 0.4735:+.2%}"
    )
    print(f"  Rest:           {fc.mean_share_rest * 100:.2f}%  (actual 2.23%)")
    print(f"  Margin:         {fc.mean_margin:+.2%}")
    print(f"  P(Petro wins):  {fc.prob_a_wins:.1%}")
    print(f"  Max R-hat:      {r2_rhat:.4f}")
    print(f"  Min ESS bulk:   {r2_ess_bulk:.0f}")
    print(f"\n-- Compared to backtest WITH R1 election result --")
    print(f"  (previous run: R1 used election likelihood, runoff had no digital signals)")
    print(f"  Full backtest: R1 MAE 1.02pp, Runoff MAE 0.72pp, P(Petro)=57.7%")
    print(f"  Forward:       R1 MAE {r1_metrics.get('mae', 0) * 100:.2f}pp, Runoff MAE ???")


if __name__ == "__main__":
    main()
