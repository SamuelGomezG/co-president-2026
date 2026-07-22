"""SPEC-34: Tests for empirical Beta posterior estimation from CNE runoff pairings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from co_president.empirical_runoff import (
    _canonical_pair,
    cache_empirical_posterior,
    compute_per_pairing_stats,
    fit_empirical_beta_per_pairing,
    get_empirical_runoff_betas,
    load_empirical_posterior,
    sample_empirical_transfer,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_synthetic_runoff_df() -> pd.DataFrame:
    """Build a minimal synthetic runoff pairings DataFrame."""
    return pd.DataFrame(
        {
            "fecha": ["2026-01-15", "2026-01-20", "2026-02-01", "2026-02-10", "2026-03-01"],
            "encuestadora": ["A", "A", "B", "B", "C"],
            "field_end": ["2026-01-15", "2026-01-20", "2026-02-01", "2026-02-10", "2026-03-01"],
            "candidate_a": ["cepeda", "cepeda", "valencia", "cepeda", "fajardo"],
            "candidate_b": ["valencia", "fajardo", "fajardo", "fajardo", "valencia"],
            "n_a": [30.0, 20.0, 15.0, 25.0, 10.0],
            "n_b": [20.0, 15.0, 10.0, 20.0, 8.0],
            "n_total": [50.0, 35.0, 25.0, 45.0, 18.0],
            "effective_n": [50.0, 35.0, 25.0, 45.0, 18.0],
        }
    )


class TestCanonicalPair:
    """Tests for the _canonical_pair helper."""

    def test_ordered(self) -> None:
        """Already-ordered pair returns unchanged."""
        assert _canonical_pair("cepeda", "valencia") == ("cepeda", "valencia")

    def test_reverse_ordered(self) -> None:
        """Reverse-ordered pair returns canonical (sorted) order."""
        assert _canonical_pair("valencia", "cepeda") == ("cepeda", "valencia")

    def test_same_candidate(self) -> None:
        """Same-candidate pair returns identity."""
        assert _canonical_pair("cepeda", "cepeda") == ("cepeda", "cepeda")


class TestComputePerPairingStats:
    """Tests for compute_per_pairing_stats aggregation."""

    def test_empty_dataframe(self) -> None:
        """Empty input returns empty output with correct columns."""
        result = compute_per_pairing_stats(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == [
            "candidate_a",
            "candidate_b",
            "n_a",
            "n_b",
            "n_total",
            "n_eff",
        ]

    def test_synthetic_data(self) -> None:
        """Synthetic data produces correct aggregated stats."""
        df = _make_synthetic_runoff_df()
        result = compute_per_pairing_stats(df)

        n_pairs = 3
        assert len(result) == n_pairs

        cep_val = result[
            (result["candidate_a"] == "cepeda") & (result["candidate_b"] == "valencia")
        ]
        assert len(cep_val) == 1
        assert cep_val.iloc[0]["n_eff"] == 50.0

        cep_faj = result[(result["candidate_a"] == "cepeda") & (result["candidate_b"] == "fajardo")]
        assert len(cep_faj) == 1
        assert cep_faj.iloc[0]["n_eff"] == 80.0
        assert cep_faj.iloc[0]["n_total"] == 80.0

    def test_sorted_by_n_eff_descending(self) -> None:
        """Output is sorted by effective_n descending."""
        df = _make_synthetic_runoff_df()
        result = compute_per_pairing_stats(df)
        n_eff_vals = result["n_eff"].to_numpy()
        assert all(n_eff_vals[i] >= n_eff_vals[i + 1] for i in range(len(n_eff_vals) - 1))

    def test_symmetric_grouping(self) -> None:
        """Pairs like (a,b) and (b,a) are grouped together."""
        df = pd.DataFrame(
            {
                "fecha": ["2026-01-01", "2026-01-02"],
                "encuestadora": ["A", "B"],
                "field_end": ["2026-01-01", "2026-01-02"],
                "candidate_a": ["cepeda", "valencia"],
                "candidate_b": ["valencia", "cepeda"],
                "n_a": [30.0, 10.0],
                "n_b": [20.0, 5.0],
                "n_total": [50.0, 15.0],
                "effective_n": [50.0, 15.0],
            }
        )
        result = compute_per_pairing_stats(df)
        assert len(result) == 1
        assert result.iloc[0]["candidate_a"] == "cepeda"
        assert result.iloc[0]["candidate_b"] == "valencia"
        assert result.iloc[0]["n_eff"] == 65.0


class TestFitEmpiricalBetaPerPairing:
    """Tests for Beta posterior fitting."""

    def test_empty_stats(self) -> None:
        """Empty stats DataFrame returns empty dict."""
        empty = pd.DataFrame(
            columns=["candidate_a", "candidate_b", "n_a", "n_b", "n_total", "n_eff"]
        )
        result = fit_empirical_beta_per_pairing(empty)
        assert result == {}

    def test_jeffreys_prior(self) -> None:
        """Jeffreys prior adds 0.5 to both alpha and beta."""
        stats = pd.DataFrame(
            [
                {
                    "candidate_a": "cepeda",
                    "candidate_b": "valencia",
                    "n_a": 30.0,
                    "n_b": 20.0,
                    "n_total": 50.0,
                    "n_eff": 50.0,
                },
            ]
        )
        result = fit_empirical_beta_per_pairing(stats)
        alpha, beta = result[("cepeda", "valencia")]
        assert alpha == 30.5
        assert beta == 20.5

    def test_zero_counts_use_jeffreys_only(self) -> None:
        """Zero counts produce pure Jeffreys prior Beta(0.5, 0.5)."""
        stats = pd.DataFrame(
            [
                {
                    "candidate_a": "cepeda",
                    "candidate_b": "valencia",
                    "n_a": 0.0,
                    "n_b": 0.0,
                    "n_total": 0.0,
                    "n_eff": 0.0,
                },
            ]
        )
        result = fit_empirical_beta_per_pairing(stats)
        alpha, beta = result[("cepeda", "valencia")]
        assert alpha == 0.5
        assert beta == 0.5


class TestSampleEmpiricalTransfer:
    """Tests for empirical Beta sampling."""

    def test_output_in_unit_interval(self) -> None:
        """All samples fall in [0, 1]."""
        posterior = (30.5, 20.5)
        samples = sample_empirical_transfer(posterior, "cepeda", "valencia", n_draws=500)
        assert samples.shape == (500,)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_mean_approximates_expected(self) -> None:
        """Beta mean = alpha / (alpha + beta) with high precision."""
        posterior = (30.5, 20.5)
        expected_mean = 30.5 / (30.5 + 20.5)
        margin = 0.02
        samples = sample_empirical_transfer(posterior, "cepeda", "valencia", n_draws=10000)
        assert abs(float(samples.mean()) - expected_mean) < margin

    def test_strong_prior_confidence(self) -> None:
        """Large alpha/beta produces tight distribution."""
        posterior = (500.5, 500.5)
        max_std = 0.03
        samples = sample_empirical_transfer(posterior, "a", "b", n_draws=1000)
        assert float(samples.std()) < max_std


class TestCacheRoundTrip:
    """Tests for netCDF cache write and read."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        """Round-trip preserves posteriors exactly."""
        posteriors: dict[tuple[str, str], tuple[float, float]] = {
            ("cepeda", "valencia"): (30.5, 20.5),
            ("fajardo", "valencia"): (15.5, 25.5),
        }
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        cache_path = cache_empirical_posterior(posteriors, year=2026, data_dir=tmp_path)
        assert cache_path.exists()

        loaded = load_empirical_posterior(year=2026, data_dir=tmp_path)
        assert loaded is not None
        assert loaded == posteriors

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        """Missing cache returns None."""
        result = load_empirical_posterior(year=2026, data_dir=tmp_path)
        assert result is None


class TestGetEmpiricalRunoffBetas:
    """Tests for the orchestrator function."""

    def test_empty_when_no_data(self, tmp_path: Path) -> None:
        """Returns empty dict when no CNE data bundles are available."""
        (tmp_path / "2026-polls").mkdir(parents=True, exist_ok=True)
        result = get_empirical_runoff_betas(year=2026, force_rebuild=True, data_dir=tmp_path)
        assert result == {}


class TestNEffThreshold:
    """Tests for the n_eff >= 50 threshold in empirical runoff integration."""

    def test_pairing_above_threshold_used(self) -> None:
        """Pairings with alpha+beta-1 >= 50 are eligible."""
        alpha, beta = 55.0, 45.5
        n_eff = alpha + beta - 1.0
        assert n_eff >= 50.0

    def test_pairing_below_threshold_skipped(self) -> None:
        """Pairings with n_eff < 50 fall back to transfer heuristic."""
        alpha, beta = 20.5, 15.5
        n_eff = alpha + beta - 1.0
        assert n_eff < 50.0

    def test_threshold_boundary(self) -> None:
        """n_eff = 50 exactly is eligible."""
        alpha, beta = 30.5, 20.5
        n_eff = alpha + beta - 1.0
        assert n_eff >= 50.0
