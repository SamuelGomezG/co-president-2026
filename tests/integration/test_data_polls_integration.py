"""Integration tests for SPEC-04: Poll data loading & cleaning.

These tests load full production CSV datasets and are marked with
``@pytest.mark.integration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from co_president.config import ModelConfig, get_active_candidates
from co_president.data import (
    CleanPolls,
    load_and_clean_all,
    load_raw_consultas,
    load_raw_polls,
)
from co_president.data_polls import (
    _SHARE_COLS_EXCLUDED,
    fix_invamer_date,
    get_computed_consultation_prior_strengths,
    infer_round_number,
    normalize_undecided,
    retain_active_candidates,
)
from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
class TestLoadRawPolls:
    """Integration tests for load_raw_polls (full CSV data)."""

    def test_returns_dataframe_with_expected_columns(self, data_dir: Path) -> None:
        """Verify load_raw_polls returns a DataFrame with key columns."""
        df = load_raw_polls(data_dir)
        assert "fecha" in df.columns
        assert "encuestadora" in df.columns
        assert "muestra" in df.columns
        assert "gustavo_petro" in df.columns
        assert "rodolfo_hernandez" in df.columns
        assert "ns_nr" in df.columns

    def test_fecha_parsed_as_datetime(self, data_dir: Path) -> None:
        """Verify fecha column is parsed to datetime (no NaT)."""
        df = load_raw_polls(data_dir)
        assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
        assert df["fecha"].isna().sum() == 0, "Found NaT dates"


@pytest.mark.integration
class TestLoadRawConsultas:
    """Integration tests for load_raw_consultas (full CSV data)."""

    def test_returns_dataframe(self, data_dir: Path) -> None:
        """Verify load_raw_consultas returns a DataFrame."""
        df = load_raw_consultas(data_dir)
        assert "fecha" in df.columns
        assert "encuestadora" in df.columns
        assert "consulta" in df.columns
        assert "candidato" in df.columns

    def test_fecha_parsed_as_datetime(self, data_dir: Path) -> None:
        """Verify fecha is parsed to datetime."""
        df = load_raw_consultas(data_dir)
        assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
        assert df["fecha"].isna().sum() == 0, "Found NaT dates"


@pytest.mark.integration
class TestLoadAndCleanAll:
    """Integration tests for the full load_and_clean_all pipeline."""

    def test_returns_clean_polls(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify load_and_clean_all returns a CleanPolls instance."""
        assert isinstance(clean_polls_fixture, CleanPolls)

    def test_round1_has_at_least_5_pollsters(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify round 1 has >= 5 unique pollsters."""
        unique_pollsters = clean_polls_fixture.round1["encuestadora"].nunique()
        assert unique_pollsters >= 5, f"Round 1 has {unique_pollsters} pollsters, need >= 5"

    def test_round2_has_at_least_2_pollsters(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify round 2 has >= 2 unique pollsters."""
        unique_pollsters = clean_polls_fixture.round2["encuestadora"].nunique()
        assert unique_pollsters >= 2, f"Round 2 has {unique_pollsters} pollsters, need >= 2"

    def test_invamer_date_corrected(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify Invamer April 19 poll is corrected to May 19."""
        invamer_round1 = clean_polls_fixture.round1[
            clean_polls_fixture.round1["encuestadora"].str.strip() == "Invamer"
        ]
        assert not invamer_round1.empty, "No Invamer polls found in round 1"
        min_date = invamer_round1["fecha"].min()
        assert min_date >= pd.Timestamp("2022-04-29")

    def test_no_duplicate_pollster_date_in_round1(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify no pollster appears more than once per date in round 1."""
        dups = clean_polls_fixture.round1.duplicated(subset=["encuestadora", "fecha"], keep=False)
        assert not dups.any(), "Duplicate pollster+date found in round 1"

    def test_no_duplicate_pollster_date_in_round2(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify no pollster appears more than once per date in round 2."""
        dups = clean_polls_fixture.round2.duplicated(subset=["encuestadora", "fecha"], keep=False)
        assert not dups.any(), "Duplicate pollster+date found in round 2"

    def test_normalized_shares_sum_to_100(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify candidate+blanco+otros sum to ~100% after normalization."""
        share_cols = [
            c
            for c in clean_polls_fixture.round1.columns
            if c in {cand.key for cand in get_active_candidates(1)} or c in ("blanco", "otros")
        ]
        for idx, row in clean_polls_fixture.round1.iterrows():
            row_sum = row[share_cols].sum()
            assert abs(row_sum - 100.0) <= 1.0, f"Row {idx} share sum = {row_sum}, expected ~100"

    def test_round2_shares_sum_to_100(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify round 2 candidate+blanco+otros sum to ~100% after normalization."""
        share_cols = [
            c
            for c in clean_polls_fixture.round2.columns
            if c in {cand.key for cand in get_active_candidates(2)} or c in ("blanco", "otros")
        ]
        for idx, row in clean_polls_fixture.round2.iterrows():
            row_sum = row[share_cols].sum()
            assert abs(row_sum - 100.0) <= 1.0, f"Row {idx} share sum = {row_sum}, expected ~100"

    def test_gad3_round2_count(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify the number of GAD3 runoff tracking polls in round 2."""
        gad3_round2 = clean_polls_fixture.round2[
            clean_polls_fixture.round2["encuestadora"].str.strip() == "GAD3"
        ]
        assert len(gad3_round2) == 10

    def test_gad3_final_wave_not_duplicated(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify the final GAD3 tracking wave appears only once in round 2."""
        gad3_round2 = clean_polls_fixture.round2[
            clean_polls_fixture.round2["encuestadora"].str.strip() == "GAD3"
        ]
        final_wave = gad3_round2[gad3_round2["fecha"] == pd.Timestamp("2022-06-10")]
        assert len(final_wave) == 1

    def test_ns_nr_is_zero_after_normalization(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify ns_nr is 0.0 in both rounds after normalization."""
        assert clean_polls_fixture.round1["ns_nr"].sum() == 0.0
        assert clean_polls_fixture.round2["ns_nr"].sum() == 0.0

    def test_all_polls_includes_unclassified_and_all_columns(
        self,
        clean_polls_fixture: CleanPolls,
    ) -> None:
        """Verify all_polls includes pre-consultation columns and unclassified rows."""
        assert "gustavo_petro" in clean_polls_fixture.all_polls.columns
        assert "round_number" in clean_polls_fixture.all_polls.columns
        unclassified = clean_polls_fixture.all_polls[
            clean_polls_fixture.all_polls["round_number"].isna()
        ]
        assert len(unclassified) > 0

    def test_all_polls_dates_monotonic(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify dates are non-decreasing in the cleaned output."""
        assert clean_polls_fixture.all_polls["fecha"].is_monotonic_increasing


@pytest.mark.integration
class TestMassiveCallerR2:
    """Tests for Round 2 forced-choice poll handling."""

    def test_massivecaller_r2_excluded(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify MassiveCaller round 2 polls are excluded (forced-choice format)."""
        mc_round2 = clean_polls_fixture.round2[
            clean_polls_fixture.round2["encuestadora"].str.strip() == "MassiveCaller"
        ]
        assert mc_round2.empty


@pytest.mark.integration
class TestPollSchemaVerification:
    """Tests to ensure poll metadata columns stay in sync with the CSV schema."""

    def test_share_cols_excluded_matches_csv_metadata(self, data_dir: Path) -> None:
        """Verify excluded share columns match the CSV metadata columns."""
        df = load_raw_polls(data_dir)
        expected_metadata = {
            "n",
            "encuestadora",
            "fecha",
            "muestra",
            "tasa_respuesta",
            "margen_error",
            "fuente",
            "link",
            "muestreo",
            "hipotesis",
            "tipo",
            "muestra_int_voto",
            "municipios",
            "ns_nr",
        }
        excluded = set(_SHARE_COLS_EXCLUDED) - {"round_number"}
        assert expected_metadata <= excluded, (
            f"CSV metadata columns not fully covered by _SHARE_COLS_EXCLUDED. "
            f"Missing: {expected_metadata - excluded}"
        )
        assert expected_metadata <= set(df.columns)
        assert expected_metadata <= set(df.columns)

        share_columns = set(df.columns) - expected_metadata - {"round_number"}
        for col in share_columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            assert (values >= 0).all()
            assert (values <= 100).all()


@pytest.mark.integration
class TestLoadAndCleanAllEmptyRounds:
    """Tests for empty round DataFrame handling in load_and_clean_all."""

    def test_empty_round2_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify empty round2 DataFrame is skipped in step 8b."""
        data_dir = resolve_data_dir(None)
        raw_polls = load_raw_polls(data_dir)
        polls = fix_invamer_date(raw_polls)
        polls = normalize_undecided(polls)
        polls = retain_active_candidates(polls, [c.key for c in get_active_candidates(1)])
        polls = infer_round_number(polls)

        round2_df = polls[polls["round_number"] == 2].copy()
        if round2_df.empty:
            pytest.skip("No round2 rows available to test empty guard.")
        remove_indices = round2_df.index.tolist()

        def fake_infer_round_number(df: pd.DataFrame) -> pd.DataFrame:
            result = infer_round_number(df)
            result.loc[remove_indices, "round_number"] = pd.NA
            result["round_number"] = result["round_number"].astype("Int64")
            return result

        monkeypatch.setattr("co_president.data_polls.infer_round_number", fake_infer_round_number)
        monkeypatch.setattr("co_president.data_polls._MIN_ROUND2_POLLSTERS", 0)

        clean = load_and_clean_all(data_dir)
        assert clean.round2.empty


@pytest.mark.integration
class TestComputedPriorStrengthsLazyEval:
    """Tests for lazy evaluation of computed consultation prior strengths."""

    def test_recovers_after_missing_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify a cached empty result does not poison subsequent calls."""
        get_computed_consultation_prior_strengths.cache_clear()

        def raise_on_call(*args: object, **kwargs: object) -> str:  # noqa: ARG001
            msg = "Simulated missing data directory"
            raise FileNotFoundError(msg)

        with monkeypatch.context() as m:
            m.setattr(
                "co_president.data_polls.resolve_data_dir",
                raise_on_call,
            )
            with pytest.raises(FileNotFoundError):
                get_computed_consultation_prior_strengths()

        result = get_computed_consultation_prior_strengths()
        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(v >= 0.10 for v in result.values())

    def test_model_config_property_lazy(self) -> None:
        """Verify the property access does not raise at instantiation time."""
        cfg = ModelConfig()
        strengths = cfg.computed_consultation_prior_strengths
        assert isinstance(strengths, dict)

    def test_model_config_property_merges_override(self) -> None:
        """Verify the property includes override values when set."""
        overrides = {"gustavo_petro": 0.05, "rodolfo_hernandez": 0.03}
        cfg = ModelConfig(consultation_prior_strength_override=overrides)
        strengths = cfg.computed_consultation_prior_strengths
        assert strengths["gustavo_petro"] == 0.05
        assert strengths["rodolfo_hernandez"] == 0.03

    def test_model_config_override_does_not_mutate_cache(self) -> None:
        """Verify that ModelConfig overrides do not mutate the memoized cache."""
        get_computed_consultation_prior_strengths.cache_clear()

        cached = get_computed_consultation_prior_strengths()
        petro_before = cached["gustavo_petro"]

        overrides = {"gustavo_petro": 0.99}
        cfg = ModelConfig(consultation_prior_strength_override=overrides)
        _ = cfg.computed_consultation_prior_strengths

        cached_after = get_computed_consultation_prior_strengths()
        assert cached_after["gustavo_petro"] == petro_before
        assert cached_after["gustavo_petro"] != 0.99

    def test_model_config_property_empty_on_missing_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the property returns empty dict when file is missing."""
        get_computed_consultation_prior_strengths.cache_clear()

        def raise_missing(*args: object, **kwargs: object) -> str:  # noqa: ARG001
            msg = "Simulated missing file"
            raise FileNotFoundError(msg)

        monkeypatch.setattr(
            "co_president.data_polls.resolve_data_dir",
            raise_missing,
        )

        cfg = ModelConfig()
        strengths = cfg.computed_consultation_prior_strengths
        assert strengths == {}
