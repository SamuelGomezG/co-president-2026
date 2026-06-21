"""SPEC-28: Tests for 2026 forecast mode with dual gating guard."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from co_president.config import ModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_roundresult() -> object:
    """Build minimal mock RoundResult for leave_2022_out."""
    from co_president.data import CandidateResult, RoundResult  # noqa: PLC0415

    candidates = tuple(
        CandidateResult(candidate_key=k, votes=100, vote_share=0.1)
        for k in ("gustavo_petro", "federico_gutierrez", "rest", "blanco")
    )
    return RoundResult(
        round_number=1,
        date=pd.Timestamp("2022-05-29"),
        total_valid_votes=400,
        total_votes_incl_blank=400,
        registered_voters=0,
        polling_stations=0,
        candidates=candidates,
        blank_votes=0,
        null_votes=0,
        unmarked_votes=0,
    )


def _make_mock_features() -> pd.DataFrame:
    """Build minimal mock features DataFrame."""
    return pd.DataFrame(
        {
            "code_dept": [1, 2],
            "code_mun": [1, 2],
            "municipio": ["A", "B"],
            "departamento": ["X", "Y"],
            "population_2020": [1000, 2000],
            "pobreza": [0.3, 0.5],
            "pct_urbano": [0.8, 0.6],
            "distancia_capital_departamento": [10.0, 20.0],
            "pct_hombres": [0.49, 0.5],
            "pct_mujeres": [0.51, 0.5],
            "pct_18_a_40": [0.4, 0.4],
            "pct_40_a_65": [0.3, 0.3],
            "pct_mayor_65": [0.1, 0.1],
            "c_postal": [0, 1],
            "gustavo_petro": [0.5, 0.4],
            "federico_gutierrez": [0.3, 0.3],
            "rodolfo_hernandez": [0.1, 0.1],
            "rest": [0.05, 0.1],
            "blanco": [0.05, 0.1],
        }
    )


def _make_mock_polls() -> object:
    """Build minimal CleanPolls stub with .round1 attribute."""
    df = pd.DataFrame(
        {
            "fecha": ["2022-05-01"],
            "gustavo_petro": [0.5],
            "federico_gutierrez": [0.3],
            "rodolfo_hernandez": [0.1],
            "rest": [0.1],
            "blanco": [0.0],
        }
    )
    return SimpleNamespace(round1=df, round2=pd.DataFrame())


def _make_mock_polls_2014() -> pd.DataFrame:
    """Build minimal pre-2014 poll DataFrame."""
    return pd.DataFrame(
        {
            "fecha": ["2014-05-01"],
            "gustavo_petro": [0.4],
            "federico_gutierrez": [0.3],
            "rodolfo_hernandez": [0.1],
            "rest": [0.15],
            "blanco": [0.05],
            "sergio_fajardo": [0.1],
            "ingrid_betancourt": [0.05],
        }
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


class TestLoadResults2018:
    """Tests for ``_load_results_2018`` helper."""

    def test_returns_none_when_cedae_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify None returned when fetch_cedae_results raises."""
        from co_president.__main__ import _load_results_2018  # noqa: PLC0415

        msg = "no data"

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise OSError(msg)

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical.fetch_cedae_results",
            _raise,
        )
        assert _load_results_2018() is None

    def test_returns_none_on_empty_dataframe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify None returned when DataFrame is empty."""
        from co_president.__main__ import _load_results_2018  # noqa: PLC0415

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical.fetch_cedae_results",
            lambda *_, **__: pd.DataFrame(),
        )
        assert _load_results_2018() is None


class TestLoadPollsTo2014:
    """Tests for ``_load_polls_to_2014`` helper."""

    def test_returns_none_when_dir_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify None returned when data/2014-polls/ does not exist."""
        from co_president.__main__ import _load_polls_to_2014  # noqa: PLC0415

        monkeypatch.setattr("pathlib.Path.exists", lambda _: False)
        assert _load_polls_to_2014() is None


# ---------------------------------------------------------------------------
# Gating test orchestration
# ---------------------------------------------------------------------------


class TestValidateBefore2026Forecast:
    """Tests for ``_validate_before_2026_forecast``."""

    def _setup_mocks(  # noqa: PLR0913
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        features_ok: bool = True,
        polls_ok: bool = True,
        results_2022_ok: bool = True,
        results_2018_ok: bool = True,
        polls_2014_ok: bool = True,
    ) -> None:
        """Install all mocks needed by ``_validate_before_2026_forecast``."""
        mock_features = _make_mock_features() if features_ok else None

        def _load_features() -> pd.DataFrame:
            if not features_ok:
                msg = "no features"
                raise OSError(msg)
            return mock_features  # type: ignore[return-value]

        monkeypatch.setattr(
            "co_president.fundamentals.features.load_features",
            _load_features,
        )

        # mock 2022 data
        mock_polls = _make_mock_polls() if polls_ok else None
        mock_results = _make_mock_roundresult() if results_2022_ok else None

        def _load_clean_all() -> object:
            if not polls_ok:
                msg = "no polls"
                raise OSError(msg)
            return mock_polls

        def _load_results() -> object:
            if not results_2022_ok:
                msg = "no results"
                raise OSError(msg)
            return (mock_results, None)

        monkeypatch.setattr(
            "co_president.data.load_and_clean_all",
            _load_clean_all,
        )
        monkeypatch.setattr(
            "co_president.data.load_canonical_results",
            _load_results,
        )

        # mock leave_2022_out
        def _leave_2022(*_args: object, **_kwargs: object) -> None:
            pass

        monkeypatch.setattr(
            "co_president.validation.municipal_oos.leave_2022_out",
            _leave_2022,
        )

        # mock 2018 data loaders
        mock_2018 = _make_mock_roundresult() if results_2018_ok else None
        monkeypatch.setattr(
            "co_president.__main__._load_results_2018",
            lambda: mock_2018,
        )

        mock_2014 = _make_mock_polls_2014() if polls_2014_ok else None
        monkeypatch.setattr(
            "co_president.__main__._load_polls_to_2014",
            lambda: mock_2014,
        )

        # mock year_2018_holdout
        def _holdout(*_args: object, **_kwargs: object) -> None:
            pass

        monkeypatch.setattr(
            "co_president.validation.municipal_oos.year_2018_holdout",
            _holdout,
        )

    def test_exits_on_feature_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SystemExit when features cannot be loaded."""
        from co_president.__main__ import _validate_before_2026_forecast  # noqa: PLC0415

        self._setup_mocks(monkeypatch, features_ok=False)
        config = ModelConfig(fundamentals_mode="prior_only")
        with pytest.raises(SystemExit):
            _validate_before_2026_forecast(config)

    def test_exits_on_poll_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SystemExit when 2022 polls cannot be loaded."""
        from co_president.__main__ import _validate_before_2026_forecast  # noqa: PLC0415

        self._setup_mocks(monkeypatch, polls_ok=False)
        config = ModelConfig(fundamentals_mode="prior_only")
        with pytest.raises(SystemExit):
            _validate_before_2026_forecast(config)

    def test_exits_on_results_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SystemExit when 2022 results cannot be loaded."""
        from co_president.__main__ import _validate_before_2026_forecast  # noqa: PLC0415

        self._setup_mocks(monkeypatch, results_2022_ok=False)
        config = ModelConfig(fundamentals_mode="prior_only")
        with pytest.raises(SystemExit):
            _validate_before_2026_forecast(config)

    def test_exits_on_leave_2022_out_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SystemExit when leave_2022_out raises ValueError."""
        from co_president.__main__ import _validate_before_2026_forecast  # noqa: PLC0415

        self._setup_mocks(monkeypatch)

        def _fail(*_args: object, **_kwargs: object) -> None:
            msg = "leave_2022_out failed"
            raise ValueError(msg)

        monkeypatch.setattr(
            "co_president.validation.municipal_oos.leave_2022_out",
            _fail,
        )
        config = ModelConfig(fundamentals_mode="prior_only")
        with pytest.raises(SystemExit):
            _validate_before_2026_forecast(config)

    def test_skips_2018_holdout_when_no_2018_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify year_2018_holdout skipped when 2018 results unavailable."""
        from co_president.__main__ import _validate_before_2026_forecast  # noqa: PLC0415

        self._setup_mocks(monkeypatch, results_2018_ok=False)
        config = ModelConfig(fundamentals_mode="prior_only")
        caplog.set_level(logging.WARNING)
        _validate_before_2026_forecast(config)
        assert "SKIP: year_2018_holdout" in caplog.text

    def test_skips_2018_holdout_when_no_pre2014_polls(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify year_2018_holdout skipped when pre-2014 polls unavailable."""
        from co_president.__main__ import _validate_before_2026_forecast  # noqa: PLC0415

        self._setup_mocks(monkeypatch, polls_2014_ok=False)
        config = ModelConfig(fundamentals_mode="prior_only")
        caplog.set_level(logging.WARNING)
        _validate_before_2026_forecast(config)
        assert "SKIP: year_2018_holdout" in caplog.text

    def test_exits_on_holdout_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify SystemExit when year_2018_holdout raises ValueError."""
        from co_president.__main__ import _validate_before_2026_forecast  # noqa: PLC0415

        self._setup_mocks(monkeypatch)

        def _fail(*_args: object, **_kwargs: object) -> None:
            msg = "year_2018_holdout failed"
            raise ValueError(msg)

        monkeypatch.setattr(
            "co_president.validation.municipal_oos.year_2018_holdout",
            _fail,
        )
        config = ModelConfig(fundamentals_mode="prior_only")
        with pytest.raises(SystemExit):
            _validate_before_2026_forecast(config)

    def test_full_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify full validation pass logs success."""
        from co_president.__main__ import _validate_before_2026_forecast  # noqa: PLC0415

        self._setup_mocks(monkeypatch)
        config = ModelConfig(fundamentals_mode="prior_only")
        caplog.set_level(logging.INFO)
        _validate_before_2026_forecast(config)
        assert "PASS: leave_2022_out" in caplog.text
        assert "PASS: year_2018_holdout" in caplog.text
        assert "2026 gating tests complete" in caplog.text


# ---------------------------------------------------------------------------
# _cmd_forecast integration
# ---------------------------------------------------------------------------


class TestCmdForecast:
    """Tests for ``_cmd_forecast`` SPEC-28 integration."""

    def test_default_year_is_2022(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify default forecast output shows 2022 mode."""
        import argparse  # noqa: PLC0415

        from co_president.__main__ import _cmd_forecast  # noqa: PLC0415

        args = argparse.Namespace(mode="prior_only", year="2022", validate_oos=False)
        _cmd_forecast(args)
        captured = capsys.readouterr()
        assert "mode='prior_only'" in captured.out

    @pytest.mark.skip(reason="FIRST_ROUND_CANDIDATES_2026 now populated — revalidate in Phase 5")
    def test_year_2026_triggers_candidate_agnostic_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify --year 2026 logs candidate-agnostic message."""
        import argparse  # noqa: PLC0415

        from co_president.__main__ import _cmd_forecast  # noqa: PLC0415

        monkeypatch.setattr(
            "co_president.__main__._validate_before_2026_forecast",
            lambda _: None,
        )
        caplog.set_level(logging.INFO)
        args = argparse.Namespace(mode="prior_only", year="2026", validate_oos=False)
        _cmd_forecast(args)
        captured = capsys.readouterr()
        assert "year=2026" in captured.out
        assert "candidate-agnostic" not in caplog.text

    def test_validate_oos_calls_gating(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify --validate-oos with --year 2026 calls gating tests."""
        import argparse  # noqa: PLC0415

        from co_president.__main__ import _cmd_forecast  # noqa: PLC0415

        called = False

        def _mock_validate(_config: ModelConfig) -> None:
            nonlocal called
            called = True
            assert _config.fundamentals_mode == "prior_only"

        monkeypatch.setattr(
            "co_president.__main__._validate_before_2026_forecast",
            _mock_validate,
        )
        args = argparse.Namespace(mode="prior_only", year="2026", validate_oos=True)
        _cmd_forecast(args)
        assert called, "_validate_before_2026_forecast was NOT called"

    def test_validate_oos_without_year_2026_does_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify --validate-oos without --year 2026 does not trigger gating."""
        import argparse  # noqa: PLC0415

        from co_president.__main__ import _cmd_forecast  # noqa: PLC0415

        called = False

        def _mock_validate(_config: ModelConfig) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(
            "co_president.__main__._validate_before_2026_forecast",
            _mock_validate,
        )
        args = argparse.Namespace(mode="prior_only", year="2022", validate_oos=True)
        _cmd_forecast(args)
        assert not called, "_validate_before_2026_forecast should NOT be called for 2022"
