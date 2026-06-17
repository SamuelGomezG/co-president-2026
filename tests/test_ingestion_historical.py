"""SPEC-12.2: Tests for historical election results ingestion."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from co_president.ingestion.ingest_historical import (
    _MMV_CODE_OVERRIDES,
    _crosswalk_mmv_municipalities,
    _load_moe_registered_voters,
    build_historical_matrix,
    compute_derived_features,
    compute_lagged_features,
    fetch_cedae_results,
    fetch_datos_gov_results,
    map_historical_candidate,
    validate_vote_shares,
)

if TYPE_CHECKING:
    from pathlib import Path


__all__: list[str] = []


def _make_historical_input() -> pd.DataFrame:
    """Synthetic 3-municipality historical data for 4 election years.

    Columns match the expected CEDAE API output format. Note that vote
    shares do NOT sum to 1.0 per (municipio, year, round) — this
    helper is designed for testing derived feature computations, not
    vote share validation.
    """
    data = {
        "codigo_municipio": [
            "05001",
            "05001",
            "05001",
            "05001",
            "05001",
            "05001",
            "05002",
            "05002",
            "05002",
            "05003",
            "05003",
            "05003",
        ],
        "year": [
            2014,
            2014,
            2018,
            2018,
            2022,
            2022,
            2014,
            2018,
            2022,
            2014,
            2018,
            2022,
        ],
        "round": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "candidate": [
            "oscar_ivan_zuluaga",
            "gustavo_petro",
            "ivan_duque",
            "gustavo_petro",
            "rodolfo_hernandez",
            "gustavo_petro",
            "oscar_ivan_zuluaga",
            "ivan_duque",
            "gustavo_petro",
            "oscar_ivan_zuluaga",
            "ivan_duque",
            "gustavo_petro",
        ],
        "votes": [
            5000,
            4000,
            6000,
            3000,
            5000,
            5500,
            3000,
            4000,
            3500,
            1000,
            2000,
            2500,
        ],
        "total_votes": [
            10000,
            10000,
            10000,
            10000,
            12000,
            12000,
            5000,
            6000,
            7000,
            3000,
            4000,
            5000,
        ],
        "registered_voters": [
            15000,
            15000,
            15000,
            15000,
            18000,
            18000,
            8000,
            9000,
            10000,
            5000,
            6000,
            7000,
        ],
    }
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════
# Fetch functions (structural tests — no real API calls)
# ═══════════════════════════════════════════════════════════════════


class TestFetchCedaeResults:
    """Contract and behavior tests for ``fetch_cedae_results``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_cedae_results)

    def test_signature_has_expected_params(self) -> None:
        """Accepts year (int) and round_num (int) parameters."""
        sig = inspect.signature(fetch_cedae_results)
        params = list(sig.parameters.keys())
        assert "year" in params, "year parameter missing"
        assert "round_num" in params, "round_num parameter missing"

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is pd.DataFrame."""
        sig = inspect.signature(fetch_cedae_results)
        assert sig.return_annotation == "pd.DataFrame"


class TestFetchDatosGovResults:
    """Structural contract for ``fetch_datos_gov_results``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_datos_gov_results)

    def test_accepts_no_args(self) -> None:
        """Accepts zero arguments."""
        sig = inspect.signature(fetch_datos_gov_results)
        assert len(sig.parameters) == 0


# ═══════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════


class TestComputeDerivedFeatures:
    """``compute_derived_features`` calculates vote shares and abstention."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        """Output contains expected column groups."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        cols = set(result.columns)
        assert "codigo_municipio" in cols
        assert "year" in cols
        assert "candidate" in cols
        assert "vote_share" in cols
        assert "abstention_rate" in cols

    def test_vote_share_range(self) -> None:
        """Vote shares are between 0 and 1."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        assert result["vote_share"].between(0, 1).all()

    def test_abstention_rate_range(self) -> None:
        """Abstention rates are between 0 and 1."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        assert result["abstention_rate"].between(0, 1).all()

    def test_abstention_calculation(self) -> None:
        """Abstention = 1 - (total_votes / registered_voters)."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        row = result[result["candidate"] == "gustavo_petro"].iloc[0]
        expected_row = df[(df["candidate"] == "gustavo_petro")].iloc[0]
        expected_abstention = 1 - (expected_row["total_votes"] / expected_row["registered_voters"])
        assert row["abstention_rate"] == pytest.approx(expected_abstention, abs=0.01)

    def test_vote_share_calculation(self) -> None:
        """Vote share = votes / total_votes."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        row = result[result["candidate"] == "gustavo_petro"].iloc[0]
        expected_row = df[(df["candidate"] == "gustavo_petro")].iloc[0]
        expected_share = expected_row["votes"] / expected_row["total_votes"]
        assert row["vote_share"] == pytest.approx(expected_share, abs=0.01)

    def test_zero_total_votes_yields_zero_share(self) -> None:
        """Rows with total_votes == 0 get vote_share == 0.0."""
        df = pd.DataFrame(
            {
                "codigo_municipio": ["05001"],
                "year": [2022],
                "round": [1],
                "candidate": ["test_candidate"],
                "votes": [0],
                "total_votes": [0],
                "registered_voters": [10000],
            }
        )
        result = compute_derived_features(df)
        assert result["vote_share"].iloc[0] == 0.0

    def test_zero_registered_yields_unknown_abstention(self) -> None:
        """Rows with registered_voters == 0 should have NA abstention (unknown)."""
        df = pd.DataFrame(
            {
                "codigo_municipio": ["05001"],
                "year": [2022],
                "round": [1],
                "candidate": ["test_candidate"],
                "votes": [500],
                "total_votes": [1000],
                "registered_voters": [0],
            }
        )
        result = compute_derived_features(df)
        assert pd.isna(result["abstention_rate"].iloc[0])


# ═══════════════════════════════════════════════════════════════════
# Candidate name mapping
# ═══════════════════════════════════════════════════════════════════


class TestMapHistoricalCandidate:
    """``map_historical_candidate`` normalises CEDAE names to canonical keys."""

    def test_known_name_passes_through(self) -> None:
        """An exact match returns the canonical key."""
        assert map_historical_candidate("gustavo_petro") == "gustavo_petro"

    def test_case_insensitive_match(self) -> None:
        """Case-insensitive name is resolved."""
        assert map_historical_candidate("Gustavo_Petro") == "gustavo_petro"

    def test_name_with_spaces(self) -> None:
        """Name with spaces is normalised to underscores."""
        assert map_historical_candidate("Gustavo Petro") == "gustavo_petro"

    def test_name_with_accents(self) -> None:
        """Accented names are stripped to ASCII before lookup."""
        assert map_historical_candidate("Álvaro Uribe") == "alvaro_uribe"

    def test_unknown_name_raises(self) -> None:
        """An unrecognised name raises ``ValueError``."""
        with pytest.raises(ValueError, match="Unrecognised historical candidate"):
            map_historical_candidate("nonexistent_candidate")


# ═══════════════════════════════════════════════════════════════════
# Lagged feature computation
# ═══════════════════════════════════════════════════════════════════


def _make_multi_year_input() -> pd.DataFrame:
    """Synthetic 2-municipality data spanning 3 election years.

    Years and candidates are chosen so that for 2018 and 2022 the
    ``_CANDIDATE_IDEOLOGY`` lookup finds matches, enabling delta
    calculation.  Municipality ``05003`` appears in both round 1 and
    round 2 of 2018 to exercise the per-(municipio, round) delta path.

    Round 2 data exercises the isolation of R1 vs R2 delta computation
    (the first R2 row must have NaN delta since there is no prior R2 row
    for that municipality).
    """
    data = {
        "codigo_municipio": [
            # 05001: present in 2014 and 2022 only, all rounds
            "05001",  # 2014 R1
            "05001",  # 2014 R1
            "05001",  # 2022 R1
            "05001",  # 2022 R1
            "05001",  # 2022 R2
            "05001",  # 2022 R2
            # 05002: present in 2014 and 2018, R1 only
            "05002",  # 2014 R1
            "05002",  # 2014 R1
            "05002",  # 2018 R1
            "05002",  # 2018 R1
            # 05003: present in both rounds of 2018 (cross-round isolation test)
            "05003",  # 2018 R1
            "05003",  # 2018 R1
            "05003",  # 2018 R2
            "05003",  # 2018 R2
        ],
        "year": [
            2014,
            2014,
            2022,
            2022,
            2022,
            2022,
            2014,
            2014,
            2018,
            2018,
            2018,
            2018,
            2018,
            2018,
        ],
        "round": [
            1,
            1,
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
        ],
        "candidate": [
            # 05001
            "clara_lopez",
            "oscar_ivan_zuluaga",
            "gustavo_petro",
            "federico_gutierrez",
            "gustavo_petro",
            "rodolfo_hernandez",
            # 05002
            "clara_lopez",
            "oscar_ivan_zuluaga",
            "gustavo_petro",
            "ivan_duque",
            # 05003
            "gustavo_petro",
            "ivan_duque",
            "gustavo_petro",
            "clara_lopez",
        ],
        "votes": [
            # 05001
            6000,
            4000,
            7000,
            5000,
            7200,
            4800,
            # 05002
            3000,
            2000,
            3500,
            2500,
            # 05003
            3600,
            2400,
            3900,
            2100,
        ],
        "total_votes": [
            # 05001
            10000,
            10000,
            12000,
            12000,
            12000,
            12000,
            # 05002
            5000,
            5000,
            6000,
            6000,
            # 05003
            6000,
            6000,
            6000,
            6000,
        ],
        "registered_voters": [
            # 05001
            15000,
            15000,
            18000,
            18000,
            18000,
            18000,
            # 05002
            8000,
            8000,
            9000,
            9000,
            # 05003
            9000,
            9000,
            9000,
            9000,
        ],
    }
    return pd.DataFrame(data)


class TestComputeLaggedFeatures:
    """``compute_lagged_features`` produces per-municipality features."""

    def test_has_expected_columns(self) -> None:
        """Output contains all expected lagged feature columns."""
        raw = _make_multi_year_input()
        derived = compute_derived_features(raw)
        result = compute_lagged_features(derived)
        expected = {
            "codigo_municipio",
            "year",
            "round",
            "left_share",
            "right_share",
            "abstention_rate",
            "delta_left",
            "delta_right",
            "delta_abstention",
        }
        assert set(result.columns) == expected

    def test_one_row_per_municipality_year_round(self) -> None:
        """Result has one row per (municipio, year, round) pair."""
        raw = _make_multi_year_input()
        derived = compute_derived_features(raw)
        result = compute_lagged_features(derived)
        unique_pairs = raw[["codigo_municipio", "year", "round"]].drop_duplicates()
        assert len(result) == len(unique_pairs)

    def test_left_share_correct(self) -> None:
        """``left_share`` matches the left candidate's vote share for 2018."""
        raw = _make_multi_year_input()
        derived = compute_derived_features(raw)
        result = compute_lagged_features(derived)
        # 05003 has 2018 R1 data with petro as left candidate
        petro_2018 = derived.loc[
            (derived["candidate"] == "gustavo_petro")
            & (derived["year"] == 2018)
            & (derived["round"] == 1)
        ]
        for _, row in petro_2018.iterrows():
            match = result.loc[
                (result["codigo_municipio"] == row["codigo_municipio"])
                & (result["year"] == 2018)
                & (result["round"] == 1)
            ]
            assert len(match) == 1
            assert match["left_share"].iloc[0] == pytest.approx(row["vote_share"], abs=0.01)

    def test_right_share_correct(self) -> None:
        """``right_share`` matches the right candidate's vote share for 2018."""
        raw = _make_multi_year_input()
        derived = compute_derived_features(raw)
        result = compute_lagged_features(derived)
        # 05003 has 2018 R1 data with ivan_duque as right candidate
        duque_2018 = derived.loc[
            (derived["candidate"] == "ivan_duque")
            & (derived["year"] == 2018)
            & (derived["round"] == 1)
        ]
        for _, row in duque_2018.iterrows():
            match = result.loc[
                (result["codigo_municipio"] == row["codigo_municipio"])
                & (result["year"] == 2018)
                & (result["round"] == 1)
            ]
            assert len(match) == 1
            assert match["right_share"].iloc[0] == pytest.approx(row["vote_share"], abs=0.01)

    def test_delta_from_previous_election(self) -> None:
        """``delta_left`` is the change from the previous election."""
        raw = _make_multi_year_input()
        derived = compute_derived_features(raw)
        result = compute_lagged_features(derived)

        # 2014 left_share for 05002: clara_lopez vote_share = 3000/5000 = 0.60
        # 2018 left_share for 05002: petro vote_share = 3500/6000 ≈ 0.5833
        # delta_left for 2018 = 0.5833 - 0.60 ≈ -0.0167
        row_2018 = result.loc[(result["codigo_municipio"] == "05002") & (result["year"] == 2018)]
        assert len(row_2018) == 1
        assert row_2018["delta_left"].iloc[0] == pytest.approx(-0.0167, abs=0.01)

    def test_deltas_do_not_cross_rounds(self) -> None:
        """Delta for first row of round 2 is NaN, not carried from round 1."""
        raw = _make_multi_year_input()
        derived = compute_derived_features(raw)
        result = compute_lagged_features(derived)

        # 05003 has both 2018 R1 and R2 data
        r2_rows = result.loc[
            (result["codigo_municipio"] == "05003")
            & (result["year"] == 2018)
            & (result["round"] == 2)
        ]
        assert not r2_rows.empty, "Fixture should include 2018 R2 data for 05003"
        for _, row in r2_rows.iterrows():
            assert pd.isna(row["delta_left"]), (
                f"delta_left for 2018 R2 should be NaN (no prior R2 row for 05003), "
                f"got {row['delta_left']}"
            )
            assert pd.isna(row["delta_right"]), (
                f"delta_right for 2018 R2 should be NaN, got {row['delta_right']}"
            )
            assert pd.isna(row["delta_abstention"]), (
                f"delta_abstention for 2018 R2 should be NaN, got {row['delta_abstention']}"
            )

    def test_empty_dataframe_returns_empty(self) -> None:
        """An empty input returns an empty DataFrame."""
        empty = pd.DataFrame(
            columns=[
                "codigo_municipio",
                "year",
                "round",
                "candidate",
                "vote_share",
                "abstention_rate",
            ]
        )
        result = compute_lagged_features(empty)
        assert result.empty


# ═══════════════════════════════════════════════════════════════════
# Vote share validation
# ═══════════════════════════════════════════════════════════════════


class TestValidateVoteShares:
    """``validate_vote_shares`` checks per-round sums."""

    def test_no_warnings_for_good_data(self) -> None:
        """Data where vote shares sum to 1.0 produces zero warnings."""
        df = pd.DataFrame(
            {
                "codigo_municipio": ["05001", "05001", "05001"],
                "year": [2022, 2022, 2022],
                "round": [1, 1, 1],
                "candidate": [
                    "gustavo_petro",
                    "rodolfo_hernandez",
                    "rest",
                ],
                "vote_share": [0.40, 0.28, 0.32],
                "abstention_rate": [0.45, 0.45, 0.45],
            }
        )
        warnings = validate_vote_shares(df)
        assert len(warnings) == 0

    def test_warning_for_bad_sums(self) -> None:
        """Data exceeding the 2 % tolerance produces a warning."""
        df = pd.DataFrame(
            {
                "codigo_municipio": ["05001", "05001"],
                "year": [2022, 2022],
                "round": [1, 1],
                "candidate": ["gustavo_petro", "rodolfo_hernandez"],
                "vote_share": [0.70, 0.40],  # sum = 1.10 > 2 % tolerance
                "abstention_rate": [0.45, 0.45],
            }
        )
        warnings = validate_vote_shares(df)
        assert len(warnings) >= 1
        assert "vote shares sum to" in warnings[0]


# ═══════════════════════════════════════════════════════════════════
# Build pipeline
# ═══════════════════════════════════════════════════════════════════


class TestBuildHistoricalMatrix:
    """``build_historical_matrix`` writes a validated CSV."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Produces ``historical_results.csv`` in the given data directory."""
        df = _make_historical_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical._fetch_all_years",
            lambda _: df,
        )

        build_historical_matrix(data_dir=tmp_path)
        target = tmp_path / "fundamentals" / "historical_results.csv"
        assert target.is_file(), f"Expected {target} to exist"

    def test_saved_file_non_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The saved CSV has rows and computed feature columns."""
        df = _make_historical_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical._fetch_all_years",
            lambda _: df,
        )

        build_historical_matrix(data_dir=tmp_path)
        saved = pd.read_csv(tmp_path / "fundamentals" / "historical_results.csv")
        assert len(saved) > 0, "Saved CSV should have at least one row"
        assert "vote_share" in saved.columns, "vote_share column missing"
        assert "abstention_rate" in saved.columns, "abstention_rate column missing"
        assert saved["vote_share"].notna().all(), "vote_share should have no nulls"
        assert saved["abstention_rate"].notna().all(), "abstention_rate should have no nulls"

    def test_creates_lagged_features_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pipeline also writes ``historical_lagged_features.csv``."""
        df = _make_historical_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical._fetch_all_years",
            lambda _: df,
        )

        build_historical_matrix(data_dir=tmp_path)
        target = tmp_path / "fundamentals" / "historical_lagged_features.csv"
        assert target.is_file(), f"Expected {target} to exist"

    def test_lagged_features_has_expected_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lagged features CSV contains the required columns."""
        df = _make_historical_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical._fetch_all_years",
            lambda _: df,
        )

        build_historical_matrix(data_dir=tmp_path)
        saved = pd.read_csv(tmp_path / "fundamentals" / "historical_lagged_features.csv")
        expected = {
            "codigo_municipio",
            "year",
            "round",
            "left_share",
            "right_share",
            "abstention_rate",
            "delta_left",
            "delta_right",
            "delta_abstention",
        }
        assert set(saved.columns) == expected


# ═══════════════════════════════════════════════════════════════════
# MMV crosswalk
# ═══════════════════════════════════════════════════════════════════


class TestCrosswalkMmvMunicipalities:
    """``_crosswalk_mmv_municipalities`` maps all MMV codes without error."""

    def test_all_municipalities_mapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        data_dir: Path,
    ) -> None:
        """All rows receive a DANE code (level 1 match + override)."""
        divipola = pd.DataFrame(
            {
                "codigo_municipio": ["05001", "20443"],
                "nombre_municipio": ["Medellin", "La Paz"],
                "departamento": ["Antioquia", "Cesar"],
            }
        )

        def mock_read_csv(path: str, **_kwargs: object) -> pd.DataFrame:
            if "divipola_master.csv" in str(path):
                return divipola
            msg = f"Unexpected path: {path}"
            raise FileNotFoundError(msg)

        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

        mmv = pd.DataFrame(
            {
                "DEP": ["05", "12"],
                "MUN": ["001", "625"],
                "DEPNOMBRE": ["ANTIOQUIA", "ARAUCA"],
                "MUNNOMBRE": ["MEDELLIN", "LA PAZ"],
            }
        )

        result = _crosswalk_mmv_municipalities(mmv, data_dir)
        assert result["codigo_municipio"].notna().all()
        assert list(result["codigo_municipio"]) == ["05001", "20443"]

    def test_unmatched_municipality_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        data_dir: Path,
    ) -> None:
        """A municipality absent from overrides raises ``ValueError``."""
        divipola = pd.DataFrame(
            {
                "codigo_municipio": ["05001"],
                "nombre_municipio": ["Medellin"],
                "departamento": ["Antioquia"],
            }
        )

        def mock_read_csv(path: str, **_kwargs: object) -> pd.DataFrame:
            if "divipola_master.csv" in str(path):
                return divipola
            msg = f"Unexpected path: {path}"
            raise FileNotFoundError(msg)

        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

        mmv = pd.DataFrame(
            {
                "DEP": ["99"],
                "MUN": ["999"],
                "DEPNOMBRE": ["NONEXIST"],
                "MUNNOMBRE": ["NOWHERE"],
            }
        )

        with pytest.raises(ValueError, match="MMV crosswalk"):
            _crosswalk_mmv_municipalities(mmv, data_dir)

    def test_override_keys_exist_in_divipola(self) -> None:
        """Representative DANE codes from ``_MMV_CODE_OVERRIDES`` match valid codes.

        Uses a minimal hardcoded fixture instead of loading the full DIVIPOLA
        master CSV to keep the test isolated from external data files.
        """
        fixture = pd.DataFrame(
            {
                "codigo_municipio": [
                    "05042",
                    "05101",
                    "05148",
                    "05585",
                    "05697",
                    "05893",
                    "13062",
                    "13600",
                    "15109",
                    "15407",
                    "20443",
                    "27025",
                ],
            }
        )
        codes = set(fixture["codigo_municipio"].unique())
        sampled = list(_MMV_CODE_OVERRIDES.values())[:10]
        for dane in sampled:
            assert dane in codes, f"Override DANE code {dane} not found in fixture"


# ═══════════════════════════════════════════════════════════════════
# MOE registered-voters backfill
# ═══════════════════════════════════════════════════════════════════


def _write_moe_csv(directory: Path, name: str, rows: list[dict[str, object]]) -> None:
    """Write a minimal MOE censo electoral CSV to *directory*/*name*."""
    pd.DataFrame(rows).to_csv(directory / name, index=False)


def _make_moe_input() -> list[dict[str, object]]:
    """Three municipalities, two election years, multiple polling stations each.

    The aggregation must sum ``total`` per (code_dane, ano) and pad the
    DANE code to 5 digits.
    """
    return [
        # 2014 presidencia — 05001 Medellín
        {
            "code_dane": 5001,
            "ano": 2014,
            "eleccion": "Presidencia",
            "total": 100,
            "mujeres": 50,
            "hombres": 50,
        },
        {
            "code_dane": 5001,
            "ano": 2014,
            "eleccion": "Presidencia",
            "total": 200,
            "mujeres": 100,
            "hombres": 100,
        },
        # 2014 — 05002 Abejorral
        {
            "code_dane": 5002,
            "ano": 2014,
            "eleccion": "Presidencia",
            "total": 50,
            "mujeres": 25,
            "hombres": 25,
        },
        # 2022 presidencia — 05001 + 05002
        {
            "code_dane": 5001,
            "ano": 2022,
            "eleccion": "Presidencia",
            "total": 400,
            "mujeres": 200,
            "hombres": 200,
        },
        {
            "code_dane": 5002,
            "ano": 2022,
            "eleccion": "Presidencia",
            "total": 80,
            "mujeres": 40,
            "hombres": 40,
        },
        # Congreso rows for years WITHOUT presidencia coverage
        {
            "code_dane": 5001,
            "ano": 2006,
            "eleccion": "Congreso",
            "total": 150,
            "mujeres": 75,
            "hombres": 75,
        },
        {
            "code_dane": 5002,
            "ano": 2006,
            "eleccion": "Congreso",
            "total": 60,
            "mujeres": 30,
            "hombres": 30,
        },
        # Congreso row for year WITH presidencia coverage (must be ignored)
        {
            "code_dane": 5001,
            "ano": 2022,
            "eleccion": "Congreso",
            "total": 999,
            "mujeres": 0,
            "hombres": 0,
        },
    ]


class TestLoadMoeRegisteredVoters:
    """``_load_moe_registered_voters`` aggregates MOE polling-station data."""

    def test_aggregates_total_per_municipality(self, tmp_path: Path) -> None:
        """Sums ``total`` per (code_dane, ano) and zero-pads DANE codes."""
        moe_dir = tmp_path / "moe_censo_electoral"
        moe_dir.mkdir()
        _write_moe_csv(moe_dir, "2014_presidencia.csv", _make_moe_input()[:3])
        _write_moe_csv(moe_dir, "2022_presidencia.csv", _make_moe_input()[3:5])

        result = _load_moe_registered_voters(moe_dir=moe_dir)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["codigo_municipio", "year", "round", "registered_voters"]
        # 05001 sum for 2014 = 100+200 = 300; 05002 = 50. Both R1 and R2 broadcasted.
        m_2014 = result[(result["codigo_municipio"] == "05001") & (result["year"] == 2014)]
        assert len(m_2014) == 2  # R1 + R2 broadcast
        assert (m_2014["registered_voters"] == 300).all()
        m_2014_5002 = result[(result["codigo_municipio"] == "05002") & (result["year"] == 2014)]
        assert (m_2014_5002["registered_voters"] == 50).all()

    def test_broadcasts_to_round_2(self, tmp_path: Path) -> None:
        """Each R1 row is duplicated as R2 (registered voters do not change)."""
        moe_dir = tmp_path / "moe_censo_electoral"
        moe_dir.mkdir()
        _write_moe_csv(moe_dir, "2022_presidencia.csv", _make_moe_input()[3:5])

        result = _load_moe_registered_voters(moe_dir=moe_dir)

        r1 = result[(result["year"] == 2022) & (result["round"] == 1)]
        r2 = result[(result["year"] == 2022) & (result["round"] == 2)]
        assert len(r1) == 2
        assert len(r2) == 2
        merged = r1.merge(r2, on=["codigo_municipio", "year"], suffixes=("_r1", "_r2"))
        assert (merged["registered_voters_r1"] == merged["registered_voters_r2"]).all()

    def test_congreso_fills_missing_years(self, tmp_path: Path) -> None:
        """Congreso data fills years without presidencia coverage."""
        moe_dir = tmp_path / "moe_censo_electoral"
        moe_dir.mkdir()
        # 2006 congreso data only (no presidencia file for 2006 exists)
        _write_moe_csv(moe_dir, "2006_congreso.csv", _make_moe_input()[5:7])

        result = _load_moe_registered_voters(moe_dir=moe_dir)
        assert not result.empty
        # 05001 should have 150 registered voters in both rounds
        m_2006_5001 = result[(result["codigo_municipio"] == "05001") & (result["year"] == 2006)]
        assert len(m_2006_5001) == 2
        assert (m_2006_5001["registered_voters"] == 150).all()
        # 05002 should have 60
        m_2006_5002 = result[(result["codigo_municipio"] == "05002") & (result["year"] == 2006)]
        assert (m_2006_5002["registered_voters"] == 60).all()

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        """An empty MOE directory yields an empty (but well-formed) DataFrame."""
        moe_dir = tmp_path / "moe_censo_electoral"
        moe_dir.mkdir()

        result = _load_moe_registered_voters(moe_dir=moe_dir)
        assert result.empty
        assert list(result.columns) == ["codigo_municipio", "year", "round", "registered_voters"]

    def test_presidencia_takes_priority_over_congreso(self, tmp_path: Path) -> None:
        """Presidencia data takes priority over congreso for same (municipio, year)."""
        moe_dir = tmp_path / "moe_censo_electoral"
        moe_dir.mkdir()
        _write_moe_csv(moe_dir, "2014_presidencia.csv", _make_moe_input()[:3])
        _write_moe_csv(moe_dir, "2022_presidencia.csv", _make_moe_input()[3:5])
        _write_moe_csv(moe_dir, "2006_congreso.csv", _make_moe_input()[5:7])
        _write_moe_csv(moe_dir, "2022_congreso.csv", _make_moe_input()[7:])

        result = _load_moe_registered_voters(moe_dir=moe_dir)

        # 2014 presidencia values — 05001 should be 300, NOT overridden by congreso
        m_2014 = result[(result["codigo_municipio"] == "05001") & (result["year"] == 2014)]
        assert (m_2014["registered_voters"] == 300).all()
        # 2006 congreso values — fill gap since no presidencia file for 2006
        m_2006_5001 = result[(result["codigo_municipio"] == "05001") & (result["year"] == 2006)]
        assert (m_2006_5001["registered_voters"] == 150).all()
        m_2006_5002 = result[(result["codigo_municipio"] == "05002") & (result["year"] == 2006)]
        assert (m_2006_5002["registered_voters"] == 60).all()
        # 2022 from presidencia (rows 3-4), NOT congreso (row 7)
        m_2022 = result[(result["codigo_municipio"] == "05001") & (result["year"] == 2022)]
        assert (m_2022["registered_voters"] == 400).all()


class TestBuildHistoricalMatrixFillsRegisteredVoters:
    """``build_historical_matrix`` backfills ``registered_voters`` from MOE."""

    def test_fills_nan_registered_voters_from_moe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NaN ``registered_voters`` in source data is replaced with MOE values."""
        # Source: 2022 R1 and R2 for 05001, with NaN registered_voters
        raw = pd.DataFrame(
            {
                "codigo_municipio": ["05001", "05001", "05001"],
                "year": [2022, 2022, 2022],
                "round": [1, 1, 2],
                "candidate": ["gustavo_petro", "rodolfo_hernandez", "gustavo_petro"],
                "votes": [300, 200, 400],
                "total_votes": [500, 500, 500],
                "registered_voters": [pd.NA, pd.NA, pd.NA],
            }
        )

        moe_dir = tmp_path / "moe_censo_electoral"
        moe_dir.mkdir()
        _write_moe_csv(
            moe_dir,
            "2022_presidencia.csv",
            [
                {
                    "code_dane": 5001,
                    "ano": 2022,
                    "eleccion": "Presidencia",
                    "total": 12345,
                    "mujeres": 0,
                    "hombres": 0,
                }
            ],
        )

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical._fetch_all_years",
            lambda _: raw,
        )
        # Point MOE discovery at our tmp directory
        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical.resolve_data_dir",
            lambda _: tmp_path,
        )

        build_historical_matrix(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "historical_results.csv", dtype={"codigo_municipio": str}
        )
        # All 05001 2022 rows (both R1 and R2) should now have 12345
        r1_2022 = saved[(saved["codigo_municipio"] == "05001") & (saved["round"] == 1)]
        r2_2022 = saved[(saved["codigo_municipio"] == "05001") & (saved["round"] == 2)]
        assert len(r1_2022) > 0, "No round-1 rows found for 05001"
        assert len(r2_2022) > 0, "No round-2 rows found for 05001"
        assert (r1_2022["registered_voters"] == 12345).all()
        assert (r2_2022["registered_voters"] == 12345).all()

    def test_preserves_existing_non_null_registered_voters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-null source values are NOT overwritten by MOE values."""
        raw = pd.DataFrame(
            {
                "codigo_municipio": ["05001", "05001"],
                "year": [2022, 2022],
                "round": [1, 1],
                "candidate": ["gustavo_petro", "rodolfo_hernandez"],
                "votes": [300, 200],
                "total_votes": [500, 500],
                "registered_voters": [7777, 7777],
            }
        )

        moe_dir = tmp_path / "moe_censo_electoral"
        moe_dir.mkdir()
        _write_moe_csv(
            moe_dir,
            "2022_presidencia.csv",
            [
                {
                    "code_dane": 5001,
                    "ano": 2022,
                    "eleccion": "Presidencia",
                    "total": 12345,
                    "mujeres": 0,
                    "hombres": 0,
                }
            ],
        )

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical._fetch_all_years",
            lambda _: raw,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical.resolve_data_dir",
            lambda _: tmp_path,
        )

        build_historical_matrix(data_dir=tmp_path)
        saved = pd.read_csv(tmp_path / "fundamentals" / "historical_results.csv")
        assert (saved["registered_voters"] == 7777).all()
