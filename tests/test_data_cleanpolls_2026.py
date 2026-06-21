"""SPEC-33: Tests for build_clean_polls_2026."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from unittest import mock

import pandas as pd
import pytest

from co_president.data_cne_2026 import build_clean_polls_2026

if TYPE_CHECKING:
    from pathlib import Path


_R1_DATE = date(2026, 5, 31)
_CANDIDATES = ["cepeda", "valencia", "fajardo", "claudia_lopez"]

# Enough unique pollsters to pass CleanPolls validation (5+ for R1, 2+ for R2)
_R1_FIRMS = ["invamer", "cifras", "gad3", "cnc", "atlas_intel"]
_R2_FIRMS = ["invamer", "cifras"]


def _make_topline(rows: list[dict]) -> pd.DataFrame:
    base = {
        "fecha": "2026-04",
        "encuestadora": "invamer",
        "field_end": date(2026, 4, 15),
        "total_weight": 100.0,
    }
    base.update(dict.fromkeys(_CANDIDATES, 0.0))
    records = []
    for row in rows:
        rec = {**base, **row}
        records.append(rec)
    return pd.DataFrame(records)


def _make_valid_topline(firms: list[str], field_end: date) -> pd.DataFrame:
    """Create topline with enough unique firms to pass validation."""
    return _make_topline([{"field_end": field_end, "encuestadora": firm} for firm in firms])


@pytest.fixture
def mock_tables(tmp_path: Path) -> Path:
    """Create fake _processed directory with valid topline and runoff parquet files."""
    processed = tmp_path / "2026-polls" / "_processed"
    processed.mkdir(parents=True)

    topline = pd.concat(
        [
            _make_valid_topline(_R1_FIRMS, date(2026, 4, 15)),
            _make_valid_topline(_R2_FIRMS, date(2026, 6, 5)),
        ],
        ignore_index=True,
    )
    topline.to_parquet(processed / "2026_topline.parquet", index=False)

    runoff = pd.DataFrame({"pairing": ["cepeda__valencia"], "value": [52.0]})
    runoff.to_parquet(processed / "2026_runoff_pairings.parquet", index=False)

    return tmp_path


class TestBuildCleanPolls2026RoundSplit:
    """Tests for round-split logic in build_clean_polls_2026."""

    def test_round_split_before_and_after_r1_date(self, mock_tables: Path) -> None:
        """Polls on/before R1 date go to round1; after go to round2."""
        cp = build_clean_polls_2026(data_dir=mock_tables)
        assert len(cp.round1) == 5
        assert len(cp.round2) == 2
        assert len(cp.all_polls) == 7

    def test_round_number_column_set(self, mock_tables: Path) -> None:
        """Round 1 gets round_number=1, round 2 gets round_number=2."""
        cp = build_clean_polls_2026(data_dir=mock_tables)
        assert (cp.round1["round_number"] == 1).all()
        assert (cp.round2["round_number"] == 2).all()

    def test_empty_topline_returns_empty_cleanpolls(self, tmp_path: Path) -> None:
        """Empty topline raises ValueError from CleanPolls validation."""
        processed = tmp_path / "2026-polls" / "_processed"
        processed.mkdir(parents=True)
        pd.DataFrame().to_parquet(processed / "2026_topline.parquet", index=False)
        pd.DataFrame().to_parquet(processed / "2026_runoff_pairings.parquet", index=False)

        with pytest.raises(ValueError, match="Round 1 needs"):
            build_clean_polls_2026(data_dir=tmp_path)

    def test_rebuild_flag_calls_build_cne_2026_tables(self, tmp_path: Path) -> None:
        """With rebuild=True, build_cne_2026_tables is called before loading."""
        processed = tmp_path / "2026-polls" / "_processed"
        processed.mkdir(parents=True)
        pd.concat(
            [
                _make_valid_topline(_R1_FIRMS, date(2026, 3, 1)),
                _make_valid_topline(_R2_FIRMS, date(2026, 6, 5)),
            ],
            ignore_index=True,
        ).to_parquet(processed / "2026_topline.parquet", index=False)
        pd.DataFrame().to_parquet(processed / "2026_runoff_pairings.parquet", index=False)

        with mock.patch("co_president.data_cne_2026.build_cne_2026_tables") as mock_build:
            build_clean_polls_2026(data_dir=tmp_path, rebuild=True)
            mock_build.assert_called_once_with(data_dir=tmp_path)

    def test_missing_parquet_triggers_rebuild(self, tmp_path: Path) -> None:
        """Missing topline parquet triggers build_cne_2026_tables."""
        with mock.patch(
            "co_president.data_cne_2026.build_cne_2026_tables",
            return_value=(pd.DataFrame(), pd.DataFrame()),
        ) as mock_build:
            with pytest.raises(ValueError, match="Round 1 needs"):
                build_clean_polls_2026(data_dir=tmp_path)
            mock_build.assert_called_once()

    def test_all_polls_before_r1_date(self, tmp_path: Path) -> None:
        """All polls before R1 date: round2 is empty, raises ValueError."""
        processed = tmp_path / "2026-polls" / "_processed"
        processed.mkdir(parents=True)

        topline = _make_valid_topline(_R1_FIRMS, date(2026, 3, 1))
        topline.to_parquet(processed / "2026_topline.parquet", index=False)
        pd.DataFrame().to_parquet(processed / "2026_runoff_pairings.parquet", index=False)

        with pytest.raises(ValueError, match="Round 2 needs"):
            build_clean_polls_2026(data_dir=tmp_path)

    def test_all_polls_after_r1_date(self, tmp_path: Path) -> None:
        """All polls after R1 date: round1 is empty, raises ValueError."""
        processed = tmp_path / "2026-polls" / "_processed"
        processed.mkdir(parents=True)

        topline = _make_valid_topline(_R2_FIRMS + _R1_FIRMS[:3], date(2026, 6, 10))
        topline.to_parquet(processed / "2026_topline.parquet", index=False)
        pd.DataFrame().to_parquet(processed / "2026_runoff_pairings.parquet", index=False)

        with pytest.raises(ValueError, match="Round 1 needs"):
            build_clean_polls_2026(data_dir=tmp_path)

    def test_agregado_only_round_bypasses_validation(self, tmp_path: Path) -> None:
        """AS/COA aggregate pollster bypasses minimum diversity check."""
        processed = tmp_path / "2026-polls" / "_processed"
        processed.mkdir(parents=True)

        topline = _make_topline(
            [
                {"field_end": date(2026, 3, 1), "encuestadora": "AGREGADO"},
                {"field_end": date(2026, 3, 1), "encuestadora": "AGREGADO"},
                {"field_end": date(2026, 6, 5), "encuestadora": "AGREGADO"},
                {"field_end": date(2026, 6, 5), "encuestadora": "AGREGADO"},
            ]
        )
        topline.to_parquet(processed / "2026_topline.parquet", index=False)
        pd.DataFrame().to_parquet(processed / "2026_runoff_pairings.parquet", index=False)

        cp = build_clean_polls_2026(data_dir=tmp_path)
        assert len(cp.round1) == 2
        assert len(cp.round2) == 2
        assert len(cp.all_polls) == 4
