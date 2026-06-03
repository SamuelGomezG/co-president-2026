"""SPEC-15: Tests for fundamental data coverage report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from co_president.ingestion.report import (
    generate_coverage_markdown,
    generate_coverage_report,
    verify_legislative_schemas,
    write_coverage_report,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__: list[str] = []


@pytest.fixture
def fundamentals_dir(tmp_path: Path) -> Path:
    """Create a mini fundamentals directory with synthetic data."""
    fund_dir = tmp_path / "fundamentals"
    fund_dir.mkdir()
    # DIVIPOLA: 3 municipalities
    pd.DataFrame({"codigo_municipio": ["00001", "00002", "00003"]}).to_csv(
        fund_dir / "divipola_master.csv", index=False
    )
    # Historical results: 6 rows for 2 municipalities
    pd.DataFrame(
        {
            "codigo_municipio": ["00001", "00001", "00002", "00002", "00001", "00001"],
            "year": [2018, 2022, 2018, 2022, 2018, 2022],
            "round": [1, 1, 1, 1, 2, 2],
            "candidate": ["A", "A", "B", "B", "A", "B"],
            "votes": [100, 150, 80, 120, 90, 110],
            "total_votes": [200, 300, 160, 240, 180, 220],
            "registered_voters": [250, 350, 200, 300, 220, 280],
            "vote_share": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "abstention_rate": [0.2, 0.14, 0.2, 0.14, 0.2, 0.14],
        }
    ).to_csv(fund_dir / "historical_results.csv", index=False)
    # Socioeconomic: 3 rows (below STUB threshold of 5)
    pd.DataFrame(
        {
            "codigo_municipio": ["00001", "00002", "00003"],
            "pct_indigenous": [0.01, 0.02, 0.03],
        }
    ).to_csv(fund_dir / "socioeconomic.csv", index=False)
    # Risk: 10 rows (below STUB threshold of 15)
    pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, 11)],
            "risk_level": ["low"] * 10,
        }
    ).to_csv(fund_dir / "risk_factors.csv", index=False)
    return tmp_path


def test_generate_coverage_report_returns_dataframe(fundamentals_dir: Path) -> None:
    df = generate_coverage_report(fundamentals_dir)
    assert isinstance(df, pd.DataFrame)
    assert "source" in df.columns
    assert "file" in df.columns
    assert "rows_actual" in df.columns
    assert "status" in df.columns


def test_coverage_report_divipola_coverage(fundamentals_dir: Path) -> None:
    df = generate_coverage_report(fundamentals_dir)
    divipola = df[df["source"] == "DIVIPOLA"]
    assert not divipola.empty
    assert divipola.iloc[0]["rows_actual"] >= 3


def test_coverage_report_identifies_socioeconomic_stub(fundamentals_dir: Path) -> None:
    df = generate_coverage_report(fundamentals_dir)
    socioeconomic = df[df["source"] == "Socioeconomic"]
    assert not socioeconomic.empty
    status = str(socioeconomic.iloc[0]["status"])
    assert "STUB" in status


def test_coverage_report_identifies_risk_stub(fundamentals_dir: Path) -> None:
    df = generate_coverage_report(fundamentals_dir)
    risk = df[df["source"] == "Risk"]
    assert not risk.empty
    status = str(risk.iloc[0]["status"])
    assert "STUB" in status


def test_generate_coverage_markdown_returns_string(fundamentals_dir: Path) -> None:
    md = generate_coverage_markdown(fundamentals_dir)
    assert isinstance(md, str)
    assert "DIVIPOLA" in md
    assert "Fundamentals Coverage" in md


def test_write_coverage_report_creates_file(fundamentals_dir: Path, tmp_path: Path) -> None:
    dest = write_coverage_report(fundamentals_dir, output_dir=tmp_path)
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")
    assert "DIVIPOLA" in content
    assert "Coverage" in content


def test_verify_legislative_schemas_returns_dict(tmp_path: Path) -> None:
    result = verify_legislative_schemas(tmp_path)
    assert isinstance(result, dict)
    assert "cedae_camara_columns" in result
    assert "moe_camara_columns" in result


def test_verify_legislative_schemas_senado(tmp_path: Path) -> None:
    result = verify_legislative_schemas(tmp_path)
    assert "cedae_senado_columns" in result
    assert "moe_senado_columns" in result


def test_generate_coverage_report_with_nonexistent_data(tmp_path: Path) -> None:
    df = generate_coverage_report(tmp_path / "nonexistent")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for _, row in df.iterrows():
        val = row["rows_actual"]
        if isinstance(val, str):
            assert val != ""
        else:
            assert isinstance(val, (int, float))
            assert val >= 0
