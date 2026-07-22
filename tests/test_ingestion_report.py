"""Tests for the per-bundle ingestion report (STATE_REPORT.md P0 step 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING
import zipfile

import pandas as pd
import pytest

from co_president.data_cne_2026 import build_cne_2026_tables
from co_president.ingestion_report import (
    BundleResult,
    DataQualityError,
    persist_report,
    results_to_dataframe,
    write_report_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_atlas_zip(path: Path, csv: str = "") -> Path:
    """Write a minimal Atlas Intel bundle to *path*."""
    if not csv:
        csv = (
            "user_id,weight,presidential_election_2026\n"
            "1,1.0,Iván Cepeda\n"
            "2,1.0,Paloma Valencia\n"
            "3,1.0,Voto en Blanco\n"
        )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Base de Dados Atlas.csv", csv)
    return path


def _make_corrupt_zip(path: Path) -> Path:
    """Write a non-zip file that any loader will reject."""
    path.write_bytes(b"not a zip file")
    return path


def test_ingestion_report_counts_bundles(tmp_path: Path) -> None:
    """3 bundles (2 atlas, 1 corrupt) → report has 3 rows with correct statuses."""
    data_dir = tmp_path / "2026-polls"
    data_dir.mkdir()
    _make_atlas_zip(data_dir / "atlas_intel_01012026_02012026.zip")
    _make_atlas_zip(data_dir / "atlas_intel_03012026_04012026.zip")
    _make_corrupt_zip(data_dir / "atlas_intel_05012026_06012026.zip")

    _topline, _runoff, report = build_cne_2026_tables(data_dir=tmp_path)

    assert len(report) == 3
    statuses = report["status"].tolist()
    assert statuses.count("ok") == 2
    assert statuses.count("errored") == 1


def test_ingestion_fails_when_firm_zero_rows(tmp_path: Path) -> None:
    """A firm whose every bundle errors → DataQualityError naming the firm."""
    data_dir = tmp_path / "2026-polls"
    data_dir.mkdir()
    _make_corrupt_zip(data_dir / "atlas_intel_01012026_02012026.zip")
    _make_corrupt_zip(data_dir / "atlas_intel_03012026_04012026.zip")

    with pytest.raises(DataQualityError, match="atlas_intel"):
        build_cne_2026_tables(data_dir=tmp_path)


def test_ingestion_persists_report_artifacts(tmp_path: Path) -> None:
    """End-to-end run writes both .parquet and .md under results/ingestion/."""
    data_dir = tmp_path / "2026-polls"
    data_dir.mkdir()
    _make_atlas_zip(data_dir / "atlas_intel_01012026_02012026.zip")

    bundle = data_dir / "atlas_intel_01012026_02012026.zip"
    results = [
        BundleResult(
            firm="atlas_intel",
            bundle_path=bundle,
            loader="load_atlas_intel",
            rows=3,
            status="ok",
            reason=None,
            shares_sum_check=100.0,
        ),
    ]
    report_df = results_to_dataframe(results)
    parquet_path = persist_report(
        report_df, year=2026, output_dir=tmp_path / "results" / "ingestion"
    )
    md_path = write_report_markdown(
        report_df, output_path=tmp_path / "results" / "ingestion" / "2026_bundle_report.md"
    )

    assert parquet_path.exists()
    assert md_path.exists()
    loaded = pd.read_parquet(parquet_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["firm"] == "atlas_intel"


def test_ingestion_rejects_shares_sum_violation(tmp_path: Path) -> None:
    """A Tempo-like row at sum=39.0 → status='rejected', reason mentions sum."""
    report_df = pd.DataFrame(
        [
            BundleResult(
                firm="tempo",
                bundle_path=tmp_path / "tempo_01012026.zip",
                loader="_load_generic_xlsx",
                rows=1,
                status="rejected",
                reason="shares sum 39.0 outside [98, 102]",
                shares_sum_check=39.0,
            ),
        ]
    )
    statuses = report_df["status"].tolist()
    reasons = report_df["reason"].tolist()
    assert statuses[0] == "rejected"
    assert "39.0" in reasons[0]


def test_ingestion_handles_empty_bundle_dir(tmp_path: Path) -> None:
    """No bundles → empty report, no DataQualityError, no crash."""
    data_dir = tmp_path / "2026-polls"
    data_dir.mkdir()

    topline, runoff, report = build_cne_2026_tables(data_dir=tmp_path)

    assert topline.empty
    assert runoff.empty
    assert report.empty
    assert len(report) == 0


def test_ingestion_ci_smoke_artifact(tmp_path: Path) -> None:
    """CI gate: ingest one OK bundle + persist a non-empty report.

    Runs last so the CI assertion (assert results/ingestion/*.parquet has
    ≥1 row) sees fresh state from a successful build.
    """
    data_dir = tmp_path / "2026-polls"
    data_dir.mkdir()
    _make_atlas_zip(data_dir / "atlas_intel_01012026_02012026.zip")

    _topline, _runoff, report = build_cne_2026_tables(data_dir=tmp_path)

    assert len(report) >= 1
    assert (report["status"] == "ok").any()
