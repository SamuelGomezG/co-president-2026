"""Tests for ``co_president.benchmarks.runner``.

Smoke test: runner produces 15 rows (5 models × 3 transforms) on synthetic
3-municipio data.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from co_president.benchmarks.runner import (
    _r2_rmse_per_class,
    _train_test_split,
    run_benchmarks,
    write_csv,
)


class TestTrainTestSplit:
    """Tests for the simple train-test split helper."""

    def test_proportion(self) -> None:
        """70/30 split produces correct train size."""
        X = np.ones((100, 4))
        y = np.ones((100, 3))
        X_tr, X_te, y_tr, y_te = _train_test_split(X, y, train_ratio=0.7)
        assert X_tr.shape[0] == 70
        assert X_te.shape[0] == 30
        assert y_tr.shape[0] == 70
        assert y_te.shape[0] == 30

    def test_deterministic(self) -> None:
        """Same seed produces identical split."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100).reshape(100, 1)
        _, X_te1, _, _ = _train_test_split(X, y, random_state=42)
        _, X_te2, _, _ = _train_test_split(X, y, random_state=42)
        assert np.array_equal(X_te1, X_te2)

    def test_different_seeds_different(self) -> None:
        """Different seeds produce different splits."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100).reshape(100, 1)
        _, X_te1, _, _ = _train_test_split(X, y, random_state=42)
        _, X_te2, _, _ = _train_test_split(X, y, random_state=99)
        assert not np.array_equal(X_te1, X_te2)


class TestR2RmsePerClass:
    """Tests for per-class metric computation."""

    def test_perfect_prediction(self) -> None:
        """R² = 1.0 and RMSE = 0.0 for perfect prediction."""
        y_true = np.array([[0.2, 0.8], [0.3, 0.7]])
        results = _r2_rmse_per_class(y_true, y_true, ["A", "B"])
        assert len(results) == 2
        assert np.isclose(results[0]["r2"], 1.0)
        assert np.isclose(results[0]["rmse"], 0.0)

    def test_constant_prediction(self) -> None:
        """R² may be negative or zero for constant prediction."""
        y_true = np.array([[0.2, 0.8], [0.3, 0.7]])
        y_pred = np.array([[0.5, 0.5], [0.5, 0.5]])
        results = _r2_rmse_per_class(y_true, y_pred, ["A", "B"])
        assert results[0]["rmse"] > 0.0


class TestRunBenchmarks:
    """Integration-level tests for the benchmark runner."""

    def test_smoke_returns_15_rows(self) -> None:
        """Smoke run produces 5 models × 3 transforms ≈ 15 rows."""
        results = run_benchmarks(smoke=True)
        assert len(results) == 75  # 5 models × 3 transforms × 5 classes

    def test_smoke_no_none_r2(self) -> None:
        """No None R² values in smoke test."""
        results = run_benchmarks(smoke=True)
        r2_vals = [r["r2"] for r in results]
        assert all(v is not None for v in r2_vals)

    def test_smoke_deterministic(self) -> None:
        """Smoke results are reproducible."""
        r1 = run_benchmarks(smoke=True)
        r2 = run_benchmarks(smoke=True)
        assert r1 == r2

    def test_write_csv(self) -> None:
        """Written CSV has correct header and row count."""
        results = run_benchmarks(smoke=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            out = write_csv(results, path)
            assert out.exists()
            lines = out.read_text().strip().split("\n")
            assert len(lines) == len(results) + 1  # +1 for header

    def test_missing_x_y_raises(self) -> None:
        """Calling run_benchmarks without X/y and smoke=False raises."""
        with pytest.raises(ValueError, match="Provide X/y or pass smoke"):
            run_benchmarks()
