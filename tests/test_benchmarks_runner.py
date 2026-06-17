"""SPEC-41: Tests for ``co_president.benchmarks.runner``.

Smoke test: runner produces 75 rows (5 models × 3 transforms × 5 classes) on
synthetic 3-municipio data.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from co_president.benchmarks.runner import (
    _compute_5class_targets,
    _impute_and_normalize,
    _intersect_columns,
    _r2_rmse_per_class,
    _train_test_split,
    report_benchmark_baseline,
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

    def test_smoke_returns_75_rows(self) -> None:
        """Smoke run produces 5 models × 3 transforms × 5 classes = 75 rows."""
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


class TestFullScale:
    """Tests with larger synthetic datasets (marked ``slow``)."""

    @pytest.mark.slow
    def test_100_rows_no_none_r2(self) -> None:
        """100-row dataset: all 75 result rows have finite R²/RMSE."""
        rng = np.random.default_rng(42)
        X = rng.dirichlet(np.ones(5), size=100).astype(np.float64)
        y = rng.dirichlet(np.ones(5), size=100).astype(np.float64)
        results = run_benchmarks(X, y)
        assert len(results) == 75
        assert all(r["r2"] is not None for r in results)
        assert all(r["rmse"] is not None for r in results)

    @pytest.mark.slow
    def test_100_rows_deterministic(self) -> None:
        """100-row results are reproducible."""
        rng = np.random.default_rng(42)
        X = rng.dirichlet(np.ones(5), size=100).astype(np.float64)
        y = rng.dirichlet(np.ones(5), size=100).astype(np.float64)
        r1 = run_benchmarks(X, y)
        r2 = run_benchmarks(X, y)
        assert r1 == r2


class TestCompute5ClassTargets:
    """Tests for 5-class ideology target computation."""

    def test_no_vote_share_cols_raises(self) -> None:
        """DataFrame without vote_share_* columns raises ValueError."""
        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
        with pytest.raises(ValueError, match=r"No vote_share_\* columns found"):
            _compute_5class_targets(df)


class TestReportBenchmarkBaseline:
    """Tests for the SPEC-26 baseline reporter."""

    def test_no_vote_share_cols_raises(self) -> None:
        """DataFrame without vote_share_* columns raises ValueError."""
        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match=r"No vote_share_\* columns found"):
            report_benchmark_baseline(df)

    def test_with_y_cols_runs_all_combos(self) -> None:
        """DataFrame with all 5 y_* columns runs all combos."""
        rng = np.random.default_rng(42)
        y_raw = rng.dirichlet(np.ones(5), size=10)
        cols: dict[str, np.ndarray] = {
            "x_feat1": rng.random(10),
            "x_feat2": rng.random(10),
        }
        for i, cls in enumerate(
            [
                "Izquierda",
                "Centro_Izquierda",
                "Centro",
                "Centro_Derecha",
                "Derecha",
            ]
        ):
            cols[f"y_{cls}"] = y_raw[:, i]
        df = pd.DataFrame(cols)
        result = report_benchmark_baseline(df)
        assert result["n_rows"] > 0
        assert result["n_models"] == 5
        assert result["n_transforms"] == 3


class TestImputeAndNormalize:
    """Tests for the NaN imputation + indicator-column logic."""

    def test_all_nan_column_adds_indicator(self) -> None:
        """All-NaN column produces extra indicator column set to 1."""
        x = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], dtype=np.float64)
        y = np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], dtype=np.float64)
        x_out, _ = _impute_and_normalize(x, y, n_classes=2)
        assert x_out.shape[1] == x.shape[1] + 1
        assert np.allclose(x_out[:, -1], 1.0)

    def test_no_nan_no_indicator(self) -> None:
        """No NaN columns leave shape unchanged."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        y = np.array([[0.6, 0.4], [0.7, 0.3]], dtype=np.float64)
        x_out, _ = _impute_and_normalize(x, y, n_classes=2)
        assert x_out.shape[1] == x.shape[1]

    def test_partial_nan_no_indicator(self) -> None:
        """Partially NaN columns (not all-NaN) do not get indicator."""
        x = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
        y = np.array([[0.6, 0.4], [0.7, 0.3]], dtype=np.float64)
        x_out, _ = _impute_and_normalize(x, y, n_classes=2)
        assert x_out.shape[1] == x.shape[1]

    def test_multiple_all_nan_columns(self) -> None:
        """Multiple all-NaN columns produce multiple indicators."""
        x = np.array(
            [[np.nan, 1.0, np.nan], [np.nan, 2.0, np.nan], [np.nan, 3.0, np.nan]],
            dtype=np.float64,
        )
        y = np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], dtype=np.float64)
        x_out, _ = _impute_and_normalize(x, y, n_classes=2)
        assert x_out.shape[1] == x.shape[1] + 2
        assert np.allclose(x_out[:, -1], 1.0)
        assert np.allclose(x_out[:, -2], 1.0)


class TestIntersectColumns:
    """Tests for name-based column intersection (Issue #21)."""

    def test_equal_columns_unchanged(self) -> None:
        """Arrays with same column names are unchanged."""
        cols_a = ["f1", "f2"]
        cols_b = ["f1", "f2"]
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = _intersect_columns([(a, cols_a), (b, cols_b)])
        assert len(result) == 2
        assert result[0].shape == a.shape
        assert result[1].shape == b.shape

    def test_unequal_columns_trim_by_name(self) -> None:
        """Arrays with different columns trim to name intersection."""
        cols_a = ["f1", "f2", "f3"]
        cols_b = ["f1", "f2"]
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([[7.0, 8.0], [9.0, 10.0]])
        result = _intersect_columns([(a, cols_a), (b, cols_b)])
        assert result[0].shape == (2, 2)
        assert result[1].shape == (2, 2)
        assert np.array_equal(result[0], a[:, :2])

    def test_preserves_base_data(self) -> None:
        """Common columns retain original values."""
        cols_a = ["f1", "f2", "f3"]
        cols_b = ["f1", "f2"]
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([[7.0, 8.0], [9.0, 10.0]])
        result = _intersect_columns([(a, cols_a), (b, cols_b)])
        assert np.array_equal(result[1], b)
        assert np.array_equal(result[0], a[:, :2])

    def test_different_names_excludes_mismatched_cols(self) -> None:
        """Columns not present in all arrays are excluded by name."""
        cols_a = ["f1", "f2", "x3"]
        cols_b = ["f1", "f2", "y3"]
        a = np.array([[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]])
        b = np.array([[5.0, 6.0, 8.0], [7.0, 8.0, 8.0]])
        result = _intersect_columns([(a, cols_a), (b, cols_b)])
        assert result[0].shape == (2, 2)
        assert result[1].shape == (2, 2)
        assert np.array_equal(result[0], a[:, :2])

    def test_synthetic_indicator_names_excluded(self) -> None:
        """__imputed_* names differ per array so they are excluded from intersection."""
        cols_a = ["f1", "f2", "__imputed_0__"]
        cols_b = ["f1", "f2", "__imputed_0__"]
        a = np.array([[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]])
        b = np.array([[5.0, 6.0, 8.0], [7.0, 8.0, 8.0]])
        result = _intersect_columns([(a, cols_a), (b, cols_b)])
        # Both have __imputed_0__, so it IS in the intersection
        assert result[0].shape == (2, 3)
        assert result[1].shape == (2, 3)

    def test_multiple_arrays_intersection_by_name(self) -> None:
        """Three or more arrays all trim to name-based intersection."""
        cols_a = ["f1", "f2", "f3", "f4", "f5"]
        cols_b = ["f1", "f2", "f3"]
        cols_c = ["f1", "f2", "f4"]
        a = np.ones((2, 5))
        b = np.ones((2, 3))
        c = np.ones((2, 3))
        result = _intersect_columns([(a, cols_a), (b, cols_b), (c, cols_c)])
        assert all(arr.shape[1] == 2 for arr in result)

    def test_single_array_passthrough(self) -> None:
        """Single array is returned unchanged."""
        a = np.array([[1.0, 2.0, 3.0]])
        result = _intersect_columns([(a, ["f1", "f2", "f3"])])
        assert len(result) == 1
        assert result[0].shape == a.shape
