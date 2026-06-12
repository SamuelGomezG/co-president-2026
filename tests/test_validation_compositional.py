"""Tests for SPEC-37: Compositional validation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from co_president.validation.compositional import (
    aitchison_mae,
    aitchison_residual,
    composite_r2,
)


class TestAitchisonResidual:
    """Tests for ``aitchison_residual``."""

    def test_perfect_match(self) -> None:
        """CLR residual is zero for identical compositions."""
        x = np.array([0.2, 0.3, 0.5])
        r = aitchison_residual(x, x)
        assert np.allclose(r, 0.0)

    def test_2d_input(self) -> None:
        """CLR residual handles batch (N, C) input."""
        x = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
        r = aitchison_residual(x, x)
        assert r.shape == (2, 3)
        assert np.allclose(r, 0.0)

    def test_nonzero_residual(self) -> None:
        """CLR residual is non-zero for different compositions."""
        y_true = np.array([0.2, 0.3, 0.5])
        y_pred = np.array([0.25, 0.25, 0.5])
        r = aitchison_residual(y_true, y_pred)
        assert np.any(np.abs(r) > 0)

    def test_shape_mismatch_raises(self) -> None:
        """Residual raises ValueError for mismatched shapes."""
        with pytest.raises(ValueError, match="same shape"):
            aitchison_residual(np.array([0.2, 0.3, 0.5]), np.array([0.2, 0.8]))


class TestCompositeR2:
    """Tests for ``composite_r2``."""

    def test_perfect_prediction(self) -> None:
        """ILR R² is 1.0 for identical compositions."""
        x = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        r2 = composite_r2(x, x)
        assert r2 == pytest.approx(1.0)

    def test_1d_perfect(self) -> None:
        """ILR R² is 1.0 for single-row perfect match."""
        x = np.array([0.2, 0.3, 0.5])
        r2 = composite_r2(x, x)
        assert r2 == pytest.approx(1.0)

    def test_random_prediction_lower_than_perfect(self) -> None:
        """Random prediction yields lower ILR R² than perfect."""
        rng = np.random.default_rng(42)
        n = 20
        alpha = rng.uniform(1, 5, size=n).astype(np.float64)
        beta = rng.uniform(1, 5, size=n).astype(np.float64)
        gamma = rng.uniform(1, 5, size=n).astype(np.float64)
        sums = alpha + beta + gamma
        y_true = np.column_stack([alpha / sums, beta / sums, gamma / sums])
        y_pred = np.column_stack([beta / sums, gamma / sums, alpha / sums])

        r2_rand = composite_r2(y_true, y_pred)
        r2_perfect = composite_r2(y_true, y_true)
        assert r2_rand < r2_perfect

    def test_shape_mismatch_raises(self) -> None:
        """R² raises ValueError for mismatched shapes."""
        with pytest.raises(ValueError, match="same shape"):
            composite_r2(np.array([[0.2, 0.3, 0.5]]), np.array([[0.2, 0.8]]))


class TestAitchisonMAE:
    """Tests for ``aitchison_mae``."""

    def test_perfect_match(self) -> None:
        """MAE is zero for identical batch compositions."""
        x = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        mae = aitchison_mae(x, x)
        assert mae == pytest.approx(0.0)

    def test_1d_perfect(self) -> None:
        """MAE is zero for single-row perfect match."""
        x = np.array([0.2, 0.3, 0.5])
        mae = aitchison_mae(x, x)
        assert mae == pytest.approx(0.0)

    def test_nonzero_distance(self) -> None:
        """MAE is positive for different compositions."""
        y_true = np.array([0.2, 0.3, 0.5])
        y_pred = np.array([0.33, 0.33, 0.34])
        mae = aitchison_mae(y_true, y_pred)
        assert mae > 0

    def test_shape_mismatch_raises(self) -> None:
        """MAE raises ValueError for mismatched shapes."""
        with pytest.raises(ValueError, match="same shape"):
            aitchison_mae(np.array([0.2, 0.3, 0.5]), np.array([0.2, 0.8]))
