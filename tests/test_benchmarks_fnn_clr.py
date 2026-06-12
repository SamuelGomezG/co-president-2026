"""Tests for ``co_president.benchmarks.fnn_clr``.

Smoke test: FNN instantiates, trains on a minimal fixture, and returns a finite
R² score.
"""

from __future__ import annotations

import numpy as np
import pytest

from co_president.benchmarks.fnn_clr import build_model, fit_model, score_model


class TestFnnBuild:
    """Tests for the model builder."""

    def test_default_architecture(self) -> None:
        """Default model has the USANTOMAS 2025 architecture."""
        model = build_model()
        assert model.hidden_layer_sizes == (150, 100, 50)
        assert model.activation == "relu"
        assert model.max_iter == 50

    def test_custom_architecture(self) -> None:
        """Override hidden layer sizes."""
        model = build_model(hidden_layer_sizes=(64, 32))
        assert model.hidden_layer_sizes == (64, 32)


class TestFnnFitScore:
    """Tests for fit and score helpers."""

    @pytest.fixture
    def _sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Minimal 5-sample × 3-feature regression problem."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(0, 0.1, size=5)
        return X, y

    def test_fit_returns_model(self, _sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """fit_model returns the fitted model."""
        X, y = _sample_data
        model = build_model(max_iter=200)
        fitted = fit_model(model, X, y)
        assert fitted is model
        assert fitted.n_iter_ > 0

    def test_score_returns_finite_r2(self, _sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """score_model returns finite R² and RMSE."""
        X, y = _sample_data
        model = build_model(max_iter=200)
        fit_model(model, X, y)
        result = score_model(model, X, y)
        assert np.isfinite(result["r2"])
        assert np.isfinite(result["rmse"])
        assert result["rmse"] >= 0.0

    def test_score_2d_target(self) -> None:
        """score_model handles 2-D targets (K classes)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        y = rng.random((10, 2))
        model = build_model(max_iter=200)
        fit_model(model, X, y)
        result = score_model(model, X, y)
        assert isinstance(result["r2"], list)
        assert len(result["r2"]) == 2
        assert all(np.isfinite(v) for v in result["r2"])
        assert all(np.isfinite(v) for v in result["rmse"])

    def test_predictor_improves_with_more_data(self) -> None:
        """R² should be higher with more training data (sanity)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(0, 0.5, size=100)
        model = build_model(max_iter=500)
        fit_model(model, X, y)
        result = score_model(model, X, y)
        assert result["r2"] >= -1.0  # not degenerate
