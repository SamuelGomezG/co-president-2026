"""Tests for ``co_president.benchmarks.transforms`` (ALR, CLR, ILR).

Mirrors the ``TestCLR`` pattern in ``tests/test_fundamentals_features.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from co_president.benchmarks.transforms import (
    _validate_composition,
    alr_transform,
    clr_transform,
    ilr_transform,
)


class TestValidateComposition:
    """Tests for the shared validation helper."""

    def test_passes_valid_1d(self) -> None:
        """Accept a valid 1-D composition."""
        _validate_composition(np.array([0.3, 0.7]))

    def test_passes_valid_2d(self) -> None:
        """Accept a valid 2-D composition matrix."""
        _validate_composition(np.array([[0.3, 0.7], [0.5, 0.5]]))

    def test_rejects_nan(self) -> None:
        """Reject composition with NaN values."""
        with pytest.raises(ValueError, match="non-finite"):
            _validate_composition(np.array([0.3, np.nan]))

    def test_rejects_inf(self) -> None:
        """Reject composition with infinity."""
        with pytest.raises(ValueError, match="non-finite"):
            _validate_composition(np.array([np.inf, 0.7]))

    def test_rejects_negative(self) -> None:
        """Reject composition with negative parts."""
        with pytest.raises(ValueError, match="negative"):
            _validate_composition(np.array([-0.1, 1.1]))

    def test_rejects_zero_row_sum(self) -> None:
        """Reject composition where a row sums to zero."""
        with pytest.raises(ValueError, match="sum to"):
            _validate_composition(np.array([0.0, 0.0]))

    def test_rejects_0d_scalar(self) -> None:
        """Reject 0-D scalar input."""
        with pytest.raises(ValueError, match="at least 1-D"):
            _validate_composition(np.array(0.5))

    def test_rejects_single_part(self) -> None:
        """Reject 1-D array with fewer than 2 parts."""
        with pytest.raises(ValueError, match="at least 2 parts"):
            _validate_composition(np.array([0.5]))


class TestAlrTransform:
    """Tests for the additive log-ratio transform."""

    def test_1d_simple(self) -> None:
        """ALR of (0.3, 0.7) -> log(0.3/0.7)."""
        result = alr_transform(np.array([0.3, 0.7]))
        expected = np.log(0.3 / 0.7)
        assert np.isclose(result, expected).all()

    def test_1d_three_parts(self) -> None:
        """ALR of (0.2, 0.3, 0.5) returns 2 elements."""
        result = alr_transform(np.array([0.2, 0.3, 0.5]))
        assert result.shape == (2,)
        expected = np.log(np.array([0.2, 0.3]) / 0.5)
        assert np.allclose(result, expected)

    def test_2d(self) -> None:
        """ALR of 3-column matrix returns 2-column matrix."""
        x = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
        result = alr_transform(x)
        assert result.shape == (2, 2)
        expected = np.log(x[:, :-1] / x[:, -1:])
        assert np.allclose(result, expected)

    def test_2d_single_row(self) -> None:
        """ALR of (1, 3) matrix returns (1, 2)."""
        x = np.array([[0.25, 0.25, 0.5]])
        result = alr_transform(x)
        assert result.shape == (1, 2)

    def test_invalid_passes_through(self) -> None:
        """Reject invalid composition passed to ALR."""
        with pytest.raises(ValueError, match="non-finite"):
            alr_transform(np.array([np.nan, 1.0]))

    def test_zero_part_produces_finite(self) -> None:
        """ALR clamps zero entries to epsilon, not -inf."""
        result = alr_transform(np.array([0.0, 0.5, 0.5]))
        assert np.all(np.isfinite(result))
        assert result.shape == (2,)

    def test_zero_denominator_produces_finite(self) -> None:
        """ALR clamps zero denominator to epsilon, not inf."""
        result = alr_transform(np.array([0.5, 0.5, 0.0]))
        assert np.all(np.isfinite(result))
        assert result.shape == (2,)


class TestClrTransform:
    """Tests for the centred log-ratio transform."""

    def test_1d_two_parts(self) -> None:
        """CLR of (0.3, 0.7) sums to zero."""
        result = clr_transform(np.array([0.3, 0.7]))
        assert np.isclose(result.sum(), 0.0)

    def test_1d_three_parts(self) -> None:
        """CLR of (0.2, 0.3, 0.5) sums to zero."""
        result = clr_transform(np.array([0.2, 0.3, 0.5]))
        assert np.isclose(result.sum(), 0.0)

    def test_2d(self) -> None:
        """CLR of 2-D matrix: each row sums to zero."""
        x = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
        result = clr_transform(x)
        assert result.shape == (2, 3)
        assert np.allclose(result.sum(axis=1), 0.0)

    def test_2d_single_row(self) -> None:
        """CLR of (1, 3) matrix returns (1, 3)."""
        x = np.array([[0.25, 0.25, 0.5]])
        result = clr_transform(x)
        assert result.shape == (1, 3)
        assert np.isclose(result.sum(), 0.0)

    def test_zero_part_produces_finite(self) -> None:
        """CLR clamps zero entries to epsilon, not -inf."""
        result = clr_transform(np.array([0.0, 0.5, 0.5]))
        assert np.all(np.isfinite(result))
        assert result.shape == (3,)


class TestIlrTransform:
    """Tests for the isometric log-ratio transform."""

    def test_1d_two_parts(self) -> None:
        """ILR of (0.3, 0.7) returns 1-D."""
        result = ilr_transform(np.array([0.3, 0.7]))
        assert result.shape == (1,)
        assert np.isfinite(result).all()

    def test_1d_three_parts(self) -> None:
        """ILR of (0.2, 0.3, 0.5) returns 2-D."""
        result = ilr_transform(np.array([0.2, 0.3, 0.5]))
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_2d(self) -> None:
        """ILR of (2, 3) -> (2, 2)."""
        x = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
        result = ilr_transform(x)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))

    def test_2d_single_row(self) -> None:
        """ILR of (1, 3) -> (1, 2)."""
        x = np.array([[0.25, 0.25, 0.5]])
        result = ilr_transform(x)
        assert result.shape == (1, 2)
        assert np.all(np.isfinite(result))

    def test_equivalence_with_clr_2d(self) -> None:
        """ILR should be recoverable via CLR + orthonormal basis for 2 parts."""
        x = np.array([0.3, 0.7])
        ilr_val = ilr_transform(x.copy())
        clr_val = clr_transform(x.copy())
        expected = (1.0 / math.sqrt(2)) * (clr_val[0] - clr_val[1])
        assert np.isclose(ilr_val[0], expected)

    def test_zero_part_produces_finite(self) -> None:
        """ILR clamps zero entries to epsilon, not -inf."""
        result = ilr_transform(np.array([0.0, 0.5, 0.5]))
        assert np.all(np.isfinite(result))
        assert result.shape == (2,)
