"""SPEC-37: Tests for compositional data transforms and metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from co_president.fundamentals.compositional import (
    aitchison_distance,
    clr_transform,
    ilr_transform,
    impute_zero_shares,
    inverse_clr,
    inverse_ilr,
)

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════════
# clr_transform / inverse_clr
# ═══════════════════════════════════════════════════════════════════════


class TestCLRTransform:
    """CLR transform contract tests."""

    def test_uniform_composition_returns_zeros(self) -> None:
        """CLR of uniform composition yields all zeros."""
        x = np.array([0.25, 0.25, 0.25, 0.25])
        result = clr_transform(x)
        assert np.allclose(result, 0.0, atol=1e-14)

    def test_two_part_bipolar(self) -> None:
        """CLR of (0.3, 0.7) returns equal-magnitude opposite signs."""
        x = np.array([0.3, 0.7])
        result = clr_transform(x)
        assert np.isclose(result[0], -result[1], atol=1e-14)

    def test_roundtrip(self) -> None:
        """Inverse CLR recovers original composition."""
        x = np.array([0.1, 0.3, 0.4, 0.2])
        z = clr_transform(x)
        recovered = inverse_clr(z)
        assert np.allclose(recovered, x, atol=1e-14)

    def test_clr_sum_zero(self) -> None:
        """Elements of CLR-transformed vector sum to zero."""
        x = np.array([0.05, 0.15, 0.6, 0.2])
        z = clr_transform(x)
        assert np.isclose(z.sum(), 0.0, atol=1e-14)

    def test_rowwise_matrix(self) -> None:
        """CLR on 2-D array operates row-wise."""
        x_mat = np.array([[0.7, 0.3], [0.2, 0.8]])
        z_mat = clr_transform(x_mat)
        assert z_mat.shape == (2, 2)
        assert np.allclose(z_mat[0].sum(), 0.0, atol=1e-14)
        assert np.allclose(z_mat[1].sum(), 0.0, atol=1e-14)

    def test_raises_on_zeros(self) -> None:
        """CLR raises ValueError if any element is zero."""
        x = np.array([0.5, 0.0, 0.5])
        with pytest.raises(ValueError, match="positive"):
            clr_transform(x)

    def test_raises_on_negative(self) -> None:
        """CLR raises ValueError on negative values."""
        x = np.array([0.5, -0.1, 0.6])
        with pytest.raises(ValueError, match="positive"):
            clr_transform(x)

    def test_raises_on_non_finite(self) -> None:
        """CLR raises ValueError on NaN or Inf."""
        x = np.array([0.5, np.nan, 0.5])
        with pytest.raises(ValueError, match="finite"):
            clr_transform(x)

    def test_scalar_input_raises(self) -> None:
        """CLR raises TypeError on scalar input."""
        with pytest.raises(TypeError):
            clr_transform(0.5)  # type: ignore[arg-type]

    def test_1d_array(self) -> None:
        """CLR accepts 1-D array returning 1-D array."""
        x = np.array([0.2, 0.3, 0.5])
        z = clr_transform(x)
        assert z.ndim == 1
        assert np.isclose(z.sum(), 0.0, atol=1e-14)

    def test_inverse_clr_produces_simplex(self) -> None:
        """Inverse CLR returns shares that sum to 1."""
        z = np.array([0.5, -0.3, -0.2])
        x = inverse_clr(z)
        assert np.all(x >= 0)
        assert np.isclose(x.sum(), 1.0, atol=1e-14)

    def test_inverse_clr_rowwise(self) -> None:
        """Inverse CLR on 2-D array operates row-wise."""
        z_mat = np.array([[0.5, -0.5], [-0.3, 0.3]])
        x_mat = inverse_clr(z_mat)
        assert x_mat.shape == (2, 2)
        assert np.allclose(x_mat[0].sum(), 1.0, atol=1e-14)
        assert np.allclose(x_mat[1].sum(), 1.0, atol=1e-14)

    def test_roundtrip_matrix(self) -> None:
        """Round-trip CLR → inverse_CLR matches on matrix input."""
        x_mat = np.array([[0.1, 0.3, 0.6], [0.25, 0.25, 0.5], [0.4, 0.4, 0.2]])
        z_mat = clr_transform(x_mat)
        recovered = inverse_clr(z_mat)
        assert np.allclose(recovered, x_mat, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# ilr_transform / inverse_ilr
# ═══════════════════════════════════════════════════════════════════════


class TestILRTransform:
    """ILR transform contract tests."""

    def test_2_part_single_row(self) -> None:
        """ILR of 2-part composition returns single value."""
        x = np.array([0.3, 0.7])
        z = ilr_transform(x)
        assert z.shape == (1,)

    def test_roundtrip_2_part(self) -> None:
        """Round-trip ILR on 2-part composition."""
        x = np.array([0.3, 0.7])
        z = ilr_transform(x)
        recovered = inverse_ilr(z)
        assert np.allclose(recovered, x, atol=1e-14)

    def test_roundtrip_3_part(self) -> None:
        """Round-trip ILR on 3-part composition."""
        x = np.array([0.1, 0.3, 0.6])
        z = ilr_transform(x)
        recovered = inverse_ilr(z)
        assert np.allclose(recovered, x, atol=1e-14)

    def test_roundtrip_4_part(self) -> None:
        """Round-trip ILR on 4-part composition."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        z = ilr_transform(x)
        recovered = inverse_ilr(z)
        assert np.allclose(recovered, x, atol=1e-14)

    def test_roundtrip_matrix(self) -> None:
        """Round-trip ILR → inverse_ILR matches on matrix input."""
        x_mat = np.array([[0.1, 0.3, 0.6], [0.25, 0.25, 0.5], [0.4, 0.4, 0.2]])
        z_mat = ilr_transform(x_mat)
        recovered = inverse_ilr(z_mat)
        assert np.allclose(recovered, x_mat, atol=1e-14)

    def test_raises_on_zeros(self) -> None:
        """ILR raises ValueError on zero values."""
        x = np.array([0.5, 0.0, 0.5])
        with pytest.raises(ValueError, match="positive"):
            ilr_transform(x)

    def test_raises_on_negative(self) -> None:
        """ILR raises ValueError on negative values."""
        x = np.array([0.5, -0.1, 0.6])
        with pytest.raises(ValueError, match="positive"):
            ilr_transform(x)

    def test_default_contrast_valid_form(self) -> None:
        """Default contrast matrix has orthonormal rows of expected shape."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        z = ilr_transform(x)
        assert z.shape == (3,), "D parts → D-1 ilr coordinates"


# ═══════════════════════════════════════════════════════════════════════
# impute_zero_shares
# ═══════════════════════════════════════════════════════════════════════


class TestImputeZeroShares:
    """Smithson-Verkuilen zero-imputation contract tests."""

    def test_positive_values_unchanged(self) -> None:
        """Already-positive shares unchanged when no zeros present."""
        y = np.array([0.2, 0.3, 0.5])
        result = impute_zero_shares(y)
        assert np.allclose(result, y, atol=1e-14)

    def test_replaces_zeros_with_positive(self) -> None:
        """Zero entries become positive after imputation."""
        y = np.array([0.5, 0.0, 0.5])
        result = impute_zero_shares(y)
        assert np.all(result > 0)

    def test_sums_to_one_after_imputation(self) -> None:
        """Imputed shares still sum to 1."""
        y = np.array([0.4, 0.0, 0.6])
        result = impute_zero_shares(y)
        assert np.isclose(result.sum(), 1.0, atol=1e-14)

    def test_multiple_zeros(self) -> None:
        """Handles multiple zero entries."""
        y = np.array([0.0, 0.0, 1.0])
        result = impute_zero_shares(y)
        assert np.all(result > 0)
        assert np.isclose(result.sum(), 1.0, atol=1e-14)

    def test_all_zeros(self) -> None:
        """All-zero input produces equal positive shares."""
        y = np.array([0.0, 0.0, 0.0])
        result = impute_zero_shares(y)
        assert np.allclose(result, 1.0 / 3, atol=1e-14)

    def test_rowwise_matrix(self) -> None:
        """Zero imputation on 2-D array operates row-wise."""
        y_mat = np.array([[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
        result = impute_zero_shares(y_mat)
        assert result.shape == (2, 3)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-14)
        assert np.all(result > 0)

    def test_raises_on_negative_values(self) -> None:
        """Imputation raises ValueError on negative values."""
        y = np.array([0.5, -0.1, 0.6])
        with pytest.raises(ValueError, match="non-negative"):
            impute_zero_shares(y)

    def test_raises_on_non_finite(self) -> None:
        """Imputation raises ValueError on NaN."""
        y = np.array([0.5, np.nan, 0.5])
        with pytest.raises(ValueError, match="finite"):
            impute_zero_shares(y)

    def test_default_delta(self) -> None:
        """Default delta = min(0.001, 0.5/C)."""
        y = np.array([0.5, 0.0, 0.5])
        result = impute_zero_shares(y)
        n_parts = 3
        expected_delta = min(0.001, 0.5 / n_parts)
        expected_scale = (1.0 - 1 * expected_delta) / 1.0
        expected_zero = expected_delta
        expected_pos = 0.5 * expected_scale
        assert np.isclose(result[0], expected_pos, atol=1e-14)
        assert np.isclose(result[1], expected_zero, atol=1e-14)
        assert np.isclose(result[2], expected_pos, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# aitchison_distance
# ═══════════════════════════════════════════════════════════════════════


class TestAitchisonDistance:
    """Aitchison distance contract tests."""

    def test_self_distance_zero(self) -> None:
        """Distance to self is zero."""
        x = np.array([0.2, 0.3, 0.5])
        d = aitchison_distance(x, x)
        assert np.isclose(d, 0.0, atol=1e-14)

    def test_symmetric(self) -> None:
        """Distance is symmetric: d(x, y) == d(y, x)."""
        x = np.array([0.2, 0.3, 0.5])
        y = np.array([0.4, 0.4, 0.2])
        assert np.isclose(
            aitchison_distance(x, y),
            aitchison_distance(y, x),
            atol=1e-14,
        )

    def test_non_negative(self) -> None:
        """Distance is non-negative."""
        x = np.array([0.2, 0.3, 0.5])
        y = np.array([0.4, 0.4, 0.2])
        assert aitchison_distance(x, y) >= 0.0

    def test_rowwise_matrix(self) -> None:
        """Distance between two matrices returns array of row distances."""
        x_mat = np.array([[0.7, 0.3], [0.2, 0.8]])
        y_mat = np.array([[0.6, 0.4], [0.3, 0.7]])
        d = aitchison_distance(x_mat, y_mat)
        assert d.shape == (2,)
        assert np.all(d >= 0)

    def test_mismatched_shape_raises(self) -> None:
        """Distance raises ValueError on mismatched shapes."""
        x = np.array([0.2, 0.3, 0.5])
        y = np.array([0.4, 0.6])
        with pytest.raises(ValueError, match="shape"):
            aitchison_distance(x, y)

    def test_known_value_2_part(self) -> None:
        """Known Aitchison distance for 2-part composition."""
        x = np.array([0.5, 0.5])
        y = np.array([0.9, 0.1])
        d = aitchison_distance(x, y)
        expected = math.sqrt(2.0) * math.log(3.0)
        assert np.isclose(d, expected, atol=1e-14)

    def test_raises_on_zeros(self) -> None:
        """Aitchison distance raises ValueError on zeros."""
        x = np.array([0.5, 0.0, 0.5])
        y = np.array([0.3, 0.4, 0.3])
        with pytest.raises(ValueError, match="positive"):
            aitchison_distance(x, y)
