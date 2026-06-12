"""SPEC-37: Compositional data transforms, zero-imputation, and Aitchison distance.

Provides CLR and ILR transforms for simplex-to-Euclidean mapping, the
corresponding inverse transforms, zero-value imputation (Smithson-Verkuilen
2006), and the Aitchison distance metric for compositional data.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

__all__ = [
    "aitchison_distance",
    "clr_transform",
    "ilr_transform",
    "impute_zero_shares",
    "inverse_clr",
    "inverse_ilr",
]

_MIN_PARTS: int = 2


# ═══════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════


def _check_positive_finite(x: np.ndarray, name: str = "x") -> None:
    """Validate that *x* contains only positive, finite values."""
    if not np.all(np.isfinite(x)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(x <= 0):
        msg = f"{name} must contain only positive values"
        raise ValueError(msg)


def _default_psi(d: int) -> np.ndarray:
    """Build default orthonormal contrast matrix for ILR.

    Uses a sequential binary partition (SBP) strategy.  Returns a ``(d-1, d)``
    matrix where each row is a unit-norm, pairwise-orthogonal contrast vector
    with entries of the form::

        c_j = +sqrt(s / (r * (r + s)))   for the *r* numerator  parts
        c_j = -sqrt(r / (s * (r + s)))   for the *s* denominator parts
        c_j =  0                          for parts not in the balance

    The resulting matrix satisfies ``psi @ psi.T = I``.
    """
    if d < _MIN_PARTS:
        msg = f"Need at least {_MIN_PARTS} parts for ILR, got {d}"
        raise ValueError(msg)

    psi = np.zeros((d - 1, d), dtype=np.float64)
    intervals: list[tuple[int, int]] = [(0, d)]
    row = 0

    while intervals and row < d - 1:
        start, end = intervals.pop(0)
        parts = end - start
        if parts <= 1:
            continue

        mid = start + parts // 2
        n_plus = mid - start
        n_minus = end - mid

        coeff_plus = math.sqrt(n_minus / (n_plus * (n_plus + n_minus)))
        coeff_minus = math.sqrt(n_plus / (n_minus * (n_plus + n_minus)))

        psi[row, start:mid] = coeff_plus
        psi[row, mid:end] = -coeff_minus

        intervals.append((start, mid))
        intervals.append((mid, end))
        row += 1

    return psi


def _to_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Promote 1-D array to (1, d); return ``(x2d, was_1d)``."""
    if x.ndim == 1:
        return x.reshape(1, -1), True
    return x, False


def _from_2d(x2d: np.ndarray, *, was_1d: bool = False) -> np.ndarray:
    """Reshape back to 1-D if input was originally 1-D."""
    if was_1d:
        return x2d.ravel()
    return x2d


# ═══════════════════════════════════════════════════════════════════════
# CLR transform
# ═══════════════════════════════════════════════════════════════════════


def clr_transform(x: np.ndarray) -> np.ndarray:
    """Centred log-ratio transform (Aitchison, 1982).

    Computes ``z_i = ln(x_i / g(x))`` where ``g(x)`` is the geometric mean of
    the composition.  Accepts a 1-D **(d,)** or 2-D **(n, d)** array and
    operates row-wise.

    Args:
        x: Positive compositional values summing to 1 along the last axis.

    Returns:
        CLR-transformed array (same shape).  Each row sums to zero.

    Raises:
        ValueError: If *x* contains non-positive, non-finite, or zero values.
        TypeError: If *x* is a scalar.

    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 0:
        msg = "clr_transform requires an array, not a scalar"
        raise TypeError(msg)

    _check_positive_finite(xa, name="x")
    log_x = np.log(xa)
    gm_log = log_x.mean(axis=-1, keepdims=True)
    return cast("np.ndarray", log_x - gm_log)


def inverse_clr(z: np.ndarray) -> np.ndarray:
    """Inverse CLR transform (softmax-based).

    Maps CLR coordinates back to the simplex via ``x_i = exp(z_i) / sum(exp(z))``.

    Args:
        z: CLR-transformed values.  Can be 1-D **(d,)** or 2-D **(n, d)**.

    Returns:
        Compositional values summing to 1 along the last axis.

    """
    za = np.asarray(z, dtype=np.float64)
    exp_z = np.exp(za)
    return cast("np.ndarray", exp_z / exp_z.sum(axis=-1, keepdims=True))


# ═══════════════════════════════════════════════════════════════════════
# ILR transform
# ═══════════════════════════════════════════════════════════════════════


def _compute_ilr(
    x: np.ndarray,
    psi: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared ILR logic: returns ``(z, psi_used)``."""
    x2d, was_1d = _to_2d(x)
    _check_positive_finite(x2d, name="x")

    d = x2d.shape[1]
    if psi is None:
        psi = _default_psi(d)
    elif psi.shape != (d - 1, d):
        msg = f"psi must be ({d - 1}, {d}), got {psi.shape}"
        raise ValueError(msg)

    log_x = np.log(x2d)
    gm_log = log_x.mean(axis=1, keepdims=True)
    clr_vals = log_x - gm_log

    z2d = clr_vals @ psi.T
    return _from_2d(z2d, was_1d=was_1d), psi


def ilr_transform(
    x: np.ndarray,
    psi: np.ndarray | None = None,
) -> np.ndarray:
    """Isometric log-ratio transform (Egozcue et al., 2003).

    Computes ``z = CLR(x) @ psi.T`` where ``psi`` is a ``(d-1, d)`` orthonormal
    contrast matrix.  When *psi* is ``None`` a default sequential binary
    partition basis is used.

    Args:
        x: Positive compositional values summing to 1.  1-D **(d,)** or
            2-D **(n, d)**.
        psi: Optional orthonormal contrast matrix ``(d-1, d)``.

    Returns:
        ILR coordinates: 1-D **(d-1,)** for a single row, 2-D **(n, d-1)** for
        a matrix.

    Raises:
        ValueError: If *x* contains non-positive, non-finite, or zero values,
            or if *psi* has incorrect shape.

    """
    z, _ = _compute_ilr(x, psi)
    return z


def inverse_ilr(
    z: np.ndarray,
    psi: np.ndarray | None = None,
    d: int | None = None,
) -> np.ndarray:
    """Inverse ILR transform.

    Maps ILR coordinates back to the simplex via
    ``x = inverse_clr(psi.T @ z)``.

    Args:
        z: ILR coordinates.  1-D **(d-1,)** or 2-D **(n, d-1)**.
        psi: Optional contrast matrix ``(d-1, d)``.  Must be provided if
            *d* is not given.
        d: Number of parts in the original composition.  Only used when
            *psi* is ``None`` to reconstruct the default contrast.

    Returns:
        Compositional values summing to 1.

    Raises:
        ValueError: If neither *psi* nor *d* is provided.

    """
    z2d, was_1d = _to_2d(np.asarray(z, dtype=np.float64))
    m = z2d.shape[1]
    d_actual = d or m + 1

    if psi is None:
        psi = _default_psi(d_actual)
    if psi.shape != (m, d_actual):
        msg = f"psi must be ({m}, {d_actual}), got {psi.shape}"
        raise ValueError(msg)

    clr_vals = z2d @ psi
    return _from_2d(inverse_clr(clr_vals), was_1d=was_1d)


# ═══════════════════════════════════════════════════════════════════════
# Zero imputation (Smithson-Verkuilen 2006)
# ═══════════════════════════════════════════════════════════════════════


def impute_zero_shares(
    y: np.ndarray,
    delta: float | None = None,
) -> np.ndarray:
    """Impute zero values in a composition (Smithson-Verkuilen, 2006).

    Replaces zeros with a small positive *delta* and rescales non-zero entries
    to preserve the unit-sum constraint.

    For each row (or the single vector), the imputation is::

        y*_zero    = delta
        y*_positive = (1 - n_zeros * delta) * y_i / sum(y_positive)

    where *n_zeros* is the count of zero entries and *delta* defaults to
    ``min(0.001, 0.5 / c)`` with *c* the number of compositional parts.

    Args:
        y: Compositional values summing to 1.  1-D **(c,)** or 2-D **(n, c)**.
        delta: Replacement value for zero entries.  Defaults to
            ``min(0.001, 0.5 / c)``.

    Returns:
        Imputed composition (same shape), all positive and summing to 1.

    Raises:
        ValueError: If *y* contains negative or non-finite values, or if
            *delta* is outside ``(0, 1)``.

    """
    y2d, was_1d = _to_2d(np.asarray(y, dtype=np.float64))

    if not np.all(np.isfinite(y2d)):
        msg = "y must contain only finite values"
        raise ValueError(msg)
    if np.any(y2d < 0):
        msg = "y must contain only non-negative values"
        raise ValueError(msg)

    c = y2d.shape[1]
    if delta is None:
        delta = min(0.001, 0.5 / c)
    if not 0 < delta < 1:
        msg = "delta must be in (0, 1)"
        raise ValueError(msg)

    zero_mask = y2d == 0
    n_zeros = zero_mask.sum(axis=1, keepdims=True)

    result = y2d.copy()
    result[zero_mask] = delta

    positive_mask = ~zero_mask
    sum_positive = np.where(
        positive_mask,
        y2d,
        0.0,
    ).sum(axis=1, keepdims=True)

    all_zero = sum_positive.ravel() == 0
    if all_zero.any():
        result[all_zero, :] = 1.0 / c

    scale = np.divide(
        1.0 - n_zeros * delta,
        sum_positive,
        where=sum_positive > 0,
        out=np.ones_like(sum_positive),
    )
    result = np.where(positive_mask, y2d * scale, result)

    return _from_2d(result, was_1d=was_1d)


# ═══════════════════════════════════════════════════════════════════════
# Aitchison distance
# ═══════════════════════════════════════════════════════════════════════


def aitchison_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Aitchison distance between two compositions (Aitchison, 1986).

    The Aitchison distance is the Euclidean distance between the CLR-transformed
    compositions::

        d_a(x, y) = sqrt(sum_i (clr(x)_i - clr(y)_i) ^ 2)

    Args:
        x: First composition.  1-D **(c,)** or 2-D **(n, c)**.
        y: Second composition (same shape as *x*).

    Returns:
        Distance(s): scalar for 1-D input, 1-D **(n,)** array for 2-D input.

    Raises:
        ValueError: If *x* and *y* have different shapes, or contain
            non-positive or non-finite values.

    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)

    if xa.shape != ya.shape:
        msg = f"x and y must have the same shape, got {xa.shape} and {ya.shape}"
        raise ValueError(msg)

    clr_x = clr_transform(xa)
    clr_y = clr_transform(ya)

    diff = clr_x - clr_y
    return cast("np.ndarray", np.sqrt((diff**2).sum(axis=-1)))
