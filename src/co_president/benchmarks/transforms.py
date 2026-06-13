"""SPEC-41: Compositional data transforms (ALR, CLR, ILR).

Implements the three standard Aitchison log-ratio transforms for compositional
data analysis.  ALR and ILR map a *D*-part composition to a (*D*-1)-dimensional
real space; CLR maps to a *D*-dimensional real space with zero-sum constraint.

The ``compositions`` PyPI package does not exist (confirmed 2026-06-12); these
pure-NumPy implementations replace the R ``compositions`` package (Boogaart &
Tolosana-Delgado) used in USANTOMAS 2025.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from co_president.fundamentals.features import clr as _clr

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EPSILON = 1e-10


def alr_transform(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Additive log-ratio transform (Aitchison, 1982).

    Maps a *D*-part composition to (*D*-1) log-ratios using the last part as
    the denominator: ``alr(x)[i] = log(x[i] / x[-1])``.

    Args:
        x: Composition vector of shape ``(D,)`` or matrix ``(N, D)``.  All
            values must be non-negative and each row must sum to a positive
            number.

    Returns:
        ALR-transformed vector of shape ``(D-1,)`` or matrix ``(N, D-1)``.

    Raises:
        ValueError: If any entry is non-finite, negative, or a row sums to
            zero.

    Example:
        >>> alr_transform(np.array([0.3, 0.7]))
        array([-0.84729786...])

    """
    _validate_composition(x)
    if x.ndim == 1:
        return np.log(x[:-1] / x[-1])

    return np.log(x[:, :-1] / x[:, -1:])


def clr_transform(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Centred log-ratio transform (Aitchison, 1982).

    Delegates to ``co_president.fundamentals.features.clr`` for 1-D inputs.
    For 2-D inputs computes row-wise CLR using NumPy vectorised operations.

    Args:
        x: Composition vector ``(D,)`` or matrix ``(N, D)``.

    Returns:
        CLR-transformed data of the same shape.

    Raises:
        ValueError: If validation fails.

    Example:
        >>> clr_transform(np.array([0.3, 0.7]))
        array([-0.423648...,  0.423648...])

    """
    _validate_composition(x)
    if x.ndim == 1:
        return np.array(_clr(tuple(x)), dtype=np.float64)

    geomean = np.exp(np.mean(np.log(np.maximum(x, _EPSILON)), axis=1, keepdims=True))
    return np.log(x / geomean)


def ilr_transform(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Isometric log-ratio transform (Egozcue et al., 2003).

    Uses sequential binary partition (SBP) with the default orthonormal basis.
    Maps a *D*-part composition to (*D*-1) orthonormal coordinates.

    Args:
        x: Composition vector ``(D,)`` or matrix ``(N, D)``.

    Returns:
        ILR-transformed vector of shape ``(D-1,)`` or matrix ``(N, D-1)``.

    Raises:
        ValueError: If validation fails.

    """
    _validate_composition(x)
    if x.ndim == 1:
        d = len(x)
        ilr_coords = np.empty(d - 1)
        for i in range(d - 1):
            r, s = 1, d - i - 1
            sqrt_val = math.sqrt((r * s) / (r + s))
            numer = math.log(x[i] + _EPSILON)
            denom = np.mean(np.log(x[i + 1 :] + _EPSILON))
            ilr_coords[i] = sqrt_val * (numer - denom)
        return ilr_coords

    d = x.shape[1]
    ilr_coords = np.empty((x.shape[0], d - 1))
    log_x = np.log(np.maximum(x, _EPSILON))
    for i in range(d - 1):
        r, s = 1, d - i - 1
        sqrt_val = math.sqrt((r * s) / (r + s))
        numer = log_x[:, i]
        denom = np.mean(log_x[:, i + 1 :], axis=1)
        ilr_coords[:, i] = sqrt_val * (numer - denom)
    return ilr_coords


def _build_ilr_contrast(d: int) -> np.ndarray:
    """Build the (D-1, D) contrast matrix for sequential SBP ILR.

    Each row is one ILR coordinate: parts before i are 0, part i gets
    the positive weight, parts after i get the negative weight.

    Args:
        d: Number of parts (columns) in the original composition.

    Returns:
        Contrast matrix of shape ``(D-1, D)`` where ``ILR(x) = V @ CLR(x)``.

    """
    V = np.zeros((d - 1, d))
    for i in range(d - 1):
        s = d - i - 1  # denominator count
        a = math.sqrt(s / (s + 1)) if s > 0 else 1.0
        b = 1.0 / math.sqrt(s * (s + 1)) if s > 0 else 0.0
        V[i, i] = a
        if s > 0:
            V[i, i + 1 :] = -b
    return V


_CLIP_MAX: float = 500.0  # prevents overflow in exp()


def alr_inv_transform(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse additive log-ratio transform.

    Args:
        y: ALR-transformed data of shape ``(D-1,)`` or ``(N, D-1)``.

    Returns:
        Composition of shape ``(D,)`` or ``(N, D)`` that sums to 1.

    """
    y_safe = np.clip(y, -_CLIP_MAX, _CLIP_MAX)
    if y.ndim == 1:
        z = np.empty(y.shape[0] + 1)
        z[:-1] = np.exp(y_safe)
        z[-1] = 1.0
        return z / z.sum()

    z = np.empty((y_safe.shape[0], y_safe.shape[1] + 1))
    z[:, :-1] = np.exp(y_safe)
    z[:, -1] = 1.0
    return z / z.sum(axis=1, keepdims=True)


def clr_inv_transform(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse centred log-ratio transform (softmax).

    Args:
        y: CLR-transformed data of shape ``(D,)`` or ``(N, D)``.

    Returns:
        Composition of shape ``(D,)`` or ``(N, D)`` that sums to 1.

    """
    y_safe = np.clip(y, -_CLIP_MAX, _CLIP_MAX)
    if y.ndim == 1:
        z = np.exp(y_safe)
        return z / z.sum()

    z = np.exp(y_safe)
    return z / z.sum(axis=1, keepdims=True)


def ilr_inv_transform(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse isometric log-ratio transform.

    Computes ``CLR = y @ V`` using the SBP contrast matrix, then applies
    ``clr_inv_transform`` (softmax).

    Args:
        y: ILR-transformed data of shape ``(D-1,)`` or ``(N, D-1)``.

    Returns:
        Composition of shape ``(D,)`` or ``(N, D)`` that sums to 1.

    """
    y_safe = np.clip(y, -_CLIP_MAX, _CLIP_MAX)
    d = y_safe.shape[-1] + 1
    V = _build_ilr_contrast(d)

    if y_safe.ndim == 1:
        clr_coords = y_safe @ V
        return clr_inv_transform(clr_coords)

    clr_coords = y_safe @ V
    return clr_inv_transform(clr_coords)


def _validate_composition(x: NDArray[np.float64]) -> None:
    """Validate that *x* is a valid composition.

    Args:
        x: Array to validate.

    Raises:
        ValueError: If any entry is non-finite, negative, or a row sums to
            (near) zero.

    """
    if not np.all(np.isfinite(x)):
        msg = "Composition contains non-finite values"
        raise ValueError(msg)
    if np.any(x < 0):
        msg = "Composition contains negative parts"
        raise ValueError(msg)
    row_sums = x.sum(axis=-1)
    if np.any(row_sums <= _EPSILON):
        msg = f"Composition rows sum to <= {_EPSILON}"
        raise ValueError(msg)
