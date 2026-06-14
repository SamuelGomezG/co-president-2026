"""SPEC-37: Compositional validation metrics.

Provides composite residuals in ILR-space, Aitchison distance aggregation,
and ILR-space R-squared for compositional target evaluation.
"""

from __future__ import annotations

import numpy as np

from co_president.fundamentals.compositional import (
    aitchison_distance as _aitchison_distance,
)
from co_president.fundamentals.compositional import (
    clr_transform,
    ilr_transform,
)

__all__ = [
    "aitchison_mae",
    "aitchison_residual",
    "composite_r2",
]


def aitchison_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Element-wise Aitchison residual in CLR space.

    Computes ``r = CLR(y_true) - CLR(y_pred)``.

    Args:
        y_true: True composition.  Shape **(n, c)** or **(c,)**.
        y_pred: Predicted composition, same shape as *y_true*.

    Returns:
        Residual array, same shape as inputs.

    Raises:
        ValueError: If *y_true* and *y_pred* have different shapes, or
            contain non-positive or non-finite values.

    Examples:
        >>> import numpy as np
        >>> from co_president.validation.compositional import aitchison_residual
        >>> y_true = np.array([[0.6, 0.3, 0.1]])
        >>> y_pred = np.array([[0.5, 0.4, 0.1]])
        >>> r = aitchison_residual(y_true, y_pred)
        >>> r.shape == y_true.shape
        True

    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        msg = f"y_true and y_pred must have same shape, got {yt.shape} and {yp.shape}"
        raise ValueError(msg)

    return clr_transform(yt) - clr_transform(yp)


def composite_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """R-squared in ILR space (symmetric compositional metric).

    Transforms both *y_true* and *y_pred* to ILR coordinates, then computes
    the standard coefficient of determination.

    Args:
        y_true: True composition.  Shape **(n, c)** or **(c,)**.
        y_pred: Predicted composition, same shape as *y_true*.

    Returns:
        R-squared in (-inf, 1]; 1 is perfect.
        Returns ``NaN`` when all true compositions are identical
        (``ss_tot == 0``).

    Raises:
        ValueError: If *y_true* and *y_pred* have different shapes, or
            contain non-positive or non-finite values.

    Examples:
        >>> import numpy as np
        >>> from co_president.validation.compositional import composite_r2
        >>> y_true = np.array([[0.6, 0.3, 0.1]])
        >>> y_pred = np.array([[0.59, 0.31, 0.1]])
        >>> r2 = composite_r2(y_true, y_pred)
        >>> r2 >= -1.0
        True

    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        msg = f"y_true and y_pred must have same shape, got {yt.shape} and {yp.shape}"
        raise ValueError(msg)

    was_1d = yt.ndim == 1
    if was_1d:
        yt = yt.reshape(1, -1)
        yp = yp.reshape(1, -1)

    ilr_true = ilr_transform(yt)
    ilr_pred = ilr_transform(yp)

    ss_res = np.sum((ilr_true - ilr_pred) ** 2)
    ss_tot = np.sum((ilr_true - ilr_true.mean(axis=0, keepdims=True)) ** 2)

    if ss_tot == 0:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def aitchison_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Aitchison distance across rows.

    Args:
        y_true: True composition.  Shape **(n, c)** or **(c,)**.
        y_pred: Predicted composition, same shape as *y_true*.

    Returns:
        Mean Aitchison distance (scalar).

    Raises:
        ValueError: If *y_true* and *y_pred* have different shapes, or
            contain non-positive or non-finite values.

    Examples:
        >>> import numpy as np
        >>> from co_president.validation.compositional import aitchison_mae
        >>> y_true = np.array([[0.6, 0.3, 0.1]])
        >>> y_pred = np.array([[0.5, 0.4, 0.1]])
        >>> mae = aitchison_mae(y_true, y_pred)
        >>> mae >= 0.0
        True

    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        msg = f"y_true and y_pred must have same shape, got {yt.shape} and {yp.shape}"
        raise ValueError(msg)

    distances = _aitchison_distance(yt, yp)
    return float(distances.mean())
