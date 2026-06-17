"""SPEC-41: Feedforward neural network benchmark models.

Wraps ``sklearn.neural_network.MLPRegressor`` with the architecture specified
in USANTOMAS 2025 Fig 17 (hidden_layer_sizes=(150, 100, 50), activation='relu',
max_iter=50).  Provides ``build_model``, ``fit_model``, and ``score_model``
helpers for use by ``runner.py``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning  # type: ignore[reportMissingTypeStubs]
from sklearn.metrics import (  # type: ignore[reportMissingTypeStubs]
    r2_score,  # type: ignore[reportUnknownVariableType]
    root_mean_squared_error,  # type: ignore[reportUnknownVariableType]
)
from sklearn.neural_network import (  # type: ignore[reportMissingTypeStubs]
    MLPRegressor,  # type: ignore[reportUnknownVariableType]
)

logger = logging.getLogger(__name__)

_RANDOM_SEED = 332211


def build_model(
    *,
    hidden_layer_sizes: tuple[int, ...] = (150, 100, 50),
    activation: str = "relu",
    max_iter: int = 50,
    random_state: int = _RANDOM_SEED,
) -> MLPRegressor:
    """Build an ``MLPRegressor`` with the USANTOMAS 2025 default architecture.

    Args:
        hidden_layer_sizes: Neurons per hidden layer.
        activation: Activation function.
        max_iter: Maximum training epochs.
        random_state: Seed for reproducible training.

    Returns:
        Unfitted ``MLPRegressor`` instance.

    """
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
    )


def fit_model(
    model: MLPRegressor,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> MLPRegressor:
    """Fit an MLPRegressor.

    Args:
        model: Unfitted or partially fitted model.
        x: Training features of shape ``(N, F)``.
        y: Training targets of shape ``(N,)`` or ``(N, K)``.

    Returns:
        Fitted model.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        model.fit(x, y)  # type: ignore[reportUnknownMemberType]
    return model


def score_model(
    model: MLPRegressor,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> dict[str, float | list[float]]:
    """Compute R² score for an already-fitted model on given data.

    Computes per-column R² when *y* is 2-D, otherwise a single scalar.

    Args:
        model: Fitted model.
        x: Features of shape ``(N, F)``.
        y: Targets of shape ``(N,)`` or ``(N, K)``.

    Returns:
        Dictionary with:
        - ``"r2"``: R² score(s) — scalar for 1-D *y*, list for 2-D *y*.
        - ``"rmse"``: Root mean squared error — same shape convention.

    """
    y_pred = model.predict(x)  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if y.ndim == 1:
        return {
            "r2": float(r2_score(y, y_pred)),  # type: ignore[reportUnknownArgumentType]
            "rmse": float(root_mean_squared_error(y, y_pred)),  # type: ignore[reportUnknownArgumentType]
        }

    r2_list: list[float] = []
    rmse_list: list[float] = []
    for k in range(y.shape[1]):
        r2_list.append(float(r2_score(y[:, k], y_pred[:, k])))  # type: ignore[reportUnknownArgumentType, reportArgumentType, reportCallIssue]
        rmse_list.append(float(root_mean_squared_error(y[:, k], y_pred[:, k])))  # type: ignore[reportUnknownArgumentType, reportArgumentType, reportCallIssue]

    return {"r2": r2_list, "rmse": rmse_list}
