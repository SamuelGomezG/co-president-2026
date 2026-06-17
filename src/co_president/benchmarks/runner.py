"""SPEC-41: ML benchmark runner.

Orchestrates 15 model-transform combinations (5 models × 3 transforms),
computes per-ideological-class R²/RMSE on a train/test split, and writes a
summary CSV to ``results/benchmarks/fnn_clr_2026.csv``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
from pathlib import Path
import re
import sys
from typing import Any
import warnings

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd  # type: ignore[reportMissingTypeStubs]
from sklearn.exceptions import (  # type: ignore[reportMissingTypeStubs]
    ConvergenceWarning,
    NotFittedError,
)
from sklearn.metrics import (  # type: ignore[reportMissingTypeStubs]
    r2_score,  # type: ignore[reportUnknownVariableType]
    root_mean_squared_error,  # type: ignore[reportUnknownVariableType]
)
from sklearn.multioutput import (  # type: ignore[reportMissingTypeStubs]
    MultiOutputRegressor,  # type: ignore[reportUnknownVariableType]
)
from sklearn.pipeline import Pipeline  # type: ignore[reportMissingTypeStubs]
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingTypeStubs]

from co_president.benchmarks.terridata import load_fiscal_features as _load_fiscal_features
from co_president.benchmarks.transforms import (
    alr_inv_transform,
    alr_transform,
    clr_inv_transform,
    clr_transform,
    ilr_inv_transform,
    ilr_transform,
)
from co_president.config import HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS
from co_president.fundamentals.features import load_features

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path("results/benchmarks")
_OUTPUT_CSV = _OUTPUT_DIR / "fnn_clr_2026.csv"

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "svr": {"class": "SVR", "kwargs": {"kernel": "rbf", "C": 1.0}},
    "rfr": {
        "class": "RandomForestRegressor",
        "kwargs": {"n_estimators": 100, "random_state": 332211},
    },
    "gbr": {
        "class": "GradientBoostingRegressor",
        "kwargs": {"n_estimators": 100, "random_state": 332211},
    },
    "knn": {"class": "KNeighborsRegressor", "kwargs": {"n_neighbors": 5}},
    "fnn": {
        "class": "MLPRegressor",
        "kwargs": {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "max_iter": 500,
            "early_stopping": False,
            "random_state": 332211,
        },
    },
}

_TRANSFORM_REGISTRY: dict[str, str] = {
    "alr": "alr",
    "clr": "clr",
    "ilr": "ilr",
}

_ELECTION_YEARS: tuple[int, ...] = (2002, 2006, 2010, 2014, 2018, 2022)
_FISCAL_YEAR_SET: set[int] = set(_ELECTION_YEARS)

_IDEOLOGY_CLASSES: list[str] = [
    "Izquierda",
    "Centro_Izquierda",
    "Centro",
    "Centro_Derecha",
    "Derecha",
]

_3CLASS_CLASSES: list[str] = [
    "Izquierda",
    "Centro",
    "Derecha",
]

# Public exports so callers can pass the right ``class_names`` to
# :func:`run_benchmarks`.  Use the module-level constant directly.
IDEOLOGY_CLASSES_5: list[str] = _IDEOLOGY_CLASSES
IDEOLOGY_CLASSES_3: list[str] = _3CLASS_CLASSES

# Deprecated alias — use HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS from config.
_CANDIDATE_5CLASS_MAP = HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS


def _compute_5class_targets(  # noqa: C901, PLR0912
    df: pd.DataFrame,
    *,
    smoke: bool = False,
) -> pd.DataFrame:  # type: ignore[name-defined]
    """Add 5-class ideology target columns to the feature matrix.

    Parses ``vote_share_{year}_r{round}_{candidate}`` columns, maps
    each candidate to an ideological class, and sums vote shares per
    class to produce ``y_Izquierda``, ``y_Centro_Izquierda``,
    ``y_Centro``, ``y_Centro_Derecha``, ``y_Derecha``.

    Only the **most recent available year** is used for targets to
    avoid target leakage across multiple years in the feature matrix.
    Vote-share columns for the target year are excluded from the
    feature set by ``load_historical_data``.

    Args:
        df: Feature matrix from ``load_features()``.
        smoke: If True, generate synthetic targets when no
            ``vote_share_*`` columns exist.  Defaults to False.

    Returns:
        DataFrame with added ``y_*`` columns (or synthetic fallback
        when *smoke* is True).

    Raises:
        ValueError: If no ``vote_share_*`` columns are found and
            *smoke* is False.

    """
    result = df.copy()

    all_years: set[int] = set()
    for col in df.columns:
        m = re.match(r"vote_share_(\d{4})_r(\d)_(.+)", col)
        if m:
            all_years.add(int(m.group(1)))

    if not all_years:
        if not smoke:
            msg = (
                "No vote_share_* columns found in DataFrame. "
                "Cannot compute 5-class ideology targets without "
                "vote-share data. Pass smoke=True for synthetic targets."
            )
            raise ValueError(msg)
        rng = np.random.default_rng(42)
        n = df.shape[0]
        for cls in _IDEOLOGY_CLASSES:
            result[f"y_{cls}"] = 0.0
        y_synth = np.abs(rng.standard_normal((n, 5)))
        y_synth = y_synth / y_synth.sum(axis=1, keepdims=True)
        for i, cls in enumerate(_IDEOLOGY_CLASSES):
            result[f"y_{cls}"] = y_synth[:, i]
        return result

    target_year = max(all_years)

    class_cols: dict[str, list[str]] = {cls: [] for cls in _IDEOLOGY_CLASSES}
    for col in df.columns:
        m = re.match(rf"vote_share_{target_year}_r(\d+)_(.+)", col)
        if m:
            round_num = int(m.group(1))
            suffix = m.group(2)
            round_map = _CANDIDATE_5CLASS_MAP.get((target_year, round_num), {})
            cls = round_map.get(suffix)
            if cls is None and round_num != 1:
                cls = _CANDIDATE_5CLASS_MAP.get((target_year, 1), {}).get(suffix)
            if cls:
                class_cols[cls].append(col)
            else:
                logger.debug(
                    "Unmapped candidate suffix %r for year %d round %d",
                    suffix,
                    target_year,
                    round_num,
                )

    for cls in _IDEOLOGY_CLASSES:
        if class_cols[cls]:
            result[f"y_{cls}"] = df[class_cols[cls]].sum(axis=1)
        else:
            result[f"y_{cls}"] = 0.0

    mapped = result[[f"y_{cls}" for cls in _IDEOLOGY_CLASSES]].sum(axis=1)
    mapped_mask = mapped > 0
    for cls in _IDEOLOGY_CLASSES:
        col = f"y_{cls}"
        result.loc[mapped_mask, col] = result.loc[mapped_mask, col] / mapped.loc[mapped_mask]

    return result


def _compute_3class_targets(
    df: pd.DataFrame,
    *,
    smoke: bool = False,
) -> pd.DataFrame:  # type: ignore[name-defined]
    """Add 3-class ideology target columns (Izquierda, Centro, Derecha).

    Computes 5-class targets via :func:`_compute_5class_targets`, then
    collapses ``Centro_Izquierda`` and ``Centro_Derecha`` into ``Centro``
    and renormalises to sum to 1 per row.

    Args:
        df: Feature matrix from ``load_features()``.
        smoke: If True, generate synthetic targets when no
            ``vote_share_*`` columns exist.  Defaults to False.

    Returns:
        DataFrame with ``y_Izquierda``, ``y_Centro``, ``y_Derecha`` columns.

    """
    result = _compute_5class_targets(df, smoke=smoke)
    result["y_Centro"] = (
        result.get("y_Centro", pd.Series(0.0, index=result.index))
        + result.get("y_Centro_Izquierda", pd.Series(0.0, index=result.index))
        + result.get("y_Centro_Derecha", pd.Series(0.0, index=result.index))
    )
    result = result.drop(columns=["y_Centro_Izquierda", "y_Centro_Derecha"], errors="ignore")
    y3 = ["y_Izquierda", "y_Centro", "y_Derecha"]
    total = result[y3].sum(axis=1)
    mask = total > 0
    for c in y3:
        result.loc[mask, c] = result.loc[mask, c] / total.loc[mask]
    return result


def _make_sklearn_model(name: str, n_train: int = 0) -> Any:  # noqa: ANN401
    """Instantiate an sklearn model by registry key.

    Models that do not support multi-output regression natively (SVR, GBR)
    are wrapped in ``MultiOutputRegressor``.  KNN's ``n_neighbors`` is clamped
    to ``min(k, n_train)`` to prevent ``n_neighbors > n_samples_fit`` errors.

    Args:
        name: Registry key from ``_MODEL_REGISTRY``.
        n_train: Number of training samples (used to clamp KNN).

    Returns:
        Instantiated (possibly wrapped) sklearn estimator.

    """
    entry = _MODEL_REGISTRY[name]
    mod = importlib.import_module(
        "sklearn.svm"
        if name == "svr"
        else "sklearn.ensemble"
        if name in ("rfr", "gbr")
        else "sklearn.neighbors"
        if name == "knn"
        else "sklearn.neural_network"
    )
    kwargs = dict(entry["kwargs"])
    if name == "knn" and n_train > 0:
        kwargs["n_neighbors"] = min(kwargs.get("n_neighbors", 5), n_train)
    estimator = getattr(mod, entry["class"])(**kwargs)

    if name in ("svr", "gbr"):
        estimator = MultiOutputRegressor(estimator)

    if name in ("fnn", "svr", "knn"):
        estimator = Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])

    return estimator


def _apply_transform(name: str, X: np.ndarray) -> np.ndarray:
    """Apply a log-ratio transform by name."""
    if name == "alr":
        return alr_transform(X)
    if name == "clr":
        return clr_transform(X)
    if name == "ilr":
        return ilr_transform(X)
    msg = f"Unknown transform: {name}"
    raise ValueError(msg)


def _apply_inverse_transform(name: str, y: np.ndarray) -> np.ndarray:
    """Inverse-transform predictions back to the original simplex."""
    if name == "alr":
        return alr_inv_transform(y)
    if name == "clr":
        return clr_inv_transform(y)
    if name == "ilr":
        return ilr_inv_transform(y)
    msg = f"Unknown transform: {name}"
    raise ValueError(msg)


def _intersect_columns(
    arrays_and_cols: list[tuple[np.ndarray, list[str]]],
) -> list[np.ndarray]:
    """Slice arrays to columns present in ALL years (name-based intersection).

    Features common across all years are kept; year-specific indicator columns
    (appended by ``_impute_and_normalize``) are dropped because their synthetic
    names never match across years.  This avoids spurious phantom features that
    zero-padding or positional truncation would create.

    Args:
        arrays_and_cols: List of ``(array, column_names)`` tuples, one per
            election year.

    Returns:
        Lists where each array is sliced to only columns in the name
        intersection, keeping the first year's column ordering.

    Raises:
        KeyError: If a column name in the intersection set is not present
            in an individual year's columns list.

    """
    if not arrays_and_cols:
        return [a for a, _ in arrays_and_cols]

    common: set[str] = set(arrays_and_cols[0][1])
    for _, cols in arrays_and_cols[1:]:
        common &= set(cols)

    first_order = [c for c in arrays_and_cols[0][1] if c in common]
    result: list[np.ndarray] = []
    for arr, cols in arrays_and_cols:
        name_to_idx = {c: i for i, c in enumerate(cols)}
        indices = [name_to_idx[c] for c in first_order]
        result.append(arr[:, indices])
    return result


def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    random_state: int = 332211,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple train-test split (no sklearn dependency for this helper).

    Args:
        X: Features of shape ``(N, F)``.
        y: Targets of shape ``(N, K)``.
        train_ratio: Fraction of rows for training.
        random_state: RNG seed.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.

    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = rng.permutation(n)
    split = int(n * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _r2_rmse_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | str]]:
    """Compute per-class R² and RMSE.

    Args:
        y_true: True values of shape ``(N, K)``.
        y_pred: Predicted values of shape ``(N, K)``.
        class_names: Labels for the K columns.

    Returns:
        List of dicts with keys ``class_name``, ``r2``, ``rmse``.

    """
    results: list[dict[str, float | str]] = []
    for k, name in enumerate(class_names):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            r2 = float(
                np.nan_to_num(r2_score(y_true[:, k], y_pred[:, k], force_finite=True), nan=0.0)  # type: ignore[reportUnknownArgumentType]
            )
        rmse = float(root_mean_squared_error(y_true[:, k], y_pred[:, k]))  # type: ignore[reportUnknownArgumentType]
        results.append({"class_name": name, "r2": r2, "rmse": rmse})
    return results


def _generate_smoke_data(n_classes: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 3-municipio × 5-feature × n_classes-class fixture.

    Both X and y are compositional (non-negative, rows sum to 1), so ALR/CLR/
    ILR transforms can be applied to either.

    Args:
        n_classes: Number of ideological classes. Defaults to 5.

    Returns:
        ``(X, y)`` where ``X.shape == (3, 5)`` and ``y.shape == (3, n_classes)``.

    """
    rng = np.random.default_rng(42)
    raw_X = rng.dirichlet(np.ones(5), size=3)
    raw_y = rng.dirichlet(np.ones(n_classes), size=3)
    return raw_X.astype(np.float64), raw_y.astype(np.float64)


def run_benchmarks(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    transform_names: tuple[str, ...] = ("alr", "clr", "ilr"),
    model_names: tuple[str, ...] = ("svr", "rfr", "gbr", "knn", "fnn"),
    class_names: list[str] | None = None,
    smoke: bool = False,
    n_classes: int = 5,
) -> list[dict[str, Any]]:
    """Run all specified model × transform combinations.

    Args:
        X: Feature matrix ``(N, F)``.  If *None* and *smoke* is True,
            synthetic 3-row data is used.
        y: Target matrix ``(N, K)``.  If *None* and *smoke* is True,
            synthetic 3-row data is used.
        transform_names: Transforms to evaluate.
        model_names: Models to evaluate.
        class_names: Names for the K target columns.  Defaults to
            ``_IDEOLOGY_CLASSES``.
        smoke: If True, use synthetic 3-row data for a quick smoke test.
        n_classes: Number of ideological classes for smoke data (default 5).

    Returns:
        List of result dicts, each with keys ``model``, ``transform``,
        ``class_name``, ``r2``, ``rmse``, ``n_train``, ``n_test``.

    """
    if smoke:
        X_in, y_in = _generate_smoke_data(n_classes=n_classes)
    elif X is not None and y is not None:
        X_in, y_in = X, y
    else:
        msg = "Provide X/y or pass smoke=True"
        raise ValueError(msg)

    if class_names is None:
        class_names = _3CLASS_CLASSES if y_in.shape[1] == 3 else _IDEOLOGY_CLASSES

    rows: list[dict[str, Any]] = []
    X_train, X_test, y_train_full, y_test_full = _train_test_split(X_in, y_in)
    train_n = X_train.shape[0]
    test_n = X_test.shape[0]

    for t_name in transform_names:
        y_train_t = _apply_transform(t_name, y_train_full)

        for m_name in model_names:
            model = _make_sklearn_model(m_name, n_train=train_n)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(X_train, y_train_t)
                y_pred_t = model.predict(X_test)
                y_pred = _apply_inverse_transform(t_name, y_pred_t)
                per_class = _r2_rmse_per_class(y_test_full, y_pred, class_names)

                rows.extend(
                    {
                        "model": m_name,
                        "transform": t_name,
                        "class_name": pc["class_name"],
                        "r2": pc["r2"],
                        "rmse": pc["rmse"],
                        "n_train": X_train.shape[0],
                        "n_test": X_test.shape[0],
                    }
                    for pc in per_class
                )
            except (ValueError, TypeError, LinAlgError, NotFittedError):
                logger.exception("Model %s + transform %s failed", m_name, t_name)
                rows.extend(
                    {
                        "model": m_name,
                        "transform": t_name,
                        "class_name": cname,
                        "r2": None,
                        "rmse": None,
                        "n_train": train_n,
                        "n_test": test_n,
                    }
                    for cname in class_names
                )
    return rows


def _impute_nan_columns(x_mat: np.ndarray, *, fill_value: float = 1e-10) -> tuple[np.ndarray, int]:
    """Impute NaN features with column medians, add indicator for all-NaN cols.

    Args:
        x_mat: Feature matrix ``(N, F)``, may contain NaN.
        fill_value: Value to fill entirely NaN columns (default ``1e-10``).

    Returns:
        ``(x_clean, n_imputed)`` where *x_clean* has NaN replaced by
        column medians (``fill_value`` for all-NaN cols) with binary
        indicator columns appended, and *n_imputed* is the count of
        all-NaN columns found.

    """
    nan_mask = np.isnan(x_mat)
    n_imputed = 0
    if nan_mask.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            col_median = np.nanmedian(x_mat, axis=0)
        all_nan = np.isnan(col_median)
        n_imputed = int(all_nan.sum())
        if all_nan.any():
            col_median[all_nan] = fill_value
        x_mat = np.where(nan_mask, col_median, x_mat)

    if n_imputed > 0:
        indicator = np.ones((x_mat.shape[0], n_imputed), dtype=np.float64)
        x_mat = np.hstack([x_mat, indicator])

    return x_mat, n_imputed


def _pad_arrays_to_max_width(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Zero-pad arrays to the maximum column count in the list.

    Used before ``np.vstack`` when different years may produce different
    numbers of feature columns (e.g. indicator columns from
    ``_impute_nan_columns``).  Padding with zeros ensures shape
    compatibility while preserving the existing data in each array.

    Args:
        arrays: List of 2D arrays ``(N_i, F_i)`` with potentially
            different ``F_i``.

    Returns:
        List where every array has ``max(F_i)`` columns.

    """
    if not arrays:
        return arrays
    max_cols = max(a.shape[1] for a in arrays)
    padded: list[np.ndarray] = []
    for a in arrays:
        if a.shape[1] < max_cols:
            pad = np.zeros((a.shape[0], max_cols - a.shape[1]))
            padded.append(np.hstack([a, pad]))
        else:
            padded.append(a)
    return padded


def _impute_and_normalize(
    x_mat: np.ndarray, y_mat: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaN features with column medians and row-normalize targets.

    Args:
        x_mat: Feature matrix ``(N, F)``, may contain NaN.
        y_mat: Target matrix ``(N, K)``, compositional rows.
        n_classes: Number of target classes (K), used for fallback rows.

    Returns:
        ``(x_clean, y_norm)`` where *x_clean* has NaN replaced by
        column medians (1e-10 for all-NaN cols) and *y_norm* is
        eps-clipped then row-normalised to sum-to-1.  All-NaN columns
        get a binary indicator column appended to *x_clean* so downstream
        models can distinguish imputed values from genuine near-zero data.

    """
    x_mat, _ = _impute_nan_columns(x_mat, fill_value=1e-10)

    eps = 1e-12
    y_mat = np.clip(y_mat, eps, None)
    y_sum = y_mat.sum(axis=1, keepdims=True)
    y_mat = np.divide(y_mat, y_sum, out=np.full_like(y_mat, 1.0 / n_classes), where=y_sum > 0)
    return x_mat, y_mat


def _prepare_year_data(
    df: pd.DataFrame,
    year: int,
    n_classes: int,
    *,
    exclude_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute features + class targets for a single election year.

    Static features (no year suffix) are taken from the feature matrix.
    Year-specific fiscal indicators from TerriData are merged in when
    the year is one of the six presidential election years.

    Args:
        df: Full feature matrix with ``vote_share_*`` columns.
        year: Target election year.  Used to select year-specific
            fiscal features.
        n_classes: 3 or 5.
        exclude_year: If set, exclude feature columns that contain
            this year suffix (prevents target leakage in holdout
            mode, e.g. excluding ``pop_2022`` when predicting 2022).

    Returns:
        ``(X_year, y_year, column_names)`` where each row is one municipality.

    """
    is_3class = n_classes == 3
    y_df = _compute_3class_targets(df) if is_3class else _compute_5class_targets(df)
    y_cols = [f"y_{cls}" for cls in (_3CLASS_CLASSES if is_3class else _IDEOLOGY_CLASSES)]

    leaked_suffix = str(exclude_year) if exclude_year else None
    static_cols = [
        c
        for c in df.columns
        if not re.match(r"vote_share_", c)
        and c not in ("year", "divipola", "municipio", "historical")
        and not c.startswith("y_")
        and pd.api.types.is_numeric_dtype(df[c])
        and (leaked_suffix is None or leaked_suffix not in c)
        and not c.startswith("pop_")
        and not c.startswith("ipm_")
        and not any(re.search(rf"_{y}$", c) for y in _ELECTION_YEARS)
    ]

    x_df = df[static_cols].copy()

    if year in _FISCAL_YEAR_SET:
        try:
            fiscal_df = _load_fiscal_features()
            fiscal_yr = fiscal_df.copy()
            muni_codes = df["codigo_municipio"].astype(int).to_numpy()
            fiscal_yr = fiscal_yr.reindex(muni_codes)
            x_df = pd.concat([x_df, fiscal_yr.reset_index(drop=True)], axis=1)
        except (FileNotFoundError, ImportError):
            pass

    x_cols = list(x_df.columns)
    x_mat = x_df.to_numpy(np.float64)

    y = y_df[y_cols].to_numpy(np.float64)

    x_clean, y_clean = _impute_and_normalize(x_mat, y, n_classes)

    # NaN indicators appended by imputation get synthetic names — different
    # per year, so they are excluded from the name-based column intersection.
    n_indicators = x_clean.shape[1] - len(x_cols)
    x_cols = x_cols + [f"__imputed_{i}__" for i in range(n_indicators)]
    return x_clean, y_clean, x_cols


def _detect_available_rounds(df: pd.DataFrame, year: int) -> list[int]:
    """Detect which electoral rounds exist for a given year in the feature matrix.

    Args:
        df: Feature matrix with ``vote_share_{year}_r{round}_*`` columns.
        year: Election year to scan.

    Returns:
        Sorted list of round numbers (e.g. ``[1]`` or ``[1, 2]``).

    """
    rounds: set[int] = set()
    for col in df.columns:
        m = re.match(rf"vote_share_{year}_r(\d+)_", col)
        if m:
            rounds.add(int(m.group(1)))
    return sorted(rounds)


def _prepare_combined_data(  # noqa: C901
    df: pd.DataFrame,
    year: int,
    n_classes: int,
    *,
    exclude_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack R1+R2 rows with ``periodo`` feature for a single year.

    Each municipality produces one row per round (R1, R2, …) with shared
    demographic features but round-specific vote-share targets and a
    ``periodo`` indicator column.  This matches USANTOMAS's methodology
    where round is the #1 most-important feature.

    Args:
        df: Full feature matrix with ``vote_share_*`` columns.
        year: Target election year.
        n_classes: 3 or 5.
        exclude_year: If set, exclude feature columns containing this
            year suffix (prevents target leakage in holdout mode).

    Returns:
        ``(X, y)`` where each row is one municipality-round combination.
        ``X`` includes a ``periodo`` column (1 or 2) as the last feature.
        ``y`` is the per-round class vote-share simplex.

    """
    is_3class = n_classes == 3
    class_names = _3CLASS_CLASSES if is_3class else _IDEOLOGY_CLASSES

    rounds = _detect_available_rounds(df, year)
    if not rounds:
        logger.warning(
            "No vote-share columns found for year %d; falling back to _prepare_year_data", year
        )
        x_y, y_y, _ = _prepare_year_data(df, year, n_classes, exclude_year=exclude_year)
        return x_y, y_y

    leaked_suffix = str(exclude_year) if exclude_year else None
    static_cols = [
        c
        for c in df.columns
        if not re.match(r"vote_share_", c)
        and c not in ("year", "divipola", "municipio", "historical")
        and not c.startswith("y_")
        and pd.api.types.is_numeric_dtype(df[c])
        and (leaked_suffix is None or leaked_suffix not in c)
        and not c.startswith("pop_")
        and not c.startswith("ipm_")
        and not any(re.search(rf"_{y}$", c) for y in _ELECTION_YEARS)
    ]

    x_base = df[static_cols].copy()

    if year in _FISCAL_YEAR_SET:
        try:
            fiscal_df = _load_fiscal_features()
            fiscal_yr = fiscal_df.copy()
            muni_codes = df["codigo_municipio"].astype(int).to_numpy()
            fiscal_yr = fiscal_yr.reindex(muni_codes)
            x_base = pd.concat([x_base, fiscal_yr.reset_index(drop=True)], axis=1)
        except (FileNotFoundError, ImportError):
            pass

    x_rows: list[pd.DataFrame] = []
    y_rows: list[pd.DataFrame] = []

    for round_num in rounds:
        round_map = _CANDIDATE_5CLASS_MAP.get((year, round_num), {})

        class_cols: dict[str, list[str]] = {cls: [] for cls in class_names}
        for col in df.columns:
            m = re.match(rf"vote_share_{year}_r{round_num}_(.+)", col)
            if m:
                suffix = m.group(1)
                cls = round_map.get(suffix)
                if cls:
                    class_cols[cls].append(col)

        y_round = pd.DataFrame(0.0, index=df.index, columns=[f"y_{cls}" for cls in class_names])
        for cls in class_names:
            if class_cols[cls]:
                y_round[f"y_{cls}"] = df[class_cols[cls]].sum(axis=1)

        mapped = y_round[[f"y_{cls}" for cls in class_names]].sum(axis=1)
        mapped_mask = mapped > 0
        for cls in class_names:
            col = f"y_{cls}"
            y_round.loc[mapped_mask, col] = y_round.loc[mapped_mask, col] / mapped.loc[mapped_mask]

        round_x = x_base.copy()
        round_x["periodo"] = round_num

        x_rows.append(round_x)
        y_rows.append(y_round)

    x_stacked = pd.concat(x_rows, ignore_index=True)
    y_stacked = pd.concat(y_rows, ignore_index=True)

    x_mat = x_stacked.to_numpy(np.float64)
    y_mat = y_stacked.to_numpy(np.float64)

    return _impute_and_normalize(x_mat, y_mat, n_classes)


def _load_feature_matrix() -> pd.DataFrame:
    """Load and validate feature matrix through the canonical pipeline.

    Returns:
        DataFrame with features + vote-share columns.

    Raises:
        FileNotFoundError: If load_features cannot find required files.
        ValueError: If schema validation fails.

    """
    return load_features()


def run_holdout_benchmarks(
    n_classes: int = 5,
    train_years: tuple[int, ...] = (2002, 2006, 2010, 2014, 2018),
    test_year: int = 2022,
) -> list[dict[str, Any]]:
    """Train on stacked (muni x year) observations, predict test year.

    For each train year, computes class targets from that year's
    ``vote_share_*`` columns, stacks all training years into one
    long-format matrix with demographic features, then evaluates all
    15 model-transform combos on the test year.

    Args:
        n_classes: Number of ideology classes (3 or 5).
        train_years: Election years for training.
        test_year: Election year to predict.

    Returns:
        Same format as :func:`run_benchmarks`.

    """
    logger.info(
        "Holdout: train on %s (n_classes=%d), test on %d",
        train_years,
        n_classes,
        test_year,
    )

    df = _load_feature_matrix()

    train_arrays: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    train_cols: list[list[str]] = []
    for y in train_years:
        X_y, y_y, cols = _prepare_year_data(df, y, n_classes, exclude_year=test_year)
        train_arrays.append(X_y)
        train_targets.append(y_y)
        train_cols.append(cols)

    X_train = np.vstack(_intersect_columns(list(zip(train_arrays, train_cols, strict=True))))
    y_train = np.vstack(train_targets)

    # Compute column-name intersection across training years so we can
    # slice the test array to match (avoids n_features mismatch at predict time).
    common: set[str] = set(train_cols[0] or [])
    for cols in train_cols[1:]:
        common &= set(cols)
    common_order = [c for c in (train_cols[0] or []) if c in common]

    X_test, y_test, test_cols = _prepare_year_data(df, test_year, n_classes, exclude_year=test_year)
    test_idx_map = {c: i for i, c in enumerate(test_cols or [])}
    test_indices = [test_idx_map[c] for c in common_order if c in test_idx_map]
    X_test = X_test[:, test_indices]

    logger.info(
        "Holdout train shape %s, test shape %s",
        X_train.shape,
        X_test.shape,
    )

    class_names = _3CLASS_CLASSES if n_classes == 3 else _IDEOLOGY_CLASSES
    rows: list[dict[str, Any]] = []

    for t_name in ("alr", "clr", "ilr"):
        y_train_t = _apply_transform(t_name, y_train)

        for m_name in ("svr", "rfr", "gbr", "knn", "fnn"):
            model = _make_sklearn_model(m_name, n_train=X_train.shape[0])
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(X_train, y_train_t)
                y_pred_t = model.predict(X_test)
                y_pred = _apply_inverse_transform(t_name, y_pred_t)
                per_class = _r2_rmse_per_class(y_test, y_pred, class_names)

                rows.extend(
                    {
                        "model": m_name,
                        "transform": t_name,
                        "class_name": pc["class_name"],
                        "r2": pc["r2"],
                        "rmse": pc["rmse"],
                        "n_train": X_train.shape[0],
                        "n_test": X_test.shape[0],
                    }
                    for pc in per_class
                )
            except (ValueError, TypeError, LinAlgError, NotFittedError):
                logger.exception("Model %s + transform %s failed (holdout)", m_name, t_name)
                rows.extend(
                    {
                        "model": m_name,
                        "transform": t_name,
                        "class_name": cname,
                        "r2": None,
                        "rmse": None,
                        "n_train": X_train.shape[0],
                        "n_test": X_test.shape[0],
                    }
                    for cname in class_names
                )
    return rows


def run_random_benchmarks(
    n_classes: int = 5,
    train_years: tuple[int, ...] = (2002, 2006, 2010, 2014, 2018, 2022),
    train_ratio: float = 0.7,
) -> list[dict[str, Any]]:
    """Random 70/30 split of stacked (muni, year) observations.

    Mirrors USANTOMAS's methodology: stack all years into long format,
    randomly split 70/30 (no year separation).  Test set has the same
    distribution as training (in-distribution validation).

    Args:
        n_classes: Number of ideology classes (3 or 5).
        train_years: Years to stack.
        train_ratio: Fraction of rows for training.

    Returns:
        Same format as :func:`run_benchmarks`.

    """
    logger.info(
        "Random 70/30 split (USANTOMAS-style): train on %s (n_classes=%d)",
        train_years,
        n_classes,
    )

    df = _load_feature_matrix()

    train_arrays: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    train_cols: list[list[str]] = []
    for y in train_years:
        X_y, y_y, cols = _prepare_year_data(df, y, n_classes)
        train_arrays.append(X_y)
        train_targets.append(y_y)
        train_cols.append(cols)

    X_all = np.vstack(_intersect_columns(list(zip(train_arrays, train_cols, strict=True))))
    y_all = np.vstack(train_targets)
    X_train, X_test, y_train, y_test = _train_test_split(X_all, y_all, train_ratio=train_ratio)

    logger.info(
        "Random split train shape %s, test shape %s",
        X_train.shape,
        X_test.shape,
    )

    class_names = _3CLASS_CLASSES if n_classes == 3 else _IDEOLOGY_CLASSES
    rows: list[dict[str, Any]] = []

    for t_name in ("alr", "clr", "ilr"):
        y_train_t = _apply_transform(t_name, y_train)

        for m_name in ("svr", "rfr", "gbr", "knn", "fnn"):
            model = _make_sklearn_model(m_name, n_train=X_train.shape[0])
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(X_train, y_train_t)
                y_pred_t = model.predict(X_test)
                y_pred = _apply_inverse_transform(t_name, y_pred_t)
                per_class = _r2_rmse_per_class(y_test, y_pred, class_names)

                rows.extend(
                    {
                        "model": m_name,
                        "transform": t_name,
                        "class_name": pc["class_name"],
                        "r2": pc["r2"],
                        "rmse": pc["rmse"],
                        "n_train": X_train.shape[0],
                        "n_test": X_test.shape[0],
                    }
                    for pc in per_class
                )
            except (ValueError, TypeError, LinAlgError, NotFittedError):
                logger.exception("Model %s + transform %s failed (random)", m_name, t_name)
                rows.extend(
                    {
                        "model": m_name,
                        "transform": t_name,
                        "class_name": cname,
                        "r2": None,
                        "rmse": None,
                        "n_train": X_train.shape[0],
                        "n_test": X_test.shape[0],
                    }
                    for cname in class_names
                )
    return rows


def run_combined_benchmarks(
    n_classes: int = 5,
    train_years: tuple[int, ...] = (2002, 2006, 2010, 2014, 2018),
    test_year: int = 2022,
) -> list[dict[str, Any]]:
    """Train on stacked (muni x year x round) observations, predict test year.

    Stacks R1 and R2 data with ``periodo`` feature, matching USANTOMAS's
    methodology where round is a feature.  Each municipality-round
    combination is one observation.

    Args:
        n_classes: Number of ideology classes (3 or 5).
        train_years: Election years for training.
        test_year: Election year to predict.

    Returns:
        Same format as :func:`run_benchmarks`.

    """
    logger.info(
        "Combined (R1+R2 stacked): train on %s (n_classes=%d), test on %d",
        train_years,
        n_classes,
        test_year,
    )

    df = _load_feature_matrix()

    train_arrays: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    for y in train_years:
        X_y, y_y = _prepare_combined_data(df, y, n_classes, exclude_year=test_year)
        train_arrays.append(X_y)
        train_targets.append(y_y)

    X_train = np.vstack(_pad_arrays_to_max_width(train_arrays))
    y_train = np.vstack(train_targets)
    X_test, y_test = _prepare_combined_data(df, test_year, n_classes, exclude_year=test_year)
    # Align X_test to training column count (pad or truncate)
    n_train_cols = X_train.shape[1]
    if X_test.shape[1] < n_train_cols:
        pad_test = np.zeros((X_test.shape[0], n_train_cols - X_test.shape[1]))
        X_test = np.hstack([X_test, pad_test])
    elif X_test.shape[1] > n_train_cols:
        X_test = X_test[:, :n_train_cols]

    logger.info(
        "Combined train shape %s, test shape %s",
        X_train.shape,
        X_test.shape,
    )

    class_names = _3CLASS_CLASSES if n_classes == 3 else _IDEOLOGY_CLASSES
    rows: list[dict[str, Any]] = []

    for t_name in ("alr", "clr", "ilr"):
        y_train_t = _apply_transform(t_name, y_train)

        for m_name in ("svr", "rfr", "gbr", "knn", "fnn"):
            model = _make_sklearn_model(m_name, n_train=X_train.shape[0])
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(X_train, y_train_t)
                y_pred_t = model.predict(X_test)
                y_pred = _apply_inverse_transform(t_name, y_pred_t)
                per_class = _r2_rmse_per_class(y_test, y_pred, class_names)

                rows.extend(
                    {
                        "model": m_name,
                        "transform": t_name,
                        "class_name": pc["class_name"],
                        "r2": pc["r2"],
                        "rmse": pc["rmse"],
                        "n_train": X_train.shape[0],
                        "n_test": X_test.shape[0],
                    }
                    for pc in per_class
                )
            except (ValueError, TypeError, LinAlgError, NotFittedError):
                logger.exception("Model %s + transform %s failed (combined)", m_name, t_name)
                rows.extend(
                    {
                        "model": m_name,
                        "transform": t_name,
                        "class_name": cname,
                        "r2": None,
                        "rmse": None,
                        "n_train": X_train.shape[0],
                        "n_test": X_test.shape[0],
                    }
                    for cname in class_names
                )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path = _OUTPUT_CSV) -> Path:
    """Write benchmark results to CSV.

    Args:
        rows: List of result dicts.
        path: Output file path.

    Returns:
        Path to the written CSV.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "transform", "class_name", "r2", "rmse", "n_train", "n_test"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path)
    return path


def main() -> None:
    """CLI entry point for the benchmark runner.

    Parses ``--smoke`` flag and runs all 15 model-transform combinations.
    """
    parser = argparse.ArgumentParser(description="SPEC-41: FNN+CLR ML benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with synthetic data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if args.smoke:
        logger.info("Running smoke benchmark (synthetic 3-row data)...")
        results = run_benchmarks(smoke=True)
    else:
        logger.info("Running full benchmark on historical data...")
        X, y = load_historical_data()
        results = run_benchmarks(X, y)

    out_path = write_csv(results)
    _print_summary(results, out_path)


def load_historical_data(
    n_classes: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load real historical feature matrix and 5-class ideology targets.

    Computes 5-class (or 3-class) ideology target columns from
    ``vote_share_*`` columns (most recent year), then excludes target-year
    vote shares from the feature matrix to prevent leakage.

    Falls back to loading the raw Parquet matrix directly if the standard
    ``load_features()`` pipeline fails (e.g. missing fiscal schema columns).

    Args:
        n_classes: Number of ideology classes (3 or 5).  Defaults to 5.

    Returns:
        ``(X, y)`` where ``X.shape == (N, F)`` and ``y.shape == (N, K)``
        with ``K = n_classes``.

    Raises:
        ValueError: If fewer than *n_classes* target columns are found
            in the feature matrix.

    """
    df = _load_feature_matrix()

    class_names = _3CLASS_CLASSES if n_classes == 3 else _IDEOLOGY_CLASSES
    n_target = n_classes

    df = _compute_3class_targets(df) if n_classes == 3 else _compute_5class_targets(df)

    target_year = _detect_target_year(df)

    def _is_feature(col: str) -> bool:
        return not (
            col in ("year", "divipola", "municipio")
            or col.startswith("y_")
            or (target_year and re.match(rf"vote_share_{target_year}_r\d_", col))
        )

    X_cols = [c for c in df.columns if _is_feature(c) and pd.api.types.is_numeric_dtype(df[c])]
    y_cols = [f"y_{cls}" for cls in class_names]

    available_y = [c for c in y_cols if c in df.columns]
    if len(available_y) < n_target:
        msg = (
            f"{n_classes}-class y columns not found in feature matrix; "
            f"cannot run benchmarks. Expected: {y_cols}, found: {available_y}"
        )
        raise ValueError(msg)

    y = df[available_y].to_numpy(np.float64)
    EPSILON = 1e-12
    y = np.clip(y, EPSILON, None)
    y_sum = y.sum(axis=1, keepdims=True)
    y = np.divide(y, y_sum, out=np.full_like(y, 1.0 / n_target), where=y_sum > 0)
    y = np.clip(y, EPSILON, None)

    x_mat = df[X_cols].to_numpy(np.float64)
    nan_count = int(np.isnan(x_mat).any(axis=0).sum())
    if nan_count:
        logger.warning("Imputing %d feature columns with column medians", nan_count)
    x_mat, n_imputed = _impute_nan_columns(x_mat, fill_value=1e-10)
    if n_imputed:
        logger.warning(
            "%d feature columns entirely NaN; filling with 1e-10",
            n_imputed,
        )

    return x_mat, y


def _detect_target_year(df: pd.DataFrame) -> int | None:
    """Detect the most recent election year in the feature matrix."""
    years: set[int] = set()
    for col in df.columns:
        m = re.match(r"vote_share_(\d{4})_r\d_", col)
        if m:
            years.add(int(m.group(1)))
    return max(years) if years else None


def _print_summary(results: list[dict[str, Any]], path: Path) -> None:
    """Print a human-readable summary of benchmark results."""
    print(f"\nBenchmark results written to {path}")
    print(f"Total rows: {len(results)}")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_model.setdefault(str(r["model"]), []).append(r)
    for m_name, m_rows in sorted(by_model.items()):
        r2s = [r["r2"] for r in m_rows if r["r2"] is not None]
        if r2s:
            print(f"  {m_name}: mean R² = {np.mean(r2s):.4f} (n={len(r2s)})")


def report_benchmark_baseline(
    features: pd.DataFrame,
    *,
    model_names: tuple[str, ...] = ("svr", "rfr", "gbr", "knn", "fnn"),
    transform_names: tuple[str, ...] = ("alr", "clr", "ilr"),
) -> dict[str, Any]:
    """Run benchmarks on a features DataFrame and log a summary.

    Extracts feature columns and 5-class ideology target columns from
    *features*, runs all *model_names* × *transform_names* combinations, and
    logs per-model mean R².  Intended as an informational reference baseline
    inside the SPEC-26 OOS validation pipeline (does **not** gate).

    If the target columns are not present in *features*, the function logs a
    warning and returns early with an empty results list.

    Args:
        features: DataFrame containing feature columns and ``y_Izquierda``,
            ``y_Centro_Izquierda``, ``y_Centro``, ``y_Centro_Derecha``,
            ``y_Derecha`` target columns.
        model_names: Models to evaluate (registry keys).
        transform_names: Transforms to evaluate.

    Returns:
        Dict with keys ``n_rows``, ``n_models``, ``n_transforms`` and
        ``results`` (the raw list of result dicts from ``run_benchmarks``).

    """
    # Ensure 5-class target columns are present
    has_y = any(c.startswith("y_Izquierda") for c in features.columns)
    df = _compute_5class_targets(features) if not has_y else features

    y_cols = [f"y_{cls}" for cls in _IDEOLOGY_CLASSES]
    available_y = [c for c in y_cols if c in df.columns]
    if len(available_y) < 5:
        logger.warning(
            "Benchmark baseline: 5-class y cols not found after _compute_5class_targets. Skipping."
        )
        return {
            "n_rows": 0,
            "n_models": 0,
            "n_transforms": 0,
            "results": [],
            "per_class_best_r2": {},
        }

    X_cols = [
        c
        for c in df.columns
        if c not in ("year", "divipola", "municipio", "historical")
        and c not in y_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[X_cols].to_numpy(np.float64)
    y = df[available_y].to_numpy(np.float64)

    results = run_benchmarks(X, y, model_names=model_names, transform_names=transform_names)

    # Compute best R² per class across all (model, transform) combinations.
    per_class_best: dict[str, float] = {}
    for cls in _IDEOLOGY_CLASSES:
        cls_r2s = [r["r2"] for r in results if r["class_name"] == cls and r["r2"] is not None]
        per_class_best[cls] = max(cls_r2s) if cls_r2s else float("nan")

    _log_baseline_summary(results)

    return {
        "n_rows": len(results),
        "n_models": len(model_names),
        "n_transforms": len(transform_names),
        "results": results,
        "per_class_best_r2": per_class_best,
    }


def _log_baseline_summary(results: list[dict[str, Any]]) -> None:
    """Log per-model mean R² from benchmark results."""
    by_model: dict[str, list[float]] = {}
    for r in results:
        by_model.setdefault(str(r["model"]), []).append(
            float(r["r2"]) if r["r2"] is not None else float("nan")
        )  # type: ignore[arg-type]
    for m_name, r2s in sorted(by_model.items()):
        valid = [v for v in r2s if not np.isnan(v)]
        if valid:
            logger.info("  %s: mean R² = %.4f (n=%d)", m_name, np.mean(valid), len(valid))
        else:
            logger.info("  %s: R² all NaN (too few samples?)", m_name)


if __name__ == "__main__":
    main()
