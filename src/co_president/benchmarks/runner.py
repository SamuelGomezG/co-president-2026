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

import numpy as np
import pandas as pd  # type: ignore[reportMissingTypeStubs]
from sklearn.metrics import (  # type: ignore[reportMissingTypeStubs]
    r2_score,  # type: ignore[reportUnknownVariableType]
    root_mean_squared_error,  # type: ignore[reportUnknownVariableType]
)
from sklearn.multioutput import (  # type: ignore[reportMissingTypeStubs]
    MultiOutputRegressor,  # type: ignore[reportUnknownVariableType]
)
from sklearn.pipeline import Pipeline  # type: ignore[reportMissingTypeStubs]
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingTypeStubs]

from co_president.benchmarks.transforms import (
    alr_inv_transform,
    alr_transform,
    clr_inv_transform,
    clr_transform,
    ilr_inv_transform,
    ilr_transform,
)
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

_IDEOLOGY_CLASSES: list[str] = [
    "Izquierda",
    "Centro_Izquierda",
    "Centro",
    "Centro_Derecha",
    "Derecha",
]

# Raw-column-suffix → ideology class mapping for each (year, round).
# Keys match the feature matrix column name after ``vote_share_{year}_r{round}_``.
# Expert-derived based on Colombian political tradition.
_CANDIDATE_5CLASS_MAP: dict[tuple[int, int], dict[str, str]] = {
    (2002, 1): {
        "alvaro_uribe": "Derecha",
        "horacio_serpa": "Centro",
        "luis_eduardo_garzon": "Izquierda",
        "ingrid_betancourt": "Centro",
        "noemi_sanin": "Centro_Derecha",
    },
    (2006, 1): {
        "alvaro_uribe": "Derecha",
        "carlos_gaviria": "Izquierda",
        "horacio_serpa": "Centro",
        "antanas_mockus": "Centro",
    },
    (2010, 1): {
        "juan_manuel_santos": "Centro_Derecha",
        "antanas_mockus": "Centro",
        "gustavo_petro": "Izquierda",
        "noemi_sanin": "Centro_Derecha",
    },
    (2014, 1): {
        "juan_manuel_santos": "Centro_Derecha",
        "oscar_ivan_zuluaga": "Derecha",
        "enrique_penalosa": "Centro",
        "LOPEZ": "Izquierda",
    },
    (2018, 1): {
        "ivan_duque": "Derecha",
        "gustavo_petro": "Izquierda",
        "sergio_fajardo": "Centro",
        "DE LA CALLE": "Centro_Izquierda",
    },
    (2022, 1): {
        "GUSTAVO PETRO": "Izquierda",
        "RODOLFO HERNÁNDEZ": "Derecha",
        "FEDERICO GUTIÉRREZ": "Centro_Derecha",
        "SERGIO FAJARDO": "Centro",
        "INGRID BETANCOURT": "Centro",
        "JOHN MILTON RODRÍGUEZ": "Derecha",
        "LUIS PÉREZ": "Centro_Derecha",
    },
}


def _compute_5class_targets(df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901  type: ignore[name-defined]
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

    Returns:
        DataFrame with added ``y_*`` columns (or synthetic fallback

    """
    result = df.copy()

    all_years: set[int] = set()
    for col in df.columns:
        m = re.match(r"vote_share_(\d{4})_r(\d)_(.+)", col)
        if m:
            all_years.add(int(m.group(1)))

    if not all_years:
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
    target_map = _CANDIDATE_5CLASS_MAP.get((target_year, 1), {})

    class_cols: dict[str, list[str]] = {cls: [] for cls in _IDEOLOGY_CLASSES}
    for col in df.columns:
        m = re.match(rf"vote_share_{target_year}_r1_(.+)", col)
        if m:
            suffix = m.group(1)
            cls = target_map.get(suffix)
            if cls:
                class_cols[cls].append(col)

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
        r2 = float(
            np.nan_to_num(r2_score(y_true[:, k], y_pred[:, k], force_finite=True), nan=0.0)  # type: ignore[reportUnknownArgumentType]
        )
        rmse = float(root_mean_squared_error(y_true[:, k], y_pred[:, k]))  # type: ignore[reportUnknownArgumentType]
        results.append({"class_name": name, "r2": r2, "rmse": rmse})
    return results


def _generate_smoke_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 3-municipio × 5-feature × 5-class fixture.

    Both X and y are compositional (non-negative, rows sum to 1), so ALR/CLR/
    ILR transforms can be applied to either.

    Returns:
        ``(X, y)`` where ``X.shape == (3, 5)`` and ``y.shape == (3, 5)``.

    """
    rng = np.random.default_rng(42)
    raw_X = rng.dirichlet(np.ones(5), size=3)
    raw_y = rng.dirichlet(np.ones(5), size=3)
    return raw_X.astype(np.float64), raw_y.astype(np.float64)


def run_benchmarks(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    transform_names: tuple[str, ...] = ("alr", "clr", "ilr"),
    model_names: tuple[str, ...] = ("svr", "rfr", "gbr", "knn", "fnn"),
    class_names: list[str] | None = None,
    smoke: bool = False,
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

    Returns:
        List of result dicts, each with keys ``model``, ``transform``,
        ``class_name``, ``r2``, ``rmse``, ``n_train``, ``n_test``.

    Returns:
        List of result dictionaries ready for CSV serialisation.

    """
    if smoke:
        X_in, y_in = _generate_smoke_data()
    elif X is not None and y is not None:
        X_in, y_in = X, y
    else:
        msg = "Provide X/y or pass smoke=True"
        raise ValueError(msg)

    if class_names is None:
        class_names = _IDEOLOGY_CLASSES

    rows: list[dict[str, Any]] = []
    X_train, X_test, y_train_full, y_test_full = _train_test_split(X_in, y_in)
    train_n = X_train.shape[0]
    test_n = X_test.shape[0]

    for t_name in transform_names:
        y_train_t = _apply_transform(t_name, y_train_full)

        for m_name in model_names:
            model = _make_sklearn_model(m_name, n_train=train_n)
            try:
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
            except Exception:
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


def load_historical_data() -> tuple[np.ndarray, np.ndarray]:
    """Load real historical feature matrix and 5-class ideology targets.

    Computes 5-class ideology target columns from ``vote_share_*`` columns
    (most recent year), then excludes target-year vote shares from the
    feature matrix to prevent leakage.

    Falls back to loading the raw Parquet matrix directly if the standard
    ``load_features()`` pipeline fails (e.g. missing fiscal schema columns).

    Returns:
        ``(X, y)`` where ``X.shape == (N, F)`` and ``y.shape == (N, 5)``.

    """
    try:
        df = load_features()  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
    except ValueError:
        logger.warning("load_features() failed; loading raw Parquet matrix")
        parquet_path = Path("data/processed/municipal_feature_matrix.parquet")
        if not parquet_path.exists():
            parquet_path = (
                Path(__file__).parents[3] / "data/processed/municipal_feature_matrix.parquet"
            )
        df = pd.read_parquet(str(parquet_path))
        logger.info("Loaded raw Parquet matrix: %d rows x %d columns", *df.shape)

    df = _compute_5class_targets(df)

    target_year = _detect_target_year(df)

    def _is_feature(col: str) -> bool:
        return not (
            col in ("year", "divipola", "municipio")
            or col.startswith("y_")
            or (target_year and re.match(rf"vote_share_{target_year}_r\d_", col))
        )

    X_cols = [c for c in df.columns if _is_feature(c) and pd.api.types.is_numeric_dtype(df[c])]
    y_cols = [f"y_{cls}" for cls in _IDEOLOGY_CLASSES]

    available_y = [c for c in y_cols if c in df.columns]
    if len(available_y) < 5:
        logger.warning("5-class y columns not found; using synthetic targets")
        rng = np.random.default_rng(42)
        n = df.shape[0]
        y_synth = np.abs(rng.standard_normal((n, 5)))
        y_synth = y_synth / y_synth.sum(axis=1, keepdims=True)
        return df[X_cols].to_numpy(np.float64), y_synth

    y = df[available_y].to_numpy(np.float64)
    EPSILON = 1e-12
    y = np.clip(y, EPSILON, None)
    y_sum = y.sum(axis=1, keepdims=True)
    y = np.divide(y, y_sum, out=np.full_like(y, 1.0 / 5), where=y_sum > 0)
    y = np.clip(y, EPSILON, None)

    x_mat = df[X_cols].to_numpy(np.float64)
    nan_mask = np.isnan(x_mat)
    if nan_mask.any():
        logger.warning(
            "Imputing %d feature columns with column medians", int(nan_mask.any(axis=0).sum())
        )
        col_median = np.nanmedian(x_mat, axis=0)
        x_mat = np.where(nan_mask, col_median, x_mat)

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
        c for c in df.columns if c not in ("year", "divipola", "municipio") and c not in y_cols
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
