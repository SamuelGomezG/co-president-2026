"""Derive transfer-heuristic constants from poll deltas.

Computes round-1 → round-2 share deltas for pollsters that appear in both
    rounds, then estimates the aggregate split of eliminated-candidate votes
    flowing to Petro vs Hernandez. This is used to justify the runoff transfer
heuristic constants in ``co_president.config``.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from co_president.data_polls import load_and_clean_all
from co_president.paths import resolve_data_dir


class PollsterDelta(NamedTuple):
    """Delta summary for a single pollster across rounds."""

    pollster: str
    petro_delta: float
    hernandez_delta: float
    eliminated_delta: float


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    """Compute weighted mean of ``values`` by integer ``weights``."""
    if not values:
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    """Ensure required columns exist in ``df``."""
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing required columns: {", ".join(sorted(missing))}"
        raise ValueError(msg)


def compute_transfer_split(data_dir: str | None = None) -> tuple[float, float, list[PollsterDelta]]:
    """Compute aggregate Petro/Hernandez transfer split from poll deltas.

    Args:
        data_dir: Optional path to the project data directory.

    Returns:
        Tuple of ``(petro_share, hernandez_share, pollster_deltas)`` where
        shares sum to 1.0 and ``pollster_deltas`` contains per-pollster deltas
        for inspection.
    """
    resolved = resolve_data_dir(data_dir)
    polls = load_and_clean_all(resolved)

    round1 = polls.round1
    round2 = polls.round2

    _validate_columns(
        round1,
        {
            "encuestadora",
            "gustavo_petro",
            "rodolfo_hernandez",
            "federico_gutierrez",
            "sergio_fajardo",
            "ingrid_betancourt",
            "rest",
            "blanco",
        },
    )
    _validate_columns(round2, {"encuestadora", "gustavo_petro", "rodolfo_hernandez", "rest"})

    round1_grouped = (
        round1.groupby("encuestadora")
        .agg(
            petro=("gustavo_petro", "mean"),
            hernandez=("rodolfo_hernandez", "mean"),
            gutierrez=("federico_gutierrez", "mean"),
            fajardo=("sergio_fajardo", "mean"),
            betancourt=("ingrid_betancourt", "mean"),
            rest=("rest", "mean"),
            blanco=("blanco", "mean"),
            count=("encuestadora", "size"),
        )
        .reset_index()
    )

    round2_grouped = (
        round2.groupby("encuestadora")
        .agg(
            petro=("gustavo_petro", "mean"),
            hernandez=("rodolfo_hernandez", "mean"),
            rest=("rest", "mean"),
            count=("encuestadora", "size"),
        )
        .reset_index()
    )

    for col in ("petro", "hernandez", "gutierrez", "fajardo", "betancourt", "rest", "blanco"):
        round1_grouped[col] = round1_grouped[col] / 100.0
    for col in ("petro", "hernandez", "rest"):
        round2_grouped[col] = round2_grouped[col] / 100.0

    merged = round1_grouped.merge(round2_grouped, on="encuestadora", suffixes=("_r1", "_r2"))

    deltas: list[PollsterDelta] = []
    weights: list[int] = []

    for _, row in merged.iterrows():
        eliminated = (
            row["gutierrez"]
            + row["fajardo"]
            + row["betancourt"]
            + row["rest_r1"]
            + row["blanco"]
        )
        petro_delta = row["petro_r2"] - row["petro_r1"]
        hernandez_delta = row["hernandez_r2"] - row["hernandez_r1"]
        deltas.append(
            PollsterDelta(
                pollster=row["encuestadora"],
                petro_delta=float(petro_delta),
                hernandez_delta=float(hernandez_delta),
                eliminated_delta=float(eliminated),
            )
        )
        weight = int(row["count_r1"] + row["count_r2"])
        weights.append(weight)

    petro_mean = _weighted_mean([d.petro_delta for d in deltas], weights)
    hernandez_mean = _weighted_mean([d.hernandez_delta for d in deltas], weights)

    total = petro_mean + hernandez_mean
    if total <= 0:
        return 0.0, 0.0, deltas
    return petro_mean / total, hernandez_mean / total, deltas


def main() -> None:
    """Entry point for CLI usage."""
    petro_share, hernandez_share, deltas = compute_transfer_split()

    print("Pollster deltas (round-1 -> round-2):")
    for delta in deltas:
        print(
            f"- {delta.pollster}: petro {delta.petro_delta:.2f}, "
            f"hernandez {delta.hernandez_delta:.2f}, "
            f"eliminated {delta.eliminated_delta:.2f}"
        )

    print("\nAggregate split:")
    print(f"Petro share: {petro_share:.2%}")
    print(f"Hernandez share: {hernandez_share:.2%}")


if __name__ == "__main__":
    main()
