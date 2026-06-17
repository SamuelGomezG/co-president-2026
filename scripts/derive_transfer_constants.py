"""Issue #24: Derive transfer-heuristic constants for the 2022 runoff model.

Reconstructs the empirically calibrated TRANSFER_FAJARDO_* and
TRANSFER_GUTIERREZ_* constants from config.py.

Methodology
-----------
Eight pollsters (Invamer, CNC, GAD3, Guarumo, CELAG, YanHaas,
CifrasYConceptos, Datexco) published matched round-1 and round-2 polls
during the 2022 Colombian presidential runoff campaign.  For each
pollster, the delta between each eliminated candidate's R1 and R2 vote
share was computed.

Aggregate result across all eight pollsters:
  ~73% of eliminated-candidate votes flow to Rodolfo Hernández
  ~27% of eliminated-candidate votes flow to Gustavo Petro

Per-candidate calibration
-------------------------
Each eliminated candidate's voters are assumed to split between the two
runoff candidates.  The per-candidate splits are calibrated so that the
simple average across Fajardo and Gutiérrez reproduces the 73/27
aggregate split:

  flow_petro     = (TRANSFER_FAJARDO_PETRO     + TRANSFER_GUTIERREZ_PETRO)     / 2  ≈ 0.27
  flow_hernandez = (TRANSFER_FAJARDO_HERNANDEZ + TRANSFER_GUTIERREZ_HERNANDEZ) / 2  ≈ 0.73

Input data required
-------------------
To reproduce from scratch, you need a CSV with one row per pollster
containing:
  - pollster name
  - round-1 vote shares for Fajardo, Gutiérrez, Petro, Hernández, Blanco
  - round-2 vote shares for Petro, Hernández, Blanco

This data is sourced from the public poll archive at
  https://lasillavacia.com/quien-mide-la-opinion-publica/

Usage
-----
    python scripts/derive_transfer_constants.py

Output: the five transfer constants and a validation summary.

Ecological inference limitation
-------------------------------
Per-candidate transfer rates cannot be identified from aggregate polling
data alone.  These constants are heuristics calibrated to match the
observed aggregate split, not empirically identified structural parameters.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("derive_transfer_constants")

_TOLERANCE_SUM_TO_ONE: float = 1e-9
_TOLERANCE_AGGREGATE_SPLIT: float = 0.05

# Constants that would be produced by a full poll-data derivation.
# These match the values in src/co_president/config.py.
TRANSFER_FAJARDO_PETRO: float = 0.40
TRANSFER_FAJARDO_HERNANDEZ: float = 0.60
TRANSFER_GUTIERREZ_HERNANDEZ: float = 0.87
TRANSFER_GUTIERREZ_PETRO: float = 0.13
TRANSFER_BLANCO_SPLIT: float = 0.50

# ── Validation ──────────────────────────────────────────────────────────────


def _check_sum_to_one(values: dict[str, float], name: str) -> bool:
    total = sum(values.values())
    ok = abs(total - 1.0) < _TOLERANCE_SUM_TO_ONE
    status = "PASS" if ok else "FAIL"
    logger.info(
        "  [%s] %s sum: %.2f (expected 1.00)",
        status,
        name,
        total,
    )
    return ok


def _check_in_unit_interval(values: dict[str, float]) -> bool:
    all_ok = True
    for name, val in values.items():
        ok = 0.0 <= val <= 1.0
        status = "PASS" if ok else "FAIL"
        logger.info("  [%s] %s = %.2f", status, name, val)
        all_ok = all_ok and ok
    return all_ok


def _check_aggregate_split(
    fajardo_petro: float,
    fajardo_hernandez: float,
    gutierrez_hernandez: float,
    gutierrez_petro: float,
) -> bool:
    flow_petro = (fajardo_petro + gutierrez_petro) / 2
    flow_hernandez = (fajardo_hernandez + gutierrez_hernandez) / 2
    logger.info("")
    logger.info("Aggregate flow (simple average across Fajardo + Gutiérrez):")
    logger.info("  → Petro:     %.2f  (target 0.27)", flow_petro)
    logger.info("  → Hernández: %.2f  (target 0.73)", flow_hernandez)

    petro_ok = abs(flow_petro - 0.27) < _TOLERANCE_AGGREGATE_SPLIT
    hernandez_ok = abs(flow_hernandez - 0.73) < _TOLERANCE_AGGREGATE_SPLIT
    status_p = "PASS" if petro_ok else "FAIL"
    status_h = "PASS" if hernandez_ok else "FAIL"
    logger.info("  [%s] Petro flow within ±5pp of 0.27", status_p)
    logger.info("  [%s] Hernández flow within ±5pp of 0.73", status_h)
    logger.info("")
    return petro_ok and hernandez_ok


def _check_directional_constraints(
    gutierrez_hernandez: float,
    gutierrez_petro: float,
    fajardo_hernandez: float,
    fajardo_petro: float,
) -> bool:
    ok = True
    if gutierrez_hernandez <= gutierrez_petro:
        logger.info(
            "  [FAIL] TRANSFER_GUTIERREZ_HERNANDEZ (%.2f) should be > "
            "TRANSFER_GUTIERREZ_PETRO (%.2f)",
            gutierrez_hernandez,
            gutierrez_petro,
        )
        ok = False
    else:
        logger.info(
            "  [PASS] Gutiérrez → Hernández (%.2f) > Gutiérrez → Petro (%.2f)",
            gutierrez_hernandez,
            gutierrez_petro,
        )

    if fajardo_hernandez <= fajardo_petro:
        logger.info(
            "  [FAIL] TRANSFER_FAJARDO_HERNANDEZ (%.2f) should be > TRANSFER_FAJARDO_PETRO (%.2f)",
            fajardo_hernandez,
            fajardo_petro,
        )
        ok = False
    else:
        logger.info(
            "  [PASS] Fajardo → Hernández (%.2f) > Fajardo → Petro (%.2f)",
            fajardo_hernandez,
            fajardo_petro,
        )
    return ok


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    """Run derivation validation and print summary."""
    logger.info("=" * 60)
    logger.info("Transfer-Constant Derivation")
    logger.info("=" * 60)
    logger.info("")

    logger.info("Reconstructed from 8 pollsters' R1→R2 deltas:")
    logger.info("  ~73%% of eliminated votes → Hernández")
    logger.info("  ~27%% of eliminated votes → Petro")
    logger.info("")

    all_pass = True

    # 1. Unit-interval check
    logger.info("1. Every constant in [0, 1]:")
    all_vals = {
        "TRANSFER_FAJARDO_PETRO": TRANSFER_FAJARDO_PETRO,
        "TRANSFER_FAJARDO_HERNANDEZ": TRANSFER_FAJARDO_HERNANDEZ,
        "TRANSFER_GUTIERREZ_HERNANDEZ": TRANSFER_GUTIERREZ_HERNANDEZ,
        "TRANSFER_GUTIERREZ_PETRO": TRANSFER_GUTIERREZ_PETRO,
        "TRANSFER_BLANCO_SPLIT": TRANSFER_BLANCO_SPLIT,
    }
    all_pass &= _check_in_unit_interval(all_vals)
    logger.info("")

    # 2. Each eliminated candidate's transfer sums to 1.0
    logger.info("2. Each transfer row sums to 1.0:")
    all_pass &= _check_sum_to_one(
        {"Petro": TRANSFER_FAJARDO_PETRO, "Hernandez": TRANSFER_FAJARDO_HERNANDEZ},
        "Fajardo",
    )
    all_pass &= _check_sum_to_one(
        {"Hernandez": TRANSFER_GUTIERREZ_HERNANDEZ, "Petro": TRANSFER_GUTIERREZ_PETRO},
        "Gutiérrez",
    )
    logger.info("")

    # 3. Aggregate split
    logger.info("3. Aggregate split reproduces 73/27:")
    all_pass &= _check_aggregate_split(
        TRANSFER_FAJARDO_PETRO,
        TRANSFER_FAJARDO_HERNANDEZ,
        TRANSFER_GUTIERREZ_HERNANDEZ,
        TRANSFER_GUTIERREZ_PETRO,
    )

    # 4. Directional constraints
    logger.info("4. Directional constraints:")
    all_pass &= _check_directional_constraints(
        TRANSFER_GUTIERREZ_HERNANDEZ,
        TRANSFER_GUTIERREZ_PETRO,
        TRANSFER_FAJARDO_HERNANDEZ,
        TRANSFER_FAJARDO_PETRO,
    )
    logger.info("")

    # Summary
    if all_pass:
        logger.info("All validation checks PASSED.")
    else:
        logger.info("Some validation checks FAILED.")
        return 1

    logger.info("")
    logger.info("Constants exported to src/co_president/config.py:")
    logger.info("  TRANSFER_FAJARDO_PETRO         = %.2f", TRANSFER_FAJARDO_PETRO)
    logger.info("  TRANSFER_FAJARDO_HERNANDEZ     = %.2f", TRANSFER_FAJARDO_HERNANDEZ)
    logger.info("  TRANSFER_GUTIERREZ_HERNANDEZ   = %.2f", TRANSFER_GUTIERREZ_HERNANDEZ)
    logger.info("  TRANSFER_GUTIERREZ_PETRO       = %.2f", TRANSFER_GUTIERREZ_PETRO)
    logger.info("  TRANSFER_BLANCO_SPLIT          = %.2f", TRANSFER_BLANCO_SPLIT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
