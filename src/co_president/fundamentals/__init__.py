"""SPEC-21: Municipal features API.

Typed, frozen ``MunicipalFeatures`` dataclass, ``HistoricalRecord``
dataclass, and ``load_features()`` entry point for the municipal
feature matrix.
"""

from co_president.fundamentals.features import (
    HistoricalRecord,
    MunicipalFeatures,
    load_features,
)

__all__ = [
    "HistoricalRecord",
    "MunicipalFeatures",
    "load_features",
]
