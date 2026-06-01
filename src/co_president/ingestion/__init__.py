"""SPEC-12 through SPEC-14: Ingestion pipelines.

SPEC-12: Geographic codes (DIVIPOLA) and historical election results
(2002--2022).
SPEC-13: Socioeconomic (DANE census, IPM, NBI, projections) and risk
(MOE, INDEPAZ, PDET, UNODC coca) data.
SPEC-14: Municipal feature matrix joining all components.
"""

from co_president.ingestion.build_feature_matrix import (
    build_feature_matrix,
    generate_data_dictionary,
    load_all_components,
    pivot_historical_wide,
    save_feature_matrix,
    validate_component_health,
)

__all__ = [
    "build_feature_matrix",
    "generate_data_dictionary",
    "load_all_components",
    "pivot_historical_wide",
    "save_feature_matrix",
    "validate_component_health",
]
