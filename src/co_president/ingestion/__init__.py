"""SPEC-12 through SPEC-16: Ingestion pipelines.

SPEC-12: Geographic codes (DIVIPOLA) and historical election results
(2002--2022).
SPEC-13: Socioeconomic (DANE census, IPM, NBI, projections) and risk
(MOE, INDEPAZ, PDET, UNODC coca) data.
SPEC-14: Municipal feature matrix joining all components.
SPEC-15: Demographic data audit, coverage report, and CEDAE-local
historical results.
SPEC-16: Sabaneta Cámara de Representantes fixture and loader.
"""

from co_president.ingestion.build_feature_matrix import (
    build_feature_matrix,
    generate_data_dictionary,
    load_all_components,
    pivot_historical_wide,
    save_feature_matrix,
    validate_component_health,
)
from co_president.ingestion.ingest_sabaneta import load_sabaneta_camara
from co_president.ingestion.report import (
    generate_coverage_markdown,
    generate_coverage_report,
    verify_legislative_schemas,
    write_coverage_report,
)

__all__ = [
    "build_feature_matrix",
    "generate_coverage_markdown",
    "generate_coverage_report",
    "generate_data_dictionary",
    "load_all_components",
    "load_sabaneta_camara",
    "pivot_historical_wide",
    "save_feature_matrix",
    "validate_component_health",
    "verify_legislative_schemas",
    "write_coverage_report",
]
