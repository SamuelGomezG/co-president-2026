"""SPEC-12 through SPEC-19: Ingestion pipelines.

SPEC-12: Geographic codes (DIVIPOLA) and historical election results
(2002--2022).
SPEC-13: Socioeconomic (DANE census, IPM, NBI, projections) and risk
(MOE, INDEPAZ, PDET, UNODC coca) data.
SPEC-14: Municipal feature matrix joining all components.
SPEC-15: Demographic data audit, coverage report, and CEDAE-local
historical results.
SPEC-16: Sabaneta Cámara de Representantes fixture and loader.
SPEC-17: CNPV 2018 census microdata ingest (F8/F9/F11).
SPEC-18: NBI and IPM poverty indicators ingest.
SPEC-19: DANE population projections ingest (2018--2026).
"""

from co_president.ingestion.build_feature_matrix import (
    build_feature_matrix,
    generate_data_dictionary,
    load_all_components,
    pivot_historical_wide,
    save_feature_matrix,
    validate_component_health,
)
from co_president.ingestion.ingest_cnpv import (
    build_cnpv_features,
    load_cnpv_data,
    validate_cnpv,
)
from co_president.ingestion.ingest_fiscal import (
    build_fiscal_features,
    load_fiscal_data,
    validate_fiscal,
)
from co_president.ingestion.ingest_ipm import (
    build_ipm_features,
    load_ipm_data,
    validate_ipm,
)
from co_president.ingestion.ingest_nbi import (
    build_nbi_features,
    load_nbi_data,
    validate_nbi,
)
from co_president.ingestion.ingest_population import (
    build_population_features,
    load_population_data,
    validate_population,
)
from co_president.ingestion.ingest_sabaneta import (
    build_sabaneta_camara_matrix,
    load_sabaneta_camara,
    validate_sabaneta,
)
from co_president.ingestion.report import (
    generate_coverage_markdown,
    generate_coverage_report,
    verify_legislative_schemas,
    write_coverage_report,
)

__all__ = [
    "build_cnpv_features",
    "build_feature_matrix",
    "build_fiscal_features",
    "build_ipm_features",
    "build_nbi_features",
    "build_population_features",
    "build_sabaneta_camara_matrix",
    "generate_coverage_markdown",
    "generate_coverage_report",
    "generate_data_dictionary",
    "load_all_components",
    "load_cnpv_data",
    "load_fiscal_data",
    "load_ipm_data",
    "load_nbi_data",
    "load_population_data",
    "load_sabaneta_camara",
    "pivot_historical_wide",
    "save_feature_matrix",
    "validate_cnpv",
    "validate_component_health",
    "validate_fiscal",
    "validate_ipm",
    "validate_nbi",
    "validate_population",
    "validate_sabaneta",
    "verify_legislative_schemas",
    "write_coverage_report",
]
