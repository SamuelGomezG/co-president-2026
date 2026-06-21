"""SPEC-32: Tests for CNE 2026 cross-validation against La Silla Vacia."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from co_president.data_cne_2026 import CANDIDATE_KEY_MAP_2026
from co_president.validation_cne_2026 import (
    compare_toplines,
    load_lsvc_detalle,
    load_lsvc_ponderacion,
    write_validation_report,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadLsvcDetalle:
    """Tests for load_lsvc_detalle with synthetic CSV data."""

    def test_loads_and_normalizes(self, tmp_path: Path) -> None:
        """Synthetic CSV loads with normalized columns."""
        csv_content = (
            "Fecha_Publicacion,Fecha_Fin_Campo,Encuestadora,Calificacion,"
            "Metodologia,Tamano_Muestral,Candidato,Porcentaje\n"
            "2025-01-30,2025-01-29,CNC,5.8,presencial,1513,Claudia López,6.90\n"
            "2025-01-30,2025-01-29,CNC,5.8,presencial,1513,Sergio Fajardo,11.80\n"
            "2025-01-30,2025-01-29,CNC,5.8,presencial,1513,Luis Gilberto Murillo,1.80\n"
        )
        detalle_path = tmp_path / "2026-polls" / "silla_vacia_ponderador"
        detalle_path.mkdir(parents=True)
        (detalle_path / "encuestas_detalle.csv").write_text(csv_content)

        df = load_lsvc_detalle(data_dir=tmp_path)
        assert list(df.columns) == ["fecha", "encuestadora", "porcentaje", "candidate_key"]
        assert len(df) == 3
        assert df["encuestadora"].iloc[0] == "cnc"
        assert df["candidate_key"].iloc[0] == "claudia_lopez"
        assert df["candidate_key"].iloc[1] == "fajardo"
        assert df["candidate_key"].iloc[2] == "rest"
        assert df["fecha"].iloc[0] == date(2025, 1, 29)
        assert df["porcentaje"].iloc[0] == 6.90

    def test_skips_header_value(self, tmp_path: Path) -> None:
        """Rows where Candidato equals 'Candidato' are dropped."""
        csv_content = (
            "Fecha_Publicacion,Fecha_Fin_Campo,Encuestadora,Calificacion,"
            "Metodologia,Tamano_Muestral,Candidato,Porcentaje\n"
            "2025-01-30,2025-01-29,CNC,5.8,presencial,1513,Candidato,50.00\n"
            "2025-01-30,2025-01-29,CNC,5.8,presencial,1513,Ivan Cepeda,20.00\n"
        )
        detalle_path = tmp_path / "2026-polls" / "silla_vacia_ponderador"
        detalle_path.mkdir(parents=True)
        (detalle_path / "encuestas_detalle.csv").write_text(csv_content)

        df = load_lsvc_detalle(data_dir=tmp_path)
        assert len(df) == 1
        assert df["candidate_key"].iloc[0] == "cepeda"


class TestLoadLsvcPonderacion:
    """Tests for load_lsvc_ponderacion with synthetic CSV data."""

    def test_loads_and_normalizes(self, tmp_path: Path) -> None:
        """Synthetic ponderacion CSV loads with normalized columns."""
        csv_content = (
            "Fecha,Candidato,Ponderado,Ponderado_Min,Ponderado_Max,Banda,"
            "Tipo,Tendencia,Cambio_pp,Tendencia_Mensual_pp\n"
            "2025-01-29,Claudia López,6.90,6.90,6.90,0.00,Historico,,,\n"
            "2025-01-29,Sergio Fajardo,11.80,11.80,11.80,0.00,Historico,,,\n"
            "2025-01-29,Voto en Blanco,5.00,5.00,5.00,0.00,Historico,,,\n"
        )
        pond_path = tmp_path / "2026-polls" / "silla_vacia_ponderador"
        pond_path.mkdir(parents=True)
        (pond_path / "ponderacion.csv").write_text(csv_content)

        df = load_lsvc_ponderacion(data_dir=tmp_path)
        assert list(df.columns) == [
            "fecha",
            "ponderado",
            "ponderado_min",
            "ponderado_max",
            "candidate_key",
        ]
        assert len(df) == 3
        assert df["candidate_key"].iloc[0] == "claudia_lopez"
        assert df["candidate_key"].iloc[1] == "fajardo"
        assert df["candidate_key"].iloc[2] == "blanco"
        assert df["ponderado"].iloc[0] == 6.90
        assert df["ponderado_min"].iloc[0] == 6.90
        assert df["ponderado_max"].iloc[0] == 6.90

    def test_skips_header_value(self, tmp_path: Path) -> None:
        """Rows where Candidato equals 'Candidato' are dropped."""
        csv_content = (
            "Fecha,Candidato,Ponderado,Ponderado_Min,Ponderado_Max,Banda,"
            "Tipo,Tendencia,Cambio_pp,Tendencia_Mensual_pp\n"
            "2025-01-29,Candidato,50.00,50.00,50.00,0.00,Historico,,,\n"
            "2025-01-29,Ivan Cepeda,20.00,20.00,20.00,0.00,Historico,,,\n"
        )
        pond_path = tmp_path / "2026-polls" / "silla_vacia_ponderador"
        pond_path.mkdir(parents=True)
        (pond_path / "ponderacion.csv").write_text(csv_content)

        df = load_lsvc_ponderacion(data_dir=tmp_path)
        assert len(df) == 1
        assert df["candidate_key"].iloc[0] == "cepeda"


class TestCompareToplines:
    """Tests for compare_toplines with synthetic DataFrames."""

    @staticmethod
    def _make_cne_topline() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fecha": [date(2025, 1, 29), date(2025, 3, 20)],
                "encuestadora": ["cnc", "cnc"],
                "field_end": [date(2025, 1, 29), date(2025, 3, 20)],
                "cepeda": [3.6, 4.1],
                "fajardo": [11.4, 9.5],
                "claudia_lopez": [8.6, 6.8],
                "de_la_espriella": [1.1, 1.7],
                "valencia": [0.0, 0.8],
                "rest": [5.0, 4.1],
                "blanco": [2.8, 4.1],
                "total_weight": [30.0, 30.0],
            }
        )

    @staticmethod
    def _make_lsvc_detalle() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fecha": [
                    date(2025, 1, 29),
                    date(2025, 1, 29),
                    date(2025, 1, 29),
                    date(2025, 3, 20),
                    date(2025, 3, 20),
                    date(2025, 3, 20),
                ],
                "encuestadora": ["cnc", "cnc", "cnc", "cnc", "cnc", "cnc"],
                "candidate_key": [
                    "cepeda",
                    "fajardo",
                    "claudia_lopez",
                    "cepeda",
                    "fajardo",
                    "claudia_lopez",
                ],
                "porcentaje": [3.6, 11.4, 8.6, 3.8, 9.7, 7.0],
            }
        )

    def test_matches_and_errors(self) -> None:
        """Synthetic comparison returns correct paired rows and error metrics."""
        cne = self._make_cne_topline()
        lsvc = self._make_lsvc_detalle()
        result = compare_toplines(cne, lsvc)

        assert result["paired_rows"] == 6
        assert result["overall_mae"] == pytest.approx(0.1167, abs=1e-4)
        assert abs(result["overall_mae"]) == result["overall_mae"]

        details = result["details"]
        assert list(details.columns) == [
            "fecha",
            "encuestadora",
            "candidate_key",
            "cne_pct",
            "lsvc_pct",
            "abs_error",
        ]

        cepeda_row = details[details["candidate_key"] == "cepeda"]
        assert len(cepeda_row) == 2
        assert cepeda_row["abs_error"].iloc[0] == 0.0  # exact match
        assert cepeda_row["abs_error"].iloc[1] == pytest.approx(0.3)  # 4.1 - 3.8

    def test_empty_when_no_overlap(self) -> None:
        """Returns empty results when no rows join."""
        cne = self._make_cne_topline()
        lsvc = pd.DataFrame(
            {
                "fecha": [date(2024, 12, 1)],
                "encuestadora": ["gad3"],
                "candidate_key": ["cepeda"],
                "porcentaje": [5.0],
            }
        )
        result = compare_toplines(cne, lsvc)
        assert result["paired_rows"] == 0
        assert result["details"].empty


class TestWriteValidationReport:
    """Tests for write_validation_report."""

    def test_writes_empty_report(self, tmp_path: Path) -> None:
        """Empty report written successfully."""
        result = {
            "paired_rows": 0,
            "mae_candidate": pd.DataFrame(),
            "rmse_candidate": pd.DataFrame(),
            "max_error_candidate": pd.DataFrame(),
            "mae_firm": pd.DataFrame(),
            "rmse_firm": pd.DataFrame(),
            "overall_mae": float("nan"),
            "overall_rmse": float("nan"),
            "details": pd.DataFrame(),
        }
        out_dir = tmp_path / "reports"
        report_path = write_validation_report(result, out_dir)
        assert report_path.exists()
        content = report_path.read_text()
        assert "No overlapping" in content

    def test_writes_full_report(self, tmp_path: Path) -> None:
        """Full report with metrics written successfully."""
        result = {
            "paired_rows": 3,
            "mae_candidate": pd.DataFrame(
                {"candidate_key": ["cepeda", "fajardo"], "mae": [0.5, 1.0]}
            ),
            "rmse_candidate": pd.DataFrame(
                {"candidate_key": ["cepeda", "fajardo"], "rmse": [0.8, 1.4]}
            ),
            "max_error_candidate": pd.DataFrame(
                {"candidate_key": ["cepeda", "fajardo"], "max_error": [1.0, 2.0]}
            ),
            "mae_firm": pd.DataFrame({"encuestadora": ["cnc"], "mae": [0.75]}),
            "rmse_firm": pd.DataFrame({"encuestadora": ["cnc"], "rmse": [1.1]}),
            "overall_mae": 0.75,
            "overall_rmse": 1.1,
            "details": pd.DataFrame(),
        }
        out_dir = tmp_path / "reports"
        report_path = write_validation_report(result, out_dir)
        content = report_path.read_text()
        assert "Overall MAE" in content
        assert "0.7500" in content
        assert "cepeda" in content


class TestLsvcCandidateNamesInCneMap:
    """Verify all LSVC candidate names are in CANDIDATE_KEY_MAP_2026."""

    def test_all_lsvc_names_mapped(self) -> None:
        """Every candidate name from LSVC data has an entry in the map."""
        lsvc_names = {
            "Abelardo De La Espriella",
            "Carlos Caicedo",
            "Clara López",
            "Claudia López",
            "Ivan Cepeda",
            "Luis Gilberto Murillo",
            "Mauricio Lizcano",
            "Miguel Uribe Londoño",
            "Ninguno",
            "No Sabe / No Responde",
            "Paloma Valencia",
            "Roy Barreras",
            "Santiago Botero",
            "Sergio Fajardo",
            "Sondra Macollins",
            "Voto en Blanco",
        }
        lower_map = {k.lower(): v for k, v in CANDIDATE_KEY_MAP_2026.items()}
        for name in lsvc_names:
            assert name in CANDIDATE_KEY_MAP_2026 or name.lower() in lower_map, (
                f"LSVC candidate name {name!r} not found in CANDIDATE_KEY_MAP_2026"
            )
