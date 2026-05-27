import pandas as pd

from co_president.config import ModelConfig
from co_president.model_round1 import build_round1_model


def test_build_round1_model_minimal() -> None:
    """Test minimal model graph construction."""
    # Synthetic data: 2 candidates, 2 polls, 1 time point (T=0)
    polls_data = {
        "fecha": ["2022-05-29", "2022-05-29"],
        "encuestadora": ["A", "B"],
        "muestra": [1000, 1000],
        "gustavo_petro": [50.0, 51.0],
        "rodolfo_hernandez": [40.0, 39.0],
        "blanco": [5.0, 5.0],
        "otros": [5.0, 5.0],
        "round_number": [1, 1],
    }
    polls = pd.DataFrame(polls_data)

    config = ModelConfig(random_walk_sigma_prior=0.5)

    model = build_round1_model(polls, None, config)

    assert model is not None
    # For Phase A: 1 RV, 0 det, 0 obs
    assert len(model.free_RVs) == 1
    assert len(model.deterministics) == 0
    assert len(model.observed_RVs) == 0
