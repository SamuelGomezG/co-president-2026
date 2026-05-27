import pandas as pd

from co_president.config import ModelConfig
from co_president.model_round1 import build_round1_model


def test_build_round1_model_house_effects() -> None:
    """Test model graph with house effects."""
    # Synthetic data: 2 candidates, 2 pollsters, 1 time point (T=0)
    # Using all candidates to match FIRST_ROUND_CANDIDATES
    polls_data = {
        "fecha": ["2022-05-29", "2022-05-29"],
        "encuestadora": ["PollsterA", "PollsterB"],
        "muestra": [1000, 1000],
        "gustavo_petro": [50.0, 51.0],
        "federico_gutierrez": [10.0, 10.0],
        "rodolfo_hernandez": [20.0, 20.0],
        "sergio_fajardo": [10.0, 10.0],
        "ingrid_betancourt": [2.0, 2.0],
        "rest": [5.0, 5.0],
        "blanco": [3.0, 2.0],
        "round_number": [1, 1],
    }
    polls = pd.DataFrame(polls_data)

    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)

    model = build_round1_model(polls, None, config)

    # 4 free RVs: sigma_rw, theta[0], sigma_house, raw_house
    # deterministics: house_effects, p_adj
    assert len(model.free_RVs) == 4
    assert len(model.deterministics) == 2
