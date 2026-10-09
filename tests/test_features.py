"""Tests for TransferIQ feature engineering."""

import numpy as np
import pandas as pd

from src.features.engineering import (
    add_age_features,
    add_discipline_features,
    add_goal_features,
    add_minutes_features,
    add_performance_features,
    clean_engineered_values,
)


def sample_player():
    return pd.DataFrame(
        [{
            "age": 24,
            "minutes": 1800,
            "goals_scored": 10,
            "assists": 5,
            "total_points": 150,
            "goals_conceded": 20,
            "clean_sheets": 8,
            "bonus": 12,
            "bps": 400,
            "red_cards": 0,
            "yellow_cards": 3,
        }]
    )


def test_goal_features_are_calculated():
    data = add_goal_features(sample_player())

    assert data.loc[0, "goal_contributions"] == 15
    assert np.isclose(data.loc[0, "goals_per_90"], 0.5)
    assert np.isclose(data.loc[0, "assists_per_90"], 0.25)


def test_age_and_minutes_features():
    data = sample_player()
    data = add_age_features(data)
    data = add_minutes_features(data)

    assert data.loc[0, "age_squared"] == 576
    assert np.isclose(data.loc[0, "minutes_ratio"], 1800 / 3420)
    assert data.loc[0, "played_full_season"] == 0


def test_performance_and_discipline_features():
    data = sample_player()
    data = add_performance_features(data)
    data = add_discipline_features(data)

    assert np.isclose(data.loc[0, "points_per_90"], 7.5)
    assert data.loc[0, "disciplinary_cards"] == 3


def test_zero_minutes_do_not_create_infinite_features():
    data = sample_player()
    data["minutes"] = 0

    data = add_goal_features(data)
    data = add_performance_features(data)
    data = add_discipline_features(data)
    data = clean_engineered_values(data)

    engineered = [
        "goals_per_90",
        "assists_per_90",
        "goal_contributions_per_90",
        "points_per_90",
        "bonus_per_90",
        "bps_per_90",
        "clean_sheets_per_90",
        "goals_conceded_per_90",
        "disciplinary_cards_per_90",
    ]

    assert np.isfinite(data[engineered].to_numpy()).all()
    assert (data[engineered] == 0).all().all()