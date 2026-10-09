"""Tests for TransferIQ prediction utilities."""

import numpy as np
import pytest

from src.models.predict import (
    build_prediction_features,
    predict_transfer_fee,
)


def sample_stats():
    return {
        "age": 24,
        "minutes": 1800,
        "goals_scored": 10,
        "assists": 5,
        "total_points": 150,
        "goals_conceded": 20,
        "clean_sheets": 8,
        "bonus": 12,
        "bps": 400,
        "influence": 500,
        "creativity": 450,
        "threat": 600,
        "ict_index": 155,
        "red_cards": 0,
        "yellow_cards": 3,
        "selected_by_percent": 12.5,
        "now_cost": 75,
        "position": "Midfielder",
        "element_type": "3",
    }


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, features):
        assert len(features) == 1
        assert "goal_contributions_per_90" in features.columns
        return np.array([self.prediction])


def test_prediction_features_include_engineered_metrics():
    features = build_prediction_features(sample_stats())

    assert features.loc[0, "goal_contributions"] == 15
    assert np.isclose(
        features.loc[0, "goal_contributions_per_90"], 0.75
    )
    assert features.loc[0, "age_squared"] == 576


def test_prediction_returns_fee_in_euros():
    prediction = predict_transfer_fee(
        sample_stats(),
        model=FakeModel(25_000_000),
    )

    assert prediction == 25_000_000


def test_negative_predictions_are_clamped_to_zero():
    prediction = predict_transfer_fee(
        sample_stats(),
        model=FakeModel(-100),
    )

    assert prediction == 0


def test_missing_numeric_input_is_rejected():
    stats = sample_stats()
    del stats["minutes"]

    with pytest.raises(ValueError, match="Missing required input"):
        build_prediction_features(stats)


def test_negative_age_is_rejected():
    stats = sample_stats()
    stats["age"] = -2

    with pytest.raises(ValueError, match="cannot be negative"):
        build_prediction_features(stats)