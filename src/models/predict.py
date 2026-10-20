"""Prediction utilities for TransferIQ."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH


NUMERIC_INPUTS = [
    "age", "minutes", "goals_scored", "assists", "total_points",
    "goals_conceded", "clean_sheets", "bonus", "bps", "influence",
    "creativity", "threat", "ict_index", "red_cards", "yellow_cards",
    "selected_by_percent", "now_cost",
]


def build_prediction_features(player_stats):
    """Convert raw player statistics into engineered model inputs."""
    stats = dict(player_stats)
    for column in NUMERIC_INPUTS:
        if column not in stats:
            raise ValueError(f"Missing required input: {column}")
        try:
            stats[column] = float(stats[column])
        except (TypeError, ValueError):
            raise ValueError(f"Input '{column}' must be numeric.") from None
        if not np.isfinite(stats[column]):
            raise ValueError(f"Input '{column}' must be a finite number.")
        if stats[column] < 0:
            raise ValueError(f"Input '{column}' cannot be negative.")

    if not 0 < stats["age"] <= 45:
        raise ValueError("Player age must be greater than zero and no more than 45.")

    minutes = stats["minutes"]
    goals = stats["goals_scored"]
    assists = stats["assists"]
    cards = stats["red_cards"] + stats["yellow_cards"]
    divisor = minutes if minutes > 0 else np.nan

    features = {
        **stats,
        "position": str(player_stats.get("position", "Unknown")),
        "element_type": str(player_stats.get("element_type", "Unknown")),
        "goal_contributions": goals + assists,
        "goals_per_90": goals / divisor * 90,
        "assists_per_90": assists / divisor * 90,
        "goal_contributions_per_90": (goals + assists) / divisor * 90,
        "points_per_90": stats["total_points"] / divisor * 90,
        "bonus_per_90": stats["bonus"] / divisor * 90,
        "bps_per_90": stats["bps"] / divisor * 90,
        "clean_sheets_per_90": stats["clean_sheets"] / divisor * 90,
        "goals_conceded_per_90": stats["goals_conceded"] / divisor * 90,
        "disciplinary_cards": cards,
        "disciplinary_cards_per_90": cards / divisor * 90,
        "age_squared": stats["age"] ** 2,
        "minutes_ratio": min(minutes / 3420, 1.0),
        "played_full_season": int(minutes >= 3000),
    }

    for name, value in features.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            features[name] = 0.0
    return pd.DataFrame([features])


def load_model(model_path=MODEL_PATH):
    """Load the trained scikit-learn model pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run the model training module first."
        )
    return joblib.load(path)


def _align_to_model_schema(features, model):
    """Align columns to the saved pipeline's original feature schema.

    Missing historical-window features are left missing, not fabricated.
    The trained preprocessing pipeline must impute them if those features exist.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return features
    aligned = features.copy()
    for column in expected:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned.reindex(columns=list(expected))


def predict_transfer_fee(player_stats, model=None):
    """Predict a non-negative transfer fee in euros."""
    if model is None:
        model = load_model()
    features = _align_to_model_schema(build_prediction_features(player_stats), model)
    try:
        prediction = float(model.predict(features)[0])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "Could not predict from these statistics. The saved model may need "
            "to be retrained with the current feature pipeline, including "
            "median imputation for unavailable historical features."
        ) from exc
    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")
    return max(prediction, 0.0)
