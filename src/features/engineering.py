"""
TransferIQ - Feature Engineering

Creates meaningful football performance features from the merged
FPL + transfer dataset.

Input:
    data/processed/transfer_training_data.csv

Output:
    data/processed/transfer_features.csv

The transfer fee remains the prediction target.
Market value is intentionally not used as a model feature.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "transfer_training_data.csv"
OUTPUT_FILE = PROCESSED_DIR / "transfer_features.csv"


def load_training_data():
    """Load the merged training dataset."""
    print("Loading merged training data...")

    data = pd.read_csv(INPUT_FILE)

    print(f"  Records: {len(data):,}")
    print(f"  Columns: {len(data.columns)}")

    return data


def convert_numeric_columns(data):
    """Make sure performance columns contain numeric values."""
    numeric_columns = [
        "age",
        "minutes",
        "goals_scored",
        "assists",
        "total_points",
        "goals_conceded",
        "clean_sheets",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "red_cards",
        "yellow_cards",
        "selected_by_percent",
        "now_cost",
        "fee",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return data


def add_goal_features(data):
    """Create goal and attacking contribution metrics."""
    print("\nCreating attacking features...")

    data["goal_contributions"] = (
        data["goals_scored"] + data["assists"]
    )

    data["goals_per_90"] = (
        data["goals_scored"] / data["minutes"] * 90
    )

    data["assists_per_90"] = (
        data["assists"] / data["minutes"] * 90
    )

    data["goal_contributions_per_90"] = (
        data["goal_contributions"] / data["minutes"] * 90
    )

    return data


def add_performance_features(data):
    """Create general performance metrics."""
    print("Creating performance features...")

    data["points_per_90"] = (
        data["total_points"] / data["minutes"] * 90
    )

    data["bonus_per_90"] = (
        data["bonus"] / data["minutes"] * 90
    )

    data["bps_per_90"] = (
        data["bps"] / data["minutes"] * 90
    )

    data["clean_sheets_per_90"] = (
        data["clean_sheets"] / data["minutes"] * 90
    )

    data["goals_conceded_per_90"] = (
        data["goals_conceded"] / data["minutes"] * 90
    )

    return data


def add_discipline_features(data):
    """Create player discipline metrics."""
    print("Creating discipline features...")

    data["disciplinary_cards"] = (
        data["red_cards"] + data["yellow_cards"]
    )

    data["disciplinary_cards_per_90"] = (
        data["disciplinary_cards"] / data["minutes"] * 90
    )

    return data


def add_age_features(data):
    """Create age-related features for transfer valuation."""
    print("Creating age features...")

    data["age_squared"] = data["age"] ** 2

    return data


def add_minutes_features(data):
    """Create availability and playing-time features."""
    print("Creating playing-time features...")

    data["minutes_ratio"] = (
        data["minutes"] / 3420
    )

    data["minutes_ratio"] = data["minutes_ratio"].clip(
        lower=0,
        upper=1,
    )

    data["played_full_season"] = (
        data["minutes"] >= 3000
    ).astype(int)

    return data


def clean_engineered_values(data):
    """
    Replace invalid values created by division.

    Players with zero minutes can otherwise produce infinity
    when calculating per-90 statistics.
    """
    feature_columns = [
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

    for column in feature_columns:
        if column in data.columns:
            data[column] = (
                data[column]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

    return data


def reorder_columns(data):
    """Place important prediction features together."""
    preferred_columns = [
        "player_name",
        "previous_season",
        "transfer_season",
        "fee",
        "age",
        "age_squared",
        "position",
        "element_type",
        "minutes",
        "minutes_ratio",
        "played_full_season",
        "goals_scored",
        "assists",
        "goal_contributions",
        "goals_per_90",
        "assists_per_90",
        "goal_contributions_per_90",
        "total_points",
        "points_per_90",
        "clean_sheets",
        "clean_sheets_per_90",
        "goals_conceded",
        "goals_conceded_per_90",
        "bonus",
        "bonus_per_90",
        "bps",
        "bps_per_90",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "red_cards",
        "yellow_cards",
        "disciplinary_cards",
        "disciplinary_cards_per_90",
        "selected_by_percent",
        "now_cost",
        "club",
        "dealing_club",
        "window",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in existing_columns
    ]

    return data[existing_columns + remaining_columns]


def save_features(data):
    """Save the engineered feature dataset."""
    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_FILE.name}")
    print(f"Rows:  {len(data):,}")
    print(f"Columns: {len(data.columns)}")


def show_summary(data):
    """Display the engineered features for verification."""
    print("\n" + "=" * 60)
    print("Feature engineering summary")
    print("=" * 60)

    feature_columns = [
        "goals_per_90",
        "assists_per_90",
        "goal_contributions",
        "goal_contributions_per_90",
        "points_per_90",
        "bonus_per_90",
        "bps_per_90",
        "clean_sheets_per_90",
        "goals_conceded_per_90",
        "disciplinary_cards_per_90",
        "age_squared",
        "minutes_ratio",
        "played_full_season",
    ]

    available_features = [
        column
        for column in feature_columns
        if column in data.columns
    ]

    print("\nEngineered features:")

    for column in available_features:
        print(f"  - {column}")

    print("\nExample records:")

    preview_columns = [
        "player_name",
        "age",
        "minutes",
        "goals_scored",
        "assists",
        "goals_per_90",
        "assists_per_90",
        "goal_contributions_per_90",
        "total_points",
        "points_per_90",
        "fee",
    ]

    available_preview = [
        column
        for column in preview_columns
        if column in data.columns
    ]

    print(
        data[available_preview]
        .head(10)
        .to_string(index=False)
    )

    print("\nMissing values in engineered features:")

    missing_values = data[available_features].isna().sum()

    print(missing_values.to_string())


def main():
    """Run the complete feature engineering pipeline."""
    print("=" * 60)
    print("TransferIQ - Feature Engineering")
    print("=" * 60)

    data = load_training_data()

    data = convert_numeric_columns(data)

    data = add_goal_features(data)
    data = add_performance_features(data)
    data = add_discipline_features(data)
    data = add_age_features(data)
    data = add_minutes_features(data)

    data = clean_engineered_values(data)

    data = reorder_columns(data)

    save_features(data)

    show_summary(data)

    print("\n" + "=" * 60)
    print("Feature engineering completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()