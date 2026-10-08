"""Evaluate the TransferIQ model on unseen transfer data."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transfer_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "transfer_value_model.pkl"


# Columns that are not model inputs.
IDENTIFIER_COLUMNS = [
    "player_name",
    "previous_season",
    "transfer_season",
    "fee",
    "club",
    "dealing_club",
    "window",
]


def load_data():
    """Load the engineered feature dataset."""
    print("Loading feature dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print(f"  Records: {len(data)}")

    return data


def load_model():
    """Load the trained model pipeline."""
    print("\nLoading trained model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run model training first."
        )

    model = joblib.load(MODEL_PATH)

    print(f"  Model loaded: {MODEL_PATH}")

    return model


def prepare_test_data(data):
    """Prepare the unseen 2025 test dataset."""
    test_data = data[data["transfer_season"] == 2025].copy()

    if test_data.empty:
        raise ValueError("No 2025 test records found.")

    features = test_data.drop(
        columns=[
            column
            for column in IDENTIFIER_COLUMNS
            if column in test_data.columns
        ]
    )

    target = test_data["fee"]

    # Keep categorical columns consistent with training.
    for column in ["position", "element_type"]:
        if column in features.columns:
            features[column] = (
                features[column]
                .fillna("Unknown")
                .astype(str)
            )

    return test_data, features, target


def calculate_metrics(actual, predicted):
    """Calculate standard regression metrics."""
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


def show_prediction_examples(test_data, predicted):
    """Display individual predictions for the test set."""
    results = test_data[
        ["player_name", "fee"]
    ].copy()

    results["predicted_fee"] = predicted
    results["error"] = (
        results["predicted_fee"] - results["fee"]
    )
    results["absolute_error"] = results["error"].abs()

    results = results.sort_values(
        "absolute_error",
        ascending=False,
    )

    print("\nLargest prediction errors:")
    print("-" * 70)

    for _, row in results.head(10).iterrows():
        actual = row["fee"] / 1_000_000
        predicted_fee = row["predicted_fee"] / 1_000_000
        error = row["error"] / 1_000_000

        print(
            f"{row['player_name']:<25} "
            f"Actual: €{actual:>7.2f}M | "
            f"Predicted: €{predicted_fee:>7.2f}M | "
            f"Error: €{error:>7.2f}M"
        )


def main():
    """Run the complete model evaluation."""
    print("=" * 60)
    print("TransferIQ - Model Evaluation")
    print("=" * 60)

    data = load_data()
    model = load_model()

    print("\nPreparing unseen 2025 test data...")

    test_data, features, target = prepare_test_data(data)

    print(f"  Test records: {len(test_data)}")
    print("  Test season: 2025")

    print("\nGenerating predictions...")

    predicted = model.predict(features)

    mae, rmse, r2 = calculate_metrics(
        target,
        predicted,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"\nMAE:  €{mae / 1_000_000:.2f}M")
    print(f"RMSE: €{rmse / 1_000_000:.2f}M")
    print(f"R²:   {r2:.4f}")

    show_prediction_examples(
        test_data,
        predicted,
    )

    print("\n" + "=" * 60)
    print("Model evaluation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
