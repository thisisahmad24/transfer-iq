"""Train and compare TransferIQ regression models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transfer_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "transfer_value_model.pkl"


# Columns that identify a transfer but should not be model inputs.
IDENTIFIER_COLUMNS = [
    "player_name",
    "previous_season",
    "transfer_season",
    "fee",
    "club",
    "dealing_club",
    "window",
]


CATEGORICAL_COLUMNS = [
    "position",
    "element_type",
]


def load_data():
    """Load the engineered dataset."""
    print("Loading feature dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}\n"
            "Run feature engineering first."
        )

    data = pd.read_csv(DATA_PATH)

    print(f"  Records: {len(data)}")
    print(f"  Columns: {len(data.columns)}")

    return data


def prepare_features(data):
    """Separate model inputs from the transfer fee target."""
    data = data.copy()

    features = data.drop(
        columns=[
            column
            for column in IDENTIFIER_COLUMNS
            if column in data.columns
        ]
    )

    target = data["fee"]

    for column in CATEGORICAL_COLUMNS:
        if column in features.columns:
            features[column] = (
                features[column]
                .fillna("Unknown")
                .astype(str)
            )

    return features, target


def split_data(data):
    """Create chronological training, validation, and test sets."""
    train_data = data[
        data["transfer_season"].isin([2021, 2022, 2023])
    ].copy()

    validation_data = data[
        data["transfer_season"] == 2024
    ].copy()

    test_data = data[
        data["transfer_season"] == 2025
    ].copy()

    if train_data.empty:
        raise ValueError("Training data is empty.")

    if validation_data.empty:
        raise ValueError("Validation data is empty.")

    if test_data.empty:
        raise ValueError("Test data is empty.")

    return train_data, validation_data, test_data


def build_preprocessor(features):
    """Build preprocessing for numerical and categorical features."""
    categorical_features = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in features.columns
    ]

    numeric_features = [
        column
        for column in features.columns
        if column not in categorical_features
    ]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ],
        remainder="drop",
    )


def build_model(model):
    """Build a complete pipeline with log-transformed target."""
    # Transfer fees are highly right-skewed.
    # log1p/expm1 lets the model learn on a compressed target scale.
    regression_model = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return regression_model


def create_model_pipeline(model, features):
    """Create preprocessing + regression pipeline."""
    preprocessor = build_preprocessor(features)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", build_model(model)),
        ]
    )


def get_models():
    """Return the models that will be compared."""
    return {
        "Log-Linear Regression": LinearRegression(),

        "Log-Ridge Regression": Ridge(
            alpha=10.0
        ),

        "Log-Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "Log-Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            random_state=42,
        ),
    }


def prepare_model_data(data):
    """Prepare features and target from a dataset."""
    features, target = prepare_features(data)
    return features, target


def make_predictions(model, features):
    """Generate realistic non-negative transfer fee predictions."""
    predictions = model.predict(features)

    # A transfer fee cannot be negative.
    return np.maximum(predictions, 0)


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE."""
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    return mae, rmse


def evaluate_on_validation(models, train_data, validation_data):
    """Compare models using the 2024 validation season."""
    print("\n" + "=" * 60)
    print("Model Selection - 2024 Validation")
    print("=" * 60)

    train_features, train_target = prepare_model_data(train_data)
    validation_features, validation_target = prepare_model_data(
        validation_data
    )

    results = []

    for name, estimator in models.items():
        print(f"\nTraining: {name}")

        model = create_model_pipeline(
            estimator,
            train_features,
        )

        model.fit(
            train_features,
            train_target,
        )

        predictions = make_predictions(
            model,
            validation_features,
        )

        mae, rmse = calculate_metrics(
            validation_target,
            predictions,
        )

        results.append(
            {
                "model": name,
                "mae": mae,
                "rmse": rmse,
            }
        )

        print(
            f"  MAE:  €{mae / 1_000_000:.2f}M"
        )
        print(
            f"  RMSE: €{rmse / 1_000_000:.2f}M"
        )

    results_df = pd.DataFrame(results)

    # MAE is the primary selection metric.
    results_df = results_df.sort_values(
        by=["mae", "rmse"]
    ).reset_index(drop=True)

    print("\n" + "-" * 60)
    print("Validation ranking:")
    print("-" * 60)

    for index, row in results_df.iterrows():
        print(
            f"{index + 1}. {row['model']:<25} "
            f"MAE: €{row['mae'] / 1_000_000:.2f}M | "
            f"RMSE: €{row['rmse'] / 1_000_000:.2f}M"
        )

    best_model_name = results_df.iloc[0]["model"]

    print(
        f"\nSelected model: {best_model_name}"
    )

    return best_model_name


def train_final_model(
    model_name,
    models,
    training_data,
):
    """Retrain the selected model using all 2021-2024 data."""
    print("\n" + "=" * 60)
    print("Final Model Training")
    print("=" * 60)

    features, target = prepare_model_data(
        training_data
    )

    print(
        f"Training selected model: {model_name}"
    )
    print(
        f"Training records: {len(features)}"
    )

    model = create_model_pipeline(
        models[model_name],
        features,
    )

    model.fit(
        features,
        target,
    )

    print("Final training completed.")

    return model


def evaluate_final_model(
    model,
    test_data,
):
    """Evaluate the selected model on untouched 2025 data."""
    print("\n" + "=" * 60)
    print("Final Evaluation - Unseen 2025 Season")
    print("=" * 60)

    features, target = prepare_model_data(
        test_data
    )

    predictions = make_predictions(
        model,
        features,
    )

    mae, rmse = calculate_metrics(
        target,
        predictions,
    )

    print(f"\nTest records: {len(test_data)}")
    print(
        f"MAE:  €{mae / 1_000_000:.2f}M"
    )
    print(
        f"RMSE: €{rmse / 1_000_000:.2f}M"
    )

    print("\nPrediction sanity check:")
    print(
        f"  Minimum prediction: "
        f"€{predictions.min() / 1_000_000:.2f}M"
    )
    print(
        f"  Maximum prediction: "
        f"€{predictions.max() / 1_000_000:.2f}M"
    )
    print(
        f"  Negative predictions: "
        f"{(predictions < 0).sum()}"
    )

    results = test_data[
        ["player_name", "fee"]
    ].copy()

    results["predicted_fee"] = predictions

    results["absolute_error"] = (
        results["predicted_fee"] - results["fee"]
    ).abs()

    results = results.sort_values(
        "absolute_error",
        ascending=False,
    )

    print("\nLargest prediction errors:")
    print("-" * 70)

    for _, row in results.head(10).iterrows():
        actual = row["fee"] / 1_000_000
        predicted = (
            row["predicted_fee"] / 1_000_000
        )
        error = (
            row["predicted_fee"] - row["fee"]
        ) / 1_000_000

        print(
            f"{row['player_name']:<25} "
            f"Actual: €{actual:>7.2f}M | "
            f"Predicted: €{predicted:>7.2f}M | "
            f"Error: €{error:>7.2f}M"
        )

    return mae, rmse


def save_model(model):
    """Save only the selected final model."""
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nSaved final model: {MODEL_PATH}"
    )


def main():
    """Run model comparison, selection, final training, and evaluation."""
    print("=" * 60)
    print("TransferIQ - Model Training & Selection")
    print("=" * 60)

    data = load_data()

    (
        train_data,
        validation_data,
        test_data,
    ) = split_data(data)

    print("\nChronological data split:")
    print(
        f"  Model training: 2021-2023 "
        f"({len(train_data)} records)"
    )
    print(
        f"  Model selection: 2024 "
        f"({len(validation_data)} records)"
    )
    print(
        f"  Final test: 2025 "
        f"({len(test_data)} records)"
    )

    models = get_models()

    # Step 1: Select the best model without touching 2025.
    best_model_name = evaluate_on_validation(
        models,
        train_data,
        validation_data,
    )

    # Step 2: Retrain the selected model on all
    # historical data available before the test season.
    final_training_data = pd.concat(
        [
            train_data,
            validation_data,
        ],
        ignore_index=True,
    )

    final_model = train_final_model(
        best_model_name,
        models,
        final_training_data,
    )

    # Step 3: Evaluate the selected model once
    # against the completely unseen 2025 season.
    evaluate_final_model(
        final_model,
        test_data,
    )

    # Step 4: Save only the selected model.
    save_model(final_model)

    print("\n" + "=" * 60)
    print("Model selection and training completed.")
    print("=" * 60)
    print(
        f"\nFinal selected model: {best_model_name}"
    )
    print(
        "\nThe 2025 test set remained unseen during "
        "model selection."
    )


if __name__ == "__main__":
    main()
