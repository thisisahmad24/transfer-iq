"""
TransferIQ - Data Cleaning

Cleans the raw datasets collected by collectors.py.

Input:
    data/raw/

Output:
    data/processed/

This file only cleans and prepares the data.
Feature engineering and model training are handled separately.
"""

from pathlib import Path
import json

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------

def clean_column_names(data):
    """Make column names consistent and easy to use."""

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return data


def save_clean_data(data, filename):
    """Save a cleaned dataframe to data/processed/."""

    output_path = PROCESSED_DATA_DIR / filename

    data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"  Saved: {filename} "
        f"({len(data):,} rows)"
    )


# ---------------------------------------------------------
# Transfer data
# ---------------------------------------------------------

def clean_transfer_data():
    """
    Combine all transfer seasons and clean the data.

    Output:
        transfers_clean.csv
    """

    print("\n[1/3] Cleaning transfer data...")

    transfer_files = sorted(
        RAW_DATA_DIR.glob("transfers_*.csv")
    )

    if not transfer_files:
        print("  No transfer files found.")
        return

    frames = []

    for file_path in transfer_files:

        data = pd.read_csv(file_path)

        data = clean_column_names(data)

        # The season is normally already present in the
        # Transfermarkt data. If it is missing, get it
        # from the filename.
        if "season" not in data.columns:

            season = file_path.stem.replace(
                "transfers_",
                "",
            )

            data["season"] = season

        frames.append(data)

    transfers = pd.concat(
        frames,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "age",
        "market_value",
        "fee",
    ]

    for column in numeric_columns:

        if column in transfers.columns:

            transfers[column] = pd.to_numeric(
                transfers[column],
                errors="coerce",
            )

    # -----------------------------------------------------
    # Text columns
    # -----------------------------------------------------

    text_columns = [
        "player_name",
        "club",
        "movement",
        "position",
        "dealing_club",
    ]

    for column in text_columns:

        if column in transfers.columns:

            transfers[column] = (
                transfers[column]
                .astype("string")
                .str.strip()
            )

    # -----------------------------------------------------
    # Transfer fee
    # -----------------------------------------------------

    # Transfers without a known fee cannot currently be
    # used as training examples.
    transfers = transfers.dropna(
        subset=["fee"]
    )

    # Keep paid transfers only.
    transfers = transfers[
        transfers["fee"] > 0
    ].copy()

    # Remove exact duplicate transfer records.
    transfers = transfers.drop_duplicates()

    # -----------------------------------------------------
    # Sorting
    # -----------------------------------------------------

    sort_columns = [
        column
        for column in [
            "season",
            "player_name",
        ]
        if column in transfers.columns
    ]

    if sort_columns:

        transfers = transfers.sort_values(
            sort_columns
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_clean_data(
        transfers,
        "transfers_clean.csv",
    )

    if "player_name" in transfers.columns:

        print(
            f"  Unique players: "
            f"{transfers['player_name'].nunique():,}"
        )


# ---------------------------------------------------------
# Historical FPL data
# ---------------------------------------------------------

def clean_historical_fpl():
    """
    Combine historical FPL player-season files.

    Output:
        fpl_historical_clean.csv
    """

    print("\n[2/3] Cleaning historical FPL data...")

    fpl_files = sorted(
        RAW_DATA_DIR.glob("fpl_20*.csv")
    )

    if not fpl_files:
        print("  No historical FPL files found.")
        return

    frames = []

    for file_path in fpl_files:

        data = pd.read_csv(file_path)

        data = clean_column_names(data)

        # Store the season explicitly.
        season = file_path.stem.replace(
            "fpl_",
            "",
        )

        data["season"] = season

        frames.append(data)

    fpl = pd.concat(
        frames,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Player name
    # -----------------------------------------------------

    if {
        "first_name",
        "second_name",
    }.issubset(fpl.columns):

        fpl["player_name"] = (
            fpl["first_name"].fillna("").astype(str)
            + " "
            + fpl["second_name"].fillna("").astype(str)
        ).str.strip()

    elif "name" in fpl.columns:

        fpl["player_name"] = (
            fpl["name"]
            .astype("string")
            .str.strip()
        )

    # -----------------------------------------------------
    # Numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "total_points",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "value",
    ]

    for column in numeric_columns:

        if column in fpl.columns:

            fpl[column] = pd.to_numeric(
                fpl[column],
                errors="coerce",
            )

    # -----------------------------------------------------
    # Text columns
    # -----------------------------------------------------

    text_columns = [
        "player_name",
        "first_name",
        "second_name",
        "position",
        "team",
    ]

    for column in text_columns:

        if column in fpl.columns:

            fpl[column] = (
                fpl[column]
                .astype("string")
                .str.strip()
            )

    # -----------------------------------------------------
    # Remove invalid player names
    # -----------------------------------------------------

    if "player_name" in fpl.columns:

        fpl = fpl[
            fpl["player_name"].notna()
            & (fpl["player_name"] != "")
        ]

    # -----------------------------------------------------
    # Remove players with no playing time
    # -----------------------------------------------------

    if "minutes" in fpl.columns:

        fpl = fpl[
            fpl["minutes"].fillna(0) > 0
        ].copy()

    # -----------------------------------------------------
    # Remove duplicate records
    # -----------------------------------------------------

    fpl = fpl.drop_duplicates()

    # -----------------------------------------------------
    # Sort data
    # -----------------------------------------------------

    sort_columns = [
        column
        for column in [
            "season",
            "player_name",
        ]
        if column in fpl.columns
    ]

    if sort_columns:

        fpl = fpl.sort_values(
            sort_columns
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_clean_data(
        fpl,
        "fpl_historical_clean.csv",
    )

    if "player_name" in fpl.columns:

        print(
            f"  Unique players: "
            f"{fpl['player_name'].nunique():,}"
        )

    print(
        f"  Player-season records: "
        f"{len(fpl):,}"
    )


# ---------------------------------------------------------
# Current FPL data
# ---------------------------------------------------------

def clean_current_fpl():
    """
    Clean the current FPL bootstrap data.

    Output:
        fpl_current_clean.csv
    """

    print("\n[3/3] Cleaning current FPL data...")

    input_path = RAW_DATA_DIR / "fpl_bootstrap.json"

    if not input_path.exists():

        print("  Current FPL file not found.")

        return

    # -----------------------------------------------------
    # Read JSON
    # -----------------------------------------------------

    try:

        with open(
            input_path,
            "r",
            encoding="utf-8",
        ) as file:

            raw_data = json.load(file)

        players = raw_data.get(
            "elements",
            [],
        )

        data = pd.DataFrame(players)

    except (OSError, json.JSONDecodeError) as error:

        print(
            f"  Could not read FPL data: {error}"
        )

        return

    if data.empty:

        print("  Current FPL dataset is empty.")

        return

    # -----------------------------------------------------
    # Column names
    # -----------------------------------------------------

    data = clean_column_names(data)

    # -----------------------------------------------------
    # Player name
    # -----------------------------------------------------

    if {
        "first_name",
        "second_name",
    }.issubset(data.columns):

        data["player_name"] = (
            data["first_name"].fillna("").astype(str)
            + " "
            + data["second_name"].fillna("").astype(str)
        ).str.strip()

    # -----------------------------------------------------
    # Numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "starts",
        "expected_goals",
        "expected_assists",
        "total_points",
        "value",
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # -----------------------------------------------------
    # Remove duplicate players
    # -----------------------------------------------------

    # FPL records contain some fields with lists.
    # Therefore, dropping duplicates across every column
    # can fail because lists are not hashable.

    if "id" in data.columns:

        data = data.drop_duplicates(
            subset=["id"]
        )

    elif "player_name" in data.columns:

        data = data.drop_duplicates(
            subset=["player_name"]
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_clean_data(
        data,
        "fpl_current_clean.csv",
    )

    print(
        f"  Current players: "
        f"{len(data):,}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    """Run all cleaning steps."""

    print("=" * 60)
    print("TransferIQ - Data Cleaning")
    print("=" * 60)

    clean_transfer_data()
    clean_historical_fpl()
    clean_current_fpl()

    print("\n" + "=" * 60)
    print("Data cleaning completed.")
    print(
        f"Processed data location: "
        f"{PROCESSED_DATA_DIR}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

