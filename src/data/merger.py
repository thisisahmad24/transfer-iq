"""
TransferIQ - Data Merger

Matches previous-season FPL performance with the following season's
Premier League transfer fees.

Example:

    FPL 2023-24 performance
            +
    2024 incoming transfer
            =
    Training record

Output:
    data/processed/transfer_training_data.csv
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FPL_FILE = PROCESSED_DIR / "fpl_historical_clean.csv"
TRANSFER_FILE = PROCESSED_DIR / "transfers_clean.csv"

OUTPUT_FILE = PROCESSED_DIR / "transfer_training_data.csv"


# ---------------------------------------------------------------------------
# Player name matching
# ---------------------------------------------------------------------------

def normalize_name(name):
    """
    Convert a player name into a consistent format for matching.

    Examples:
        Alexander Sørloth -> alexander sorloth
        Aaron Ramsdale   -> aaron ramsdale
    """

    if pd.isna(name):
        return ""

    name = str(name).strip().lower()

    # Normalize accented characters.
    name = unicodedata.normalize("NFKD", name)

    name = "".join(
        character
        for character in name
        if not unicodedata.combining(character)
    )

    # Handle characters that are not converted by Unicode normalization.
    replacements = {
        "ø": "o",
        "đ": "d",
        "ð": "d",
        "þ": "th",
        "ł": "l",
        "æ": "ae",
        "œ": "oe",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Remove punctuation.
    name = re.sub(r"[^a-z0-9\s]", " ", name)

    # Remove repeated spaces.
    name = re.sub(r"\s+", " ", name).strip()

    return name


# ---------------------------------------------------------------------------
# Season handling
# ---------------------------------------------------------------------------

def transfer_year_to_fpl_season(year):
    """
    Convert a transfer year into the previous FPL season.

    Examples:
        2021 -> 2020-21
        2022 -> 2021-22
        2023 -> 2022-23
        2024 -> 2023-24
        2025 -> 2024-25
    """

    year = int(year)

    return f"{year - 1}-{str(year)[-2:]}"


# ---------------------------------------------------------------------------
# Load cleaned datasets
# ---------------------------------------------------------------------------

def load_data():
    """Load the cleaned FPL and transfer datasets."""

    print("Loading cleaned datasets...")

    fpl = pd.read_csv(FPL_FILE)
    transfers = pd.read_csv(TRANSFER_FILE)

    print(f"  FPL records:      {len(fpl):,}")
    print(f"  Transfer records: {len(transfers):,}")

    return fpl, transfers


# ---------------------------------------------------------------------------
# Prepare FPL data
# ---------------------------------------------------------------------------

def prepare_fpl_data(fpl):
    """Prepare historical FPL data for player-season matching."""

    print("\nPreparing historical FPL data...")

    fpl = fpl.copy()

    # Create a normalized name used only for matching.
    fpl["name_key"] = fpl["player_name"].apply(normalize_name)

    # These are the FPL seasons for which we have historical data.
    valid_seasons = {
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
    }

    fpl = fpl[
        fpl["season"].astype(str).isin(valid_seasons)
    ].copy()

    # Remove rows without a usable player name.
    fpl = fpl[fpl["name_key"] != ""].copy()

    # Keep one record per player per season.
    fpl = fpl.drop_duplicates(
        subset=["name_key", "season"],
        keep="first",
    )

    print(f"  Usable FPL records: {len(fpl):,}")
    print(f"  Unique players:     {fpl['name_key'].nunique():,}")

    return fpl


# ---------------------------------------------------------------------------
# Prepare transfer data
# ---------------------------------------------------------------------------

def prepare_transfer_data(transfers):
    """
    Keep permanent incoming Premier League transfers with a positive fee.
    """

    print("\nPreparing transfer data...")

    transfers = transfers.copy()

    # Make sure important fields are numeric.
    transfers["season"] = pd.to_numeric(
        transfers["season"],
        errors="coerce",
    )

    transfers["fee"] = pd.to_numeric(
        transfers["fee"],
        errors="coerce",
    )

    transfers["is_loan"] = pd.to_numeric(
        transfers["is_loan"],
        errors="coerce",
    ).fillna(0)

    # We want permanent incoming transfers only.
    transfers = transfers[
        (transfers["movement"].astype(str).str.lower() == "in")
        & (transfers["is_loan"] == 0)
        & (transfers["fee"] > 0)
    ].copy()

    # 2020 requires the 2019-20 FPL season, which we do not currently have.
    # Therefore training starts with the 2021 transfer season.
    transfers = transfers[
        transfers["season"].between(2021, 2025)
    ].copy()

    # Map transfer year to the preceding FPL season.
    transfers["fpl_season"] = transfers["season"].apply(
        transfer_year_to_fpl_season
    )

    # Create normalized player names for matching.
    transfers["name_key"] = transfers["player_name"].apply(
        normalize_name
    )

    transfers = transfers[
        transfers["name_key"] != ""
    ].copy()

    print(f"  Eligible transfers: {len(transfers):,}")
    print(f"  Unique players:     {transfers['name_key'].nunique():,}")

    return transfers


# ---------------------------------------------------------------------------
# Merge FPL and transfer data
# ---------------------------------------------------------------------------

def merge_datasets(fpl, transfers):
    """
    Match a player's previous-season FPL performance to their
    subsequent incoming transfer.
    """

    print("\nMatching FPL performance with transfer fees...")

    merged = transfers.merge(
        fpl,
        how="inner",
        left_on=["name_key", "fpl_season"],
        right_on=["name_key", "season"],
        suffixes=("_transfer", "_fpl"),
    )

    print(f"  Matched training records: {len(merged):,}")

    if len(transfers) > 0:
        match_rate = len(merged) / len(transfers) * 100
        print(f"  Match rate:              {match_rate:.1f}%")

    return merged


# ---------------------------------------------------------------------------
# Select final training columns
# ---------------------------------------------------------------------------

def select_training_columns(merged):
    """Create the clean final training dataset."""

    columns = [
        # Player
        "player_name_transfer",

        # Transfer information
        "season_transfer",
        "fee",
        "age",
        "position",
        "club",
        "dealing_club",
        "window",

        # Previous-season FPL performance
        "season_fpl",
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
        "element_type",
    ]

    # Only select columns that actually exist.
    available_columns = [
        column
        for column in columns
        if column in merged.columns
    ]

    result = merged[available_columns].copy()

    # Use simple names in the final dataset.
    result = result.rename(
        columns={
            "player_name_transfer": "player_name",
            "season_transfer": "transfer_season",
            "season_fpl": "previous_season",
        }
    )

    # Sort chronologically.
    result = result.sort_values(
        by=["transfer_season", "player_name"]
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_training_data(data):
    """Save the final merged dataset."""

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_FILE.name}")
    print(f"Rows:  {len(data):,}")
    print(f"Columns: {len(data.columns)}")


# ---------------------------------------------------------------------------
# Display summary
# ---------------------------------------------------------------------------

def show_summary(data):
    """Display a useful summary of the merged training dataset."""

    print("\n" + "=" * 60)
    print("Training dataset summary")
    print("=" * 60)

    if data.empty:
        print("No matched records were found.")
        return

    print("\nTransfers by season:")

    season_counts = (
        data["transfer_season"]
        .value_counts()
        .sort_index()
    )

    for season, count in season_counts.items():
        print(f"  {season}: {count:,}")

    print("\nTransfer fee statistics:")

    print(
        data["fee"]
        .describe()[
            ["count", "mean", "50%", "min", "max"]
        ]
        .to_string()
    )

    print("\nExample records:")

    preview_columns = [
        "player_name",
        "previous_season",
        "transfer_season",
        "minutes",
        "goals_scored",
        "assists",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the complete data-merging pipeline."""

    print("=" * 60)
    print("TransferIQ - Data Merger")
    print("=" * 60)

    # 1. Load cleaned datasets.
    fpl, transfers = load_data()

    # 2. Prepare historical FPL data.
    fpl = prepare_fpl_data(fpl)

    # 3. Prepare transfer data.
    transfers = prepare_transfer_data(transfers)

    # 4. Match previous-season performance to transfers.
    merged = merge_datasets(fpl, transfers)

    # 5. Select final training features.
    training_data = select_training_columns(merged)

    # 6. Save the training dataset.
    save_training_data(training_data)

    # 7. Display summary.
    show_summary(training_data)

    print("\n" + "=" * 60)
    print("Data merging completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
