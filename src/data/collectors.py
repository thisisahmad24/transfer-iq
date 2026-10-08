"""
TransferIQ - Data Collector

Downloads the raw football data used by the project.

Sources:
- Fantasy Premier League API
- Vaastav Fantasy Premier League historical dataset
- Transfermarkt data
- football-data.co.uk

All downloaded files are saved in:
    data/raw/
"""

from pathlib import Path
import json
import time

import pandas as pd
import requests


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Data sources
# ---------------------------------------------------------

FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

HISTORICAL_FPL_URL = (
    "https://raw.githubusercontent.com/"
    "vaastav/Fantasy-Premier-League/master/data"
)

TRANSFER_URL = (
    "https://raw.githubusercontent.com/"
    "eordo/transfermarkt-data/master/premier_league"
)

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281"


# Historical FPL seasons used for model training.
#
# 2025-26 is required because:
#
#   2025-26 performance -> 2026 transfer market
#
HISTORICAL_SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


# Transfermarkt seasons.
#
# The 2026 transfer dataset is our latest historical
# transfer market and will be paired with 2025-26 FPL data.
TRANSFER_SEASONS = [
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]


# football-data.co.uk season codes.
MATCH_SEASONS = [
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]


# ---------------------------------------------------------
# HTTP session
# ---------------------------------------------------------

session = requests.Session()

session.headers.update(
    {
        "User-Agent": "TransferIQ/1.0"
    }
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def download_file(url, output_path):
    """Download a file and save it locally."""

    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()

        output_path.write_bytes(response.content)

        return True

    except requests.RequestException as error:
        print(f"  Failed: {error}")
        return False


def download_csv(url, output_path):
    """Download a CSV file and make sure it can be read."""

    if not download_file(url, output_path):
        return False

    try:
        data = pd.read_csv(output_path)

        print(
            f"  Saved: {output_path.name} "
            f"({len(data):,} rows)"
        )

        return True

    except Exception as error:
        print(f"  Invalid CSV: {error}")

        # Remove incomplete/invalid file.
        if output_path.exists():
            output_path.unlink()

        return False


# ---------------------------------------------------------
# Current FPL data
# ---------------------------------------------------------

def collect_current_fpl():
    """Download the current FPL player dataset."""

    print("\n[1/4] Collecting current FPL data...")

    output_path = RAW_DATA_DIR / "fpl_bootstrap.json"

    try:
        response = session.get(FPL_API_URL, timeout=60)
        response.raise_for_status()

        data = response.json()

        output_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

        players = data.get("elements", [])

        print(f"  Saved {len(players):,} current players.")

    except requests.RequestException as error:
        print(f"  Failed: {error}")


# ---------------------------------------------------------
# Historical FPL data
# ---------------------------------------------------------

def collect_historical_fpl():
    """Download historical FPL player statistics."""

    print("\n[2/4] Collecting historical FPL data...")

    total_players = 0

    for season in HISTORICAL_SEASONS:
        url = (
            f"{HISTORICAL_FPL_URL}/"
            f"{season}/cleaned_players.csv"
        )

        output_path = (
            RAW_DATA_DIR / f"fpl_{season}.csv"
        )

        print(f"  {season}...")

        if download_csv(url, output_path):
            data = pd.read_csv(output_path)
            total_players += len(data)

        # Small delay between GitHub requests.
        time.sleep(0.5)

    print(
        f"  Historical records collected: "
        f"{total_players:,}"
    )


# ---------------------------------------------------------
# Transfer data
# ---------------------------------------------------------

def collect_transfer_data():
    """Download Premier League transfer records."""

    print("\n[3/4] Collecting transfer data...")

    total_transfers = 0

    for season in TRANSFER_SEASONS:
        url = f"{TRANSFER_URL}/{season}.csv"

        output_path = (
            RAW_DATA_DIR / f"transfers_{season}.csv"
        )

        print(f"  {season}...")

        if download_csv(url, output_path):
            data = pd.read_csv(output_path)
            total_transfers += len(data)

        time.sleep(0.5)

    print(
        f"  Transfer records collected: "
        f"{total_transfers:,}"
    )


# ---------------------------------------------------------
# Premier League match data
# ---------------------------------------------------------

def collect_match_data():
    """Download Premier League match results."""

    print("\n[4/4] Collecting match data...")

    total_matches = 0

    for season in MATCH_SEASONS:
        url = (
            f"{FOOTBALL_DATA_URL}/"
            f"{season}/E0.csv"
        )

        output_path = (
            RAW_DATA_DIR
            / f"premier_league_{season}.csv"
        )

        print(f"  {season}...")

        if download_csv(url, output_path):
            data = pd.read_csv(output_path)
            total_matches += len(data)

        time.sleep(0.5)

    print(
        f"  Match records collected: "
        f"{total_matches:,}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    """Run the complete data collection process."""

    print("=" * 60)
    print("TransferIQ - Data Collection")
    print("=" * 60)

    collect_current_fpl()
    collect_historical_fpl()
    collect_transfer_data()
    collect_match_data()

    print("\n" + "=" * 60)
    print("Data collection completed.")
    print(f"Raw data location: {RAW_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
