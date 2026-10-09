"""Tests for TransferIQ data collection configuration."""

from src.data.collectors import (
    HISTORICAL_SEASONS,
    TRANSFER_SEASONS,
)


def test_historical_seasons_are_configured():
    assert "2024-25" in HISTORICAL_SEASONS
    assert "2025-26" in HISTORICAL_SEASONS


def test_transfer_seasons_are_configured():
    assert 2025 in TRANSFER_SEASONS
    assert 2026 in TRANSFER_SEASONS


def test_seasons_are_unique():
    assert len(HISTORICAL_SEASONS) == len(set(HISTORICAL_SEASONS))
    assert len(TRANSFER_SEASONS) == len(set(TRANSFER_SEASONS))