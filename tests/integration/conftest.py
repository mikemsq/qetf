"""Shared fixtures for integration tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
import yaml


@pytest.fixture
def mock_data_access():
    """Create mock data access context for integration tests.

    This mock provides realistic data patterns for testing the
    entire regime system without requiring actual data files.
    """
    ctx = MagicMock()

    # Create realistic SPY price data (uptrend over the year)
    dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
    base_price = 500
    trend = np.linspace(0, 100, len(dates))  # 100 point uptrend
    noise = np.random.randn(len(dates)) * 5
    spy_close = base_price + trend + noise

    spy_prices = pd.DataFrame(
        spy_close,
        index=dates,
        columns=pd.MultiIndex.from_product(
            [["SPY"], ["Close"]], names=["Ticker", "Price"]
        ),
    )
    ctx.prices.read_prices_as_of.return_value = spy_prices

    # Create VIX data (low volatility environment)
    vix_dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
    vix_values = 15 + np.random.randn(len(vix_dates)) * 3
    vix_data = pd.DataFrame({"VIX": vix_values}, index=vix_dates)
    ctx.macro.read_macro_indicator.return_value = vix_data

    return ctx


@pytest.fixture
def high_vol_data_access():
    """Create mock data access for high volatility regime."""
    ctx = MagicMock()

    dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
    spy_close = 500 + np.linspace(0, 100, len(dates))
    spy_prices = pd.DataFrame(
        spy_close,
        index=dates,
        columns=pd.MultiIndex.from_product(
            [["SPY"], ["Close"]], names=["Ticker", "Price"]
        ),
    )
    ctx.prices.read_prices_as_of.return_value = spy_prices

    # High VIX
    vix_dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
    vix_data = pd.DataFrame({"VIX": [30] * len(vix_dates)}, index=vix_dates)
    ctx.macro.read_macro_indicator.return_value = vix_data

    return ctx


@pytest.fixture
def downtrend_data_access():
    """Create mock data access for downtrend regime."""
    ctx = MagicMock()

    dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
    # Price below 200MA
    spy_close = 400 + np.linspace(50, 0, len(dates))  # Declining from 450 to 400
    spy_prices = pd.DataFrame(
        spy_close,
        index=dates,
        columns=pd.MultiIndex.from_product(
            [["SPY"], ["Close"]], names=["Ticker", "Price"]
        ),
    )
    ctx.prices.read_prices_as_of.return_value = spy_prices

    vix_dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
    vix_data = pd.DataFrame({"VIX": [18] * len(vix_dates)}, index=vix_dates)
    ctx.macro.read_macro_indicator.return_value = vix_data

    return ctx


@pytest.fixture
def clean_state_dir(tmp_path):
    """Provide clean state directory for each test."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return state_dir


@pytest.fixture
def sample_regime_mapping(tmp_path):
    """Create sample regime mapping for tests."""
    mapping = {
        "version": "1.0",
        "mapping": {
            "uptrend_low_vol": {
                "strategy": "momentum_acceleration",
                "config_path": "configs/strategies/momentum_acceleration.yaml",
                "rationale": "Clean trends, aggressive momentum",
            },
            "uptrend_high_vol": {
                "strategy": "vol_adjusted_momentum",
                "config_path": "configs/strategies/vol_adjusted_momentum.yaml",
                "rationale": "Trends exist but noisy",
            },
            "downtrend_low_vol": {
                "strategy": "momentum",
                "config_path": "configs/strategies/momentum.yaml",
                "rationale": "Shorter lookback for faster signals",
            },
            "downtrend_high_vol": {
                "strategy": "vol_adjusted_momentum",
                "config_path": "configs/strategies/vol_adjusted_momentum.yaml",
                "rationale": "Defensive positioning",
            },
        },
        "fallback": {
            "strategy": "momentum_acceleration",
            "config_path": "configs/strategies/momentum_acceleration.yaml",
            "rationale": "Best overall performer",
        },
    }

    path = tmp_path / "regime_mapping.yaml"
    with open(path, "w") as f:
        yaml.dump(mapping, f)

    return path


@pytest.fixture
def differentiated_regime_mapping(tmp_path):
    """Create regime mapping with unique strategy per regime for testing."""
    mapping = {
        "version": "1.0",
        "mapping": {
            "uptrend_low_vol": {
                "strategy": "strategy_uptrend_low",
                "config_path": "configs/strategies/strategy_a.yaml",
            },
            "uptrend_high_vol": {
                "strategy": "strategy_uptrend_high",
                "config_path": "configs/strategies/strategy_b.yaml",
            },
            "downtrend_low_vol": {
                "strategy": "strategy_downtrend_low",
                "config_path": "configs/strategies/strategy_c.yaml",
            },
            "downtrend_high_vol": {
                "strategy": "strategy_downtrend_high",
                "config_path": "configs/strategies/strategy_d.yaml",
            },
        },
        "fallback": {
            "strategy": "strategy_fallback",
            "config_path": "configs/strategies/fallback.yaml",
        },
    }

    path = tmp_path / "differentiated_mapping.yaml"
    with open(path, "w") as f:
        yaml.dump(mapping, f)

    return path
