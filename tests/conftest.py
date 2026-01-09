"""Shared test fixtures for QuantETF tests.

This module provides pytest fixtures that can be used across all test files.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_etf_tickers():
    """Return list of sample ETF tickers for testing.
    
    Returns:
        List of commonly used ETF ticker symbols
    """
    return ["SPY", "QQQ", "IWM", "EFA", "AGG"]


@pytest.fixture
def date_range():
    """Return standard test date range (5 years of history).
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    end = datetime.now()
    start = end - timedelta(days=365*5)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@pytest.fixture
def short_date_range():
    """Return short test date range (1 year of history).
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    return "2020-01-01", "2020-12-31"


@pytest.fixture
def sample_price_data():
    """Return sample ETF price data for testing.

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price_Field) for single ticker 'SPY'
    """
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
    n = len(dates)

    # Create realistic-looking price data
    base_price = 100
    returns = np.arange(n) * 0.1  # Slight upward trend
    noise = np.arange(n) % 5 - 2  # Add some noise

    close_prices = base_price + returns + noise

    # Create simple DataFrame first
    simple_df = pd.DataFrame({
        "Open": close_prices - 0.5,
        "High": close_prices + 1.0,
        "Low": close_prices - 1.0,
        "Close": close_prices,
        "Volume": 1000000,
    }, index=dates)

    # Convert to MultiIndex format (Ticker, Price_Field)
    simple_df.columns = pd.MultiIndex.from_product(
        [['SPY'], simple_df.columns],
        names=['Ticker', 'Price']
    )

    return simple_df


@pytest.fixture
def sample_price_data_multi_ticker():
    """Return sample price data for multiple tickers.

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price_Field)
    """
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
    n = len(dates)

    tickers = ["SPY", "QQQ"]
    data_dict = {}

    for ticker in tickers:
        base_price = 100 if ticker == "SPY" else 200
        returns = np.arange(n) * 0.1
        noise = np.arange(n) % 5 - 2

        close_prices = base_price + returns + noise

        data_dict[ticker] = pd.DataFrame({
            "Open": close_prices - 0.5,
            "High": close_prices + 1.0,
            "Low": close_prices - 1.0,
            "Close": close_prices,
            "Volume": 1000000,
        }, index=dates)

    # Combine into MultiIndex DataFrame with names
    combined = pd.concat(data_dict, axis=1)
    combined.columns.names = ['Ticker', 'Price']
    return combined
