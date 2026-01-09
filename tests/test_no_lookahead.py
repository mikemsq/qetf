"""Tests to verify no lookahead bias in data access and models.

This test suite is CRITICAL for ensuring backtest credibility. Lookahead bias
(using future data in historical decisions) is the #1 cause of failed quant
strategies in production.

These tests use synthetic data with known properties to verify that we NEVER
see future information when making decisions.
"""

import pandas as pd
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.types import Universe


def create_synthetic_prices():
    """Create synthetic price data for testing.

    Creates a simple dataset where prices increment daily:
    - Date 2023-01-01: all prices = 100
    - Date 2023-01-02: all prices = 101
    - Date 2023-01-03: all prices = 102
    - etc.

    This makes it easy to verify: if we see price=105 on 2023-01-03,
    we know we're using future data!

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price) and datetime index
    """
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    tickers = ['TICKER_A', 'TICKER_B', 'TICKER_C']

    # Create price data where each ticker has the same pattern
    # Price = 100 + day_index (day 0 = 100, day 1 = 101, etc.)
    data = {}
    for ticker in tickers:
        ticker_data = pd.DataFrame({
            'Open': [100 + i for i in range(len(dates))],
            'High': [100 + i + 0.5 for i in range(len(dates))],
            'Low': [100 + i - 0.5 for i in range(len(dates))],
            'Close': [100 + i for i in range(len(dates))],
            'Volume': [1000000] * len(dates),
        }, index=dates)
        data[ticker] = ticker_data

    # Combine into MultiIndex format: (date, (ticker, field))
    combined = pd.concat(data, axis=1)
    combined.columns.names = ['Ticker', 'Price']

    return combined


def test_snapshot_store_t1_access():
    """Verify SnapshotDataStore enforces T-1 data access.

    CRITICAL: This test ensures we NEVER see data from the as_of date itself.
    When making decisions on 2023-01-10, we should only see data through
    2023-01-09 (T-1).
    """
    # Create synthetic data
    prices = create_synthetic_prices()

    # Save to temp parquet file
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / 'test_snapshot'
        snapshot_path.mkdir()
        parquet_path = snapshot_path / 'prices.parquet'
        prices.to_parquet(parquet_path)

        # Load with SnapshotDataStore
        store = SnapshotDataStore(snapshot_path)

        # Request data as of 2023-01-10 (day 9, price should be 109)
        # But T-1 means we should only see data through 2023-01-09 (day 8, price=108)
        as_of = pd.Timestamp('2023-01-10')
        data = store.get_close_prices(as_of=as_of)

        # Latest date in data should be 2023-01-09 (day before as_of)
        assert data.index.max() < as_of, "Data includes as_of date (lookahead!)"
        assert data.index.max() == pd.Timestamp('2023-01-09'), "Latest date should be T-1"

        # Latest price should be 108 (day 8), NOT 109 (day 9)
        # This proves we're not seeing data from as_of date
        latest_prices = data.loc[data.index.max()]
        assert all(latest_prices == 108), f"Expected price 108, got {latest_prices.values}"


def test_strict_inequality_t1():
    """Verify we use < not <= for T-1 filtering.

    CRITICAL: Using <= would include the as_of date itself, which would be
    lookahead bias. We must use strict inequality (<).
    """
    prices = create_synthetic_prices()

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / 'test_snapshot'
        snapshot_path.mkdir()
        parquet_path = snapshot_path / 'prices.parquet'
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        # Request data as of 2023-01-10
        as_of = pd.Timestamp('2023-01-10')
        data = store.get_close_prices(as_of=as_of)

        # Should NOT include as_of date (strict <)
        assert as_of not in data.index, "as_of date included in data (lookahead!)"

        # Should include day before
        assert pd.Timestamp('2023-01-09') in data.index, "T-1 date should be included"


def test_momentum_alpha_no_lookahead():
    """Verify MomentumAlpha doesn't use future data.

    CRITICAL: This test ensures the momentum calculation only uses historical
    data. The momentum should be calculated using prices from T-1 and earlier,
    never from the decision date itself.
    """
    # Create synthetic data
    prices = create_synthetic_prices()

    # Save to temp snapshot
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / 'test_snapshot'
        snapshot_path.mkdir()
        parquet_path = snapshot_path / 'prices.parquet'
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        # Create universe
        universe = Universe(
            as_of=pd.Timestamp('2023-01-10'),
            tickers=('TICKER_A', 'TICKER_B', 'TICKER_C')
        )

        # Calculate momentum as of 2023-01-10
        # Set min_periods low for this test
        alpha = MomentumAlpha(lookback_days=5, min_periods=5)
        scores = alpha.score(
            as_of=pd.Timestamp('2023-01-10'),
            universe=universe,
            features=None,
            store=store
        )

        # Momentum should be calculated using:
        # - Last 5 days of available data (ending at T-1):
        #   2023-01-05 (104), 2023-01-06 (105), 2023-01-07 (106),
        #   2023-01-08 (107), 2023-01-09 (108)
        # - Current price (last in window): 2023-01-09 = 108
        # - Lookback price (first in window): 2023-01-05 = 104
        # - Return = (108 / 104) - 1 = 0.038462

        # All tickers should have same momentum (same price pattern)
        expected_return = (108.0 / 104.0) - 1.0
        for ticker in universe.tickers:
            actual = scores.scores[ticker]
            assert actual == pytest.approx(expected_return, rel=1e-4), \
                f"{ticker}: expected {expected_return:.6f}, got {actual:.6f}"

        # Verify we're not accidentally using day 9 price (109)
        # That would give momentum = (109 / 105) - 1 = 0.038095
        wrong_return = (109.0 / 105.0) - 1.0
        for ticker in universe.tickers:
            assert abs(scores.scores[ticker] - wrong_return) > 1e-5, \
                f"{ticker}: appears to be using future data (price 109 instead of 108)"


def test_lookback_window_no_lookahead():
    """Verify lookback windows don't include future data.

    CRITICAL: When requesting a lookback window, the window should end at T-1,
    not at T. This test verifies the window boundaries are correct.
    """
    prices = create_synthetic_prices()

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / 'test_snapshot'
        snapshot_path.mkdir()
        parquet_path = snapshot_path / 'prices.parquet'
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        # Request 10 days of data as of 2023-01-20
        as_of = pd.Timestamp('2023-01-20')
        data = store.read_prices(as_of=as_of, lookback_days=10)

        # Should have 10 days of data
        assert len(data) == 10, f"Expected 10 days, got {len(data)}"

        # Latest date should be 2023-01-19 (T-1)
        assert data.index.max() == pd.Timestamp('2023-01-19'), \
            f"Latest date should be T-1, got {data.index.max()}"

        # Earliest date should be 2023-01-10 (T-1 minus 9 more days)
        assert data.index.min() == pd.Timestamp('2023-01-10'), \
            f"Earliest date should be 10 days before T, got {data.index.min()}"

        # Should NOT include as_of date
        assert as_of not in data.index, "Lookback window includes as_of date (lookahead!)"


def test_momentum_with_different_lookbacks():
    """Test momentum calculation with various lookback periods.

    This verifies that different lookback periods all respect the T-1 boundary
    and produce correct results based on the synthetic data.
    """
    prices = create_synthetic_prices()

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / 'test_snapshot'
        snapshot_path.mkdir()
        parquet_path = snapshot_path / 'prices.parquet'
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        universe = Universe(
            as_of=pd.Timestamp('2023-02-01'),  # Day 31, T-1 price = 130
            tickers=('TICKER_A',)
        )

        # Test different lookback periods
        # For 2023-02-01, we see data up to 2023-01-31 (day 30, price 130)
        test_cases = [
            (5, 130.0, 126.0),    # 5-day: last 5 days ending at 130
            (10, 130.0, 121.0),   # 10-day: last 10 days ending at 130
            (20, 130.0, 111.0),   # 20-day: last 20 days ending at 130
        ]

        for lookback, current_price, old_price in test_cases:
            # Set min_periods equal to lookback for this test
            alpha = MomentumAlpha(lookback_days=lookback, min_periods=lookback)
            scores = alpha.score(
                as_of=pd.Timestamp('2023-02-01'),
                universe=universe,
                features=None,
                store=store
            )

            expected = (current_price / old_price) - 1.0
            actual = scores.scores['TICKER_A']
            assert actual == pytest.approx(expected, rel=1e-4), \
                f"Lookback {lookback}: expected {expected:.6f}, got {actual:.6f}"


def test_edge_case_insufficient_data():
    """Test handling when there's insufficient historical data.

    This ensures we handle edge cases gracefully without introducing lookahead.
    """
    # Create very short history
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    tickers = ['TICKER_A']

    data = {}
    for ticker in tickers:
        ticker_data = pd.DataFrame({
            'Open': [100 + i for i in range(len(dates))],
            'High': [100 + i + 0.5 for i in range(len(dates))],
            'Low': [100 + i - 0.5 for i in range(len(dates))],
            'Close': [100 + i for i in range(len(dates))],
            'Volume': [1000000] * len(dates),
        }, index=dates)
        data[ticker] = ticker_data

    prices = pd.concat(data, axis=1)
    prices.columns.names = ['Ticker', 'Price']

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / 'test_snapshot'
        snapshot_path.mkdir()
        parquet_path = snapshot_path / 'prices.parquet'
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        universe = Universe(
            as_of=pd.Timestamp('2023-01-05'),
            tickers=('TICKER_A',)
        )

        # Request momentum with lookback longer than available history
        # Should handle gracefully (return NaN or use available data)
        alpha = MomentumAlpha(lookback_days=252, min_periods=200)
        scores = alpha.score(
            as_of=pd.Timestamp('2023-01-05'),
            universe=universe,
            features=None,
            store=store
        )

        # Should return NaN due to insufficient data
        assert pd.isna(scores.scores['TICKER_A']), \
            "Should return NaN when insufficient data available"


def test_price_progression_verification():
    """Verify synthetic data has expected properties.

    This is a sanity check to ensure our synthetic data is set up correctly
    for the other tests to be meaningful.
    """
    prices = create_synthetic_prices()

    # Check date range
    assert len(prices) == 300, f"Expected 300 days, got {len(prices)}"

    # Check all tickers present
    tickers = prices.columns.get_level_values('Ticker').unique()
    assert len(tickers) == 3, f"Expected 3 tickers, got {len(tickers)}"

    # Check price progression for each ticker
    for ticker in ['TICKER_A', 'TICKER_B', 'TICKER_C']:
        close_prices = prices[(ticker, 'Close')]

        # First day should be 100
        assert close_prices.iloc[0] == 100, \
            f"{ticker}: First price should be 100, got {close_prices.iloc[0]}"

        # Last day should be 399 (100 + 299)
        assert close_prices.iloc[-1] == 399, \
            f"{ticker}: Last price should be 399, got {close_prices.iloc[-1]}"

        # Prices should increment by 1 each day
        diffs = close_prices.diff().dropna()
        assert all(diffs == 1.0), \
            f"{ticker}: Prices should increment by 1 daily"


def test_no_lookahead_with_multiple_dates():
    """Test T-1 enforcement across multiple decision dates.

    This verifies that the T-1 rule is consistently applied regardless of
    which date we're making decisions on.
    """
    prices = create_synthetic_prices()

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / 'test_snapshot'
        snapshot_path.mkdir()
        parquet_path = snapshot_path / 'prices.parquet'
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        # Test multiple decision dates
        test_dates = [
            pd.Timestamp('2023-01-10'),
            pd.Timestamp('2023-02-15'),
            pd.Timestamp('2023-06-01'),
        ]

        for as_of in test_dates:
            data = store.get_close_prices(as_of=as_of)

            # Should never include as_of date
            assert as_of not in data.index, \
                f"Data for {as_of} includes the as_of date (lookahead!)"

            # Latest date should be exactly T-1
            expected_latest = as_of - pd.Timedelta(days=1)
            assert data.index.max() == expected_latest, \
                f"For as_of={as_of}, expected latest={expected_latest}, got {data.index.max()}"

            # Verify price matches expected value
            day_index = (expected_latest - pd.Timestamp('2023-01-01')).days
            expected_price = 100 + day_index
            actual_price = data.loc[expected_latest, 'TICKER_A']
            assert actual_price == expected_price, \
                f"For {as_of}: expected price {expected_price}, got {actual_price}"
