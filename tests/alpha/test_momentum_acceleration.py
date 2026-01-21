"""Tests for Momentum Acceleration alpha model."""

import pandas as pd
import numpy as np
import pytest
import tempfile
from pathlib import Path

from quantetf.alpha.momentum_acceleration import MomentumAccelerationAlpha
from quantetf.data.access import DataAccessFactory
from quantetf.types import Universe


def create_test_prices(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    price_paths: dict[str, list[float]]
) -> pd.DataFrame:
    """Create synthetic price data for testing.

    Args:
        tickers: List of ticker symbols
        dates: DatetimeIndex for the data
        price_paths: Dict mapping ticker to list of close prices

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price) ready for SnapshotDataStore
    """
    data = {}
    for ticker in tickers:
        prices = price_paths[ticker]
        ticker_data = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(prices),
        }, index=dates)
        data[ticker] = ticker_data

    combined = pd.concat(data, axis=1)
    combined.columns.names = ['Ticker', 'Price']
    return combined


class TestMomentumAccelerationBasics:
    """Test basic functionality of momentum acceleration."""

    def test_parameter_validation(self):
        """Test that parameter validation works correctly."""
        # Valid parameters
        alpha = MomentumAccelerationAlpha(
            short_lookback_days=63,
            long_lookback_days=252,
            min_periods=200
        )
        assert alpha.short_lookback_days == 63
        assert alpha.long_lookback_days == 252

        # Invalid: short >= long
        with pytest.raises(ValueError, match="short_lookback.*must be <"):
            MomentumAccelerationAlpha(
                short_lookback_days=252,
                long_lookback_days=63
            )

        # Invalid: short == long
        with pytest.raises(ValueError, match="short_lookback.*must be <"):
            MomentumAccelerationAlpha(
                short_lookback_days=100,
                long_lookback_days=100
            )

        # Invalid: short < 20
        with pytest.raises(ValueError, match="short_lookback must be >= 20"):
            MomentumAccelerationAlpha(short_lookback_days=10, long_lookback_days=100)

        # Invalid: min_periods < short_lookback
        with pytest.raises(ValueError, match="min_periods.*must be >="):
            MomentumAccelerationAlpha(
                short_lookback_days=100,
                long_lookback_days=200,
                min_periods=50
            )

    def test_positive_acceleration_scenario(self):
        """Test ticker with accelerating momentum (positive score)."""
        # Create 300 days of data
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # TICKER_A: Steady growth for 12M, then rapid growth last 3M
        # First 237 days (252-63): +10% total (237 days to get from 100 to 110)
        # Last 63 days: +20% (from 110 to 132)
        prices_a = []
        for i in range(237):
            prices_a.append(100 + (10 * i / 237))  # Gradual to 110
        for i in range(63):
            prices_a.append(110 + (22 * i / 63))  # Faster to 132

        prices = create_test_prices(
            tickers=['TICKER_A'],
            dates=dates,
            price_paths={'TICKER_A': prices_a}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            parquet_path = snapshot_path / 'data.parquet'
            prices.to_parquet(parquet_path)

            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": str(parquet_path)},
                enable_caching=False
            )
            alpha = MomentumAccelerationAlpha(
                short_lookback_days=63,
                long_lookback_days=252,
                min_periods=200
            )

            universe = Universe(as_of=dates[-1], tickers=('TICKER_A',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Check that acceleration is positive
            assert scores.scores['TICKER_A'] > 0, "Expected positive acceleration"

    def test_positive_acceleration_corrected(self):
        """Test ticker with true accelerating momentum."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Build price path that achieves positive acceleration
        # Need: (price_end / price_252d_ago) - 1 = 0.20
        # Need: (price_end / price_63d_ago) - 1 = 0.25
        # price_end = 120, price_252d_ago = 100, price_63d_ago = 96

        prices_a = []
        # First 189 days: 100 down to 96
        for i in range(189):
            prices_a.append(100 - (4 * i / 189))
        # Next 48 days: 96 stable (to pad to 237)
        for i in range(48):
            prices_a.append(96)
        # Last 63 days: 96 up to 120
        for i in range(63):
            prices_a.append(96 + (24 * i / 62))

        prices = create_test_prices(
            tickers=['TICKER_A'],
            dates=dates,
            price_paths={'TICKER_A': prices_a}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            parquet_path = snapshot_path / 'data.parquet'
            prices.to_parquet(parquet_path)

            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": str(parquet_path)},
                enable_caching=False
            )
            alpha = MomentumAccelerationAlpha(
                short_lookback_days=63,
                long_lookback_days=252,
                min_periods=200
            )

            universe = Universe(as_of=dates[-1], tickers=('TICKER_A',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Should have positive acceleration
            assert scores.scores['TICKER_A'] > 0, "Expected positive acceleration"
            # Should be approximately 0.05 (5%)
            assert abs(scores.scores['TICKER_A'] - 0.05) < 0.02

    def test_negative_acceleration_scenario(self):
        """Test ticker with decelerating momentum (negative score)."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # TICKER_B: Strong 12M performance (+30%), but weak recent 3M (+5%)
        prices_b = []
        # First 189 days: 100 to 123.8
        for i in range(189):
            prices_b.append(100 + (23.8 * i / 188))
        # Pad to 237
        for i in range(48):
            prices_b.append(123.8)
        # Last 63 days: 123.8 to 130
        for i in range(63):
            prices_b.append(123.8 + (6.2 * i / 62))

        prices = create_test_prices(
            tickers=['TICKER_B'],
            dates=dates,
            price_paths={'TICKER_B': prices_b}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            parquet_path = snapshot_path / 'data.parquet'
            prices.to_parquet(parquet_path)

            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": str(parquet_path)},
                enable_caching=False
            )
            alpha = MomentumAccelerationAlpha(short_lookback_days=63, long_lookback_days=252)

            universe = Universe(as_of=dates[-1], tickers=('TICKER_B',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Should have negative acceleration (decelerating)
            assert scores.scores['TICKER_B'] < 0
            # Acceleration should be approximately 5% - 30% = -25%
            assert scores.scores['TICKER_B'] < -0.20

    def test_insufficient_data_returns_nan(self):
        """Test that insufficient data returns NaN."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')  # Only 100 days

        prices = create_test_prices(
            tickers=['TICKER_SHORT'],
            dates=dates,
            price_paths={'TICKER_SHORT': [100] * 100}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            parquet_path = snapshot_path / 'data.parquet'
            prices.to_parquet(parquet_path)

            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": str(parquet_path)},
                enable_caching=False
            )
            alpha = MomentumAccelerationAlpha(min_periods=200)  # Require 200 days

            universe = Universe(as_of=dates[-1], tickers=('TICKER_SHORT',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Should return NaN for insufficient data
            assert pd.isna(scores.scores['TICKER_SHORT'])


class TestMomentumAccelerationEdgeCases:
    """Test edge cases and error handling."""

    def test_no_lookahead_bias(self):
        """Verify that scores don't use data from as_of date."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Create price that jumps 50% on the last day
        prices_normal = [100] * 299
        prices_with_jump = prices_normal + [150]

        prices = create_test_prices(
            tickers=['TICKER_A'],
            dates=dates,
            price_paths={'TICKER_A': prices_with_jump}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            parquet_path = snapshot_path / 'data.parquet'
            prices.to_parquet(parquet_path)

            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": str(parquet_path)},
                enable_caching=False
            )
            alpha = MomentumAccelerationAlpha(
                short_lookback_days=63,
                long_lookback_days=252,
                min_periods=200
            )

            universe = Universe(as_of=dates[-1], tickers=('TICKER_A',))

            # Score as of the jump date
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Score should be ~0 (flat prices), NOT influenced by the jump
            assert abs(scores.scores['TICKER_A']) < 0.01, \
                "Score should not be influenced by as_of date price jump"

    def test_missing_ticker_returns_nan(self):
        """Test that missing ticker returns NaN."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        prices = create_test_prices(
            tickers=['TICKER_EXISTS'],
            dates=dates,
            price_paths={'TICKER_EXISTS': [100] * 300}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            parquet_path = snapshot_path / 'data.parquet'
            prices.to_parquet(parquet_path)

            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": str(parquet_path)},
                enable_caching=False
            )
            alpha = MomentumAccelerationAlpha()

            # Request score for ticker not in data
            universe = Universe(as_of=dates[-1], tickers=('TICKER_EXISTS', 'TICKER_MISSING'))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Missing ticker should have NaN
            assert pd.isna(scores.scores['TICKER_MISSING'])
            # Existing ticker should have a value
            assert pd.notna(scores.scores['TICKER_EXISTS'])

    def test_wrong_store_type_raises_error(self):
        """Test that using wrong data_access type raises TypeError."""
        alpha = MomentumAccelerationAlpha()
        as_of = pd.Timestamp('2023-01-01')
        universe = Universe(as_of=as_of, tickers=('TICKER_A',))

        # Use None as data_access (wrong type)
        with pytest.raises((TypeError, AttributeError)):
            alpha.score(
                as_of=as_of,
                universe=universe,
                features=None,
                data_access=None  # Wrong type!
            )


class TestMomentumAccelerationRanking:
    """Test that acceleration correctly ranks tickers."""

    def test_ranking_multiple_tickers(self):
        """Test that tickers are correctly ranked by acceleration."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Create three tickers with different acceleration patterns
        # TICKER_ACCEL: Accelerating (positive)
        # TICKER_STEADY: Steady momentum (near zero)
        # TICKER_DECEL: Decelerating (negative)

        # TICKER_ACCEL: 12M=+10%, 3M=+20%, accel=+10%
        prices_accel = []
        for i in range(189):
            prices_accel.append(100)
        for i in range(48):
            prices_accel.append(100)
        for i in range(63):
            prices_accel.append(100 + (20 * i / 62))

        # TICKER_STEADY: 12M=+10%, 3M=+10%, accel=0%
        prices_steady = []
        for i in range(300):
            prices_steady.append(100 + (10 * i / 299))

        # TICKER_DECEL: 12M=+20%, 3M=+5%, accel=-15%
        prices_decel = []
        for i in range(237):
            prices_decel.append(100 + (20 * i / 236))
        for i in range(63):
            prices_decel.append(120 + (5 * i / 62) - 15)  # Slight pullback

        prices = create_test_prices(
            tickers=['TICKER_ACCEL', 'TICKER_STEADY', 'TICKER_DECEL'],
            dates=dates,
            price_paths={
                'TICKER_ACCEL': prices_accel,
                'TICKER_STEADY': prices_steady,
                'TICKER_DECEL': prices_decel
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            parquet_path = snapshot_path / 'data.parquet'
            prices.to_parquet(parquet_path)

            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": str(parquet_path)},
                enable_caching=False
            )
            alpha = MomentumAccelerationAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_ACCEL', 'TICKER_STEADY', 'TICKER_DECEL'))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Verify ranking
            assert scores.scores['TICKER_ACCEL'] > scores.scores['TICKER_STEADY']
            assert scores.scores['TICKER_STEADY'] > scores.scores['TICKER_DECEL']

            # Verify signs
            assert scores.scores['TICKER_ACCEL'] > 0, "Accelerating should be positive"
            assert abs(scores.scores['TICKER_STEADY']) < 0.05, "Steady should be near zero"
            assert scores.scores['TICKER_DECEL'] < 0, "Decelerating should be negative"
