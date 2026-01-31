"""Tests for TrendFilteredMomentum alpha model.

These tests verify the trend-filtered momentum strategy correctly:
1. Detects bullish/bearish regimes based on trend ticker vs MA
2. Uses momentum scores in bullish regime
3. Uses defensive scores in bearish regime
4. Handles edge cases gracefully
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantetf.alpha.trend_filtered_momentum import TrendFilteredMomentum
from quantetf.data.access import DataAccessFactory
from quantetf.types import Universe


def create_test_data_access(prices_df: pd.DataFrame, tmp_path: Path):
    """Create a DataAccessContext from price data.

    Args:
        prices_df: DataFrame with MultiIndex columns (Ticker, Price)
        tmp_path: Temporary directory for snapshot

    Returns:
        DataAccessContext for testing
    """
    snapshot_path = tmp_path / 'test_snapshot'
    snapshot_path.mkdir(exist_ok=True)
    parquet_path = snapshot_path / 'data.parquet'
    prices_df.to_parquet(parquet_path)

    return DataAccessFactory.create_context(
        config={"snapshot_path": str(parquet_path)},
        enable_caching=False
    )


def create_bullish_data(num_days: int = 300) -> pd.DataFrame:
    """Create price data where SPY is trending up (above MA200).

    Returns DataFrame with SPY, QQQ, AGG, TLT all with known patterns.
    """
    dates = pd.date_range('2020-01-01', periods=num_days, freq='D')
    np.random.seed(42)

    # SPY trending up strongly - will be above MA200
    spy_prices = 100 * np.cumprod(1 + np.random.randn(num_days) * 0.01 + 0.001)

    # QQQ trending up even more (higher momentum)
    qqq_prices = 100 * np.cumprod(1 + np.random.randn(num_days) * 0.015 + 0.0015)

    # Defensive assets - relatively flat
    agg_prices = 100 * np.cumprod(1 + np.random.randn(num_days) * 0.003 + 0.0001)
    tlt_prices = 100 * np.cumprod(1 + np.random.randn(num_days) * 0.008 + 0.0002)

    data = {}
    for ticker, prices in [('SPY', spy_prices), ('QQQ', qqq_prices),
                           ('AGG', agg_prices), ('TLT', tlt_prices)]:
        data[ticker] = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': [1000000] * num_days,
        }, index=dates)

    combined = pd.concat(data, axis=1)
    combined.columns.names = ['Ticker', 'Price']
    return combined


def create_bearish_data(num_days: int = 300) -> pd.DataFrame:
    """Create price data where SPY is trending down (below MA200).

    SPY starts high, stays flat for 200 days, then drops.
    """
    dates = pd.date_range('2020-01-01', periods=num_days, freq='D')

    # SPY: flat for 200 days then drops - will be below MA200
    spy_prices = np.concatenate([
        100 * np.ones(200),  # Flat for 200 days
        100 * np.linspace(1, 0.75, 100)  # Drop 25% over 100 days
    ])

    # QQQ follows similar pattern
    qqq_prices = spy_prices * 1.1

    # Defensive assets stay flat
    agg_prices = 100 * np.ones(num_days)
    tlt_prices = 100 * np.ones(num_days)

    data = {}
    for ticker, prices in [('SPY', spy_prices), ('QQQ', qqq_prices),
                           ('AGG', agg_prices), ('TLT', tlt_prices)]:
        data[ticker] = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': [1000000] * num_days,
        }, index=dates)

    combined = pd.concat(data, axis=1)
    combined.columns.names = ['Ticker', 'Price']
    return combined


class TestTrendFilteredMomentum:
    """Tests for TrendFilteredMomentum alpha model."""

    def test_bullish_regime_uses_momentum(self, tmp_path):
        """When SPY > MA200, should use momentum scores."""
        prices = create_bullish_data()
        data_access = create_test_data_access(prices, tmp_path)

        model = TrendFilteredMomentum(
            momentum_lookback=60,
            ma_period=50,  # Short MA to ensure bullish
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-01'),
            tickers=('SPY', 'QQQ', 'AGG', 'TLT')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-10-01'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # Should have scores for all tickers
        assert len(scores.scores) == 4
        # Momentum scores should vary (not all the same)
        assert scores.scores.nunique() > 1

    def test_bearish_regime_uses_defensive(self, tmp_path):
        """When SPY < MA200, should score defensive assets."""
        prices = create_bearish_data()
        data_access = create_test_data_access(prices, tmp_path)

        model = TrendFilteredMomentum(
            ma_period=200,
            defensive_tickers=['AGG', 'TLT'],
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-20'),
            tickers=('SPY', 'QQQ', 'AGG', 'TLT')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-10-20'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # Defensive tickers should have high scores
        assert scores.scores['AGG'] == 1.0
        assert scores.scores['TLT'] == 1.0
        # Non-defensive should have zero
        assert scores.scores['SPY'] == 0.0
        assert scores.scores['QQQ'] == 0.0

    def test_regime_detection_bullish(self, tmp_path):
        """Test regime detection returns BULLISH correctly."""
        # Create deterministic bullish data where SPY is clearly above MA
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # SPY steadily trending up (guaranteed to be above MA50)
        spy_prices = 100 + np.arange(300) * 0.5  # Goes from 100 to 249.5

        data = {}
        data['SPY'] = pd.DataFrame({
            'Open': spy_prices,
            'High': spy_prices,
            'Low': spy_prices,
            'Close': spy_prices,
            'Volume': [1000000] * 300,
        }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        data_access = create_test_data_access(combined, tmp_path)

        model = TrendFilteredMomentum(ma_period=50)

        regime = model.get_regime(
            data_access=data_access,
            as_of=pd.Timestamp('2020-10-01')
        )

        assert regime == "BULLISH"

    def test_regime_detection_defensive(self, tmp_path):
        """Test regime detection returns DEFENSIVE correctly."""
        prices = create_bearish_data()
        data_access = create_test_data_access(prices, tmp_path)

        model = TrendFilteredMomentum(ma_period=200)

        regime = model.get_regime(
            data_access=data_access,
            as_of=pd.Timestamp('2020-10-20')
        )

        assert regime == "DEFENSIVE"

    def test_insufficient_data_returns_nan(self, tmp_path):
        """Should return NaN when insufficient data for momentum."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')

        data = {}
        for ticker in ['SPY', 'QQQ']:
            data[ticker] = pd.DataFrame({
                'Open': np.random.randn(50).cumsum() + 100,
                'High': np.random.randn(50).cumsum() + 101,
                'Low': np.random.randn(50).cumsum() + 99,
                'Close': np.random.randn(50).cumsum() + 100,
                'Volume': [1000000] * 50,
            }, index=dates)

        prices = pd.concat(data, axis=1)
        prices.columns.names = ['Ticker', 'Price']

        data_access = create_test_data_access(prices, tmp_path)

        model = TrendFilteredMomentum(min_periods=200)

        universe = Universe(
            as_of=pd.Timestamp('2020-02-15'),
            tickers=('SPY', 'QQQ')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-02-15'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # Should return NaN due to insufficient data
        assert pd.isna(scores.scores['QQQ'])

    def test_missing_trend_ticker_defaults_to_bullish(self, tmp_path):
        """Should default to bullish if trend ticker is missing."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Only QQQ, no SPY - create steadily rising prices
        qqq_prices = 100 + np.arange(300) * 0.3  # Steady uptrend

        data = {}
        data['QQQ'] = pd.DataFrame({
            'Open': qqq_prices,
            'High': qqq_prices,
            'Low': qqq_prices,
            'Close': qqq_prices,
            'Volume': [1000000] * 300,
        }, index=dates)

        prices = pd.concat(data, axis=1)
        prices.columns.names = ['Ticker', 'Price']

        data_access = create_test_data_access(prices, tmp_path)

        model = TrendFilteredMomentum(
            trend_ticker='SPY',  # SPY not in data
            min_periods=50,
            momentum_lookback=60,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-01'),
            tickers=('QQQ',)
        )

        # Should not raise, should use momentum (bullish default)
        scores = model.score(
            as_of=pd.Timestamp('2020-10-01'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # Should have a score
        assert len(scores.scores) > 0
        # With missing trend ticker, defaults to bullish, so we should get momentum scores
        # With steadily rising prices over 60 days, we should have positive momentum
        assert not pd.isna(scores.scores['QQQ'])
        assert scores.scores['QQQ'] > 0  # Should be positive (price went up)

    def test_custom_defensive_tickers(self, tmp_path):
        """Test that custom defensive tickers are respected."""
        prices = create_bearish_data()
        data_access = create_test_data_access(prices, tmp_path)

        # Only AGG as defensive (not TLT)
        model = TrendFilteredMomentum(
            ma_period=200,
            defensive_tickers=['AGG'],
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-20'),
            tickers=('SPY', 'QQQ', 'AGG', 'TLT')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-10-20'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # Only AGG should have high score
        assert scores.scores['AGG'] == 1.0
        # TLT should be zero (not in defensive list)
        assert scores.scores['TLT'] == 0.0

    def test_momentum_calculation_correctness(self, tmp_path):
        """Verify momentum is calculated correctly in bullish regime."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Create predictable price data
        # Ticker A: +20% over last 60 days
        # Ticker B: +10% over last 60 days
        spy_prices = np.concatenate([
            100 * np.ones(240),
            100 * np.linspace(1, 1.3, 60)  # Trending up to stay above MA
        ])

        ticker_a_prices = np.concatenate([
            100 * np.ones(240),
            100 * np.linspace(1, 1.2, 60)  # +20%
        ])

        ticker_b_prices = np.concatenate([
            100 * np.ones(240),
            100 * np.linspace(1, 1.1, 60)  # +10%
        ])

        data = {}
        for ticker, prices in [('SPY', spy_prices), ('A', ticker_a_prices), ('B', ticker_b_prices)]:
            data[ticker] = pd.DataFrame({
                'Open': prices,
                'High': prices,
                'Low': prices,
                'Close': prices,
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        data_access = create_test_data_access(combined, tmp_path)

        model = TrendFilteredMomentum(
            momentum_lookback=60,
            ma_period=50,
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-27'),  # After 300 days
            tickers=('A', 'B')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-10-27'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # A should have higher score than B
        assert scores.scores['A'] > scores.scores['B']
        # A should be close to 0.20 (20% return)
        assert scores.scores['A'] == pytest.approx(0.20, rel=0.1)
        # B should be close to 0.10 (10% return)
        assert scores.scores['B'] == pytest.approx(0.10, rel=0.1)

    def test_model_initialization(self):
        """Test model can be initialized with various parameters."""
        model = TrendFilteredMomentum()
        assert model.momentum_lookback == 252
        assert model.ma_period == 200
        assert model.trend_ticker == 'SPY'

        model2 = TrendFilteredMomentum(
            momentum_lookback=60,
            ma_period=100,
            trend_ticker='QQQ',
            defensive_tickers=['TLT'],
            min_periods=30
        )
        assert model2.momentum_lookback == 60
        assert model2.ma_period == 100
        assert model2.trend_ticker == 'QQQ'
        assert model2.defensive_tickers == ['TLT']
        assert model2.min_periods == 30
