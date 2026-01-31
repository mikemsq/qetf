"""Tests for ValueMomentum alpha model.

These tests verify the value-momentum blend strategy correctly:
1. Blends momentum and value signals with configurable weights
2. Z-scores signals before blending
3. Handles different lookback periods
4. Handles edge cases gracefully
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantetf.alpha.value_momentum import ValueMomentum
from quantetf.data.access import DataAccessFactory
from quantetf.types import Universe


def create_test_data_access(prices_df: pd.DataFrame, tmp_path: Path):
    """Create a DataAccessContext from price data."""
    snapshot_path = tmp_path / 'test_snapshot'
    snapshot_path.mkdir(exist_ok=True)
    parquet_path = snapshot_path / 'data.parquet'
    prices_df.to_parquet(parquet_path)

    return DataAccessFactory.create_context(
        config={"snapshot_path": str(parquet_path)},
        enable_caching=False
    )


class TestValueMomentum:
    """Tests for ValueMomentum alpha model."""

    def test_equal_weight_blend(self, tmp_path):
        """Test 50/50 blend of momentum and value signals."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Create assets with different momentum profiles
        # A: high momentum (winner)
        # B: low momentum (loser)
        # C: medium momentum
        data = {}
        ticker_a_prices = 100 * np.cumprod(1 + np.ones(300) * 0.001)  # Winner
        ticker_b_prices = 100 * np.cumprod(1 - np.ones(300) * 0.0005)  # Loser
        ticker_c_prices = 100 * np.ones(300)  # Flat

        for ticker, prices in [('A', ticker_a_prices), ('B', ticker_b_prices), ('C', ticker_c_prices)]:
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

        model = ValueMomentum(
            momentum_weight=0.5,
            value_weight=0.5,
            momentum_lookback=252,
            value_lookback=252,
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-27'),
            tickers=('A', 'B', 'C')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-10-27'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # All three should have scores
        assert not pd.isna(scores.scores['A'])
        assert not pd.isna(scores.scores['B'])
        assert not pd.isna(scores.scores['C'])

    def test_momentum_only(self, tmp_path):
        """Test with 100% momentum weight."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        data = {}
        ticker_a_prices = 100 * np.cumprod(1 + np.ones(300) * 0.001)  # Winner
        ticker_b_prices = 100 * np.cumprod(1 - np.ones(300) * 0.0005)  # Loser

        for ticker, prices in [('A', ticker_a_prices), ('B', ticker_b_prices)]:
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

        model = ValueMomentum(
            momentum_weight=1.0,
            value_weight=0.0,
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-27'),
            tickers=('A', 'B')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-10-27'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # A (winner) should have higher score than B (loser)
        assert scores.scores['A'] > scores.scores['B']

    def test_value_only(self, tmp_path):
        """Test with 100% value weight."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        data = {}
        ticker_a_prices = 100 * np.cumprod(1 + np.ones(300) * 0.001)  # Winner
        ticker_b_prices = 100 * np.cumprod(1 - np.ones(300) * 0.0005)  # Loser

        for ticker, prices in [('A', ticker_a_prices), ('B', ticker_b_prices)]:
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

        model = ValueMomentum(
            momentum_weight=0.0,
            value_weight=1.0,
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-27'),
            tickers=('A', 'B')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-10-27'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # B (loser) should have higher value score than A (winner)
        assert scores.scores['B'] > scores.scores['A']

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        model = ValueMomentum(
            momentum_weight=3.0,
            value_weight=1.0,
        )

        # Weights should be normalized
        assert model.momentum_weight == pytest.approx(0.75)
        assert model.value_weight == pytest.approx(0.25)

    def test_different_lookbacks(self, tmp_path):
        """Test using different lookback periods for momentum and value."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Create price data with different short vs long term patterns
        # First 200 days: trending up
        # Last 100 days: trending down
        prices = np.concatenate([
            100 * np.cumprod(1 + np.ones(200) * 0.002),  # Up
            100 * np.cumprod(1 + np.ones(200) * 0.002)[-1] * np.cumprod(1 - np.ones(100) * 0.003)  # Down
        ])

        data = {}
        data['A'] = pd.DataFrame({
            'Open': prices,
            'High': prices,
            'Low': prices,
            'Close': prices,
            'Volume': [1000000] * 300,
        }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        data_access = create_test_data_access(combined, tmp_path)

        model = ValueMomentum(
            momentum_lookback=60,  # Short-term (will see downtrend)
            value_lookback=200,  # Long-term (will see uptrend)
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-27'),
            tickers=('A',)
        )

        # Should be able to get scores
        scores = model.score(
            as_of=pd.Timestamp('2020-10-27'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        assert not pd.isna(scores.scores['A'])

    def test_signal_components(self, tmp_path):
        """Test get_signal_components returns individual signals."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        data = {}
        ticker_a_prices = 100 * np.cumprod(1 + np.ones(300) * 0.001)
        ticker_b_prices = 100 * np.cumprod(1 - np.ones(300) * 0.0005)

        for ticker, prices in [('A', ticker_a_prices), ('B', ticker_b_prices)]:
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

        model = ValueMomentum(min_periods=50)

        universe = Universe(
            as_of=pd.Timestamp('2020-10-27'),
            tickers=('A', 'B')
        )

        components = model.get_signal_components(
            data_access=data_access,
            as_of=pd.Timestamp('2020-10-27'),
            universe=universe
        )

        assert 'momentum_z' in components
        assert 'value_z' in components
        assert 'blended' in components

        # Momentum z-scores should show A > B
        assert components['momentum_z']['A'] > components['momentum_z']['B']

        # Value z-scores should show B > A
        assert components['value_z']['B'] > components['value_z']['A']

    def test_insufficient_data(self, tmp_path):
        """Test handling of insufficient data."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')

        data = {}
        for ticker in ['A', 'B']:
            data[ticker] = pd.DataFrame({
                'Open': np.random.randn(50).cumsum() + 100,
                'High': np.random.randn(50).cumsum() + 101,
                'Low': np.random.randn(50).cumsum() + 99,
                'Close': np.random.randn(50).cumsum() + 100,
                'Volume': [1000000] * 50,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        data_access = create_test_data_access(combined, tmp_path)

        model = ValueMomentum(min_periods=200)

        universe = Universe(
            as_of=pd.Timestamp('2020-02-15'),
            tickers=('A', 'B')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-02-15'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # Should return NaN for both tickers
        assert pd.isna(scores.scores['A'])
        assert pd.isna(scores.scores['B'])

    def test_zscore_with_no_variance(self, tmp_path):
        """Test z-score handles zero variance gracefully."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # All same price - no variance
        data = {}
        for ticker in ['A', 'B', 'C']:
            data[ticker] = pd.DataFrame({
                'Open': 100 * np.ones(300),
                'High': 100 * np.ones(300),
                'Low': 100 * np.ones(300),
                'Close': 100 * np.ones(300),
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        data_access = create_test_data_access(combined, tmp_path)

        model = ValueMomentum(min_periods=50)

        universe = Universe(
            as_of=pd.Timestamp('2020-10-27'),
            tickers=('A', 'B', 'C')
        )

        # Should not raise, should return zeros
        scores = model.score(
            as_of=pd.Timestamp('2020-10-27'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # All should be zero (no variance to z-score)
        assert scores.scores['A'] == 0.0
        assert scores.scores['B'] == 0.0
        assert scores.scores['C'] == 0.0

    def test_blended_score_correctness(self, tmp_path):
        """Verify blended scores are calculated correctly."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Create three assets with known characteristics
        # A: +30% return (high momentum, low value)
        # B: -10% return (low momentum, high value)
        # C: +10% return (medium both)
        data = {}
        ticker_a_prices = 100 * np.linspace(1, 1.3, 300)  # +30%
        ticker_b_prices = 100 * np.linspace(1, 0.9, 300)  # -10%
        ticker_c_prices = 100 * np.linspace(1, 1.1, 300)  # +10%

        for ticker, prices in [('A', ticker_a_prices), ('B', ticker_b_prices), ('C', ticker_c_prices)]:
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

        # Use 60/40 momentum/value blend
        model = ValueMomentum(
            momentum_weight=0.6,
            value_weight=0.4,
            min_periods=50,
        )

        universe = Universe(
            as_of=pd.Timestamp('2020-10-27'),
            tickers=('A', 'B', 'C')
        )

        scores = model.score(
            as_of=pd.Timestamp('2020-10-27'),
            universe=universe,
            features=None,
            data_access=data_access
        )

        # With 60% momentum weight:
        # A should still have relatively high score (momentum dominates)
        # B should have moderate score (value partially compensates)
        assert scores.scores['A'] > scores.scores['C']
        # C is middle ground
        assert scores.scores['C'] > scores.scores['B']
