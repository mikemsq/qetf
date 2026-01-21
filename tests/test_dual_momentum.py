"""Tests for DualMomentum alpha model.

These tests verify the dual momentum strategy correctly:
1. Uses relative momentum when assets have positive absolute momentum
2. Falls back to safe assets when all momentum is negative
3. Correctly applies the risk-free rate threshold
4. Handles edge cases gracefully
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantetf.alpha.dual_momentum import DualMomentum
from quantetf.data.access import DataAccessFactory
from quantetf.types import Universe


def create_test_context(prices_df: pd.DataFrame):
    """Create a DataAccessContext from price data."""
    tmpdir = tempfile.mkdtemp()
    snapshot_path = Path(tmpdir) / 'test_snapshot'
    snapshot_path.mkdir()
    parquet_path = snapshot_path / 'data.parquet'
    prices_df.to_parquet(parquet_path)
    ctx = DataAccessFactory.create_context(
        config={"snapshot_path": str(parquet_path)},
        enable_caching=False
    )
    return ctx, snapshot_path


class TestDualMomentum:
    """Tests for DualMomentum alpha model."""

    def test_positive_momentum_uses_ranking(self):
        """When assets have positive momentum, rank by return."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Create assets with clear return ranking (all above 2% rf threshold)
        # Daily return needed for 35% annual over 300 days
        data = {}
        ticker_a_prices = 100 * np.cumprod(1 + np.ones(300) * 0.001)  # ~35% annual
        ticker_b_prices = 100 * np.cumprod(1 + np.ones(300) * 0.0006)  # ~20% annual
        ticker_c_prices = 100 * np.cumprod(1 + np.ones(300) * 0.0003)  # ~9% annual

        for ticker, prices in [('A', ticker_a_prices), ('B', ticker_b_prices),
                               ('C', ticker_c_prices), ('AGG', 100 * np.ones(300))]:
            data[ticker] = pd.DataFrame({
                'Open': prices,
                'High': prices,
                'Low': prices,
                'Close': prices,
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        ctx, snapshot_path = create_test_context(combined)

        try:
            model = DualMomentum(
                lookback=252,
                risk_free_rate=0.02,
                min_periods=50,
            )

            universe = Universe(
                as_of=pd.Timestamp('2020-10-27'),
                tickers=('A', 'B', 'C', 'AGG')
            )

            scores = model.score(
                as_of=pd.Timestamp('2020-10-27'),
                universe=universe,
                features=None,
                data_access=ctx
            )

            # A should have highest score, then B, then C
            assert scores.scores['A'] > scores.scores['B'] > scores.scores['C']
            # AGG is safe ticker, should have 0 score when momentum is positive
            assert scores.scores['AGG'] == 0.0
        finally:
            import shutil
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)

    def test_negative_momentum_uses_safe(self):
        """When all momentum negative, use safe assets."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # All assets declining
        data = {}
        ticker_a_prices = 100 * np.cumprod(1 - np.ones(300) * 0.001)  # Declining
        ticker_b_prices = 100 * np.cumprod(1 - np.ones(300) * 0.0005)  # Declining

        for ticker, prices in [('A', ticker_a_prices), ('B', ticker_b_prices),
                               ('AGG', 100 * np.ones(300)), ('BND', 100 * np.ones(300))]:
            data[ticker] = pd.DataFrame({
                'Open': prices,
                'High': prices,
                'Low': prices,
                'Close': prices,
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        ctx, snapshot_path = create_test_context(combined)

        try:
            model = DualMomentum(
                lookback=252,
                risk_free_rate=0.02,
                safe_tickers=['AGG', 'BND'],
                min_periods=50,
            )

            universe = Universe(
                as_of=pd.Timestamp('2020-10-27'),
                tickers=('A', 'B', 'AGG', 'BND')
            )

            scores = model.score(
                as_of=pd.Timestamp('2020-10-27'),
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Safe assets should have high scores
            assert scores.scores['AGG'] == 1.0
            assert scores.scores['BND'] == 1.0
            # Risky assets should have zero
            assert scores.scores['A'] == 0.0
            assert scores.scores['B'] == 0.0
        finally:
            import shutil
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)

    def test_absolute_momentum_threshold(self):
        """Test that absolute momentum filter works correctly."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Asset with 1% return (below 2% threshold)
        # Daily return for ~1% annual over 252 days
        data = {}
        ticker_a_prices = 100 * np.cumprod(1 + np.ones(300) * 0.00004)  # ~1% annual

        for ticker, prices in [('A', ticker_a_prices), ('AGG', 100 * np.ones(300))]:
            data[ticker] = pd.DataFrame({
                'Open': prices,
                'High': prices,
                'Low': prices,
                'Close': prices,
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        ctx, snapshot_path = create_test_context(combined)

        try:
            model = DualMomentum(
                lookback=252,
                risk_free_rate=0.02,  # 2% threshold
                min_periods=50,
            )

            universe = Universe(
                as_of=pd.Timestamp('2020-10-27'),
                tickers=('A', 'AGG')
            )

            scores = model.score(
                as_of=pd.Timestamp('2020-10-27'),
                universe=universe,
                features=None,
                data_access=ctx
            )

            # A should be filtered out (below threshold)
            # Should fall back to safe assets
            assert scores.scores['A'] == 0.0
            assert scores.scores['AGG'] == 1.0
        finally:
            import shutil
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)

    def test_partial_positive_momentum(self):
        """Test when some assets have positive momentum and some don't."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # A has positive momentum, B has negative
        data = {}
        ticker_a_prices = 100 * np.cumprod(1 + np.ones(300) * 0.001)  # +35% annual
        ticker_b_prices = 100 * np.cumprod(1 - np.ones(300) * 0.001)  # Declining

        for ticker, prices in [('A', ticker_a_prices), ('B', ticker_b_prices),
                               ('AGG', 100 * np.ones(300))]:
            data[ticker] = pd.DataFrame({
                'Open': prices,
                'High': prices,
                'Low': prices,
                'Close': prices,
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        ctx, snapshot_path = create_test_context(combined)

        try:
            model = DualMomentum(
                lookback=252,
                risk_free_rate=0.02,
                min_periods=50,
            )

            universe = Universe(
                as_of=pd.Timestamp('2020-10-27'),
                tickers=('A', 'B', 'AGG')
            )

            scores = model.score(
                as_of=pd.Timestamp('2020-10-27'),
                universe=universe,
                features=None,
                data_access=ctx
            )

            # A should have positive score (above threshold)
            assert scores.scores['A'] > 0
            # B should have zero (below threshold)
            assert scores.scores['B'] == 0.0
            # AGG should have zero (safe tickers only used when all negative)
            assert scores.scores['AGG'] == 0.0
        finally:
            import shutil
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)

    def test_signal_type_momentum(self):
        """Test signal type detection for momentum regime."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        data = {}
        ticker_a_prices = 100 * np.cumprod(1 + np.ones(300) * 0.001)

        for ticker, prices in [('A', ticker_a_prices), ('AGG', 100 * np.ones(300))]:
            data[ticker] = pd.DataFrame({
                'Open': prices,
                'High': prices,
                'Low': prices,
                'Close': prices,
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        ctx, snapshot_path = create_test_context(combined)

        try:
            model = DualMomentum(
                lookback=252,
                risk_free_rate=0.02,
                min_periods=50,
            )

            universe = Universe(
                as_of=pd.Timestamp('2020-10-27'),
                tickers=('A', 'AGG')
            )

            signal_type = model.get_signal_type(
                data_access=ctx,
                as_of=pd.Timestamp('2020-10-27'),
                universe=universe
            )

            assert signal_type == "MOMENTUM"
        finally:
            import shutil
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)

    def test_signal_type_safe(self):
        """Test signal type detection for safe regime."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        data = {}
        ticker_a_prices = 100 * np.cumprod(1 - np.ones(300) * 0.001)

        for ticker, prices in [('A', ticker_a_prices), ('AGG', 100 * np.ones(300))]:
            data[ticker] = pd.DataFrame({
                'Open': prices,
                'High': prices,
                'Low': prices,
                'Close': prices,
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        ctx, snapshot_path = create_test_context(combined)

        try:
            model = DualMomentum(
                lookback=252,
                risk_free_rate=0.02,
                min_periods=50,
            )

            universe = Universe(
                as_of=pd.Timestamp('2020-10-27'),
                tickers=('A', 'AGG')
            )

            signal_type = model.get_signal_type(
                data_access=ctx,
                as_of=pd.Timestamp('2020-10-27'),
                universe=universe
            )

            assert signal_type == "SAFE"
        finally:
            import shutil
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)

    def test_custom_safe_tickers(self):
        """Test that custom safe tickers are respected."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # All declining
        data = {}
        for ticker in ['A', 'B', 'AGG', 'CUSTOM']:
            data[ticker] = pd.DataFrame({
                'Open': 100 * np.cumprod(1 - np.ones(300) * 0.001),
                'High': 100 * np.cumprod(1 - np.ones(300) * 0.001),
                'Low': 100 * np.cumprod(1 - np.ones(300) * 0.001),
                'Close': 100 * np.cumprod(1 - np.ones(300) * 0.001),
                'Volume': [1000000] * 300,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        ctx, snapshot_path = create_test_context(combined)

        try:
            # Only CUSTOM as safe (not AGG)
            model = DualMomentum(
                lookback=252,
                risk_free_rate=0.02,
                safe_tickers=['CUSTOM'],
                min_periods=50,
            )

            universe = Universe(
                as_of=pd.Timestamp('2020-10-27'),
                tickers=('A', 'B', 'AGG', 'CUSTOM')
            )

            scores = model.score(
                as_of=pd.Timestamp('2020-10-27'),
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Only CUSTOM should have high score
            assert scores.scores['CUSTOM'] == 1.0
            # AGG should be zero (not in safe list, and it's declining)
            assert scores.scores['AGG'] == 0.0
        finally:
            import shutil
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')

        data = {}
        for ticker in ['A', 'AGG']:
            data[ticker] = pd.DataFrame({
                'Open': np.random.randn(50).cumsum() + 100,
                'High': np.random.randn(50).cumsum() + 101,
                'Low': np.random.randn(50).cumsum() + 99,
                'Close': np.random.randn(50).cumsum() + 100,
                'Volume': [1000000] * 50,
            }, index=dates)

        combined = pd.concat(data, axis=1)
        combined.columns.names = ['Ticker', 'Price']

        ctx, snapshot_path = create_test_context(combined)

        try:
            model = DualMomentum(
                lookback=252,
                min_periods=200,  # More than available
            )

            universe = Universe(
                as_of=pd.Timestamp('2020-02-15'),
                tickers=('A', 'AGG')
            )

            scores = model.score(
                as_of=pd.Timestamp('2020-02-15'),
                universe=universe,
                features=None,
                data_access=ctx
            )

            # Should fall back to safe assets when no valid momentum data
            assert scores.scores['AGG'] == 1.0
        finally:
            import shutil
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)

    def test_wrong_store_type_raises(self):
        """Should raise TypeError if not using DataAccessContext."""
        model = DualMomentum()

        universe = Universe(
            as_of=pd.Timestamp('2020-10-01'),
            tickers=('A', 'B')
        )

        with pytest.raises((TypeError, AttributeError)):
            model.score(
                as_of=pd.Timestamp('2020-10-01'),
                universe=universe,
                features=None,
                data_access=None  # Wrong type!
            )
