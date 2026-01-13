"""Tests for Residual Momentum alpha model."""

import pandas as pd
import numpy as np
import pytest
import tempfile
from pathlib import Path

from quantetf.alpha.residual_momentum import ResidualMomentumAlpha
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.types import Universe


def create_test_prices(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    price_paths: dict[str, list[float]]
) -> pd.DataFrame:
    """Create synthetic price data for testing."""
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


class TestResidualMomentumBasics:
    """Test basic functionality."""

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        alpha = ResidualMomentumAlpha(
            lookback_days=252,
            min_periods=200,
            spy_ticker="SPY"
        )
        assert alpha.lookback_days == 252
        assert alpha.spy_ticker == "SPY"

        # Invalid: lookback < min_periods
        with pytest.raises(ValueError, match="lookback_days.*must be >= min_periods"):
            ResidualMomentumAlpha(lookback_days=100, min_periods=200)

        # Invalid: min_periods too small
        with pytest.raises(ValueError, match="min_periods must be >= 50"):
            ResidualMomentumAlpha(min_periods=30)

    def test_extracts_residual_momentum(self):
        """Test that residual momentum is correctly extracted."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # SPY: steady linear growth
        spy_prices = [100 + i * 0.1 for i in range(300)]

        # TICKER_A: follows SPY exactly (beta=1.0, no residuals)
        ticker_a_prices = [100 + i * 0.1 for i in range(300)]

        # TICKER_B: follows SPY but with positive residuals (outperforms)
        # Same beta but consistently higher
        ticker_b_prices = [105 + i * 0.1 for i in range(300)]

        prices = create_test_prices(
            tickers=['SPY', 'TICKER_A', 'TICKER_B'],
            dates=dates,
            price_paths={
                'SPY': spy_prices,
                'TICKER_A': ticker_a_prices,
                'TICKER_B': ticker_b_prices
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = ResidualMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_A', 'TICKER_B'))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # TICKER_A should have near-zero score (follows SPY exactly)
            assert abs(scores.scores['TICKER_A']) < 0.01

            # TICKER_B should have positive score (outperforms SPY)
            assert scores.scores['TICKER_B'] > 0

    def test_beta_neutral_property(self):
        """Test that residuals are beta-neutral."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # SPY: linear growth
        spy_prices = [100 * (1 + 0.001 * i) for i in range(300)]

        # TICKER_HIGH_BETA: 2x SPY sensitivity but pure beta (no alpha)
        ticker_high_beta = [100 * (1 + 0.002 * i) for i in range(300)]

        prices = create_test_prices(
            tickers=['SPY', 'TICKER_HIGH_BETA'],
            dates=dates,
            price_paths={
                'SPY': spy_prices,
                'TICKER_HIGH_BETA': ticker_high_beta
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = ResidualMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_HIGH_BETA',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # High beta but no idiosyncratic return -> near-zero residuals
            assert abs(scores.scores['TICKER_HIGH_BETA']) < 0.02

    def test_spy_in_universe_returns_nan(self):
        """Test that SPY itself gets NaN score."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        spy_prices = [100 + i * 0.1 for i in range(300)]

        prices = create_test_prices(
            tickers=['SPY'],
            dates=dates,
            price_paths={'SPY': spy_prices}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = ResidualMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('SPY',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # SPY can't be regressed on itself
            assert pd.isna(scores.scores['SPY'])

    def test_insufficient_data_returns_nan(self):
        """Test that insufficient data returns NaN."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        prices = create_test_prices(
            tickers=['SPY', 'TICKER_SHORT'],
            dates=dates,
            price_paths={
                'SPY': [100] * 100,
                'TICKER_SHORT': [100] * 100
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = ResidualMomentumAlpha(min_periods=200)

            universe = Universe(as_of=dates[-1], tickers=('TICKER_SHORT',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            assert pd.isna(scores.scores['TICKER_SHORT'])


class TestResidualMomentumEdgeCases:
    """Test edge cases and error handling."""

    def test_no_lookahead_bias(self):
        """Verify that scores don't use data from as_of date."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Flat prices except huge jump on last day
        spy_prices = [100] * 299 + [100]
        ticker_prices = [100] * 299 + [200]  # Huge jump on as_of date

        prices = create_test_prices(
            tickers=['SPY', 'TICKER_A'],
            dates=dates,
            price_paths={
                'SPY': spy_prices,
                'TICKER_A': ticker_prices
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = ResidualMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_A',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # Score should be near 0 (flat prices), NOT influenced by jump
            assert abs(scores.scores['TICKER_A']) < 0.01

    def test_spy_missing_raises_error(self):
        """Test that missing SPY data raises error."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Only ticker, no SPY
        prices = create_test_prices(
            tickers=['TICKER_A'],
            dates=dates,
            price_paths={'TICKER_A': [100] * 300}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = ResidualMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_A',))

            # Should raise ValueError about missing SPY
            with pytest.raises(ValueError):
                alpha.score(
                    as_of=dates[-1],
                    universe=universe,
                    features=None,
                    store=store
                )

    def test_wrong_store_type_raises_error(self):
        """Test that using wrong store type raises TypeError."""
        alpha = ResidualMomentumAlpha()
        as_of = pd.Timestamp('2023-01-01')
        universe = Universe(as_of=as_of, tickers=('TICKER_A',))

        with pytest.raises(TypeError, match="requires SnapshotDataStore"):
            alpha.score(
                as_of=as_of,
                universe=universe,
                features=None,
                store=None
            )


class TestResidualMomentumRanking:
    """Test that tickers are correctly ranked by residual momentum."""

    def test_ranks_by_idiosyncratic_performance(self):
        """Test that ranking is based on idiosyncratic returns, not beta."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # SPY: linear growth
        spy_prices = [100 + i * 0.05 for i in range(300)]

        # TICKER_HIGH_BETA_LOW_ALPHA: High beta (2x), small positive residuals
        # Follows SPY*2 but with slight consistent outperformance
        ticker_hb_la = [100 + i * 0.10 + i * 0.01 for i in range(300)]

        # TICKER_LOW_BETA_HIGH_ALPHA: Low beta (0.5x), large positive residuals
        # Follows SPY*0.5 but with strong consistent outperformance
        ticker_lb_ha = [100 + i * 0.025 + i * 0.1 for i in range(300)]

        prices = create_test_prices(
            tickers=['SPY', 'TICKER_HB_LA', 'TICKER_LB_HA'],
            dates=dates,
            price_paths={
                'SPY': spy_prices,
                'TICKER_HB_LA': ticker_hb_la,
                'TICKER_LB_HA': ticker_lb_ha
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = ResidualMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_HB_LA', 'TICKER_LB_HA'))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # Low beta but high alpha should score higher than high beta low alpha
            assert scores.scores['TICKER_LB_HA'] > scores.scores['TICKER_HB_LA']
