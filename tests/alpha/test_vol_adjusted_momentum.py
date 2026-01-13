"""Tests for Volatility-Adjusted Momentum alpha model."""

import pandas as pd
import numpy as np
import pytest
import tempfile
from pathlib import Path

from quantetf.alpha.vol_adjusted_momentum import VolAdjustedMomentumAlpha
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


class TestVolAdjustedMomentumBasics:
    """Test basic functionality."""

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        alpha = VolAdjustedMomentumAlpha(
            lookback_days=252,
            min_periods=200,
            vol_floor=0.01
        )
        assert alpha.lookback_days == 252
        assert alpha.vol_floor == 0.01

        # Invalid: lookback < min_periods
        with pytest.raises(ValueError, match="lookback_days must be >= min_periods"):
            VolAdjustedMomentumAlpha(lookback_days=100, min_periods=200)

        # Invalid: min_periods too small
        with pytest.raises(ValueError, match="min_periods must be >= 20"):
            VolAdjustedMomentumAlpha(min_periods=10)

        # Invalid: vol_floor <= 0
        with pytest.raises(ValueError, match="vol_floor must be > 0"):
            VolAdjustedMomentumAlpha(vol_floor=0)

    def test_ranks_by_sharpe_ratio(self):
        """Test that tickers are ranked by risk-adjusted returns."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # TICKER_A: +20% return, 10% vol → score ≈ 2.0
        # Linear growth with small noise
        prices_a = []
        for i in range(300):
            base = 100 * (1 + 0.20 * i / 299)
            noise = np.sin(i * 0.5) * 0.5  # Small noise for 10% vol
            prices_a.append(base + noise)

        # TICKER_B: +20% return, 40% vol → score ≈ 0.5
        # Linear growth with large noise
        prices_b = []
        for i in range(300):
            base = 100 * (1 + 0.20 * i / 299)
            noise = np.sin(i * 0.5) * 10  # Large noise for 40% vol
            prices_b.append(max(1, base + noise))  # Ensure positive

        prices = create_test_prices(
            tickers=['TICKER_A', 'TICKER_B'],
            dates=dates,
            price_paths={'TICKER_A': prices_a, 'TICKER_B': prices_b}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = VolAdjustedMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_A', 'TICKER_B'))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # TICKER_A should have higher score (better risk-adjusted return)
            assert scores.scores['TICKER_A'] > scores.scores['TICKER_B']

    def test_vol_floor_prevents_division_by_zero(self):
        """Test that vol_floor prevents division by zero for constant prices."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Constant price (zero volatility)
        prices_const = [100] * 300

        prices = create_test_prices(
            tickers=['TICKER_CONST'],
            dates=dates,
            price_paths={'TICKER_CONST': prices_const}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = VolAdjustedMomentumAlpha(vol_floor=0.01)

            universe = Universe(as_of=dates[-1], tickers=('TICKER_CONST',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # Score should be finite (not inf/nan from division by zero)
            assert np.isfinite(scores.scores['TICKER_CONST'])
            # With 0% return and vol_floor, score should be 0
            assert scores.scores['TICKER_CONST'] == 0.0

    def test_negative_returns_negative_scores(self):
        """Test that negative returns produce negative scores."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Declining prices: -10% return
        prices_down = []
        for i in range(300):
            prices_down.append(100 * (1 - 0.10 * i / 299))

        prices = create_test_prices(
            tickers=['TICKER_DOWN'],
            dates=dates,
            price_paths={'TICKER_DOWN': prices_down}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = VolAdjustedMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_DOWN',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # Score should be negative (negative return)
            assert scores.scores['TICKER_DOWN'] < 0

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
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = VolAdjustedMomentumAlpha(min_periods=200)

            universe = Universe(as_of=dates[-1], tickers=('TICKER_SHORT',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            assert pd.isna(scores.scores['TICKER_SHORT'])


class TestVolAdjustedMomentumEdgeCases:
    """Test edge cases and error handling."""

    def test_no_lookahead_bias(self):
        """Verify that scores don't use data from as_of date."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Stable prices except huge jump on last day
        prices_normal = [100] * 299
        prices_with_jump = prices_normal + [200]

        prices = create_test_prices(
            tickers=['TICKER_A'],
            dates=dates,
            price_paths={'TICKER_A': prices_with_jump}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = VolAdjustedMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_A',))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # Score should be near 0 (flat prices), NOT influenced by jump
            # Return should be 0% (100 to 100), vol should be near 0 (floor applied)
            assert abs(scores.scores['TICKER_A']) < 0.1

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
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = VolAdjustedMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_EXISTS', 'TICKER_MISSING'))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            assert pd.isna(scores.scores['TICKER_MISSING'])
            assert pd.notna(scores.scores['TICKER_EXISTS'])

    def test_wrong_store_type_raises_error(self):
        """Test that using wrong store type raises TypeError."""
        alpha = VolAdjustedMomentumAlpha()
        as_of = pd.Timestamp('2023-01-01')
        universe = Universe(as_of=as_of, tickers=('TICKER_A',))

        with pytest.raises(TypeError, match="requires SnapshotDataStore"):
            alpha.score(
                as_of=as_of,
                universe=universe,
                features=None,
                store=None
            )


class TestVolAdjustedMomentumComparison:
    """Test comparison with vanilla momentum."""

    def test_penalizes_high_volatility(self):
        """Test that high volatility is penalized."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Both have +20% return, but different volatility
        # TICKER_SMOOTH: Low vol (smooth growth)
        prices_smooth = [100 * (1 + 0.20 * i / 299) for i in range(300)]

        # TICKER_VOLATILE: High vol (erratic growth to same endpoint)
        prices_volatile = []
        for i in range(300):
            base = 100 * (1 + 0.20 * i / 299)
            volatility = 20 * np.sin(i * 0.3)  # Large swings
            prices_volatile.append(max(1, base + volatility))

        prices = create_test_prices(
            tickers=['TICKER_SMOOTH', 'TICKER_VOLATILE'],
            dates=dates,
            price_paths={
                'TICKER_SMOOTH': prices_smooth,
                'TICKER_VOLATILE': prices_volatile
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / 'test_snapshot'
            snapshot_path.mkdir()
            prices.to_parquet(snapshot_path / 'prices.parquet')

            store = SnapshotDataStore(snapshot_path)
            alpha = VolAdjustedMomentumAlpha()

            universe = Universe(as_of=dates[-1], tickers=('TICKER_SMOOTH', 'TICKER_VOLATILE'))
            scores = alpha.score(
                as_of=dates[-1],
                universe=universe,
                features=None,
                store=store
            )

            # Smooth should score higher than volatile
            assert scores.scores['TICKER_SMOOTH'] > scores.scores['TICKER_VOLATILE']
