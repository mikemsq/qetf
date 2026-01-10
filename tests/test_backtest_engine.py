"""Tests for SimpleBacktestEngine.

These tests verify the event-driven backtest engine works correctly with
Phase 2 components (momentum alpha, equal-weight portfolio, transaction costs).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from quantetf.backtest.simple_engine import (
    SimpleBacktestEngine,
    BacktestConfig,
    _generate_rebalance_dates,
    _calculate_sharpe,
    _calculate_max_drawdown,
)
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.types import Universe


# ============================================================================
# Helper Functions Tests
# ============================================================================


def test_generate_rebalance_dates_monthly():
    """Test monthly rebalance date generation."""
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-12-31")

    dates = _generate_rebalance_dates(start, end, frequency='monthly')

    assert len(dates) == 12  # One per month
    assert all(isinstance(d, pd.Timestamp) for d in dates)
    # Check that dates are month-end business days
    assert dates[0].month == 1
    assert dates[-1].month == 12


def test_generate_rebalance_dates_weekly():
    """Test weekly rebalance date generation."""
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-03-31")

    dates = _generate_rebalance_dates(start, end, frequency='weekly')

    # Should have ~13 weeks in 3 months
    assert len(dates) >= 12
    assert len(dates) <= 14
    assert all(isinstance(d, pd.Timestamp) for d in dates)
    # Check that all dates are Fridays
    assert all(d.dayofweek == 4 for d in dates)  # 4 = Friday


def test_generate_rebalance_dates_invalid_frequency():
    """Test that invalid frequency raises error."""
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-12-31")

    with pytest.raises(ValueError, match="Unknown frequency"):
        _generate_rebalance_dates(start, end, frequency='daily')


def test_calculate_sharpe_normal_returns():
    """Test Sharpe ratio calculation with normal returns."""
    # Monthly returns with positive mean
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, 0.02])

    sharpe = _calculate_sharpe(returns, periods_per_year=12)

    # Should be positive since mean is positive
    assert sharpe > 0
    # Rough sanity check
    assert 0.5 < sharpe < 5.0


def test_calculate_sharpe_zero_volatility():
    """Test Sharpe ratio with zero volatility (all same returns)."""
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    sharpe = _calculate_sharpe(returns, periods_per_year=12)

    # Zero volatility should return 0
    assert sharpe == 0.0


def test_calculate_sharpe_empty_returns():
    """Test Sharpe ratio with empty returns."""
    returns = pd.Series([], dtype=float)

    sharpe = _calculate_sharpe(returns, periods_per_year=12)

    assert sharpe == 0.0


def test_calculate_max_drawdown_with_drawdown():
    """Test max drawdown calculation with actual drawdown."""
    nav = pd.Series([100, 110, 105, 90, 95, 100])

    max_dd = _calculate_max_drawdown(nav)

    # Max drawdown is from 110 to 90 = -18.18%
    expected = (90 - 110) / 110
    assert abs(max_dd - expected) < 0.001


def test_calculate_max_drawdown_no_drawdown():
    """Test max drawdown with monotonically increasing NAV."""
    nav = pd.Series([100, 105, 110, 115, 120])

    max_dd = _calculate_max_drawdown(nav)

    # No drawdown
    assert max_dd == 0.0


def test_calculate_max_drawdown_empty():
    """Test max drawdown with empty series."""
    nav = pd.Series([], dtype=float)

    max_dd = _calculate_max_drawdown(nav)

    assert max_dd == 0.0


# ============================================================================
# Integration Tests with Synthetic Data
# ============================================================================


@pytest.fixture
def synthetic_snapshot(tmp_path):
    """Create a synthetic snapshot for testing.

    Creates 3 tickers with 3 years of daily data for sufficient lookback.
    """
    dates = pd.date_range("2021-01-01", "2023-12-31", freq='B')

    # Create synthetic prices with different trends
    # SPY: steady uptrend
    # QQQ: volatile uptrend
    # IWM: flat/choppy

    np.random.seed(42)

    spy_prices = 100 * (1 + np.arange(len(dates)) * 0.0003)  # ~30% per year
    spy_prices = spy_prices * (1 + np.random.randn(len(dates)) * 0.01)  # Add noise

    qqq_prices = 100 * (1 + np.arange(len(dates)) * 0.0004)  # ~40% per year
    qqq_prices = qqq_prices * (1 + np.random.randn(len(dates)) * 0.02)  # More volatile

    iwm_prices = 100 * (1 + np.random.randn(len(dates)) * 0.015)  # Noisy, no trend

    # Create MultiIndex DataFrame
    data = pd.DataFrame(index=dates)

    for ticker, prices in [('SPY', spy_prices), ('QQQ', qqq_prices), ('IWM', iwm_prices)]:
        data[(ticker, 'Open')] = prices
        data[(ticker, 'High')] = prices * 1.01
        data[(ticker, 'Low')] = prices * 0.99
        data[(ticker, 'Close')] = prices
        data[(ticker, 'Volume')] = 1_000_000

    data.columns = pd.MultiIndex.from_tuples(data.columns, names=['Ticker', 'Price'])

    # Save to parquet
    snapshot_dir = tmp_path / "test_snapshot"
    snapshot_dir.mkdir()
    parquet_path = snapshot_dir / "data.parquet"
    data.to_parquet(parquet_path)

    return snapshot_dir


def test_backtest_runs_successfully(synthetic_snapshot):
    """Test that a basic backtest completes successfully."""
    from quantetf.data.snapshot_store import SnapshotDataStore

    # Setup
    store = SnapshotDataStore(synthetic_snapshot / "data.parquet")
    universe = Universe(
        as_of=pd.Timestamp('2023-12-31'),
        tickers=('SPY', 'QQQ', 'IWM')
    )

    config = BacktestConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2023-12-31'),
        universe=universe,
        initial_capital=100_000.0,
        rebalance_frequency='monthly'
    )

    # Run backtest
    engine = SimpleBacktestEngine()
    result = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60, min_periods=50),
        portfolio=EqualWeightTopN(top_n=2),
        cost_model=FlatTransactionCost(cost_bps=10.0),
        store=store
    )

    # Verify structure
    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0
    assert 'nav' in result.equity_curve.columns
    assert 'cost' in result.equity_curve.columns
    assert 'returns' in result.equity_curve.columns

    # Verify metrics exist
    assert 'total_return' in result.metrics
    assert 'sharpe_ratio' in result.metrics
    assert 'max_drawdown' in result.metrics
    assert 'total_costs' in result.metrics
    assert 'num_rebalances' in result.metrics

    # Verify holdings and weights
    assert result.holdings_history is not None
    assert result.weights_history is not None
    assert len(result.holdings_history) == len(result.equity_curve)
    assert len(result.weights_history) == len(result.equity_curve)


def test_backtest_costs_applied(synthetic_snapshot):
    """Test that transaction costs are properly applied."""
    from quantetf.data.snapshot_store import SnapshotDataStore

    store = SnapshotDataStore(synthetic_snapshot / "data.parquet")
    universe = Universe(
        as_of=pd.Timestamp('2023-12-31'),
        tickers=('SPY', 'QQQ', 'IWM')
    )

    config = BacktestConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2023-06-30'),
        universe=universe,
        initial_capital=100_000.0,
        rebalance_frequency='monthly'
    )

    # Run with costs
    engine = SimpleBacktestEngine()
    result_with_costs = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60, min_periods=50),
        portfolio=EqualWeightTopN(top_n=2),
        cost_model=FlatTransactionCost(cost_bps=50.0),  # High costs
        store=store
    )

    # Run without costs
    result_no_costs = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60, min_periods=50),
        portfolio=EqualWeightTopN(top_n=2),
        cost_model=FlatTransactionCost(cost_bps=0.0),  # Zero costs
        store=store
    )

    # With costs should have lower final NAV
    assert result_with_costs.metrics['final_nav'] < result_no_costs.metrics['final_nav']

    # Total costs should be positive
    assert result_with_costs.metrics['total_costs'] > 0

    # No-cost run should have zero costs
    assert result_no_costs.metrics['total_costs'] == 0.0


def test_backtest_weights_sum_to_one(synthetic_snapshot):
    """Test that portfolio weights always sum to ~1.0."""
    from quantetf.data.snapshot_store import SnapshotDataStore

    store = SnapshotDataStore(synthetic_snapshot / "data.parquet")
    universe = Universe(
        as_of=pd.Timestamp('2023-12-31'),
        tickers=('SPY', 'QQQ', 'IWM')
    )

    config = BacktestConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2023-06-30'),
        universe=universe,
        initial_capital=100_000.0
    )

    engine = SimpleBacktestEngine()
    result = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60, min_periods=50),
        portfolio=EqualWeightTopN(top_n=2),
        cost_model=FlatTransactionCost(cost_bps=10.0),
        store=store
    )

    # Check all weight rows sum to ~1.0
    weight_sums = result.weights_history.sum(axis=1)

    for date, weight_sum in weight_sums.items():
        assert abs(weight_sum - 1.0) < 0.01, f"Weights don't sum to 1.0 on {date}: {weight_sum}"


def test_backtest_top_n_positions(synthetic_snapshot):
    """Test that we hold exactly top_n positions."""
    from quantetf.data.snapshot_store import SnapshotDataStore

    store = SnapshotDataStore(synthetic_snapshot / "data.parquet")
    universe = Universe(
        as_of=pd.Timestamp('2023-12-31'),
        tickers=('SPY', 'QQQ', 'IWM')
    )

    config = BacktestConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2023-06-30'),
        universe=universe,
        initial_capital=100_000.0
    )

    top_n = 2
    engine = SimpleBacktestEngine()
    result = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60, min_periods=50),
        portfolio=EqualWeightTopN(top_n=top_n),
        cost_model=FlatTransactionCost(cost_bps=10.0),
        store=store
    )

    # Check that each rebalance has exactly top_n positions
    for date, weights in result.weights_history.iterrows():
        num_positions = (weights > 0).sum()
        assert num_positions == top_n, f"Expected {top_n} positions on {date}, got {num_positions}"


def test_backtest_nav_evolution(synthetic_snapshot):
    """Test that NAV evolves correctly over time."""
    from quantetf.data.snapshot_store import SnapshotDataStore

    store = SnapshotDataStore(synthetic_snapshot / "data.parquet")
    universe = Universe(
        as_of=pd.Timestamp('2023-12-31'),
        tickers=('SPY', 'QQQ', 'IWM')
    )

    config = BacktestConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2023-06-30'),
        universe=universe,
        initial_capital=100_000.0
    )

    engine = SimpleBacktestEngine()
    result = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60, min_periods=50),
        portfolio=EqualWeightTopN(top_n=2),
        cost_model=FlatTransactionCost(cost_bps=10.0),
        store=store
    )

    # NAV should start near initial capital (after first rebalance costs)
    first_nav = result.equity_curve['nav'].iloc[0]
    assert first_nav <= config.initial_capital  # May be lower due to costs
    assert first_nav > config.initial_capital * 0.99  # But not much lower

    # NAV should be monotonic or have reasonable changes
    nav_series = result.equity_curve['nav']
    for i in range(1, len(nav_series)):
        # NAV shouldn't change by more than 50% between rebalances (sanity check)
        pct_change = abs(nav_series.iloc[i] / nav_series.iloc[i-1] - 1.0)
        assert pct_change < 0.5, f"Unrealistic NAV change: {pct_change:.2%}"


def test_backtest_empty_universe():
    """Test that backtest handles empty universe gracefully."""
    # This test would require a special setup, skipping for now
    # TODO: Implement once we have better error handling
    pass


def test_backtest_insufficient_data(tmp_path):
    """Test backtest with insufficient historical data."""
    # Create snapshot with very limited data
    dates = pd.date_range("2023-12-01", "2023-12-31", freq='B')  # Only 1 month

    data = pd.DataFrame(index=dates)
    for ticker in ['SPY', 'QQQ']:
        prices = 100 * (1 + np.arange(len(dates)) * 0.001)
        data[(ticker, 'Close')] = prices

    data.columns = pd.MultiIndex.from_tuples(data.columns, names=['Ticker', 'Price'])

    snapshot_dir = tmp_path / "limited_snapshot"
    snapshot_dir.mkdir()
    data.to_parquet(snapshot_dir / "data.parquet")

    from quantetf.data.snapshot_store import SnapshotDataStore

    store = SnapshotDataStore(snapshot_dir / "data.parquet")
    universe = Universe(
        as_of=pd.Timestamp('2023-12-31'),
        tickers=('SPY', 'QQQ')
    )

    config = BacktestConfig(
        start_date=pd.Timestamp('2023-12-01'),
        end_date=pd.Timestamp('2023-12-31'),
        universe=universe,
        initial_capital=100_000.0
    )

    engine = SimpleBacktestEngine()

    # Should still run, but may have limited rebalances
    result = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=252, min_periods=10),  # Lower min_periods
        portfolio=EqualWeightTopN(top_n=1),
        cost_model=FlatTransactionCost(cost_bps=10.0),
        store=store
    )

    # Should have at least one rebalance
    assert len(result.equity_curve) >= 1


def test_backtest_reproducibility(synthetic_snapshot):
    """Test that running same backtest twice gives same results."""
    from quantetf.data.snapshot_store import SnapshotDataStore

    store = SnapshotDataStore(synthetic_snapshot / "data.parquet")
    universe = Universe(
        as_of=pd.Timestamp('2023-12-31'),
        tickers=('SPY', 'QQQ', 'IWM')
    )

    config = BacktestConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2023-06-30'),
        universe=universe,
        initial_capital=100_000.0
    )

    engine = SimpleBacktestEngine()

    # Run twice
    result1 = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60, min_periods=50),
        portfolio=EqualWeightTopN(top_n=2),
        cost_model=FlatTransactionCost(cost_bps=10.0),
        store=store
    )

    result2 = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60, min_periods=50),
        portfolio=EqualWeightTopN(top_n=2),
        cost_model=FlatTransactionCost(cost_bps=10.0),
        store=store
    )

    # Results should be identical
    assert result1.metrics['total_return'] == result2.metrics['total_return']
    assert result1.metrics['sharpe_ratio'] == result2.metrics['sharpe_ratio']
    assert result1.metrics['final_nav'] == result2.metrics['final_nav']

    # Equity curves should match
    pd.testing.assert_series_equal(result1.equity_curve['nav'], result2.equity_curve['nav'])
