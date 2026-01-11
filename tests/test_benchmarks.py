"""Tests for benchmark strategies."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from quantetf.evaluation import benchmarks
from quantetf.backtest.simple_engine import BacktestConfig
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.types import Universe


@pytest.fixture
def mock_store(tmp_path):
    """Create a mock SnapshotDataStore with synthetic data."""
    # Create synthetic price data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')  # Business days
    tickers = ['SPY', 'AGG', 'ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']

    # Generate random walk prices
    np.random.seed(42)

    # Create MultiIndex DataFrame
    data = pd.DataFrame(index=dates)

    for ticker in tickers:
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.01, len(dates))
        prices = base_price * (1 + returns).cumprod()

        data[(ticker, 'Open')] = prices
        data[(ticker, 'High')] = prices * 1.01
        data[(ticker, 'Low')] = prices * 0.99
        data[(ticker, 'Close')] = prices
        data[(ticker, 'Volume')] = 1_000_000

    data.columns = pd.MultiIndex.from_tuples(data.columns, names=['Ticker', 'Price'])

    # Save to parquet
    snapshot_dir = tmp_path / 'snapshot'
    snapshot_dir.mkdir()
    parquet_path = snapshot_dir / 'data.parquet'
    data.to_parquet(parquet_path)

    return SnapshotDataStore(parquet_path)


@pytest.fixture
def basic_config():
    """Create a basic backtest config."""
    return BacktestConfig(
        start_date=pd.Timestamp('2020-01-31'),
        end_date=pd.Timestamp('2020-12-31'),
        universe=Universe(
            as_of=pd.Timestamp('2020-12-31'),
            tickers=['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
        ),
        initial_capital=100_000.0,
        rebalance_frequency='monthly'
    )


# Test SPY Benchmark

def test_spy_benchmark_basic(mock_store, basic_config):
    """Test SPY benchmark runs successfully."""
    result = benchmarks.run_spy_benchmark(config=basic_config, store=mock_store)

    assert result.name == 'SPY Buy-and-Hold'
    assert result.description == '100% SPY passive allocation'
    assert len(result.equity_curve) > 0
    assert 'nav' in result.equity_curve.columns
    assert 'cost' in result.equity_curve.columns
    assert 'total_return' in result.metrics
    assert 'final_nav' in result.metrics


def test_spy_benchmark_returns_positive_nav(mock_store, basic_config):
    """Test SPY benchmark returns positive NAV values."""
    result = benchmarks.run_spy_benchmark(config=basic_config, store=mock_store)

    assert (result.equity_curve['nav'] > 0).all()
    assert result.metrics['final_nav'] > 0


def test_spy_benchmark_holdings_constant(mock_store, basic_config):
    """Test SPY benchmark maintains constant holdings (buy-and-hold)."""
    result = benchmarks.run_spy_benchmark(config=basic_config, store=mock_store)

    # SPY shares should be constant (buy and hold)
    spy_holdings = result.holdings_history['SPY']
    assert (spy_holdings == spy_holdings.iloc[0]).all()


# Test 60/40 Benchmark

def test_60_40_benchmark_basic(mock_store, basic_config):
    """Test 60/40 benchmark runs successfully."""
    result = benchmarks.run_60_40_benchmark(config=basic_config, store=mock_store)

    assert result.name == '60/40 Portfolio'
    assert '60% SPY, 40% AGG' in result.description
    assert len(result.equity_curve) > 0
    assert 'total_return' in result.metrics


def test_60_40_benchmark_weights(mock_store, basic_config):
    """Test 60/40 benchmark maintains target weights."""
    result = benchmarks.run_60_40_benchmark(
        config=basic_config,
        store=mock_store,
        rebalance_frequency='monthly'
    )

    # Check weights at rebalance dates
    weights = result.weights_history
    assert 'SPY' in weights.columns
    assert 'AGG' in weights.columns

    # Weights should be close to 60/40 at rebalance dates
    spy_weights = weights['SPY']
    agg_weights = weights['AGG']

    # Allow small numerical errors
    assert np.allclose(spy_weights, 0.6, atol=1e-6)
    assert np.allclose(agg_weights, 0.4, atol=1e-6)


# Test Equal Weight Benchmark

def test_equal_weight_benchmark_basic(mock_store, basic_config):
    """Test equal weight benchmark runs successfully."""
    result = benchmarks.run_equal_weight_benchmark(config=basic_config, store=mock_store)

    assert result.name == 'Equal Weight Universe'
    assert len(result.equity_curve) > 0
    assert result.metrics['universe_size'] == len(basic_config.universe.tickers)


def test_equal_weight_benchmark_weights_sum_to_one(mock_store, basic_config):
    """Test equal weight benchmark weights sum to 1."""
    result = benchmarks.run_equal_weight_benchmark(config=basic_config, store=mock_store)

    weights = result.weights_history
    # Exclude CASH column if present
    from quantetf.types import CASH_TICKER
    etf_columns = [col for col in weights.columns if col != CASH_TICKER]

    # Sum of weights should be ~1.0 at each rebalance date
    weight_sums = weights[etf_columns].sum(axis=1)
    assert np.allclose(weight_sums, 1.0, atol=1e-6)


def test_equal_weight_benchmark_equal_allocation(mock_store, basic_config):
    """Test equal weight benchmark allocates equally."""
    result = benchmarks.run_equal_weight_benchmark(config=basic_config, store=mock_store)

    weights = result.weights_history
    # Exclude cash column
    from quantetf.types import CASH_TICKER
    etf_columns = [col for col in weights.columns if col != CASH_TICKER]

    # At rebalance dates, all weights should be equal
    # Check that all non-zero weights are equal to each other
    first_weight = weights[etf_columns[0]].iloc[0]

    for col in etf_columns:
        assert np.allclose(weights[col], first_weight, atol=1e-6)

    # Also check that weights sum to 1
    weight_sums = weights[etf_columns].sum(axis=1)
    assert np.allclose(weight_sums, 1.0, atol=1e-6)


# Test Random Selection Benchmark

def test_random_selection_benchmark_basic(mock_store, basic_config):
    """Test random selection benchmark runs successfully."""
    result = benchmarks.run_random_selection_benchmark(
        config=basic_config,
        store=mock_store,
        n_selections=3,
        n_trials=10,
        seed=42
    )

    assert 'Random Selection' in result.name
    assert len(result.equity_curve) > 0
    assert result.metrics['n_selections'] == 3
    assert result.metrics['n_trials'] == 10


def test_random_selection_benchmark_reproducible(mock_store, basic_config):
    """Test random selection benchmark is reproducible with seed."""
    result1 = benchmarks.run_random_selection_benchmark(
        config=basic_config,
        store=mock_store,
        n_selections=3,
        n_trials=10,
        seed=42
    )

    result2 = benchmarks.run_random_selection_benchmark(
        config=basic_config,
        store=mock_store,
        n_selections=3,
        n_trials=10,
        seed=42
    )

    # Results should be identical
    assert np.allclose(result1.equity_curve['nav'], result2.equity_curve['nav'])


def test_random_selection_benchmark_includes_std(mock_store, basic_config):
    """Test random selection benchmark includes standard deviation."""
    result = benchmarks.run_random_selection_benchmark(
        config=basic_config,
        store=mock_store,
        n_selections=3,
        n_trials=10,
        seed=42
    )

    assert 'std' in result.equity_curve.columns
    assert 'std_final_nav' in result.metrics


def test_random_selection_benchmark_invalid_n(mock_store, basic_config):
    """Test random selection benchmark raises error for invalid N."""
    with pytest.raises(ValueError, match="n_selections.*universe size"):
        benchmarks.run_random_selection_benchmark(
            config=basic_config,
            store=mock_store,
            n_selections=100,  # More than universe size
            n_trials=10
        )


# Test Oracle Benchmark

def test_oracle_benchmark_basic(mock_store, basic_config):
    """Test oracle benchmark runs successfully."""
    result = benchmarks.run_oracle_benchmark(
        config=basic_config,
        store=mock_store,
        n_selections=3
    )

    assert 'Oracle' in result.name
    assert 'perfect foresight' in result.description.lower()
    assert len(result.equity_curve) > 0
    assert result.metrics['n_selections'] == 3


def test_oracle_benchmark_outperforms_random(mock_store, basic_config):
    """Test oracle benchmark should outperform random selection."""
    random_result = benchmarks.run_random_selection_benchmark(
        config=basic_config,
        store=mock_store,
        n_selections=3,
        n_trials=50,
        seed=42
    )

    oracle_result = benchmarks.run_oracle_benchmark(
        config=basic_config,
        store=mock_store,
        n_selections=3
    )

    # Oracle should have higher total return (in most cases)
    # Note: This is probabilistic, but with 252 days, oracle should win
    assert oracle_result.metrics['total_return'] >= random_result.metrics['total_return']


# Test Date Generation

def test_generate_rebalance_dates_monthly():
    """Test monthly rebalance date generation."""
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2020-12-31')

    dates = benchmarks._generate_rebalance_dates_simple(start, end, 'monthly')

    assert len(dates) == 12  # One per month
    assert dates[0].month == 1
    assert dates[-1].month == 12


def test_generate_rebalance_dates_quarterly():
    """Test quarterly rebalance date generation."""
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2020-12-31')

    dates = benchmarks._generate_rebalance_dates_simple(start, end, 'quarterly')

    assert len(dates) == 4  # One per quarter


def test_generate_rebalance_dates_weekly():
    """Test weekly rebalance date generation."""
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2020-01-31')

    dates = benchmarks._generate_rebalance_dates_simple(start, end, 'weekly')

    # Should have ~4-5 weeks in January
    assert 4 <= len(dates) <= 5


def test_generate_rebalance_dates_invalid_frequency():
    """Test invalid frequency raises error."""
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2020-12-31')

    with pytest.raises(ValueError, match="Unknown frequency"):
        benchmarks._generate_rebalance_dates_simple(start, end, 'daily')


# Edge Cases

def test_benchmark_with_empty_date_range(mock_store):
    """Test benchmarks handle empty date range."""
    config = BacktestConfig(
        start_date=pd.Timestamp('2020-06-01'),
        end_date=pd.Timestamp('2020-05-01'),  # End before start
        universe=Universe(
            as_of=pd.Timestamp('2020-05-01'),
            tickers=['ETF1', 'ETF2']
        ),
        initial_capital=100_000.0
    )

    with pytest.raises(ValueError, match="No valid dates"):
        benchmarks.run_spy_benchmark(config=config, store=mock_store)


def test_benchmark_result_dataclass():
    """Test BenchmarkResult dataclass can be created."""
    result = benchmarks.BenchmarkResult(
        name='Test Benchmark',
        equity_curve=pd.DataFrame({'nav': [100, 110], 'cost': [0, 0]}),
        holdings_history=pd.DataFrame(),
        weights_history=pd.DataFrame(),
        metrics={'total_return': 0.1},
        description='Test description'
    )

    assert result.name == 'Test Benchmark'
    assert result.metrics['total_return'] == 0.1
    assert result.description == 'Test description'
