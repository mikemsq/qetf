"""Tests for walk-forward validation module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
import yaml
from datetime import datetime, timedelta

from quantetf.evaluation.walk_forward import (
    WalkForwardConfig,
    WalkForwardWindow,
    WalkForwardWindowResult,
    WalkForwardAnalysis,
    generate_walk_forward_windows,
    run_walk_forward_validation,
    analyze_walk_forward_results,
    create_walk_forward_summary_table,
)
from quantetf.backtest.simple_engine import BacktestResult, BacktestConfig
from quantetf.types import CASH_TICKER


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_wf_config():
    """Basic walk-forward configuration."""
    return WalkForwardConfig(train_years=2, test_years=1, step_months=6)


@pytest.fixture
def mock_snapshot_dir():
    """Create a temporary snapshot directory with mock data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create metadata
        metadata = {
            "tickers": ["SPY", "QQQ", "IWM"],
            "start_date": "2020-01-01",
            "end_date": "2025-12-31",
        }
        with open(tmpdir / "manifest.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Create mock price data in the format expected by SnapshotDataStore
        # MultiIndex columns: (Ticker, Price)
        dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")
        tickers = metadata["tickers"]

        # Create simple synthetic prices (upward trending with noise)
        np.random.seed(42)

        # Build MultiIndex columns (Ticker, Price)
        columns = pd.MultiIndex.from_product(
            [tickers, ['Open', 'High', 'Low', 'Close', 'Volume']],
            names=['Ticker', 'Price']
        )

        # Create data
        data = {}
        for ticker in tickers:
            base_price = 100.0
            returns = np.random.normal(0.0005, 0.01, len(dates))  # Slight positive drift
            close_prices = base_price * (1 + returns).cumprod()

            data[(ticker, 'Open')] = close_prices * 0.99
            data[(ticker, 'High')] = close_prices * 1.01
            data[(ticker, 'Low')] = close_prices * 0.99
            data[(ticker, 'Close')] = close_prices
            data[(ticker, 'Volume')] = np.full(len(dates), 1000000)

        df = pd.DataFrame(data, index=dates, columns=columns)
        df.index.name = 'Date'

        # Save to parquet with proper name
        df.to_parquet(tmpdir / "data.parquet")

        yield tmpdir


@pytest.fixture
def mock_window_result():
    """Create a mock WalkForwardWindowResult."""
    window = WalkForwardWindow(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2022-01-01"),
        test_start=pd.Timestamp("2022-01-01"),
        test_end=pd.Timestamp("2023-01-01"),
        window_id=0,
    )

    # Create mock equity curves
    train_dates = pd.date_range("2020-01-01", "2022-01-01", freq="B")
    test_dates = pd.date_range("2022-01-01", "2023-01-01", freq="B")

    train_equity = pd.DataFrame(
        {"nav": 100000 * (1 + np.random.normal(0.001, 0.01, len(train_dates))).cumprod()},
        index=train_dates,
    )
    test_equity = pd.DataFrame(
        {"nav": 100000 * (1 + np.random.normal(0.0005, 0.01, len(test_dates))).cumprod()},
        index=test_dates,
    )

    train_config = BacktestConfig(
        start_date=window.train_start,
        end_date=window.train_end,
        universe=["SPY", "QQQ"],
        initial_capital=100000.0,
    )
    test_config = BacktestConfig(
        start_date=window.test_start,
        end_date=window.test_end,
        universe=["SPY", "QQQ"],
        initial_capital=100000.0,
    )

    train_result = BacktestResult(
        equity_curve=train_equity,
        holdings_history=pd.DataFrame(),
        weights_history=pd.DataFrame(),
        metrics={
            "total_return": 0.20,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
            "annualized_return": 0.10,
        },
        config=train_config,
    )

    test_result = BacktestResult(
        equity_curve=test_equity,
        holdings_history=pd.DataFrame(),
        weights_history=pd.DataFrame(),
        metrics={
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.12,
            "annualized_return": 0.08,
        },
        config=test_config,
    )

    return WalkForwardWindowResult(
        window=window,
        train_result=train_result,
        test_result=test_result,
        train_metrics=train_result.metrics,
        test_metrics=test_result.metrics,
    )


# ============================================================================
# Tests: WalkForwardConfig
# ============================================================================


def test_walk_forward_config_defaults():
    """Test default configuration values."""
    config = WalkForwardConfig()
    assert config.train_years == 2
    assert config.test_years == 1
    assert config.step_months == 6
    assert config.min_train_periods == 126
    assert config.min_test_periods == 21


def test_walk_forward_config_custom():
    """Test custom configuration values."""
    config = WalkForwardConfig(train_years=3, test_years=2, step_months=3)
    assert config.train_years == 3
    assert config.test_years == 2
    assert config.step_months == 3


# ============================================================================
# Tests: generate_walk_forward_windows
# ============================================================================


def test_generate_windows_basic(basic_wf_config):
    """Test basic window generation."""
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2025-12-31")

    windows = generate_walk_forward_windows(start, end, basic_wf_config)

    # Should generate multiple windows
    assert len(windows) > 0
    assert all(isinstance(w, WalkForwardWindow) for w in windows)

    # Check first window
    first = windows[0]
    assert first.train_start == start
    assert first.train_end == start + pd.DateOffset(years=2)
    assert first.test_start == first.train_end
    assert first.test_end == first.test_start + pd.DateOffset(years=1)
    assert first.window_id == 0


def test_generate_windows_sequential_ids(basic_wf_config):
    """Test that window IDs are sequential."""
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2025-12-31")

    windows = generate_walk_forward_windows(start, end, basic_wf_config)

    for i, window in enumerate(windows):
        assert window.window_id == i


def test_generate_windows_non_overlapping_test_periods(basic_wf_config):
    """Test that test periods may overlap but train periods step forward consistently."""
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2025-12-31")

    windows = generate_walk_forward_windows(start, end, basic_wf_config)

    for i in range(len(windows) - 1):
        # Train start should step forward by step_months
        expected_step = pd.DateOffset(months=basic_wf_config.step_months)
        assert windows[i + 1].train_start == windows[i].train_start + expected_step


def test_generate_windows_insufficient_data(basic_wf_config):
    """Test error when date range is too short."""
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-12-31")  # Only 1 year, need 3 (2 train + 1 test)

    with pytest.raises(ValueError, match="Insufficient data"):
        generate_walk_forward_windows(start, end, basic_wf_config)


def test_generate_windows_custom_step():
    """Test window generation with custom step size."""
    config = WalkForwardConfig(train_years=2, test_years=1, step_months=3)
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2024-12-31")

    windows = generate_walk_forward_windows(start, end, config)

    # With 3-month steps, should get more windows than 6-month steps
    assert len(windows) > 5

    # Check that windows step by 3 months
    for i in range(len(windows) - 1):
        expected_next_start = windows[i].train_start + pd.DateOffset(months=3)
        assert windows[i + 1].train_start == expected_next_start


def test_generate_windows_exact_fit():
    """Test window generation when data exactly fits windows."""
    config = WalkForwardConfig(train_years=2, test_years=1, step_months=12)
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2024-01-01")  # Exactly 4 years

    windows = generate_walk_forward_windows(start, end, config)

    # Should get at least one complete window
    assert len(windows) >= 1


# ============================================================================
# Tests: analyze_walk_forward_results
# ============================================================================


def test_analyze_results_single_window(mock_window_result):
    """Test analysis with single window."""
    results = [mock_window_result]

    analysis = analyze_walk_forward_results(results)

    assert isinstance(analysis, WalkForwardAnalysis)
    assert analysis.summary_stats["num_windows"] == 1
    assert "is_sharpe_mean" in analysis.summary_stats
    assert "oos_sharpe_mean" in analysis.summary_stats
    assert "sharpe_degradation" in analysis.degradation_metrics


def test_analyze_results_multiple_windows(mock_window_result):
    """Test analysis with multiple windows."""
    # Create 3 windows with varying performance
    results = []
    for i in range(3):
        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01") + pd.DateOffset(months=i * 6),
            train_end=pd.Timestamp("2022-01-01") + pd.DateOffset(months=i * 6),
            test_start=pd.Timestamp("2022-01-01") + pd.DateOffset(months=i * 6),
            test_end=pd.Timestamp("2023-01-01") + pd.DateOffset(months=i * 6),
            window_id=i,
        )

        train_config = BacktestConfig(
            start_date=window.train_start,
            end_date=window.train_end,
            universe=["SPY"],
            initial_capital=100000.0,
        )
        test_config = BacktestConfig(
            start_date=window.test_start,
            end_date=window.test_end,
            universe=["SPY"],
            initial_capital=100000.0,
        )

        train_result = BacktestResult(
            equity_curve=pd.DataFrame({"nav": [100000, 110000]}),
            holdings_history=pd.DataFrame(),
            weights_history=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.5 + i * 0.1, "total_return": 0.15 + i * 0.05},
            config=train_config,
        )
        test_result = BacktestResult(
            equity_curve=pd.DataFrame({"nav": [100000, 105000]}),
            holdings_history=pd.DataFrame(),
            weights_history=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.0 + i * 0.1, "total_return": 0.10 + i * 0.03},
            config=test_config,
        )

        results.append(
            WalkForwardWindowResult(
                window=window,
                train_result=train_result,
                test_result=test_result,
                train_metrics=train_result.metrics,
                test_metrics=test_result.metrics,
            )
        )

    analysis = analyze_walk_forward_results(results)

    assert analysis.summary_stats["num_windows"] == 3
    assert analysis.summary_stats["is_sharpe_mean"] > analysis.summary_stats["oos_sharpe_mean"]
    assert analysis.degradation_metrics["sharpe_degradation"] > 0


def test_analyze_results_empty_list():
    """Test that empty results list raises error."""
    with pytest.raises(ValueError, match="No results to analyze"):
        analyze_walk_forward_results([])


def test_analyze_results_degradation_calculation(mock_window_result):
    """Test degradation calculation."""
    results = [mock_window_result]

    analysis = analyze_walk_forward_results(results)

    # Degradation should be IS - OOS
    expected_degradation = (
        mock_window_result.train_metrics["sharpe_ratio"]
        - mock_window_result.test_metrics["sharpe_ratio"]
    )
    assert abs(analysis.degradation_metrics["sharpe_degradation"] - expected_degradation) < 0.001


def test_analyze_results_handles_nan():
    """Test that analysis handles NaN values gracefully."""
    window = WalkForwardWindow(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2022-01-01"),
        test_start=pd.Timestamp("2022-01-01"),
        test_end=pd.Timestamp("2023-01-01"),
        window_id=0,
    )

    train_config = BacktestConfig(
        start_date=window.train_start,
        end_date=window.train_end,
        universe=["SPY"],
        initial_capital=100000.0,
    )
    test_config = BacktestConfig(
        start_date=window.test_start,
        end_date=window.test_end,
        universe=["SPY"],
        initial_capital=100000.0,
    )

    train_result = BacktestResult(
        equity_curve=pd.DataFrame(),
        holdings_history=pd.DataFrame(),
        weights_history=pd.DataFrame(),
        metrics={"sharpe_ratio": np.nan, "total_return": 0.10},
        config=train_config,
    )
    test_result = BacktestResult(
        equity_curve=pd.DataFrame(),
        holdings_history=pd.DataFrame(),
        weights_history=pd.DataFrame(),
        metrics={"sharpe_ratio": 1.2, "total_return": np.nan},
        config=test_config,
    )

    result = WalkForwardWindowResult(
        window=window,
        train_result=train_result,
        test_result=test_result,
        train_metrics=train_result.metrics,
        test_metrics=test_result.metrics,
    )

    # Should not raise, just handle NaN appropriately
    analysis = analyze_walk_forward_results([result])
    assert isinstance(analysis, WalkForwardAnalysis)


# ============================================================================
# Tests: create_walk_forward_summary_table
# ============================================================================


def test_summary_table_creation(mock_window_result):
    """Test summary table creation."""
    results = [mock_window_result]
    analysis = analyze_walk_forward_results(results)

    df = create_walk_forward_summary_table(analysis)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "window_id" in df.columns
    assert "train_sharpe" in df.columns
    assert "test_sharpe" in df.columns
    assert "sharpe_degradation" in df.columns


def test_summary_table_multiple_windows(mock_window_result):
    """Test summary table with multiple windows."""
    # Create 3 mock results
    results = []
    for i in range(3):
        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01") + pd.DateOffset(months=i * 6),
            train_end=pd.Timestamp("2022-01-01") + pd.DateOffset(months=i * 6),
            test_start=pd.Timestamp("2022-01-01") + pd.DateOffset(months=i * 6),
            test_end=pd.Timestamp("2023-01-01") + pd.DateOffset(months=i * 6),
            window_id=i,
        )
        train_config = BacktestConfig(
            start_date=window.train_start,
            end_date=window.train_end,
            universe=["SPY"],
            initial_capital=100000.0,
        )
        test_config = BacktestConfig(
            start_date=window.test_start,
            end_date=window.test_end,
            universe=["SPY"],
            initial_capital=100000.0,
        )
        train_result = BacktestResult(
            equity_curve=pd.DataFrame(),
            holdings_history=pd.DataFrame(),
            weights_history=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.5, "total_return": 0.15, "max_drawdown": -0.10},
            config=train_config,
        )
        test_result = BacktestResult(
            equity_curve=pd.DataFrame(),
            holdings_history=pd.DataFrame(),
            weights_history=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.2, "total_return": 0.12, "max_drawdown": -0.12},
            config=test_config,
        )
        results.append(
            WalkForwardWindowResult(
                window=window,
                train_result=train_result,
                test_result=test_result,
                train_metrics=train_result.metrics,
                test_metrics=test_result.metrics,
            )
        )

    analysis = analyze_walk_forward_results(results)
    df = create_walk_forward_summary_table(analysis)

    assert len(df) == 3
    assert df["window_id"].tolist() == [0, 1, 2]


# ============================================================================
# Tests: Integration (run_walk_forward_validation)
# ============================================================================


def test_run_walk_forward_integration(mock_snapshot_dir):
    """Test end-to-end walk-forward validation."""
    wf_config = WalkForwardConfig(train_years=2, test_years=1, step_months=12)

    strategy_params = {"top_n": 2, "lookback_days": 126, "cost_bps": 10.0}

    results = run_walk_forward_validation(
        snapshot_path=mock_snapshot_dir,
        start_date="2020-01-01",
        end_date="2024-12-31",
        wf_config=wf_config,
        strategy_params=strategy_params,
        initial_capital=100000.0,
    )

    # Should get at least one window
    assert len(results) >= 1
    assert all(isinstance(r, WalkForwardWindowResult) for r in results)

    # Each result should have train and test results
    for r in results:
        assert r.train_result is not None
        assert r.test_result is not None
        assert "sharpe_ratio" in r.train_metrics
        assert "sharpe_ratio" in r.test_metrics


def test_run_walk_forward_invalid_snapshot():
    """Test error handling for invalid snapshot path."""
    wf_config = WalkForwardConfig()
    strategy_params = {"top_n": 5, "lookback_days": 252}

    with pytest.raises(FileNotFoundError):
        run_walk_forward_validation(
            snapshot_path="/nonexistent/path",
            start_date="2020-01-01",
            end_date="2024-12-31",
            wf_config=wf_config,
            strategy_params=strategy_params,
        )


def test_run_walk_forward_date_conversion(mock_snapshot_dir):
    """Test that string dates are properly converted."""
    wf_config = WalkForwardConfig(train_years=2, test_years=1, step_months=12)
    strategy_params = {"top_n": 2, "lookback_days": 126}

    # Pass dates as strings
    results = run_walk_forward_validation(
        snapshot_path=mock_snapshot_dir,
        start_date="2020-01-01",  # String
        end_date="2024-12-31",  # String
        wf_config=wf_config,
        strategy_params=strategy_params,
    )

    assert len(results) >= 1


# ============================================================================
# Tests: Edge Cases
# ============================================================================


def test_stability_metrics_calculation(mock_window_result):
    """Test stability metrics are calculated correctly."""
    results = [mock_window_result]
    analysis = analyze_walk_forward_results(results)

    assert "is_sharpe_cv" in analysis.stability_metrics
    assert "oos_sharpe_cv" in analysis.stability_metrics
    assert "oos_sharpe_positive_pct" in analysis.stability_metrics


def test_positive_window_percentage(mock_window_result):
    """Test calculation of positive OOS window percentage."""
    # Create windows with mix of positive and negative OOS performance
    results = []
    for i in range(4):
        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2023-01-01"),
            window_id=i,
        )
        train_config = BacktestConfig(
            start_date=window.train_start,
            end_date=window.train_end,
            universe=["SPY"],
            initial_capital=100000.0,
        )
        test_config = BacktestConfig(
            start_date=window.test_start,
            end_date=window.test_end,
            universe=["SPY"],
            initial_capital=100000.0,
        )
        train_result = BacktestResult(
            equity_curve=pd.DataFrame(),
            holdings_history=pd.DataFrame(),
            weights_history=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.5, "total_return": 0.15},
            config=train_config,
        )
        # 2 positive, 2 negative OOS returns
        oos_return = 0.10 if i < 2 else -0.05
        test_result = BacktestResult(
            equity_curve=pd.DataFrame(),
            holdings_history=pd.DataFrame(),
            weights_history=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.0, "total_return": oos_return},
            config=test_config,
        )
        results.append(
            WalkForwardWindowResult(
                window=window,
                train_result=train_result,
                test_result=test_result,
                train_metrics=train_result.metrics,
                test_metrics=test_result.metrics,
            )
        )

    analysis = analyze_walk_forward_results(results)

    # Should be 50% positive (2 out of 4)
    assert analysis.degradation_metrics["pct_windows_oos_positive"] == 0.5
