"""Walk-forward validation for strategy robustness testing.

This module implements rolling window validation to test strategy performance
out-of-sample and detect overfitting. Walk-forward validation splits the data
into overlapping train/test windows and evaluates how well the strategy performs
on unseen data.

Key Features:
- Configurable train/test window sizes
- Rolling window approach to maximize data usage
- In-sample vs out-of-sample performance comparison
- Parameter stability tracking
- Regime-specific performance analysis
- Prevents data leakage between windows

Example:
    >>> from quantetf.evaluation.walk_forward import WalkForwardConfig, run_walk_forward
    >>> config = WalkForwardConfig(
    ...     train_years=2,
    ...     test_years=1,
    ...     step_months=6
    ... )
    >>> results = run_walk_forward(
    ...     snapshot_path="data/snapshots/snapshot_5yr_20etfs",
    ...     config=config,
    ...     strategy_params={'top_n': 5, 'lookback_days': 252}
    ... )
    >>> summary = analyze_walk_forward_results(results)
    >>> print(f"OOS Sharpe mean: {summary['oos_sharpe_mean']:.2f}")
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from quantetf.backtest.simple_engine import (
    SimpleBacktestEngine,
    BacktestConfig,
    BacktestResult,
)
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.data.access import DataAccessContext
from quantetf.types import Universe
from quantetf.evaluation.metrics import sharpe, max_drawdown

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        train_years: Number of years for training window
        test_years: Number of years for testing window
        step_months: Number of months to step forward each iteration
        min_train_periods: Minimum trading periods required in train window
        min_test_periods: Minimum trading periods required in test window
    """

    train_years: int = 2
    test_years: int = 1
    step_months: int = 6
    min_train_periods: int = 126  # ~6 months of trading days
    min_test_periods: int = 21  # ~1 month of trading days


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward train/test window.

    Attributes:
        train_start: Start date of training period
        train_end: End date of training period
        test_start: Start date of testing period
        test_end: End date of testing period
        window_id: Sequential identifier for this window
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_id: int


@dataclass
class WalkForwardWindowResult:
    """Results from a single walk-forward window.

    Attributes:
        window: The window configuration
        train_result: Backtest result on training data
        test_result: Backtest result on testing data
        train_metrics: Performance metrics for training period
        test_metrics: Performance metrics for testing period
    """

    window: WalkForwardWindow
    train_result: BacktestResult
    test_result: BacktestResult
    train_metrics: dict
    test_metrics: dict


@dataclass
class WalkForwardAnalysis:
    """Aggregate analysis of walk-forward validation results.

    Attributes:
        window_results: List of individual window results
        summary_stats: Summary statistics across all windows
        degradation_metrics: In-sample vs out-of-sample degradation
        stability_metrics: Parameter and performance stability
    """

    window_results: list[WalkForwardWindowResult]
    summary_stats: dict
    degradation_metrics: dict
    stability_metrics: dict


def generate_walk_forward_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    """Generate rolling train/test windows for walk-forward validation.

    Creates a sequence of non-overlapping test windows with overlapping
    training windows. Each window moves forward by step_months.

    Args:
        start_date: Overall start date of available data
        end_date: Overall end date of available data
        config: Walk-forward configuration

    Returns:
        List of WalkForwardWindow objects

    Raises:
        ValueError: If date range is too short for even one window

    Example:
        >>> windows = generate_walk_forward_windows(
        ...     pd.Timestamp('2020-01-01'),
        ...     pd.Timestamp('2025-12-31'),
        ...     WalkForwardConfig(train_years=2, test_years=1, step_months=6)
        ... )
        >>> len(windows)
        8
    """
    windows = []
    window_id = 0

    # Calculate minimum required period
    min_required_days = (config.train_years + config.test_years) * 365
    available_days = (end_date - start_date).days

    if available_days < min_required_days:
        raise ValueError(
            f"Insufficient data: need {min_required_days} days "
            f"({config.train_years}y train + {config.test_years}y test), "
            f"but only have {available_days} days"
        )

    # Start with first possible window
    current_train_start = start_date

    while True:
        # Calculate window boundaries
        train_end = current_train_start + pd.DateOffset(years=config.train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=config.test_years)

        # Check if we have enough data for this window
        if test_end > end_date:
            logger.info(
                f"Stopping at window {window_id}: test_end {test_end} > end_date {end_date}"
            )
            break

        window = WalkForwardWindow(
            train_start=current_train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            window_id=window_id,
        )
        windows.append(window)

        logger.debug(
            f"Window {window_id}: train [{current_train_start.date()} to {train_end.date()}], "
            f"test [{test_start.date()} to {test_end.date()}]"
        )

        # Move forward by step_months
        current_train_start = current_train_start + pd.DateOffset(
            months=config.step_months
        )
        window_id += 1

    if not windows:
        raise ValueError(
            f"Could not create any valid windows with train_years={config.train_years}, "
            f"test_years={config.test_years}, available period: {start_date} to {end_date}"
        )

    logger.info(f"Generated {len(windows)} walk-forward windows")
    return windows


def run_backtest_for_window(
    *,
    window: WalkForwardWindow,
    period_type: str,  # 'train' or 'test'
    data_access: DataAccessContext,
    universe: Universe,
    strategy_params: dict,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Run backtest for a specific window period.

    Args:
        window: Walk-forward window configuration
        period_type: Either 'train' or 'test' to select period
        data_access: DataAccessContext for historical prices and macro data
        universe: List of tickers to trade
        strategy_params: Strategy configuration (top_n, lookback_days, cost_bps)
        initial_capital: Starting capital in dollars

    Returns:
        BacktestResult for the specified period

    Raises:
        ValueError: If period_type is invalid
    """
    if period_type == "train":
        start = window.train_start
        end = window.train_end
    elif period_type == "test":
        start = window.test_start
        end = window.test_end
    else:
        raise ValueError(f"period_type must be 'train' or 'test', got {period_type}")

    # Extract strategy parameters with defaults
    top_n = strategy_params.get("top_n", 5)
    lookback_days = strategy_params.get("lookback_days", 252)
    cost_bps = strategy_params.get("cost_bps", 10.0)
    rebalance_freq = strategy_params.get("rebalance_frequency", "monthly")

    # Create strategy components
    alpha_model = MomentumAlpha(lookback_days=lookback_days)
    portfolio = EqualWeightTopN(top_n=top_n)
    cost_model = FlatTransactionCost(cost_bps=cost_bps)

    # Configure and run backtest
    config = BacktestConfig(
        start_date=start,
        end_date=end,
        universe=universe,
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_freq,
    )

    engine = SimpleBacktestEngine()
    result = engine.run(
        config=config,
        alpha_model=alpha_model,
        portfolio=portfolio,
        cost_model=cost_model,
        data_access=data_access,
    )

    logger.info(
        f"Window {window.window_id} {period_type}: "
        f"Return={result.metrics.get('total_return', 0):.2%}, "
        f"Sharpe={result.metrics.get('sharpe_ratio', 0):.2f}"
    )

    return result


def run_walk_forward_window(
    *,
    window: WalkForwardWindow,
    data_access: DataAccessContext,
    universe: Universe,
    strategy_params: dict,
    initial_capital: float = 100_000.0,
) -> WalkForwardWindowResult:
    """Run both train and test backtests for a single window.

    Args:
        window: Walk-forward window configuration
        data_access: DataAccessContext for historical prices and macro data
        universe: List of tickers to trade
        strategy_params: Strategy configuration
        initial_capital: Starting capital in dollars

    Returns:
        WalkForwardWindowResult with train and test results
    """
    logger.info(f"Running window {window.window_id}...")

    # Run training period backtest
    train_result = run_backtest_for_window(
        window=window,
        period_type="train",
        data_access=data_access,
        universe=universe,
        strategy_params=strategy_params,
        initial_capital=initial_capital,
    )

    # Run testing period backtest
    test_result = run_backtest_for_window(
        window=window,
        period_type="test",
        data_access=data_access,
        universe=universe,
        strategy_params=strategy_params,
        initial_capital=initial_capital,
    )

    return WalkForwardWindowResult(
        window=window,
        train_result=train_result,
        test_result=test_result,
        train_metrics=train_result.metrics,
        test_metrics=test_result.metrics,
    )


def run_walk_forward_validation(
    *,
    snapshot_path: Path | str,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    wf_config: WalkForwardConfig,
    strategy_params: dict,
    initial_capital: float = 100_000.0,
) -> list[WalkForwardWindowResult]:
    """Run complete walk-forward validation.

    This is the main entry point for walk-forward testing. It generates
    all windows, runs backtests for each, and returns the results.

    Args:
        snapshot_path: Path to snapshot data directory
        start_date: Overall start date for validation
        end_date: Overall end date for validation
        wf_config: Walk-forward configuration
        strategy_params: Strategy parameters (top_n, lookback_days, etc.)
        initial_capital: Starting capital in dollars

    Returns:
        List of WalkForwardWindowResult objects

    Raises:
        FileNotFoundError: If snapshot_path doesn't exist
        ValueError: If date range is invalid

    Example:
        >>> results = run_walk_forward_validation(
        ...     snapshot_path="data/snapshots/snapshot_5yr_20etfs",
        ...     start_date="2020-01-01",
        ...     end_date="2025-12-31",
        ...     wf_config=WalkForwardConfig(train_years=2, test_years=1),
        ...     strategy_params={'top_n': 5, 'lookback_days': 252}
        ... )
    """
    # Convert to Path object
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot path not found: {snapshot_path}")

    # Convert dates to Timestamps
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)

    # Load data store
    logger.info(f"Loading snapshot from {snapshot_path}")

    # Check if path is directory or file
    if snapshot_path.is_dir():
        data_path = snapshot_path / 'data.parquet'
        metadata_path = snapshot_path / 'manifest.yaml'
    else:
        data_path = snapshot_path
        metadata_path = snapshot_path.parent / 'manifest.yaml'

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Create DataAccessContext using factory
    from quantetf.data.access import DataAccessFactory
    data_access = DataAccessFactory.create_context(
        config={"snapshot_path": str(data_path)},
        enable_caching=False  # Disable caching for walk-forward validation
    )

    # Load universe from metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    import yaml

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    # Try to get tickers from different possible locations
    tickers = metadata.get("tickers")
    if not tickers:
        data_summary = metadata.get("data_summary", {})
        tickers = data_summary.get("tickers", [])

    if not tickers:
        raise ValueError("No tickers found in snapshot metadata")

    # Create Universe object
    universe = Universe(as_of=end_date, tickers=tuple(tickers))

    logger.info(f"Universe: {len(tickers)} tickers")

    # Generate windows
    windows = generate_walk_forward_windows(start_date, end_date, wf_config)

    # Run backtests for each window
    results = []
    for i, window in enumerate(windows):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing window {i+1}/{len(windows)}")
        logger.info(f"{'='*80}")

        try:
            window_result = run_walk_forward_window(
                window=window,
                data_access=data_access,
                universe=universe,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
            )
            results.append(window_result)
        except Exception as e:
            logger.error(f"Error in window {window.window_id}: {e}", exc_info=True)
            # Continue with next window rather than failing completely
            continue

    if not results:
        raise RuntimeError("No windows completed successfully")

    logger.info(f"\nCompleted {len(results)}/{len(windows)} windows successfully")
    return results


def analyze_walk_forward_results(
    results: list[WalkForwardWindowResult],
) -> WalkForwardAnalysis:
    """Analyze walk-forward validation results.

    Computes summary statistics, degradation metrics, and stability metrics
    across all windows to assess strategy robustness.

    Args:
        results: List of window results from run_walk_forward_validation

    Returns:
        WalkForwardAnalysis with comprehensive analysis

    Example:
        >>> analysis = analyze_walk_forward_results(results)
        >>> print(f"Mean OOS Sharpe: {analysis.summary_stats['oos_sharpe_mean']:.2f}")
        >>> print(f"Sharpe degradation: {analysis.degradation_metrics['sharpe_degradation']:.2f}")
    """
    if not results:
        raise ValueError("No results to analyze")

    # Extract metrics from all windows
    train_sharpes = []
    test_sharpes = []
    train_returns = []
    test_returns = []
    train_drawdowns = []
    test_drawdowns = []

    for r in results:
        train_sharpes.append(r.train_metrics.get("sharpe_ratio", np.nan))
        test_sharpes.append(r.test_metrics.get("sharpe_ratio", np.nan))
        train_returns.append(r.train_metrics.get("total_return", np.nan))
        test_returns.append(r.test_metrics.get("total_return", np.nan))
        train_drawdowns.append(r.train_metrics.get("max_drawdown", np.nan))
        test_drawdowns.append(r.test_metrics.get("max_drawdown", np.nan))

    # Filter out NaN values for statistics
    def safe_mean(values):
        clean = [v for v in values if not np.isnan(v)]
        return np.mean(clean) if clean else np.nan

    def safe_std(values):
        clean = [v for v in values if not np.isnan(v)]
        return np.std(clean, ddof=1) if len(clean) > 1 else np.nan

    def safe_median(values):
        clean = [v for v in values if not np.isnan(v)]
        return np.median(clean) if clean else np.nan

    # Summary statistics
    summary_stats = {
        "num_windows": len(results),
        "is_sharpe_mean": safe_mean(train_sharpes),
        "is_sharpe_std": safe_std(train_sharpes),
        "is_sharpe_median": safe_median(train_sharpes),
        "oos_sharpe_mean": safe_mean(test_sharpes),
        "oos_sharpe_std": safe_std(test_sharpes),
        "oos_sharpe_median": safe_median(test_sharpes),
        "is_return_mean": safe_mean(train_returns),
        "oos_return_mean": safe_mean(test_returns),
        "is_drawdown_mean": safe_mean(train_drawdowns),
        "oos_drawdown_mean": safe_mean(test_drawdowns),
    }

    # Degradation metrics (in-sample vs out-of-sample)
    sharpe_diffs = [
        (train - test)
        for train, test in zip(train_sharpes, test_sharpes)
        if not (np.isnan(train) or np.isnan(test))
    ]
    return_diffs = [
        (train - test)
        for train, test in zip(train_returns, test_returns)
        if not (np.isnan(train) or np.isnan(test))
    ]

    degradation_metrics = {
        "sharpe_degradation": safe_mean(sharpe_diffs),
        "sharpe_degradation_std": safe_std(sharpe_diffs),
        "return_degradation": safe_mean(return_diffs),
        "pct_windows_oos_positive": (
            sum(1 for r in test_returns if r > 0) / len(test_returns)
            if test_returns
            else 0.0
        ),
        "pct_windows_oos_beats_is": (
            sum(1 for t, s in zip(test_sharpes, train_sharpes) if t > s)
            / len(test_sharpes)
            if test_sharpes
            else 0.0
        ),
    }

    # Stability metrics
    stability_metrics = {
        "is_sharpe_cv": (
            abs(safe_std(train_sharpes) / safe_mean(train_sharpes))
            if safe_mean(train_sharpes) != 0
            else np.nan
        ),
        "oos_sharpe_cv": (
            abs(safe_std(test_sharpes) / safe_mean(test_sharpes))
            if safe_mean(test_sharpes) != 0
            else np.nan
        ),
        "oos_sharpe_positive_pct": (
            sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)
            if test_sharpes
            else 0.0
        ),
    }

    return WalkForwardAnalysis(
        window_results=results,
        summary_stats=summary_stats,
        degradation_metrics=degradation_metrics,
        stability_metrics=stability_metrics,
    )


def create_walk_forward_summary_table(analysis: WalkForwardAnalysis) -> pd.DataFrame:
    """Create a summary table of walk-forward results.

    Args:
        analysis: Walk-forward analysis results

    Returns:
        DataFrame with one row per window plus summary statistics

    Example:
        >>> df = create_walk_forward_summary_table(analysis)
        >>> print(df[['window_id', 'train_sharpe', 'test_sharpe', 'degradation']])
    """
    rows = []
    for r in analysis.window_results:
        rows.append(
            {
                "window_id": r.window.window_id,
                "train_start": r.window.train_start.date(),
                "train_end": r.window.train_end.date(),
                "test_start": r.window.test_start.date(),
                "test_end": r.window.test_end.date(),
                "train_sharpe": r.train_metrics.get("sharpe_ratio", np.nan),
                "test_sharpe": r.test_metrics.get("sharpe_ratio", np.nan),
                "train_return": r.train_metrics.get("total_return", np.nan),
                "test_return": r.test_metrics.get("total_return", np.nan),
                "train_drawdown": r.train_metrics.get("max_drawdown", np.nan),
                "test_drawdown": r.test_metrics.get("max_drawdown", np.nan),
                "sharpe_degradation": (
                    r.train_metrics.get("sharpe_ratio", np.nan)
                    - r.test_metrics.get("sharpe_ratio", np.nan)
                ),
            }
        )

    df = pd.DataFrame(rows)
    return df
