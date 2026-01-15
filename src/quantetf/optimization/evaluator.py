"""Multi-period strategy evaluator.

This module provides evaluation of strategy configurations across multiple time
periods (3yr, 5yr, 10yr) to determine if a strategy consistently beats SPY.

A strategy "beats SPY" when it achieves:
1. Positive excess return (strategy return > SPY return)
2. Information Ratio > 0 (positive risk-adjusted excess return)
3. Consistent across ALL evaluated periods

Example:
    >>> from quantetf.optimization.evaluator import MultiPeriodEvaluator
    >>> from quantetf.optimization.grid import generate_configs
    >>> from quantetf.data.snapshot_store import SnapshotDataStore
    >>> from pathlib import Path
    >>>
    >>> store = SnapshotDataStore(Path("data/snapshots/snapshot_20260113/data.parquet"))
    >>> evaluator = MultiPeriodEvaluator(store)
    >>> config = generate_configs()[0]
    >>> result = evaluator.evaluate(config)
    >>> print(f"Beats SPY: {result.beats_spy_all_periods}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from quantetf.alpha.factory import create_alpha_model
from quantetf.backtest.simple_engine import BacktestConfig, BacktestResult, SimpleBacktestEngine
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.evaluation.metrics import (
    calculate_active_metrics,
    max_drawdown,
    sharpe,
)
from quantetf.optimization.grid import StrategyConfig
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.types import Universe

logger = logging.getLogger(__name__)


@dataclass
class PeriodMetrics:
    """Metrics for a single evaluation period.

    Captures all relevant performance metrics for a strategy evaluated
    over a specific time period (e.g., 3yr, 5yr, 10yr).

    Attributes:
        period_name: Human-readable period identifier (e.g., '3yr', '5yr')
        start_date: Start date of the evaluation period
        end_date: End date of the evaluation period
        strategy_return: Total cumulative return of the strategy
        spy_return: Total cumulative return of SPY benchmark
        active_return: Excess return (strategy - SPY)
        strategy_volatility: Annualized volatility of strategy returns
        tracking_error: Annualized volatility of excess returns
        information_ratio: Active return / tracking error
        max_drawdown: Maximum peak-to-trough decline
        sharpe_ratio: Risk-adjusted return (annualized)
        num_rebalances: Number of portfolio rebalances in period
        evaluation_success: Whether evaluation completed without errors
    """

    period_name: str
    start_date: datetime
    end_date: datetime
    strategy_return: float
    spy_return: float
    active_return: float
    strategy_volatility: float
    tracking_error: float
    information_ratio: float
    max_drawdown: float
    sharpe_ratio: float
    num_rebalances: int = 0
    evaluation_success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'period_name': self.period_name,
            'start_date': self.start_date.isoformat() if isinstance(self.start_date, datetime) else str(self.start_date),
            'end_date': self.end_date.isoformat() if isinstance(self.end_date, datetime) else str(self.end_date),
            'strategy_return': self.strategy_return,
            'spy_return': self.spy_return,
            'active_return': self.active_return,
            'strategy_volatility': self.strategy_volatility,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'num_rebalances': self.num_rebalances,
            'evaluation_success': self.evaluation_success,
        }


@dataclass
class MultiPeriodResult:
    """Results across all evaluation periods for a strategy configuration.

    Aggregates metrics from multiple time periods and provides summary
    statistics for ranking and comparison.

    Attributes:
        config_name: Unique identifier for the strategy configuration
        config: The StrategyConfig that was evaluated
        periods: Dictionary mapping period names to PeriodMetrics
        beats_spy_all_periods: True if strategy beats SPY in ALL periods
        composite_score: Combined score for ranking strategies
        error_message: Error message if evaluation failed, None otherwise
    """

    config_name: str
    config: Optional[StrategyConfig]
    periods: Dict[str, PeriodMetrics]
    beats_spy_all_periods: bool
    composite_score: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export.

        Returns a flat dictionary suitable for DataFrame creation with
        columns for each period's metrics.
        """
        result: Dict[str, Any] = {
            'config_name': self.config_name,
            'beats_spy_all_periods': self.beats_spy_all_periods,
            'composite_score': self.composite_score,
            'error_message': self.error_message,
        }

        # Add per-period metrics with prefixed column names
        for period_name, metrics in self.periods.items():
            # Convert '3yr' -> '3y' for shorter column names
            prefix = period_name.replace('yr', 'y')
            result[f'{prefix}_strategy_return'] = metrics.strategy_return
            result[f'{prefix}_spy_return'] = metrics.spy_return
            result[f'{prefix}_active_return'] = metrics.active_return
            result[f'{prefix}_information_ratio'] = metrics.information_ratio
            result[f'{prefix}_sharpe_ratio'] = metrics.sharpe_ratio
            result[f'{prefix}_max_drawdown'] = metrics.max_drawdown
            result[f'{prefix}_tracking_error'] = metrics.tracking_error
            result[f'{prefix}_num_rebalances'] = metrics.num_rebalances

        return result

    def summary(self) -> str:
        """Return a human-readable summary of the result."""
        lines = [
            f"Strategy: {self.config_name}",
            f"Beats SPY (all periods): {self.beats_spy_all_periods}",
            f"Composite Score: {self.composite_score:.4f}",
            "",
        ]

        for period_name, metrics in sorted(self.periods.items()):
            lines.append(f"  {period_name}:")
            lines.append(f"    Strategy Return: {metrics.strategy_return:+.2%}")
            lines.append(f"    SPY Return:      {metrics.spy_return:+.2%}")
            lines.append(f"    Active Return:   {metrics.active_return:+.2%}")
            lines.append(f"    Info Ratio:      {metrics.information_ratio:.3f}")
            lines.append(f"    Sharpe Ratio:    {metrics.sharpe_ratio:.3f}")
            lines.append("")

        if self.error_message:
            lines.append(f"Error: {self.error_message}")

        return "\n".join(lines)


class MultiPeriodEvaluator:
    """Evaluates strategies across multiple time periods.

    This class runs backtests for a strategy configuration across multiple
    time windows (default: 3yr, 5yr, 10yr) ending at a common end date.
    It calculates comprehensive metrics and determines if the strategy
    consistently beats the SPY benchmark.

    Example:
        >>> store = SnapshotDataStore(Path("data/snapshots/snapshot/data.parquet"))
        >>> evaluator = MultiPeriodEvaluator(store)
        >>> result = evaluator.evaluate(config, periods_years=[3, 5, 10])
        >>> print(f"Composite score: {result.composite_score:.3f}")
    """

    def __init__(
        self,
        snapshot_path: Union[str, Path],
        end_date: Optional[pd.Timestamp] = None,
        cost_bps: float = 10.0,
    ):
        """Initialize the evaluator.

        Args:
            snapshot_path: Path to the snapshot data.parquet file
            end_date: End date for all evaluation periods (defaults to snapshot end)
            cost_bps: Transaction cost in basis points (default: 10)
        """
        self.snapshot_path = Path(snapshot_path)
        self.store = SnapshotDataStore(self.snapshot_path)
        self.cost_bps = cost_bps

        # Determine end date from snapshot if not provided
        if end_date is not None:
            self.end_date = pd.Timestamp(end_date)
        else:
            _, snapshot_end = self.store.date_range
            self.end_date = pd.Timestamp(snapshot_end)

        logger.info(f"MultiPeriodEvaluator initialized: end_date={self.end_date}")

    def evaluate(
        self,
        config: StrategyConfig,
        periods_years: Optional[List[int]] = None,
    ) -> MultiPeriodResult:
        """Evaluate strategy across multiple time periods.

        Args:
            config: Strategy configuration to evaluate
            periods_years: List of lookback periods in years (default: [3, 5, 10])

        Returns:
            MultiPeriodResult with metrics for each period and summary statistics
        """
        if periods_years is None:
            periods_years = [3, 5, 10]

        config_name = config.generate_name()
        logger.info(f"Evaluating config: {config_name}")
        logger.info(f"Periods: {periods_years} years")

        period_results: Dict[str, PeriodMetrics] = {}

        for years in periods_years:
            period_name = f'{years}yr'
            try:
                metrics = self._evaluate_period(config, years)
                period_results[period_name] = metrics
                logger.info(
                    f"  {period_name}: active_return={metrics.active_return:+.2%}, "
                    f"IR={metrics.information_ratio:.3f}"
                )
            except Exception as e:
                logger.warning(f"Failed to evaluate {period_name} for {config_name}: {e}")
                period_results[period_name] = self._create_failed_metrics(period_name, years, str(e))

        # Determine if strategy beats SPY in all periods
        beats_spy = self._beats_spy_all_periods(period_results)

        # Calculate composite score for ranking
        composite_score = self._calculate_composite_score(period_results)

        return MultiPeriodResult(
            config_name=config_name,
            config=config,
            periods=period_results,
            beats_spy_all_periods=beats_spy,
            composite_score=composite_score,
        )

    def _evaluate_period(self, config: StrategyConfig, years: int) -> PeriodMetrics:
        """Evaluate strategy for a single time period.

        Args:
            config: Strategy configuration
            years: Number of years for this period

        Returns:
            PeriodMetrics with all calculated metrics
        """
        period_name = f'{years}yr'

        # Calculate start date (account for warmup period)
        # Add extra days for warmup based on longest lookback
        warmup_days = self._get_warmup_days(config)
        start_date = self.end_date - pd.Timedelta(days=years * 365 + warmup_days)

        logger.debug(f"Period {period_name}: {start_date} to {self.end_date}")

        # Get tickers from universe (simplified - using stored tickers)
        tickers = self.store.tickers

        # Create backtest components
        alpha_config = {
            'type': config.alpha_type,
            **config.alpha_params,
        }
        alpha_model = create_alpha_model(alpha_config)
        portfolio_constructor = EqualWeightTopN(top_n=config.top_n)
        cost_model = FlatTransactionCost(cost_bps=self.cost_bps)

        # Create universe
        universe = Universe(as_of=self.end_date, tickers=tuple(tickers))

        # Create backtest config
        backtest_config = BacktestConfig(
            start_date=start_date,
            end_date=self.end_date,
            universe=universe,
            initial_capital=100_000.0,
            rebalance_frequency=config.schedule_name,
        )

        # Run backtest
        engine = SimpleBacktestEngine()
        result = engine.run(
            config=backtest_config,
            alpha_model=alpha_model,
            portfolio=portfolio_constructor,
            cost_model=cost_model,
            store=self.store,
        )

        # Calculate SPY benchmark returns for same period
        spy_returns = self._get_spy_returns(result.equity_curve.index)

        # Calculate strategy returns from equity curve
        strategy_returns = result.equity_curve['returns'].dropna()

        # Align returns
        common_dates = strategy_returns.index.intersection(spy_returns.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between strategy and SPY")

        aligned_strategy = strategy_returns.loc[common_dates]
        aligned_spy = spy_returns.loc[common_dates]

        # Calculate active metrics
        active_metrics = calculate_active_metrics(
            strategy_returns=aligned_strategy,
            benchmark_returns=aligned_spy,
            periods_per_year=self._get_periods_per_year(config.schedule_name),
        )

        # Calculate cumulative returns
        strategy_total_return = (1 + aligned_strategy).prod() - 1
        spy_total_return = (1 + aligned_spy).prod() - 1

        return PeriodMetrics(
            period_name=period_name,
            start_date=start_date.to_pydatetime(),
            end_date=self.end_date.to_pydatetime(),
            strategy_return=float(strategy_total_return),
            spy_return=float(spy_total_return),
            active_return=float(strategy_total_return - spy_total_return),
            strategy_volatility=active_metrics.get('strategy_volatility',
                                                    float(aligned_strategy.std() * np.sqrt(252))),
            tracking_error=active_metrics['tracking_error'],
            information_ratio=active_metrics['information_ratio'],
            max_drawdown=result.metrics['max_drawdown'],
            sharpe_ratio=active_metrics['strategy_sharpe'],
            num_rebalances=result.metrics['num_rebalances'],
            evaluation_success=True,
        )

    def _get_spy_returns(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Get SPY returns for the given dates.

        Args:
            dates: DatetimeIndex of dates to get returns for

        Returns:
            Series of SPY returns aligned to dates
        """
        # Get SPY prices for entire available range
        start_date = dates.min()
        end_date = dates.max() + pd.Timedelta(days=1)

        try:
            spy_prices = self.store.get_close_prices(
                as_of=end_date,
                tickers=['SPY'],
            )
        except (ValueError, KeyError):
            # SPY might not be in snapshot, create synthetic benchmark
            logger.warning("SPY not found in snapshot, using equal-weight proxy")
            all_prices = self.store.get_close_prices(as_of=end_date)
            spy_prices = all_prices.mean(axis=1).to_frame('SPY')

        # Calculate returns
        spy_returns = spy_prices['SPY'].pct_change().dropna()

        # Filter to dates range
        spy_returns = spy_returns[spy_returns.index >= start_date]

        return spy_returns

    def _get_warmup_days(self, config: StrategyConfig) -> int:
        """Calculate warmup period needed for alpha model.

        Args:
            config: Strategy configuration

        Returns:
            Number of warmup days needed
        """
        params = config.alpha_params

        if config.alpha_type == 'momentum_acceleration':
            lookback = params.get('long_lookback_days', 252)
        else:
            lookback = params.get('lookback_days', 252)

        # Add buffer for sufficient data
        return int(lookback * 1.5)

    def _get_periods_per_year(self, schedule_name: str) -> int:
        """Get number of rebalance periods per year.

        Args:
            schedule_name: 'weekly' or 'monthly'

        Returns:
            Number of periods per year (52 for weekly, 12 for monthly)
        """
        if schedule_name == 'weekly':
            return 52
        elif schedule_name == 'monthly':
            return 12
        else:
            return 12  # Default to monthly

    def _create_failed_metrics(
        self,
        period_name: str,
        years: int,
        error_msg: str,
    ) -> PeriodMetrics:
        """Create metrics for a failed evaluation.

        Args:
            period_name: Name of the period (e.g., '3yr')
            years: Number of years
            error_msg: Error message describing the failure

        Returns:
            PeriodMetrics with failure indicators
        """
        start_date = self.end_date - pd.Timedelta(days=years * 365)

        return PeriodMetrics(
            period_name=period_name,
            start_date=start_date.to_pydatetime(),
            end_date=self.end_date.to_pydatetime(),
            strategy_return=float('-inf'),
            spy_return=0.0,
            active_return=float('-inf'),
            strategy_volatility=float('inf'),
            tracking_error=float('inf'),
            information_ratio=float('-inf'),
            max_drawdown=-1.0,
            sharpe_ratio=float('-inf'),
            num_rebalances=0,
            evaluation_success=False,
        )

    def _beats_spy_all_periods(self, periods: Dict[str, PeriodMetrics]) -> bool:
        """Check if strategy beats SPY in ALL periods.

        A strategy beats SPY if it has:
        1. Positive active return (strategy > SPY)
        2. Positive information ratio

        Args:
            periods: Dictionary of period metrics

        Returns:
            True if strategy beats SPY in all periods
        """
        for period_name, metrics in periods.items():
            # Skip failed evaluations
            if not metrics.evaluation_success:
                return False

            # Must have positive active return
            if metrics.active_return <= 0:
                logger.debug(f"{period_name}: Failed - negative active return")
                return False

            # Must have positive information ratio
            if metrics.information_ratio <= 0:
                logger.debug(f"{period_name}: Failed - negative IR")
                return False

        return True

    def _calculate_composite_score(self, periods: Dict[str, PeriodMetrics]) -> float:
        """Calculate composite score for ranking strategies.

        Score formula:
            composite = avg(IR) - consistency_penalty + winner_bonus

        Where:
        - avg(IR): Average information ratio across all periods
        - consistency_penalty: 0.5 * std(IR) penalizes volatile performance
        - winner_bonus: +0.5 if strategy beats SPY in all periods

        Args:
            periods: Dictionary of period metrics

        Returns:
            Composite score (higher is better)
        """
        # Collect valid IRs
        irs = [
            m.information_ratio
            for m in periods.values()
            if m.evaluation_success and m.information_ratio != float('-inf')
        ]

        if not irs:
            return float('-inf')

        avg_ir = sum(irs) / len(irs)

        # Consistency penalty (std of IRs)
        if len(irs) > 1:
            ir_variance = sum((ir - avg_ir) ** 2 for ir in irs) / len(irs)
            ir_std = ir_variance ** 0.5
            consistency_penalty = ir_std * 0.5
        else:
            consistency_penalty = 0.0

        # Winner bonus
        beats_all = all(
            m.evaluation_success and m.active_return > 0 and m.information_ratio > 0
            for m in periods.values()
        )
        winner_bonus = 0.5 if beats_all else 0.0

        composite = avg_ir - consistency_penalty + winner_bonus

        logger.debug(
            f"Composite score: avg_IR={avg_ir:.3f}, penalty={consistency_penalty:.3f}, "
            f"bonus={winner_bonus:.1f}, total={composite:.3f}"
        )

        return composite
