"""Multi-period strategy evaluator.

This module provides evaluation of strategy configurations across multiple time
periods (3yr, 5yr, 10yr) to determine if a strategy consistently beats SPY.

A strategy "beats SPY" when it achieves:
1. Positive excess return (strategy return > SPY return)
2. Information Ratio > 0 (positive risk-adjusted excess return)
3. Consistent across ALL evaluated periods

Uses DataAccessContext (DAL) for all data access, enabling:
- Decoupling from specific data storage implementations
- Easy mocking in tests
- Transparent caching

Example:
    >>> from quantetf.optimization.evaluator import MultiPeriodEvaluator
    >>> from quantetf.optimization.grid import generate_configs
    >>> from quantetf.data.access import DataAccessFactory
    >>>
    >>> ctx = DataAccessFactory.create_context(
    ...     config={"snapshot_path": "data/snapshots/snapshot_20260113/data.parquet"}
    ... )
    >>> evaluator = MultiPeriodEvaluator(data_access=ctx)
    >>> config = generate_configs()[0]
    >>> result = evaluator.evaluate(config)
    >>> print(f"Beats SPY: {result.beats_spy_all_periods}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantetf.alpha.factory import create_alpha_model
from quantetf.backtest.simple_engine import BacktestConfig, BacktestResult, SimpleBacktestEngine
from quantetf.data.access import DataAccessContext
from quantetf.evaluation.metrics import (
    calculate_active_metrics,
    max_drawdown,
    sharpe,
)
from quantetf.optimization.grid import StrategyConfig
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.types import RegimeMetrics, Universe

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
        daily_returns: Optional daily returns series for regime analysis
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
    daily_returns: Optional[pd.Series] = None

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

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/snapshot/data.parquet"}
        ... )
        >>> evaluator = MultiPeriodEvaluator(data_access=ctx)
        >>> result = evaluator.evaluate(config, periods_years=[3, 5, 10])
        >>> print(f"Composite score: {result.composite_score:.3f}")
    """

    def __init__(
        self,
        data_access: DataAccessContext,
        end_date: Optional[pd.Timestamp] = None,
        cost_bps: float = 10.0,
    ):
        """Initialize the evaluator.

        Args:
            data_access: DataAccessContext for historical prices and macro data
            end_date: End date for all evaluation periods (defaults to latest price date)
            cost_bps: Transaction cost in basis points (default: 10)
        """
        self.data_access = data_access
        self.cost_bps = cost_bps

        # Determine end date from data accessor if not provided
        if end_date is not None:
            self.end_date = pd.Timestamp(end_date)
        else:
            latest_date = self.data_access.prices.get_latest_price_date()
            self.end_date = pd.Timestamp(latest_date)

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

        # Get tickers from price accessor
        tickers = self.data_access.prices.get_available_tickers()

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
            data_access=self.data_access,
        )

        # Calculate strategy returns from equity curve
        strategy_returns = result.equity_curve['returns'].dropna()

        # Get the actual date range from strategy returns
        eval_start = strategy_returns.index.min()
        eval_end = strategy_returns.index.max()

        # Calculate SPY return using prices (not sparse aligned daily returns)
        # This fixes BUG-001: SPY return was incorrectly calculated using only
        # returns on rebalance dates, missing ~250 of ~260 trading days per year
        spy_prices = self._get_spy_prices(eval_start, eval_end)
        if len(spy_prices) < 2:
            raise ValueError("Insufficient SPY price data for return calculation")
        spy_total_return = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1

        # Get SPY returns for active metrics calculation (tracking error, IR)
        # These metrics still need aligned returns at rebalance frequency
        spy_returns = self._get_spy_returns(result.equity_curve.index)
        common_dates = strategy_returns.index.intersection(spy_returns.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between strategy and SPY")

        aligned_strategy = strategy_returns.loc[common_dates]
        aligned_spy = spy_returns.loc[common_dates]

        # Calculate active metrics using aligned returns at rebalance frequency
        active_metrics = calculate_active_metrics(
            strategy_returns=aligned_strategy,
            benchmark_returns=aligned_spy,
            periods_per_year=self._get_periods_per_year(config.schedule_name),
        )

        # Strategy total return from compounded strategy returns
        strategy_total_return = (1 + aligned_strategy).prod() - 1

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
            daily_returns=strategy_returns,  # Store for regime analysis
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
            ohlcv_data = self.data_access.prices.read_prices_as_of(
                as_of=end_date,
                tickers=['SPY'],
            )
            spy_prices = ohlcv_data.xs('Close', level='Price', axis=1)
        except (ValueError, KeyError):
            # SPY might not be in snapshot, create synthetic benchmark
            logger.warning("SPY not found in data, using equal-weight proxy")
            ohlcv_data = self.data_access.prices.read_prices_as_of(as_of=end_date)
            all_prices = ohlcv_data.xs('Close', level='Price', axis=1)
            spy_prices = all_prices.mean(axis=1).to_frame('SPY')

        # Calculate returns
        spy_returns = spy_prices['SPY'].pct_change().dropna()

        # Filter to dates range
        spy_returns = spy_returns[spy_returns.index >= start_date]

        return spy_returns

    def _get_spy_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        """Get SPY closing prices for date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Series of SPY closing prices
        """
        try:
            ohlcv_data = self.data_access.prices.read_prices_as_of(
                as_of=end_date + pd.Timedelta(days=1),
                tickers=['SPY'],
            )
            spy_prices = ohlcv_data.xs('Close', level='Price', axis=1)['SPY']
        except (ValueError, KeyError):
            # SPY might not be in snapshot, create synthetic benchmark
            logger.warning("SPY not found in data, using equal-weight proxy for prices")
            ohlcv_data = self.data_access.prices.read_prices_as_of(
                as_of=end_date + pd.Timedelta(days=1)
            )
            all_prices = ohlcv_data.xs('Close', level='Price', axis=1)
            spy_prices = all_prices.mean(axis=1)

        return spy_prices[(spy_prices.index >= start_date) & (spy_prices.index <= end_date)]

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

    def _calculate_trailing_score(
        self,
        periods: Dict[str, PeriodMetrics],
        trailing_days: int = 252,
    ) -> float:
        """Calculate score based on trailing window performance only.

        This method is optimized for quarterly re-runs where we want to
        score strategies based on recent performance rather than long-term
        multi-period averages.

        Args:
            periods: Dictionary of period metrics (uses shortest period)
            trailing_days: Days for trailing window (default: 252 = 1 year)

        Returns:
            Trailing Sharpe ratio (higher is better)
        """
        # Use the shortest evaluation period's returns for trailing calculation
        min_period = None
        min_years = float('inf')

        for period_name, metrics in periods.items():
            if not metrics.evaluation_success:
                continue
            # Parse years from period name (e.g., '1yr' -> 1)
            try:
                years = int(period_name.replace('yr', ''))
                if years < min_years:
                    min_years = years
                    min_period = metrics
            except ValueError:
                continue

        if min_period is None or min_period.daily_returns is None:
            logger.debug("No valid period with daily returns for trailing score")
            return float('-inf')

        # Calculate trailing Sharpe ratio
        returns = min_period.daily_returns
        if len(returns) < trailing_days:
            trailing_returns = returns
        else:
            trailing_returns = returns.iloc[-trailing_days:]

        if len(trailing_returns) < 20:
            logger.debug("Insufficient data for trailing score calculation")
            return float('-inf')

        # Annualize based on rebalance frequency
        periods_per_year = self._get_periods_per_year(
            'monthly' if len(returns) < 100 else 'weekly'
        )
        mean_return = trailing_returns.mean()
        std_return = trailing_returns.std()

        if std_return == 0:
            return 0.0

        trailing_sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)

        logger.debug(f"Trailing score ({trailing_days}d): Sharpe={trailing_sharpe:.3f}")
        return trailing_sharpe

    def _calculate_regime_weighted_score(
        self,
        regime_metrics: Dict[str, RegimeMetrics],
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate regime-weighted composite score.

        This method scores strategies based on their performance across
        different market regimes, weighted by historical regime frequency.

        Args:
            regime_metrics: Dict mapping regime name to RegimeMetrics
            regime_weights: Optional weights for each regime (sum to 1.0).
                           If not provided, uses equal weights.

        Returns:
            Regime-weighted score (higher is better)
        """
        if not regime_metrics:
            return float('-inf')

        # Default to equal weights if not provided
        if regime_weights is None:
            n_regimes = len(regime_metrics)
            regime_weights = {r: 1.0 / n_regimes for r in regime_metrics.keys()}

        score = 0.0
        total_weight = 0.0

        for regime, metrics in regime_metrics.items():
            weight = regime_weights.get(regime, 0.0)
            if weight > 0 and metrics.sharpe_ratio != float('-inf'):
                score += weight * metrics.sharpe_ratio
                total_weight += weight

        if total_weight == 0:
            return float('-inf')

        # Normalize by total weight used (in case some regimes are missing)
        normalized_score = score / total_weight

        logger.debug(
            f"Regime-weighted score: {normalized_score:.3f} "
            f"(weights used: {total_weight:.2f})"
        )
        return normalized_score
