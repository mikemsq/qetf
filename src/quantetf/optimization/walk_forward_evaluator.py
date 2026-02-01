"""Walk-forward evaluation for strategy optimization.

This module provides walk-forward validation to evaluate strategies
on out-of-sample data, preventing overfitting to recent market regimes.

Walk-forward validation evaluates strategies by:
1. Training on historical data (warmup period)
2. Testing on unseen future data (out-of-sample period)
3. Rolling windows forward to get multiple OOS evaluations
4. Scoring strategies based on OOS performance (not in-sample!)

Example:
    >>> from quantetf.optimization.walk_forward_evaluator import (
    ...     WalkForwardEvaluator,
    ...     WalkForwardEvaluatorConfig,
    ... )
    >>> evaluator = WalkForwardEvaluator(
    ...     data_access=data_access,
    ...     wf_config=WalkForwardEvaluatorConfig(train_years=3, test_years=1),
    ...     cost_bps=10.0,
    ... )
    >>> result = evaluator.evaluate(strategy_config)
    >>> print(f"OOS Sharpe: {result.oos_sharpe_mean:.2f}")
    >>> print(f"Composite Score: {result.composite_score:.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from quantetf.alpha.factory import create_alpha_model
from quantetf.backtest.simple_engine import (
    BacktestConfig,
    BacktestResult,
    SimpleBacktestEngine,
)
from quantetf.data.access import DataAccessContext
from quantetf.evaluation.walk_forward import (
    WalkForwardConfig,
    WalkForwardWindow,
    generate_walk_forward_windows,
)
from quantetf.optimization.grid import StrategyConfig
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.types import Universe

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Result Dataclasses (IMPL-036-A)
# =============================================================================


@dataclass
class WalkForwardEvaluatorConfig:
    """Configuration for walk-forward evaluation.

    Attributes:
        train_years: Number of years for each training window.
        test_years: Number of years for each test window.
        step_months: Months to slide window forward each iteration.
        min_windows: Minimum number of windows required for valid evaluation.
        require_positive_oos: If True, filter out strategies with negative OOS return.

    Example:
        With train_years=3, test_years=1, step_months=6:
        Window 1: Train [2016-2019] → Test [2019-2020]
        Window 2: Train [2016.5-2019.5] → Test [2019.5-2020.5]
        etc.
    """

    train_years: int = 3
    test_years: int = 1
    step_months: int = 6
    min_windows: int = 4
    require_positive_oos: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.train_years < 1:
            raise ValueError(f"train_years must be >= 1, got {self.train_years}")
        if self.test_years < 1:
            raise ValueError(f"test_years must be >= 1, got {self.test_years}")
        if self.step_months < 1 or self.step_months > 12:
            raise ValueError(f"step_months must be 1-12, got {self.step_months}")
        if self.min_windows < 1:
            raise ValueError(f"min_windows must be >= 1, got {self.min_windows}")


@dataclass
class WindowResult:
    """Results from a single walk-forward window.

    Attributes:
        window_id: Sequential identifier for this window.
        train_start: Start date of training period.
        train_end: End date of training period.
        test_start: Start date of test period (same as train_end).
        test_end: End date of test period.

        # In-sample (training) metrics - for reference only
        is_return: Total return during training period.
        is_sharpe: Sharpe ratio during training period.
        is_volatility: Annualized volatility during training.

        # Out-of-sample (test) metrics - PRIMARY for ranking
        oos_return: Total return during test period.
        oos_sharpe: Sharpe ratio during test period.
        oos_volatility: Annualized volatility during test.
        oos_max_drawdown: Maximum drawdown during test.

        # Benchmark comparison (test period only)
        spy_return: SPY total return during test period.
        active_return: oos_return - spy_return

        # Daily returns for further analysis
        oos_daily_returns: Series of daily returns during test period.
    """

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # In-sample metrics (reference only)
    is_return: float
    is_sharpe: float
    is_volatility: float

    # Out-of-sample metrics (PRIMARY)
    oos_return: float
    oos_sharpe: float
    oos_volatility: float
    oos_max_drawdown: float

    # Benchmark comparison
    spy_return: float
    active_return: float  # oos_return - spy_return

    # Daily returns for analysis
    oos_daily_returns: Optional[pd.Series] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "test_start": self.test_start.isoformat() if self.test_start else None,
            "test_end": self.test_end.isoformat() if self.test_end else None,
            "is_return": self.is_return,
            "is_sharpe": self.is_sharpe,
            "is_volatility": self.is_volatility,
            "oos_return": self.oos_return,
            "oos_sharpe": self.oos_sharpe,
            "oos_volatility": self.oos_volatility,
            "oos_max_drawdown": self.oos_max_drawdown,
            "spy_return": self.spy_return,
            "active_return": self.active_return,
        }


@dataclass
class WalkForwardEvaluationResult:
    """Results from walk-forward evaluation of a single strategy.

    This aggregates results across all walk-forward windows and provides
    summary statistics for strategy ranking.

    Attributes:
        config_name: Unique identifier for the strategy configuration.
        config: The StrategyConfig object (for reference).
        num_windows: Number of walk-forward windows evaluated.

        # Out-of-sample metrics (PRIMARY - use these for ranking)
        oos_sharpe_mean: Mean Sharpe ratio across test windows.
        oos_sharpe_std: Standard deviation of Sharpe across windows.
        oos_return_mean: Mean total return per test window.
        oos_active_return_mean: Mean active return vs SPY per window.
        oos_win_rate: Fraction of windows with positive active return.

        # In-sample metrics (for reference/debugging)
        is_sharpe_mean: Mean Sharpe during training windows.
        is_return_mean: Mean return during training windows.

        # Degradation metrics (IS - OOS; lower is better)
        sharpe_degradation: is_sharpe_mean - oos_sharpe_mean
        return_degradation: is_return_mean - oos_return_mean

        # Per-window details
        window_results: List of WindowResult objects.

        # Composite score for ranking (based on OOS metrics)
        composite_score: Final score for ranking strategies.
    """

    config_name: str
    config: Optional[StrategyConfig] = None
    num_windows: int = 0

    # Out-of-sample metrics (PRIMARY)
    oos_sharpe_mean: float = 0.0
    oos_sharpe_std: float = 0.0
    oos_return_mean: float = 0.0
    oos_active_return_mean: float = 0.0
    oos_win_rate: float = 0.0

    # In-sample metrics (reference)
    is_sharpe_mean: float = 0.0
    is_return_mean: float = 0.0

    # Degradation metrics
    sharpe_degradation: float = 0.0
    return_degradation: float = 0.0

    # Per-window details
    window_results: List[WindowResult] = field(default_factory=list)

    # Composite score
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            "config_name": self.config_name,
            "num_windows": self.num_windows,
            "oos_sharpe_mean": self.oos_sharpe_mean,
            "oos_sharpe_std": self.oos_sharpe_std,
            "oos_return_mean": self.oos_return_mean,
            "oos_active_return_mean": self.oos_active_return_mean,
            "oos_win_rate": self.oos_win_rate,
            "is_sharpe_mean": self.is_sharpe_mean,
            "is_return_mean": self.is_return_mean,
            "sharpe_degradation": self.sharpe_degradation,
            "return_degradation": self.return_degradation,
            "composite_score": self.composite_score,
        }

    def passed_filter(self) -> bool:
        """Check if this strategy passes OOS filters."""
        return self.oos_active_return_mean > 0


# =============================================================================
# WalkForwardEvaluator Class (IMPL-036-B/C/D)
# =============================================================================


class WalkForwardEvaluator:
    """Evaluates strategies using walk-forward validation.

    Walk-forward validation evaluates strategies on out-of-sample data
    to prevent overfitting. Each strategy is tested across multiple
    train/test windows, and the composite score is based on OOS metrics.

    Attributes:
        data_access: DataAccessContext for historical data.
        wf_config: Walk-forward configuration (window sizes, etc.).
        cost_bps: Transaction cost in basis points.
        _windows: Cached list of walk-forward windows.

    Example:
        >>> evaluator = WalkForwardEvaluator(
        ...     data_access=data_access,
        ...     wf_config=WalkForwardEvaluatorConfig(train_years=3, test_years=1),
        ...     cost_bps=10.0,
        ... )
        >>> result = evaluator.evaluate(strategy_config)
        >>> print(f"OOS Sharpe: {result.oos_sharpe_mean:.2f}")
        >>> print(f"Composite Score: {result.composite_score:.2f}")
    """

    def __init__(
        self,
        data_access: DataAccessContext,
        wf_config: Optional[WalkForwardEvaluatorConfig] = None,
        cost_bps: float = 10.0,
    ) -> None:
        """Initialize the walk-forward evaluator.

        Args:
            data_access: DataAccessContext for accessing price data.
            wf_config: Walk-forward configuration. Uses defaults if None.
            cost_bps: Transaction cost in basis points (default: 10).
        """
        self.data_access = data_access
        self.wf_config = wf_config or WalkForwardEvaluatorConfig()
        self.cost_bps = cost_bps
        self._windows: Optional[List[WalkForwardWindow]] = None

    def _get_data_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get the available date range from the data source.

        Returns:
            Tuple of (start_date, end_date) as pd.Timestamp.
        """
        # Get date range from price data using SPY as proxy
        prices = self.data_access.prices.read_prices_as_of(
            as_of=pd.Timestamp.now(),
            tickers=["SPY"],
        )
        if prices.empty:
            raise ValueError("No price data available")

        return pd.Timestamp(prices.index.min()), pd.Timestamp(prices.index.max())

    def _generate_windows(self) -> List[WalkForwardWindow]:
        """Generate walk-forward windows based on available data.

        Caches windows for reuse across multiple strategy evaluations.

        Returns:
            List of WalkForwardWindow objects.

        Raises:
            ValueError: If insufficient data for minimum number of windows.
        """
        if self._windows is not None:
            return self._windows

        start_date, end_date = self._get_data_date_range()

        # Map our config to the existing WalkForwardConfig
        wf_config = WalkForwardConfig(
            train_years=self.wf_config.train_years,
            test_years=self.wf_config.test_years,
            step_months=self.wf_config.step_months,
        )

        self._windows = generate_walk_forward_windows(
            start_date=start_date,
            end_date=end_date,
            config=wf_config,
        )

        if len(self._windows) < self.wf_config.min_windows:
            raise ValueError(
                f"Insufficient data for walk-forward validation. "
                f"Need {self.wf_config.min_windows} windows but could only generate "
                f"{len(self._windows)}. Try reducing train_years or test_years."
            )

        logger.info(f"Generated {len(self._windows)} walk-forward windows")
        return self._windows

    def evaluate(self, strategy_config: StrategyConfig) -> WalkForwardEvaluationResult:
        """Evaluate a strategy using walk-forward validation.

        For each window:
        1. Run backtest starting from train_start (for warmup)
        2. Extract metrics only from the test period
        3. Compare to SPY benchmark for the same test period

        Args:
            strategy_config: The StrategyConfig to evaluate.

        Returns:
            WalkForwardEvaluationResult with OOS-based scoring.
        """
        windows = self._generate_windows()
        window_results: List[WindowResult] = []

        for window in windows:
            try:
                window_result = self._evaluate_window(strategy_config, window)
                window_results.append(window_result)
            except Exception as e:
                logger.warning(
                    f"Failed to evaluate window {window.window_id} "
                    f"for {strategy_config.generate_name()}: {e}"
                )
                continue

        if not window_results:
            # Return a failed result
            return WalkForwardEvaluationResult(
                config_name=strategy_config.generate_name(),
                config=strategy_config,
                num_windows=0,
                oos_sharpe_mean=float("-inf"),
                composite_score=float("-inf"),
            )

        # Aggregate results across windows
        return self._aggregate_results(
            config_name=strategy_config.generate_name(),
            config=strategy_config,
            window_results=window_results,
        )

    def _evaluate_window(
        self,
        strategy_config: StrategyConfig,
        window: WalkForwardWindow,
    ) -> WindowResult:
        """Evaluate a strategy on a single walk-forward window.

        Runs backtest starting from train_start (for warmup) but only
        scores the test period.

        Args:
            strategy_config: Strategy configuration to test.
            window: Walk-forward window with train/test periods.

        Returns:
            WindowResult with IS and OOS metrics.
        """
        # Load universe from config
        universe = self._load_universe(strategy_config)

        # Create strategy components
        alpha_config = {
            "type": strategy_config.alpha_type,
            **strategy_config.alpha_params,
        }
        alpha_model = create_alpha_model(alpha_config)
        portfolio = EqualWeightTopN(top_n=strategy_config.top_n)
        cost_model = FlatTransactionCost(cost_bps=self.cost_bps)

        # Run full backtest (train + test) for warmup
        full_config = BacktestConfig(
            start_date=window.train_start,
            end_date=window.test_end,
            universe=universe,
            initial_capital=100_000.0,
            rebalance_frequency=strategy_config.schedule_name,
        )

        engine = SimpleBacktestEngine()
        full_result = engine.run(
            config=full_config,
            alpha_model=alpha_model,
            portfolio=portfolio,
            cost_model=cost_model,
            data_access=self.data_access,
        )

        # Extract train period metrics
        train_returns = self._extract_period_returns(
            full_result, window.train_start, window.train_end
        )
        is_metrics = self._calculate_metrics(train_returns)

        # Extract test period metrics (PRIMARY)
        test_returns = self._extract_period_returns(
            full_result, window.test_start, window.test_end
        )
        oos_metrics = self._calculate_metrics(test_returns)

        # Get SPY return for test period
        spy_return = self._get_spy_return_for_window(window)
        active_return = oos_metrics["total_return"] - spy_return

        return WindowResult(
            window_id=window.window_id,
            train_start=window.train_start.to_pydatetime(),
            train_end=window.train_end.to_pydatetime(),
            test_start=window.test_start.to_pydatetime(),
            test_end=window.test_end.to_pydatetime(),
            is_return=is_metrics["total_return"],
            is_sharpe=is_metrics["sharpe"],
            is_volatility=is_metrics["volatility"],
            oos_return=oos_metrics["total_return"],
            oos_sharpe=oos_metrics["sharpe"],
            oos_volatility=oos_metrics["volatility"],
            oos_max_drawdown=oos_metrics["max_drawdown"],
            spy_return=spy_return,
            active_return=active_return,
            oos_daily_returns=test_returns,
        )

    def _extract_period_returns(
        self,
        result: BacktestResult,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.Series:
        """Extract daily returns for a specific period from backtest result.

        Args:
            result: Full backtest result.
            start_date: Start of period to extract.
            end_date: End of period to extract.

        Returns:
            Series of daily returns for the period.
        """
        # Get equity curve and filter to period
        equity = result.equity_curve
        mask = (equity.index >= start_date) & (equity.index <= end_date)
        period_equity = equity[mask]

        if period_equity.empty:
            return pd.Series(dtype=float)

        # Use returns column if available, otherwise calculate from portfolio_value
        if "returns" in period_equity.columns:
            returns = period_equity["returns"].dropna()
        elif "portfolio_value" in period_equity.columns:
            returns = period_equity["portfolio_value"].pct_change().dropna()
        else:
            # Assume equity curve is a Series
            returns = period_equity.pct_change().dropna()

        return returns

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics from a return series.

        Args:
            returns: Daily return series.

        Returns:
            Dictionary with total_return, sharpe, volatility, max_drawdown.
        """
        if returns.empty or len(returns) < 2:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
            }

        # Total return
        total_return = (1 + returns).prod() - 1

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming risk-free rate = 0)
        if volatility > 0:
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()  # Most negative = worst drawdown

        return {
            "total_return": float(total_return),
            "sharpe": float(sharpe_ratio),
            "volatility": float(volatility),
            "max_drawdown": float(max_dd),
        }

    def _get_spy_return_for_window(self, window: WalkForwardWindow) -> float:
        """Get SPY total return for the test period only.

        Args:
            window: Walk-forward window.

        Returns:
            SPY total return as a decimal (e.g., 0.10 for 10%).
        """
        try:
            spy_prices = self.data_access.prices.read_prices_as_of(
                as_of=window.test_end,
                tickers=["SPY"],
            )
        except Exception as e:
            logger.warning(f"Failed to get SPY prices: {e}")
            return 0.0

        if spy_prices.empty:
            logger.warning("No SPY data available for benchmark")
            return 0.0

        # Handle MultiIndex columns (ticker, price_type)
        if isinstance(spy_prices.columns, pd.MultiIndex):
            try:
                spy_close = spy_prices.xs("Close", level="Price", axis=1)["SPY"]
            except KeyError:
                # Try alternative column access
                spy_close = spy_prices["SPY"]["Close"]
        else:
            spy_close = spy_prices["SPY"] if "SPY" in spy_prices.columns else spy_prices.iloc[:, 0]

        # Filter to test period
        mask = (spy_close.index >= window.test_start) & (
            spy_close.index <= window.test_end
        )
        period_prices = spy_close[mask]

        if len(period_prices) < 2:
            return 0.0

        # Calculate total return
        start_price = period_prices.iloc[0]
        end_price = period_prices.iloc[-1]
        return float((end_price / start_price) - 1)

    def _aggregate_results(
        self,
        config_name: str,
        config: StrategyConfig,
        window_results: List[WindowResult],
    ) -> WalkForwardEvaluationResult:
        """Aggregate window results into a single evaluation result.

        Args:
            config_name: Strategy configuration name.
            config: Strategy configuration object.
            window_results: List of individual window results.

        Returns:
            Aggregated WalkForwardEvaluationResult.
        """
        # Extract OOS metrics from all windows
        oos_sharpes = [w.oos_sharpe for w in window_results]
        oos_returns = [w.oos_return for w in window_results]
        oos_active_returns = [w.active_return for w in window_results]

        # Extract IS metrics
        is_sharpes = [w.is_sharpe for w in window_results]
        is_returns = [w.is_return for w in window_results]

        # Calculate aggregates
        oos_sharpe_mean = float(np.mean(oos_sharpes))
        oos_sharpe_std = (
            float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) > 1 else 0.0
        )
        oos_return_mean = float(np.mean(oos_returns))
        oos_active_return_mean = float(np.mean(oos_active_returns))

        # Win rate: fraction of windows with positive active return
        oos_win_rate = sum(1 for ar in oos_active_returns if ar > 0) / len(
            oos_active_returns
        )

        # IS metrics for reference
        is_sharpe_mean = float(np.mean(is_sharpes))
        is_return_mean = float(np.mean(is_returns))

        # Degradation metrics
        sharpe_degradation = is_sharpe_mean - oos_sharpe_mean
        return_degradation = is_return_mean - oos_return_mean

        # Create result
        result = WalkForwardEvaluationResult(
            config_name=config_name,
            config=config,
            num_windows=len(window_results),
            oos_sharpe_mean=oos_sharpe_mean,
            oos_sharpe_std=oos_sharpe_std,
            oos_return_mean=oos_return_mean,
            oos_active_return_mean=oos_active_return_mean,
            oos_win_rate=oos_win_rate,
            is_sharpe_mean=is_sharpe_mean,
            is_return_mean=is_return_mean,
            sharpe_degradation=sharpe_degradation,
            return_degradation=return_degradation,
            window_results=window_results,
        )

        # Calculate composite score
        result.composite_score = self._calculate_composite_score(result)

        return result

    def _calculate_composite_score(
        self, result: WalkForwardEvaluationResult
    ) -> float:
        """Calculate composite score for ranking strategies.

        Uses OOS metrics only. Formula:
        score = 0.4 * active_return_scaled + 0.3 * sharpe + 0.3 * win_rate

        Args:
            result: Aggregated evaluation result.

        Returns:
            Composite score for ranking (higher is better).
        """
        # Scale active return to ~0-1 range (assuming 10% is good)
        active_return_scaled = result.oos_active_return_mean * 10

        score = (
            0.4 * active_return_scaled
            + 0.3 * result.oos_sharpe_mean
            + 0.3 * result.oos_win_rate
        )

        return float(score)

    def _load_universe(self, strategy_config: StrategyConfig) -> Universe:
        """Load universe from strategy configuration.

        Args:
            strategy_config: Strategy configuration with universe_path.

        Returns:
            Universe object with tickers.
        """
        universe_path = Path(strategy_config.universe_path)
        if not universe_path.exists():
            raise FileNotFoundError(f"Universe config not found: {universe_path}")

        with open(universe_path) as f:
            universe_config = yaml.safe_load(f)

        # Get tickers from the source section
        if "source" in universe_config and "tickers" in universe_config["source"]:
            tickers = universe_config["source"]["tickers"]
        else:
            raise ValueError(
                f"No tickers found in universe config '{strategy_config.universe_path}'"
            )

        if not tickers:
            raise ValueError(
                f"Empty ticker list in universe '{strategy_config.universe_name}'"
            )

        # Filter to tickers that exist in our data
        available_tickers = set(self.data_access.prices.get_available_tickers())
        valid_tickers = [t for t in tickers if t in available_tickers]

        if not valid_tickers:
            raise ValueError(
                f"No valid tickers found for universe '{strategy_config.universe_name}'. "
                f"Universe has {len(tickers)} tickers, none available in data."
            )

        return Universe(
            as_of=pd.Timestamp.now(),
            tickers=tuple(valid_tickers),
        )
