# Task IMPL-036-B/C/D: Walk-Forward Evaluator Core Implementation

## File to Modify
`src/quantetf/optimization/walk_forward_evaluator.py` (add to file created in IMPL-036-A)

## Purpose
Implement the `WalkForwardEvaluator` class that evaluates a single strategy across multiple walk-forward windows and calculates OOS-based metrics for ranking.

---

## Background

This evaluator replaces the in-sample scoring approach with walk-forward validation:
- Each strategy is tested on **out-of-sample** data (test windows)
- Training windows provide warmup data but are NOT used for ranking
- Composite score is based on OOS metrics (Sharpe, active return, win rate)

---

## Key Design Decisions

### 1. Warmup Period Handling
The training period provides warmup for indicators. When evaluating:
- Run backtest starting from `train_start`
- Only extract metrics from the `test_start` to `test_end` period
- This ensures indicators are "warm" before the test period

### 2. SPY Benchmark
Calculate SPY return for each TEST window separately (not full period):
```
Window 1 Test: SPY return from 2019-01-01 to 2019-12-31
Window 2 Test: SPY return from 2019-07-01 to 2020-06-30
```

### 3. Composite Scoring
Use OOS metrics only. Recommended formula (Option A from master handout):
```python
score = (
    0.4 * oos_active_return_mean * 10 +  # Scale to ~0-1 range
    0.3 * oos_sharpe_mean +
    0.3 * oos_win_rate
)
```

---

## Implementation

Add the following to `src/quantetf/optimization/walk_forward_evaluator.py`:

```python
"""Walk-forward evaluation for strategy optimization."""
import logging
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from quantetf.data.access import DataAccessContext
from quantetf.backtest.simple_engine import (
    SimpleBacktestEngine,
    BacktestConfig,
    BacktestResult,
)
from quantetf.evaluation.metrics import sharpe, max_drawdown
from quantetf.alpha.factory import create_alpha_model
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.types import Universe

# Import from existing walk_forward module
from quantetf.evaluation.walk_forward import (
    WalkForwardWindow,
    generate_walk_forward_windows,
)

# Import dataclasses from IMPL-036-A
# (These should already be in this file)

logger = logging.getLogger(__name__)


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
    ):
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
        # Get date range from price data
        # This depends on your DataAccessContext implementation
        prices = self.data_access.prices.read_prices_as_of(
            as_of=pd.Timestamp.now(),
            tickers=['SPY'],  # Use SPY as proxy for date range
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

        # Use existing function from walk_forward.py
        from quantetf.evaluation.walk_forward import (
            WalkForwardConfig,
            generate_walk_forward_windows,
        )

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

    def evaluate(self, strategy_config: 'StrategyConfig') -> WalkForwardEvaluationResult:
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
                oos_sharpe_mean=float('-inf'),
                composite_score=float('-inf'),
            )

        # Aggregate results across windows
        return self._aggregate_results(
            config_name=strategy_config.generate_name(),
            config=strategy_config,
            window_results=window_results,
        )

    def _evaluate_window(
        self,
        strategy_config: 'StrategyConfig',
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
        alpha_model = self._create_alpha_model(strategy_config)
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
        active_return = oos_metrics['total_return'] - spy_return

        return WindowResult(
            window_id=window.window_id,
            train_start=window.train_start.to_pydatetime(),
            train_end=window.train_end.to_pydatetime(),
            test_start=window.test_start.to_pydatetime(),
            test_end=window.test_end.to_pydatetime(),
            is_return=is_metrics['total_return'],
            is_sharpe=is_metrics['sharpe'],
            is_volatility=is_metrics['volatility'],
            oos_return=oos_metrics['total_return'],
            oos_sharpe=oos_metrics['sharpe'],
            oos_volatility=oos_metrics['volatility'],
            oos_max_drawdown=oos_metrics['max_drawdown'],
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

        # Calculate daily returns from equity curve
        returns = period_equity.pct_change().dropna()
        return returns

    def _calculate_metrics(self, returns: pd.Series) -> dict:
        """Calculate performance metrics from a return series.

        Args:
            returns: Daily return series.

        Returns:
            Dictionary with total_return, sharpe, volatility, max_drawdown.
        """
        if returns.empty:
            return {
                'total_return': 0.0,
                'sharpe': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
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
            'total_return': total_return,
            'sharpe': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_dd,
        }

    def _get_spy_return_for_window(self, window: WalkForwardWindow) -> float:
        """Get SPY total return for the test period only.

        Args:
            window: Walk-forward window.

        Returns:
            SPY total return as a decimal (e.g., 0.10 for 10%).
        """
        spy_prices = self.data_access.prices.read_prices_as_of(
            as_of=window.test_end,
            tickers=['SPY'],
        )

        if spy_prices.empty or 'SPY' not in spy_prices.columns:
            logger.warning("No SPY data available for benchmark")
            return 0.0

        # Filter to test period
        mask = (spy_prices.index >= window.test_start) & (spy_prices.index <= window.test_end)
        period_prices = spy_prices.loc[mask, 'SPY']

        if len(period_prices) < 2:
            return 0.0

        # Calculate total return
        start_price = period_prices.iloc[0]
        end_price = period_prices.iloc[-1]
        return (end_price / start_price) - 1

    def _aggregate_results(
        self,
        config_name: str,
        config: 'StrategyConfig',
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
        oos_sharpe_mean = np.mean(oos_sharpes)
        oos_sharpe_std = np.std(oos_sharpes, ddof=1) if len(oos_sharpes) > 1 else 0.0
        oos_return_mean = np.mean(oos_returns)
        oos_active_return_mean = np.mean(oos_active_returns)

        # Win rate: fraction of windows with positive active return
        oos_win_rate = sum(1 for ar in oos_active_returns if ar > 0) / len(oos_active_returns)

        # IS metrics for reference
        is_sharpe_mean = np.mean(is_sharpes)
        is_return_mean = np.mean(is_returns)

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

    def _calculate_composite_score(self, result: WalkForwardEvaluationResult) -> float:
        """Calculate composite score for ranking strategies.

        Uses Option A from master handout (Weighted Sum):
        score = 0.4 * active_return_scaled + 0.3 * sharpe + 0.3 * win_rate

        Args:
            result: Aggregated evaluation result.

        Returns:
            Composite score for ranking (higher is better).
        """
        # Scale active return to ~0-1 range (assuming 10% is good)
        active_return_scaled = result.oos_active_return_mean * 10

        score = (
            0.4 * active_return_scaled +
            0.3 * result.oos_sharpe_mean +
            0.3 * result.oos_win_rate
        )

        return score

    def _load_universe(self, strategy_config: 'StrategyConfig') -> Universe:
        """Load universe from strategy configuration.

        Args:
            strategy_config: Strategy configuration with universe_path.

        Returns:
            Universe object with tickers.
        """
        import yaml
        from pathlib import Path

        universe_path = Path(strategy_config.universe_path)
        if not universe_path.exists():
            raise FileNotFoundError(f"Universe config not found: {universe_path}")

        with open(universe_path) as f:
            universe_config = yaml.safe_load(f)

        # Get tickers for the specified universe name
        universe_data = universe_config.get(strategy_config.universe_name, {})
        tickers = universe_data.get('tickers', [])

        if not tickers:
            raise ValueError(
                f"No tickers found for universe '{strategy_config.universe_name}' "
                f"in {universe_path}"
            )

        return Universe(
            as_of=pd.Timestamp.now(),
            tickers=tuple(tickers),
        )

    def _create_alpha_model(self, strategy_config: 'StrategyConfig'):
        """Create alpha model from strategy configuration.

        Args:
            strategy_config: Strategy configuration with alpha_type and alpha_params.

        Returns:
            Alpha model instance.
        """
        # Use factory pattern if available, otherwise fallback
        try:
            from quantetf.alpha.factory import create_alpha_model
            return create_alpha_model(
                alpha_type=strategy_config.alpha_type,
                **strategy_config.alpha_params,
            )
        except ImportError:
            # Fallback: manual instantiation
            return self._create_alpha_model_manual(strategy_config)

    def _create_alpha_model_manual(self, strategy_config: 'StrategyConfig'):
        """Manual alpha model creation as fallback."""
        alpha_type = strategy_config.alpha_type
        params = strategy_config.alpha_params

        if alpha_type == 'momentum':
            from quantetf.alpha.momentum import MomentumAlpha
            return MomentumAlpha(**params)
        elif alpha_type == 'momentum_acceleration':
            from quantetf.alpha.momentum_acceleration import MomentumAccelerationAlpha
            return MomentumAccelerationAlpha(**params)
        elif alpha_type == 'vol_adjusted_momentum':
            from quantetf.alpha.vol_adjusted_momentum import VolAdjustedMomentumAlpha
            return VolAdjustedMomentumAlpha(**params)
        elif alpha_type == 'residual_momentum':
            from quantetf.alpha.residual_momentum import ResidualMomentumAlpha
            return ResidualMomentumAlpha(**params)
        else:
            raise ValueError(f"Unknown alpha type: {alpha_type}")
```

---

## Key Files to Reference

| File | Purpose |
|------|---------|
| `src/quantetf/evaluation/walk_forward.py` | Reuse `generate_walk_forward_windows()` |
| `src/quantetf/optimization/evaluator.py` | Pattern for `MultiPeriodEvaluator` |
| `src/quantetf/backtest/simple_engine.py` | `BacktestConfig`, `SimpleBacktestEngine` |
| `src/quantetf/optimization/grid.py` | `StrategyConfig` |

---

## Testing

Add to `tests/optimization/test_walk_forward_evaluator.py`:

```python
"""Tests for WalkForwardEvaluator class."""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

from quantetf.optimization.walk_forward_evaluator import (
    WalkForwardEvaluator,
    WalkForwardEvaluatorConfig,
    WalkForwardEvaluationResult,
    WindowResult,
)


class TestWalkForwardEvaluator:
    """Tests for WalkForwardEvaluator."""

    @pytest.fixture
    def mock_data_access(self):
        """Create mock DataAccessContext."""
        mock = Mock()
        # Create price data spanning 10 years
        dates = pd.date_range('2016-01-01', '2026-01-01', freq='B')
        prices = pd.DataFrame({
            'SPY': np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01)),
            'QQQ': np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015)),
        }, index=dates)
        mock.prices.read_prices_as_of = Mock(return_value=prices)
        return mock

    @pytest.fixture
    def evaluator(self, mock_data_access):
        """Create evaluator instance."""
        return WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(
                train_years=2,
                test_years=1,
                step_months=6,
                min_windows=2,
            ),
            cost_bps=10.0,
        )

    def test_window_generation(self, evaluator):
        """Test that windows are generated correctly."""
        windows = evaluator._generate_windows()

        assert len(windows) >= 2
        for w in windows:
            # Test end should equal train end (no gap)
            assert w.test_start == w.train_end
            # Test period should be ~1 year
            test_days = (w.test_end - w.test_start).days
            assert 300 < test_days < 400

    def test_windows_cached(self, evaluator):
        """Test that windows are cached after first generation."""
        windows1 = evaluator._generate_windows()
        windows2 = evaluator._generate_windows()
        assert windows1 is windows2

    def test_calculate_metrics_basic(self, evaluator):
        """Test metric calculation with known values."""
        # Create simple return series: 1% daily for 252 days
        returns = pd.Series([0.01] * 252)
        metrics = evaluator._calculate_metrics(returns)

        assert metrics['total_return'] > 0
        assert metrics['sharpe'] > 0
        assert metrics['volatility'] > 0
        assert metrics['max_drawdown'] == 0  # No drawdown with positive returns

    def test_calculate_metrics_empty(self, evaluator):
        """Test metric calculation with empty series."""
        returns = pd.Series(dtype=float)
        metrics = evaluator._calculate_metrics(returns)

        assert metrics['total_return'] == 0.0
        assert metrics['sharpe'] == 0.0
        assert metrics['volatility'] == 0.0
        assert metrics['max_drawdown'] == 0.0

    def test_calculate_metrics_with_drawdown(self, evaluator):
        """Test max drawdown calculation."""
        # Create series with a 10% drawdown
        returns = pd.Series([0.05, 0.05, -0.15, 0.05, 0.05])
        metrics = evaluator._calculate_metrics(returns)

        assert metrics['max_drawdown'] < 0  # Should be negative

    def test_composite_score_calculation(self, evaluator):
        """Test composite score uses OOS metrics only."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_sharpe_mean=0.5,
            oos_active_return_mean=0.05,  # 5%
            oos_win_rate=0.7,
            is_sharpe_mean=2.0,  # High IS should NOT inflate score
            is_return_mean=0.30,
        )
        score = evaluator._calculate_composite_score(result)

        # Score should be moderate despite high IS
        # 0.4 * (0.05 * 10) + 0.3 * 0.5 + 0.3 * 0.7 = 0.2 + 0.15 + 0.21 = 0.56
        assert 0.4 < score < 0.8

    def test_composite_score_negative_active_return(self, evaluator):
        """Test composite score with negative active return."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_sharpe_mean=0.5,
            oos_active_return_mean=-0.05,  # -5%
            oos_win_rate=0.3,
        )
        score = evaluator._calculate_composite_score(result)

        # Should be low due to negative active return
        assert score < 0.5

    def test_aggregate_results_win_rate(self, evaluator):
        """Test win rate calculation across windows."""
        window_results = [
            WindowResult(
                window_id=i,
                train_start=datetime(2016, 1, 1),
                train_end=datetime(2019, 1, 1),
                test_start=datetime(2019, 1, 1),
                test_end=datetime(2020, 1, 1),
                is_return=0.20,
                is_sharpe=1.0,
                is_volatility=0.15,
                oos_return=0.10,
                oos_sharpe=0.7,
                oos_volatility=0.18,
                oos_max_drawdown=-0.10,
                spy_return=0.08,
                active_return=0.02 if i % 2 == 0 else -0.01,  # Alternating
            )
            for i in range(4)
        ]

        result = evaluator._aggregate_results(
            config_name="test",
            config=None,
            window_results=window_results,
        )

        # 2 out of 4 windows have positive active return
        assert result.oos_win_rate == 0.5


class TestOverfitDetection:
    """Tests for detecting overfit strategies."""

    def test_overfit_strategy_shows_degradation(self):
        """Test that overfit strategy shows IS >> OOS."""
        # Simulate an overfit strategy result
        result = WalkForwardEvaluationResult(
            config_name="overfit_strategy",
            num_windows=7,
            oos_sharpe_mean=0.21,
            is_sharpe_mean=2.72,
            sharpe_degradation=2.51,  # Large degradation!
        )

        # Should show significant degradation
        assert result.sharpe_degradation > 0.5
        assert result.is_sharpe_mean > result.oos_sharpe_mean

    def test_robust_strategy_low_degradation(self):
        """Test that robust strategy shows low degradation."""
        result = WalkForwardEvaluationResult(
            config_name="robust_strategy",
            num_windows=7,
            oos_sharpe_mean=0.82,
            is_sharpe_mean=0.95,
            sharpe_degradation=0.13,  # Low degradation
        )

        assert result.sharpe_degradation < 0.5
```

---

## Edge Cases

### 1. Insufficient Data
```python
if len(self._windows) < self.wf_config.min_windows:
    raise ValueError(
        f"Insufficient data for walk-forward validation. "
        f"Need {self.wf_config.min_windows} windows..."
    )
```

### 2. Missing SPY Data
Return 0.0 and log a warning - don't fail the entire evaluation.

### 3. Individual Window Failure
Log warning and continue to next window. Only fail if ALL windows fail.

### 4. Empty Return Series
Return zero metrics (not NaN) to avoid breaking aggregation.

---

## Acceptance Checklist

- [ ] `_generate_windows()` reuses `generate_walk_forward_windows()` from walk_forward.py
- [ ] Windows are cached after first generation
- [ ] `evaluate()` returns `WalkForwardEvaluationResult` with all fields populated
- [ ] `_evaluate_window()` runs backtest from train_start for warmup
- [ ] `_evaluate_window()` only scores test period returns
- [ ] `_get_spy_return_for_window()` calculates SPY return for test period only
- [ ] `_calculate_metrics()` handles empty series gracefully
- [ ] `_calculate_composite_score()` uses only OOS metrics
- [ ] `_aggregate_results()` correctly calculates win_rate
- [ ] Individual window failures don't crash the evaluation
- [ ] All tests pass
- [ ] Type hints on all methods
- [ ] Docstrings on all public methods

---

## Performance Requirements

- Single strategy evaluation: < 30 seconds
- Should work with 7+ windows efficiently

---

## Dependencies

- `quantetf.evaluation.walk_forward` (for `generate_walk_forward_windows`)
- `quantetf.backtest.simple_engine`
- `quantetf.alpha.*` (all alpha models)
- `quantetf.portfolio.equal_weight`
- `quantetf.portfolio.costs`

---

## Next Task

After completing this task, proceed to **IMPL-036-E**: Modify `StrategyOptimizer` to use `WalkForwardEvaluator`.
