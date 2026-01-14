# Task 2: Multi-Period Evaluator

## File to Create
`src/quantetf/optimization/evaluator.py`

## Purpose
Evaluate a single strategy configuration across multiple time periods (3yr, 5yr, 10yr) and determine if it beats SPY.

## Implementation

```python
"""Multi-period strategy evaluator."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from quantetf.backtest.simple_engine import SimpleBacktestEngine
from quantetf.evaluation.metrics import calculate_active_metrics
from quantetf.evaluation.benchmarks import get_spy_returns
from quantetf.data.snapshot import SnapshotDataStore

logger = logging.getLogger(__name__)


@dataclass
class PeriodMetrics:
    """Metrics for a single evaluation period."""
    period_name: str  # '3yr', '5yr', '10yr'
    start_date: datetime
    end_date: datetime
    strategy_return: float
    spy_return: float
    active_return: float  # strategy - spy
    strategy_volatility: float
    tracking_error: float
    information_ratio: float
    max_drawdown: float
    sharpe_ratio: float


@dataclass
class MultiPeriodResult:
    """Results across all evaluation periods."""
    config_name: str
    periods: Dict[str, PeriodMetrics]
    beats_spy_all_periods: bool
    composite_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        result = {
            'config_name': self.config_name,
            'beats_spy_all_periods': self.beats_spy_all_periods,
            'composite_score': self.composite_score
        }
        for period_name, metrics in self.periods.items():
            prefix = period_name.replace('yr', 'y')
            result[f'{prefix}_strategy_return'] = metrics.strategy_return
            result[f'{prefix}_spy_return'] = metrics.spy_return
            result[f'{prefix}_active_return'] = metrics.active_return
            result[f'{prefix}_information_ratio'] = metrics.information_ratio
            result[f'{prefix}_sharpe_ratio'] = metrics.sharpe_ratio
            result[f'{prefix}_max_drawdown'] = metrics.max_drawdown
        return result


class MultiPeriodEvaluator:
    """Evaluates strategies across multiple time periods."""

    def __init__(self, snapshot: SnapshotDataStore, end_date: Optional[datetime] = None):
        """
        Args:
            snapshot: Data snapshot for backtesting
            end_date: End date for all periods (defaults to snapshot end date)
        """
        self.snapshot = snapshot
        self.end_date = end_date or snapshot.get_end_date()

    def evaluate(
        self,
        config: 'StrategyConfig',
        periods_years: List[int] = [3, 5, 10]
    ) -> MultiPeriodResult:
        """
        Evaluate strategy across multiple time periods.

        Args:
            config: Strategy configuration to evaluate
            periods_years: List of lookback periods in years

        Returns:
            MultiPeriodResult with metrics for each period
        """
        period_results = {}

        for years in periods_years:
            period_name = f'{years}yr'
            try:
                metrics = self._evaluate_period(config, years)
                period_results[period_name] = metrics
            except Exception as e:
                logger.warning(f"Failed to evaluate {period_name} for {config.generate_name()}: {e}")
                # Create failed metrics
                period_results[period_name] = self._create_failed_metrics(period_name, years)

        # Determine if strategy beats SPY in all periods
        beats_spy = self._beats_spy_all_periods(period_results)

        # Calculate composite score
        composite_score = self._calculate_composite_score(period_results)

        return MultiPeriodResult(
            config_name=config.generate_name(),
            periods=period_results,
            beats_spy_all_periods=beats_spy,
            composite_score=composite_score
        )

    def _evaluate_period(self, config: 'StrategyConfig', years: int) -> PeriodMetrics:
        """Evaluate strategy for a single period."""
        # Calculate start date
        start_date = self.end_date - timedelta(days=years * 365)

        # Run backtest
        engine = SimpleBacktestEngine(self.snapshot)
        backtest_result = engine.run(
            config=config.to_dict(),
            start_date=start_date,
            end_date=self.end_date
        )

        # Get SPY benchmark returns for same period
        spy_returns = get_spy_returns(
            self.snapshot,
            start_date=start_date,
            end_date=self.end_date
        )

        # Calculate active metrics
        active_metrics = calculate_active_metrics(
            strategy_returns=backtest_result.returns,
            benchmark_returns=spy_returns
        )

        # Calculate total returns
        strategy_total_return = (1 + backtest_result.returns).prod() - 1
        spy_total_return = (1 + spy_returns).prod() - 1

        return PeriodMetrics(
            period_name=f'{years}yr',
            start_date=start_date,
            end_date=self.end_date,
            strategy_return=strategy_total_return,
            spy_return=spy_total_return,
            active_return=strategy_total_return - spy_total_return,
            strategy_volatility=backtest_result.returns.std() * (252 ** 0.5),
            tracking_error=active_metrics['tracking_error'],
            information_ratio=active_metrics['information_ratio'],
            max_drawdown=backtest_result.max_drawdown,
            sharpe_ratio=active_metrics.get('sharpe_ratio', 0)
        )

    def _create_failed_metrics(self, period_name: str, years: int) -> PeriodMetrics:
        """Create metrics for a failed evaluation."""
        start_date = self.end_date - timedelta(days=years * 365)
        return PeriodMetrics(
            period_name=period_name,
            start_date=start_date,
            end_date=self.end_date,
            strategy_return=float('-inf'),
            spy_return=0,
            active_return=float('-inf'),
            strategy_volatility=float('inf'),
            tracking_error=float('inf'),
            information_ratio=float('-inf'),
            max_drawdown=-1.0,
            sharpe_ratio=float('-inf')
        )

    def _beats_spy_all_periods(self, periods: Dict[str, PeriodMetrics]) -> bool:
        """Check if strategy beats SPY in ALL periods."""
        for period_name, metrics in periods.items():
            # Must have positive active return
            if metrics.active_return <= 0:
                return False
            # Must have positive information ratio
            if metrics.information_ratio <= 0:
                return False
        return True

    def _calculate_composite_score(self, periods: Dict[str, PeriodMetrics]) -> float:
        """
        Calculate composite score for ranking strategies.

        Score = average(IR across periods) with penalty for inconsistency.
        """
        irs = [m.information_ratio for m in periods.values()
               if m.information_ratio != float('-inf')]

        if not irs:
            return float('-inf')

        avg_ir = sum(irs) / len(irs)

        # Penalty for inconsistency (std of IRs)
        if len(irs) > 1:
            ir_std = (sum((ir - avg_ir) ** 2 for ir in irs) / len(irs)) ** 0.5
            consistency_penalty = ir_std * 0.5
        else:
            consistency_penalty = 0

        # Bonus for beating SPY in all periods
        if all(m.active_return > 0 and m.information_ratio > 0
               for m in periods.values()):
            winner_bonus = 0.5
        else:
            winner_bonus = 0

        return avg_ir - consistency_penalty + winner_bonus
```

## Key Files to Reference

| File | What to Import |
|------|----------------|
| `src/quantetf/backtest/simple_engine.py` | `SimpleBacktestEngine` |
| `src/quantetf/evaluation/metrics.py` | `calculate_active_metrics` |
| `src/quantetf/evaluation/benchmarks.py` | `get_spy_returns` |
| `src/quantetf/data/snapshot.py` | `SnapshotDataStore` |

## Testing

```python
def test_multi_period_evaluator():
    from quantetf.optimization.grid import generate_configs

    # Load snapshot
    snapshot = SnapshotDataStore('data/snapshots/snapshot_20260113_232157')

    # Create evaluator
    evaluator = MultiPeriodEvaluator(snapshot)

    # Test with first config
    configs = generate_configs()
    config = configs[0]

    result = evaluator.evaluate(config)

    assert result.config_name == config.generate_name()
    assert '3yr' in result.periods
    assert '5yr' in result.periods
    assert '10yr' in result.periods

    print(f"Config: {result.config_name}")
    print(f"Beats SPY all periods: {result.beats_spy_all_periods}")
    print(f"Composite score: {result.composite_score:.3f}")
```

## Scoring Logic

The composite score rewards:
1. **High average Information Ratio** across all periods
2. **Consistency** - penalizes strategies with volatile IR across periods
3. **Winner bonus** - extra points for beating SPY in all periods

This ensures the top-ranked strategies are both good AND consistent.

## Edge Cases

1. **Insufficient data**: If a period requires more history than available, evaluation fails gracefully
2. **Division by zero**: Tracking error of 0 would cause IR to be infinite - handle this
3. **Missing SPY data**: Should never happen but handle gracefully

## Dependencies

- `src/quantetf/backtest/simple_engine.py`
- `src/quantetf/evaluation/metrics.py`
- `src/quantetf/evaluation/benchmarks.py`
- `src/quantetf/data/snapshot.py`
- `src/quantetf/optimization/grid.py` (for StrategyConfig type hint)
