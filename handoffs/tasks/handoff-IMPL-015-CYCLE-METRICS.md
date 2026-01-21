# Task Handoff: IMPL-015 - Cycle Win Rate Metrics

**Task ID:** IMPL-015
**Status:** ready
**Priority:** critical
**Estimated Effort:** 4-6 hours
**Dependencies:** None

---

## Quick Context

You are implementing **per-rebalance-cycle metrics** to measure the primary success criterion:

> **The investment process must demonstrate positive active returns (outperformance vs SPY) for at least 80% of rebalance cycles during the backtest period.**

Current metrics track daily/aggregate performance. We need cycle-level decomposition to answer: "What percentage of individual rebalance periods beat SPY?"

**Why this matters:** This is the PRIMARY SUCCESS CRITERION. Without this metric, we cannot evaluate whether a strategy meets the project goals.

---

## What You Need to Know

### Current State

The existing `win_rate()` function in `src/quantetf/evaluation/metrics.py` calculates percentage of positive daily returns. This is NOT what we need.

We need:
- Track returns for each rebalance cycle (e.g., if monthly rebalancing, each month is one cycle)
- Compare strategy return vs SPY return for each cycle
- Calculate: `cycles_where_strategy > SPY / total_cycles`

### Key Concepts

**Rebalance Cycle:** The period between two consecutive rebalance dates.
- If rebalancing monthly: ~12 cycles per year
- If rebalancing weekly: ~52 cycles per year

**Active Return per Cycle:** `strategy_return_cycle - spy_return_cycle`

**Cycle Win Rate:** Percentage of cycles with positive active return

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Coding standards
2. **`/workspaces/qetf/README.md`** - Success criterion definition
3. **`/workspaces/qetf/src/quantetf/evaluation/metrics.py`** - Existing metrics (extend this)
4. **`/workspaces/qetf/src/quantetf/backtest/simple_engine.py`** - BacktestResult structure
5. **`/workspaces/qetf/src/quantetf/evaluation/benchmarks.py`** - SPY benchmark calculation

---

## Implementation Steps

### 1. Create CycleMetrics dataclass

Add to `src/quantetf/evaluation/metrics.py` or create new file `src/quantetf/evaluation/cycle_metrics.py`:

```python
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

@dataclass
class CycleResult:
    """Result for a single rebalance cycle."""
    cycle_start: pd.Timestamp
    cycle_end: pd.Timestamp
    strategy_return: float
    benchmark_return: float
    active_return: float
    is_win: bool  # True if active_return > 0


@dataclass
class CycleMetrics:
    """Aggregate metrics across all rebalance cycles."""
    total_cycles: int
    winning_cycles: int
    losing_cycles: int
    win_rate: float  # winning_cycles / total_cycles
    avg_active_return: float
    avg_win_return: float  # Average active return when winning
    avg_loss_return: float  # Average active return when losing
    best_cycle: CycleResult
    worst_cycle: CycleResult
    cycle_results: List[CycleResult]

    def meets_success_criterion(self, threshold: float = 0.80) -> bool:
        """Check if win rate meets the 80% threshold."""
        return self.win_rate >= threshold
```

### 2. Implement cycle decomposition function

```python
def decompose_by_cycles(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    rebalance_dates: List[pd.Timestamp],
) -> CycleMetrics:
    """
    Decompose backtest results into per-rebalance-cycle metrics.

    Args:
        equity_curve: Strategy NAV series (DatetimeIndex)
        benchmark_curve: Benchmark NAV series (DatetimeIndex, e.g., SPY)
        rebalance_dates: List of rebalance dates from backtest

    Returns:
        CycleMetrics with per-cycle breakdown
    """
    cycle_results = []

    for i in range(len(rebalance_dates) - 1):
        cycle_start = rebalance_dates[i]
        cycle_end = rebalance_dates[i + 1]

        # Get returns for this cycle
        # Use .loc to get values at or after the dates
        strat_start = equity_curve.loc[:cycle_start].iloc[-1]
        strat_end = equity_curve.loc[:cycle_end].iloc[-1]
        strategy_return = (strat_end / strat_start) - 1

        bench_start = benchmark_curve.loc[:cycle_start].iloc[-1]
        bench_end = benchmark_curve.loc[:cycle_end].iloc[-1]
        benchmark_return = (bench_end / bench_start) - 1

        active_return = strategy_return - benchmark_return

        cycle_results.append(CycleResult(
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            strategy_return=strategy_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            is_win=active_return > 0,
        ))

    # Aggregate metrics
    winning = [c for c in cycle_results if c.is_win]
    losing = [c for c in cycle_results if not c.is_win]

    return CycleMetrics(
        total_cycles=len(cycle_results),
        winning_cycles=len(winning),
        losing_cycles=len(losing),
        win_rate=len(winning) / len(cycle_results) if cycle_results else 0.0,
        avg_active_return=np.mean([c.active_return for c in cycle_results]),
        avg_win_return=np.mean([c.active_return for c in winning]) if winning else 0.0,
        avg_loss_return=np.mean([c.active_return for c in losing]) if losing else 0.0,
        best_cycle=max(cycle_results, key=lambda c: c.active_return),
        worst_cycle=min(cycle_results, key=lambda c: c.active_return),
        cycle_results=cycle_results,
    )
```

### 3. Add convenience function for BacktestResult

```python
def calculate_cycle_metrics(
    backtest_result: "BacktestResult",
    benchmark_prices: pd.Series,  # SPY prices
) -> CycleMetrics:
    """
    Calculate cycle metrics from a backtest result.

    Args:
        backtest_result: Result from SimpleBacktestEngine
        benchmark_prices: SPY adjusted close prices

    Returns:
        CycleMetrics
    """
    # Convert benchmark prices to NAV (normalized to 1.0 at start)
    benchmark_nav = benchmark_prices / benchmark_prices.iloc[0]

    return decompose_by_cycles(
        equity_curve=backtest_result.equity_curve,
        benchmark_curve=benchmark_nav,
        rebalance_dates=backtest_result.rebalance_dates,
    )
```

### 4. Add regime-conditional cycle metrics (optional enhancement)

```python
def calculate_regime_cycle_metrics(
    backtest_result: "BacktestResult",
    benchmark_prices: pd.Series,
    regime_series: pd.Series,  # Regime classification per date
) -> Dict[str, CycleMetrics]:
    """
    Calculate cycle metrics broken down by market regime.

    Returns dict mapping regime name to CycleMetrics for cycles in that regime.
    """
    base_metrics = calculate_cycle_metrics(backtest_result, benchmark_prices)

    # Group cycles by regime at cycle start
    regime_cycles = {}
    for cycle in base_metrics.cycle_results:
        regime = regime_series.loc[:cycle.cycle_start].iloc[-1]
        if regime not in regime_cycles:
            regime_cycles[regime] = []
        regime_cycles[regime].append(cycle)

    # Calculate metrics per regime
    return {
        regime: _aggregate_cycle_results(cycles)
        for regime, cycles in regime_cycles.items()
    }
```

### 5. Write tests

Create `tests/evaluation/test_cycle_metrics.py`:

```python
import pytest
import pandas as pd
import numpy as np
from quantetf.evaluation.cycle_metrics import (
    CycleResult,
    CycleMetrics,
    decompose_by_cycles,
)


class TestCycleMetrics:
    """Tests for per-cycle win rate metrics."""

    def test_basic_decomposition(self):
        """Test basic cycle decomposition with synthetic data."""
        # Create monthly dates
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-12-31", freq="MS").tolist()

        # Strategy outperforms in 10/12 months = 83.3% win rate
        strategy_nav = pd.Series(
            [1.0 + 0.01 * i + (0.005 if i % 30 < 20 else -0.002) for i in range(len(dates))],
            index=dates,
        )
        benchmark_nav = pd.Series(
            [1.0 + 0.008 * i for i in range(len(dates))],
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        assert metrics.total_cycles == len(rebalance_dates) - 1
        assert 0 <= metrics.win_rate <= 1
        assert len(metrics.cycle_results) == metrics.total_cycles

    def test_success_criterion_check(self):
        """Test the 80% threshold check."""
        # Create metrics with exactly 80% win rate
        metrics = CycleMetrics(
            total_cycles=10,
            winning_cycles=8,
            losing_cycles=2,
            win_rate=0.80,
            avg_active_return=0.01,
            avg_win_return=0.02,
            avg_loss_return=-0.03,
            best_cycle=None,
            worst_cycle=None,
            cycle_results=[],
        )

        assert metrics.meets_success_criterion(0.80) is True
        assert metrics.meets_success_criterion(0.81) is False

    def test_all_winning_cycles(self):
        """Test when strategy beats benchmark every cycle."""
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-06-30", freq="MS").tolist()

        # Strategy always outperforms
        strategy_nav = pd.Series([1.0 + 0.02 * i for i in range(len(dates))], index=dates)
        benchmark_nav = pd.Series([1.0 + 0.01 * i for i in range(len(dates))], index=dates)

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        assert metrics.win_rate == 1.0
        assert metrics.losing_cycles == 0
        assert metrics.meets_success_criterion() is True

    def test_all_losing_cycles(self):
        """Test when benchmark beats strategy every cycle."""
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-06-30", freq="MS").tolist()

        # Benchmark always outperforms
        strategy_nav = pd.Series([1.0 + 0.01 * i for i in range(len(dates))], index=dates)
        benchmark_nav = pd.Series([1.0 + 0.02 * i for i in range(len(dates))], index=dates)

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        assert metrics.win_rate == 0.0
        assert metrics.winning_cycles == 0
        assert metrics.meets_success_criterion() is False
```

### 6. Integrate with backtest reporting

Update `scripts/run_backtest.py` or create helper to print cycle metrics:

```python
def print_cycle_summary(metrics: CycleMetrics) -> None:
    """Print formatted cycle metrics summary."""
    print("\n" + "=" * 60)
    print("CYCLE WIN RATE ANALYSIS")
    print("=" * 60)
    print(f"Total Rebalance Cycles: {metrics.total_cycles}")
    print(f"Winning Cycles: {metrics.winning_cycles}")
    print(f"Losing Cycles: {metrics.losing_cycles}")
    print(f"\n>>> WIN RATE: {metrics.win_rate:.1%} <<<")
    print(f">>> SUCCESS CRITERION (≥80%): {'✓ PASS' if metrics.meets_success_criterion() else '✗ FAIL'} <<<")
    print(f"\nAverage Active Return per Cycle: {metrics.avg_active_return:.2%}")
    print(f"Average Win: +{metrics.avg_win_return:.2%}")
    print(f"Average Loss: {metrics.avg_loss_return:.2%}")
    print(f"\nBest Cycle: {metrics.best_cycle.cycle_start:%Y-%m-%d} → +{metrics.best_cycle.active_return:.2%}")
    print(f"Worst Cycle: {metrics.worst_cycle.cycle_start:%Y-%m-%d} → {metrics.worst_cycle.active_return:.2%}")
    print("=" * 60)
```

---

## Acceptance Criteria

- [ ] `CycleResult` and `CycleMetrics` dataclasses implemented
- [ ] `decompose_by_cycles()` function correctly computes per-cycle returns
- [ ] `calculate_cycle_metrics()` works with BacktestResult
- [ ] `meets_success_criterion()` method checks 80% threshold
- [ ] Unit tests pass with synthetic data
- [ ] Integration test with real backtest result
- [ ] Cycle metrics printed in backtest summary output
- [ ] All code has type hints and docstrings
- [ ] Code follows CLAUDE_CONTEXT.md standards

---

## Definition of Done

1. All acceptance criteria met
2. `pytest tests/evaluation/test_cycle_metrics.py` passes
3. Running a backtest shows cycle win rate in output
4. PROGRESS_LOG.md updated
5. Completion note created: `handoffs/completion-IMPL-015.md`
6. TASKS.md status updated to `completed`
7. Code committed with clear message

---

## Notes

- This is the **most critical metric** for project success
- The 80% threshold is configurable but 80% is the default
- Consider edge cases: single cycle, empty cycles, missing dates
- Regime-conditional metrics are optional but valuable for future analysis
