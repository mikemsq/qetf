"""Per-rebalance-cycle metrics for strategy evaluation.

This module provides cycle-level decomposition of backtest results to measure
the primary success criterion: percentage of rebalance cycles that beat the
benchmark (SPY).

Key Concept:
    A "cycle" is the period between two consecutive rebalance dates.
    For monthly rebalancing, each month is one cycle.
    For weekly rebalancing, each week is one cycle.

Success Criterion:
    A strategy is successful if it beats the benchmark in at least 80% of cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantetf.backtest.simple_engine import BacktestResult


@dataclass
class CycleResult:
    """Result for a single rebalance cycle."""

    cycle_start: pd.Timestamp
    cycle_end: pd.Timestamp
    strategy_return: float
    benchmark_return: float
    active_return: float
    is_win: bool  # True if active_return > 0

    def __repr__(self) -> str:
        """String representation with formatted returns."""
        return (
            f"CycleResult({self.cycle_start.date()} → {self.cycle_end.date()}, "
            f"strat={self.strategy_return:+.2%}, bench={self.benchmark_return:+.2%}, "
            f"active={self.active_return:+.2%}, win={self.is_win})"
        )


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
        """Check if win rate meets the 80% threshold.

        Args:
            threshold: Required win rate to pass (default 0.80 = 80%)

        Returns:
            True if win_rate >= threshold, False otherwise
        """
        return self.win_rate >= threshold

    def summary(self, threshold: float = 0.80) -> dict:
        """Return summary statistics as dictionary.

        Args:
            threshold: Required win rate for success criterion

        Returns:
            Dictionary with key metrics
        """
        return {
            'total_cycles': self.total_cycles,
            'winning_cycles': self.winning_cycles,
            'losing_cycles': self.losing_cycles,
            'win_rate': self.win_rate,
            'avg_active_return': self.avg_active_return,
            'avg_win_return': self.avg_win_return,
            'avg_loss_return': self.avg_loss_return,
            'best_cycle_active_return': self.best_cycle.active_return,
            'worst_cycle_active_return': self.worst_cycle.active_return,
            'meets_success_criterion': self.meets_success_criterion(threshold),
        }


def decompose_by_cycles(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    rebalance_dates: List[pd.Timestamp],
) -> CycleMetrics:
    """Decompose backtest results into per-rebalance-cycle metrics.

    Args:
        equity_curve: Strategy NAV series (DatetimeIndex, values = portfolio value)
        benchmark_curve: Benchmark NAV series (DatetimeIndex, normalized to 1.0 at start)
        rebalance_dates: List of rebalance dates from backtest, in chronological order

    Returns:
        CycleMetrics with per-cycle breakdown and aggregate statistics

    Raises:
        ValueError: If inputs are invalid (empty, mismatched, single cycle, etc.)

    Example:
        >>> dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        >>> strat_nav = pd.Series([1.0 + 0.001*i for i in range(len(dates))], index=dates)
        >>> bench_nav = pd.Series([1.0 + 0.0008*i for i in range(len(dates))], index=dates)
        >>> rebalance_dates = pd.date_range("2024-01-01", "2024-12-31", freq="MS").tolist()
        >>> metrics = decompose_by_cycles(strat_nav, bench_nav, rebalance_dates)
        >>> print(f"Win Rate: {metrics.win_rate:.1%}")
    """
    if len(rebalance_dates) < 2:
        raise ValueError(
            f"Need at least 2 rebalance dates to form cycles, got {len(rebalance_dates)}"
        )

    if len(equity_curve) == 0 or len(benchmark_curve) == 0:
        raise ValueError("Equity curve and benchmark curve cannot be empty")

    cycle_results = []

    # Iterate through consecutive rebalance dates
    for i in range(len(rebalance_dates) - 1):
        cycle_start = rebalance_dates[i]
        cycle_end = rebalance_dates[i + 1]

        # Get NAV values at cycle boundaries
        # Use .loc to access by timestamp, handling cases where exact dates might not exist
        # Use forward fill or nearest value to handle missing dates
        try:
            strat_start_val = equity_curve.loc[:cycle_start].iloc[-1]
            strat_end_val = equity_curve.loc[:cycle_end].iloc[-1]
        except (IndexError, KeyError) as e:
            raise ValueError(
                f"Cannot extract strategy NAV for cycle {i} ({cycle_start} → {cycle_end}): {e}"
            )

        try:
            bench_start_val = benchmark_curve.loc[:cycle_start].iloc[-1]
            bench_end_val = benchmark_curve.loc[:cycle_end].iloc[-1]
        except (IndexError, KeyError) as e:
            raise ValueError(
                f"Cannot extract benchmark NAV for cycle {i} ({cycle_start} → {cycle_end}): {e}"
            )

        # Calculate returns for this cycle
        if strat_start_val <= 0 or bench_start_val <= 0:
            raise ValueError(
                f"Invalid NAV values at cycle start {cycle_start}: "
                f"strat={strat_start_val}, bench={bench_start_val}"
            )

        strategy_return = float((strat_end_val / strat_start_val) - 1.0)
        benchmark_return = float((bench_end_val / bench_start_val) - 1.0)
        active_return = float(strategy_return - benchmark_return)

        cycle_results.append(
            CycleResult(
                cycle_start=cycle_start,
                cycle_end=cycle_end,
                strategy_return=strategy_return,
                benchmark_return=benchmark_return,
                active_return=active_return,
                is_win=active_return > 0,
            )
        )

    # Aggregate metrics
    winning = [c for c in cycle_results if c.is_win]
    losing = [c for c in cycle_results if not c.is_win]

    if not cycle_results:
        raise ValueError("No cycles generated from rebalance dates")

    win_rate = len(winning) / len(cycle_results)
    avg_active_return = float(np.mean([c.active_return for c in cycle_results]))
    avg_win_return = (
        float(np.mean([c.active_return for c in winning]))
        if winning
        else 0.0
    )
    avg_loss_return = (
        float(np.mean([c.active_return for c in losing]))
        if losing
        else 0.0
    )
    best_cycle = max(cycle_results, key=lambda c: c.active_return)
    worst_cycle = min(cycle_results, key=lambda c: c.active_return)

    return CycleMetrics(
        total_cycles=len(cycle_results),
        winning_cycles=len(winning),
        losing_cycles=len(losing),
        win_rate=win_rate,
        avg_active_return=avg_active_return,
        avg_win_return=avg_win_return,
        avg_loss_return=avg_loss_return,
        best_cycle=best_cycle,
        worst_cycle=worst_cycle,
        cycle_results=cycle_results,
    )


def calculate_cycle_metrics(
    backtest_result: BacktestResult,
    benchmark_prices: pd.Series,
    rebalance_dates: List[pd.Timestamp],
) -> CycleMetrics:
    """Calculate cycle metrics from a backtest result.

    Args:
        backtest_result: Result from SimpleBacktestEngine
        benchmark_prices: SPY adjusted close prices (Series with DatetimeIndex)
        rebalance_dates: List of rebalance dates used in the backtest

    Returns:
        CycleMetrics with per-cycle and aggregate statistics

    Raises:
        ValueError: If inputs are invalid or inconsistent

    Example:
        >>> from quantetf.backtest.simple_engine import SimpleBacktestEngine
        >>> result = engine.run(...)
        >>> spy_prices = store.get_prices('SPY')
        >>> metrics = calculate_cycle_metrics(result, spy_prices, rebalance_dates)
        >>> print(f"Success: {metrics.meets_success_criterion()}")
    """
    if benchmark_prices.empty:
        raise ValueError("Benchmark prices cannot be empty")

    # Extract strategy NAV (equity curve is a DataFrame with 'nav' column)
    if isinstance(backtest_result.equity_curve, pd.DataFrame):
        equity_curve = backtest_result.equity_curve['nav']
    else:
        equity_curve = backtest_result.equity_curve

    # Normalize benchmark prices to NAV (1.0 at start)
    benchmark_nav = benchmark_prices / benchmark_prices.iloc[0]

    # Align indices and handle mismatches
    if not equity_curve.index.equals(benchmark_nav.index):
        # Reindex benchmark to match strategy (forward fill missing dates)
        benchmark_nav = benchmark_nav.reindex(equity_curve.index, method='ffill')

    return decompose_by_cycles(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_nav,
        rebalance_dates=rebalance_dates,
    )


def print_cycle_summary(metrics: CycleMetrics, threshold: float = 0.80) -> None:
    """Print formatted cycle metrics summary to stdout.

    Args:
        metrics: CycleMetrics from decompose_by_cycles
        threshold: Success criterion threshold (default 0.80 = 80%)

    Example:
        >>> print_cycle_summary(metrics)
        >>> print_cycle_summary(metrics, threshold=0.75)
    """
    passes_criterion = metrics.meets_success_criterion(threshold)
    criterion_status = "✓ PASS" if passes_criterion else "✗ FAIL"

    print("\n" + "=" * 70)
    print("CYCLE WIN RATE ANALYSIS")
    print("=" * 70)
    print(f"Total Rebalance Cycles: {metrics.total_cycles}")
    print(f"Winning Cycles: {metrics.winning_cycles}")
    print(f"Losing Cycles: {metrics.losing_cycles}")
    print(f"\n>>> WIN RATE: {metrics.win_rate:.1%} <<<")
    print(f">>> SUCCESS CRITERION (≥{threshold:.0%}): {criterion_status} <<<")
    print(f"\nAverage Active Return per Cycle: {metrics.avg_active_return:+.2%}")
    if metrics.winning_cycles > 0:
        print(f"Average Win Return: +{metrics.avg_win_return:.2%}")
    if metrics.losing_cycles > 0:
        print(f"Average Loss Return: {metrics.avg_loss_return:.2%}")
    print(f"\nBest Cycle ({metrics.best_cycle.cycle_start.date()} → "
          f"{metrics.best_cycle.cycle_end.date()}): "
          f"+{metrics.best_cycle.active_return:.2%}")
    print(f"Worst Cycle ({metrics.worst_cycle.cycle_start.date()} → "
          f"{metrics.worst_cycle.cycle_end.date()}): "
          f"{metrics.worst_cycle.active_return:.2%}")
    print("=" * 70)


def cycle_metrics_dataframe(metrics: CycleMetrics) -> pd.DataFrame:
    """Convert cycle results to pandas DataFrame for analysis.

    Args:
        metrics: CycleMetrics from decompose_by_cycles

    Returns:
        DataFrame with columns: start_date, end_date, strat_return, bench_return,
                                active_return, is_win

    Example:
        >>> df = cycle_metrics_dataframe(metrics)
        >>> print(df.describe())
    """
    data = {
        'start_date': [c.cycle_start for c in metrics.cycle_results],
        'end_date': [c.cycle_end for c in metrics.cycle_results],
        'strat_return': [c.strategy_return for c in metrics.cycle_results],
        'bench_return': [c.benchmark_return for c in metrics.cycle_results],
        'active_return': [c.active_return for c in metrics.cycle_results],
        'is_win': [c.is_win for c in metrics.cycle_results],
    }
    return pd.DataFrame(data)
