# Performance Analysis Refactor - Active Returns Focus

**Created:** January 12, 2026
**Priority:** HIGH
**Type:** Refactoring / Enhancement

---

## Problem Statement

Current implementation shows portfolio performance in isolation. This is **not aligned with project goals**.

**The goal of QuantETF is to beat SPY.** Therefore, all performance analysis must emphasize:
1. Portfolio performance
2. Benchmark performance (SPY)
3. **Active performance (the difference)**

Pure portfolio metrics are less interesting than relative performance vs benchmark.

---

## User Requirement

> "I always want to see the portfolio performance, benchmark performance and active performance. The pure portfolio performance is less interesting than comparison to buy and hold spy strategy. The goal of this project is to create a strategy that can beat spy."

---

## Current State Analysis

### What needs updating:

**1. VIZ-001: Backtest Analysis Notebook** (`notebooks/backtest_analysis.ipynb`)
- Currently shows only strategy equity curve
- Should overlay SPY benchmark automatically
- Should show active return (shaded area between curves)
- Should lead with "Beat SPY by X%" summary

**2. ANALYSIS-003: Strategy Comparison Script** (`scripts/compare_strategies.py`)
- Already has some benchmark comparison capability
- Should make SPY comparison the **default**, not optional
- Metrics table should show strategy vs SPY side-by-side

**3. ANALYSIS-001: Enhanced Metrics Module** (`src/quantetf/evaluation/metrics.py`)
- Add helper function: `calculate_active_metrics(strategy_returns, benchmark_returns)`
- Should return: active return, tracking error, information ratio, beta, alpha
- Make this the **standard output format**

**4. Future: VIZ-004: Auto-Report Generation**
- Must have prominent "Active Performance" section
- Lead with "Strategy beat/underperformed SPY by X%"
- All charts should show strategy vs benchmark overlays

---

## Implementation Guidelines

### Standard Reporting Format

Every performance report should follow this structure:

```
Performance Summary
-------------------
Strategy Return:     +45.2%
SPY Return:          +35.1%
Active Return:       +10.1%  ← THIS IS THE KEY METRIC
Information Ratio:   1.25

Strategy Sharpe:     1.20
SPY Sharpe:          0.90
Sharpe Difference:   +0.30
```

### Visualization Standards

**Equity Curve Chart:**
```python
# ✅ CORRECT: Show strategy vs benchmark overlaid
plt.plot(strategy_equity, label='Strategy', linewidth=2)
plt.plot(spy_equity, label='SPY (Benchmark)', linewidth=2, alpha=0.7)
plt.fill_between(dates, strategy_equity, spy_equity,
                 where=(strategy_equity >= spy_equity),
                 alpha=0.2, color='green', label='Outperformance')
plt.fill_between(dates, strategy_equity, spy_equity,
                 where=(strategy_equity < spy_equity),
                 alpha=0.2, color='red', label='Underperformance')
```

**NOT:**
```python
# ❌ WRONG: Strategy alone (missing context)
plt.plot(strategy_equity, label='Strategy')
```

### Code Helper Pattern

Add to `src/quantetf/evaluation/metrics.py`:

```python
def calculate_active_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> dict[str, float]:
    """Calculate active performance metrics vs benchmark.

    This is the STANDARD format for all performance reporting.

    Args:
        strategy_returns: Strategy period returns
        benchmark_returns: Benchmark period returns (typically SPY)
        periods_per_year: Trading periods per year

    Returns:
        Dictionary with:
        - strategy_return: Total strategy return
        - benchmark_return: Total benchmark return
        - active_return: Excess return vs benchmark
        - tracking_error: Volatility of active returns
        - information_ratio: Active return / tracking error
        - beta: Strategy beta to benchmark
        - alpha: Jensen's alpha (risk-adjusted excess return)
        - strategy_sharpe: Strategy Sharpe ratio
        - benchmark_sharpe: Benchmark Sharpe ratio
    """
    # Implementation here
    pass
```

---

## Files to Update

### High Priority (Phase 3 completed tasks):

1. **notebooks/backtest_analysis.ipynb**
   - Add SPY data loading at start
   - Update all charts to show strategy vs SPY
   - Add active return summary section
   - Use `calculate_active_metrics()` helper

2. **src/quantetf/evaluation/metrics.py**
   - Add `calculate_active_metrics()` function
   - Add tests in `tests/test_advanced_metrics.py`

3. **scripts/compare_strategies.py**
   - Make SPY comparison default (not optional)
   - Show active metrics in output tables

### Medium Priority (future):

4. **src/quantetf/evaluation/benchmarks.py**
   - Already has SPY benchmark, ensure it's used by default

5. **VIZ-004 (when implemented)**
   - Follow new standards from the start

---

## Acceptance Criteria

- [x] `calculate_active_metrics()` function implemented and tested (16 tests in TestCalculateActiveMetrics)
- [x] Backtest analysis notebook shows strategy vs SPY overlaid in all relevant charts
- [x] Notebook leads with active return summary ("🎯 ACTIVE PERFORMANCE SUMMARY")
- [x] All metrics tables show strategy, benchmark, and active metrics side-by-side
- [x] Documentation updated to reflect new standard format
- [x] CLAUDE_CONTEXT.md updated with performance analysis standards (✅ DONE)
- [x] PROJECT_BRIEF.md updated with success criteria (✅ DONE)

---

## Example Output (Target)

### Notebook Summary Cell:
```
🎯 Active Performance Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy beat SPY by +12.3% over the backtest period

Strategy:  +48.5% return, 1.25 Sharpe, -15.2% max drawdown
SPY:       +36.2% return, 0.92 Sharpe, -18.5% max drawdown

Active:    +12.3% excess return
           1.45 Information Ratio
           8.5% tracking error
```

---

## Notes

- This refactor aligns implementation with project goals
- SPY is the default benchmark (can be parameterized for other benchmarks later)
- Active performance = "Is the strategy worth the complexity vs simple SPY buy-and-hold?"
- Tracking error and information ratio are key metrics for active management

---

## Related Files

- [CLAUDE_CONTEXT.md](../CLAUDE_CONTEXT.md) - Performance Analysis Standards section
- [PROJECT_BRIEF.md](../PROJECT_BRIEF.md) - Success Criteria section
- [TASKS.md](../TASKS.md) - Phase 3 tasks list

---

**Status:** ✅ COMPLETED (2026-01-15)
**Estimated Effort:** 2-3 hours (actual: already implemented in prior sessions)
**Dependencies:** None (uses existing code)
