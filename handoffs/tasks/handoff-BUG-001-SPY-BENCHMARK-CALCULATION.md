# BUG-001: Fix SPY Benchmark Calculation in Strategy Optimizer

**Priority:** CRITICAL
**Estimated Effort:** Small (1-2 hours)
**Dependencies:** None

---

## Summary

The strategy optimizer's multi-period evaluator calculates SPY benchmark returns incorrectly, resulting in **drastically underestimated benchmark performance**. This causes all "active returns" to be overstated, making strategies appear to beat SPY when they may actually be underperforming.

---

## The Bug

### Location
[src/quantetf/optimization/evaluator.py](src/quantetf/optimization/evaluator.py) lines 338-368

### Root Cause

The equity curve from `SimpleBacktestEngine` only contains **rebalance dates** (e.g., 14 monthly dates for a 1-year period), not daily NAV values. When calculating SPY returns, the code aligns SPY daily returns to these sparse rebalance dates:

```python
# Line 339: Gets SPY daily returns
spy_returns = self._get_spy_returns(result.equity_curve.index)

# Line 345: Aligns to equity curve dates (only rebalance dates!)
common_dates = strategy_returns.index.intersection(spy_returns.index)

# Line 350: Only uses returns on those 14 dates
aligned_spy = spy_returns.loc[common_dates]

# Line 361: Compounds only 14 daily returns, missing 258 other days
spy_total_return = (1 + aligned_spy).prod() - 1
```

### Impact

| Metric | Buggy Value | Correct Value |
|--------|-------------|---------------|
| SPY Return (Nov 2024 - Dec 2025) | 1.89% | 14.89% |
| Missing trading days | 258 of 272 | 0 |
| Active return error | +13% overstated | - |

**All optimization results are invalid** - strategies that appear to beat SPY by 10-50% may actually be underperforming.

---

## Required Fix

### Option A: Use Price-Based Return (Recommended)

Calculate SPY return from prices at start and end of the aligned period:

```python
def _evaluate_period(self, config: StrategyConfig, years: int) -> PeriodMetrics:
    # ... existing backtest code ...

    # Get the actual date range from strategy returns
    strategy_returns = result.equity_curve['returns'].dropna()
    eval_start = strategy_returns.index.min()
    eval_end = strategy_returns.index.max()

    # Calculate SPY return using prices (not aligned daily returns)
    spy_prices = self._get_spy_prices(eval_start, eval_end)
    spy_total_return = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1

    # Strategy return is still compounded from its returns
    strategy_total_return = (1 + strategy_returns).prod() - 1

    # ... rest of metrics calculation ...
```

Add helper method:

```python
def _get_spy_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    """Get SPY closing prices for date range."""
    ohlcv_data = self.data_access.prices.read_prices_as_of(
        as_of=end_date + pd.Timedelta(days=1),
        tickers=['SPY'],
    )
    spy_prices = ohlcv_data.xs('Close', level='Price', axis=1)['SPY']
    return spy_prices[(spy_prices.index >= start_date) & (spy_prices.index <= end_date)]
```

### Option B: Use All Trading Days for SPY

Calculate SPY return using all daily returns in the period:

```python
# Get SPY returns for ALL trading days (not aligned to strategy)
spy_returns = self._get_spy_returns_full_period(eval_start, eval_end)
spy_total_return = (1 + spy_returns).prod() - 1
```

### Recommendation

**Use Option A** - it's simpler, more intuitive, and matches how investors typically calculate benchmark returns (price at end / price at start - 1).

---

## Files to Modify

1. **`src/quantetf/optimization/evaluator.py`**
   - Fix `_evaluate_period()` method (lines 285-378)
   - Add `_get_spy_prices()` helper method
   - Update `_get_spy_returns()` or deprecate it

2. **`tests/test_evaluator.py`** (if exists, or create)
   - Add test verifying SPY return matches expected value
   - Test with known date range and expected return

---

## Test Cases

### Test 1: Verify SPY Return Calculation
```python
def test_spy_benchmark_return_calculation():
    """SPY return should match price-based calculation."""
    # Period: 2024-11-29 to 2025-12-31
    # Expected SPY return: ~14.89% (from $593.56 to $681.92)

    evaluator = MultiPeriodEvaluator(data_access, end_date=pd.Timestamp('2025-12-31'))

    # Create a simple config
    config = StrategyConfig(...)
    metrics = evaluator._evaluate_period(config, years=1)

    # SPY return should be close to actual market return
    assert metrics.spy_return > 0.10  # At least 10%, not 1.9%
    assert abs(metrics.spy_return - 0.1489) < 0.02  # Within 2% of expected
```

### Test 2: Active Return Sanity Check
```python
def test_active_return_sanity():
    """Active return should be strategy - benchmark, not inflated."""
    # Run evaluation and verify active return is reasonable
    # A strategy returning 15% vs SPY 14.89% should have ~0% active return
    # Not +13% as currently calculated
```

---

## Acceptance Criteria

- [x] SPY benchmark return calculated from prices, not sparse aligned returns
- [x] Test case verifies SPY return for known period matches expected (~14.89% for Nov 2024 - Dec 2025)
- [x] Re-run optimization produces corrected benchmark figures
- [x] Active returns are realistic (most strategies should NOT beat SPY by 50%+)
- [x] All existing tests pass

**Status: COMPLETED** (2026-01-28)

---

## Verification Steps

After fix, run:

```bash
# Quick verification
python -c "
import pandas as pd
import sys
sys.path.insert(0, 'src')
from quantetf.optimization.evaluator import MultiPeriodEvaluator
from quantetf.data.access import DataAccessFactory

data_access = DataAccessFactory.create_context(
    config={'snapshot_path': 'data/snapshots/snapshot_20260122_010523/data.parquet'},
    enable_caching=True
)
evaluator = MultiPeriodEvaluator(data_access, end_date=pd.Timestamp('2026-01-20'))

# The _get_spy_prices or fixed calculation should return ~14-15% for trailing 1yr
# Not 1.9%
"

# Re-run optimizer on small subset to verify
python scripts/find_best_strategy.py --periods 1 --max-configs 10
```

---

## Notes

- This bug affects ALL previous optimization results
- Consider re-running full optimization after fix
- The strategy returns themselves are correct; only the benchmark calculation is wrong
- Information Ratio and other relative metrics will change significantly after fix
