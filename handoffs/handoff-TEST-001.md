# Task Handoff: TEST-001 - No-Lookahead Validation Tests

**Task ID:** TEST-001
**Status:** ready
**Priority:** critical
**Estimated Time:** 2 hours

---

## Quick Context

You are creating a comprehensive test suite to verify that our backtest system has no lookahead bias. This is **critical** for credible backtesting - using future data invalidates all results.

**Why this matters:** Lookahead bias is the #1 cause of failed quant strategies in production. These tests act as our safety net to prevent accidentally using future information.

---

## What You Need to Know

### What is Lookahead Bias?

Using information that wouldn't have been available at the decision time:
- ❌ Using today's close to make today's decision
- ❌ Using tomorrow's data to make today's decision
- ✅ Using yesterday's close to make today's decision (T-1)

### Our Design Principles

1. **T-1 Data Access:** All decisions use data from *before* the decision date
2. **Strict Inequality:** `data.index < as_of` (not `<=`)
3. **Point-in-Time:** Snapshots represent what was known at a specific date

### Common Lookahead Mistakes

- Forgetting to filter data by `as_of`
- Using `<=` instead of `<`
- Calculating features with future data
- Using unadjusted prices that include future splits

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Lookahead bias section
2. **`/workspaces/qetf/src/quantetf/data/snapshot_store.py`** - How we enforce T-1
3. **`/workspaces/qetf/src/quantetf/alpha/momentum.py`** - Example using T-1 correctly
4. **`/workspaces/qetf/tests/test_yfinance_provider.py`** - Test patterns

---

## Implementation Steps

### 1. Create test file

Create `/workspaces/qetf/tests/test_no_lookahead.py`

### 2. Create synthetic test data

**Key idea:** Create fake price data where we *know* what the answer should be, then verify we never see future data.

```python
"""Tests to verify no lookahead bias in data access and models."""

import pandas as pd
import numpy as np
import pytest
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.types import Universe


def create_synthetic_prices():
    """Create synthetic price data for testing.

    Creates a simple dataset where prices increment daily:
    - Date 2023-01-01: all prices = 100
    - Date 2023-01-02: all prices = 101
    - Date 2023-01-03: all prices = 102
    - etc.

    This makes it easy to verify: if we see price=105 on 2023-01-03,
    we know we're using future data!
    """
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    tickers = ['TICKER_A', 'TICKER_B', 'TICKER_C']

    # Create MultiIndex DataFrame
    arrays = [
        np.repeat(tickers, len(dates)),  # Ticker level
        np.tile(dates, len(tickers))     # Date level
    ]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=['ticker', 'date'])

    # Prices increment daily: date index + 100
    # Jan 1 (day 0) = 100, Jan 2 (day 1) = 101, etc.
    prices = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            prices.append(100 + i)

    df = pd.DataFrame({
        'close': prices
    }, index=multi_index)

    # Pivot to (Date, (Ticker, Field)) format
    df_reset = df.reset_index()
    df_pivot = df_reset.pivot_table(
        index='date',
        columns='ticker',
        values='close'
    )

    # Convert to MultiIndex columns
    df_pivot.columns = pd.MultiIndex.from_product(
        [df_pivot.columns, ['close']],
        names=['ticker', 'field']
    )

    return df_pivot
```

### 3. Test SnapshotDataStore T-1 enforcement

```python
def test_snapshot_store_t1_access():
    """Verify SnapshotDataStore enforces T-1 data access."""
    # Create synthetic data
    prices = create_synthetic_prices()

    # Save to temp parquet file
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = os.path.join(tmpdir, 'test_snapshot')
        os.makedirs(snapshot_path)
        parquet_path = os.path.join(snapshot_path, 'prices.parquet')
        prices.to_parquet(parquet_path)

        # Load with SnapshotDataStore
        store = SnapshotDataStore(snapshot_path)

        # Request data as of 2023-01-10 (day 9, price should be 109)
        # But T-1 means we should only see data through 2023-01-09 (day 8, price=108)
        as_of = pd.Timestamp('2023-01-10')
        data = store.get_close_prices(as_of=as_of)

        # Latest date in data should be 2023-01-09 (day before as_of)
        assert data.index.max() < as_of
        assert data.index.max() == pd.Timestamp('2023-01-09')

        # Latest price should be 108 (day 8), NOT 109 (day 9)
        # This proves we're not seeing data from as_of date
        latest_prices = data.loc[data.index.max()]
        assert all(latest_prices == 108)
```

### 4. Test MomentumAlpha doesn't use future data

```python
def test_momentum_alpha_no_lookahead():
    """Verify MomentumAlpha doesn't use future data."""
    # Create synthetic data
    prices = create_synthetic_prices()

    # Save to temp snapshot
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = os.path.join(tmpdir, 'test_snapshot')
        os.makedirs(snapshot_path)
        parquet_path = os.path.join(snapshot_path, 'prices.parquet')
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        # Create universe
        universe = Universe(
            as_of=pd.Timestamp('2023-01-10'),
            tickers=('TICKER_A', 'TICKER_B', 'TICKER_C')
        )

        # Calculate momentum as of 2023-01-10
        alpha = MomentumAlpha(lookback_days=5)
        scores = alpha.score(
            as_of=pd.Timestamp('2023-01-10'),
            universe=universe,
            features=None,
            store=store
        )

        # Momentum should be calculated using:
        # - Current price (T-1): 2023-01-09 = 108
        # - Lookback price (T-1-5): 2023-01-04 = 103
        # - Return = (108 / 103) - 1 = 0.0485 (4.85%)

        # All tickers should have same momentum (same price pattern)
        expected_return = (108 / 103) - 1.0
        for ticker in universe.tickers:
            assert scores.scores[ticker] == pytest.approx(expected_return, rel=1e-4)
```

### 5. Test strict inequality (< not <=)

```python
def test_strict_inequality_t1():
    """Verify we use < not <= for T-1 filtering."""
    prices = create_synthetic_prices()

    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = os.path.join(tmpdir, 'test_snapshot')
        os.makedirs(snapshot_path)
        parquet_path = os.path.join(snapshot_path, 'prices.parquet')
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        # Request data as of 2023-01-10
        as_of = pd.Timestamp('2023-01-10')
        data = store.get_close_prices(as_of=as_of)

        # Should NOT include as_of date (strict <)
        assert as_of not in data.index

        # Should include day before
        assert pd.Timestamp('2023-01-09') in data.index
```

### 6. Test with lookback window

```python
def test_lookback_window_no_lookahead():
    """Verify lookback windows don't include future data."""
    prices = create_synthetic_prices()

    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = os.path.join(tmpdir, 'test_snapshot')
        os.makedirs(snapshot_path)
        parquet_path = os.path.join(snapshot_path, 'prices.parquet')
        prices.to_parquet(parquet_path)

        store = SnapshotDataStore(snapshot_path)

        # Request 10 days of data as of 2023-01-20
        as_of = pd.Timestamp('2023-01-20')
        data = store.read_prices(as_of=as_of, lookback_days=10)

        # Should have 10 days of data
        assert len(data) == 10

        # Latest date should be 2023-01-19 (T-1)
        assert data.index.max() == pd.Timestamp('2023-01-19')

        # Earliest date should be 2023-01-10 (T-1 minus 9 more days)
        assert data.index.min() == pd.Timestamp('2023-01-10')
```

---

## Acceptance Criteria

- [ ] Test suite verifies SnapshotDataStore enforces T-1
- [ ] Tests verify strict inequality (< not <=)
- [ ] Tests verify MomentumAlpha doesn't use future data
- [ ] Tests use synthetic data with known answers
- [ ] Tests verify lookback windows are correct
- [ ] All edge cases covered (as_of on weekend, missing data, etc.)
- [ ] Tests are well-documented explaining what they verify
- [ ] All tests pass: `pytest tests/test_no_lookahead.py -v`

---

## Success Looks Like

```bash
$ pytest tests/test_no_lookahead.py -v

tests/test_no_lookahead.py::test_snapshot_store_t1_access PASSED
tests/test_no_lookahead.py::test_momentum_alpha_no_lookahead PASSED
tests/test_no_lookahead.py::test_strict_inequality_t1 PASSED
tests/test_no_lookahead.py::test_lookback_window_no_lookahead PASSED

All tests verify we NEVER see future data!
```

---

## Additional Test Ideas

- Test that as_of date on weekend uses Friday data
- Test with missing data (some tickers have gaps)
- Test with insufficient lookback (not enough history)
- Test edge case: as_of = first date in dataset

---

## Questions? Issues?

If blocked or unclear:
1. **Check CLAUDE_CONTEXT.md** - "No Lookahead Bias" section
2. **Study SnapshotDataStore** - See how it filters data
3. **Look at existing tests** - Pattern in test_yfinance_provider.py
4. **Document questions** in `handoffs/completion-TEST-001.md`

---

## When Done

1. Verify all tests pass
2. Create `handoffs/completion-TEST-001.md` with:
   - What you tested
   - Test results
   - Any edge cases discovered
   - Confidence level in no-lookahead enforcement
3. Update `TASKS.md`: change status to `completed`
4. Commit with message: "Add no-lookahead validation tests (TEST-001)"

---

## Important Reminder

**Save progress frequently!** Update your completion note as you go:
- After creating synthetic data helper
- After each test function
- After tests pass
- After finding any issues

This way if you hit quota limits, the next session can resume easily.

---

## Why This Is Critical

These tests are our **credibility insurance**. When your backtest shows great returns, these tests prove it's not because of lookahead bias. Without them, we can't trust any results.
