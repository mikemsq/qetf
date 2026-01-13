# HANDOUT: Momentum Acceleration Strategy Implementation

**Strategy ID**: `momentum_acceleration`
**File**: `/workspaces/qetf/src/quantetf/alpha/momentum_acceleration.py`
**Priority**: HIGH (Phase 1.3)
**Complexity**: LOW (Simplest strategy!)
**Implementation Order**: Best to start here

---

## 1. Executive Summary

### What
Momentum Acceleration measures the *change* in momentum by comparing recent (3-month) returns to longer-term (12-month) returns. The score is `returns_3m - returns_12m`, capturing trend strength and inflection points.

### Why
- **Early signal detection**: Identifies momentum changes before they fully develop
- **Trend strength**: Positive acceleration = strengthening trend
- **Mean reversion protection**: Negative acceleration = weakening momentum (exit before crash)
- **Lower lag**: Reacts faster than traditional 12-month momentum
- **Simple computation**: Just two return calculations and a subtraction!

### Complexity
**LOW** - Simplest of all Phase 1 strategies. Just subtracts two momentum values. No regression, no volatility calculations.

### Priority
**HIGH** - Start with this strategy! It provides tactical timing signals and is the easiest to implement and debug.

---

## 2. Background & Context

### Mathematical Definition

For each ticker `i` at decision time `T`:

1. **Calculate Short-Term Momentum** (3 months = 63 trading days):
   ```python
   returns_3m = (price_T-1 / price_T-63) - 1
   ```

2. **Calculate Long-Term Momentum** (12 months = 252 trading days):
   ```python
   returns_12m = (price_T-1 / price_T-252) - 1
   ```

3. **Calculate Acceleration Score**:
   ```python
   score_i = returns_3m - returns_12m
   ```

4. **Interpretation**:
   - **Positive score**: Recent returns > long-term average → accelerating uptrend
   - **Negative score**: Recent returns < long-term average → decelerating (or reversing)
   - **High positive**: Strong acceleration → buy signal
   - **High negative**: Deceleration → avoid or sell

### Examples

```python
# Example 1: Accelerating uptrend
price_12mo_ago = 100
price_3mo_ago = 110
price_today = 125

returns_12m = (125 / 100) - 1 = 0.25  # +25%
returns_3m = (125 / 110) - 1 = 0.136  # +13.6%
acceleration = 0.136 - 0.25 = -0.114  # Negative (decelerating)

# Example 2: Strong acceleration
price_12mo_ago = 100
price_3mo_ago = 105
price_today = 120

returns_12m = (120 / 100) - 1 = 0.20  # +20%
returns_3m = (120 / 105) - 1 = 0.143  # +14.3%
# Wait, this doesn't work...

# Correct Example 2: Strong acceleration
# The 3M and 12M windows OVERLAP (both end at T-1)
# So we need to think differently:
# If 12M return = +20% total
# And last 3M contributed +15% to that
# Then first 9M only contributed +5%
# Score = +15% - +5% = +10% (accelerating!)
```

### Research Rationale

- **Momentum Lifecycle**: Trends go through phases: acceleration → cruise → deceleration → reversal
- **Early Entry**: Acceleration captures trends early (higher returns)
- **Early Exit**: Deceleration signals exit before crashes (lower drawdowns)
- **Behavioral**: Captures herding behavior (acceleration phase) and profit-taking (deceleration)
- **Academic**: Related to "momentum factor timing" literature (Moskowitz et al.)

### Expected Performance Characteristics

- **CAGR**: 9-13% (higher than vanilla momentum due to early entry)
- **Sharpe Ratio**: 0.7-1.1 (similar to momentum)
- **Max Drawdown**: 18-28% (better than momentum due to early exits)
- **Turnover**: Higher than 12M momentum (more frequent signals)
- **Best Performance**: During regime changes (bull→bear or bear→bull transitions)
- **Correlation to SPY**: 0.4-0.6

---

## 3. Implementation Specification

### File Path
```
/workspaces/qetf/src/quantetf/alpha/momentum_acceleration.py
```

### Class Definition
```python
class MomentumAccelerationAlpha(AlphaModel)
```

### Inheritance
- **Parent**: `quantetf.alpha.base.AlphaModel`
- **Must implement**: `score()` method
- **Pattern**: Very similar to `MomentumAlpha` but with two lookback windows

### Constructor Signature
```python
def __init__(
    self,
    short_lookback_days: int = 63,   # ~3 months
    long_lookback_days: int = 252,    # ~12 months
    min_periods: int = 200
) -> None:
    """Initialize momentum acceleration alpha model.

    Args:
        short_lookback_days: Days for short-term momentum (default: 63 ~ 3 months)
        long_lookback_days: Days for long-term momentum (default: 252 ~ 12 months)
        min_periods: Minimum valid prices required (default: 200)

    Raises:
        ValueError: If short_lookback >= long_lookback
    """
```

### score() Method Signature
```python
def score(
    self,
    *,
    as_of: pd.Timestamp,
    universe: Universe,
    features: FeatureFrame,
    store: DataStore,
    dataset_version: Optional[DatasetVersion] = None,
) -> AlphaScores:
    """Compute momentum acceleration scores.

    Args:
        as_of: Decision date (use T-1 data)
        universe: Set of eligible tickers
        features: Pre-computed features (not used)
        store: SnapshotDataStore for price history
        dataset_version: Optional dataset version

    Returns:
        AlphaScores with acceleration scores.
        Positive = accelerating, Negative = decelerating.

    Algorithm:
        1. Validate store type
        2. Validate short_lookback < long_lookback
        3. Get prices [T-long_lookback, T-1]
        4. For each ticker:
            a. Check min_periods
            b. Calculate short-term return (last short_lookback days)
            c. Calculate long-term return (last long_lookback days)
            d. Score = short_return - long_return
        5. Return AlphaScores
    """
```

### Algorithm Pseudocode

```python
# Step 1: Validate
if not isinstance(store, SnapshotDataStore):
    raise TypeError("MomentumAccelerationAlpha requires SnapshotDataStore")

# Step 2: Get prices (need enough for long lookback)
prices = store.get_close_prices(
    as_of=as_of,
    tickers=list(universe.tickers),
    lookback_days=long_lookback_days + 50  # Buffer
)

# Step 3: Calculate acceleration for each ticker
scores = {}

for ticker in universe.tickers:
    if ticker not in prices.columns:
        scores[ticker] = np.nan
        continue

    ticker_px = prices[ticker].dropna()

    if len(ticker_px) < min_periods:
        scores[ticker] = np.nan
        continue

    # Need enough data for long lookback
    if len(ticker_px) < long_lookback_days:
        scores[ticker] = np.nan
        continue

    # Calculate long-term momentum (full lookback)
    long_window = ticker_px.iloc[-long_lookback_days:]

    if len(long_window) < 2:
        scores[ticker] = np.nan
        continue

    long_return = (long_window.iloc[-1] / long_window.iloc[0]) - 1.0

    # Calculate short-term momentum (recent period)
    short_window = ticker_px.iloc[-short_lookback_days:]

    if len(short_window) < 2:
        scores[ticker] = np.nan
        continue

    short_return = (short_window.iloc[-1] / short_window.iloc[0]) - 1.0

    # Acceleration = short - long
    acceleration = short_return - long_return

    scores[ticker] = acceleration

    logger.debug(
        f"{ticker}: short={short_return:.4f}, long={long_return:.4f}, "
        f"accel={acceleration:.4f}"
    )

# Step 4: Return AlphaScores
scores_series = pd.Series(scores)
return AlphaScores(as_of=as_of, scores=scores_series)
```

### Dependencies and Imports

```python
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from quantetf.alpha.base import AlphaModel
from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore
from quantetf.data.snapshot_store import SnapshotDataStore
```

---

## 4. Code Template

```python
"""Momentum acceleration alpha model.

Ranks tickers by the difference between short-term and long-term momentum,
capturing trend acceleration or deceleration signals.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from quantetf.alpha.base import AlphaModel
from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore
from quantetf.data.snapshot_store import SnapshotDataStore

logger = logging.getLogger(__name__)


class MomentumAccelerationAlpha(AlphaModel):
    """Momentum acceleration: rank by (short_returns - long_returns).

    This model captures trend strength by comparing recent momentum to
    longer-term momentum. Positive acceleration indicates strengthening
    trends (buy signal), while negative acceleration indicates weakening
    trends (sell/avoid signal).

    Mathematical approach:
        score_i = returns_3m_i - returns_12m_i

    Where:
        - returns_3m = (price_T-1 / price_T-63) - 1   # Short-term
        - returns_12m = (price_T-1 / price_T-252) - 1  # Long-term

    Interpretation:
        - score > 0: Accelerating (recent > long-term) → Strong buy
        - score ≈ 0: Steady momentum → Hold
        - score < 0: Decelerating (recent < long-term) → Weak/sell

    Use cases:
        - Early trend detection: Enter during acceleration phase
        - Exit signals: Detect momentum weakening before reversal
        - Regime changes: Capture bull/bear transitions

    Point-in-time compliance:
        - Uses only data BEFORE as_of date (T-1 and earlier)

    Example:
        >>> alpha = MomentumAccelerationAlpha(
        ...     short_lookback_days=63,
        ...     long_lookback_days=252
        ... )
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=features,
        ...     store=store
        ... )
    """

    def __init__(
        self,
        short_lookback_days: int = 63,
        long_lookback_days: int = 252,
        min_periods: int = 200
    ) -> None:
        """Initialize momentum acceleration alpha model.

        Args:
            short_lookback_days: Days for short-term momentum (default: 63 ~ 3M)
            long_lookback_days: Days for long-term momentum (default: 252 ~ 12M)
            min_periods: Minimum valid prices required (default: 200)

        Raises:
            ValueError: If parameters are invalid
        """
        if short_lookback_days >= long_lookback_days:
            raise ValueError(
                f"short_lookback ({short_lookback_days}) must be < "
                f"long_lookback ({long_lookback_days})"
            )
        if min_periods < short_lookback_days:
            raise ValueError(
                f"min_periods ({min_periods}) must be >= "
                f"short_lookback ({short_lookback_days})"
            )
        if short_lookback_days < 20:
            raise ValueError(f"short_lookback must be >= 20 days")

        self.short_lookback_days = short_lookback_days
        self.long_lookback_days = long_lookback_days
        self.min_periods = min_periods

        logger.info(
            f"Initialized MomentumAccelerationAlpha: "
            f"short_lookback={short_lookback_days}, "
            f"long_lookback={long_lookback_days}, "
            f"min_periods={min_periods}"
        )

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> AlphaScores:
        """Compute momentum acceleration scores.

        CRITICAL: Uses only data BEFORE as_of (T-1 and earlier).

        Args:
            as_of: Decision date
            universe: Set of eligible tickers
            features: Pre-computed features (not used)
            store: Data store for price history
            dataset_version: Optional dataset version

        Returns:
            AlphaScores with acceleration scores.
            Positive = accelerating trend, Negative = decelerating.
            NaN = insufficient data.

        Raises:
            TypeError: If store is not SnapshotDataStore
        """
        logger.info(
            f"Computing momentum acceleration as of {as_of} "
            f"for {len(universe.tickers)} tickers"
        )

        # Validate store type
        from quantetf.data.snapshot_store import SnapshotDataStore
        if not isinstance(store, SnapshotDataStore):
            raise TypeError(
                f"MomentumAccelerationAlpha requires SnapshotDataStore, "
                f"got {type(store)}"
            )

        # Get prices
        prices = store.get_close_prices(
            as_of=as_of,
            tickers=list(universe.tickers),
            lookback_days=self.long_lookback_days + 50
        )

        # Calculate scores
        scores = {}
        valid_count = 0

        for ticker in universe.tickers:
            if ticker not in prices.columns:
                logger.debug(f"{ticker}: No price data available")
                scores[ticker] = np.nan
                continue

            ticker_px = prices[ticker].dropna()

            if len(ticker_px) < self.min_periods:
                logger.debug(
                    f"{ticker}: Insufficient data "
                    f"({len(ticker_px)} < {self.min_periods})"
                )
                scores[ticker] = np.nan
                continue

            # Calculate long-term return
            long_window = ticker_px.iloc[-self.long_lookback_days:]
            if len(long_window) < 2:
                scores[ticker] = np.nan
                continue

            long_return = (long_window.iloc[-1] / long_window.iloc[0]) - 1.0

            # Calculate short-term return
            short_window = ticker_px.iloc[-self.short_lookback_days:]
            if len(short_window) < 2:
                scores[ticker] = np.nan
                continue

            short_return = (short_window.iloc[-1] / short_window.iloc[0]) - 1.0

            # Acceleration
            acceleration = short_return - long_return
            scores[ticker] = acceleration
            valid_count += 1

            logger.debug(
                f"{ticker}: short={short_return:.4f}, long={long_return:.4f}, "
                f"accel={acceleration:.4f}"
            )

        logger.info(
            f"Computed acceleration scores for {valid_count}/{len(universe.tickers)} tickers"
        )

        return AlphaScores(as_of=as_of, scores=pd.Series(scores))
```

---

## 5. Testing Specification

### Test File Path
```
/workspaces/qetf/tests/alpha/test_momentum_acceleration.py
```

### Required Test Cases

#### Test 1: `test_positive_acceleration_scenario`
**Setup**:
- TICKER_A:
  - 12M return = +10%
  - 3M return = +20%
  - Expected score = +20% - 10% = +10% (accelerating)

**Expected Behavior**:
- Positive score indicating acceleration

**Assertions**:
```python
assert scores['TICKER_A'] > 0
assert abs(scores['TICKER_A'] - 0.10) < 0.02
```

#### Test 2: `test_negative_acceleration_scenario`
**Setup**:
- TICKER_B:
  - 12M return = +20%
  - 3M return = +5%
  - Expected score = +5% - 20% = -15% (decelerating)

**Expected Behavior**:
- Negative score indicating deceleration

**Assertions**:
```python
assert scores['TICKER_B'] < 0
assert abs(scores['TICKER_B'] - (-0.15)) < 0.02
```

#### Test 3: `test_steady_momentum_zero_acceleration`
**Setup**:
- TICKER_C: Linear uptrend (constant % gain per month)
  - Should have relatively small acceleration value

**Expected Behavior**:
- Score close to zero

**Assertions**:
```python
assert abs(scores['TICKER_C']) < 0.10  # Small acceleration
```

#### Test 4: `test_reversal_detection`
**Setup**:
- TICKER_D: Was rising, now falling
  - 12M return = +15%
  - 3M return = -10%
  - Expected score = -10% - 15% = -25% (strong deceleration)

**Expected Behavior**:
- Strong negative score

**Assertions**:
```python
assert scores['TICKER_D'] < -0.20
```

#### Test 5: `test_insufficient_data_returns_nan`
**Setup**:
- TICKER_E: Only 100 days (< min_periods=200)

**Expected Behavior**:
- NaN score

**Assertions**:
```python
assert pd.isna(scores['TICKER_E'])
```

#### Test 6: `test_no_lookahead_bias`
**Setup**:
- Price spike on as_of date (+20%)
- Test that score doesn't include this spike

**Expected Behavior**:
- Score based only on T-1 data

**Assertions**:
```python
expected = calculate_score_without_spike()
assert abs(scores['TICKER_A'] - expected) < 0.01
```

#### Test 7: `test_parameter_validation`
**Setup**:
- Try creating with short_lookback >= long_lookback

**Expected Behavior**:
- ValueError raised

**Assertions**:
```python
with pytest.raises(ValueError, match="short_lookback.*must be <"):
    MomentumAccelerationAlpha(short_lookback_days=252, long_lookback_days=63)
```

#### Test 8: `test_integration_backtest`
**Setup**:
- Run backtest 2021-2023

**Expected Behavior**:
- Completes successfully
- Competitive Sharpe ratio

**Assertions**:
```python
assert result.metrics['sharpe_ratio'] > 0.4
```

---

## 6. Integration & Deployment

### Configuration File Example

**Path**: `/workspaces/qetf/configs/strategies/momentum_acceleration_top5.yaml`

```yaml
name: momentum_acceleration_top5_ew
universe: configs/universes/tier1_initial_20.yaml
schedule: configs/schedules/monthly_rebalance.yaml
cost_model: configs/costs/flat_10bps.yaml

alpha_model:
  type: momentum_acceleration
  short_lookback_days: 63   # ~3 months
  long_lookback_days: 252    # ~12 months
  min_periods: 200

portfolio_construction:
  type: equal_weight_top_n
  top_n: 5
  constraints:
    max_weight: 0.60
    min_weight: 0.00

description: |
  Momentum acceleration: rank by (3M returns - 12M returns).
  Captures trend strength and inflection points.
```

### Command to Run Backtest

```bash
python scripts/run_backtest.py \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --strategy momentum-acceleration-top5 \
  --output artifacts/backtests
```

### Success Criteria

1. All tests pass
2. Sharpe ratio > 0.5
3. Max drawdown < 35%
4. Information ratio vs SPY > 0.2
5. Code coverage > 90%
6. No lookahead bias

---

## 7. Edge Cases & Gotchas

### Common Pitfalls

1. **Lookback Window Confusion**
   - ❌ WRONG: short_lookback > long_lookback
   - ✅ RIGHT: Validate in `__init__`

2. **Window Overlap Understanding**
   - Both windows end at T-1 (they overlap!)
   - This is correct - we want recent vs overall momentum

3. **Subtraction Order**
   - ❌ WRONG: `long - short` (inverted signal)
   - ✅ RIGHT: `short - long` (positive = acceleration)

### Edge Cases to Handle

| Edge Case | Handling |
|-----------|----------|
| short_lookback >= long_lookback | Raise ValueError in __init__ |
| < min_periods data | Return NaN |
| Negative acceleration (deceleration) | Valid! Keep negative score |
| Both returns negative | Valid! Score = difference |

---

## 8. Acceptance Checklist

- [ ] Inherits from `AlphaModel`
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] Validates short < long lookback
- [ ] Returns `AlphaScores` correctly
- [ ] Uses T-1 data only
- [ ] All 8 tests passing
- [ ] Coverage > 90%
- [ ] Works with backtest engine

---

## 9. Reference Files

- `/workspaces/qetf/src/quantetf/alpha/momentum.py` - Primary reference
- `/workspaces/qetf/src/quantetf/alpha/base.py` - AlphaModel interface
- `/workspaces/qetf/src/quantetf/data/snapshot_store.py` - Data access
- `/workspaces/qetf/tests/test_no_lookahead.py` - Testing patterns

---

## 10. Implementation Checklist

- [ ] Create file `/workspaces/qetf/src/quantetf/alpha/momentum_acceleration.py`
- [ ] Implement `__init__()` with validation (short < long)
- [ ] Implement `score()` method
- [ ] Add logging
- [ ] Create test file
- [ ] Implement tests 1-4
- [ ] Run tests, fix bugs
- [ ] Implement tests 5-8
- [ ] Achieve >90% coverage
- [ ] Run mini backtest
- [ ] Type check + formatting
- [ ] Create config file
- [ ] Run full backtest
- [ ] Complete checklist

**Estimated time**: 2-3 hours (simplest strategy!)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-13
**Author**: Quant Research Architect
**Status**: Ready for Implementation
