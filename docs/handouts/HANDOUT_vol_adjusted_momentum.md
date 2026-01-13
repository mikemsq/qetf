# HANDOUT: Volatility-Adjusted Momentum Strategy Implementation

**Strategy ID**: `vol_adjusted_momentum`
**File**: `/workspaces/qetf/src/quantetf/alpha/vol_adjusted_momentum.py`
**Priority**: HIGH (Phase 1.2)
**Complexity**: LOW
**Implementation Order**: 2 of 3 (or implement first if you prefer defensive strategies)

---

## 1. Executive Summary

### What
Volatility-Adjusted Momentum ranks ETFs by risk-adjusted returns using a Sharpe-style metric: `score = returns / realized_volatility`. This rewards consistent performers over volatile ones.

### Why
- **Better drawdown control**: Penalizes high-volatility assets that can cause large losses
- **Smoother returns**: Favors stable uptrends over erratic gains
- **Risk-aware allocation**: Automatically reduces exposure to volatile periods
- **Simple to implement**: Single division operation, no regression required
- **Defensive characteristics**: Naturally tilts portfolio toward lower-risk assets

### Complexity
**LOW** - Straightforward calculation requiring only returns and volatility. Simpler than residual momentum, slightly more complex than acceleration (needs std dev calculation).

### Priority
**HIGH** - Phase 1, Strategy 2 of 3. Great second choice after momentum acceleration, or implement first if you want defensive characteristics.

---

## 2. Background & Context

### Mathematical Definition

For each ticker `i` at decision time `T`:

1. **Calculate Cumulative Returns**:
   ```python
   cum_return_i = (price_T-1 / price_T-252) - 1
   ```

2. **Calculate Realized Volatility**:
   ```python
   daily_returns = prices.pct_change()  # Last 252 days
   realized_vol_i = daily_returns.std() * sqrt(252)  # Annualized
   ```

3. **Calculate Sharpe-Style Score**:
   ```python
   score_i = cum_return_i / realized_vol_i
   ```

4. **Ranking**: Higher score = better risk-adjusted momentum

### Examples

```python
# Example 1: High return, low vol (BEST)
ETF_A: return = +20%, vol = 10%
score_A = 0.20 / 0.10 = 2.0

# Example 2: High return, high vol (NEUTRAL)
ETF_B: return = +25%, vol = 25%
score_B = 0.25 / 0.25 = 1.0

# Example 3: Moderate return, low vol (GOOD)
ETF_C: return = +15%, vol = 8%
score_C = 0.15 / 0.08 = 1.875

# Ranking: A (2.0) > C (1.875) > B (1.0)
# A wins despite lower return than B due to much lower vol
```

### Research Rationale

- **Risk-Return Tradeoff**: Traditional momentum ignores volatility. A 20% return with 5% vol is better than 25% return with 15% vol.
- **Volatility Clustering**: High volatility tends to persist → avoiding volatile assets reduces future downside risk
- **Behavioral Finance**: Volatile assets have higher crash risk due to panic selling
- **Empirical Evidence**: Sharpe ratio ranking has shown 10-20% better Sharpe than raw momentum in academic studies

### Expected Performance Characteristics

- **CAGR**: 7-10% (slightly lower than price momentum)
- **Sharpe Ratio**: 0.9-1.3 (significantly higher than price momentum's 0.6-0.9)
- **Max Drawdown**: 12-20% (better than price momentum's 20-30%)
- **Correlation to SPY**: 0.5-0.7
- **Win Rate**: 55-60% (days beating SPY)
- **Best Use Case**: Bear markets and high-volatility regimes

---

## 3. Implementation Specification

### File Path
```
/workspaces/qetf/src/quantetf/alpha/vol_adjusted_momentum.py
```

### Class Definition
```python
class VolAdjustedMomentumAlpha(AlphaModel)
```

### Inheritance
- **Parent**: `quantetf.alpha.base.AlphaModel`
- **Must implement**: `score()` method
- **Pattern**: Follow `MomentumAlpha` structure (simpler - no regression)

### Constructor Signature
```python
def __init__(
    self,
    lookback_days: int = 252,
    min_periods: int = 200,
    vol_floor: float = 0.01,
    annualization_factor: float = None  # Defaults to sqrt(252)
) -> None:
    """Initialize volatility-adjusted momentum alpha model.

    Args:
        lookback_days: Days for return and volatility calculation (default: 252)
        min_periods: Minimum valid prices required (default: 200)
        vol_floor: Minimum volatility to prevent division by zero (default: 0.01 = 1%)
        annualization_factor: Factor to annualize vol (default: sqrt(252) ≈ 15.87)
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
    """Compute volatility-adjusted momentum scores.

    Args:
        as_of: Decision date (use data up to T-1)
        universe: Set of eligible tickers
        features: Pre-computed features (not used)
        store: SnapshotDataStore for price history
        dataset_version: Optional dataset version

    Returns:
        AlphaScores with vol-adjusted scores (higher = better risk-adjusted return)

    Algorithm:
        1. Validate store type
        2. Get prices [T-lookback_days, T-1] for universe
        3. For each ticker:
            a. Check min_periods
            b. Calculate cumulative return over lookback
            c. Calculate realized volatility (annualized std of daily returns)
            d. Apply vol_floor to prevent division by zero
            e. Score = cumulative_return / realized_vol
        4. Return AlphaScores
    """
```

### Algorithm Pseudocode

```python
# Step 1: Validate inputs
if not isinstance(store, SnapshotDataStore):
    raise TypeError("VolAdjustedMomentumAlpha requires SnapshotDataStore")

# Step 2: Get prices for universe (T-1 and earlier)
prices = store.get_close_prices(
    as_of=as_of,
    tickers=list(universe.tickers),
    lookback_days=lookback_days + 50  # Buffer
)

# Step 3: Calculate scores for each ticker
scores = {}

for ticker in universe.tickers:
    if ticker not in prices.columns:
        scores[ticker] = np.nan
        continue

    ticker_px = prices[ticker].dropna()

    if len(ticker_px) < min_periods:
        scores[ticker] = np.nan
        continue

    # Use last N days
    window = ticker_px.iloc[-lookback_days:]

    if len(window) < 2:
        scores[ticker] = np.nan
        continue

    # Calculate cumulative return
    cum_return = (window.iloc[-1] / window.iloc[0]) - 1.0

    # Calculate daily returns and realized volatility
    daily_returns = window.pct_change().dropna()

    if len(daily_returns) < 10:  # Need sufficient returns for vol
        scores[ticker] = np.nan
        continue

    realized_vol = daily_returns.std()

    # Annualize volatility: σ_annual = σ_daily * sqrt(252)
    realized_vol_annual = realized_vol * np.sqrt(annualization_factor)

    # Apply floor to prevent division by near-zero
    realized_vol_annual = max(realized_vol_annual, vol_floor)

    # Sharpe-style score
    score = cum_return / realized_vol_annual

    scores[ticker] = score

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
"""Volatility-adjusted momentum alpha model.

Ranks tickers by risk-adjusted returns using a Sharpe-style metric.
Penalizes volatile assets in favor of smooth, consistent performers.
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


class VolAdjustedMomentumAlpha(AlphaModel):
    """Volatility-adjusted momentum: rank by returns / realized_vol.

    This model computes a Sharpe-style score for each ticker by dividing
    cumulative returns by realized volatility. Higher scores indicate
    better risk-adjusted performance.

    Mathematical approach:
        score_i = (cumulative_return_i) / (realized_volatility_i)

    Where:
        - cumulative_return = (price_T-1 / price_T-lookback) - 1
        - realized_volatility = std(daily_returns) * sqrt(252)

    This naturally favors:
        - High returns with low volatility (best)
        - Moderate returns with low volatility (good)
        - High returns with high volatility (neutral)
        - Low returns with high volatility (worst)

    Point-in-time compliance:
        - Uses only data BEFORE as_of date (T-1 and earlier)
        - No lookahead bias

    Example:
        >>> alpha = VolAdjustedMomentumAlpha(lookback_days=252)
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=features,
        ...     store=store
        ... )
    """

    def __init__(
        self,
        lookback_days: int = 252,
        min_periods: int = 200,
        vol_floor: float = 0.01,
        annualization_factor: float = None
    ) -> None:
        """Initialize volatility-adjusted momentum alpha model.

        Args:
            lookback_days: Trading days for return/vol calculation (default: 252)
            min_periods: Minimum valid prices required (default: 200)
            vol_floor: Minimum volatility to prevent div-by-zero (default: 0.01 = 1% annual)
            annualization_factor: Factor to annualize vol (default: 252 trading days)

        Raises:
            ValueError: If parameters are invalid
        """
        if lookback_days < min_periods:
            raise ValueError(f"lookback_days must be >= min_periods")
        if min_periods < 20:
            raise ValueError(f"min_periods must be >= 20 for stable vol estimate")
        if vol_floor <= 0:
            raise ValueError(f"vol_floor must be > 0")

        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.vol_floor = vol_floor
        self.annualization_factor = annualization_factor if annualization_factor is not None else 252.0

        logger.info(
            f"Initialized VolAdjustedMomentumAlpha: "
            f"lookback={lookback_days}, min_periods={min_periods}, "
            f"vol_floor={vol_floor}, annualization={self.annualization_factor}"
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
        """Compute volatility-adjusted momentum scores.

        CRITICAL: Uses only data BEFORE as_of (T-1 and earlier).

        Args:
            as_of: Decision date
            universe: Set of eligible tickers
            features: Pre-computed features (not used)
            store: Data store for price history
            dataset_version: Optional dataset version

        Returns:
            AlphaScores with vol-adjusted momentum scores.
            Higher scores = better risk-adjusted returns.
            NaN scores = insufficient data.

        Raises:
            TypeError: If store is not SnapshotDataStore
        """
        logger.info(
            f"Computing vol-adjusted momentum as of {as_of} "
            f"for {len(universe.tickers)} tickers"
        )

        # Validate store type
        from quantetf.data.snapshot_store import SnapshotDataStore
        if not isinstance(store, SnapshotDataStore):
            raise TypeError(
                f"VolAdjustedMomentumAlpha requires SnapshotDataStore, "
                f"got {type(store)}"
            )

        # Get prices
        prices = store.get_close_prices(
            as_of=as_of,
            tickers=list(universe.tickers),
            lookback_days=self.lookback_days + 50
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

            # Use last N days
            window = ticker_px.iloc[-self.lookback_days:]

            if len(window) < 2:
                scores[ticker] = np.nan
                continue

            # Calculate cumulative return
            cum_return = (window.iloc[-1] / window.iloc[0]) - 1.0

            # Calculate daily returns
            daily_returns = window.pct_change().dropna()

            if len(daily_returns) < 10:
                scores[ticker] = np.nan
                continue

            # Realized volatility (annualized)
            realized_vol = daily_returns.std()
            realized_vol_annual = realized_vol * np.sqrt(self.annualization_factor)

            # Apply floor to prevent division by zero
            realized_vol_annual = max(realized_vol_annual, self.vol_floor)

            # Sharpe-style score
            score = cum_return / realized_vol_annual

            scores[ticker] = score
            valid_count += 1

            logger.debug(
                f"{ticker}: return={cum_return:.4f}, vol={realized_vol_annual:.4f}, "
                f"score={score:.4f}"
            )

        logger.info(
            f"Computed vol-adjusted scores for {valid_count}/{len(universe.tickers)} tickers"
        )

        return AlphaScores(as_of=as_of, scores=pd.Series(scores))
```

---

## 5. Testing Specification

### Test File Path
```
/workspaces/qetf/tests/alpha/test_vol_adjusted_momentum.py
```

### Required Test Cases

#### Test 1: `test_vol_adjusted_ranks_correctly`
**Setup**:
- TICKER_A: +20% return, 10% volatility → score = 2.0
- TICKER_B: +10% return, 5% volatility → score = 2.0
- TICKER_C: +20% return, 40% volatility → score = 0.5

**Expected**:
- TICKER_A and TICKER_B should have similar scores (both ~2.0)
- TICKER_C should have lower score (~0.5)

**Assertions**:
```python
assert abs(scores['TICKER_A'] - 2.0) < 0.3
assert abs(scores['TICKER_B'] - 2.0) < 0.3
assert scores['TICKER_C'] < 1.0
```

#### Test 2: `test_vol_floor_prevents_division_by_zero`
**Setup**:
- TICKER_A: constant price (zero volatility)

**Expected**:
- Should use vol_floor instead of zero
- Score should be finite (not inf)

**Assertions**:
```python
assert np.isfinite(scores['TICKER_A'])
assert scores['TICKER_A'] < 1000  # Not extremely high
```

#### Test 3: `test_negative_returns_negative_scores`
**Setup**:
- TICKER_A: -10% return, 5% volatility

**Expected**:
- Score should be negative (-2.0)

**Assertions**:
```python
assert scores['TICKER_A'] < 0
assert abs(scores['TICKER_A'] - (-2.0)) < 0.3
```

#### Test 4: `test_insufficient_data_returns_nan`
**Setup**:
- TICKER_A: only 50 days (< min_periods=200)

**Expected**:
- Score = NaN

**Assertions**:
```python
assert pd.isna(scores['TICKER_A'])
```

#### Test 5: `test_no_lookahead_bias`
**Setup**:
- Prices jump 10% on as_of date
- Test that score doesn't include this jump

**Expected**:
- Score based only on T-1 and earlier

**Assertions**:
```python
# Manual calculation with T-1 data
expected_score = calculate_with_t_minus_1_data()
assert abs(scores['TICKER_A'] - expected_score) < 0.01
```

#### Test 6: `test_annualization_factor_applied`
**Setup**:
- Daily vol = 0.01 (1%)
- annualization_factor = 252

**Expected**:
- Annual vol = 0.01 * sqrt(252) ≈ 0.1587 (15.87%)

**Assertions**:
```python
# Verify annualization is applied correctly
```

#### Test 7: `test_integration_with_backtest`
**Setup**:
- Run full backtest 2021-2023

**Expected**:
- Completes without errors
- Sharpe > 0.5 (should be higher than momentum due to vol adjustment)

**Assertions**:
```python
assert result.metrics['sharpe_ratio'] > 0.5
assert result.metrics['max_drawdown'] < 0.30
```

#### Test 8: `test_comparison_to_vanilla_momentum`
**Setup**:
- Run both VolAdjustedMomentum and vanilla Momentum on same period

**Expected**:
- Vol-adjusted should have higher Sharpe (lower drawdown)

**Assertions**:
```python
assert vol_adj_sharpe > vanilla_sharpe
assert vol_adj_drawdown < vanilla_drawdown
```

---

## 6. Integration & Deployment

### Configuration File Example

**Path**: `/workspaces/qetf/configs/strategies/vol_adjusted_momentum_top5.yaml`

```yaml
name: vol_adjusted_momentum_top5_ew
universe: configs/universes/tier1_initial_20.yaml
schedule: configs/schedules/monthly_rebalance.yaml
cost_model: configs/costs/flat_10bps.yaml

alpha_model:
  type: vol_adjusted_momentum
  lookback_days: 252
  min_periods: 200
  vol_floor: 0.01
  annualization_factor: 252.0

portfolio_construction:
  type: equal_weight_top_n
  top_n: 5
  constraints:
    max_weight: 0.60
    min_weight: 0.00

description: |
  Volatility-adjusted momentum: rank by Sharpe-style returns/vol ratio.
  Favors consistent performers over volatile assets.
```

### Command to Run Backtest

```bash
python scripts/run_backtest.py \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --strategy vol-adjusted-momentum-top5 \
  --output artifacts/backtests
```

### Success Criteria

1. All tests pass
2. Sharpe ratio > 0.6 (better than vanilla momentum)
3. Max drawdown < 30%
4. Information ratio vs SPY > 0.3
5. Code coverage > 90%
6. No lookahead bias

---

## 7. Edge Cases & Gotchas

### Common Pitfalls

1. **Division by Zero**
   - ❌ WRONG: `score = returns / vol` (vol can be 0)
   - ✅ RIGHT: `vol = max(vol, vol_floor)`

2. **Volatility Units**
   - ❌ WRONG: Using daily vol directly (too small, scores too high)
   - ✅ RIGHT: Annualize with `vol * sqrt(252)`

3. **Negative Returns**
   - ❌ WRONG: Taking absolute value of returns
   - ✅ RIGHT: Keep negative (negative score is valid!)

4. **Insufficient Returns for Vol**
   - ❌ WRONG: Calculating vol with < 10 returns
   - ✅ RIGHT: Return NaN if insufficient data

### Edge Cases to Handle

| Edge Case | Handling |
|-----------|----------|
| Zero volatility (constant price) | Apply vol_floor, return finite score |
| Negative returns | Valid! Score will be negative |
| Very high volatility (> 100%) | Valid! Score will be very low |
| < min_periods data | Return NaN |
| All NaN scores | Portfolio → 100% cash |

---

## 8. Acceptance Checklist

- [ ] Inherits from `AlphaModel`
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] Handles zero volatility (vol_floor)
- [ ] Handles negative returns
- [ ] Uses T-1 data only
- [ ] All 8 tests passing
- [ ] Coverage > 90%
- [ ] Sharpe > 0.6
- [ ] IR vs SPY > 0.3
- [ ] Max DD < 30%

---

## 9. Reference Files

- `/workspaces/qetf/src/quantetf/alpha/momentum.py` - Primary reference
- `/workspaces/qetf/src/quantetf/alpha/base.py` - AlphaModel interface
- `/workspaces/qetf/src/quantetf/data/snapshot_store.py` - Data access
- `/workspaces/qetf/tests/test_no_lookahead.py` - Testing patterns

---

## 10. Implementation Checklist

- [ ] Create file `/workspaces/qetf/src/quantetf/alpha/vol_adjusted_momentum.py`
- [ ] Implement `__init__()` with validation
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
- [ ] Compare to SPY + vanilla momentum
- [ ] Complete checklist

**Estimated time**: 2-3 hours

---

**Document Version**: 1.0
**Last Updated**: 2026-01-13
**Author**: Quant Research Architect
**Status**: Ready for Implementation
