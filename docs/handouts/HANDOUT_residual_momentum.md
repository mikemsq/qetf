# HANDOUT: Residual Momentum Strategy Implementation

**Strategy ID**: `residual_momentum`
**File**: `/workspaces/qetf/src/quantetf/alpha/residual_momentum.py`
**Priority**: HIGH (Phase 1.1)
**Complexity**: MEDIUM
**Implementation Order**: 1 of 3

---

## 1. Executive Summary

### What
Residual Momentum (also called "Beta-Neutral Momentum") extracts pure alpha signals by regressing each ETF's returns against SPY (market beta) and ranking by the residuals. This isolates momentum independent of market exposure.

### Why
- **Lower correlation to SPY**: By removing market beta, we capture idiosyncratic momentum
- **Better risk-adjusted returns**: Expected to produce higher Information Ratio vs SPY
- **Smoother equity curve**: Reduces exposure to broad market crashes
- **Academic backing**: Consistent with factor investing literature (Fama-French residual momentum)

### Complexity
**MEDIUM** - Requires linear regression for each ticker, careful handling of numerical edge cases, and point-in-time data alignment.

### Priority
**HIGH** - Phase 1, Strategy 1 of 3. Implement first as it has highest expected alpha generation potential.

---

## 2. Background & Context

### Mathematical Definition

For each ticker `i` at decision time `T`:

1. **Regression Model**:
   ```
   R_i(t) = α_i + β_i * R_SPY(t) + ε_i(t)
   ```
   Where:
   - `R_i(t)` = daily return of ticker i on day t
   - `R_SPY(t)` = daily return of SPY on day t
   - `β_i` = market beta (sensitivity to SPY)
   - `α_i` = intercept (average excess return)
   - `ε_i(t)` = residual (idiosyncratic return)

2. **Lookback Window**: Use 252 trading days (~1 year) of daily returns

3. **Score Calculation**:
   ```python
   # Sum residuals over the lookback period
   residual_momentum_score = sum(ε_i(t) for t in [T-252, T-1])
   ```

4. **Ranking**: Higher cumulative residuals = higher score = stronger idiosyncratic momentum

### Research Rationale

- **Factor Isolation**: Traditional momentum conflates market beta with alpha. Residual momentum removes beta.
- **Diversification**: Lower correlation to market returns improves portfolio diversification
- **Drawdown Protection**: Market-neutral component should reduce drawdowns during market crashes
- **Empirical Evidence**: Academic literature shows residual momentum has similar returns to price momentum but with lower volatility

### Expected Performance Characteristics

- **CAGR**: 8-12% (slightly lower than price momentum)
- **Sharpe Ratio**: 0.8-1.2 (higher than price momentum due to lower vol)
- **Max Drawdown**: 15-25% (better than price momentum's 20-30%)
- **Correlation to SPY**: 0.3-0.5 (vs 0.6-0.8 for price momentum)
- **Information Ratio vs SPY**: 0.5-0.9 (higher than price momentum)

---

## 3. Implementation Specification

### File Path
```
/workspaces/qetf/src/quantetf/alpha/residual_momentum.py
```

### Class Definition
```python
class ResidualMomentumAlpha(AlphaModel)
```

### Inheritance
- **Parent**: `quantetf.alpha.base.AlphaModel` (abstract base class)
- **Must implement**: `score()` method
- **Pattern**: Follow `quantetf.alpha.momentum.MomentumAlpha` as reference

### Constructor Signature
```python
def __init__(
    self,
    lookback_days: int = 252,
    min_periods: int = 200,
    spy_ticker: str = "SPY"
) -> None:
    """Initialize residual momentum alpha model.

    Args:
        lookback_days: Number of trading days for regression (default: 252 ~ 1 year)
        min_periods: Minimum valid prices required for regression (default: 200)
        spy_ticker: Market benchmark ticker for beta regression (default: "SPY")
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
    """Compute residual momentum scores.

    CRITICAL: Must use ONLY data before as_of (T-1 and earlier).

    Args:
        as_of: Decision date (use data up to T-1 only)
        universe: Set of eligible tickers
        features: Pre-computed features (not used in this model)
        store: SnapshotDataStore for accessing price history
        dataset_version: Optional dataset version for reproducibility

    Returns:
        AlphaScores with residual momentum score for each ticker

    Algorithm:
        1. Validate store is SnapshotDataStore
        2. Get SPY prices [T-lookback_days, T-1]
        3. For each ticker in universe:
            a. Get ticker prices [T-lookback_days, T-1]
            b. Check min_periods requirement
            c. Calculate daily returns for ticker and SPY
            d. Run OLS regression: returns_i ~ returns_spy
            e. Extract residuals
            f. Sum residuals over lookback period
            g. Handle edge cases (insufficient data, regression failure)
        4. Return AlphaScores with cumulative residuals
    """
```

### Algorithm Pseudocode

```python
# Step 1: Validate inputs
if not isinstance(store, SnapshotDataStore):
    raise TypeError("ResidualMomentumAlpha requires SnapshotDataStore")

# Step 2: Get SPY benchmark prices (T-1 and earlier!)
spy_prices = store.get_close_prices(
    as_of=as_of,
    tickers=[spy_ticker],
    lookback_days=lookback_days + 50  # Buffer for returns calculation
)

if spy_ticker not in spy_prices.columns or spy_prices.empty:
    raise ValueError(f"SPY data not available as of {as_of}")

# Calculate SPY daily returns
spy_returns = spy_prices[spy_ticker].pct_change().dropna()

if len(spy_returns) < min_periods:
    raise ValueError(f"Insufficient SPY data: {len(spy_returns)} < {min_periods}")

# Step 3: Get ticker universe prices
ticker_prices = store.get_close_prices(
    as_of=as_of,
    tickers=list(universe.tickers),
    lookback_days=lookback_days + 50
)

# Step 4: Calculate residual momentum for each ticker
scores = {}

for ticker in universe.tickers:
    if ticker not in ticker_prices.columns:
        logger.warning(f"No data for {ticker}")
        scores[ticker] = np.nan
        continue

    # Get ticker returns
    ticker_px = ticker_prices[ticker].dropna()

    if len(ticker_px) < min_periods:
        logger.debug(f"{ticker}: Insufficient data ({len(ticker_px)} < {min_periods})")
        scores[ticker] = np.nan
        continue

    ticker_returns = ticker_px.pct_change().dropna()

    # Align ticker and SPY returns on common dates
    common_dates = ticker_returns.index.intersection(spy_returns.index)

    if len(common_dates) < min_periods:
        scores[ticker] = np.nan
        continue

    y = ticker_returns.loc[common_dates].values  # Ticker returns
    X = spy_returns.loc[common_dates].values     # SPY returns

    # Run OLS regression
    try:
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # Solve: β = (X'X)^-1 X'y
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

        # Calculate residuals: ε = y - Xβ
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred

        # Score = sum of residuals over lookback period
        residual_momentum = residuals.sum()

        scores[ticker] = residual_momentum

    except np.linalg.LinAlgError:
        logger.warning(f"{ticker}: Regression failed (singular matrix)")
        scores[ticker] = np.nan

# Step 5: Return AlphaScores
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
"""Residual momentum alpha model.

This module implements beta-neutral momentum by regressing ticker returns
against SPY and ranking by residuals. This isolates idiosyncratic momentum
independent of market beta exposure.
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


class ResidualMomentumAlpha(AlphaModel):
    """Residual momentum alpha: rank by beta-neutral return residuals.

    This model removes market (SPY) exposure from ticker returns via OLS regression,
    then ranks tickers by the sum of their residuals over the lookback period.

    The intuition: tickers with high residual momentum have strong idiosyncratic
    trends independent of the overall market direction.

    Mathematical approach:
        1. For each ticker, regress daily returns on SPY returns
        2. Extract residuals (returns unexplained by market beta)
        3. Sum residuals over lookback period
        4. Rank tickers by cumulative residuals (higher = better)

    Point-in-time compliance:
        - Uses only data BEFORE as_of date (T-1 and earlier)
        - No lookahead bias in regression or scoring

    Example:
        >>> alpha = ResidualMomentumAlpha(lookback_days=252)
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
        spy_ticker: str = "SPY"
    ) -> None:
        """Initialize residual momentum alpha model.

        Args:
            lookback_days: Number of trading days for regression window (default: 252)
            min_periods: Minimum valid returns required for regression (default: 200)
            spy_ticker: Market benchmark ticker for beta calculation (default: "SPY")

        Raises:
            ValueError: If parameters are invalid
        """
        if lookback_days < min_periods:
            raise ValueError(f"lookback_days ({lookback_days}) must be >= min_periods ({min_periods})")
        if min_periods < 50:
            raise ValueError(f"min_periods must be >= 50 for stable regression")

        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.spy_ticker = spy_ticker

        logger.info(
            f"Initialized ResidualMomentumAlpha: "
            f"lookback={lookback_days}, min_periods={min_periods}, "
            f"benchmark={spy_ticker}"
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
        """Compute residual momentum scores for the universe.

        CRITICAL: Uses only data BEFORE as_of (T-1 and earlier) to prevent lookahead.

        Args:
            as_of: Decision date (score computed using data up to T-1)
            universe: Set of eligible tickers to score
            features: Pre-computed features (not used in this model)
            store: Data store for accessing price history
            dataset_version: Optional dataset version for reproducibility

        Returns:
            AlphaScores with residual momentum score for each ticker.
            NaN scores indicate insufficient data for regression.

        Raises:
            TypeError: If store is not SnapshotDataStore
            ValueError: If SPY data is unavailable or insufficient
        """
        logger.info(f"Computing residual momentum as of {as_of} for {len(universe.tickers)} tickers")

        # TODO: Validate store type
        # TODO: Get SPY prices and calculate returns
        # TODO: Get universe ticker prices
        # TODO: For each ticker:
        #       - Calculate returns
        #       - Align with SPY returns on common dates
        #       - Run OLS regression
        #       - Sum residuals
        #       - Handle edge cases
        # TODO: Return AlphaScores with log summary

        raise NotImplementedError("ResidualMomentumAlpha.score() not yet implemented")

    def _run_regression(
        self,
        ticker_returns: pd.Series,
        spy_returns: pd.Series
    ) -> Optional[np.ndarray]:
        """Run OLS regression and return residuals.

        Helper method to isolate regression logic for testing.

        Args:
            ticker_returns: Daily returns for the ticker
            spy_returns: Daily returns for SPY benchmark

        Returns:
            Array of residuals, or None if regression fails

        Algorithm:
            1. Align returns on common dates
            2. Build design matrix X = [1, SPY_returns]
            3. Solve: β = (X'X)^-1 X'y
            4. Calculate residuals: ε = y - Xβ
        """
        # TODO: Implement OLS regression
        # TODO: Handle numerical errors (singular matrix, etc.)
        # TODO: Return residuals array
        raise NotImplementedError("_run_regression() not yet implemented")
```

---

## 5. Testing Specification

### Test File Path
```
/workspaces/qetf/tests/alpha/test_residual_momentum.py
```

### Required Test Cases

#### Test 1: `test_residual_momentum_basic_functionality`
**Setup**:
- Create synthetic price data for 3 tickers + SPY
- SPY: steady uptrend (+0.1% daily)
- TICKER_A: follows SPY exactly (beta=1.0, no residual)
- TICKER_B: outperforms SPY (+0.15% daily, positive residual)
- TICKER_C: underperforms SPY (+0.05% daily, negative residual)

**Expected Behavior**:
- TICKER_B should have highest score (positive residual momentum)
- TICKER_C should have lowest score (negative residual momentum)
- TICKER_A should have near-zero score

**Assertions**:
```python
assert scores['TICKER_B'] > scores['TICKER_A'] > scores['TICKER_C']
assert abs(scores['TICKER_A']) < 0.01  # Near-zero residual
```

#### Test 2: `test_insufficient_data_returns_nan`
**Setup**:
- Create ticker with only 50 days of data (< min_periods=200)

**Expected Behavior**:
- Score should be NaN for tickers with insufficient history

**Assertions**:
```python
assert pd.isna(scores['SHORT_HISTORY_TICKER'])
```

#### Test 3: `test_no_lookahead_bias`
**Setup**:
- Synthetic data where prices jump 10% on as_of date
- Create SnapshotDataStore with this data

**Expected Behavior**:
- Score should NOT reflect the 10% jump on as_of date
- Regression should use only T-1 and earlier returns

**Assertions**:
```python
# Calculate what score would be with lookahead
score_with_lookahead = calculate_score_including_as_of_date()
# Actual score should be different
assert abs(actual_score - score_with_lookahead) > 0.05
```

#### Test 4: `test_spy_missing_raises_error`
**Setup**:
- Universe with tickers but SPY not in snapshot

**Expected Behavior**:
- Should raise ValueError with clear message

**Assertions**:
```python
with pytest.raises(ValueError, match="SPY data not available"):
    alpha.score(as_of=date, universe=universe, ...)
```

#### Test 5: `test_regression_singular_matrix_handling`
**Setup**:
- Create degenerate data: ticker returns = constant (zero variance)

**Expected Behavior**:
- Regression should fail gracefully
- Score should be NaN, not crash

**Assertions**:
```python
assert pd.isna(scores['CONSTANT_TICKER'])
# Check warning was logged
assert "Regression failed" in caplog.text
```

#### Test 6: `test_date_alignment_between_ticker_and_spy`
**Setup**:
- TICKER_A has data Mon-Fri
- SPY has data Mon-Thu (missing Friday)
- Test with as_of = Saturday

**Expected Behavior**:
- Should only use Mon-Thu common dates for regression
- Should still compute valid score if min_periods met

**Assertions**:
```python
assert not pd.isna(scores['TICKER_A'])
# Verify only common dates used (check via mock/spy)
```

#### Test 7: `test_scores_sum_correctly_over_residuals`
**Setup**:
- Hand-crafted data where we can calculate expected residuals manually
- Example: SPY returns = [0.01, 0.02, -0.01]
          TICKER returns = [0.02, 0.03, 0.00]
- Calculate expected residuals and sum

**Expected Behavior**:
- Score should match hand-calculated sum of residuals

**Assertions**:
```python
expected_residual_sum = 0.123  # Calculated by hand
assert abs(scores['TICKER_A'] - expected_residual_sum) < 0.001
```

#### Test 8: `test_integration_with_backtest_engine`
**Setup**:
- Run mini backtest with ResidualMomentumAlpha
- 2021-2023 period, 3 rebalances
- Use real snapshot data

**Expected Behavior**:
- Backtest completes without errors
- Equity curve is monotonic or has reasonable drawdowns
- Metrics computed successfully

**Assertions**:
```python
assert result.equity_curve is not None
assert len(result.equity_curve) == expected_length
assert 'sharpe_ratio' in result.metrics
assert result.metrics['sharpe_ratio'] > -2.0  # Sanity check
```

---

## 6. Integration & Deployment

### Configuration File Example

**Path**: `/workspaces/qetf/configs/strategies/residual_momentum_top5.yaml`

```yaml
name: residual_momentum_top5_ew
universe: configs/universes/etf_20.yaml
schedule: configs/schedules/monthly_rebalance.yaml
cost_model: configs/costs/flat_10bps.yaml

alpha_model:
  type: residual_momentum
  lookback_days: 252
  min_periods: 200
  spy_ticker: SPY

portfolio_construction:
  type: equal_weight_top_n
  top_n: 5
  constraints:
    max_weight: 0.60
    min_weight: 0.00

description: |
  Residual momentum strategy: regress returns on SPY, rank by residuals.
  Beta-neutral approach to capture idiosyncratic momentum.
```

### Command to Run Backtest

```bash
# Basic backtest
python scripts/run_backtest.py \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --strategy residual-momentum-top5 \
  --top-n 5 \
  --lookback 252 \
  --cost-bps 10.0 \
  --output artifacts/backtests

# Expected output directory:
# artifacts/backtests/20260113_HHMMSS_residual-momentum-top5/
```

### Expected Output Artifacts

After successful backtest:

```
artifacts/backtests/20260113_HHMMSS_residual-momentum-top5/
├── config.yaml              # Strategy configuration
├── metrics.json             # Performance metrics JSON
├── equity_curve.csv         # Time series of NAV
├── weights_history.csv      # Portfolio weights over time
├── holdings_history.csv     # Share positions over time
├── results.pkl              # Full BacktestResult object
└── out.txt                  # Execution log
```

### Success Criteria

**Must Pass All**:
1. All unit tests pass (`pytest tests/alpha/test_residual_momentum.py -v`)
2. No lookahead bias detected (test_no_lookahead_bias passes)
3. Backtest completes without errors
4. Metrics computed:
   - Sharpe Ratio > 0.3
   - Max Drawdown < 40%
   - CAGR > 0% (positive returns)
5. Active returns vs SPY:
   - Information Ratio > 0.0 (preferably > 0.3)
6. Code coverage > 90% for residual_momentum.py
7. Type hints pass `mypy` check (no errors)
8. Logging is informative (INFO level shows progress)

---

## 7. Edge Cases & Gotchas

### Common Pitfalls

1. **Lookahead Bias in Regression**
   - ❌ WRONG: Including as_of date in regression window
   - ✅ RIGHT: Use `as_of=as_of` which gives T-1 data only

2. **Date Alignment Issues**
   - ❌ WRONG: Assuming ticker and SPY have same dates
   - ✅ RIGHT: Use `.intersection()` to find common dates before regression

3. **Singular Matrix in Regression**
   - ❌ WRONG: Crash on `np.linalg.LinAlgError`
   - ✅ RIGHT: Wrap in try/except, return NaN, log warning

4. **Division by Zero in Returns**
   - ❌ WRONG: `returns = prices.diff() / prices.shift(1)` can have NaN/inf
   - ✅ RIGHT: Use `.pct_change()` and `.dropna()`

5. **Insufficient Warmup Period**
   - ❌ WRONG: Running backtest from 2021-01-01 with 252-day lookback
   - ✅ RIGHT: Ensure snapshot starts at least 252 days before first rebalance

### Edge Cases to Handle

| Edge Case | Handling Strategy |
|-----------|------------------|
| SPY not in snapshot | Raise `ValueError` with clear message |
| Ticker has < min_periods data | Return `NaN` score, log DEBUG message |
| All tickers have NaN scores | Portfolio allocates 100% to cash |
| Regression singular matrix | Return `NaN`, log WARNING |
| as_of date has no data | `store.get_close_prices()` raises error (by design) |
| Ticker returns = constant | Regression fails → NaN score |
| SPY returns = constant | Raise `ValueError` (market data issue) |
| Negative residuals for all tickers | Valid! Rank them (least negative = best) |

### Point-in-Time Compliance Notes

**CRITICAL**: This strategy MUST use T-1 data only.

```python
# When scoring on 2023-12-31 (Monday):
as_of = pd.Timestamp("2023-12-31")

# This call returns data UP TO 2023-12-30 (Sunday/Friday depending on weekends)
prices = store.get_close_prices(as_of=as_of, ...)

# Verify in tests:
assert prices.index.max() < as_of  # Strict inequality!
```

### Performance Considerations

1. **Regression Complexity**: O(n) where n = lookback_days
   - For 252 days: ~milliseconds per ticker
   - For 20 tickers: ~20ms total (acceptable)

2. **Memory Usage**:
   - Storing 252 days × 20 tickers × 5 fields = ~25KB
   - Not a concern for ETF universe

3. **Numerical Stability**:
   - Use `np.linalg.lstsq()` with `rcond=None` for stable solve
   - Alternative: `scipy.stats.linregress()` (simpler but slower)

---

## 8. Acceptance Checklist

Use this checklist for code review:

### Code Quality
- [ ] Inherits from `AlphaModel` base class
- [ ] Implements `score()` method with correct signature
- [ ] All type hints present (`-> AlphaScores`, `Optional[DatasetVersion]`, etc.)
- [ ] Docstrings present for class and all public methods
- [ ] Follows existing code style (see `momentum.py`)
- [ ] No hardcoded paths or magic numbers (use constants)
- [ ] Passes `mypy` type checking (no errors)
- [ ] Passes `black` code formatting
- [ ] Passes `flake8` linting

### Functionality
- [ ] Returns `AlphaScores` object with correct `as_of` date
- [ ] Scores are numeric (float) or NaN (not None, not string)
- [ ] Handles universe with 1 ticker (edge case)
- [ ] Handles universe with 20 tickers (normal case)
- [ ] Returns NaN for tickers with insufficient data
- [ ] Logs informative messages (INFO level for summary, DEBUG for details)
- [ ] Raises `TypeError` if store is not SnapshotDataStore
- [ ] Raises `ValueError` if SPY data unavailable

### Point-in-Time Compliance
- [ ] Uses `store.get_close_prices(as_of=...)` correctly
- [ ] Never accesses data >= as_of date
- [ ] test_no_lookahead_bias passes
- [ ] Regression uses only T-1 and earlier returns
- [ ] No `.shift(-1)` or forward-looking operations

### Testing
- [ ] All 8 test cases implemented and passing
- [ ] Test coverage > 90% (check with `pytest --cov`)
- [ ] Integration test with backtest engine passes
- [ ] Edge cases tested (NaN handling, errors, etc.)
- [ ] Tests use synthetic data (not dependent on external data)

### Integration
- [ ] Can be imported: `from quantetf.alpha.residual_momentum import ResidualMomentumAlpha`
- [ ] Works with `SimpleBacktestEngine`
- [ ] Works with `EqualWeightTopN` portfolio constructor
- [ ] Produces valid backtest results
- [ ] Metrics computed successfully

### Documentation
- [ ] Module docstring explains strategy concept
- [ ] Class docstring includes example usage
- [ ] All parameters documented in `__init__` docstring
- [ ] `score()` method documents the algorithm
- [ ] Edge cases documented in docstrings
- [ ] README or strategy doc updated (if applicable)

### Performance Validation
- [ ] Sharpe ratio > 0.3 (or documented reason if lower)
- [ ] Information ratio vs SPY computed
- [ ] Max drawdown < 50% (reasonable for aggressive strategy)
- [ ] No obvious bugs in equity curve (e.g., sudden jumps)
- [ ] Comparison to vanilla momentum shows expected characteristics

---

## 9. Reference Files

Study these files to understand patterns and best practices:

### Primary Reference: Momentum Alpha
**File**: `/workspaces/qetf/src/quantetf/alpha/momentum.py`
**Lines to study**:
- Lines 46-73: `MomentumAlpha` class definition and `__init__`
- Lines 74-120: `score()` method implementation
- Lines 106-120: Price data retrieval pattern
- Lines 121-150: Per-ticker scoring loop with error handling
- Lines 151-169: Summary logging and return statement

**Key patterns**:
- Store type validation (line 103-104)
- Price retrieval with lookback buffer (line 108-112)
- Try/except for data errors (line 107-119)
- Per-ticker loop with NaN handling (lines 124-149)
- Logging summary statistics (lines 156-162)

### Alpha Model Base Class
**File**: `/workspaces/qetf/src/quantetf/alpha/base.py`
**Lines to study**:
- Lines 19-32: `AlphaModel` abstract base class definition
- Line 23-31: `score()` method signature (must match exactly!)

### Data Store Interface
**File**: `/workspaces/qetf/src/quantetf/data/snapshot_store.py`
**Lines to study**:
- Lines 96-120: `get_close_prices()` method (returns T-1 data)
- Lines 74-94: `read_prices()` T-1 enforcement (line 75: `< as_of`)
- Point-in-time compliance comments (lines 59-72)

### Type Definitions
**File**: `/workspaces/qetf/src/quantetf/types.py`
**Lines to study**:
- Lines 35-39: `AlphaScores` dataclass definition
- Lines 22-25: `Universe` dataclass
- Lines 29-32: `FeatureFrame` dataclass
- Lines 13-18: `DatasetVersion` dataclass

### Testing Patterns
**File**: `/workspaces/qetf/tests/test_no_lookahead.py`
**Lines to study**:
- Lines 23-58: `create_synthetic_prices()` helper (how to create test data)
- Lines 61-94: `test_snapshot_store_t1_access()` (T-1 validation)
- Lines 96-120: `test_strict_inequality_t1()` (lookahead bias test pattern)

**File**: `/workspaces/qetf/tests/test_backtest_engine.py`
**Lines to study**:
- Lines 30-57: Rebalance date generation tests
- Lines 68-98: Helper function tests (Sharpe, drawdown)

### Backtest Integration
**File**: `/workspaces/qetf/scripts/run_backtest.py`
**Lines to study**:
- Lines 39-44: Imports (how to import alpha models)
- Lines 142-200: `run_backtest()` function (integration pattern)
- Lines 250-280: Metrics computation and output

### Portfolio Constructor Reference
**File**: `/workspaces/qetf/src/quantetf/portfolio/equal_weight.py`
**Lines to study**:
- Lines 55-85: `construct()` method signature (what portfolio expects from alpha)
- Lines 96-98: Valid score handling (`.dropna()` pattern)
- Lines 117-122: Ranking and selection logic

---

## 10. Implementation Checklist

**Step-by-step implementation guide**:

- [ ] **Step 1**: Create file `/workspaces/qetf/src/quantetf/alpha/residual_momentum.py`
- [ ] **Step 2**: Copy code template from Section 4
- [ ] **Step 3**: Implement `__init__()` method (validate parameters)
- [ ] **Step 4**: Implement `_run_regression()` helper method
  - [ ] Test with simple synthetic data
  - [ ] Handle singular matrix case
- [ ] **Step 5**: Implement `score()` method main logic
  - [ ] Store type validation
  - [ ] SPY data retrieval
  - [ ] Universe data retrieval
  - [ ] Per-ticker loop with regression
- [ ] **Step 6**: Add comprehensive logging
- [ ] **Step 7**: Create test file `/workspaces/qetf/tests/alpha/test_residual_momentum.py`
- [ ] **Step 8**: Implement Test 1-4 (basic functionality, data validation)
- [ ] **Step 9**: Run tests, fix bugs
- [ ] **Step 10**: Implement Test 5-8 (edge cases, integration)
- [ ] **Step 11**: Run full test suite, achieve >90% coverage
- [ ] **Step 12**: Run mini backtest manually (2021-2023)
- [ ] **Step 13**: Review equity curve, check for anomalies
- [ ] **Step 14**: Run type checking (`mypy`)
- [ ] **Step 15**: Run code formatting (`black`, `flake8`)
- [ ] **Step 16**: Document any deviations from spec
- [ ] **Step 17**: Create config file (Section 6)
- [ ] **Step 18**: Run full 5-year backtest
- [ ] **Step 19**: Compare to SPY benchmark
- [ ] **Step 20**: Complete acceptance checklist (Section 8)

**Estimated implementation time**: 3-4 hours for experienced developer

---

**Document Version**: 1.0
**Last Updated**: 2026-01-13
**Author**: Quant Research Architect
**Status**: Ready for Implementation
