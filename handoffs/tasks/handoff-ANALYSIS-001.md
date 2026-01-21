# Handoff: ANALYSIS-001 - Enhanced Metrics Module

**Task ID:** ANALYSIS-001
**Status:** ready
**Priority:** HIGH (Foundation task - blocks many others)
**Estimated Time:** 2-3 hours
**Dependencies:** None
**Assigned to:** [Available for pickup]

---

## Context & Motivation

### What are we building?

We're expanding the existing `src/quantetf/evaluation/metrics.py` module with 6 advanced performance metrics that are standard in quantitative finance. These metrics will provide deeper insights into strategy performance beyond basic Sharpe ratio and returns.

### Why does this matter?

The current metrics.py has basic functionality (Sharpe, returns, drawdowns). Advanced metrics like Sortino ratio and VaR are **essential** for:

1. **Risk-adjusted performance:** Sortino focuses on downside risk (more relevant than total volatility)
2. **Drawdown analysis:** Calmar ratio helps assess return vs worst-case scenarios
3. **Tail risk:** VaR and CVaR quantify extreme loss potential
4. **Rolling analysis:** Rolling Sharpe shows performance stability over time
5. **Benchmark comparison:** Information ratio measures skill vs benchmark

**Impact:** This task **blocks** 9 other Phase 3 tasks that depend on these metrics.

---

## Current State

### Existing Files

**`src/quantetf/evaluation/metrics.py`** - Check what's already implemented:
```bash
# Read this file first to understand existing patterns
cat src/quantetf/evaluation/metrics.py
```

Expected existing functions:
- Basic return calculations
- Sharpe ratio
- Maximum drawdown
- Possibly some others

**`tests/test_metrics.py`** - Existing test patterns:
```bash
cat tests/test_metrics.py
```

Study the existing test structure and patterns to maintain consistency.

---

## Task Specification

### New Metrics to Implement

#### 1. Sortino Ratio
**Formula:** `(R_p - R_f) / σ_downside`

Where:
- `R_p` = portfolio return (annualized)
- `R_f` = risk-free rate (default 0)
- `σ_downside` = standard deviation of negative returns only

**Why:** Better than Sharpe for strategies with asymmetric returns. Only penalizes downside volatility.

**Function signature:**
```python
def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio (return / downside deviation).

    Measures risk-adjusted return using only downside volatility.
    Higher values indicate better risk-adjusted performance.

    Args:
        returns: Series of period returns (e.g., daily)
        risk_free_rate: Annualized risk-free rate (default 0.0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Sortino ratio (annualized)

    Raises:
        ValueError: If returns is empty or all NaN

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> sortino_ratio(returns)
        1.23
    """
```

**Test cases:**
- Typical case: Mixed positive/negative returns
- Edge case: All positive returns (downside_dev = 0, handle carefully)
- Edge case: All negative returns
- Edge case: Empty series (should raise ValueError)
- Edge case: All NaN (should raise ValueError)

---

#### 2. Calmar Ratio
**Formula:** `Annualized Return / Max Drawdown`

**Why:** Measures return per unit of worst-case drawdown risk. Popular for hedge funds.

**Function signature:**
```python
def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """Calculate Calmar ratio (return / max drawdown).

    Measures annualized return per unit of maximum drawdown.
    Higher values indicate better risk-adjusted performance.

    Args:
        returns: Series of period returns (e.g., daily)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Calmar ratio

    Raises:
        ValueError: If returns is empty or all NaN
        ValueError: If max drawdown is zero (no drawdown occurred)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        >>> calmar_ratio(returns)
        2.45
    """
```

**Dependencies:** Will likely use existing `max_drawdown()` function. Check if it exists in current metrics.py.

**Test cases:**
- Typical case: Returns with moderate drawdown
- Edge case: No drawdown (all positive returns) - should handle gracefully
- Edge case: Very small drawdown - ratio can be very large
- Edge case: Empty series

---

#### 3. Win Rate
**Formula:** `Count(positive returns) / Count(all returns)`

**Why:** Simple metric showing percentage of winning periods. Useful for understanding strategy consistency.

**Function signature:**
```python
def win_rate(returns: pd.Series) -> float:
    """Calculate win rate (percentage of positive return periods).

    Args:
        returns: Series of period returns

    Returns:
        Win rate as percentage (0-100)

    Raises:
        ValueError: If returns is empty or all NaN

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> win_rate(returns)
        60.0  # 3 out of 5 positive
    """
```

**Test cases:**
- Typical case: Mixed wins/losses
- Edge case: All wins (100%)
- Edge case: All losses (0%)
- Edge case: Returns with zeros (decide how to treat them - suggest excluding)
- Edge case: Empty series

---

#### 4. Value at Risk (VaR)
**Formula:** `Percentile(returns, 1 - confidence_level)`

**Why:** Answers "What's the maximum loss I can expect with X% confidence?" Standard risk metric.

**Function signature:**
```python
def value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95,
    periods_per_year: int = 252
) -> float:
    """Calculate Value at Risk (VaR) at given confidence level.

    VaR estimates the maximum loss over a time period at a given
    confidence level. For example, 95% VaR of -2% means there's a
    5% chance of losing more than 2% in a period.

    Uses historical simulation (empirical percentile method).

    Args:
        returns: Series of period returns
        confidence_level: Confidence level (0.95 = 95%)
        periods_per_year: Trading periods per year (for annualization)

    Returns:
        VaR as a negative number (e.g., -0.02 = 2% loss)

    Raises:
        ValueError: If returns is empty or all NaN
        ValueError: If confidence_level not in (0, 1)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.03, 0.02])
        >>> value_at_risk(returns, confidence_level=0.95)
        -0.025  # 95% VaR is 2.5% loss
    """
```

**Implementation note:** Use `np.percentile` or `pd.Series.quantile()`. VaR should be returned as a **negative** number for losses.

**Test cases:**
- Typical case: Various confidence levels (90%, 95%, 99%)
- Edge case: Confidence level validation (must be 0 < x < 1)
- Edge case: Insufficient data (< 20 observations) - consider warning
- Edge case: All positive returns (VaR could be positive)
- Validation: Known distribution (e.g., normal) with calculated VaR

---

#### 5. Conditional Value at Risk (CVaR / Expected Shortfall)
**Formula:** `Mean(returns where returns < VaR)`

**Why:** CVaR is the average loss when VaR is exceeded. More informative than VaR alone.

**Function signature:**
```python
def conditional_value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95,
    periods_per_year: int = 252
) -> float:
    """Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that VaR threshold is exceeded.
    It captures tail risk better than VaR alone.

    Args:
        returns: Series of period returns
        confidence_level: Confidence level (0.95 = 95%)
        periods_per_year: Trading periods per year (for annualization)

    Returns:
        CVaR as a negative number (expected loss in tail)

    Raises:
        ValueError: If returns is empty or all NaN
        ValueError: If confidence_level not in (0, 1)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.03, 0.02])
        >>> conditional_value_at_risk(returns, confidence_level=0.95)
        -0.030  # Average loss when VaR exceeded
    """
```

**Dependencies:** Will use `value_at_risk()` function defined above.

**Test cases:**
- Typical case: Calculate CVaR given VaR threshold
- Comparison: CVaR should be >= VaR (more extreme)
- Edge case: No returns exceed VaR (very small dataset)
- Validation: Known distribution with calculated CVaR

---

#### 6. Rolling Sharpe Ratio
**Formula:** Sharpe ratio calculated over rolling windows

**Why:** Shows how Sharpe ratio changes over time. Helps identify if performance is consistent or concentrated in certain periods.

**Function signature:**
```python
def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
    min_periods: int = 126
) -> pd.Series:
    """Calculate rolling Sharpe ratio over time.

    Computes Sharpe ratio over a rolling window. Useful for
    visualizing performance stability.

    Args:
        returns: Series of period returns (e.g., daily)
        window: Rolling window size in periods (default 252 = 1 year daily)
        risk_free_rate: Annualized risk-free rate (default 0.0)
        min_periods: Minimum periods required for calculation

    Returns:
        Series of rolling Sharpe ratios (same index as returns)

    Raises:
        ValueError: If returns is empty or all NaN

    Example:
        >>> returns = pd.Series([...])  # 500 days of returns
        >>> rolling_sharpe = rolling_sharpe_ratio(returns, window=252)
        >>> rolling_sharpe.plot()  # Visualize Sharpe over time
    """
```

**Implementation note:**
- Use `returns.rolling(window, min_periods=min_periods)`
- Apply Sharpe calculation to each window
- Return as Series (not single value)

**Test cases:**
- Typical case: 500+ days of returns, 252-day window
- Edge case: Returns shorter than window (should handle with min_periods)
- Edge case: All NaN in a window
- Validation: Check first non-NaN value matches manual calculation

---

### Information Ratio (Bonus 7th Metric - Recommended)

**Formula:** `(R_p - R_b) / Tracking Error`

Where:
- `R_p` = portfolio return
- `R_b` = benchmark return
- `Tracking Error` = std(R_p - R_b)

**Function signature:**
```python
def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """Calculate Information Ratio vs benchmark.

    Measures excess return per unit of tracking error.
    Indicates skill in active management.

    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Information ratio (annualized)

    Raises:
        ValueError: If returns and benchmark not aligned
        ValueError: If tracking error is zero (perfect tracking)

    Example:
        >>> portfolio_returns = pd.Series([0.01, 0.02, -0.01])
        >>> spy_returns = pd.Series([0.008, 0.015, -0.008])
        >>> information_ratio(portfolio_returns, spy_returns)
        0.85
    """
```

**Why recommended:** Very useful for comparing strategies to SPY benchmark. Will be heavily used in Phase 3 benchmark comparison tasks.

---

## Implementation Guidelines

### File Structure

Update `src/quantetf/evaluation/metrics.py`:

```python
"""Performance metrics for strategy evaluation.

This module provides comprehensive metrics for analyzing backtest
and live trading performance.
"""

import numpy as np
import pandas as pd
from typing import Union

# ... existing code ...

# Add new functions below existing ones
# Keep alphabetical order for readability

def calmar_ratio(...):
    ...

def conditional_value_at_risk(...):
    ...

def information_ratio(...):
    ...

def rolling_sharpe_ratio(...):
    ...

def sortino_ratio(...):
    ...

def value_at_risk(...):
    ...

def win_rate(...):
    ...
```

### Testing Strategy

Create `tests/test_advanced_metrics.py`:

```python
"""Tests for advanced performance metrics."""

import numpy as np
import pandas as pd
import pytest
from quantetf.evaluation.metrics import (
    sortino_ratio,
    calmar_ratio,
    win_rate,
    value_at_risk,
    conditional_value_at_risk,
    rolling_sharpe_ratio,
    information_ratio,
)

class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_typical_case(self):
        """Test Sortino with mixed returns."""
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, 0.01])
        ratio = sortino_ratio(returns)
        assert isinstance(ratio, float)
        assert ratio > 0  # Positive expected return

    def test_sortino_all_positive(self):
        """Test Sortino when no downside (all gains)."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.005])
        # Should handle gracefully (downside_dev = 0)
        # Either return inf or a very large number
        ratio = sortino_ratio(returns)
        assert ratio > 0 or np.isinf(ratio)

    def test_sortino_empty_raises(self):
        """Test Sortino with empty series raises ValueError."""
        with pytest.raises(ValueError):
            sortino_ratio(pd.Series([]))

# ... similar structure for other metrics ...

class TestValueAtRisk:
    def test_var_95_confidence(self):
        """Test VaR at 95% confidence."""
        # Known distribution test
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        var_95 = value_at_risk(returns, confidence_level=0.95)

        # VaR should be negative (it's a loss)
        assert var_95 < 0

        # Approximately 5% of returns should be worse than VaR
        worse_than_var = (returns < var_95).sum()
        pct_worse = worse_than_var / len(returns)
        assert 0.03 < pct_worse < 0.07  # ~5% with some tolerance

# Add 3+ tests per metric (typical + edge cases)
```

**Test coverage target:** Minimum 3 tests per metric = 21 tests total (7 metrics × 3 tests)

---

## Acceptance Criteria

Mark complete when ALL of these are true:

- [ ] All 6 required metrics implemented (Sortino, Calmar, Win Rate, VaR, CVaR, Rolling Sharpe)
- [ ] Information Ratio implemented (7th metric - recommended)
- [ ] Each metric has comprehensive docstring with:
  - [ ] Clear description of what it measures
  - [ ] Formula or methodology
  - [ ] Args with types
  - [ ] Returns with type
  - [ ] Raises section for error cases
  - [ ] Example usage
- [ ] Each metric has 3+ tests covering:
  - [ ] Typical use case
  - [ ] At least 2 edge cases
  - [ ] Error handling (ValueError for invalid inputs)
- [ ] All tests pass: `pytest tests/test_advanced_metrics.py -v`
- [ ] Integration with existing metrics.py (no breaking changes)
- [ ] Code follows project style (see CLAUDE_CONTEXT.md)
- [ ] Type hints for all function signatures

---

## Testing & Validation

### How to verify correctness

1. **Run test suite:**
```bash
pytest tests/test_advanced_metrics.py -v
```

2. **Manual validation with known values:**
```python
# In Python REPL or notebook
import pandas as pd
from quantetf.evaluation.metrics import sortino_ratio, value_at_risk

# Test with known distribution
import numpy as np
np.random.seed(42)
returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

# Calculate metrics
sortino = sortino_ratio(returns)
var_95 = value_at_risk(returns, 0.95)

print(f"Sortino: {sortino:.2f}")
print(f"VaR (95%): {var_95:.4f}")
```

3. **Cross-reference with literature:**
- Compare calculations with standard definitions (e.g., Investopedia, quantitative finance textbooks)
- Validate against other libraries if available (e.g., `empyrical` package)

4. **Test with real backtest data:**
```python
# Load actual backtest results
import pandas as pd
results = pd.read_csv('artifacts/backtests/latest_backtest_equity.csv')
returns = results['daily_return']

# Calculate all metrics
from quantetf.evaluation.metrics import *

print("Sortino:", sortino_ratio(returns))
print("Calmar:", calmar_ratio(returns))
print("Win Rate:", win_rate(returns))
print("VaR (95%):", value_at_risk(returns))
print("CVaR (95%):", conditional_value_at_risk(returns))
```

---

## Dependencies

**None** - This is a foundation task.

**Blocks these tasks:**
- VIZ-001 (Backtest Analysis Notebook)
- VIZ-002 (Alpha Diagnostics)
- VIZ-003 (Stress Test Notebook)
- ANALYSIS-002 (Risk Analytics Module)
- ANALYSIS-003 (Strategy Comparison)
- ANALYSIS-004 (Parameter Sensitivity)
- ANALYSIS-005 (Benchmark Comparison)
- ANALYSIS-006 (Walk-Forward Validation)
- VIZ-004 (Auto-Report Generation)

**Priority:** Complete this ASAP to unblock parallel work.

---

## Inputs

**Existing files to read:**
- `src/quantetf/evaluation/metrics.py` - Current implementation
- `tests/test_metrics.py` - Existing test patterns
- `CLAUDE_CONTEXT.md` - Coding standards
- `artifacts/backtests/latest_backtest_equity.csv` - Real data for testing

**Sample backtest data location:**
```
artifacts/backtests/
├── latest_backtest_equity.csv
├── latest_backtest_metrics.yaml
└── ...
```

---

## Outputs

**Files to create/modify:**
1. `src/quantetf/evaluation/metrics.py` - Add 6-7 new functions
2. `tests/test_advanced_metrics.py` - Create with 21+ tests

**Expected test results:**
```
tests/test_advanced_metrics.py::TestSortinoRatio::test_sortino_typical_case PASSED
tests/test_advanced_metrics.py::TestSortinoRatio::test_sortino_all_positive PASSED
...
tests/test_advanced_metrics.py::TestInformationRatio::test_ir_vs_benchmark PASSED

======================== 21 passed in 0.85s ========================
```

---

## Examples & References

### Example usage (for documentation)

```python
from quantetf.evaluation.metrics import *
import pandas as pd

# Load backtest results
returns = pd.read_csv('backtest_equity.csv')['daily_return']

# Calculate comprehensive metrics
metrics = {
    'sharpe': sharpe_ratio(returns),  # Existing
    'sortino': sortino_ratio(returns),  # New
    'calmar': calmar_ratio(returns),  # New
    'win_rate': win_rate(returns),  # New
    'var_95': value_at_risk(returns, 0.95),  # New
    'cvar_95': conditional_value_at_risk(returns, 0.95),  # New
}

# Rolling analysis
rolling_sharpe = rolling_sharpe_ratio(returns, window=252)
rolling_sharpe.plot(title='Rolling 1-Year Sharpe Ratio')

# Benchmark comparison
spy_returns = pd.read_csv('spy_returns.csv')['daily_return']
ir = information_ratio(returns, spy_returns)
print(f"Information Ratio vs SPY: {ir:.2f}")
```

### Mathematical References

**Sortino Ratio:**
- Original paper: Sortino & Price (1994)
- Formula: (R - MAR) / DD where DD = downside deviation
- Our simplified version uses 0 as MAR (Minimum Acceptable Return)

**VaR & CVaR:**
- Methodology: Historical simulation (empirical percentile)
- Alternative methods exist (parametric, Monte Carlo) - we use simplest
- Reference: "Value at Risk" by Jorion (2006)

**Calmar Ratio:**
- Named after California Managed Account Reports
- Common in hedge fund industry
- Reference: Young (1991)

**Information Ratio:**
- Measures active management skill
- IR = α / TE where α = excess return, TE = tracking error
- Reference: Grinold & Kahn, "Active Portfolio Management"

---

## Notes & Tips

### Common Pitfalls

1. **Annualization:** Make sure to annualize consistently
   - Returns: multiply by `periods_per_year`
   - Volatility: multiply by `sqrt(periods_per_year)`

2. **VaR sign convention:** VaR is typically expressed as a **positive** number representing loss, but internally we'll return it **negative** for consistency with return calculations.

3. **Division by zero:** Handle cases where:
   - Downside deviation = 0 (all positive returns)
   - Max drawdown = 0 (no drawdown)
   - Tracking error = 0 (perfect tracking)

4. **Empty data:** Always validate input is non-empty and not all NaN

5. **Rolling calculations:** Be careful with `min_periods` parameter to avoid NaN at start

### Performance Considerations

- These metrics operate on relatively small datasets (< 10K points typically)
- No need for optimization - clarity over speed
- Use pandas/numpy built-in functions (they're already optimized)

---

## Success Criteria Summary

**This task is complete when:**

✅ All 6 metrics implemented with clear docstrings
✅ Information ratio implemented (7th metric)
✅ 21+ tests written and passing
✅ No breaking changes to existing metrics.py
✅ Code follows CLAUDE_CONTEXT.md standards
✅ Can be imported and used by downstream tasks

**Expected time:** 2-3 hours

**Agent should:**
1. Read existing metrics.py and tests
2. Implement all metrics with docstrings
3. Create comprehensive test suite
4. Validate with real backtest data
5. Mark task as `completed` in TASKS.md

---

**Ready to begin!** Pick up this task by updating TASKS.md status to `in_progress` and assigning to your agent ID.
