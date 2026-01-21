# Task Handoff: IMPL-002 - EqualWeightTopN Portfolio Constructor

**Task ID:** IMPL-002
**Status:** ready
**Priority:** high
**Estimated Time:** 1-2 hours

---

## Quick Context

You are implementing a portfolio constructor that takes the top N ETFs by alpha score and assigns them equal weights (1/N each). This is part of Phase 2 (Backtest Engine) - building the pluggable components for backtesting.

**Why this matters:** This is the simplest portfolio construction method and serves as our MVP. Later we'll add optimization-based methods (mean-variance, risk parity), but equal-weight is a good baseline.

---

## What You Need to Know

### Architecture
- Portfolio constructors implement the `PortfolioConstructor` abstract base class
- They take alpha scores as input and return target weights
- Must be pluggable - easy to swap with other constructors

### Design Decisions (from architecture discussion)
- **Top N = 5** for our 20-ETF universe
- **Equal weight** means each selected ETF gets 1/N weight
- **Scores** are continuous values (not ranks) - we rank them internally
- Weights must sum to approximately 1.0

### Point-in-Time Requirement
- Constructor uses only the alpha scores provided (already point-in-time)
- No additional data access needed
- The `as_of` date is for logging/debugging only

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Coding standards, Python conventions
2. **`/workspaces/qetf/src/quantetf/portfolio/base.py`** - Base class to implement
3. **`/workspaces/qetf/src/quantetf/types.py`** - Data structures (AlphaScores, TargetWeights)
4. **`/workspaces/qetf/src/quantetf/alpha/momentum.py`** - Example of similar pattern

---

## Implementation Steps

### 1. Create the file
Create `/workspaces/qetf/src/quantetf/portfolio/equal_weight.py`

### 2. Implement EqualWeightTopN class

```python
"""Equal-weight portfolio construction."""

from typing import Optional
import pandas as pd
import numpy as np

from quantetf.portfolio.base import PortfolioConstructor
from quantetf.types import AlphaScores, RiskModelOutput, TargetWeights, Universe
from quantetf.data.store import DataStore


class EqualWeightTopN(PortfolioConstructor):
    """Portfolio constructor that selects top N assets by alpha score and assigns equal weights.

    This is the simplest portfolio construction method: rank assets by alpha score,
    select the top N, and give each an equal weight of 1/N.

    Example:
        >>> constructor = EqualWeightTopN(top_n=5)
        >>> weights = constructor.construct(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     alpha=alpha_scores,
        ...     risk=risk_output,  # Not used, but required by interface
        ...     store=store  # Not used, but required by interface
        ... )
    """

    def __init__(self, top_n: int = 5):
        """Initialize equal-weight constructor.

        Args:
            top_n: Number of top-scoring assets to select (default: 5)
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        self.top_n = top_n

    def construct(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        alpha: AlphaScores,
        risk: RiskModelOutput,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
        prev_weights: Optional[pd.Series] = None,
    ) -> TargetWeights:
        """Construct equal-weight portfolio from top N alpha scores.

        Args:
            as_of: Date as of which to construct portfolio
            universe: Set of eligible tickers
            alpha: Alpha scores for the universe
            risk: Risk model output (not used in equal-weight)
            store: Data store (not used in equal-weight)
            dataset_version: Optional dataset version
            prev_weights: Optional previous weights (not used in equal-weight)

        Returns:
            TargetWeights with equal weights for top N tickers
        """
        # TODO: Implement this method
        # 1. Get valid alpha scores (drop NaN)
        # 2. Rank scores (highest score = best)
        # 3. Select top N tickers
        # 4. Assign equal weight (1/N) to each
        # 5. Create weights Series (all tickers in universe, 0.0 for non-selected)
        # 6. Return TargetWeights
        pass
```

### 3. Add proper imports
Make sure you import `DatasetVersion` from `quantetf.types`

### 4. Implement the logic

**Key steps:**
```python
# Drop NaN scores (tickers with insufficient data)
valid_scores = alpha.scores.dropna()

# Rank by score (descending - higher is better)
ranked = valid_scores.sort_values(ascending=False)

# Select top N
top_tickers = ranked.head(self.top_n).index.tolist()

# Calculate equal weight
weight = 1.0 / len(top_tickers) if top_tickers else 0.0

# Create weights series for entire universe
weights = pd.Series(0.0, index=list(universe.tickers))
weights[top_tickers] = weight

# Return TargetWeights
return TargetWeights(
    as_of=as_of,
    weights=weights,
    diagnostics={
        'top_n': self.top_n,
        'selected': top_tickers,
        'num_valid_scores': len(valid_scores)
    }
)
```

### 5. Add logging (optional but recommended)
```python
import logging
logger = logging.getLogger(__name__)

# In construct():
logger.info(f"Constructing equal-weight portfolio: top {self.top_n} from {len(valid_scores)} valid scores")
logger.info(f"Selected: {top_tickers}, weight={weight:.4f} each")
```

### 6. Create tests
Create `/workspaces/qetf/tests/test_equal_weight.py`

**Test cases needed:**
- Test with 10 scores, top_n=5 → should select top 5, each gets 0.2 weight
- Test with NaN scores → should skip NaN, select top N from valid
- Test with top_n > available scores → should select all available
- Test weights sum to ~1.0
- Test all non-selected tickers have 0.0 weight

Example test:
```python
def test_equal_weight_top_n():
    """Test basic equal-weight selection."""
    from quantetf.portfolio.equal_weight import EqualWeightTopN
    from quantetf.types import AlphaScores, Universe
    import pandas as pd

    # Create test data
    tickers = ['A', 'B', 'C', 'D', 'E', 'F']
    scores = pd.Series([0.5, 0.3, 0.8, 0.1, 0.6, 0.2], index=tickers)

    alpha = AlphaScores(
        as_of=pd.Timestamp("2023-12-31"),
        scores=scores
    )

    universe = Universe(
        as_of=pd.Timestamp("2023-12-31"),
        tickers=tuple(tickers)
    )

    # Construct portfolio
    constructor = EqualWeightTopN(top_n=3)
    weights = constructor.construct(
        as_of=pd.Timestamp("2023-12-31"),
        universe=universe,
        alpha=alpha,
        risk=None,  # Not used
        store=None  # Not used
    )

    # Verify top 3: C (0.8), E (0.6), A (0.5)
    assert weights.weights['C'] == pytest.approx(1/3)
    assert weights.weights['E'] == pytest.approx(1/3)
    assert weights.weights['A'] == pytest.approx(1/3)
    assert weights.weights['B'] == 0.0
    assert weights.weights['D'] == 0.0
    assert weights.weights['F'] == 0.0

    # Verify sum
    assert weights.weights.sum() == pytest.approx(1.0)
```

### 7. Run tests
```bash
pytest tests/test_equal_weight.py -v
```

---

## Acceptance Criteria

- [ ] `EqualWeightTopN` class implements `PortfolioConstructor` interface
- [ ] Constructor takes `top_n` parameter (default=5)
- [ ] `construct()` method selects top N by alpha score
- [ ] Selected tickers get equal weight (1/N)
- [ ] Non-selected tickers get 0.0 weight
- [ ] Handles NaN scores correctly (skips them)
- [ ] Weights sum to approximately 1.0
- [ ] Includes comprehensive docstrings
- [ ] Follows CLAUDE_CONTEXT.md standards (snake_case, type hints, etc.)
- [ ] Tests cover all edge cases
- [ ] All tests pass: `pytest tests/test_equal_weight.py`

---

## Success Looks Like

```python
# Usage example
from quantetf.portfolio.equal_weight import EqualWeightTopN

constructor = EqualWeightTopN(top_n=5)

# Given alpha scores for 20 ETFs, selects top 5 and gives each 20% weight
weights = constructor.construct(
    as_of=pd.Timestamp("2023-12-31"),
    universe=universe,  # 20 ETFs
    alpha=alpha_scores,  # Momentum scores
    risk=None,  # Not needed for equal-weight
    store=None  # Not needed for equal-weight
)

# weights.weights is a Series with 20 entries:
# - Top 5 tickers: 0.20 each
# - Other 15 tickers: 0.00 each
# - Sum = 1.0
```

---

## Questions? Issues?

If blocked or unclear:
1. **Check CLAUDE_CONTEXT.md** for Python coding standards
2. **Look at `/workspaces/qetf/src/quantetf/alpha/momentum.py`** for similar pattern
3. **Check `/workspaces/qetf/src/quantetf/portfolio/base.py`** for the interface
4. **Document any questions** in `handoffs/completion-IMPL-002.md`

---

## Related Files

- Base class: `src/quantetf/portfolio/base.py`
- Types: `src/quantetf/types.py`
- Example: `src/quantetf/alpha/momentum.py`
- Standards: `CLAUDE_CONTEXT.md`
- Tests example: `tests/test_yfinance_provider.py`

---

## Estimated Time

**1-2 hours:**
- 15 min: Read context files
- 30 min: Implement class
- 30 min: Write tests
- 15 min: Run tests and fix issues
- 15 min: Write completion note

---

## When Done

1. Verify all tests pass
2. Create `handoffs/completion-IMPL-002.md` with:
   - What you implemented
   - Any design decisions
   - Test results
   - Any issues or learnings
3. Update `TASKS.md`: change status to `completed`
4. Commit with message: "Implement EqualWeightTopN portfolio constructor (IMPL-002)"
