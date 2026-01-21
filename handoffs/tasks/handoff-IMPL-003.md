# Task Handoff: IMPL-003 - FlatTransactionCost Model

**Task ID:** IMPL-003
**Status:** ready
**Priority:** high
**Estimated Time:** 1 hour

---

## Quick Context

You are implementing a simple flat transaction cost model that charges 10 basis points (0.10%) per trade. This is part of Phase 2 (Backtest Engine) - building realistic cost modeling for backtests.

**Why this matters:** Transaction costs can significantly impact strategy performance. We start with a conservative flat-fee model (10 bps is typical for liquid ETFs through discount brokers).

---

## What You Need to Know

### Architecture
- Cost models implement the `TransactionCostModel` abstract base class
- They calculate the dollar cost of executing a set of trades
- Must be realistic but conservative (better to overestimate than underestimate)

### Design Decisions
- **10 basis points (0.10%)** flat fee per trade
- Cost = |trade_value| × 0.0010
- Applied to both buys and sells
- No minimum fee, no tiered pricing (keep it simple for MVP)

### Point-in-Time Requirement
- Cost model uses portfolio values and target weights (already point-in-time)
- No additional data access needed
- The `as_of` date is for logging/debugging only

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Coding standards
2. **`/workspaces/qetf/src/quantetf/portfolio/base.py`** - Base class to implement
3. **`/workspaces/qetf/src/quantetf/types.py`** - Data structures (TargetWeights, Trades)
4. **`/workspaces/qetf/src/quantetf/portfolio/costs.py`** - File to update (already has base class)

---

## Implementation Steps

### 1. Check existing file
Look at `/workspaces/qetf/src/quantetf/portfolio/costs.py` - it likely has the base class defined.

### 2. Implement FlatTransactionCost class

```python
"""Transaction cost models."""

import logging
from typing import Optional
import pandas as pd

from quantetf.portfolio.base import TransactionCostModel
from quantetf.types import TargetWeights, Trades, DatasetVersion
from quantetf.data.store import DataStore

logger = logging.getLogger(__name__)


class FlatTransactionCost(TransactionCostModel):
    """Flat transaction cost model - charges a fixed percentage per trade.

    This is the simplest cost model: charge a flat percentage of the trade value,
    regardless of trade size or asset type. Common for discount brokers with ETFs.

    Example:
        >>> cost_model = FlatTransactionCost(cost_bps=10)  # 10 bps = 0.10%
        >>> cost = cost_model.calculate_cost(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     trades=trades,
        ...     store=store
        ... )
    """

    def __init__(self, cost_bps: float = 10.0):
        """Initialize flat cost model.

        Args:
            cost_bps: Cost in basis points (default: 10 bps = 0.10%)
        """
        if cost_bps < 0:
            raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
        self.cost_bps = cost_bps
        self.cost_rate = cost_bps / 10000.0  # Convert bps to decimal

    def calculate_cost(
        self,
        *,
        as_of: pd.Timestamp,
        trades: Trades,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> float:
        """Calculate total transaction cost for a set of trades.

        Cost = sum(|trade_value| * cost_rate) for all trades

        Args:
            as_of: Date as of which to calculate costs
            trades: Trade information (tickers, shares, prices)
            store: Data store (not used in flat model)
            dataset_version: Optional dataset version

        Returns:
            Total transaction cost in dollars
        """
        # Calculate trade values
        trade_values = trades.shares * trades.prices

        # Cost is absolute value of trade * cost rate
        total_cost = (trade_values.abs() * self.cost_rate).sum()

        logger.info(
            f"Transaction costs: {len(trades.shares)} trades, "
            f"{self.cost_bps} bps, total=${total_cost:.2f}"
        )

        return total_cost
```

### 3. Handle edge cases

**Important considerations:**
- Empty trades (no cost)
- Zero shares (no cost)
- Negative shares (sells) - use absolute value

### 4. Create tests

Create `/workspaces/qetf/tests/test_transaction_costs.py`

**Test cases needed:**
- Test with simple trade: 100 shares @ $50, 10 bps → cost = $5.00
- Test with multiple trades → sum correctly
- Test with buy and sell → both incur costs
- Test with zero shares → zero cost
- Test with empty trades → zero cost
- Test cost_bps parameter validation

Example test:
```python
def test_flat_cost_simple_trade():
    """Test flat cost calculation with a simple trade."""
    from quantetf.portfolio.costs import FlatTransactionCost
    from quantetf.types import Trades
    import pandas as pd

    # Create test trade: buy 100 shares @ $50 = $5,000
    # Cost at 10 bps = $5,000 * 0.0010 = $5.00
    trades = Trades(
        as_of=pd.Timestamp("2023-12-31"),
        shares=pd.Series([100], index=['SPY']),
        prices=pd.Series([50.0], index=['SPY'])
    )

    cost_model = FlatTransactionCost(cost_bps=10.0)
    cost = cost_model.calculate_cost(
        as_of=pd.Timestamp("2023-12-31"),
        trades=trades,
        store=None  # Not used
    )

    assert cost == pytest.approx(5.00)


def test_flat_cost_buy_and_sell():
    """Test that both buys and sells incur costs."""
    from quantetf.portfolio.costs import FlatTransactionCost
    from quantetf.types import Trades
    import pandas as pd

    # Buy 100 SPY @ $50 = $5,000 → cost $5
    # Sell 50 QQQ @ $60 = $3,000 → cost $3
    # Total cost = $8
    trades = Trades(
        as_of=pd.Timestamp("2023-12-31"),
        shares=pd.Series([100, -50], index=['SPY', 'QQQ']),
        prices=pd.Series([50.0, 60.0], index=['SPY', 'QQQ'])
    )

    cost_model = FlatTransactionCost(cost_bps=10.0)
    cost = cost_model.calculate_cost(
        as_of=pd.Timestamp("2023-12-31"),
        trades=trades,
        store=None
    )

    # (5000 * 0.001) + (3000 * 0.001) = 8.00
    assert cost == pytest.approx(8.00)
```

### 5. Run tests
```bash
pytest tests/test_transaction_costs.py -v
```

---

## Acceptance Criteria

- [ ] `FlatTransactionCost` class implements `TransactionCostModel` interface
- [ ] Constructor takes `cost_bps` parameter (default=10.0)
- [ ] Validates cost_bps >= 0
- [ ] `calculate_cost()` method calculates sum(|trade_value| * cost_rate)
- [ ] Handles empty trades (returns 0.0)
- [ ] Handles both buys (positive shares) and sells (negative shares)
- [ ] Includes comprehensive docstrings
- [ ] Follows CLAUDE_CONTEXT.md standards
- [ ] Tests cover all edge cases
- [ ] All tests pass: `pytest tests/test_transaction_costs.py`

---

## Success Looks Like

```python
# Usage example
from quantetf.portfolio.costs import FlatTransactionCost

cost_model = FlatTransactionCost(cost_bps=10)  # 10 bps = 0.10%

# Calculate cost for a set of trades
total_cost = cost_model.calculate_cost(
    as_of=pd.Timestamp("2023-12-31"),
    trades=trades,  # Buy 100 SPY @ $400, sell 50 QQQ @ $350
    store=store
)

# For trades totaling $57,500 in absolute value:
# Cost = $57,500 * 0.0010 = $57.50
```

---

## Questions? Issues?

If blocked or unclear:
1. **Check CLAUDE_CONTEXT.md** for coding standards
2. **Look at base class** in `src/quantetf/portfolio/base.py`
3. **Check types** in `src/quantetf/types.py` for Trades structure
4. **Document questions** in `handoffs/completion-IMPL-003.md`

---

## Related Files

- Base class: `src/quantetf/portfolio/base.py`
- Types: `src/quantetf/types.py`
- Update file: `src/quantetf/portfolio/costs.py`
- Standards: `CLAUDE_CONTEXT.md`

---

## When Done

1. Verify all tests pass
2. Create `handoffs/completion-IMPL-003.md` with:
   - What you implemented
   - Test results
   - Any issues or learnings
3. Update `TASKS.md`: change status to `completed`
4. Commit with message: "Implement FlatTransactionCost model (IMPL-003)"

---

## Important Reminder

**Save progress frequently!** Update your completion note as you go:
- After reading context files
- After implementing the class
- After writing each test
- After tests pass

This way if you hit quota limits, the next session can pick up where you left off.
