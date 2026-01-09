# Task Handoff: IMPL-004 - SimpleBacktestEngine

**Task ID:** IMPL-004
**Status:** ready
**Priority:** high
**Estimated Time:** 3-4 hours

---

## Quick Context

You are implementing the SimpleBacktestEngine - the orchestrator that ties together all Phase 2 components into a working backtest. This is the most important piece of Phase 2 because it makes everything else come together.

**Why this matters:** This is where the magic happens. The engine loops through time, makes decisions using historical data only, and tracks portfolio performance. Get this right and we have a working backtest system!

---

## What You Need to Know

### Architecture

The backtest engine is **event-driven** (not vectorized):
- Iterate through rebalance dates chronologically
- At each date, make decisions using only T-1 data
- Track holdings, costs, and performance over time
- Return complete backtest results

### Components Already Built ✅

1. **SnapshotDataStore** - Point-in-time data access (T-1 enforcement)
2. **MomentumAlpha** - Generates alpha scores from historical data
3. **EqualWeightTopN** - Converts scores to target weights
4. **FlatTransactionCost** - Calculates rebalancing costs
5. **No-lookahead tests** - Validates we never use future data

### Design Decisions

- **Event-driven loop:** One iteration per rebalance date
- **T-1 data access:** All decisions use previous day's close
- **Monthly rebalancing:** Default schedule (configurable)
- **Track everything:** Holdings, weights, NAV, costs, returns
- **Return structured results:** Easy to analyze and visualize

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Coding standards, no-lookahead rules
2. **`/workspaces/qetf/src/quantetf/backtest/base.py`** - BacktestEngine base class
3. **`/workspaces/qetf/src/quantetf/types.py`** - Data structures (BacktestResult, etc.)
4. **`/workspaces/qetf/src/quantetf/alpha/momentum.py`** - Example component usage
5. **`/workspaces/qetf/src/quantetf/portfolio/equal_weight.py`** - Portfolio constructor
6. **`/workspaces/qetf/src/quantetf/portfolio/costs.py`** - Cost model

---

## Implementation Steps

### 1. Create the file

Create `/workspaces/qetf/src/quantetf/backtest/simple_engine.py`

### 2. Import dependencies

```python
"""Simple event-driven backtest engine."""

import logging
from typing import List
import pandas as pd
import numpy as np

from quantetf.backtest.base import BacktestEngine
from quantetf.types import (
    AlphaModel,
    PortfolioConstructor,
    CostModel,
    Universe,
    BacktestResult,
    BacktestConfig,
)
from quantetf.data.store import DataStore

logger = logging.getLogger(__name__)
```

### 3. Implement SimpleBacktestEngine class

**Key methods:**

```python
class SimpleBacktestEngine(BacktestEngine):
    """Simple event-driven backtest engine.

    Iterates through rebalance dates chronologically, making portfolio decisions
    using only historical data (T-1), and tracking performance over time.

    Example:
        >>> engine = SimpleBacktestEngine()
        >>> result = engine.run(
        ...     config=config,
        ...     alpha_model=MomentumAlpha(lookback_days=252),
        ...     portfolio=EqualWeightTopN(top_n=5),
        ...     cost_model=FlatTransactionCost(cost_bps=10.0),
        ...     store=store
        ... )
    """

    def run(
        self,
        *,
        config: BacktestConfig,
        alpha_model: AlphaModel,
        portfolio: PortfolioConstructor,
        cost_model: CostModel,
        store: DataStore,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            config: Backtest configuration (dates, universe, initial capital)
            alpha_model: Alpha model to generate signals
            portfolio: Portfolio constructor for target weights
            cost_model: Transaction cost model
            store: Data store for historical prices

        Returns:
            BacktestResult with equity curve, holdings, metrics
        """
        # TODO: Implement backtest loop
        pass
```

### 4. Implement the backtest loop

**Core algorithm:**

```python
def run(self, *, config, alpha_model, portfolio, cost_model, store):
    """Run the backtest."""

    # 1. Initialize tracking
    nav = config.initial_capital
    holdings = pd.Series(0.0, index=list(config.universe.tickers))  # shares
    weights = pd.Series(0.0, index=list(config.universe.tickers))  # portfolio weights

    # History tracking
    nav_history = []
    holdings_history = []
    weights_history = []
    costs_history = []

    # 2. Generate rebalance dates
    rebalance_dates = _generate_rebalance_dates(
        start=config.start_date,
        end=config.end_date,
        frequency='monthly'  # or from config
    )

    logger.info(f"Running backtest: {len(rebalance_dates)} rebalance dates")

    # 3. Event loop - iterate through rebalance dates
    for i, rebalance_date in enumerate(rebalance_dates):
        logger.info(f"Rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}")

        # 3a. Get prices (T-1 data only!)
        prices = store.get_close_prices(as_of=rebalance_date)
        if prices.empty:
            logger.warning(f"No data available for {rebalance_date}, skipping")
            continue

        # Get latest prices for universe tickers
        latest_prices = prices.iloc[-1]  # Most recent (T-1)

        # 3b. Mark to market - update NAV with current holdings
        if i > 0:  # Not first iteration
            portfolio_value = (holdings * latest_prices).sum()
            nav = portfolio_value

        # 3c. Generate alpha scores
        alpha_scores = alpha_model.score(
            as_of=rebalance_date,
            universe=config.universe,
            features=None,  # Not used yet
            store=store
        )

        # 3d. Construct target portfolio
        target_weights = portfolio.construct(
            as_of=rebalance_date,
            universe=config.universe,
            alpha=alpha_scores,
            risk=None,  # Not used yet
            store=store,
            prev_weights=weights
        )

        # 3e. Calculate costs
        cost = cost_model.estimate_rebalance_cost(
            prev_weights=weights,
            next_weights=target_weights.weights,
            prices=latest_prices
        )
        cost_dollars = cost * nav  # Convert from fraction to dollars

        # 3f. Apply costs to NAV
        nav -= cost_dollars

        # 3g. Calculate new holdings (shares to buy/sell)
        target_dollars = target_weights.weights * nav
        new_holdings = target_dollars / latest_prices
        new_holdings = new_holdings.fillna(0.0)

        # 3h. Update state
        holdings = new_holdings
        weights = target_weights.weights

        # 3i. Track history
        nav_history.append({
            'date': rebalance_date,
            'nav': nav,
            'cost': cost_dollars
        })
        holdings_history.append(holdings.copy())
        weights_history.append(weights.copy())
        costs_history.append(cost_dollars)

    # 4. Calculate final metrics
    nav_df = pd.DataFrame(nav_history).set_index('date')
    nav_df['returns'] = nav_df['nav'].pct_change()

    total_return = (nav_df['nav'].iloc[-1] / config.initial_capital) - 1.0
    sharpe_ratio = _calculate_sharpe(nav_df['returns'])
    max_drawdown = _calculate_max_drawdown(nav_df['nav'])

    # 5. Return results
    return BacktestResult(
        equity_curve=nav_df,
        holdings_history=pd.DataFrame(holdings_history, index=rebalance_dates),
        weights_history=pd.DataFrame(weights_history, index=rebalance_dates),
        metrics={
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_costs': nav_df['cost'].sum(),
            'num_rebalances': len(rebalance_dates)
        },
        config=config
    )
```

### 5. Add helper functions

```python
def _generate_rebalance_dates(start, end, frequency='monthly'):
    """Generate rebalance dates between start and end."""
    if frequency == 'monthly':
        # Last business day of each month
        dates = pd.date_range(start, end, freq='BME')  # Business Month End
    elif frequency == 'weekly':
        dates = pd.date_range(start, end, freq='W-FRI')
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    return dates.tolist()


def _calculate_sharpe(returns, periods_per_year=12):
    """Calculate annualized Sharpe ratio."""
    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe


def _calculate_max_drawdown(nav_series):
    """Calculate maximum drawdown."""
    running_max = nav_series.expanding().max()
    drawdown = (nav_series - running_max) / running_max
    return drawdown.min()
```

### 6. Create tests

Create `/workspaces/qetf/tests/test_backtest_engine.py`

**Test cases needed:**

1. **Basic backtest runs successfully**
   - Simple 1-year backtest
   - Verify equity curve generated
   - Verify metrics calculated

2. **No lookahead bias**
   - Use synthetic data
   - Verify decisions made with T-1 data only

3. **Costs applied correctly**
   - Track cumulative costs
   - Verify NAV reduced by costs

4. **Holdings tracked correctly**
   - Verify shares calculated from weights
   - Verify portfolio value = sum(holdings * prices)

5. **Edge cases**
   - Empty universe
   - Insufficient data for some dates
   - All NaN alpha scores

Example test:

```python
def test_simple_backtest_runs():
    """Test that a basic backtest completes successfully."""
    from quantetf.backtest.simple_engine import SimpleBacktestEngine
    from quantetf.alpha.momentum import MomentumAlpha
    from quantetf.portfolio.equal_weight import EqualWeightTopN
    from quantetf.portfolio.costs import FlatTransactionCost
    from quantetf.data.snapshot_store import SnapshotDataStore
    from quantetf.types import BacktestConfig, Universe
    import pandas as pd

    # Setup
    store = SnapshotDataStore('/path/to/snapshot')
    universe = Universe(
        as_of=pd.Timestamp('2023-12-31'),
        tickers=('SPY', 'QQQ', 'IWM')
    )

    config = BacktestConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2023-12-31'),
        universe=universe,
        initial_capital=100000.0
    )

    # Run backtest
    engine = SimpleBacktestEngine()
    result = engine.run(
        config=config,
        alpha_model=MomentumAlpha(lookback_days=60),
        portfolio=EqualWeightTopN(top_n=2),
        cost_model=FlatTransactionCost(cost_bps=10.0),
        store=store
    )

    # Verify
    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0
    assert 'total_return' in result.metrics
    assert 'sharpe_ratio' in result.metrics
    assert result.metrics['total_return'] is not None
```

### 7. Run tests

```bash
pytest tests/test_backtest_engine.py -v
```

---

## Acceptance Criteria

- [ ] SimpleBacktestEngine class implements BacktestEngine interface
- [ ] Event-driven loop iterates through rebalance dates chronologically
- [ ] Uses T-1 data for all decisions (enforced by SnapshotDataStore)
- [ ] Calculates holdings from weights and NAV
- [ ] Applies transaction costs correctly
- [ ] Tracks NAV, holdings, weights history
- [ ] Calculates metrics: total return, Sharpe ratio, max drawdown
- [ ] Returns BacktestResult with all required fields
- [ ] Comprehensive docstrings and logging
- [ ] Tests cover basic backtest and edge cases
- [ ] All tests pass

---

## Success Looks Like

```python
# Run a complete backtest
from quantetf.backtest.simple_engine import SimpleBacktestEngine
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost

engine = SimpleBacktestEngine()

result = engine.run(
    config=config,  # 2019-2024, 20 ETFs, $100K
    alpha_model=MomentumAlpha(lookback_days=252),
    portfolio=EqualWeightTopN(top_n=5),
    cost_model=FlatTransactionCost(cost_bps=10.0),
    store=snapshot_store
)

print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
print(f"Total Costs: ${result.metrics['total_costs']:,.2f}")

# Plot equity curve
result.equity_curve['nav'].plot()
```

---

## Questions? Issues?

If blocked or unclear:
1. **Check CLAUDE_CONTEXT.md** - Event-driven backtest patterns
2. **Look at component implementations** - See how alpha/portfolio/costs work
3. **Check types.py** - BacktestResult, BacktestConfig structures
4. **Document questions** in `handoffs/completion-IMPL-004.md`

---

## Related Files

- Base class: `src/quantetf/backtest/base.py`
- Types: `src/quantetf/types.py`
- Components: `src/quantetf/alpha/`, `src/quantetf/portfolio/`
- Data access: `src/quantetf/data/snapshot_store.py`
- Standards: `CLAUDE_CONTEXT.md`

---

## Important Notes

### No Lookahead!

The most critical requirement: **NEVER use future data**

- Always use `store.get_close_prices(as_of=date)` - returns data < date
- Never access data from rebalance_date itself
- TEST-001 validates this - run it after implementation

### Event-Driven (Not Vectorized)

We iterate through dates explicitly:
- Easier to debug (can print state at each date)
- Clearer logic (one decision per loop iteration)
- Prevents accidental lookahead (can't "vectorize into future")

### State Management

Track state carefully:
- `holdings` (shares) - physical position
- `weights` (fractions) - portfolio allocation
- `nav` (dollars) - total portfolio value

All three must stay synchronized!

---

## When Done

1. Verify all tests pass (including TEST-001 no-lookahead tests)
2. Run a test backtest on real snapshot data
3. Create `handoffs/completion-IMPL-004.md` with:
   - Implementation details
   - Test results
   - Sample backtest results
   - Any issues or learnings
4. Update `TASKS.md`: change status to `completed`
5. Commit with message: "Implement SimpleBacktestEngine (IMPL-004)"

---

## Important Reminder

**Save progress frequently!** Update your completion note as you go:
- After implementing core loop
- After adding each helper function
- After writing each test
- After tests pass

This way if you hit quota limits, the next session can pick up where you left off.

---

## Estimated Breakdown

**3-4 hours total:**
- 20 min: Read context and understand components
- 90 min: Implement SimpleBacktestEngine class
- 60 min: Write comprehensive tests
- 30 min: Debug and fix issues
- 20 min: Create completion note and update docs

This is the biggest single task in Phase 2, but all the pieces are ready!
