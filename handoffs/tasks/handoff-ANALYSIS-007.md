# Handoff: ANALYSIS-007 - Transaction Cost Analysis

**Task ID:** ANALYSIS-007
**Status:** ready
**Priority:** MEDIUM (Enhancement - Independent)
**Estimated Time:** 2-3 hours
**Dependencies:** None
**Assigned to:** [Available for pickup]

---

## Context & Motivation

### What are we building?

Expanding the transaction cost modeling system with three realistic cost models:

1. **SlippageCostModel** - Volume-based slippage (larger trades = more slippage)
2. **SpreadCostModel** - Bid-ask spread costs
3. **ImpactCostModel** - Market impact for large trades (square-root impact)

Plus a cost sensitivity analysis notebook to understand how costs affect strategy performance.

### Why does this matter?

**Transaction costs can make or break a strategy.**

Current state: We only have `FlatTransactionCost` (10 bps per trade)

Problem: Flat costs are unrealistic because:
- Liquid ETFs (SPY, QQQ) have < 1 bp spread
- Illiquid ETFs can have 50+ bp spread
- Large trades incur slippage and market impact
- Rebalancing frequency dramatically affects cost drag

**Example impact:**
- Strategy with 10% monthly turnover:
  - With 1 bp costs: ~1.2% annual drag
  - With 10 bp costs: ~12% annual drag (massive!)
  - With 50 bp costs: Strategy likely not viable

**This task enables:**
- Realistic cost modeling based on ETF characteristics
- Cost-aware strategy optimization
- Rebalance frequency analysis (daily vs weekly vs monthly)
- Identification of expensive trades to avoid

---

## Current State

### Existing Cost Infrastructure

**Current file: `src/quantetf/portfolio/costs.py`**

Read this first:
```bash
cat src/quantetf/portfolio/costs.py
```

Expected existing content:
```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class CostModel:
    """Base class for transaction cost models."""
    pass

@dataclass
class FlatTransactionCost(CostModel):
    """Flat cost per trade (e.g., 10 bps)."""
    cost_bps: float = 10.0

    def calculate_cost(self, old_weights, new_weights, nav):
        """Calculate transaction cost as fraction of NAV."""
        # Implementation exists...
        pass
```

**Existing tests:**
```bash
cat tests/test_transaction_costs.py
```

Expected: 22 tests for FlatTransactionCost

### How Costs Are Used in Backtest

In `SimpleBacktestEngine`:
```python
# After portfolio construction
cost_model = FlatTransactionCost(cost_bps=10.0)
cost_frac = cost_model.calculate_cost(old_weights, new_weights, nav)

# Deduct from NAV
nav_after_cost = nav * (1 - cost_frac)
```

---

## Task Specification

### Cost Model 1: SlippageCostModel

**Concept:** Slippage increases with trade size relative to average daily volume (ADV)

**Formula:**
```
slippage_bps = base_spread + (trade_size_usd / (ADV * adv_fraction)) * impact_coefficient

Where:
- base_spread: minimum cost (e.g., 1 bp for liquid ETFs)
- trade_size_usd: dollar value of trade
- ADV: average daily volume (dollars)
- adv_fraction: fraction of ADV we assume can be traded (e.g., 0.1 = 10% of daily volume)
- impact_coefficient: linear impact factor (e.g., 10 bps per 10% ADV)
```

**Simplified version** (if volume data not available):
```
slippage_bps = base_spread + (abs(weight_change) * impact_coefficient)

Example: If we go from 0% to 5% of portfolio in SPY:
slippage = 1 bp (base) + 5% * 2 = 1 + 0.10 = 11 bps
```

**Implementation:**

```python
@dataclass
class SlippageCostModel(CostModel):
    """Volume-based slippage cost model.

    Costs increase with trade size. Useful for modeling impact
    of large rebalances.

    Attributes:
        base_spread_bps: Minimum spread cost (default 5.0 bps)
        impact_coefficient: Impact per percentage point of position change (default 2.0)
    """
    base_spread_bps: float = 5.0
    impact_coefficient: float = 2.0

    def calculate_cost(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        nav: float
    ) -> float:
        """Calculate slippage cost based on position changes.

        Cost = base_spread + |weight_change| * impact_coefficient

        Args:
            old_weights: Series of old portfolio weights (ticker -> weight)
            new_weights: Series of new portfolio weights (ticker -> weight)
            nav: Current NAV (not used in this simplified model)

        Returns:
            Total cost as fraction of NAV

        Example:
            >>> old_weights = pd.Series({'SPY': 0.5, 'QQQ': 0.5})
            >>> new_weights = pd.Series({'SPY': 0.7, 'QQQ': 0.3})
            >>> model = SlippageCostModel(base_spread_bps=5.0, impact_coefficient=2.0)
            >>> cost = model.calculate_cost(old_weights, new_weights, nav=100000)
            >>> # SPY: |0.7 - 0.5| = 0.2 -> 5 + 0.2*2 = 5.4 bps
            >>> # QQQ: |0.3 - 0.5| = 0.2 -> 5 + 0.2*2 = 5.4 bps
            >>> # Avg cost ~ 5.4 bps
        """
        # Align weights
        all_tickers = old_weights.index.union(new_weights.index)
        old_w = old_weights.reindex(all_tickers, fill_value=0.0)
        new_w = new_weights.reindex(all_tickers, fill_value=0.0)

        # Calculate per-ticker costs
        weight_changes = (new_w - old_w).abs()

        costs_bps = self.base_spread_bps + weight_changes * self.impact_coefficient * 100

        # Weight-average cost by trade size
        total_trade_size = weight_changes.sum()
        if total_trade_size == 0:
            return 0.0

        avg_cost_bps = (costs_bps * weight_changes).sum() / total_trade_size

        # Return as fraction
        return avg_cost_bps / 10000.0
```

**Test cases:**
- No trade (old_weights == new_weights) → 0 cost
- Small trade (1% change) → base_spread + small impact
- Large trade (20% change) → base_spread + large impact
- Complete rebalance (100% turnover) → high cost
- Edge case: empty weights

---

### Cost Model 2: SpreadCostModel

**Concept:** Each ETF has a bid-ask spread (liquid ETFs ~1 bp, illiquid ~50+ bp)

**Implementation approach:**

```python
@dataclass
class SpreadCostModel(CostModel):
    """Bid-ask spread cost model.

    Different ETFs have different spreads based on liquidity.

    Attributes:
        spread_map: Dict mapping ticker -> spread in bps (default spreads if not specified)
        default_spread_bps: Spread for tickers not in map (default 10.0)
    """
    spread_map: dict = None
    default_spread_bps: float = 10.0

    def __post_init__(self):
        """Initialize with default spreads if not provided."""
        if self.spread_map is None:
            # Default spreads for common ETFs (based on typical bid-ask)
            self.spread_map = {
                'SPY': 1.0,
                'QQQ': 1.0,
                'IWM': 2.0,
                'VOO': 1.5,
                'VTI': 1.0,
                'EEM': 5.0,
                'TLT': 3.0,
                'GLD': 2.0,
                'VNQ': 5.0,
                # Add more as needed
            }

    def calculate_cost(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        nav: float
    ) -> float:
        """Calculate cost based on bid-ask spreads.

        For each traded ticker, apply its spread cost.

        Args:
            old_weights: Old portfolio weights
            new_weights: New portfolio weights
            nav: Current NAV

        Returns:
            Total cost as fraction of NAV

        Example:
            >>> old_weights = pd.Series({'SPY': 1.0})
            >>> new_weights = pd.Series({'QQQ': 1.0})
            >>> model = SpreadCostModel()
            >>> cost = model.calculate_cost(old_weights, new_weights, nav=100000)
            >>> # Trade 100% out of SPY (1 bp) and 100% into QQQ (1 bp)
            >>> # Cost = (1 + 1) / 2 = 1 bp average
        """
        all_tickers = old_weights.index.union(new_weights.index)
        old_w = old_weights.reindex(all_tickers, fill_value=0.0)
        new_w = new_weights.reindex(all_tickers, fill_value=0.0)

        # Calculate turnover per ticker
        turnover = (new_w - old_w).abs()

        # Get spread for each ticker
        spreads = pd.Series({
            ticker: self.spread_map.get(ticker, self.default_spread_bps)
            for ticker in all_tickers
        })

        # Cost = turnover-weighted average spread
        total_turnover = turnover.sum()
        if total_turnover == 0:
            return 0.0

        avg_spread_bps = (spreads * turnover).sum() / total_turnover

        # Return as fraction
        return avg_spread_bps / 10000.0
```

**Test cases:**
- Trade in liquid ETF (SPY) → 1 bp cost
- Trade in illiquid ETF → higher cost
- Mixed trade → weighted average
- Unknown ticker → use default_spread_bps
- Custom spread_map override

---

### Cost Model 3: ImpactCostModel

**Concept:** Market impact follows square-root law (larger trades = disproportionately higher impact)

**Formula (academic):**
```
impact_bps = sigma * sqrt(trade_size_pct / ADV_pct)

Where:
- sigma: volatility of the asset (daily)
- trade_size_pct: trade as % of portfolio
- ADV_pct: average daily volume as % of market cap
```

**Simplified implementation:**
```python
@dataclass
class ImpactCostModel(CostModel):
    """Market impact cost model using square-root law.

    Larger trades incur disproportionately higher costs.

    Attributes:
        impact_coefficient: Scaling factor for impact (default 5.0)
    """
    impact_coefficient: float = 5.0

    def calculate_cost(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        nav: float
    ) -> float:
        """Calculate market impact using square-root law.

        Formula: impact_bps = coefficient * sqrt(abs(weight_change))

        This captures the non-linear nature of market impact:
        - 1% trade → ~0.05% cost
        - 4% trade → ~0.10% cost (sqrt scaling)
        - 16% trade → ~0.20% cost

        Args:
            old_weights: Old portfolio weights
            new_weights: New portfolio weights
            nav: Current NAV

        Returns:
            Total cost as fraction of NAV

        Example:
            >>> old_weights = pd.Series({'SPY': 0.5})
            >>> new_weights = pd.Series({'SPY': 0.7})
            >>> model = ImpactCostModel(impact_coefficient=5.0)
            >>> cost = model.calculate_cost(old_weights, new_weights, nav=100000)
            >>> # weight_change = 0.2 (20%)
            >>> # impact = 5.0 * sqrt(0.2) = 2.24 bps
        """
        all_tickers = old_weights.index.union(new_weights.index)
        old_w = old_weights.reindex(all_tickers, fill_value=0.0)
        new_w = new_weights.reindex(all_tickers, fill_value=0.0)

        # Calculate weight changes
        weight_changes = (new_w - old_w).abs()

        # Apply square-root law
        import numpy as np
        impacts_bps = self.impact_coefficient * np.sqrt(weight_changes)

        # Weight-average by trade size
        total_trade_size = weight_changes.sum()
        if total_trade_size == 0:
            return 0.0

        avg_impact_bps = (impacts_bps * weight_changes).sum() / total_trade_size

        return avg_impact_bps / 10000.0
```

**Test cases:**
- Small trade (1% change) → low impact
- Large trade (25% change) → higher impact (but less than linear)
- Verify square-root relationship: 4x trade → 2x cost
- Edge case: zero trade

---

## Cost Sensitivity Notebook

**File:** `notebooks/cost_sensitivity.ipynb`

### Notebook Structure

**Section 1: Cost Model Comparison**

Compare all 4 cost models on same backtest:
```python
# Run backtest with different cost models
cost_models = {
    'Flat_1bp': FlatTransactionCost(cost_bps=1.0),
    'Flat_10bp': FlatTransactionCost(cost_bps=10.0),
    'Flat_50bp': FlatTransactionCost(cost_bps=50.0),
    'Slippage': SlippageCostModel(base_spread_bps=5.0, impact_coefficient=2.0),
    'Spread': SpreadCostModel(),
    'Impact': ImpactCostModel(impact_coefficient=5.0),
}

results = {}
for name, model in cost_models.items():
    # Run backtest with this cost model
    # Store results
    results[name] = run_backtest(cost_model=model)

# Compare final returns and Sharpe ratios
comparison_df = pd.DataFrame({
    name: {
        'Total Return': r['total_return'],
        'Sharpe Ratio': r['sharpe'],
        'Total Costs': r['total_costs'],
    }
    for name, r in results.items()
})

print(comparison_df)
```

**Section 2: Rebalance Frequency vs Cost Drag**

Test daily, weekly, monthly, quarterly rebalancing:
```python
frequencies = ['daily', 'weekly', 'monthly', 'quarterly']

results = []
for freq in frequencies:
    result = run_backtest(rebalance_freq=freq, cost_model=SpreadCostModel())
    results.append({
        'Frequency': freq,
        'Return': result['total_return'],
        'Costs': result['total_costs'],
        'Net Return': result['total_return'] - result['total_costs'],
        'Sharpe': result['sharpe'],
    })

# Plot return vs cost by frequency
df = pd.DataFrame(results)
df.plot(x='Frequency', y=['Return', 'Costs', 'Net Return'], kind='bar')
```

**Section 3: High-Turnover vs Low-Turnover Strategies**

Compare aggressive rebalancing vs lazy rebalancing:
```python
# Aggressive: Rebalance to exact weights every week
aggressive_result = run_backtest(
    rebalance_freq='weekly',
    rebalance_threshold=0.0,  # Always rebalance
    cost_model=SpreadCostModel()
)

# Lazy: Only rebalance if drift > 5%
lazy_result = run_backtest(
    rebalance_freq='monthly',
    rebalance_threshold=0.05,  # Only if >5% drift
    cost_model=SpreadCostModel()
)

# Compare
print(f"Aggressive: {aggressive_result['sharpe']:.2f} Sharpe, {aggressive_result['total_costs']:.2%} costs")
print(f"Lazy: {lazy_result['sharpe']:.2f} Sharpe, {lazy_result['total_costs']:.2%} costs")
```

**Section 4: Identify Expensive Trades**

Analyze which tickers/trades incur highest costs:
```python
# From backtest trade log, calculate cost per trade
trade_log = results['trade_log']

# Add cost estimate
trade_log['cost_estimate'] = trade_log.apply(
    lambda row: estimate_trade_cost(row['ticker'], row['size_pct'], model=SpreadCostModel()),
    axis=1
)

# Find most expensive trades
expensive_trades = trade_log.nlargest(10, 'cost_estimate')
print(expensive_trades[['date', 'ticker', 'size_pct', 'cost_estimate']])

# Summarize by ticker
cost_by_ticker = trade_log.groupby('ticker')['cost_estimate'].sum().sort_values(ascending=False)
cost_by_ticker.plot(kind='bar', title='Total Cost by Ticker')
```

**Section 5: Break-Even Analysis**

"How much alpha do I need to overcome transaction costs?"

```python
# For different cost levels, calculate required alpha to break even
cost_levels_bps = [1, 5, 10, 20, 50, 100]
rebalance_monthly = 12  # trades per year

breakeven_alpha = []
for cost_bps in cost_levels_bps:
    annual_cost = (cost_bps / 10000) * rebalance_monthly
    breakeven_alpha.append({
        'Cost_bps': cost_bps,
        'Annual_Cost': annual_cost,
        'Required_Alpha': annual_cost,
    })

df = pd.DataFrame(breakeven_alpha)
print(df)

# Interpretation: If costs are 10 bps per trade and you rebalance monthly,
# you need 1.2% annual alpha just to break even
```

---

## Implementation Guidelines

### Update `src/quantetf/portfolio/costs.py`

Add three new classes after `FlatTransactionCost`:
```python
@dataclass
class SlippageCostModel(CostModel):
    ...

@dataclass
class SpreadCostModel(CostModel):
    ...

@dataclass
class ImpactCostModel(CostModel):
    ...
```

Keep existing `FlatTransactionCost` unchanged (don't break existing backtests).

---

### Update `tests/test_transaction_costs.py`

Add new test classes:
```python
class TestSlippageCostModel:
    def test_no_trade_zero_cost(self):
        ...

    def test_small_trade_low_cost(self):
        ...

    def test_large_trade_high_cost(self):
        ...

    def test_impact_increases_with_size(self):
        """Verify larger trades cost more."""
        ...

    def test_empty_weights(self):
        ...

class TestSpreadCostModel:
    def test_liquid_etf_low_cost(self):
        """SPY should have ~1 bp cost."""
        ...

    def test_illiquid_etf_high_cost(self):
        """Unknown ticker should use default spread."""
        ...

    def test_custom_spread_map(self):
        """Test providing custom spread map."""
        ...

class TestImpactCostModel:
    def test_square_root_scaling(self):
        """Verify 4x trade has 2x cost (sqrt relationship)."""
        model = ImpactCostModel(impact_coefficient=10.0)

        # 1% trade
        cost_1pct = model.calculate_cost(
            pd.Series({'SPY': 0.0}),
            pd.Series({'SPY': 0.01}),
            nav=100000
        )

        # 4% trade (4x larger)
        cost_4pct = model.calculate_cost(
            pd.Series({'SPY': 0.0}),
            pd.Series({'SPY': 0.04}),
            nav=100000
        )

        # Should be approximately 2x (sqrt(4) = 2)
        assert abs(cost_4pct / cost_1pct - 2.0) < 0.1

# Add 5+ tests per model = 15+ new tests
```

**Target:** Add 15 new tests (5 per model)

---

## Acceptance Criteria

- [ ] Three new cost models implemented:
  - [ ] SlippageCostModel with base spread + linear impact
  - [ ] SpreadCostModel with per-ticker spreads
  - [ ] ImpactCostModel with square-root law
- [ ] Each model has:
  - [ ] Proper dataclass structure matching FlatTransactionCost pattern
  - [ ] calculate_cost() method with same signature
  - [ ] Comprehensive docstring with formula and example
  - [ ] 5+ tests (typical + edge cases)
- [ ] Cost sensitivity notebook created:
  - [ ] `notebooks/cost_sensitivity.ipynb`
  - [ ] All 5 sections implemented
  - [ ] Runs end-to-end without errors
  - [ ] Clear markdown explanations
- [ ] All tests pass: `pytest tests/test_transaction_costs.py -v`
- [ ] Existing FlatTransactionCost tests still pass (no breaking changes)
- [ ] Total test count: 22 (existing) + 15 (new) = 37+ tests

---

## Dependencies

**None** - Independent task

**Benefits downstream:**
- More realistic backtests
- Better parameter tuning (rebalance frequency)
- Cost-aware strategy optimization

---

## Inputs

**Existing files:**
- `src/quantetf/portfolio/costs.py` - Existing FlatTransactionCost
- `tests/test_transaction_costs.py` - Existing 22 tests
- Backtest results for notebook testing

---

## Outputs

**Modified files:**
1. `src/quantetf/portfolio/costs.py` - Add 3 new models (~150 lines)
2. `tests/test_transaction_costs.py` - Add 15+ tests (~200 lines)

**Created files:**
3. `notebooks/cost_sensitivity.ipynb` - Analysis notebook

---

## Examples

### Example Usage

```python
from quantetf.portfolio.costs import *
import pandas as pd

# Define portfolio change
old_weights = pd.Series({'SPY': 0.5, 'QQQ': 0.3, 'IWM': 0.2})
new_weights = pd.Series({'SPY': 0.4, 'QQQ': 0.4, 'IWM': 0.2})
nav = 100000

# Compare cost models
models = {
    'Flat 10bp': FlatTransactionCost(cost_bps=10.0),
    'Slippage': SlippageCostModel(base_spread_bps=5.0, impact_coefficient=2.0),
    'Spread': SpreadCostModel(),
    'Impact': ImpactCostModel(impact_coefficient=5.0),
}

for name, model in models.items():
    cost_frac = model.calculate_cost(old_weights, new_weights, nav)
    cost_dollar = cost_frac * nav
    print(f"{name:15s}: {cost_frac*10000:5.2f} bps (${cost_dollar:.2f})")

# Output:
# Flat 10bp      : 10.00 bps ($100.00)
# Slippage       :  8.50 bps ($85.00)
# Spread         :  1.20 bps ($12.00)
# Impact         :  3.45 bps ($34.50)
```

### Notebook Preview

```python
# notebooks/cost_sensitivity.ipynb

import pandas as pd
from quantetf.backtest.simple_engine import SimpleBacktestEngine
from quantetf.portfolio.costs import *

# Load data
snapshot = load_snapshot('data/snapshots/snapshot_5yr_20etfs')

# Compare cost models
for cost_model in [FlatTransactionCost(10), SpreadCostModel(), ImpactCostModel()]:
    engine = SimpleBacktestEngine(
        # ... config ...
        cost_model=cost_model
    )
    results = engine.run()

    print(f"{cost_model.__class__.__name__}:")
    print(f"  Total Return: {results.total_return:.2%}")
    print(f"  Total Costs:  {results.total_costs:.2%}")
    print(f"  Net Return:   {results.net_return:.2%}")
    print()
```

---

## Testing & Validation

### Run tests
```bash
pytest tests/test_transaction_costs.py -v
```

### Validate with backtest
```python
# Test in actual backtest
from scripts.run_backtest import run_backtest

result = run_backtest(
    snapshot_path='data/snapshots/snapshot_5yr_20etfs',
    cost_model='spread',  # Use SpreadCostModel
)

print(f"Total costs: {result['total_costs']:.2%}")
```

---

## References

**Academic literature:**
- Almgren & Chriss (2000) - "Optimal execution of portfolio transactions"
- Square-root market impact law: Barra Research

**Practical:**
- ETF bid-ask spreads: ETF.com, ETFdb.com
- Typical spreads: SPY ~0.01%, illiquid ETFs ~0.50%

---

## Success Criteria

✅ 3 new cost models implemented with proper signatures
✅ 15+ new tests passing (total 37+)
✅ Cost sensitivity notebook functional
✅ No breaking changes to existing code
✅ Documentation complete

**Expected time:** 2-3 hours

---

**Ready to begin!** This is an independent Wave 1 task that can run in parallel with ANALYSIS-001 and INFRA-002.
