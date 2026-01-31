# Bug Fix: Universe Configuration Ignored in Strategy Optimizer

## Summary

The `MultiPeriodEvaluator` ignores the universe configuration specified in `StrategyConfig`, causing all universe tiers (tier2/tier3/tier4) to produce identical backtest results. This prevents meaningful analysis of whether wider vs narrower ETF universes provide alpha benefits.

## Evidence

Analysis of optimization run `20260130_213658`:
- 1,314 configurations tested (438 unique strategies × 3 universe tiers)
- **Zero variance** in results across tiers for identical strategy parameters
- Max absolute difference in returns between any tier pair: `0.0000000000`

## Root Cause

### Location
`src/quantetf/optimization/evaluator.py` lines 307-319

### Current (Buggy) Code
```python
def _evaluate_period(self, config: StrategyConfig, years: int) -> PeriodMetrics:
    ...
    # BUG: Gets ALL tickers from data accessor, ignoring config.universe_path
    tickers = self.data_access.prices.get_available_tickers()

    # Creates universe from ALL available tickers, not from config
    universe = Universe(as_of=self.end_date, tickers=tuple(tickers))
```

### The Problem
1. `StrategyConfig` contains `universe_path` (e.g., `configs/universes/tier2_core_50.yaml`)
2. `StrategyConfig` contains `universe_name` (e.g., `tier2`)
3. Neither field is ever read or used in `_evaluate_period()`
4. Instead, the evaluator fetches ALL tickers from the price data accessor

## Expected Behavior

The evaluator should:
1. Load the universe YAML file specified in `config.universe_path`
2. Extract the ticker list from the YAML
3. Filter to only those tickers when creating the `Universe` object
4. Pass the filtered universe to the backtest engine

## Proposed Fix

### Option A: Load Universe from YAML (Recommended)

```python
def _evaluate_period(self, config: StrategyConfig, years: int) -> PeriodMetrics:
    ...
    # Load universe from config
    universe_tickers = self._load_universe_tickers(config.universe_path)

    # Filter to tickers that exist in our data
    available_tickers = set(self.data_access.prices.get_available_tickers())
    valid_tickers = [t for t in universe_tickers if t in available_tickers]

    # Create universe with filtered tickers
    universe = Universe(as_of=self.end_date, tickers=tuple(valid_tickers))
```

New helper method:
```python
def _load_universe_tickers(self, universe_path: str) -> List[str]:
    """Load ticker list from universe YAML config."""
    import yaml
    from pathlib import Path

    config_path = Path(universe_path)
    if not config_path.exists():
        raise ValueError(f"Universe config not found: {universe_path}")

    with open(config_path) as f:
        universe_config = yaml.safe_load(f)

    return universe_config['source']['tickers']
```

### Option B: Use Existing Universe Loader

Check if `src/quantetf/data/access/universe.py` or `src/quantetf/universe/` already has universe loading logic that can be reused.

## Files to Modify

| File | Change |
|------|--------|
| `src/quantetf/optimization/evaluator.py` | Add universe loading logic in `_evaluate_period()` |
| `src/quantetf/optimization/evaluator.py` | Add `_load_universe_tickers()` helper method |

## Files to Reference

| File | Purpose |
|------|---------|
| `src/quantetf/optimization/grid.py` | `StrategyConfig` dataclass with `universe_path` field |
| `configs/universes/tier2_core_50.yaml` | Example universe YAML structure |
| `configs/universes/tier3_expanded_100.yaml` | Example universe YAML structure |
| `configs/universes/tier4_broad_200.yaml` | Example universe YAML structure |

## Universe YAML Structure

```yaml
name: tier2_core_50_etfs
source:
  type: static_list
  tickers:
    - SPY
    - QQQ
    - IWM
    # ... etc
```

## Testing Requirements

### Unit Tests
1. Test that `_load_universe_tickers()` correctly parses YAML
2. Test that missing tickers in data are filtered out gracefully
3. Test that empty universe raises appropriate error

### Integration Tests
1. Run optimization with tier2 only → verify ~50 tickers used
2. Run optimization with tier4 only → verify ~200 tickers used
3. Run same strategy on tier2 vs tier4 → verify **different** results

### Validation
Re-run optimization sweep after fix and verify:
- Different tiers produce different composite scores
- Larger universes may (or may not) find different winning ETFs
- Results are reproducible

## Impact Assessment

- **Low risk**: Change is isolated to evaluator's data loading
- **No API changes**: `StrategyConfig` already has the fields
- **Backward compatible**: Fix uses existing config structure

## Success Criteria

After fix, running the same optimization should show:
1. Variance > 0 between tier results for same strategy params
2. Potentially different top-N winning strategies per tier
3. Ability to answer: "Does a wider universe provide alpha benefits?"
