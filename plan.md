# Bug Fix: Universe Configuration Ignored in Strategy Optimizer

## Status: ✅ FIXED (2026-01-31)

## Summary

The `MultiPeriodEvaluator` was ignoring the universe configuration specified in `StrategyConfig`, causing all universe tiers (tier2/tier3/tier4) to produce identical backtest results. This prevented meaningful analysis of whether wider vs narrower ETF universes provide alpha benefits.

## Evidence (Before Fix)

Analysis of optimization run `20260130_213658`:
- 1,314 configurations tested (438 unique strategies × 3 universe tiers)
- **Zero variance** in results across tiers for identical strategy parameters
- Max absolute difference in returns between any tier pair: `0.0000000000`

## Root Cause

### Location
`src/quantetf/optimization/evaluator.py` lines 307-319

### Original (Buggy) Code
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
3. Neither field was ever read or used in `_evaluate_period()`
4. Instead, the evaluator fetched ALL tickers from the price data accessor

## Fix Applied

### Changes Made

**1. `src/quantetf/optimization/evaluator.py`**

Added `_load_universe_tickers()` helper method (lines 489-519):
```python
def _load_universe_tickers(self, universe_path: str) -> List[str]:
    """Load ticker list from universe YAML config."""
    config_path = Path(universe_path)
    if not config_path.exists():
        raise ValueError(f"Universe config not found: {universe_path}")

    with open(config_path) as f:
        universe_config = yaml.safe_load(f)

    if 'source' not in universe_config:
        raise ValueError(f"Universe config missing 'source' section: {universe_path}")

    if 'tickers' not in universe_config['source']:
        raise ValueError(f"Universe config missing 'source.tickers': {universe_path}")

    tickers = universe_config['source']['tickers']
    if not tickers:
        raise ValueError(f"Universe config has empty ticker list: {universe_path}")

    return tickers
```

Modified `_evaluate_period()` (lines 306-334):
```python
# Load universe tickers from config
universe_tickers = self._load_universe_tickers(config.universe_path)

# Filter to tickers that exist in our data
available_tickers = set(self.data_access.prices.get_available_tickers())
valid_tickers = [t for t in universe_tickers if t in available_tickers]

if not valid_tickers:
    raise ValueError(
        f"No valid tickers found for universe '{config.universe_name}'. "
        f"Universe has {len(universe_tickers)} tickers, none available in data."
    )

# Create universe with filtered tickers from config
universe = Universe(as_of=self.end_date, tickers=tuple(valid_tickers))
```

**2. Test Fixes**
- `tests/optimization/test_evaluator.py` - Added `universe_name` to fixtures + new tests for `_load_universe_tickers()`
- `tests/optimization/test_grid.py` - Updated fixtures + corrected expected alpha type count (4→7)
- `tests/optimization/test_optimizer.py` - Added `universe_name` to all StrategyConfig instances

## Verification

```
$ python -c "from quantetf.optimization.evaluator import MultiPeriodEvaluator; ..."
Tier2 tickers: 50
Tier4 tickers: 200
SUCCESS: Universe loading works correctly!
```

## Test Results

| Test Suite | Result |
|------------|--------|
| `tests/optimization/test_evaluator.py` | **27 passed** |
| `tests/optimization/test_grid.py` | **37 passed** |
| `tests/optimization/test_optimizer.py` | **35 passed** |
| **Total optimization tests** | **99 passed** |

## Next Steps

Re-run the optimization sweep to validate that:
1. Different tiers produce **different** composite scores
2. Larger universes may find different winning strategies
3. Results are reproducible
