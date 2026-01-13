# Quick Start: Implementing Phase 1 Momentum Strategies

This guide will get you started implementing the three Phase 1 momentum strategies with minimal friction.

## TL;DR - Get Started in 5 Minutes

**Simplest path**: Start with Momentum Acceleration

```bash
# 1. Pick a handout
cat docs/handouts/HANDOUT_momentum_acceleration.md

# 2. Create the file
touch src/quantetf/alpha/momentum_acceleration.py

# 3. Copy the code template (Section 4 of handout)
# 4. Replace TODOs with implementation (Section 3 has pseudocode)
# 5. Create test file
touch tests/alpha/test_momentum_acceleration.py

# 6. Run tests
pytest tests/alpha/test_momentum_acceleration.py -v

# 7. Run backtest
python scripts/run_backtest.py --help
```

---

## Strategy Comparison

| Strategy | Complexity | Computation | Time Est | Start Here? |
|----------|-----------|-------------|----------|-------------|
| Momentum Acceleration | LOW | 2 returns + subtract | 2-3 hrs | ✅ YES |
| Vol-Adjusted Momentum | LOW | Return + std dev | 2-3 hrs | ✅ Good 2nd |
| Residual Momentum | MEDIUM | OLS regression | 3-4 hrs | Later |

---

## Implementation Path

### Option A: Simplest First (Recommended)

**Best for**: Quick wins, building confidence, iterative development

1. **Momentum Acceleration** (2-3 hours)
   - File: `src/quantetf/alpha/momentum_acceleration.py`
   - Handout: `HANDOUT_momentum_acceleration.md`
   - Why first: Simplest computation, easiest to debug
   - Test: Only 2 return calculations

2. **Volatility-Adjusted Momentum** (2-3 hours)
   - File: `src/quantetf/alpha/vol_adjusted_momentum.py`
   - Handout: `HANDOUT_vol_adjusted_momentum.md`
   - Why second: Adds volatility calculation, still simple
   - Test: Introduces risk-adjustment concept

3. **Residual Momentum** (3-4 hours)
   - File: `src/quantetf/alpha/residual_momentum.py`
   - Handout: `HANDOUT_residual_momentum.md`
   - Why last: Most complex (regression), builds on prior learning
   - Test: Full statistical regression implementation

**Total time**: 7-10 hours

---

### Option B: Alpha Potential First

**Best for**: Prioritizing performance, experienced implementers

1. Residual Momentum (highest alpha)
2. Vol-Adjusted Momentum (defensive)
3. Momentum Acceleration (tactical timing)

---

## Step-by-Step: Momentum Acceleration

### 1. Read the Handout (10 min)

```bash
less docs/handouts/HANDOUT_momentum_acceleration.md
```

Pay special attention to:
- Section 2: Mathematical definition
- Section 3: Algorithm pseudocode
- Section 4: Code template
- Section 5: Test cases

### 2. Study Reference Files (15 min)

```bash
# Primary reference
cat src/quantetf/alpha/momentum.py

# Focus on:
# - Class structure (lines 46-73)
# - score() method (lines 74-150)
# - Error handling patterns
```

### 3. Create Implementation File (90 min)

```bash
# Create file
touch src/quantetf/alpha/momentum_acceleration.py

# Copy template from Section 4 of handout
# Then fill in the TODOs:
```

Key implementation points:
- Validate `short_lookback < long_lookback` in `__init__`
- Get prices with proper lookback
- Calculate two returns (short and long)
- Subtract: `score = short - long`
- Handle NaN for insufficient data

### 4. Create Tests (60 min)

```bash
# Create test file
touch tests/alpha/test_momentum_acceleration.py

# Implement 8 test cases from Section 5:
# 1. Positive acceleration
# 2. Negative acceleration (deceleration)
# 3. Steady momentum
# 4. Reversal detection
# 5. Insufficient data
# 6. No lookahead bias
# 7. Parameter validation
# 8. Integration backtest
```

### 5. Run Tests (15 min)

```bash
# Run tests
pytest tests/alpha/test_momentum_acceleration.py -v

# Check coverage
pytest tests/alpha/test_momentum_acceleration.py --cov=src/quantetf/alpha/momentum_acceleration --cov-report=term-missing

# Should be >90% coverage
```

### 6. Create Config (10 min)

```bash
# Create strategy config
cat > configs/strategies/momentum_acceleration_top5.yaml << 'EOF'
name: momentum_acceleration_top5_ew
universe: configs/universes/tier1_initial_20.yaml
schedule: configs/schedules/monthly_rebalance.yaml
cost_model: configs/costs/flat_10bps.yaml

alpha_model:
  type: momentum_acceleration
  short_lookback_days: 63
  long_lookback_days: 252
  min_periods: 200

portfolio_construction:
  type: equal_weight_top_n
  top_n: 5
  constraints:
    max_weight: 0.60
    min_weight: 0.00
EOF
```

### 7. Run Backtest (10 min)

```bash
# Check if snapshot exists
ls -lh data/snapshots/

# Run backtest
python scripts/run_backtest.py \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --strategy momentum-acceleration-top5 \
  --output artifacts/backtests

# Review results
cat artifacts/backtests/latest/metrics.json
```

### 8. Validate Results (10 min)

Check success criteria (from Section 6):
- ✅ All tests pass
- ✅ Sharpe ratio > 0.5
- ✅ Max drawdown < 35%
- ✅ Information ratio vs SPY > 0.2
- ✅ Code coverage > 90%

---

## Common Issues & Solutions

### Issue: Import errors

```python
# Solution: Check your imports match these
from quantetf.alpha.base import AlphaModel
from quantetf.types import AlphaScores, Universe
from quantetf.data.snapshot_store import SnapshotDataStore
```

### Issue: Type hints failing

```bash
# Run mypy
mypy src/quantetf/alpha/momentum_acceleration.py

# Common fixes:
# - Add Optional[...] for nullable types
# - Use pd.Timestamp not datetime
# - Return AlphaScores not dict
```

### Issue: Tests failing on lookahead bias

```python
# Ensure you use store correctly
prices = store.get_close_prices(
    as_of=as_of,  # This enforces T-1 cutoff
    tickers=list(universe.tickers),
    lookback_days=self.long_lookback_days + 50
)

# Verify:
assert prices.index.max() < as_of  # Strict inequality!
```

### Issue: Backtest fails

Check:
1. Did you register the alpha model in the factory?
2. Does config YAML match your class parameters?
3. Is snapshot path correct?
4. Do you have enough warmup period?

---

## Parallel Implementation

If you have multiple agents, split work:

**Agent A**: Momentum Acceleration
- Implements simplest strategy
- Sets pattern for others
- ~2-3 hours

**Agent B**: Vol-Adjusted Momentum
- Implements while A works
- Can reference A's patterns
- ~2-3 hours

**Agent C**: Residual Momentum
- Can start after A or B complete
- Most complex, benefits from seeing patterns
- ~3-4 hours

**Total parallel time**: 3-4 hours vs 7-10 hours serial

---

## Validation Checklist

Before considering a strategy "done":

- [ ] Code
  - [ ] File created with correct name
  - [ ] Inherits from AlphaModel
  - [ ] Implements score() method
  - [ ] Type hints present
  - [ ] Docstrings complete

- [ ] Tests
  - [ ] Test file created
  - [ ] All 8 test cases implemented
  - [ ] Tests pass
  - [ ] Coverage > 90%

- [ ] Integration
  - [ ] Config YAML created
  - [ ] Backtest runs successfully
  - [ ] Metrics meet success criteria

- [ ] Quality
  - [ ] No mypy errors
  - [ ] Black formatting applied
  - [ ] No flake8 warnings
  - [ ] No lookahead bias

---

## Next Steps After Implementation

Once all 3 strategies are implemented:

1. **Compare Performance**
   ```bash
   # Run comparison across all strategies
   python scripts/compare_strategies.py \
     --strategies momentum_acceleration vol_adjusted residual_momentum \
     --output artifacts/phase1_comparison/
   ```

2. **Walk-Forward Validation**
   ```bash
   # Test robustness
   python scripts/walk_forward_test.py \
     --strategy momentum_acceleration_top5
   ```

3. **Ensemble Testing**
   ```bash
   # Try combining strategies
   # See STRATEGY_RESEARCH_PLAN.md Phase 3
   ```

---

## Getting Help

- **Handout unclear?**: Check reference files (Section 9)
- **Pattern question?**: Look at `src/quantetf/alpha/momentum.py`
- **Test question?**: Look at `tests/test_no_lookahead.py`
- **Integration question?**: Check existing configs in `configs/strategies/`

---

## Success Metrics Summary

| Strategy | Sharpe | IR vs SPY | Max DD | Time |
|----------|--------|-----------|--------|------|
| Momentum Acceleration | >0.5 | >0.2 | <35% | 2-3h |
| Vol-Adjusted Momentum | >0.6 | >0.3 | <30% | 2-3h |
| Residual Momentum | >0.3 | >0.3 | <40% | 3-4h |

If your implementation meets these targets, you're done! 🎉

---

**Last Updated**: 2026-01-13
**Author**: Quant Research Team
