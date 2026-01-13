# Strategy Comparison Guide

This guide explains how to run parameter sweeps, compare strategies, and choose the best performing alpha model using the QuantETF framework.

## Overview

The strategy comparison system provides three main capabilities:

1. **Single Strategy Backtest** - Run one config and analyze results
2. **Strategy Sweep** - Run multiple configs and compare them
3. **Parameter Sweep** - Test different parameter combinations for a single strategy

## Quick Start

### Run a Single Strategy

```bash
# Run momentum acceleration strategy
python scripts/run_backtest_from_config.py \
  --config configs/strategies/momentum_acceleration_top5.yaml
```

### Compare Multiple Strategies

```bash
# Compare all Phase 1 momentum strategies
python scripts/run_strategy_sweep.py \
  --configs configs/strategies/momentum_*.yaml \
           configs/strategies/residual_*.yaml \
  --output artifacts/comparisons/phase1_comparison
```

### Compare with Custom Date Range

```bash
# Test strategies on recent market period
python scripts/run_strategy_sweep.py \
  --configs configs/strategies/*.yaml \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --output artifacts/comparisons/recent_period
```

## Detailed Usage

### 1. Running Individual Strategies

Use `run_backtest_from_config.py` to test a single strategy configuration:

```bash
python scripts/run_backtest_from_config.py \
  --config configs/strategies/residual_momentum_top5.yaml \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --capital 100000 \
  --output-dir artifacts/backtests
```

**Parameters:**
- `--config`: Path to strategy YAML file (required)
- `--snapshot`: Path to snapshot data directory (default: `data/snapshots/snapshot_5yr_20etfs`)
- `--start`: Backtest start date YYYY-MM-DD (default: `2021-01-01`)
- `--end`: Backtest end date YYYY-MM-DD (default: `2025-12-31`)
- `--capital`: Initial capital in dollars (default: `100000`)
- `--output-dir`: Output directory for results (default: `artifacts/backtests`)
- `--verbose`: Enable verbose logging

**Output Files:**
- `equity_curve.csv` - NAV over time
- `holdings_history.csv` - Share holdings at each rebalance
- `weights_history.csv` - Portfolio weights at each rebalance
- `metrics.json` - Performance metrics (Sharpe, returns, drawdown, etc.)
- `config.json` - Complete configuration used for the run
- `out.txt` - Log output

### 2. Strategy Sweeps (Comparing Multiple Strategies)

Use `run_strategy_sweep.py` to run and compare multiple strategy configurations:

```bash
python scripts/run_strategy_sweep.py \
  --configs configs/strategies/momentum_*.yaml \
  --output artifacts/comparisons/momentum_sweep \
  --start 2021-01-01 \
  --end 2025-12-31
```

**Glob Pattern Support:**
```bash
# All strategies
--configs configs/strategies/*.yaml

# All momentum variants
--configs configs/strategies/momentum_*.yaml

# Specific strategies
--configs configs/strategies/momentum_acceleration_top5.yaml \
          configs/strategies/vol_adjusted_momentum_top5.yaml

# Mix patterns and specific files
--configs configs/strategies/momentum_*.yaml \
          configs/strategies/residual_momentum_top5.yaml
```

**Parameters:**
- `--configs`: List of strategy config files or glob patterns (required)
- `--snapshot`: Snapshot data directory (default: `data/snapshots/snapshot_5yr_20etfs`)
- `--start`: Backtest start date (default: `2021-01-01`)
- `--end`: Backtest end date (default: `2025-12-31`)
- `--capital`: Initial capital (default: `100000`)
- `--output`: Output directory (default: `artifacts/comparisons/TIMESTAMP`)
- `--no-spy-benchmark`: Disable automatic SPY benchmark comparison
- `--verbose`: Enable verbose logging

**Output Structure:**
```
artifacts/comparisons/TIMESTAMP/
├── backtests/                           # Individual backtest results
│   ├── momentum_acceleration_top5_ew/
│   │   ├── equity_curve.csv
│   │   ├── metrics.json
│   │   └── ...
│   ├── vol_adjusted_momentum_top5_ew/
│   └── residual_momentum_top5_ew/
├── strategy_comparison_report.md        # Markdown comparison report
├── strategy_comparison_metrics.csv      # Metrics table
├── strategy_comparison_equity.png       # Equity curve overlay chart
└── strategy_comparison_scatter.png      # Risk-return scatter plot
```

**Console Output Example:**
```
================================================================================
STRATEGY COMPARISON SUMMARY
================================================================================
                                 total_return    cagr  sharpe_ratio  max_drawdown  volatility
momentum_acceleration_top5_ew          45.23%  10.12%         1.234        -23.45%      15.67%
vol_adjusted_momentum_top5_ew          38.91%   8.76%         1.456        -18.23%      11.23%
residual_momentum_top5_ew              52.34%  11.45%         1.123        -25.67%      17.89%
SPY Benchmark                          35.67%   8.12%         0.987        -20.34%      14.56%
================================================================================
```

### 3. Parameter Sweeps (Testing Different Parameters)

To test different parameters for a single strategy, create multiple config variants:

**Example: Testing different lookback periods for momentum acceleration**

Create config files with different parameters:

```yaml
# configs/strategies/momentum_accel_short.yaml
name: momentum_accel_3m_vs_9m
alpha_model:
  type: momentum_acceleration
  short_lookback_days: 63   # 3 months
  long_lookback_days: 189   # 9 months
# ... rest of config

# configs/strategies/momentum_accel_medium.yaml
name: momentum_accel_3m_vs_12m
alpha_model:
  type: momentum_acceleration
  short_lookback_days: 63   # 3 months
  long_lookback_days: 252   # 12 months

# configs/strategies/momentum_accel_long.yaml
name: momentum_accel_6m_vs_12m
alpha_model:
  type: momentum_acceleration
  short_lookback_days: 126  # 6 months
  long_lookback_days: 252   # 12 months
```

Then run the sweep:

```bash
python scripts/run_strategy_sweep.py \
  --configs configs/strategies/momentum_accel_*.yaml \
  --output artifacts/sweeps/momentum_accel_lookback_sweep
```

## Comparing Existing Backtests

If you already have backtest results, use the comparison script:

```bash
python scripts/compare_strategies.py \
  --backtest-dirs artifacts/backtests/20260113_*/  \
  --output artifacts/comparisons/latest
```

## Interpretation Guide

### Key Metrics

**Returns:**
- `total_return`: Cumulative return over the period
- `cagr`: Compound Annual Growth Rate

**Risk-Adjusted Performance:**
- `sharpe_ratio`: Return per unit of volatility (higher is better, >1.0 is good)
- `sortino_ratio`: Like Sharpe but only penalizes downside volatility
- `information_ratio`: Excess return vs SPY per unit of tracking error

**Risk Metrics:**
- `max_drawdown`: Worst peak-to-trough decline (lower magnitude is better)
- `volatility`: Annualized standard deviation of returns
- `win_rate`: Percentage of rebalance periods with positive returns

**Other:**
- `num_rebalances`: Number of portfolio rebalances
- `total_costs`: Total transaction costs paid

### Choosing the Best Strategy

Consider these factors:

1. **Sharpe Ratio** - Primary metric for risk-adjusted performance
   - Target: >0.5 (acceptable), >1.0 (good), >1.5 (excellent)

2. **Maximum Drawdown** - How much you can tolerate losing
   - Conservative: <20%
   - Moderate: 20-30%
   - Aggressive: >30%

3. **Correlation to SPY** - Diversification benefit
   - Low correlation (<0.5) means more diversification
   - High correlation (>0.8) means similar to holding SPY

4. **Consistency** - Win rate and distribution of returns
   - Higher win rate = more consistent strategy

5. **Statistical Significance** - Check t-test results
   - p-value <0.05 means significant outperformance

**Example Decision Matrix:**

| Strategy | Sharpe | Max DD | vs SPY | Best For |
|----------|--------|--------|--------|----------|
| Momentum Acceleration | 1.2 | -24% | Higher risk, higher return | Bull markets, trend followers |
| Vol-Adjusted Momentum | 1.4 | -18% | Lower risk, stable | Bear markets, risk-averse |
| Residual Momentum | 1.1 | -26% | Beta-neutral alpha | Low correlation to market |

## Advanced Usage

### Testing Different Universes

Compare strategy performance across different universes:

```bash
# Create config variants with different universes
for universe in tier1_initial_20 tier2_core_50 tier3_expanded_100; do
  # Modify config to use different universe
  sed "s/tier1_initial_20/$universe/" \
    configs/strategies/momentum_acceleration_top5.yaml \
    > configs/strategies/momentum_accel_${universe}.yaml
done

# Run sweep
python scripts/run_strategy_sweep.py \
  --configs configs/strategies/momentum_accel_tier*.yaml \
  --output artifacts/sweeps/universe_comparison
```

### Testing Different Portfolio Sizes

```yaml
# Top 3
portfolio_construction:
  type: equal_weight_top_n
  top_n: 3

# Top 5 (default)
portfolio_construction:
  type: equal_weight_top_n
  top_n: 5

# Top 10
portfolio_construction:
  type: equal_weight_top_n
  top_n: 10
```

### Walk-Forward Analysis

For production-ready validation:

```bash
# In-sample period
python scripts/run_strategy_sweep.py \
  --configs configs/strategies/*.yaml \
  --start 2018-01-01 --end 2021-12-31 \
  --output artifacts/analysis/in_sample

# Out-of-sample period
python scripts/run_strategy_sweep.py \
  --configs configs/strategies/*.yaml \
  --start 2022-01-01 --end 2025-12-31 \
  --output artifacts/analysis/out_of_sample

# Compare degradation
python scripts/compare_strategies.py \
  --backtest-dirs \
    artifacts/analysis/in_sample/backtests/* \
    artifacts/analysis/out_of_sample/backtests/* \
  --output artifacts/analysis/is_vs_oos
```

## Troubleshooting

### Config Not Loading

Check that your YAML file has required fields:
```yaml
name: strategy_name
universe: configs/universes/tier1_initial_20.yaml
schedule: configs/schedules/monthly_rebalance.yaml
cost_model: configs/costs/flat_10bps.yaml

alpha_model:
  type: momentum_acceleration  # Must be registered in factory
  # ... parameters

portfolio_construction:
  type: equal_weight_top_n
  top_n: 5
```

### Missing Data

Ensure snapshot covers the backtest period:
```bash
# Check snapshot date range
python -c "
from quantetf.data.snapshot_store import SnapshotDataStore
store = SnapshotDataStore('data/snapshots/snapshot_5yr_20etfs/data.parquet')
print(f'Start: {store.start_date}')
print(f'End: {store.end_date}')
"
```

### Alpha Model Not Found

Register custom alpha models in `src/quantetf/alpha/factory.py`:
```python
from quantetf.alpha.my_custom_alpha import MyCustomAlpha
AlphaModelRegistry.register('my_custom', MyCustomAlpha)
```

## Examples

### Example 1: Choose Best Momentum Strategy

```bash
# Compare all momentum variants
python scripts/run_strategy_sweep.py \
  --configs \
    configs/strategies/momentum_acceleration_top5.yaml \
    configs/strategies/vol_adjusted_momentum_top5.yaml \
    configs/strategies/residual_momentum_top5.yaml \
  --output artifacts/comparisons/phase1_momentum \
  --verbose

# Review results
cat artifacts/comparisons/phase1_momentum/strategy_comparison_report.md
```

### Example 2: Optimize Lookback Period

```bash
# Create parameter grid (you need to create these configs first)
for short in 42 63 84; do
  for long in 189 252 315; do
    # Create config with these parameters
    # ... (config generation code)
  done
done

# Run parameter sweep
python scripts/run_strategy_sweep.py \
  --configs configs/strategies/param_sweep_*.yaml \
  --output artifacts/sweeps/lookback_optimization
```

### Example 3: Portfolio Size Optimization

```bash
# Compare top-3, top-5, top-10
python scripts/run_strategy_sweep.py \
  --configs \
    configs/strategies/momentum_accel_top3.yaml \
    configs/strategies/momentum_accel_top5.yaml \
    configs/strategies/momentum_accel_top10.yaml \
  --output artifacts/sweeps/portfolio_size_sweep
```

## Next Steps

1. Run initial comparison of Phase 1 strategies
2. Identify best performing strategy
3. Run parameter sweeps to optimize performance
4. Validate with walk-forward analysis
5. Deploy chosen strategy to production

## References

- Strategy configs: `configs/strategies/`
- Universe definitions: `configs/universes/`
- Backtest engine: `src/quantetf/backtest/simple_engine.py`
- Comparison tools: `src/quantetf/evaluation/comparison.py`
