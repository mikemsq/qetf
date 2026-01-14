# Strategy Optimizer System - Main Handout

## Overview

Build an automated system to find ETF strategies that beat SPY across multiple time periods (3, 5, and 10 years). The system systematically searches across alpha models, parameters, schedules, and portfolio construction options.

## Success Criteria

A strategy "beats SPY" when it achieves:
1. **Positive excess return** (strategy return > SPY return)
2. **Information Ratio > 0** (positive risk-adjusted excess return)
3. **Consistent across ALL three periods** (3yr, 5yr, 10yr)

## Architecture

```
src/quantetf/optimization/
├── __init__.py
├── grid.py          # Parameter grid generator
├── evaluator.py     # Multi-period evaluation
└── optimizer.py     # Main orchestrator

scripts/
└── find_best_strategy.py  # CLI entry point
```

## Component Breakdown

### Task 1: Parameter Grid Generator (`grid.py`)
See: `HANDOUT_grid_generator.md`

Defines schedule-specific parameter search spaces for all 4 alpha models.

### Task 2: Multi-Period Evaluator (`evaluator.py`)
See: `HANDOUT_multi_period_evaluator.md`

Evaluates a single strategy config across 3yr, 5yr, and 10yr windows.

### Task 3: Strategy Optimizer (`optimizer.py`)
See: `HANDOUT_optimizer.md`

Main orchestrator that generates configs, runs evaluations, and ranks results.

### Task 4: CLI Script (`find_best_strategy.py`)
See: `HANDOUT_cli_script.md`

Command-line interface for running optimization.

## Key Design Decisions

### 1. Schedule-Specific Parameters
Weekly rebalancing uses shorter lookback periods (faster signals match frequent trading).
Monthly rebalancing uses longer lookback periods (stable signals match infrequent trading).

### 2. Universe Constraint
Use Tier 3 only (`configs/universes/tier3_expanded_100.yaml`) - 100 liquid ETFs.

### 3. Top N Holdings
Limited to [3, 5, 7] - avoids over-diversification.

### 4. Validation Constraints
- `momentum_acceleration`: short_lookback_days must be < long_lookback_days
- min_periods should be less than lookback_days

## Expected Configuration Count

| Schedule | Alpha Models | Params | × top_n | Total |
|----------|-------------|--------|---------|-------|
| Weekly   | 4           | ~46    | 3       | ~138  |
| Monthly  | 4           | ~72    | 3       | ~216  |
| **Total**|             |        |         | **~354** |

## Output Structure

```
artifacts/optimization/TIMESTAMP/
├── all_results.csv           # All configs with metrics
├── winners.csv               # Configs that beat SPY in all periods
├── best_strategy.yaml        # Top-ranked config file
├── optimization_report.md    # Summary report
└── backtests/                # Individual backtest results
```

## Key Files to Reference

| File | Purpose |
|------|---------|
| `src/quantetf/alpha/factory.py` | Alpha model registration |
| `src/quantetf/config/loader.py` | Config loading |
| `src/quantetf/backtest/simple_engine.py` | Backtest execution |
| `src/quantetf/evaluation/metrics.py` | `calculate_active_metrics()` |
| `src/quantetf/evaluation/benchmarks.py` | SPY benchmark |
| `configs/universes/tier3_expanded_100.yaml` | Universe definition |
| `configs/schedules/weekly_rebalance.yaml` | Weekly schedule |
| `configs/schedules/monthly_rebalance.yaml` | Monthly schedule |

## Implementation Order

1. **grid.py** - No dependencies, pure config generation
2. **evaluator.py** - Depends on existing backtest infrastructure
3. **optimizer.py** - Depends on grid.py and evaluator.py
4. **find_best_strategy.py** - Depends on optimizer.py

## Risk Mitigation

1. **Overfitting**: Use 3 different time periods as pseudo-out-of-sample validation
2. **Data snooping**: Report ALL results, not just winners
3. **Computational cost**: ~354 configs is manageable (~15-25 min)
4. **Error handling**: Log and skip failed configs, don't abort entire run
