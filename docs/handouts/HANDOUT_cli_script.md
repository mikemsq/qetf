# Task 4: CLI Scripts for Strategy Optimization

## Overview

The strategy optimization pipeline is split into three modular scripts, orchestrated by a shell script:

| Script | Purpose |
|--------|---------|
| `run_backtests.py` | Run backtests, save results to pickle |
| `rank_strategies.py` | Load results, apply ranking criteria |
| `analyze_regimes.py` | Load results, run regime analysis |
| `optimize.sh` | Orchestrate all three steps |

## Files

```
scripts/
├── run_backtests.py      # Step 1: Run backtests
├── rank_strategies.py    # Step 2: Rank strategies
├── analyze_regimes.py    # Step 3: Regime analysis
└── optimize.sh           # Orchestration script
```

## Usage

### Full Pipeline (Recommended)

```bash
# Run complete optimization pipeline
./scripts/optimize.sh

# With options
./scripts/optimize.sh --max-configs 100 --scoring-method trailing_1y

# Quick test
./scripts/optimize.sh --max-configs 20

# Show help
./scripts/optimize.sh --help
```

### Individual Scripts

```bash
# Step 1: Run backtests only
python scripts/run_backtests.py \
    --snapshot data/snapshots/snapshot_latest \
    --periods 1,3 \
    --parallel 8

# Step 2: Rank with different criteria (no rerun needed)
python scripts/rank_strategies.py \
    --results artifacts/optimization/.../backtest_results.pkl \
    --scoring-method multi_period

python scripts/rank_strategies.py \
    --results artifacts/optimization/.../backtest_results.pkl \
    --scoring-method trailing_1y

# Step 3: Regime analysis
python scripts/analyze_regimes.py \
    --results artifacts/optimization/.../backtest_results.pkl \
    --snapshot data/snapshots/snapshot_latest
```

## Script Details

### run_backtests.py

Runs all strategy backtests and saves comprehensive results.

```bash
python scripts/run_backtests.py --help
```

**Key Options:**
- `--snapshot PATH` - Data snapshot path (required)
- `--output PATH` - Output directory (default: artifacts/optimization)
- `--periods LIST` - Evaluation periods in years (default: 1,3)
- `--parallel N` - Number of parallel workers (default: 1)
- `--max-configs N` - Limit configs for testing
- `--dry-run` - Just count configs

**Output:**
- `backtest_results.pkl` - Full results with daily returns (for regime analysis)
- `backtest_results.csv` - Human-readable metrics

### rank_strategies.py

Loads saved backtest results and applies ranking without rerunning backtests.

```bash
python scripts/rank_strategies.py --help
```

**Key Options:**
- `--results PATH` - Path to backtest_results.pkl (required)
- `--output PATH` - Output directory
- `--scoring-method METHOD` - One of: multi_period, trailing_1y, regime_weighted

**Output:**
- `all_results.csv` - All strategies ranked
- `winners.csv` - Strategies beating SPY
- `best_strategy.yaml` - Top strategy config

### analyze_regimes.py

Runs regime analysis on saved backtest results.

```bash
python scripts/analyze_regimes.py --help
```

**Key Options:**
- `--results PATH` - Path to backtest_results.pkl (required)
- `--snapshot PATH` - Data snapshot for macro data (required)
- `--output PATH` - Output directory

**Output:**
- `regime_mapping.yaml` - Regime → strategy mapping
- `regime_analysis.csv` - Per-strategy per-regime metrics
- `regime_history.parquet` - Daily regime labels

### optimize.sh

Orchestrates all three steps in sequence.

```bash
./scripts/optimize.sh --help
```

**Key Options:**
- `--snapshot PATH` - Data snapshot (default: data/snapshots/snapshot_latest)
- `--output PATH` - Output directory (default: artifacts/optimization)
- `--periods LIST` - Evaluation periods (default: 1)
- `--parallel N` - Parallel workers (default: 8)
- `--scoring-method M` - Scoring method (default: regime_weighted)
- `--max-configs N` - Limit configs
- `--skip-regime` - Skip regime analysis step
- `--dry-run` - Preview only

## Example Output

```
============================================================
STRATEGY OPTIMIZATION PIPELINE
============================================================
Snapshot:       data/snapshots/snapshot_latest
Output:         artifacts/optimization
Periods:        1 years
Parallel:       8 workers
Scoring:        regime_weighted
Started:        2026-01-31 20:30:14
============================================================

>>> Step 1/3: Running backtests...
[Progress output...]

============================================================
BACKTEST RUN COMPLETE
============================================================
Total configs tested:     1314
Failed configs:           0
Strategies that beat SPY: 245

>>> Step 2/3: Ranking strategies...
[Ranking output...]

>>> Step 3/3: Running regime analysis...
[Regime output...]

============================================================
PIPELINE COMPLETE
============================================================
Started:  2026-01-31 20:30:14
Ended:    2026-01-31 21:45:32
Duration: 01:15:18

Results:  artifacts/optimization/20260131_203014
============================================================
```

## Benefits of Modular Design

1. **No rerunning backtests** - Rank with different methods using saved results
2. **Separate regime analysis** - Can run independently
3. **Composable** - Run steps individually or combine
4. **Shell orchestration** - Easy to modify workflow
5. **Debugging** - Run just the failing step

## Output Directory Structure

```
artifacts/optimization/[timestamp]/
├── [backtest_timestamp]/
│   ├── backtest_results.pkl     # Raw results with daily returns
│   └── backtest_results.csv     # Human-readable metrics
├── ranked_[timestamp]/
│   ├── all_results.csv          # Ranked strategies
│   ├── winners.csv              # SPY-beaters
│   └── best_strategy.yaml       # Top strategy
└── regime_[timestamp]/
    ├── regime_mapping.yaml      # Regime → strategy
    ├── regime_analysis.csv      # Per-regime metrics
    └── regime_history.parquet   # Daily regime labels
```

## Dependencies

- Python 3.8+
- pandas, numpy, pyyaml
- quantetf package

## Notes

- Always test with `--max-configs 20` first
- Use `--dry-run` to verify config count
- Results are timestamped, won't overwrite
- Pickle format preserves daily returns for regime analysis
