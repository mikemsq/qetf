# QuantETF Bash Scripts Guide

Complete set of executable bash scripts for running QuantETF operations from the terminal without writing code.

## Quick Start

```bash
# View current status
./check_status.sh --all

# Run complete workflow (optimize → backtest → walk-forward)
./run_all.sh

# Or run individual operations
./run_optimization.sh --parallel 8
./run_backtest.sh --analysis
./run_walk_forward.sh --plots

# View results
./view_results.sh --walk-forward latest
```

## Scripts Overview

### Main Operation Scripts

#### `run_optimization.sh` - Strategy Parameter Optimization
Perform grid search across strategy parameter space to find optimal configurations.

**Usage:**
```bash
./run_optimization.sh [OPTIONS]
```

**Key Options:**
- `--parallel N` - Number of parallel jobs (default: 4)
- `--max-configs N` - Limit number of configurations tested
- `--dry-run` - Show command without executing
- `--verbose` - Enable verbose output
- `--help` - Show full help

**Examples:**
```bash
# Standard optimization with 4 parallel jobs
./run_optimization.sh

# Fast optimization with 8 parallel jobs
./run_optimization.sh --parallel 8

# Test only 50 configurations
./run_optimization.sh --max-configs 50

# Preview commands without running
./run_optimization.sh --dry-run
```

**Output:**
- `artifacts/optimization/[timestamp]/` - Results directory
- `best_strategy.yaml` - Optimal parameter configuration
- `results_summary.json` - Performance metrics across all configurations

**Runtime:** 15-60 minutes depending on parallel jobs and configuration count

---

#### `run_backtest.sh` - Full Backtest with Analysis
Execute comprehensive backtest with cycle metrics, drawdown analysis, and performance reports.

**Usage:**
```bash
./run_backtest.sh [OPTIONS]
```

**Key Options:**
- `-s, --snapshot PATH` - Snapshot to use (default: latest)
- `--strategy NAME` - Strategy type
- `--top-n N` - Number of holdings (default: 5)
- `--lookback N` - Momentum lookback days (default: 252)
- `--cost-bps N` - Transaction costs in basis points (default: 10)
- `--rebalance FREQ` - Rebalance frequency: weekly/monthly/quarterly
- `--start DATE` - Start date (YYYY-MM-DD)
- `--end DATE` - End date (YYYY-MM-DD)
- `-a, --analysis` - Generate enhanced analysis report
- `--verbose` - Enable verbose output

**Examples:**
```bash
# Standard backtest
./run_backtest.sh

# Backtest with analysis
./run_backtest.sh --analysis

# Custom date range
./run_backtest.sh --start 2020-01-01 --end 2024-12-31 --analysis

# Custom strategy parameters
./run_backtest.sh --top-n 7 --lookback 126 --cost-bps 5
```

**Output:**
- `artifacts/backtests/[timestamp]/` - Results directory
- `performance_analysis.json` - Sharpe, Sortino, max drawdown, returns
- `cycle_metrics.json` - Monthly and daily performance metrics
- `backtest_results.csv` - Full price and return history

**Runtime:** 5-15 minutes depending on snapshot size

---

#### `run_walk_forward.sh` - Out-of-Sample Validation
Perform rolling window walk-forward test to validate strategy robustness on unseen data.

**Usage:**
```bash
./run_walk_forward.sh [OPTIONS]
```

**Key Options:**
- `-s, --snapshot PATH` - Snapshot to use
- `--train-years N` - Training window size (default: 5)
- `--test-years N` - Testing window size (default: 1)
- `--step-months N` - Step between windows (default: 12)
- `--top-n N` - Number of holdings (default: 5)
- `--lookback N` - Momentum lookback days (default: 252)
- `--cost-bps N` - Transaction costs in bps (default: 10)
- `--rebalance FREQ` - Rebalance frequency
- `-p, --plots` - Save visualization plots
- `--verbose` - Enable verbose output

**Examples:**
```bash
# Standard walk-forward (5-year train, 1-year test)
./run_walk_forward.sh

# With visualization plots
./run_walk_forward.sh --plots

# Custom windows (3-year train, 1-year test)
./run_walk_forward.sh --train-years 3 --test-years 1 --step-months 6 --plots
```

**Output:**
- `artifacts/walk_forward/[timestamp]/` - Results directory
- `summary.json` - Validation statistics
  - `num_windows` - Number of rolling windows
  - `oos_sharpe_mean` - Out-of-sample Sharpe ratio
  - `oos_return_mean` - Out-of-sample returns
  - `pct_windows_oos_positive` - % of windows with positive returns
  - `pct_windows_oos_beats_is` - % of windows beating training
  - `sharpe_degradation` - In-sample to out-of-sample degradation
- `window_results.csv` - Per-window metrics
- `walk_forward_analysis.png` - Performance visualization
- `window_N/` directories - Detailed per-window results

**Runtime:** 30-120 minutes depending on data size and window count

---

#### `run_monitoring.sh` - Real-Time Performance Tracking
Monitor current strategy performance and compare to benchmarks.

**Usage:**
```bash
./run_monitoring.sh [OPTIONS]
```

**Key Options:**
- `-c, --config PATH` - Strategy config file
- `--monitor-days N` - Recent days to monitor (default: 30)
- `--continuous` - Run continuous monitoring (every 5 minutes)
- `--email-alerts` - Send email for drawdowns > 5%
- `--verbose` - Enable verbose output

**Examples:**
```bash
# Monitor last 30 days
./run_monitoring.sh

# Monitor last 90 days
./run_monitoring.sh --monitor-days 90

# Continuous monitoring
./run_monitoring.sh --continuous

# With email alerts
./run_monitoring.sh --continuous --email-alerts
```

**Output:**
- `artifacts/monitoring/[timestamp]/` - Results directory
- `latest_metrics.json` - Current performance metrics
- Real-time alerts if drawdowns exceed thresholds

**Runtime:** 1-5 minutes per check, continuous mode runs indefinitely

---

#### `run_all.sh` - Complete Workflow Orchestrator
Execute full workflow in sequence: optimize → backtest → walk-forward → monitor.

**Usage:**
```bash
./run_all.sh [OPTIONS]
```

**Key Options:**
- `--steps STEPS` - Comma-separated steps (default: optimize,backtest,walk-forward)
  - Available: `optimize`, `backtest`, `walk-forward`, `monitor`
- `--parallel N` - Parallel jobs for optimization
- `--skip-validation` - Skip walk-forward validation
- `--dry-run` - Show commands without executing

**Examples:**
```bash
# Complete workflow
./run_all.sh

# Only optimization and backtest
./run_all.sh --steps optimize,backtest

# Just validation of existing strategy
./run_all.sh --steps walk-forward

# Preview before running
./run_all.sh --dry-run
```

**Output:**
- Combined results from all executed steps
- Summary report with total runtime
- Progress tracking for each step

**Runtime:** 60-240 minutes for complete workflow

---

### Utility Scripts

#### `check_status.sh` - Process and Output Status
Check running processes, view latest results, and monitor logs.

**Usage:**
```bash
./check_status.sh [OPTIONS]
```

**Key Options:**
- `--all` - Show all status (default)
- `--processes` - Show running quantetf processes
- `--optimization` - Show latest optimization results
- `--backtest` - Show latest backtest results
- `--walk-forward` - Show latest walk-forward results
- `--logs N` - Show last N lines of logs

**Examples:**
```bash
# Full status check
./check_status.sh

# Check for running processes
./check_status.sh --processes

# View latest walk-forward results
./check_status.sh --walk-forward

# Show last 50 lines of logs
./check_status.sh --logs 50
```

---

#### `view_results.sh` - Detailed Results Viewer
Display detailed analysis results and metrics from completed runs.

**Usage:**
```bash
./view_results.sh [OPTIONS]
```

**Key Options:**
- `--optimization` - View latest optimization results
- `--backtest [ID]` - View backtest (ID or 'latest')
- `--walk-forward [ID]` - View walk-forward results (ID or 'latest')
- `--list-all` - List all available results
- `--metrics-only` - Show key metrics only (no tables)

**Examples:**
```bash
# View latest optimization
./view_results.sh --optimization

# View latest backtest
./view_results.sh --backtest latest

# View specific walk-forward by ID
./view_results.sh --walk-forward 20260117_182400

# List all available results
./view_results.sh --list-all

# View metrics only (compact output)
./view_results.sh --walk-forward latest --metrics-only
```

---

## Workflow Recipes

### Recipe 1: Find and Validate Best Strategy
```bash
# 1. Optimize to find best parameters
./run_optimization.sh --parallel 8

# 2. Run backtest on best strategy
./run_backtest.sh --analysis

# 3. Validate on out-of-sample data
./run_walk_forward.sh --plots

# 4. View results
./view_results.sh --walk-forward latest
```

### Recipe 2: Quick Validation Check
```bash
# Validate current best strategy without re-optimizing
./run_walk_forward.sh --plots --verbose

# Check status
./check_status.sh --walk-forward
```

### Recipe 3: Period-Specific Analysis
```bash
# Test strategy on recent data (2023-2026)
./run_backtest.sh \
  --start 2023-01-01 \
  --end 2026-01-15 \
  --analysis

# Compare to longer period
./run_backtest.sh \
  --start 2016-01-15 \
  --end 2026-01-15 \
  --analysis
```

### Recipe 4: Parameter Sensitivity Analysis
```bash
# Test with fewer holdings
./run_backtest.sh --top-n 3 --analysis

# Test with more holdings
./run_backtest.sh --top-n 10 --analysis

# Test with lower lookback
./run_backtest.sh --lookback 126 --analysis

# Test with higher costs
./run_backtest.sh --cost-bps 15 --analysis
```

### Recipe 5: Continuous Production Monitoring
```bash
# Start continuous monitoring with alerts
./run_monitoring.sh --continuous --email-alerts

# In another terminal, check status
watch -n 60 './check_status.sh --processes'
```

---

## Key Directories

```
artifacts/
├── optimization/        # Grid search results
│   └── [timestamp]/
│       ├── best_strategy.yaml
│       └── results_summary.json
├── backtests/          # Backtest outputs
│   └── [timestamp]/
│       ├── performance_analysis.json
│       ├── cycle_metrics.json
│       └── backtest_results.csv
└── walk_forward/       # Walk-forward validation results
    └── [timestamp]/
        ├── summary.json
        ├── window_results.csv
        ├── walk_forward_analysis.png
        └── window_N/
```

---

## Performance Guidelines

| Operation | Time | CPU | Memory | Output Size |
|-----------|------|-----|--------|-------------|
| Optimization | 15-60 min | 100% × N | 2-4 GB | 50-200 MB |
| Backtest | 5-15 min | 50-100% | 1-2 GB | 10-50 MB |
| Walk-Forward | 30-120 min | 80-100% | 2-4 GB | 100-500 MB |
| Monitoring | 1-5 min | 20% | 500 MB | 5-10 MB |

---

## Troubleshooting

### Script Not Executable
```bash
chmod +x scripts/*.sh
```

### Permission Denied
```bash
ls -la scripts/run_*.sh
# If not executable, run chmod above
```

### Python Command Not Found
Scripts assume `python` or `python3` is in PATH. Verify:
```bash
which python
which python3
```

### Process Still Running
Check status before re-running:
```bash
./check_status.sh --processes
```

### View Recent Logs
```bash
ls -lt *.log | head -5
tail -f walk_forward_*.log  # Follow latest log
```

### Cleanup Old Results
```bash
# Remove results older than 7 days
find artifacts/ -maxdepth 2 -type d -mtime +7 -exec rm -rf {} \;
```

---

## Environment Requirements

- **Python:** 3.8+
- **System:** Linux/Mac (bash 4.0+)
- **Storage:** 2-5 GB for artifacts
- **Memory:** 4-8 GB RAM recommended
- **CPU:** Multi-core (4+ cores recommended for parallel optimization)

---

## Getting Help

```bash
# Each script has detailed help
./run_optimization.sh --help
./run_backtest.sh --help
./run_walk_forward.sh --help
./run_monitoring.sh --help
./run_all.sh --help
./check_status.sh --help
./view_results.sh --help
```

---

## Notes

- All timestamps are YYYYMMDD_HHMMSS format
- Results are automatically dated and organized by operation type
- Scripts create necessary directories automatically
- Color output helps identify warnings (yellow) and errors (red)
- Use `--dry-run` flag to preview commands before execution
- Walk-forward validation should be run before deploying to production
