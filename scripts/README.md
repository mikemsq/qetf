# scripts/

One-off utilities and operational scripts.

Prefer adding stable functionality to the CLI in `src/quantetf/cli/` once it becomes reusable.

## Available Scripts

### [harmonize_etf_lists.py](harmonize_etf_lists.py)
Harmonizes multiple ETF list sources into a unified universe CSV file.

**Purpose:** Merges data from `etf_list.csv` and `ishares_list.csv`, resolves conflicts, infers issuer families, and creates a standardized ETF universe with consistent metadata.

**Usage:**
```bash
python scripts/harmonize_etf_lists.py \
    --etf-list data/raw/etf_list.csv \
    --ishares-list data/raw/ishares_list.csv \
    --out data/curated/etf_universe.csv
```

**Output:** A unified CSV file with columns: ticker, name, provider, issuer_family, inception_date, expense_ratio_pct, asset_class, and more.

---

### [download_ohlcv_stooq.py](download_ohlcv_stooq.py)
Free daily OHLCV downloader for ETFs via Stooq.

**Purpose:** Downloads historical price data from Stooq for ETFs defined in a universe CSV. Stores per-ticker parquet files and maintains a manifest for incremental updates.

**Usage:**
```bash
# Download full universe
python scripts/download_ohlcv_stooq.py \
    --universe data/curated/etf_universe.csv \
    --out-dir data/raw/prices/stooq

# Download with limits (testing)
python scripts/download_ohlcv_stooq.py \
    --universe data/curated/etf_universe.csv \
    --max-tickers 10 \
    --sleep 0.2
```

**Features:**
- Incremental downloads (only fetches new data)
- Politeness throttling with configurable sleep
- Manifest tracking with download statistics
- Automatic retry and error handling

---

### [ingest_etf_data.py](ingest_etf_data.py)
ETF data ingestion script using Yahoo Finance.

**Purpose:** Fetches historical price data for ETFs from Yahoo Finance, validates data quality, detects anomalies, and stores curated data with comprehensive metadata.

**Usage:**
```bash
# Fetch specific date range
python scripts/ingest_etf_data.py \
    --universe initial_20_etfs \
    --start-date 2021-01-01 \
    --end-date 2026-01-09

# Fetch last N years
python scripts/ingest_etf_data.py \
    --universe initial_20_etfs \
    --lookback-years 5

# Validate tickers only
python scripts/ingest_etf_data.py \
    --universe initial_20_etfs \
    --validate-only
```

**Output:**
- Curated parquet file with MultiIndex columns (Ticker, Price)
- Metadata YAML with validation results and anomaly detection
- Stored in `data/curated/` with timestamp

---

### [create_snapshot.py](create_snapshot.py)
Creates immutable, versioned data snapshots for reproducible backtesting.

**Purpose:** Takes curated data and creates a versioned snapshot with manifest, validation details, and git commit hash for reproducibility.

**Usage:**
```bash
# From specific curated file
python scripts/create_snapshot.py \
    --input data/curated/initial_20_etfs_2021-01-10_2026-01-09_20260109_022324.parquet

# From universe (uses latest)
python scripts/create_snapshot.py \
    --universe initial_20_etfs

# With custom name
python scripts/create_snapshot.py \
    --universe initial_20_etfs \
    --name snapshot_5yr_20etfs
```

**Output:**
- `data/snapshots/{snapshot_name}/`
  - `data.parquet`: Immutable price data
  - `manifest.yaml`: Metadata, date ranges, tickers, git commit
  - `validation.yaml`: Quality checks and anomaly details

---

### [run_backtest.py](run_backtest.py)
Orchestrates a complete backtest run using the SimpleBacktestEngine.

**Purpose:** Loads snapshot data, configures strategy components (alpha model, portfolio, cost model), executes backtest, and saves comprehensive results.

**Usage:**
```bash
# Run with defaults (5-year snapshot, momentum top-5 strategy)
python scripts/run_backtest.py

# Run with custom parameters
python scripts/run_backtest.py \
    --snapshot data/snapshots/snapshot_5yr_20etfs \
    --start 2021-01-01 \
    --end 2025-12-31 \
    --top-n 5 \
    --lookback 252 \
    --cost-bps 10.0

# Run with smaller universe
python scripts/run_backtest.py \
    --top-n 3 \
    --lookback 126 \
    --cost-bps 5 \
    --rebalance monthly
```

**Output:**
- `artifacts/backtests/{timestamp}_{strategy}/`
  - `equity_curve.csv`: NAV and costs over time
  - `holdings_history.csv`: Share holdings at each rebalance
  - `weights_history.csv`: Portfolio weights at each rebalance
  - `metrics.json`: Performance metrics (returns, Sharpe, drawdown)
  - `config.json`: Backtest configuration

**Strategy Components:**
- **Alpha Model:** Momentum-based ranking (configurable lookback)
- **Portfolio:** Equal-weight top-N ETFs
- **Cost Model:** Flat transaction costs in basis points

---

### [compare_strategies.py](compare_strategies.py)
Compare multiple strategy backtests and generate comparative analysis.

**Purpose:** Loads multiple backtest results, computes comparative metrics, performs statistical significance tests (Sharpe ratio t-tests), and generates comprehensive reports with charts, tables, and correlation analysis.

**Usage:**
```bash
# Compare all backtests in a directory
python scripts/compare_strategies.py \
    --backtest-dirs artifacts/backtests/*/ \
    --output artifacts/comparisons/latest

# Compare specific strategies
python scripts/compare_strategies.py \
    --backtest-dirs \
        artifacts/backtests/20260111_023934_momentum-ew-top5 \
        artifacts/backtests/20260111_024110_momentum-ew-top3 \
    --output artifacts/comparisons/momentum_comparison \
    --report-name momentum_3v5

# Compare with verbose logging
python scripts/compare_strategies.py \
    --backtest-dirs artifacts/backtests/momentum_* \
    --output artifacts/comparisons/ \
    --report-name momentum_comparison \
    --verbose
```

**Output:**
- `{output_dir}/{report_name}_metrics.csv`: Comparison table with all metrics
- `{output_dir}/{report_name}_table.html`: Styled HTML comparison table
- `{output_dir}/{report_name}_equity.png`: Equity curve overlay chart
- `{output_dir}/{report_name}_risk_return.png`: Risk-return scatter plot
- `{output_dir}/{report_name}_correlation.csv`: Returns correlation matrix
- `{output_dir}/{report_name}_significance_tests.csv`: Pairwise Sharpe ratio t-tests

**Features:**
- Comprehensive metrics comparison (CAGR, Sharpe, Sortino, Calmar, etc.)
- Statistical significance testing (Jobson-Korkie test for Sharpe ratios)
- Correlation analysis of strategy returns
- Professional visualizations (equity overlay, risk-return scatter)
- HTML report generation with styling
- Console summary with formatted tables

**Metrics Compared:**
- Total return, CAGR
- Sharpe ratio, Sortino ratio, Calmar ratio
- Max drawdown, volatility
- Win rate, VaR, CVaR
- Total costs, number of rebalances
- Final NAV

---

### [walk_forward_test.py](walk_forward_test.py)
Walk-forward validation framework for testing strategy robustness.

**Purpose:** Performs rolling window validation to assess out-of-sample performance and detect overfitting. Tests the strategy on unseen data using a configurable train/test window approach.

**Usage:**
```bash
# Run with defaults (2y train, 1y test, 6m step)
python scripts/walk_forward_test.py

# Custom train/test windows
python scripts/walk_forward_test.py \
    --train-years 3 \
    --test-years 1 \
    --step-months 3

# Custom strategy parameters
python scripts/walk_forward_test.py \
    --top-n 7 \
    --lookback 126 \
    --cost-bps 5

# Save detailed output with visualizations
python scripts/walk_forward_test.py \
    --output artifacts/walk_forward/test1 \
    --save-plots
```

**Output:**
- Summary statistics (in-sample vs out-of-sample performance)
- Degradation metrics (Sharpe degradation, return degradation)
- Stability metrics (coefficient of variation, positive window percentage)
- Window-by-window results CSV
- Visualization plots (Sharpe ratio evolution, degradation by window)
- Individual window equity curves

**Key Metrics:**
- In-sample vs out-of-sample Sharpe ratio
- Performance degradation (IS - OOS)
- Percentage of positive OOS windows
- Stability of returns across windows

**Interpretation:**
- **Low degradation:** Strategy is robust and not overfit
- **High degradation:** Strategy may be overfit to training data
- **Positive OOS Sharpe:** Strategy shows genuine predictive power
- **Stable performance:** Consistent results across different time periods

---

### [find_best_strategy.py](find_best_strategy.py)
Strategy optimization CLI that searches across all parameter combinations to find strategies that beat SPY.

**Purpose:** Runs the strategy optimizer to systematically search through alpha models, parameters, and portfolio construction options. Identifies strategies that consistently outperform SPY across multiple time periods (3yr, 5yr, 10yr by default).

**Usage:**
```bash
# Basic run (all 324 configurations)
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157

# Quick test with limited configs
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --max-configs 20 --verbose

# Parallel execution (faster)
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --parallel 4

# Dry run (count configs only)
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --dry-run

# Custom evaluation periods
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --periods 1,3,5

# Filter by schedule and alpha type
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --schedules weekly \
    --alpha-types momentum,vol_adjusted_momentum
```

**Output:**
- `artifacts/optimization/{timestamp}/`
  - `all_results.csv`: Every configuration with metrics
  - `winners.csv`: Configurations that beat SPY in all periods
  - `best_strategy.yaml`: Ready-to-use config for best strategy
  - `optimization_report.md`: Human-readable summary

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--snapshot` | (required) | Path to data snapshot directory |
| `--output` | `artifacts/optimization` | Output directory for results |
| `--periods` | `3,5,10` | Comma-separated evaluation periods in years |
| `--max-configs` | None | Maximum configs to test (for debugging) |
| `--parallel` | `1` | Number of parallel workers |
| `--cost-bps` | `10.0` | Transaction cost in basis points |
| `--schedules` | all | Filter by schedule (weekly, monthly) |
| `--alpha-types` | all | Filter by alpha type |
| `--verbose/-v` | False | Enable DEBUG logging |
| `--dry-run` | False | Count configs without running |

**Alpha Types Available:**
- `momentum`: Classic price momentum
- `momentum_acceleration`: Short vs long momentum comparison
- `vol_adjusted_momentum`: Volatility-scaled momentum
- `residual_momentum`: Market-adjusted momentum

**What "Beats SPY" Means:**
A strategy beats SPY when it achieves:
1. Positive excess return (strategy return > SPY return)
2. Positive information ratio (risk-adjusted excess return > 0)
3. Consistent across ALL evaluated periods

---

## Typical Workflow

1. **Harmonize universe:**
   ```bash
   python scripts/harmonize_etf_lists.py
   ```

2. **Ingest data:**
   ```bash
   python scripts/ingest_etf_data.py --universe initial_20_etfs --lookback-years 5
   ```

3. **Create snapshot:**
   ```bash
   python scripts/create_snapshot.py --universe initial_20_etfs
   ```

4. **Run backtest:**
   ```bash
   python scripts/run_backtest.py --snapshot data/snapshots/snapshot_5yr_20etfs
   ```

5. **Compare strategies:**
   ```bash
   python scripts/compare_strategies.py \
       --backtest-dirs artifacts/backtests/*/ \
       --output artifacts/comparisons/latest
   ```

6. **Validate strategy robustness (walk-forward):**
   ```bash
   python scripts/walk_forward_test.py \
       --snapshot data/snapshots/snapshot_5yr_20etfs \
       --train-years 2 \
       --test-years 1 \
       --save-plots
   ```

7. **Find best strategy (optimization):**
   ```bash
   python scripts/find_best_strategy.py \
       --snapshot data/snapshots/snapshot_5yr_20etfs \
       --periods 3,5,10
   ```
