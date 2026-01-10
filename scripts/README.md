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
