# Handoff: INFRA-002 - Data Quality Monitoring

**Task ID:** INFRA-002
**Status:** ready
**Priority:** MEDIUM (Infrastructure - Independent)
**Estimated Time:** 2-3 hours
**Dependencies:** None
**Assigned to:** [Available for pickup]

---

## Context & Motivation

### What are we building?

A comprehensive data quality monitoring system that validates ETF price data and detects anomalies. This includes:

1. A script (`scripts/data_health_check.py`) to run quality checks on snapshots
2. A module (`src/quantetf/data/quality.py`) with reusable quality check functions
3. Automated quality scoring and reporting

### Why does this matter?

**Garbage in, garbage out.** Bad data destroys backtests and can lead to catastrophic trading decisions. Common issues include:

- Missing data (gaps, NaN values)
- Price spikes from bad ticks or stock splits
- Stale data (not updated recently)
- Volume anomalies (zeros, impossibly high values)
- Duplicate tickers (same ETF listed twice)
- Delisted ETFs still in dataset

**This task prevents:** Bad backtests, false alpha signals, production failures

**Use cases:**
- Pre-backtest validation: "Is this snapshot clean enough to use?"
- Production monitoring: "Is our data feed working?"
- Data provider comparison: "Which source has better quality?"

---

## Current State

### Existing Data Structure

**Snapshot format** (from Phase 1):
```
data/snapshots/snapshot_5yr_20etfs/
├── prices.parquet          # MultiIndex (date, ticker)
├── metadata.yaml           # Snapshot metadata
└── universe.txt            # List of tickers
```

**Parquet schema:**
```python
# MultiIndex DataFrame
# Index: (date, ticker)
# Columns: open, high, low, close, volume, adjusted_close
```

Example:
```python
import pandas as pd
df = pd.read_parquet('data/snapshots/snapshot_5yr_20etfs/prices.parquet')
# DatetimeIndex × Index (date, ticker)
# Columns: open, high, low, close, volume
```

### Inspect Existing Data

```bash
# View snapshot structure
ls -la data/snapshots/snapshot_5yr_20etfs/

# Load and inspect
python -c "
import pandas as pd
df = pd.read_parquet('data/snapshots/snapshot_5yr_20etfs/prices.parquet')
print(df.head(20))
print(df.info())
print(df.describe())
"
```

---

## Task Specification

### Quality Checks to Implement

#### 1. Missing Data Summary

**What:** Calculate % of NaN values for each ticker and each column

**Output:**
```
Ticker    Close_NaN%   Volume_NaN%   Total_Days   Missing_Days
SPY       0.0%         0.0%          1260         0
QQQ       0.2%         0.0%          1260         3
ARKK      5.1%         2.3%          1260         64
```

**Thresholds:**
- < 1% missing: GOOD
- 1-5% missing: WARNING
- > 5% missing: POOR

**Function signature:**
```python
def check_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing data across tickers.

    Args:
        df: MultiIndex DataFrame (date, ticker) with OHLCV columns

    Returns:
        DataFrame with missing data summary per ticker
        Columns: ticker, close_nan_pct, volume_nan_pct, total_days, missing_days

    Example:
        >>> df = pd.read_parquet('snapshot.parquet')
        >>> missing_summary = check_missing_data(df)
        >>> print(missing_summary[missing_summary['close_nan_pct'] > 5.0])
    """
```

---

#### 2. Price Spike Detection

**What:** Detect single-day price moves > 10% (potential bad data)

**Logic:**
```python
# Calculate daily returns
returns = prices.pct_change()

# Flag spikes
spikes = returns.abs() > 0.10  # 10% threshold
```

**Edge cases to handle:**
- Legitimate moves (COVID crash had 10%+ days) - don't flag as errors
- Stock splits (should be in adjusted_close already)
- First day of ticker (NaN return) - ignore

**Output:**
```
Date          Ticker   Price_Change   Likely_Issue
2020-03-16    SPY      -12.0%        Legitimate (COVID)
2022-01-03    ARKK     +45.3%        Suspicious spike ⚠
```

**Function signature:**
```python
def detect_price_spikes(
    df: pd.DataFrame,
    threshold: float = 0.10,
    column: str = 'close'
) -> pd.DataFrame:
    """Detect suspicious single-day price spikes.

    Args:
        df: MultiIndex DataFrame (date, ticker)
        threshold: Return threshold for spike (default 0.10 = 10%)
        column: Price column to check (default 'close')

    Returns:
        DataFrame of detected spikes with date, ticker, return, price_before, price_after

    Example:
        >>> spikes = detect_price_spikes(df, threshold=0.15)
        >>> print(f"Found {len(spikes)} suspicious spikes")
    """
```

**Threshold recommendations:**
- 10% = flag for review
- 20% = almost certainly bad data (unless known event)

---

#### 3. Stale Data Detection

**What:** Identify tickers with gaps > 5 business days (no data)

**Logic:**
```python
# For each ticker, find date gaps
# Business days between consecutive dates
gaps = date_diff - expected_diff

if gaps > 5:
    flag_as_stale
```

**Output:**
```
Ticker   Gap_Start     Gap_End       Gap_Days   Status
ARKK     2021-01-15    2021-02-03    14         STALE ⚠
IWM      2023-12-20    2023-12-27    5          OK (holiday)
```

**Function signature:**
```python
def detect_stale_data(
    df: pd.DataFrame,
    max_gap_days: int = 5
) -> pd.DataFrame:
    """Detect gaps in price data (stale/missing periods).

    Args:
        df: MultiIndex DataFrame (date, ticker)
        max_gap_days: Maximum acceptable gap in business days

    Returns:
        DataFrame of gaps exceeding threshold with ticker, gap_start, gap_end, gap_days

    Example:
        >>> gaps = detect_stale_data(df, max_gap_days=7)
        >>> critical_gaps = gaps[gaps['gap_days'] > 10]
    """
```

**Note:** Use business days, not calendar days. `pd.bdate_range()` may be helpful.

---

#### 4. Volume Anomalies

**What:** Detect volume issues (zeros, extreme values)

**Checks:**
1. **Zero volume days** (suspicious for liquid ETFs like SPY)
2. **Extreme volume** (> 10x median volume for that ticker)
3. **Constant volume** (same value many days in a row - sign of stale data)

**Output:**
```
Ticker   Issue_Type        Date         Volume      Median_Volume
SPY      Zero_Volume       2022-05-03   0           75000000
ARKK     Extreme_Volume    2021-02-12   550000000   5000000
```

**Function signature:**
```python
def detect_volume_anomalies(
    df: pd.DataFrame,
    zero_volume_flag: bool = True,
    extreme_multiple: float = 10.0
) -> pd.DataFrame:
    """Detect volume anomalies (zeros, extremes, constant values).

    Args:
        df: MultiIndex DataFrame (date, ticker) with 'volume' column
        zero_volume_flag: Whether to flag zero volume days
        extreme_multiple: Flag volume > X times median as extreme

    Returns:
        DataFrame of volume anomalies with ticker, date, volume, issue_type

    Example:
        >>> volume_issues = detect_volume_anomalies(df)
        >>> zero_vol = volume_issues[volume_issues['issue_type'] == 'zero_volume']
    """
```

---

#### 5. Correlation Matrix (Duplicate Detection)

**What:** Compute correlation matrix to detect duplicate or highly correlated tickers

**Why:** Sometimes same ETF is included twice under different tickers, or two tickers track same index

**Logic:**
```python
# Pivot to wide format: dates × tickers
prices_wide = df['close'].unstack()

# Compute correlation
corr_matrix = prices_wide.corr()

# Flag perfect or near-perfect correlations (> 0.99)
high_corr = corr_matrix[(corr_matrix > 0.99) & (corr_matrix < 1.0)]
```

**Output:**
```
Ticker_A   Ticker_B   Correlation   Likely_Duplicate
SPY        VOO        0.998         Yes (both track S&P 500)
QQQ        QQQM       0.999         Yes (same index)
```

**Function signature:**
```python
def detect_duplicate_tickers(
    df: pd.DataFrame,
    correlation_threshold: float = 0.99
) -> pd.DataFrame:
    """Detect duplicate or highly correlated tickers.

    Args:
        df: MultiIndex DataFrame (date, ticker) with 'close' column
        correlation_threshold: Correlation above which to flag (default 0.99)

    Returns:
        DataFrame of ticker pairs with correlation > threshold

    Example:
        >>> duplicates = detect_duplicate_tickers(df)
        >>> print(f"Found {len(duplicates)} potential duplicates")
    """
```

---

#### 6. Delisting Detection

**What:** Identify tickers that stop reporting data (potentially delisted)

**Logic:**
```python
# Find last date for each ticker
last_dates = df.groupby('ticker').apply(lambda x: x.index.get_level_values('date').max())

# Compare to end of dataset
dataset_end = df.index.get_level_values('date').max()

gap = (dataset_end - last_dates).days

if gap > 30:
    likely_delisted
```

**Output:**
```
Ticker   Last_Date     Dataset_End   Gap_Days   Status
ARKK     2025-12-15    2026-01-11    27         Active
TQQQ     2024-06-20    2026-01-11    570        Likely Delisted ⚠
```

**Function signature:**
```python
def detect_delisted_tickers(
    df: pd.DataFrame,
    gap_threshold_days: int = 30
) -> pd.DataFrame:
    """Detect tickers that may have been delisted (stopped reporting).

    Args:
        df: MultiIndex DataFrame (date, ticker)
        gap_threshold_days: Days from dataset end to consider delisted

    Returns:
        DataFrame with ticker, last_date, gap_days, status

    Example:
        >>> delisted = detect_delisted_tickers(df)
        >>> delisted_only = delisted[delisted['status'] == 'Delisted']
    """
```

---

### Quality Scoring Function

Aggregate all checks into a single quality score (0-100) per ticker.

```python
def calculate_quality_score(
    df: pd.DataFrame,
    ticker: str
) -> dict:
    """Calculate overall quality score for a ticker.

    Scoring:
    - Missing data: -10 points per 1% missing (max -30)
    - Price spikes: -5 points per spike (max -20)
    - Volume issues: -5 points per issue (max -20)
    - Stale data: -10 points per gap > 5 days (max -30)
    - Base score: 100

    Args:
        df: MultiIndex DataFrame (date, ticker)
        ticker: Ticker symbol to score

    Returns:
        Dict with score (0-100) and breakdown by category

    Example:
        >>> score = calculate_quality_score(df, 'SPY')
        >>> print(f"SPY quality score: {score['total']}/100")
        >>> print(score['breakdown'])
    """
```

**Score interpretation:**
- 90-100: Excellent
- 75-89: Good
- 60-74: Fair (usable with caution)
- < 60: Poor (investigate before using)

---

### Quality Report Generation

Main script: `scripts/data_health_check.py`

**CLI Interface:**
```bash
# Run on snapshot
python scripts/data_health_check.py --snapshot data/snapshots/snapshot_5yr_20etfs

# Output location
--output artifacts/data_quality/quality_report_20260111.html

# Options
--format [text|html|json]
--threshold 75  # Only show tickers below this score
```

**Output format (text):**
```
=== Data Quality Report ===
Snapshot: snapshot_5yr_20etfs
Generated: 2026-01-11 10:30:00
Tickers analyzed: 20

--- Overall Summary ---
Average quality score: 87/100
Tickers with score > 90: 15 (75%)
Tickers with score < 75: 2 (10%)

--- Issues Found ---
Missing Data: 3 tickers with > 1% missing
Price Spikes: 5 suspicious spikes detected
Volume Anomalies: 8 zero-volume days
Stale Data: 1 ticker with gap > 5 days
Delisted: 0 tickers

--- Ticker Scores ---
Ticker   Score   Grade   Issues
SPY      98      A       None
QQQ      95      A       1 price spike
ARKK     72      C       5% missing, 2 spikes, 1 gap
...

--- Detailed Issues ---
[Expand each check with full DataFrame output]

--- Recommendations ---
- Remove ticker ARKK (score 72) or fill missing data
- Investigate price spike in QQQ on 2022-01-03
- Overall data quality is GOOD (avg 87/100)
```

---

## Implementation Guidelines

### File Structure

**Create new module:**
```python
# src/quantetf/data/quality.py
"""Data quality checks for ETF price data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def check_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """..."""
    pass

def detect_price_spikes(...) -> pd.DataFrame:
    """..."""
    pass

# ... other checks ...

def calculate_quality_score(df: pd.DataFrame, ticker: str) -> Dict:
    """..."""
    pass

def generate_quality_report(df: pd.DataFrame, output_path: str = None) -> str:
    """Generate comprehensive quality report.

    Runs all checks and compiles results.

    Args:
        df: MultiIndex DataFrame to analyze
        output_path: Optional path to save report (HTML or text)

    Returns:
        Report as string
    """
    pass
```

**Create script:**
```python
# scripts/data_health_check.py
"""Script to run data quality checks on snapshots."""

import argparse
from pathlib import Path
import pandas as pd
from quantetf.data.quality import generate_quality_report

def main():
    parser = argparse.ArgumentParser(description='Run data quality checks')
    parser.add_argument('--snapshot', required=True, help='Path to snapshot directory')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--format', choices=['text', 'html', 'json'], default='text')
    parser.add_argument('--threshold', type=int, default=0, help='Only show tickers below this score')

    args = parser.parse_args()

    # Load snapshot
    snapshot_path = Path(args.snapshot)
    df = pd.read_parquet(snapshot_path / 'prices.parquet')

    # Generate report
    report = generate_quality_report(df, output_path=args.output, format=args.format)

    # Print to console
    print(report)

if __name__ == '__main__':
    main()
```

---

## Testing Strategy

Create `tests/test_data_quality.py`:

```python
"""Tests for data quality checks."""

import pandas as pd
import numpy as np
import pytest
from quantetf.data.quality import (
    check_missing_data,
    detect_price_spikes,
    detect_stale_data,
    detect_volume_anomalies,
    detect_duplicate_tickers,
    detect_delisted_tickers,
    calculate_quality_score,
)

@pytest.fixture
def clean_data():
    """Generate clean synthetic dataset."""
    dates = pd.bdate_range('2024-01-01', '2024-12-31')
    tickers = ['SPY', 'QQQ', 'IWM']

    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'date': date,
                'ticker': ticker,
                'close': 100 + np.random.randn(),
                'volume': 1000000 + np.random.randint(0, 100000),
            })

    df = pd.DataFrame(data)
    df = df.set_index(['date', 'ticker'])
    return df

@pytest.fixture
def dirty_data():
    """Generate dataset with known issues."""
    dates = pd.bdate_range('2024-01-01', '2024-12-31')

    data = []
    for i, date in enumerate(dates):
        # SPY: clean
        data.append({'date': date, 'ticker': 'SPY', 'close': 100 + i*0.1, 'volume': 1000000})

        # QQQ: has price spike on day 50
        if i == 50:
            price = 200  # Spike
        else:
            price = 100 + i*0.1
        data.append({'date': date, 'ticker': 'QQQ', 'close': price, 'volume': 1000000})

        # ARKK: missing data on days 100-110
        if 100 <= i <= 110:
            continue
        data.append({'date': date, 'ticker': 'ARKK', 'close': 100 + i*0.1, 'volume': 1000000})

    df = pd.DataFrame(data).set_index(['date', 'ticker'])
    return df

class TestMissingData:
    def test_clean_data_no_missing(self, clean_data):
        """Test that clean data shows 0% missing."""
        result = check_missing_data(clean_data)
        assert (result['close_nan_pct'] == 0.0).all()

    def test_dirty_data_detects_missing(self, dirty_data):
        """Test detection of missing data in ARKK."""
        result = check_missing_data(dirty_data)
        arkk_missing = result[result['ticker'] == 'ARKK']['close_nan_pct'].iloc[0]
        assert arkk_missing > 0  # Should detect missing days

class TestPriceSpikes:
    def test_detect_spike(self, dirty_data):
        """Test detection of QQQ price spike."""
        spikes = detect_price_spikes(dirty_data, threshold=0.10)
        assert len(spikes) > 0
        assert 'QQQ' in spikes['ticker'].values

    def test_no_spikes_in_clean_data(self, clean_data):
        """Test no false positives on clean data."""
        spikes = detect_price_spikes(clean_data, threshold=0.10)
        assert len(spikes) == 0

class TestQualityScore:
    def test_clean_data_high_score(self, clean_data):
        """Test that clean data gets high score."""
        score = calculate_quality_score(clean_data, 'SPY')
        assert score['total'] >= 95

    def test_dirty_data_lower_score(self, dirty_data):
        """Test that data with issues gets lower score."""
        score = calculate_quality_score(dirty_data, 'ARKK')
        assert score['total'] < 90  # Missing data should lower score

# Add tests for other checks...
```

**Test coverage target:** 12+ tests covering all checks

---

## Acceptance Criteria

- [ ] All 6 quality check functions implemented:
  - [ ] check_missing_data()
  - [ ] detect_price_spikes()
  - [ ] detect_stale_data()
  - [ ] detect_volume_anomalies()
  - [ ] detect_duplicate_tickers()
  - [ ] detect_delisted_tickers()
- [ ] Quality scoring function implemented
- [ ] Report generation function implemented
- [ ] CLI script `scripts/data_health_check.py` works
- [ ] Can run on real snapshot: `python scripts/data_health_check.py --snapshot data/snapshots/snapshot_5yr_20etfs`
- [ ] 12+ tests pass
- [ ] Generates readable report (text or HTML)
- [ ] Documentation in docstrings

---

## Dependencies

**None** - Independent task

**Blocks:** Nothing critical, but useful for all data-related work

---

## Inputs

- Real snapshot: `data/snapshots/snapshot_5yr_20etfs/prices.parquet`
- Existing code patterns in `src/quantetf/data/`

---

## Outputs

**Files to create:**
1. `src/quantetf/data/quality.py` - Quality check functions
2. `scripts/data_health_check.py` - CLI script
3. `tests/test_data_quality.py` - Test suite
4. `artifacts/data_quality/quality_report_*.html` - Generated reports

---

## Examples

### Example CLI Usage

```bash
# Run basic quality check
python scripts/data_health_check.py \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --format text

# Generate HTML report
python scripts/data_health_check.py \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --output artifacts/data_quality/report_20260111.html \
  --format html

# Show only problematic tickers
python scripts/data_health_check.py \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --threshold 75
```

### Example Programmatic Usage

```python
import pandas as pd
from quantetf.data.quality import *

# Load snapshot
df = pd.read_parquet('data/snapshots/snapshot_5yr_20etfs/prices.parquet')

# Run individual checks
missing = check_missing_data(df)
spikes = detect_price_spikes(df)
gaps = detect_stale_data(df)

# Calculate scores
scores = {ticker: calculate_quality_score(df, ticker)
          for ticker in df.index.get_level_values('ticker').unique()}

# Print summary
for ticker, score in scores.items():
    print(f"{ticker}: {score['total']}/100")
```

---

## Success Criteria

✅ All 6 check functions + scoring + reporting implemented
✅ 12+ tests passing
✅ CLI script functional on real data
✅ Generates readable, actionable reports
✅ Code follows CLAUDE_CONTEXT.md standards

**Expected time:** 2-3 hours

---

**Ready to begin!** This is an independent task that can run in parallel with ANALYSIS-001 and ANALYSIS-007.
