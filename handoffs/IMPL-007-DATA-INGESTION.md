# Handoff: IMPL-007 - FRED and Additional Data Ingestion

**Task ID:** IMPL-007
**Status:** ready
**Priority:** MEDIUM
**Estimated Effort:** Medium (2-3 files, ~300 lines)
**Dependencies:** None
**Assigned to:** Coding Agent

---

## Context & Motivation

### What are we building?

Data ingestion scripts for additional free data sources:

1. **FRED Economic Indicators** - Macro signals for regime detection
2. **Enhanced Stooq Integration** - Additional ETF coverage
3. **Data Quality Validation** - Ensure data integrity

### Why does this matter?

Research (RESEARCH-001) identified that regime detection is key, but requires more than price data:
- VIX for volatility regime
- Treasury yields for risk-on/risk-off
- Credit spreads for stress signals
- Economic indicators for macro regime

FRED provides all of this for free.

---

## Data Sources Summary

### 1. FRED (Federal Reserve Economic Data)

**URL:** https://fred.stlouisfed.org/
**API:** Free with registration
**Python:** `fredapi` package

| Indicator | FRED Code | Use Case |
|-----------|-----------|----------|
| VIX | VIXCLS | Volatility regime |
| 10Y Treasury | DGS10 | Risk-free rate, duration |
| 2Y Treasury | DGS2 | Yield curve |
| 2Y-10Y Spread | T10Y2Y | Recession indicator |
| High Yield Spread | BAMLH0A0HYM2 | Credit stress |
| Fed Funds Rate | FEDFUNDS | Monetary policy |
| Unemployment | UNRATE | Economic health |
| CPI YoY | CPIAUCSL | Inflation |

### 2. Existing Sources

| Source | Package | Current Use |
|--------|---------|-------------|
| Stooq | `scripts/download_ohlcv_stooq.py` | ETF OHLCV |
| Yahoo | `yfinance` | Backup source |

---

## Specification 1: FRED Data Ingestion

### Implementation

**File:** `scripts/ingest_fred_data.py`

```python
#!/usr/bin/env python3
"""Ingest macroeconomic data from FRED.

Downloads key economic indicators for regime detection and macro analysis.

Usage:
    python scripts/ingest_fred_data.py --start-date 2015-01-01
    python scripts/ingest_fred_data.py --api-key YOUR_KEY --indicators VIX,DGS10

Environment:
    FRED_API_KEY: Your FRED API key (get free at https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


# Default indicators to download
DEFAULT_INDICATORS: Dict[str, str] = {
    # Volatility
    'VIX': 'VIXCLS',

    # Interest Rates
    'TREASURY_10Y': 'DGS10',
    'TREASURY_2Y': 'DGS2',
    'TREASURY_3M': 'DTB3',
    'TREASURY_SPREAD_10Y2Y': 'T10Y2Y',
    'FED_FUNDS': 'FEDFUNDS',

    # Credit
    'HY_SPREAD': 'BAMLH0A0HYM2',  # ICE BofA High Yield Spread
    'IG_SPREAD': 'BAMLC0A0CM',    # Investment Grade Spread

    # Economic
    'UNEMPLOYMENT': 'UNRATE',
    'CPI': 'CPIAUCSL',
    'INDUSTRIAL_PROD': 'INDPRO',

    # Market
    'SP500': 'SP500',  # S&P 500 index
}


def get_fred_api_key() -> str:
    """Get FRED API key from environment or prompt."""
    key = os.environ.get('FRED_API_KEY')
    if not key:
        raise ValueError(
            "FRED_API_KEY not found in environment. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set FRED_API_KEY environment variable."
        )
    return key


def download_fred_series(
    api_key: str,
    series_id: str,
    start_date: str,
    end_date: Optional[str] = None
) -> pd.Series:
    """Download a single FRED series.

    Args:
        api_key: FRED API key
        series_id: FRED series code (e.g., 'VIXCLS')
        start_date: Start date (YYYY-MM-DD)
        end_date: Optional end date

    Returns:
        pandas Series with data
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Please install fredapi: pip install fredapi")

    fred = Fred(api_key=api_key)

    kwargs = {'observation_start': start_date}
    if end_date:
        kwargs['observation_end'] = end_date

    data = fred.get_series(series_id, **kwargs)
    data.name = series_id

    return data


def download_all_indicators(
    api_key: str,
    indicators: Dict[str, str],
    start_date: str,
    end_date: Optional[str] = None,
    output_dir: Path = Path('data/raw/macro')
) -> Dict[str, pd.Series]:
    """Download all specified indicators.

    Args:
        api_key: FRED API key
        indicators: Dict mapping name -> FRED code
        start_date: Start date
        end_date: Optional end date
        output_dir: Directory to save files

    Returns:
        Dict of downloaded series
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    failed = []

    for name, code in indicators.items():
        print(f"Downloading {name} ({code})...", end=' ')
        try:
            data = download_fred_series(api_key, code, start_date, end_date)
            results[name] = data

            # Save to parquet
            df = data.to_frame()
            df.to_parquet(output_dir / f"{name}.parquet")
            print(f"OK ({len(data)} observations)")

        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)

    # Save metadata
    metadata = {
        'download_date': datetime.now().isoformat(),
        'start_date': start_date,
        'end_date': end_date or 'latest',
        'indicators': {name: code for name, code in indicators.items()},
        'successful': list(results.keys()),
        'failed': failed,
    }

    with open(output_dir / 'manifest.yaml', 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"\nDownloaded {len(results)}/{len(indicators)} indicators")
    print(f"Saved to {output_dir}")

    return results


def create_combined_macro_dataset(
    data_dir: Path = Path('data/raw/macro'),
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Combine all macro indicators into single DataFrame.

    Args:
        data_dir: Directory with individual parquet files
        output_path: Optional path to save combined file

    Returns:
        Combined DataFrame with all indicators
    """
    dfs = []
    for parquet_file in data_dir.glob('*.parquet'):
        if parquet_file.name == 'combined.parquet':
            continue
        df = pd.read_parquet(parquet_file)
        df.columns = [parquet_file.stem]
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No parquet files found in {data_dir}")

    combined = pd.concat(dfs, axis=1)
    combined = combined.sort_index()

    if output_path:
        combined.to_parquet(output_path)
        print(f"Combined dataset saved to {output_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Download FRED economic indicators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all default indicators
    python scripts/ingest_fred_data.py --start-date 2015-01-01

    # Download specific indicators
    python scripts/ingest_fred_data.py --start-date 2015-01-01 --indicators VIX,DGS10,T10Y2Y

    # Use specific API key
    FRED_API_KEY=your_key python scripts/ingest_fred_data.py --start-date 2015-01-01
        """
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2015-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), default is latest'
    )
    parser.add_argument(
        '--indicators',
        type=str,
        default=None,
        help='Comma-separated list of indicator names to download'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/macro',
        help='Output directory'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='FRED API key (or set FRED_API_KEY env var)'
    )
    parser.add_argument(
        '--list-indicators',
        action='store_true',
        help='List available indicators and exit'
    )
    parser.add_argument(
        '--combine',
        action='store_true',
        help='Also create combined dataset'
    )

    args = parser.parse_args()

    if args.list_indicators:
        print("Available indicators:")
        print("-" * 50)
        for name, code in DEFAULT_INDICATORS.items():
            print(f"  {name:25s} -> {code}")
        return

    # Get API key
    api_key = args.api_key or get_fred_api_key()

    # Select indicators
    if args.indicators:
        selected = args.indicators.split(',')
        indicators = {k: v for k, v in DEFAULT_INDICATORS.items() if k in selected}
        if not indicators:
            print(f"Warning: None of {selected} found in default indicators")
            # Try treating as FRED codes directly
            indicators = {s: s for s in selected}
    else:
        indicators = DEFAULT_INDICATORS

    # Download
    output_dir = Path(args.output_dir)
    results = download_all_indicators(
        api_key=api_key,
        indicators=indicators,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=output_dir
    )

    # Optionally combine
    if args.combine and results:
        create_combined_macro_dataset(
            data_dir=output_dir,
            output_path=output_dir / 'combined.parquet'
        )


if __name__ == '__main__':
    main()
```

---

## Specification 2: Macro Data Loader

**File:** `src/quantetf/data/macro_loader.py`

```python
"""Macro data loader for regime detection.

Loads FRED data and provides regime signals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class MacroDataLoader:
    """Load and process macroeconomic data.

    Attributes:
        data_dir: Directory containing macro parquet files
    """
    data_dir: Path = Path('data/raw/macro')

    def load_indicator(self, name: str) -> pd.Series:
        """Load a single indicator.

        Args:
            name: Indicator name (e.g., 'VIX', 'TREASURY_10Y')

        Returns:
            pandas Series with indicator values
        """
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Indicator {name} not found at {path}")

        df = pd.read_parquet(path)
        return df.iloc[:, 0]  # Return first column as Series

    def load_all(self) -> pd.DataFrame:
        """Load all available indicators.

        Returns:
            DataFrame with all indicators as columns
        """
        path = self.data_dir / 'combined.parquet'
        if path.exists():
            return pd.read_parquet(path)

        # Combine individual files
        dfs = []
        for parquet_file in self.data_dir.glob('*.parquet'):
            if parquet_file.name in ['combined.parquet', 'manifest.yaml']:
                continue
            df = pd.read_parquet(parquet_file)
            df.columns = [parquet_file.stem]
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No data found in {self.data_dir}")

        return pd.concat(dfs, axis=1).sort_index()

    def get_vix(self, date: Optional[str] = None) -> float:
        """Get VIX value for date.

        Args:
            date: Optional date (returns latest if None)

        Returns:
            VIX value
        """
        vix = self.load_indicator('VIX')
        if date:
            vix = vix.loc[:date]
        return vix.iloc[-1]

    def get_yield_curve_spread(self, date: Optional[str] = None) -> float:
        """Get 10Y-2Y Treasury spread.

        Args:
            date: Optional date

        Returns:
            Spread in percentage points
        """
        spread = self.load_indicator('TREASURY_SPREAD_10Y2Y')
        if date:
            spread = spread.loc[:date]
        return spread.iloc[-1]

    def is_high_vol_regime(
        self,
        date: Optional[str] = None,
        threshold: float = 25.0
    ) -> bool:
        """Check if VIX indicates high volatility.

        Args:
            date: Date to check
            threshold: VIX threshold (default 25)

        Returns:
            True if VIX > threshold
        """
        return self.get_vix(date) > threshold

    def is_yield_curve_inverted(self, date: Optional[str] = None) -> bool:
        """Check if yield curve is inverted (recession signal).

        Args:
            date: Date to check

        Returns:
            True if 10Y-2Y spread is negative
        """
        return self.get_yield_curve_spread(date) < 0


class RegimeDetector:
    """Detect market regimes using macro data.

    Combines multiple signals to classify regime.
    """

    def __init__(self, macro_loader: MacroDataLoader):
        self.macro = macro_loader

    def detect_regime(self, date: str) -> str:
        """Detect regime for a given date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Regime string: 'RISK_ON', 'RISK_OFF', 'HIGH_VOL', 'RECESSION_WARNING'
        """
        try:
            vix = self.macro.get_vix(date)
            spread = self.macro.get_yield_curve_spread(date)
        except Exception:
            return 'UNKNOWN'

        # Inverted yield curve = recession warning
        if spread < 0:
            return 'RECESSION_WARNING'

        # High VIX = high volatility regime
        if vix > 30:
            return 'HIGH_VOL'
        elif vix > 20:
            return 'ELEVATED_VOL'

        # Normal conditions
        return 'RISK_ON'
```

---

## Specification 3: Requirements Update

**Update:** `requirements.txt`

Add:
```
fredapi>=0.5.1
```

---

## Directory Structure

After implementation:

```
scripts/
├── ingest_fred_data.py      # NEW

src/quantetf/data/
├── __init__.py
├── snapshot.py
├── macro_loader.py          # NEW

data/
├── raw/
│   ├── prices/              # Existing
│   └── macro/               # NEW
│       ├── VIX.parquet
│       ├── TREASURY_10Y.parquet
│       ├── TREASURY_SPREAD_10Y2Y.parquet
│       ├── HY_SPREAD.parquet
│       ├── combined.parquet
│       └── manifest.yaml
```

---

## Usage Examples

### Download FRED Data

```bash
# Set API key (get free at https://fred.stlouisfed.org/docs/api/api_key.html)
export FRED_API_KEY=your_api_key_here

# Download all default indicators
python scripts/ingest_fred_data.py --start-date 2015-01-01 --combine

# Download specific indicators
python scripts/ingest_fred_data.py --start-date 2015-01-01 --indicators VIX,DGS10,T10Y2Y

# List available indicators
python scripts/ingest_fred_data.py --list-indicators
```

### Use in Code

```python
from quantetf.data.macro_loader import MacroDataLoader, RegimeDetector

# Load macro data
macro = MacroDataLoader()

# Get specific indicator
vix = macro.load_indicator('VIX')
print(f"Current VIX: {vix.iloc[-1]:.1f}")

# Check regime
detector = RegimeDetector(macro)
regime = detector.detect_regime('2024-01-15')
print(f"Current regime: {regime}")

# Use in alpha model
if macro.is_high_vol_regime():
    print("High volatility - use defensive strategy")
```

### Integration with Alpha Models

```python
from quantetf.alpha import TrendFilteredMomentum
from quantetf.data.macro_loader import RegimeDetector

class MacroAwareMomentum:
    """Momentum strategy that considers macro regime."""

    def __init__(self):
        self.momentum = TrendFilteredMomentum()
        self.regime_detector = RegimeDetector(MacroDataLoader())

    def score(self, prices, date):
        regime = self.regime_detector.detect_regime(date)

        if regime == 'RECESSION_WARNING':
            # Very defensive
            return self.momentum._defensive_scores(prices, None)
        elif regime == 'HIGH_VOL':
            # Reduce exposure
            scores = self.momentum.score(prices, date)
            return scores * 0.5  # Half position
        else:
            return self.momentum.score(prices, date)
```

---

## Acceptance Criteria

- [ ] `scripts/ingest_fred_data.py` implemented with:
  - [ ] Command-line interface
  - [ ] Support for all default indicators
  - [ ] Manifest file generation
  - [ ] Combined dataset creation
- [ ] `src/quantetf/data/macro_loader.py` implemented with:
  - [ ] MacroDataLoader class
  - [ ] RegimeDetector class
  - [ ] Helper methods (is_high_vol_regime, etc.)
- [ ] `fredapi` added to requirements.txt
- [ ] Data directory structure documented
- [ ] Integration example working

---

## Testing

```bash
# Test download (requires API key)
FRED_API_KEY=your_key python scripts/ingest_fred_data.py --start-date 2020-01-01 --indicators VIX

# Test loader
python -c "
from quantetf.data.macro_loader import MacroDataLoader
loader = MacroDataLoader()
print(loader.load_indicator('VIX').tail())
"
```

---

## Notes for Coding Agent

1. **API Key Security:** Never commit API keys. Use environment variables.
2. **Rate Limits:** FRED has generous limits, but add sleep between requests if needed.
3. **Data Gaps:** Some indicators have gaps (weekends, holidays). Handle with forward-fill.
4. **Timezone:** FRED data is typically daily. Align with price data dates.

---

## Additional Free Data Sources (Reference)

For future expansion:

| Source | Data | Python Package |
|--------|------|----------------|
| Yahoo Finance | OHLCV, fundamentals | `yfinance` |
| Alpha Vantage | OHLCV, technicals | `alpha_vantage` |
| Tiingo | OHLCV, news | `tiingo` |
| Finnhub | Real-time, news | `finnhub-python` |
| Polygon.io | Market data | `polygon-api-client` |

Most have free tiers with rate limits. FRED is the most generous for macro data.

---

**Ready to implement!**
