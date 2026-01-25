# Task Handoff: IMPL-035c - VIX Data Verification & Ingestion

**Task ID:** IMPL-035c
**Parent Task:** IMPL-035 (Regime-Based Strategy Selection System)
**Status:** ready
**Priority:** HIGH
**Type:** Data Infrastructure
**Estimated Effort:** 1-2 hours
**Dependencies:** None (can run in parallel with IMPL-035a)

---

## Summary

Verify that VIX data is available in the existing macro data infrastructure. If not, implement ingestion from FRED. The regime detector needs daily VIX values for volatility state classification.

---

## Deliverables

1. **Verification:** Confirm VIX data exists and is accessible via DataAccessContext
2. **If missing:** Script to ingest VIX from FRED API
3. **Helper function:** `get_vix_as_of(date)` for regime detector use
4. **Tests:** Verify VIX data coverage and accessibility

---

## Step 1: Verification (Do This First)

Before implementing anything, check if VIX already exists:

```bash
# Check if VIX is in macro data
ls -la data/curated/macro/
ls -la data/raw/macro/

# Check existing FRED ingestion
grep -r "VIX" src/quantetf/data/ --include="*.py"
grep -r "VIXCLS" src/quantetf/ --include="*.py"

# Check if macro accessor handles VIX
cat src/quantetf/data/access/macro_accessor.py
```

### Expected Locations

| Location | What to look for |
|----------|------------------|
| `data/curated/macro/VIX.parquet` | Curated VIX data |
| `data/raw/macro/VIX.parquet` | Raw VIX data |
| `src/quantetf/macro/` | Macro data module |
| `scripts/ingest_fred_data.py` | FRED ingestion script |

---

## Step 2a: If VIX Exists

Create a helper to access it easily:

```python
# src/quantetf/regime/indicators.py
"""Helper functions to get regime indicators."""

from typing import Optional
import pandas as pd

from quantetf.data.access import DataAccessContext


class RegimeIndicators:
    """Fetches indicators needed for regime detection."""

    def __init__(self, data_access: DataAccessContext):
        self.data_access = data_access

    def get_spy_data(
        self,
        as_of: pd.Timestamp,
        lookback_days: int = 200,
    ) -> pd.DataFrame:
        """
        Get SPY price and 200-day moving average.

        Returns DataFrame with columns: ['close', 'ma_200']
        """
        # Get enough history for 200MA
        start_date = as_of - pd.Timedelta(days=lookback_days + 50)

        prices = self.data_access.prices.read_prices_as_of(
            as_of=as_of,
            tickers=["SPY"],
        )
        spy_close = prices.xs("Close", level="Price", axis=1)["SPY"]

        # Filter to date range
        spy_close = spy_close[spy_close.index >= start_date]

        # Calculate 200MA
        ma_200 = spy_close.rolling(window=200, min_periods=200).mean()

        return pd.DataFrame({
            "close": spy_close,
            "ma_200": ma_200,
        })

    def get_vix(
        self,
        as_of: pd.Timestamp,
        lookback_days: int = 30,
    ) -> pd.Series:
        """
        Get VIX values up to as_of date.

        Returns Series indexed by date.
        """
        # Use existing macro data accessor
        vix = self.data_access.macro.read_series_as_of(
            series_id="VIXCLS",  # or "VIX" depending on naming
            as_of=as_of,
        )
        start_date = as_of - pd.Timedelta(days=lookback_days)
        return vix[vix.index >= start_date]

    def get_current_indicators(
        self,
        as_of: pd.Timestamp,
    ) -> dict:
        """
        Get all indicators needed for regime detection.

        Returns:
            {
                "spy_price": float,
                "spy_200ma": float,
                "vix": float,
                "as_of": Timestamp,
            }
        """
        spy_data = self.get_spy_data(as_of)
        vix_data = self.get_vix(as_of)

        # Get most recent values at or before as_of
        spy_row = spy_data.loc[:as_of].iloc[-1]
        vix_value = vix_data.loc[:as_of].iloc[-1]

        return {
            "spy_price": spy_row["close"],
            "spy_200ma": spy_row["ma_200"],
            "vix": vix_value,
            "as_of": as_of,
        }
```

---

## Step 2b: If VIX Missing - Implement Ingestion

```python
# scripts/ingest_vix.py
"""Ingest VIX data from FRED."""

import os
from pathlib import Path
import pandas as pd
from fredapi import Fred

DEFAULT_OUTPUT_DIR = Path("data/curated/macro")
FRED_SERIES_ID = "VIXCLS"  # CBOE Volatility Index


def ingest_vix(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_date: str = "2015-01-01",
    api_key: str | None = None,
) -> Path:
    """
    Ingest VIX data from FRED API.

    Args:
        output_dir: Directory to save parquet file
        start_date: Start date for data (YYYY-MM-DD)
        api_key: FRED API key. Uses FRED_API_KEY env var if not provided.

    Returns:
        Path to saved parquet file
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY env var or pass api_key."
        )

    fred = Fred(api_key=api_key)

    print(f"Fetching {FRED_SERIES_ID} from FRED...")
    vix = fred.get_series(FRED_SERIES_ID, observation_start=start_date)

    # Convert to DataFrame
    df = pd.DataFrame({"vix": vix})
    df.index.name = "date"

    # Fill forward missing values (weekends, holidays)
    df = df.ffill()

    # Save to parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "VIX.parquet"
    df.to_parquet(output_path)

    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest VIX data from FRED")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    ingest_vix(
        output_dir=args.output_dir,
        start_date=args.start_date,
    )
```

### Update Macro Accessor (if needed)

If the existing macro accessor doesn't handle VIX, add support:

```python
# In src/quantetf/data/access/macro_accessor.py

def read_vix_as_of(self, as_of: pd.Timestamp) -> pd.Series:
    """Read VIX data up to as_of date."""
    vix_path = self.macro_dir / "VIX.parquet"
    if not vix_path.exists():
        raise FileNotFoundError(
            f"VIX data not found at {vix_path}. "
            "Run scripts/ingest_vix.py to download."
        )

    df = pd.read_parquet(vix_path)
    return df["vix"].loc[:as_of]
```

---

## Test Cases

```python
# tests/regime/test_indicators.py
import pytest
import pandas as pd

from quantetf.regime.indicators import RegimeIndicators


class TestRegimeIndicators:
    """Test indicator fetching for regime detection."""

    @pytest.fixture
    def data_access(self):
        """Create DataAccessContext with test data."""
        # Use existing test fixtures or create mock
        from quantetf.data.access import DataAccessFactory

        return DataAccessFactory.create_context(
            config={"snapshot_path": "data/snapshots/snapshot_20260115_*/data.parquet"},
            enable_caching=True,
        )

    def test_get_spy_data_returns_close_and_ma(self, data_access):
        """SPY data should include close price and 200MA."""
        indicators = RegimeIndicators(data_access)
        spy_data = indicators.get_spy_data(
            as_of=pd.Timestamp("2026-01-20"),
            lookback_days=200,
        )

        assert "close" in spy_data.columns
        assert "ma_200" in spy_data.columns
        assert len(spy_data) > 0

    def test_get_vix_returns_series(self, data_access):
        """VIX should return a pandas Series."""
        indicators = RegimeIndicators(data_access)
        vix = indicators.get_vix(
            as_of=pd.Timestamp("2026-01-20"),
            lookback_days=30,
        )

        assert isinstance(vix, pd.Series)
        assert len(vix) > 0
        # VIX should be positive and reasonable
        assert vix.min() > 0
        assert vix.max() < 100

    def test_get_current_indicators(self, data_access):
        """Current indicators should return all required values."""
        indicators = RegimeIndicators(data_access)
        current = indicators.get_current_indicators(
            as_of=pd.Timestamp("2026-01-20"),
        )

        assert "spy_price" in current
        assert "spy_200ma" in current
        assert "vix" in current
        assert current["spy_price"] > 0
        assert current["spy_200ma"] > 0
        assert 0 < current["vix"] < 100

    def test_no_lookahead_in_indicators(self, data_access):
        """Indicators should not use future data."""
        indicators = RegimeIndicators(data_access)

        # Get indicators for historical date
        historical = indicators.get_current_indicators(
            as_of=pd.Timestamp("2025-06-15"),
        )

        # Should not contain data after as_of
        assert historical["as_of"] <= pd.Timestamp("2025-06-15")
```

---

## Acceptance Criteria

- [ ] VIX data is available via DataAccessContext (verify or implement)
- [ ] `RegimeIndicators` class provides SPY price, 200MA, and VIX
- [ ] No lookahead bias (all data is as-of date)
- [ ] VIX data covers required historical period (2016-2026)
- [ ] Tests verify data accessibility and correctness
- [ ] If ingestion needed, script works with FRED API

---

## Notes for Implementer

1. **Check first:** Don't duplicate work if VIX already exists
2. **FRED API key:** May need to be configured (check .env or secrets)
3. **Data alignment:** VIX has trading days only; handle weekends/holidays
4. **Point-in-time:** All data access must respect as_of date

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Coding Agent
