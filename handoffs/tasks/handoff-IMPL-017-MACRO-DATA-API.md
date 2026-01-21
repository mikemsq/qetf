# Task Handoff: IMPL-017 - Enhanced Macro Data API

**Task ID:** IMPL-017
**Status:** ready
**Priority:** high
**Estimated Effort:** 4-6 hours
**Dependencies:** None (can run parallel with IMPL-015, IMPL-016)

---

## Quick Context

You are enhancing the **MacroDataLoader** API to expose all available macro indicators for use in regime detection and alpha models.

Currently:
- 12 macro indicators exist in `data/raw/macro/` (VIX, rates, credit spreads, economic data)
- Only `get_vix()` and `get_yield_curve_spread()` are exposed
- Credit spreads, Fed Funds, CPI, unemployment, industrial production are unused

**Goal:** Create a complete API so alpha models and regime detectors can access any macro indicator as-of any date.

---

## What You Need to Know

### Available Macro Data

Files in `data/raw/macro/`:
```
VIX.parquet           - CBOE Volatility Index
DGS3MO.parquet        - 3-Month Treasury Yield
DGS2.parquet          - 2-Year Treasury Yield
DGS10.parquet         - 10-Year Treasury Yield
T10Y2Y.parquet        - 10Y-2Y Spread (yield curve)
T10Y3M.parquet        - 10Y-3M Spread
HY_SPREAD.parquet     - High Yield Credit Spread
IG_SPREAD.parquet     - Investment Grade Credit Spread
FED_FUNDS.parquet     - Federal Funds Rate
CPI.parquet           - Consumer Price Index
UNRATE.parquet        - Unemployment Rate
INDPRO.parquet        - Industrial Production Index
SP500.parquet         - S&P 500 (for reference)
```

### Current Implementation

`src/quantetf/data/macro_loader.py` has:
- `MacroDataLoader` class that loads all parquet files
- `get_vix(as_of)` - returns VIX value
- `get_yield_curve_spread(as_of)` - returns 10Y-2Y spread
- Internal `_data` dict with all loaded DataFrames

### Design Requirements

1. **Point-in-time access** - All getters must respect `as_of` date
2. **Lookback windows** - Support getting rolling windows (e.g., VIX over last 20 days)
3. **Z-scores** - Support standardized values for regime detection
4. **Composite indicators** - Support derived indicators (e.g., financial conditions index)

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Coding standards
2. **`/workspaces/qetf/src/quantetf/data/macro_loader.py`** - Current implementation
3. **`/workspaces/qetf/data/raw/macro/`** - Available data files
4. **`/workspaces/qetf/src/quantetf/monitoring/regime.py`** - How regime uses macro

---

## Implementation Steps

### 1. Define MacroIndicator enum

Add to `src/quantetf/data/macro_loader.py`:

```python
from enum import Enum
from typing import Optional, List
import pandas as pd
import numpy as np


class MacroIndicator(Enum):
    """Available macro indicators."""
    # Volatility
    VIX = "VIX"

    # Interest Rates
    TREASURY_3M = "DGS3MO"
    TREASURY_2Y = "DGS2"
    TREASURY_10Y = "DGS10"
    FED_FUNDS = "FED_FUNDS"

    # Yield Spreads
    YIELD_CURVE_10Y2Y = "T10Y2Y"
    YIELD_CURVE_10Y3M = "T10Y3M"

    # Credit Spreads
    HIGH_YIELD_SPREAD = "HY_SPREAD"
    INVESTMENT_GRADE_SPREAD = "IG_SPREAD"

    # Economic Indicators
    CPI = "CPI"
    UNEMPLOYMENT = "UNRATE"
    INDUSTRIAL_PRODUCTION = "INDPRO"

    # Market
    SP500 = "SP500"
```

### 2. Enhance MacroDataLoader with full API

```python
class MacroDataLoader:
    """Load and access macroeconomic indicators.

    Provides point-in-time access to macro data for regime detection
    and alpha model conditioning.
    """

    def __init__(self, macro_dir: str = "data/raw/macro"):
        self.macro_dir = Path(macro_dir)
        self._data: Dict[str, pd.DataFrame] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all available macro data files."""
        for indicator in MacroIndicator:
            filepath = self.macro_dir / f"{indicator.value}.parquet"
            if filepath.exists():
                df = pd.read_parquet(filepath)
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                self._data[indicator.value] = df

    def get_available_indicators(self) -> List[MacroIndicator]:
        """Return list of indicators with loaded data."""
        return [
            ind for ind in MacroIndicator
            if ind.value in self._data
        ]

    # ==================== Core Getters ====================

    def get(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        column: str = "value",
    ) -> Optional[float]:
        """
        Get indicator value as-of a date.

        Args:
            indicator: Which indicator to get
            as_of: Point-in-time date
            column: Column name (default 'value')

        Returns:
            Most recent value on or before as_of, or None if no data
        """
        if indicator.value not in self._data:
            return None

        df = self._data[indicator.value]
        valid = df.loc[:as_of]

        if valid.empty:
            return None

        return float(valid[column].iloc[-1])

    def get_series(
        self,
        indicator: MacroIndicator,
        start: pd.Timestamp,
        end: pd.Timestamp,
        column: str = "value",
    ) -> pd.Series:
        """
        Get indicator time series for date range.

        Args:
            indicator: Which indicator
            start: Start date (inclusive)
            end: End date (inclusive)
            column: Column name

        Returns:
            Series of values within date range
        """
        if indicator.value not in self._data:
            return pd.Series(dtype=float)

        df = self._data[indicator.value]
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask, column]

    def get_lookback(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        lookback_days: int,
        column: str = "value",
    ) -> pd.Series:
        """
        Get indicator values for lookback window.

        Args:
            indicator: Which indicator
            as_of: End date of lookback
            lookback_days: Number of calendar days to look back
            column: Column name

        Returns:
            Series of values in lookback window
        """
        start = as_of - pd.Timedelta(days=lookback_days)
        return self.get_series(indicator, start, as_of, column)

    # ==================== Convenience Getters ====================

    def get_vix(self, as_of: pd.Timestamp) -> Optional[float]:
        """Get VIX value as-of date."""
        return self.get(MacroIndicator.VIX, as_of)

    def get_yield_curve_spread(self, as_of: pd.Timestamp) -> Optional[float]:
        """Get 10Y-2Y yield spread as-of date."""
        return self.get(MacroIndicator.YIELD_CURVE_10Y2Y, as_of)

    def get_credit_spread(
        self,
        as_of: pd.Timestamp,
        high_yield: bool = True,
    ) -> Optional[float]:
        """Get credit spread as-of date.

        Args:
            as_of: Point-in-time date
            high_yield: If True, return HY spread; else IG spread
        """
        indicator = (
            MacroIndicator.HIGH_YIELD_SPREAD
            if high_yield
            else MacroIndicator.INVESTMENT_GRADE_SPREAD
        )
        return self.get(indicator, as_of)

    def get_fed_funds(self, as_of: pd.Timestamp) -> Optional[float]:
        """Get Federal Funds rate as-of date."""
        return self.get(MacroIndicator.FED_FUNDS, as_of)

    def get_treasury_rate(
        self,
        as_of: pd.Timestamp,
        maturity: str = "10Y",
    ) -> Optional[float]:
        """Get Treasury yield as-of date.

        Args:
            as_of: Point-in-time date
            maturity: "3M", "2Y", or "10Y"
        """
        indicator_map = {
            "3M": MacroIndicator.TREASURY_3M,
            "2Y": MacroIndicator.TREASURY_2Y,
            "10Y": MacroIndicator.TREASURY_10Y,
        }
        if maturity not in indicator_map:
            raise ValueError(f"Invalid maturity: {maturity}. Use 3M, 2Y, or 10Y")
        return self.get(indicator_map[maturity], as_of)

    # ==================== Statistical Methods ====================

    def get_zscore(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        lookback_days: int = 252,
    ) -> Optional[float]:
        """
        Get z-score of current value relative to lookback period.

        Args:
            indicator: Which indicator
            as_of: Point-in-time date
            lookback_days: Days for calculating mean/std

        Returns:
            Z-score or None if insufficient data
        """
        series = self.get_lookback(indicator, as_of, lookback_days)

        if len(series) < 20:  # Minimum observations
            return None

        current = series.iloc[-1]
        mean = series.mean()
        std = series.std()

        if std == 0:
            return 0.0

        return (current - mean) / std

    def get_percentile(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        lookback_days: int = 252,
    ) -> Optional[float]:
        """
        Get percentile rank of current value.

        Args:
            indicator: Which indicator
            as_of: Point-in-time date
            lookback_days: Days for calculating percentile

        Returns:
            Percentile (0-100) or None if insufficient data
        """
        series = self.get_lookback(indicator, as_of, lookback_days)

        if len(series) < 20:
            return None

        current = series.iloc[-1]
        return float((series < current).sum() / len(series) * 100)

    def get_change(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        days: int = 20,
    ) -> Optional[float]:
        """
        Get change in indicator over period.

        Args:
            indicator: Which indicator
            as_of: Point-in-time date
            days: Number of days for change calculation

        Returns:
            Absolute change or None
        """
        series = self.get_lookback(indicator, as_of, days + 5)  # Buffer for missing days

        if len(series) < 2:
            return None

        return float(series.iloc[-1] - series.iloc[0])

    # ==================== Composite Indicators ====================

    def get_financial_conditions(
        self,
        as_of: pd.Timestamp,
        lookback_days: int = 252,
    ) -> Optional[float]:
        """
        Compute composite financial conditions indicator.

        Combines VIX, credit spreads, and yield curve into single score.
        Higher = tighter conditions (more stress).

        Returns:
            Composite z-score or None if data missing
        """
        components = []

        # VIX z-score (higher = more stress)
        vix_z = self.get_zscore(MacroIndicator.VIX, as_of, lookback_days)
        if vix_z is not None:
            components.append(vix_z)

        # HY spread z-score (higher = more stress)
        hy_z = self.get_zscore(MacroIndicator.HIGH_YIELD_SPREAD, as_of, lookback_days)
        if hy_z is not None:
            components.append(hy_z)

        # Yield curve z-score (lower/negative = more stress)
        yc_z = self.get_zscore(MacroIndicator.YIELD_CURVE_10Y2Y, as_of, lookback_days)
        if yc_z is not None:
            components.append(-yc_z)  # Invert: flat/inverted = stress

        if not components:
            return None

        return float(np.mean(components))

    def get_macro_snapshot(self, as_of: pd.Timestamp) -> dict:
        """
        Get snapshot of all macro indicators as-of date.

        Returns:
            Dict with all available indicator values
        """
        snapshot = {"as_of": as_of.isoformat()}

        for indicator in self.get_available_indicators():
            value = self.get(indicator, as_of)
            if value is not None:
                snapshot[indicator.value] = value

        return snapshot
```

### 3. Write tests

Create `tests/data/test_macro_loader.py`:

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from quantetf.data.macro_loader import MacroDataLoader, MacroIndicator


class TestMacroDataLoader:
    """Tests for enhanced macro data API."""

    @pytest.fixture
    def mock_loader(self, tmp_path):
        """Create loader with mock data."""
        # Create mock VIX data
        dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
        vix_data = pd.DataFrame(
            {"value": np.random.uniform(15, 30, len(dates))},
            index=dates,
        )
        vix_path = tmp_path / "VIX.parquet"
        vix_data.to_parquet(vix_path)

        # Create mock yield curve data
        yc_data = pd.DataFrame(
            {"value": np.random.uniform(-0.5, 2.0, len(dates))},
            index=dates,
        )
        yc_path = tmp_path / "T10Y2Y.parquet"
        yc_data.to_parquet(yc_path)

        return MacroDataLoader(str(tmp_path))

    def test_get_available_indicators(self, mock_loader):
        """Test listing available indicators."""
        available = mock_loader.get_available_indicators()
        assert MacroIndicator.VIX in available
        assert MacroIndicator.YIELD_CURVE_10Y2Y in available

    def test_get_value_as_of(self, mock_loader):
        """Test point-in-time value retrieval."""
        value = mock_loader.get(
            MacroIndicator.VIX,
            pd.Timestamp("2023-06-15"),
        )
        assert value is not None
        assert 10 < value < 40

    def test_get_returns_none_for_future_date(self, mock_loader):
        """Test that future dates return last known value."""
        value = mock_loader.get(
            MacroIndicator.VIX,
            pd.Timestamp("2022-01-01"),  # Before data starts
        )
        assert value is None

    def test_get_lookback(self, mock_loader):
        """Test lookback window retrieval."""
        series = mock_loader.get_lookback(
            MacroIndicator.VIX,
            pd.Timestamp("2023-06-15"),
            lookback_days=30,
        )
        assert len(series) > 0
        assert len(series) <= 30

    def test_get_zscore(self, mock_loader):
        """Test z-score calculation."""
        zscore = mock_loader.get_zscore(
            MacroIndicator.VIX,
            pd.Timestamp("2023-12-01"),
            lookback_days=252,
        )
        assert zscore is not None
        assert -4 < zscore < 4  # Reasonable z-score range

    def test_get_percentile(self, mock_loader):
        """Test percentile calculation."""
        pct = mock_loader.get_percentile(
            MacroIndicator.VIX,
            pd.Timestamp("2023-12-01"),
            lookback_days=252,
        )
        assert pct is not None
        assert 0 <= pct <= 100

    def test_get_macro_snapshot(self, mock_loader):
        """Test snapshot of all indicators."""
        snapshot = mock_loader.get_macro_snapshot(pd.Timestamp("2023-06-15"))
        assert "as_of" in snapshot
        assert "VIX" in snapshot


class TestMacroIndicatorEnum:
    """Tests for MacroIndicator enum."""

    def test_all_indicators_have_values(self):
        """Test all enum members have string values."""
        for indicator in MacroIndicator:
            assert isinstance(indicator.value, str)
            assert len(indicator.value) > 0

    def test_indicator_categories(self):
        """Test indicator groupings make sense."""
        volatility = [MacroIndicator.VIX]
        rates = [
            MacroIndicator.TREASURY_3M,
            MacroIndicator.TREASURY_2Y,
            MacroIndicator.TREASURY_10Y,
            MacroIndicator.FED_FUNDS,
        ]
        spreads = [
            MacroIndicator.YIELD_CURVE_10Y2Y,
            MacroIndicator.YIELD_CURVE_10Y3M,
            MacroIndicator.HIGH_YIELD_SPREAD,
            MacroIndicator.INVESTMENT_GRADE_SPREAD,
        ]

        # All should be valid enum members
        for ind in volatility + rates + spreads:
            assert ind in MacroIndicator
```

---

## Acceptance Criteria

- [ ] `MacroIndicator` enum with all available indicators
- [ ] `get()` method for any indicator with point-in-time access
- [ ] `get_series()` for date range queries
- [ ] `get_lookback()` for rolling window access
- [ ] `get_zscore()` for standardized values
- [ ] `get_percentile()` for percentile ranking
- [ ] `get_change()` for period changes
- [ ] `get_financial_conditions()` composite indicator
- [ ] `get_macro_snapshot()` for full state dump
- [ ] Convenience getters: `get_vix()`, `get_credit_spread()`, `get_fed_funds()`, `get_treasury_rate()`
- [ ] All methods respect point-in-time (no lookahead)
- [ ] Unit tests pass
- [ ] Type hints and docstrings complete

---

## Definition of Done

1. All acceptance criteria met
2. `pytest tests/data/test_macro_loader.py` passes
3. Existing code using old API still works (backward compatible)
4. PROGRESS_LOG.md updated
5. Completion note created: `handoffs/completion-IMPL-017.md`
6. TASKS.md status updated to `completed`
7. Code committed with clear message

---

## Notes

- Maintain backward compatibility with existing `get_vix()` and `get_yield_curve_spread()`
- Consider adding caching for frequently accessed lookback windows
- Z-scores and percentiles are useful for regime thresholds
- Financial conditions composite is just one example - others can be added
- This API enables regime detection to use rich macro context
