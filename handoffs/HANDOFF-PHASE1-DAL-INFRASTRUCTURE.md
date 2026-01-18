# Handoff: Phase 1 DAL Infrastructure Implementation (IMPL-019 through IMPL-024)

**Created:** January 18, 2026  
**Target Audience:** Coding agents implementing Phase 1  
**Status:** Ready for implementation  

---

## Overview

This handoff contains everything needed to implement Phase 1 of the Data Access Layer refactoring. Phase 1 is foundational and blocks all subsequent work.

**Success Criteria:** All 6 tasks complete with 100% test pass rate and no circular imports.

**Estimated Effort:** 12-14 hours total (can parallelize)

---

## Context

The QuantETF project currently uses a "snapshot" pattern where scripts directly pass snapshot file paths to components. This creates tight coupling and makes testing difficult.

**Goal:** Replace snapshots with a unified Data Access Layer that:
1. Provides clean interfaces for all data access
2. Centralizes caching and optimization
3. Enables easy testing with mocks
4. Supports multiple data sources (snapshot, live API, etc.)

---

## IMPL-019: DAL Core Interfaces & Types

**Priority:** CRITICAL (blocks all others)  
**Effort:** 3 hours | 400 LOC  
**Must start here.**

### What to Build

Create 4 new Python files that define the foundation:

**File 1: `src/quantetf/data/access/__init__.py`**
- New directory `src/quantetf/data/access/`
- Package initialization file
- Expose public interfaces:
  ```python
  from .abstract import (
      PriceDataAccessor,
      MacroDataAccessor,
      UniverseDataAccessor,
      ReferenceDataAccessor,
  )
  from .context import DataAccessContext
  from .factory import DataAccessFactory
  from .types import Regime, TickerMetadata, ExchangeInfo
  
  __all__ = [...]
  ```

**File 2: `src/quantetf/data/access/types.py`**

Define these enums and dataclasses:

```python
from enum import Enum
from dataclasses import dataclass

class Regime(Enum):
    """Market regime classification."""
    RISK_ON = "risk_on"
    ELEVATED_VOL = "elevated_vol"
    HIGH_VOL = "high_vol"
    RECESSION_WARNING = "recession_warning"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class TickerMetadata:
    """Metadata about a ticker/ETF."""
    ticker: str
    name: str
    sector: str
    exchange: str
    currency: str

@dataclass(frozen=True)
class ExchangeInfo:
    """Metadata about an exchange."""
    name: str
    trading_hours: str  # e.g., "09:30-16:00 EST"
    timezone: str  # e.g., "US/Eastern"
    settlement_days: int  # Usually 2 or 3

@dataclass(frozen=True)
class DataAccessMetadata:
    """Metadata about data from an accessor."""
    source: str
    timestamp: pd.Timestamp
    lookback_date: Optional[pd.Timestamp]
    data_quality_score: float  # 0.0 to 1.0
```

**File 3: `src/quantetf/data/access/abstract.py`**

Define abstract base classes. These are your interfaces - all implementations must satisfy these contracts.

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd

class PriceDataAccessor(ABC):
    """Abstract interface for price data access.
    
    All implementations must:
    - Provide point-in-time data (no lookahead bias)
    - Filter to dates < as_of (strict inequality)
    - Support optional ticker filtering
    - Support optional lookback windows
    """
    
    @abstractmethod
    def read_prices_as_of(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return OHLCV prices for all dates < as_of.
        
        Critical: Data on/after as_of must be excluded (lookahead prevention).
        
        Args:
            as_of: Cutoff date (exclusive - not included)
            tickers: Optional subset of tickers to return
            lookback_days: Optional window of days to look back
            
        Returns:
            DataFrame with:
            - Index: datetime (dates <= as_of)
            - Columns: MultiIndex (Ticker, Field) where Field in [Open, High, Low, Close, Volume]
            
        Raises:
            ValueError: If no data available before as_of
        """
        pass
    
    @abstractmethod
    def read_ohlcv_range(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return OHLCV for closed date range [start, end]."""
        pass
    
    @abstractmethod
    def get_latest_price_date(self) -> pd.Timestamp:
        """Return most recent date with available price data."""
        pass
    
    @abstractmethod
    def validate_data_availability(
        self,
        tickers: list[str],
        as_of: pd.Timestamp,
    ) -> dict[str, bool]:
        """Check which tickers have data available as of date.
        
        Returns:
            Dict mapping ticker → True if available, False if missing
        """
        pass

class MacroDataAccessor(ABC):
    """Abstract interface for macro data access."""
    
    @abstractmethod
    def read_macro_indicator(
        self,
        indicator: str,
        as_of: pd.Timestamp,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return time series for a macro indicator.
        
        Returns data up to as_of (point-in-time).
        
        Args:
            indicator: Indicator name (e.g., "VIX", "SPX_YIELD")
            as_of: Cutoff date (exclusive)
            lookback_days: Optional lookback window
            
        Returns:
            DataFrame with DatetimeIndex and indicator values
            
        Raises:
            ValueError: If indicator not available
        """
        pass
    
    @abstractmethod
    def get_regime(self, as_of: pd.Timestamp) -> Regime:
        """Detect current market regime as of date.
        
        Returns: One of RISK_ON, ELEVATED_VOL, HIGH_VOL, RECESSION_WARNING, UNKNOWN
        """
        pass
    
    @abstractmethod
    def get_available_indicators(self) -> list[str]:
        """Return list of available macro indicators."""
        pass

class UniverseDataAccessor(ABC):
    """Abstract interface for universe (ticker set) definitions."""
    
    @abstractmethod
    def get_universe(self, universe_name: str) -> list[str]:
        """Get current/latest universe tickers."""
        pass
    
    @abstractmethod
    def get_universe_as_of(
        self,
        universe_name: str,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """Get universe membership at specific point in time.
        
        For graduated universes, only includes tickers added by as_of.
        """
        pass
    
    @abstractmethod
    def list_available_universes(self) -> list[str]:
        """Return list of available universe names."""
        pass

class ReferenceDataAccessor(ABC):
    """Abstract interface for static reference data."""
    
    @abstractmethod
    def get_ticker_info(self, ticker: str) -> TickerMetadata:
        """Get metadata for a ticker."""
        pass
    
    @abstractmethod
    def get_sector_mapping(self) -> dict[str, str]:
        """Return ticker → sector mapping for all tickers."""
        pass
    
    @abstractmethod
    def get_exchange_info(self) -> dict[str, ExchangeInfo]:
        """Return exchange → metadata mapping."""
        pass
```

**File 4: `src/quantetf/data/access/context.py`**

```python
from dataclasses import dataclass
from .abstract import (
    PriceDataAccessor,
    MacroDataAccessor,
    UniverseDataAccessor,
    ReferenceDataAccessor,
)

@dataclass(frozen=True)
class DataAccessContext:
    """Container for all DAL accessors.
    
    Usage:
        ctx = DataAccessContext(
            prices=price_accessor,
            macro=macro_accessor,
            universes=universe_accessor,
            references=ref_accessor,
        )
        
        # Pass to components
        engine = SimpleBacktestEngine(data_access=ctx)
    """
    
    prices: PriceDataAccessor
    macro: MacroDataAccessor
    universes: UniverseDataAccessor
    references: ReferenceDataAccessor
```

**File 5: `src/quantetf/data/access/factory.py`**

```python
from pathlib import Path
from typing import Optional, Dict, Any
from .context import DataAccessContext
from .abstract import (
    PriceDataAccessor,
    MacroDataAccessor,
    UniverseDataAccessor,
    ReferenceDataAccessor,
)

class DataAccessFactory:
    """Factory for creating configured DAL accessors.
    
    Usage:
        # Create from defaults
        ctx = DataAccessFactory.create_context()
        
        # Create with custom config
        ctx = DataAccessFactory.create_context(
            config_file="configs/data_access.yaml"
        )
        
        # Create individual accessors
        prices = DataAccessFactory.create_price_accessor(
            source="snapshot",
            config={"snapshot_dir": "data/snapshots/latest"}
        )
    """
    
    @staticmethod
    def create_price_accessor(
        source: str = "snapshot",
        config: Optional[Dict[str, Any]] = None,
    ) -> PriceDataAccessor:
        """Create price accessor from source type.
        
        Args:
            source: "snapshot" (default), "live", or "mock"
            config: Source-specific configuration dict
            
        Returns:
            Configured PriceDataAccessor instance
        """
        # Implementation in IMPL-020, IMPL-032
        pass
    
    @staticmethod
    def create_macro_accessor(
        source: str = "fred",
        config: Optional[Dict[str, Any]] = None,
    ) -> MacroDataAccessor:
        """Create macro accessor from source type.
        
        Args:
            source: "fred" (default) or "mock"
            config: Source-specific configuration
            
        Returns:
            Configured MacroDataAccessor instance
        """
        # Implementation in IMPL-021
        pass
    
    @staticmethod
    def create_universe_accessor(
        config: Optional[Dict[str, Any]] = None,
    ) -> UniverseDataAccessor:
        """Create universe accessor.
        
        Args:
            config: Universe configuration (config_dir, etc.)
            
        Returns:
            Configured UniverseDataAccessor instance
        """
        # Implementation in IMPL-022
        pass
    
    @staticmethod
    def create_reference_accessor(
        config: Optional[Dict[str, Any]] = None,
    ) -> ReferenceDataAccessor:
        """Create reference data accessor.
        
        Args:
            config: Reference data configuration
            
        Returns:
            Configured ReferenceDataAccessor instance
        """
        # Implementation in IMPL-023
        pass
    
    @staticmethod
    def create_context(
        config_file: Optional[Path] = None,
    ) -> DataAccessContext:
        """Create a complete DataAccessContext.
        
        Args:
            config_file: Path to data_access.yaml config (optional)
            
        Returns:
            Fully configured DataAccessContext with all accessors
        """
        # Will use individual create_* methods above
        pass
```

### Testing Requirements

**File:** `tests/data/access/test_dal_core.py`

Create unit tests:

```python
import pytest
from quantetf.data.access import (
    DataAccessContext,
    DataAccessFactory,
    Regime,
    TickerMetadata,
)

def test_regime_enum_has_expected_values():
    """Verify Regime enum has all required values."""
    assert hasattr(Regime, 'RISK_ON')
    assert hasattr(Regime, 'ELEVATED_VOL')
    assert hasattr(Regime, 'HIGH_VOL')
    assert hasattr(Regime, 'RECESSION_WARNING')
    assert hasattr(Regime, 'UNKNOWN')

def test_ticker_metadata_is_frozen():
    """Verify TickerMetadata is immutable."""
    meta = TickerMetadata(
        ticker="SPY",
        name="SPDR S&P 500",
        sector="Broad Market",
        exchange="NYSE",
        currency="USD",
    )
    with pytest.raises(AttributeError):
        meta.ticker = "QQQ"

def test_data_access_context_holds_accessors():
    """Verify DataAccessContext can hold accessor instances."""
    # Use mock accessors (from IMPL-031)
    from tests.data.access.mocks import (
        MockPriceAccessor,
        MockMacroAccessor,
        MockUniverseAccessor,
        MockReferenceAccessor,
    )
    
    ctx = DataAccessContext(
        prices=MockPriceAccessor(),
        macro=MockMacroAccessor(),
        universes=MockUniverseAccessor(),
        references=MockReferenceAccessor(),
    )
    
    assert ctx.prices is not None
    assert ctx.macro is not None
    assert ctx.universes is not None
    assert ctx.references is not None

def test_data_access_context_is_frozen():
    """Verify context is immutable."""
    # Should raise error if trying to modify
    pass

def test_factory_creates_accessors():
    """Verify factory can create accessor instances."""
    # Will need mock accessors to exist first
    pass
```

### Acceptance Criteria

✓ All 5 files created  
✓ No circular imports (`python -m py_compile src/quantetf/data/access/*.py`)  
✓ Type checking passes (`mypy src/quantetf/data/access/`)  
✓ All unit tests pass  
✓ Docstrings complete and clear  
✓ Abstract methods cannot be instantiated  

---

## IMPL-020: SnapshotPriceAccessor Implementation

**Priority:** CRITICAL  
**Effort:** 3 hours | 250 LOC  
**Depends on:** IMPL-019

### What to Build

Implement the concrete price accessor that wraps the existing SnapshotDataStore.

**File:** `src/quantetf/data/access/snapshot_price.py`

```python
from pathlib import Path
from typing import Optional
import pandas as pd
from .abstract import PriceDataAccessor

class SnapshotPriceAccessor(PriceDataAccessor):
    """Access price data from a snapshot parquet file.
    
    Wraps SnapshotDataStore to provide PriceDataAccessor interface.
    Guarantees point-in-time data access (no lookahead bias).
    """
    
    def __init__(self, snapshot_path: Path):
        """Initialize with path to snapshot parquet file.
        
        Args:
            snapshot_path: Path to data.parquet file in snapshot directory
            
        Raises:
            FileNotFoundError: If snapshot doesn't exist
        """
        from quantetf.data.snapshot_store import SnapshotDataStore
        
        self.snapshot_path = Path(snapshot_path)
        self._store = SnapshotDataStore(self.snapshot_path)
        self._latest_date = self._store._data.index.max()
    
    def read_prices_as_of(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return prices for all dates < as_of (point-in-time)."""
        # Use existing SnapshotDataStore.read_prices() method
        # Make sure to filter to dates BEFORE as_of (strict inequality)
        result = self._store.read_prices(
            as_of=as_of,
            tickers=tickers,
            lookback_days=lookback_days,
        )
        if result.empty:
            raise ValueError(f"No price data available before {as_of}")
        return result
    
    def read_ohlcv_range(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return OHLCV for date range [start, end]."""
        # Filter _store._data to range [start, end]
        # Apply ticker filter if specified
        pass
    
    def get_latest_price_date(self) -> pd.Timestamp:
        """Return latest date with price data."""
        return self._latest_date
    
    def validate_data_availability(
        self,
        tickers: list[str],
        as_of: pd.Timestamp,
    ) -> dict[str, bool]:
        """Check which tickers have data before as_of."""
        result = {}
        try:
            pit_data = self._store._data[self._store._data.index < as_of]
            available = pit_data.columns.get_level_values('Ticker').unique()
            for ticker in tickers:
                result[ticker] = ticker in available
        except Exception:
            result = {t: False for t in tickers}
        return result
```

### Key Implementation Points

1. **Point-in-Time Access:** When user calls `read_prices_as_of(as_of=2023-12-31)`, return data where `index < 2023-12-31` only. No data on/after the cutoff date.

2. **Lookahead Bias Prevention:** This is critical. The cutoff is strict inequality (<, not <=).

3. **Reuse Existing Code:** Wrap/compose with SnapshotDataStore, don't rewrite it.

4. **Lazy Loading:** Cache the latest_date but don't load entire snapshot into memory until first query.

5. **Error Handling:** Raise informative errors if snapshot missing or corrupted.

### Testing Requirements

**File:** `tests/data/access/test_snapshot_price.py`

Critical tests:

```python
import pytest
import pandas as pd
from pathlib import Path
from quantetf.data.access.snapshot_price import SnapshotPriceAccessor

def test_read_prices_as_of_returns_point_in_time():
    """Verify data returned only for dates < as_of."""
    accessor = SnapshotPriceAccessor(Path("data/snapshots/snapshot_5yr_20etfs/data.parquet"))
    as_of = pd.Timestamp("2023-12-31")
    
    prices = accessor.read_prices_as_of(as_of)
    
    # All dates must be < as_of
    assert (prices.index < as_of).all()
    # No data on as_of date itself
    assert (prices.index != as_of).all()

def test_read_prices_as_of_with_ticker_filter():
    """Verify ticker filtering works."""
    accessor = SnapshotPriceAccessor(Path("data/snapshots/snapshot_5yr_20etfs/data.parquet"))
    prices = accessor.read_prices_as_of(
        as_of=pd.Timestamp("2023-12-31"),
        tickers=["SPY", "QQQ"],
    )
    
    available_tickers = prices.columns.get_level_values('Ticker').unique()
    assert set(available_tickers) <= {"SPY", "QQQ"}

def test_get_latest_price_date():
    """Verify latest date is returned correctly."""
    accessor = SnapshotPriceAccessor(Path("data/snapshots/snapshot_5yr_20etfs/data.parquet"))
    latest = accessor.get_latest_price_date()
    assert isinstance(latest, pd.Timestamp)

def test_validate_data_availability():
    """Verify availability check works."""
    accessor = SnapshotPriceAccessor(Path("data/snapshots/snapshot_5yr_20etfs/data.parquet"))
    result = accessor.validate_data_availability(
        ["SPY", "QQQ", "NONEXISTENT"],
        as_of=pd.Timestamp("2023-12-31"),
    )
    
    assert result["SPY"] == True
    assert result["QQQ"] == True
    assert result["NONEXISTENT"] == False
```

### Acceptance Criteria

✓ Wraps SnapshotDataStore without rewriting  
✓ Point-in-time access guaranteed  
✓ No lookahead bias possible  
✓ All tests pass  
✓ Type hints complete  
✓ Performance acceptable (< 100ms typical query)  

---

## IMPL-021: FREDMacroAccessor Implementation

**Priority:** CRITICAL  
**Effort:** 3 hours | 300 LOC  
**Depends on:** IMPL-019

### What to Build

Implement macro data accessor that wraps existing MacroDataLoader and adds regime detection.

**File:** `src/quantetf/data/access/fred_macro.py`

```python
import pandas as pd
from typing import Optional
from .abstract import MacroDataAccessor
from .types import Regime

class FREDMacroAccessor(MacroDataAccessor):
    """Access macro data and detect market regimes.
    
    Wraps existing MacroDataLoader to provide clean interface.
    Adds regime detection logic.
    """
    
    def __init__(self, macro_loader):
        """Initialize with MacroDataLoader instance.
        
        Args:
            macro_loader: Existing MacroDataLoader from quantetf.data.macro_loader
        """
        self.macro_loader = macro_loader
    
    def read_macro_indicator(
        self,
        indicator: str,
        as_of: pd.Timestamp,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return time series for macro indicator.
        
        Args:
            indicator: "VIX", "SPX_YIELD", "US_2Y_10Y_SPREAD", etc.
            as_of: Cutoff date (return data <= as_of)
            lookback_days: Optional lookback window
        """
        # Load from macro_loader
        # Filter to dates <= as_of
        # Apply lookback if specified
        pass
    
    def get_regime(self, as_of: pd.Timestamp) -> Regime:
        """Detect market regime as of date.
        
        Regime logic:
        1. If VIX > 30 AND US_2Y_10Y_SPREAD < -0.5: RECESSION_WARNING
        2. Else if VIX > 30: HIGH_VOL
        3. Else if VIX > 20: ELEVATED_VOL
        4. Else if US_2Y_10Y_SPREAD < 0: ELEVATED_VOL
        5. Else: RISK_ON
        6. If any indicator missing: UNKNOWN
        """
        try:
            # Read VIX and spread
            vix_data = self.read_macro_indicator("VIX", as_of, lookback_days=1)
            spread_data = self.read_macro_indicator("US_2Y_10Y_SPREAD", as_of, lookback_days=1)
            
            # Get latest values
            vix = vix_data.iloc[-1].values[0] if not vix_data.empty else None
            spread = spread_data.iloc[-1].values[0] if not spread_data.empty else None
            
            if vix is None or spread is None:
                return Regime.UNKNOWN
            
            # Apply regime logic
            if vix > 30 and spread < -0.5:
                return Regime.RECESSION_WARNING
            elif vix > 30:
                return Regime.HIGH_VOL
            elif vix > 20:
                return Regime.ELEVATED_VOL
            elif spread < 0:
                return Regime.ELEVATED_VOL
            else:
                return Regime.RISK_ON
        except Exception:
            return Regime.UNKNOWN
    
    def get_available_indicators(self) -> list[str]:
        """Return list of available macro indicators."""
        # Return list from macro_loader
        pass
```

### Key Implementation Points

1. **Wrap MacroDataLoader:** Reuse existing functionality.

2. **Regime Detection:** Implement the logic shown above. Handle missing data gracefully by returning UNKNOWN.

3. **Point-in-Time:** Ensure data returned is up to as_of date (not after).

4. **Error Handling:** Regime detection should never crash - return UNKNOWN on any error.

### Testing Requirements

**File:** `tests/data/access/test_fred_macro.py`

Critical tests:

```python
import pytest
import pandas as pd
from quantetf.data.access.fred_macro import FREDMacroAccessor
from quantetf.data.access.types import Regime

def test_regime_recession_warning():
    """VIX > 30 and spread < -0.5 → RECESSION_WARNING."""
    # Mock macro data with VIX=35, spread=-0.6
    # Verify regime returns RECESSION_WARNING
    pass

def test_regime_high_vol():
    """VIX > 30 (but good spread) → HIGH_VOL."""
    pass

def test_regime_elevated_vol_from_vix():
    """VIX > 20 (but < 30) → ELEVATED_VOL."""
    pass

def test_regime_elevated_vol_from_spread():
    """Negative spread → ELEVATED_VOL."""
    pass

def test_regime_risk_on():
    """Low VIX, positive spread → RISK_ON."""
    pass

def test_regime_unknown_on_missing_data():
    """Missing indicators → UNKNOWN."""
    pass

def test_read_macro_indicator():
    """Verify macro data read works."""
    pass
```

### Acceptance Criteria

✓ Regime detection implemented  
✓ Regime logic correct for all cases  
✓ Handles missing data gracefully  
✓ Point-in-time access  
✓ All tests pass  

---

## IMPL-022, IMPL-023, IMPL-024

These three tasks follow the same pattern as IMPL-020 and IMPL-021:

1. **IMPL-022:** Implement `ConfigFileUniverseAccessor` wrapping config file reading
2. **IMPL-023:** Implement `StaticReferenceDataAccessor` for reference data
3. **IMPL-024:** Implement `CachedPriceAccessor` and `CachedMacroAccessor` decorators

See the detailed specifications in [IMPLEMENTATION_TASKS_DATA_ACCESS_LAYER.md](./IMPLEMENTATION_TASKS_DATA_ACCESS_LAYER.md) for complete requirements.

---

## What NOT to Do

❌ Don't rewrite SnapshotDataStore or MacroDataLoader - wrap them  
❌ Don't create circular imports (careful with imports)  
❌ Don't hardcode file paths - make them configurable  
❌ Don't forget point-in-time logic (< operator, not <=)  
❌ Don't skip type hints  
❌ Don't skip docstrings  
❌ Don't skip tests  

---

## Integration with Existing Code

The following existing modules are used:

- `quantetf.data.snapshot_store.SnapshotDataStore` - wrap in IMPL-020
- `quantetf.data.macro_loader.MacroDataLoader` - wrap in IMPL-021
- `quantetf.data.providers.*` - use for universe loading
- `quantetf.config.loader` - use for loading reference configs

Don't modify these existing modules. Just wrap them.

---

## Success Definition

**Phase 1 is complete when:**

1. All 6 tasks (IMPL-019 through IMPL-024) have working code
2. All unit tests pass (minimum 80% coverage)
3. `mypy src/quantetf/data/access/` passes cleanly
4. No circular imports
5. All docstrings complete
6. Code ready for Phase 2 migration work

**Estimated time:** 12-14 hours total (can parallelize: IMPL-019 first, then 020-023 in parallel, then 024)

---

## Next Steps After Phase 1

Once Phase 1 complete, the 6 new DAL components will be used by:

- IMPL-025: Backtest engine refactoring
- IMPL-026: Alpha models refactoring  
- IMPL-027: Optimization refactoring
- IMPL-028: Production pipeline refactoring
- IMPL-029: Research scripts refactoring
- IMPL-030: Monitoring refactoring
- IMPL-031: Test utilities and mocks

Each of these will be simpler because the DAL foundation is in place.

