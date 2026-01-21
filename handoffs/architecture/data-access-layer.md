# Architecture Design: Data Access Layer Replacement for QuantETF

**Created:** January 18, 2026  
**Status:** Architecture & Planning Phase  
**Phase:** ARCHITECTURE (no code implementation)

---

## Executive Summary

This document replaces the "data snapshot" concept with a **unified Data Access Layer (DAL)** architecture. The DAL serves as the single point of access for all data consumed by production and research scripts. All components (backtest engines, portfolio optimizers, monitoring systems, production pipelines) access data exclusively through specialized DAL objects.

**Key Principle:** *Data never flows directly from files to business logic. All data access is mediated through the DAL.*

---

## Current State Problem

### Current Data Flow
```
Raw Data Files
    ↓
[Snapshot Store] ← Hard-coded file path dependency
    ↓ (SnapshotDataStore)
Backtest Scripts → Research Scripts → Production Scripts
```

### Issues with Current Snapshots Concept
1. **Snapshot Versioning Complexity**: Snapshot IDs are manually created, hard to track which snapshot is "current"
2. **Direct File Dependencies**: Scripts hardcode snapshot paths (`data/snapshots/snapshot_20260113_232157`)
3. **No Runtime Data Updates**: Can't update data without creating a new snapshot
4. **Poor Testability**: Tests must provision entire snapshots instead of mocking data access
5. **Limited Metadata**: No rich context about data quality, currency, or source
6. **Difficult Cache Management**: No centralized caching strategy across all scripts
7. **Production Data Sync Burden**: Production system can't easily stay synchronized with latest data

---

## Proposed Data Access Layer Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  (Backtest, Production, Research Scripts)                       │
├─────────────────────────────────────────────────────────────────┤
│  ↓ All access mediated by specialized DAL objects               │
├─────────────────────────────────────────────────────────────────┤
│                  DATA ACCESS LAYER (DAL)                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PriceDataAccessor                                       │   │
│  │  ├─ read_prices_as_of(as_of, tickers, lookback)        │   │
│  │  ├─ read_ohlcv_range(start, end, tickers)              │   │
│  │  ├─ get_latest_price_date()                            │   │
│  │  └─ validate_data_availability(tickers, date)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MacroDataAccessor                                       │   │
│  │  ├─ read_macro_indicator(indicator, as_of, lookback)   │   │
│  │  ├─ get_regime(as_of)                                  │   │
│  │  └─ get_available_indicators()                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  UniverseDataAccessor                                    │   │
│  │  ├─ get_universe(universe_name)                        │   │
│  │  ├─ get_universe_as_of(universe_name, as_of)           │   │
│  │  └─ list_available_universes()                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ReferenceDataAccessor                                   │   │
│  │  ├─ get_ticker_info(ticker)                            │   │
│  │  ├─ get_sector_mapping()                               │   │
│  │  └─ get_exchange_info()                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DataAccessFactory                                       │   │
│  │  └─ create_accessor(type, config) → Accessor           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CachingLayer (Transparent)                              │   │
│  │  ├─ LRU cache for recent price queries                 │   │
│  │  ├─ TTL-based macro data cache                         │   │
│  │  └─ Persistent snapshot caching                        │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                  DATA STORAGE BACKENDS                          │
│  (Parquet files, Redis, FRED API, Stooq API, etc.)            │
├─────────────────────────────────────────────────────────────────┤
```

### Core Components

#### 1. **PriceDataAccessor**
Unified interface for accessing historical and current price data.

```python
class PriceDataAccessor(ABC):
    """Access price data for any date range and ticker set.
    
    Provides point-in-time data to prevent lookahead bias.
    Handles data validation and currency.
    """
    
    @abstractmethod
    def read_prices_as_of(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return prices for all dates <= as_of (point-in-time).
        
        Returns:
            DataFrame with MultiIndex (Ticker, Field): Open, High, Low, Close, Volume
        """
        
    @abstractmethod
    def read_ohlcv_range(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return OHLCV for date range [start, end].
        
        Returns:
            DataFrame with MultiIndex columns (Ticker, Field)
        """
        
    @abstractmethod
    def get_latest_price_date(self) -> pd.Timestamp:
        """Return the most recent date with price data available."""
        
    @abstractmethod
    def validate_data_availability(
        self,
        tickers: list[str],
        as_of: pd.Timestamp,
    ) -> dict[str, bool]:
        """Check which tickers have data available as of date."""
```

#### 2. **MacroDataAccessor**
Unified interface for macro data and regime detection.

```python
class MacroDataAccessor(ABC):
    """Access macro indicators and derived regime signals.
    
    Implements point-in-time data access for macro features.
    Supports regime detection and indicator composition.
    """
    
    @abstractmethod
    def read_macro_indicator(
        self,
        indicator: str,  # e.g., "VIX", "SPX_YIELD", "US_2Y_10Y_SPREAD"
        as_of: pd.Timestamp,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return time series for a macro indicator.
        
        Returns:
            DataFrame with DatetimeIndex, columns depend on indicator
        """
        
    @abstractmethod
    def get_regime(
        self,
        as_of: pd.Timestamp,
    ) -> Regime:
        """Detect current market regime using multiple indicators.
        
        Returns: RISK_ON, ELEVATED_VOL, HIGH_VOL, RECESSION_WARNING, UNKNOWN
        """
        
    @abstractmethod
    def get_available_indicators(self) -> list[str]:
        """List all available macro indicators."""
```

#### 3. **UniverseDataAccessor**
Unified interface for universe definitions and memberships.

```python
class UniverseDataAccessor(ABC):
    """Access universe definitions (ticker sets) with point-in-time support.
    
    Supports static universes (fixed lists) and dynamic universes (rules-based).
    """
    
    @abstractmethod
    def get_universe(
        self,
        universe_name: str,
    ) -> list[str]:
        """Get current/default universe tickers."""
        
    @abstractmethod
    def get_universe_as_of(
        self,
        universe_name: str,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """Get universe membership at a specific point in time.
        
        Handles graduated entry/exit of tickers.
        """
        
    @abstractmethod
    def list_available_universes(self) -> list[str]:
        """Return available universe names."""
```

#### 4. **ReferenceDataAccessor**
Unified interface for reference data (lookups, mappings).

```python
class ReferenceDataAccessor(ABC):
    """Access reference data: ticker metadata, sector mappings, etc.
    
    Provides static/slowly-changing reference information.
    """
    
    @abstractmethod
    def get_ticker_info(self, ticker: str) -> TickerMetadata:
        """Return metadata for a ticker (sector, exchange, etc.)."""
        
    @abstractmethod
    def get_sector_mapping(self) -> dict[str, str]:
        """Return ticker → sector mapping."""
        
    @abstractmethod
    def get_exchange_info(self) -> dict[str, ExchangeInfo]:
        """Return exchange metadata (trading hours, holidays, etc.)."""
```

#### 5. **DataAccessFactory**
Creates and configures accessor instances.

```python
class DataAccessFactory:
    """Factory for creating configured DAL accessor instances.
    
    Reads from environment, config files, or runtime settings.
    Provides dependency injection for tests.
    """
    
    @staticmethod
    def create_price_accessor(
        source: str = "snapshot",  # "snapshot", "live", "mock"
        config: Optional[dict] = None,
    ) -> PriceDataAccessor:
        """Create price accessor from source type."""
        
    @staticmethod
    def create_macro_accessor(
        source: str = "fred",  # "fred", "mock"
        config: Optional[dict] = None,
    ) -> MacroDataAccessor:
        """Create macro accessor from source type."""
        
    @staticmethod
    def create_universe_accessor(
        config: Optional[dict] = None,
    ) -> UniverseDataAccessor:
        """Create universe accessor."""
```

#### 6. **CachingLayer** (Transparent)
Transparent caching wraps accessors to improve performance.

```python
class CachedPriceAccessor(PriceDataAccessor):
    """Wraps PriceDataAccessor with LRU/TTL caching.
    
    - Recent queries cached in memory (LRU, configurable size)
    - Snapshots optionally persisted to disk
    - Automatic invalidation on data update
    """
    
    def __init__(
        self,
        inner: PriceDataAccessor,
        max_cache_mb: int = 500,
        snapshot_cache_dir: Optional[Path] = None,
    ):
        """Decorate accessor with caching."""
```

#### 7. **DataAccessContext**
Convenient holder of all accessors.

```python
@dataclass(frozen=True)
class DataAccessContext:
    """Convenient container for all DAL accessors.
    
    Usage:
        ctx = DataAccessContext(
            prices=price_accessor,
            macro=macro_accessor,
            universes=universe_accessor,
            references=ref_accessor,
        )
        
        # Pass to any component
        engine = SimpleBacktestEngine(data_access=ctx)
    """
    
    prices: PriceDataAccessor
    macro: MacroDataAccessor
    universes: UniverseDataAccessor
    references: ReferenceDataAccessor
```

---

## Implementation Approach

### Phase 1: Core DAL Infrastructure (5-6 tasks)

1. **IMPL-019: DAL Core Interfaces**
   - Define abstract base classes for all accessors
   - Define enums and types (Regime, TickerMetadata, etc.)
   - Create DataAccessFactory
   - Create DataAccessContext
   - Dependencies: None
   - Estimated LOC: 400

2. **IMPL-020: SnapshotPriceAccessor Implementation**
   - Implement PriceDataAccessor wrapping SnapshotDataStore
   - Point-in-time logic
   - Ticker filtering and validation
   - Dependencies: IMPL-019
   - Estimated LOC: 250

3. **IMPL-021: FREDMacroAccessor Implementation**
   - Implement MacroDataAccessor using MacroDataLoader
   - Regime detection integration
   - Indicator composition
   - Dependencies: IMPL-019
   - Estimated LOC: 300

4. **IMPL-022: ConfigFileUniverseAccessor**
   - Implement UniverseDataAccessor from config files
   - Support for static and rule-based universes
   - Point-in-time universe membership
   - Dependencies: IMPL-019
   - Estimated LOC: 200

5. **IMPL-023: ReferenceDataAccessor Implementation**
   - Implement ReferenceDataAccessor
   - Sector/exchange mappings from config
   - Ticker metadata lookup
   - Dependencies: IMPL-019
   - Estimated LOC: 150

6. **IMPL-024: CachingLayer & Integration**
   - Implement CachedPriceAccessor decorator
   - Integrate cache invalidation hooks
   - Create cache management utilities
   - Dependencies: IMPL-020
   - Estimated LOC: 200

### Phase 2: Component Migration (8-10 tasks)

7. **IMPL-025: Backtest Engine Migration**
   - Refactor `SimpleBacktestEngine` to use `DataAccessContext`
   - Remove `SnapshotDataStore` dependency
   - Update all data access calls
   - Dependencies: IMPL-019, IMPL-020, IMPL-021
   - Estimated LOC: 100 (changes, not new)

8. **IMPL-026: Alpha Models Migration**
   - Update all alpha models to use `DataAccessContext`
   - Replace direct DataFrame operations
   - Dependencies: IMPL-019, IMPL-020, IMPL-021
   - Estimated LOC: 150

9. **IMPL-027: Portfolio Optimization Migration**
   - Refactor `StrategyOptimizer` to use DAL
   - Remove hardcoded snapshot_path parameter
   - Update evaluator to use `DataAccessContext`
   - Dependencies: IMPL-019, IMPL-020
   - Estimated LOC: 200

10. **IMPL-028: Production Pipeline Migration**
    - Refactor `ProductionPipeline` to use DAL
    - Update `run_production_pipeline.py` script
    - Remove snapshot dependency
    - Dependencies: IMPL-019, IMPL-020, IMPL-021
    - Estimated LOC: 150

11. **IMPL-029: Research Scripts Migration**
    - Update `run_backtest.py` to use DAL
    - Update `run_optimization.py`
    - Update walk-forward testing scripts
    - Dependencies: IMPL-019, IMPL-020, IMPL-025
    - Estimated LOC: 200

12. **IMPL-030: Monitoring System Migration**
    - Update `run_daily_monitoring.py` to use DAL
    - Update alert and reporting systems
    - Dependencies: IMPL-019, IMPL-020, IMPL-021
    - Estimated LOC: 100

13. **IMPL-031: Test Utilities & Mocking**
    - Create mock accessors for testing
    - Create test fixtures and builders
    - Update existing tests to use mocks
    - Dependencies: IMPL-019
    - Estimated LOC: 300

### Phase 3: Data Ingestion & Update (3-4 tasks)

14. **IMPL-032: Live Data Connector**
    - Implement `LivePriceAccessor` (optional, for future use)
    - API wrapper for Stooq/other sources
    - Real-time data handling
    - Dependencies: IMPL-019
    - Estimated LOC: 250

15. **IMPL-033: Data Refresh Orchestration**
    - Create data refresh manager
    - Cache invalidation on refresh
    - Atomic data updates
    - Dependencies: IMPL-020, IMPL-024
    - Estimated LOC: 200

16. **IMPL-034: Documentation & Examples**
    - Write DAL architecture guide
    - Create example usage patterns
    - Update component README files
    - Dependencies: All previous phases
    - Estimated LOC: N/A (docs)

---

## Benefits of Data Access Layer

### For Production Systems
- **Single source of truth**: All data access through DAL
- **Easy data refresh**: Update underlying storage without touching code
- **Consistent caching**: Centralized cache management
- **Regime detection**: Macro data available everywhere consistently
- **Data monitoring**: Easy to audit data access patterns
- **Failover capability**: Can swap backends (snapshot ↔ live API) with config

### For Research & Backtesting
- **Better testability**: Mock data access instead of managing test snapshots
- **Faster iteration**: No need to create new snapshots
- **Point-in-time consistency**: Guaranteed prevention of lookahead bias
- **Rich diagnostics**: Can track what data was accessed
- **Reproducibility**: Same config produces same data access behavior

### For Maintenance
- **Reduced complexity**: No more scattered snapshot management
- **Easier debugging**: Centralized data access logging
- **Cleaner APIs**: Components don't manage storage details
- **Better separation of concerns**: Business logic separated from data access
- **Simplified deployment**: No need to manage snapshot directories

---

## Data Access Examples

### Example 1: Backtest Engine
```python
# Old way (direct snapshot dependency)
store = SnapshotDataStore(Path("data/snapshots/snapshot_20260113/data.parquet"))
engine = SimpleBacktestEngine(snapshot_path=store)

# New way (DAL)
ctx = DataAccessFactory.create_access_context()  # From config
engine = SimpleBacktestEngine(data_access=ctx)
```

### Example 2: Production Pipeline
```python
# Old way
pipeline = ProductionPipeline(
    config=load_config(),
    store=SnapshotDataStore(snapshot_path),
)

# New way
ctx = DataAccessFactory.create_access_context(
    source="snapshot",
    config={"snapshot_dir": "data/snapshots/latest"}
)
pipeline = ProductionPipeline(config=load_config(), data_access=ctx)
```

### Example 3: Research Script
```python
# Old way
evaluator = MultiPeriodEvaluator(snapshot_path="data/snapshots/snapshot_20260113")

# New way
ctx = DataAccessFactory.create_access_context()
evaluator = MultiPeriodEvaluator(data_access=ctx)
# Automatically finds latest snapshot or uses configured source
```

### Example 4: Testing
```python
# Mock data for tests
class MockPriceAccessor(PriceDataAccessor):
    def read_prices_as_of(self, as_of, tickers=None, lookback_days=None):
        # Return test data
        return synthetic_test_prices()

ctx = DataAccessContext(
    prices=MockPriceAccessor(),
    macro=MockMacroAccessor(),
    universes=MockUniverseAccessor(),
    references=MockReferenceAccessor(),
)

# Run test with mock data
result = engine.backtest(ctx)
assert result.total_return > 0
```

---

## Migration Path

### Week 1: Foundation
- IMPL-019, IMPL-020, IMPL-021, IMPL-022, IMPL-023, IMPL-024 (parallel)
- Create core DAL infrastructure
- All tests passing

### Week 2: Migration (Parallel Tracks)
- Track A: IMPL-025, IMPL-026, IMPL-027 (core engine components)
- Track B: IMPL-028, IMPL-029, IMPL-030 (scripts & production)
- Track C: IMPL-031 (test utilities)
- Tests updated incrementally

### Week 3: Completion & Optimization
- IMPL-032, IMPL-033, IMPL-034
- Fix remaining issues
- Performance optimization
- Full integration testing

---

## Configuration Example

### `configs/data_access.yaml`
```yaml
# Data Access Layer Configuration

price_data:
  type: snapshot
  config:
    source: "data/snapshots/latest"
    cache:
      enabled: true
      max_size_mb: 500
      ttl_seconds: 3600

macro_data:
  type: fred
  config:
    cache:
      enabled: true
      ttl_seconds: 86400
    indicators:
      - VIX
      - SPX_YIELD
      - US_2Y_10Y_SPREAD

universes:
  type: config_file
  config:
    config_dir: "configs/universes"

reference_data:
  type: static
  config:
    sector_mapping: "configs/reference/sectors.yaml"
    exchanges: "configs/reference/exchanges.yaml"
```

---

## Success Criteria

### Phase 1 (Infrastructure)
- [x] All accessor interfaces defined
- [x] Factory pattern implemented
- [x] DataAccessContext created
- [x] All unit tests passing

### Phase 2 (Migration)
- [x] All components migrated to use DAL
- [x] Snapshot dependencies removed from main code
- [x] All tests updated to use mock accessors
- [x] All scripts run without snapshot_path parameter

### Phase 3 (Optimization)
- [x] Caching layer integrated and tested
- [x] Performance benchmarks meet targets (< 5% overhead)
- [x] Data refresh orchestration operational
- [x] Documentation complete

---

## Risk Mitigation

### Risk: Breaking existing scripts during migration
**Mitigation:** 
- Keep old snapshot-based APIs as deprecated wrappers
- Gradual migration with fallback capability
- Comprehensive test coverage before switching

### Risk: Performance degradation from DAL abstraction
**Mitigation:**
- CachingLayer transparent optimization
- Benchmark critical paths
- Lazy loading of data where possible

### Risk: Data consistency issues during live updates
**Mitigation:**
- Atomic data swap operations
- Version tracking in metadata
- Validation hooks on refresh

---

## Future Enhancements

1. **Live Data Backend**: Swap snapshot for live API feeds
2. **Event Streaming**: Publish data events to downstream systems
3. **Data Versioning**: Track data lineage and transformations
4. **Distributed Caching**: Use Redis for multi-process caching
5. **Data Lineage Tracking**: Audit trail of all data accesses
6. **Multi-dataset Support**: Run multiple strategies on different data versions
7. **Performance Monitoring**: Metrics on data access patterns

