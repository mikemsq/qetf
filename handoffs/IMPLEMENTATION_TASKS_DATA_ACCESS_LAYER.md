# Implementation Tasks: Data Access Layer

**Created:** January 18, 2026  
**Total Tasks:** 16  
**Total Estimated LOC:** 3,200+  
**Estimated Total Effort:** 40-45 hours

---

## Task Summary by Phase

### Phase 1: Core DAL Infrastructure (6 tasks)
**Duration:** 12-14 hours  
**Sequence:** Can execute all in parallel or sequential  
**Output:** Complete, testable DAL infrastructure with all accessor implementations

| Task ID | Title | Dependencies | Est. Hours | Est. LOC |
|---------|-------|--------------|-----------|----------|
| IMPL-019 | DAL Core Interfaces & Types | None | 3 | 400 |
| IMPL-020 | SnapshotPriceAccessor | IMPL-019 | 3 | 250 |
| IMPL-021 | FREDMacroAccessor | IMPL-019 | 3 | 300 |
| IMPL-022 | ConfigFileUniverseAccessor | IMPL-019 | 2 | 200 |
| IMPL-023 | ReferenceDataAccessor | IMPL-019 | 2 | 150 |
| IMPL-024 | CachingLayer & Integration | IMPL-020 | 2.5 | 200 |

### Phase 2: Component Migration (7 tasks)
**Duration:** 20-24 hours  
**Sequence:** Can execute in parallel or dependent order  
**Output:** All major components refactored to use DAL

| Task ID | Title | Dependencies | Est. Hours | Est. LOC |
|---------|-------|--------------|-----------|----------|
| IMPL-025 | Backtest Engine Migration | IMPL-019,020,021 | 2 | 100 |
| IMPL-026 | Alpha Models Migration | IMPL-019,020,021 | 2.5 | 150 |
| IMPL-027 | Portfolio Optimization Migration | IMPL-019,020 | 3 | 200 |
| IMPL-028 | Production Pipeline Migration | IMPL-019,020,021 | 2 | 150 |
| IMPL-029 | Research Scripts Migration | IMPL-019,020,025 | 3 | 200 |
| IMPL-030 | Monitoring System Migration | IMPL-019,020,021 | 2 | 100 |
| IMPL-031 | Test Utilities & Mocking | IMPL-019 | 4 | 300 |

### Phase 3: Data Ingestion & Finalization (3 tasks)
**Duration:** 8-10 hours  
**Sequence:** Sequential, depends on Phases 1-2  
**Output:** Complete system with data refresh capability and documentation

| Task ID | Title | Dependencies | Est. Hours | Est. LOC |
|---------|-------|--------------|-----------|----------|
| IMPL-032 | Live Data Connector | IMPL-019 | 3 | 250 |
| IMPL-033 | Data Refresh Orchestration | IMPL-020,024 | 2.5 | 200 |
| IMPL-034 | Documentation & Examples | All | 2.5 | N/A |

---

## Detailed Task Specifications

---

## IMPL-019: DAL Core Interfaces & Types

**Priority:** P0 (CRITICAL - Blocks all others)  
**Effort:** 3 hours | 400 LOC  
**Dependencies:** None

### Overview
Create the fundamental type definitions and abstract interfaces for the entire DAL system. This is the foundation that all other components depend on.

### Deliverables

#### New File: `src/quantetf/data/access/__init__.py`
Initialize DAL package and expose public interfaces.

#### New File: `src/quantetf/data/access/types.py`
Define all enums, dataclasses, and types:

**Content:**
- `Regime` enum: RISK_ON, ELEVATED_VOL, HIGH_VOL, RECESSION_WARNING, UNKNOWN
- `TickerMetadata` dataclass: ticker, name, sector, exchange, currency
- `ExchangeInfo` dataclass: name, trading_hours, holidays, settlement_days
- `DataAccessMetadata` dataclass: source, timestamp, lookback_date, data_quality_score

#### New File: `src/quantetf/data/access/abstract.py`
Define abstract base classes for all accessors:

**Content:**
- `PriceDataAccessor(ABC)`:
  - `read_prices_as_of(as_of, tickers, lookback_days) → DataFrame`
  - `read_ohlcv_range(start, end, tickers) → DataFrame`
  - `get_latest_price_date() → Timestamp`
  - `validate_data_availability(tickers, as_of) → dict`

- `MacroDataAccessor(ABC)`:
  - `read_macro_indicator(indicator, as_of, lookback_days) → DataFrame`
  - `get_regime(as_of) → Regime`
  - `get_available_indicators() → list[str]`

- `UniverseDataAccessor(ABC)`:
  - `get_universe(universe_name) → list[str]`
  - `get_universe_as_of(universe_name, as_of) → list[str]`
  - `list_available_universes() → list[str]`

- `ReferenceDataAccessor(ABC)`:
  - `get_ticker_info(ticker) → TickerMetadata`
  - `get_sector_mapping() → dict[str, str]`
  - `get_exchange_info() → dict[str, ExchangeInfo]`

#### New File: `src/quantetf/data/access/context.py`
Create unified context object:

**Content:**
- `DataAccessContext(frozen=True)`:
  - `prices: PriceDataAccessor`
  - `macro: MacroDataAccessor`
  - `universes: UniverseDataAccessor`
  - `references: ReferenceDataAccessor`

#### New File: `src/quantetf/data/access/factory.py`
Create factory for instantiating accessors:

**Content:**
- `DataAccessFactory` class:
  - Static methods: `create_price_accessor(source, config)`
  - Static methods: `create_macro_accessor(source, config)`
  - Static methods: `create_universe_accessor(config)`
  - Static methods: `create_reference_accessor(config)`
  - Static methods: `create_context(config_file)` - convenience wrapper

### Testing Requirements

**File:** `tests/data/test_dal_interfaces.py`

Test cases:
- [ ] All abstract methods defined
- [ ] DataAccessContext can be instantiated with valid accessors
- [ ] Factory creates valid accessor instances
- [ ] Enums have expected values
- [ ] Dataclasses can be serialized/deserialized

### Acceptance Criteria

- [ ] No circular imports
- [ ] All abstract methods documented with docstrings
- [ ] Type hints complete and correct
- [ ] Factory pattern correctly implemented
- [ ] 100% of new code passes type checking (mypy)
- [ ] All tests pass

---

## IMPL-020: SnapshotPriceAccessor Implementation

**Priority:** P0 (CRITICAL)  
**Effort:** 3 hours | 250 LOC  
**Dependencies:** IMPL-019

### Overview
Implement concrete PriceDataAccessor that wraps existing SnapshotDataStore. This enables backward compatibility while accessing data through the DAL interface.

### Deliverables

#### New File: `src/quantetf/data/access/snapshot_price.py`

**Content:**

`SnapshotPriceAccessor(PriceDataAccessor)`:
- Wraps `SnapshotDataStore` from existing code
- Implements point-in-time data access
- Handles ticker filtering and validation
- Provides metadata about snapshot

**Methods:**
- `__init__(snapshot_path: Path, cache_enabled: bool = True)`
- `read_prices_as_of(as_of, tickers, lookback_days) → DataFrame`
  - Returns only data where `index < as_of` (strict inequality)
  - Prevents lookahead bias
  - Filters to requested tickers
  - Applies lookback window if specified
- `read_ohlcv_range(start, end, tickers) → DataFrame`
  - Returns OHLCV for closed date range [start, end]
  - Supports efficient date filtering
- `get_latest_price_date() → Timestamp`
  - Returns maximum date in snapshot
- `validate_data_availability(tickers, as_of) → dict[str, bool]`
  - Returns True/False for each ticker
  - Checks if data exists before as_of date
- `get_available_tickers() → list[str]`
  - List all tickers in snapshot

**Key Design Points:**
- Use composition (wrap SnapshotDataStore) not inheritance
- Cache latest_date at init time
- Lazy-load snapshot data on first access (if possible)
- Store metadata about snapshot quality/currency

### Testing Requirements

**File:** `tests/data/access/test_snapshot_price.py`

Test cases:
- [ ] `read_prices_as_of()` returns point-in-time data
- [ ] Data before as_of is included, data on/after as_of is excluded
- [ ] Ticker filtering works correctly
- [ ] Lookback window applied correctly
- [ ] Missing tickers raise appropriate error
- [ ] `get_latest_price_date()` returns correct value
- [ ] `validate_data_availability()` works for all tickers
- [ ] Data shape is correct (MultiIndex columns)
- [ ] Empty result raises error instead of returning empty DataFrame
- [ ] Performance: can load 10-year snapshot with millions of rows

### Acceptance Criteria

- [ ] All abstract methods implemented
- [ ] Point-in-time data access guaranteed (no lookahead)
- [ ] All tests pass
- [ ] Backward compatible with SnapshotDataStore
- [ ] Performance acceptable (< 100ms for typical queries)
- [ ] Type hints complete

---

## IMPL-021: FREDMacroAccessor Implementation

**Priority:** P0 (CRITICAL)  
**Effort:** 3 hours | 300 LOC  
**Dependencies:** IMPL-019

### Overview
Implement concrete MacroDataAccessor that integrates with existing MacroDataLoader. Provides regime detection and macro indicator access through DAL interface.

### Deliverables

#### New File: `src/quantetf/data/access/fred_macro.py`

**Content:**

`FREDMacroAccessor(MacroDataAccessor)`:
- Wraps existing `MacroDataLoader` from `src/quantetf/data/macro_loader.py`
- Implements regime detection
- Caches frequently accessed indicators
- Provides point-in-time macro data

**Methods:**
- `__init__(macro_loader: MacroDataLoader, cache_ttl_seconds: int = 86400)`
- `read_macro_indicator(indicator, as_of, lookback_days) → DataFrame`
  - Reads from MacroDataLoader
  - Returns data up to as_of (point-in-time)
  - Supports all FRED indicators: VIX, yields, spreads, etc.
  - Applies lookback window if specified
- `get_regime(as_of) → Regime`
  - Detects current regime using multiple indicators
  - Uses existing `RegimeDetector` from codebase
  - Returns: RISK_ON, ELEVATED_VOL, HIGH_VOL, RECESSION_WARNING, or UNKNOWN
- `get_available_indicators() → list[str]`
  - Returns list of available macro indicators
  - From MacroDataLoader configuration
- `get_indicator_metadata(indicator) → dict`
  - Returns info about indicator (source, units, frequency)

**Regime Detection Logic:**
```
If VIX > 30 and US_2Y_10Y_SPREAD < -0.5:
  → RECESSION_WARNING
Else if VIX > 30:
  → HIGH_VOL
Else if VIX > 20:
  → ELEVATED_VOL
Else if US_2Y_10Y_SPREAD < 0:
  → ELEVATED_VOL
Else:
  → RISK_ON
```

### Testing Requirements

**File:** `tests/data/access/test_fred_macro.py`

Test cases:
- [ ] `read_macro_indicator()` returns correct data
- [ ] Point-in-time filtering works (data up to as_of only)
- [ ] Lookback window applied correctly
- [ ] Regime detection returns expected values
- [ ] Regime detection handles edge cases (VIX = 30 exactly)
- [ ] `get_available_indicators()` returns non-empty list
- [ ] Caching works (same query returns same object)
- [ ] Cache TTL expiration works
- [ ] Missing indicator raises error
- [ ] Integration with MacroDataLoader works

### Acceptance Criteria

- [ ] All abstract methods implemented
- [ ] Regime detection accurate and tested
- [ ] Caching functional and tested
- [ ] Point-in-time access guaranteed
- [ ] All tests pass
- [ ] Type hints complete
- [ ] Integration with existing MacroDataLoader verified

---

## IMPL-022: ConfigFileUniverseAccessor

**Priority:** P0  
**Effort:** 2 hours | 200 LOC  
**Dependencies:** IMPL-019

### Overview
Implement concrete UniverseDataAccessor that reads universe definitions from YAML config files. Supports static (fixed) and dynamic (rule-based) universes with point-in-time support.

### Deliverables

#### New File: `src/quantetf/data/access/universe.py`

**Content:**

`ConfigFileUniverseAccessor(UniverseDataAccessor)`:
- Reads universe definitions from `configs/universes/*.yaml`
- Supports static universes (fixed ticker lists)
- Supports graduated universe (tickers added over time)
- Caches universe definitions

**Methods:**
- `__init__(config_dir: Path, cache: bool = True)`
- `get_universe(universe_name) → list[str]`
  - Returns latest/current universe tickers
  - Case-insensitive name matching
- `get_universe_as_of(universe_name, as_of) → list[str]`
  - Returns universe membership at specific date
  - Handles graduated addition of tickers
- `list_available_universes() → list[str]`
  - Returns names of all defined universes
- `get_universe_metadata(universe_name) → dict`
  - Returns metadata: description, size, last_updated

**Config File Format:**

Static universe example (`tier1_initial_20.yaml`):
```yaml
name: tier1_initial_20
description: "Core 20 highly liquid ETFs"
type: static
tickers:
  - SPY
  - QQQ
  - IWM
  - ... (20 total)
```

Graduated universe example:
```yaml
name: tier4_broad_200
description: "200-ETF broad universe"
type: graduated
tickers:
  - ticker: SPY
    added_date: 2016-01-13
  - ticker: QQQ
    added_date: 2016-01-13
  - ticker: XYZ_NEW
    added_date: 2023-01-01
```

### Testing Requirements

**File:** `tests/data/access/test_universe.py`

Test cases:
- [ ] `get_universe()` returns correct static universe
- [ ] `get_universe_as_of()` returns correct historical universe
- [ ] Graduated universe includes only tickers added by as_of date
- [ ] Missing universe raises error
- [ ] `list_available_universes()` returns all defined universes
- [ ] Case-insensitive universe name matching
- [ ] Caching works correctly
- [ ] Invalid config file raises error
- [ ] Empty universe raises warning

### Acceptance Criteria

- [ ] All abstract methods implemented
- [ ] Supports both static and graduated universes
- [ ] Point-in-time universe membership correct
- [ ] All tests pass
- [ ] Type hints complete
- [ ] Works with existing universe configs

---

## IMPL-023: ReferenceDataAccessor Implementation

**Priority:** P1  
**Effort:** 2 hours | 150 LOC  
**Dependencies:** IMPL-019

### Overview
Implement concrete ReferenceDataAccessor for static/slow-changing reference data like sector mappings, ticker metadata, exchange information.

### Deliverables

#### New File: `src/quantetf/data/access/reference.py`

**Content:**

`StaticReferenceDataAccessor(ReferenceDataAccessor)`:
- Reads reference data from YAML config files
- Supports sector mappings, ticker metadata, exchange info
- Caches all data in memory

**Methods:**
- `__init__(config_dir: Path)`
- `get_ticker_info(ticker) → TickerMetadata`
  - Returns metadata for ticker
  - Fields: ticker, name, sector, exchange, currency
  - Raises error if ticker not found
- `get_sector_mapping() → dict[str, str]`
  - Returns ticker → sector mapping for all tickers
- `get_exchange_info() → dict[str, ExchangeInfo]`
  - Returns exchange → metadata mapping
  - Includes trading hours, holidays, etc.
- `get_sectors() → list[str]`
  - Returns unique list of sectors
- `get_tickers_by_sector(sector) → list[str]`
  - Returns all tickers in a sector

**Config File Structure:**

`configs/reference/tickers.yaml`:
```yaml
tickers:
  SPY:
    name: "SPDR S&P 500 ETF Trust"
    sector: "Broad Market"
    exchange: "NASDAQ"
    currency: "USD"
  QQQ:
    name: "Invesco QQQ Trust"
    sector: "Technology"
    exchange: "NASDAQ"
    currency: "USD"
```

`configs/reference/exchanges.yaml`:
```yaml
exchanges:
  NASDAQ:
    name: "NASDAQ"
    trading_hours: "09:30-16:00 EST"
    timezone: "US/Eastern"
    settlement_days: 2
```

### Testing Requirements

**File:** `tests/data/access/test_reference.py`

Test cases:
- [ ] `get_ticker_info()` returns correct metadata
- [ ] Missing ticker raises error
- [ ] `get_sector_mapping()` returns complete mapping
- [ ] `get_exchange_info()` returns correct info
- [ ] `get_sectors()` returns unique list
- [ ] `get_tickers_by_sector()` returns correct tickers
- [ ] Invalid config raises error

### Acceptance Criteria

- [ ] All abstract methods implemented
- [ ] All tests pass
- [ ] Type hints complete
- [ ] Works with existing reference data

---

## IMPL-024: CachingLayer & Integration

**Priority:** P0  
**Effort:** 2.5 hours | 200 LOC  
**Dependencies:** IMPL-020

### Overview
Implement transparent caching layer that wraps accessors to improve performance. Provides LRU in-memory caching for prices and TTL-based caching for macro data.

### Deliverables

#### New File: `src/quantetf/data/access/caching.py`

**Content:**

`CachedPriceAccessor(PriceDataAccessor)`:
- Decorator pattern wrapping `PriceDataAccessor`
- LRU in-memory cache for recent queries
- Optional persistent snapshot caching
- Automatic cache invalidation

**Methods:**
- `__init__(inner: PriceDataAccessor, max_cache_mb: int = 500, snapshot_cache_dir: Optional[Path] = None)`
- Cache all methods from PriceDataAccessor
- Override `read_prices_as_of()` and `read_ohlcv_range()` with caching logic
- Add cache management methods:
  - `clear_cache()`
  - `get_cache_stats() → dict`
  - `invalidate_after(date: Timestamp)` - clear queries on/after date

**Cache Key Strategy:**
- Price cache key: `(as_of, tuple(sorted(tickers)), lookback_days, method)`
- Use MD5 hash of tickers to avoid long keys

`CachedMacroAccessor(MacroDataAccessor)`:
- Similar pattern for macro data
- TTL-based cache (default 24 hours)
- Per-indicator cache TTL configuration

#### New File: `src/quantetf/data/access/cache_manager.py`

**Content:**

`CacheManager`:
- Centralized cache management
- Track cache stats across all accessors
- Methods:
  - `clear_all_caches()`
  - `get_global_stats() → dict`
  - `configure_ttl(accessor_type, ttl_seconds)`

### Testing Requirements

**File:** `tests/data/access/test_caching.py`

Test cases:
- [ ] Cache hit returns same object
- [ ] Cache miss calls inner accessor
- [ ] Cache respects size limits (LRU eviction)
- [ ] TTL expiration works for macro cache
- [ ] `clear_cache()` removes all entries
- [ ] `invalidate_after()` removes appropriate entries
- [ ] Cache stats tracked correctly
- [ ] Performance: cache reduces latency by 90%+
- [ ] Memory usage within configured limits

### Acceptance Criteria

- [ ] Transparent caching works
- [ ] All tests pass
- [ ] Performance goals met (< 5% overhead with cache enabled)
- [ ] Type hints complete
- [ ] Integration with existing accessors

---

## IMPL-025: Backtest Engine Migration

**Priority:** P1  
**Effort:** 2 hours | 100 LOC changes  
**Dependencies:** IMPL-019, IMPL-020, IMPL-021

### Overview
Refactor `SimpleBacktestEngine` to use `DataAccessContext` instead of taking `snapshot_path` parameter. Remove SnapshotDataStore dependency from core engine.

### Current State
```python
class SimpleBacktestEngine:
    def __init__(self, snapshot_path: Union[str, Path], ...):
        self.store = SnapshotDataStore(snapshot_path)
```

### Target State
```python
class SimpleBacktestEngine:
    def __init__(self, data_access: DataAccessContext, ...):
        self.data_access = data_access
```

### Changes Required

**File:** `src/quantetf/backtest/simple_engine.py`

Changes:
- [ ] Remove `snapshot_path` parameter from `__init__`
- [ ] Add `data_access: DataAccessContext` parameter
- [ ] Replace all `self.store` calls with `self.data_access.prices`
- [ ] Update backtest logic to use accessor methods
- [ ] Remove SnapshotDataStore import
- [ ] Update all docstrings

### Testing Requirements

**File:** `tests/backtest/test_simple_engine.py`

Changes:
- [ ] Update all tests to pass DataAccessContext
- [ ] Create mock accessors for tests
- [ ] Verify backward compatibility if needed
- [ ] Performance tests pass

### Acceptance Criteria

- [ ] No snapshot_path parameter in engine
- [ ] Uses DataAccessContext exclusively
- [ ] All tests pass
- [ ] No performance degradation
- [ ] SnapshotDataStore no longer imported

---

## IMPL-026: Alpha Models Migration

**Priority:** P1  
**Effort:** 2.5 hours | 150 LOC changes  
**Dependencies:** IMPL-019, IMPL-020, IMPL-021

### Overview
Update all alpha models (Momentum, TrendFiltered, Dual, ValueMomentum) to use `DataAccessContext` instead of directly accessing SnapshotDataStore.

### Affected Files
- `src/quantetf/alpha/momentum.py`
- `src/quantetf/alpha/trend_filtered.py`
- `src/quantetf/alpha/dual_momentum.py`
- `src/quantetf/alpha/value_momentum.py`

### Changes Required

For each alpha model:
- [ ] Change `score()` method signature to accept `data_access: DataAccessContext`
- [ ] Replace `store.read_prices()` with `data_access.prices.read_prices_as_of()`
- [ ] Update any macro data access to use `data_access.macro`
- [ ] Update docstrings and examples
- [ ] Remove SnapshotDataStore imports

### Testing Requirements

**Files:** `tests/alpha/test_*.py`

Changes:
- [ ] Update all tests to create and pass DataAccessContext
- [ ] Create mock accessors
- [ ] Verify signal quality unchanged
- [ ] Performance tests pass

### Acceptance Criteria

- [ ] All alpha models accept DataAccessContext
- [ ] No SnapshotDataStore imports in alpha module
- [ ] All tests pass
- [ ] Signal quality metrics identical to before
- [ ] No performance degradation

---

## IMPL-027: Portfolio Optimization Migration

**Priority:** P1  
**Effort:** 3 hours | 200 LOC changes  
**Dependencies:** IMPL-019, IMPL-020

### Overview
Refactor portfolio optimization system (StrategyOptimizer, MultiPeriodEvaluator) to use DataAccessContext. Remove hardcoded snapshot_path parameters.

### Affected Files
- `src/quantetf/optimization/optimizer.py`
- `src/quantetf/optimization/evaluator.py`
- `scripts/run_optimization.py`

### Changes Required

**StrategyOptimizer:**
- [ ] Remove `snapshot_path` parameter
- [ ] Accept `data_access: DataAccessContext` instead
- [ ] Pass data_access to all sub-components
- [ ] Update logging and diagnostics

**MultiPeriodEvaluator:**
- [ ] Remove `snapshot_path` parameter  
- [ ] Accept `data_access: DataAccessContext`
- [ ] Use accessor methods for all data reads
- [ ] Update docstrings

**run_optimization.py script:**
- [ ] Remove `--snapshot` argument
- [ ] Use factory to create DataAccessContext
- [ ] Get snapshot from config if needed

### Testing Requirements

**Files:** `tests/optimization/test_*.py`

Changes:
- [ ] All tests create DataAccessContext
- [ ] Mock accessors for isolated testing
- [ ] Verify optimization results unchanged
- [ ] Performance tests pass

### Acceptance Criteria

- [ ] No snapshot_path in API
- [ ] Uses DataAccessContext throughout
- [ ] Optimization results identical to before
- [ ] All tests pass
- [ ] run_optimization.py works without snapshot argument

---

## IMPL-028: Production Pipeline Migration

**Priority:** P1  
**Effort:** 2 hours | 150 LOC changes  
**Dependencies:** IMPL-019, IMPL-020, IMPL-021

### Overview
Refactor `ProductionPipeline` and related components to use DataAccessContext. Remove snapshot-based data access.

### Affected Files
- `src/quantetf/production/pipeline.py`
- `scripts/run_production_pipeline.py`
- Production config system

### Changes Required

**ProductionPipeline:**
- [ ] Remove `store` parameter or change to `data_access: DataAccessContext`
- [ ] Update all data access to use context
- [ ] Use `data_access.macro` for regime detection
- [ ] Update pre-trade checks to use context

**run_production_pipeline.py:**
- [ ] Remove `--snapshot` argument
- [ ] Create DataAccessContext from config
- [ ] Pass context to pipeline
- [ ] Update usage documentation

**Production config:**
- [ ] Add data_access section to config
- [ ] Define default data source

### Testing Requirements

**Files:** `tests/production/test_*.py`

Changes:
- [ ] All tests create DataAccessContext
- [ ] Mock accessors for testing
- [ ] Verify portfolio recommendations unchanged
- [ ] Risk overlay behavior unchanged

### Acceptance Criteria

- [ ] Pipeline accepts DataAccessContext
- [ ] No snapshot dependency in pipeline
- [ ] run_production_pipeline.py works
- [ ] All tests pass
- [ ] Portfolio recommendations identical

---

## IMPL-029: Research Scripts Migration

**Priority:** P1  
**Effort:** 3 hours | 200 LOC changes  
**Dependencies:** IMPL-019, IMPL-020, IMPL-025

### Overview
Update all research and analysis scripts to use DataAccessContext. These are often entry points for users, so they should have clean interfaces.

### Affected Files
- `scripts/run_backtest.py`
- `scripts/walk_forward_test.py`
- `scripts/compare_strategies.py`
- `scripts/run_experiment.py`
- Other analysis scripts

### Changes Required

**For each script:**
- [ ] Remove `--snapshot` argument
- [ ] Add option to specify data source (config file or default)
- [ ] Create DataAccessContext using factory
- [ ] Pass context to all components
- [ ] Update usage docs and help text

**Example: run_backtest.py**

Before:
```bash
python scripts/run_backtest.py --snapshot data/snapshots/snapshot_20260113
```

After:
```bash
python scripts/run_backtest.py  # Uses config/data_access.yaml by default
# or
python scripts/run_backtest.py --data-config custom_data.yaml
```

### Testing Requirements

**File:** `tests/scripts/test_*.py`

Changes:
- [ ] Script arguments work correctly
- [ ] DataAccessContext created properly
- [ ] All outputs generated correctly
- [ ] Help text accurate

### Acceptance Criteria

- [ ] All scripts work without snapshot argument
- [ ] Scripts have clean, simple interfaces
- [ ] Help text updated
- [ ] All tests pass
- [ ] Results identical to before

---

## IMPL-030: Monitoring System Migration

**Priority:** P1  
**Effort:** 2 hours | 100 LOC changes  
**Dependencies:** IMPL-019, IMPL-020, IMPL-021

### Overview
Update monitoring and alerting systems to use DataAccessContext. Includes daily monitoring, alert generation, and performance tracking.

### Affected Files
- `scripts/run_daily_monitoring.py`
- `src/quantetf/monitoring/` module
- Alert and reporting systems

### Changes Required

- [ ] MonitoringSystem accepts `data_access: DataAccessContext`
- [ ] All data reads use accessor methods
- [ ] Remove snapshot dependencies
- [ ] Update scripts to create DataAccessContext
- [ ] Update monitoring config format if needed

### Testing Requirements

- [ ] Monitoring functionality unchanged
- [ ] Alerts generated correctly
- [ ] Reports generated with current data

### Acceptance Criteria

- [ ] Monitoring uses DataAccessContext
- [ ] No snapshot dependencies
- [ ] All tests pass
- [ ] Monitoring functionality verified

---

## IMPL-031: Test Utilities & Mocking

**Priority:** P1  
**Effort:** 4 hours | 300 LOC  
**Dependencies:** IMPL-019

### Overview
Create comprehensive test utilities: mock accessors, test fixtures, and builders for creating test data. This enables easy testing without managing test snapshots.

### Deliverables

#### New File: `tests/data/access/mocks.py`

**Content:**

Mock accessor classes for testing:

`MockPriceAccessor(PriceDataAccessor)`:
- Takes synthetic price data as input
- Supports configurable date range
- Useful for unit tests

`MockMacroAccessor(MacroDataAccessor)`:
- Returns synthetic macro data
- Supports configurable regimes
- Useful for regime-dependent tests

`MockUniverseAccessor(UniverseDataAccessor)`:
- Returns configurable universe lists
- Supports historical universe membership
- Useful for universe-dependent tests

`MockReferenceAccessor(ReferenceDataAccessor)`:
- Returns synthetic reference data
- Configurable sector mappings
- Useful for sector/exchange tests

#### New File: `tests/data/access/builders.py`

**Content:**

Builder classes for constructing test data:

`SyntheticPriceDataBuilder`:
- Create synthetic OHLCV data
- Specify date range, tickers
- Realistic or controlled price movements
- Method chaining for fluent API

Example:
```python
prices = (SyntheticPriceDataBuilder()
    .date_range('2023-01-01', '2023-12-31')
    .tickers(['SPY', 'QQQ'])
    .random_walk()  # or .constant(), .trending(), etc.
    .build())
```

`SyntheticMacroDataBuilder`:
- Create synthetic macro data
- Specify indicators
- Return as DataFrame

`TestDataFixture`:
- Pre-built common test datasets
- S&P 500 sample data
- VIX sample data
- Standard universes

#### New File: `tests/conftest.py`

**Content:**

Pytest fixtures for common test setups:
- `mock_access_context` fixture
- `synthetic_price_data` fixture
- `test_universes` fixture
- `test_reference_data` fixture

### Testing Requirements

**File:** `tests/test_utilities.py`

Test cases:
- [ ] Mock accessors return configured data
- [ ] Builders create valid synthetic data
- [ ] Fixtures provide consistent test data
- [ ] Mocks work with all accessor methods

### Acceptance Criteria

- [ ] All mock accessors implemented
- [ ] Builders create valid data
- [ ] Fixtures available and documented
- [ ] Used in IMPL-025+ migrations
- [ ] Simplifies test writing

---

## IMPL-032: Live Data Connector

**Priority:** P2  
**Effort:** 3 hours | 250 LOC  
**Dependencies:** IMPL-019

### Overview
Implement optional `LivePriceAccessor` for future use with live data feeds (Stooq API, etc.). Not required for initial deployment but enables future real-time capabilities.

### Deliverables

#### New File: `src/quantetf/data/access/live_price.py`

**Content:**

`LivePriceAccessor(PriceDataAccessor)`:
- Fetches live data from Stooq or other API
- Caches recent data locally
- Falls back to cached data if API unavailable
- Implements rate limiting

**Methods:**
- `__init__(api_key: Optional[str] = None, cache_dir: Optional[Path] = None)`
- All standard PriceDataAccessor methods
- Additional `fetch_latest() → DataFrame` for real-time data

**Integration Points:**
- Uses existing `StooqConnector` from `src/quantetf/data/connectors.py`
- Falls back to snapshot if API fails
- Caches locally to disk

### Testing Requirements

- [ ] Live API calls work correctly
- [ ] Caching works
- [ ] Fallback to snapshot works
- [ ] Rate limiting works
- [ ] Error handling for API failures

### Acceptance Criteria

- [ ] Live price accessor implemented
- [ ] Passes all tests
- [ ] Suitable for future production use
- [ ] Documentation complete

---

## IMPL-033: Data Refresh Orchestration

**Priority:** P2  
**Effort:** 2.5 hours | 200 LOC  
**Dependencies:** IMPL-020, IMPL-024

### Overview
Create data refresh orchestration system. Manages updating underlying data while maintaining cache consistency and data integrity.

### Deliverables

#### New File: `src/quantetf/data/refresh_manager.py`

**Content:**

`DataRefreshManager`:
- Coordinates data updates
- Manages cache invalidation
- Ensures atomic updates
- Provides rollback capability

**Methods:**
- `refresh_price_data(snapshot_path: Path) → bool`
  - Load new snapshot
  - Validate data quality
  - Swap with current snapshot
  - Invalidate caches
- `schedule_refresh(cron_expr: str, snapshot_dir: Path)`
  - Schedule periodic refreshes
- `get_last_refresh() → pd.Timestamp`
  - Track when data was last updated
- `validate_data_quality(snapshot: Path) → ValidationResult`
  - Check data completeness, date range, etc.

#### Integration Points
- Works with `CachedPriceAccessor`
- Invalidates caches on refresh
- Logs refresh operations

### Testing Requirements

- [ ] Refresh updates accessor state
- [ ] Caches invalidated correctly
- [ ] Data quality validated
- [ ] Rollback works
- [ ] Atomic updates guaranteed
- [ ] Scheduling works

### Acceptance Criteria

- [ ] Data refresh system operational
- [ ] Cache consistency maintained
- [ ] All tests pass
- [ ] Ready for production use

---

## IMPL-034: Documentation & Examples

**Priority:** P2  
**Effort:** 2.5 hours  
**Dependencies:** All

### Overview
Comprehensive documentation and examples for the DAL system. Helps users and developers understand and use the system correctly.

### Deliverables

#### New File: `docs/DATA_ACCESS_LAYER.md`

**Content:**
- Architecture overview
- Component descriptions
- Design patterns
- Configuration guide

#### Updated File: `docs/ARCHITECTURE.md`

**Content:**
- Updated with DAL architecture diagram
- Explain how it replaces snapshots
- Benefits and design decisions

#### New File: `src/quantetf/data/access/README.md`

**Content:**
- DAL module overview
- Quick start guide
- Common patterns
- API reference

#### New File: `examples/dal_examples.py`

**Content:**
- Example 1: Creating a DataAccessContext
- Example 2: Using accessors directly
- Example 3: Writing tests with mocks
- Example 4: Configuring data sources
- Example 5: Caching configuration

#### New File: `docs/MIGRATION_GUIDE.md`

**Content:**
- Step-by-step guide for migrating old code
- Before/after comparisons
- Troubleshooting section
- FAQ

#### Updated: `README.md`

**Changes:**
- Add section on data access layer
- Update architecture diagram
- Link to DAL docs

### Acceptance Criteria

- [ ] Architecture document complete
- [ ] README updated
- [ ] Examples cover common use cases
- [ ] Migration guide helpful
- [ ] All links work
- [ ] Diagrams clear and accurate

---

## Task Execution Guide

### Recommended Parallel Execution Strategy

#### Week 1: Foundation (Parallel Batch)
Execute all Phase 1 tasks in parallel or quick sequence:
- **Batch 1 (Hours 0-3):** IMPL-019 (foundation for all)
- **Batch 2 (Hours 3-9):** IMPL-020, IMPL-021, IMPL-022, IMPL-023 in parallel (no deps on each other)
- **Batch 3 (Hours 9-11):** IMPL-024 (depends on IMPL-020)

**Sequential dependency required:** IMPL-019 → everything else

#### Week 2: Migration (Parallel Tracks)

**Track A (Core Components):**
- IMPL-025 (Backtest Engine)
- IMPL-026 (Alpha Models)
- IMPL-027 (Optimization)

**Track B (Scripts & Production):**
- IMPL-028 (Production Pipeline)
- IMPL-029 (Research Scripts)
- IMPL-030 (Monitoring)

**Track C (Testing):**
- IMPL-031 (Test Utilities) - in parallel with other phases

**Dependencies:** All depend on IMPL-019,020,021 from Week 1

**Parallelization benefit:** Can run 3 tracks in parallel = 3x speedup

#### Week 3: Finalization
- IMPL-032 (Live Connector)
- IMPL-033 (Refresh Orchestration)
- IMPL-034 (Documentation)

**Sequential:** IMPL-032, IMPL-033 can run in parallel; IMPL-034 last

---

## Quality Gates & Verification

### Per-Task Quality Checklist

Each task must satisfy:
- [ ] All abstract methods implemented (if implementing interface)
- [ ] All unit tests pass (minimum 80% coverage)
- [ ] Type hints complete (mypy clean)
- [ ] Docstrings comprehensive
- [ ] No deprecated patterns or APIs
- [ ] Performance acceptable (benchmarks pass)
- [ ] Integration tests pass (where applicable)
- [ ] Backward compatibility maintained (if applicable)

### Integration Test Checklist

After all Phase 1-2 tasks complete:
- [ ] run_backtest.py works with DataAccessContext
- [ ] run_production_pipeline.py works
- [ ] walk_forward_test.py works
- [ ] Optimization produces same results
- [ ] Performance within acceptable bounds
- [ ] All 300+ existing tests pass
- [ ] New tests achieve 80%+ coverage

### Production Readiness Checklist

After Phase 3 complete:
- [ ] All code reviewed and approved
- [ ] All tests pass (100%)
- [ ] Documentation complete and reviewed
- [ ] Performance benchmarked
- [ ] Rollback procedure documented
- [ ] Deployment plan created
- [ ] Training materials prepared
- [ ] Monitoring and alerts in place

