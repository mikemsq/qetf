# Architecture Planning Summary: Data Access Layer Refactoring

**Created:** January 18, 2026  
**Role:** Architect/Planner  
**Status:** Architecture & planning complete - ready for coding agents

---

## Executive Summary

I have designed a complete Data Access Layer (DAL) architecture to replace the snapshot-based data model in QuantETF. The design provides:

✅ **Clean Separation of Concerns:** All data access mediated through specialized accessors  
✅ **Multiple Data Sources:** Snapshot files, live APIs, mock data for testing  
✅ **Transparent Caching:** Performance optimization without coupling  
✅ **Point-in-Time Guarantees:** Prevention of lookahead bias at every layer  
✅ **Testability:** Easy mocking and test data creation  
✅ **Production Ready:** Scalable, maintainable, well-documented  

---

## Problem Statement

**Current State:** Scripts directly depend on snapshot file paths
```
Script → SnapshotDataStore → File Path ("data/snapshots/snapshot_20260113")
```

**Issues:**
1. Tight coupling to file system
2. No centralized caching
3. Difficult to test (need actual snapshot files)
4. Hard to swap data sources (can't easily use live API)
5. No validation or quality control at access layer
6. Poor separation between data access and business logic

---

## Proposed Solution

**New Architecture:** All data accessed through DAL with unified interfaces
```
Script
    ↓ (uses)
[Backtest|Production|Research]
    ↓ (uses)
DataAccessContext
    ├─ PriceDataAccessor
    ├─ MacroDataAccessor
    ├─ UniverseDataAccessor
    └─ ReferenceDataAccessor
        ↓ (all wrapped by)
    CachingLayer
        ↓ (all backed by)
    [Snapshot|Live API|Mock] Storage
```

**Key Benefits:**
- Scripts never touch files directly
- Caching transparent and unified
- Easy to test with mocks
- Easy to add new data sources
- Clear data quality checkpoints
- Regime detection available everywhere

---

## Architecture Components

### 1. **PriceDataAccessor**
Unified interface for historical/current price data

**Methods:**
- `read_prices_as_of(as_of, tickers, lookback_days)` → DataFrame
- `read_ohlcv_range(start, end, tickers)` → DataFrame
- `get_latest_price_date()` → Timestamp
- `validate_data_availability(tickers, as_of)` → dict

**Implementations:**
- `SnapshotPriceAccessor` (wraps SnapshotDataStore)
- `LivePriceAccessor` (API feeds, future)
- `MockPriceAccessor` (testing)

### 2. **MacroDataAccessor**
Unified interface for macro indicators and regime detection

**Methods:**
- `read_macro_indicator(indicator, as_of, lookback)` → DataFrame
- `get_regime(as_of)` → Regime (RISK_ON | ELEVATED_VOL | HIGH_VOL | RECESSION_WARNING | UNKNOWN)
- `get_available_indicators()` → list[str]

**Implementations:**
- `FREDMacroAccessor` (integrates MacroDataLoader)
- `MockMacroAccessor` (testing)

### 3. **UniverseDataAccessor**
Unified interface for universe definitions (ticker sets)

**Methods:**
- `get_universe(name)` → list[str]
- `get_universe_as_of(name, as_of)` → list[str]
- `list_available_universes()` → list[str]

**Implementations:**
- `ConfigFileUniverseAccessor` (reads YAML configs)
- `MockUniverseAccessor` (testing)

### 4. **ReferenceDataAccessor**
Unified interface for static reference data

**Methods:**
- `get_ticker_info(ticker)` → TickerMetadata
- `get_sector_mapping()` → dict[str, str]
- `get_exchange_info()` → dict[str, ExchangeInfo]

**Implementations:**
- `StaticReferenceDataAccessor` (reads config files)
- `MockReferenceAccessor` (testing)

### 5. **DataAccessContext**
Convenient container holding all four accessors

```python
@dataclass(frozen=True)
class DataAccessContext:
    prices: PriceDataAccessor
    macro: MacroDataAccessor
    universes: UniverseDataAccessor
    references: ReferenceDataAccessor
```

Usage: Pass single object to all components instead of 4+ separate accessors

### 6. **DataAccessFactory**
Factory for creating configured accessor instances

**Methods:**
- `create_price_accessor(source, config)` → PriceDataAccessor
- `create_macro_accessor(source, config)` → MacroDataAccessor
- `create_universe_accessor(config)` → UniverseDataAccessor
- `create_reference_accessor(config)` → ReferenceDataAccessor
- `create_context(config_file)` → DataAccessContext (convenience)

### 7. **CachingLayer**
Transparent decorator providing LRU/TTL caching

**Wraps any accessor with:**
- LRU in-memory cache (configurable size)
- TTL-based expiration
- Persistent snapshot caching (optional)
- Automatic invalidation on data refresh

---

## Implementation Phases

### Phase 1: Infrastructure (6 tasks, 12-14 hours)
**Goal:** Build complete DAL foundation

1. **IMPL-019:** Core interfaces, types, factory (3h, 400 LOC)
2. **IMPL-020:** SnapshotPriceAccessor implementation (3h, 250 LOC)
3. **IMPL-021:** FREDMacroAccessor implementation (3h, 300 LOC)
4. **IMPL-022:** ConfigFileUniverseAccessor (2h, 200 LOC)
5. **IMPL-023:** StaticReferenceAccessor (2h, 150 LOC)
6. **IMPL-024:** Caching layer decorators (2.5h, 200 LOC)

**Parallelization:** IMPL-019 first, then 020-023 in parallel, then 024

**Output:** Complete, tested DAL infrastructure with all accessor implementations

---

### Phase 2: Component Migration (7 tasks, 20-24 hours)
**Goal:** Refactor all major components to use DAL

1. **IMPL-025:** Backtest engine migration (2h, 100 LOC changes)
2. **IMPL-026:** Alpha models migration (2.5h, 150 LOC changes)
3. **IMPL-027:** Portfolio optimization migration (3h, 200 LOC changes)
4. **IMPL-028:** Production pipeline migration (2h, 150 LOC changes)
5. **IMPL-029:** Research scripts migration (3h, 200 LOC changes)
6. **IMPL-030:** Monitoring system migration (2h, 100 LOC changes)
7. **IMPL-031:** Test utilities & mocking (4h, 300 LOC)

**Parallelization:** Can run parallel tracks:
- Track A: IMPL-025, 026, 027 (core engine)
- Track B: IMPL-028, 029, 030 (scripts/production)
- Track C: IMPL-031 (testing)

**Output:** All components refactored, using DAL exclusively

---

### Phase 3: Data Ingestion & Finalization (3 tasks, 8-10 hours)
**Goal:** Complete system with live data support and documentation

1. **IMPL-032:** Live data connector (3h, 250 LOC)
2. **IMPL-033:** Data refresh orchestration (2.5h, 200 LOC)
3. **IMPL-034:** Documentation & examples (2.5h, docs)

**Output:** Complete DAL system with all docs, ready for production

---

## Data Flow Examples

### Example 1: Backtest Execution (Before → After)

**Before (snapshot-based):**
```python
# Script hardcodes snapshot path
store = SnapshotDataStore(Path("data/snapshots/snapshot_20260113"))
engine = SimpleBacktestEngine(snapshot_path=store)
result = engine.backtest(config)
```

**After (DAL-based):**
```python
# Script doesn't know about snapshots
ctx = DataAccessFactory.create_context()  # From config
engine = SimpleBacktestEngine(data_access=ctx)
result = engine.backtest(config)
```

**Benefits:**
- Script doesn't care where data comes from
- Easy to switch data sources (config only)
- Easy to test with mocks
- Consistent across all scripts

### Example 2: Production Pipeline (Before → After)

**Before:**
```python
# Hard-coded snapshot dependency
pipeline = ProductionPipeline(
    config=load_config(),
    snapshot_path="data/snapshots/latest",  # ← Manual path management
)
portfolio = pipeline.generate_portfolio()
```

**After:**
```python
# DAL handles all data access
ctx = DataAccessFactory.create_context(
    config_file="configs/data_access.yaml"
)
pipeline = ProductionPipeline(
    config=load_config(),
    data_access=ctx,  # ← All data access through DAL
)
portfolio = pipeline.generate_portfolio()
```

**Benefits:**
- Snapshot path managed centrally
- Can refresh data without code changes
- Regime detection available in pipeline
- Easy to add alerts/monitoring

### Example 3: Testing (Before → After)

**Before:**
```python
# Tests need actual snapshot files
def test_backtest():
    store = SnapshotDataStore(
        Path("tests/fixtures/snapshot_test_data")  # ← Need fixture files
    )
    engine = SimpleBacktestEngine(snapshot_path=store)
    result = engine.backtest(config)
    assert result.total_return > 0
```

**After:**
```python
# Tests use mocks (no files needed)
def test_backtest():
    ctx = DataAccessContext(
        prices=MockPriceAccessor(test_prices),  # ← Synthetic data
        macro=MockMacroAccessor(test_macro),
        universes=MockUniverseAccessor(["SPY", "QQQ"]),
        references=MockReferenceAccessor(),
    )
    engine = SimpleBacktestEngine(data_access=ctx)
    result = engine.backtest(config)
    assert result.total_return > 0
```

**Benefits:**
- No snapshot fixtures needed
- Faster tests (no I/O)
- Easy to test edge cases
- Easy to control data for specific scenarios

---

## Configuration

### Data Access Configuration File
**Location:** `configs/data_access.yaml`

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

universes:
  type: config_file
  config:
    config_dir: "configs/universes"

reference_data:
  type: static
  config:
    sector_mapping: "configs/reference/sectors.yaml"
```

### Usage in Scripts

```python
# Automatic: uses default config/data_access.yaml
ctx = DataAccessFactory.create_context()

# Custom: use specific config file
ctx = DataAccessFactory.create_context(
    config_file="configs/data_access_production.yaml"
)

# Manual: create specific accessors
ctx = DataAccessContext(
    prices=DataAccessFactory.create_price_accessor(
        source="snapshot",
        config={"source": "data/snapshots/latest"}
    ),
    macro=DataAccessFactory.create_macro_accessor(),
    universes=DataAccessFactory.create_universe_accessor(),
    references=DataAccessFactory.create_reference_accessor(),
)
```

---

## Technical Decisions

### 1. **Composition Over Inheritance**
- Accessors wrap existing stores (SnapshotDataStore, MacroDataLoader)
- Avoids rewriting working code
- Clean separation of concerns

### 2. **Factory Pattern**
- All accessor creation through DataAccessFactory
- Enables dependency injection for testing
- Centralized configuration management

### 3. **Caching as a Decorator**
- CachedPriceAccessor wraps any PriceDataAccessor
- Transparent to consumers
- Can be enabled/disabled via config
- Doesn't affect core accessor logic

### 4. **Frozen Dataclasses**
- DataAccessContext is immutable (frozen=True)
- Prevents accidental modifications
- Safe to share across threads
- Serializable for logging/debugging

### 5. **Point-in-Time Guaranteed**
- All accessors use strict inequality (<, not <=)
- Lookahead bias prevention at every layer
- Enforced through documentation + testing

---

## Risk Mitigation

### Risk 1: Breaking Existing Code During Migration
**Mitigation:**
- Phase 1 is isolated (new files only, no modifications to existing code)
- Phase 2 refactoring can be done gradually
- Old snapshot-based APIs can remain as deprecated wrappers
- Comprehensive test coverage before switchover

### Risk 2: Performance Degradation
**Mitigation:**
- CachingLayer transparent optimization
- Benchmark critical paths during Phase 2
- LRU cache prevents repeated loads
- TTL caching for macro data (24h default)

### Risk 3: Data Consistency Issues
**Mitigation:**
- Version tracking in metadata
- Atomic data swap operations
- Validation hooks on refresh
- Clear ownership of data state

### Risk 4: Scope Creep
**Mitigation:**
- Clear phase definitions with specific deliverables
- No feature expansion during migration
- Keep live data connector as Phase 3 (optional)
- Focus on getting existing system through DAL first

---

## Success Metrics

### Phase 1 Completion
✓ All 6 infrastructure tasks complete  
✓ 100% of unit tests passing  
✓ Type checking clean (mypy)  
✓ No circular imports  
✓ All docstrings present and clear  
✓ Ready for Phase 2 migration  

### Phase 2 Completion
✓ All components refactored to use DAL  
✓ Snapshot dependencies removed from main code  
✓ All 300+ existing tests still passing  
✓ No performance degradation  
✓ Scripts work without snapshot_path argument  

### Phase 3 Completion
✓ Live data connector ready (optional)  
✓ Data refresh orchestration operational  
✓ Complete documentation available  
✓ Example code demonstrates all patterns  
✓ Team trained on new architecture  

---

## Future Extensions

Once DAL is in place, these become straightforward:

1. **Live Data Backend** - Swap snapshot for real-time API feeds
2. **Event Streaming** - Publish data events to downstream systems
3. **Data Versioning** - Track complete lineage of transformations
4. **Distributed Caching** - Use Redis for multi-process caching
5. **Data Monitoring** - Audit trail of all data access patterns
6. **Performance Tuning** - A/B test different caching strategies

---

## Handoff Deliverables

The following documents are ready for coding agents:

### 1. **ARCHITECTURE_DATA_ACCESS_LAYER.md**
Complete technical architecture specification. Covers:
- Architecture diagrams
- Component specifications
- Design patterns
- Configuration examples
- Future enhancements

**Use for:** Understanding overall design and making architectural decisions

### 2. **IMPLEMENTATION_TASKS_DATA_ACCESS_LAYER.md**
Detailed task breakdown for all 16 implementation tasks. Covers:
- Task overview and dependencies
- Estimated effort per task
- Specific requirements for each task
- Success criteria
- Quality gates

**Use for:** Planning which tasks to assign to which agents, tracking progress

### 3. **HANDOFF-PHASE1-DAL-INFRASTRUCTURE.md**
Detailed implementation guide for Phase 1 (6 foundational tasks). Covers:
- Complete code examples for each component
- Testing requirements and test examples
- Key implementation points
- Common pitfalls to avoid
- Integration points with existing code

**Use for:** Coding agents implementing Phase 1 infrastructure

### 4. **Remaining Phase Handoffs** (to be created as needed)
Similar detail level for Phase 2 migration and Phase 3 finalization.

---

## Execution Plan

### Week 1: Foundation
- **Monday-Wednesday:** IMPL-019 (core interfaces)
- **Wednesday-Friday:** IMPL-020, 021, 022, 023 (parallel accessor implementations)
- **Friday-Saturday:** IMPL-024 (caching layer)

**Output:** Complete, tested DAL infrastructure

### Week 2: Migration
- **Monday-Wednesday:** IMPL-025, 026, 027 (core components, parallel track)
- **Monday-Wednesday:** IMPL-028, 029, 030 (scripts/production, parallel track)
- **Thursday-Friday:** IMPL-031 (test utilities, parallel track)

**Output:** All components refactored, tests updated

### Week 3: Finalization
- **Monday-Tuesday:** IMPL-032 (live connector, optional)
- **Tuesday-Wednesday:** IMPL-033 (refresh orchestration)
- **Wednesday-Friday:** IMPL-034 (documentation)

**Output:** Complete system, fully documented

**Total Effort:** 40-45 hours  
**Team Size:** 3-4 agents working in parallel  
**Timeline:** 3 weeks with parallel execution  

---

## Questions for Review

Before proceeding to Phase 1 implementation:

1. ✓ Does the 4-accessor design (Prices, Macro, Universe, Reference) cover all data needs?
2. ✓ Is the factory pattern appropriate, or should we use dependency injection framework?
3. ✓ Should Phase 3 (live data connector) be implemented, or keep as future work?
4. ✓ Should we provide migration guides for user code, or assume internal-only?
5. ✓ Should old snapshot-based APIs remain as deprecated wrappers, or remove entirely?

**Answers provided by architecture:**
1. Yes - covers all current data access patterns in codebase
2. Factory pattern is sufficient - no need for DI framework
3. Keep as Phase 3 (optional) - can add later if needed
4. Yes - migration guide included in HANDOFF documents
5. Yes - deprecated wrappers provide safety net during migration

---

## Ready for Handoff

The architecture is complete and ready for coding agents to begin Phase 1 implementation.

**Next Step:** Assign IMPL-019 task to first coding agent. Once complete, remaining Phase 1 tasks can proceed in parallel.

