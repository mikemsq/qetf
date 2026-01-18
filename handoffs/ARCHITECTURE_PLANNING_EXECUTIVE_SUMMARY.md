# Executive Summary: Data Access Layer Refactoring Initiative

**Date:** January 18, 2026  
**Prepared by:** Architect/Planner  
**Audience:** Technical leads, project managers  
**Status:** Ready for implementation

---

## Problem Statement

QuantETF currently uses a **snapshot-based data model** where scripts directly depend on hardcoded file paths:

```python
# Current (problematic)
store = SnapshotDataStore(Path("data/snapshots/snapshot_20260113_232157"))
engine = SimpleBacktestEngine(snapshot_path=store)
```

**Issues:**
- ❌ Tight coupling to file system
- ❌ No centralized caching strategy
- ❌ Difficult to test (requires actual snapshot files)
- ❌ Hard to support multiple data sources (live API, mock data)
- ❌ Poor separation between data access and business logic
- ❌ No data quality validation at access layer

---

## Solution: Data Access Layer (DAL)

Implement a **unified Data Access Layer** providing clean interfaces for all data access. Scripts never touch files directly.

```python
# New (clean)
ctx = DataAccessFactory.create_context()  # From config
engine = SimpleBacktestEngine(data_access=ctx)
```

### Four Core Accessors

| Accessor | Purpose | Methods |
|----------|---------|---------|
| **PriceDataAccessor** | Historical/current prices | `read_prices_as_of()`, `read_ohlcv_range()`, `validate_data_availability()` |
| **MacroDataAccessor** | Macro indicators & regimes | `read_macro_indicator()`, `get_regime()` |
| **UniverseDataAccessor** | Ticker set definitions | `get_universe()`, `get_universe_as_of()` |
| **ReferenceDataAccessor** | Static reference data | `get_ticker_info()`, `get_sector_mapping()` |

All wrapped in **DataAccessContext** for convenient single-object passing.

---

## Key Benefits

### For Development
- ✅ **Cleaner APIs:** Scripts don't manage file paths
- ✅ **Easier Testing:** Mock data instead of snapshot files
- ✅ **Better Separation:** Business logic independent from storage
- ✅ **Transparent Caching:** Performance optimization without coupling

### For Production
- ✅ **Multiple Data Sources:** Snapshot files, live APIs, databases
- ✅ **Data Quality Control:** Validation at access layer
- ✅ **Centralized Configuration:** Single source of truth for data settings
- ✅ **Automated Refresh:** Orchestrated data updates with cache management
- ✅ **Regime Detection:** Macro data available everywhere consistently

### For Testing
- ✅ **Mock Accessors:** Synthetic data without files
- ✅ **Fast Tests:** No disk I/O
- ✅ **Edge Cases:** Easy to control data for specific scenarios
- ✅ **Test Builders:** Fluent API for creating test datasets

---

## Implementation Plan

### Phase 1: Foundation (Weeks 1, ~12-14 hours)
Build DAL core infrastructure. **Completely isolated - no changes to existing code.**

**6 tasks:**
1. Core interfaces & types (3h)
2. SnapshotPriceAccessor (3h)
3. FREDMacroAccessor (3h)
4. ConfigFileUniverseAccessor (2h)
5. StaticReferenceAccessor (2h)
6. CachingLayer (2.5h)

**Can parallelize:** IMPL-019 → then 020-023 parallel → then 024

**Output:** Complete, tested DAL infrastructure. Zero impact on existing code.

---

### Phase 2: Migration (Week 2, ~20-24 hours)
Refactor all major components to use DAL. **Can execute in parallel tracks.**

**7 tasks across 3 parallel tracks:**

**Track A (Core Engine):**
- Backtest engine refactoring (2h)
- Alpha models refactoring (2.5h)
- Portfolio optimization refactoring (3h)

**Track B (Scripts & Production):**
- Production pipeline refactoring (2h)
- Research scripts refactoring (3h)
- Monitoring system refactoring (2h)

**Track C (Testing):**
- Test utilities & mocks (4h)

**Output:** All components use DAL exclusively. Snapshots removed from main code.

---

### Phase 3: Completion (Week 3, ~8-10 hours)
Add live data support and complete documentation.

**3 tasks:**
- Live data connector (3h)
- Data refresh orchestration (2.5h)
- Documentation & examples (2.5h)

**Output:** Production-ready system with full documentation.

---

## Timeline & Resources

| Phase | Duration | Agents | Dependencies |
|-------|----------|--------|--------------|
| Phase 1 | 3-4 days | 2-3 (can parallelize) | None - starts immediately |
| Phase 2 | 5-6 days | 3-4 (parallel tracks) | Phase 1 complete |
| Phase 3 | 3-4 days | 1-2 | Phase 2 complete |
| **Total** | **10-12 days** | **3-4 agents** | - |

**Critical Path:** Phase 1 → Phase 2 → Phase 3 (sequential phases, but tasks within phases parallel)

---

## What's in the Handoffs

Three comprehensive documents prepared for coding agents:

### 1. Architecture Specification
**File:** `handoffs/ARCHITECTURE_DATA_ACCESS_LAYER.md`
- Technical architecture diagrams
- Component specifications
- Design decisions and rationale
- Configuration examples
- Future enhancement opportunities

### 2. Task Breakdown
**File:** `handoffs/IMPLEMENTATION_TASKS_DATA_ACCESS_LAYER.md`
- All 16 tasks with specific requirements
- Estimated effort and LOC for each
- Dependency graph
- Success criteria and quality gates
- Parallelization strategy

### 3. Phase 1 Implementation Guide
**File:** `handoffs/HANDOFF-PHASE1-DAL-INFRASTRUCTURE.md`
- Complete code examples for Phase 1
- Step-by-step implementation guidance
- Testing requirements and examples
- Common pitfalls to avoid
- Integration points with existing code

---

## Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Breaking existing code | Low | High | Phase 1 isolated, gradual Phase 2 migration |
| Performance degradation | Low | Medium | CachingLayer transparent optimization, benchmarking |
| Scope creep | Medium | Medium | Clear phase definitions, no feature expansion |
| Data consistency issues | Low | High | Version tracking, atomic updates, validation |
| Team learning curve | Medium | Low | Comprehensive documentation, clear examples |

**Overall Risk Level:** ✅ **LOW** - Well-defined scope, isolated phases, clear deliverables

---

## Success Metrics

### Phase 1: Infrastructure Complete
- ✅ All 6 tasks implemented and tested
- ✅ 100% test pass rate
- ✅ Type checking clean
- ✅ No circular imports
- ✅ Production-ready code

### Phase 2: Migration Complete
- ✅ All components use DAL
- ✅ Snapshot dependencies removed
- ✅ All 300+ existing tests passing
- ✅ Performance verified (no degradation)
- ✅ Scripts work without snapshot_path argument

### Phase 3: Production Ready
- ✅ Live data connector available
- ✅ Refresh orchestration operational
- ✅ Complete documentation
- ✅ Team trained
- ✅ Deployment plan executed

---

## Immediate Next Steps

1. **Review & Approve Architecture**
   - Review `ARCHITECTURE_PLANNING_SUMMARY.md`
   - Confirm 4-accessor design is appropriate
   - Approve Phase 1 scope

2. **Assign Phase 1 Tasks**
   - Agent 1: IMPL-019 (core interfaces) - critical path
   - Agents 2-3: IMPL-020, 021, 022, 023 (wait for IMPL-019, then parallel)
   - Agent 1 or 2: IMPL-024 (after IMPL-020)

3. **Prepare Development Environment**
   - Create branch for DAL work (e.g., `feature/data-access-layer`)
   - Ensure test infrastructure ready
   - Set up CI/CD for new test suite

4. **Monitor Phase 1 Progress**
   - Daily standup during Phase 1
   - Code review of each task
   - Integration testing when complete

---

## Why This Approach

### Why Replace Snapshots with DAL?

**Current Snapshot Model:**
- Snapshots are immutable dataset versions
- Created manually when data changes
- Scripts hardcode snapshot IDs
- No way to gracefully handle data updates

**DAL Model:**
- DAL abstracts data source
- Data updates transparent to code
- Configuration-driven data source
- Supports live data seamlessly

### Why This DAL Design?

**Four Accessors Instead of One:**
- Separation of concerns (prices ≠ macro ≠ universe)
- Different caching strategies per accessor
- Different quality control per data type
- Easier to extend with new data sources

**Factory Pattern:**
- Enables dependency injection for testing
- Centralized configuration
- Easy to swap implementations
- Clear ownership of object creation

**CachingLayer as Decorator:**
- Transparent optimization
- No coupling to specific accessor implementations
- Can enable/disable with config
- Testable in isolation

---

## Comparison: Before vs. After

### Before: Snapshot-Based
```python
# run_backtest.py
store = SnapshotDataStore(Path("data/snapshots/snapshot_20260113"))
engine = SimpleBacktestEngine(snapshot_path=store)

# run_production_pipeline.py
store = SnapshotDataStore(Path("data/snapshots/snapshot_20260115"))
pipeline = ProductionPipeline(store=store)

# run_optimization.py
store = SnapshotDataStore(Path("data/snapshots/snapshot_20260110"))
optimizer = StrategyOptimizer(snapshot_path=store)

# Issues:
# - Manual snapshot path management
# - Different paths in different scripts
# - Can't easily test without snapshot files
# - Hard to update data atomically
```

### After: DAL-Based
```python
# All scripts identical
ctx = DataAccessFactory.create_context()
# or
ctx = DataAccessFactory.create_context(
    config_file="configs/data_access.yaml"
)

# run_backtest.py
engine = SimpleBacktestEngine(data_access=ctx)

# run_production_pipeline.py
pipeline = ProductionPipeline(data_access=ctx)

# run_optimization.py
optimizer = StrategyOptimizer(data_access=ctx)

# Benefits:
# - Single configuration source
# - Easy to test (use mocks)
# - Easy to update data (refresh manager)
# - Easy to add live data support
```

---

## FAQ

**Q: How long until we can start coding?**  
A: Immediately. All planning and architecture is complete. Code Phase 1 can start today.

**Q: Will this break existing code?**  
A: No. Phase 1 is completely isolated. Phase 2 migration is gradual and well-documented.

**Q: What's the impact on performance?**  
A: None expected. CachingLayer provides transparent optimization. Benchmarks during Phase 2.

**Q: Can we use this with live data?**  
A: Yes, Phase 3 includes `LivePriceAccessor` implementation for future use.

**Q: What if we need to change the design?**  
A: Phase 1 is sealed by week 1. Design changes after Phase 1 start would require re-planning.

**Q: Can we parallelize more?**  
A: Yes - Phase 2 has 3 parallel tracks. Phase 1 limited by IMPL-019 → IMPL-024 dependency.

**Q: What happens to old snapshot code?**  
A: Will be wrapped as deprecated APIs during Phase 2 migration. Can be removed once all migrations complete.

---

## Conclusion

This Data Access Layer refactoring provides:

✅ **Cleaner architecture** - Separation of concerns  
✅ **Better testability** - Mock data, no files needed  
✅ **Production readiness** - Multiple data sources, quality control  
✅ **Future-proof** - Easy to add live data, streaming, etc.  
✅ **Low risk** - Isolated phases, comprehensive documentation  
✅ **Clear path** - 16 well-defined tasks, estimated timelines  

**The architecture is complete. Ready for Phase 1 implementation to begin.**

---

## Document Index

| Document | Purpose | Audience |
|----------|---------|----------|
| `ARCHITECTURE_DATA_ACCESS_LAYER.md` | Technical architecture specification | Architects, senior engineers |
| `IMPLEMENTATION_TASKS_DATA_ACCESS_LAYER.md` | Task breakdown and requirements | Project managers, task assigners |
| `HANDOFF-PHASE1-DAL-INFRASTRUCTURE.md` | Implementation guide for Phase 1 | Coding agents working on Phase 1 |
| `ARCHITECTURE_PLANNING_SUMMARY.md` | Complete planning overview | Architecture review, technical leads |
| `ARCHITECTURE_PLANNING_EXECUTIVE_SUMMARY.md` (this document) | High-level overview | Executives, stakeholders, decision makers |

---

**Status:** ✅ Ready for Implementation  
**Recommendation:** Proceed with Phase 1  
**Next Action:** Assign IMPL-019 to first coding agent

