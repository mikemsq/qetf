# Task Queue - QuantETF

**Last Updated:** January 9, 2026
**Active Phase:** Phase 2 - Backtest Engine

## Task Status Legend

- `ready` - Available for pickup by any agent
- `in_progress` - Currently being worked on
- `blocked` - Waiting on dependencies
- `completed` - Implementation done, needs review
- `merged` - Reviewed and merged to main

---

## Current Sprint: Phase 2 Backtest Engine

### IMPL-002: EqualWeightTopN Portfolio Constructor
**Status:** completed
**Priority:** high
**Completed:** 2026-01-09
**Dependencies:** []
**Assigned:** Session-IMPL-002

**Description:**
Implement EqualWeightTopN portfolio constructor that takes top N ETFs by alpha score and assigns equal weights.

**Files:**
- Created: `src/quantetf/portfolio/equal_weight.py`
- Created: `tests/test_equal_weight.py`

**Handoff:** `handoffs/handoff-IMPL-002.md`
**Completion:** `handoffs/completion-IMPL-002.md`

**Notes:**
- All 14 tests pass (covers edge cases, NaN handling, large universes)
- Handles fewer valid scores than top_n gracefully
- Comprehensive diagnostics for debugging
- Ready for integration in backtest engine

---

### IMPL-003: FlatTransactionCost Model
**Status:** completed
**Priority:** high
**Estimated:** 1 hour
**Completed:** 2026-01-09
**Dependencies:** []
**Assigned:** Session-IMPL-003

**Description:**
Implement simple flat transaction cost model (10 bps per trade).

**Files:**
- Update: `src/quantetf/portfolio/costs.py`
- Tests: `tests/test_transaction_costs.py`

**Handoff:** `handoffs/handoff-IMPL-003.md`

**Notes:**
- Implemented as dataclass following existing CostModel pattern
- 22 comprehensive tests, all passing
- Handles edge cases (empty/None weights, NaN values, misaligned tickers)
- Returns cost as fraction of NAV (e.g., 0.001 = 10 bps)

---

### IMPL-004: Simple Backtest Engine
**Status:** completed
**Priority:** high
**Completed:** 2026-01-10
**Dependencies:** [IMPL-002, IMPL-003] (both completed)
**Assigned:** Session-IMPL-004-Resume

**Description:**
Implement SimpleBacktestEngine that orchestrates the event-driven backtest loop.

**Files:**
- Created: `src/quantetf/backtest/simple_engine.py`
- Created: `tests/test_backtest_engine.py`

**Handoff:** `handoffs/handoff-IMPL-004.md`
**Completion:** `handoffs/completion-IMPL-004.md`

**Notes:**
- 353 lines of implementation + 475 lines of tests
- 17 comprehensive tests, all passing
- Event-driven architecture with T-1 data access
- Integrates MomentumAlpha, EqualWeightTopN, FlatTransactionCost
- Total test count: 68 → 85 (+17)
- Ready for IMPL-005

---

### TEST-001: No-Lookahead Tests
**Status:** completed
**Priority:** critical
**Estimated:** 2 hours
**Completed:** 2026-01-09
**Dependencies:** []
**Assigned:** Testing Agent

**Description:**
Create synthetic data tests to verify no lookahead bias in data access and alpha models.

**Files:**
- Created: `tests/test_no_lookahead.py`

**Handoff:** `handoffs/handoff-TEST-001.md`
**Completion:** `handoffs/completion-TEST-001.md`

**Notes:**
- 8 comprehensive tests all passing
- Synthetic data approach makes lookahead bugs visible
- Verifies SnapshotDataStore T-1 enforcement
- Verifies MomentumAlpha uses only historical data
- Strong confidence in no-lookahead enforcement

---

### IMPL-005: End-to-End Backtest Script
**Status:** completed
**Priority:** high
**Estimated:** 2 hours
**Completed:** 2026-01-10
**Dependencies:** [IMPL-004] (completed)
**Assigned:** Session-IMPL-005

**Description:**
Create script to run complete backtest on 5-year snapshot and generate results.

**Files:**
- Created: `scripts/run_backtest.py`
- Created: `tests/test_run_backtest.py`

**Handoff:** `handoffs/handoff-IMPL-005.md`
**Completion:** `handoffs/completion-IMPL-005.md`

**Notes:**
- 350+ lines of implementation with CLI interface
- 16 comprehensive tests, all passing
- Successfully runs on real 5yr snapshot data
- Generates equity curve, metrics, holdings, weights, config
- Sample results: 66.9% return, 1.50 Sharpe, -9.8% max drawdown (2023-2025)
- Phase 2 complete (100%)

---

## Completed Tasks

### IMPL-001: MomentumAlpha Model
**Status:** completed
**Priority:** high
**Completed:** 2026-01-09
**Agent:** Session-001

**Description:**
Implemented MomentumAlpha class with 252-day lookback and T-1 data access.

**Files:**
- Updated: `src/quantetf/alpha/momentum.py`
- Created: `src/quantetf/data/snapshot_store.py`

**Notes:**
- Includes comprehensive logging
- Handles missing data gracefully
- Ready for integration testing

---

### INFRA-001: Snapshot Data Store
**Status:** completed
**Priority:** high
**Completed:** 2026-01-09
**Agent:** Session-001

**Description:**
Implemented SnapshotDataStore for point-in-time data access from parquet files.

**Files:**
- Created: `src/quantetf/data/snapshot_store.py`

**Notes:**
- Enforces T-1 data access (no lookahead)
- Supports lookback windows
- Handles MultiIndex format correctly

---

## Backlog (Future Phases)

### Phase 3: Strategy Development
- IMPL-006: Mean reversion alpha model
- IMPL-007: Multi-factor alpha combiner
- IMPL-008: Mean-variance portfolio optimizer
- IMPL-009: Risk parity constructor
- IMPL-010: Covariance estimation

### Phase 4: Production Pipeline
- IMPL-011: Production recommendation generator
- IMPL-012: Run manifest creation
- IMPL-013: Alerting system
- IMPL-014: Data freshness checks

### Phase 5: Documentation & Polish
- DOC-001: Tutorial notebook
- DOC-002: API documentation
- DOC-003: Strategy comparison framework
- TEST-002: Golden test suite

---

## Task Creation Guidelines

When creating new tasks:

1. **Keep tasks focused** - 1-3 hours of work
2. **Make independent** - Minimize dependencies
3. **Provide context** - Create handoff file
4. **Clear acceptance criteria** - How to verify done
5. **Include examples** - Show expected usage

## Task Pickup Process

1. Find a `ready` task that interests you
2. Update status to `in_progress`
3. Add your agent ID to `Assigned`
4. Read the handoff file completely
5. Read CLAUDE_CONTEXT.md for standards
6. Implement and test
7. Update status to `completed`
8. Create completion note

## Need Help?

- **Blocked?** Update task status and note blocker
- **Unclear requirements?** Check handoff file and CLAUDE_CONTEXT.md
- **Architecture questions?** See PROJECT_BRIEF.md
- **Can't find pattern?** Look at similar completed tasks
