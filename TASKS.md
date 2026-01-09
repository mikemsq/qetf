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
**Status:** ready
**Priority:** high
**Estimated:** 1-2 hours
**Dependencies:** []
**Assigned:** (available)

**Description:**
Implement EqualWeightTopN portfolio constructor that takes top N ETFs by alpha score and assigns equal weights.

**Files:**
- Create: `src/quantetf/portfolio/equal_weight.py`
- Tests: `tests/test_equal_weight.py`

**Handoff:** `handoffs/handoff-IMPL-002.md`

---

### IMPL-003: FlatTransactionCost Model
**Status:** ready
**Priority:** high
**Estimated:** 1 hour
**Dependencies:** []
**Assigned:** (available)

**Description:**
Implement simple flat transaction cost model (10 bps per trade).

**Files:**
- Update: `src/quantetf/portfolio/costs.py`
- Tests: `tests/test_transaction_costs.py`

**Handoff:** `handoffs/handoff-IMPL-003.md`

---

### IMPL-004: Simple Backtest Engine
**Status:** blocked
**Priority:** high
**Estimated:** 3-4 hours
**Dependencies:** [IMPL-002, IMPL-003]
**Assigned:** (available)

**Description:**
Implement SimpleBacktestEngine that orchestrates the event-driven backtest loop.

**Files:**
- Create: `src/quantetf/backtest/simple_engine.py`
- Tests: `tests/test_backtest_engine.py`

**Handoff:** `handoffs/handoff-IMPL-004.md`

---

### TEST-001: No-Lookahead Tests
**Status:** ready
**Priority:** critical
**Estimated:** 2 hours
**Dependencies:** []
**Assigned:** (available)

**Description:**
Create synthetic data tests to verify no lookahead bias in data access and alpha models.

**Files:**
- Create: `tests/test_no_lookahead.py`

**Handoff:** `handoffs/handoff-TEST-001.md`

---

### IMPL-005: End-to-End Backtest Script
**Status:** blocked
**Priority:** high
**Estimated:** 2 hours
**Dependencies:** [IMPL-004]
**Assigned:** (available)

**Description:**
Create script to run complete backtest on 5-year snapshot and generate results.

**Files:**
- Create: `scripts/run_backtest.py`
- Create: `notebooks/backtest_analysis.ipynb`

**Handoff:** `handoffs/handoff-IMPL-005.md`

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
