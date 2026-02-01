# Task Queue - QuantETF

**Last Updated:** February 1, 2026
**Active Phase:** Strategy Optimization Execution

## Primary Goal

**Build a systematic investment process that consistently outperforms SPY on a universe of ETFs.**

| Requirement | Value |
|-------------|-------|
| Primary Universe | Tier 4 (200 ETFs) |
| Data Period | 10 years (2016-2026) |
| Success Criterion | >=80% of rebalance cycles with positive active return |

---

## Task Status Legend

- `ready` - Available for pickup by any agent
- `in_progress` - Currently being worked on
- `blocked` - Waiting on dependencies
- `completed` - Done (see handoffs/completions/ for details)

---

## Active Tasks

### Data Access Layer - Phase 1 (Foundation)

---

### Strategy Optimizer

#### IMPL-036: Walk-Forward Optimizer Implementation
**Status:** ready
**Priority:** CRITICAL
**Dependencies:** None
**Handout:** [docs/handouts/HANDOUT_walk_forward_optimizer.md](docs/handouts/HANDOUT_walk_forward_optimizer.md)

Modify the strategy optimizer to use walk-forward validation internally, scoring strategies by out-of-sample (OOS) performance instead of in-sample metrics. This addresses the overfitting problem discovered when a 1-year optimized strategy underperformed over 10 years.

**Key changes:**
1. Create `WalkForwardEvaluator` that generates train/test windows
2. Evaluate strategies on TEST periods only (train provides warmup)
3. Score by average OOS active return, OOS Sharpe, and win rate
4. Filter strategies with negative OOS active return

**Sub-tasks:**
- IMPL-036-A: Create dataclasses (WalkForwardEvaluatorConfig, WalkForwardEvaluationResult)
- IMPL-036-B: Implement WalkForwardEvaluator.evaluate()
- IMPL-036-C: Unit tests for window generation
- IMPL-036-D: Unit tests for single-window evaluation
- IMPL-036-E: Modify StrategyOptimizer to use WalkForwardEvaluator
- IMPL-036-F: Update composite scoring to use OOS metrics
- IMPL-036-G: Add filtering for negative OOS strategies
- IMPL-036-H: Update CLI with walk-forward arguments
- IMPL-036-I: Update output format
- IMPL-036-J: Validation run and documentation

---

#### SEARCH-001: Run Strategy Optimizer on Tier 4
**Status:** blocked
**Priority:** HIGH
**Dependencies:** IMPL-036 (walk-forward optimizer required)
**Blocked Reason:** Current optimizer uses in-sample evaluation which leads to overfitting. Must complete IMPL-036 first.

Run strategy optimizer on Tier 4 data using walk-forward validation.

```bash
python scripts/run_backtests.py \
    --snapshot data/snapshots/snapshot_latest \
    --walk-forward \
    --train-years 3 \
    --test-years 1 \
    --step-months 6 \
    --parallel 4
```

---

### Analytics & Visualization (Phase 3 Backlog)

#### VIZ-002: Alpha Diagnostics Notebook
**Status:** ready
**Priority:** MEDIUM
**Dependencies:** ANALYSIS-001 (completed)

Create notebook for analyzing alpha signal quality.

---

#### ANALYSIS-004: Parameter Sensitivity Analysis
**Status:** ready
**Priority:** MEDIUM
**Dependencies:** ANALYSIS-003 (completed)

Create parameter sweep and sensitivity testing notebook.

---

#### ANALYSIS-007: Transaction Cost Analysis
**Status:** ready
**Priority:** MEDIUM
**Dependencies:** ANALYSIS-001 (completed)

Enhanced cost modeling with realistic cost structures.

---

#### VIZ-003: Stress Test Notebook
**Status:** ready
**Priority:** MEDIUM
**Dependencies:** VIZ-001 (completed)

Analyze performance during crisis periods.

---

#### VIZ-004: Auto-Report Generation
**Status:** ready
**Priority:** MEDIUM
**Dependencies:** VIZ-001, ANALYSIS-001, ANALYSIS-002 (all completed)

Create automated HTML report generator.

---

#### INFRA-002: Data Quality Monitoring
**Status:** ready
**Priority:** MEDIUM
**Dependencies:** None

Script to audit data quality and detect anomalies.

---

### Experiments (Blocked)

#### EXP-001: Monthly vs Annual Rebalancing
**Status:** blocked
**Priority:** HIGH
**Dependencies:** IMPL-006 (completed), SEARCH-001 results

---

#### EXP-002: Trend Filter Backtest
**Status:** blocked
**Priority:** HIGH
**Dependencies:** IMPL-006 (completed), SEARCH-001 results

---

#### EXP-003: Ensemble vs Switching
**Status:** blocked
**Priority:** MEDIUM
**Dependencies:** IMPL-006, IMPL-007 (completed), SEARCH-001 results

---

#### EXP-004: Walk-Forward Validation
**Status:** superseded
**Priority:** N/A
**Dependencies:** N/A
**Note:** Superseded by IMPL-036 which integrates walk-forward into the optimizer

---

## Completed Tasks (Summary)

For detailed completion reports, see [handoffs/completions/](handoffs/completions/).

### Data Access Layer
| ID | Task | Completed |
|----|------|-----------|
| IMPL-019 | DAL Core Interfaces & Types | 2026-01-18 |
| IMPL-020 | SnapshotPriceAccessor | 2026-01-18 |
| IMPL-021 | FREDMacroAccessor | 2026-01-18 |
| IMPL-022 | ConfigFileUniverseAccessor | 2026-01-21 |
| IMPL-023 | ReferenceDataAccessor | 2026-01-21 |
| IMPL-024 | CachingLayer & Integration | 2026-01-21 |
| IMPL-025 | Backtest Engine Migration | 2026-01-21 |
| IMPL-026 | Alpha Models Migration | 2026-01-21 |
| IMPL-027 | Portfolio Optimization Migration | 2026-01-21 |
| IMPL-028 | Production Pipeline Migration | 2026-01-22 |
| IMPL-029 | Research Scripts Migration | 2026-01-22 |
| IMPL-030 | Monitoring System Migration | 2026-01-22 |
| IMPL-031 | Test Utilities & Mocking | 2026-01-22 |

### Regime-Aware Infrastructure
| ID | Task | Completed |
|----|------|-----------|
| IMPL-015 | Cycle Win Rate Metrics | 2026-01-17 |
| IMPL-016 | Alpha Selector Framework | 2026-01-17 |
| IMPL-017 | Enhanced Macro Data API | 2026-01-17 |
| IMPL-018 | Regime-Alpha Pipeline | 2026-01-17 |

### Strategy Optimizer
| ID | Task | Completed |
|----|------|-----------|
| OPT-001 | Parameter Grid Generator | 2026-01-14 |
| OPT-002 | Multi-Period Evaluator | 2026-01-15 |
| OPT-003 | Strategy Optimizer | 2026-01-15 |
| OPT-004 | CLI Script | 2026-01-15 |
| OPT-005 | Update Exports | 2026-01-15 |

### Backtest Engine (Phase 2)
| ID | Task | Completed |
|----|------|-----------|
| IMPL-001 | MomentumAlpha Model | 2026-01-09 |
| IMPL-002 | EqualWeightTopN | 2026-01-09 |
| IMPL-003 | FlatTransactionCost | 2026-01-09 |
| IMPL-004 | SimpleBacktestEngine | 2026-01-10 |
| IMPL-005 | E2E Backtest Script | 2026-01-10 |
| TEST-001 | No-Lookahead Tests | 2026-01-09 |

### Alpha Models & Data
| ID | Task | Completed |
|----|------|-----------|
| IMPL-006 | New Alpha Models | 2026-01-15 |
| IMPL-007 | FRED Macro Ingestion | 2026-01-15 |
| DATA-001 | Tier 4 10-Year Data | 2026-01-15 |

### Analytics (Phase 3)
| ID | Task | Completed |
|----|------|-----------|
| ANALYSIS-001 | Enhanced Metrics | 2026-01-11 |
| ANALYSIS-002 | Risk Analytics | 2026-01-11 |
| ANALYSIS-003 | Strategy Comparison | 2026-01-11 |
| ANALYSIS-005 | Benchmark Comparison | 2026-01-11 |
| ANALYSIS-006 | Walk-Forward Validation | 2026-01-11 |
| REFACTOR-001 | Active Returns Focus | 2026-01-15 |
| VIZ-001 | Backtest Analysis Notebook | 2026-01-11 |

### Production System
| ID | Task | Completed |
|----|------|-----------|
| IMPL-035 | Regime-Based Strategy Selection System | 2026-01-28 |

### Bug Fixes
| ID | Task | Completed |
|----|------|-----------|
| BUG-001 | Fix SPY Benchmark Calculation | 2026-01-28 |

---

## Task Pickup Process

1. Find a `ready` task
2. Update status to `in_progress` in this file
3. Read handoff file from [handoffs/tasks/](handoffs/tasks/) or [handoffs/architecture/](handoffs/architecture/)
4. Read [CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md) for coding standards
5. Implement and test
6. Create completion note in [handoffs/completions/](handoffs/completions/)
7. Update status to `completed` in this file

---

## Backlog (Future Phases)

### Data Access Layer - Phase 2 (Migration) - COMPLETE
- ~~IMPL-026: Alpha models migration~~ (completed)
- ~~IMPL-027: Portfolio optimization migration~~ (completed)
- ~~IMPL-028: Production pipeline migration~~ (completed)
- ~~IMPL-029: Research scripts migration~~ (completed)
- ~~IMPL-030: Monitoring system migration~~ (completed)
- ~~IMPL-031: Test utilities & mocking~~ (completed)

### Data Access Layer - Phase 3 (Optional)
- IMPL-032: Live data connector
- IMPL-033: Data refresh orchestration
- IMPL-034: Documentation & examples

### Production Pipeline
- IMPL-010: Risk Overlays Module (completed)
- IMPL-011: Portfolio State Management (completed)
- IMPL-012: Enhanced Production Pipeline
- IMPL-013: Monitoring & Alerts
- IMPL-014: Production Config

### Documentation
- DOC-001: Tutorial notebook
- DOC-002: API documentation
- TEST-002: Golden test suite
