# Architect Analysis: IMPL-035 vs IMPL-040 Implementation Order

**Author:** Architect/Planner Agent
**Date:** 2026-01-29
**Status:** APPROVED FOR IMPLEMENTATION

---

## Executive Summary

IMPL-035 (Regime-Based Strategy Selection System) and IMPL-040 (Optimizer Redesign) have overlapping scope in the optimizer component. After analysis, **IMPL-040 supersedes IMPL-035e** and should be implemented instead.

**Key Decision:** Skip IMPL-035e and implement IMPL-040 in its place.

---

## Conflict Analysis

### Files Modified by Both Tasks

| File | IMPL-035e | IMPL-040 |
|------|-----------|----------|
| `src/quantetf/optimization/optimizer.py` | ✓ | ✓ |
| `src/quantetf/optimization/types.py` | ✓ | ✓ |
| `src/quantetf/optimization/evaluator.py` | | ✓ |
| `scripts/find_best_strategy.py` | | ✓ |
| `scripts/optimize.sh` | | ✓ |
| `configs/optimization/defaults.yaml` | | ✓ |

### Why IMPL-040 Supersedes IMPL-035e

IMPL-035e was the original plan to add regime analysis to the optimizer. However, IMPL-040 was created to address three fundamental issues discovered in the optimizer:

1. **Regime Analysis is Currently a Stub**
   The existing `_create_regime_analysis_stub()` function generates **fake data** using composite scores + random noise. IMPL-035e would have built on this stub. IMPL-040 replaces it with real analysis.

2. **Multi-Period Scoring is Broken for Quarterly Runs**
   The composite score formula `avg(IR) - 0.5*std(IR) + 0.5*winner_bonus` was designed for one-time discovery. With quarterly re-runs using 1 period, `std(IR)=0`, making the score perfectly correlated with IR. IMPL-040 introduces regime-weighted scoring.

3. **Single Strategy Output vs. Regime Mapping**
   IMPL-035e planned to add regime mapping as an output. IMPL-040 makes `regime_mapping.yaml` the PRIMARY output with proper per-regime metrics.

**Implementing IMPL-035e first would create code that IMPL-040 immediately rewrites** - this is wasted effort.

---

## Recommended Implementation Order

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 1: FOUNDATION                       │
│                        (Can run in parallel)                     │
├─────────────────────────────────────────────────────────────────┤
│  IMPL-035a: Regime Detector          (2-3h)  ───┐              │
│  IMPL-035b: Configuration Files      (1h)    ───┼─→ No deps    │
│  IMPL-035c: VIX Data Verification    (1-2h)  ───┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 2: ANALYSIS LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  IMPL-035d: Regime Analyzer          (3-4h)                     │
│             Depends on: IMPL-035a                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 3: OPTIMIZER INTEGRATION                 │
├─────────────────────────────────────────────────────────────────┤
│  ██ IMPL-035e: SKIP (superseded by IMPL-040) ██                 │
│                                                                  │
│  IMPL-040: Optimizer Redesign        (4-6h)                     │
│            Depends on: IMPL-035d                                 │
│            Produces: regime_mapping.yaml (PRIMARY OUTPUT)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 4: PRODUCTION COMPONENTS                 │
│                        (Can run in parallel)                     │
├─────────────────────────────────────────────────────────────────┤
│  IMPL-035f: Daily Regime Monitor     (2-3h)                     │
│             Depends on: IMPL-035a, IMPL-035c                     │
│                                                                  │
│  IMPL-035g: Production Rebalancer    (3-4h)                     │
│             Depends on: IMPL-035f                                │
│             Requires: regime_mapping.yaml from IMPL-040          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 5: TESTING & TOOLING                     │
│                        (Can run in parallel)                     │
├─────────────────────────────────────────────────────────────────┤
│  IMPL-035h: Integration Tests        (2-3h)                     │
│  IMPL-035i: CLI Scripts              (2h)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task Assignments for Coder Agents

### Batch 1: Foundation (Parallelizable)

| Task | Priority | Effort | Handoff Document |
|------|----------|--------|------------------|
| IMPL-035a | HIGH | 2-3h | [handoff-IMPL-035a-REGIME-DETECTOR.md](../tasks/handoff-IMPL-035a-REGIME-DETECTOR.md) |
| IMPL-035b | HIGH | 1h | [handoff-IMPL-035b-CONFIG-FILES.md](../tasks/handoff-IMPL-035b-CONFIG-FILES.md) |
| IMPL-035c | HIGH | 1-2h | [handoff-IMPL-035c-VIX-DATA.md](../tasks/handoff-IMPL-035c-VIX-DATA.md) |

### Batch 2: Analysis (Sequential)

| Task | Priority | Effort | Handoff Document | Blocked By |
|------|----------|--------|------------------|------------|
| IMPL-035d | HIGH | 3-4h | [handoff-IMPL-035d-REGIME-ANALYZER.md](../tasks/handoff-IMPL-035d-REGIME-ANALYZER.md) | IMPL-035a |

### Batch 3: Optimizer (Sequential)

| Task | Priority | Effort | Handoff Document | Blocked By |
|------|----------|--------|------------------|------------|
| ~~IMPL-035e~~ | ~~SKIP~~ | — | ~~superseded~~ | — |
| **IMPL-040** | **CRITICAL** | 4-6h | [handoff-IMPL-040-OPTIMIZER-REDESIGN.md](../tasks/handoff-IMPL-040-OPTIMIZER-REDESIGN.md) | IMPL-035d |

### Batch 4: Production (Parallelizable after IMPL-040)

| Task | Priority | Effort | Handoff Document | Blocked By |
|------|----------|--------|------------------|------------|
| IMPL-035f | HIGH | 2-3h | [handoff-IMPL-035f-DAILY-MONITOR.md](../tasks/handoff-IMPL-035f-DAILY-MONITOR.md) | IMPL-035a, IMPL-035c |
| IMPL-035g | HIGH | 3-4h | [handoff-IMPL-035g-PRODUCTION-REBALANCER.md](../tasks/handoff-IMPL-035g-PRODUCTION-REBALANCER.md) | IMPL-035f, IMPL-040 |

### Batch 5: Testing & Tooling (Parallelizable)

| Task | Priority | Effort | Handoff Document | Blocked By |
|------|----------|--------|------------------|------------|
| IMPL-035h | MEDIUM | 2-3h | [handoff-IMPL-035h-INTEGRATION-TESTS.md](../tasks/handoff-IMPL-035h-INTEGRATION-TESTS.md) | Batches 1-4 |
| IMPL-035i | MEDIUM | 2h | [handoff-IMPL-035i-CLI-SCRIPTS.md](../tasks/handoff-IMPL-035i-CLI-SCRIPTS.md) | Batches 1-4 |

---

## Total Effort Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| Phase 1 | 035a, 035b, 035c | 4-6h |
| Phase 2 | 035d | 3-4h |
| Phase 3 | 040 | 4-6h |
| Phase 4 | 035f, 035g | 5-7h |
| Phase 5 | 035h, 035i | 4-5h |
| **TOTAL** | | **20-28h** |

Note: By skipping IMPL-035e (2-3h), we save effort that would otherwise be immediately reworked by IMPL-040.

---

## Critical Path

The critical path through the system is:

```
IMPL-035a (2-3h) → IMPL-035d (3-4h) → IMPL-040 (4-6h) → IMPL-035g (3-4h)
                                                      ↑
                                            IMPL-035f (2-3h) ←── IMPL-035c (1-2h)
```

**Minimum time to production-ready system:** ~15-20h (assuming sequential execution of critical path)

---

## Handoff Document Updates Required

1. **IMPL-035e**: Mark as `SUPERSEDED` with pointer to IMPL-040
2. **IMPL-040**: Update dependencies section to clarify it replaces IMPL-035e
3. **IMPL-035g**: Add note that it requires `regime_mapping.yaml` from IMPL-040

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Implementing tasks out of order | Follow batch ordering strictly |
| IMPL-040 has hidden complexity | Document contains detailed specs; follow acceptance criteria |
| Integration issues between components | IMPL-035h (integration tests) should catch these |

---

## Approval

This analysis recommends:
- ✅ Skip IMPL-035e (superseded)
- ✅ Implement IMPL-040 in Phase 3
- ✅ Follow batch ordering as specified

Ready for coder agents to begin with **Batch 1: IMPL-035a, IMPL-035b, IMPL-035c** (parallelizable).
