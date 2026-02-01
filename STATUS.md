# QuantETF Project Status

**Last Updated:** February 1, 2026
**Current Phase:** Walk-Forward Optimizer Implementation
**Branch:** main

---

## Primary Goal

**Build a systematic investment process that consistently outperforms SPY on a universe of ETFs.**

| Requirement | Value |
|-------------|-------|
| Primary Universe | Tier 4 (200 ETFs) |
| Data Period | 10 years (2016-2026) |
| Success Criterion | >=80% of rebalance cycles with positive active return |
| Evaluation Periods | Walk-forward OOS validation |

---

## Quick Status

| Area | Status | Notes |
|------|--------|-------|
| Data Infrastructure | Complete | Tier 4, 10yr history, FRED macro data |
| Backtest Engine | Complete | Event-driven, T-1 enforcement, 300+ tests |
| Analytics | Complete | Metrics, risk analysis, walk-forward validation |
| Regime Detection | Complete | IMPL-015 through IMPL-018 |
| Data Access Layer | Complete | IMPL-019 through IMPL-031 (13 tasks) |
| Regime Strategy System | Complete | IMPL-035 (4-regime system) |
| Bug Fixes | Complete | BUG-001 SPY benchmark calculation fixed |
| Strategy Optimizer | **BLOCKED** | Requires IMPL-036 (walk-forward integration) |

---

## Recent Progress

### February 1, 2026: Overfitting Problem Identified

**Critical Finding**: The current optimizer selects strategies based on in-sample performance, leading to severe overfitting.

**Evidence**:
- Optimization selected `momentum_acceleration_126_63_top7_tier3` with +12.4% active return (1yr)
- 10-year backtest of same strategy: +46.6% total return vs SPY ~200% (massive underperformance)
- Strategy shows Sharpe 0.35 over 10 years (very poor)

**Root Cause**: Optimizer evaluated strategies on trailing 1-year window and selected parameters that fit that specific period.

**Solution**: IMPL-036 - Integrate walk-forward validation into optimizer
- Evaluate strategies on out-of-sample (OOS) test windows
- Score by average OOS active return, not in-sample
- Filter strategies with negative OOS performance

**Handout**: [docs/handouts/HANDOUT_walk_forward_optimizer.md](docs/handouts/HANDOUT_walk_forward_optimizer.md)

---

### January 28, 2026: IMPL-035 & BUG-001 Complete

- **IMPL-035**: Regime-Based Strategy Selection System
  - 4-regime system (trend × volatility matrix)
  - SPY/200MA + VIX for regime detection
  - Hysteresis thresholds to prevent whipsawing
  - Regime detection analysis notebook added

- **BUG-001**: Fixed SPY Benchmark Calculation
  - SPY returns now calculated from prices (not sparse aligned returns)
  - All optimization results are now valid

### January 21-22, 2026: Data Access Layer Complete

Completed all 13 DAL tasks (IMPL-019 through IMPL-031):
- Core interfaces, price accessor, macro accessor
- Universe and reference data accessors
- Caching layer, all component migrations
- Test utilities and mocking infrastructure

### January 17, 2026: Regime-Aware Infrastructure Complete

Completed IMPL-015 through IMPL-018:
- Per-rebalance-cycle metrics for 80% win rate validation
- Alpha selector framework for regime-based model selection
- Enhanced macro data API with point-in-time access
- Regime-aware alpha integration layer

---

## What's Working

- Complete end-to-end backtesting system
- 4 alpha models (Momentum, TrendFiltered, Dual, ValueMomentum)
- Regime detection from macro data (VIX, yield curve, credit spreads)
- Regime-aware model selection
- Per-cycle win rate metrics
- Walk-forward validation framework
- 300+ passing tests

---

## What's Next

### Immediate Priority

1. **SEARCH-001: Run Strategy Optimizer on Tier 4**
   ```bash
   python scripts/find_best_strategy.py \
       --snapshot data/snapshots/snapshot_latest \
       --periods 1,3 \
       --parallel 4
   ```

2. **Validate Best Strategy**
   - Walk-forward test
   - Cycle metrics (80% criterion)
   - Stress test analysis

3. **Verify regime-to-strategy mapping**
   - Determine which strategies perform best in each regime
   - Configure production system

### Backlog

- VIZ-002: Alpha diagnostics notebook
- ANALYSIS-004: Parameter sensitivity
- ANALYSIS-007: Transaction cost analysis
- Production deployment
- DAL Phase 3: Live connector, documentation

---

## Architecture Overview

```
Data Layer
├── FRED macro data (VIX, yields, spreads)
├── Tier 4 universe (200 ETFs, 10-year history)
└── Point-in-time snapshots

Regime Detection Layer
├── RegimeDetector (classifies market conditions)
└── MacroDataLoader → FREDMacroAccessor (DAL)

Alpha Layer
├── MomentumAlpha, TrendFilteredMomentum
├── DualMomentum, ValueMomentum
└── RegimeAwareAlpha (adaptive selection)

Portfolio Layer
├── EqualWeightTopN
├── FlatTransactionCost
└── RiskOverlays

Backtest & Analytics Layer
├── SimpleBacktestEngine
├── CycleMetrics (80% win rate validation)
├── Walk-forward validation
└── Strategy comparison
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 300+ |
| Test Pass Rate | 100% |
| Code Coverage | ~80% |
| Implementations Complete | 32 |
| Tasks Remaining | SEARCH-001 + validation |

---

## Documentation Guide

| File | Purpose |
|------|---------|
| [README.md](README.md) | Public overview, quickstart |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Vision, goals, architecture |
| [STATUS.md](STATUS.md) | This file - current state |
| [TASKS.md](TASKS.md) | Task queue |
| [CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md) | Coding standards |
| [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) | Agent roles |
| [handoffs/](handoffs/) | Task handoffs and completions |

---

## For New Agents

1. Read [CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md) for coding standards
2. Read this file for current status
3. Check [TASKS.md](TASKS.md) for available tasks
4. Read relevant handoff from [handoffs/tasks/](handoffs/tasks/)

---

**Repository:** https://github.com/mikemsq/qetf
