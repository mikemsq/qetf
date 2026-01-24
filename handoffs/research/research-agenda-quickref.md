# Research Agenda Quick Reference

**Last Updated:** 2026-01-24
**Status:** Active Research Program → **PRODUCTION DESIGN APPROVED**

---

## Core Hypothesis

> "Different market regimes require different strategies. By detecting regimes and adapting, we can beat SPY over rolling 1-year periods."

**Evidence:**
- Perfect foresight: 10/10 years beatable (avg +55.8%/year spread)
- Simple momentum: 3/9 years win rate (33%)
- Regimes confirmed: Bull (7), Bear (1), High Vol (1), Moderate (2)

## APPROVED PRODUCTION DESIGN (2026-01-24)

Architecture Decision Record: `handoffs/architecture/ADR-001-regime-based-strategy-selection.md`

| Decision | Choice |
|----------|--------|
| Regimes | 4 (trend × vol matrix) |
| Detection | SPY vs 200MA + VIX with hysteresis |
| Thresholds | MA ±2%, VIX 20/25 |
| Mapping | Empirical (from optimization) |
| Frequency | Daily check, rebalance-day action |

**Implementation Task:** IMPL-035 (ready for architect pickup)

---

## Priority Experiments

| ID | Experiment | Hypothesis | Status |
|----|------------|------------|--------|
| EXP-001 | Monthly Rebalancing | Faster adaptation captures more alpha | **TODO** |
| EXP-002 | Trend Filter | SPY > MA200 filter reduces crashes | **TODO** |
| EXP-003 | Ensemble Blend | Blending beats switching | **TODO** |
| EXP-004 | Vol Targeting | Risk-scaled positions smooth returns | **TODO** |
| EXP-005 | Walk-Forward | Validate rules out-of-sample | **TODO** |

---

## Implementation Tasks

### Wave 1 (Immediate)

| Task | File | Status |
|------|------|--------|
| TrendFilteredMomentum | `src/quantetf/alpha/trend_filtered_momentum.py` | [IMPL-006] |
| DualMomentum | `src/quantetf/alpha/dual_momentum.py` | [IMPL-006] |
| ValueMomentum | `src/quantetf/alpha/value_momentum.py` | [IMPL-006] |
| FRED Data Ingestion | `scripts/ingest_fred_data.py` | [IMPL-007] |

### Wave 2 (Next)

| Task | Depends On |
|------|------------|
| Run EXP-001 | Wave 1 complete |
| Run EXP-002 | TrendFilteredMomentum |
| Macro regime detection | FRED data |

---

## Key Findings Summary

### What Works
- Trend filtering (SPY > MA200)
- Monthly rebalancing (vs annual)
- Ensemble blending (vs switching)
- Vol targeting (risk-adjusted positions)

### What Doesn't Work
- Pure momentum with annual rebalance (too much variance)
- Perfect regime switching (detection too slow)
- Over-optimization on historical regimes (overfitting)

### Strategy-Regime Map (Observed)

| Regime | Best Strategy | Why |
|--------|---------------|-----|
| Bull | Value / Mean-Reversion | Momentum crashes (2021) |
| Bear | 12M Momentum | Energy/commodities worked |
| High Vol | 12M Momentum | Clean trends emerge |
| Transition | Ensemble | Hedges uncertainty |

---

## Data Sources

### Available
- **ETF Prices:** Stooq (300+ ETFs, 10 years)
- **Tier 4 Universe:** 200 ETFs (178 non-leveraged)

### To Add
- **FRED Macro:** VIX, Treasury yields, credit spreads
- **Get API Key:** https://fred.stlouisfed.org/docs/api/api_key.html

---

## Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Win Rate | >50% | Beat SPY more often than not |
| Sharpe | >0.8 | Risk-adjusted outperformance |
| Max DD | <25% | Better than SPY's -34% (2020) |
| IR | >0.3 | Consistent alpha vs benchmark |

---

## Documents

| Document | Purpose |
|----------|---------|
| [ADR-001-regime-based-strategy-selection.md](../architecture/ADR-001-regime-based-strategy-selection.md) | **Production design decisions** |
| [handoff-IMPL-035-REGIME-STRATEGY-SYSTEM.md](../tasks/handoff-IMPL-035-REGIME-STRATEGY-SYSTEM.md) | **Implementation task handoff** |
| [regime-hypothesis.md](./regime-hypothesis.md) | Full research findings |
| [IMPL-006-NEW-ALPHA-MODELS.md](../tasks/IMPL-006-NEW-ALPHA-MODELS.md) | New strategy specs |
| [IMPL-007-DATA-INGESTION.md](../tasks/IMPL-007-DATA-INGESTION.md) | FRED data ingestion |

---

## Quick Commands

```bash
# Run existing momentum backtest
python scripts/run_backtest.py --strategy momentum --start-date 2016-01-01

# When FRED is set up
export FRED_API_KEY=your_key
python scripts/ingest_fred_data.py --start-date 2015-01-01 --combine

# Test new alpha models (after implementation)
pytest tests/test_*momentum*.py -v
```

---

## Next Steps for Quant Agent

1. **After Wave 1 Implementation:**
   - Run EXP-001 (monthly rebalance comparison)
   - Run EXP-002 (trend filter backtest)
   - Document results in new research note

2. **After FRED Data Available:**
   - Test macro regime detection
   - Integrate with alpha models
   - Run EXP-003 (ensemble vs switching)

3. **Validation Phase:**
   - Walk-forward on 2021-2025
   - Parameter sensitivity analysis
   - Final strategy recommendation

---

*This is a living document. Update as experiments complete.*
