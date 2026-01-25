# ADR-001: Regime-Based Strategy Selection System

**Status:** APPROVED
**Date:** 2026-01-24
**Decision Makers:** User (Product Owner), Claude (Quant Researcher)
**Supersedes:** N/A

---

## Context

The current optimization system finds strategies that perform well historically, but uses a single "best" strategy regardless of market conditions. Market regimes change (bull/bear, high-vol/low-vol), and strategies that excel in one regime may underperform in another.

**Problem Statement:** How do we adapt strategy selection to changing market conditions without overfitting to recent performance?

**Approaches Considered:**

| Approach | Description | Tradeoff |
|----------|-------------|----------|
| 1. Regime Detection + Strategy Mapping | Detect regime, look up pre-assigned strategy | Simple, binary switching |
| 2. Strategy Ensemble with Dynamic Weights | Run all strategies, blend with regime-influenced weights | More flexible, gradual transitions |
| 3. Adaptive Parameters | Same strategy type, adjust parameters per regime | Lower turnover, limited adaptability |

**Decision:** Start with **Approach 1** (regime detection + strategy mapping) for simplicity, with architecture designed to evolve into **Approach 2** (ensemble) in the future. Approach 1 is a special case of Approach 2 where weights are binary (0 or 1).

---

## Decisions

### D1: Number of Regimes

**Decision:** 4 regimes (2x2 matrix: trend × volatility)

```
                    LOW VOLATILITY          HIGH VOLATILITY
                ┌─────────────────────┬─────────────────────┐
   UPTREND      │ uptrend_low_vol     │ uptrend_high_vol    │
                │ (calm bull)         │ (volatile rally)    │
                ├─────────────────────┼─────────────────────┤
   DOWNTREND    │ downtrend_low_vol   │ downtrend_high_vol  │
                │ (grinding bear)     │ (crisis/panic)      │
                └─────────────────────┴─────────────────────┘
```

**Rationale:**
- Captures two independent dimensions that affect strategy performance differently
- Maps naturally to existing alpha models (momentum for trends, vol-adjusted for high-vol)
- Complexity is manageable (4 strategies needed)

---

### D2: Regime Indicators

**Decision:** Two indicators

| Indicator | Purpose | Data Source |
|-----------|---------|-------------|
| SPY vs 200-day Moving Average | Trend detection | Price data (already available) |
| VIX level | Volatility detection | Macro data (FRED or similar) |

**Rationale:**
- Both are observable and forward-looking (VIX is implied vol)
- SPY/MA is a proven trend indicator
- VIX is the market standard for volatility regime
- Minimal additional data requirements

---

### D3: Regime Thresholds

**Decision:** Use hysteresis to prevent whipsawing

**Trend (SPY vs 200MA):**
```python
# Enter downtrend:  SPY < 200MA × 0.98  (2% below)
# Exit downtrend:   SPY > 200MA × 1.02  (2% above)
# Otherwise: stay in current trend state
TREND_HYSTERESIS_PCT = 0.02
```

**Volatility (VIX):**
```python
# Enter high_vol:  VIX > 25
# Exit high_vol:   VIX < 20
# Otherwise: stay in current vol state
VIX_HIGH_THRESHOLD = 25
VIX_LOW_THRESHOLD = 20
```

**Rationale:**
- Hysteresis prevents rapid switching at boundaries
- VIX 20-25 is the historical "elevated" zone
- 2% buffer around MA avoids noise-driven switches

---

### D4: Strategy-to-Regime Mapping

**Decision:** Determined empirically via regime-segmented analysis after each optimization run

**Process:**
1. Optimization produces ranked strategies (finalists)
2. Label historical periods by regime
3. Analyze each finalist's performance within each regime
4. Assign best-performing finalist to each regime

**Default Mapping (Hypothesis, to be validated):**

| Regime | Strategy Type | Rationale |
|--------|---------------|-----------|
| uptrend_low_vol | Aggressive momentum, longer lookback | Clean trends, ride them |
| uptrend_high_vol | Vol-adjusted momentum | Trends exist but noisy |
| downtrend_low_vol | Defensive / shorter momentum | Reduce exposure, faster signals |
| downtrend_high_vol | Minimal exposure / cash-like | Preserve capital |

**Important:** This mapping is recalculated after each optimization run since the finalist strategies may change.

---

### D5: Transition Rules

**Decision:** Hysteresis (different thresholds for entry/exit)

**Implementation:**
```python
class RegimeState:
    current_trend: str  # "uptrend" or "downtrend"
    current_vol: str    # "low_vol" or "high_vol"

def update_regime(state, spy_price, spy_200ma, vix):
    # Trend with hysteresis
    if spy_price < spy_200ma * 0.98:
        state.current_trend = "downtrend"
    elif spy_price > spy_200ma * 1.02:
        state.current_trend = "uptrend"
    # else: keep current_trend

    # Vol with hysteresis
    if vix > 25:
        state.current_vol = "high_vol"
    elif vix < 20:
        state.current_vol = "low_vol"
    # else: keep current_vol

    return f"{state.current_trend}_{state.current_vol}"
```

**Rationale:** Prevents thrashing at regime boundaries while still responding to clear regime changes.

---

### D6: Fallback Strategy

**Decision:** Dynamic fallback = strategy with highest composite_score from latest optimization

**Implementation:**
- Read from `best_strategy.yaml` produced by optimization
- Used when regime detection is uncertain or data is missing
- Automatically updates when optimization is re-run

**Rationale:** The most robust strategy (highest composite score across all periods) is the safest default when conditions are unclear.

---

### D7: Regime Check Frequency

**Decision:** Daily check, act on rebalance dates

**Implementation:**
```
Daily:
  1. Ingest latest data
  2. Calculate regime indicators (SPY/MA, VIX)
  3. Update regime state (with hysteresis)
  4. Log regime if changed
  5. Check circuit breakers (future: risk monitoring)

On Rebalance Date (weekly/monthly):
  1. Look up strategy for current regime
  2. Run that strategy's alpha model
  3. Construct portfolio
  4. Generate trades
```

**Rationale:** Daily awareness of regime changes, but trading only on scheduled rebalance dates minimizes turnover while maintaining responsiveness.

---

## Production Process (Target Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│  QUARTERLY: Strategy Validation & Mapping                       │
│                                                                 │
│  1. Run full optimization across all alpha models               │
│  2. Select top N finalists by composite score                   │
│  3. Run regime-segmented analysis on finalists                  │
│  4. Produce regime → strategy mapping                           │
│                                                                 │
│  Outputs:                                                       │
│  ├── artifacts/optimization/{timestamp}/finalists.yaml          │
│  ├── artifacts/optimization/{timestamp}/regime_mapping.yaml     │
│  └── artifacts/optimization/{timestamp}/fallback_strategy.yaml  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  DAILY: Data Ingestion & Monitoring                             │
│                                                                 │
│  1. Ingest latest price data                                    │
│  2. Ingest latest macro data (VIX)                              │
│  3. Update regime state (with hysteresis)                       │
│  4. Check circuit breakers (risk monitoring)                    │
│  5. Log alerts if regime changed or circuit breaker triggered   │
│                                                                 │
│  Outputs:                                                       │
│  └── data/state/current_regime.json                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (on rebalance dates only)
┌─────────────────────────────────────────────────────────────────┐
│  REBALANCE: Execute Strategy                                    │
│                                                                 │
│  1. Load current regime from state                              │
│  2. Look up strategy in regime_mapping.yaml                     │
│  3. Run selected strategy's alpha model                         │
│  4. Construct portfolio (equal weight top N)                    │
│  5. Generate trades (current → target)                          │
│  6. Log execution details                                       │
│                                                                 │
│  Outputs:                                                       │
│  ├── artifacts/rebalance/{date}/portfolio.yaml                  │
│  ├── artifacts/rebalance/{date}/trades.csv                      │
│  └── artifacts/rebalance/{date}/execution_log.json              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Future Evolution (Approach 2: Ensemble)

When ready to evolve to ensemble approach:

1. **Change weights from binary to continuous:**
   ```python
   # Approach 1 (current):
   weights = {regime_strategy: 1.0, others: 0.0}

   # Approach 2 (future):
   weights = {
       strategy_1: 0.4,  # Base weight + regime tilt
       strategy_2: 0.3,
       strategy_3: 0.2,
       strategy_4: 0.1,
   }
   ```

2. **Add weight calculation logic:**
   - Base: Equal weight (1/N)
   - Tilt: Adjust ±20% based on regime indicators
   - Constraints: No single strategy > 50%

3. **Architecture supports this:** The regime mapping can return weights instead of single strategy name.

---

## Data Requirements

### Must Be Preserved in Git

| File/Directory | Purpose | Update Frequency |
|----------------|---------|------------------|
| `configs/regimes/thresholds.yaml` | Regime threshold configuration | Rarely |
| `configs/regimes/default_mapping.yaml` | Default regime→strategy mapping | After optimization |

### Generated Artifacts (Git-ignored, but backed up)

| File/Directory | Purpose | Update Frequency |
|----------------|---------|------------------|
| `artifacts/optimization/{timestamp}/` | Optimization results | Per optimization run |
| `artifacts/optimization/{timestamp}/regime_mapping.yaml` | Computed mapping | Per optimization run |
| `data/state/current_regime.json` | Current regime state | Daily |
| `data/state/regime_history.parquet` | Regime change log | Daily |

### External Data Dependencies

| Data | Source | Frequency | Storage |
|------|--------|-----------|---------|
| VIX | FRED (VIXCLS) or similar | Daily | `data/raw/macro/VIX.parquet` |
| SPY prices | Existing price data | Daily | Existing snapshot |

---

## Implementation Tasks

See: `handoffs/tasks/handoff-IMPL-035-REGIME-STRATEGY-SYSTEM.md`

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Regime detection lags actual regime change | Hysteresis accepts some lag; focus on avoiding whipsaws over speed |
| Not enough historical data per regime | Start with conservative mapping; refine with more data |
| Strategy-regime mapping overfits | Use walk-forward validation; mapping based on financial intuition |
| Frequent regime changes cause high turnover | Hysteresis reduces switches; only act on rebalance dates |

---

## Success Metrics

1. **Implementation Success:**
   - [ ] Regime detector correctly classifies historical periods
   - [ ] Strategy mapping matches intuition (momentum in trends, defensive in crisis)
   - [ ] Production pipeline executes without errors

2. **Performance Success (to be measured):**
   - [ ] Regime-aware system has higher Sharpe than single-strategy baseline
   - [ ] Max drawdown reduced in crisis regimes
   - [ ] Win rate vs SPY improved

---

## References

- [regime-hypothesis.md](../research/regime-hypothesis.md) - Prior research on regimes
- [HANDOUT_strategy_optimizer.md](../../docs/handouts/HANDOUT_strategy_optimizer.md) - Optimizer documentation
- [handoff-IMPL-018-REGIME-ALPHA-INTEGRATION.md](../tasks/handoff-IMPL-018-REGIME-ALPHA-INTEGRATION.md) - Related implementation task

---

**Document Version:** 1.0
**Last Updated:** 2026-01-24
