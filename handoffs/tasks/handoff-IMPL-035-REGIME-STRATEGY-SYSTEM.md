# Task Handoff: IMPL-035 - Regime-Based Strategy Selection System

**Task ID:** IMPL-035
**Status:** ready
**Priority:** HIGH
**Type:** Architecture/Implementation Planning
**Estimated Effort:** Architect to break down into subtasks
**Dependencies:** None (supersedes/refines IMPL-018)

---

## Executive Summary

Design and implement a production-ready regime-based strategy selection system that:
1. Detects market regimes (4 states: trend × volatility)
2. Maps regimes to pre-validated strategies
3. Executes the appropriate strategy on rebalance dates
4. Supports evolution to ensemble-weighted approach in future

**Key Decision Document:** `handoffs/architecture/ADR-001-regime-based-strategy-selection.md`

---

## Architectural Decisions (Already Made)

The following decisions have been approved by the product owner. Do NOT revisit these unless explicitly requested:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| D1: Regimes | 4 (trend × vol matrix) | Captures two independent dimensions |
| D2: Indicators | SPY vs 200MA + VIX | Observable, proven, minimal data needs |
| D3: Thresholds | Hysteresis (±2% MA, VIX 20/25) | Prevents whipsawing |
| D4: Strategy mapping | Empirical (from optimization) | Recalculated per optimization run |
| D5: Transitions | Hysteresis | Different entry/exit thresholds |
| D6: Fallback | Dynamic (highest composite score) | Auto-updates with optimization |
| D7: Check frequency | Daily check, rebalance-day action | Balance responsiveness vs turnover |

---

## System Components to Implement

### Component 1: Regime Detector

**Purpose:** Classify current market into one of 4 regimes

**Location:** `src/quantetf/regime/detector.py`

**Interface:**
```python
class RegimeDetector:
    def __init__(self, config: RegimeConfig):
        """
        config contains:
        - trend_hysteresis_pct: float = 0.02
        - vix_high_threshold: float = 25
        - vix_low_threshold: float = 20
        """

    def detect(
        self,
        spy_price: float,
        spy_200ma: float,
        vix: float,
        previous_state: RegimeState,
    ) -> RegimeState:
        """Returns new regime state with hysteresis applied."""

    def get_regime_name(self, state: RegimeState) -> str:
        """Returns e.g., 'uptrend_low_vol'"""
```

**Inputs needed:**
- SPY price (from existing price data)
- SPY 200-day MA (calculate from price data)
- VIX (from macro data - may need ingestion task)

**Tests:**
- Correct regime classification for known scenarios
- Hysteresis prevents rapid switching
- Handles missing data gracefully

---

### Component 2: Regime-Segmented Analyzer

**Purpose:** Analyze historical performance of strategies by regime

**Location:** `src/quantetf/optimization/regime_analyzer.py`

**Interface:**
```python
class RegimeAnalyzer:
    def __init__(self, detector: RegimeDetector, data_access: DataAccessContext):
        pass

    def label_history(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        - date
        - regime (one of 4)
        - spy_price, spy_200ma, vix (for verification)
        """

    def analyze_strategy_by_regime(
        self,
        strategy_results: pd.DataFrame,  # from backtest
        regime_labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Returns performance metrics per strategy per regime:
        - strategy_name, regime, return, sharpe, max_drawdown, etc.
        """

    def compute_regime_mapping(
        self,
        analysis_results: pd.DataFrame,
        finalists: List[str],
    ) -> Dict[str, str]:
        """
        Returns {regime_name: best_strategy_name}
        Picks strategy with best risk-adjusted performance per regime.
        """
```

**Outputs:**
- `regime_mapping.yaml` in optimization artifacts

---

### Component 3: Extended Optimizer Output

**Purpose:** Add regime analysis to optimization pipeline

**Location:** Update `src/quantetf/optimization/optimizer.py`

**Changes:**
1. After finding winners, run regime-segmented analysis
2. Compute regime → strategy mapping
3. Output additional files:
   - `finalists.yaml` - Top N strategies
   - `regime_mapping.yaml` - Regime to strategy lookup
   - `fallback_strategy.yaml` - Best overall (already exists as `best_strategy.yaml`)

**Configuration:**
```yaml
# configs/optimization/regime_analysis.yaml
regime_analysis:
  enabled: true
  num_finalists: 6
  performance_metric: "information_ratio"  # or "sharpe", "sortino"
```

---

### Component 4: Daily Regime Monitor

**Purpose:** Track regime state daily, persist for rebalance use

**Location:** `src/quantetf/production/regime_monitor.py`

**Interface:**
```python
class DailyRegimeMonitor:
    def __init__(
        self,
        detector: RegimeDetector,
        data_access: DataAccessContext,
        state_path: Path = Path("data/state/current_regime.json"),
    ):
        pass

    def update(self, as_of: pd.Timestamp) -> RegimeState:
        """
        1. Load previous state from state_path
        2. Get current indicators (SPY, VIX)
        3. Compute new state with hysteresis
        4. Save to state_path
        5. Log if regime changed
        6. Return new state
        """

    def get_current_regime(self) -> str:
        """Read from state file, return regime name."""

    def get_history(self) -> pd.DataFrame:
        """Return regime history from parquet log."""
```

**State file format:**
```json
{
  "as_of": "2026-01-24",
  "regime": "uptrend_low_vol",
  "trend_state": "uptrend",
  "vol_state": "low_vol",
  "indicators": {
    "spy_price": 585.42,
    "spy_200ma": 542.18,
    "vix": 14.5
  },
  "previous_regime": "uptrend_low_vol",
  "regime_changed": false
}
```

---

### Component 5: Production Rebalancer

**Purpose:** Execute rebalance using regime-selected strategy

**Location:** `src/quantetf/production/rebalancer.py`

**Interface:**
```python
class RegimeAwareRebalancer:
    def __init__(
        self,
        regime_monitor: DailyRegimeMonitor,
        regime_mapping_path: Path,
        strategy_configs: Dict[str, Path],  # strategy_name -> config yaml
        data_access: DataAccessContext,
    ):
        pass

    def rebalance(self, as_of: pd.Timestamp) -> RebalanceResult:
        """
        1. Get current regime from monitor
        2. Look up strategy in regime_mapping
        3. Load strategy config
        4. Run alpha model, construct portfolio
        5. Generate trades (diff from current holdings)
        6. Return result with trades, logs
        """

    def dry_run(self, as_of: pd.Timestamp) -> RebalanceResult:
        """Same as rebalance but don't persist changes."""
```

---

### Component 6: VIX Data Ingestion (if not already available)

**Purpose:** Ensure VIX data is available for regime detection

**Location:** `scripts/ingest_vix.py` or extend existing macro ingestion

**Check first:** Does `src/quantetf/macro/` already handle VIX? If so, just ensure it's being ingested.

**If not:**
```python
# scripts/ingest_vix.py
from fredapi import Fred

def ingest_vix(output_path: Path, start_date: str = "2015-01-01"):
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    vix = fred.get_series("VIXCLS", start=start_date)
    vix.to_frame("vix").to_parquet(output_path / "VIX.parquet")
```

---

## Configuration Files to Create

### `configs/regimes/thresholds.yaml`

```yaml
# Regime detection thresholds
# See ADR-001 for rationale

trend:
  indicator: "spy_vs_200ma"
  hysteresis_pct: 0.02  # ±2% around 200MA

volatility:
  indicator: "vix"
  high_threshold: 25  # Enter high_vol when VIX > 25
  low_threshold: 20   # Exit high_vol when VIX < 20

regimes:
  - name: "uptrend_low_vol"
    trend: "uptrend"
    vol: "low_vol"
  - name: "uptrend_high_vol"
    trend: "uptrend"
    vol: "high_vol"
  - name: "downtrend_low_vol"
    trend: "downtrend"
    vol: "low_vol"
  - name: "downtrend_high_vol"
    trend: "downtrend"
    vol: "high_vol"
```

### `configs/regimes/default_mapping.yaml`

```yaml
# Default regime → strategy mapping
# Updated by optimization; this is fallback

mapping:
  uptrend_low_vol: "momentum_acceleration"
  uptrend_high_vol: "vol_adjusted_momentum"
  downtrend_low_vol: "momentum"  # shorter lookback
  downtrend_high_vol: "vol_adjusted_momentum"  # defensive

fallback: "momentum_acceleration"  # if regime unknown

# Metadata
generated_by: "default"
generated_at: "2026-01-24"
```

---

## Suggested Task Breakdown

For the architect/planner to refine:

| Task ID | Component | Description | Estimate | Dependencies |
|---------|-----------|-------------|----------|--------------|
| IMPL-035a | Regime Detector | Core regime classification with hysteresis | 2-3h | None |
| IMPL-035b | Config Files | Create threshold and mapping configs | 1h | IMPL-035a |
| IMPL-035c | VIX Ingestion | Ensure VIX data available (may already exist) | 1-2h | None |
| IMPL-035d | Regime Analyzer | Historical regime labeling and strategy analysis | 3-4h | IMPL-035a |
| IMPL-035e | Extended Optimizer | Add regime analysis to optimization output | 2-3h | IMPL-035d |
| IMPL-035f | Daily Monitor | Regime state tracking and persistence | 2-3h | IMPL-035a, IMPL-035c |
| IMPL-035g | Production Rebalancer | Regime-aware rebalance execution | 3-4h | IMPL-035f |
| IMPL-035h | Integration Tests | End-to-end testing | 2-3h | All above |
| IMPL-035i | CLI Scripts | Scripts for daily monitor and rebalance | 2h | IMPL-035f, IMPL-035g |

**Total estimated:** 18-25 hours

**Critical path:** IMPL-035a → IMPL-035d → IMPL-035e (regime analysis in optimizer)

---

## Relationship to Existing Tasks

| Existing Task | Relationship |
|---------------|--------------|
| IMPL-018 (Regime-Alpha Integration) | IMPL-030 supersedes and refines IMPL-018 |
| IMPL-016 (Alpha Selector) | May reuse `AlphaSelector` interface |
| IMPL-017 (Macro Data API) | May already provide VIX access |

**Recommendation:** Check status of IMPL-016, IMPL-017, IMPL-018 and determine what can be reused vs. needs rework.

---

## Files to Read Before Starting

1. `handoffs/architecture/ADR-001-regime-based-strategy-selection.md` - Decisions
2. `src/quantetf/optimization/optimizer.py` - Current optimizer
3. `src/quantetf/optimization/evaluator.py` - How composite score is calculated
4. `src/quantetf/alpha/selector.py` - Existing selector (if from IMPL-016)
5. `src/quantetf/macro/` - Existing macro data handling
6. `handoffs/research/regime-hypothesis.md` - Prior regime research

---

## Acceptance Criteria (Overall)

- [ ] Regime detector correctly classifies all 4 regimes with hysteresis
- [ ] Regime analyzer produces mapping from optimization results
- [ ] Optimizer outputs `finalists.yaml` and `regime_mapping.yaml`
- [ ] Daily monitor tracks regime state with persistence
- [ ] Production rebalancer uses regime-selected strategy
- [ ] All components have unit tests (>80% coverage)
- [ ] Integration test runs full cycle: optimize → monitor → rebalance
- [ ] Configuration is externalized (no hardcoded thresholds)
- [ ] Logging captures regime changes and strategy selections

---

## Definition of Done

1. All subtasks completed
2. Tests pass (unit + integration)
3. Documentation updated (README in relevant dirs)
4. Example usage documented
5. Completion report created
6. ADR-001 marked as IMPLEMENTED

---

## Notes for Architect

- **Keep it simple for v1:** Binary regime → strategy mapping, not weighted ensemble
- **Design for evolution:** Interfaces should support weights (future Approach 2)
- **Reuse existing code:** Check what IMPL-016/17/18 already built
- **Test with real data:** Use existing optimization results for validation
- **Circuit breakers:** Consider adding risk monitoring hooks (future task)

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Architect/Planner Agent
