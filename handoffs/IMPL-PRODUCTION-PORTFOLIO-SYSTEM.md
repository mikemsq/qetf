# Implementation Handoff: Production Portfolio Management System

**Date:** January 16, 2026
**Source:** Research findings from `handoffs/RESEARCH-FINDINGS-STRATEGY-OPTIMIZATION.md`
**Status:** READY FOR IMPLEMENTATION

---

## Executive Summary

Implement a production-ready portfolio management system based on validated research findings. The winning strategy (Value-Momentum blend) achieved Sharpe 2.85 and +106% active return vs SPY. This handoff defines 5 implementation tasks for coder agents.

---

## Task Overview

| Task ID | Description | Priority | Dependencies | Est. LOC |
|---------|-------------|----------|--------------|----------|
| IMPL-010 | Risk Overlays Module | P0 | None | ~300 |
| IMPL-011 | Portfolio State Management | P0 | None | ~250 |
| IMPL-012 | Enhanced Production Pipeline | P1 | IMPL-010, IMPL-011 | ~200 |
| IMPL-013 | Monitoring & Alerts Module | P1 | IMPL-011 | ~300 |
| IMPL-014 | Production Config & Scripts | P2 | IMPL-010-013 | ~200 |

---

## IMPL-010: Risk Overlays Module

**File:** `src/quantetf/risk/overlays.py` (PARTIALLY CREATED - needs review/completion)

### Description
Implement 4 risk overlays using chain-of-responsibility pattern to modify target weights based on risk constraints.

### Requirements

1. **VolatilityTargeting**
   - Scale exposure to target 15% annual volatility
   - Use EWMA volatility estimation (halflife=20 days, lookback=60 days)
   - Clamp scale factor between 0.25 and 1.50
   - Integration: Use pattern from `src/quantetf/risk/covariance.py:14` (EWMACovariance)

2. **PositionLimitOverlay**
   - Cap any single position at 25% max weight
   - Redistribute excess weight proportionally to other positions
   - Track which tickers were capped in diagnostics

3. **DrawdownCircuitBreaker**
   - Soft threshold (10% DD) → 75% exposure
   - Hard threshold (20% DD) → 50% exposure
   - Exit threshold (30% DD) → 25% exposure
   - Requires `PortfolioState` with `peak_nav` attribute from IMPL-011

4. **VIXRegimeOverlay**
   - High VIX (>30) → 50% to defensive assets
   - Elevated VIX (>25) → 25% to defensive assets
   - Defensive tickers: AGG, TLT, GLD, USMV, SPLV
   - Integration: Use `MacroDataLoader` from `src/quantetf/data/macro_loader.py:15`

### Interface
```python
class RiskOverlay(ABC):
    @abstractmethod
    def apply(
        self,
        target_weights: pd.Series,
        as_of: pd.Timestamp,
        store: DataStore,
        portfolio_state: Optional[PortfolioState],
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Return (adjusted_weights, diagnostics)."""

def apply_overlay_chain(
    target_weights: pd.Series,
    overlays: list[RiskOverlay],
    as_of: pd.Timestamp,
    store: DataStore,
    portfolio_state: Optional[PortfolioState],
) -> tuple[pd.Series, dict[str, dict[str, Any]]]:
    """Apply overlays sequentially, return final weights + all diagnostics."""
```

### Tests Required
- `tests/risk/test_overlays.py`
  - `test_volatility_targeting_scales_correctly` - verify scale factor calculation
  - `test_volatility_targeting_clamps_bounds` - verify min/max scale limits
  - `test_position_limit_caps_weights` - verify 25% cap applied
  - `test_position_limit_redistributes` - verify excess redistributed
  - `test_drawdown_circuit_breaker_thresholds` - verify each level triggers correctly
  - `test_vix_regime_shifts_to_defensive` - verify defensive allocation

### Acceptance Criteria
- [ ] All 4 overlays implemented with frozen dataclass pattern
- [ ] `apply_overlay_chain()` function works correctly
- [ ] Exports added to `src/quantetf/risk/__init__.py`
- [ ] All unit tests pass

---

## IMPL-011: Portfolio State Management

**File:** `src/quantetf/production/state.py` (NEW)

### Description
Implement portfolio state persistence with support for SQLite, JSON, and in-memory backends.

### Requirements

1. **PortfolioState dataclass**
   ```python
   @dataclass(frozen=True)
   class PortfolioState:
       as_of: pd.Timestamp
       holdings: pd.Series      # ticker -> shares
       weights: pd.Series       # ticker -> weight
       nav: float
       cost_basis: pd.Series    # ticker -> avg cost
       peak_nav: float          # For drawdown calculation
       created_at: datetime
   ```

2. **PortfolioHistory**
   ```python
   @dataclass
   class PortfolioHistory:
       states: list[PortfolioState]

       def get_peak_nav(self) -> float: ...
       def get_current_drawdown(self) -> float: ...
       def get_nav_series(self) -> pd.Series: ...
   ```

3. **PortfolioStateManager (ABC)**
   ```python
   class PortfolioStateManager(ABC):
       @abstractmethod
       def save_state(self, state: PortfolioState) -> None: ...

       @abstractmethod
       def get_latest_state(self) -> Optional[PortfolioState]: ...

       @abstractmethod
       def get_state_as_of(self, as_of: pd.Timestamp) -> Optional[PortfolioState]: ...

       @abstractmethod
       def get_history(self, lookback_days: int = 365) -> PortfolioHistory: ...

       def calculate_drift(
           self, current: pd.Series, target: pd.Series, threshold: float = 0.02
       ) -> pd.DataFrame: ...
   ```

4. **SQLiteStateManager** - Production backend
   - Schema:
     ```sql
     CREATE TABLE portfolio_states (
         id INTEGER PRIMARY KEY,
         as_of TEXT NOT NULL UNIQUE,
         nav REAL NOT NULL,
         peak_nav REAL NOT NULL,
         created_at TEXT NOT NULL
     );
     CREATE TABLE holdings (
         state_id INTEGER REFERENCES portfolio_states(id),
         ticker TEXT NOT NULL,
         shares REAL NOT NULL,
         weight REAL NOT NULL,
         cost_basis REAL,
         PRIMARY KEY(state_id, ticker)
     );
     ```

5. **JSONStateManager** - Simple file-based backend

6. **InMemoryStateManager** - For testing

### Tests Required
- `tests/production/test_state.py`
  - `test_sqlite_state_save_and_load`
  - `test_sqlite_state_history`
  - `test_json_state_persistence`
  - `test_portfolio_history_peak_nav`
  - `test_portfolio_history_drawdown`
  - `test_drift_detection`

### Acceptance Criteria
- [ ] All 3 backend implementations working
- [ ] SQLite schema auto-creates on first use
- [ ] History correctly tracks peak NAV
- [ ] Drift detection works with configurable threshold
- [ ] All unit tests pass

---

## IMPL-012: Enhanced Production Pipeline

**File:** `src/quantetf/production/pipeline.py` (ENHANCE existing)

### Description
Enhance the existing thin production pipeline to integrate risk overlays, pre-trade checks, and state management.

### Current State
The existing pipeline at `src/quantetf/production/pipeline.py` is minimal (~50 LOC). It needs enhancement.

### Requirements

1. **PipelineConfig dataclass**
   ```python
   @dataclass
   class PipelineConfig:
       strategy_config_path: Path
       risk_overlays: list[RiskOverlay]
       pre_trade_checks: list[PreTradeCheck]
       state_manager: PortfolioStateManager
       rebalance_schedule: str = "monthly"  # "monthly", "weekly", "daily"
       trade_threshold: float = 0.005  # Ignore trades < 0.5%
       dry_run: bool = False
   ```

2. **PreTradeCheck base class**
   ```python
   class PreTradeCheck(ABC):
       @abstractmethod
       def check(
           self, trades: pd.DataFrame, state: PortfolioState, as_of: pd.Timestamp
       ) -> tuple[bool, str]:
           """Return (passed, reason)."""
   ```

3. **Pre-trade check implementations:**
   - `MaxTurnoverCheck` - Block if turnover > 50%
   - `SectorConcentrationCheck` - Block if any sector > 40%

4. **PipelineResult dataclass**
   ```python
   @dataclass
   class PipelineResult:
       as_of: pd.Timestamp
       target_weights: pd.Series
       adjusted_weights: pd.Series  # After risk overlays
       trades: pd.DataFrame
       pre_trade_checks_passed: bool
       check_results: dict[str, Any]
       overlay_diagnostics: dict[str, Any]
       execution_status: str  # "pending", "executed", "blocked"
   ```

5. **Enhanced ProductionPipeline.run() method:**
   - Load current state from state_manager
   - Generate alpha scores
   - Construct raw target weights
   - Apply risk overlay chain
   - Generate trades using existing `diff_trades()`
   - Run pre-trade checks
   - Return PipelineResult

6. **Rebalance scheduling:**
   - `should_rebalance(as_of: pd.Timestamp) -> bool`
   - Monthly: Last business day of month
   - Weekly: Friday

### Integration Points
- Use `diff_trades()` from `src/quantetf/production/recommendations.py`
- Use `apply_overlay_chain()` from IMPL-010
- Use `PortfolioStateManager` from IMPL-011
- Use `AlphaModelRegistry` from `src/quantetf/alpha/factory.py`

### Tests Required
- `tests/production/test_pipeline_enhanced.py`
  - `test_overlay_chain_applied`
  - `test_pre_trade_checks_block_high_turnover`
  - `test_rebalance_schedule_monthly`
  - `test_pipeline_result_structure`

### Acceptance Criteria
- [ ] Pipeline integrates all risk overlays
- [ ] Pre-trade checks can block execution
- [ ] Rebalance scheduling works correctly
- [ ] Backward compatible with existing code
- [ ] All tests pass

---

## IMPL-013: Monitoring & Alerts Module

**Directory:** `src/quantetf/monitoring/` (NEW)

### Description
Create a monitoring module for NAV tracking, drawdown alerts, regime change detection, and data quality checks.

### Files to Create

1. **`__init__.py`** - Module exports

2. **`alerts.py`** - Alert system
   ```python
   @dataclass(frozen=True)
   class Alert:
       timestamp: datetime
       level: str  # "INFO", "WARNING", "CRITICAL"
       category: str  # "DRAWDOWN", "REGIME", "DATA_QUALITY", "REBALANCE"
       message: str
       data: dict[str, Any]

   class AlertHandler(ABC):
       @abstractmethod
       def send(self, alert: Alert) -> None: ...

   class ConsoleAlertHandler(AlertHandler): ...
   class FileAlertHandler(AlertHandler): ...  # Append to log file

   class AlertManager:
       handlers: list[AlertHandler]
       def emit(self, alert: Alert) -> None: ...
   ```

3. **`nav_tracker.py`** - NAV and drawdown monitoring
   ```python
   class NAVTracker:
       def __init__(
           self,
           state_manager: PortfolioStateManager,
           alert_manager: AlertManager,
           drawdown_thresholds: tuple[float, ...] = (0.10, 0.20, 0.30),
       ): ...

       def update(self, as_of: pd.Timestamp, nav: float) -> dict[str, Any]:
           """Update NAV, check thresholds, emit alerts."""
   ```

4. **`regime.py`** - Regime change monitoring
   ```python
   class RegimeMonitor:
       def __init__(
           self,
           macro_loader: MacroDataLoader,
           alert_manager: AlertManager,
       ): ...

       def check(self, as_of: str) -> dict[str, Any]:
           """Check for regime changes, emit alerts."""
   ```
   - Integration: Use `RegimeDetector` from `src/quantetf/data/macro_loader.py:124`

5. **`quality.py`** - Data quality checks
   ```python
   class DataQualityChecker:
       def check_price_staleness(
           self, store: DataStore, tickers: list[str], max_stale_days: int = 3
       ) -> list[str]:
           """Return tickers with stale data."""

       def check_price_gaps(
           self, store: DataStore, tickers: list[str], max_gap_days: int = 5
       ) -> dict[str, int]:
           """Return tickers with data gaps."""
   ```

### Tests Required
- `tests/monitoring/test_alerts.py`
- `tests/monitoring/test_nav_tracker.py`
- `tests/monitoring/test_regime_monitor.py`

### Acceptance Criteria
- [ ] Alert system with pluggable handlers
- [ ] NAV tracker emits alerts at thresholds
- [ ] Regime monitor detects and alerts on changes
- [ ] Data quality checker identifies issues
- [ ] All tests pass

---

## IMPL-014: Production Config & Scripts

**Files:**
- `configs/strategies/production_value_momentum.yaml` (NEW)
- `scripts/run_production_pipeline.py` (NEW)
- `scripts/run_daily_monitoring.py` (NEW)

### Description
Create the production strategy configuration and operational scripts.

### Requirements

1. **Production Strategy Config** (`configs/strategies/production_value_momentum.yaml`)
   ```yaml
   name: production_value_momentum
   description: Production value-momentum strategy (Sharpe 2.85, +106% vs SPY)

   universe: configs/universes/tier4_broad_200.yaml
   schedule: configs/schedules/monthly_rebalance.yaml
   cost_model: configs/costs/flat_10bps.yaml

   alpha_model:
     type: value_momentum
     momentum_weight: 0.5
     value_weight: 0.5
     momentum_lookback: 252
     value_lookback: 252
     min_periods: 200

   portfolio_construction:
     type: equal_weight_top_n
     top_n: 5

   risk_overlays:
     volatility_targeting:
       enabled: true
       target_vol: 0.15
       lookback_days: 60
       min_scale: 0.25
       max_scale: 1.50
     position_limits:
       enabled: true
       max_weight: 0.25
       redistribute: true
     drawdown_circuit_breaker:
       enabled: true
       soft_threshold: 0.10
       hard_threshold: 0.20
       exit_threshold: 0.30
     vix_regime:
       enabled: true
       high_vix_threshold: 30.0
       elevated_vix_threshold: 25.0
       defensive_tickers: [AGG, TLT, GLD, USMV, SPLV]

   state:
     backend: sqlite
     path: data/production/portfolio_state.db

   monitoring:
     drawdown_alerts: [0.10, 0.20, 0.30]
     regime_change_alerts: true
   ```

2. **Production Pipeline Script** (`scripts/run_production_pipeline.py`)
   ```bash
   python scripts/run_production_pipeline.py \
     --config configs/strategies/production_value_momentum.yaml \
     --snapshot data/snapshots/snapshot_20260115_170559 \
     --dry-run  # or --execute
   ```
   - Load config
   - Initialize state manager
   - Run pipeline
   - Output recommendations to JSON/CSV

3. **Daily Monitoring Script** (`scripts/run_daily_monitoring.py`)
   ```bash
   python scripts/run_daily_monitoring.py \
     --config configs/strategies/production_value_momentum.yaml \
     --check-quality \
     --check-regime \
     --update-nav
   ```

### Acceptance Criteria
- [ ] Production config validates against schema
- [ ] Pipeline script runs with --dry-run
- [ ] Monitoring script performs all checks
- [ ] Output is human-readable and machine-parseable

---

## Implementation Order

```
IMPL-010 (Risk Overlays)  ──┐
                            ├──> IMPL-012 (Pipeline) ──> IMPL-014 (Config/Scripts)
IMPL-011 (State Mgmt)     ──┤
                            └──> IMPL-013 (Monitoring)
```

**Recommended sequence:**
1. IMPL-010 + IMPL-011 (can be parallel)
2. IMPL-012 (depends on 010, 011)
3. IMPL-013 (depends on 011)
4. IMPL-014 (depends on all)

---

## Notes for Coder Agents

1. **Follow existing patterns:**
   - Use `@dataclass(frozen=True)` for immutable data structures (see `src/quantetf/types.py`)
   - Use abstract base classes with `ABC` (see `src/quantetf/risk/base.py`)
   - Return diagnostics dicts for observability

2. **Code style:**
   - Type hints on all public functions
   - Docstrings with Args/Returns
   - No external dependencies beyond existing (pandas, numpy, sqlite3)

3. **Testing:**
   - Use pytest fixtures for common setup
   - Mock external dependencies (DataStore, MacroDataLoader)
   - Test edge cases (empty data, missing tickers)

4. **File `src/quantetf/risk/overlays.py` already exists:**
   - Review and complete/fix as needed
   - Ensure it matches the interface spec above

---

**Document Author:** Planner Agent
**Review Status:** Ready for Coder Agents
