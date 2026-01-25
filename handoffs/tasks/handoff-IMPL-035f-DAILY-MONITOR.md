# Task Handoff: IMPL-035f - Daily Regime Monitor

**Task ID:** IMPL-035f
**Parent Task:** IMPL-035 (Regime-Based Strategy Selection System)
**Status:** ready
**Priority:** HIGH
**Type:** Production Infrastructure
**Estimated Effort:** 2-3 hours
**Dependencies:** IMPL-035a (Regime Detector), IMPL-035c (VIX Data)

---

## Summary

Implement the Daily Regime Monitor that tracks regime state with persistence. This component runs daily, detects regime changes, and maintains state for the production rebalancer to use.

---

## Deliverables

1. **`src/quantetf/production/regime_monitor.py`** - DailyRegimeMonitor class
2. **`data/state/`** - State persistence directory
3. **`tests/production/test_regime_monitor.py`** - Unit tests
4. **State files:**
   - `data/state/current_regime.json` - Current regime state
   - `data/state/regime_history.parquet` - Historical regime log

---

## Technical Specification

### Interface Design

```python
# src/quantetf/production/regime_monitor.py
"""Daily regime monitoring with state persistence."""

from pathlib import Path
from typing import Optional
import json
import pandas as pd
import logging
from datetime import datetime

from quantetf.regime.detector import RegimeDetector
from quantetf.regime.indicators import RegimeIndicators
from quantetf.regime.types import RegimeState, RegimeConfig
from quantetf.regime.config import load_thresholds

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path("data/state")


class DailyRegimeMonitor:
    """
    Monitors market regime daily with state persistence.

    This component:
    1. Loads previous regime state from disk
    2. Gets current market indicators (SPY, VIX)
    3. Detects current regime using hysteresis
    4. Persists new state to disk
    5. Logs regime changes for alerting

    Usage:
        monitor = DailyRegimeMonitor(data_access)
        state = monitor.update(as_of=pd.Timestamp.now())
        print(f"Current regime: {state.name}")
    """

    def __init__(
        self,
        data_access: "DataAccessContext",
        state_dir: Path = DEFAULT_STATE_DIR,
        config: Optional[RegimeConfig] = None,
    ):
        """
        Initialize monitor.

        Args:
            data_access: Data access context for indicators
            state_dir: Directory for state persistence
            config: Regime detection config (uses defaults if not provided)
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.state_dir / "current_regime.json"
        self.history_file = self.state_dir / "regime_history.parquet"

        # Initialize components
        self.config = config or load_thresholds()
        self.detector = RegimeDetector(self.config)
        self.indicators = RegimeIndicators(data_access)

    def update(self, as_of: Optional[pd.Timestamp] = None) -> RegimeState:
        """
        Update regime state for given date.

        Args:
            as_of: Date to evaluate (defaults to today)

        Returns:
            Current RegimeState after update
        """
        as_of = as_of or pd.Timestamp.now().normalize()

        # Load previous state
        previous_state = self.load_state()

        # Get current indicators
        try:
            indicators = self.indicators.get_current_indicators(as_of)
        except Exception as e:
            logger.error(f"Failed to get indicators: {e}")
            if previous_state:
                logger.warning("Using previous state as fallback")
                return previous_state
            raise

        # Detect current regime
        new_state = self.detector.detect(
            spy_price=indicators["spy_price"],
            spy_200ma=indicators["spy_200ma"],
            vix=indicators["vix"],
            previous_state=previous_state,
            as_of=as_of,
        )

        # Check for regime change
        regime_changed = (
            previous_state is None or
            new_state.name != previous_state.name
        )

        if regime_changed:
            self._log_regime_change(previous_state, new_state)

        # Persist state
        self.save_state(new_state)
        self._append_history(new_state, regime_changed)

        return new_state

    def load_state(self) -> Optional[RegimeState]:
        """Load current regime state from disk."""
        if not self.state_file.exists():
            logger.info("No previous state found, starting fresh")
            return None

        try:
            with open(self.state_file) as f:
                data = json.load(f)
            return RegimeState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def save_state(self, state: RegimeState) -> None:
        """Save current regime state to disk."""
        data = state.to_dict()
        data["updated_at"] = datetime.now().isoformat()

        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved state: {state.name}")

    def get_current_regime(self) -> str:
        """Get current regime name from persisted state."""
        state = self.load_state()
        if state is None:
            raise ValueError("No regime state available. Run update() first.")
        return state.name

    def get_history(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Get regime history from log.

        Args:
            start_date: Filter start (optional)
            end_date: Filter end (optional)

        Returns:
            DataFrame with columns: date, regime, regime_changed, spy_price, vix
        """
        if not self.history_file.exists():
            return pd.DataFrame()

        df = pd.read_parquet(self.history_file)

        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        return df

    def _append_history(self, state: RegimeState, regime_changed: bool) -> None:
        """Append state to history log."""
        record = pd.DataFrame([{
            "date": state.as_of,
            "regime": state.name,
            "trend": state.trend.value,
            "vol": state.vol.value,
            "spy_price": state.spy_price,
            "spy_200ma": state.spy_200ma,
            "vix": state.vix,
            "regime_changed": regime_changed,
            "recorded_at": pd.Timestamp.now(),
        }])

        if self.history_file.exists():
            existing = pd.read_parquet(self.history_file)
            # Avoid duplicate entries for same date
            existing = existing[existing["date"] != state.as_of]
            df = pd.concat([existing, record], ignore_index=True)
        else:
            df = record

        df.to_parquet(self.history_file, index=False)

    def _log_regime_change(
        self,
        previous: Optional[RegimeState],
        current: RegimeState,
    ) -> None:
        """Log regime change for alerting."""
        prev_name = previous.name if previous else "none"
        logger.warning(
            f"REGIME CHANGE: {prev_name} -> {current.name} "
            f"(SPY={current.spy_price:.2f}, VIX={current.vix:.2f})"
        )

        # Could integrate with alerting system here
        # e.g., send email, Slack notification, etc.

    def get_regime_summary(self) -> dict:
        """
        Get summary of current regime state.

        Returns:
            Dict suitable for display or API response
        """
        state = self.load_state()
        if state is None:
            return {"error": "No state available"}

        return {
            "regime": state.name,
            "trend": state.trend.value,
            "volatility": state.vol.value,
            "as_of": state.as_of.isoformat(),
            "indicators": {
                "spy_price": state.spy_price,
                "spy_200ma": state.spy_200ma,
                "spy_vs_ma_pct": (
                    (state.spy_price / state.spy_200ma - 1) * 100
                    if state.spy_200ma else None
                ),
                "vix": state.vix,
            },
        }
```

### State File Format

```json
// data/state/current_regime.json
{
  "regime": "uptrend_low_vol",
  "trend": "uptrend",
  "vol": "low_vol",
  "as_of": "2026-01-24T00:00:00",
  "indicators": {
    "spy_price": 585.42,
    "spy_200ma": 542.18,
    "vix": 14.5
  },
  "updated_at": "2026-01-24T08:30:00"
}
```

### History File Schema

```
regime_history.parquet:
- date: timestamp
- regime: string
- trend: string
- vol: string
- spy_price: float
- spy_200ma: float
- vix: float
- regime_changed: bool
- recorded_at: timestamp
```

---

## Test Cases

```python
# tests/production/test_regime_monitor.py
import pytest
import pandas as pd
from pathlib import Path
import json

from quantetf.production.regime_monitor import DailyRegimeMonitor


class TestDailyRegimeMonitor:
    """Test daily regime monitoring."""

    @pytest.fixture
    def state_dir(self, tmp_path):
        return tmp_path / "state"

    @pytest.fixture
    def monitor(self, data_access, state_dir):
        return DailyRegimeMonitor(
            data_access=data_access,
            state_dir=state_dir,
        )

    def test_update_creates_state_file(self, monitor, state_dir):
        """First update should create state file."""
        state = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        assert (state_dir / "current_regime.json").exists()
        assert state.name in [
            "uptrend_low_vol",
            "uptrend_high_vol",
            "downtrend_low_vol",
            "downtrend_high_vol",
        ]

    def test_update_persists_and_loads(self, monitor):
        """State should survive save/load cycle."""
        state1 = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        # Create new monitor instance (simulates restart)
        monitor2 = DailyRegimeMonitor(
            data_access=monitor.indicators.data_access,
            state_dir=monitor.state_dir,
        )
        loaded = monitor2.load_state()

        assert loaded is not None
        assert loaded.name == state1.name

    def test_history_appended(self, monitor, state_dir):
        """Each update should append to history."""
        monitor.update(as_of=pd.Timestamp("2026-01-18"))
        monitor.update(as_of=pd.Timestamp("2026-01-19"))
        monitor.update(as_of=pd.Timestamp("2026-01-20"))

        history = monitor.get_history()
        assert len(history) == 3

    def test_get_current_regime(self, monitor):
        """get_current_regime should return regime name."""
        monitor.update(as_of=pd.Timestamp("2026-01-20"))
        regime = monitor.get_current_regime()

        assert isinstance(regime, str)
        assert "_" in regime  # e.g., "uptrend_low_vol"

    def test_no_state_raises(self, monitor):
        """get_current_regime should raise if no state."""
        with pytest.raises(ValueError, match="No regime state"):
            monitor.get_current_regime()

    def test_hysteresis_maintained_across_updates(self, monitor):
        """Hysteresis should persist across monitor restarts."""
        # First update establishes state
        monitor.update(as_of=pd.Timestamp("2026-01-15"))

        # Simulate restart with new monitor
        monitor2 = DailyRegimeMonitor(
            data_access=monitor.indicators.data_access,
            state_dir=monitor.state_dir,
        )

        # Previous state should be loaded for hysteresis
        state = monitor2.update(as_of=pd.Timestamp("2026-01-16"))
        assert state is not None

    def test_regime_change_logged(self, monitor, caplog):
        """Regime changes should be logged as warnings."""
        # This test may need mock data that causes regime change
        import logging
        caplog.set_level(logging.WARNING)

        # Update multiple times...
        monitor.update(as_of=pd.Timestamp("2026-01-20"))

        # Check if any regime change was logged
        # (depends on actual data causing change)

    def test_get_regime_summary(self, monitor):
        """Summary should include all relevant info."""
        monitor.update(as_of=pd.Timestamp("2026-01-20"))
        summary = monitor.get_regime_summary()

        assert "regime" in summary
        assert "indicators" in summary
        assert "spy_price" in summary["indicators"]
        assert "vix" in summary["indicators"]
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantetf/production/regime_monitor.py` | Monitor class |
| `tests/production/test_regime_monitor.py` | Unit tests |
| `data/state/.gitkeep` | Ensure directory exists |

---

## Acceptance Criteria

- [ ] `DailyRegimeMonitor` correctly updates and persists regime state
- [ ] State survives process restart (load from disk)
- [ ] History is appended correctly without duplicates
- [ ] Regime changes are logged at WARNING level
- [ ] Hysteresis is maintained across updates
- [ ] `get_regime_summary()` returns useful info
- [ ] All tests pass

---

## Integration Notes

1. **Daily cron job:** This will be called by `scripts/run_daily_monitor.py` (IMPL-035i)
2. **Alerting:** Consider adding Slack/email integration for regime changes
3. **State directory:** Should be in `.gitignore` but backed up

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Coding Agent
