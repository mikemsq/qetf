# Task Handoff: IMPL-035a - Regime Detector

**Task ID:** IMPL-035a
**Parent Task:** IMPL-035 (Regime-Based Strategy Selection System)
**Status:** ready
**Priority:** HIGH
**Type:** Core Implementation
**Estimated Effort:** 2-3 hours
**Dependencies:** None

---

## Summary

Implement the core regime detection module that classifies markets into 4 regimes based on trend (SPY vs 200MA) and volatility (VIX level). The detector must support hysteresis to prevent whipsawing at regime boundaries.

---

## Deliverables

1. **`src/quantetf/regime/detector.py`** - Core regime detector class
2. **`src/quantetf/regime/__init__.py`** - Package initialization
3. **`src/quantetf/regime/types.py`** - Regime-specific dataclasses
4. **`tests/regime/test_detector.py`** - Comprehensive unit tests

---

## Technical Specification

### Regime States (4 regimes)

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

### Hysteresis Rules (from ADR-001)

**Trend (SPY vs 200MA):**
- Enter downtrend: `SPY < 200MA × 0.98` (2% below)
- Exit downtrend: `SPY > 200MA × 1.02` (2% above)
- Otherwise: maintain current trend state

**Volatility (VIX):**
- Enter high_vol: `VIX > 25`
- Exit high_vol: `VIX < 20`
- Otherwise: maintain current vol state

### Interface Design

```python
# src/quantetf/regime/types.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class TrendState(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"


class VolatilityState(Enum):
    LOW_VOL = "low_vol"
    HIGH_VOL = "high_vol"


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for regime detection thresholds."""
    trend_hysteresis_pct: float = 0.02  # ±2% around 200MA
    vix_high_threshold: float = 25.0    # Enter high_vol when VIX > 25
    vix_low_threshold: float = 20.0     # Exit high_vol when VIX < 20


@dataclass
class RegimeState:
    """Current regime state with hysteresis memory."""
    trend: TrendState
    vol: VolatilityState
    as_of: pd.Timestamp

    # Indicator values for debugging/logging
    spy_price: Optional[float] = None
    spy_200ma: Optional[float] = None
    vix: Optional[float] = None

    @property
    def name(self) -> str:
        """Return regime name, e.g., 'uptrend_low_vol'."""
        return f"{self.trend.value}_{self.vol.value}"

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "regime": self.name,
            "trend": self.trend.value,
            "vol": self.vol.value,
            "as_of": self.as_of.isoformat(),
            "indicators": {
                "spy_price": self.spy_price,
                "spy_200ma": self.spy_200ma,
                "vix": self.vix,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegimeState":
        """Deserialize from JSON storage."""
        return cls(
            trend=TrendState(data["trend"]),
            vol=VolatilityState(data["vol"]),
            as_of=pd.Timestamp(data["as_of"]),
            spy_price=data.get("indicators", {}).get("spy_price"),
            spy_200ma=data.get("indicators", {}).get("spy_200ma"),
            vix=data.get("indicators", {}).get("vix"),
        )


# src/quantetf/regime/detector.py
from typing import Optional
import pandas as pd
import logging

from .types import RegimeConfig, RegimeState, TrendState, VolatilityState

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regime based on trend (SPY vs 200MA) and volatility (VIX).
    Uses hysteresis to prevent rapid switching at boundaries.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize detector with threshold configuration.

        Args:
            config: Regime thresholds. Uses defaults if not provided.
        """
        self.config = config or RegimeConfig()

    def detect(
        self,
        spy_price: float,
        spy_200ma: float,
        vix: float,
        previous_state: Optional[RegimeState],
        as_of: pd.Timestamp,
    ) -> RegimeState:
        """
        Detect current regime with hysteresis.

        Args:
            spy_price: Current SPY closing price
            spy_200ma: 200-day moving average of SPY
            vix: Current VIX level
            previous_state: Previous regime state (for hysteresis)
            as_of: Current date

        Returns:
            New RegimeState with updated values
        """
        # Determine trend with hysteresis
        trend = self._detect_trend(spy_price, spy_200ma, previous_state)

        # Determine volatility with hysteresis
        vol = self._detect_volatility(vix, previous_state)

        new_state = RegimeState(
            trend=trend,
            vol=vol,
            as_of=as_of,
            spy_price=spy_price,
            spy_200ma=spy_200ma,
            vix=vix,
        )

        # Log regime change
        if previous_state and new_state.name != previous_state.name:
            logger.info(
                f"Regime change: {previous_state.name} -> {new_state.name} "
                f"(SPY={spy_price:.2f}, 200MA={spy_200ma:.2f}, VIX={vix:.2f})"
            )

        return new_state

    def _detect_trend(
        self,
        spy_price: float,
        spy_200ma: float,
        previous_state: Optional[RegimeState],
    ) -> TrendState:
        """Detect trend state with hysteresis."""
        hysteresis = self.config.trend_hysteresis_pct

        # Clear signals (beyond hysteresis bands)
        if spy_price < spy_200ma * (1 - hysteresis):
            return TrendState.DOWNTREND
        elif spy_price > spy_200ma * (1 + hysteresis):
            return TrendState.UPTREND

        # Inside hysteresis band - maintain previous state
        if previous_state:
            return previous_state.trend

        # No previous state, default to uptrend if above MA
        return TrendState.UPTREND if spy_price >= spy_200ma else TrendState.DOWNTREND

    def _detect_volatility(
        self,
        vix: float,
        previous_state: Optional[RegimeState],
    ) -> VolatilityState:
        """Detect volatility state with hysteresis."""
        # Clear signals (beyond hysteresis bands)
        if vix > self.config.vix_high_threshold:
            return VolatilityState.HIGH_VOL
        elif vix < self.config.vix_low_threshold:
            return VolatilityState.LOW_VOL

        # Inside hysteresis band (20-25) - maintain previous state
        if previous_state:
            return previous_state.vol

        # No previous state, default to low_vol
        return VolatilityState.LOW_VOL

    @staticmethod
    def get_default_state(as_of: pd.Timestamp) -> RegimeState:
        """Return default regime state (uptrend_low_vol)."""
        return RegimeState(
            trend=TrendState.UPTREND,
            vol=VolatilityState.LOW_VOL,
            as_of=as_of,
        )
```

---

## Test Cases

### Required Tests (minimum)

```python
# tests/regime/test_detector.py
import pytest
import pandas as pd
from quantetf.regime.detector import RegimeDetector
from quantetf.regime.types import RegimeConfig, RegimeState, TrendState, VolatilityState


class TestRegimeDetector:
    """Test regime detection with hysteresis."""

    @pytest.fixture
    def detector(self):
        return RegimeDetector()

    @pytest.fixture
    def as_of(self):
        return pd.Timestamp("2026-01-24")

    # --- Trend Detection Tests ---

    def test_clear_uptrend(self, detector, as_of):
        """SPY well above 200MA should be uptrend."""
        state = detector.detect(
            spy_price=600, spy_200ma=500, vix=15, previous_state=None, as_of=as_of
        )
        assert state.trend == TrendState.UPTREND

    def test_clear_downtrend(self, detector, as_of):
        """SPY well below 200MA should be downtrend."""
        state = detector.detect(
            spy_price=480, spy_200ma=500, vix=15, previous_state=None, as_of=as_of
        )
        assert state.trend == TrendState.DOWNTREND

    def test_trend_hysteresis_maintains_uptrend(self, detector, as_of):
        """SPY slightly below 200MA but inside band should maintain uptrend."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 499 is only 0.2% below 500 (inside 2% band)
        state = detector.detect(
            spy_price=499, spy_200ma=500, vix=15, previous_state=previous, as_of=as_of
        )
        assert state.trend == TrendState.UPTREND

    def test_trend_hysteresis_maintains_downtrend(self, detector, as_of):
        """SPY slightly above 200MA but inside band should maintain downtrend."""
        previous = RegimeState(
            trend=TrendState.DOWNTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 501 is only 0.2% above 500 (inside 2% band)
        state = detector.detect(
            spy_price=501, spy_200ma=500, vix=15, previous_state=previous, as_of=as_of
        )
        assert state.trend == TrendState.DOWNTREND

    def test_trend_crosses_hysteresis_to_downtrend(self, detector, as_of):
        """SPY dropping below lower band should trigger downtrend."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 489 is 2.2% below 500 (outside 2% band)
        state = detector.detect(
            spy_price=489, spy_200ma=500, vix=15, previous_state=previous, as_of=as_of
        )
        assert state.trend == TrendState.DOWNTREND

    # --- Volatility Detection Tests ---

    def test_clear_low_vol(self, detector, as_of):
        """VIX well below 20 should be low_vol."""
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=12, previous_state=None, as_of=as_of
        )
        assert state.vol == VolatilityState.LOW_VOL

    def test_clear_high_vol(self, detector, as_of):
        """VIX above 25 should be high_vol."""
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=30, previous_state=None, as_of=as_of
        )
        assert state.vol == VolatilityState.HIGH_VOL

    def test_vol_hysteresis_maintains_low_vol(self, detector, as_of):
        """VIX in 20-25 band should maintain low_vol."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=22, previous_state=previous, as_of=as_of
        )
        assert state.vol == VolatilityState.LOW_VOL

    def test_vol_hysteresis_maintains_high_vol(self, detector, as_of):
        """VIX in 20-25 band should maintain high_vol."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.HIGH_VOL, as_of=as_of
        )
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=22, previous_state=previous, as_of=as_of
        )
        assert state.vol == VolatilityState.HIGH_VOL

    # --- Combined Regime Tests ---

    def test_all_four_regimes(self, detector, as_of):
        """Verify all 4 regime combinations work."""
        # Uptrend + Low Vol
        state = detector.detect(600, 500, 15, None, as_of)
        assert state.name == "uptrend_low_vol"

        # Uptrend + High Vol
        state = detector.detect(600, 500, 30, None, as_of)
        assert state.name == "uptrend_high_vol"

        # Downtrend + Low Vol
        state = detector.detect(480, 500, 15, None, as_of)
        assert state.name == "downtrend_low_vol"

        # Downtrend + High Vol
        state = detector.detect(480, 500, 30, None, as_of)
        assert state.name == "downtrend_high_vol"

    # --- Serialization Tests ---

    def test_state_to_dict_roundtrip(self, as_of):
        """RegimeState should survive JSON serialization."""
        state = RegimeState(
            trend=TrendState.UPTREND,
            vol=VolatilityState.HIGH_VOL,
            as_of=as_of,
            spy_price=600.0,
            spy_200ma=550.0,
            vix=28.5,
        )
        data = state.to_dict()
        restored = RegimeState.from_dict(data)

        assert restored.name == state.name
        assert restored.spy_price == state.spy_price
        assert restored.vix == state.vix

    # --- Edge Cases ---

    def test_no_previous_state_defaults(self, detector, as_of):
        """Without previous state, should make reasonable defaults."""
        # Exactly at MA, low VIX -> uptrend_low_vol
        state = detector.detect(500, 500, 15, None, as_of)
        assert state.name == "uptrend_low_vol"

    def test_custom_config(self, as_of):
        """Custom thresholds should be respected."""
        config = RegimeConfig(
            trend_hysteresis_pct=0.05,  # 5% instead of 2%
            vix_high_threshold=30,      # 30 instead of 25
            vix_low_threshold=15,       # 15 instead of 20
        )
        detector = RegimeDetector(config)

        # VIX 25 should be low_vol with higher threshold
        state = detector.detect(500, 500, 25, None, as_of)
        assert state.vol == VolatilityState.LOW_VOL
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantetf/regime/__init__.py` | Package exports |
| `src/quantetf/regime/types.py` | RegimeConfig, RegimeState, enums |
| `src/quantetf/regime/detector.py` | RegimeDetector class |
| `tests/regime/__init__.py` | Test package |
| `tests/regime/test_detector.py` | Unit tests |

---

## Acceptance Criteria

- [ ] `RegimeDetector` class correctly classifies all 4 regimes
- [ ] Hysteresis prevents rapid switching at boundaries
- [ ] Works with no previous state (first detection)
- [ ] `RegimeState.to_dict()` / `from_dict()` roundtrip works
- [ ] Configuration is externalized via `RegimeConfig`
- [ ] Logging captures regime changes
- [ ] All tests pass (minimum 15 tests)
- [ ] No hardcoded thresholds in detector logic

---

## Notes for Implementer

1. **Keep it simple:** This is just the detection logic, not the indicator calculation
2. **Indicators come from caller:** The detector doesn't fetch SPY/VIX data itself
3. **Design for testability:** Pure functions where possible
4. **Follow existing patterns:** Look at `src/quantetf/alpha/` for similar class structures

---

## Definition of Done

1. All files created
2. All tests pass
3. `pytest tests/regime/test_detector.py` runs clean
4. No type errors (if running mypy)
5. Can instantiate detector and detect all 4 regimes

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Coding Agent
