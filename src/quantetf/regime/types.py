"""Regime detection types and data structures.

This module defines the core types for the regime-based strategy selection system
as specified in ADR-001.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class TrendState(Enum):
    """Trend state based on SPY vs 200-day moving average."""

    UPTREND = "uptrend"
    DOWNTREND = "downtrend"


class VolatilityState(Enum):
    """Volatility state based on VIX level."""

    LOW_VOL = "low_vol"
    HIGH_VOL = "high_vol"


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for regime detection thresholds.

    Attributes:
        trend_hysteresis_pct: Percentage band around 200MA for trend hysteresis (default 2%)
        vix_high_threshold: VIX level to enter high_vol state (default 25)
        vix_low_threshold: VIX level to exit high_vol state (default 20)
    """

    trend_hysteresis_pct: float = 0.02  # +/- 2% around 200MA
    vix_high_threshold: float = 25.0  # Enter high_vol when VIX > 25
    vix_low_threshold: float = 20.0  # Exit high_vol when VIX < 20


@dataclass
class RegimeState:
    """Current regime state with hysteresis memory.

    Attributes:
        trend: Current trend state (uptrend/downtrend)
        vol: Current volatility state (low_vol/high_vol)
        as_of: Timestamp when this state was determined
        spy_price: SPY price used for detection (optional, for debugging)
        spy_200ma: SPY 200-day MA used for detection (optional, for debugging)
        vix: VIX level used for detection (optional, for debugging)
    """

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
            },
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
