"""Regime detector with hysteresis support.

This module implements the core regime detection logic as specified in ADR-001.
The detector classifies markets into 4 regimes based on trend (SPY vs 200MA)
and volatility (VIX level), with hysteresis to prevent whipsawing at boundaries.
"""

from typing import Optional
import logging

import pandas as pd

from .types import RegimeConfig, RegimeState, TrendState, VolatilityState

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regime based on trend (SPY vs 200MA) and volatility (VIX).

    Uses hysteresis to prevent rapid switching at boundaries:

    Trend (SPY vs 200MA):
    - Enter downtrend: SPY < 200MA * 0.98 (2% below)
    - Exit downtrend: SPY > 200MA * 1.02 (2% above)
    - Otherwise: maintain current trend state

    Volatility (VIX):
    - Enter high_vol: VIX > 25
    - Exit high_vol: VIX < 20
    - Otherwise: maintain current vol state
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """Initialize detector with threshold configuration.

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
        """Detect current regime with hysteresis.

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
