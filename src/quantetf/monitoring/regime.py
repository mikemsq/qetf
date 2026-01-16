"""Regime change monitoring.

This module provides regime monitoring functionality that detects market
regime changes and emits alerts when conditions shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from quantetf.monitoring.alerts import Alert, AlertManager

if TYPE_CHECKING:
    from quantetf.data.macro_loader import MacroDataLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeState:
    """Current market regime state.

    Attributes:
        regime: Current regime classification.
        vix: Current VIX level.
        yield_curve_spread: Current 10Y-2Y Treasury spread.
        timestamp: When this state was observed.
        previous_regime: Previous regime (for change detection).
    """

    regime: str
    vix: float | None
    yield_curve_spread: float | None
    timestamp: datetime
    previous_regime: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "regime": self.regime,
            "vix": self.vix,
            "yield_curve_spread": self.yield_curve_spread,
            "timestamp": self.timestamp.isoformat(),
            "previous_regime": self.previous_regime,
        }


@dataclass
class RegimeCheckResult:
    """Result of a regime check operation.

    Attributes:
        current_state: Current regime state.
        regime_changed: Whether regime changed from previous check.
        alerts_emitted: List of alerts that were emitted.
    """

    current_state: RegimeState
    regime_changed: bool
    alerts_emitted: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "current_state": self.current_state.to_dict(),
            "regime_changed": self.regime_changed,
            "alerts_emitted": [a.to_dict() for a in self.alerts_emitted],
        }


class RegimeMonitor:
    """Monitor market regime and emit alerts on changes.

    The RegimeMonitor uses the MacroDataLoader and RegimeDetector to
    classify the current market regime and emit alerts when the regime
    changes (e.g., from RISK_ON to HIGH_VOL).

    Regime Classifications:
        - RISK_ON: Normal market conditions
        - ELEVATED_VOL: VIX between 20-30
        - HIGH_VOL: VIX above 30
        - RECESSION_WARNING: Inverted yield curve

    Example:
        >>> from quantetf.data import MacroDataLoader
        >>> from quantetf.monitoring import AlertManager, RegimeMonitor
        >>> macro_loader = MacroDataLoader()
        >>> alert_manager = AlertManager()
        >>> monitor = RegimeMonitor(macro_loader, alert_manager)
        >>> result = monitor.check("2024-01-15")
    """

    # Regime severity for determining alert levels
    REGIME_SEVERITY = {
        "RISK_ON": 0,
        "ELEVATED_VOL": 1,
        "HIGH_VOL": 2,
        "RECESSION_WARNING": 3,
        "UNKNOWN": -1,
    }

    def __init__(
        self,
        macro_loader: "MacroDataLoader",
        alert_manager: AlertManager,
        vix_elevated_threshold: float = 20.0,
        vix_high_threshold: float = 30.0,
    ) -> None:
        """Initialize regime monitor.

        Args:
            macro_loader: MacroDataLoader for accessing VIX and yield curve data.
            alert_manager: AlertManager for emitting notifications.
            vix_elevated_threshold: VIX level for elevated volatility regime.
            vix_high_threshold: VIX level for high volatility regime.
        """
        self._macro_loader = macro_loader
        self._alert_manager = alert_manager
        self._vix_elevated_threshold = vix_elevated_threshold
        self._vix_high_threshold = vix_high_threshold
        self._last_regime: str | None = None
        self._regime_history: list[RegimeState] = []

    @property
    def last_regime(self) -> str | None:
        """Get the last detected regime."""
        return self._last_regime

    @property
    def regime_history(self) -> list[RegimeState]:
        """Get history of regime states."""
        return self._regime_history.copy()

    def _detect_regime(self, date: str) -> tuple[str, float | None, float | None]:
        """Detect current regime using macro data.

        Args:
            date: Date string (YYYY-MM-DD).

        Returns:
            Tuple of (regime, vix, yield_curve_spread).
        """
        vix: float | None = None
        spread: float | None = None

        try:
            vix = self._macro_loader.get_vix(date)
        except Exception as e:
            logger.warning(f"Could not load VIX data: {e}")

        try:
            spread = self._macro_loader.get_yield_curve_spread(date)
        except Exception as e:
            logger.warning(f"Could not load yield curve spread: {e}")

        # Determine regime
        if spread is not None and spread < 0:
            regime = "RECESSION_WARNING"
        elif vix is not None:
            if vix > self._vix_high_threshold:
                regime = "HIGH_VOL"
            elif vix > self._vix_elevated_threshold:
                regime = "ELEVATED_VOL"
            else:
                regime = "RISK_ON"
        else:
            regime = "UNKNOWN"

        return regime, vix, spread

    def _get_alert_level(self, old_regime: str | None, new_regime: str) -> str:
        """Determine alert level based on regime transition.

        Args:
            old_regime: Previous regime.
            new_regime: New regime.

        Returns:
            Alert level (INFO, WARNING, CRITICAL).
        """
        old_severity = self.REGIME_SEVERITY.get(old_regime or "RISK_ON", 0)
        new_severity = self.REGIME_SEVERITY.get(new_regime, 0)

        # Transitioning to more severe regime
        if new_severity > old_severity:
            if new_regime in ("HIGH_VOL", "RECESSION_WARNING"):
                return "CRITICAL"
            elif new_regime == "ELEVATED_VOL":
                return "WARNING"

        # Returning to normal
        if new_regime == "RISK_ON" and old_regime in ("HIGH_VOL", "RECESSION_WARNING"):
            return "INFO"

        return "INFO"

    def check(self, as_of: str) -> RegimeCheckResult:
        """Check current regime and emit alerts if changed.

        Args:
            as_of: Date to check (YYYY-MM-DD format).

        Returns:
            RegimeCheckResult with current state and any alerts.
        """
        regime, vix, spread = self._detect_regime(as_of)

        regime_changed = self._last_regime is not None and regime != self._last_regime

        state = RegimeState(
            regime=regime,
            vix=vix,
            yield_curve_spread=spread,
            timestamp=datetime.now(timezone.utc),
            previous_regime=self._last_regime,
        )

        alerts_emitted: list[Alert] = []

        if regime_changed:
            alert_level = self._get_alert_level(self._last_regime, regime)

            # Build descriptive message
            vix_str = f"VIX={vix:.1f}" if vix is not None else "VIX=N/A"
            spread_str = f"10Y-2Y={spread:.2f}%" if spread is not None else "10Y-2Y=N/A"

            alert = Alert(
                timestamp=datetime.now(timezone.utc),
                level=alert_level,
                category="REGIME",
                message=f"Market regime changed: {self._last_regime} -> {regime} "
                f"({vix_str}, {spread_str})",
                data={
                    "previous_regime": self._last_regime,
                    "new_regime": regime,
                    "vix": vix,
                    "yield_curve_spread": spread,
                    "as_of": as_of,
                },
            )
            self._alert_manager.emit(alert)
            alerts_emitted.append(alert)
            logger.info(f"Regime change detected: {self._last_regime} -> {regime}")

        # Update state
        self._last_regime = regime
        self._regime_history.append(state)

        # Keep history bounded
        if len(self._regime_history) > 1000:
            self._regime_history = self._regime_history[-1000:]

        return RegimeCheckResult(
            current_state=state,
            regime_changed=regime_changed,
            alerts_emitted=alerts_emitted,
        )

    def get_current_regime(self, as_of: str | None = None) -> dict[str, Any]:
        """Get current regime status without emitting alerts.

        Args:
            as_of: Optional date to check. Uses latest data if None.

        Returns:
            Dictionary with current regime status.
        """
        regime, vix, spread = self._detect_regime(as_of or "")

        return {
            "regime": regime,
            "vix": vix,
            "yield_curve_spread": spread,
            "vix_elevated_threshold": self._vix_elevated_threshold,
            "vix_high_threshold": self._vix_high_threshold,
            "description": self._get_regime_description(regime),
        }

    def _get_regime_description(self, regime: str) -> str:
        """Get human-readable description for a regime.

        Args:
            regime: Regime classification.

        Returns:
            Description string.
        """
        descriptions = {
            "RISK_ON": "Normal market conditions, favorable for risk assets",
            "ELEVATED_VOL": "Elevated volatility, consider reducing exposure",
            "HIGH_VOL": "High volatility regime, defensive positioning recommended",
            "RECESSION_WARNING": "Inverted yield curve signals recession risk",
            "UNKNOWN": "Unable to determine regime (data unavailable)",
        }
        return descriptions.get(regime, "Unknown regime")

    def get_regime_stats(self, lookback_days: int = 252) -> dict[str, Any]:
        """Get regime statistics over a lookback period.

        Args:
            lookback_days: Number of states to consider.

        Returns:
            Dictionary with regime statistics.
        """
        recent = self._regime_history[-lookback_days:] if self._regime_history else []

        if not recent:
            return {
                "total_observations": 0,
                "regime_counts": {},
                "regime_percentages": {},
                "current_regime": self._last_regime,
            }

        regime_counts: dict[str, int] = {}
        for state in recent:
            regime_counts[state.regime] = regime_counts.get(state.regime, 0) + 1

        total = len(recent)
        regime_percentages = {k: v / total for k, v in regime_counts.items()}

        return {
            "total_observations": total,
            "regime_counts": regime_counts,
            "regime_percentages": regime_percentages,
            "current_regime": self._last_regime,
            "lookback_days": lookback_days,
        }

    def reset(self) -> None:
        """Reset monitor state.

        Clears last regime and history, useful for testing.
        """
        self._last_regime = None
        self._regime_history.clear()
        logger.info("Regime monitor reset")
