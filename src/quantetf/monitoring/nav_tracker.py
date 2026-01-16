"""NAV tracking and drawdown monitoring.

This module provides NAV tracking functionality that monitors portfolio value
and emits alerts when drawdown thresholds are breached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from quantetf.monitoring.alerts import Alert, AlertManager

if TYPE_CHECKING:
    from quantetf.production.state import PortfolioStateManager

logger = logging.getLogger(__name__)


@dataclass
class DrawdownThreshold:
    """Configuration for a drawdown alert threshold.

    Attributes:
        level: Drawdown level as a fraction (e.g., 0.10 = 10%).
        alert_level: Alert severity when this threshold is breached.
        name: Human-readable name for this threshold.
    """

    level: float
    alert_level: str
    name: str

    def __post_init__(self) -> None:
        """Validate threshold configuration."""
        if not 0 < self.level < 1:
            raise ValueError(f"Drawdown level must be between 0 and 1: {self.level}")
        valid_levels = {"INFO", "WARNING", "CRITICAL"}
        if self.alert_level not in valid_levels:
            raise ValueError(f"Invalid alert level: {self.alert_level}")


@dataclass
class NAVUpdateResult:
    """Result of a NAV update operation.

    Attributes:
        as_of: Date of the NAV update.
        nav: Current NAV value.
        peak_nav: Historical peak NAV.
        drawdown: Current drawdown as a fraction.
        alerts_emitted: List of alerts that were emitted.
        thresholds_breached: List of threshold names that were breached.
    """

    as_of: pd.Timestamp
    nav: float
    peak_nav: float
    drawdown: float
    alerts_emitted: list[Alert] = field(default_factory=list)
    thresholds_breached: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "as_of": self.as_of.isoformat(),
            "nav": self.nav,
            "peak_nav": self.peak_nav,
            "drawdown": self.drawdown,
            "alerts_emitted": [a.to_dict() for a in self.alerts_emitted],
            "thresholds_breached": self.thresholds_breached,
        }


class NAVTracker:
    """Track portfolio NAV and monitor drawdown thresholds.

    The NAVTracker maintains portfolio value history and emits alerts when
    drawdown thresholds are breached. It integrates with the PortfolioStateManager
    for state persistence and the AlertManager for notifications.

    Example:
        >>> from quantetf.production import InMemoryStateManager
        >>> from quantetf.monitoring import AlertManager, NAVTracker
        >>> state_manager = InMemoryStateManager()
        >>> alert_manager = AlertManager()
        >>> tracker = NAVTracker(state_manager, alert_manager)
        >>> result = tracker.update(pd.Timestamp("2024-01-15"), nav=100000.0)
    """

    DEFAULT_THRESHOLDS = (
        DrawdownThreshold(level=0.05, alert_level="INFO", name="5% Drawdown"),
        DrawdownThreshold(level=0.10, alert_level="WARNING", name="10% Drawdown"),
        DrawdownThreshold(level=0.20, alert_level="WARNING", name="20% Drawdown"),
        DrawdownThreshold(level=0.30, alert_level="CRITICAL", name="30% Drawdown"),
    )

    def __init__(
        self,
        state_manager: "PortfolioStateManager",
        alert_manager: AlertManager,
        drawdown_thresholds: tuple[DrawdownThreshold, ...] | None = None,
    ) -> None:
        """Initialize NAV tracker.

        Args:
            state_manager: Portfolio state manager for history access.
            alert_manager: Alert manager for emitting notifications.
            drawdown_thresholds: Custom drawdown thresholds. Uses defaults if None.
        """
        self._state_manager = state_manager
        self._alert_manager = alert_manager
        self._thresholds = (
            drawdown_thresholds if drawdown_thresholds is not None else self.DEFAULT_THRESHOLDS
        )
        # Track which thresholds have been triggered to avoid duplicate alerts
        self._triggered_thresholds: set[str] = set()
        # Track last known drawdown for detecting threshold crossings
        self._last_drawdown: float = 0.0

    @property
    def thresholds(self) -> tuple[DrawdownThreshold, ...]:
        """Get configured drawdown thresholds."""
        return self._thresholds

    def get_current_drawdown(self) -> float:
        """Get current portfolio drawdown.

        Returns:
            Current drawdown as a positive fraction (0.10 = 10%).
        """
        latest = self._state_manager.get_latest_state()
        if latest is None:
            return 0.0
        return latest.get_current_drawdown()

    def get_peak_nav(self) -> float:
        """Get historical peak NAV.

        Returns:
            Peak NAV value, or 0.0 if no history.
        """
        history = self._state_manager.get_history(lookback_days=365 * 10)
        return history.get_peak_nav()

    def update(
        self,
        as_of: pd.Timestamp,
        nav: float,
        holdings: pd.Series | None = None,
        weights: pd.Series | None = None,
        cost_basis: pd.Series | None = None,
    ) -> NAVUpdateResult:
        """Update NAV and check drawdown thresholds.

        This method updates the portfolio NAV, calculates the current drawdown,
        and emits alerts for any thresholds that are newly breached.

        Args:
            as_of: Date of the NAV update.
            nav: Current NAV value.
            holdings: Optional current holdings (ticker -> shares).
            weights: Optional current weights (ticker -> weight).
            cost_basis: Optional cost basis (ticker -> avg cost).

        Returns:
            NAVUpdateResult with details of the update and any alerts emitted.
        """
        from quantetf.production.state import PortfolioState

        # Get peak NAV from history
        history = self._state_manager.get_history(lookback_days=365 * 10)
        peak_nav = max(history.get_peak_nav(), nav)  # Update peak if current is higher

        # Calculate drawdown
        drawdown = 0.0 if peak_nav <= 0 else max(0.0, (peak_nav - nav) / peak_nav)

        # Create and save state
        state = PortfolioState(
            as_of=as_of,
            holdings=holdings if holdings is not None else pd.Series(dtype=float),
            weights=weights if weights is not None else pd.Series(dtype=float),
            nav=nav,
            cost_basis=cost_basis if cost_basis is not None else pd.Series(dtype=float),
            peak_nav=peak_nav,
            created_at=datetime.now(timezone.utc),
        )
        self._state_manager.save_state(state)

        # Check thresholds and emit alerts
        alerts_emitted: list[Alert] = []
        thresholds_breached: list[str] = []

        for threshold in self._thresholds:
            # Check if we've crossed this threshold
            if drawdown >= threshold.level:
                thresholds_breached.append(threshold.name)

                # Only alert if this is a new breach (crossing from below)
                if threshold.name not in self._triggered_thresholds:
                    self._triggered_thresholds.add(threshold.name)

                    alert = Alert(
                        timestamp=datetime.now(timezone.utc),
                        level=threshold.alert_level,
                        category="DRAWDOWN",
                        message=f"Drawdown threshold breached: {threshold.name} "
                        f"({drawdown:.1%} drawdown from peak ${peak_nav:,.2f})",
                        data={
                            "threshold_name": threshold.name,
                            "threshold_level": threshold.level,
                            "current_drawdown": drawdown,
                            "nav": nav,
                            "peak_nav": peak_nav,
                            "as_of": as_of.isoformat(),
                        },
                    )
                    self._alert_manager.emit(alert)
                    alerts_emitted.append(alert)
                    logger.warning(f"Drawdown alert: {threshold.name} breached at {drawdown:.1%}")
            else:
                # Clear trigger if we've recovered above this threshold
                if threshold.name in self._triggered_thresholds:
                    self._triggered_thresholds.discard(threshold.name)
                    # Emit recovery alert
                    alert = Alert(
                        timestamp=datetime.now(timezone.utc),
                        level="INFO",
                        category="DRAWDOWN",
                        message=f"Recovered above {threshold.name} threshold "
                        f"(now {drawdown:.1%} drawdown)",
                        data={
                            "threshold_name": threshold.name,
                            "threshold_level": threshold.level,
                            "current_drawdown": drawdown,
                            "nav": nav,
                            "peak_nav": peak_nav,
                            "as_of": as_of.isoformat(),
                        },
                    )
                    self._alert_manager.emit(alert)
                    alerts_emitted.append(alert)
                    logger.info(f"Recovery alert: Recovered above {threshold.name}")

        self._last_drawdown = drawdown

        return NAVUpdateResult(
            as_of=as_of,
            nav=nav,
            peak_nav=peak_nav,
            drawdown=drawdown,
            alerts_emitted=alerts_emitted,
            thresholds_breached=thresholds_breached,
        )

    def check_drawdown(self) -> dict[str, Any]:
        """Check current drawdown status without updating NAV.

        Returns:
            Dictionary with current drawdown status including:
            - current_nav: Current NAV
            - peak_nav: Peak NAV
            - drawdown: Current drawdown fraction
            - drawdown_pct: Drawdown as percentage string
            - thresholds_breached: List of breached threshold names
            - status: Overall status (OK, WARNING, CRITICAL)
        """
        latest = self._state_manager.get_latest_state()
        if latest is None:
            return {
                "current_nav": 0.0,
                "peak_nav": 0.0,
                "drawdown": 0.0,
                "drawdown_pct": "0.0%",
                "thresholds_breached": [],
                "status": "NO_DATA",
            }

        drawdown = latest.get_current_drawdown()
        peak_nav = latest.peak_nav

        thresholds_breached = [t.name for t in self._thresholds if drawdown >= t.level]

        # Determine overall status
        status = "OK"
        for threshold in sorted(self._thresholds, key=lambda t: t.level, reverse=True):
            if drawdown >= threshold.level:
                status = threshold.alert_level
                break

        return {
            "current_nav": latest.nav,
            "peak_nav": peak_nav,
            "drawdown": drawdown,
            "drawdown_pct": f"{drawdown:.1%}",
            "thresholds_breached": thresholds_breached,
            "status": status,
            "as_of": latest.as_of.isoformat(),
        }

    def get_nav_history(self, lookback_days: int = 365) -> pd.Series:
        """Get NAV time series.

        Args:
            lookback_days: Number of days to look back.

        Returns:
            Series with datetime index and NAV values.
        """
        history = self._state_manager.get_history(lookback_days=lookback_days)
        return history.get_nav_series()

    def reset_triggered_thresholds(self) -> None:
        """Reset all triggered threshold flags.

        Call this to re-enable alerts for thresholds that have already been triggered.
        """
        self._triggered_thresholds.clear()
        logger.info("Reset triggered drawdown thresholds")
