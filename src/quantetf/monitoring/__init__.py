"""Monitoring and alerts module for production portfolio management.

This module provides monitoring capabilities including:
- Alert system with pluggable handlers
- NAV tracking and drawdown monitoring
- Market regime change detection
- Data quality checks

Example:
    >>> from quantetf.monitoring import AlertManager, NAVTracker, RegimeMonitor
    >>> from quantetf.production import InMemoryStateManager
    >>> from quantetf.data import MacroDataLoader
    >>>
    >>> # Set up alert system
    >>> alert_manager = AlertManager()
    >>>
    >>> # Set up NAV tracking
    >>> state_manager = InMemoryStateManager()
    >>> nav_tracker = NAVTracker(state_manager, alert_manager)
    >>>
    >>> # Set up regime monitoring
    >>> macro_loader = MacroDataLoader()
    >>> regime_monitor = RegimeMonitor(macro_loader, alert_manager)
"""

from quantetf.monitoring.alerts import (
    Alert,
    AlertHandler,
    AlertManager,
    ConsoleAlertHandler,
    FileAlertHandler,
    InMemoryAlertHandler,
    create_default_alert_manager,
)
from quantetf.monitoring.nav_tracker import (
    DrawdownThreshold,
    NAVTracker,
    NAVUpdateResult,
)
from quantetf.monitoring.quality import (
    AnomalyResult,
    DataQualityChecker,
    GapResult,
    QualityCheckResult,
    StalenessResult,
)
from quantetf.monitoring.regime import (
    RegimeCheckResult,
    RegimeMonitor,
    RegimeState,
)

__all__ = [
    # Alerts
    "Alert",
    "AlertHandler",
    "AlertManager",
    "ConsoleAlertHandler",
    "FileAlertHandler",
    "InMemoryAlertHandler",
    "create_default_alert_manager",
    # NAV Tracking
    "NAVTracker",
    "NAVUpdateResult",
    "DrawdownThreshold",
    # Regime Monitoring
    "RegimeMonitor",
    "RegimeState",
    "RegimeCheckResult",
    # Data Quality
    "DataQualityChecker",
    "QualityCheckResult",
    "StalenessResult",
    "GapResult",
    "AnomalyResult",
]
