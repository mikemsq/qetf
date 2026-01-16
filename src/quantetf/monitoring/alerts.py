"""Alert system for monitoring and notifications.

This module provides a pluggable alert system for emitting notifications about
portfolio events such as drawdown breaches, regime changes, and data quality issues.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    """Immutable alert record.

    Attributes:
        timestamp: When the alert was generated.
        level: Severity level (INFO, WARNING, CRITICAL).
        category: Alert category (DRAWDOWN, REGIME, DATA_QUALITY, REBALANCE).
        message: Human-readable alert message.
        data: Additional structured data for the alert.
    """

    timestamp: datetime
    level: str
    category: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alert fields."""
        valid_levels = {"INFO", "WARNING", "CRITICAL"}
        if self.level not in valid_levels:
            raise ValueError(f"Invalid alert level: {self.level}. Must be one of {valid_levels}")

        valid_categories = {"DRAWDOWN", "REGIME", "DATA_QUALITY", "REBALANCE", "SYSTEM"}
        if self.category not in valid_categories:
            raise ValueError(
                f"Invalid alert category: {self.category}. Must be one of {valid_categories}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization.

        Returns:
            Dictionary representation of the alert.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Alert:
        """Create Alert from dictionary.

        Args:
            data: Dictionary with alert data.

        Returns:
            Alert instance.
        """
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=data["level"],
            category=data["category"],
            message=data["message"],
            data=data.get("data", {}),
        )

    def format(self) -> str:
        """Format alert as a human-readable string.

        Returns:
            Formatted alert string.
        """
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts}] [{self.level}] [{self.category}] {self.message}"


class AlertHandler(ABC):
    """Abstract base class for alert handlers.

    Handlers receive alerts and process them (log, write to file, send email, etc.).
    """

    @abstractmethod
    def send(self, alert: Alert) -> None:
        """Send an alert through this handler.

        Args:
            alert: Alert to send.
        """


class ConsoleAlertHandler(AlertHandler):
    """Handler that prints alerts to console/stdout.

    Uses Python logging to output alerts with appropriate log levels.
    """

    def __init__(self, logger_name: str = "quantetf.monitoring.alerts") -> None:
        """Initialize console handler.

        Args:
            logger_name: Name of the logger to use.
        """
        self._logger = logging.getLogger(logger_name)

    def send(self, alert: Alert) -> None:
        """Print alert to console via logging.

        Args:
            alert: Alert to send.
        """
        log_level = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "CRITICAL": logging.CRITICAL,
        }.get(alert.level, logging.INFO)

        self._logger.log(log_level, alert.format())


class FileAlertHandler(AlertHandler):
    """Handler that appends alerts to a log file.

    Each alert is written as a JSON line for easy parsing.
    """

    def __init__(self, log_path: Union[str, Path]) -> None:
        """Initialize file handler.

        Args:
            log_path: Path to the alert log file.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        """Append alert to log file as JSON line.

        Args:
            alert: Alert to send.
        """
        with open(self.log_path, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")

    def read_alerts(self, limit: int = 100) -> list[Alert]:
        """Read recent alerts from log file.

        Args:
            limit: Maximum number of recent alerts to return.

        Returns:
            List of Alert objects, most recent first.
        """
        if not self.log_path.exists():
            return []

        alerts: list[Alert] = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        alerts.append(Alert.from_dict(data))
                    except (json.JSONDecodeError, KeyError):
                        continue

        return alerts[-limit:][::-1]  # Return most recent first


class InMemoryAlertHandler(AlertHandler):
    """Handler that stores alerts in memory.

    Useful for testing and for applications that need to query recent alerts.
    """

    def __init__(self, max_alerts: int = 1000) -> None:
        """Initialize in-memory handler.

        Args:
            max_alerts: Maximum number of alerts to retain.
        """
        self.max_alerts = max_alerts
        self._alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        """Store alert in memory.

        Args:
            alert: Alert to store.
        """
        self._alerts.append(alert)
        if len(self._alerts) > self.max_alerts:
            self._alerts = self._alerts[-self.max_alerts :]

    def get_alerts(
        self,
        level: str | None = None,
        category: str | None = None,
        since: datetime | None = None,
    ) -> list[Alert]:
        """Get stored alerts with optional filters.

        Args:
            level: Filter by alert level.
            category: Filter by category.
            since: Filter alerts after this timestamp.

        Returns:
            List of matching alerts.
        """
        result = self._alerts.copy()

        if level:
            result = [a for a in result if a.level == level]
        if category:
            result = [a for a in result if a.category == category]
        if since:
            result = [a for a in result if a.timestamp >= since]

        return result

    def clear(self) -> None:
        """Clear all stored alerts."""
        self._alerts.clear()


class AlertManager:
    """Central alert manager that routes alerts to registered handlers.

    The AlertManager acts as a facade for the alert system, allowing
    components to emit alerts without knowing the specific handlers.
    """

    def __init__(self, handlers: list[AlertHandler] | None = None) -> None:
        """Initialize alert manager.

        Args:
            handlers: List of handlers to register. If None, uses ConsoleAlertHandler.
        """
        self._handlers: list[AlertHandler] = handlers if handlers is not None else []

    def add_handler(self, handler: AlertHandler) -> None:
        """Register an alert handler.

        Args:
            handler: Handler to register.
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: AlertHandler) -> None:
        """Remove a registered handler.

        Args:
            handler: Handler to remove.
        """
        if handler in self._handlers:
            self._handlers.remove(handler)

    def emit(self, alert: Alert) -> None:
        """Emit an alert to all registered handlers.

        Args:
            alert: Alert to emit.
        """
        for handler in self._handlers:
            try:
                handler.send(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {type(handler).__name__}: {e}")

    def info(
        self,
        category: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> Alert:
        """Emit an INFO level alert.

        Args:
            category: Alert category.
            message: Alert message.
            data: Optional additional data.

        Returns:
            The emitted Alert.
        """
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            level="INFO",
            category=category,
            message=message,
            data=data or {},
        )
        self.emit(alert)
        return alert

    def warning(
        self,
        category: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> Alert:
        """Emit a WARNING level alert.

        Args:
            category: Alert category.
            message: Alert message.
            data: Optional additional data.

        Returns:
            The emitted Alert.
        """
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            level="WARNING",
            category=category,
            message=message,
            data=data or {},
        )
        self.emit(alert)
        return alert

    def critical(
        self,
        category: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> Alert:
        """Emit a CRITICAL level alert.

        Args:
            category: Alert category.
            message: Alert message.
            data: Optional additional data.

        Returns:
            The emitted Alert.
        """
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            level="CRITICAL",
            category=category,
            message=message,
            data=data or {},
        )
        self.emit(alert)
        return alert


def create_default_alert_manager(
    log_path: Union[str, Path] | None = None,
    console: bool = True,
) -> AlertManager:
    """Create an AlertManager with default handlers.

    Args:
        log_path: Optional path for file logging.
        console: Whether to include console handler.

    Returns:
        Configured AlertManager.
    """
    handlers: list[AlertHandler] = []

    if console:
        handlers.append(ConsoleAlertHandler())

    if log_path:
        handlers.append(FileAlertHandler(log_path))

    return AlertManager(handlers=handlers)
