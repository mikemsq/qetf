"""Tests for the alerts module."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quantetf.monitoring.alerts import (
    Alert,
    AlertHandler,
    AlertManager,
    ConsoleAlertHandler,
    FileAlertHandler,
    InMemoryAlertHandler,
    create_default_alert_manager,
)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_create_alert(self):
        """Test creating a basic alert."""
        alert = Alert(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            level="WARNING",
            category="DRAWDOWN",
            message="Test alert message",
            data={"key": "value"},
        )

        assert alert.level == "WARNING"
        assert alert.category == "DRAWDOWN"
        assert alert.message == "Test alert message"
        assert alert.data == {"key": "value"}

    def test_alert_validates_level(self):
        """Test that invalid levels raise ValueError."""
        with pytest.raises(ValueError, match="Invalid alert level"):
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="INVALID",
                category="DRAWDOWN",
                message="Test",
            )

    def test_alert_validates_category(self):
        """Test that invalid categories raise ValueError."""
        with pytest.raises(ValueError, match="Invalid alert category"):
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="WARNING",
                category="INVALID",
                message="Test",
            )

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        alert = Alert(
            timestamp=ts,
            level="INFO",
            category="REGIME",
            message="Test message",
            data={"foo": "bar"},
        )

        result = alert.to_dict()

        assert result["timestamp"] == ts.isoformat()
        assert result["level"] == "INFO"
        assert result["category"] == "REGIME"
        assert result["message"] == "Test message"
        assert result["data"] == {"foo": "bar"}

    def test_alert_from_dict(self):
        """Test creating alert from dictionary."""
        data = {
            "timestamp": "2024-01-15T12:00:00+00:00",
            "level": "CRITICAL",
            "category": "DATA_QUALITY",
            "message": "Data issue",
            "data": {"ticker": "SPY"},
        }

        alert = Alert.from_dict(data)

        assert alert.level == "CRITICAL"
        assert alert.category == "DATA_QUALITY"
        assert alert.message == "Data issue"
        assert alert.data == {"ticker": "SPY"}

    def test_alert_format(self):
        """Test alert formatting."""
        ts = datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)
        alert = Alert(
            timestamp=ts,
            level="WARNING",
            category="DRAWDOWN",
            message="10% drawdown reached",
        )

        formatted = alert.format()

        assert "[2024-01-15 12:30:45]" in formatted
        assert "[WARNING]" in formatted
        assert "[DRAWDOWN]" in formatted
        assert "10% drawdown reached" in formatted

    def test_alert_default_data(self):
        """Test that data defaults to empty dict."""
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            level="INFO",
            category="SYSTEM",
            message="Test",
        )

        assert alert.data == {}


class TestConsoleAlertHandler:
    """Tests for ConsoleAlertHandler."""

    def test_send_alert(self, caplog):
        """Test sending alert to console."""
        handler = ConsoleAlertHandler()
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            level="WARNING",
            category="DRAWDOWN",
            message="Test drawdown alert",
        )

        with caplog.at_level("WARNING"):
            handler.send(alert)

        assert "Test drawdown alert" in caplog.text


class TestFileAlertHandler:
    """Tests for FileAlertHandler."""

    def test_send_alert_creates_file(self):
        """Test that sending an alert creates the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.jsonl"
            handler = FileAlertHandler(log_path)

            alert = Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="REBALANCE",
                message="Rebalance complete",
            )

            handler.send(alert)

            assert log_path.exists()

    def test_send_alert_appends_json(self):
        """Test that alerts are appended as JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.jsonl"
            handler = FileAlertHandler(log_path)

            # Send two alerts
            for i in range(2):
                alert = Alert(
                    timestamp=datetime.now(timezone.utc),
                    level="INFO",
                    category="SYSTEM",
                    message=f"Alert {i}",
                )
                handler.send(alert)

            # Read and verify
            with open(log_path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            assert json.loads(lines[0])["message"] == "Alert 0"
            assert json.loads(lines[1])["message"] == "Alert 1"

    def test_read_alerts(self):
        """Test reading alerts from log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.jsonl"
            handler = FileAlertHandler(log_path)

            # Send some alerts
            for i in range(5):
                alert = Alert(
                    timestamp=datetime.now(timezone.utc),
                    level="INFO",
                    category="SYSTEM",
                    message=f"Alert {i}",
                )
                handler.send(alert)

            # Read alerts
            alerts = handler.read_alerts(limit=3)

            assert len(alerts) == 3
            # Most recent first
            assert alerts[0].message == "Alert 4"
            assert alerts[1].message == "Alert 3"
            assert alerts[2].message == "Alert 2"

    def test_read_alerts_empty_file(self):
        """Test reading from non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nonexistent.jsonl"
            handler = FileAlertHandler(log_path)

            alerts = handler.read_alerts()

            assert alerts == []


class TestInMemoryAlertHandler:
    """Tests for InMemoryAlertHandler."""

    def test_send_and_retrieve_alerts(self):
        """Test sending and retrieving alerts."""
        handler = InMemoryAlertHandler()

        for i in range(3):
            alert = Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="SYSTEM",
                message=f"Alert {i}",
            )
            handler.send(alert)

        alerts = handler.get_alerts()
        assert len(alerts) == 3

    def test_max_alerts_limit(self):
        """Test that handler respects max_alerts limit."""
        handler = InMemoryAlertHandler(max_alerts=5)

        for i in range(10):
            alert = Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="SYSTEM",
                message=f"Alert {i}",
            )
            handler.send(alert)

        alerts = handler.get_alerts()
        assert len(alerts) == 5
        # Should have most recent alerts
        assert alerts[0].message == "Alert 5"

    def test_filter_by_level(self):
        """Test filtering alerts by level."""
        handler = InMemoryAlertHandler()

        handler.send(
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="SYSTEM",
                message="Info",
            )
        )
        handler.send(
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="WARNING",
                category="SYSTEM",
                message="Warning",
            )
        )

        warnings = handler.get_alerts(level="WARNING")
        assert len(warnings) == 1
        assert warnings[0].message == "Warning"

    def test_filter_by_category(self):
        """Test filtering alerts by category."""
        handler = InMemoryAlertHandler()

        handler.send(
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="DRAWDOWN",
                message="Drawdown",
            )
        )
        handler.send(
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="REGIME",
                message="Regime",
            )
        )

        drawdown_alerts = handler.get_alerts(category="DRAWDOWN")
        assert len(drawdown_alerts) == 1
        assert drawdown_alerts[0].message == "Drawdown"

    def test_filter_by_timestamp(self):
        """Test filtering alerts by timestamp."""
        handler = InMemoryAlertHandler()

        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2024, 1, 15, tzinfo=timezone.utc)

        handler.send(
            Alert(
                timestamp=old_time,
                level="INFO",
                category="SYSTEM",
                message="Old",
            )
        )
        handler.send(
            Alert(
                timestamp=new_time,
                level="INFO",
                category="SYSTEM",
                message="New",
            )
        )

        recent = handler.get_alerts(since=datetime(2024, 1, 10, tzinfo=timezone.utc))
        assert len(recent) == 1
        assert recent[0].message == "New"

    def test_clear_alerts(self):
        """Test clearing all alerts."""
        handler = InMemoryAlertHandler()

        handler.send(
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="SYSTEM",
                message="Test",
            )
        )

        assert len(handler.get_alerts()) == 1

        handler.clear()

        assert len(handler.get_alerts()) == 0


class TestAlertManager:
    """Tests for AlertManager."""

    def test_emit_to_multiple_handlers(self):
        """Test emitting to multiple handlers."""
        handler1 = InMemoryAlertHandler()
        handler2 = InMemoryAlertHandler()

        manager = AlertManager(handlers=[handler1, handler2])

        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            level="INFO",
            category="SYSTEM",
            message="Test",
        )

        manager.emit(alert)

        assert len(handler1.get_alerts()) == 1
        assert len(handler2.get_alerts()) == 1

    def test_add_handler(self):
        """Test adding a handler."""
        manager = AlertManager()
        handler = InMemoryAlertHandler()

        manager.add_handler(handler)

        manager.emit(
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="SYSTEM",
                message="Test",
            )
        )

        assert len(handler.get_alerts()) == 1

    def test_remove_handler(self):
        """Test removing a handler."""
        handler = InMemoryAlertHandler()
        manager = AlertManager(handlers=[handler])

        manager.remove_handler(handler)

        manager.emit(
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="SYSTEM",
                message="Test",
            )
        )

        assert len(handler.get_alerts()) == 0

    def test_info_convenience_method(self):
        """Test info convenience method."""
        handler = InMemoryAlertHandler()
        manager = AlertManager(handlers=[handler])

        alert = manager.info("SYSTEM", "Info message", {"key": "value"})

        assert alert.level == "INFO"
        assert alert.category == "SYSTEM"
        assert alert.message == "Info message"
        assert len(handler.get_alerts()) == 1

    def test_warning_convenience_method(self):
        """Test warning convenience method."""
        handler = InMemoryAlertHandler()
        manager = AlertManager(handlers=[handler])

        alert = manager.warning("DRAWDOWN", "Warning message")

        assert alert.level == "WARNING"
        assert alert.category == "DRAWDOWN"

    def test_critical_convenience_method(self):
        """Test critical convenience method."""
        handler = InMemoryAlertHandler()
        manager = AlertManager(handlers=[handler])

        alert = manager.critical("DATA_QUALITY", "Critical message")

        assert alert.level == "CRITICAL"
        assert alert.category == "DATA_QUALITY"

    def test_handler_error_doesnt_break_others(self):
        """Test that one handler error doesn't break other handlers."""

        class FailingHandler(AlertHandler):
            def send(self, alert):
                raise RuntimeError("Handler failed")

        handler1 = FailingHandler()
        handler2 = InMemoryAlertHandler()

        manager = AlertManager(handlers=[handler1, handler2])

        # Should not raise, and handler2 should still receive the alert
        manager.emit(
            Alert(
                timestamp=datetime.now(timezone.utc),
                level="INFO",
                category="SYSTEM",
                message="Test",
            )
        )

        assert len(handler2.get_alerts()) == 1


class TestCreateDefaultAlertManager:
    """Tests for create_default_alert_manager."""

    def test_creates_console_handler(self):
        """Test creating manager with console handler."""
        manager = create_default_alert_manager(console=True)

        # Should have one handler (console)
        assert len(manager._handlers) == 1
        assert isinstance(manager._handlers[0], ConsoleAlertHandler)

    def test_creates_file_handler(self):
        """Test creating manager with file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.jsonl"
            manager = create_default_alert_manager(log_path=log_path, console=False)

            assert len(manager._handlers) == 1
            assert isinstance(manager._handlers[0], FileAlertHandler)

    def test_creates_both_handlers(self):
        """Test creating manager with both handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.jsonl"
            manager = create_default_alert_manager(log_path=log_path, console=True)

            assert len(manager._handlers) == 2
