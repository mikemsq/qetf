"""Tests for the nav_tracker module."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from quantetf.monitoring.alerts import AlertManager, InMemoryAlertHandler
from quantetf.monitoring.nav_tracker import (
    DrawdownThreshold,
    NAVTracker,
    NAVUpdateResult,
)
from quantetf.production.state import InMemoryStateManager


class TestDrawdownThreshold:
    """Tests for DrawdownThreshold dataclass."""

    def test_create_valid_threshold(self):
        """Test creating a valid threshold."""
        threshold = DrawdownThreshold(
            level=0.10, alert_level="WARNING", name="10% Drawdown"
        )

        assert threshold.level == 0.10
        assert threshold.alert_level == "WARNING"
        assert threshold.name == "10% Drawdown"

    def test_validates_level_range(self):
        """Test that level must be between 0 and 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            DrawdownThreshold(level=1.5, alert_level="WARNING", name="Invalid")

        with pytest.raises(ValueError, match="between 0 and 1"):
            DrawdownThreshold(level=0, alert_level="WARNING", name="Invalid")

        with pytest.raises(ValueError, match="between 0 and 1"):
            DrawdownThreshold(level=-0.1, alert_level="WARNING", name="Invalid")

    def test_validates_alert_level(self):
        """Test that alert_level must be valid."""
        with pytest.raises(ValueError, match="Invalid alert level"):
            DrawdownThreshold(level=0.10, alert_level="INVALID", name="Invalid")


class TestNAVUpdateResult:
    """Tests for NAVUpdateResult dataclass."""

    def test_create_result(self):
        """Test creating a NAV update result."""
        result = NAVUpdateResult(
            as_of=pd.Timestamp("2024-01-15"),
            nav=95000.0,
            peak_nav=100000.0,
            drawdown=0.05,
            thresholds_breached=["5% Drawdown"],
        )

        assert result.nav == 95000.0
        assert result.drawdown == 0.05
        assert "5% Drawdown" in result.thresholds_breached

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = NAVUpdateResult(
            as_of=pd.Timestamp("2024-01-15"),
            nav=95000.0,
            peak_nav=100000.0,
            drawdown=0.05,
        )

        data = result.to_dict()

        assert data["nav"] == 95000.0
        assert data["peak_nav"] == 100000.0
        assert data["drawdown"] == 0.05
        assert "as_of" in data


class TestNAVTracker:
    """Tests for NAVTracker class."""

    @pytest.fixture
    def state_manager(self):
        """Create in-memory state manager."""
        return InMemoryStateManager()

    @pytest.fixture
    def alert_handler(self):
        """Create in-memory alert handler."""
        return InMemoryAlertHandler()

    @pytest.fixture
    def alert_manager(self, alert_handler):
        """Create alert manager with in-memory handler."""
        return AlertManager(handlers=[alert_handler])

    @pytest.fixture
    def tracker(self, state_manager, alert_manager):
        """Create NAV tracker with default thresholds."""
        return NAVTracker(state_manager, alert_manager)

    def test_update_saves_state(self, tracker, state_manager):
        """Test that update saves portfolio state."""
        result = tracker.update(
            as_of=pd.Timestamp("2024-01-15"),
            nav=100000.0,
            holdings=pd.Series({"SPY": 100}),
            weights=pd.Series({"SPY": 1.0}),
        )

        assert result.nav == 100000.0
        assert result.peak_nav == 100000.0
        assert result.drawdown == 0.0

        # Check state was saved
        state = state_manager.get_latest_state()
        assert state is not None
        assert state.nav == 100000.0

    def test_tracks_peak_nav(self, tracker):
        """Test that peak NAV is tracked correctly."""
        # First update at 100k
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)

        # Second update at 110k (new peak)
        result = tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=110000.0)
        assert result.peak_nav == 110000.0

        # Third update at 105k (below peak)
        result = tracker.update(as_of=pd.Timestamp("2024-01-17"), nav=105000.0)
        assert result.peak_nav == 110000.0
        assert result.drawdown == pytest.approx((110000 - 105000) / 110000)

    def test_emits_alert_on_threshold_breach(self, tracker, alert_handler):
        """Test that alerts are emitted when drawdown thresholds are breached."""
        # Start at 100k
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)

        # Drop to 88k (12% drawdown, breaches 10% threshold)
        result = tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=88000.0)

        assert len(result.alerts_emitted) > 0
        assert any("10% Drawdown" in a.message for a in result.alerts_emitted)
        assert len(alert_handler.get_alerts()) > 0

    def test_does_not_repeat_alerts_for_same_threshold(self, tracker, alert_handler):
        """Test that alerts are not repeated for the same threshold."""
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)

        # First breach of 10%
        result1 = tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=88000.0)
        alert_count1 = len([a for a in result1.alerts_emitted if "10% Drawdown" in a.message])

        # Still below 10%
        result2 = tracker.update(as_of=pd.Timestamp("2024-01-17"), nav=87000.0)
        alert_count2 = len([a for a in result2.alerts_emitted if "10% Drawdown" in a.message])

        assert alert_count1 >= 1
        assert alert_count2 == 0  # No repeat alert

    def test_emits_recovery_alert(self, tracker, alert_handler):
        """Test that recovery alerts are emitted when recovering above threshold."""
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)

        # Drop below 10%
        tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=88000.0)

        # Recover above 10%
        result = tracker.update(as_of=pd.Timestamp("2024-01-17"), nav=95000.0)

        # Should have recovery alert
        recovery_alerts = [a for a in result.alerts_emitted if "Recovered" in a.message]
        assert len(recovery_alerts) > 0

    def test_tracks_thresholds_breached(self, tracker):
        """Test that breached thresholds are tracked."""
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)

        # 25% drawdown should breach multiple thresholds
        result = tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=75000.0)

        assert "5% Drawdown" in result.thresholds_breached
        assert "10% Drawdown" in result.thresholds_breached
        assert "20% Drawdown" in result.thresholds_breached

    def test_custom_thresholds(self, state_manager, alert_manager):
        """Test using custom thresholds."""
        custom_thresholds = (
            DrawdownThreshold(level=0.15, alert_level="WARNING", name="15% Custom"),
        )
        tracker = NAVTracker(
            state_manager, alert_manager, drawdown_thresholds=custom_thresholds
        )

        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)

        # 10% drawdown shouldn't trigger custom 15% threshold
        result = tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=90000.0)
        assert len(result.alerts_emitted) == 0

        # 20% drawdown should trigger
        result = tracker.update(as_of=pd.Timestamp("2024-01-17"), nav=80000.0)
        assert len(result.alerts_emitted) == 1
        assert "15% Custom" in result.alerts_emitted[0].message

    def test_check_drawdown_status(self, tracker):
        """Test checking drawdown status without updating."""
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)
        tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=85000.0)

        status = tracker.check_drawdown()

        assert status["current_nav"] == 85000.0
        assert status["peak_nav"] == 100000.0
        assert status["drawdown"] == pytest.approx(0.15)
        assert status["status"] == "WARNING"
        assert len(status["thresholds_breached"]) > 0

    def test_check_drawdown_no_data(self, tracker):
        """Test checking drawdown with no data."""
        status = tracker.check_drawdown()

        assert status["status"] == "NO_DATA"
        assert status["drawdown"] == 0.0

    def test_get_nav_history(self, tracker):
        """Test getting NAV time series."""
        # Use recent dates to avoid lookback filtering
        today = pd.Timestamp.now().normalize()
        tracker.update(as_of=today - pd.Timedelta(days=2), nav=100000.0)
        tracker.update(as_of=today - pd.Timedelta(days=1), nav=105000.0)
        tracker.update(as_of=today, nav=103000.0)

        history = tracker.get_nav_history()

        assert len(history) == 3
        assert history.iloc[0] == 100000.0
        assert history.iloc[-1] == 103000.0

    def test_get_current_drawdown(self, tracker):
        """Test getting current drawdown."""
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)
        tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=90000.0)

        drawdown = tracker.get_current_drawdown()

        assert drawdown == pytest.approx(0.10)

    def test_get_peak_nav(self, tracker):
        """Test getting peak NAV."""
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)
        tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=110000.0)
        tracker.update(as_of=pd.Timestamp("2024-01-17"), nav=105000.0)

        peak = tracker.get_peak_nav()

        assert peak == 110000.0

    def test_reset_triggered_thresholds(self, tracker, alert_handler):
        """Test resetting triggered thresholds."""
        tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=100000.0)
        tracker.update(as_of=pd.Timestamp("2024-01-16"), nav=88000.0)

        initial_alert_count = len(alert_handler.get_alerts())

        # Reset and trigger again
        tracker.reset_triggered_thresholds()
        tracker.update(as_of=pd.Timestamp("2024-01-17"), nav=87000.0)

        # Should emit new alerts
        final_alert_count = len(alert_handler.get_alerts())
        assert final_alert_count > initial_alert_count

    def test_handles_zero_nav(self, tracker):
        """Test handling zero NAV edge case."""
        result = tracker.update(as_of=pd.Timestamp("2024-01-15"), nav=0.0)

        assert result.nav == 0.0
        # Should not raise

    def test_handles_empty_holdings(self, tracker):
        """Test handling empty holdings."""
        result = tracker.update(
            as_of=pd.Timestamp("2024-01-15"),
            nav=100000.0,
            holdings=None,
            weights=None,
        )

        assert result.nav == 100000.0

    def test_thresholds_property(self, tracker):
        """Test accessing thresholds property."""
        thresholds = tracker.thresholds

        assert len(thresholds) > 0
        assert all(isinstance(t, DrawdownThreshold) for t in thresholds)
