"""Tests for the regime monitoring module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from quantetf.monitoring.alerts import AlertManager, InMemoryAlertHandler
from quantetf.monitoring.regime import (
    RegimeCheckResult,
    RegimeMonitor,
    RegimeState,
)


class TestRegimeState:
    """Tests for RegimeState dataclass."""

    def test_create_state(self):
        """Test creating a regime state."""
        state = RegimeState(
            regime="RISK_ON",
            vix=15.5,
            yield_curve_spread=0.50,
            timestamp=datetime.now(timezone.utc),
        )

        assert state.regime == "RISK_ON"
        assert state.vix == 15.5
        assert state.yield_curve_spread == 0.50

    def test_to_dict(self):
        """Test converting state to dictionary."""
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        state = RegimeState(
            regime="HIGH_VOL",
            vix=35.0,
            yield_curve_spread=0.25,
            timestamp=ts,
            previous_regime="RISK_ON",
        )

        data = state.to_dict()

        assert data["regime"] == "HIGH_VOL"
        assert data["vix"] == 35.0
        assert data["previous_regime"] == "RISK_ON"


class TestRegimeCheckResult:
    """Tests for RegimeCheckResult dataclass."""

    def test_create_result(self):
        """Test creating a check result."""
        state = RegimeState(
            regime="ELEVATED_VOL",
            vix=25.0,
            yield_curve_spread=0.30,
            timestamp=datetime.now(timezone.utc),
        )

        result = RegimeCheckResult(
            current_state=state,
            regime_changed=True,
        )

        assert result.regime_changed is True
        assert result.current_state.regime == "ELEVATED_VOL"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        state = RegimeState(
            regime="RISK_ON",
            vix=18.0,
            yield_curve_spread=0.50,
            timestamp=datetime.now(timezone.utc),
        )

        result = RegimeCheckResult(
            current_state=state,
            regime_changed=False,
        )

        data = result.to_dict()

        assert data["regime_changed"] is False
        assert "current_state" in data


class TestRegimeMonitor:
    """Tests for RegimeMonitor class."""

    @pytest.fixture
    def mock_macro_loader(self):
        """Create a mock macro data loader."""
        loader = MagicMock()
        loader.get_vix.return_value = 15.0
        loader.get_yield_curve_spread.return_value = 0.50
        return loader

    @pytest.fixture
    def alert_handler(self):
        """Create in-memory alert handler."""
        return InMemoryAlertHandler()

    @pytest.fixture
    def alert_manager(self, alert_handler):
        """Create alert manager with in-memory handler."""
        return AlertManager(handlers=[alert_handler])

    @pytest.fixture
    def monitor(self, mock_macro_loader, alert_manager):
        """Create regime monitor."""
        return RegimeMonitor(mock_macro_loader, alert_manager)

    def test_detect_risk_on_regime(self, monitor, mock_macro_loader):
        """Test detecting RISK_ON regime."""
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50

        result = monitor.check("2024-01-15")

        assert result.current_state.regime == "RISK_ON"
        assert result.current_state.vix == 15.0

    def test_detect_elevated_vol_regime(self, monitor, mock_macro_loader):
        """Test detecting ELEVATED_VOL regime."""
        mock_macro_loader.get_vix.return_value = 25.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.30

        result = monitor.check("2024-01-15")

        assert result.current_state.regime == "ELEVATED_VOL"

    def test_detect_high_vol_regime(self, monitor, mock_macro_loader):
        """Test detecting HIGH_VOL regime."""
        mock_macro_loader.get_vix.return_value = 35.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.20

        result = monitor.check("2024-01-15")

        assert result.current_state.regime == "HIGH_VOL"

    def test_detect_recession_warning(self, monitor, mock_macro_loader):
        """Test detecting RECESSION_WARNING regime (inverted yield curve)."""
        mock_macro_loader.get_vix.return_value = 20.0
        mock_macro_loader.get_yield_curve_spread.return_value = -0.25

        result = monitor.check("2024-01-15")

        assert result.current_state.regime == "RECESSION_WARNING"

    def test_recession_warning_takes_priority(self, monitor, mock_macro_loader):
        """Test that recession warning takes priority over VIX."""
        mock_macro_loader.get_vix.return_value = 35.0  # Would be HIGH_VOL
        mock_macro_loader.get_yield_curve_spread.return_value = -0.10  # But inverted

        result = monitor.check("2024-01-15")

        assert result.current_state.regime == "RECESSION_WARNING"

    def test_detects_regime_change(self, monitor, mock_macro_loader, alert_handler):
        """Test detecting regime changes."""
        # First check: RISK_ON
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50
        result1 = monitor.check("2024-01-15")

        assert result1.regime_changed is False  # First check, no previous

        # Second check: HIGH_VOL
        mock_macro_loader.get_vix.return_value = 35.0
        result2 = monitor.check("2024-01-16")

        assert result2.regime_changed is True
        assert result2.current_state.previous_regime == "RISK_ON"
        assert result2.current_state.regime == "HIGH_VOL"

    def test_emits_alert_on_regime_change(self, monitor, mock_macro_loader, alert_handler):
        """Test that alerts are emitted on regime change."""
        # First check
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50
        monitor.check("2024-01-15")

        # Change to HIGH_VOL
        mock_macro_loader.get_vix.return_value = 35.0
        result = monitor.check("2024-01-16")

        assert len(result.alerts_emitted) == 1
        assert result.alerts_emitted[0].category == "REGIME"
        assert "RISK_ON -> HIGH_VOL" in result.alerts_emitted[0].message

    def test_critical_alert_for_high_vol(self, monitor, mock_macro_loader, alert_handler):
        """Test that transitioning to HIGH_VOL emits CRITICAL alert."""
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50
        monitor.check("2024-01-15")

        mock_macro_loader.get_vix.return_value = 35.0
        result = monitor.check("2024-01-16")

        assert result.alerts_emitted[0].level == "CRITICAL"

    def test_info_alert_for_recovery(self, monitor, mock_macro_loader, alert_handler):
        """Test that recovering to RISK_ON emits INFO alert."""
        # Start in HIGH_VOL
        mock_macro_loader.get_vix.return_value = 35.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50
        monitor.check("2024-01-15")

        # Recover to RISK_ON
        mock_macro_loader.get_vix.return_value = 15.0
        result = monitor.check("2024-01-16")

        assert result.alerts_emitted[0].level == "INFO"

    def test_no_alert_when_regime_unchanged(self, monitor, mock_macro_loader, alert_handler):
        """Test that no alert is emitted when regime is unchanged."""
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50

        monitor.check("2024-01-15")
        result = monitor.check("2024-01-16")

        assert len(result.alerts_emitted) == 0

    def test_handles_vix_data_unavailable(self, monitor, mock_macro_loader):
        """Test handling when VIX data is unavailable."""
        mock_macro_loader.get_vix.side_effect = Exception("Data unavailable")
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50

        result = monitor.check("2024-01-15")

        assert result.current_state.regime == "UNKNOWN"
        assert result.current_state.vix is None

    def test_handles_yield_curve_data_unavailable(self, monitor, mock_macro_loader):
        """Test handling when yield curve data is unavailable."""
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.side_effect = Exception("Data unavailable")

        result = monitor.check("2024-01-15")

        # Should still detect based on VIX
        assert result.current_state.regime == "RISK_ON"
        assert result.current_state.yield_curve_spread is None

    def test_get_current_regime(self, monitor, mock_macro_loader):
        """Test getting current regime without emitting alerts."""
        mock_macro_loader.get_vix.return_value = 25.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.30

        status = monitor.get_current_regime("2024-01-15")

        assert status["regime"] == "ELEVATED_VOL"
        assert status["vix"] == 25.0
        assert "description" in status

    def test_get_regime_description(self, monitor):
        """Test getting regime descriptions."""
        assert "favorable" in monitor._get_regime_description("RISK_ON").lower()
        assert "elevated" in monitor._get_regime_description("ELEVATED_VOL").lower()
        assert "high" in monitor._get_regime_description("HIGH_VOL").lower()
        assert "recession" in monitor._get_regime_description("RECESSION_WARNING").lower()

    def test_regime_history_tracking(self, monitor, mock_macro_loader):
        """Test that regime history is tracked."""
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50

        monitor.check("2024-01-15")
        monitor.check("2024-01-16")
        monitor.check("2024-01-17")

        history = monitor.regime_history
        assert len(history) == 3

    def test_regime_history_bounded(self, monitor, mock_macro_loader):
        """Test that regime history doesn't grow unbounded."""
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50

        # Add more than 1000 states
        for i in range(1100):
            monitor.check(f"2024-01-{(i % 28) + 1:02d}")

        history = monitor.regime_history
        assert len(history) <= 1000

    def test_get_regime_stats(self, monitor, mock_macro_loader):
        """Test getting regime statistics."""
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50

        for _ in range(10):
            monitor.check("2024-01-15")

        stats = monitor.get_regime_stats()

        assert stats["total_observations"] == 10
        assert "RISK_ON" in stats["regime_counts"]
        assert stats["regime_counts"]["RISK_ON"] == 10
        assert stats["regime_percentages"]["RISK_ON"] == 1.0

    def test_get_regime_stats_empty(self, monitor):
        """Test getting regime stats with no history."""
        stats = monitor.get_regime_stats()

        assert stats["total_observations"] == 0
        assert stats["regime_counts"] == {}

    def test_reset(self, monitor, mock_macro_loader):
        """Test resetting monitor state."""
        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50

        monitor.check("2024-01-15")
        monitor.check("2024-01-16")

        monitor.reset()

        assert monitor.last_regime is None
        assert len(monitor.regime_history) == 0

    def test_custom_vix_thresholds(self, mock_macro_loader, alert_manager):
        """Test using custom VIX thresholds."""
        monitor = RegimeMonitor(
            mock_macro_loader,
            alert_manager,
            vix_elevated_threshold=15.0,
            vix_high_threshold=25.0,
        )

        mock_macro_loader.get_vix.return_value = 18.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50

        result = monitor.check("2024-01-15")

        # With custom thresholds, 18 VIX should be ELEVATED_VOL
        assert result.current_state.regime == "ELEVATED_VOL"

    def test_last_regime_property(self, monitor, mock_macro_loader):
        """Test accessing last regime property."""
        assert monitor.last_regime is None

        mock_macro_loader.get_vix.return_value = 15.0
        mock_macro_loader.get_yield_curve_spread.return_value = 0.50
        monitor.check("2024-01-15")

        assert monitor.last_regime == "RISK_ON"
