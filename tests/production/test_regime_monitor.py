"""Tests for daily regime monitoring."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
import json
import logging

from quantetf.production.regime_monitor import DailyRegimeMonitor
from quantetf.regime.types import RegimeState, TrendState, VolatilityState


class MockIndicators:
    """Mock indicators that return controlled data."""

    def __init__(self, spy_price=600, spy_200ma=550, vix=15):
        self.spy_price = spy_price
        self.spy_200ma = spy_200ma
        self.vix = vix
        self.data_access = MagicMock()

    def get_current_indicators(self, as_of):
        return {
            "spy_price": self.spy_price,
            "spy_200ma": self.spy_200ma,
            "vix": self.vix,
            "as_of": as_of,
        }


class TestDailyRegimeMonitor:
    """Test daily regime monitoring."""

    @pytest.fixture
    def state_dir(self, tmp_path):
        return tmp_path / "state"

    @pytest.fixture
    def mock_data_access(self):
        """Create mock data access context."""
        ctx = MagicMock()

        # Mock SPY prices
        dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
        spy_prices = pd.DataFrame(
            np.linspace(500, 600, len(dates)),
            index=dates,
            columns=pd.MultiIndex.from_product(
                [["SPY"], ["Close"]], names=["Ticker", "Price"]
            ),
        )
        ctx.prices.read_prices_as_of.return_value = spy_prices

        # Mock VIX data
        vix_dates = pd.date_range("2025-12-01", "2026-01-20", freq="B")
        vix_data = pd.DataFrame(
            {"VIX": np.random.uniform(12, 20, len(vix_dates))},
            index=vix_dates,
        )
        ctx.macro.read_macro_indicator.return_value = vix_data

        return ctx

    @pytest.fixture
    def monitor(self, mock_data_access, state_dir):
        return DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=state_dir,
        )

    def test_init_creates_state_dir(self, mock_data_access, tmp_path):
        """Monitor should create state directory on init."""
        state_dir = tmp_path / "new_state_dir"
        assert not state_dir.exists()

        DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=state_dir,
        )

        assert state_dir.exists()

    def test_update_creates_state_file(self, monitor, state_dir):
        """First update should create state file."""
        state = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        assert (state_dir / "current_regime.json").exists()
        assert state.name in [
            "uptrend_low_vol",
            "uptrend_high_vol",
            "downtrend_low_vol",
            "downtrend_high_vol",
        ]

    def test_update_persists_and_loads(self, monitor, mock_data_access, state_dir):
        """State should survive save/load cycle."""
        state1 = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        # Create new monitor instance (simulates restart)
        monitor2 = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=state_dir,
        )
        loaded = monitor2.load_state()

        assert loaded is not None
        assert loaded.name == state1.name

    def test_history_appended(self, monitor, state_dir):
        """Each update should append to history."""
        monitor.update(as_of=pd.Timestamp("2026-01-16"))
        monitor.update(as_of=pd.Timestamp("2026-01-17"))
        monitor.update(as_of=pd.Timestamp("2026-01-20"))

        history = monitor.get_history()
        assert len(history) == 3

    def test_history_no_duplicates(self, monitor, state_dir):
        """Updating same date twice should not create duplicates."""
        monitor.update(as_of=pd.Timestamp("2026-01-20"))
        monitor.update(as_of=pd.Timestamp("2026-01-20"))

        history = monitor.get_history()
        assert len(history) == 1

    def test_get_current_regime(self, monitor):
        """get_current_regime should return regime name."""
        monitor.update(as_of=pd.Timestamp("2026-01-20"))
        regime = monitor.get_current_regime()

        assert isinstance(regime, str)
        assert "_" in regime  # e.g., "uptrend_low_vol"

    def test_no_state_raises(self, monitor):
        """get_current_regime should raise if no state."""
        with pytest.raises(ValueError, match="No regime state"):
            monitor.get_current_regime()

    def test_load_state_no_file(self, monitor):
        """load_state should return None if no file exists."""
        result = monitor.load_state()
        assert result is None

    def test_save_and_load_state(self, monitor):
        """State should survive save/load cycle."""
        state = RegimeState(
            trend=TrendState.UPTREND,
            vol=VolatilityState.LOW_VOL,
            as_of=pd.Timestamp("2026-01-20"),
            spy_price=600.0,
            spy_200ma=550.0,
            vix=15.0,
        )

        monitor.save_state(state)
        loaded = monitor.load_state()

        assert loaded is not None
        assert loaded.name == state.name
        assert loaded.spy_price == state.spy_price

    def test_state_file_contains_updated_at(self, monitor, state_dir):
        """State file should include updated_at timestamp."""
        monitor.update(as_of=pd.Timestamp("2026-01-20"))

        with open(state_dir / "current_regime.json") as f:
            data = json.load(f)

        assert "updated_at" in data

    def test_get_regime_summary(self, monitor):
        """Summary should include all relevant info."""
        monitor.update(as_of=pd.Timestamp("2026-01-20"))
        summary = monitor.get_regime_summary()

        assert "regime" in summary
        assert "trend" in summary
        assert "volatility" in summary
        assert "indicators" in summary
        assert "spy_price" in summary["indicators"]
        assert "vix" in summary["indicators"]
        assert "spy_vs_ma_pct" in summary["indicators"]

    def test_get_regime_summary_no_state(self, monitor):
        """Summary should return error if no state."""
        summary = monitor.get_regime_summary()
        assert "error" in summary

    def test_history_filter_by_date(self, monitor):
        """get_history should filter by date range."""
        monitor.update(as_of=pd.Timestamp("2026-01-15"))
        monitor.update(as_of=pd.Timestamp("2026-01-17"))
        monitor.update(as_of=pd.Timestamp("2026-01-20"))

        history = monitor.get_history(
            start_date=pd.Timestamp("2026-01-16"),
            end_date=pd.Timestamp("2026-01-18"),
        )

        assert len(history) == 1
        assert history.iloc[0]["date"] == pd.Timestamp("2026-01-17")

    def test_regime_change_logged(self, state_dir, caplog):
        """Regime changes should be logged as warnings."""
        ctx = MagicMock()

        # First update: uptrend
        dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
        spy_prices = pd.DataFrame(
            np.linspace(600, 650, len(dates)),  # Uptrend
            index=dates,
            columns=pd.MultiIndex.from_product(
                [["SPY"], ["Close"]], names=["Ticker", "Price"]
            ),
        )
        ctx.prices.read_prices_as_of.return_value = spy_prices

        vix_dates = pd.date_range("2025-12-01", "2026-01-20", freq="B")
        vix_data = pd.DataFrame({"VIX": [15.0] * len(vix_dates)}, index=vix_dates)
        ctx.macro.read_macro_indicator.return_value = vix_data

        monitor = DailyRegimeMonitor(data_access=ctx, state_dir=state_dir)

        with caplog.at_level(logging.WARNING):
            # First update logs as regime change (from none)
            monitor.update(as_of=pd.Timestamp("2026-01-20"))

        assert any("REGIME CHANGE" in record.message for record in caplog.records)


class TestRegimeMonitorIndicatorErrors:
    """Test error handling when indicators fail."""

    @pytest.fixture
    def state_dir(self, tmp_path):
        return tmp_path / "state"

    def test_indicator_failure_uses_previous_state(self, state_dir):
        """If indicators fail, should use previous state."""
        ctx = MagicMock()

        # Setup initial working state
        dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
        spy_prices = pd.DataFrame(
            np.linspace(500, 600, len(dates)),
            index=dates,
            columns=pd.MultiIndex.from_product(
                [["SPY"], ["Close"]], names=["Ticker", "Price"]
            ),
        )
        ctx.prices.read_prices_as_of.return_value = spy_prices

        vix_dates = pd.date_range("2025-12-01", "2026-01-20", freq="B")
        vix_data = pd.DataFrame({"VIX": [15.0] * len(vix_dates)}, index=vix_dates)
        ctx.macro.read_macro_indicator.return_value = vix_data

        monitor = DailyRegimeMonitor(data_access=ctx, state_dir=state_dir)
        first_state = monitor.update(as_of=pd.Timestamp("2026-01-19"))

        # Now make indicators fail
        ctx.prices.read_prices_as_of.side_effect = ValueError("Connection failed")

        # Should return previous state as fallback
        second_state = monitor.update(as_of=pd.Timestamp("2026-01-20"))
        assert second_state.name == first_state.name

    def test_indicator_failure_no_previous_raises(self, state_dir):
        """If indicators fail and no previous state, should raise."""
        ctx = MagicMock()
        ctx.prices.read_prices_as_of.side_effect = ValueError("Connection failed")

        monitor = DailyRegimeMonitor(data_access=ctx, state_dir=state_dir)

        with pytest.raises(ValueError):
            monitor.update(as_of=pd.Timestamp("2026-01-20"))
