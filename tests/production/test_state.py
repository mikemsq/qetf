"""Tests for Portfolio State Management module.

Tests cover:
- PortfolioState dataclass functionality
- PortfolioHistory methods
- InMemoryStateManager
- JSONStateManager
- SQLiteStateManager
- Drift detection
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from quantetf.production.state import (
    InMemoryStateManager,
    JSONStateManager,
    PortfolioHistory,
    PortfolioState,
    SQLiteStateManager,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_state() -> PortfolioState:
    """Create a sample portfolio state for testing."""
    return PortfolioState(
        as_of=pd.Timestamp("2026-01-15"),
        holdings=pd.Series({"SPY": 100.0, "QQQ": 50.0, "AGG": 200.0}),
        weights=pd.Series({"SPY": 0.50, "QQQ": 0.25, "AGG": 0.25}),
        nav=100000.0,
        cost_basis=pd.Series({"SPY": 450.0, "QQQ": 380.0, "AGG": 100.0}),
        peak_nav=110000.0,
        created_at=datetime(2026, 1, 15, 10, 0, 0),
    )


@pytest.fixture
def sample_state_2() -> PortfolioState:
    """Create a second sample portfolio state."""
    return PortfolioState(
        as_of=pd.Timestamp("2026-01-16"),
        holdings=pd.Series({"SPY": 105.0, "QQQ": 48.0, "AGG": 195.0}),
        weights=pd.Series({"SPY": 0.52, "QQQ": 0.23, "AGG": 0.25}),
        nav=102000.0,
        cost_basis=pd.Series({"SPY": 450.0, "QQQ": 380.0, "AGG": 100.0}),
        peak_nav=110000.0,
        created_at=datetime(2026, 1, 16, 10, 0, 0),
    )


@pytest.fixture
def temp_json_dir():
    """Create a temporary directory for JSON state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "portfolio_state.db"


# -----------------------------------------------------------------------------
# PortfolioState Tests
# -----------------------------------------------------------------------------


class TestPortfolioState:
    """Tests for PortfolioState dataclass."""

    def test_create_state(self, sample_state):
        """Test basic state creation."""
        assert sample_state.nav == 100000.0
        assert sample_state.peak_nav == 110000.0
        assert len(sample_state.holdings) == 3
        assert len(sample_state.weights) == 3

    def test_state_is_frozen(self, sample_state):
        """Test that state is immutable."""
        with pytest.raises(AttributeError):
            sample_state.nav = 200000.0

    def test_state_negative_nav_raises(self):
        """Test that negative NAV raises ValueError."""
        with pytest.raises(ValueError, match="NAV cannot be negative"):
            PortfolioState(
                as_of=pd.Timestamp("2026-01-15"),
                holdings=pd.Series({"SPY": 100.0}),
                weights=pd.Series({"SPY": 1.0}),
                nav=-1000.0,
                cost_basis=pd.Series({"SPY": 450.0}),
                peak_nav=100000.0,
            )

    def test_state_negative_peak_nav_raises(self):
        """Test that negative peak NAV raises ValueError."""
        with pytest.raises(ValueError, match="Peak NAV cannot be negative"):
            PortfolioState(
                as_of=pd.Timestamp("2026-01-15"),
                holdings=pd.Series({"SPY": 100.0}),
                weights=pd.Series({"SPY": 1.0}),
                nav=100000.0,
                cost_basis=pd.Series({"SPY": 450.0}),
                peak_nav=-1000.0,
            )

    def test_get_current_drawdown(self, sample_state):
        """Test drawdown calculation."""
        # NAV=100000, peak=110000, DD = (110000-100000)/110000 = 0.0909
        expected = (110000.0 - 100000.0) / 110000.0
        assert abs(sample_state.get_current_drawdown() - expected) < 0.0001

    def test_get_current_drawdown_no_drawdown(self):
        """Test drawdown when at peak."""
        state = PortfolioState(
            as_of=pd.Timestamp("2026-01-15"),
            holdings=pd.Series({"SPY": 100.0}),
            weights=pd.Series({"SPY": 1.0}),
            nav=100000.0,
            cost_basis=pd.Series({"SPY": 450.0}),
            peak_nav=100000.0,  # At peak
        )
        assert state.get_current_drawdown() == 0.0

    def test_to_dict_and_from_dict(self, sample_state):
        """Test serialization round-trip."""
        data = sample_state.to_dict()
        restored = PortfolioState.from_dict(data)

        assert restored.as_of == sample_state.as_of
        assert restored.nav == sample_state.nav
        assert restored.peak_nav == sample_state.peak_nav
        pd.testing.assert_series_equal(restored.holdings, sample_state.holdings)
        pd.testing.assert_series_equal(restored.weights, sample_state.weights)


# -----------------------------------------------------------------------------
# PortfolioHistory Tests
# -----------------------------------------------------------------------------


class TestPortfolioHistory:
    """Tests for PortfolioHistory class."""

    def test_empty_history(self):
        """Test empty history behavior."""
        history = PortfolioHistory()
        assert len(history) == 0
        assert not history  # bool check
        assert history.get_peak_nav() == 0.0
        assert history.get_current_drawdown() == 0.0
        assert history.get_latest_state() is None
        assert len(history.get_nav_series()) == 0

    def test_history_with_states(self, sample_state, sample_state_2):
        """Test history with multiple states."""
        history = PortfolioHistory(states=[sample_state, sample_state_2])

        assert len(history) == 2
        assert history  # bool check
        assert history.get_latest_state() == sample_state_2

    def test_get_peak_nav(self, sample_state, sample_state_2):
        """Test peak NAV calculation from history."""
        history = PortfolioHistory(states=[sample_state, sample_state_2])
        # sample_state_2 has higher nav (102000) vs sample_state (100000)
        assert history.get_peak_nav() == 102000.0

    def test_get_current_drawdown(self):
        """Test current drawdown from history."""
        state1 = PortfolioState(
            as_of=pd.Timestamp("2026-01-10"),
            holdings=pd.Series({"SPY": 100.0}),
            weights=pd.Series({"SPY": 1.0}),
            nav=100000.0,
            cost_basis=pd.Series({"SPY": 450.0}),
            peak_nav=100000.0,
        )
        state2 = PortfolioState(
            as_of=pd.Timestamp("2026-01-15"),
            holdings=pd.Series({"SPY": 100.0}),
            weights=pd.Series({"SPY": 1.0}),
            nav=90000.0,  # Dropped
            cost_basis=pd.Series({"SPY": 450.0}),
            peak_nav=100000.0,
        )
        history = PortfolioHistory(states=[state1, state2])

        # Peak is 100000, current is 90000, DD = 10%
        assert abs(history.get_current_drawdown() - 0.10) < 0.0001

    def test_get_nav_series(self, sample_state, sample_state_2):
        """Test NAV series extraction."""
        history = PortfolioHistory(states=[sample_state, sample_state_2])
        nav_series = history.get_nav_series()

        assert isinstance(nav_series, pd.Series)
        assert len(nav_series) == 2
        assert nav_series.iloc[0] == 100000.0
        assert nav_series.iloc[1] == 102000.0

    def test_add_state(self, sample_state, sample_state_2):
        """Test adding states to history."""
        history = PortfolioHistory()
        history.add_state(sample_state_2)  # Add later date first
        history.add_state(sample_state)  # Add earlier date second

        # Should be sorted chronologically
        assert history.states[0].as_of < history.states[1].as_of


# -----------------------------------------------------------------------------
# InMemoryStateManager Tests
# -----------------------------------------------------------------------------


class TestInMemoryStateManager:
    """Tests for InMemoryStateManager."""

    def test_save_and_get_latest(self, sample_state):
        """Test saving and retrieving latest state."""
        manager = InMemoryStateManager()
        manager.save_state(sample_state)

        latest = manager.get_latest_state()
        assert latest is not None
        assert latest.as_of == sample_state.as_of
        assert latest.nav == sample_state.nav

    def test_get_latest_empty(self):
        """Test getting latest from empty manager."""
        manager = InMemoryStateManager()
        assert manager.get_latest_state() is None

    def test_get_state_as_of(self, sample_state, sample_state_2):
        """Test retrieving state by date."""
        manager = InMemoryStateManager()
        manager.save_state(sample_state)
        manager.save_state(sample_state_2)

        found = manager.get_state_as_of(sample_state.as_of)
        assert found is not None
        assert found.nav == sample_state.nav

    def test_get_state_as_of_not_found(self, sample_state):
        """Test retrieving non-existent state."""
        manager = InMemoryStateManager()
        manager.save_state(sample_state)

        found = manager.get_state_as_of(pd.Timestamp("2020-01-01"))
        assert found is None

    def test_get_history(self, sample_state, sample_state_2):
        """Test getting history within lookback period."""
        manager = InMemoryStateManager()
        manager.save_state(sample_state)
        manager.save_state(sample_state_2)

        history = manager.get_history(lookback_days=365)
        assert len(history) == 2

    def test_clear(self, sample_state):
        """Test clearing all states."""
        manager = InMemoryStateManager()
        manager.save_state(sample_state)
        manager.clear()

        assert manager.get_latest_state() is None


# -----------------------------------------------------------------------------
# JSONStateManager Tests
# -----------------------------------------------------------------------------


class TestJSONStateManager:
    """Tests for JSONStateManager."""

    def test_save_and_get_latest(self, temp_json_dir, sample_state):
        """Test saving and retrieving latest state."""
        manager = JSONStateManager(temp_json_dir)
        manager.save_state(sample_state)

        latest = manager.get_latest_state()
        assert latest is not None
        assert latest.as_of == sample_state.as_of
        assert latest.nav == sample_state.nav

    def test_get_latest_empty(self, temp_json_dir):
        """Test getting latest from empty manager."""
        manager = JSONStateManager(temp_json_dir)
        assert manager.get_latest_state() is None

    def test_get_state_as_of(self, temp_json_dir, sample_state, sample_state_2):
        """Test retrieving state by date."""
        manager = JSONStateManager(temp_json_dir)
        manager.save_state(sample_state)
        manager.save_state(sample_state_2)

        found = manager.get_state_as_of(sample_state.as_of)
        assert found is not None
        assert found.nav == sample_state.nav

    def test_get_state_as_of_not_found(self, temp_json_dir, sample_state):
        """Test retrieving non-existent state."""
        manager = JSONStateManager(temp_json_dir)
        manager.save_state(sample_state)

        found = manager.get_state_as_of(pd.Timestamp("2020-01-01"))
        assert found is None

    def test_json_file_created(self, temp_json_dir, sample_state):
        """Test that JSON file is created."""
        manager = JSONStateManager(temp_json_dir)
        manager.save_state(sample_state)

        files = list(temp_json_dir.glob("state_*.json"))
        assert len(files) == 1
        assert "20260115" in files[0].name

    def test_json_state_persistence(self, temp_json_dir, sample_state):
        """Test state persists across manager instances."""
        manager1 = JSONStateManager(temp_json_dir)
        manager1.save_state(sample_state)

        # Create new manager pointing to same dir
        manager2 = JSONStateManager(temp_json_dir)
        latest = manager2.get_latest_state()

        assert latest is not None
        assert latest.nav == sample_state.nav

    def test_get_history(self, temp_json_dir, sample_state, sample_state_2):
        """Test getting history from JSON files."""
        manager = JSONStateManager(temp_json_dir)
        manager.save_state(sample_state)
        manager.save_state(sample_state_2)

        history = manager.get_history(lookback_days=365)
        assert len(history) == 2

    def test_string_path_conversion(self, temp_json_dir):
        """Test that string path is converted to Path."""
        manager = JSONStateManager(str(temp_json_dir))
        assert isinstance(manager.data_dir, Path)


# -----------------------------------------------------------------------------
# SQLiteStateManager Tests
# -----------------------------------------------------------------------------


class TestSQLiteStateManager:
    """Tests for SQLiteStateManager."""

    def test_sqlite_state_save_and_load(self, temp_sqlite_db, sample_state):
        """Test saving and loading state from SQLite."""
        manager = SQLiteStateManager(temp_sqlite_db)
        manager.save_state(sample_state)

        latest = manager.get_latest_state()
        assert latest is not None
        assert latest.as_of == sample_state.as_of
        assert latest.nav == sample_state.nav
        pd.testing.assert_series_equal(
            latest.holdings.sort_index(), sample_state.holdings.sort_index()
        )

    def test_sqlite_get_latest_empty(self, temp_sqlite_db):
        """Test getting latest from empty database."""
        manager = SQLiteStateManager(temp_sqlite_db)
        assert manager.get_latest_state() is None

    def test_sqlite_state_history(self, temp_sqlite_db, sample_state, sample_state_2):
        """Test retrieving state history from SQLite."""
        manager = SQLiteStateManager(temp_sqlite_db)
        manager.save_state(sample_state)
        manager.save_state(sample_state_2)

        history = manager.get_history(lookback_days=365)
        assert len(history) == 2
        assert history.states[0].as_of == sample_state.as_of
        assert history.states[1].as_of == sample_state_2.as_of

    def test_sqlite_get_state_as_of(
        self, temp_sqlite_db, sample_state, sample_state_2
    ):
        """Test retrieving state by date from SQLite."""
        manager = SQLiteStateManager(temp_sqlite_db)
        manager.save_state(sample_state)
        manager.save_state(sample_state_2)

        found = manager.get_state_as_of(sample_state.as_of)
        assert found is not None
        assert found.nav == sample_state.nav

    def test_sqlite_get_state_as_of_not_found(self, temp_sqlite_db, sample_state):
        """Test retrieving non-existent state from SQLite."""
        manager = SQLiteStateManager(temp_sqlite_db)
        manager.save_state(sample_state)

        found = manager.get_state_as_of(pd.Timestamp("2020-01-01"))
        assert found is None

    def test_sqlite_database_created(self, temp_sqlite_db):
        """Test that SQLite database file is created."""
        manager = SQLiteStateManager(temp_sqlite_db)
        assert temp_sqlite_db.exists()

    def test_sqlite_state_persistence(self, temp_sqlite_db, sample_state):
        """Test state persists across manager instances."""
        manager1 = SQLiteStateManager(temp_sqlite_db)
        manager1.save_state(sample_state)

        # Create new manager pointing to same db
        manager2 = SQLiteStateManager(temp_sqlite_db)
        latest = manager2.get_latest_state()

        assert latest is not None
        assert latest.nav == sample_state.nav

    def test_sqlite_upsert_same_date(self, temp_sqlite_db, sample_state):
        """Test that saving same date updates existing record."""
        manager = SQLiteStateManager(temp_sqlite_db)
        manager.save_state(sample_state)

        # Modify and save again
        new_state = PortfolioState(
            as_of=sample_state.as_of,  # Same date
            holdings=pd.Series({"SPY": 120.0}),
            weights=pd.Series({"SPY": 1.0}),
            nav=120000.0,  # Different NAV
            cost_basis=pd.Series({"SPY": 450.0}),
            peak_nav=120000.0,
        )
        manager.save_state(new_state)

        # Should only have one state
        history = manager.get_history(lookback_days=365)
        assert len(history) == 1
        assert history.states[0].nav == 120000.0

    def test_sqlite_delete_state(self, temp_sqlite_db, sample_state):
        """Test deleting a state."""
        manager = SQLiteStateManager(temp_sqlite_db)
        manager.save_state(sample_state)

        result = manager.delete_state(sample_state.as_of)
        assert result is True
        assert manager.get_latest_state() is None

    def test_sqlite_delete_nonexistent(self, temp_sqlite_db):
        """Test deleting non-existent state returns False."""
        manager = SQLiteStateManager(temp_sqlite_db)
        result = manager.delete_state(pd.Timestamp("2020-01-01"))
        assert result is False


# -----------------------------------------------------------------------------
# Drift Detection Tests
# -----------------------------------------------------------------------------


class TestDriftDetection:
    """Tests for drift detection functionality."""

    def test_drift_detection(self):
        """Test basic drift detection."""
        manager = InMemoryStateManager()

        current = pd.Series({"SPY": 0.50, "QQQ": 0.30, "AGG": 0.20})
        target = pd.Series({"SPY": 0.40, "QQQ": 0.35, "AGG": 0.25})

        drift = manager.calculate_drift(current, target, threshold=0.02)

        assert len(drift) == 3  # All exceed 2%
        assert "SPY" in drift["ticker"].values
        assert drift.loc[drift["ticker"] == "SPY", "drift"].values[0] == pytest.approx(
            -0.10
        )

    def test_drift_detection_threshold(self):
        """Test drift threshold filtering."""
        manager = InMemoryStateManager()

        current = pd.Series({"SPY": 0.50, "QQQ": 0.30, "AGG": 0.20})
        target = pd.Series({"SPY": 0.49, "QQQ": 0.30, "AGG": 0.21})  # Small drifts

        drift = manager.calculate_drift(current, target, threshold=0.05)

        # No positions exceed 5% threshold
        assert len(drift) == 0

    def test_drift_detection_new_positions(self):
        """Test drift detection with new positions."""
        manager = InMemoryStateManager()

        current = pd.Series({"SPY": 0.50, "QQQ": 0.50})
        target = pd.Series({"SPY": 0.40, "IWM": 0.30, "AGG": 0.30})  # New: IWM, AGG

        drift = manager.calculate_drift(current, target, threshold=0.02)

        tickers = set(drift["ticker"].values)
        assert "IWM" in tickers  # New position
        assert "AGG" in tickers  # New position
        assert "QQQ" in tickers  # Removed position

    def test_drift_sorted_by_abs_drift(self):
        """Test drift results are sorted by absolute drift."""
        manager = InMemoryStateManager()

        current = pd.Series({"SPY": 0.50, "QQQ": 0.30, "AGG": 0.20})
        target = pd.Series({"SPY": 0.35, "QQQ": 0.35, "AGG": 0.30})

        drift = manager.calculate_drift(current, target, threshold=0.02)

        # SPY has largest absolute drift (0.15)
        assert drift.iloc[0]["ticker"] == "SPY"


# -----------------------------------------------------------------------------
# PortfolioHistory Peak NAV Tests
# -----------------------------------------------------------------------------


class TestPortfolioHistoryPeakNav:
    """Additional tests for peak NAV tracking."""

    def test_portfolio_history_peak_nav_increases(self):
        """Test peak NAV updates with new highs."""
        states = []
        for i, nav in enumerate([100, 110, 105, 120, 115]):
            state = PortfolioState(
                as_of=pd.Timestamp(f"2026-01-{10+i}"),
                holdings=pd.Series({"SPY": 100.0}),
                weights=pd.Series({"SPY": 1.0}),
                nav=float(nav * 1000),
                cost_basis=pd.Series({"SPY": 450.0}),
                peak_nav=float(max(100, nav) * 1000),
            )
            states.append(state)

        history = PortfolioHistory(states=states)
        assert history.get_peak_nav() == 120000.0

    def test_portfolio_history_drawdown_from_peak(self):
        """Test drawdown calculation from history peak."""
        states = [
            PortfolioState(
                as_of=pd.Timestamp("2026-01-10"),
                holdings=pd.Series({"SPY": 100.0}),
                weights=pd.Series({"SPY": 1.0}),
                nav=100000.0,
                cost_basis=pd.Series({"SPY": 450.0}),
                peak_nav=100000.0,
            ),
            PortfolioState(
                as_of=pd.Timestamp("2026-01-11"),
                holdings=pd.Series({"SPY": 100.0}),
                weights=pd.Series({"SPY": 1.0}),
                nav=110000.0,  # New peak
                cost_basis=pd.Series({"SPY": 450.0}),
                peak_nav=110000.0,
            ),
            PortfolioState(
                as_of=pd.Timestamp("2026-01-12"),
                holdings=pd.Series({"SPY": 100.0}),
                weights=pd.Series({"SPY": 1.0}),
                nav=99000.0,  # Drop below first
                cost_basis=pd.Series({"SPY": 450.0}),
                peak_nav=110000.0,
            ),
        ]

        history = PortfolioHistory(states=states)

        # Peak is 110000, current is 99000
        expected_dd = (110000 - 99000) / 110000
        assert abs(history.get_current_drawdown() - expected_dd) < 0.0001
