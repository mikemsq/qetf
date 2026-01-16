"""Portfolio state management for production systems.

This module provides portfolio state persistence with support for SQLite, JSON,
and in-memory backends. It enables tracking of portfolio state over time,
drawdown calculations, and drift detection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioState:
    """Immutable snapshot of portfolio state at a point in time.

    Attributes:
        as_of: The timestamp this state represents.
        holdings: Series mapping ticker to number of shares held.
        weights: Series mapping ticker to portfolio weight (should sum to ~1.0).
        nav: Net asset value of the portfolio.
        cost_basis: Series mapping ticker to average cost per share.
        peak_nav: Historical peak NAV for drawdown calculation.
        created_at: When this state record was created.
    """

    as_of: pd.Timestamp
    holdings: pd.Series
    weights: pd.Series
    nav: float
    cost_basis: pd.Series
    peak_nav: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate state after initialization."""
        if self.nav < 0:
            raise ValueError(f"NAV cannot be negative: {self.nav}")
        if self.peak_nav < 0:
            raise ValueError(f"Peak NAV cannot be negative: {self.peak_nav}")

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak.

        Returns:
            Drawdown as a positive fraction (e.g., 0.10 = 10% drawdown).
        """
        if self.peak_nav <= 0:
            return 0.0
        return max(0.0, (self.peak_nav - self.nav) / self.peak_nav)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of the state.
        """
        return {
            "as_of": self.as_of.isoformat(),
            "holdings": self.holdings.to_dict(),
            "weights": self.weights.to_dict(),
            "nav": self.nav,
            "cost_basis": self.cost_basis.to_dict(),
            "peak_nav": self.peak_nav,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PortfolioState:
        """Create PortfolioState from dictionary.

        Args:
            data: Dictionary with state data.

        Returns:
            PortfolioState instance.
        """
        return cls(
            as_of=pd.Timestamp(data["as_of"]),
            holdings=pd.Series(data["holdings"]),
            weights=pd.Series(data["weights"]),
            nav=data["nav"],
            cost_basis=pd.Series(data["cost_basis"]),
            peak_nav=data["peak_nav"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class PortfolioHistory:
    """Collection of portfolio states over time.

    Provides methods for analyzing portfolio history including peak NAV
    tracking, drawdown analysis, and NAV time series.

    Attributes:
        states: List of PortfolioState objects, ordered by as_of date.
    """

    states: list[PortfolioState] = field(default_factory=list)

    def __len__(self) -> int:
        """Return number of states in history."""
        return len(self.states)

    def __bool__(self) -> bool:
        """Return True if history has any states."""
        return len(self.states) > 0

    def get_peak_nav(self) -> float:
        """Get historical peak NAV across all states.

        Returns:
            Peak NAV value, or 0.0 if no states exist.
        """
        if not self.states:
            return 0.0
        return max(state.nav for state in self.states)

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak.

        Returns:
            Current drawdown as positive fraction (e.g., 0.10 = 10%),
            or 0.0 if no states exist.
        """
        if not self.states:
            return 0.0

        peak = self.get_peak_nav()
        if peak <= 0:
            return 0.0

        current_nav = self.states[-1].nav
        return max(0.0, (peak - current_nav) / peak)

    def get_nav_series(self) -> pd.Series:
        """Get NAV as a time series.

        Returns:
            Series with datetime index and NAV values.
        """
        if not self.states:
            return pd.Series(dtype=float)

        dates = [state.as_of for state in self.states]
        navs = [state.nav for state in self.states]
        return pd.Series(navs, index=pd.DatetimeIndex(dates), name="nav")

    def get_latest_state(self) -> Optional[PortfolioState]:
        """Get the most recent state.

        Returns:
            Latest PortfolioState or None if no states exist.
        """
        if not self.states:
            return None
        return self.states[-1]

    def add_state(self, state: PortfolioState) -> None:
        """Add a state to history, maintaining chronological order.

        Args:
            state: PortfolioState to add.
        """
        self.states.append(state)
        # Sort by as_of date to maintain order
        self.states.sort(key=lambda s: s.as_of)


class PortfolioStateManager(ABC):
    """Abstract base class for portfolio state persistence.

    Implementations provide different storage backends (SQLite, JSON, in-memory)
    for persisting and retrieving portfolio state over time.
    """

    @abstractmethod
    def save_state(self, state: PortfolioState) -> None:
        """Persist a portfolio state.

        Args:
            state: PortfolioState to save.
        """

    @abstractmethod
    def get_latest_state(self) -> Optional[PortfolioState]:
        """Get the most recent portfolio state.

        Returns:
            Latest PortfolioState or None if no states exist.
        """

    @abstractmethod
    def get_state_as_of(self, as_of: pd.Timestamp) -> Optional[PortfolioState]:
        """Get portfolio state as of a specific date.

        Args:
            as_of: The date to retrieve state for.

        Returns:
            PortfolioState for the given date, or None if not found.
        """

    @abstractmethod
    def get_history(self, lookback_days: int = 365) -> PortfolioHistory:
        """Get portfolio history for a lookback period.

        Args:
            lookback_days: Number of days to look back from today.

        Returns:
            PortfolioHistory containing states within the lookback period.
        """

    def calculate_drift(
        self,
        current: pd.Series,
        target: pd.Series,
        threshold: float = 0.02,
    ) -> pd.DataFrame:
        """Calculate drift between current and target weights.

        Identifies positions where the weight difference exceeds the threshold,
        indicating a potential need for rebalancing.

        Args:
            current: Current portfolio weights (ticker -> weight).
            target: Target portfolio weights (ticker -> weight).
            threshold: Minimum absolute drift to report (default 2%).

        Returns:
            DataFrame with columns: ticker, current_weight, target_weight,
            drift, abs_drift. Only includes positions exceeding threshold.
        """
        # Align series to same index
        all_tickers = sorted(set(current.index) | set(target.index))
        current = current.reindex(all_tickers, fill_value=0.0)
        target = target.reindex(all_tickers, fill_value=0.0)

        # Calculate drift
        drift = target - current
        abs_drift = drift.abs()

        # Create DataFrame
        df = pd.DataFrame(
            {
                "ticker": all_tickers,
                "current_weight": current.values,
                "target_weight": target.values,
                "drift": drift.values,
                "abs_drift": abs_drift.values,
            }
        )

        # Filter by threshold
        df = df[df["abs_drift"] >= threshold]
        df = df.sort_values("abs_drift", ascending=False).reset_index(drop=True)

        return df


class InMemoryStateManager(PortfolioStateManager):
    """In-memory portfolio state manager for testing.

    Stores states in a list without any persistence. Useful for unit tests
    and development.
    """

    def __init__(self) -> None:
        """Initialize in-memory state manager."""
        self._states: list[PortfolioState] = []

    def save_state(self, state: PortfolioState) -> None:
        """Save state to memory.

        Args:
            state: PortfolioState to save.
        """
        self._states.append(state)
        # Sort by date to maintain order
        self._states.sort(key=lambda s: s.as_of)
        logger.debug(f"Saved state for {state.as_of}, total states: {len(self._states)}")

    def get_latest_state(self) -> Optional[PortfolioState]:
        """Get most recent state.

        Returns:
            Latest PortfolioState or None if empty.
        """
        if not self._states:
            return None
        return self._states[-1]

    def get_state_as_of(self, as_of: pd.Timestamp) -> Optional[PortfolioState]:
        """Get state for a specific date.

        Args:
            as_of: Date to retrieve state for.

        Returns:
            PortfolioState for the date or None if not found.
        """
        for state in self._states:
            if state.as_of == as_of:
                return state
        return None

    def get_history(self, lookback_days: int = 365) -> PortfolioHistory:
        """Get history within lookback period.

        Args:
            lookback_days: Days to look back from today.

        Returns:
            PortfolioHistory with matching states.
        """
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        filtered = [s for s in self._states if s.as_of >= cutoff]
        return PortfolioHistory(states=filtered)

    def clear(self) -> None:
        """Clear all stored states."""
        self._states.clear()


class JSONStateManager(PortfolioStateManager):
    """JSON file-based portfolio state manager.

    Stores each state as a JSON file with the as_of date as filename.
    Simple and human-readable, suitable for small-scale usage.
    """

    def __init__(self, data_dir: Union[str, Path]) -> None:
        """Initialize JSON state manager.

        Args:
            data_dir: Directory to store JSON state files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JSONStateManager initialized at {self.data_dir}")

    def _get_state_path(self, as_of: pd.Timestamp) -> Path:
        """Get file path for a state date.

        Args:
            as_of: Date of the state.

        Returns:
            Path to the state file.
        """
        date_str = as_of.strftime("%Y%m%d")
        return self.data_dir / f"state_{date_str}.json"

    def save_state(self, state: PortfolioState) -> None:
        """Save state to JSON file.

        Args:
            state: PortfolioState to save.
        """
        path = self._get_state_path(state.as_of)
        data = state.to_dict()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved state to {path}")

    def get_latest_state(self) -> Optional[PortfolioState]:
        """Get most recent state from files.

        Returns:
            Latest PortfolioState or None if no files exist.
        """
        state_files = sorted(self.data_dir.glob("state_*.json"))
        if not state_files:
            return None

        # Load most recent file (sorted alphabetically = chronologically)
        with open(state_files[-1]) as f:
            data = json.load(f)

        return PortfolioState.from_dict(data)

    def get_state_as_of(self, as_of: pd.Timestamp) -> Optional[PortfolioState]:
        """Get state for a specific date.

        Args:
            as_of: Date to retrieve state for.

        Returns:
            PortfolioState or None if file doesn't exist.
        """
        path = self._get_state_path(as_of)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return PortfolioState.from_dict(data)

    def get_history(self, lookback_days: int = 365) -> PortfolioHistory:
        """Get history from JSON files within lookback period.

        Args:
            lookback_days: Days to look back from today.

        Returns:
            PortfolioHistory with matching states.
        """
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        states: list[PortfolioState] = []

        for path in sorted(self.data_dir.glob("state_*.json")):
            with open(path) as f:
                data = json.load(f)

            state = PortfolioState.from_dict(data)
            if state.as_of >= cutoff:
                states.append(state)

        return PortfolioHistory(states=states)


class SQLiteStateManager(PortfolioStateManager):
    """SQLite-based portfolio state manager for production use.

    Stores states in a SQLite database with proper schema. Suitable for
    production systems requiring durable storage and efficient queries.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        """Initialize SQLite state manager.

        Creates the database and tables if they don't exist.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"SQLiteStateManager initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            SQLite connection object.
        """
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Main portfolio states table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    as_of TEXT NOT NULL UNIQUE,
                    nav REAL NOT NULL,
                    peak_nav REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Holdings table (one-to-many relationship)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS holdings (
                    state_id INTEGER REFERENCES portfolio_states(id),
                    ticker TEXT NOT NULL,
                    shares REAL NOT NULL,
                    weight REAL NOT NULL,
                    cost_basis REAL,
                    PRIMARY KEY(state_id, ticker)
                )
                """
            )

            # Create index for efficient lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_as_of ON portfolio_states(as_of)
                """
            )

            conn.commit()

    def save_state(self, state: PortfolioState) -> None:
        """Save state to SQLite database.

        Args:
            state: PortfolioState to save.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert or replace the portfolio state
            cursor.execute(
                """
                INSERT OR REPLACE INTO portfolio_states (as_of, nav, peak_nav, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    state.as_of.isoformat(),
                    state.nav,
                    state.peak_nav,
                    state.created_at.isoformat(),
                ),
            )

            # Get the state ID
            cursor.execute(
                "SELECT id FROM portfolio_states WHERE as_of = ?",
                (state.as_of.isoformat(),),
            )
            state_id = cursor.fetchone()[0]

            # Delete existing holdings for this state
            cursor.execute("DELETE FROM holdings WHERE state_id = ?", (state_id,))

            # Insert holdings
            for ticker in state.holdings.index:
                shares = float(state.holdings[ticker])
                weight = float(state.weights.get(ticker, 0.0))
                cost = float(state.cost_basis.get(ticker, 0.0))

                cursor.execute(
                    """
                    INSERT INTO holdings (state_id, ticker, shares, weight, cost_basis)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (state_id, ticker, shares, weight, cost),
                )

            conn.commit()
            logger.debug(f"Saved state to SQLite for {state.as_of}")

    def _load_state_from_row(
        self, row: tuple, holdings_data: list[tuple]
    ) -> PortfolioState:
        """Create PortfolioState from database row data.

        Args:
            row: Tuple of (id, as_of, nav, peak_nav, created_at).
            holdings_data: List of (ticker, shares, weight, cost_basis) tuples.

        Returns:
            PortfolioState instance.
        """
        state_id, as_of_str, nav, peak_nav, created_at_str = row

        # Build Series from holdings
        holdings_dict = {}
        weights_dict = {}
        cost_basis_dict = {}

        for ticker, shares, weight, cost_basis in holdings_data:
            holdings_dict[ticker] = shares
            weights_dict[ticker] = weight
            cost_basis_dict[ticker] = cost_basis if cost_basis is not None else 0.0

        return PortfolioState(
            as_of=pd.Timestamp(as_of_str),
            holdings=pd.Series(holdings_dict),
            weights=pd.Series(weights_dict),
            nav=nav,
            cost_basis=pd.Series(cost_basis_dict),
            peak_nav=peak_nav,
            created_at=datetime.fromisoformat(created_at_str),
        )

    def get_latest_state(self) -> Optional[PortfolioState]:
        """Get most recent state from database.

        Returns:
            Latest PortfolioState or None if empty.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get latest state
            cursor.execute(
                """
                SELECT id, as_of, nav, peak_nav, created_at
                FROM portfolio_states
                ORDER BY as_of DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()

            if row is None:
                return None

            state_id = row[0]

            # Get holdings for this state
            cursor.execute(
                """
                SELECT ticker, shares, weight, cost_basis
                FROM holdings
                WHERE state_id = ?
                """,
                (state_id,),
            )
            holdings_data = cursor.fetchall()

            return self._load_state_from_row(row, holdings_data)

    def get_state_as_of(self, as_of: pd.Timestamp) -> Optional[PortfolioState]:
        """Get state for a specific date.

        Args:
            as_of: Date to retrieve state for.

        Returns:
            PortfolioState or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, as_of, nav, peak_nav, created_at
                FROM portfolio_states
                WHERE as_of = ?
                """,
                (as_of.isoformat(),),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            state_id = row[0]

            cursor.execute(
                """
                SELECT ticker, shares, weight, cost_basis
                FROM holdings
                WHERE state_id = ?
                """,
                (state_id,),
            )
            holdings_data = cursor.fetchall()

            return self._load_state_from_row(row, holdings_data)

    def get_history(self, lookback_days: int = 365) -> PortfolioHistory:
        """Get history from database within lookback period.

        Args:
            lookback_days: Days to look back from today.

        Returns:
            PortfolioHistory with matching states.
        """
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        states: list[PortfolioState] = []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get all states within range
            cursor.execute(
                """
                SELECT id, as_of, nav, peak_nav, created_at
                FROM portfolio_states
                WHERE as_of >= ?
                ORDER BY as_of ASC
                """,
                (cutoff.isoformat(),),
            )
            rows = cursor.fetchall()

            for row in rows:
                state_id = row[0]

                cursor.execute(
                    """
                    SELECT ticker, shares, weight, cost_basis
                    FROM holdings
                    WHERE state_id = ?
                    """,
                    (state_id,),
                )
                holdings_data = cursor.fetchall()

                state = self._load_state_from_row(row, holdings_data)
                states.append(state)

        return PortfolioHistory(states=states)

    def delete_state(self, as_of: pd.Timestamp) -> bool:
        """Delete a state from the database.

        Args:
            as_of: Date of state to delete.

        Returns:
            True if state was deleted, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id FROM portfolio_states WHERE as_of = ?",
                (as_of.isoformat(),),
            )
            row = cursor.fetchone()

            if row is None:
                return False

            state_id = row[0]

            cursor.execute("DELETE FROM holdings WHERE state_id = ?", (state_id,))
            cursor.execute("DELETE FROM portfolio_states WHERE id = ?", (state_id,))

            conn.commit()
            logger.debug(f"Deleted state for {as_of}")
            return True
