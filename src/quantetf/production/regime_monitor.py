"""Daily regime monitoring with state persistence."""

from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import json
import logging

import pandas as pd

from quantetf.regime.detector import RegimeDetector
from quantetf.regime.indicators import RegimeIndicators
from quantetf.regime.types import RegimeState, RegimeConfig
from quantetf.regime.config import load_thresholds

if TYPE_CHECKING:
    from quantetf.data.access import DataAccessContext

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path("data/state")


class DailyRegimeMonitor:
    """Monitors market regime daily with state persistence.

    This component:
    1. Loads previous regime state from disk
    2. Gets current market indicators (SPY, VIX)
    3. Detects current regime using hysteresis
    4. Persists new state to disk
    5. Logs regime changes for alerting

    Usage:
        from quantetf.data.access import DataAccessFactory

        ctx = DataAccessFactory.create_context(...)
        monitor = DailyRegimeMonitor(ctx)
        state = monitor.update(as_of=pd.Timestamp.now())
        print(f"Current regime: {state.name}")
    """

    def __init__(
        self,
        data_access: "DataAccessContext",
        state_dir: Path = DEFAULT_STATE_DIR,
        config: Optional[RegimeConfig] = None,
    ):
        """Initialize monitor.

        Args:
            data_access: Data access context for indicators
            state_dir: Directory for state persistence
            config: Regime detection config (uses defaults if not provided)
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.state_dir / "current_regime.json"
        self.history_file = self.state_dir / "regime_history.parquet"

        # Initialize components
        self.config = config or load_thresholds()
        self.detector = RegimeDetector(self.config)
        self.indicators = RegimeIndicators(data_access)

    def update(self, as_of: Optional[pd.Timestamp] = None) -> RegimeState:
        """Update regime state for given date.

        Args:
            as_of: Date to evaluate (defaults to today)

        Returns:
            Current RegimeState after update
        """
        as_of = as_of or pd.Timestamp.now().normalize()

        # Load previous state
        previous_state = self.load_state()

        # Get current indicators
        try:
            indicators = self.indicators.get_current_indicators(as_of)
        except Exception as e:
            logger.error(f"Failed to get indicators: {e}")
            if previous_state:
                logger.warning("Using previous state as fallback")
                return previous_state
            raise

        # Detect current regime
        new_state = self.detector.detect(
            spy_price=indicators["spy_price"],
            spy_200ma=indicators["spy_200ma"],
            vix=indicators["vix"],
            previous_state=previous_state,
            as_of=as_of,
        )

        # Check for regime change
        regime_changed = (
            previous_state is None or new_state.name != previous_state.name
        )

        if regime_changed:
            self._log_regime_change(previous_state, new_state)

        # Persist state
        self.save_state(new_state)
        self._append_history(new_state, regime_changed)

        return new_state

    def load_state(self) -> Optional[RegimeState]:
        """Load current regime state from disk."""
        if not self.state_file.exists():
            logger.info("No previous state found, starting fresh")
            return None

        try:
            with open(self.state_file) as f:
                data = json.load(f)
            return RegimeState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def save_state(self, state: RegimeState) -> None:
        """Save current regime state to disk."""
        data = state.to_dict()
        data["updated_at"] = datetime.now().isoformat()

        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved state: {state.name}")

    def get_current_regime(self) -> str:
        """Get current regime name from persisted state."""
        state = self.load_state()
        if state is None:
            raise ValueError("No regime state available. Run update() first.")
        return state.name

    def get_history(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Get regime history from log.

        Args:
            start_date: Filter start (optional)
            end_date: Filter end (optional)

        Returns:
            DataFrame with columns: date, regime, regime_changed, spy_price, vix
        """
        if not self.history_file.exists():
            return pd.DataFrame()

        df = pd.read_parquet(self.history_file)

        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        return df

    def _append_history(self, state: RegimeState, regime_changed: bool) -> None:
        """Append state to history log."""
        record = pd.DataFrame([{
            "date": state.as_of,
            "regime": state.name,
            "trend": state.trend.value,
            "vol": state.vol.value,
            "spy_price": state.spy_price,
            "spy_200ma": state.spy_200ma,
            "vix": state.vix,
            "regime_changed": regime_changed,
            "recorded_at": pd.Timestamp.now(),
        }])

        if self.history_file.exists():
            existing = pd.read_parquet(self.history_file)
            # Avoid duplicate entries for same date
            existing = existing[existing["date"] != state.as_of]
            df = pd.concat([existing, record], ignore_index=True)
        else:
            df = record

        df.to_parquet(self.history_file, index=False)

    def _log_regime_change(
        self,
        previous: Optional[RegimeState],
        current: RegimeState,
    ) -> None:
        """Log regime change for alerting."""
        prev_name = previous.name if previous else "none"
        logger.warning(
            f"REGIME CHANGE: {prev_name} -> {current.name} "
            f"(SPY={current.spy_price:.2f}, VIX={current.vix:.2f})"
        )

    def get_regime_summary(self) -> dict:
        """Get summary of current regime state.

        Returns:
            Dict suitable for display or API response
        """
        state = self.load_state()
        if state is None:
            return {"error": "No state available"}

        spy_vs_ma_pct = None
        if state.spy_200ma and state.spy_200ma > 0:
            spy_vs_ma_pct = (state.spy_price / state.spy_200ma - 1) * 100

        return {
            "regime": state.name,
            "trend": state.trend.value,
            "volatility": state.vol.value,
            "as_of": state.as_of.isoformat(),
            "indicators": {
                "spy_price": state.spy_price,
                "spy_200ma": state.spy_200ma,
                "spy_vs_ma_pct": spy_vs_ma_pct,
                "vix": state.vix,
            },
        }
