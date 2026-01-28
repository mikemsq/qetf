"""Helper functions to get regime indicators."""

from typing import Optional
import logging

import pandas as pd

from quantetf.data.access import DataAccessContext

logger = logging.getLogger(__name__)


class RegimeIndicators:
    """Fetches indicators needed for regime detection.

    Provides SPY price, 200-day moving average, and VIX data
    for use with the RegimeDetector.

    Usage:
        from quantetf.data.access import DataAccessFactory
        from quantetf.regime.indicators import RegimeIndicators

        ctx = DataAccessFactory.create_context(...)
        indicators = RegimeIndicators(ctx)

        current = indicators.get_current_indicators(pd.Timestamp("2026-01-24"))
        # Returns: {"spy_price": 600.5, "spy_200ma": 550.2, "vix": 15.3, ...}
    """

    def __init__(self, data_access: DataAccessContext):
        """Initialize with data access context.

        Args:
            data_access: DataAccessContext with prices and macro accessors
        """
        self.data_access = data_access

    def get_spy_data(
        self,
        as_of: pd.Timestamp,
        lookback_days: int = 250,
    ) -> pd.DataFrame:
        """Get SPY price and 200-day moving average.

        Args:
            as_of: Point-in-time date
            lookback_days: Days of history to fetch (default 250 for 200MA)

        Returns:
            DataFrame with columns: ['close', 'ma_200']
        """
        # Calculate start date for enough history
        start_date = as_of - pd.Timedelta(days=lookback_days + 50)

        # Get SPY prices
        prices = self.data_access.prices.read_prices_as_of(
            as_of=as_of,
            tickers=["SPY"],
        )

        # Handle empty DataFrame
        if prices.empty:
            return pd.DataFrame(columns=["close", "ma_200"])

        # Extract close prices - handle various column formats
        if isinstance(prices.columns, pd.MultiIndex):
            # Try to extract by level name or position
            level_names = prices.columns.names
            if "Price" in level_names:
                spy_close = prices.xs("Close", level="Price", axis=1)["SPY"]
            elif len(level_names) >= 2:
                # Assume format is (Ticker, Field) - get Close for SPY
                spy_close = prices[("SPY", "Close")]
            else:
                spy_close = prices["SPY"]
        else:
            spy_close = prices["SPY"]

        # Filter to date range
        spy_close = spy_close[spy_close.index >= start_date]

        # Calculate 200MA
        ma_200 = spy_close.rolling(window=200, min_periods=200).mean()

        return pd.DataFrame({
            "close": spy_close,
            "ma_200": ma_200,
        })

    def get_vix(
        self,
        as_of: pd.Timestamp,
        lookback_days: int = 30,
    ) -> pd.Series:
        """Get VIX values up to as_of date.

        Args:
            as_of: Point-in-time date
            lookback_days: Days of history to return

        Returns:
            Series of VIX values indexed by date
        """
        vix_df = self.data_access.macro.read_macro_indicator(
            indicator="VIX",
            as_of=as_of,
            lookback_days=lookback_days,
        )

        # Return as Series
        return vix_df.iloc[:, 0] if not vix_df.empty else pd.Series(dtype=float)

    def get_current_indicators(
        self,
        as_of: pd.Timestamp,
    ) -> dict:
        """Get all indicators needed for regime detection.

        Args:
            as_of: Point-in-time date

        Returns:
            Dictionary with:
            - spy_price: Current SPY closing price
            - spy_200ma: 200-day moving average of SPY
            - vix: Current VIX level
            - as_of: The as_of date used

        Raises:
            ValueError: If any required indicator is unavailable
        """
        # Get SPY data
        spy_data = self.get_spy_data(as_of)

        if spy_data.empty:
            raise ValueError(f"No SPY data available as of {as_of}")

        # Get most recent values at or before as_of
        spy_valid = spy_data.dropna(subset=["ma_200"])
        if spy_valid.empty:
            raise ValueError(f"Not enough SPY history for 200MA as of {as_of}")

        spy_row = spy_valid.loc[:as_of].iloc[-1]

        # Get VIX data
        vix_data = self.get_vix(as_of)
        if vix_data.empty:
            raise ValueError(f"No VIX data available as of {as_of}")

        vix_value = float(vix_data.loc[:as_of].iloc[-1])

        return {
            "spy_price": float(spy_row["close"]),
            "spy_200ma": float(spy_row["ma_200"]),
            "vix": vix_value,
            "as_of": as_of,
        }

    def get_indicators_safe(
        self,
        as_of: pd.Timestamp,
    ) -> Optional[dict]:
        """Get indicators with error handling.

        Like get_current_indicators but returns None on errors
        instead of raising exceptions.

        Args:
            as_of: Point-in-time date

        Returns:
            Dictionary of indicators, or None if unavailable
        """
        try:
            return self.get_current_indicators(as_of)
        except (ValueError, KeyError, IndexError) as e:
            logger.warning(f"Failed to get indicators as of {as_of}: {e}")
            return None
