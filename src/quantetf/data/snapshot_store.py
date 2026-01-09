"""Simple data store that loads from a snapshot parquet file.

This provides point-in-time data access from our standardized snapshot format.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from quantetf.data.store import DataStore
from quantetf.types import DatasetVersion


class SnapshotDataStore(DataStore):
    """Loads price data from a snapshot parquet file.

    This store provides point-in-time access to price data in our standardized
    MultiIndex format: (Ticker, Price_Field).

    Example:
        >>> store = SnapshotDataStore(Path("data/snapshots/snapshot_5yr_20etfs/data.parquet"))
        >>> prices = store.read_prices(as_of=pd.Timestamp("2023-12-31"))
    """

    def __init__(self, snapshot_path: Path):
        """Initialize store with path to snapshot parquet file.

        Args:
            snapshot_path: Path to the snapshot data.parquet file
        """
        self.snapshot_path = snapshot_path
        self._data: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self):
        """Load the snapshot data into memory."""
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {self.snapshot_path}")

        self._data = pd.read_parquet(self.snapshot_path)

        # Verify MultiIndex format
        if not isinstance(self._data.columns, pd.MultiIndex):
            raise ValueError(
                f"Expected MultiIndex columns, got: {type(self._data.columns)}"
            )

        if self._data.columns.names != ['Ticker', 'Price']:
            raise ValueError(
                f"Expected column names ['Ticker', 'Price'], got: {self._data.columns.names}"
            )

    def read_prices(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Read price data as-of a specific date (point-in-time).

        CRITICAL: Returns only data BEFORE as_of date (strict inequality) to
        prevent lookahead bias. When making decisions on date T, you only see
        data up through T-1 close.

        Args:
            as_of: The date as of which to return data (exclusive - not included)
            tickers: Optional list of tickers to filter (None = all)
            lookback_days: Optional number of days to look back from as_of

        Returns:
            DataFrame with MultiIndex columns (Ticker, Price_Field) and
            datetime index, containing only data before as_of date.
        """
        # Filter to only data BEFORE as_of (T-1 and earlier)
        pit_data = self._data[self._data.index < as_of].copy()

        if pit_data.empty:
            raise ValueError(f"No data available before {as_of}")

        # Apply lookback window if specified
        if lookback_days is not None:
            start_date = as_of - pd.Timedelta(days=lookback_days)
            pit_data = pit_data[pit_data.index >= start_date]

        # Filter tickers if specified
        if tickers is not None:
            available_tickers = pit_data.columns.get_level_values('Ticker').unique()
            missing_tickers = set(tickers) - set(available_tickers)
            if missing_tickers:
                raise ValueError(f"Tickers not found in snapshot: {missing_tickers}")

            pit_data = pit_data[[t for t in tickers if t in available_tickers]]

        return pit_data

    def get_close_prices(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Get close prices in simple format (date × ticker).

        This is a convenience method that extracts just the Close prices
        and returns them in a simpler format for calculations.

        Args:
            as_of: The date as of which to return data (exclusive)
            tickers: Optional list of tickers to filter
            lookback_days: Optional number of days to look back

        Returns:
            DataFrame with datetime index and ticker columns, values are Close prices
        """
        data = self.read_prices(as_of, tickers, lookback_days)

        # Extract Close prices and pivot to simple format
        close_prices = data.xs('Close', level='Price', axis=1)

        return close_prices

    def read_prices_total_return(
        self,
        version: Optional[DatasetVersion] = None
    ) -> pd.DataFrame:
        """Return DataFrame of daily total returns (date × ticker).

        Note: For now this returns simple returns, not accounting for dividends.
        Future enhancement: incorporate dividend adjustments.
        """
        # Get all close prices
        close_prices = self._data.xs('Close', level='Price', axis=1)

        # Calculate daily returns
        returns = close_prices.pct_change()

        return returns

    def read_instrument_master(
        self,
        version: Optional[DatasetVersion] = None
    ) -> pd.DataFrame:
        """Return instrument metadata.

        For MVP, returns basic metadata inferred from the data.
        """
        tickers = self._data.columns.get_level_values('Ticker').unique().tolist()

        metadata = pd.DataFrame({
            'ticker': tickers,
            'first_date': [self._data[ticker]['Close'].first_valid_index() for ticker in tickers],
            'last_date': [self._data[ticker]['Close'].last_valid_index() for ticker in tickers],
        })

        return metadata.set_index('ticker')

    def create_snapshot(
        self,
        *,
        snapshot_id: str,
        as_of: pd.Timestamp,
        description: str = ""
    ):
        """Not implemented for this simple store."""
        raise NotImplementedError("SnapshotDataStore is read-only")

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the date range covered by this snapshot."""
        return self._data.index.min(), self._data.index.max()

    @property
    def tickers(self) -> list[str]:
        """Return list of tickers in this snapshot."""
        return self._data.columns.get_level_values('Ticker').unique().tolist()
