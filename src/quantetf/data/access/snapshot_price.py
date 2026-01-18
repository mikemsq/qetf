"""SnapshotPriceAccessor - price data access from snapshot parquet files."""

from pathlib import Path
from typing import Optional
import pandas as pd

from .abstract import PriceDataAccessor


class SnapshotPriceAccessor(PriceDataAccessor):
    """Access price data from a snapshot parquet file.
    
    Wraps SnapshotDataStore to provide PriceDataAccessor interface.
    Guarantees point-in-time data access (no lookahead bias).
    
    Usage:
        accessor = SnapshotPriceAccessor(
            Path("data/snapshots/snapshot_5yr_20etfs/data.parquet")
        )
        prices = accessor.read_prices_as_of(pd.Timestamp("2023-12-31"))
    """
    
    def __init__(self, snapshot_path: Path):
        """Initialize with path to snapshot parquet file.
        
        Args:
            snapshot_path: Path to data.parquet file in snapshot directory
            
        Raises:
            FileNotFoundError: If snapshot doesn't exist
            ValueError: If snapshot format invalid
        """
        from quantetf.data.snapshot_store import SnapshotDataStore
        
        self.snapshot_path = Path(snapshot_path)
        self._store = SnapshotDataStore(self.snapshot_path)
        
        # Cache latest date for efficient access
        self._latest_date: pd.Timestamp = self._store._data.index.max()
    
    def read_prices_as_of(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return OHLCV prices for all dates < as_of.
        
        Critical: Data on/after as_of must be excluded (lookahead prevention).
        Returns data in MultiIndex format (Ticker, Price_Field).
        
        Args:
            as_of: Cutoff date (exclusive - not included)
            tickers: Optional subset of tickers to return
            lookback_days: Optional window of days to look back
            
        Returns:
            DataFrame with:
            - Index: datetime (dates < as_of)
            - Columns: MultiIndex (Ticker, Field) where Field in [Open, High, Low, Close, Volume]
            
        Raises:
            ValueError: If no data available before as_of
        """
        # Use existing SnapshotDataStore.read_prices() method
        # It already implements point-in-time (strict inequality)
        result = self._store.read_prices(
            as_of=as_of,
            tickers=tickers,
            lookback_days=lookback_days,
        )
        
        return result
    
    def read_ohlcv_range(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return OHLCV for closed date range [start, end].
        
        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            tickers: Optional subset of tickers to return
            
        Returns:
            DataFrame with:
            - Index: datetime (dates in [start, end])
            - Columns: MultiIndex (Ticker, Price_Field)
            
        Raises:
            ValueError: If no data in range
        """
        # Filter snapshot data to the date range [start, end]
        range_data = self._store._data[
            (self._store._data.index >= start) & (self._store._data.index <= end)
        ].copy()
        
        if range_data.empty:
            raise ValueError(f"No price data available in range [{start}, {end}]")
        
        # Filter tickers if specified
        if tickers is not None:
            available_tickers = range_data.columns.get_level_values('Ticker').unique()
            missing_tickers = set(tickers) - set(available_tickers)
            if missing_tickers:
                raise ValueError(f"Tickers not found in snapshot: {missing_tickers}")
            
            range_data = range_data[[t for t in tickers if t in available_tickers]]
        
        return range_data
    
    def get_latest_price_date(self) -> pd.Timestamp:
        """Return most recent date with available price data."""
        return self._latest_date
    
    def validate_data_availability(
        self,
        tickers: list[str],
        as_of: pd.Timestamp,
    ) -> dict[str, bool]:
        """Check which tickers have data available as of date.
        
        Args:
            tickers: List of tickers to validate
            as_of: Cutoff date
            
        Returns:
            Dict mapping ticker → True if available before as_of, False if missing
        """
        result = {}
        
        try:
            # Get data available before as_of
            pit_data = self._store._data[self._store._data.index < as_of]
            available_tickers = pit_data.columns.get_level_values('Ticker').unique()
            
            # Check each requested ticker
            for ticker in tickers:
                result[ticker] = ticker in available_tickers
        except Exception:
            # If anything goes wrong, mark all as unavailable
            result = {t: False for t in tickers}
        
        return result
    
    def get_available_tickers(self) -> list[str]:
        """Return list of all tickers in this snapshot.
        
        Returns:
            List of unique ticker symbols
        """
        return self._store.tickers
    
    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the (start_date, end_date) of this snapshot.
        
        Returns:
            Tuple of (earliest_date, latest_date)
        """
        return self._store.date_range
