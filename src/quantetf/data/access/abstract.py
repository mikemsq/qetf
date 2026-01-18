"""Abstract base classes for Data Access Layer interfaces."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from .types import Regime, TickerMetadata, ExchangeInfo


class PriceDataAccessor(ABC):
    """Abstract interface for price data access.
    
    All implementations must:
    - Provide point-in-time data (no lookahead bias)
    - Filter to dates < as_of (strict inequality)
    - Support optional ticker filtering
    - Support optional lookback windows
    """
    
    @abstractmethod
    def read_prices_as_of(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return OHLCV prices for all dates < as_of.
        
        Critical: Data on/after as_of must be excluded (lookahead prevention).
        
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
        pass
    
    @abstractmethod
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
            - Columns: MultiIndex (Ticker, Field)
        """
        pass
    
    @abstractmethod
    def get_latest_price_date(self) -> pd.Timestamp:
        """Return most recent date with available price data."""
        pass
    
    @abstractmethod
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
        pass


class MacroDataAccessor(ABC):
    """Abstract interface for macro data access."""
    
    @abstractmethod
    def read_macro_indicator(
        self,
        indicator: str,
        as_of: pd.Timestamp,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return time series for a macro indicator.
        
        Returns data up to as_of (point-in-time).
        
        Args:
            indicator: Indicator name (e.g., "VIX", "SPX_YIELD")
            as_of: Cutoff date (exclusive)
            lookback_days: Optional lookback window
            
        Returns:
            DataFrame with DatetimeIndex and indicator values
            
        Raises:
            ValueError: If indicator not available
        """
        pass
    
    @abstractmethod
    def get_regime(self, as_of: pd.Timestamp) -> Regime:
        """Detect current market regime as of date.
        
        Args:
            as_of: Date for regime detection
            
        Returns:
            One of RISK_ON, ELEVATED_VOL, HIGH_VOL, RECESSION_WARNING, UNKNOWN
        """
        pass
    
    @abstractmethod
    def get_available_indicators(self) -> list[str]:
        """Return list of available macro indicators."""
        pass


class UniverseDataAccessor(ABC):
    """Abstract interface for universe (ticker set) definitions."""
    
    @abstractmethod
    def get_universe(self, universe_name: str) -> list[str]:
        """Get current/latest universe tickers.
        
        Args:
            universe_name: Name of the universe
            
        Returns:
            List of tickers in the universe
        """
        pass
    
    @abstractmethod
    def get_universe_as_of(
        self,
        universe_name: str,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """Get universe membership at specific point in time.
        
        For graduated universes, only includes tickers added by as_of.
        
        Args:
            universe_name: Name of the universe
            as_of: Point-in-time date
            
        Returns:
            List of tickers in universe as of that date
        """
        pass
    
    @abstractmethod
    def list_available_universes(self) -> list[str]:
        """Return list of available universe names."""
        pass


class ReferenceDataAccessor(ABC):
    """Abstract interface for static reference data."""
    
    @abstractmethod
    def get_ticker_info(self, ticker: str) -> TickerMetadata:
        """Get metadata for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            TickerMetadata with ticker information
            
        Raises:
            ValueError: If ticker not found
        """
        pass
    
    @abstractmethod
    def get_sector_mapping(self) -> dict[str, str]:
        """Return ticker → sector mapping for all tickers.
        
        Returns:
            Dictionary mapping ticker symbols to sector names
        """
        pass
    
    @abstractmethod
    def get_exchange_info(self) -> dict[str, ExchangeInfo]:
        """Return exchange → metadata mapping.
        
        Returns:
            Dictionary mapping exchange names to ExchangeInfo
        """
        pass
