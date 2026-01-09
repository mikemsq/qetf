"""Abstract base class for ETF data providers.

This module defines the interface that all data providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class DataProvider(ABC):
    """Abstract base class for ETF data providers.
    
    All data providers must implement the methods defined here to ensure
    consistent interface across different data sources.
    """
    
    @abstractmethod
    def fetch_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV price data for ETF tickers.
        
        Args:
            tickers: List of ETF ticker symbols (e.g., ['SPY', 'QQQ'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with datetime index and OHLCV columns, or None if failed.
            For single ticker: columns are ['Open', 'High', 'Low', 'Close', 'Volume']
            For multiple tickers: MultiIndex columns with (ticker, column) pairs
            
        Raises:
            ValueError: If date format is invalid or dates are out of range
        """
        pass
    
    @abstractmethod
    def get_ticker_info(self, ticker: str) -> Optional[dict]:
        """Get metadata and information about an ETF ticker.
        
        Args:
            ticker: ETF ticker symbol
            
        Returns:
            Dictionary with ticker information, or None if not found.
            May include: name, exchange, sector, expense_ratio, aum, etc.
        """
        pass
    
    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker symbol is valid and available.
        
        Args:
            ticker: ETF ticker symbol to validate
            
        Returns:
            True if ticker exists and data is available, False otherwise
        """
        pass
