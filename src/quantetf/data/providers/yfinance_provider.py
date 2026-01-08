"""yfinance data provider implementation for ETF price data.

This module implements the DataProvider interface using the yfinance library
to fetch ETF price data from Yahoo Finance.
"""

import logging
from typing import List, Optional
import pandas as pd
from datetime import datetime

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from .base import DataProvider


logger = logging.getLogger(__name__)


class YFinanceProvider(DataProvider):
    """Data provider implementation using yfinance (Yahoo Finance).
    
    This provider fetches ETF price data from Yahoo Finance using the yfinance
    library. It provides daily OHLCV data with adjusted close prices that account
    for splits and dividends.
    
    Note: yfinance is an unofficial API and Yahoo Finance's terms of service
    restrict usage to personal, non-commercial purposes.
    
    Example:
        >>> provider = YFinanceProvider()
        >>> data = provider.fetch_prices(['SPY'], '2020-01-01', '2021-01-01')
        >>> print(data.head())
    """
    
    def __init__(self, auto_adjust: bool = True, progress: bool = False):
        """Initialize yfinance provider.
        
        Args:
            auto_adjust: If True, use adjusted close prices (default: True)
            progress: If True, show download progress bar (default: False)
        
        Raises:
            ImportError: If yfinance is not installed
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is not installed. "
                "Install it with: pip install yfinance"
            )
        self.auto_adjust = auto_adjust
        self.progress = progress
    
    def fetch_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV price data from Yahoo Finance.
        
        Args:
            tickers: List of ETF ticker symbols (e.g., ['SPY', 'QQQ'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with datetime index and OHLCV columns.
            For single ticker: columns are ['Open', 'High', 'Low', 'Close', 'Volume']
            For multiple tickers: MultiIndex columns with (ticker, column) pairs
            Returns None if fetch fails.
            
        Raises:
            ValueError: If date format is invalid
        
        Example:
            >>> provider = YFinanceProvider()
            >>> data = provider.fetch_prices(['SPY', 'QQQ'], '2020-01-01', '2021-01-01')
            >>> print(data.columns.levels[0])  # Shows tickers
            >>> print(data.columns.levels[1])  # Shows OHLCV columns
        """
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
        
        # Validate tickers is not empty
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        try:
            # Download data using yfinance
            logger.info(f"Fetching data for {len(tickers)} ticker(s) from {start_date} to {end_date}")
            
            data = yf.download(
                tickers=tickers if len(tickers) > 1 else tickers[0],
                start=start_date,
                end=end_date,
                auto_adjust=self.auto_adjust,
                progress=self.progress,
                actions=False  # Don't include dividend/split data
            )
            
            # Check if data was retrieved
            if data.empty:
                logger.warning(f"No data retrieved for {tickers}")
                return None
            
            # Ensure consistent column structure
            if len(tickers) == 1:
                # For single ticker, yfinance returns simple columns
                # Ensure we have the expected column names
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in expected_cols):
                    logger.error(f"Missing required columns for {tickers[0]}")
                    return None
            else:
                # For multiple tickers, yfinance returns MultiIndex columns
                # Format: (column, ticker) - we want (ticker, column)
                if isinstance(data.columns, pd.MultiIndex):
                    # Swap levels if needed
                    if data.columns.names[0] != 'Ticker':
                        data.columns = data.columns.swaplevel(0, 1)
            
            logger.info(f"Successfully fetched {len(data)} rows for {len(tickers)} ticker(s)")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            return None
    
    def get_ticker_info(self, ticker: str) -> Optional[dict]:
        """Get metadata and information about an ETF ticker.
        
        Args:
            ticker: ETF ticker symbol
            
        Returns:
            Dictionary with ticker information including:
            - longName: Full name of the ETF
            - symbol: Ticker symbol
            - quoteType: Asset type (ETF, EQUITY, etc.)
            - exchange: Exchange where traded
            - currency: Trading currency
            And other fields available from yfinance
            Returns None if ticker not found or error occurs.
            
        Example:
            >>> provider = YFinanceProvider()
            >>> info = provider.get_ticker_info('SPY')
            >>> print(info.get('longName'))
            'SPDR S&P 500 ETF Trust'
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if not info or 'symbol' not in info:
                logger.warning(f"No information found for ticker: {ticker}")
                return None
            
            logger.info(f"Retrieved info for {ticker}")
            return info
            
        except Exception as e:
            logger.error(f"Error fetching ticker info for {ticker}: {e}")
            return None
    
    def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker symbol is valid and data is available.
        
        Args:
            ticker: ETF ticker symbol to validate
            
        Returns:
            True if ticker exists and has available data, False otherwise
            
        Example:
            >>> provider = YFinanceProvider()
            >>> provider.validate_ticker('SPY')
            True
            >>> provider.validate_ticker('INVALIDTICKER123')
            False
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            # Try to get basic info
            info = ticker_obj.info
            
            # Check if we got valid information
            if not info or 'symbol' not in info:
                return False
            
            # Try to fetch a small amount of recent data
            hist = ticker_obj.history(period='5d')
            
            # Ticker is valid if we got some historical data
            return not hist.empty
            
        except Exception as e:
            logger.debug(f"Ticker validation failed for {ticker}: {e}")
            return False
