"""Validation utilities for QuantETF data quality checks.

This module provides functions to validate ETF price data and ensure
data quality before using in backtests or calculations.
"""

from typing import List
import pandas as pd
import numpy as np


def validate_ohlcv_dataframe(
    df: pd.DataFrame,
    ticker: str = "Unknown",
    required_cols: List[str] = None
) -> bool:
    """Validate OHLCV DataFrame structure and data quality.

    Performs comprehensive validation including:
    - Presence of required columns
    - No negative prices
    - Valid OHLC relationships (High >= Low, etc.)
    - Reasonable data completeness

    Args:
        df: DataFrame with MultiIndex columns (Ticker, Price_Field)
        ticker: Ticker symbol to validate (used to select from MultiIndex)
        required_cols: List of required price field names (default: OHLCV)

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails with specific error message

    Example:
        >>> data = fetch_prices(['SPY'], '2020-01-01', '2021-01-01')
        >>> validate_ohlcv_dataframe(data, 'SPY')
        True
    """
    if required_cols is None:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Check DataFrame is not empty
    if df.empty:
        raise ValueError(f"DataFrame is empty for {ticker}")

    # Expect MultiIndex columns with format (Ticker, Price_Field)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            f"Expected MultiIndex columns with format (Ticker, Price_Field), "
            f"but got simple columns: {df.columns.tolist()}"
        )

    # Get the price field level (should be level 1 with name 'Price')
    price_level = 1 if df.columns.names[0] == 'Ticker' else 0
    available_cols = set(df.columns.get_level_values(price_level))

    # Check required columns exist
    missing_cols = set(required_cols) - available_cols
    if missing_cols:
        raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")

    # Check if the specific ticker exists in the DataFrame
    ticker_level = 0 if df.columns.names[0] == 'Ticker' else 1
    available_tickers = df.columns.get_level_values(ticker_level).unique().tolist()

    if ticker not in available_tickers:
        raise ValueError(
            f"Ticker '{ticker}' not found in DataFrame. "
            f"Available tickers: {available_tickers}"
        )

    # Extract data for the specific ticker
    ticker_data = df[ticker]

    # Get OHLC data
    open_data = ticker_data.get('Open')
    high_data = ticker_data.get('High')
    low_data = ticker_data.get('Low')
    close_data = ticker_data.get('Close')

    # Check for negative prices
    price_cols_data = [d for d in [open_data, high_data, low_data, close_data] if d is not None]
    for price_data in price_cols_data:
        if (price_data < 0).any():
            raise ValueError(f"Negative prices detected for {ticker}")

    # Check OHLC relationships
    if all(d is not None for d in [open_data, high_data, low_data, close_data]):
        # High should be >= all other prices
        high_invalid = ((high_data < low_data) |
                       (high_data < open_data) |
                       (high_data < close_data)).any()

        # Low should be <= all other prices
        low_invalid = ((low_data > open_data) |
                      (low_data > close_data)).any()

        if high_invalid or low_invalid:
            raise ValueError(f"Invalid OHLC relationships detected for {ticker}")
    
    # Check for excessive missing data
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_pct = missing_cells / total_cells if total_cells > 0 else 0
    
    if missing_pct > 0.05:  # More than 5% missing
        raise ValueError(
            f"Excessive missing data ({missing_pct:.1%}) for {ticker}. "
            f"Found {missing_cells} missing values out of {total_cells} cells."
        )
    
    return True


def detect_price_anomalies(
    df: pd.DataFrame,
    ticker: str = "Unknown",
    max_daily_change: float = 0.5
) -> dict:
    """Detect potential anomalies in price data.

    Args:
        df: DataFrame with MultiIndex columns (Ticker, Price_Field)
        ticker: Ticker symbol to analyze
        max_daily_change: Maximum reasonable daily price change (default: 50%)

    Returns:
        Dictionary with anomaly counts:
        - suspicious_days: Days where OHLC are all equal
        - extreme_changes: Days with price changes > max_daily_change
        - zero_volume: Days with zero trading volume

    Example:
        >>> anomalies = detect_price_anomalies(data, 'SPY')
        >>> if anomalies['extreme_changes'] > 0:
        ...     print(f"Warning: {anomalies['extreme_changes']} extreme price changes")
    """
    issues = {
        'suspicious_days': 0,
        'extreme_changes': 0,
        'zero_volume': 0
    }

    # Expect MultiIndex columns
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            f"Expected MultiIndex columns with format (Ticker, Price_Field), "
            f"but got simple columns: {df.columns.tolist()}"
        )

    # Check if ticker exists
    ticker_level = 0 if df.columns.names[0] == 'Ticker' else 1
    available_tickers = df.columns.get_level_values(ticker_level).unique().tolist()

    if ticker not in available_tickers:
        raise ValueError(
            f"Ticker '{ticker}' not found in DataFrame. "
            f"Available tickers: {available_tickers}"
        )

    # Extract data for the specific ticker
    ticker_data = df[ticker]

    # Get OHLC and volume data
    open_data = ticker_data.get('Open')
    high_data = ticker_data.get('High')
    low_data = ticker_data.get('Low')
    close_data = ticker_data.get('Close')
    volume_data = ticker_data.get('Volume')

    # Check for suspicious days (OHLC all equal)
    if all(d is not None for d in [open_data, high_data, low_data, close_data]):
        mask = ((open_data == high_data) &
                (open_data == low_data) &
                (open_data == close_data))
        issues['suspicious_days'] = mask.sum()

    # Check for extreme price changes
    if close_data is not None:
        returns = close_data.pct_change(fill_method=None)
        issues['extreme_changes'] = (abs(returns) > max_daily_change).sum()

    # Check for zero volume days
    if volume_data is not None:
        issues['zero_volume'] = (volume_data == 0).sum()

    return issues


def validate_date_range(
    df: pd.DataFrame,
    expected_start: str,
    expected_end: str,
    tolerance_days: int = 5
) -> bool:
    """Validate that DataFrame covers expected date range.
    
    Args:
        df: DataFrame with datetime index
        expected_start: Expected start date (YYYY-MM-DD)
        expected_end: Expected end date (YYYY-MM-DD)
        tolerance_days: Allowed difference in days (for weekends/holidays)
        
    Returns:
        True if date range is approximately correct
        
    Raises:
        ValueError: If date range is significantly different than expected
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    actual_start = df.index.min()
    actual_end = df.index.max()
    
    expected_start_dt = pd.to_datetime(expected_start)
    expected_end_dt = pd.to_datetime(expected_end)
    
    start_diff = abs((actual_start - expected_start_dt).days)
    end_diff = abs((actual_end - expected_end_dt).days)
    
    if start_diff > tolerance_days:
        raise ValueError(
            f"Start date mismatch: expected {expected_start}, "
            f"got {actual_start.strftime('%Y-%m-%d')} "
            f"(difference: {start_diff} days)"
        )
    
    if end_diff > tolerance_days:
        raise ValueError(
            f"End date mismatch: expected {expected_end}, "
            f"got {actual_end.strftime('%Y-%m-%d')} "
            f"(difference: {end_diff} days)"
        )
    
    return True
