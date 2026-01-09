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
        df: DataFrame to validate with OHLCV data
        ticker: Ticker symbol for error messages (optional)
        required_cols: List of required column names (default: OHLCV)
        
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
    
    # Check required columns exist
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")
    
    # Check for negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    available_price_cols = [col for col in price_cols if col in df.columns]
    
    if (df[available_price_cols] < 0).any().any():
        raise ValueError(f"Negative prices detected for {ticker}")
    
    # Check OHLC relationships
    if all(col in df.columns for col in price_cols):
        # High should be >= all other prices
        high_invalid = ((df['High'] < df['Low']) |
                       (df['High'] < df['Open']) |
                       (df['High'] < df['Close'])).any()
        
        # Low should be <= all other prices  
        low_invalid = ((df['Low'] > df['Open']) |
                      (df['Low'] > df['Close'])).any()
        
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
        df: DataFrame with price data (must have 'Close' column)
        ticker: Ticker symbol for reporting
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
    
    # Check for suspicious days (OHLC all equal)
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        mask = ((df['Open'] == df['High']) & 
                (df['Open'] == df['Low']) & 
                (df['Open'] == df['Close']))
        issues['suspicious_days'] = mask.sum()
    
    # Check for extreme price changes
    if 'Close' in df.columns:
        returns = df['Close'].pct_change()
        issues['extreme_changes'] = (abs(returns) > max_daily_change).sum()
    
    # Check for zero volume days
    if 'Volume' in df.columns:
        issues['zero_volume'] = (df['Volume'] == 0).sum()
    
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
