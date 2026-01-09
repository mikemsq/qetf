"""Tests for yfinance data provider.

This module tests the YFinanceProvider class to ensure it correctly fetches
and validates ETF price data from Yahoo Finance.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from src.quantetf.data.providers.yfinance_provider import YFinanceProvider
from src.quantetf.utils.validation import (
    validate_ohlcv_dataframe,
    detect_price_anomalies,
    validate_date_range
)


class TestYFinanceProvider:
    """Test suite for YFinanceProvider class."""
    
    def test_initialization(self):
        """Test provider initialization with default parameters."""
        try:
            provider = YFinanceProvider()
            assert provider.auto_adjust == True
            assert provider.progress == False
        except ImportError:
            pytest.skip("yfinance not installed")
    
    def test_initialization_with_params(self):
        """Test provider initialization with custom parameters."""
        try:
            provider = YFinanceProvider(auto_adjust=False, progress=True)
            assert provider.auto_adjust == False
            assert provider.progress == True
        except ImportError:
            pytest.skip("yfinance not installed")
    
    def test_fetch_prices_invalid_date_format(self):
        """Test that invalid date format raises ValueError."""
        try:
            provider = YFinanceProvider()
            
            with pytest.raises(ValueError, match="Invalid date format"):
                provider.fetch_prices(['SPY'], '2020/01/01', '2021-01-01')
            
            with pytest.raises(ValueError, match="Invalid date format"):
                provider.fetch_prices(['SPY'], '2020-01-01', 'invalid-date')
        except ImportError:
            pytest.skip("yfinance not installed")
    
    def test_fetch_prices_empty_tickers(self):
        """Test that empty tickers list raises ValueError."""
        try:
            provider = YFinanceProvider()
            
            with pytest.raises(ValueError, match="Tickers list cannot be empty"):
                provider.fetch_prices([], '2020-01-01', '2021-01-01')
        except ImportError:
            pytest.skip("yfinance not installed")
    
    @pytest.mark.integration
    def test_fetch_prices_single_ticker(self, short_date_range):
        """Test fetching data for a single ticker.
        
        Note: This is an integration test that makes real API calls.
        """
        try:
            provider = YFinanceProvider()
            start, end = short_date_range
            
            data = provider.fetch_prices(['SPY'], start, end)
            
            assert data is not None
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            
            # Check required columns exist
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            assert all(col in data.columns for col in expected_cols)
            
            # Check datetime index
            assert isinstance(data.index, pd.DatetimeIndex)
            
            # Validate data quality
            assert validate_ohlcv_dataframe(data, 'SPY')
        except ImportError:
            pytest.skip("yfinance not installed")
    
    @pytest.mark.integration
    def test_fetch_prices_multiple_tickers(self):
        """Test fetching data for multiple tickers.
        
        Note: This is an integration test that makes real API calls.
        """
        try:
            provider = YFinanceProvider()
            tickers = ['SPY', 'QQQ']
            
            data = provider.fetch_prices(tickers, '2020-01-01', '2020-03-31')
            
            assert data is not None
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            
            # For multiple tickers, should have MultiIndex columns
            assert isinstance(data.columns, pd.MultiIndex)
            
            # Check datetime index
            assert isinstance(data.index, pd.DatetimeIndex)
        except ImportError:
            pytest.skip("yfinance not installed")
    
    @pytest.mark.integration
    def test_fetch_prices_invalid_ticker(self):
        """Test fetching data for invalid ticker returns None or empty."""
        try:
            provider = YFinanceProvider()
            
            # Use obviously invalid ticker
            data = provider.fetch_prices(
                ['THISISNOTAVALIDTICKER123456'],
                '2020-01-01',
                '2020-12-31'
            )
            
            # Should return None or empty DataFrame for invalid ticker
            assert data is None or data.empty
        except ImportError:
            pytest.skip("yfinance not installed")
    
    @pytest.mark.integration
    def test_get_ticker_info_valid(self):
        """Test getting ticker info for valid ETF."""
        try:
            provider = YFinanceProvider()
            
            info = provider.get_ticker_info('SPY')
            
            assert info is not None
            assert isinstance(info, dict)
            assert 'symbol' in info
            assert info['symbol'] == 'SPY'
        except ImportError:
            pytest.skip("yfinance not installed")
    
    @pytest.mark.integration  
    def test_get_ticker_info_invalid(self):
        """Test getting ticker info for invalid ticker."""
        try:
            provider = YFinanceProvider()
            
            info = provider.get_ticker_info('INVALIDTICKER123456')
            
            # Should return None for invalid ticker
            assert info is None
        except ImportError:
            pytest.skip("yfinance not installed")
    
    @pytest.mark.integration
    def test_validate_ticker_valid(self):
        """Test validating a valid ticker."""
        try:
            provider = YFinanceProvider()
            
            is_valid = provider.validate_ticker('SPY')
            
            assert is_valid == True
        except ImportError:
            pytest.skip("yfinance not installed")
    
    @pytest.mark.integration
    def test_validate_ticker_invalid(self):
        """Test validating an invalid ticker."""
        try:
            provider = YFinanceProvider()
            
            is_valid = provider.validate_ticker('INVALIDTICKER123456')
            
            assert is_valid == False
        except ImportError:
            pytest.skip("yfinance not installed")
    
    def test_fetch_prices_with_mock(self, sample_price_data):
        """Test fetch_prices with mocked yfinance download."""
        try:
            provider = YFinanceProvider()
            
            with patch('src.quantetf.data.providers.yfinance_provider.yf.download') as mock_download:
                mock_download.return_value = sample_price_data
                
                data = provider.fetch_prices(['SPY'], '2020-01-01', '2020-12-31')
                
                assert data is not None
                assert not data.empty
                mock_download.assert_called_once()
        except ImportError:
            pytest.skip("yfinance not installed")


class TestValidationUtilities:
    """Test suite for validation utility functions."""
    
    def test_validate_ohlcv_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_ohlcv_dataframe(empty_df, 'TEST')
    
    def test_validate_ohlcv_missing_columns(self, sample_price_data):
        """Test validation fails when required columns are missing."""
        # Remove Close column
        df = sample_price_data.drop('Close', axis=1)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv_dataframe(df, 'TEST')
    
    def test_validate_ohlcv_negative_prices(self, sample_price_data):
        """Test validation fails for negative prices."""
        df = sample_price_data.copy()
        df.loc[df.index[0], 'Close'] = -10  # Introduce negative price
        
        with pytest.raises(ValueError, match="Negative prices"):
            validate_ohlcv_dataframe(df, 'TEST')
    
    def test_validate_ohlcv_invalid_ohlc_relationship(self, sample_price_data):
        """Test validation fails for invalid OHLC relationships."""
        df = sample_price_data.copy()
        # Make High less than Low (invalid)
        df.loc[df.index[0], 'High'] = df.loc[df.index[0], 'Low'] - 1
        
        with pytest.raises(ValueError, match="Invalid OHLC relationships"):
            validate_ohlcv_dataframe(df, 'TEST')
    
    def test_validate_ohlcv_excessive_missing_data(self, sample_price_data):
        """Test validation fails for excessive missing data."""
        df = sample_price_data.copy()
        # Make 10% of data NaN (exceeds 5% threshold)
        df.iloc[:int(len(df)*0.1), :] = pd.NA
        
        with pytest.raises(ValueError, match="Excessive missing data"):
            validate_ohlcv_dataframe(df, 'TEST')
    
    def test_validate_ohlcv_valid_data(self, sample_price_data):
        """Test validation passes for valid data."""
        result = validate_ohlcv_dataframe(sample_price_data, 'TEST')
        assert result == True
    
    def test_detect_price_anomalies_clean_data(self, sample_price_data):
        """Test anomaly detection on clean data."""
        anomalies = detect_price_anomalies(sample_price_data, 'TEST')
        
        assert anomalies['suspicious_days'] == 0
        assert anomalies['extreme_changes'] == 0
        assert anomalies['zero_volume'] == 0
    
    def test_detect_price_anomalies_suspicious_day(self, sample_price_data):
        """Test detection of suspicious day (OHLC all equal)."""
        df = sample_price_data.copy()
        # Make OHLC all equal for one day
        df.loc[df.index[0], ['Open', 'High', 'Low', 'Close']] = 100
        
        anomalies = detect_price_anomalies(df, 'TEST')
        
        assert anomalies['suspicious_days'] == 1
    
    def test_detect_price_anomalies_extreme_change(self, sample_price_data):
        """Test detection of extreme price changes."""
        df = sample_price_data.copy()
        # Create extreme price change (>50%)
        df.loc[df.index[1], 'Close'] = df.loc[df.index[0], 'Close'] * 1.6
        
        anomalies = detect_price_anomalies(df, 'TEST', max_daily_change=0.5)
        
        assert anomalies['extreme_changes'] >= 1
    
    def test_detect_price_anomalies_zero_volume(self, sample_price_data):
        """Test detection of zero volume days."""
        df = sample_price_data.copy()
        df.loc[df.index[0], 'Volume'] = 0
        
        anomalies = detect_price_anomalies(df, 'TEST')
        
        assert anomalies['zero_volume'] == 1
    
    def test_validate_date_range_non_datetime_index(self):
        """Test validation fails for non-datetime index."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="must be DatetimeIndex"):
            validate_date_range(df, '2020-01-01', '2020-12-31')
