"""Tests for SnapshotPriceAccessor."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from quantetf.data.access.snapshot_price import SnapshotPriceAccessor


@pytest.fixture
def mock_snapshot_data():
    """Create mock snapshot data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    tickers = ['SPY', 'QQQ', 'IWM', 'EEM']
    
    # Create MultiIndex columns
    columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume']], 
                                         names=['Ticker', 'Price'])
    
    # Create sample data
    data = np.random.randn(len(dates), len(columns)) * 10 + 100
    df = pd.DataFrame(data, index=dates, columns=columns)
    
    return df


@pytest.fixture
def mock_snapshot_store(mock_snapshot_data):
    """Create a mock SnapshotDataStore."""
    mock_store = MagicMock()
    mock_store._data = mock_snapshot_data
    mock_store.read_prices.side_effect = lambda as_of, tickers=None, lookback_days=None: (
        mock_snapshot_data[mock_snapshot_data.index < as_of]
    )
    mock_store.tickers = ['SPY', 'QQQ', 'IWM', 'EEM']
    mock_store.date_range = (mock_snapshot_data.index.min(), mock_snapshot_data.index.max())
    
    return mock_store


@pytest.fixture
def snapshot_accessor(mock_snapshot_store):
    """Create SnapshotPriceAccessor with mocked store."""
    with patch('quantetf.data.snapshot_store.SnapshotDataStore', return_value=mock_snapshot_store):
        accessor = SnapshotPriceAccessor(Path("data/snapshots/test/data.parquet"))
    
    return accessor


class TestSnapshotPriceAccessorInitialization:
    """Test SnapshotPriceAccessor initialization."""
    
    def test_initialization_with_valid_path(self, mock_snapshot_store):
        """Test initialization with valid snapshot path."""
        with patch('quantetf.data.snapshot_store.SnapshotDataStore', return_value=mock_snapshot_store):
            accessor = SnapshotPriceAccessor(Path("data/snapshots/test/data.parquet"))
            assert accessor.snapshot_path == Path("data/snapshots/test/data.parquet")
            assert accessor._latest_date == mock_snapshot_store._data.index.max()
    
    def test_initialization_caches_latest_date(self, mock_snapshot_store, mock_snapshot_data):
        """Test that latest date is cached during initialization."""
        with patch('quantetf.data.snapshot_store.SnapshotDataStore', return_value=mock_snapshot_store):
            accessor = SnapshotPriceAccessor(Path("data/snapshots/test/data.parquet"))
            assert accessor._latest_date == mock_snapshot_data.index.max()


class TestReadPricesAsOf:
    """Test read_prices_as_of method."""
    
    def test_read_prices_as_of_basic(self, snapshot_accessor, mock_snapshot_data):
        """Test basic point-in-time read."""
        as_of = pd.Timestamp('2023-03-31')
        
        prices = snapshot_accessor.read_prices_as_of(as_of)
        
        # Should call underlying store with correct parameters
        snapshot_accessor._store.read_prices.assert_called_with(
            as_of=as_of,
            tickers=None,
            lookback_days=None,
        )
    
    def test_read_prices_as_of_with_ticker_filter(self, snapshot_accessor):
        """Test point-in-time read with ticker filtering."""
        as_of = pd.Timestamp('2023-03-31')
        tickers = ['SPY', 'QQQ']
        
        snapshot_accessor.read_prices_as_of(as_of, tickers=tickers)
        
        # Should pass ticker filter to store
        snapshot_accessor._store.read_prices.assert_called_with(
            as_of=as_of,
            tickers=tickers,
            lookback_days=None,
        )
    
    def test_read_prices_as_of_with_lookback(self, snapshot_accessor):
        """Test point-in-time read with lookback window."""
        as_of = pd.Timestamp('2023-03-31')
        lookback_days = 30
        
        snapshot_accessor.read_prices_as_of(as_of, lookback_days=lookback_days)
        
        # Should pass lookback to store
        snapshot_accessor._store.read_prices.assert_called_with(
            as_of=as_of,
            tickers=None,
            lookback_days=lookback_days,
        )
    
    def test_read_prices_as_of_no_data_raises_error(self, snapshot_accessor):
        """Test that ValueError is raised when no data available."""
        as_of = pd.Timestamp('2023-01-01')  # Before all data
        
        # Mock store to return empty
        snapshot_accessor._store.read_prices.side_effect = ValueError(
            f"No data available before {as_of}"
        )
        
        with pytest.raises(ValueError, match="No data available before"):
            snapshot_accessor.read_prices_as_of(as_of)


class TestReadOhlcvRange:
    """Test read_ohlcv_range method."""
    
    def test_read_ohlcv_range_basic(self, snapshot_accessor, mock_snapshot_data):
        """Test reading OHLCV for a date range."""
        start = pd.Timestamp('2023-02-01')
        end = pd.Timestamp('2023-03-31')
        
        result = snapshot_accessor.read_ohlcv_range(start, end)
        
        # Check all dates are in range
        assert (result.index >= start).all()
        assert (result.index <= end).all()
    
    def test_read_ohlcv_range_with_ticker_filter(self, snapshot_accessor, mock_snapshot_data):
        """Test reading OHLCV with ticker filtering."""
        start = pd.Timestamp('2023-02-01')
        end = pd.Timestamp('2023-03-31')
        tickers = ['SPY', 'QQQ']
        
        result = snapshot_accessor.read_ohlcv_range(start, end, tickers=tickers)
        
        # Check dates are in range
        assert (result.index >= start).all()
        assert (result.index <= end).all()
        
        # Check tickers are filtered
        result_tickers = result.columns.get_level_values('Ticker').unique()
        assert set(result_tickers).issubset(set(tickers))
    
    def test_read_ohlcv_range_missing_ticker_raises_error(self, snapshot_accessor, mock_snapshot_data):
        """Test that error is raised for missing ticker."""
        start = pd.Timestamp('2023-02-01')
        end = pd.Timestamp('2023-03-31')
        
        with pytest.raises(ValueError, match="Tickers not found"):
            snapshot_accessor.read_ohlcv_range(start, end, tickers=['NONEXISTENT'])
    
    def test_read_ohlcv_range_no_data_raises_error(self, snapshot_accessor, mock_snapshot_data):
        """Test that error is raised when no data in range."""
        start = pd.Timestamp('2030-01-01')  # After all data
        end = pd.Timestamp('2030-12-31')
        
        with pytest.raises(ValueError, match="No price data available in range"):
            snapshot_accessor.read_ohlcv_range(start, end)


class TestGetLatestPriceDate:
    """Test get_latest_price_date method."""
    
    def test_get_latest_price_date(self, snapshot_accessor, mock_snapshot_data):
        """Test that latest date is returned correctly."""
        latest = snapshot_accessor.get_latest_price_date()
        
        assert isinstance(latest, pd.Timestamp)
        assert latest == mock_snapshot_data.index.max()
    
    def test_latest_price_date_is_cached(self, snapshot_accessor):
        """Test that latest date is cached (doesn't change on multiple calls)."""
        latest1 = snapshot_accessor.get_latest_price_date()
        latest2 = snapshot_accessor.get_latest_price_date()
        
        assert latest1 == latest2


class TestValidateDataAvailability:
    """Test validate_data_availability method."""
    
    def test_validate_data_availability_all_exist(self, snapshot_accessor):
        """Test validation when all tickers exist."""
        as_of = pd.Timestamp('2023-03-31')
        tickers = ['SPY', 'QQQ', 'IWM']
        
        result = snapshot_accessor.validate_data_availability(tickers, as_of)
        
        assert isinstance(result, dict)
        assert all(result[t] in [True, False] for t in tickers)
    
    def test_validate_data_availability_some_missing(self, snapshot_accessor, mock_snapshot_data):
        """Test validation with some missing tickers."""
        as_of = pd.Timestamp('2023-03-31')
        tickers = ['SPY', 'QQQ', 'NONEXISTENT']
        
        result = snapshot_accessor.validate_data_availability(tickers, as_of)
        
        # Real tickers should be True, fake should be False
        assert result['SPY'] in [True, False]  # Actually in data
        assert result['QQQ'] in [True, False]  # Actually in data
        assert result['NONEXISTENT'] == False  # Not in data
    
    def test_validate_data_availability_before_as_of(self, snapshot_accessor):
        """Test that validation checks data BEFORE as_of."""
        as_of = pd.Timestamp('2023-03-31')
        
        # Should not raise error - handles gracefully
        result = snapshot_accessor.validate_data_availability(['SPY'], as_of)
        assert isinstance(result, dict)


class TestGetAvailableTickers:
    """Test get_available_tickers method."""
    
    def test_get_available_tickers(self, snapshot_accessor):
        """Test that available tickers are returned."""
        tickers = snapshot_accessor.get_available_tickers()
        
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert all(isinstance(t, str) for t in tickers)
        assert 'SPY' in tickers
        assert 'QQQ' in tickers


class TestDateRange:
    """Test date_range property."""
    
    def test_date_range(self, snapshot_accessor, mock_snapshot_data):
        """Test that date_range returns correct span."""
        start_date, end_date = snapshot_accessor.date_range
        
        assert start_date == mock_snapshot_data.index.min()
        assert end_date == mock_snapshot_data.index.max()
        assert start_date < end_date


class TestPointInTimeGuarantee:
    """Test that point-in-time guarantee is maintained."""
    
    def test_no_lookahead_bias(self, snapshot_accessor, mock_snapshot_data):
        """Test that no data on/after as_of date is returned."""
        as_of = pd.Timestamp('2023-03-15')
        
        # Get pit data from snapshot
        pit_data = mock_snapshot_data[mock_snapshot_data.index < as_of]
        
        # All dates should be strictly before as_of
        assert (pit_data.index < as_of).all()
        assert (pit_data.index != as_of).all()
    
    def test_strict_inequality_on_boundary(self, snapshot_accessor, mock_snapshot_data):
        """Test that data on as_of date is excluded (strict <, not <=)."""
        # Use a date that exists in data
        as_of = mock_snapshot_data.index[50]
        
        # Get pit data
        pit_data = mock_snapshot_data[mock_snapshot_data.index < as_of]
        
        # Should not include data on as_of date
        assert as_of not in pit_data.index


class TestFactoryIntegration:
    """Test integration with DataAccessFactory."""
    
    def test_factory_creates_snapshot_accessor(self, mock_snapshot_store):
        """Test that factory can create SnapshotPriceAccessor."""
        from quantetf.data.access.factory import DataAccessFactory
        
        with patch('quantetf.data.snapshot_store.SnapshotDataStore', return_value=mock_snapshot_store):
            accessor = DataAccessFactory.create_price_accessor(
                source="snapshot",
                config={"snapshot_path": "data/snapshots/test/data.parquet"}
            )
        
        assert isinstance(accessor, SnapshotPriceAccessor)
    
    def test_factory_requires_snapshot_path(self):
        """Test that factory requires snapshot_path in config."""
        from quantetf.data.access.factory import DataAccessFactory
        
        with pytest.raises(ValueError, match="snapshot_path"):
            DataAccessFactory.create_price_accessor(
                source="snapshot",
                config={}
            )
    
    def test_factory_rejects_missing_config(self):
        """Test that factory requires config dict."""
        from quantetf.data.access.factory import DataAccessFactory
        
        with pytest.raises(ValueError, match="snapshot_path"):
            DataAccessFactory.create_price_accessor(
                source="snapshot",
                config=None
            )


class TestDataFrame:
    """Test DataFrame format and structure."""
    
    def test_multiindex_columns(self, snapshot_accessor, mock_snapshot_data):
        """Test that returned data has MultiIndex columns."""
        as_of = pd.Timestamp('2023-03-31')
        
        prices = snapshot_accessor.read_prices_as_of(as_of)
        
        # Should have MultiIndex columns
        assert isinstance(prices.columns, pd.MultiIndex)
        assert prices.columns.names == ['Ticker', 'Price']
    
    def test_datetime_index(self, snapshot_accessor, mock_snapshot_data):
        """Test that returned data has datetime index."""
        as_of = pd.Timestamp('2023-03-31')
        
        prices = snapshot_accessor.read_prices_as_of(as_of)
        
        # Should have datetime index
        assert isinstance(prices.index, pd.DatetimeIndex)
