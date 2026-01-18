"""Tests for FREDMacroAccessor."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from quantetf.data.access.fred_macro import FREDMacroAccessor
from quantetf.data.access.types import Regime


@pytest.fixture
def mock_macro_loader():
    """Create a mock MacroDataLoader."""
    mock_loader = MagicMock()
    
    # Mock get_lookback to return a series with VIX data
    def get_lookback_side_effect(indicator, as_of, lookback_days, column=None):
        # Return mock VIX or spread data
        dates = pd.date_range(as_of - pd.Timedelta(days=lookback_days), as_of, freq='D')
        if 'VIX' in str(indicator):
            # Return VIX-like data (25-35)
            values = np.linspace(25, 30, len(dates))
        elif 'T10Y2Y' in str(indicator) or 'SPREAD' in str(indicator):
            # Return spread-like data (-0.2 to 0.5)
            values = np.linspace(-0.2, 0.5, len(dates))
        else:
            values = np.random.randn(len(dates))
        
        return pd.Series(values, index=dates)
    
    # Mock get_series similarly
    def get_series_side_effect(indicator, start, end, column=None):
        dates = pd.date_range(start, end, freq='D')
        if 'VIX' in str(indicator):
            values = np.linspace(25, 30, len(dates))
        elif 'T10Y2Y' in str(indicator) or 'SPREAD' in str(indicator):
            values = np.linspace(-0.2, 0.5, len(dates))
        else:
            values = np.random.randn(len(dates))
        
        return pd.Series(values, index=dates)
    
    # Mock _load_dataframe
    def load_dataframe_side_effect(indicator_value):
        # Return a dataframe with mock data
        dates = pd.date_range('2023-01-01', '2024-01-31', freq='D')
        if 'VIX' in indicator_value:
            values = np.linspace(15, 35, len(dates))
        elif 'T10Y2Y' in indicator_value or 'SPREAD' in indicator_value:
            values = np.linspace(-0.5, 1.0, len(dates))
        else:
            values = np.random.randn(len(dates))
        
        df = pd.DataFrame(values, index=dates, columns=[indicator_value])
        return df
    
    mock_loader.get_lookback.side_effect = get_lookback_side_effect
    mock_loader.get_series.side_effect = get_series_side_effect
    mock_loader._load_dataframe.side_effect = load_dataframe_side_effect
    
    return mock_loader


@pytest.fixture
def macro_accessor(mock_macro_loader):
    """Create FREDMacroAccessor with mocked loader."""
    return FREDMacroAccessor(mock_macro_loader)


class TestFREDMacroAccessorInitialization:
    """Test FREDMacroAccessor initialization."""
    
    def test_initialization(self, mock_macro_loader):
        """Test accessor initialization with MacroDataLoader."""
        accessor = FREDMacroAccessor(mock_macro_loader)
        assert accessor.macro_loader is mock_macro_loader


class TestReadMacroIndicator:
    """Test read_macro_indicator method."""
    
    def test_read_vix_with_lookback(self, macro_accessor):
        """Test reading VIX indicator with lookback."""
        as_of = pd.Timestamp('2024-01-31')
        result = macro_accessor.read_macro_indicator('VIX', as_of, lookback_days=20)
        
        assert isinstance(result, pd.DataFrame)
        assert 'VIX' in result.columns
        assert not result.empty
    
    def test_read_spread_with_lookback(self, macro_accessor):
        """Test reading yield spread with lookback."""
        as_of = pd.Timestamp('2024-01-31')
        result = macro_accessor.read_macro_indicator('T10Y2Y', as_of, lookback_days=20)
        
        assert isinstance(result, pd.DataFrame)
        assert 'T10Y2Y' in result.columns
        assert not result.empty
    
    def test_read_indicator_without_lookback(self, macro_accessor):
        """Test reading indicator without lookback window."""
        as_of = pd.Timestamp('2024-01-31')
        result = macro_accessor.read_macro_indicator('VIX', as_of)
        
        assert isinstance(result, pd.DataFrame)
        assert 'VIX' in result.columns
        assert not result.empty
    
    def test_read_invalid_indicator_raises_error(self, macro_accessor):
        """Test that invalid indicator raises ValueError."""
        as_of = pd.Timestamp('2024-01-31')
        
        with pytest.raises(ValueError, match="Indicator not found"):
            macro_accessor.read_macro_indicator('INVALID_IND', as_of)
    
    def test_read_indicator_returns_dataframe(self, macro_accessor):
        """Test that returned data is DataFrame with correct format."""
        as_of = pd.Timestamp('2024-01-31')
        result = macro_accessor.read_macro_indicator('VIX', as_of, lookback_days=10)
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result.columns) == 1


class TestGetRegime:
    """Test get_regime method."""
    
    def test_regime_risk_on(self, macro_accessor, mock_macro_loader):
        """Test RISK_ON regime detection."""
        # Mock low VIX and positive spread
        as_of = pd.Timestamp('2024-01-31')
        
        def get_lookback_vix(indicator, as_of_val, lookback_days, column=None):
            # Low VIX (< 20)
            dates = pd.date_range(as_of_val - pd.Timedelta(days=lookback_days), as_of_val, freq='D')
            return pd.Series(np.full(len(dates), 15), index=dates)
        
        def get_lookback_spread(indicator, as_of_val, lookback_days, column=None):
            # Positive spread (> 0)
            dates = pd.date_range(as_of_val - pd.Timedelta(days=lookback_days), as_of_val, freq='D')
            return pd.Series(np.full(len(dates), 0.5), index=dates)
        
        # Set up side effect to return different values for VIX vs spread
        call_count = [0]
        def side_effect_wrapper(indicator, as_of_val, lookback_days, column=None):
            if 'VIX' in str(indicator):
                return get_lookback_vix(indicator, as_of_val, lookback_days, column)
            else:
                return get_lookback_spread(indicator, as_of_val, lookback_days, column)
        
        mock_macro_loader.get_lookback.side_effect = side_effect_wrapper
        
        regime = macro_accessor.get_regime(as_of)
        assert regime == Regime.RISK_ON
    
    def test_regime_elevated_vol_from_spread(self, macro_accessor, mock_macro_loader):
        """Test ELEVATED_VOL regime from negative spread."""
        as_of = pd.Timestamp('2024-01-31')
        
        def side_effect_wrapper(indicator, as_of_val, lookback_days, column=None):
            dates = pd.date_range(as_of_val - pd.Timedelta(days=lookback_days), as_of_val, freq='D')
            if 'VIX' in str(indicator):
                # Low VIX
                return pd.Series(np.full(len(dates), 18), index=dates)
            else:
                # Negative spread
                return pd.Series(np.full(len(dates), -0.1), index=dates)
        
        mock_macro_loader.get_lookback.side_effect = side_effect_wrapper
        
        regime = macro_accessor.get_regime(as_of)
        assert regime == Regime.ELEVATED_VOL
    
    def test_regime_elevated_vol_from_vix(self, macro_accessor, mock_macro_loader):
        """Test ELEVATED_VOL regime from elevated VIX."""
        as_of = pd.Timestamp('2024-01-31')
        
        def side_effect_wrapper(indicator, as_of_val, lookback_days, column=None):
            dates = pd.date_range(as_of_val - pd.Timedelta(days=lookback_days), as_of_val, freq='D')
            if 'VIX' in str(indicator):
                # Elevated VIX (20-30)
                return pd.Series(np.full(len(dates), 25), index=dates)
            else:
                # Positive spread
                return pd.Series(np.full(len(dates), 0.5), index=dates)
        
        mock_macro_loader.get_lookback.side_effect = side_effect_wrapper
        
        regime = macro_accessor.get_regime(as_of)
        assert regime == Regime.ELEVATED_VOL
    
    def test_regime_high_vol(self, macro_accessor, mock_macro_loader):
        """Test HIGH_VOL regime detection."""
        as_of = pd.Timestamp('2024-01-31')
        
        def side_effect_wrapper(indicator, as_of_val, lookback_days, column=None):
            dates = pd.date_range(as_of_val - pd.Timedelta(days=lookback_days), as_of_val, freq='D')
            if 'VIX' in str(indicator):
                # High VIX (> 30)
                return pd.Series(np.full(len(dates), 35), index=dates)
            else:
                # Positive spread (not negative)
                return pd.Series(np.full(len(dates), 0.3), index=dates)
        
        mock_macro_loader.get_lookback.side_effect = side_effect_wrapper
        
        regime = macro_accessor.get_regime(as_of)
        assert regime == Regime.HIGH_VOL
    
    def test_regime_recession_warning(self, macro_accessor, mock_macro_loader):
        """Test RECESSION_WARNING regime detection."""
        as_of = pd.Timestamp('2024-01-31')
        
        def side_effect_wrapper(indicator, as_of_val, lookback_days, column=None):
            dates = pd.date_range(as_of_val - pd.Timedelta(days=lookback_days), as_of_val, freq='D')
            if 'VIX' in str(indicator):
                # High VIX (> 30)
                return pd.Series(np.full(len(dates), 35), index=dates)
            else:
                # Negative spread (< -0.5)
                return pd.Series(np.full(len(dates), -0.6), index=dates)
        
        mock_macro_loader.get_lookback.side_effect = side_effect_wrapper
        
        regime = macro_accessor.get_regime(as_of)
        assert regime == Regime.RECESSION_WARNING
    
    def test_regime_unknown_on_error(self, macro_accessor, mock_macro_loader):
        """Test UNKNOWN regime when data unavailable."""
        as_of = pd.Timestamp('2024-01-31')
        
        # Make loader raise an exception
        mock_macro_loader.get_lookback.side_effect = ValueError("No data")
        
        regime = macro_accessor.get_regime(as_of)
        assert regime == Regime.UNKNOWN
    
    def test_regime_returns_regime_enum(self, macro_accessor):
        """Test that get_regime always returns Regime enum."""
        as_of = pd.Timestamp('2024-01-31')
        
        regime = macro_accessor.get_regime(as_of)
        assert isinstance(regime, Regime)
        assert regime in [
            Regime.RISK_ON,
            Regime.ELEVATED_VOL,
            Regime.HIGH_VOL,
            Regime.RECESSION_WARNING,
            Regime.UNKNOWN,
        ]


class TestGetAvailableIndicators:
    """Test get_available_indicators method."""
    
    def test_available_indicators_returns_list(self, macro_accessor):
        """Test that available indicators returns non-empty list."""
        indicators = macro_accessor.get_available_indicators()
        
        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert all(isinstance(ind, str) for ind in indicators)
    
    def test_available_indicators_includes_vix(self, macro_accessor):
        """Test that VIX is in available indicators."""
        indicators = macro_accessor.get_available_indicators()
        
        assert 'VIX' in indicators
    
    def test_available_indicators_includes_spreads(self, macro_accessor):
        """Test that yield spreads are available."""
        indicators = macro_accessor.get_available_indicators()
        
        # Should have some spread indicators
        spread_indicators = [ind for ind in indicators if 'Y' in ind or '10Y' in ind or 'SPREAD' in ind]
        assert len(spread_indicators) > 0


class TestFactoryIntegration:
    """Test integration with DataAccessFactory."""
    
    def test_factory_creates_macro_accessor(self, mock_macro_loader):
        """Test that factory can create FREDMacroAccessor."""
        from quantetf.data.access.factory import DataAccessFactory
        
        with patch('quantetf.data.macro_loader.MacroDataLoader', return_value=mock_macro_loader):
            accessor = DataAccessFactory.create_macro_accessor(source="fred")
        
        assert isinstance(accessor, FREDMacroAccessor)
    
    def test_factory_with_custom_data_dir(self, mock_macro_loader):
        """Test factory with custom data directory."""
        from quantetf.data.access.factory import DataAccessFactory
        
        with patch('quantetf.data.macro_loader.MacroDataLoader') as MockLoader:
            MockLoader.return_value = mock_macro_loader
            
            accessor = DataAccessFactory.create_macro_accessor(
                source="fred",
                config={"data_dir": "custom/path"}
            )
            
            assert isinstance(accessor, FREDMacroAccessor)
            # Verify constructor was called with data_dir
            MockLoader.assert_called_with(data_dir=Path("custom/path"))


class TestRegimeBoundaryConditions:
    """Test regime detection edge cases."""
    
    def test_regime_vix_exactly_30(self, macro_accessor, mock_macro_loader):
        """Test regime when VIX = 30 exactly."""
        as_of = pd.Timestamp('2024-01-31')
        
        def side_effect_wrapper(indicator, as_of_val, lookback_days, column=None):
            dates = pd.date_range(as_of_val - pd.Timedelta(days=lookback_days), as_of_val, freq='D')
            if 'VIX' in str(indicator):
                # VIX exactly at boundary
                return pd.Series(np.full(len(dates), 30.0), index=dates)
            else:
                # Positive spread
                return pd.Series(np.full(len(dates), 0.5), index=dates)
        
        mock_macro_loader.get_lookback.side_effect = side_effect_wrapper
        
        regime = macro_accessor.get_regime(as_of)
        # At 30, should be ELEVATED_VOL (> 30 needed for HIGH_VOL)
        assert regime == Regime.ELEVATED_VOL
    
    def test_regime_spread_exactly_0(self, macro_accessor, mock_macro_loader):
        """Test regime when spread = 0 exactly."""
        as_of = pd.Timestamp('2024-01-31')
        
        def side_effect_wrapper(indicator, as_of_val, lookback_days, column=None):
            dates = pd.date_range(as_of_val - pd.Timedelta(days=lookback_days), as_of_val, freq='D')
            if 'VIX' in str(indicator):
                # Low VIX
                return pd.Series(np.full(len(dates), 15), index=dates)
            else:
                # Spread exactly at 0 (boundary)
                return pd.Series(np.full(len(dates), 0.0), index=dates)
        
        mock_macro_loader.get_lookback.side_effect = side_effect_wrapper
        
        regime = macro_accessor.get_regime(as_of)
        # At 0, should be RISK_ON (not ELEVATED_VOL)
        assert regime == Regime.RISK_ON


class TestIndicatorNames:
    """Test indicator name handling."""
    
    def test_vix_by_enum_name(self, macro_accessor):
        """Test reading VIX using enum name."""
        as_of = pd.Timestamp('2024-01-31')
        result = macro_accessor.read_macro_indicator('VIX', as_of, lookback_days=10)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    
    def test_spread_by_enum_value(self, macro_accessor):
        """Test reading spread using enum value."""
        as_of = pd.Timestamp('2024-01-31')
        result = macro_accessor.read_macro_indicator('T10Y2Y', as_of, lookback_days=10)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
