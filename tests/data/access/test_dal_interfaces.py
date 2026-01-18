"""Tests for DAL core interfaces and types."""

import pytest
import pandas as pd
from datetime import datetime
from typing import Optional

from quantetf.data.access import (
    PriceDataAccessor,
    MacroDataAccessor,
    UniverseDataAccessor,
    ReferenceDataAccessor,
    DataAccessContext,
    DataAccessFactory,
    Regime,
    TickerMetadata,
    ExchangeInfo,
    DataAccessMetadata,
)


class TestRegimeEnum:
    """Test Regime enum values and behavior."""
    
    def test_regime_values_defined(self):
        """Test that all expected regime values are defined."""
        assert Regime.RISK_ON.value == "risk_on"
        assert Regime.ELEVATED_VOL.value == "elevated_vol"
        assert Regime.HIGH_VOL.value == "high_vol"
        assert Regime.RECESSION_WARNING.value == "recession_warning"
        assert Regime.UNKNOWN.value == "unknown"
    
    def test_regime_all_members(self):
        """Test that all expected regime members exist."""
        members = {r.name for r in Regime}
        expected = {"RISK_ON", "ELEVATED_VOL", "HIGH_VOL", "RECESSION_WARNING", "UNKNOWN"}
        assert members == expected


class TestTickerMetadata:
    """Test TickerMetadata dataclass."""
    
    def test_ticker_metadata_creation(self):
        """Test creating a TickerMetadata instance."""
        meta = TickerMetadata(
            ticker="SPY",
            name="SPDR S&P 500 ETF",
            sector="Equities",
            exchange="NYSE",
            currency="USD",
        )
        assert meta.ticker == "SPY"
        assert meta.name == "SPDR S&P 500 ETF"
        assert meta.sector == "Equities"
        assert meta.exchange == "NYSE"
        assert meta.currency == "USD"
    
    def test_ticker_metadata_frozen(self):
        """Test that TickerMetadata is frozen (immutable)."""
        meta = TickerMetadata(
            ticker="SPY",
            name="SPDR S&P 500 ETF",
            sector="Equities",
            exchange="NYSE",
            currency="USD",
        )
        with pytest.raises(AttributeError):
            meta.ticker = "QQQ"


class TestExchangeInfo:
    """Test ExchangeInfo dataclass."""
    
    def test_exchange_info_creation(self):
        """Test creating an ExchangeInfo instance."""
        info = ExchangeInfo(
            name="New York Stock Exchange",
            trading_hours="09:30-16:00 EST",
            timezone="US/Eastern",
            settlement_days=2,
        )
        assert info.name == "New York Stock Exchange"
        assert info.trading_hours == "09:30-16:00 EST"
        assert info.timezone == "US/Eastern"
        assert info.settlement_days == 2
    
    def test_exchange_info_frozen(self):
        """Test that ExchangeInfo is frozen (immutable)."""
        info = ExchangeInfo(
            name="NYSE",
            trading_hours="09:30-16:00 EST",
            timezone="US/Eastern",
            settlement_days=2,
        )
        with pytest.raises(AttributeError):
            info.name = "New York Stock Exchange"


class TestDataAccessMetadata:
    """Test DataAccessMetadata dataclass."""
    
    def test_data_access_metadata_creation(self):
        """Test creating a DataAccessMetadata instance."""
        ts = pd.Timestamp("2024-01-31")
        meta = DataAccessMetadata(
            source="snapshot",
            timestamp=ts,
            lookback_date=pd.Timestamp("2024-01-30"),
            data_quality_score=0.98,
        )
        assert meta.source == "snapshot"
        assert meta.timestamp == ts
        assert meta.lookback_date == pd.Timestamp("2024-01-30")
        assert meta.data_quality_score == 0.98
    
    def test_data_access_metadata_none_lookback(self):
        """Test DataAccessMetadata with None lookback_date."""
        ts = pd.Timestamp("2024-01-31")
        meta = DataAccessMetadata(
            source="snapshot",
            timestamp=ts,
            lookback_date=None,
            data_quality_score=0.95,
        )
        assert meta.lookback_date is None


class TestPriceDataAccessor:
    """Test PriceDataAccessor interface."""
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        methods = {
            "read_prices_as_of",
            "read_ohlcv_range",
            "get_latest_price_date",
            "validate_data_availability",
        }
        abstract_methods = {
            m for m in dir(PriceDataAccessor)
            if getattr(getattr(PriceDataAccessor, m), "__isabstractmethod__", False)
        }
        assert abstract_methods == methods
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            PriceDataAccessor()
    
    def test_subclass_must_implement_all_methods(self):
        """Test that subclass must implement all abstract methods."""
        class IncompletePriceAccessor(PriceDataAccessor):
            def read_prices_as_of(self, as_of, tickers=None, lookback_days=None):
                pass
            # Missing other methods
        
        with pytest.raises(TypeError):
            IncompletePriceAccessor()


class TestMacroDataAccessor:
    """Test MacroDataAccessor interface."""
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        methods = {
            "read_macro_indicator",
            "get_regime",
            "get_available_indicators",
        }
        abstract_methods = {
            m for m in dir(MacroDataAccessor)
            if getattr(getattr(MacroDataAccessor, m), "__isabstractmethod__", False)
        }
        assert abstract_methods == methods
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            MacroDataAccessor()


class TestUniverseDataAccessor:
    """Test UniverseDataAccessor interface."""
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        methods = {
            "get_universe",
            "get_universe_as_of",
            "list_available_universes",
        }
        abstract_methods = {
            m for m in dir(UniverseDataAccessor)
            if getattr(getattr(UniverseDataAccessor, m), "__isabstractmethod__", False)
        }
        assert abstract_methods == methods
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            UniverseDataAccessor()


class TestReferenceDataAccessor:
    """Test ReferenceDataAccessor interface."""
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        methods = {
            "get_ticker_info",
            "get_sector_mapping",
            "get_exchange_info",
        }
        abstract_methods = {
            m for m in dir(ReferenceDataAccessor)
            if getattr(getattr(ReferenceDataAccessor, m), "__isabstractmethod__", False)
        }
        assert abstract_methods == methods
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            ReferenceDataAccessor()


class TestDataAccessContext:
    """Test DataAccessContext."""
    
    def test_context_creation_with_mock_accessors(self):
        """Test creating context with mock accessors."""
        # Create minimal mock implementations
        class MockPrice(PriceDataAccessor):
            def read_prices_as_of(self, as_of, tickers=None, lookback_days=None):
                return pd.DataFrame()
            def read_ohlcv_range(self, start, end, tickers=None):
                return pd.DataFrame()
            def get_latest_price_date(self):
                return pd.Timestamp.now()
            def validate_data_availability(self, tickers, as_of):
                return {t: True for t in tickers}
        
        class MockMacro(MacroDataAccessor):
            def read_macro_indicator(self, indicator, as_of, lookback_days=None):
                return pd.DataFrame()
            def get_regime(self, as_of):
                return Regime.RISK_ON
            def get_available_indicators(self):
                return []
        
        class MockUniverse(UniverseDataAccessor):
            def get_universe(self, universe_name):
                return []
            def get_universe_as_of(self, universe_name, as_of):
                return []
            def list_available_universes(self):
                return []
        
        class MockReference(ReferenceDataAccessor):
            def get_ticker_info(self, ticker):
                raise ValueError("Not implemented in mock")
            def get_sector_mapping(self):
                return {}
            def get_exchange_info(self):
                return {}
        
        ctx = DataAccessContext(
            prices=MockPrice(),
            macro=MockMacro(),
            universes=MockUniverse(),
            references=MockReference(),
        )
        
        assert ctx.prices is not None
        assert ctx.macro is not None
        assert ctx.universes is not None
        assert ctx.references is not None
    
    def test_context_frozen(self):
        """Test that DataAccessContext is frozen (immutable)."""
        class MockPrice(PriceDataAccessor):
            def read_prices_as_of(self, as_of, tickers=None, lookback_days=None):
                return pd.DataFrame()
            def read_ohlcv_range(self, start, end, tickers=None):
                return pd.DataFrame()
            def get_latest_price_date(self):
                return pd.Timestamp.now()
            def validate_data_availability(self, tickers, as_of):
                return {}
        
        class MockMacro(MacroDataAccessor):
            def read_macro_indicator(self, indicator, as_of, lookback_days=None):
                return pd.DataFrame()
            def get_regime(self, as_of):
                return Regime.UNKNOWN
            def get_available_indicators(self):
                return []
        
        class MockUniverse(UniverseDataAccessor):
            def get_universe(self, universe_name):
                return []
            def get_universe_as_of(self, universe_name, as_of):
                return []
            def list_available_universes(self):
                return []
        
        class MockReference(ReferenceDataAccessor):
            def get_ticker_info(self, ticker):
                raise ValueError("Not implemented")
            def get_sector_mapping(self):
                return {}
            def get_exchange_info(self):
                return {}
        
        ctx = DataAccessContext(
            prices=MockPrice(),
            macro=MockMacro(),
            universes=MockUniverse(),
            references=MockReference(),
        )
        
        # Try to modify - should raise error
        with pytest.raises(AttributeError):
            ctx.prices = None


class TestDataAccessFactory:
    """Test DataAccessFactory."""
    
    def test_factory_methods_exist(self):
        """Test that factory has required methods."""
        assert hasattr(DataAccessFactory, "create_price_accessor")
        assert hasattr(DataAccessFactory, "create_macro_accessor")
        assert hasattr(DataAccessFactory, "create_universe_accessor")
        assert hasattr(DataAccessFactory, "create_reference_accessor")
        assert hasattr(DataAccessFactory, "create_context")
    
    def test_create_price_accessor_not_implemented(self):
        """Test that price accessor creation raises ValueError when config missing."""
        # Snapshot is implemented but requires config
        with pytest.raises(ValueError, match="snapshot_path"):
            DataAccessFactory.create_price_accessor(source="snapshot")
    
    def test_create_macro_accessor_not_implemented(self):
        """Test that macro accessor creation raises NotImplementedError for now."""
        with pytest.raises(NotImplementedError):
            DataAccessFactory.create_macro_accessor(source="fred")
    
    def test_create_universe_accessor_not_implemented(self):
        """Test that universe accessor creation raises NotImplementedError for now."""
        with pytest.raises(NotImplementedError):
            DataAccessFactory.create_universe_accessor()
    
    def test_create_reference_accessor_not_implemented(self):
        """Test that reference accessor creation raises NotImplementedError for now."""
        with pytest.raises(NotImplementedError):
            DataAccessFactory.create_reference_accessor()
    
    def test_create_context_not_implemented(self):
        """Test that context creation raises NotImplementedError for now."""
        with pytest.raises(NotImplementedError):
            DataAccessFactory.create_context()
    
    def test_create_price_accessor_invalid_source(self):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown price accessor source"):
            DataAccessFactory.create_price_accessor(source="invalid")
    
    def test_create_macro_accessor_invalid_source(self):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown macro accessor source"):
            DataAccessFactory.create_macro_accessor(source="invalid")


class TestPublicExports:
    """Test that all expected items are exported."""
    
    def test_package_exports_abstract_classes(self):
        """Test that abstract classes are exported."""
        from quantetf.data import access
        assert hasattr(access, "PriceDataAccessor")
        assert hasattr(access, "MacroDataAccessor")
        assert hasattr(access, "UniverseDataAccessor")
        assert hasattr(access, "ReferenceDataAccessor")
    
    def test_package_exports_context_and_factory(self):
        """Test that context and factory are exported."""
        from quantetf.data import access
        assert hasattr(access, "DataAccessContext")
        assert hasattr(access, "DataAccessFactory")
    
    def test_package_exports_types(self):
        """Test that types are exported."""
        from quantetf.data import access
        assert hasattr(access, "Regime")
        assert hasattr(access, "TickerMetadata")
        assert hasattr(access, "ExchangeInfo")
        assert hasattr(access, "DataAccessMetadata")
