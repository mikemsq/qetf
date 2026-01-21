"""Tests for StaticReferenceDataAccessor."""

import pytest
from pathlib import Path
import tempfile
import shutil
import yaml

from quantetf.data.access.reference import StaticReferenceDataAccessor
from quantetf.data.access.types import TickerMetadata, ExchangeInfo


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with test reference configs."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create tickers.yaml
    tickers_config = {
        "tickers": {
            "SPY": {
                "name": "SPDR S&P 500 ETF Trust",
                "sector": "Broad Market",
                "exchange": "NYSE",
                "currency": "USD",
            },
            "QQQ": {
                "name": "Invesco QQQ Trust",
                "sector": "Technology",
                "exchange": "NASDAQ",
                "currency": "USD",
            },
            "EFA": {
                "name": "iShares MSCI EAFE ETF",
                "sector": "International Developed",
                "exchange": "NYSE",
                "currency": "USD",
            },
            "TLT": {
                "name": "iShares 20+ Year Treasury Bond ETF",
                "sector": "Fixed Income",
                "exchange": "NASDAQ",
                "currency": "USD",
            },
            "GLD": {
                "name": "SPDR Gold Trust",
                "sector": "Commodities",
                "exchange": "NYSE",
                "currency": "USD",
            },
        }
    }
    with open(temp_dir / "tickers.yaml", "w") as f:
        yaml.safe_dump(tickers_config, f)

    # Create exchanges.yaml
    exchanges_config = {
        "exchanges": {
            "NYSE": {
                "name": "New York Stock Exchange",
                "trading_hours": "09:30-16:00",
                "timezone": "US/Eastern",
                "settlement_days": 2,
            },
            "NASDAQ": {
                "name": "NASDAQ Stock Market",
                "trading_hours": "09:30-16:00",
                "timezone": "US/Eastern",
                "settlement_days": 2,
            },
        }
    }
    with open(temp_dir / "exchanges.yaml", "w") as f:
        yaml.safe_dump(exchanges_config, f)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def accessor(temp_config_dir):
    """Create StaticReferenceDataAccessor with test configs."""
    return StaticReferenceDataAccessor(temp_config_dir)


class TestStaticReferenceDataAccessorInitialization:
    """Test StaticReferenceDataAccessor initialization."""

    def test_initialization_with_valid_directory(self, temp_config_dir):
        """Test initialization with valid config directory."""
        accessor = StaticReferenceDataAccessor(temp_config_dir)
        assert accessor._config_dir == temp_config_dir

    def test_initialization_with_string_path(self, temp_config_dir):
        """Test initialization with string path (converted to Path)."""
        accessor = StaticReferenceDataAccessor(str(temp_config_dir))
        assert accessor._config_dir == temp_config_dir

    def test_initialization_with_nonexistent_directory_raises_error(self):
        """Test that ValueError is raised for nonexistent directory."""
        with pytest.raises(ValueError, match="does not exist"):
            StaticReferenceDataAccessor(Path("/nonexistent/directory"))


class TestGetTickerInfo:
    """Test get_ticker_info method."""

    def test_get_ticker_info_returns_metadata(self, accessor):
        """Test getting ticker info returns TickerMetadata."""
        info = accessor.get_ticker_info("SPY")

        assert isinstance(info, TickerMetadata)
        assert info.ticker == "SPY"
        assert info.name == "SPDR S&P 500 ETF Trust"
        assert info.sector == "Broad Market"
        assert info.exchange == "NYSE"
        assert info.currency == "USD"

    def test_get_ticker_info_case_insensitive(self, accessor):
        """Test that ticker lookup is case-insensitive."""
        info1 = accessor.get_ticker_info("spy")
        info2 = accessor.get_ticker_info("SPY")
        info3 = accessor.get_ticker_info("Spy")

        assert info1 == info2 == info3

    def test_get_ticker_info_different_exchanges(self, accessor):
        """Test getting tickers from different exchanges."""
        nyse_ticker = accessor.get_ticker_info("SPY")
        nasdaq_ticker = accessor.get_ticker_info("QQQ")

        assert nyse_ticker.exchange == "NYSE"
        assert nasdaq_ticker.exchange == "NASDAQ"

    def test_get_ticker_info_missing_raises_error(self, accessor):
        """Test that ValueError is raised for unknown ticker."""
        with pytest.raises(ValueError, match="not found"):
            accessor.get_ticker_info("INVALID_TICKER")

    def test_get_ticker_info_error_message_helpful(self, accessor):
        """Test that error message provides helpful info."""
        with pytest.raises(ValueError) as exc_info:
            accessor.get_ticker_info("INVALID")

        assert "INVALID" in str(exc_info.value)
        assert "5 total" in str(exc_info.value)  # Number of available tickers


class TestGetSectorMapping:
    """Test get_sector_mapping method."""

    def test_get_sector_mapping_returns_dict(self, accessor):
        """Test that sector mapping returns a dictionary."""
        mapping = accessor.get_sector_mapping()

        assert isinstance(mapping, dict)
        assert len(mapping) == 5

    def test_get_sector_mapping_correct_values(self, accessor):
        """Test that sector mapping contains correct values."""
        mapping = accessor.get_sector_mapping()

        assert mapping["SPY"] == "Broad Market"
        assert mapping["QQQ"] == "Technology"
        assert mapping["EFA"] == "International Developed"
        assert mapping["TLT"] == "Fixed Income"
        assert mapping["GLD"] == "Commodities"

    def test_get_sector_mapping_returns_copy(self, accessor):
        """Test that sector mapping returns a copy, not the cached dict."""
        mapping1 = accessor.get_sector_mapping()
        mapping2 = accessor.get_sector_mapping()

        # Modify mapping1
        mapping1["NEW"] = "Test"

        # mapping2 should not be affected
        assert "NEW" not in mapping2


class TestGetExchangeInfo:
    """Test get_exchange_info method."""

    def test_get_exchange_info_returns_dict(self, accessor):
        """Test that exchange info returns a dictionary."""
        exchanges = accessor.get_exchange_info()

        assert isinstance(exchanges, dict)
        assert len(exchanges) == 2

    def test_get_exchange_info_returns_exchange_info_objects(self, accessor):
        """Test that exchange info contains ExchangeInfo objects."""
        exchanges = accessor.get_exchange_info()

        nyse = exchanges["NYSE"]
        assert isinstance(nyse, ExchangeInfo)
        assert nyse.name == "New York Stock Exchange"
        assert nyse.trading_hours == "09:30-16:00"
        assert nyse.timezone == "US/Eastern"
        assert nyse.settlement_days == 2

    def test_get_exchange_info_returns_copy(self, accessor):
        """Test that exchange info returns a copy."""
        exchanges1 = accessor.get_exchange_info()
        exchanges2 = accessor.get_exchange_info()

        # Modify exchanges1
        del exchanges1["NYSE"]

        # exchanges2 should not be affected
        assert "NYSE" in exchanges2


class TestGetSectors:
    """Test get_sectors method."""

    def test_get_sectors_returns_list(self, accessor):
        """Test that get_sectors returns a list."""
        sectors = accessor.get_sectors()

        assert isinstance(sectors, list)
        assert len(sectors) == 5

    def test_get_sectors_unique(self, accessor):
        """Test that sectors list has no duplicates."""
        sectors = accessor.get_sectors()

        assert len(sectors) == len(set(sectors))

    def test_get_sectors_sorted(self, accessor):
        """Test that sectors list is sorted."""
        sectors = accessor.get_sectors()

        assert sectors == sorted(sectors)

    def test_get_sectors_contains_expected_values(self, accessor):
        """Test that sectors contains expected values."""
        sectors = accessor.get_sectors()

        assert "Broad Market" in sectors
        assert "Technology" in sectors
        assert "Fixed Income" in sectors


class TestGetTickersBySector:
    """Test get_tickers_by_sector method."""

    def test_get_tickers_by_sector_returns_list(self, accessor):
        """Test that get_tickers_by_sector returns a list."""
        tickers = accessor.get_tickers_by_sector("Broad Market")

        assert isinstance(tickers, list)

    def test_get_tickers_by_sector_correct_tickers(self, accessor):
        """Test that correct tickers are returned for sector."""
        broad_market = accessor.get_tickers_by_sector("Broad Market")
        tech = accessor.get_tickers_by_sector("Technology")

        assert broad_market == ["SPY"]
        assert tech == ["QQQ"]

    def test_get_tickers_by_sector_sorted(self, accessor):
        """Test that returned tickers are sorted."""
        # Add more tickers to the same sector for testing sorting
        tickers = accessor.get_tickers_by_sector("Broad Market")

        assert tickers == sorted(tickers)

    def test_get_tickers_by_sector_missing_raises_error(self, accessor):
        """Test that ValueError is raised for unknown sector."""
        with pytest.raises(ValueError, match="not found"):
            accessor.get_tickers_by_sector("Nonexistent Sector")

    def test_get_tickers_by_sector_error_lists_available(self, accessor):
        """Test that error message lists available sectors."""
        with pytest.raises(ValueError) as exc_info:
            accessor.get_tickers_by_sector("Bad Sector")

        assert "Available sectors" in str(exc_info.value)


class TestGetAvailableTickers:
    """Test get_available_tickers method."""

    def test_get_available_tickers_returns_list(self, accessor):
        """Test that get_available_tickers returns a list."""
        tickers = accessor.get_available_tickers()

        assert isinstance(tickers, list)
        assert len(tickers) == 5

    def test_get_available_tickers_sorted(self, accessor):
        """Test that tickers are sorted."""
        tickers = accessor.get_available_tickers()

        assert tickers == sorted(tickers)

    def test_get_available_tickers_contains_expected(self, accessor):
        """Test that expected tickers are in list."""
        tickers = accessor.get_available_tickers()

        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "GLD" in tickers


class TestCaching:
    """Test caching behavior."""

    def test_lazy_loading_tickers(self, accessor):
        """Test that tickers are lazy-loaded."""
        assert accessor._tickers_cache is None

        accessor.get_ticker_info("SPY")

        assert accessor._tickers_cache is not None

    def test_lazy_loading_exchanges(self, accessor):
        """Test that exchanges are lazy-loaded."""
        assert accessor._exchanges_cache is None

        accessor.get_exchange_info()

        assert accessor._exchanges_cache is not None

    def test_lazy_loading_sector_mapping(self, accessor):
        """Test that sector mapping is lazy-loaded."""
        assert accessor._sector_mapping_cache is None

        accessor.get_sector_mapping()

        assert accessor._sector_mapping_cache is not None

    def test_cache_hit_returns_same_data(self, accessor):
        """Test that cache returns consistent data."""
        info1 = accessor.get_ticker_info("SPY")
        info2 = accessor.get_ticker_info("SPY")

        assert info1 is info2  # Same object from cache

    def test_clear_cache(self, accessor):
        """Test that clear_cache removes all cached data."""
        # Load all caches
        accessor.get_ticker_info("SPY")
        accessor.get_exchange_info()
        accessor.get_sector_mapping()

        # Verify caches are populated
        assert accessor._tickers_cache is not None
        assert accessor._exchanges_cache is not None
        assert accessor._sector_mapping_cache is not None

        # Clear caches
        accessor.clear_cache()

        # Verify caches are cleared
        assert accessor._tickers_cache is None
        assert accessor._exchanges_cache is None
        assert accessor._sector_mapping_cache is None


class TestInvalidConfig:
    """Test handling of invalid configuration files."""

    def test_missing_tickers_file_raises_error(self):
        """Test that missing tickers.yaml raises FileNotFoundError."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Only create exchanges.yaml
            exchanges_config = {"exchanges": {"NYSE": {"name": "NYSE"}}}
            with open(temp_dir / "exchanges.yaml", "w") as f:
                yaml.safe_dump(exchanges_config, f)

            accessor = StaticReferenceDataAccessor(temp_dir)

            with pytest.raises(FileNotFoundError, match="tickers.yaml"):
                accessor.get_ticker_info("SPY")
        finally:
            shutil.rmtree(temp_dir)

    def test_missing_exchanges_file_raises_error(self):
        """Test that missing exchanges.yaml raises FileNotFoundError."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Only create tickers.yaml
            tickers_config = {"tickers": {"SPY": {"name": "SPY"}}}
            with open(temp_dir / "tickers.yaml", "w") as f:
                yaml.safe_dump(tickers_config, f)

            accessor = StaticReferenceDataAccessor(temp_dir)

            with pytest.raises(FileNotFoundError, match="exchanges.yaml"):
                accessor.get_exchange_info()
        finally:
            shutil.rmtree(temp_dir)

    def test_invalid_tickers_format_raises_error(self):
        """Test that invalid tickers.yaml format raises ValueError."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create tickers.yaml without 'tickers' key
            bad_config = {"bad_key": {"SPY": {"name": "SPY"}}}
            with open(temp_dir / "tickers.yaml", "w") as f:
                yaml.safe_dump(bad_config, f)

            accessor = StaticReferenceDataAccessor(temp_dir)

            with pytest.raises(ValueError, match="missing 'tickers' key"):
                accessor.get_ticker_info("SPY")
        finally:
            shutil.rmtree(temp_dir)

    def test_invalid_exchanges_format_raises_error(self):
        """Test that invalid exchanges.yaml format raises ValueError."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create valid tickers.yaml
            tickers_config = {"tickers": {"SPY": {"name": "SPY"}}}
            with open(temp_dir / "tickers.yaml", "w") as f:
                yaml.safe_dump(tickers_config, f)

            # Create exchanges.yaml without 'exchanges' key
            bad_config = {"bad_key": {"NYSE": {"name": "NYSE"}}}
            with open(temp_dir / "exchanges.yaml", "w") as f:
                yaml.safe_dump(bad_config, f)

            accessor = StaticReferenceDataAccessor(temp_dir)

            with pytest.raises(ValueError, match="missing 'exchanges' key"):
                accessor.get_exchange_info()
        finally:
            shutil.rmtree(temp_dir)


class TestDefaultValues:
    """Test default values when optional fields are missing."""

    def test_missing_ticker_fields_use_defaults(self):
        """Test that missing ticker fields use sensible defaults."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create minimal tickers.yaml
            tickers_config = {"tickers": {"ABC": {}}}
            with open(temp_dir / "tickers.yaml", "w") as f:
                yaml.safe_dump(tickers_config, f)

            accessor = StaticReferenceDataAccessor(temp_dir)
            info = accessor.get_ticker_info("ABC")

            assert info.ticker == "ABC"
            assert info.name == "ABC"  # Defaults to ticker
            assert info.sector == "Unknown"
            assert info.exchange == "Unknown"
            assert info.currency == "USD"
        finally:
            shutil.rmtree(temp_dir)

    def test_missing_exchange_fields_use_defaults(self):
        """Test that missing exchange fields use sensible defaults."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create minimal exchanges.yaml
            exchanges_config = {"exchanges": {"TEST": {}}}
            with open(temp_dir / "exchanges.yaml", "w") as f:
                yaml.safe_dump(exchanges_config, f)

            # Also need valid tickers.yaml
            tickers_config = {"tickers": {"SPY": {"name": "SPY"}}}
            with open(temp_dir / "tickers.yaml", "w") as f:
                yaml.safe_dump(tickers_config, f)

            accessor = StaticReferenceDataAccessor(temp_dir)
            exchanges = accessor.get_exchange_info()

            test_exchange = exchanges["TEST"]
            assert test_exchange.name == "TEST"
            assert test_exchange.trading_hours == "09:30-16:00"
            assert test_exchange.timezone == "US/Eastern"
            assert test_exchange.settlement_days == 2
        finally:
            shutil.rmtree(temp_dir)


class TestFactoryIntegration:
    """Test integration with DataAccessFactory."""

    def test_factory_creates_reference_accessor(self, temp_config_dir):
        """Test that factory can create StaticReferenceDataAccessor."""
        from quantetf.data.access.factory import DataAccessFactory

        accessor = DataAccessFactory.create_reference_accessor(
            config={"config_dir": str(temp_config_dir)}
        )

        assert isinstance(accessor, StaticReferenceDataAccessor)

    def test_factory_uses_default_config_dir(self):
        """Test that factory uses default config directory."""
        from quantetf.data.access.factory import DataAccessFactory

        accessor = DataAccessFactory.create_reference_accessor()

        assert accessor._config_dir == Path("configs/reference")

    def test_factory_accepts_custom_config_dir(self, temp_config_dir):
        """Test that factory accepts custom config directory."""
        from quantetf.data.access.factory import DataAccessFactory

        accessor = DataAccessFactory.create_reference_accessor(
            config={"config_dir": str(temp_config_dir)}
        )

        assert accessor._config_dir == temp_config_dir


class TestRealReferenceConfigs:
    """Test with real reference config files (integration tests)."""

    @pytest.fixture
    def real_accessor(self):
        """Create accessor with real config files."""
        config_dir = Path("configs/reference")
        if not config_dir.exists():
            pytest.skip("Real config directory not found")
        return StaticReferenceDataAccessor(config_dir)

    def test_spy_ticker_info(self, real_accessor):
        """Test that SPY ticker info can be loaded."""
        info = real_accessor.get_ticker_info("SPY")

        assert info.ticker == "SPY"
        assert info.sector == "Broad Market"
        assert info.exchange in ["NYSE", "NASDAQ"]

    def test_multiple_sectors_exist(self, real_accessor):
        """Test that multiple sectors are defined."""
        sectors = real_accessor.get_sectors()

        assert len(sectors) > 5
        assert "Technology" in sectors
        assert "Fixed Income" in sectors

    def test_exchange_info_exists(self, real_accessor):
        """Test that exchange info is defined."""
        exchanges = real_accessor.get_exchange_info()

        assert "NYSE" in exchanges
        assert "NASDAQ" in exchanges

    def test_all_tier4_tickers_have_info(self, real_accessor):
        """Test that all tier4 tickers have reference data."""
        # Load tier4 universe tickers
        tier4_config_path = Path("configs/universes/tier4_broad_200.yaml")
        if not tier4_config_path.exists():
            pytest.skip("Tier4 config not found")

        with open(tier4_config_path) as f:
            tier4_config = yaml.safe_load(f)

        tickers = tier4_config["source"]["tickers"]

        # Check all tickers have reference data
        missing = []
        for ticker in tickers:
            try:
                real_accessor.get_ticker_info(ticker)
            except ValueError:
                missing.append(ticker)

        if missing:
            pytest.fail(
                f"Missing reference data for {len(missing)} tickers: {missing[:10]}..."
            )
