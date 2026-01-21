"""Tests for ConfigFileUniverseAccessor."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import yaml

from quantetf.data.access.universe import ConfigFileUniverseAccessor


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with test universe configs."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create static universe config
    static_config = {
        "name": "test_static_universe",
        "description": "Test static universe",
        "tier": 1,
        "size": 4,
        "liquidity_profile": "high",
        "source": {
            "type": "static_list",
            "tickers": ["SPY", "QQQ", "IWM", "EEM"],
        },
        "eligibility": {
            "min_history_days": 252,
            "min_avg_dollar_volume": 50_000_000,
        },
    }
    with open(temp_dir / "static_universe.yaml", "w") as f:
        yaml.safe_dump(static_config, f)

    # Create graduated universe config
    graduated_config = {
        "name": "test_graduated_universe",
        "description": "Test graduated universe",
        "tier": 2,
        "source": {
            "type": "graduated",
            "tickers": [
                {"ticker": "SPY", "added_date": "2016-01-01"},
                {"ticker": "QQQ", "added_date": "2016-01-01"},
                {"ticker": "IWM", "added_date": "2018-01-01"},
                {"ticker": "ARKK", "added_date": "2020-01-01"},
            ],
        },
    }
    with open(temp_dir / "graduated_universe.yaml", "w") as f:
        yaml.safe_dump(graduated_config, f)

    # Create empty universe config (edge case)
    empty_config = {
        "name": "test_empty_universe",
        "source": {
            "type": "static_list",
            "tickers": [],
        },
    }
    with open(temp_dir / "empty_universe.yaml", "w") as f:
        yaml.safe_dump(empty_config, f)

    # Create legacy format config (tickers at root level)
    legacy_config = {
        "name": "test_legacy_universe",
        "tickers": ["VTI", "BND", "GLD"],
        "source": {"type": "static_list"},
    }
    with open(temp_dir / "legacy_universe.yaml", "w") as f:
        yaml.safe_dump(legacy_config, f)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def accessor(temp_config_dir):
    """Create ConfigFileUniverseAccessor with test configs."""
    return ConfigFileUniverseAccessor(temp_config_dir)


class TestConfigFileUniverseAccessorInitialization:
    """Test ConfigFileUniverseAccessor initialization."""

    def test_initialization_with_valid_directory(self, temp_config_dir):
        """Test initialization with valid config directory."""
        accessor = ConfigFileUniverseAccessor(temp_config_dir)
        assert accessor.config_dir == temp_config_dir

    def test_initialization_with_string_path(self, temp_config_dir):
        """Test initialization with string path (converted to Path)."""
        accessor = ConfigFileUniverseAccessor(str(temp_config_dir))
        assert accessor.config_dir == temp_config_dir

    def test_initialization_with_nonexistent_directory_raises_error(self):
        """Test that FileNotFoundError is raised for nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="not found"):
            ConfigFileUniverseAccessor(Path("/nonexistent/directory"))

    def test_initialization_with_file_raises_error(self, temp_config_dir):
        """Test that ValueError is raised when path is a file, not directory."""
        file_path = temp_config_dir / "static_universe.yaml"
        with pytest.raises(ValueError, match="must be a directory"):
            ConfigFileUniverseAccessor(file_path)

    def test_initialization_scans_universe_files(self, accessor):
        """Test that initialization scans for universe files."""
        universes = accessor.list_available_universes()
        assert len(universes) >= 4  # At least our test universes


class TestGetUniverse:
    """Test get_universe method."""

    def test_get_universe_static(self, accessor):
        """Test getting a static universe returns correct tickers."""
        tickers = accessor.get_universe("test_static_universe")

        assert isinstance(tickers, list)
        assert len(tickers) == 4
        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "IWM" in tickers
        assert "EEM" in tickers

    def test_get_universe_graduated_returns_all(self, accessor):
        """Test getting graduated universe without as_of returns all tickers."""
        tickers = accessor.get_universe("test_graduated_universe")

        assert len(tickers) == 4
        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "IWM" in tickers
        assert "ARKK" in tickers

    def test_get_universe_case_insensitive(self, accessor):
        """Test that universe names are case-insensitive."""
        tickers1 = accessor.get_universe("test_static_universe")
        tickers2 = accessor.get_universe("TEST_STATIC_UNIVERSE")
        tickers3 = accessor.get_universe("Test_Static_Universe")

        assert tickers1 == tickers2 == tickers3

    def test_get_universe_by_filename(self, accessor):
        """Test getting universe by filename (without .yaml extension)."""
        tickers = accessor.get_universe("static_universe")

        assert len(tickers) == 4
        assert "SPY" in tickers

    def test_get_universe_missing_raises_error(self, accessor):
        """Test that ValueError is raised for nonexistent universe."""
        with pytest.raises(ValueError, match="not found"):
            accessor.get_universe("nonexistent_universe")

    def test_get_universe_error_message_lists_available(self, accessor):
        """Test that error message lists available universes."""
        with pytest.raises(ValueError) as exc_info:
            accessor.get_universe("nonexistent_universe")

        assert "Available universes" in str(exc_info.value)

    def test_get_universe_empty_warns(self, accessor):
        """Test that warning is raised for empty universe."""
        with pytest.warns(UserWarning, match="empty"):
            tickers = accessor.get_universe("test_empty_universe")
            assert tickers == []

    def test_get_universe_tickers_uppercase(self, accessor):
        """Test that returned tickers are uppercase."""
        tickers = accessor.get_universe("test_static_universe")

        for ticker in tickers:
            assert ticker == ticker.upper()

    def test_get_universe_legacy_format(self, accessor):
        """Test getting universe with legacy config format."""
        tickers = accessor.get_universe("test_legacy_universe")

        assert len(tickers) == 3
        assert "VTI" in tickers
        assert "BND" in tickers
        assert "GLD" in tickers


class TestGetUniverseAsOf:
    """Test get_universe_as_of method."""

    def test_get_universe_as_of_graduated_early(self, accessor):
        """Test graduated universe as of early date returns subset."""
        as_of = pd.Timestamp("2017-01-01")
        tickers = accessor.get_universe_as_of("test_graduated_universe", as_of)

        assert len(tickers) == 2
        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "IWM" not in tickers  # Added 2018
        assert "ARKK" not in tickers  # Added 2020

    def test_get_universe_as_of_graduated_mid(self, accessor):
        """Test graduated universe as of mid date returns partial."""
        as_of = pd.Timestamp("2019-06-15")
        tickers = accessor.get_universe_as_of("test_graduated_universe", as_of)

        assert len(tickers) == 3
        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "IWM" in tickers
        assert "ARKK" not in tickers  # Added 2020

    def test_get_universe_as_of_graduated_late(self, accessor):
        """Test graduated universe as of late date returns all."""
        as_of = pd.Timestamp("2025-01-01")
        tickers = accessor.get_universe_as_of("test_graduated_universe", as_of)

        assert len(tickers) == 4
        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "IWM" in tickers
        assert "ARKK" in tickers

    def test_get_universe_as_of_static_unchanged(self, accessor):
        """Test static universe returns same result regardless of as_of."""
        early = pd.Timestamp("2010-01-01")
        late = pd.Timestamp("2025-01-01")

        tickers_early = accessor.get_universe_as_of("test_static_universe", early)
        tickers_late = accessor.get_universe_as_of("test_static_universe", late)

        assert tickers_early == tickers_late
        assert len(tickers_early) == 4

    def test_get_universe_as_of_boundary_inclusive(self, accessor):
        """Test that tickers added ON as_of date are included."""
        # IWM added on 2018-01-01
        as_of = pd.Timestamp("2018-01-01")
        tickers = accessor.get_universe_as_of("test_graduated_universe", as_of)

        assert "IWM" in tickers

    def test_get_universe_as_of_boundary_exclusive(self, accessor):
        """Test that tickers added AFTER as_of date are excluded."""
        # IWM added on 2018-01-01
        as_of = pd.Timestamp("2017-12-31")
        tickers = accessor.get_universe_as_of("test_graduated_universe", as_of)

        assert "IWM" not in tickers

    def test_get_universe_as_of_empty_warns(self, accessor):
        """Test that warning is raised when no tickers as of date."""
        as_of = pd.Timestamp("2010-01-01")  # Before any tickers added
        with pytest.warns(UserWarning, match="empty"):
            tickers = accessor.get_universe_as_of("test_graduated_universe", as_of)
            assert tickers == []


class TestListAvailableUniverses:
    """Test list_available_universes method."""

    def test_list_available_universes(self, accessor):
        """Test listing available universes."""
        universes = accessor.list_available_universes()

        assert isinstance(universes, list)
        assert len(universes) >= 4
        assert "test_static_universe" in universes
        assert "test_graduated_universe" in universes

    def test_list_available_universes_sorted(self, accessor):
        """Test that universe list is sorted."""
        universes = accessor.list_available_universes()

        assert universes == sorted(universes)

    def test_list_available_universes_unique(self, accessor):
        """Test that universe list has no duplicates."""
        universes = accessor.list_available_universes()

        assert len(universes) == len(set(universes))


class TestGetUniverseMetadata:
    """Test get_universe_metadata method."""

    def test_get_universe_metadata_static(self, accessor):
        """Test getting metadata for static universe."""
        metadata = accessor.get_universe_metadata("test_static_universe")

        assert metadata["name"] == "test_static_universe"
        assert metadata["description"] == "Test static universe"
        assert metadata["size"] == 4
        assert metadata["tier"] == 1
        assert metadata["type"] == "static_list"
        assert metadata["liquidity_profile"] == "high"

    def test_get_universe_metadata_graduated(self, accessor):
        """Test getting metadata for graduated universe."""
        metadata = accessor.get_universe_metadata("test_graduated_universe")

        assert metadata["name"] == "test_graduated_universe"
        assert metadata["type"] == "graduated"
        assert metadata["size"] == 4

    def test_get_universe_metadata_eligibility(self, accessor):
        """Test that eligibility criteria are included in metadata."""
        metadata = accessor.get_universe_metadata("test_static_universe")

        assert "eligibility" in metadata
        assert metadata["eligibility"]["min_history_days"] == 252

    def test_get_universe_metadata_missing_raises_error(self, accessor):
        """Test that error is raised for nonexistent universe."""
        with pytest.raises(ValueError, match="not found"):
            accessor.get_universe_metadata("nonexistent_universe")


class TestCaching:
    """Test caching behavior."""

    def test_caching_enabled_by_default(self, temp_config_dir):
        """Test that caching is enabled by default."""
        accessor = ConfigFileUniverseAccessor(temp_config_dir)
        assert accessor._cache_enabled is True

    def test_caching_can_be_disabled(self, temp_config_dir):
        """Test that caching can be disabled."""
        accessor = ConfigFileUniverseAccessor(temp_config_dir, cache=False)
        assert accessor._cache_enabled is False

    def test_cache_stores_config(self, accessor):
        """Test that config is cached after first access."""
        assert len(accessor._cache) == 0

        accessor.get_universe("test_static_universe")

        assert len(accessor._cache) > 0
        assert "test_static_universe" in accessor._cache

    def test_cache_hit_returns_same_object(self, accessor):
        """Test that cache returns same config object."""
        accessor.get_universe("test_static_universe")
        config1 = accessor._cache.get("test_static_universe")

        accessor.get_universe("test_static_universe")
        config2 = accessor._cache.get("test_static_universe")

        assert config1 is config2

    def test_clear_cache(self, accessor):
        """Test that clear_cache removes all cached configs."""
        accessor.get_universe("test_static_universe")
        assert len(accessor._cache) > 0

        accessor.clear_cache()

        assert len(accessor._cache) == 0

    def test_cache_disabled_does_not_store(self, temp_config_dir):
        """Test that disabled cache does not store configs."""
        accessor = ConfigFileUniverseAccessor(temp_config_dir, cache=False)

        accessor.get_universe("test_static_universe")

        assert len(accessor._cache) == 0


class TestInvalidConfig:
    """Test handling of invalid configuration files."""

    def test_invalid_yaml_ignored_during_scan(self, temp_config_dir):
        """Test that invalid YAML files are ignored during scan."""
        # Create invalid YAML file
        with open(temp_config_dir / "invalid.yaml", "w") as f:
            f.write("invalid: yaml: content: [")

        # Should not raise during initialization
        accessor = ConfigFileUniverseAccessor(temp_config_dir)
        universes = accessor.list_available_universes()

        assert len(universes) >= 4  # Still has valid universes

    def test_missing_tickers_raises_error(self, temp_config_dir):
        """Test that missing tickers field raises error."""
        # Create config without tickers
        bad_config = {
            "name": "bad_universe",
            "source": {"type": "static_list"},
        }
        with open(temp_config_dir / "bad_universe.yaml", "w") as f:
            yaml.safe_dump(bad_config, f)

        accessor = ConfigFileUniverseAccessor(temp_config_dir)

        with pytest.raises(ValueError, match="missing 'tickers' field"):
            accessor.get_universe("bad_universe")


class TestFactoryIntegration:
    """Test integration with DataAccessFactory."""

    def test_factory_creates_universe_accessor(self):
        """Test that factory can create ConfigFileUniverseAccessor."""
        from quantetf.data.access.factory import DataAccessFactory

        accessor = DataAccessFactory.create_universe_accessor()

        assert isinstance(accessor, ConfigFileUniverseAccessor)

    def test_factory_uses_default_config_dir(self):
        """Test that factory uses default config directory."""
        from quantetf.data.access.factory import DataAccessFactory

        accessor = DataAccessFactory.create_universe_accessor()

        assert accessor.config_dir == Path("configs/universes")

    def test_factory_accepts_custom_config_dir(self, temp_config_dir):
        """Test that factory accepts custom config directory."""
        from quantetf.data.access.factory import DataAccessFactory

        accessor = DataAccessFactory.create_universe_accessor(
            config={"config_dir": str(temp_config_dir)}
        )

        assert accessor.config_dir == temp_config_dir

    def test_factory_accepts_cache_config(self, temp_config_dir):
        """Test that factory passes cache config."""
        from quantetf.data.access.factory import DataAccessFactory

        accessor = DataAccessFactory.create_universe_accessor(
            config={"config_dir": str(temp_config_dir), "cache": False}
        )

        assert accessor._cache_enabled is False


class TestRealUniverseConfigs:
    """Test with real universe config files (integration tests)."""

    @pytest.fixture
    def real_accessor(self):
        """Create accessor with real config files."""
        config_dir = Path("configs/universes")
        if not config_dir.exists():
            pytest.skip("Real config directory not found")
        return ConfigFileUniverseAccessor(config_dir)

    def test_tier1_universe_exists(self, real_accessor):
        """Test that tier1 universe can be loaded."""
        tickers = real_accessor.get_universe("tier1_initial_20_etfs")

        assert len(tickers) == 20
        assert "SPY" in tickers
        assert "QQQ" in tickers

    def test_tier4_universe_exists(self, real_accessor):
        """Test that tier4 universe can be loaded."""
        tickers = real_accessor.get_universe("tier4_broad_200_etfs")

        assert len(tickers) == 200
        assert "SPY" in tickers

    def test_list_real_universes(self, real_accessor):
        """Test listing real universe files."""
        universes = real_accessor.list_available_universes()

        assert len(universes) > 0
        # Check some expected universes
        assert any("tier1" in u.lower() for u in universes)
