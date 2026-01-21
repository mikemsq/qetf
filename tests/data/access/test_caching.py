"""Tests for caching layer (CachedPriceAccessor, CachedMacroAccessor, CacheManager)."""

import time
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from quantetf.data.access.caching import (
    CachedPriceAccessor,
    CachedMacroAccessor,
    CacheStats,
    CacheEntry,
    _estimate_df_size,
    _make_cache_key,
    _make_range_cache_key,
)
from quantetf.data.access.cache_manager import (
    CacheManager,
    get_global_cache_manager,
    reset_global_cache_manager,
)
from quantetf.data.access.types import Regime


@pytest.fixture
def mock_price_data():
    """Create mock price data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = ["SPY", "QQQ", "IWM"]

    columns = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Price"],
    )

    data = np.random.randn(len(dates), len(columns)) * 10 + 100
    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture
def mock_inner_price_accessor(mock_price_data):
    """Create a mock PriceDataAccessor."""
    mock = MagicMock()
    mock.read_prices_as_of.return_value = mock_price_data
    mock.read_ohlcv_range.return_value = mock_price_data
    mock.get_latest_price_date.return_value = mock_price_data.index.max()
    mock.validate_data_availability.return_value = {"SPY": True, "QQQ": True}
    mock.date_range = (mock_price_data.index.min(), mock_price_data.index.max())
    mock.get_available_tickers.return_value = ["SPY", "QQQ", "IWM"]
    return mock


@pytest.fixture
def mock_macro_data():
    """Create mock macro data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    return pd.DataFrame({"VIX": np.random.randn(len(dates)) * 5 + 20}, index=dates)


@pytest.fixture
def mock_inner_macro_accessor(mock_macro_data):
    """Create a mock MacroDataAccessor."""
    mock = MagicMock()
    mock.read_macro_indicator.return_value = mock_macro_data
    mock.get_regime.return_value = Regime.RISK_ON
    mock.get_available_indicators.return_value = ["VIX", "T10Y2Y", "UNRATE"]
    return mock


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_initial_values(self):
        """Test default values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_bytes == 0

    def test_hit_rate_zero_requests(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 80.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(hits=10, misses=5, evictions=2, total_bytes=1000)
        result = stats.to_dict()
        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["evictions"] == 2
        assert result["total_bytes"] == 1000
        assert "hit_rate_pct" in result


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(data="test", size_bytes=100)
        assert entry.data == "test"
        assert entry.size_bytes == 100
        assert entry.created_at > 0
        assert entry.last_accessed > 0

    def test_touch(self):
        """Test touch updates last_accessed."""
        entry = CacheEntry(data="test", size_bytes=100)
        old_time = entry.last_accessed
        time.sleep(0.01)
        entry.touch()
        assert entry.last_accessed > old_time


class TestCacheKeyFunctions:
    """Test cache key generation functions."""

    def test_make_cache_key_consistency(self):
        """Test that same inputs produce same key."""
        as_of = pd.Timestamp("2023-06-15")
        tickers = ["SPY", "QQQ"]

        key1 = _make_cache_key("read_prices_as_of", as_of, tickers, 30)
        key2 = _make_cache_key("read_prices_as_of", as_of, tickers, 30)

        assert key1 == key2

    def test_make_cache_key_ticker_order_invariant(self):
        """Test that ticker order doesn't matter."""
        as_of = pd.Timestamp("2023-06-15")

        key1 = _make_cache_key("read_prices_as_of", as_of, ["SPY", "QQQ"], None)
        key2 = _make_cache_key("read_prices_as_of", as_of, ["QQQ", "SPY"], None)

        assert key1 == key2

    def test_make_cache_key_different_params(self):
        """Test that different params produce different keys."""
        as_of = pd.Timestamp("2023-06-15")

        key1 = _make_cache_key("read_prices_as_of", as_of, ["SPY"], 30)
        key2 = _make_cache_key("read_prices_as_of", as_of, ["SPY"], 60)

        assert key1 != key2

    def test_make_range_cache_key(self):
        """Test range cache key generation."""
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-06-30")

        key1 = _make_range_cache_key(start, end, ["SPY"])
        key2 = _make_range_cache_key(start, end, ["SPY"])

        assert key1 == key2


class TestCachedPriceAccessor:
    """Test CachedPriceAccessor."""

    def test_initialization(self, mock_inner_price_accessor):
        """Test initialization."""
        cached = CachedPriceAccessor(mock_inner_price_accessor, max_cache_mb=100)
        assert cached._max_cache_bytes == 100 * 1024 * 1024
        assert len(cached._cache) == 0

    def test_cache_hit(self, mock_inner_price_accessor, mock_price_data):
        """Test that repeated calls return cached data."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)
        as_of = pd.Timestamp("2023-06-15")

        # First call - cache miss
        result1 = cached.read_prices_as_of(as_of)
        assert mock_inner_price_accessor.read_prices_as_of.call_count == 1

        # Second call - cache hit
        result2 = cached.read_prices_as_of(as_of)
        assert mock_inner_price_accessor.read_prices_as_of.call_count == 1

        # Results should be equal
        pd.testing.assert_frame_equal(result1, result2)

    def test_cache_miss_different_params(self, mock_inner_price_accessor):
        """Test that different params cause cache miss."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)

        cached.read_prices_as_of(pd.Timestamp("2023-06-15"))
        cached.read_prices_as_of(pd.Timestamp("2023-06-16"))

        assert mock_inner_price_accessor.read_prices_as_of.call_count == 2

    def test_cache_stats_tracking(self, mock_inner_price_accessor):
        """Test that cache stats are tracked."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)
        as_of = pd.Timestamp("2023-06-15")

        # First call - miss
        cached.read_prices_as_of(as_of)
        stats = cached.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Second call - hit
        cached.read_prices_as_of(as_of)
        stats = cached.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1

    def test_lru_eviction(self, mock_inner_price_accessor, mock_price_data):
        """Test LRU eviction when cache exceeds limit."""
        # Create small cache (1 KB)
        cached = CachedPriceAccessor(mock_inner_price_accessor, max_cache_mb=0.001)

        # Make multiple requests to trigger eviction
        for i in range(5):
            as_of = pd.Timestamp(f"2023-06-{10+i}")
            cached.read_prices_as_of(as_of)

        stats = cached.get_cache_stats()
        assert stats["evictions"] > 0

    def test_clear_cache(self, mock_inner_price_accessor):
        """Test cache clearing."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)
        as_of = pd.Timestamp("2023-06-15")

        cached.read_prices_as_of(as_of)
        assert len(cached._cache) > 0

        cached.clear_cache()
        assert len(cached._cache) == 0
        assert cached._stats.total_bytes == 0

    def test_read_ohlcv_range_cached(self, mock_inner_price_accessor):
        """Test that read_ohlcv_range is also cached."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-06-30")

        cached.read_ohlcv_range(start, end)
        cached.read_ohlcv_range(start, end)

        assert mock_inner_price_accessor.read_ohlcv_range.call_count == 1

    def test_passthrough_methods(self, mock_inner_price_accessor):
        """Test that some methods pass through without caching."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)

        # These should delegate directly
        cached.get_latest_price_date()
        cached.validate_data_availability(["SPY"], pd.Timestamp("2023-06-15"))
        cached.get_available_tickers()

        assert mock_inner_price_accessor.get_latest_price_date.called
        assert mock_inner_price_accessor.validate_data_availability.called
        assert mock_inner_price_accessor.get_available_tickers.called

    def test_date_range_property(self, mock_inner_price_accessor, mock_price_data):
        """Test date_range property delegation."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)
        start, end = cached.date_range

        assert start == mock_price_data.index.min()
        assert end == mock_price_data.index.max()

    def test_invalidate_after(self, mock_inner_price_accessor):
        """Test invalidate_after clears cache."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)
        as_of = pd.Timestamp("2023-06-15")

        cached.read_prices_as_of(as_of)
        assert len(cached._cache) > 0

        count = cached.invalidate_after(pd.Timestamp("2023-06-01"))
        assert count > 0
        assert len(cached._cache) == 0


class TestCachedMacroAccessor:
    """Test CachedMacroAccessor."""

    def test_initialization(self, mock_inner_macro_accessor):
        """Test initialization."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor, cache_ttl_seconds=3600)
        assert cached._default_ttl == 3600
        assert len(cached._cache) == 0

    def test_cache_hit(self, mock_inner_macro_accessor):
        """Test that repeated calls return cached data."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor)
        as_of = pd.Timestamp("2023-06-15")

        cached.read_macro_indicator("VIX", as_of)
        cached.read_macro_indicator("VIX", as_of)

        assert mock_inner_macro_accessor.read_macro_indicator.call_count == 1

    def test_cache_miss_different_indicator(self, mock_inner_macro_accessor):
        """Test that different indicators cause cache miss."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor)
        as_of = pd.Timestamp("2023-06-15")

        cached.read_macro_indicator("VIX", as_of)
        cached.read_macro_indicator("T10Y2Y", as_of)

        assert mock_inner_macro_accessor.read_macro_indicator.call_count == 2

    def test_ttl_expiration(self, mock_inner_macro_accessor):
        """Test that entries expire after TTL."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor, cache_ttl_seconds=0.05)
        as_of = pd.Timestamp("2023-06-15")

        cached.read_macro_indicator("VIX", as_of)
        assert mock_inner_macro_accessor.read_macro_indicator.call_count == 1

        # Wait for TTL to expire
        time.sleep(0.1)

        cached.read_macro_indicator("VIX", as_of)
        assert mock_inner_macro_accessor.read_macro_indicator.call_count == 2

    def test_regime_cached(self, mock_inner_macro_accessor):
        """Test that get_regime is cached."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor)
        as_of = pd.Timestamp("2023-06-15")

        result1 = cached.get_regime(as_of)
        result2 = cached.get_regime(as_of)

        assert mock_inner_macro_accessor.get_regime.call_count == 1
        assert result1 == result2

    def test_get_available_indicators_not_cached(self, mock_inner_macro_accessor):
        """Test that get_available_indicators passes through."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor)

        cached.get_available_indicators()
        cached.get_available_indicators()

        assert mock_inner_macro_accessor.get_available_indicators.call_count == 2

    def test_set_indicator_ttl(self, mock_inner_macro_accessor):
        """Test per-indicator TTL configuration."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor)
        cached.set_indicator_ttl("VIX", 60)

        assert cached._indicator_ttls["VIX"] == 60

    def test_purge_expired(self, mock_inner_macro_accessor):
        """Test purging expired entries."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor, cache_ttl_seconds=0.01)
        as_of = pd.Timestamp("2023-06-15")

        cached.read_macro_indicator("VIX", as_of)
        assert len(cached._cache) > 0

        # Wait for expiration
        time.sleep(0.05)

        count = cached.purge_expired()
        assert count > 0

    def test_clear_cache(self, mock_inner_macro_accessor):
        """Test cache clearing."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor)
        as_of = pd.Timestamp("2023-06-15")

        cached.read_macro_indicator("VIX", as_of)
        assert len(cached._cache) > 0

        cached.clear_cache()
        assert len(cached._cache) == 0

    def test_cache_stats(self, mock_inner_macro_accessor):
        """Test cache statistics."""
        cached = CachedMacroAccessor(mock_inner_macro_accessor)
        as_of = pd.Timestamp("2023-06-15")

        cached.read_macro_indicator("VIX", as_of)
        cached.read_macro_indicator("VIX", as_of)

        stats = cached.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert "valid_entries" in stats
        assert "expired_entries" in stats


class TestCacheManager:
    """Test CacheManager."""

    def test_initialization(self):
        """Test CacheManager initialization."""
        manager = CacheManager()
        assert len(manager._accessors) == 0

    def test_register_accessor(self, mock_inner_price_accessor):
        """Test registering an accessor."""
        manager = CacheManager()
        cached = CachedPriceAccessor(mock_inner_price_accessor)

        manager.register("prices", cached)
        assert "prices" in manager.list_accessors()

    def test_register_duplicate_raises(self, mock_inner_price_accessor):
        """Test that duplicate registration raises."""
        manager = CacheManager()
        cached = CachedPriceAccessor(mock_inner_price_accessor)

        manager.register("prices", cached)
        with pytest.raises(ValueError, match="already registered"):
            manager.register("prices", cached)

    def test_register_wrong_type_raises(self):
        """Test that registering wrong type raises."""
        manager = CacheManager()

        with pytest.raises(TypeError, match="Expected"):
            manager.register("bad", MagicMock())

    def test_unregister(self, mock_inner_price_accessor):
        """Test unregistering an accessor."""
        manager = CacheManager()
        cached = CachedPriceAccessor(mock_inner_price_accessor)

        manager.register("prices", cached)
        manager.unregister("prices")

        assert "prices" not in manager.list_accessors()

    def test_unregister_unknown_raises(self):
        """Test unregistering unknown accessor raises."""
        manager = CacheManager()

        with pytest.raises(KeyError, match="not registered"):
            manager.unregister("unknown")

    def test_clear_all_caches(self, mock_inner_price_accessor, mock_inner_macro_accessor):
        """Test clearing all caches."""
        manager = CacheManager()

        cached_prices = CachedPriceAccessor(mock_inner_price_accessor)
        cached_macro = CachedMacroAccessor(mock_inner_macro_accessor)

        manager.register("prices", cached_prices)
        manager.register("macro", cached_macro)

        # Populate caches
        cached_prices.read_prices_as_of(pd.Timestamp("2023-06-15"))
        cached_macro.read_macro_indicator("VIX", pd.Timestamp("2023-06-15"))

        results = manager.clear_all_caches()

        assert results["prices"] > 0
        assert results["macro"] > 0

    def test_get_global_stats(self, mock_inner_price_accessor, mock_inner_macro_accessor):
        """Test global statistics."""
        manager = CacheManager()

        cached_prices = CachedPriceAccessor(mock_inner_price_accessor)
        cached_macro = CachedMacroAccessor(mock_inner_macro_accessor)

        manager.register("prices", cached_prices)
        manager.register("macro", cached_macro)

        # Generate some activity
        cached_prices.read_prices_as_of(pd.Timestamp("2023-06-15"))
        cached_prices.read_prices_as_of(pd.Timestamp("2023-06-15"))  # hit
        cached_macro.read_macro_indicator("VIX", pd.Timestamp("2023-06-15"))

        stats = manager.get_global_stats()

        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 2
        assert "global_hit_rate_pct" in stats
        assert "per_accessor" in stats
        assert "prices" in stats["per_accessor"]
        assert "macro" in stats["per_accessor"]

    def test_configure_ttl(self, mock_inner_macro_accessor):
        """Test TTL configuration."""
        manager = CacheManager()
        cached_macro = CachedMacroAccessor(mock_inner_macro_accessor)

        manager.register("macro", cached_macro)
        manager.configure_ttl("macro", 7200)

        assert cached_macro._default_ttl == 7200

    def test_configure_ttl_with_indicator(self, mock_inner_macro_accessor):
        """Test per-indicator TTL configuration."""
        manager = CacheManager()
        cached_macro = CachedMacroAccessor(mock_inner_macro_accessor)

        manager.register("macro", cached_macro)
        manager.configure_ttl("macro", 300, indicator="VIX")

        assert cached_macro._indicator_ttls["VIX"] == 300

    def test_configure_ttl_wrong_type_raises(self):
        """Test that TTL config for non-macro raises."""
        manager = CacheManager()

        with pytest.raises(ValueError, match="only applies to"):
            manager.configure_ttl("prices", 3600)

    def test_get_accessor(self, mock_inner_price_accessor):
        """Test getting accessor by name."""
        manager = CacheManager()
        cached = CachedPriceAccessor(mock_inner_price_accessor)

        manager.register("prices", cached)
        result = manager.get_accessor("prices")

        assert result is cached

    def test_get_accessor_unknown_raises(self):
        """Test getting unknown accessor raises."""
        manager = CacheManager()

        with pytest.raises(KeyError, match="not registered"):
            manager.get_accessor("unknown")

    def test_purge_expired(self, mock_inner_macro_accessor):
        """Test purging expired entries."""
        manager = CacheManager()
        cached_macro = CachedMacroAccessor(
            mock_inner_macro_accessor, cache_ttl_seconds=0.01
        )

        manager.register("macro", cached_macro)

        # Populate and let expire
        cached_macro.read_macro_indicator("VIX", pd.Timestamp("2023-06-15"))
        time.sleep(0.05)

        results = manager.purge_expired()
        assert results["macro"] > 0


class TestGlobalCacheManager:
    """Test global cache manager singleton."""

    def test_get_global_cache_manager(self):
        """Test getting global manager."""
        reset_global_cache_manager()

        manager1 = get_global_cache_manager()
        manager2 = get_global_cache_manager()

        assert manager1 is manager2

    def test_reset_global_cache_manager(self):
        """Test resetting global manager."""
        manager1 = get_global_cache_manager()
        reset_global_cache_manager()
        manager2 = get_global_cache_manager()

        assert manager1 is not manager2


class TestCachingPerformance:
    """Performance-related tests."""

    def test_cache_reduces_latency(self, mock_inner_price_accessor, mock_price_data):
        """Test that cache significantly reduces latency."""
        cached = CachedPriceAccessor(mock_inner_price_accessor)
        as_of = pd.Timestamp("2023-06-15")

        # Add artificial delay to inner accessor
        def slow_read(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            return mock_price_data

        mock_inner_price_accessor.read_prices_as_of.side_effect = slow_read

        # First call (cache miss)
        start = time.time()
        cached.read_prices_as_of(as_of)
        miss_time = time.time() - start

        # Second call (cache hit)
        start = time.time()
        cached.read_prices_as_of(as_of)
        hit_time = time.time() - start

        # Cache hit should be much faster (at least 5x)
        assert hit_time < miss_time / 5

    def test_memory_limit_respected(self, mock_inner_price_accessor, mock_price_data):
        """Test that memory limit is approximately respected."""
        # Set 1 MB limit
        cached = CachedPriceAccessor(mock_inner_price_accessor, max_cache_mb=1)

        # Make many requests to exceed limit
        for i in range(20):
            as_of = pd.Timestamp(f"2023-06-{(i % 28) + 1}")
            cached.read_prices_as_of(as_of)

        stats = cached.get_cache_stats()

        # Total bytes should not exceed limit by much
        assert stats["total_bytes"] <= 1.5 * 1024 * 1024


class TestEstimateDfSize:
    """Test DataFrame size estimation."""

    def test_estimate_df_size(self):
        """Test DataFrame size estimation."""
        df = pd.DataFrame({"a": range(1000), "b": range(1000)})
        size = _estimate_df_size(df)
        assert size > 0
        assert isinstance(size, (int, np.integer))

    def test_estimate_empty_df(self):
        """Test empty DataFrame estimation."""
        df = pd.DataFrame()
        size = _estimate_df_size(df)
        assert size >= 0
