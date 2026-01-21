"""Caching layer for Data Access Layer accessors.

Provides transparent caching wrappers for PriceDataAccessor and MacroDataAccessor
to improve performance through LRU and TTL-based caching strategies.

Usage:
    from quantetf.data.access import SnapshotPriceAccessor, FREDMacroAccessor
    from quantetf.data.access.caching import CachedPriceAccessor, CachedMacroAccessor

    # Wrap a price accessor with caching
    inner_price = SnapshotPriceAccessor(snapshot_path)
    cached_price = CachedPriceAccessor(inner_price, max_cache_mb=500)

    # Wrap a macro accessor with caching
    inner_macro = FREDMacroAccessor(loader)
    cached_macro = CachedMacroAccessor(inner_macro, cache_ttl_seconds=3600)
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .abstract import MacroDataAccessor, PriceDataAccessor
from .types import Regime


@dataclass
class CacheStats:
    """Statistics for cache performance tracking."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Return stats as dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_bytes": self.total_bytes,
            "hit_rate_pct": round(self.hit_rate, 2),
        }


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    data: Any
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()


def _estimate_df_size(df: pd.DataFrame) -> int:
    """Estimate DataFrame size in bytes."""
    return df.memory_usage(deep=True).sum()


def _make_cache_key(
    method: str,
    as_of: pd.Timestamp,
    tickers: Optional[list[str]],
    lookback_days: Optional[int],
) -> str:
    """Create a consistent cache key for price queries.

    Args:
        method: Method name (e.g., "read_prices_as_of")
        as_of: Cutoff date
        tickers: Optional ticker list (will be sorted)
        lookback_days: Optional lookback window

    Returns:
        String cache key (MD5 hash for compact keys)
    """
    # Sort tickers for consistent keys
    ticker_str = ",".join(sorted(tickers)) if tickers else "ALL"

    # Create key components
    key_parts = [
        method,
        str(as_of),
        ticker_str,
        str(lookback_days or "NONE"),
    ]

    # Hash for compact key
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _make_range_cache_key(
    start: pd.Timestamp,
    end: pd.Timestamp,
    tickers: Optional[list[str]],
) -> str:
    """Create a consistent cache key for range queries.

    Args:
        start: Start date
        end: End date
        tickers: Optional ticker list (will be sorted)

    Returns:
        String cache key (MD5 hash)
    """
    ticker_str = ",".join(sorted(tickers)) if tickers else "ALL"

    key_parts = [
        "read_ohlcv_range",
        str(start),
        str(end),
        ticker_str,
    ]

    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


class CachedPriceAccessor(PriceDataAccessor):
    """Caching wrapper for PriceDataAccessor with LRU eviction.

    Provides transparent caching with configurable memory limits.
    Uses LRU (Least Recently Used) eviction when cache exceeds size limit.

    Usage:
        inner = SnapshotPriceAccessor(path)
        cached = CachedPriceAccessor(inner, max_cache_mb=500)

        # First call hits inner accessor
        prices1 = cached.read_prices_as_of(as_of)

        # Second call returns cached result
        prices2 = cached.read_prices_as_of(as_of)

        # Check stats
        print(cached.get_cache_stats())
    """

    def __init__(
        self,
        inner: PriceDataAccessor,
        max_cache_mb: int = 500,
        snapshot_cache_dir: Optional[Path] = None,
    ):
        """Initialize caching wrapper.

        Args:
            inner: The underlying PriceDataAccessor to wrap
            max_cache_mb: Maximum cache size in megabytes (default: 500)
            snapshot_cache_dir: Optional directory for persistent cache (not implemented)
        """
        self._inner = inner
        self._max_cache_bytes = max_cache_mb * 1024 * 1024
        self._snapshot_cache_dir = snapshot_cache_dir

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()

    def read_prices_as_of(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return cached OHLCV prices for dates < as_of.

        Args:
            as_of: Cutoff date (exclusive)
            tickers: Optional subset of tickers
            lookback_days: Optional lookback window

        Returns:
            DataFrame with OHLCV data (from cache if available)
        """
        cache_key = _make_cache_key("read_prices_as_of", as_of, tickers, lookback_days)

        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            entry.touch()
            # Move to end for LRU
            self._cache.move_to_end(cache_key)
            self._stats.hits += 1
            return entry.data.copy()

        # Cache miss - fetch from inner
        self._stats.misses += 1
        result = self._inner.read_prices_as_of(as_of, tickers, lookback_days)

        # Store in cache
        self._add_to_cache(cache_key, result)

        return result

    def read_ohlcv_range(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return cached OHLCV for date range [start, end].

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            tickers: Optional subset of tickers

        Returns:
            DataFrame with OHLCV data (from cache if available)
        """
        cache_key = _make_range_cache_key(start, end, tickers)

        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            entry.touch()
            self._cache.move_to_end(cache_key)
            self._stats.hits += 1
            return entry.data.copy()

        # Cache miss
        self._stats.misses += 1
        result = self._inner.read_ohlcv_range(start, end, tickers)

        # Store in cache
        self._add_to_cache(cache_key, result)

        return result

    def get_latest_price_date(self) -> pd.Timestamp:
        """Return most recent date with available price data.

        This is a lightweight call, not cached.
        """
        return self._inner.get_latest_price_date()

    def validate_data_availability(
        self,
        tickers: list[str],
        as_of: pd.Timestamp,
    ) -> dict[str, bool]:
        """Check which tickers have data available.

        This is a lightweight call, not cached.
        """
        return self._inner.validate_data_availability(tickers, as_of)

    def _add_to_cache(self, key: str, data: pd.DataFrame) -> None:
        """Add entry to cache with LRU eviction.

        Args:
            key: Cache key
            data: DataFrame to cache
        """
        size_bytes = _estimate_df_size(data)

        # Evict if necessary
        while (
            self._stats.total_bytes + size_bytes > self._max_cache_bytes
            and self._cache
        ):
            self._evict_oldest()

        # Add to cache
        entry = CacheEntry(data=data.copy(), size_bytes=size_bytes)
        self._cache[key] = entry
        self._stats.total_bytes += size_bytes

    def _evict_oldest(self) -> None:
        """Remove the oldest (least recently used) entry."""
        if not self._cache:
            return

        # Get oldest (first item in OrderedDict)
        oldest_key = next(iter(self._cache))
        oldest_entry = self._cache.pop(oldest_key)

        self._stats.total_bytes -= oldest_entry.size_bytes
        self._stats.evictions += 1

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._stats.total_bytes = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, evictions, size, hit_rate
        """
        return {
            **self._stats.to_dict(),
            "entries": len(self._cache),
            "max_bytes": self._max_cache_bytes,
        }

    def invalidate_after(self, date: pd.Timestamp) -> int:
        """Remove cached entries for queries on/after the given date.

        Useful when data is refreshed and queries after a certain date
        need to be recomputed.

        Args:
            date: Invalidate queries where as_of >= date

        Returns:
            Number of entries invalidated
        """
        # This requires parsing keys, which is complex with hashed keys
        # For now, do a full clear - in production, track as_of dates separately
        count = len(self._cache)
        self.clear_cache()
        return count

    # Delegate additional methods if the inner accessor has them
    @property
    def date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Return date range from inner accessor."""
        return self._inner.date_range

    def get_available_tickers(self) -> list[str]:
        """Return available tickers from inner accessor."""
        return self._inner.get_available_tickers()


class CachedMacroAccessor(MacroDataAccessor):
    """Caching wrapper for MacroDataAccessor with TTL-based expiration.

    Provides transparent caching with configurable TTL (time-to-live).
    Useful for macro data that changes infrequently.

    Usage:
        inner = FREDMacroAccessor(loader)
        cached = CachedMacroAccessor(inner, cache_ttl_seconds=3600)

        # First call hits inner accessor
        vix1 = cached.read_macro_indicator("VIX", as_of)

        # Second call (within TTL) returns cached result
        vix2 = cached.read_macro_indicator("VIX", as_of)
    """

    def __init__(
        self,
        inner: MacroDataAccessor,
        cache_ttl_seconds: int = 86400,
        indicator_ttls: Optional[Dict[str, int]] = None,
    ):
        """Initialize caching wrapper.

        Args:
            inner: The underlying MacroDataAccessor to wrap
            cache_ttl_seconds: Default TTL in seconds (default: 24 hours)
            indicator_ttls: Optional per-indicator TTL overrides
        """
        self._inner = inner
        self._default_ttl = cache_ttl_seconds
        self._indicator_ttls = indicator_ttls or {}

        # Cache: key -> (data, expiry_time)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._stats = CacheStats()

    def read_macro_indicator(
        self,
        indicator: str,
        as_of: pd.Timestamp,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return cached macro indicator data.

        Args:
            indicator: Indicator name
            as_of: Cutoff date
            lookback_days: Optional lookback window

        Returns:
            DataFrame with indicator values (from cache if valid)
        """
        cache_key = self._make_indicator_key(indicator, as_of, lookback_days)

        # Check cache
        if cache_key in self._cache:
            data, expiry = self._cache[cache_key]
            if time.time() < expiry:
                self._stats.hits += 1
                return data.copy()
            else:
                # Expired
                del self._cache[cache_key]

        # Cache miss
        self._stats.misses += 1
        result = self._inner.read_macro_indicator(indicator, as_of, lookback_days)

        # Store in cache with TTL
        ttl = self._indicator_ttls.get(indicator, self._default_ttl)
        expiry = time.time() + ttl
        self._cache[cache_key] = (result.copy(), expiry)
        self._stats.total_bytes += _estimate_df_size(result)

        return result

    def get_regime(self, as_of: pd.Timestamp) -> Regime:
        """Return cached market regime.

        Regime detection is cached with default TTL.

        Args:
            as_of: Date for regime detection

        Returns:
            Regime enum value
        """
        cache_key = f"regime_{as_of}"

        # Check cache
        if cache_key in self._cache:
            data, expiry = self._cache[cache_key]
            if time.time() < expiry:
                self._stats.hits += 1
                return data
            else:
                del self._cache[cache_key]

        # Cache miss
        self._stats.misses += 1
        result = self._inner.get_regime(as_of)

        # Store with TTL
        expiry = time.time() + self._default_ttl
        self._cache[cache_key] = (result, expiry)

        return result

    def get_available_indicators(self) -> list[str]:
        """Return available indicators from inner accessor.

        Not cached - lightweight call.
        """
        return self._inner.get_available_indicators()

    def _make_indicator_key(
        self,
        indicator: str,
        as_of: pd.Timestamp,
        lookback_days: Optional[int],
    ) -> str:
        """Create cache key for indicator query."""
        return f"{indicator}_{as_of}_{lookback_days or 'ALL'}"

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._stats.total_bytes = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        # Count valid entries
        now = time.time()
        valid_entries = sum(1 for _, (_, exp) in self._cache.items() if exp > now)
        expired_entries = len(self._cache) - valid_entries

        return {
            **self._stats.to_dict(),
            "entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "default_ttl_seconds": self._default_ttl,
        }

    def set_indicator_ttl(self, indicator: str, ttl_seconds: int) -> None:
        """Configure TTL for a specific indicator.

        Args:
            indicator: Indicator name
            ttl_seconds: TTL in seconds
        """
        self._indicator_ttls[indicator] = ttl_seconds

    def purge_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries purged
        """
        now = time.time()
        expired_keys = [k for k, (_, exp) in self._cache.items() if exp <= now]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)
