"""Centralized cache management for Data Access Layer.

Provides a single point of control for managing caches across all
accessor types in the DAL system.

Usage:
    from quantetf.data.access.cache_manager import CacheManager
    from quantetf.data.access.caching import CachedPriceAccessor, CachedMacroAccessor

    # Create cached accessors
    cached_prices = CachedPriceAccessor(inner_prices)
    cached_macro = CachedMacroAccessor(inner_macro)

    # Register with manager
    manager = CacheManager()
    manager.register("prices", cached_prices)
    manager.register("macro", cached_macro)

    # Global operations
    manager.clear_all_caches()
    stats = manager.get_global_stats()
"""

from typing import Any, Dict, Optional, Union

from .caching import CachedMacroAccessor, CachedPriceAccessor


class CacheManager:
    """Centralized cache management for DAL accessors.

    Tracks all cached accessors and provides unified operations:
    - Clear all caches
    - Get combined statistics
    - Configure TTLs across accessors
    - Monitor cache health

    Usage:
        manager = CacheManager()
        manager.register("prices", cached_price_accessor)
        manager.register("macro", cached_macro_accessor)

        # Get stats
        stats = manager.get_global_stats()

        # Clear everything
        manager.clear_all_caches()
    """

    def __init__(self):
        """Initialize cache manager."""
        self._accessors: Dict[str, Union[CachedPriceAccessor, CachedMacroAccessor]] = {}

    def register(
        self,
        name: str,
        accessor: Union[CachedPriceAccessor, CachedMacroAccessor],
    ) -> None:
        """Register a cached accessor for management.

        Args:
            name: Unique name for this accessor
            accessor: CachedPriceAccessor or CachedMacroAccessor instance

        Raises:
            ValueError: If name already registered
            TypeError: If accessor is not a cached accessor type
        """
        if name in self._accessors:
            raise ValueError(f"Accessor already registered: {name}")

        if not isinstance(accessor, (CachedPriceAccessor, CachedMacroAccessor)):
            raise TypeError(
                f"Expected CachedPriceAccessor or CachedMacroAccessor, "
                f"got {type(accessor).__name__}"
            )

        self._accessors[name] = accessor

    def unregister(self, name: str) -> None:
        """Unregister a cached accessor.

        Args:
            name: Name of accessor to unregister

        Raises:
            KeyError: If name not registered
        """
        if name not in self._accessors:
            raise KeyError(f"Accessor not registered: {name}")

        del self._accessors[name]

    def clear_all_caches(self) -> Dict[str, int]:
        """Clear caches for all registered accessors.

        Returns:
            Dictionary mapping accessor name to number of entries cleared
        """
        results = {}
        for name, accessor in self._accessors.items():
            stats = accessor.get_cache_stats()
            entries_before = stats.get("entries", 0)
            accessor.clear_cache()
            results[name] = entries_before

        return results

    def get_global_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics across all accessors.

        Returns:
            Dictionary with:
            - total_hits: Sum of hits across all caches
            - total_misses: Sum of misses
            - total_bytes: Total memory used
            - global_hit_rate: Overall hit rate percentage
            - per_accessor: Individual stats for each accessor
        """
        total_hits = 0
        total_misses = 0
        total_bytes = 0
        total_evictions = 0
        per_accessor = {}

        for name, accessor in self._accessors.items():
            stats = accessor.get_cache_stats()
            total_hits += stats.get("hits", 0)
            total_misses += stats.get("misses", 0)
            total_bytes += stats.get("total_bytes", 0)
            total_evictions += stats.get("evictions", 0)
            per_accessor[name] = stats

        total_requests = total_hits + total_misses
        global_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_bytes": total_bytes,
            "total_evictions": total_evictions,
            "global_hit_rate_pct": round(global_hit_rate, 2),
            "registered_accessors": len(self._accessors),
            "per_accessor": per_accessor,
        }

    def configure_ttl(
        self,
        accessor_type: str,
        ttl_seconds: int,
        indicator: Optional[str] = None,
    ) -> None:
        """Configure TTL for macro accessors.

        Args:
            accessor_type: "macro" to target macro accessors
            ttl_seconds: New TTL in seconds
            indicator: Optional specific indicator (for MacroAccessor)

        Raises:
            ValueError: If accessor_type not recognized
        """
        if accessor_type != "macro":
            raise ValueError(f"TTL configuration only applies to 'macro' accessors")

        for accessor in self._accessors.values():
            if isinstance(accessor, CachedMacroAccessor):
                if indicator:
                    accessor.set_indicator_ttl(indicator, ttl_seconds)
                else:
                    accessor._default_ttl = ttl_seconds

    def get_accessor(
        self,
        name: str,
    ) -> Union[CachedPriceAccessor, CachedMacroAccessor]:
        """Get a registered accessor by name.

        Args:
            name: Accessor name

        Returns:
            The registered accessor

        Raises:
            KeyError: If name not registered
        """
        if name not in self._accessors:
            raise KeyError(f"Accessor not registered: {name}")

        return self._accessors[name]

    def list_accessors(self) -> list[str]:
        """List all registered accessor names.

        Returns:
            List of accessor names
        """
        return list(self._accessors.keys())

    def purge_expired(self) -> Dict[str, int]:
        """Purge expired entries from all macro caches.

        Returns:
            Dictionary mapping accessor name to count of purged entries
        """
        results = {}
        for name, accessor in self._accessors.items():
            if isinstance(accessor, CachedMacroAccessor):
                count = accessor.purge_expired()
                results[name] = count

        return results


# Global cache manager singleton (optional convenience)
_global_manager: Optional[CacheManager] = None


def get_global_cache_manager() -> CacheManager:
    """Get or create the global cache manager singleton.

    Returns:
        The global CacheManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = CacheManager()
    return _global_manager


def reset_global_cache_manager() -> None:
    """Reset the global cache manager (mainly for testing)."""
    global _global_manager
    _global_manager = None
