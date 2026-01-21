"""Data Access Layer (DAL) - unified interface for all data access operations.

This package provides a clean, type-safe interface for accessing data from
various sources (snapshots, live APIs, etc.) without tight coupling to
specific implementations.

Core Components:
- PriceDataAccessor: OHLCV price data
- MacroDataAccessor: Macro indicators and market regime
- UniverseDataAccessor: Ticker set definitions
- ReferenceDataAccessor: Static reference data
- DataAccessContext: Unified container for all accessors
- DataAccessFactory: Factory for creating configured accessors

Caching Components:
- CachedPriceAccessor: LRU-cached price accessor
- CachedMacroAccessor: TTL-cached macro accessor
- CacheManager: Centralized cache management

Usage:
    from quantetf.data.access import DataAccessFactory

    # Create context with caching enabled (default)
    ctx = DataAccessFactory.create_context(
        config={"snapshot_path": "data/snapshots/latest/data.parquet"}
    )

    # Use accessors
    prices = ctx.prices.read_prices_as_of(as_of="2024-01-31")
    regime = ctx.macro.get_regime(as_of="2024-01-31")
    universe = ctx.universes.get_universe("etf_tier1")
"""

from .abstract import (
    PriceDataAccessor,
    MacroDataAccessor,
    UniverseDataAccessor,
    ReferenceDataAccessor,
)
from .context import DataAccessContext
from .factory import DataAccessFactory
from .types import Regime, TickerMetadata, ExchangeInfo, DataAccessMetadata
from .universe import ConfigFileUniverseAccessor
from .reference import StaticReferenceDataAccessor
from .caching import CachedPriceAccessor, CachedMacroAccessor
from .cache_manager import CacheManager, get_global_cache_manager

__all__ = [
    # Abstract interfaces
    "PriceDataAccessor",
    "MacroDataAccessor",
    "UniverseDataAccessor",
    "ReferenceDataAccessor",
    # Context and factory
    "DataAccessContext",
    "DataAccessFactory",
    # Concrete implementations
    "ConfigFileUniverseAccessor",
    "StaticReferenceDataAccessor",
    # Caching
    "CachedPriceAccessor",
    "CachedMacroAccessor",
    "CacheManager",
    "get_global_cache_manager",
    # Types
    "Regime",
    "TickerMetadata",
    "ExchangeInfo",
    "DataAccessMetadata",
]
