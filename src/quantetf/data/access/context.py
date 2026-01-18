"""Data Access Context - unified container for all DAL accessors."""

from dataclasses import dataclass
from .abstract import (
    PriceDataAccessor,
    MacroDataAccessor,
    UniverseDataAccessor,
    ReferenceDataAccessor,
)


@dataclass(frozen=True)
class DataAccessContext:
    """Container for all DAL accessors.
    
    Provides a single, unified interface to all data access functionality.
    Immutable to ensure consistency across components.
    
    Usage:
        ctx = DataAccessContext(
            prices=price_accessor,
            macro=macro_accessor,
            universes=universe_accessor,
            references=ref_accessor,
        )
        
        # Pass to components
        engine = SimpleBacktestEngine(data_access=ctx)
        
        # Use accessors
        prices = ctx.prices.read_prices_as_of(as_of)
        macro = ctx.macro.read_macro_indicator("VIX", as_of)
        universe = ctx.universes.get_universe("etf_tier1")
    """
    
    prices: PriceDataAccessor
    macro: MacroDataAccessor
    universes: UniverseDataAccessor
    references: ReferenceDataAccessor
