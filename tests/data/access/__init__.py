"""Tests for data access module.

This package contains:
- mocks: Mock accessor implementations for testing
- builders: Data builders for generating synthetic test data
"""

from .mocks import (
    MockPriceAccessor,
    MockMacroAccessor,
    MockUniverseAccessor,
    MockReferenceAccessor,
    create_mock_context,
)
from .builders import (
    PriceDataBuilder,
    MacroDataBuilder,
    UniverseBuilder,
    TickerMetadataBuilder,
)

__all__ = [
    # Mocks
    'MockPriceAccessor',
    'MockMacroAccessor',
    'MockUniverseAccessor',
    'MockReferenceAccessor',
    'create_mock_context',
    # Builders
    'PriceDataBuilder',
    'MacroDataBuilder',
    'UniverseBuilder',
    'TickerMetadataBuilder',
]
