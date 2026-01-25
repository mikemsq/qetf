"""Shared test fixtures for QuantETF tests.

This module provides pytest fixtures that can be used across all test files.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tests.data.access.builders import (
    PriceDataBuilder,
    MacroDataBuilder,
    UniverseBuilder,
    TickerMetadataBuilder,
)
from tests.data.access.mocks import (
    MockPriceAccessor,
    MockMacroAccessor,
    MockUniverseAccessor,
    MockReferenceAccessor,
    create_mock_context,
)
from quantetf.data.access.context import DataAccessContext
from quantetf.data.access.types import Regime


# =============================================================================
# Legacy Fixtures (backward compatibility)
# =============================================================================

@pytest.fixture
def sample_etf_tickers():
    """Return list of sample ETF tickers for testing.
    
    Returns:
        List of commonly used ETF ticker symbols
    """
    return ["SPY", "QQQ", "IWM", "EFA", "AGG"]


@pytest.fixture
def date_range():
    """Return standard test date range (5 years of history).
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    end = datetime.now()
    start = end - timedelta(days=365*5)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@pytest.fixture
def short_date_range():
    """Return short test date range (1 year of history).
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    return "2020-01-01", "2020-12-31"


@pytest.fixture
def sample_price_data():
    """Return sample ETF price data for testing.

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price_Field) for single ticker 'SPY'
    """
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
    n = len(dates)

    # Create realistic-looking price data
    base_price = 100
    returns = np.arange(n) * 0.1  # Slight upward trend
    noise = np.arange(n) % 5 - 2  # Add some noise

    close_prices = base_price + returns + noise

    # Create simple DataFrame first
    simple_df = pd.DataFrame({
        "Open": close_prices - 0.5,
        "High": close_prices + 1.0,
        "Low": close_prices - 1.0,
        "Close": close_prices,
        "Volume": 1000000,
    }, index=dates)

    # Convert to MultiIndex format (Ticker, Price_Field)
    simple_df.columns = pd.MultiIndex.from_product(
        [['SPY'], simple_df.columns],
        names=['Ticker', 'Price']
    )

    return simple_df


@pytest.fixture
def sample_price_data_multi_ticker():
    """Return sample price data for multiple tickers.

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price_Field)
    """
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
    n = len(dates)

    tickers = ["SPY", "QQQ"]
    data_dict = {}

    for ticker in tickers:
        base_price = 100 if ticker == "SPY" else 200
        returns = np.arange(n) * 0.1
        noise = np.arange(n) % 5 - 2

        close_prices = base_price + returns + noise

        data_dict[ticker] = pd.DataFrame({
            "Open": close_prices - 0.5,
            "High": close_prices + 1.0,
            "Low": close_prices - 1.0,
            "Close": close_prices,
            "Volume": 1000000,
        }, index=dates)

    # Combine into MultiIndex DataFrame with names
    combined = pd.concat(data_dict, axis=1)
    combined.columns.names = ['Ticker', 'Price']
    return combined


# =============================================================================
# Data Access Layer (DAL) Fixtures
# =============================================================================

@pytest.fixture
def dal_price_data() -> pd.DataFrame:
    """Return synthetic price data for DAL testing.

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price) for 15 ETFs
    """
    tickers = UniverseBuilder.default_etf_universe()
    return (
        PriceDataBuilder()
        .with_tickers(tickers)
        .with_date_range('2020-01-01', '2023-12-31')
        .with_seed(42)
        .build()
    )


@pytest.fixture
def dal_price_data_short() -> pd.DataFrame:
    """Return short price data for quick tests.

    Returns:
        DataFrame with 1 year of data for 5 tickers
    """
    return (
        PriceDataBuilder()
        .with_tickers(['SPY', 'QQQ', 'IWM', 'AGG', 'GLD'])
        .with_date_range('2023-01-01', '2023-12-31')
        .with_seed(42)
        .build()
    )


@pytest.fixture
def dal_macro_data() -> dict[str, pd.DataFrame]:
    """Return synthetic macro indicator data.

    Returns:
        Dict with VIX, YIELD_SPREAD, and UNEMPLOYMENT indicators
    """
    builder = MacroDataBuilder().with_date_range('2020-01-01', '2023-12-31').with_seed(42)
    return {
        'VIX': builder.build_vix(),
        'YIELD_SPREAD': builder.build_yield_spread(),
        'UNEMPLOYMENT': builder.build_unemployment(),
    }


@pytest.fixture
def dal_universes() -> dict[str, list[str]]:
    """Return test universe definitions.

    Returns:
        Dict mapping universe names to ticker lists
    """
    return {
        'etf_tier1': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
        'etf_tier2': UniverseBuilder.default_etf_universe(),
        'bonds': ['AGG', 'LQD', 'HYG', 'TLT'],
        'sectors': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI'],
    }


@pytest.fixture
def dal_ticker_metadata() -> dict:
    """Return test ticker metadata.

    Returns:
        Dict mapping tickers to TickerMetadata
    """
    return TickerMetadataBuilder.default_etf_metadata()


@pytest.fixture
def mock_price_accessor(dal_price_data) -> MockPriceAccessor:
    """Return MockPriceAccessor with synthetic data."""
    return MockPriceAccessor(dal_price_data)


@pytest.fixture
def mock_price_accessor_short(dal_price_data_short) -> MockPriceAccessor:
    """Return MockPriceAccessor with short synthetic data."""
    return MockPriceAccessor(dal_price_data_short)


@pytest.fixture
def mock_macro_accessor(dal_macro_data) -> MockMacroAccessor:
    """Return MockMacroAccessor with synthetic data."""
    return MockMacroAccessor(dal_macro_data)


@pytest.fixture
def mock_universe_accessor(dal_universes) -> MockUniverseAccessor:
    """Return MockUniverseAccessor with test universes."""
    return MockUniverseAccessor(dal_universes)


@pytest.fixture
def mock_reference_accessor(dal_ticker_metadata) -> MockReferenceAccessor:
    """Return MockReferenceAccessor with test metadata."""
    return MockReferenceAccessor(dal_ticker_metadata)


@pytest.fixture
def mock_data_context(
    mock_price_accessor,
    mock_macro_accessor,
    mock_universe_accessor,
    mock_reference_accessor,
) -> DataAccessContext:
    """Return fully configured mock DataAccessContext.

    This is the primary fixture for testing components that
    depend on the Data Access Layer.
    """
    return DataAccessContext(
        prices=mock_price_accessor,
        macro=mock_macro_accessor,
        universes=mock_universe_accessor,
        references=mock_reference_accessor,
    )


@pytest.fixture
def mock_data_context_short(dal_price_data_short, dal_macro_data) -> DataAccessContext:
    """Return mock DataAccessContext with short data for quick tests."""
    tickers = list(dal_price_data_short.columns.get_level_values('Ticker').unique())

    return DataAccessContext(
        prices=MockPriceAccessor(dal_price_data_short),
        macro=MockMacroAccessor(dal_macro_data),
        universes=MockUniverseAccessor({'default': tickers}),
        references=MockReferenceAccessor(
            TickerMetadataBuilder().with_tickers(tickers).build()
        ),
    )


@pytest.fixture
def mock_regime_accessor() -> MockMacroAccessor:
    """Return MockMacroAccessor with regime testing data.

    Pre-configured with regime periods for testing regime-aware strategies.
    """
    builder = MacroDataBuilder().with_date_range('2020-01-01', '2023-12-31').with_seed(42)

    accessor = MockMacroAccessor(
        indicators={
            'VIX': builder.build_vix(
                spike_dates=['2020-03-16', '2022-01-24'],
                spike_levels=[82.0, 38.0],
            ),
        },
        default_regime=Regime.RISK_ON,
    )

    # Set regime periods
    accessor.set_regime(
        pd.Timestamp('2020-03-01'),
        pd.Timestamp('2020-04-30'),
        Regime.HIGH_VOL,
    )
    accessor.set_regime(
        pd.Timestamp('2022-01-01'),
        pd.Timestamp('2022-03-31'),
        Regime.ELEVATED_VOL,
    )

    return accessor


@pytest.fixture
def crash_price_data() -> pd.DataFrame:
    """Return price data with a market crash event for drawdown testing."""
    return (
        PriceDataBuilder()
        .with_tickers(['SPY', 'QQQ', 'IWM'])
        .with_date_range('2020-01-01', '2020-12-31')
        .with_seed(42)
        .build_with_crash(
            crash_date='2020-03-16',
            crash_magnitude=0.35,
            recovery_days=120,
        )
    )


@pytest.fixture
def gappy_price_data() -> pd.DataFrame:
    """Return price data with gaps for incomplete data handling tests."""
    return (
        PriceDataBuilder()
        .with_tickers(['SPY', 'QQQ'])
        .with_date_range('2023-01-01', '2023-12-31')
        .with_seed(42)
        .build_with_gaps(gap_ratio=0.05)
    )
