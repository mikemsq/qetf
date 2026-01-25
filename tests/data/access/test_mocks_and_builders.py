"""Tests for DAL test utilities (mocks and builders).

These tests verify that the mock accessors and data builders work correctly.
"""

import pytest
import pandas as pd
import numpy as np

from quantetf.data.access.types import Regime, TickerMetadata
from quantetf.data.access.context import DataAccessContext

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


class TestPriceDataBuilder:
    """Test PriceDataBuilder functionality."""

    def test_build_basic(self):
        """Test basic price data generation."""
        data = PriceDataBuilder().build()

        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.columns, pd.MultiIndex)
        assert data.columns.names == ['Ticker', 'Price']
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_build_with_tickers(self):
        """Test building with specific tickers."""
        tickers = ['SPY', 'QQQ', 'IWM']
        data = PriceDataBuilder().with_tickers(tickers).build()

        result_tickers = list(data.columns.get_level_values('Ticker').unique())
        assert result_tickers == tickers

    def test_build_with_date_range(self):
        """Test building with specific date range."""
        data = (
            PriceDataBuilder()
            .with_date_range('2022-01-01', '2022-12-31')
            .build()
        )

        assert data.index.min() >= pd.Timestamp('2022-01-01')
        assert data.index.max() <= pd.Timestamp('2022-12-31')

    def test_build_with_seed_reproducibility(self):
        """Test that seed produces reproducible data."""
        data1 = PriceDataBuilder().with_seed(42).build()
        data2 = PriceDataBuilder().with_seed(42).build()

        pd.testing.assert_frame_equal(data1, data2)

    def test_build_with_different_seeds(self):
        """Test that different seeds produce different data."""
        data1 = PriceDataBuilder().with_seed(42).build()
        data2 = PriceDataBuilder().with_seed(123).build()

        # Data should be different
        assert not data1.equals(data2)

    def test_ohlc_constraints(self):
        """Test that OHLC constraints are satisfied."""
        data = PriceDataBuilder().with_seed(42).build()

        for ticker in data.columns.get_level_values('Ticker').unique():
            opens = data[(ticker, 'Open')]
            highs = data[(ticker, 'High')]
            lows = data[(ticker, 'Low')]
            closes = data[(ticker, 'Close')]

            # High should be >= Open, Close
            assert (highs >= opens).all()
            assert (highs >= closes).all()

            # Low should be <= Open, Close
            assert (lows <= opens).all()
            assert (lows <= closes).all()

    def test_build_with_gaps(self):
        """Test building data with gaps."""
        full_data = PriceDataBuilder().with_seed(42).build()
        gappy_data = PriceDataBuilder().with_seed(42).build_with_gaps(gap_ratio=0.1)

        # Gappy data should have fewer rows
        assert len(gappy_data) < len(full_data)
        assert len(gappy_data) > len(full_data) * 0.85  # Should be roughly 90%

    def test_build_with_crash(self):
        """Test building data with crash event."""
        data = (
            PriceDataBuilder()
            .with_tickers(['SPY'])
            .with_date_range('2020-01-01', '2020-12-31')
            .with_seed(42)
            .build_with_crash(
                crash_date='2020-03-16',
                crash_magnitude=0.30,
                recovery_days=60,
            )
        )

        crash_date = pd.Timestamp('2020-03-16')
        pre_crash = data[data.index < crash_date - pd.Timedelta(days=5)]
        at_crash = data[data.index == crash_date]

        if not at_crash.empty:
            pre_level = pre_crash[('SPY', 'Close')].iloc[-1]
            crash_level = at_crash[('SPY', 'Close')].iloc[0]
            # Should show significant drop
            drop = (pre_level - crash_level) / pre_level
            assert drop > 0.20  # At least 20% drop


class TestMacroDataBuilder:
    """Test MacroDataBuilder functionality."""

    def test_build_vix_basic(self):
        """Test basic VIX generation."""
        data = MacroDataBuilder().build_vix()

        assert isinstance(data, pd.DataFrame)
        assert 'VIX' in data.columns
        assert isinstance(data.index, pd.DatetimeIndex)
        # VIX should be positive
        assert (data['VIX'] > 0).all()

    def test_build_vix_with_spikes(self):
        """Test VIX generation with volatility spikes."""
        data = MacroDataBuilder().build_vix(
            spike_dates=['2023-03-15'],
            spike_levels=[50.0],
        )

        spike_date = pd.Timestamp('2023-03-15')
        if spike_date in data.index:
            assert data.loc[spike_date, 'VIX'] == 50.0

    def test_build_yield_spread(self):
        """Test yield spread generation."""
        data = MacroDataBuilder().build_yield_spread()

        assert isinstance(data, pd.DataFrame)
        assert 'YIELD_SPREAD' in data.columns

    def test_build_unemployment(self):
        """Test unemployment rate generation."""
        data = MacroDataBuilder().build_unemployment()

        assert isinstance(data, pd.DataFrame)
        assert 'UNEMPLOYMENT' in data.columns
        # Unemployment should be positive and reasonable
        assert (data['UNEMPLOYMENT'] >= 2.0).all()
        assert (data['UNEMPLOYMENT'] <= 15.0).all()


class TestMockPriceAccessor:
    """Test MockPriceAccessor functionality."""

    @pytest.fixture
    def price_data(self):
        return (
            PriceDataBuilder()
            .with_tickers(['SPY', 'QQQ', 'IWM'])
            .with_date_range('2023-01-01', '2023-12-31')
            .with_seed(42)
            .build()
        )

    @pytest.fixture
    def accessor(self, price_data):
        return MockPriceAccessor(price_data)

    def test_initialization(self, accessor, price_data):
        """Test accessor initialization."""
        assert accessor._data.equals(price_data)
        assert set(accessor._tickers) == {'SPY', 'QQQ', 'IWM'}

    def test_initialization_requires_multiindex(self):
        """Test that initialization fails without MultiIndex columns."""
        bad_data = pd.DataFrame({'A': [1, 2, 3]}, index=pd.date_range('2023-01-01', periods=3))
        with pytest.raises(ValueError, match="MultiIndex"):
            MockPriceAccessor(bad_data)

    def test_read_prices_as_of_basic(self, accessor):
        """Test basic point-in-time read."""
        as_of = pd.Timestamp('2023-06-15')
        result = accessor.read_prices_as_of(as_of)

        # All dates should be strictly before as_of
        assert (result.index < as_of).all()
        assert isinstance(result.columns, pd.MultiIndex)

    def test_read_prices_as_of_no_lookahead(self, accessor):
        """Test that no data on/after as_of is returned."""
        as_of = pd.Timestamp('2023-06-15')
        result = accessor.read_prices_as_of(as_of)

        # Strict inequality - as_of date should not be included
        assert as_of not in result.index

    def test_read_prices_as_of_with_tickers(self, accessor):
        """Test read with ticker filter."""
        result = accessor.read_prices_as_of(
            pd.Timestamp('2023-06-15'),
            tickers=['SPY', 'QQQ'],
        )

        result_tickers = list(result.columns.get_level_values('Ticker').unique())
        assert set(result_tickers) == {'SPY', 'QQQ'}

    def test_read_prices_as_of_with_lookback(self, accessor):
        """Test read with lookback window."""
        as_of = pd.Timestamp('2023-06-15')
        result = accessor.read_prices_as_of(as_of, lookback_days=30)

        cutoff = as_of - pd.Timedelta(days=30)
        assert (result.index >= cutoff).all()
        assert (result.index < as_of).all()

    def test_read_prices_as_of_missing_ticker(self, accessor):
        """Test that error is raised for missing ticker."""
        with pytest.raises(ValueError, match="not found"):
            accessor.read_prices_as_of(
                pd.Timestamp('2023-06-15'),
                tickers=['NONEXISTENT'],
            )

    def test_read_ohlcv_range(self, accessor):
        """Test reading date range."""
        start = pd.Timestamp('2023-03-01')
        end = pd.Timestamp('2023-03-31')
        result = accessor.read_ohlcv_range(start, end)

        assert (result.index >= start).all()
        assert (result.index <= end).all()

    def test_get_latest_price_date(self, accessor, price_data):
        """Test getting latest date."""
        latest = accessor.get_latest_price_date()
        assert latest == price_data.index.max()

    def test_validate_data_availability(self, accessor):
        """Test data availability validation."""
        result = accessor.validate_data_availability(
            ['SPY', 'QQQ', 'NONEXISTENT'],
            pd.Timestamp('2023-06-15'),
        )

        assert result['SPY'] == True
        assert result['QQQ'] == True
        assert result['NONEXISTENT'] == False


class TestMockMacroAccessor:
    """Test MockMacroAccessor functionality."""

    @pytest.fixture
    def accessor(self):
        indicators = {
            'VIX': MacroDataBuilder().build_vix(),
            'YIELD_SPREAD': MacroDataBuilder().build_yield_spread(),
        }
        return MockMacroAccessor(indicators)

    def test_read_macro_indicator(self, accessor):
        """Test reading macro indicator."""
        result = accessor.read_macro_indicator(
            'VIX',
            pd.Timestamp('2023-06-15'),
        )

        assert isinstance(result, pd.DataFrame)
        assert (result.index < pd.Timestamp('2023-06-15')).all()

    def test_read_macro_indicator_missing(self, accessor):
        """Test error for missing indicator."""
        with pytest.raises(ValueError, match="not available"):
            accessor.read_macro_indicator(
                'NONEXISTENT',
                pd.Timestamp('2023-06-15'),
            )

    def test_get_regime_default(self, accessor):
        """Test default regime."""
        regime = accessor.get_regime(pd.Timestamp('2023-06-15'))
        assert regime == Regime.RISK_ON

    def test_get_regime_configured(self):
        """Test configured regime periods."""
        accessor = MockMacroAccessor(default_regime=Regime.RISK_ON)
        accessor.set_regime(
            pd.Timestamp('2023-03-01'),
            pd.Timestamp('2023-03-31'),
            Regime.HIGH_VOL,
        )

        # Within period
        assert accessor.get_regime(pd.Timestamp('2023-03-15')) == Regime.HIGH_VOL
        # Outside period
        assert accessor.get_regime(pd.Timestamp('2023-04-15')) == Regime.RISK_ON

    def test_get_available_indicators(self, accessor):
        """Test getting available indicators."""
        indicators = accessor.get_available_indicators()
        assert set(indicators) == {'VIX', 'YIELD_SPREAD'}


class TestMockUniverseAccessor:
    """Test MockUniverseAccessor functionality."""

    @pytest.fixture
    def accessor(self):
        return MockUniverseAccessor(
            universes={
                'tier1': ['SPY', 'QQQ', 'IWM'],
                'bonds': ['AGG', 'TLT'],
            }
        )

    def test_get_universe(self, accessor):
        """Test getting static universe."""
        result = accessor.get_universe('tier1')
        assert result == ['SPY', 'QQQ', 'IWM']

    def test_get_universe_missing(self, accessor):
        """Test error for missing universe."""
        with pytest.raises(ValueError, match="not found"):
            accessor.get_universe('nonexistent')

    def test_list_available_universes(self, accessor):
        """Test listing universes."""
        universes = accessor.list_available_universes()
        assert set(universes) == {'tier1', 'bonds'}

    def test_graduated_universe(self):
        """Test time-based universe membership."""
        accessor = MockUniverseAccessor(
            graduated_universes={
                'growing': {
                    pd.Timestamp('2023-01-01'): ['SPY'],
                    pd.Timestamp('2023-06-01'): ['SPY', 'QQQ'],
                    pd.Timestamp('2023-12-01'): ['SPY', 'QQQ', 'IWM'],
                }
            }
        )

        # Before first snapshot
        result = accessor.get_universe_as_of('growing', pd.Timestamp('2022-12-01'))
        assert result == []

        # After first snapshot
        result = accessor.get_universe_as_of('growing', pd.Timestamp('2023-03-01'))
        assert result == ['SPY']

        # After second snapshot
        result = accessor.get_universe_as_of('growing', pd.Timestamp('2023-08-01'))
        assert result == ['SPY', 'QQQ']


class TestMockReferenceAccessor:
    """Test MockReferenceAccessor functionality."""

    @pytest.fixture
    def accessor(self):
        metadata = TickerMetadataBuilder.default_etf_metadata()
        return MockReferenceAccessor(metadata)

    def test_get_ticker_info(self, accessor):
        """Test getting ticker metadata."""
        info = accessor.get_ticker_info('SPY')

        assert isinstance(info, TickerMetadata)
        assert info.ticker == 'SPY'
        assert info.name == 'SPDR S&P 500 ETF'

    def test_get_ticker_info_missing(self, accessor):
        """Test error for missing ticker."""
        with pytest.raises(ValueError, match="not found"):
            accessor.get_ticker_info('NONEXISTENT')

    def test_get_sector_mapping(self, accessor):
        """Test getting sector mapping."""
        mapping = accessor.get_sector_mapping()

        assert isinstance(mapping, dict)
        assert 'SPY' in mapping

    def test_get_exchange_info(self, accessor):
        """Test getting exchange info."""
        exchanges = accessor.get_exchange_info()

        assert 'NYSE' in exchanges
        assert 'NASDAQ' in exchanges


class TestCreateMockContext:
    """Test create_mock_context helper."""

    def test_create_with_defaults(self):
        """Test creating context with default data."""
        ctx = create_mock_context()

        assert isinstance(ctx, DataAccessContext)
        assert isinstance(ctx.prices, MockPriceAccessor)
        assert isinstance(ctx.macro, MockMacroAccessor)
        assert isinstance(ctx.universes, MockUniverseAccessor)
        assert isinstance(ctx.references, MockReferenceAccessor)

    def test_create_with_custom_prices(self):
        """Test creating context with custom price data."""
        prices = (
            PriceDataBuilder()
            .with_tickers(['AAPL', 'MSFT'])
            .with_seed(42)
            .build()
        )
        ctx = create_mock_context(prices=prices)

        tickers = ctx.prices.get_available_tickers()
        assert set(tickers) == {'AAPL', 'MSFT'}


class TestConfTestFixtures:
    """Test that conftest fixtures work correctly."""

    def test_dal_price_data(self, dal_price_data):
        """Test dal_price_data fixture."""
        assert isinstance(dal_price_data, pd.DataFrame)
        assert len(dal_price_data.columns.get_level_values('Ticker').unique()) == 15

    def test_mock_data_context(self, mock_data_context):
        """Test mock_data_context fixture."""
        assert isinstance(mock_data_context, DataAccessContext)

        # Test that accessors work
        prices = mock_data_context.prices.read_prices_as_of(pd.Timestamp('2023-06-15'))
        assert len(prices) > 0

    def test_mock_data_context_short(self, mock_data_context_short):
        """Test mock_data_context_short fixture."""
        assert isinstance(mock_data_context_short, DataAccessContext)

        tickers = mock_data_context_short.prices.get_available_tickers()
        assert len(tickers) == 5

    def test_crash_price_data(self, crash_price_data):
        """Test crash_price_data fixture."""
        assert isinstance(crash_price_data, pd.DataFrame)
        # Data should cover 2020
        assert crash_price_data.index.min().year == 2020

    def test_gappy_price_data(self, gappy_price_data):
        """Test gappy_price_data fixture."""
        assert isinstance(gappy_price_data, pd.DataFrame)

        # Should have fewer rows than a full year (252 business days)
        assert len(gappy_price_data) < 252
