"""Mock implementations of DAL accessors for testing.

These mock accessors implement the abstract DAL interfaces with in-memory
data storage, allowing tests to run without external dependencies.

Usage:
    from tests.data.access.mocks import MockPriceAccessor, MockMacroAccessor
    from tests.data.access.builders import PriceDataBuilder

    # Build test data
    prices = PriceDataBuilder().with_tickers(['SPY', 'QQQ']).build()

    # Create mock accessor
    accessor = MockPriceAccessor(prices)

    # Use in tests
    data = accessor.read_prices_as_of(pd.Timestamp('2023-06-01'))
"""

from typing import Optional
import pandas as pd

from quantetf.data.access.abstract import (
    PriceDataAccessor,
    MacroDataAccessor,
    UniverseDataAccessor,
    ReferenceDataAccessor,
)
from quantetf.data.access.types import Regime, TickerMetadata, ExchangeInfo


class MockPriceAccessor(PriceDataAccessor):
    """Mock price data accessor for testing.

    Stores price data in memory and implements point-in-time access
    with strict lookahead prevention.

    Args:
        data: DataFrame with MultiIndex columns (Ticker, Price) and DatetimeIndex
    """

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex columns (Ticker, Price)")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        self._data = data
        self._tickers = list(data.columns.get_level_values('Ticker').unique())

    def read_prices_as_of(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return OHLCV prices for all dates < as_of."""
        # Strict inequality - no lookahead
        mask = self._data.index < as_of

        if lookback_days is not None:
            cutoff = as_of - pd.Timedelta(days=lookback_days)
            mask = mask & (self._data.index >= cutoff)

        result = self._data.loc[mask].copy()

        if result.empty:
            raise ValueError(f"No data available before {as_of}")

        if tickers is not None:
            missing = set(tickers) - set(self._tickers)
            if missing:
                raise ValueError(f"Tickers not found: {missing}")
            result = result.loc[:, tickers]

        return result

    def read_ohlcv_range(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return OHLCV for closed date range [start, end]."""
        mask = (self._data.index >= start) & (self._data.index <= end)
        result = self._data.loc[mask].copy()

        if result.empty:
            raise ValueError(f"No price data available in range [{start}, {end}]")

        if tickers is not None:
            missing = set(tickers) - set(self._tickers)
            if missing:
                raise ValueError(f"Tickers not found: {missing}")
            result = result.loc[:, tickers]

        return result

    def get_latest_price_date(self) -> pd.Timestamp:
        """Return most recent date with available price data."""
        return self._data.index.max()

    def validate_data_availability(
        self,
        tickers: list[str],
        as_of: pd.Timestamp,
    ) -> dict[str, bool]:
        """Check which tickers have data available as of date."""
        result = {}
        pit_data = self._data[self._data.index < as_of]

        for ticker in tickers:
            if ticker not in self._tickers:
                result[ticker] = False
            elif pit_data.empty:
                result[ticker] = False
            else:
                # Check if ticker has non-null data
                ticker_data = pit_data.loc[:, ticker]
                result[ticker] = not ticker_data.isnull().all().all()

        return result

    def get_available_tickers(self) -> list[str]:
        """Return list of available tickers."""
        return self._tickers.copy()

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return (start_date, end_date) of available data."""
        return (self._data.index.min(), self._data.index.max())


class MockMacroAccessor(MacroDataAccessor):
    """Mock macro data accessor for testing.

    Stores macro indicator data in memory with configurable regime detection.

    Args:
        indicators: Dict mapping indicator name to DataFrame with values
        regime_map: Optional dict mapping date ranges to regimes
    """

    def __init__(
        self,
        indicators: Optional[dict[str, pd.DataFrame]] = None,
        regime_map: Optional[dict[tuple[pd.Timestamp, pd.Timestamp], Regime]] = None,
        default_regime: Regime = Regime.RISK_ON,
    ):
        self._indicators = indicators or {}
        self._regime_map = regime_map or {}
        self._default_regime = default_regime

    def read_macro_indicator(
        self,
        indicator: str,
        as_of: pd.Timestamp,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return time series for a macro indicator."""
        if indicator not in self._indicators:
            raise ValueError(f"Indicator not available: {indicator}")

        data = self._indicators[indicator]
        mask = data.index < as_of

        if lookback_days is not None:
            cutoff = as_of - pd.Timedelta(days=lookback_days)
            mask = mask & (data.index >= cutoff)

        result = data.loc[mask].copy()

        if result.empty:
            raise ValueError(f"No data available for {indicator} before {as_of}")

        return result

    def get_regime(self, as_of: pd.Timestamp) -> Regime:
        """Detect current market regime as of date."""
        for (start, end), regime in self._regime_map.items():
            if start <= as_of <= end:
                return regime
        return self._default_regime

    def get_available_indicators(self) -> list[str]:
        """Return list of available macro indicators."""
        return list(self._indicators.keys())

    def set_regime(self, start: pd.Timestamp, end: pd.Timestamp, regime: Regime):
        """Set regime for a date range (test helper)."""
        self._regime_map[(start, end)] = regime

    def add_indicator(self, name: str, data: pd.DataFrame):
        """Add indicator data (test helper)."""
        self._indicators[name] = data


class MockUniverseAccessor(UniverseDataAccessor):
    """Mock universe accessor for testing.

    Stores universe definitions in memory with optional time-based filtering.

    Args:
        universes: Dict mapping universe name to list of tickers
        graduated_universes: Dict for time-based universe membership
    """

    def __init__(
        self,
        universes: Optional[dict[str, list[str]]] = None,
        graduated_universes: Optional[dict[str, dict[pd.Timestamp, list[str]]]] = None,
    ):
        self._universes = universes or {}
        self._graduated = graduated_universes or {}

    def get_universe(self, universe_name: str) -> list[str]:
        """Get current/latest universe tickers."""
        if universe_name in self._universes:
            return self._universes[universe_name].copy()

        if universe_name in self._graduated:
            # Return latest snapshot
            dates = sorted(self._graduated[universe_name].keys())
            if dates:
                return self._graduated[universe_name][dates[-1]].copy()

        raise ValueError(f"Universe not found: {universe_name}")

    def get_universe_as_of(
        self,
        universe_name: str,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """Get universe membership at specific point in time."""
        if universe_name in self._graduated:
            # Find most recent snapshot <= as_of
            snapshots = self._graduated[universe_name]
            valid_dates = [d for d in snapshots.keys() if d <= as_of]
            if valid_dates:
                latest = max(valid_dates)
                return snapshots[latest].copy()
            return []

        # Non-graduated universe - return static membership
        if universe_name in self._universes:
            return self._universes[universe_name].copy()

        raise ValueError(f"Universe not found: {universe_name}")

    def list_available_universes(self) -> list[str]:
        """Return list of available universe names."""
        return list(set(self._universes.keys()) | set(self._graduated.keys()))

    def add_universe(self, name: str, tickers: list[str]):
        """Add a static universe (test helper)."""
        self._universes[name] = tickers

    def add_graduated_snapshot(
        self,
        universe_name: str,
        as_of: pd.Timestamp,
        tickers: list[str],
    ):
        """Add a graduated universe snapshot (test helper)."""
        if universe_name not in self._graduated:
            self._graduated[universe_name] = {}
        self._graduated[universe_name][as_of] = tickers


class MockReferenceAccessor(ReferenceDataAccessor):
    """Mock reference data accessor for testing.

    Stores ticker metadata and sector mappings in memory.

    Args:
        ticker_info: Dict mapping ticker to TickerMetadata
        sector_mapping: Dict mapping ticker to sector
        exchange_info: Dict mapping exchange name to ExchangeInfo
    """

    def __init__(
        self,
        ticker_info: Optional[dict[str, TickerMetadata]] = None,
        sector_mapping: Optional[dict[str, str]] = None,
        exchange_info: Optional[dict[str, ExchangeInfo]] = None,
    ):
        self._ticker_info = ticker_info or {}
        # Auto-populate sector mapping from ticker_info if not provided
        if sector_mapping is not None:
            self._sector_mapping = sector_mapping
        else:
            self._sector_mapping = {
                ticker: meta.sector
                for ticker, meta in self._ticker_info.items()
            }
        self._exchange_info = exchange_info or self._default_exchange_info()

    def _default_exchange_info(self) -> dict[str, ExchangeInfo]:
        """Create default exchange info."""
        return {
            "NYSE": ExchangeInfo(
                name="New York Stock Exchange",
                trading_hours="09:30-16:00 EST",
                timezone="US/Eastern",
                settlement_days=2,
            ),
            "NASDAQ": ExchangeInfo(
                name="NASDAQ Stock Market",
                trading_hours="09:30-16:00 EST",
                timezone="US/Eastern",
                settlement_days=2,
            ),
            "ARCA": ExchangeInfo(
                name="NYSE Arca",
                trading_hours="09:30-16:00 EST",
                timezone="US/Eastern",
                settlement_days=2,
            ),
        }

    def get_ticker_info(self, ticker: str) -> TickerMetadata:
        """Get metadata for a ticker."""
        if ticker not in self._ticker_info:
            raise ValueError(f"Ticker not found: {ticker}")
        return self._ticker_info[ticker]

    def get_sector_mapping(self) -> dict[str, str]:
        """Return ticker -> sector mapping for all tickers."""
        return self._sector_mapping.copy()

    def get_exchange_info(self) -> dict[str, ExchangeInfo]:
        """Return exchange -> metadata mapping."""
        return self._exchange_info.copy()

    def add_ticker(self, metadata: TickerMetadata):
        """Add ticker metadata (test helper)."""
        self._ticker_info[metadata.ticker] = metadata
        self._sector_mapping[metadata.ticker] = metadata.sector

    def set_sector(self, ticker: str, sector: str):
        """Set sector for ticker (test helper)."""
        self._sector_mapping[ticker] = sector


def create_mock_context(
    prices: Optional[pd.DataFrame] = None,
    indicators: Optional[dict[str, pd.DataFrame]] = None,
    universes: Optional[dict[str, list[str]]] = None,
    ticker_info: Optional[dict[str, TickerMetadata]] = None,
) -> "DataAccessContext":
    """Create a DataAccessContext with mock accessors.

    Convenience function for creating a fully mocked context for testing.

    Args:
        prices: Price data for MockPriceAccessor
        indicators: Macro indicator data for MockMacroAccessor
        universes: Universe definitions for MockUniverseAccessor
        ticker_info: Ticker metadata for MockReferenceAccessor

    Returns:
        DataAccessContext with all mock accessors configured
    """
    from quantetf.data.access.context import DataAccessContext
    from .builders import PriceDataBuilder, MacroDataBuilder

    # Use builders for defaults if not provided
    if prices is None:
        prices = PriceDataBuilder().build()

    if indicators is None:
        indicators = {"VIX": MacroDataBuilder().build_vix()}

    if universes is None:
        tickers = list(prices.columns.get_level_values('Ticker').unique())
        universes = {"default": tickers}

    if ticker_info is None:
        ticker_info = {}
        for ticker in prices.columns.get_level_values('Ticker').unique():
            ticker_info[ticker] = TickerMetadata(
                ticker=ticker,
                name=f"{ticker} ETF",
                sector="Unknown",
                exchange="ARCA",
                currency="USD",
            )

    return DataAccessContext(
        prices=MockPriceAccessor(prices),
        macro=MockMacroAccessor(indicators),
        universes=MockUniverseAccessor(universes),
        references=MockReferenceAccessor(ticker_info),
    )
