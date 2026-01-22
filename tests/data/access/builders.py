"""Data builders for creating synthetic test data.

These builders use the builder pattern to create realistic test datasets
with customizable parameters.

Usage:
    from tests.data.access.builders import PriceDataBuilder

    # Create price data with specific tickers and date range
    prices = (
        PriceDataBuilder()
        .with_tickers(['SPY', 'QQQ', 'IWM'])
        .with_date_range('2020-01-01', '2023-12-31')
        .with_trend(0.0001)  # Slight upward trend
        .build()
    )
"""

from typing import Optional
import pandas as pd
import numpy as np


class PriceDataBuilder:
    """Builder for creating synthetic OHLCV price data.

    Creates realistic price data with configurable parameters including
    trend, volatility, and price levels.
    """

    def __init__(self):
        self._tickers = ['SPY', 'QQQ']
        self._start_date = '2020-01-01'
        self._end_date = '2023-12-31'
        self._freq = 'B'  # Business days
        self._base_prices: dict[str, float] = {}
        self._trend = 0.0001  # Daily return drift
        self._volatility = 0.02  # Daily volatility
        self._seed: Optional[int] = None

    def with_tickers(self, tickers: list[str]) -> 'PriceDataBuilder':
        """Set tickers to include in the data."""
        self._tickers = tickers
        return self

    def with_date_range(
        self,
        start: str,
        end: str,
        freq: str = 'B',
    ) -> 'PriceDataBuilder':
        """Set date range and frequency."""
        self._start_date = start
        self._end_date = end
        self._freq = freq
        return self

    def with_base_price(self, ticker: str, price: float) -> 'PriceDataBuilder':
        """Set base price for a specific ticker."""
        self._base_prices[ticker] = price
        return self

    def with_trend(self, daily_drift: float) -> 'PriceDataBuilder':
        """Set daily return drift (e.g., 0.0001 for ~2.5% annual)."""
        self._trend = daily_drift
        return self

    def with_volatility(self, daily_vol: float) -> 'PriceDataBuilder':
        """Set daily volatility (e.g., 0.02 for ~32% annual)."""
        self._volatility = daily_vol
        return self

    def with_seed(self, seed: int) -> 'PriceDataBuilder':
        """Set random seed for reproducibility."""
        self._seed = seed
        return self

    def build(self) -> pd.DataFrame:
        """Build the price DataFrame."""
        if self._seed is not None:
            np.random.seed(self._seed)

        dates = pd.date_range(self._start_date, self._end_date, freq=self._freq)
        n_days = len(dates)

        data_dict = {}
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']

        for ticker in self._tickers:
            base_price = self._base_prices.get(ticker, 100.0 + hash(ticker) % 200)

            # Generate returns using geometric brownian motion
            returns = np.random.normal(self._trend, self._volatility, n_days)
            cumulative = np.exp(np.cumsum(returns))
            close_prices = base_price * cumulative

            # Generate OHLC from close
            daily_range = close_prices * np.random.uniform(0.005, 0.02, n_days)
            high_prices = close_prices + daily_range * np.random.uniform(0.3, 0.7, n_days)
            low_prices = close_prices - daily_range * np.random.uniform(0.3, 0.7, n_days)
            open_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0.2, 0.8, n_days)

            # Ensure OHLC constraints
            high_prices = np.maximum.reduce([open_prices, high_prices, close_prices])
            low_prices = np.minimum.reduce([open_prices, low_prices, close_prices])

            # Volume with some variation
            base_volume = 1_000_000 * (1 + hash(ticker) % 10)
            volume = base_volume * np.random.uniform(0.5, 1.5, n_days)

            ticker_df = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volume.astype(int),
            }, index=dates)

            data_dict[ticker] = ticker_df

        # Combine into MultiIndex DataFrame
        result = pd.concat(data_dict, axis=1)
        result.columns.names = ['Ticker', 'Price']

        return result

    def build_with_gaps(
        self,
        gap_ratio: float = 0.05,
    ) -> pd.DataFrame:
        """Build price data with random gaps (missing dates).

        Useful for testing handling of incomplete data.

        Args:
            gap_ratio: Fraction of dates to remove (0.05 = 5%)

        Returns:
            DataFrame with some dates randomly removed
        """
        data = self.build()

        if self._seed is not None:
            np.random.seed(self._seed + 1)

        n_gaps = int(len(data) * gap_ratio)
        gap_indices = np.random.choice(len(data), n_gaps, replace=False)
        keep_mask = ~np.isin(np.arange(len(data)), gap_indices)

        return data.iloc[keep_mask].copy()

    def build_with_crash(
        self,
        crash_date: str,
        crash_magnitude: float = 0.20,
        recovery_days: int = 60,
    ) -> pd.DataFrame:
        """Build price data with a market crash event.

        Useful for testing drawdown calculations and crash detection.

        Args:
            crash_date: Date of the crash
            crash_magnitude: Size of crash (0.20 = 20% drop)
            recovery_days: Days to recover

        Returns:
            DataFrame with crash pattern embedded
        """
        data = self.build()
        crash_ts = pd.Timestamp(crash_date)

        if crash_ts not in data.index:
            # Find nearest date
            crash_ts = data.index[data.index.get_indexer([crash_ts], method='nearest')[0]]

        crash_idx = data.index.get_loc(crash_ts)
        n_days = len(data)

        # Create crash multiplier
        multiplier = np.ones(n_days)
        for i in range(crash_idx, n_days):
            days_since_crash = i - crash_idx
            if days_since_crash <= recovery_days:
                # V-shaped recovery
                bottom = 1 - crash_magnitude
                recovery = (days_since_crash / recovery_days) * crash_magnitude
                multiplier[i] = bottom + recovery
            else:
                multiplier[i] = 1.0

        # Apply to all price columns
        for ticker in self._tickers:
            for field in ['Open', 'High', 'Low', 'Close']:
                data[(ticker, field)] = data[(ticker, field)] * multiplier

        return data


class MacroDataBuilder:
    """Builder for creating synthetic macro indicator data."""

    def __init__(self):
        self._start_date = '2020-01-01'
        self._end_date = '2023-12-31'
        self._freq = 'B'
        self._seed: Optional[int] = None

    def with_date_range(
        self,
        start: str,
        end: str,
        freq: str = 'B',
    ) -> 'MacroDataBuilder':
        """Set date range and frequency."""
        self._start_date = start
        self._end_date = end
        self._freq = freq
        return self

    def with_seed(self, seed: int) -> 'MacroDataBuilder':
        """Set random seed for reproducibility."""
        self._seed = seed
        return self

    def build_vix(
        self,
        base_level: float = 18.0,
        spike_dates: Optional[list[str]] = None,
        spike_levels: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """Build synthetic VIX data.

        Args:
            base_level: Normal VIX level
            spike_dates: Dates of volatility spikes
            spike_levels: VIX levels during spikes

        Returns:
            DataFrame with VIX values
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        dates = pd.date_range(self._start_date, self._end_date, freq=self._freq)
        n_days = len(dates)

        # Generate mean-reverting VIX
        vix = np.zeros(n_days)
        vix[0] = base_level

        mean_reversion = 0.03
        vol_of_vol = 0.1

        for i in range(1, n_days):
            shock = np.random.normal(0, vol_of_vol)
            reversion = mean_reversion * (base_level - vix[i - 1])
            vix[i] = max(10, vix[i - 1] + reversion + shock * vix[i - 1])

        # Add spikes
        if spike_dates and spike_levels:
            for date_str, level in zip(spike_dates, spike_levels):
                spike_ts = pd.Timestamp(date_str)
                if spike_ts in dates:
                    idx = dates.get_loc(spike_ts)
                    vix[idx] = level
                    # Gradual decay over 20 days
                    for j in range(1, 21):
                        if idx + j < n_days:
                            decay = 0.9 ** j
                            vix[idx + j] = max(vix[idx + j], level * decay)

        return pd.DataFrame({'VIX': vix}, index=dates)

    def build_yield_spread(
        self,
        base_spread: float = 0.5,
    ) -> pd.DataFrame:
        """Build synthetic yield spread data (10Y-2Y)."""
        if self._seed is not None:
            np.random.seed(self._seed + 100)

        dates = pd.date_range(self._start_date, self._end_date, freq=self._freq)
        n_days = len(dates)

        # Random walk with mean reversion
        spread = np.zeros(n_days)
        spread[0] = base_spread

        for i in range(1, n_days):
            shock = np.random.normal(0, 0.02)
            reversion = 0.01 * (base_spread - spread[i - 1])
            spread[i] = spread[i - 1] + reversion + shock

        return pd.DataFrame({'YIELD_SPREAD': spread}, index=dates)

    def build_unemployment(
        self,
        base_rate: float = 4.0,
        recession_start: Optional[str] = None,
        recession_peak: float = 10.0,
    ) -> pd.DataFrame:
        """Build synthetic unemployment rate data."""
        if self._seed is not None:
            np.random.seed(self._seed + 200)

        dates = pd.date_range(self._start_date, self._end_date, freq=self._freq)
        n_days = len(dates)

        unemployment = np.zeros(n_days)
        unemployment[0] = base_rate

        for i in range(1, n_days):
            shock = np.random.normal(0, 0.05)
            reversion = 0.005 * (base_rate - unemployment[i - 1])
            unemployment[i] = max(2.0, unemployment[i - 1] + reversion + shock)

        # Add recession spike if specified
        if recession_start:
            recession_ts = pd.Timestamp(recession_start)
            if recession_ts in dates:
                idx = dates.get_loc(recession_ts)
                # Spike up over 6 months
                spike_days = min(126, n_days - idx)
                for j in range(spike_days):
                    progress = j / spike_days
                    target = base_rate + (recession_peak - base_rate) * progress
                    unemployment[idx + j] = max(unemployment[idx + j], target)
                # Slow recovery
                for j in range(spike_days, n_days - idx):
                    decay = 0.995 ** (j - spike_days)
                    excess = recession_peak - base_rate
                    unemployment[idx + j] = base_rate + excess * decay

        return pd.DataFrame({'UNEMPLOYMENT': unemployment}, index=dates)


class UniverseBuilder:
    """Builder for creating test universe definitions."""

    def __init__(self):
        self._universes: dict[str, list[str]] = {}
        self._graduated: dict[str, dict[pd.Timestamp, list[str]]] = {}

    def with_static_universe(
        self,
        name: str,
        tickers: list[str],
    ) -> 'UniverseBuilder':
        """Add a static universe."""
        self._universes[name] = tickers
        return self

    def with_graduated_universe(
        self,
        name: str,
        snapshots: dict[str, list[str]],
    ) -> 'UniverseBuilder':
        """Add a graduated universe with time-based membership.

        Args:
            name: Universe name
            snapshots: Dict mapping date strings to ticker lists
        """
        self._graduated[name] = {
            pd.Timestamp(date): tickers
            for date, tickers in snapshots.items()
        }
        return self

    def build_universes(self) -> dict[str, list[str]]:
        """Build static universes dict."""
        return self._universes.copy()

    def build_graduated(self) -> dict[str, dict[pd.Timestamp, list[str]]]:
        """Build graduated universes dict."""
        return self._graduated.copy()

    @staticmethod
    def default_etf_universe() -> list[str]:
        """Return default ETF universe for testing."""
        return [
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
            'AGG', 'LQD', 'HYG', 'TLT', 'GLD',
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI',
        ]


class TickerMetadataBuilder:
    """Builder for creating test ticker metadata."""

    def __init__(self):
        self._metadata: dict[str, dict] = {}

    def with_ticker(
        self,
        ticker: str,
        name: Optional[str] = None,
        sector: str = 'Unknown',
        exchange: str = 'ARCA',
        currency: str = 'USD',
    ) -> 'TickerMetadataBuilder':
        """Add metadata for a ticker."""
        from quantetf.data.access.types import TickerMetadata

        self._metadata[ticker] = TickerMetadata(
            ticker=ticker,
            name=name or f"{ticker} ETF",
            sector=sector,
            exchange=exchange,
            currency=currency,
        )
        return self

    def with_tickers(self, tickers: list[str]) -> 'TickerMetadataBuilder':
        """Add default metadata for multiple tickers."""
        for ticker in tickers:
            self.with_ticker(ticker)
        return self

    def build(self) -> dict[str, "TickerMetadata"]:
        """Build ticker metadata dict."""
        from quantetf.data.access.types import TickerMetadata
        return self._metadata.copy()

    @staticmethod
    def default_etf_metadata() -> dict[str, "TickerMetadata"]:
        """Return default ETF metadata for testing."""
        from quantetf.data.access.types import TickerMetadata

        etf_info = {
            'SPY': ('SPDR S&P 500 ETF', 'US Equity'),
            'QQQ': ('Invesco QQQ Trust', 'US Equity'),
            'IWM': ('iShares Russell 2000', 'US Equity'),
            'EFA': ('iShares MSCI EAFE', 'Intl Equity'),
            'EEM': ('iShares MSCI Emerging Markets', 'Emerging Markets'),
            'AGG': ('iShares Core US Aggregate Bond', 'Fixed Income'),
            'LQD': ('iShares Investment Grade Corporate', 'Fixed Income'),
            'HYG': ('iShares High Yield Corporate', 'Fixed Income'),
            'TLT': ('iShares 20+ Year Treasury', 'Fixed Income'),
            'GLD': ('SPDR Gold Trust', 'Commodities'),
            'XLF': ('Financial Select Sector SPDR', 'Financials'),
            'XLK': ('Technology Select Sector SPDR', 'Technology'),
            'XLE': ('Energy Select Sector SPDR', 'Energy'),
            'XLV': ('Health Care Select Sector SPDR', 'Healthcare'),
            'XLI': ('Industrial Select Sector SPDR', 'Industrials'),
        }

        return {
            ticker: TickerMetadata(
                ticker=ticker,
                name=name,
                sector=sector,
                exchange='ARCA',
                currency='USD',
            )
            for ticker, (name, sector) in etf_info.items()
        }
