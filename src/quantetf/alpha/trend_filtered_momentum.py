"""Trend-filtered momentum alpha model.

This model applies a trend filter before using momentum signals.
When SPY is above its 200-day moving average (bullish), use momentum.
When SPY is below (bearish), allocate to defensive assets.

Migrated to use DataAccessContext (DAL) instead of direct SnapshotDataStore dependency.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from quantetf.alpha.base import AlphaModel
from quantetf.data.access import DataAccessContext
from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class TrendFilteredMomentum(AlphaModel):
    """Momentum strategy with trend filter.

    Only use momentum when market trend is bullish (SPY > MA200).
    When bearish, allocate to defensive assets.

    This model combines a trend-following overlay with cross-sectional momentum.
    The intuition is that momentum works best in bull markets, so we should
    avoid momentum exposure when the broad market is in a downtrend.

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/latest/data.parquet"}
        ... )
        >>> alpha = TrendFilteredMomentum(momentum_lookback=252, ma_period=200)
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=None,
        ...     data_access=ctx
        ... )

    Attributes:
        momentum_lookback: Days for momentum calculation (default: 252)
        ma_period: Moving average period for trend filter (default: 200)
        trend_ticker: Ticker to use for trend detection (default: 'SPY')
        defensive_tickers: Tickers to use in defensive mode
        min_periods: Minimum data points required
    """

    DEFAULT_DEFENSIVE_TICKERS = ['AGG', 'TLT', 'GLD', 'USMV', 'SPLV']

    def __init__(
        self,
        momentum_lookback: int = 252,
        ma_period: int = 200,
        trend_ticker: str = 'SPY',
        defensive_tickers: Optional[List[str]] = None,
        min_periods: int = 200,
    ):
        """Initialize trend-filtered momentum model.

        Args:
            momentum_lookback: Number of trading days for momentum calculation
            ma_period: Moving average period for trend filter
            trend_ticker: Ticker to use for trend detection (typically SPY)
            defensive_tickers: Tickers to score highly in bearish regime
            min_periods: Minimum number of valid prices required
        """
        self.momentum_lookback = momentum_lookback
        self.ma_period = ma_period
        self.trend_ticker = trend_ticker
        self.defensive_tickers = defensive_tickers or self.DEFAULT_DEFENSIVE_TICKERS
        self.min_periods = min_periods

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        data_access: DataAccessContext,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> AlphaScores:
        """Calculate scores based on regime.

        Args:
            as_of: Date for which to calculate scores
            universe: Set of eligible tickers
            features: Pre-computed features (not used)
            data_access: DataAccessContext for accessing price history
            dataset_version: Optional dataset version

        Returns:
            AlphaScores with scores for each ticker
        """
        logger.info(
            f"Computing trend-filtered momentum scores as of {as_of} "
            f"for {len(universe.tickers)} tickers"
        )

        # Get price data for trend detection and momentum calculation
        # Need enough data for MA and momentum lookback
        required_lookback = max(self.ma_period, self.momentum_lookback) + 50

        # Get universe tickers first
        try:
            ohlcv_data = data_access.prices.read_prices_as_of(
                as_of=as_of,
                tickers=list(universe.tickers),
                lookback_days=required_lookback
            )
            prices = ohlcv_data.xs('Close', level='Price', axis=1)
        except ValueError as e:
            logger.error(f"Failed to get prices: {e}")
            return AlphaScores(
                as_of=as_of,
                scores=pd.Series(0.0, index=list(universe.tickers))
            )

        # Try to get trend ticker separately if not already in universe
        if self.trend_ticker not in prices.columns:
            try:
                trend_ohlcv = data_access.prices.read_prices_as_of(
                    as_of=as_of,
                    tickers=[self.trend_ticker],
                    lookback_days=required_lookback
                )
                trend_prices = trend_ohlcv.xs('Close', level='Price', axis=1)
                prices = pd.concat([prices, trend_prices], axis=1)
            except ValueError:
                logger.warning(
                    f"Trend ticker {self.trend_ticker} not available, defaulting to bullish"
                )

        # Determine regime
        is_bullish = self._is_bullish(prices, as_of)
        regime = "BULLISH" if is_bullish else "DEFENSIVE"
        logger.info(f"Current regime: {regime}")

        if is_bullish:
            scores = self._momentum_scores(prices, universe)
        else:
            scores = self._defensive_scores(universe)

        # Log summary
        valid_scores = scores.dropna()
        if len(valid_scores) > 0:
            logger.info(
                f"Scores ({regime}): mean={valid_scores.mean():.4f}, "
                f"std={valid_scores.std():.4f}, "
                f"valid={len(valid_scores)}/{len(scores)}"
            )

        return AlphaScores(as_of=as_of, scores=scores)

    def _is_bullish(self, prices: pd.DataFrame, as_of: pd.Timestamp) -> bool:
        """Check if market is in bullish regime.

        Returns True if trend_ticker is above its moving average.

        Args:
            prices: DataFrame with close prices
            as_of: Date for calculation (prices already filtered to before this)

        Returns:
            True if in bullish regime, False otherwise
        """
        if self.trend_ticker not in prices.columns:
            logger.warning(
                f"Trend ticker {self.trend_ticker} not in data, defaulting to bullish"
            )
            return True

        trend_prices = prices[self.trend_ticker].dropna()

        if len(trend_prices) < self.ma_period:
            logger.warning(
                f"Not enough data for MA ({len(trend_prices)} < {self.ma_period}), "
                "defaulting to bullish"
            )
            return True

        current_price = trend_prices.iloc[-1]
        ma_value = trend_prices.rolling(self.ma_period).mean().iloc[-1]

        is_bullish = current_price > ma_value
        logger.debug(
            f"Trend: {self.trend_ticker} = {current_price:.2f}, "
            f"MA{self.ma_period} = {ma_value:.2f}, bullish = {is_bullish}"
        )

        return is_bullish

    def _momentum_scores(
        self,
        prices: pd.DataFrame,
        universe: Universe,
    ) -> pd.Series:
        """Calculate momentum scores (trailing return).

        Args:
            prices: DataFrame with close prices
            universe: Set of eligible tickers

        Returns:
            Series of momentum scores
        """
        scores = {}

        for ticker in universe.tickers:
            if ticker not in prices.columns:
                logger.warning(f"No price data for {ticker}")
                scores[ticker] = np.nan
                continue

            ticker_prices = prices[ticker].dropna()

            if len(ticker_prices) < self.min_periods:
                logger.debug(
                    f"{ticker}: Insufficient data ({len(ticker_prices)} < {self.min_periods})"
                )
                scores[ticker] = np.nan
                continue

            # Calculate lookback return
            lookback_prices = ticker_prices.iloc[-self.momentum_lookback:]

            if len(lookback_prices) < 2:
                scores[ticker] = np.nan
                continue

            momentum = (lookback_prices.iloc[-1] / lookback_prices.iloc[0]) - 1.0
            scores[ticker] = momentum

        return pd.Series(scores)

    def _defensive_scores(self, universe: Universe) -> pd.Series:
        """Score defensive assets high, others zero.

        In defensive mode, we want to hold safe assets.

        Args:
            universe: Set of eligible tickers

        Returns:
            Series of scores (defensive tickers = 1.0, others = 0.0)
        """
        scores = pd.Series(0.0, index=list(universe.tickers))

        for ticker in self.defensive_tickers:
            if ticker in scores.index:
                scores[ticker] = 1.0

        return scores

    def get_regime(
        self,
        data_access: DataAccessContext,
        as_of: pd.Timestamp,
    ) -> str:
        """Return current regime for logging/analysis.

        Args:
            data_access: DataAccessContext for accessing prices
            as_of: Date for regime detection

        Returns:
            "BULLISH" or "DEFENSIVE"
        """
        required_lookback = self.ma_period + 50

        try:
            ohlcv_data = data_access.prices.read_prices_as_of(
                as_of=as_of,
                tickers=[self.trend_ticker],
                lookback_days=required_lookback
            )
            prices = ohlcv_data.xs('Close', level='Price', axis=1)
        except ValueError:
            return "BULLISH"  # Default

        return "BULLISH" if self._is_bullish(prices, as_of) else "DEFENSIVE"
