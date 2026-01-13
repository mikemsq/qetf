"""Momentum acceleration alpha model.

Ranks tickers by the difference between short-term and long-term momentum,
capturing trend acceleration or deceleration signals.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from quantetf.alpha.base import AlphaModel
from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore

logger = logging.getLogger(__name__)


class MomentumAccelerationAlpha(AlphaModel):
    """Momentum acceleration: rank by (short_returns - long_returns).

    This model captures trend strength by comparing recent momentum to
    longer-term momentum. Positive acceleration indicates strengthening
    trends (buy signal), while negative acceleration indicates weakening
    trends (sell/avoid signal).

    Mathematical approach:
        score_i = returns_3m_i - returns_12m_i

    Where:
        - returns_3m = (price_T-1 / price_T-63) - 1   # Short-term
        - returns_12m = (price_T-1 / price_T-252) - 1  # Long-term

    Interpretation:
        - score > 0: Accelerating (recent > long-term) → Strong buy
        - score ≈ 0: Steady momentum → Hold
        - score < 0: Decelerating (recent < long-term) → Weak/sell

    Use cases:
        - Early trend detection: Enter during acceleration phase
        - Exit signals: Detect momentum weakening before reversal
        - Regime changes: Capture bull/bear transitions

    Point-in-time compliance:
        - Uses only data BEFORE as_of date (T-1 and earlier)

    Example:
        >>> alpha = MomentumAccelerationAlpha(
        ...     short_lookback_days=63,
        ...     long_lookback_days=252
        ... )
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=features,
        ...     store=store
        ... )
    """

    def __init__(
        self,
        short_lookback_days: int = 63,
        long_lookback_days: int = 252,
        min_periods: int = 200
    ) -> None:
        """Initialize momentum acceleration alpha model.

        Args:
            short_lookback_days: Days for short-term momentum (default: 63 ~ 3M)
            long_lookback_days: Days for long-term momentum (default: 252 ~ 12M)
            min_periods: Minimum valid prices required (default: 200)

        Raises:
            ValueError: If parameters are invalid
        """
        if short_lookback_days >= long_lookback_days:
            raise ValueError(
                f"short_lookback ({short_lookback_days}) must be < "
                f"long_lookback ({long_lookback_days})"
            )
        if min_periods < short_lookback_days:
            raise ValueError(
                f"min_periods ({min_periods}) must be >= "
                f"short_lookback ({short_lookback_days})"
            )
        if short_lookback_days < 20:
            raise ValueError(f"short_lookback must be >= 20 days")

        self.short_lookback_days = short_lookback_days
        self.long_lookback_days = long_lookback_days
        self.min_periods = min_periods

        logger.info(
            f"Initialized MomentumAccelerationAlpha: "
            f"short_lookback={short_lookback_days}, "
            f"long_lookback={long_lookback_days}, "
            f"min_periods={min_periods}"
        )

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> AlphaScores:
        """Compute momentum acceleration scores.

        CRITICAL: Uses only data BEFORE as_of (T-1 and earlier).

        Args:
            as_of: Decision date
            universe: Set of eligible tickers
            features: Pre-computed features (not used)
            store: Data store for price history
            dataset_version: Optional dataset version

        Returns:
            AlphaScores with acceleration scores.
            Positive = accelerating trend, Negative = decelerating.
            NaN = insufficient data.

        Raises:
            TypeError: If store is not SnapshotDataStore
        """
        logger.info(
            f"Computing momentum acceleration as of {as_of} "
            f"for {len(universe.tickers)} tickers"
        )

        # Validate store type
        from quantetf.data.snapshot_store import SnapshotDataStore
        if not isinstance(store, SnapshotDataStore):
            raise TypeError(
                f"MomentumAccelerationAlpha requires SnapshotDataStore, "
                f"got {type(store)}"
            )

        # Get prices - add buffer for calculations
        try:
            prices = store.get_close_prices(
                as_of=as_of,
                tickers=list(universe.tickers),
                lookback_days=self.long_lookback_days + 50
            )
        except ValueError as e:
            logger.error(f"Failed to get prices: {e}")
            # Return NaN scores if we can't get data
            return AlphaScores(
                as_of=as_of,
                scores=pd.Series(np.nan, index=list(universe.tickers))
            )

        # Calculate scores
        scores = {}
        valid_count = 0

        for ticker in universe.tickers:
            if ticker not in prices.columns:
                logger.debug(f"{ticker}: No price data available")
                scores[ticker] = np.nan
                continue

            ticker_px = prices[ticker].dropna()

            if len(ticker_px) < self.min_periods:
                logger.debug(
                    f"{ticker}: Insufficient data "
                    f"({len(ticker_px)} < {self.min_periods})"
                )
                scores[ticker] = np.nan
                continue

            # Calculate long-term return (use all available data up to lookback)
            long_window = ticker_px.iloc[-self.long_lookback_days:]
            if len(long_window) < 2:
                scores[ticker] = np.nan
                continue

            long_return = (long_window.iloc[-1] / long_window.iloc[0]) - 1.0

            # Calculate short-term return (most recent period)
            short_window = ticker_px.iloc[-self.short_lookback_days:]
            if len(short_window) < 2:
                scores[ticker] = np.nan
                continue

            short_return = (short_window.iloc[-1] / short_window.iloc[0]) - 1.0

            # Acceleration = short - long
            acceleration = short_return - long_return
            scores[ticker] = acceleration
            valid_count += 1

            logger.debug(
                f"{ticker}: short={short_return:.4f}, long={long_return:.4f}, "
                f"accel={acceleration:.4f}"
            )

        # Log summary statistics
        scores_series = pd.Series(scores)
        valid_scores = scores_series.dropna()

        if len(valid_scores) > 0:
            logger.info(
                f"Acceleration scores: mean={valid_scores.mean():.4f}, "
                f"std={valid_scores.std():.4f}, "
                f"min={valid_scores.min():.4f}, "
                f"max={valid_scores.max():.4f}, "
                f"valid={valid_count}/{len(universe.tickers)}"
            )
        else:
            logger.warning("No valid acceleration scores computed")

        return AlphaScores(as_of=as_of, scores=scores_series)
