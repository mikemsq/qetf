"""Momentum alpha models for ETF strategies.

Momentum strategies bet on the continuation of recent price trends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from quantetf.alpha.base import AlphaModel
from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossSectionalMomentum(AlphaModel):
    """Alpha model that reads a precomputed momentum feature column.

    This is the original stub - kept for backwards compatibility.
    """

    feature_name: str = "momentum"

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        store: DataStore,
        dataset_version: DatasetVersion | None = None,
    ) -> AlphaScores:
        if self.feature_name not in features.frame.columns:
            raise KeyError(f"Missing feature '{self.feature_name}' in feature frame")
        scores = features.frame[self.feature_name].astype(float).copy()
        scores = scores.loc[list(universe.tickers)]
        return AlphaScores(as_of=as_of, scores=scores)


class MomentumAlpha(AlphaModel):
    """Simple momentum alpha based on trailing total return.

    This model computes momentum as the cumulative return over a lookback
    window. Higher momentum = stronger uptrend = higher alpha score.

    The model uses point-in-time data to ensure no lookahead bias.

    Example:
        >>> alpha = MomentumAlpha(lookback_days=252)  # 12-month momentum
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=features,
        ...     store=store
        ... )
    """

    def __init__(self, lookback_days: int = 252, min_periods: int = 200):
        """Initialize momentum alpha model.

        Args:
            lookback_days: Number of trading days to look back (default: 252 ~ 1 year)
            min_periods: Minimum number of valid prices required (default: 200)
        """
        self.lookback_days = lookback_days
        self.min_periods = min_periods

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> AlphaScores:
        """Compute momentum scores for the universe.

        CRITICAL: This method must ONLY use data available before as_of date.
        It retrieves prices up to (but not including) as_of, ensuring no lookahead.

        Args:
            as_of: Date as of which to compute scores (decisions made on this date)
            universe: Set of eligible tickers
            features: Pre-computed features (not used in simple momentum)
            store: Data store for accessing price history
            dataset_version: Optional dataset version for reproducibility

        Returns:
            AlphaScores with momentum scores for each ticker in universe
        """
        logger.info(f"Computing momentum scores as of {as_of} for {len(universe.tickers)} tickers")

        # Import here to avoid circular dependency
        from quantetf.data.snapshot_store import SnapshotDataStore

        if not isinstance(store, SnapshotDataStore):
            raise TypeError(f"MomentumAlpha requires SnapshotDataStore, got {type(store)}")

        # Get price data - this automatically uses T-1 and earlier (no lookahead)
        try:
            prices = store.get_close_prices(
                as_of=as_of,
                tickers=list(universe.tickers),
                lookback_days=self.lookback_days + 50  # Extra buffer for rolling calculations
            )
        except ValueError as e:
            logger.error(f"Failed to get prices: {e}")
            # Return zero scores if we can't get data
            return AlphaScores(
                as_of=as_of,
                scores=pd.Series(0.0, index=list(universe.tickers))
            )

        # Calculate momentum for each ticker
        scores = {}

        for ticker in universe.tickers:
            if ticker not in prices.columns:
                logger.warning(f"No price data for {ticker}")
                scores[ticker] = np.nan
                continue

            ticker_prices = prices[ticker].dropna()

            # Check if we have enough data
            if len(ticker_prices) < self.min_periods:
                logger.debug(f"{ticker}: Insufficient data ({len(ticker_prices)} < {self.min_periods})")
                scores[ticker] = np.nan
                continue

            # Calculate momentum as cumulative return over lookback period
            # Use last N days of available data
            lookback_prices = ticker_prices.iloc[-self.lookback_days:]

            if len(lookback_prices) < 2:
                scores[ticker] = np.nan
                continue

            # Momentum = (most recent price / oldest price in window) - 1
            momentum = (lookback_prices.iloc[-1] / lookback_prices.iloc[0]) - 1.0

            scores[ticker] = momentum

        scores_series = pd.Series(scores)

        # Log summary statistics
        valid_scores = scores_series.dropna()
        if len(valid_scores) > 0:
            logger.info(
                f"Momentum scores: mean={valid_scores.mean():.4f}, "
                f"std={valid_scores.std():.4f}, "
                f"min={valid_scores.min():.4f}, "
                f"max={valid_scores.max():.4f}, "
                f"valid={len(valid_scores)}/{len(scores_series)}"
            )
        else:
            logger.warning("No valid momentum scores computed")

        return AlphaScores(
            as_of=as_of,
            scores=scores_series
        )
