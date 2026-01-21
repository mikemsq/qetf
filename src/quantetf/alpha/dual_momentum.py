"""Dual momentum alpha model (Gary Antonacci style).

Combines absolute momentum (vs risk-free) and relative momentum (vs peers).

The dual momentum strategy was popularized by Gary Antonacci and combines:
1. Absolute momentum: Only invest if asset return exceeds risk-free rate
2. Relative momentum: Among qualifying assets, select the best performers

When no assets pass the absolute momentum filter, allocate to safe assets.

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


class DualMomentum(AlphaModel):
    """Dual momentum strategy combining absolute and relative momentum.

    1. Absolute momentum: Filter out assets with return < risk-free rate
    2. Relative momentum: Rank remaining assets by return

    If no assets pass absolute filter, allocate to safe assets.

    This approach aims to avoid holding assets in downtrends while still
    capturing relative momentum among assets that are trending up.

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/latest/data.parquet"}
        ... )
        >>> alpha = DualMomentum(lookback=252, risk_free_rate=0.02)
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=None,
        ...     data_access=ctx
        ... )

    Attributes:
        lookback: Days for momentum calculation (default: 252)
        risk_free_rate: Annual risk-free rate for absolute filter (default: 0.02)
        safe_tickers: Tickers to use when all momentum is negative
        min_periods: Minimum data points required
    """

    DEFAULT_SAFE_TICKERS = ['AGG', 'BND', 'SHY']

    def __init__(
        self,
        lookback: int = 252,
        risk_free_rate: float = 0.02,
        safe_tickers: Optional[List[str]] = None,
        min_periods: int = 200,
    ):
        """Initialize dual momentum model.

        Args:
            lookback: Number of trading days for momentum calculation
            risk_free_rate: Annual risk-free rate as threshold (e.g., 0.02 = 2%)
            safe_tickers: Tickers to score highly when all momentum is negative
            min_periods: Minimum number of valid prices required
        """
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        self.safe_tickers = safe_tickers or self.DEFAULT_SAFE_TICKERS
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
        """Calculate dual momentum scores.

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
            f"Computing dual momentum scores as of {as_of} "
            f"for {len(universe.tickers)} tickers"
        )

        # Filter universe to exclude safe tickers from momentum calculation
        # Safe tickers are only used when momentum is negative
        risky_tickers = [t for t in universe.tickers if t not in self.safe_tickers]

        try:
            ohlcv_data = data_access.prices.read_prices_as_of(
                as_of=as_of,
                tickers=list(universe.tickers),
                lookback_days=self.lookback + 50
            )
            prices = ohlcv_data.xs('Close', level='Price', axis=1)
        except ValueError as e:
            logger.error(f"Failed to get prices: {e}")
            return AlphaScores(
                as_of=as_of,
                scores=pd.Series(0.0, index=list(universe.tickers))
            )

        # Calculate returns for risky assets
        returns = self._calculate_returns(prices, risky_tickers)

        # Absolute momentum threshold (annualized rf rate scaled to lookback period)
        threshold = self.risk_free_rate * (self.lookback / 252)

        # Filter: only positive absolute momentum (above threshold)
        positive_momentum = returns[returns > threshold]

        if len(positive_momentum) == 0:
            logger.info(
                f"No assets with positive momentum (threshold={threshold:.4f}), "
                f"using safe assets"
            )
            scores = self._safe_scores(list(universe.tickers))
            signal_type = "SAFE"
        else:
            # Relative momentum: use returns as scores (higher = better)
            # Start with zeros for all tickers
            scores = pd.Series(0.0, index=list(universe.tickers))
            # Fill in momentum scores for qualifying assets
            for ticker in positive_momentum.index:
                if ticker in scores.index:
                    scores[ticker] = positive_momentum[ticker]
            signal_type = "MOMENTUM"

        # Log summary
        valid_scores = scores.dropna()
        positive_count = (scores > 0).sum()
        logger.info(
            f"Dual momentum ({signal_type}): {positive_count} assets with positive scores, "
            f"threshold={threshold:.4f}"
        )

        return AlphaScores(as_of=as_of, scores=scores)

    def _calculate_returns(
        self,
        prices: pd.DataFrame,
        tickers: List[str],
    ) -> pd.Series:
        """Calculate lookback returns for given tickers.

        Args:
            prices: DataFrame with close prices
            tickers: List of tickers to calculate returns for

        Returns:
            Series of returns indexed by ticker
        """
        returns = {}

        for ticker in tickers:
            if ticker not in prices.columns:
                logger.warning(f"No price data for {ticker}")
                returns[ticker] = np.nan
                continue

            ticker_prices = prices[ticker].dropna()

            if len(ticker_prices) < self.min_periods:
                logger.debug(
                    f"{ticker}: Insufficient data ({len(ticker_prices)} < {self.min_periods})"
                )
                returns[ticker] = np.nan
                continue

            # Calculate lookback return
            lookback_prices = ticker_prices.iloc[-self.lookback:]

            if len(lookback_prices) < 2:
                returns[ticker] = np.nan
                continue

            ret = (lookback_prices.iloc[-1] / lookback_prices.iloc[0]) - 1.0
            returns[ticker] = ret

        return pd.Series(returns).dropna()

    def _safe_scores(self, all_tickers: List[str]) -> pd.Series:
        """Score safe assets high when momentum is negative.

        Args:
            all_tickers: List of all tickers in universe

        Returns:
            Series with 1.0 for safe tickers, 0.0 for others
        """
        scores = pd.Series(0.0, index=all_tickers)

        for ticker in self.safe_tickers:
            if ticker in scores.index:
                scores[ticker] = 1.0

        return scores

    def get_signal_type(
        self,
        data_access: DataAccessContext,
        as_of: pd.Timestamp,
        universe: Universe,
    ) -> str:
        """Return whether using momentum or safe assets.

        Args:
            data_access: DataAccessContext for accessing prices
            as_of: Date for calculation
            universe: Set of eligible tickers

        Returns:
            "MOMENTUM" or "SAFE"
        """
        scores = self.score(
            as_of=as_of,
            universe=universe,
            features=None,
            data_access=data_access
        )

        # Check if we're in safe mode
        safe_scores = [scores.scores.get(t, 0) for t in self.safe_tickers if t in scores.scores.index]
        risky_tickers = [t for t in scores.scores.index if t not in self.safe_tickers]
        risky_scores = [scores.scores.get(t, 0) for t in risky_tickers]

        if any(s > 0 for s in safe_scores) and not any(s > 0 for s in risky_scores):
            return "SAFE"
        else:
            return "MOMENTUM"
