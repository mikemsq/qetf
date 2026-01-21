"""Value-momentum blend alpha model.

Combines momentum (trend-following) with value (mean-reversion) signals.

The value-momentum blend attempts to balance two opposing forces:
- Momentum: Buy recent winners (trend continuation)
- Value: Buy recent losers (mean reversion)

By blending these signals, the strategy aims to capture both trend-following
and contrarian opportunities while reducing exposure to each signal's failure modes.

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


class ValueMomentum(AlphaModel):
    """Blended value and momentum strategy.

    Score = momentum_weight * momentum_score + value_weight * value_score

    Where:
    - momentum_score: Z-score of trailing return (higher past return = higher score)
    - value_score: Z-score of negative trailing return (lower past return = higher score)

    Both signals are z-scored before blending to ensure comparable scales.

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/latest/data.parquet"}
        ... )
        >>> alpha = ValueMomentum(momentum_weight=0.6, value_weight=0.4)
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=None,
        ...     data_access=ctx
        ... )

    Attributes:
        momentum_weight: Weight for momentum signal (default: 0.5)
        value_weight: Weight for value signal (default: 0.5)
        momentum_lookback: Days for momentum calculation (default: 252)
        value_lookback: Days for value calculation (default: 252)
        min_periods: Minimum data required
    """

    def __init__(
        self,
        momentum_weight: float = 0.5,
        value_weight: float = 0.5,
        momentum_lookback: int = 252,
        value_lookback: int = 252,
        min_periods: int = 200,
    ):
        """Initialize value-momentum blend model.

        Args:
            momentum_weight: Weight for momentum signal (0-1)
            value_weight: Weight for value signal (0-1)
            momentum_lookback: Days for momentum calculation
            value_lookback: Days for value calculation
            min_periods: Minimum number of valid prices required

        Note:
            Weights are normalized to sum to 1 if they don't already.
        """
        # Normalize weights to sum to 1
        total = momentum_weight + value_weight
        if abs(total - 1.0) > 0.001 and total > 0:
            momentum_weight = momentum_weight / total
            value_weight = value_weight / total

        self.momentum_weight = momentum_weight
        self.value_weight = value_weight
        self.momentum_lookback = momentum_lookback
        self.value_lookback = value_lookback
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
        """Calculate blended value-momentum scores.

        Args:
            as_of: Date for which to calculate scores
            universe: Set of eligible tickers
            features: Pre-computed features (not used)
            data_access: DataAccessContext for accessing price history
            dataset_version: Optional dataset version

        Returns:
            AlphaScores with blended scores (z-scored)
        """
        logger.info(
            f"Computing value-momentum scores as of {as_of} "
            f"for {len(universe.tickers)} tickers "
            f"(mom_weight={self.momentum_weight:.2f}, val_weight={self.value_weight:.2f})"
        )

        # Get price data
        required_lookback = max(self.momentum_lookback, self.value_lookback) + 50

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

        # Calculate momentum returns
        mom_returns = self._calculate_returns(prices, universe.tickers, self.momentum_lookback)

        # Calculate value returns (using potentially different lookback)
        val_returns = self._calculate_returns(prices, universe.tickers, self.value_lookback)

        # Z-score both signals
        mom_z = self._zscore(mom_returns)
        val_z = self._zscore(-val_returns)  # Negative: losers get high value score

        # Blend signals
        # Only include tickers that have both signals
        common_tickers = mom_z.index.intersection(val_z.index)

        if len(common_tickers) == 0:
            logger.warning("No tickers with both momentum and value signals")
            return AlphaScores(
                as_of=as_of,
                scores=pd.Series(np.nan, index=list(universe.tickers))
            )

        blended = (
            self.momentum_weight * mom_z[common_tickers] +
            self.value_weight * val_z[common_tickers]
        )

        # Fill in NaN for tickers without scores
        final_scores = pd.Series(np.nan, index=list(universe.tickers))
        for ticker in blended.index:
            final_scores[ticker] = blended[ticker]

        # Log summary
        valid_scores = final_scores.dropna()
        if len(valid_scores) > 0:
            logger.info(
                f"Value-momentum scores: mean={valid_scores.mean():.4f}, "
                f"std={valid_scores.std():.4f}, "
                f"valid={len(valid_scores)}/{len(final_scores)}"
            )

        return AlphaScores(as_of=as_of, scores=final_scores)

    def _calculate_returns(
        self,
        prices: pd.DataFrame,
        tickers: tuple,
        lookback: int,
    ) -> pd.Series:
        """Calculate lookback returns for given tickers.

        Args:
            prices: DataFrame with close prices
            tickers: Tuple of tickers to calculate returns for
            lookback: Number of days for return calculation

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
            lookback_prices = ticker_prices.iloc[-lookback:]

            if len(lookback_prices) < 2:
                returns[ticker] = np.nan
                continue

            ret = (lookback_prices.iloc[-1] / lookback_prices.iloc[0]) - 1.0
            returns[ticker] = ret

        return pd.Series(returns).dropna()

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Convert to z-scores.

        Args:
            series: Series of values to standardize

        Returns:
            Z-scored series (mean=0, std=1)
        """
        if len(series) == 0:
            return series

        std = series.std()
        if std == 0 or np.isnan(std):
            # If no variance, return zeros
            return pd.Series(0.0, index=series.index)

        return (series - series.mean()) / std

    def get_signal_components(
        self,
        data_access: DataAccessContext,
        as_of: pd.Timestamp,
        universe: Universe,
    ) -> dict:
        """Get individual momentum and value signal components.

        Useful for debugging and understanding the blend.

        Args:
            data_access: DataAccessContext for accessing prices
            as_of: Date for calculation
            universe: Set of eligible tickers

        Returns:
            Dict with 'momentum_z', 'value_z', and 'blended' Series
        """
        required_lookback = max(self.momentum_lookback, self.value_lookback) + 50

        try:
            ohlcv_data = data_access.prices.read_prices_as_of(
                as_of=as_of,
                tickers=list(universe.tickers),
                lookback_days=required_lookback
            )
            prices = ohlcv_data.xs('Close', level='Price', axis=1)
        except ValueError:
            return {
                'momentum_z': pd.Series(dtype=float),
                'value_z': pd.Series(dtype=float),
                'blended': pd.Series(dtype=float),
            }

        mom_returns = self._calculate_returns(prices, universe.tickers, self.momentum_lookback)
        val_returns = self._calculate_returns(prices, universe.tickers, self.value_lookback)

        mom_z = self._zscore(mom_returns)
        val_z = self._zscore(-val_returns)

        common = mom_z.index.intersection(val_z.index)
        blended = (
            self.momentum_weight * mom_z[common] +
            self.value_weight * val_z[common]
        ) if len(common) > 0 else pd.Series(dtype=float)

        return {
            'momentum_z': mom_z,
            'value_z': val_z,
            'blended': blended,
        }
