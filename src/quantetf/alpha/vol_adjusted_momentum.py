"""Volatility-adjusted momentum alpha model.

Ranks tickers by risk-adjusted returns using a Sharpe-style metric.
Penalizes volatile assets in favor of smooth, consistent performers.

Migrated to use DataAccessContext (DAL) instead of direct SnapshotDataStore dependency.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from quantetf.alpha.base import AlphaModel
from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.access import DataAccessContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class VolAdjustedMomentumAlpha(AlphaModel):
    """Volatility-adjusted momentum: rank by returns / realized_vol.

    This model computes a Sharpe-style score for each ticker by dividing
    cumulative returns by realized volatility. Higher scores indicate
    better risk-adjusted performance.

    Mathematical approach:
        score_i = (cumulative_return_i) / (realized_volatility_i)

    Where:
        - cumulative_return = (price_T-1 / price_T-lookback) - 1
        - realized_volatility = std(daily_returns) * sqrt(252)

    This naturally favors:
        - High returns with low volatility (best)
        - Moderate returns with low volatility (good)
        - High returns with high volatility (neutral)
        - Low returns with high volatility (worst)

    Point-in-time compliance:
        - Uses only data BEFORE as_of date (T-1 and earlier)
        - No lookahead bias

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/latest/data.parquet"}
        ... )
        >>> alpha = VolAdjustedMomentumAlpha(lookback_days=252)
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=None,
        ...     data_access=ctx
        ... )
    """

    def __init__(
        self,
        lookback_days: int = 252,
        min_periods: int = 200,
        vol_floor: float = 0.01,
        annualization_factor: float = None
    ) -> None:
        """Initialize volatility-adjusted momentum alpha model.

        Args:
            lookback_days: Trading days for return/vol calculation (default: 252)
            min_periods: Minimum valid prices required (default: 200)
            vol_floor: Minimum volatility to prevent div-by-zero (default: 0.01 = 1% annual)
            annualization_factor: Factor to annualize vol (default: 252 trading days)

        Raises:
            ValueError: If parameters are invalid
        """
        if lookback_days < min_periods:
            raise ValueError(f"lookback_days must be >= min_periods")
        if min_periods < 20:
            raise ValueError(f"min_periods must be >= 20 for stable vol estimate")
        if vol_floor <= 0:
            raise ValueError(f"vol_floor must be > 0")

        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.vol_floor = vol_floor
        self.annualization_factor = annualization_factor if annualization_factor is not None else 252.0

        logger.info(
            f"Initialized VolAdjustedMomentumAlpha: "
            f"lookback={lookback_days}, min_periods={min_periods}, "
            f"vol_floor={vol_floor}, annualization={self.annualization_factor}"
        )

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        data_access: DataAccessContext,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> AlphaScores:
        """Compute volatility-adjusted momentum scores.

        CRITICAL: Uses only data BEFORE as_of (T-1 and earlier).

        Args:
            as_of: Decision date
            universe: Set of eligible tickers
            features: Pre-computed features (not used)
            data_access: DataAccessContext for price history
            dataset_version: Optional dataset version

        Returns:
            AlphaScores with vol-adjusted momentum scores.
            Higher scores = better risk-adjusted returns.
            NaN scores = insufficient data.
        """
        logger.info(
            f"Computing vol-adjusted momentum as of {as_of} "
            f"for {len(universe.tickers)} tickers"
        )

        # Get prices
        try:
            ohlcv_data = data_access.prices.read_prices_as_of(
                as_of=as_of,
                tickers=list(universe.tickers),
                lookback_days=self.lookback_days + 50
            )
            prices = ohlcv_data.xs('Close', level='Price', axis=1)
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

            # Use last N days
            window = ticker_px.iloc[-self.lookback_days:]

            if len(window) < 2:
                scores[ticker] = np.nan
                continue

            # Calculate cumulative return
            cum_return = (window.iloc[-1] / window.iloc[0]) - 1.0

            # Calculate daily returns
            daily_returns = window.pct_change().dropna()

            if len(daily_returns) < 10:
                scores[ticker] = np.nan
                continue

            # Realized volatility (annualized)
            realized_vol = daily_returns.std()
            realized_vol_annual = realized_vol * np.sqrt(self.annualization_factor)

            # Apply floor to prevent division by zero
            realized_vol_annual = max(realized_vol_annual, self.vol_floor)

            # Sharpe-style score
            score = cum_return / realized_vol_annual

            scores[ticker] = score
            valid_count += 1

            logger.debug(
                f"{ticker}: return={cum_return:.4f}, vol={realized_vol_annual:.4f}, "
                f"score={score:.4f}"
            )

        # Log summary statistics
        scores_series = pd.Series(scores)
        valid_scores = scores_series.dropna()

        if len(valid_scores) > 0:
            logger.info(
                f"Vol-adjusted scores: mean={valid_scores.mean():.4f}, "
                f"std={valid_scores.std():.4f}, "
                f"min={valid_scores.min():.4f}, "
                f"max={valid_scores.max():.4f}, "
                f"valid={valid_count}/{len(universe.tickers)}"
            )
        else:
            logger.warning("No valid vol-adjusted scores computed")

        return AlphaScores(as_of=as_of, scores=scores_series)
