"""Residual momentum alpha model.

This module implements beta-neutral momentum by regressing ticker returns
against SPY and ranking by residuals. This isolates idiosyncratic momentum
independent of market beta exposure.

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


class ResidualMomentumAlpha(AlphaModel):
    """Residual momentum alpha: rank by beta-neutral return residuals.

    This model removes market (SPY) exposure from ticker returns via OLS regression,
    then ranks tickers by the sum of their residuals over the lookback period.

    The intuition: tickers with high residual momentum have strong idiosyncratic
    trends independent of the overall market direction.

    Mathematical approach:
        1. For each ticker, regress daily returns on SPY returns
        2. Extract residuals (returns unexplained by market beta)
        3. Sum residuals over lookback period
        4. Rank tickers by cumulative residuals (higher = better)

    Point-in-time compliance:
        - Uses only data BEFORE as_of date (T-1 and earlier)
        - No lookahead bias in regression or scoring

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/latest/data.parquet"}
        ... )
        >>> alpha = ResidualMomentumAlpha(lookback_days=252)
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
        spy_ticker: str = "SPY"
    ) -> None:
        """Initialize residual momentum alpha model.

        Args:
            lookback_days: Number of trading days for regression window (default: 252)
            min_periods: Minimum valid returns required for regression (default: 200)
            spy_ticker: Market benchmark ticker for beta calculation (default: "SPY")

        Raises:
            ValueError: If parameters are invalid
        """
        if lookback_days < min_periods:
            raise ValueError(f"lookback_days ({lookback_days}) must be >= min_periods ({min_periods})")
        if min_periods < 50:
            raise ValueError(f"min_periods must be >= 50 for stable regression")

        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.spy_ticker = spy_ticker

        logger.info(
            f"Initialized ResidualMomentumAlpha: "
            f"lookback={lookback_days}, min_periods={min_periods}, "
            f"benchmark={spy_ticker}"
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
        """Compute residual momentum scores for the universe.

        CRITICAL: Uses only data BEFORE as_of (T-1 and earlier) to prevent lookahead.

        Args:
            as_of: Decision date (score computed using data up to T-1)
            universe: Set of eligible tickers to score
            features: Pre-computed features (not used in this model)
            data_access: DataAccessContext for accessing price history
            dataset_version: Optional dataset version for reproducibility

        Returns:
            AlphaScores with residual momentum score for each ticker.
            NaN scores indicate insufficient data for regression.

        Raises:
            ValueError: If SPY data is unavailable or insufficient
        """
        logger.info(f"Computing residual momentum as of {as_of} for {len(universe.tickers)} tickers")

        # Get prices for SPY and universe tickers
        tickers_with_spy = list(set(list(universe.tickers) + [self.spy_ticker]))

        try:
            ohlcv_data = data_access.prices.read_prices_as_of(
                as_of=as_of,
                tickers=tickers_with_spy,
                lookback_days=self.lookback_days + 50
            )
            prices = ohlcv_data.xs('Close', level='Price', axis=1)
        except ValueError as e:
            # If error mentions SPY specifically, raise it
            if self.spy_ticker in str(e) or "SPY" in str(e):
                raise ValueError(f"SPY data not available as of {as_of}") from e
            # Otherwise return NaN for all tickers
            logger.error(f"Failed to get prices: {e}")
            return AlphaScores(
                as_of=as_of,
                scores=pd.Series(np.nan, index=list(universe.tickers))
            )

        # Verify SPY exists
        if self.spy_ticker not in prices.columns:
            raise ValueError(f"SPY data not available as of {as_of}")

        # Calculate daily returns
        returns = prices.pct_change().dropna()

        # Get SPY returns
        spy_returns = returns[self.spy_ticker].dropna()

        if len(spy_returns) < self.min_periods:
            logger.warning(f"Insufficient SPY data: {len(spy_returns)} < {self.min_periods}")
            return AlphaScores(
                as_of=as_of,
                scores=pd.Series(np.nan, index=list(universe.tickers))
            )

        # Calculate residual momentum for each ticker
        scores = {}
        valid_count = 0

        for ticker in universe.tickers:
            # Skip if ticker is SPY itself
            if ticker == self.spy_ticker:
                scores[ticker] = np.nan
                logger.debug(f"{ticker}: Skipping (same as benchmark)")
                continue

            if ticker not in returns.columns:
                logger.debug(f"{ticker}: No price data available")
                scores[ticker] = np.nan
                continue

            ticker_returns = returns[ticker].dropna()

            # Align ticker and SPY returns on common dates
            common_dates = ticker_returns.index.intersection(spy_returns.index)

            if len(common_dates) < self.min_periods:
                logger.debug(
                    f"{ticker}: Insufficient aligned data "
                    f"({len(common_dates)} < {self.min_periods})"
                )
                scores[ticker] = np.nan
                continue

            # Get aligned returns
            y = ticker_returns.loc[common_dates].values
            x = spy_returns.loc[common_dates].values

            # Run OLS regression: y = alpha + beta * x + residuals
            try:
                # Add intercept term
                X = np.column_stack([np.ones(len(x)), x])

                # Solve: β = (X'X)^-1 X'y
                beta = np.linalg.lstsq(X, y, rcond=None)[0]

                # Calculate residuals: ε = y - Xβ
                y_pred = X @ beta
                residuals = y - y_pred

                # Score = sum of residuals over lookback period
                residual_momentum = residuals.sum()

                scores[ticker] = residual_momentum
                valid_count += 1

                logger.debug(
                    f"{ticker}: beta={beta[1]:.4f}, alpha={beta[0]:.6f}, "
                    f"residual_sum={residual_momentum:.6f}"
                )

            except np.linalg.LinAlgError:
                logger.warning(f"{ticker}: Regression failed (singular matrix)")
                scores[ticker] = np.nan

        # Log summary statistics
        scores_series = pd.Series(scores)
        valid_scores = scores_series.dropna()

        if len(valid_scores) > 0:
            logger.info(
                f"Residual momentum scores: mean={valid_scores.mean():.6f}, "
                f"std={valid_scores.std():.6f}, "
                f"min={valid_scores.min():.6f}, "
                f"max={valid_scores.max():.6f}, "
                f"valid={valid_count}/{len(universe.tickers)}"
            )
        else:
            logger.warning("No valid residual momentum scores computed")

        return AlphaScores(as_of=as_of, scores=scores_series)
