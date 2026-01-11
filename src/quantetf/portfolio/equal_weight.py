"""Equal-weight portfolio construction.

This module provides a simple portfolio constructor that selects the top N
assets by alpha score and assigns equal weights to each.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from quantetf.portfolio.base import PortfolioConstructor
from quantetf.types import AlphaScores, DatasetVersion, RiskModelOutput, TargetWeights, Universe, CASH_TICKER
from quantetf.data.store import DataStore

logger = logging.getLogger(__name__)


class EqualWeightTopN(PortfolioConstructor):
    """Portfolio constructor that selects top N assets by alpha score and assigns equal weights.

    This is the simplest portfolio construction method: rank assets by alpha score,
    select the top N, and give each an equal weight of 1/N.

    The constructor ignores risk model output, previous weights, and does not access
    the data store - it operates purely on the provided alpha scores.

    Example:
        >>> constructor = EqualWeightTopN(top_n=5)
        >>> weights = constructor.construct(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     alpha=alpha_scores,
        ...     risk=None,  # Not used
        ...     store=None  # Not used
        ... )
        >>> # weights.weights will have 5 non-zero entries summing to 1.0
    """

    def __init__(self, top_n: int = 5):
        """Initialize equal-weight constructor.

        Args:
            top_n: Number of top-scoring assets to select (default: 5)

        Raises:
            ValueError: If top_n is less than 1
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        self.top_n = top_n

    def construct(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        alpha: AlphaScores,
        risk: RiskModelOutput,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
        prev_weights: Optional[pd.Series] = None,
    ) -> TargetWeights:
        """Construct equal-weight portfolio from top N alpha scores.

        Selects the top N assets by alpha score (highest scores win) and assigns
        each an equal weight of 1/N. All other assets in the universe receive
        zero weight.

        Args:
            as_of: Date as of which to construct portfolio
            universe: Set of eligible tickers
            alpha: Alpha scores for the universe
            risk: Risk model output (not used in equal-weight)
            store: Data store (not used in equal-weight)
            dataset_version: Optional dataset version
            prev_weights: Optional previous weights (not used in equal-weight)

        Returns:
            TargetWeights with equal weights for top N tickers and diagnostics

        Raises:
            ValueError: If alpha scores and universe tickers don't align
        """
        logger.info(f"Constructing equal-weight portfolio as of {as_of}: top_n={self.top_n}")

        # Validate that alpha scores align with universe
        alpha_tickers = set(alpha.scores.index)
        universe_tickers = set(universe.tickers)
        if not alpha_tickers.issuperset(universe_tickers):
            missing = universe_tickers - alpha_tickers
            logger.warning(f"Universe contains tickers not in alpha scores: {missing}")

        # Get valid alpha scores (drop NaN - these are tickers with insufficient data)
        valid_scores = alpha.scores.dropna()
        logger.info(f"Valid scores: {len(valid_scores)}/{len(alpha.scores)}")

        if len(valid_scores) == 0:
            logger.warning("No valid alpha scores available, allocating 100% to cash")
            all_tickers = list(universe.tickers) + [CASH_TICKER]
            weights = pd.Series(0.0, index=all_tickers)
            weights[CASH_TICKER] = 1.0
            return TargetWeights(
                as_of=as_of,
                weights=weights,
                diagnostics={
                    'top_n': self.top_n,
                    'selected': [],
                    'num_valid_scores': 0,
                    'num_universe_tickers': len(universe.tickers),
                    'cash_weight': 1.0
                }
            )

        # Rank by score (descending - higher score is better)
        ranked = valid_scores.sort_values(ascending=False)

        # Select top N tickers
        num_selected = min(self.top_n, len(ranked))
        top_tickers = ranked.head(num_selected).index.tolist()

        # Calculate equal weight across top_n positions (not num_selected)
        # This ensures total weight always sums to 1.0
        weight = 1.0 / self.top_n

        logger.info(
            f"Selected {num_selected} tickers: {top_tickers} "
            f"(weight={weight:.4f} each)"
        )

        # Create weights series for entire universe plus cash (0.0 for non-selected)
        all_tickers = list(universe.tickers) + [CASH_TICKER]
        weights = pd.Series(0.0, index=all_tickers)
        weights[top_tickers] = weight

        # Allocate remaining weight to cash if we have fewer than top_n positions
        if num_selected < self.top_n:
            cash_weight = (self.top_n - num_selected) * weight
            weights[CASH_TICKER] = cash_weight
            logger.info(f"Allocated {cash_weight:.4f} to cash (insufficient valid scores)")

        # Verify sum (should be ~1.0)
        weight_sum = weights.sum()
        logger.info(f"Total weight: {weight_sum:.6f}")

        return TargetWeights(
            as_of=as_of,
            weights=weights,
            diagnostics={
                'top_n': self.top_n,
                'selected': top_tickers,
                'num_valid_scores': len(valid_scores),
                'num_universe_tickers': len(universe.tickers),
                'num_selected': num_selected,
                'weight_per_ticker': weight,
                'total_weight': float(weight_sum),
                'cash_weight': float(weights.get(CASH_TICKER, 0.0))
            }
        )
