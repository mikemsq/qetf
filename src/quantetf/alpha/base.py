from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import pandas as pd

from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.access import DataAccessContext, PriceDataAccessor


@dataclass(frozen=True)
class AlphaModelSpec:
    name: str
    params: Mapping[str, Any]


class AlphaModel(ABC):
    """Produces alpha scores (rankings or expected returns) for a universe.

    Alpha models use DataAccessContext for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching
    - Point-in-time data access (no lookahead bias)

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/latest/data.parquet"}
        ... )
        >>> alpha = MomentumAlpha(lookback_days=252)
        >>> scores = alpha.score(
        ...     as_of=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     features=None,
        ...     data_access=ctx
        ... )
    """

    @abstractmethod
    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        data_access: DataAccessContext,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> AlphaScores:
        """Compute alpha scores for the universe.

        CRITICAL: This method must ONLY use data available before as_of date.
        It should retrieve prices up to (but not including) as_of, ensuring no lookahead.

        Args:
            as_of: Date as of which to compute scores (decisions made on this date)
            universe: Set of eligible tickers
            features: Pre-computed features (may be None for simple models)
            data_access: DataAccessContext for accessing price and macro data
            dataset_version: Optional dataset version for reproducibility

        Returns:
            AlphaScores with scores for each ticker in universe
        """
        raise NotImplementedError
