from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

from quantetf.types import AlphaScores, DatasetVersion, RiskModelOutput, TargetWeights, Universe
from quantetf.data.access import DataAccessContext


@dataclass(frozen=True)
class PortfolioConstructionSpec:
    name: str
    params: Mapping[str, Any]


class CostModel(ABC):
    """Estimates transaction costs given turnover or trade list."""

    @abstractmethod
    def estimate_rebalance_cost(
        self,
        *,
        prev_weights: pd.Series,
        next_weights: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> float:
        """Return a cost as a fraction of NAV (for example 0.001 = 10 bps)."""
        raise NotImplementedError


class PortfolioConstructor(ABC):
    """Builds target weights from alpha and risk, subject to constraints.

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching
    """

    @abstractmethod
    def construct(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        alpha: AlphaScores,
        risk: RiskModelOutput,
        data_access: DataAccessContext,
        dataset_version: Optional[DatasetVersion] = None,
        prev_weights: Optional[pd.Series] = None,
    ) -> TargetWeights:
        raise NotImplementedError
