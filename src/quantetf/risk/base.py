from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

from quantetf.types import DatasetVersion, RiskModelOutput, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class RiskModelSpec:
    name: str
    params: Mapping[str, Any]


class RiskModel(ABC):
    """Produces a covariance matrix and optional exposures for a universe."""

    @abstractmethod
    def estimate(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> RiskModelOutput:
        raise NotImplementedError
