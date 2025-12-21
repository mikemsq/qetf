from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class AlphaModelSpec:
    name: str
    params: Mapping[str, Any]


class AlphaModel(ABC):
    """Produces alpha scores (rankings or expected returns) for a universe."""

    @abstractmethod
    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> AlphaScores:
        raise NotImplementedError
