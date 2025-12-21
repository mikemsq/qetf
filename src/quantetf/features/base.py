from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from quantetf.types import DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    params: Mapping[str, Any]


class FeatureComputer(ABC):
    """Computes features for a universe as-of a given timestamp."""

    @abstractmethod
    def compute(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> FeatureFrame:
        raise NotImplementedError
