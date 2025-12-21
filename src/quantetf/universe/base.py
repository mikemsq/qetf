from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from quantetf.types import DatasetVersion, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class UniverseSpec:
    name: str
    params: Mapping[str, Any]


class UniverseBuilder(ABC):
    """Builds an eligible universe as-of a given timestamp."""

    @abstractmethod
    def build(
        self,
        *,
        as_of: pd.Timestamp,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> Universe:
        raise NotImplementedError
