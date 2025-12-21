from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class IngestionResult:
    source: str
    retrieved_at: pd.Timestamp
    artifacts: Mapping[str, str]  # name -> path
    metadata: Mapping[str, Any]


class DataConnector(ABC):
    """Fetches raw data from a source and writes it into data/raw/.

    Connectors should be side-effectful but deterministic given parameters.
    """

    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @abstractmethod
    def ingest(self, *, as_of: Optional[pd.Timestamp] = None, **kwargs: Any) -> IngestionResult: ...
