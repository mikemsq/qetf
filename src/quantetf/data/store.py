from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

from quantetf.types import DatasetVersion


@dataclass(frozen=True)
class SnapshotManifest:
    version: DatasetVersion
    tables: Mapping[str, str]  # logical name -> path
    metadata: Mapping[str, Any]


class DataStore(ABC):
    """Reads curated tables and creates immutable snapshots."""

    @abstractmethod
    def read_prices_total_return(self, version: Optional[DatasetVersion] = None) -> pd.DataFrame:
        """Return a DataFrame of daily total returns.

        Expected shape:
        - index: date (pd.Timestamp)
        - columns: tickers
        - values: daily return (decimal)
        """
        raise NotImplementedError

    @abstractmethod
    def read_instrument_master(self, version: Optional[DatasetVersion] = None) -> pd.DataFrame:
        """Return instrument metadata table, point-in-time if possible."""
        raise NotImplementedError

    @abstractmethod
    def create_snapshot(self, *, snapshot_id: str, as_of: pd.Timestamp, description: str = "") -> SnapshotManifest:
        """Create an immutable dataset snapshot and return its manifest."""
        raise NotImplementedError
