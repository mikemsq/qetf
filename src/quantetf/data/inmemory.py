from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quantetf.data.store import DataStore, SnapshotManifest
from quantetf.types import DatasetVersion


@dataclass
class InMemoryDataStore(DataStore):
    """A tiny in-memory store used for tests and examples."""

    prices_total_return: pd.DataFrame
    instrument_master: pd.DataFrame

    def read_prices_total_return(self, version: Optional[DatasetVersion] = None) -> pd.DataFrame:
        return self.prices_total_return

    def read_instrument_master(self, version: Optional[DatasetVersion] = None) -> pd.DataFrame:
        return self.instrument_master

    def create_snapshot(self, *, snapshot_id: str, as_of: pd.Timestamp, description: str = "") -> SnapshotManifest:
        version = DatasetVersion(id=snapshot_id, created_at=pd.Timestamp.utcnow().to_pydatetime(), description=description)
        return SnapshotManifest(version=version, tables={}, metadata={"as_of": str(as_of)})
