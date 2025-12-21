from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from quantetf.types import DatasetVersion, FeatureFrame


@dataclass(frozen=True)
class FeatureCacheKey:
    dataset_id: str
    name: str
    as_of: str  # ISO date string


class FeatureStore:
    """Lightweight on-disk cache for feature frames."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: FeatureCacheKey) -> Path:
        return self.root / "features" / key.dataset_id / key.name / f"{key.as_of}.parquet"

    def write(self, key: FeatureCacheKey, frame: FeatureFrame) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.frame.to_parquet(path)
        return path

    def read(self, key: FeatureCacheKey) -> Optional[pd.DataFrame]:
        path = self._path(key)
        if not path.exists():
            return None
        return pd.read_parquet(path)
