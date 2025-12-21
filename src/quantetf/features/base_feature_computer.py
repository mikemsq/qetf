from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quantetf.features.base import FeatureComputer
from quantetf.features.library import compute_momentum_score, compute_realized_vol
from quantetf.types import DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class BasicFeatureComputer(FeatureComputer):
    """Computes a minimal feature set used by starter strategies."""

    momentum_lookback_days: int = 126
    momentum_skip_recent_days: int = 5
    vol_lookback_days: int = 60

    def compute(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> FeatureFrame:
        ret = store.read_prices_total_return(version=dataset_version)

        momentum = compute_momentum_score(
            ret,
            as_of=as_of,
            universe=universe,
            lookback_days=self.momentum_lookback_days,
            skip_recent_days=self.momentum_skip_recent_days,
        )
        vol = compute_realized_vol(ret, as_of=as_of, universe=universe, lookback_days=self.vol_lookback_days)

        frame = pd.DataFrame({"momentum": momentum, "vol": vol})
        frame.index.name = "ticker"
        return FeatureFrame(as_of=as_of, frame=frame)
