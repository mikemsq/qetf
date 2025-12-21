from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantetf.risk.base import RiskModel
from quantetf.types import DatasetVersion, RiskModelOutput, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class EWMACovariance(RiskModel):
    halflife_days: int = 60
    lookback_days: int = 252

    def estimate(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        store: DataStore,
        dataset_version: DatasetVersion | None = None,
    ) -> RiskModelOutput:
        ret = store.read_prices_total_return(version=dataset_version)
        end = as_of
        start = end - pd.tseries.offsets.BDay(self.lookback_days)
        window = ret.loc[start:end, list(universe.tickers)].dropna(how="all")

        # EWMA covariance
        lam = 0.5 ** (1.0 / float(self.halflife_days))
        w = np.array([(1 - lam) * lam ** i for i in range(len(window) - 1, -1, -1)], dtype=float)
        w = w / w.sum()

        X = window.fillna(0.0).to_numpy()
        mu = (w[:, None] * X).sum(axis=0)
        Xc = X - mu[None, :]
        cov = (Xc.T * w) @ Xc

        cov_df = pd.DataFrame(cov, index=window.columns, columns=window.columns)
        return RiskModelOutput(as_of=as_of, cov=cov_df, exposures=None, diagnostics={"method": "ewma"})
