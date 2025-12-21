from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quantetf.portfolio.base import PortfolioConstructor
from quantetf.portfolio.constraints import WeightConstraints, clip_weights
from quantetf.types import AlphaScores, DatasetVersion, RiskModelOutput, TargetWeights, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class TopXEqualWeight(PortfolioConstructor):
    x: int
    constraints: WeightConstraints = WeightConstraints()

    def construct(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        alpha: AlphaScores,
        risk: RiskModelOutput,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
        prev_weights: Optional[pd.Series] = None,
    ) -> TargetWeights:
        scores = alpha.scores.dropna().sort_values(ascending=False)
        top = scores.head(self.x).index.tolist()

        w = pd.Series(0.0, index=list(universe.tickers), dtype=float)
        if len(top) > 0:
            w.loc[top] = 1.0 / float(len(top))

        w = clip_weights(w, constraints=self.constraints)

        diagnostics = {
            "selected": top,
            "x": self.x,
        }
        return TargetWeights(as_of=as_of, weights=w, diagnostics=diagnostics)
