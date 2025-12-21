from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from quantetf.alpha.base import AlphaModel
from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class WeightedEnsemble(AlphaModel):
    models: Sequence[AlphaModel]
    weights: Sequence[float]

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        store: DataStore,
        dataset_version: DatasetVersion | None = None,
    ) -> AlphaScores:
        if len(self.models) != len(self.weights):
            raise ValueError("models and weights must have the same length")
        combined = None
        for m, w in zip(self.models, self.weights):
            s = m.score(
                as_of=as_of,
                universe=universe,
                features=features,
                store=store,
                dataset_version=dataset_version,
            ).scores * float(w)
            combined = s if combined is None else (combined.add(s, fill_value=0.0))
        return AlphaScores(as_of=as_of, scores=combined)
