from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantetf.alpha.base import AlphaModel
from quantetf.types import AlphaScores, DatasetVersion, FeatureFrame, Universe
from quantetf.data.store import DataStore


@dataclass(frozen=True)
class CrossSectionalMomentum(AlphaModel):
    """Alpha model that reads a precomputed momentum feature column."""

    feature_name: str = "momentum"

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        store: DataStore,
        dataset_version: DatasetVersion | None = None,
    ) -> AlphaScores:
        if self.feature_name not in features.frame.columns:
            raise KeyError(f"Missing feature '{self.feature_name}' in feature frame")
        scores = features.frame[self.feature_name].astype(float).copy()
        scores = scores.loc[list(universe.tickers)]
        return AlphaScores(as_of=as_of, scores=scores)
