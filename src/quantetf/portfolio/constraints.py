from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class WeightConstraints:
    min_weight: float = 0.0
    max_weight: float = 1.0

def clip_weights(w: pd.Series, *, constraints: WeightConstraints) -> pd.Series:
    clipped = w.clip(lower=constraints.min_weight, upper=constraints.max_weight)
    if clipped.sum() != 0:
        clipped = clipped / clipped.sum()
    return clipped
