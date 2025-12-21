from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from quantetf.types import DatasetVersion, RecommendationPacket
from quantetf.production.recommendations import diff_trades


@dataclass
class ProductionPipeline:
    """Runs the end-to-end pipeline to produce trading recommendations.

    This class is intentionally thin. It should orchestrate:
    - loading the current portfolio state
    - computing target weights for 'as_of'
    - producing a trade list and summary
    - writing an immutable recommendation packet
    """

    artifacts_root: Path

    def run(
        self,
        *,
        as_of: pd.Timestamp,
        dataset_version: DatasetVersion,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> RecommendationPacket:
        trades = diff_trades(current_weights, target_weights, threshold=0.0)

        summary = {
            "as_of": str(as_of),
            "dataset_id": dataset_version.id,
            "num_trades": int(len(trades)),
            "gross_turnover": float(0.5 * trades["delta_weight"].abs().sum()),
        }

        manifest = {
            "dataset_id": dataset_version.id,
            "as_of": str(as_of),
        }

        packet = RecommendationPacket(
            as_of=as_of,
            target_weights=target_weights,
            trades=trades,
            summary=summary,
            manifest=manifest,
        )
        return packet
