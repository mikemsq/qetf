from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from quantetf.types import RecommendationPacket


@dataclass
class RecommendationWriter:
    root: Path

    def write(self, *, run_id: str, packet: RecommendationPacket) -> Path:
        out = self.root / run_id
        out.mkdir(parents=True, exist_ok=True)

        packet.trades.to_csv(out / "trades.csv", index=False)
        packet.target_weights.to_csv(out / "target_weights.csv", header=True)

        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(dict(packet.summary), f, indent=2)

        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(dict(packet.manifest), f, indent=2)

        return out
