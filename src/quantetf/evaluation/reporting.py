from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import pandas as pd

from quantetf.types import BacktestResult


@dataclass
class ReportWriter:
    """Writes a minimal set of run artifacts to a folder."""

    root: Path

    def write(self, *, run_id: str, result: BacktestResult) -> Path:
        out = self.root / run_id
        out.mkdir(parents=True, exist_ok=True)

        result.returns.to_csv(out / "returns.csv", header=True)
        result.equity_curve.to_csv(out / "equity_curve.csv", header=True)
        result.positions.to_csv(out / "positions.csv")
        result.trades.to_csv(out / "trades.csv")

        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(dict(result.metrics), f, indent=2)

        with open(out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(dict(result.metadata), f, indent=2)

        return out
