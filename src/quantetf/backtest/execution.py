from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Execution assumptions for the backtest simulator."""
    fills_at: str = "close"  # "close" or "next_open" placeholder
    allow_partial_fills: bool = False


def compute_turnover(prev_w: pd.Series, next_w: pd.Series) -> float:
    prev = prev_w.fillna(0.0)
    nxt = next_w.fillna(0.0)
    tickers = sorted(set(prev.index).union(set(nxt.index)))
    prev = prev.reindex(tickers, fill_value=0.0)
    nxt = nxt.reindex(tickers, fill_value=0.0)
    return float(0.5 * (nxt - prev).abs().sum())
