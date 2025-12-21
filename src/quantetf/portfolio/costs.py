from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from quantetf.portfolio.base import CostModel


@dataclass(frozen=True)
class SimpleLinearCostModel(CostModel):
    """Linear cost model using a single bps-per-dollar-turnover style assumption."""

    total_bps_per_turnover: float = 10.0  # combined commission + slippage + spread

    def estimate_rebalance_cost(
        self,
        *,
        prev_weights: pd.Series,
        next_weights: pd.Series,
        prices: pd.Series | None = None,
    ) -> float:
        prev = prev_weights.fillna(0.0)
        nxt = next_weights.fillna(0.0)
        tickers = sorted(set(prev.index).union(set(nxt.index)))
        prev = prev.reindex(tickers, fill_value=0.0)
        nxt = nxt.reindex(tickers, fill_value=0.0)

        turnover = 0.5 * (nxt - prev).abs().sum()  # 0..1 for fully invested portfolios
        return (self.total_bps_per_turnover / 10_000.0) * float(turnover)
