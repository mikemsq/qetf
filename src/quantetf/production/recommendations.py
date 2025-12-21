from __future__ import annotations

import pandas as pd

from quantetf.types import RecommendationPacket


def diff_trades(current_weights: pd.Series, target_weights: pd.Series, *, threshold: float = 0.0) -> pd.DataFrame:
    """Convert current and target weights into a trade list."""
    cur = current_weights.fillna(0.0)
    tgt = target_weights.fillna(0.0)
    tickers = sorted(set(cur.index).union(set(tgt.index)))
    cur = cur.reindex(tickers, fill_value=0.0)
    tgt = tgt.reindex(tickers, fill_value=0.0)

    delta = (tgt - cur)
    trades = pd.DataFrame(
        {
            "ticker": tickers,
            "current_weight": cur.values,
            "target_weight": tgt.values,
            "delta_weight": delta.values,
        }
    )
    if threshold > 0:
        trades = trades[trades["delta_weight"].abs() >= threshold]
    trades = trades.sort_values("delta_weight", ascending=False).reset_index(drop=True)
    return trades
