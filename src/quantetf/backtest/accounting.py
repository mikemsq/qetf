from __future__ import annotations

import pandas as pd


def compute_portfolio_returns(returns: pd.DataFrame, positions: pd.DataFrame) -> pd.Series:
    """Compute daily portfolio returns given asset returns and daily weights.

    Both inputs are expected to share:
    - index: date
    - columns: tickers
    """
    aligned_r = returns.reindex_like(positions).fillna(0.0)
    aligned_w = positions.fillna(0.0)
    return (aligned_w * aligned_r).sum(axis=1)
